#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP Course — Lecture 8
Large Language Models in practice: sizes, prices, sampling, LoRA, quantization.

Lecture 7 fine-tuned encoders of 80–280 M parameters on a laptop. This lecture is
about the models that are 10 to 10 000 times larger — what the size means, what
they cost to call, how to steer them, and what you can still do to them without a
GPU.

Nothing here trains a large model. Every measurement is either arithmetic you can
check, a tokenizer count, a parameter count, or a generation from a 0.5 B model
that runs on a laptop.

Setup:
    pip install -r requirements.txt

Steps:
    1. tokens    — what a token is, and what a language costs
    2. memory    — parameters -> gigabytes, at each precision
    3. sampling  — temperature, top-k, top-p, on a real model
    4. lora      — how few parameters LoRA actually trains (counted, not trained)
    5. quantize  — what int8 does to size and to output

    python3 lesson8.py --tokens        # tokenizer economics, kk / ru / en
    python3 lesson8.py --memory        # size tables and KV cache arithmetic
    python3 lesson8.py --sampling      # generation under different settings
    python3 lesson8.py --lora          # LoRA parameter counts
    python3 lesson8.py --quantize      # int8 weights: size and quality
    python3 lesson8.py --all
"""

import argparse
import math
import os
import time

import torch

SMALL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# One meaning, three languages. The Kazakh and Russian are translations of the
# English, so any difference in token count is about the tokenizer, not content.
PARALLEL = {
    "en": "Hello, I cannot log in to the application. Please help me restore access to my account.",
    "ru": "Здравствуйте, я не могу войти в приложение. Пожалуйста, помогите восстановить доступ к моему аккаунту.",
    "kk": "Сәлеметсіз бе, мен қосымшаға кіре алмай жатырмын. Аккаунтыма қолжетімділікті қалпына келтіруге көмектесіңізші.",
}

# Anthropic first-party list prices, US dollars per million tokens.
# Cached 2026-06-24 — prices change; re-check before quoting these anywhere.
CLAUDE_PRICES = [
    # (model, context, input $/Mtok, output $/Mtok)
    ("claude-opus-5", "1M", 5.00, 25.00),
    ("claude-sonnet-5", "1M", 2.00, 10.00),
    ("claude-haiku-4-5", "200K", 1.00, 5.00),
]


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = device()


# ---------------------------------------------------------------------------
# 1. Tokens — the unit you are billed in
# ---------------------------------------------------------------------------

def count_tiktoken(encoding_name, text):
    import tiktoken
    return len(tiktoken.get_encoding(encoding_name).encode(text))


def count_hf(name, text):
    from transformers import AutoTokenizer
    return len(AutoTokenizer.from_pretrained(name).tokenize(text))


def tokens_demo():
    """How many tokens does the same sentence cost in each language?"""
    print("=" * 74)
    print("WHAT A TOKEN IS, AND WHAT A LANGUAGE COSTS")
    print("=" * 74)

    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    sample = "Fine-tuning is expensive."
    print(f"\n  «{sample}» under o200k_base:")
    for piece in enc.encode(sample):
        print(f"      {piece:>7}  ->  {enc.decode([piece])!r}")
    print("  A token is a piece of a word. Billing counts these, not words.")

    encoders = [
        ("o200k_base  (GPT-4o family)", lambda t: count_tiktoken("o200k_base", t)),
        ("cl100k_base (GPT-4 / 3.5)", lambda t: count_tiktoken("cl100k_base", t)),
        ("Qwen2.5", lambda t: count_hf("Qwen/Qwen2.5-0.5B-Instruct", t)),
    ]

    print("\n  the same message, three languages, tokens per message:")
    header = f"  {'tokenizer':<30}" + "".join(f"{k:>8}" for k in PARALLEL)
    print(header)
    print("  " + "-" * (len(header) - 2))
    baselines = {}
    for label, fn in encoders:
        counts = {}
        for lang, text in PARALLEL.items():
            try:
                counts[lang] = fn(text)
            except Exception as error:
                counts[lang] = None
                print(f"    {label}: {type(error).__name__}")
        baselines[label] = counts
        row = "".join(f"{counts[k]:>8}" if counts[k] else f"{'-':>8}" for k in PARALLEL)
        print(f"  {label:<30}{row}")

    print("\n  cost multiplier against English (same meaning):")
    for label, counts in baselines.items():
        if not counts.get("en"):
            continue
        row = "".join(f"{counts[k] / counts['en']:>8.2f}" if counts[k] else f"{'-':>8}"
                      for k in PARALLEL)
        print(f"  {label:<30}{row}")

    words = {k: len(v.split()) for k, v in PARALLEL.items()}
    print(f"\n  words per message: " + "  ".join(f"{k} {v}" for k, v in words.items()))
    print("  (so the difference below is the tokenizer, not a longer sentence)")
    for label, counts in baselines.items():
        if not counts.get("en"):
            continue
        row = "".join(f"{counts[k] / words[k]:>8.2f}" for k in PARALLEL if counts[k])
        print(f"  tokens/word  {label:<17}{row}")


def pricing_demo(calls_per_month=100_000):
    """Work a monthly bill, in each language, from measured token counts.

    The reply is written in the user's language too, so the language multiplier
    applies to the output as well — assuming a fixed output length regardless of
    language would understate the effect.
    """
    print("\n" + "=" * 74)
    print("WORKING OUT AN API BILL")
    print("=" * 74)
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")

    # Measured multiplier: tokens per word, relative to English.
    per_word = {lang: len(enc.encode(text)) / len(text.split())
                for lang, text in PARALLEL.items()}
    factor = {lang: per_word[lang] / per_word["en"] for lang in PARALLEL}

    system_prompt = ("You are a helpful customer-support assistant for a Kazakhstan "
                     "bank. Answer briefly and politely. If the user reports a login "
                     "problem, ask for the phone number on the account.")
    system_tokens = len(enc.encode(system_prompt))

    # Two workload shapes with very different input/output balance.
    workloads = [
        ("chat reply", 20, 100),           # short question in, paragraph out
        ("classify a document", 2_000, 5),  # long document in, one label out
    ]

    print(f"\n  {calls_per_month:,} calls per month, o200k_base, system prompt"
          f" {system_tokens} tokens")
    print(f"  measured language multiplier (tokens per word vs English): "
          + ", ".join(f"{k} {factor[k]:.2f}x" for k in PARALLEL))

    for name, words_in, words_out in workloads:
        print(f"\n  --- {name}: {words_in} words in, {words_out} words out ---")
        print(f"  {'model':<20}" + "".join(f"{lang:>13}" for lang in PARALLEL)
              + f"{'kk vs en':>11}")
        print("  " + "-" * 64)
        for model, _ctx, price_in, price_out in CLAUDE_PRICES:
            costs = {}
            for lang in PARALLEL:
                tok_in = system_tokens + words_in * per_word["en"] * factor[lang]
                tok_out = words_out * per_word["en"] * factor[lang]
                costs[lang] = calls_per_month * (
                    tok_in / 1e6 * price_in + tok_out / 1e6 * price_out)
            row = "".join(f"{costs[l]:>12,.0f}$" for l in PARALLEL)
            print(f"  {model:<20}{row}{costs['kk'] / costs['en']:>10.2f}x")

    print("\n  Read the two blocks against each other:")
    print("   - the Kazakh penalty is ~2.6x in both, because the tokenizer inflates")
    print("     the input and the output alike — it is a property of the language,")
    print("     not of the workload")
    print("   - but the absolute bill is 3.7x higher for the classification job even")
    print("     though it emits five words: 2 000 words of input is what costs money")
    print("\n  Three levers, all free:")
    print("   1. output tokens cost 5x input — 'answer briefly' is a budget decision")
    print("   2. the system prompt is resent every call — prompt caching exists for this")
    print("   3. a cheaper model on the easy half of your traffic")


# ---------------------------------------------------------------------------
# 2. Memory — what a parameter count means in gigabytes
# ---------------------------------------------------------------------------

BYTES = {"fp32": 4, "fp16/bf16": 2, "int8": 1, "int4": 0.5}


def memory_demo():
    print("=" * 74)
    print("PARAMETERS TO GIGABYTES")
    print("=" * 74)
    sizes = [("0.5 B", 0.5e9), ("7 B", 7e9), ("70 B", 70e9),
             ("405 B", 405e9), ("1 T", 1e12)]
    print(f"\n  weights only, no activations, no KV cache\n")
    print(f"  {'model':<10}" + "".join(f"{p:>14}" for p in BYTES))
    print("  " + "-" * 66)
    for label, n in sizes:
        row = "".join(f"{n * b / 1e9:>11.1f} GB" for b in BYTES.values())
        print(f"  {label:<10}{row}")

    ram = 24
    print(f"\n  what fits in {ram} GB of unified memory (this laptop), weights only:")
    for label, n in sizes:
        fits = [p for p, b in BYTES.items() if n * b / 1e9 < ram * 0.8]
        print(f"    {label:<8} {', '.join(fits) if fits else 'nothing — needs a bigger machine'}")

    print("\n  Rule of thumb: a model in fp16 needs about 2 GB per billion parameters,")
    print("  and you need headroom on top for activations and the KV cache.")


def kv_cache_demo():
    print("\n" + "=" * 74)
    print("THE KV CACHE — why long chats get expensive")
    print("=" * 74)
    # Llama-3-8B-ish geometry, grouped-query attention
    layers, kv_heads, head_dim, bytes_per = 32, 8, 128, 2
    per_token = 2 * layers * kv_heads * head_dim * bytes_per
    print(f"\n  a 8 B model: {layers} layers, {kv_heads} KV heads, head dim {head_dim}, bf16")
    print(f"  per token, per sequence: 2 x {layers} x {kv_heads} x {head_dim} x {bytes_per}"
          f" = {per_token:,} bytes = {per_token / 1024:.0f} KB\n")
    print(f"  {'context':>10}{'1 user':>12}{'10 users':>12}{'100 users':>12}")
    print("  " + "-" * 44)
    for ctx in (1_000, 8_000, 32_000, 128_000):
        row = "".join(f"{per_token * ctx * u / 1e9:>11.2f} GB" for u in (1, 10, 100))
        print(f"  {ctx:>10,}{row}")
    print("\n  The weights are loaded once and shared. The KV cache is per user, per")
    print("  token, and it is why serving long contexts to many people is the hard part.")
    print("  Grouped-query attention (Lecture 6) exists to shrink exactly this table.")


# ---------------------------------------------------------------------------
# 3. Sampling — the knobs you actually turn
# ---------------------------------------------------------------------------

def load_small():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        SMALL_MODEL, dtype=torch.float32).to(DEVICE).eval()
    return tokenizer, model


def generate(tokenizer, model, prompt, seed=0, max_new_tokens=40, **kwargs):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").to(DEVICE)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.eos_token_id, **kwargs)
    return tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def sampling_demo():
    print("=" * 74)
    print("SAMPLING PARAMETERS")
    print("=" * 74)
    tokenizer, model = load_small()
    prompt = "Write one sentence about Almaty."

    print(f"\n  model {SMALL_MODEL}, prompt: «{prompt}»")

    print("\n  --- greedy (do_sample=False): the argmax every time ---")
    for seed in (0, 1, 2):
        print(f"    seed {seed}: {generate(tokenizer, model, prompt, seed, do_sample=False)}")
    print("    Identical, because no randomness is involved at all.")

    print("\n  --- temperature, with sampling on ---")
    for temp in (0.1, 0.7, 1.5):
        print(f"\n    temperature={temp}")
        for seed in (0, 1):
            print(f"      seed {seed}: {generate(tokenizer, model, prompt, seed, do_sample=True, temperature=temp, top_p=1.0, top_k=0)}")

    print("\n  --- top-k: keep only the k most likely tokens ---")
    for k in (1, 5, 50):
        print(f"    top_k={k:<4}{generate(tokenizer, model, prompt, 0, do_sample=True, temperature=1.0, top_k=k, top_p=1.0)}")

    print("\n  --- top-p (nucleus): keep the smallest set summing to p ---")
    for p in (0.1, 0.9, 1.0):
        print(f"    top_p={p:<5}{generate(tokenizer, model, prompt, 0, do_sample=True, temperature=1.0, top_p=p, top_k=0)}")

    print("\n  --- what the distribution looks like at each temperature ---")
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**ids).logits[0, -1]
    for temp in (0.1, 0.7, 1.0, 2.0):
        probs = (logits / temp).softmax(-1)
        top = probs.topk(5)
        entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum())
        shown = ", ".join(f"{tokenizer.decode([i]).strip()!r} {p:.3f}"
                          for p, i in zip(top.values.tolist(), top.indices.tolist()))
        print(f"    T={temp:<4} entropy {entropy:>5.2f}   {shown}")
    print("\n  Temperature does not add knowledge. It reshapes one distribution:")
    print("  low temperature sharpens it, high temperature flattens it.")


# ---------------------------------------------------------------------------
# 4. LoRA — counted, not trained
# ---------------------------------------------------------------------------

def lora_demo():
    print("=" * 74)
    print("LoRA — HOW FEW PARAMETERS IT ACTUALLY TRAINS")
    print("=" * 74)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(SMALL_MODEL, dtype=torch.float32)
    total = sum(p.numel() for p in base.parameters())
    print(f"\n  base model {SMALL_MODEL}: {total:,} parameters")
    print(f"  full fine-tuning would train all {total:,} of them,")
    print(f"  and the optimiser needs roughly 12–16 bytes per parameter on top.\n")

    print(f"  {'rank r':>8}{'trainable':>14}{'% of model':>13}{'adapter MB':>13}")
    print("  " + "-" * 50)
    for r in (4, 8, 16, 64):
        model = get_peft_model(
            AutoModelForCausalLM.from_pretrained(SMALL_MODEL, dtype=torch.float32),
            LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, bias="none",
                       task_type="CAUSAL_LM",
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {r:>8}{trainable:>14,}{100 * trainable / total:>12.3f}%"
              f"{trainable * 4 / 1e6:>12.1f}")
        del model

    print("\n  The maths: a frozen weight W of shape (d, k) gets a side path B·A with")
    print("  A of shape (r, k) and B of shape (d, r). Instead of d·k numbers you")
    print("  train r·(d + k). At d = k = 896 and r = 8 that is 802 816 against 14 336.")
    print("\n  Two consequences worth remembering:")
    print("   1. the adapter is a few megabytes, so one base model can serve many tasks")
    print("   2. after training you can fold B·A back into W, so inference costs nothing extra")


# ---------------------------------------------------------------------------
# 5. Quantization — measured
# ---------------------------------------------------------------------------

def tensor_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


def quantize_int8(tensor):
    """Symmetric per-row int8: the simplest scheme that actually works."""
    scale = tensor.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    q = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def quantize_demo():
    print("=" * 74)
    print("QUANTIZATION")
    print("=" * 74)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(SMALL_MODEL, dtype=torch.float32)
    n = sum(p.numel() for p in model.parameters())
    print(f"\n  {SMALL_MODEL}: {n:,} parameters")
    print(f"  {'precision':<14}{'bytes/param':>13}{'total':>12}{'vs fp32':>10}")
    print("  " + "-" * 50)
    base = n * 4
    for label, b in BYTES.items():
        print(f"  {label:<14}{b:>13}{n * b / 1e9:>9.2f} GB{base / (n * b):>9.1f}x")

    # What int8 rounding does to one real weight matrix
    layer = model.model.layers[0].self_attn.q_proj.weight.data
    q, scale = quantize_int8(layer)
    restored = q.float() * scale
    error = (restored - layer).abs()
    print(f"\n  one real weight matrix, {tuple(layer.shape)}, rounded to int8:")
    print(f"    original range      {layer.min():+.4f} .. {layer.max():+.4f}")
    print(f"    mean absolute error {error.mean():.6f}")
    print(f"    worst single error  {error.max():.6f}")
    print(f"    relative error      {100 * error.mean() / layer.abs().mean():.2f}%")
    print(f"    distinct values     {layer.unique().numel():,} -> {q.unique().numel()}")

    # Does the model still say sensible things with every attention matrix rounded?
    prompt = "The capital of Kazakhstan is"
    ids = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        before = tokenizer.decode(
            model.generate(**ids, max_new_tokens=12, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)[0][ids["input_ids"].shape[1]:],
            skip_special_tokens=True).strip()

    changed = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and module.weight.shape[0] > 64:
                q, scale = quantize_int8(module.weight.data)
                module.weight.data = q.float() * scale
                changed += 1
    with torch.no_grad():
        after = tokenizer.decode(
            model.generate(**ids, max_new_tokens=12, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)[0][ids["input_ids"].shape[1]:],
            skip_special_tokens=True).strip()

    print(f"\n  rounded {changed} linear layers to int8 and back:")
    print(f"    «{prompt}»")
    print(f"    fp32 : {before}")
    print(f"    int8 : {after}")
    print(f"    identical: {before == after}")
    print("\n  int8 stores the weights in a quarter of the memory. The arithmetic is")
    print("  still done in floating point here — real int8 kernels also make it faster.")


def main():
    parser = argparse.ArgumentParser(description="Large language models, measured")
    for flag in ("tokens", "memory", "sampling", "lora", "quantize", "all"):
        parser.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    chosen = [f for f in ("tokens", "memory", "sampling", "lora", "quantize")
              if getattr(args, f) or args.all]
    if not chosen:
        parser.print_help()
        return
    for name in chosen:
        if name == "tokens":
            tokens_demo(); pricing_demo()
        elif name == "memory":
            memory_demo(); kv_cache_demo()
        elif name == "sampling":
            sampling_demo()
        elif name == "lora":
            lora_demo()
        elif name == "quantize":
            quantize_demo()
        print()


if __name__ == "__main__":
    main()

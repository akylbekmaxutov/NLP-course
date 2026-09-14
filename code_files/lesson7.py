#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP Course — Lecture 7
Pretrained language models: BERT and its family, fine-tuned.

Lecture 6 built a Transformer from scratch and it reached 0.750 on polarity —
below TF-IDF's 0.780. The diagnosis was data: 8 000 labelled reviews cannot
teach a model what Kazakh is *and* what sentiment is at the same time.

This lecture changes one thing. The encoder no longer starts from random
weights: it starts from a model that has already read a great deal of text and
learned the language from it. Everything else — the same balanced split, the
same test set, the same measurement — stays identical, so the difference is
attributable.

Setup:
    pip install -r requirements.txt
    HF_TOKEN=hf_... in a .env file (KazSAnDRA is a gated dataset)

Steps:
    1. data       — the balanced KazSAnDRA splits of Lectures 3-6, or SST-2
    2. tokenizer  — what a subword tokenizer does to Kazakh
    3. model      — a classification head on a pretrained encoder
    4. train      — fine-tune every layer, with a hand-written loop
    5. compare    — several pretrained models under one identical recipe
    6. inspect    — what pretraining alone already knows

    python3 lesson7.py --task polarity          # one model, one task
    python3 lesson7.py --task score
    python3 lesson7.py --dataset sst2           # the same recipe, in English
    python3 lesson7.py --compare                # every model in the table
    python3 lesson7.py --mlm-demo               # fill in the blank, no fine-tuning
    python3 lesson7.py --frozen                 # features only, head trained
    python3 lesson7.py --tokenizer-demo         # how Kazakh gets split up
"""

import argparse
import os
import time

from dotenv import load_dotenv

# Read the .env before importing `datasets`: huggingface_hub reads HF_TOKEN from
# the environment when it is imported, so setting it later is too late.
HERE = os.path.dirname(os.path.abspath(__file__))
for _folder in (HERE, os.path.dirname(HERE)):          # code_files/.env or ../.env
    load_dotenv(os.path.join(_folder, ".env"))

import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import (AutoModelForMaskedLM, AutoModelForSequenceClassification,
                          AutoTokenizer, get_linear_schedule_with_warmup)

DATASET = "issai/kazsandra"
TASKS = {
    "polarity": ("polarity_classification", ["negative", "positive"]),
    "score": ("score_classification", ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]),
}

SEED = 42

# The models compared on the Kazakh tasks. Kept in one place so the table on the
# lecture page and the table this script prints cannot drift apart.
KAZAKH_MODELS = [
    ("mBERT", "bert-base-multilingual-cased"),
    ("DistilmBERT", "distilbert-base-multilingual-cased"),
    ("MiniLM", "microsoft/Multilingual-MiniLM-L12-H384"),
    ("XLM-R", "xlm-roberta-base"),
    ("kaz-RoBERTa", "kz-transformers/kaz-roberta-conversational"),
]

ENGLISH_MODELS = [
    ("BERT", "bert-base-uncased"),
    ("DistilBERT", "distilbert-base-uncased"),
    ("RoBERTa", "roberta-base"),
    ("ALBERT", "albert-base-v2"),
    ("ELECTRA-small", "google/electra-small-discriminator"),
]


def device():
    """Apple Silicon has MPS; everything else falls back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = device()


# ---------------------------------------------------------------------------
# 1. Data — exactly the splits of Lectures 3 to 6, so the numbers compare
# ---------------------------------------------------------------------------

def load_kazsandra(task, split):
    config, _ = TASKS[task]
    cached = os.path.join(HERE, "data", f"{split}_{'pc' if task == 'polarity' else 'sc'}.csv")
    if os.path.exists(cached):
        return pd.read_csv(cached)
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN not found. Accept the terms at\n"
            f"  https://huggingface.co/datasets/{DATASET}\n"
            "then put HF_TOKEN=hf_... in a .env file.")
    return load_dataset(DATASET, config, split=split, token=token).to_pandas()


def balance(df, per_class, seed=SEED):
    """The same balancing as Lecture 3 onwards: equal classes, shuffled, seeded."""
    per_class = min(per_class, df["label"].value_counts().min())
    parts = [g.sample(per_class, random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def kazsandra_split(task, train_per_class, test_per_class):
    train = balance(load_kazsandra(task, "train"), train_per_class)
    test = balance(load_kazsandra(task, "test"), test_per_class)
    columns = ["text_cleaned", "label"]
    return train[columns].rename(columns={"text_cleaned": "text"}), \
        test[columns].rename(columns={"text_cleaned": "text"})


def sst2_split(train_per_class, test_per_class):
    """Stanford Sentiment Treebank v2 — the benchmark BERT was measured on.

    The GLUE test split is unlabelled, so the published convention is to report
    on the validation split. We balance both sides the same way as KazSAnDRA.
    """
    data = load_dataset("stanfordnlp/sst2")
    train = balance(data["train"].to_pandas(), train_per_class)
    test = balance(data["validation"].to_pandas(), test_per_class)
    columns = ["sentence", "label"]
    return train[columns].rename(columns={"sentence": "text"}), \
        test[columns].rename(columns={"sentence": "text"})


def get_data(args):
    if args.dataset == "sst2":
        train, test = sst2_split(args.train_size, args.test_size)
        return train, test, ["negative", "positive"], "SST-2 (English)"
    names = TASKS[args.task][1]
    train, test = kazsandra_split(args.task, args.train_size, args.test_size)
    return train, test, names, f"KazSAnDRA {args.task} (Kazakh)"


# ---------------------------------------------------------------------------
# 2. Tokenizer — the pretrained model brings its own vocabulary
# ---------------------------------------------------------------------------

def encode(texts, labels, tokenizer, max_len):
    batch = tokenizer(list(texts), truncation=True, max_length=max_len,
                      padding="max_length", return_tensors="pt")
    return torch.utils.data.TensorDataset(
        batch["input_ids"],
        batch["attention_mask"],
        torch.tensor(list(labels), dtype=torch.long))


def loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(SEED) if shuffle else None)


def readable(tokenizer, piece):
    """Byte-level BPE stores tokens as mojibake; decode them back for printing."""
    text = tokenizer.convert_tokens_to_string([piece])
    return text.strip() or piece


def tokenizer_demo(name, texts):
    """Show what a sub-word vocabulary does to Kazakh morphology."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    print(f"\n  {name}   vocabulary {tokenizer.vocab_size:,}")
    for text in texts:
        pieces = [readable(tokenizer, p) for p in tokenizer.tokenize(text)]
        print(f"    «{text}»")
        print(f"      {len(pieces):>2} pieces: {' · '.join(pieces)}")


# ---------------------------------------------------------------------------
# 3 + 4. Fine-tuning — a head on a pretrained encoder, then train everything
# ---------------------------------------------------------------------------

def accuracy(model, batches):
    model.eval()
    right = total = 0
    with torch.no_grad():
        for ids, mask, labels in batches:
            ids, mask, labels = ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
            logits = model(input_ids=ids, attention_mask=mask).logits
            right += int((logits.argmax(1) == labels).sum())
            total += len(labels)
    return right / total


def predict(model, batches):
    model.eval()
    out, gold = [], []
    with torch.no_grad():
        for ids, mask, labels in batches:
            logits = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE)).logits
            out.append(logits.argmax(1).cpu())
            gold.append(labels)
    return torch.cat(out).numpy(), torch.cat(gold).numpy()


def fine_tune(name, train_batches, test_batches, classes, args, frozen=False):
    """Load a pretrained encoder, attach a fresh classification head, train."""
    torch.manual_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=classes).to(DEVICE)

    if frozen:
        # Feature extraction: the encoder is a frozen feature function and only
        # the new head learns. This is the comparison that shows why we do not
        # stop here.
        for parameter in model.base_model.parameters():
            parameter.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  {total:,} parameters, {trainable:,} trainable"
          f"{' (encoder frozen)' if frozen else ''}")

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)
    steps = len(train_batches) * args.epochs
    schedule = get_linear_schedule_with_warmup(optimiser, int(0.1 * steps), steps)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        started, running, seen = time.time(), 0.0, 0
        for ids, mask, labels in train_batches:
            ids, mask, labels = ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
            optimiser.zero_grad()
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimiser.step()
            schedule.step()
            running += loss.item() * len(labels)
            seen += len(labels)
        print(f"    epoch {epoch}  loss={running / seen:.4f}"
              f"  test={accuracy(model, test_batches):.3f}"
              f"  ({time.time() - started:.0f}s)")

    return model, accuracy(model, test_batches)


# ---------------------------------------------------------------------------
# 5. What pretraining alone already knows
# ---------------------------------------------------------------------------

def mlm_demo(name, sentences, top_k=5):
    """Fill in the blank with the pretrained model, before any fine-tuning."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForMaskedLM.from_pretrained(name).to(DEVICE).eval()
    print(f"\n  {name}")
    for sentence in sentences:
        text = sentence.replace("[MASK]", tokenizer.mask_token)
        batch = tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**batch).logits
        position = (batch["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0, 0]
        top = logits[0, position].softmax(-1).topk(top_k)
        guesses = ", ".join(
            f"{readable(tokenizer, tokenizer.convert_ids_to_tokens(i))} {p:.2f}"
            for p, i in zip(top.values.tolist(), top.indices.tolist()))
        print(f"    {sentence}")
        print(f"      → {guesses}")


# ---------------------------------------------------------------------------
# 6. Runners
# ---------------------------------------------------------------------------

def run_one(name, label, args):
    train_df, test_df, names, title = get_data(args)
    print("=" * 74, f"\n{label}  ·  {title}\n", "=" * 74, sep="")

    tokenizer = AutoTokenizer.from_pretrained(name)
    train_batches = loader(encode(train_df["text"].astype(str), train_df["label"],
                                  tokenizer, args.max_len), args.batch_size, True)
    test_batches = loader(encode(test_df["text"].astype(str), test_df["label"],
                                 tokenizer, args.max_len), args.batch_size, False)
    print(f"  train {len(train_df)}  test {len(test_df)}"
          f"  vocabulary {tokenizer.vocab_size:,}  device {DEVICE.type}")

    model, score = fine_tune(name, train_batches, test_batches, len(names),
                             args, frozen=args.frozen)

    predicted, gold = predict(model, test_batches)
    print(f"\n  final: {score:.3f}   (random guessing would be {1 / len(names):.3f})\n")
    print(classification_report(gold, predicted, target_names=names,
                                digits=3, zero_division=0))
    print("confusion matrix (rows = true, columns = predicted)")
    print(confusion_matrix(gold, predicted))
    return score


def run_compare(args):
    models = ENGLISH_MODELS if args.dataset == "sst2" else KAZAKH_MODELS
    train_df, test_df, names, title = get_data(args)
    print("=" * 74, f"\nCOMPARING {len(models)} PRETRAINED MODELS  ·  {title}\n",
          "=" * 74, sep="")
    print(f"  train {len(train_df)}  test {len(test_df)}"
          f"  {args.epochs} epochs  lr {args.lr}  device {DEVICE.type}\n")

    results = []
    for label, name in models:
        print(f"— {label}  ({name})")
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            train_batches = loader(encode(train_df["text"].astype(str), train_df["label"],
                                          tokenizer, args.max_len), args.batch_size, True)
            test_batches = loader(encode(test_df["text"].astype(str), test_df["label"],
                                         tokenizer, args.max_len), args.batch_size, False)
            started = time.time()
            model, score = fine_tune(name, train_batches, test_batches,
                                     len(names), args, frozen=args.frozen)
            size = sum(p.numel() for p in model.parameters())
            results.append((label, name, size, score, time.time() - started))
            del model
        except Exception as error:                       # a model may be gated
            print(f"    skipped: {type(error).__name__}: {str(error)[:120]}")
        print()

    print("=" * 74, "\nSUMMARY\n", "=" * 74, sep="")
    print(f"  {'model':<14}{'parameters':>13}{'minutes':>10}{'accuracy':>11}")
    for label, _, size, score, seconds in results:
        print(f"  {label:<14}{size:>13,}{seconds / 60:>10.1f}{score:>11.3f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning pretrained models")
    parser.add_argument("--dataset", choices=["kazsandra", "sst2"], default="kazsandra")
    parser.add_argument("--task", choices=["polarity", "score"], default="polarity")
    parser.add_argument("--model", default="bert-base-multilingual-cased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train-size", type=int, default=4000, help="rows per class")
    parser.add_argument("--test-size", type=int, default=1000, help="rows per class")
    parser.add_argument("--frozen", action="store_true",
                        help="freeze the encoder and train only the head")
    parser.add_argument("--compare", action="store_true",
                        help="fine-tune every model in the table")
    parser.add_argument("--mlm-demo", action="store_true")
    parser.add_argument("--tokenizer-demo", action="store_true")
    args = parser.parse_args()

    if args.tokenizer_demo:
        samples = ["қосымша өте жақсы емес", "кітаптарымыздан", "Астана қаласы әдемі"]
        for _, name in KAZAKH_MODELS:
            try:
                tokenizer_demo(name, samples)
            except Exception as error:
                print(f"  {name}: {type(error).__name__}")
        return

    if args.mlm_demo:
        sentences = [
            "Қазақстанның астанасы [MASK] қаласы.",
            "Бұл қосымша маған қатты [MASK].",
            "Мен кітап [MASK].",
        ]
        for name in ("bert-base-multilingual-cased", "xlm-roberta-base",
                     "kz-transformers/kaz-roberta-conversational"):
            try:
                mlm_demo(name, sentences)
            except Exception as error:
                print(f"  {name}: {type(error).__name__}: {str(error)[:100]}")
        return

    if args.compare:
        run_compare(args)
        return

    short_name = {name: label for label, name in KAZAKH_MODELS + ENGLISH_MODELS}
    run_one(args.model, short_name.get(args.model, args.model), args)


if __name__ == "__main__":
    main()

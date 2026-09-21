"""One thin client for the locally served model. Everything else imports this.

The server is llama.cpp's `llama-server`, which speaks the OpenAI chat API.
That is the point: the code below would talk to vLLM, to Ollama, or to a
commercial endpoint with nothing changed but the base URL and the key.

    ./serve.sh                    # terminal 1
    python3 prompting.py --all    # terminal 2
"""
import os
import time

from openai import OpenAI

BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen3.5-4b")

# A local server ignores the key, but the client library insists on one.
client = OpenAI(base_url=BASE_URL, api_key="no-key-needed")

# Sampling presets published on the Qwen3.5 model card. They differ by mode,
# which is worth noticing: the same weights want different knobs depending on
# whether they are thinking or answering directly.
PRESETS = {
    "thinking": dict(temperature=1.0, top_p=0.95, presence_penalty=1.5),
    "instruct": dict(temperature=0.7, top_p=0.8, presence_penalty=1.5),
}


class Reply:
    """What came back: the answer, the thinking, and what it cost."""

    def __init__(self, text, thinking, usage, seconds):
        self.text = text
        self.thinking = thinking or ""
        self.usage = usage
        self.seconds = seconds

    @property
    def prompt_tokens(self):
        return getattr(self.usage, "prompt_tokens", 0) or 0

    @property
    def output_tokens(self):
        return getattr(self.usage, "completion_tokens", 0) or 0

    @property
    def thinking_tokens(self):
        """Rough, but comparable across runs: thinking is billed as output."""
        if not self.thinking:
            return 0
        return round(self.output_tokens * len(self.thinking) /
                     max(1, len(self.thinking) + len(self.text)))

    def __str__(self):
        return self.text


def ask(prompt, system=None, think=True, max_tokens=1024, seed=None, **sampling):
    """One turn. `think=False` switches the model out of reasoning mode."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, think=think, max_tokens=max_tokens, seed=seed, **sampling)


def chat(messages, think=True, max_tokens=1024, seed=None, **sampling):
    """A whole conversation. Returns a Reply."""
    params = dict(PRESETS["thinking" if think else "instruct"])
    params.update(sampling)

    # Both of these travel outside the OpenAI schema. `enable_thinking` is read
    # by the chat template itself — it is the switch that decides whether the
    # prompt ends with an open <think> block or with a closed, empty one.
    extra = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": think}}
    if seed is not None:
        extra["seed"] = seed

    started = time.time()
    r = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        extra_body=extra,
        **params,
    )
    seconds = time.time() - started

    msg = r.choices[0].message
    text = (msg.content or "").strip()
    thinking = getattr(msg, "reasoning_content", None) or ""

    # Some builds return the reasoning inline instead of in its own field.
    if not thinking and text.startswith("<think>"):
        end = text.find("</think>")
        if end != -1:
            thinking = text[len("<think>"):end].strip()
            text = text[end + len("</think>"):].strip()

    return Reply(text, thinking, r.usage, seconds)


def stream(messages, think=True, max_tokens=1024, seed=None, **sampling):
    """Same call, yielded piece by piece. Yields ("thinking"|"answer", text)."""
    params = dict(PRESETS["thinking" if think else "instruct"])
    params.update(sampling)
    extra = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": think}}
    if seed is not None:
        extra["seed"] = seed

    events = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=max_tokens,
        stream=True, extra_body=extra, **params)

    inside_think = False
    for event in events:
        if not event.choices:
            continue
        delta = event.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            yield "thinking", reasoning
        piece = delta.content or ""
        if not piece:
            continue
        # Builds that do not split reasoning out send it inline instead.
        while piece:
            if not inside_think:
                start = piece.find("<think>")
                if start == -1:
                    yield "answer", piece
                    break
                yield "answer", piece[:start]
                piece = piece[start + len("<think>"):]
                inside_think = True
            else:
                end = piece.find("</think>")
                if end == -1:
                    yield "thinking", piece
                    break
                yield "thinking", piece[:end]
                piece = piece[end + len("</think>"):]
                inside_think = False


def server_is_up():
    try:
        client.models.list()
        return True
    except Exception:
        return False


def require_server():
    if not server_is_up():
        raise SystemExit(
            "No model server at %s.\n"
            "Start one in another terminal:  ./serve.sh" % BASE_URL)

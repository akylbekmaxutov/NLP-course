# Lecture 9 — Prompt engineering and RAG

Everything runs on a laptop. No API key, no GPU rental, nothing leaves the
machine: a 4-billion-parameter model in 4-bit, served locally, and a retrieval
index built from this course's own lecture pages.

## Install

```bash
brew install llama.cpp                 # macOS, builds with Metal
pip install -r requirements.txt
```

## Run — two terminals

```bash
./serve.sh                             # 1. the model server, port 8080
python3 app.py                         # 2. the lab,          port 7860
```

`serve.sh` downloads `unsloth/Qwen3.5-4B-GGUF` (Q4_K_M, 2.7 GB) the first time
and starts `llama-server`, which exposes an **OpenAI-compatible API** at
`http://127.0.0.1:8080/v1`. Everything else here is an ordinary OpenAI client
pointed at that address — the same code works against vLLM, Ollama, or a
commercial endpoint with one line changed:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="no-key-needed")

client.chat.completions.create(
    model="qwen3.5-4b",
    messages=[{"role": "user", "content": "..."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

## The lab — http://127.0.0.1:7860

| tab | what it is for |
|---|---|
| **Chat** | one conversation, every knob exposed, the model's thinking shown in a panel |
| **Techniques** | nine prompt-engineering techniques, each sent both the naive way and the engineered way, side by side, with tokens and seconds |
| **RAG** | the same question with retrieval on and off, showing the passages that were fetched |

The Techniques tab is the teaching surface. Pick a technique, press **Run
both**, and the two answers appear next to each other. Every prompt is editable
in the page, so a question from the room can be answered by changing the prompt
and pressing the button again.

1. Reasoning on / off — `enable_thinking`, and what it costs
2. Role and output format
3. Few-shot examples
4. Structured output (JSON)
5. Chain of thought written into the prompt
6. Say what to do, not what to avoid
7. Language and length control
8. Delimiters and prompt injection
9. Self-consistency (sample N, take the vote)

## The reasoning switch

Qwen3.5 thinks by default. The switch is not a sampling parameter — it is an
argument to the **chat template**, which decides whether the prompt ends with
an open `<think>` block:

```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

`serve.sh` passes `--jinja` for exactly this reason: without it, llama-server
uses a built-in template and the flag is silently ignored.

Give reasoning room. A thinking reply that runs out of `max_tokens` does not
come back short — it comes back **empty**, because generation stopped while
still inside the `<think>` block. 4096 is a sane default here; 2048 is not.

## RAG

```bash
python3 corpus.py                      # cut the course pages into passages
python3 retrieve.py --build            # embed them once (~1 min)
python3 retrieve.py --compare          # bm25 vs dense vs hybrid
python3 rag.py --ask "your question"   # one question, all the way through
```

The knowledge base is this course's own HTML, in all three languages — 1 924
passages. That is deliberate: the numbers in those pages were measured in
Lectures 3–8 and appear nowhere else, so the model cannot have memorised them.
An ungrounded answer is provably invented, and you can show it in the app by
unticking **Retrieve passages**.

Retrieval is three-way: BM25 (Lecture 3's bag of words, better weighted), dense
embeddings (`multilingual-e5-small`), and reciprocal-rank fusion of the two.

## The files

| file | what it is |
|---|---|
| `serve.sh` | starts `llama-server` with Qwen3.5-4B (Q4_K_M) |
| `llm.py` | the OpenAI client every other file imports: `ask`, `chat`, `stream` |
| `techniques.py` | the nine techniques, as editable prompt pairs |
| `app.py` | the Gradio lab |
| `corpus.py` | cuts the course's HTML pages into retrievable passages |
| `retrieve.py` | BM25, dense embeddings, hybrid fusion |
| `rag.py` | retrieval-augmented answering over the passages |
| `tasks.py` | graded task sets, shared with the optional batch script |
| `prompting.py` | optional: runs the techniques as a batch and scores them |

`prompting.py` is not needed for the lecture. It exists so a claim made in the
room can be checked over hundreds of examples instead of one, and it takes
tens of minutes on this hardware.

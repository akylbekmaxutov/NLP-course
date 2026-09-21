"""The Lecture 9 lab: Gradio in front of the locally served model.

    ./serve.sh                # terminal 1 — llama-server on :8080
    python3 app.py            # terminal 2 — http://127.0.0.1:7860

Three tabs:
  Chat        one conversation, every knob exposed, the thinking visible
  Techniques  a prompt-engineering technique, run both ways, side by side
  RAG         the same question with and without the course material attached
"""
import argparse
import re
from collections import Counter

import gradio as gr

import llm
import techniques

DEFAULT_SYSTEM = "You are a helpful assistant. Answer in the user's language."

CHAT_EXAMPLES = [
    "Астана мен Алматының арасы қанша шақырым?",
    "A shop lists a book at 4 500 tenge, takes 15% off, then adds 12% VAT. "
    "What is the final price?",
    "Объясни простыми словами, что такое attention в трансформерах.",
    "Write a one-sentence product review in Kazakh and label its sentiment.",
]

CSS = """
.thinking textarea { font-size: 13px; color: #555; background: #f8f9fa; }
.note { background: #f8f9fa; border-left: 3px solid #2563eb; padding: 10px 14px; }
footer { display: none !important; }
"""


def stat_line(reply, cap):
    return ("`%d` prompt tokens · `%d` output tokens (`%d` thinking) · `%.1f` s%s"
            % (reply.prompt_tokens, reply.output_tokens, reply.thinking_tokens,
               reply.seconds,
               "  ⚠️ hit max_tokens" if reply.output_tokens >= cap else ""))


# --------------------------------------------------------------------------
# Tab 1 — chat
# --------------------------------------------------------------------------
def respond(message, history, system, think, temperature, top_p, max_tokens):
    messages = [{"role": "system", "content": system}] if system.strip() else []
    messages += [{"role": t["role"], "content": t["content"]} for t in history]
    messages.append({"role": "user", "content": message})

    history = history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": ""}]
    thinking, answer = "", ""
    try:
        for kind, piece in llm.stream(messages, think=think,
                                      max_tokens=int(max_tokens),
                                      temperature=temperature, top_p=top_p):
            if kind == "thinking":
                thinking += piece
            else:
                answer += piece
            history[-1]["content"] = answer or "_thinking…_"
            yield history, "", thinking, "generating…"
    except Exception as exc:
        history[-1]["content"] = "Error talking to the server: %s" % exc
        yield history, "", thinking, "no server — run ./serve.sh"
        return

    history[-1]["content"] = answer
    yield history, "", thinking, ("thinking %d chars · answer %d chars"
                                  % (len(thinking), len(answer)))


def chat_tab():
    with gr.Tab("Chat"):
        gr.Markdown(
            "One conversation with the local model. **Reasoning** is the "
            "`enable_thinking` flag in the chat template: with it on, the "
            "model writes a private scratchpad before answering."
        )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=440, label="Conversation")
                box = gr.Textbox(placeholder="Ask something…", lines=2,
                                 label="Message", submit_btn=True)
                stats = gr.Markdown("")
                with gr.Accordion("Thinking (what users never see)", open=False):
                    thinking = gr.Textbox(lines=10, label="",
                                          elem_classes="thinking")
            with gr.Column(scale=2):
                system = gr.Textbox(DEFAULT_SYSTEM, lines=6, label="System prompt")
                think = gr.Checkbox(True, label="Reasoning (enable_thinking)")
                temperature = gr.Slider(0.0, 1.5, 0.7, step=0.05, label="temperature")
                top_p = gr.Slider(0.1, 1.0, 0.8, step=0.05, label="top_p")
                # Generous by default: a reasoning reply that runs out of
                # budget is not a short answer, it is no answer at all.
                max_tokens = gr.Slider(64, 8192, 4096, step=64, label="max_tokens")
                clear = gr.Button("Clear conversation")
                gr.Examples(CHAT_EXAMPLES, inputs=box, label="Try one")

        box.submit(respond,
                   [box, chatbot, system, think, temperature, top_p, max_tokens],
                   [chatbot, box, thinking, stats])
        clear.click(lambda: ([], "", ""), None, [chatbot, thinking, stats])


# --------------------------------------------------------------------------
# Tab 2 — techniques, run both ways
# --------------------------------------------------------------------------
ANSWER_LINE = re.compile(r"ANSWER\s*[:\-]\s*(.+)", re.I)


def load_technique(name):
    t = techniques.get(name)
    return (gr.Markdown(t["note"], elem_classes="note"),
            t["task"],
            t["system_a"] or "", t["user_a"], t["think_a"], "### A — %s" % t["label_a"],
            t["system_b"] or "", t["user_b"], t["think_b"], "### B — %s" % t["label_b"],
            5 if t["vote"] else 1)


def run_pair(task, system_a, user_a, think_a, system_b, user_b, think_b,
             temperature, max_tokens, samples):
    cap = int(max_tokens)
    outputs = []
    for system, user, think, votes in (
            (system_a, user_a, think_a, 1),
            (system_b, user_b, think_b, int(samples))):
        prompt = user.replace("{task}", task)
        try:
            replies = [llm.ask(prompt, system=system or None, think=think,
                               max_tokens=cap, temperature=temperature,
                               seed=100 + i)
                       for i in range(votes)]
        except Exception as exc:
            outputs += ["Error talking to the server: %s" % exc, ""]
            continue

        if votes > 1:
            picked = [(ANSWER_LINE.findall(r.text) or [r.text.strip()])[-1].strip()
                      for r in replies]
            winner, count = Counter(picked).most_common(1)[0]
            body = ("**Vote over %d samples → `%s`** (%d/%d agreed)\n\n"
                    "samples: %s\n\n---\n\n%s"
                    % (votes, winner, count, votes,
                       ", ".join("`%s`" % p for p in picked), replies[0].text))
            note = ("`%d` calls · `%d` output tokens total · `%.1f` s"
                    % (votes, sum(r.output_tokens for r in replies),
                       sum(r.seconds for r in replies)))
        else:
            r = replies[0]
            body = r.text or ("_(empty — generation stopped inside the thinking "
                              "block; raise max_tokens)_")
            note = stat_line(r, cap)
        outputs += [body, note]
    return outputs[0], outputs[1], outputs[2], outputs[3]


def techniques_tab():
    with gr.Tab("Techniques"):
        gr.Markdown(
            "Pick a technique. The same task is sent two ways — the way people "
            "write it first, and the way it is written after. Edit either side "
            "and run again; `{task}` is replaced by the box below."
        )
        picker = gr.Dropdown(list(techniques.TECHNIQUES), value=techniques.DEFAULT,
                             label="Technique")
        note = gr.Markdown(elem_classes="note")
        task = gr.Textbox(lines=3, label="Task / input  (fills {task})")
        with gr.Row():
            temperature = gr.Slider(0.0, 1.5, 0.7, step=0.05, label="temperature")
            max_tokens = gr.Slider(64, 8192, 4096, step=64, label="max_tokens")
            samples = gr.Slider(1, 9, 1, step=2, label="samples on side B (vote)")
        run = gr.Button("Run both", variant="primary")
        with gr.Row():
            with gr.Column():
                head_a = gr.Markdown("### A")
                system_a = gr.Textbox(lines=5, label="System prompt A")
                user_a = gr.Textbox(lines=3, label="User prompt A")
                think_a = gr.Checkbox(label="Reasoning A")
                out_a = gr.Markdown()
                note_a = gr.Markdown()
            with gr.Column():
                head_b = gr.Markdown("### B")
                system_b = gr.Textbox(lines=5, label="System prompt B")
                user_b = gr.Textbox(lines=3, label="User prompt B")
                think_b = gr.Checkbox(label="Reasoning B")
                out_b = gr.Markdown()
                note_b = gr.Markdown()

        fields = [note, task, system_a, user_a, think_a, head_a,
                  system_b, user_b, think_b, head_b, samples]
        picker.change(load_technique, picker, fields)
        run.click(run_pair,
                  [task, system_a, user_a, think_a, system_b, user_b, think_b,
                   temperature, max_tokens, samples],
                  [out_a, note_a, out_b, note_b])
        return picker, fields


# --------------------------------------------------------------------------
# Tab 3 — RAG
# --------------------------------------------------------------------------
_retriever = None


def retriever():
    global _retriever
    if _retriever is None:
        import retrieve
        _retriever = retrieve.Retriever()
    return _retriever


def rag_answer(question, use_rag, method, k, think):
    import rag
    import retrieve

    if not use_rag:
        reply = rag.closed_book(question, think=think)
        return (reply.text, "_retrieval is off — the model is answering from "
                            "memory alone_", stat_line(reply, 400))

    hits = retriever().search(question, k=int(k), method=method)
    shown = ["**[%d]** `%.4f` · Lecture %d · %s · `%s`\n\n> %s"
             % (n, score, c["lecture"], c["heading"], c["source"],
                c["text"][:400].replace("\n", " "))
             for n, (c, score) in enumerate(hits, 1)]
    prompt = rag.TEMPLATE % (retrieve.format_context(hits), question)
    reply = llm.ask(prompt, system=rag.SYSTEM, think=think, max_tokens=400,
                    temperature=0.3)
    return reply.text, "\n\n---\n\n".join(shown), stat_line(reply, 400)


RAG_EXAMPLES = [
    "What accuracy did mBERT reach on the KazSAnDRA polarity task?",
    "How many parameters does LoRA train at rank 4 on a 0.5B model?",
    "What relative error did rounding one weight matrix to int8 cause?",
    "Attention дегеніміз не?",
    "Чем BERT отличается от GPT?",
    "How many students are enrolled in this course?",
]


def rag_tab():
    with gr.Tab("RAG"):
        gr.Markdown(
            "The knowledge base is **this course's own pages**, in all three "
            "languages — 1 924 passages. The numbers in them were measured in "
            "Lectures 3–8 and exist nowhere else, so an ungrounded answer is "
            "provably invented. Untick **Retrieve passages** and ask again to "
            "watch it happen. The last example is not in the material at all: "
            "the only correct answer is a refusal."
        )
        question = gr.Textbox(RAG_EXAMPLES[0], lines=2, label="Question")
        with gr.Row():
            use_rag = gr.Checkbox(True, label="Retrieve passages")
            method = gr.Radio(["hybrid", "bm25", "dense"], value="hybrid",
                              label="Retriever")
            k = gr.Slider(1, 10, 5, step=1, label="passages (k)")
            think = gr.Checkbox(False, label="Reasoning")
        run = gr.Button("Answer", variant="primary")
        answer = gr.Markdown()
        note = gr.Markdown()
        with gr.Accordion("Retrieved passages", open=True):
            passages = gr.Markdown()
        gr.Examples(RAG_EXAMPLES, inputs=question, label="Try one")
        run.click(rag_answer, [question, use_rag, method, k, think],
                  [answer, passages, note])


def build():
    with gr.Blocks(title="Lecture 9 — Prompting and RAG") as demo:
        gr.Markdown("# Lecture 9 — Prompt engineering and RAG\n"
                    "Qwen3.5-4B, 4-bit, served locally by llama.cpp and called "
                    "through the OpenAI client. Nothing leaves this machine.")
        chat_tab()
        picker, fields = techniques_tab()
        rag_tab()
        # Fill the Techniques tab with the first technique on page load.
        demo.load(load_technique, picker, fields)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    if not llm.server_is_up():
        print("Warning: no model server at %s — start ./serve.sh first."
              % llm.BASE_URL)
    build().launch(server_name="127.0.0.1", server_port=args.port,
                   share=args.share, css=CSS, theme=gr.themes.Soft())

"""The agent, in the browser, with its reasoning shown.

    python3 app.py        # http://127.0.0.1:7861

The point of the interface is the left-hand column: every tool call, its
arguments, and what came back. An agent whose steps you cannot see is an agent
you cannot debug or trust.
"""
import argparse
import json

import gradio as gr

import agent
import llm
import tools

EXAMPLES = [
    "Астанадан Алматыға дейін қанша шақырым және Алматыда ауа райы қандай?",
    "I have 500 USD for three days in Shymkent. What is that per day in tenge?",
    "Что такое Чарынский каньон и далеко ли он от Алматы?",
    "Plan two days in Almaty: what is it known for, and what should I pack?",
    "What will the weather be in Astana on 1 March next year?",
    "Compare Almaty and Shymkent: population, elevation, and the weather this week.",
]

CSS = """
.trace { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
.trace .ok { border-left: 3px solid #15803d; padding-left: 10px; margin-bottom: 10px; }
.trace .bad { border-left: 3px solid #b91c1c; padding-left: 10px; margin-bottom: 10px; }
footer { display: none !important; }
"""


def render(step):
    body = json.dumps(step.result, ensure_ascii=False, indent=1)
    if len(body) > 700:
        body = body[:700] + "\n …"
    args = ", ".join("%s=%r" % kv for kv in step.arguments.items())
    return ('<div class="%s"><b>[%d] %s</b>(%s)<br><pre style="margin:4px 0">%s</pre></div>'
            % ("bad" if step.failed else "ok", step.n, step.name, args,
               body.replace("&", "&amp;").replace("<", "&lt;")))


def ask(question, max_steps, temperature):
    if not question.strip():
        yield "", "", ""
        return

    collected = []
    yield "", '<div class="trace"><i>thinking…</i></div>', "_running_"

    # The loop is synchronous, so collect the trace through the callback and
    # show it when the run finishes. (Streaming it live would need threads;
    # the lecture keeps the loop readable instead.)
    try:
        result = agent.run(question, max_steps=int(max_steps),
                           temperature=temperature,
                           on_event=collected.append)
    except Exception as exc:
        yield ("", '<div class="trace bad">Could not reach the model.<br>%s</div>'
               % str(exc).replace("<", "&lt;"),
               "**No backend.** Put `OPENAI_API_KEY=...` in the repository's "
               "`.env`, or start Lecture 9's `./serve.sh`.")
        return

    trace = ('<div class="trace">%s</div>'
             % ("".join(render(s) for s in result.steps) or "<i>no tools were used</i>"))
    stats = ("`%d` tool calls · `%d` prompt tokens · `%d` output tokens · "
             "`%.1f` s · %s%s"
             % (len(result.steps), result.prompt_tokens, result.output_tokens,
                result.seconds, result.backend,
                "  ⚠️ **hit the step limit**" if result.stopped_early else ""))
    if len(result.per_call) > 1:
        stats += ("\n\nPrompt tokens per turn: %s — the whole history is resent "
                  "every time, which is why agents get expensive."
                  % " → ".join(str(n) for n in result.per_call))
    yield result.answer, trace, stats


def build():
    with gr.Blocks(title="Lecture 10 — AI Agents") as demo:
        gr.Markdown(
            "# Lecture 10 — an AI agent\n"
            "A travel assistant for Kazakhstan. It knows nothing: every "
            "distance, forecast, exchange rate and description is fetched with "
            "a tool while you watch. **Backend: %s**" % llm.describe())
        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(EXAMPLES[0], lines=2, label="Ask something",
                                      submit_btn=True)
                answer = gr.Markdown(label="Answer")
                stats = gr.Markdown()
                gr.Examples(EXAMPLES, inputs=question, label="Try one")
            with gr.Column(scale=2):
                gr.Markdown("### What the agent did")
                trace = gr.HTML()
                with gr.Accordion("Settings", open=False):
                    max_steps = gr.Slider(1, 12, agent.MAX_STEPS, step=1,
                                          label="max steps")
                    temperature = gr.Slider(0.0, 1.0, 0.2, step=0.05,
                                            label="temperature")
                with gr.Accordion("The six tools", open=False):
                    gr.Markdown("\n".join(
                        "- **%s** — %s" % (n, t["schema"]["description"])
                        for n, t in tools.TOOLS.items()))
        question.submit(ask, [question, max_steps, temperature],
                        [answer, trace, stats])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()
    print("backend:", llm.describe())
    build().launch(server_name="127.0.0.1", server_port=args.port, css=CSS,
                   theme=gr.themes.Soft())

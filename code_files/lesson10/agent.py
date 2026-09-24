"""The agent loop.

This is the whole idea, and it is about twenty lines:

    while True:
        ask the model, handing it the list of tools
        if it replied with text        -> done
        if it asked to call tools      -> call them, append the results, repeat

Everything else in this file is bookkeeping: a trace so you can see what it
did, a step limit so it cannot spin forever, and token accounting so you can
watch the cost grow.

What makes this an agent rather than a pipeline: nobody wrote down which tools
to call, in what order, or how many times. The model decides, and the decision
changes with the question.

    python3 agent.py "How far is Astana from Almaty, and what is the weather there?"
    python3 agent.py --demo
"""
import argparse
import json
import sys
import time

import llm
import tools

SYSTEM = """You are a travel planning assistant for trips in Kazakhstan.

You know nothing reliable about specific places, prices, distances or weather.
Look everything up with the tools. Never state a number you did not get from a
tool, and never describe a place from memory.

Rules:
- Use calculate for every sum, total or per-day figure. Do not do arithmetic
  in your head.
- Use wikipedia before describing what somewhere is known for, and mention the
  article you used.
- Weather is only available about two weeks ahead. If someone asks about a
  date further away, say so rather than inventing a forecast.
- Currency rates move; when you quote money, say the date of the rate.
- When you have enough, answer in the user's language, briefly, in plain words.
"""

MAX_STEPS = 8


class Step:
    """One tool call and what came back, for the trace."""

    def __init__(self, n, name, arguments, result, seconds):
        self.n, self.name, self.arguments = n, name, arguments
        self.result, self.seconds = result, seconds

    @property
    def failed(self):
        return isinstance(self.result, dict) and "error" in self.result

    def __str__(self):
        args = ", ".join("%s=%r" % kv for kv in self.arguments.items())
        body = json.dumps(self.result, ensure_ascii=False)
        if len(body) > 160:
            body = body[:157] + "..."
        return "[%d] %s(%s)\n    -> %s" % (self.n, self.name, args, body)


class Result:
    def __init__(self):
        self.answer = ""
        self.steps = []
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.per_call = []          # prompt tokens at each turn, to show growth
        self.seconds = 0.0
        self.stopped_early = False
        self.backend = ""


def run(question, max_steps=MAX_STEPS, temperature=0.2, on_event=None,
        system=SYSTEM):
    """Run the agent to an answer. `on_event` is called with each Step."""
    client, model, label = llm.backend()
    out = Result()
    out.backend = label

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": question}]
    started = time.time()

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools.openai_tools(),
            tool_choice="auto",
            temperature=temperature,
        )
        usage = response.usage
        if usage:
            out.prompt_tokens += usage.prompt_tokens or 0
            out.output_tokens += usage.completion_tokens or 0
            out.per_call.append(usage.prompt_tokens or 0)

        message = response.choices[0].message
        calls = message.tool_calls or []

        # The assistant turn has to go back verbatim, tool calls and all,
        # or the next request will not line up with the tool replies.
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name,
                                         "arguments": c.function.arguments}}
                           for c in calls] or None,
        })
        if not calls:
            messages[-1].pop("tool_calls")
            out.answer = (message.content or "").strip()
            break

        for call in calls:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
                result = {"error": "arguments were not valid JSON: %r"
                                   % call.function.arguments}
            else:
                began = time.time()
                result = tools.call(name, arguments)
            seconds = result.pop("_seconds", 0) if isinstance(result, dict) else 0

            record = Step(len(out.steps) + 1, name, arguments, result, seconds)
            out.steps.append(record)
            if on_event:
                on_event(record)

            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, ensure_ascii=False)})
    else:
        out.stopped_early = True
        out.answer = ("I ran out of steps (%d) before finishing. Here is what I "
                      "found so far:\n\n%s"
                      % (max_steps, "\n".join(str(s) for s in out.steps)))

    out.seconds = time.time() - started
    return out


# --------------------------------------------------------------------------
DEMO = [
    "Астанадан Алматыға дейін қанша шақырым және Алматыда ауа райы қандай?",
    "I have 500 USD for three days in Shymkent. What is that per day in tenge?",
    "Что такое Чарынский каньон и далеко ли он от Алматы?",
    "Plan two days in Almaty: what is it known for, and what should I pack?",
    "What will the weather be in Astana on 1 March next year?",
]


def show(result, question):
    print("\n" + "=" * 74)
    print("Q: %s" % question)
    print("=" * 74)
    for step in result.steps:
        print(step)
    print("\nANSWER\n%s" % result.answer)
    print("\n%d tool calls, %d prompt tokens, %d output tokens, %.1f s%s"
          % (len(result.steps), result.prompt_tokens, result.output_tokens,
             result.seconds, "  (STOPPED EARLY)" if result.stopped_early else ""))
    if len(result.per_call) > 1:
        print("prompt tokens per turn: %s  <- the history is resent every time"
              % " -> ".join(str(n) for n in result.per_call))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", nargs="*")
    ap.add_argument("--demo", action="store_true", help="run the example questions")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = ap.parse_args()

    print("backend:", llm.describe())
    questions = DEMO if args.demo else [" ".join(args.question)]
    if not questions[0]:
        ap.error("give a question, or --demo")
    for q in questions:
        show(run(q, max_steps=args.max_steps), q)


if __name__ == "__main__":
    main()

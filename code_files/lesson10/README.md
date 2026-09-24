# Lecture 10 — AI Agents

A travel assistant for Kazakhstan that **knows nothing**. Every distance,
forecast, exchange rate and description in its answers was fetched with a tool
while you watched. That is deliberate twice over: it is what an agent is, and
it means the lecture states no fact about Kazakhstan that it cannot source.

## What makes this an agent

RAG, in Lecture 9, was a fixed pipeline: always retrieve once, then answer. An
agent decides. Nobody wrote down which tools to call, in what order, or how
many times — the model chooses, and the choice changes with the question.

The whole idea is about twenty lines of `agent.py`:

```python
while True:
    ask the model, handing it the list of tools
    if it replied with text   -> done
    if it asked for tools     -> call them, append the results, repeat
```

## Run it

Pick a backend. **OpenAI:** put your key in the repository's `.env`:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini      # optional
```

**Or, with no key at all**, reuse Lecture 9's local server:

```bash
cd ../lesson9 && ./serve.sh   # terminal 1
```

Then:

```bash
pip install -r requirements.txt
python3 agent.py --demo                       # the example questions, in the terminal
python3 agent.py "How far is Turkistan from Shymkent?"
python3 app.py                                # http://127.0.0.1:7861
```

`llm.py` picks the backend: OpenAI if a key is set, the local server otherwise.
The agent code never finds out which it got.

## The six tools

| tool | what it does | source |
|---|---|---|
| `geocode` | coordinates, country, population, elevation, timezone | Open-Meteo |
| `distance_km` | great-circle distance between two places | computed from the above |
| `weather_forecast` | daily high, low and rainfall, up to 16 days | Open-Meteo |
| `wikipedia` | a short summary and its URL, in en / kk / ru | Wikipedia |
| `convert_currency` | today's rate, with the date it was published | open.er-api.com |
| `calculate` | arithmetic, parsed rather than `eval`-ed | standard library |

None of them needs an API key.

## The files

| file | what it is |
|---|---|
| `tools.py` | the six tools, their JSON schemas, and the dispatcher |
| `agent.py` | the loop, the trace, the step limit, the token accounting |
| `llm.py` | backend selection and a five-line `.env` reader |
| `app.py` | the browser interface, with the trace shown beside the answer |

## Things the lecture is built around

- **A tool description is a prompt.** `geocode`'s description is the only thing
  telling the model to look a city up instead of guessing its population.
- **Never `eval` model output.** `calculate` walks the syntax tree by hand.
  Asked to run `__import__('os').system('echo pwned')` it returns
  `only + - * / // % ** and numbers are allowed`.
- **Tool errors are data, not exceptions.** A wrong tool name, wrong arguments
  or unparseable JSON all come back to the model as `{"error": ...}` so it can
  correct itself. A tool that raises kills the loop.
- **The history is resent every turn.** A three-step run here goes
  400 → 650 → 900 prompt tokens. Cost grows with the square of the steps,
  which is the single most surprising thing about running agents.
- **Always cap the steps.** `MAX_STEPS = 8`. Without it, a model that keeps
  calling the same tool bills you until you notice.
- **An agent that cannot show its work cannot be trusted.** Hence the trace.

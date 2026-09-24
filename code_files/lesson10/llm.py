"""Which model the agent talks to.

Two backends, one code path, because both speak the OpenAI chat API:

  OpenAI       put OPENAI_API_KEY in the repository's .env file
  local        Lecture 9's llama-server, if no key is set

The agent code below does not know or care which one it got. That is worth
noticing: "an agent" is not a product you buy, it is a loop you write around
whatever model answers.
"""
import os

from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))

# Looked at in order; the first file to define a key wins. Three places
# because people reasonably put .env next to the code, next to the lesson
# folders, or at the top of the repository.
ENV_FILES = [
    os.path.join(HERE, ".env"),
    os.path.abspath(os.path.join(HERE, "..", ".env")),
    os.path.abspath(os.path.join(HERE, "..", "..", ".env")),
]


def load_env(paths=ENV_FILES):
    """A small .env reader, so the folder needs no extra dependency."""
    found = []
    for path in paths:
        if not os.path.exists(path):
            continue
        found.append(path)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                # setdefault: a real environment variable beats the file, and
                # the first file that mentions a key beats later ones.
                os.environ.setdefault(key.strip(),
                                      value.strip().strip('"').strip("'"))
    return found


ENV_LOADED = load_env()

LOCAL_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8080/v1")


def backend():
    """Return (client, model_name, label)."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAI(api_key=key), model, "OpenAI %s" % model
    model = os.environ.get("LLM_MODEL", "qwen3.5-4b")
    return (OpenAI(base_url=LOCAL_URL, api_key="no-key-needed"), model,
            "local %s at %s" % (model, LOCAL_URL))


def describe():
    try:
        _, _, label = backend()
        return label
    except Exception as exc:
        return "no backend (%s)" % exc

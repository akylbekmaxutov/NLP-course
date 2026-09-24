"""The tools the agent can call.

A tool is three things: a Python function, a JSON schema describing it, and a
sentence telling the model when to reach for it. The third one is a prompt, and
it decides whether the agent works.

Nothing here is invented. Every fact comes back from a public API at call time,
which is the point: the agent does not know anything about Kazakhstan, it looks
things up. That also means every answer can be traced to a source.
"""
import ast
import json
import math
import operator
import time
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "NLP-Course-Lecture10/1.0 (teaching example)"
TIMEOUT = 20


class ToolError(Exception):
    """Raised when a tool cannot answer. The agent sees the message and retries."""


def _get(url, params=None, attempts=3):
    """One GET, retried on the transient failures every public API has."""
    if params:
        url += "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except Exception as exc:
            last = exc
        time.sleep(0.6 * (attempt + 1))
    raise ToolError("could not reach %s (%s)" % (url.split("?")[0], last))


# --------------------------------------------------------------------------
# 1. geocode
# --------------------------------------------------------------------------
# Kazakh and Russian letters to the Latin the gazetteer indexes. The geocoding
# API finds "Алматы" when asked in Russian but not "Түркістан" in any language,
# so a transliterated second attempt is what makes Kazakh place names work.
_TRANSLIT = {
    "а": "a", "ә": "a", "б": "b", "в": "v", "г": "g", "ғ": "g", "д": "d",
    "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k",
    "қ": "q", "л": "l", "м": "m", "н": "n", "ң": "n", "о": "o", "ө": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ұ": "u", "ү": "u",
    "ф": "f", "х": "kh", "һ": "h", "ц": "ts", "ч": "ch", "ш": "sh",
    "щ": "shch", "ъ": "", "ы": "y", "і": "i", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}

# Spellings the gazetteer does not connect on its own.
_ALIASES = {
    "нур-султан": "Astana", "нұр-сұлтан": "Astana", "nur-sultan": "Astana",
    "türkistan": "Turkestan", "turkistan": "Turkestan",
    "туркестан": "Turkestan", "түркістан": "Turkestan",
    "өскемен": "Ust-Kamenogorsk", "oskemen": "Ust-Kamenogorsk",
    "қарағанды": "Karaganda", "qaraghandy": "Karaganda",
    "ақтөбе": "Aktobe", "орал": "Oral", "семей": "Semey",
}


def _transliterate(text):
    out = []
    for ch in text:
        low = ch.lower()
        if low in _TRANSLIT:
            mapped = _TRANSLIT[low]
            out.append(mapped.title() if ch.isupper() else mapped)
        else:
            out.append(ch)
    return "".join(out)


def _rank(entry):
    """Prefer real settlements over airports, peaks and districts."""
    code = (entry.get("feature_code") or "")
    bonus = {"PPLC": 3, "PPLA": 2, "PPLA2": 1}.get(code, 0)
    return (bonus, entry.get("population") or 0)


def geocode(place, country=None):
    """Turn a place name into coordinates and basic facts.

    Tries the name as written in English, Russian and Kazakh, then tries a
    transliteration. Without that, a question asked in Kazakh spends half the
    agent's steps failing to find its own cities.
    """
    candidates = [place]
    alias = _ALIASES.get(place.strip().lower())
    if alias:
        candidates.append(alias)
    latin = _transliterate(place)
    if latin.lower() != place.lower():
        candidates.append(latin)
        if _ALIASES.get(latin.strip().lower()):
            candidates.append(_ALIASES[latin.strip().lower()])

    seen = []
    for candidate in candidates:
        for language in ("en", "ru", "kk"):
            data = _get("https://geocoding-api.open-meteo.com/v1/search",
                        {"name": candidate, "count": 10, "language": language})
            results = data.get("results") or []
            if country:
                results = [r for r in results
                           if (r.get("country_code") or "").upper() == country.upper()]
            seen.extend(results)
            if results:
                break
        if seen:
            break

    if not seen:
        raise ToolError(
            "no place called %r was found. Try the Latin spelling "
            "(for example Almaty, Astana, Shymkent, Turkestan)." % place)

    r = max(seen, key=_rank)
    return {
        "name": r["name"],
        "country": r.get("country"),
        "latitude": round(r["latitude"], 4),
        "longitude": round(r["longitude"], 4),
        "population": r.get("population"),
        "elevation_m": r.get("elevation"),
        "timezone": r.get("timezone"),
        "asked_for": place,
        "source": "open-meteo geocoding",
    }


# --------------------------------------------------------------------------
# 2. distance
# --------------------------------------------------------------------------
def distance_km(origin, destination):
    """Great-circle distance between two place names, in kilometres."""
    a, b = geocode(origin), geocode(destination)
    r = 6371.0
    p1, p2 = math.radians(a["latitude"]), math.radians(b["latitude"])
    dp = p2 - p1
    dl = math.radians(b["longitude"] - a["longitude"])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    km = 2 * r * math.asin(math.sqrt(h))
    return {
        "from": a["name"], "to": b["name"],
        "straight_line_km": round(km, 1),
        "note": "straight line, not road distance; a road route is typically "
                "15-30% longer",
    }


# --------------------------------------------------------------------------
# 3. weather
# --------------------------------------------------------------------------
def weather_forecast(place, days=3):
    """Daily forecast for a place. Open-Meteo gives at most 16 days ahead."""
    days = max(1, min(int(days), 16))
    spot = geocode(place)
    data = _get("https://api.open-meteo.com/v1/forecast", {
        "latitude": spot["latitude"], "longitude": spot["longitude"],
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto", "forecast_days": days,
    })
    daily = data["daily"]
    return {
        "place": spot["name"],
        "unit": "celsius",
        "days": [
            {"date": d, "max_c": hi, "min_c": lo, "precipitation_mm": mm}
            for d, hi, lo, mm in zip(daily["time"], daily["temperature_2m_max"],
                                     daily["temperature_2m_min"],
                                     daily["precipitation_sum"])
        ],
        "source": "open-meteo forecast",
    }


# --------------------------------------------------------------------------
# 4. wikipedia
# --------------------------------------------------------------------------
def wikipedia(topic, lang="en"):
    """A short encyclopaedia summary, with the URL so the answer can cite it."""
    if lang not in ("en", "kk", "ru"):
        lang = "en"
    title = urllib.parse.quote(topic.replace(" ", "_"), safe="")
    data = _get("https://%s.wikipedia.org/api/rest_v1/page/summary/%s" % (lang, title))
    if data.get("type", "").endswith("not_found"):
        raise ToolError("no %s Wikipedia article called %r" % (lang, topic))
    extract = data.get("extract") or ""
    if not extract:
        raise ToolError("the %s article %r has no summary" % (lang, topic))
    return {
        "title": data.get("title"),
        "lang": lang,
        "summary": extract[:900],
        "url": (data.get("content_urls", {}).get("desktop", {}) or {}).get("page"),
        "source": "%s.wikipedia.org" % lang,
    }


# --------------------------------------------------------------------------
# 5. currency
# --------------------------------------------------------------------------
def convert_currency(amount, from_currency, to_currency="KZT"):
    """Convert money at today's published rate. Always returns the rate date."""
    base = from_currency.upper()
    target = to_currency.upper()
    data = _get("https://open.er-api.com/v6/latest/%s" % base)
    if data.get("result") != "success":
        raise ToolError("no rates published for %s" % base)
    rate = data.get("rates", {}).get(target)
    if rate is None:
        raise ToolError("no rate from %s to %s" % (base, target))
    return {
        "amount": float(amount), "from": base, "to": target,
        "rate": rate, "converted": round(float(amount) * rate, 2),
        "rate_date": data.get("time_last_update_utc"),
        "source": "open.er-api.com",
    }


# --------------------------------------------------------------------------
# 6. calculate
# --------------------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow,
        ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node):
    """Walk the syntax tree by hand. Never use eval() on model output: the
    model is an untrusted source, exactly like the user."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ToolError("only + - * / // % ** and numbers are allowed")


def calculate(expression):
    """Arithmetic. Language models are unreliable at this and tireless at
    deciding they are not."""
    try:
        tree = ast.parse(str(expression), mode="eval")
    except SyntaxError:
        raise ToolError("%r is not an arithmetic expression" % expression)
    value = _eval(tree.body)
    return {"expression": str(expression), "result": round(value, 6)}


# --------------------------------------------------------------------------
# The registry: function, schema, and the sentence that decides when it is used
# --------------------------------------------------------------------------
TOOLS = {
    "geocode": {
        "fn": geocode,
        "schema": {
            "name": "geocode",
            "description": "Look up a place: coordinates, country, population, "
                           "elevation and timezone. Use this whenever you need "
                           "facts about a city rather than guessing them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "place": {"type": "string", "description": "City or town name, e.g. Shymkent"},
                    "country": {"type": "string", "description": "Optional ISO country code to disambiguate, e.g. KZ"},
                },
                "required": ["place"],
            },
        },
    },
    "distance_km": {
        "fn": distance_km,
        "schema": {
            "name": "distance_km",
            "description": "Straight-line distance in kilometres between two "
                           "places. Use for 'how far is X from Y'. Road "
                           "distance is longer; say so.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                },
                "required": ["origin", "destination"],
            },
        },
    },
    "weather_forecast": {
        "fn": weather_forecast,
        "schema": {
            "name": "weather_forecast",
            "description": "Daily high, low and rainfall for the next 1-16 days "
                           "in a place. Only the near future is available; you "
                           "cannot get weather for a date months away.",
            "parameters": {
                "type": "object",
                "properties": {
                    "place": {"type": "string"},
                    "days": {"type": "integer", "description": "1 to 16"},
                },
                "required": ["place"],
            },
        },
    },
    "wikipedia": {
        "fn": wikipedia,
        "schema": {
            "name": "wikipedia",
            "description": "A short encyclopaedia summary with a URL. Use it "
                           "for what a place is known for, its history, or any "
                           "fact you would otherwise be guessing at. Available "
                           "in English, Kazakh and Russian.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Article title, e.g. Charyn Canyon"},
                    "lang": {"type": "string", "enum": ["en", "kk", "ru"]},
                },
                "required": ["topic"],
            },
        },
    },
    "convert_currency": {
        "fn": convert_currency,
        "schema": {
            "name": "convert_currency",
            "description": "Convert an amount between currencies at today's "
                           "published rate. Returns the rate and the date it "
                           "was published.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string", "description": "ISO code, e.g. USD"},
                    "to_currency": {"type": "string", "description": "ISO code, e.g. KZT"},
                },
                "required": ["amount", "from_currency"],
            },
        },
    },
    "calculate": {
        "fn": calculate,
        "schema": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression. Use this for "
                           "every sum, total, per-day figure or percentage "
                           "instead of doing the arithmetic yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. 200000/3"},
                },
                "required": ["expression"],
            },
        },
    },
}


def openai_tools():
    """The tool list in the shape the chat API expects."""
    return [{"type": "function", "function": t["schema"]} for t in TOOLS.values()]


def call(name, arguments):
    """Run one tool. Errors come back as data, so the agent can react to them."""
    if name not in TOOLS:
        return {"error": "there is no tool called %r. Available: %s"
                         % (name, ", ".join(TOOLS))}
    started = time.time()
    try:
        result = TOOLS[name]["fn"](**arguments)
        result["_seconds"] = round(time.time() - started, 2)
        return result
    except ToolError as exc:
        return {"error": str(exc)}
    except TypeError as exc:
        return {"error": "wrong arguments for %s: %s" % (name, exc)}
    except Exception as exc:                       # a tool must never crash the loop
        return {"error": "%s failed: %s" % (name, exc)}

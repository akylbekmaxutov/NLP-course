"""The graded task sets. Kept apart from the experiments so that every
experiment is scored against the same, visible, checkable answers."""
import os
import random
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))

ANSWER_LINE = re.compile(r"ANSWER\s*[:\-]\s*(.+)", re.I)
NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def extract(text):
    """Take the model's last ANSWER: line, or its last number, or its last line."""
    matches = ANSWER_LINE.findall(text or "")
    if matches:
        return matches[-1].strip().strip(".*`\"' ")
    numbers = NUMBER.findall(text or "")
    if numbers:
        return numbers[-1]
    return (text or "").strip().split("\n")[-1][:80]


def num(value):
    try:
        return float(str(value).replace(",", ".").replace(" ", "").replace("$", ""))
    except ValueError:
        return None


def close(target, tolerance=0.01):
    return lambda got: (num(got) is not None
                        and abs(num(got) - target) <= tolerance)


def text_is(*accepted):
    lowered = [a.lower() for a in accepted]
    return lambda got: any(a in (got or "").lower() for a in lowered)


# --------------------------------------------------------------------------
# REASONING — twelve questions with one checkable answer each. None of them
# needs knowledge; all of them need two or three steps held in order.
# --------------------------------------------------------------------------
REASONING = [
    ("A night bus leaves Astana at 21:40 and the journey to Almaty takes "
     "18 hours. At what time does it arrive? Answer as HH:MM.",
     text_is("15:40")),

    ("A bookshop in Shymkent lists a book at 4 500 tenge. A 15% discount is "
     "applied first, then 12% VAT is added to the discounted price. What is "
     "the final price in tenge?",
     close(4284, 1)),

    ("A course has 11 lectures. Eight of them are finished. What percentage "
     "of the course remains? Round to the nearest whole percent.",
     close(27, 0.5)),

    ("How many times does the letter 'а' appear in the word «Қарағанды»? "
     "Count only 'а', not 'ғ' or 'ы'.",
     close(3, 0)),

    ("Sort these years in increasing order: 2017, 1991, 2026, 1997. "
     "Which one is the second smallest?",
     close(1997, 0)),

    ("Aigul is older than Bolat. Bolat is older than Chingiz. Chingiz is "
     "older than Dana. Who is the youngest?",
     text_is("dana")),

    ("A balanced training set has 8 000 examples spread equally over 5 "
     "classes. How many examples are in each class?",
     close(1600, 0)),

    ("A model has 4 billion parameters stored at 4 bits each. How many "
     "gigabytes is that, using 1 GB = 1 000 000 000 bytes?",
     close(2, 0.05)),

    ("An API charges $2.00 per million input tokens. You send 250 000 input "
     "tokens. What is the cost in dollars?",
     close(0.5, 0.005)),

    ("Which number is larger, 9.11 or 9.9?",
     text_is("9.9")),

    ("The course began on 24.08.2026 and Lecture 9 is on 21.09.2026. How "
     "many days are there between those two dates?",
     close(28, 0)),

    ("A review set has 1 000 reviews. 62% are positive. Of the positive ones, "
     "half mention delivery. How many reviews are positive and mention "
     "delivery?",
     close(310, 0)),
]

REASONING_RULE = ("Solve the problem. End your reply with a final line in "
                  "exactly this form:\nANSWER: <value>")


# --------------------------------------------------------------------------
# CLASSIFICATION — real Kazakh reviews from KazSAnDRA, the same test split
# Lectures 3 to 7 were scored on, so the numbers are directly comparable.
# --------------------------------------------------------------------------
SEED = 42


def balance(df, per_class, seed=SEED):
    """Copied unchanged from Lecture 3 and reused by every lecture since."""
    per_class = min(per_class, df["label"].value_counts().min())
    parts = [g.sample(per_class, random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def kazsandra(n=100, split="test", per_class=1000):
    """The first n rows of exactly the set Lecture 7 scored its models on.

    Same file, same balance(), same seed, same 1 000-per-class test split. We
    then take the first n/2 of each class in that already-shuffled order, so
    every review the LLM sees is a review a fine-tuned mBERT was also graded
    on. Without that, a comparison against 0.799 would be a comparison between
    two different test sets.
    """
    path = os.path.join(DATA, "%s_pc.csv" % split)
    if not os.path.exists(path):
        raise SystemExit("Missing %s — run Lecture 3's script first." % path)
    df = pd.read_csv(path)[["text_cleaned", "label"]]
    df = df.rename(columns={"text_cleaned": "text"}).dropna()
    full = balance(df, per_class)

    rows = []
    for label in (0, 1):
        part = full[full.label == label].head(n // 2)
        rows.extend((str(t), int(l)) for t, l in zip(part["text"], part["label"]))
    random.Random(SEED).shuffle(rows)
    return rows


def few_shot_examples(k=8, seed=7):
    """Drawn from train, never from test."""
    path = os.path.join(DATA, "train_pc.csv")
    df = pd.read_csv(path)[["text_cleaned", "label"]]
    df = df.rename(columns={"text_cleaned": "text"}).dropna()
    picked = balance(df, 4000, seed=seed).head(k)
    return [(str(t), int(l)) for t, l in zip(picked["text"], picked["label"])]

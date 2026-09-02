#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP Course — Lecture 4
Training word embeddings: word2vec and fastText, from scratch.

This script stands on its own. It loads the KazSAnDRA reviews, trains its own
word2vec and fastText models on them, saves both to models/, and only at the
very end loads them back to reuse as features for the classifier of Lecture 3.

Setup:
    pip install -r requirements.txt
    HF_TOKEN=hf_... in a .env file

Steps:
    1. corpus   — load the reviews and tokenize them; no labels are used
    2. word2vec — train it, save it
    3. fastText — train it, save it
    4. explore  — load both back and look at what they learned
    5. reuse    — document vectors as features for the Lecture 3 classifier

    python3 lesson4.py
    python3 lesson4.py --retrain              # ignore the saved models
    python3 lesson4.py --dim 200 --epochs 20
    python3 lesson4.py --skip-cv              # faster
"""

import argparse
import os
import re

from dotenv import load_dotenv

# Read the .env before importing `datasets`: huggingface_hub reads HF_TOKEN from
# the environment when it is imported, so setting it later is too late.
HERE = os.path.dirname(os.path.abspath(__file__))
for _folder in (HERE, os.path.dirname(HERE)):         # code_files/.env or ../.env
    load_dotenv(os.path.join(_folder, ".env"))

import numpy as np
from datasets import load_dataset
from gensim.models import FastText, Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

DATASET = "issai/kazsandra"
CONFIG = "polarity_classification"
LABELS = {0: "negative", 1: "positive"}

MODEL_DIR = os.path.join(HERE, "models")
W2V_PATH = os.path.join(MODEL_DIR, "word2vec.model")
FT_PATH = os.path.join(MODEL_DIR, "fasttext.model")


# ---------------------------------------------------------------------------
# 1. The corpus
# ---------------------------------------------------------------------------

def load(split):
    """One split of KazSAnDRA as a DataFrame. The dataset is gated, so it
    needs a token; put HF_TOKEN=hf_... in a .env file next to the project."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN not found. Accept the terms at\n"
            f"  https://huggingface.co/datasets/{DATASET}\n"
            "then put HF_TOKEN=hf_... in a .env file.")
    return load_dataset(DATASET, CONFIG, split=split, token=token).to_pandas()


def balance(df, per_class, seed=42):
    """Take the same number of reviews from each class."""
    import pandas as pd
    parts = [group.sample(per_class, random_state=seed)
             for _, group in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


TOKEN = re.compile(r"\w+")


def tokenize(text):
    """Lowercase, then keep runs of letters and digits. Deliberately simple:
    the embeddings should see the words the classifier will see later."""
    return TOKEN.findall(str(text).lower())


def build_corpus(texts):
    """A list of token lists — the input format gensim expects."""
    return [tokenize(text) for text in texts]


# ---------------------------------------------------------------------------
# 2-3. Training
# ---------------------------------------------------------------------------
#
# workers=1 looks wasteful, and it is: with several threads gensim updates the
# shared weights in a racy order and two runs of the same code give different
# vectors. One worker plus a fixed seed makes the result reproducible, which
# matters more in a lecture than the speed-up.

SETTINGS = dict(vector_size=100, window=5, min_count=3, sg=1, workers=1,
                epochs=10, seed=42)


def train_word2vec(corpus, **overrides):
    settings = dict(SETTINGS, **overrides)
    model = Word2Vec(corpus, **settings)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(W2V_PATH)
    return model


def train_fasttext(corpus, **overrides):
    settings = dict(SETTINGS, **overrides)
    model = FastText(corpus, min_n=3, max_n=5, **settings)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(FT_PATH)
    return model


# ---------------------------------------------------------------------------
# 4. What the vectors learned
# ---------------------------------------------------------------------------

def explore(word2vec, fasttext, words=("жақсы", "нашар", "интернет", "рахмет")):
    for word in words:
        if word not in word2vec.wv:
            continue
        print(f"\n  {word}")
        print("    word2vec:", ", ".join(w for w, _ in word2vec.wv.most_similar(word, topn=5)))
        print("    fastText:", ", ".join(w for w, _ in fasttext.wv.most_similar(word, topn=5)))

    print("\n  similarity between pairs (word2vec):")
    for a, b in (("жақсы", "керемет"), ("жақсы", "нашар"),
                 ("интернет", "байланыс"), ("интернет", "тамақ")):
        if a in word2vec.wv and b in word2vec.wv:
            print(f"    {a:>9} ~ {b:<10} {word2vec.wv.similarity(a, b):.3f}")

    print("\n  words that were never in the training text:")
    for word in ("кітаптарымыздан", "ұнамағандықтан", "интернетсіздіктен"):
        known = word in word2vec.wv
        nearest = ", ".join(w for w, _ in fasttext.wv.most_similar(word, topn=3))
        print(f"    {word:<20} word2vec knows it: {str(known):<5}  fastText → {nearest}")


# ---------------------------------------------------------------------------
# 5. Reuse: document vectors as features
# ---------------------------------------------------------------------------

class MeanEmbedding(BaseEstimator, TransformerMixin):
    """Average the word vectors of a document.

    Same interface as TfidfVectorizer, so it slots into a Pipeline unchanged.
    word2vec can only look up words it saw during training; fastText builds a
    vector for anything out of its character n-grams, so nothing is skipped.
    """

    def __init__(self, model=None):
        self.model = model

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        vectors = self.model.wv
        out = np.zeros((len(texts), vectors.vector_size), dtype=np.float32)
        self.skipped_ = 0
        for i, text in enumerate(texts):
            found = []
            for word in tokenize(text):
                try:
                    found.append(vectors[word])
                except KeyError:          # word2vec only: unknown word
                    self.skipped_ += 1
            if found:
                out[i] = np.mean(found, axis=0)
        return out


def build_pipelines(word2vec, fasttext):
    classifier = lambda: LogisticRegression(max_iter=2000, C=5.0)
    return {
        "TF-IDF": Pipeline([
            ("features", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", classifier()),
        ]),
        "word2vec": Pipeline([("features", MeanEmbedding(word2vec)), ("clf", classifier())]),
        "fastText": Pipeline([("features", MeanEmbedding(fasttext)), ("clf", classifier())]),
    }


def evaluate(name, model, x_train, y_train, x_test, y_test, cross_validate=True):
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    line = f"  {name:<10} test={accuracy:.3f}"

    features = model.named_steps["features"]
    if hasattr(features, "skipped_"):
        line += f"   skipped {features.skipped_} unknown tokens"
    if cross_validate:
        scores = cross_val_score(model, x_train, y_train, cv=5)
        line += f"   CV={scores.mean():.3f} ± {scores.std():.3f}"
    print(line)
    return accuracy, model.predict(x_test)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Training word embeddings on KazSAnDRA")
    parser.add_argument("--dim", type=int, default=100, help="vector size")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--retrain", action="store_true", help="ignore models/")
    parser.add_argument("--train-size", type=int, default=4000, help="reviews per class")
    parser.add_argument("--test-size", type=int, default=1000, help="reviews per class")
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    settings = dict(vector_size=args.dim, epochs=args.epochs, min_count=args.min_count)
    saved = os.path.exists(W2V_PATH) and os.path.exists(FT_PATH)

    print("=" * 70, "\nSTEP 1 — the corpus\n", "=" * 70, sep="")
    reviews = load("train")["text_cleaned"].astype(str)
    print(f"  reviews: {len(reviews)}  (the labels are not used for training vectors)")
    corpus = build_corpus(reviews)
    print(f"  tokens : {sum(len(s) for s in corpus)}")
    print(f"  example: {corpus[0][:8]}")

    if saved and not args.retrain:
        print("\n" + "=" * 70, "\nSTEP 2-3 — load the saved models\n", "=" * 70, sep="")
        print("  models/ already exists — run with --retrain to train again")
        word2vec = Word2Vec.load(W2V_PATH)
        fasttext = FastText.load(FT_PATH)
    else:
        print("\n" + "=" * 70, "\nSTEP 2 — train word2vec\n", "=" * 70, sep="")
        word2vec = train_word2vec(corpus, **settings)
        print(f"  saved to models/{os.path.basename(W2V_PATH)}")

        print("\n" + "=" * 70, "\nSTEP 3 — train fastText\n", "=" * 70, sep="")
        fasttext = train_fasttext(corpus, **settings)
        print(f"  saved to models/{os.path.basename(FT_PATH)}")

    print(f"  vocabulary: {len(word2vec.wv)} words × {word2vec.wv.vector_size} dimensions")

    print("\n" + "=" * 70, "\nSTEP 4 — what the vectors learned\n", "=" * 70, sep="")
    explore(word2vec, fasttext)

    print("\n" + "=" * 70, "\nSTEP 5 — reuse them for the Lecture 3 classifier\n", "=" * 70, sep="")
    train = balance(load("train"), args.train_size)
    test = balance(load("test"), args.test_size)
    x_train, y_train = train["text_cleaned"].astype(str), train["label"]
    x_test, y_test = test["text_cleaned"].astype(str), test["label"]
    print(f"  train: {len(train)}  test: {len(test)}  (balanced, as in Lecture 3)\n")

    results, predictions = {}, {}
    for name, model in build_pipelines(word2vec, fasttext).items():
        results[name], predictions[name] = evaluate(
            name, model, x_train, y_train, x_test, y_test, not args.skip_cv)

    best = max(results, key=results.get)
    print(f"\n  best: {best} ({results[best]:.3f})\n")
    print(classification_report(y_test, predictions[best],
                                target_names=[LABELS[0], LABELS[1]], digits=3))
    print("confusion matrix (rows = true, columns = predicted)")
    print(confusion_matrix(y_test, predictions[best]))

    fixed = [(t, y) for t, y, a, b in zip(x_test, y_test, predictions["TF-IDF"],
                                          predictions["fastText"]) if a != y and b == y]
    print(f"\n  reviews fastText got right and TF-IDF got wrong: {len(fixed)}")
    for text, label in fixed[:3]:
        print(f"    {LABELS[label]:>8}  «{text[:55]}»")


if __name__ == "__main__":
    main()

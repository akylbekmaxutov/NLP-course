#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP Course — Lecture 3
Classical Machine Learning for text: polarity classification on KazSAnDRA.

Dataset: issai/kazsandra, polarity_classification split.
Labels: 0 = negative, 1 = positive.

Setup:
    pip install -r requirements.txt
    echo "HF_TOKEN=hf_..." > .env        # after accepting the terms on the dataset page

Steps, in order:
    1. load      — download the CSV files once, then read them from disk
    2. explore   — EDA: class balance, domains, review lengths, examples
    3. balance   — equal number of positive and negative, for train and for test
    4. train     — TF-IDF + five classifiers
    5. test      — accuracy, precision, recall, F1, confusion matrix
    6. inspect   — the words each model relies on, and the mistakes it makes

    python3 lesson3.py
    python3 lesson3.py --train-size 1000 --test-size 500
    python3 lesson3.py --model logreg
"""

import argparse
import os

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

DATASET = "issai/kazsandra"
CONFIG = "polarity_classification"
LABELS = {0: "negative", 1: "positive"}


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def load(split):
    """One split of KazSAnDRA as a DataFrame.

    The dataset is gated, so `datasets` needs a token. Put it in a .env file
    next to the project and python-dotenv will pick it up.
    """
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN not found. Accept the terms at\n"
            f"  https://huggingface.co/datasets/{DATASET}\n"
            "then put HF_TOKEN=hf_... in a .env file.")

    dataset = load_dataset(DATASET, CONFIG, split=split, token=token)
    return dataset.to_pandas()


# ---------------------------------------------------------------------------
# 2. Explore
# ---------------------------------------------------------------------------

def explore(df, name):
    print(f"\n{name}: {len(df)} reviews")

    counts = df["label"].value_counts().sort_index()
    for label, n in counts.items():
        print(f"  label {label} ({LABELS[label]:>8}): {n:>6}  {100 * n / len(df):5.1f}%")

    print("  domains:", df["domain"].value_counts().to_dict())

    words = df["text_cleaned"].astype(str).str.split().str.len()
    print(f"  words per review: median={words.median():.0f}  mean={words.mean():.1f}  max={words.max()}")
    print(f"  very short (<= 3 words): {100 * (words <= 3).mean():.1f}%")

    for label in counts.index:
        example = df.loc[df["label"] == label, "text"].iloc[0]
        print(f"  label {label} ({LABELS[label]}): «{str(example)[:60]}»")


# ---------------------------------------------------------------------------
# 3. Balance
# ---------------------------------------------------------------------------

def balance(df, per_class, seed=42):
    """Take the same number of reviews from each class."""
    parts = [group.sample(per_class, random_state=seed)
             for _, group in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------

def build(name):
    """TF-IDF features plus one classifier, as a single Pipeline."""
    classifiers = {
        "nb": MultinomialNB(alpha=1.0),
        "logreg": LogisticRegression(max_iter=1000, C=5.0),
        "svm": LinearSVC(C=1.0),
        # Trees split on one feature at a time, which is a poor fit for tens of
        # thousands of sparse columns. They are here to be measured, not assumed.
        "tree": DecisionTreeClassifier(max_depth=40, random_state=42),
        "forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    }
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", classifiers[name]),
    ])


# ---------------------------------------------------------------------------
# 5. Test
# ---------------------------------------------------------------------------

def evaluate(model, x_test, y_test, name):
    predicted = model.predict(x_test)
    print(f"\n--- {name} ---")
    print(classification_report(y_test, predicted,
                                target_names=[LABELS[0], LABELS[1]], digits=3))
    print("confusion matrix (rows = true, columns = predicted)")
    print(confusion_matrix(y_test, predicted))
    return predicted


# ---------------------------------------------------------------------------
# 6. Inspect
# ---------------------------------------------------------------------------

# Real reviews contain profanity; these are skipped in the printed lists.
SKIP = {}


def top_words(model, n=10):
    """The features pushing hardest towards each class.

    Logistic Regression and LinearSVC expose one weight per feature in coef_.
    MultinomialNB has no weights — it stores log P(word | class), so the
    difference between the two rows plays the same role. Trees only report
    feature_importances_, which has no direction, so they return nothing here.
    """
    names = model.named_steps["tfidf"].get_feature_names_out()
    classifier = model.named_steps["clf"]
    if hasattr(classifier, "coef_"):                     # LogisticRegression, LinearSVC
        weights = classifier.coef_[0]
    elif hasattr(classifier, "feature_log_prob_"):       # MultinomialNB
        weights = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
    else:                                                # trees: importance has no sign
        return [], []
    order = weights.argsort()
    negative = [names[i] for i in order if names[i] not in SKIP][:n]
    positive = [names[i] for i in order[::-1] if names[i] not in SKIP][:n]
    return negative, positive


def show_mistakes(x_test, y_test, predicted, limit=4):
    wrong = [(t, y, p) for t, y, p in zip(x_test, y_test, predicted) if y != p]
    for text, true, got in wrong[:limit]:
        print(f"  true={LABELS[true]:>8}  predicted={LABELS[got]:>8}  «{str(text)[:55]}»")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Polarity classification on KazSAnDRA")
    parser.add_argument("--train-size", type=int, default=4000, help="reviews per class")
    parser.add_argument("--test-size", type=int, default=1000, help="reviews per class")
    parser.add_argument("--model", default="all",
                        choices=["nb", "logreg", "svm", "tree", "forest", "all"])
    parser.add_argument("--figure", metavar="PNG", help="save the confusion matrix")
    args = parser.parse_args()

    print("=" * 70, "\nSTEP 1 — load\n", "=" * 70, sep="")
    train_raw, test_raw = load("train"), load("test")

    print("\n" + "=" * 70, "\nSTEP 2 — explore\n", "=" * 70, sep="")
    explore(train_raw, "train (raw)")
    explore(test_raw, "test (raw)")

    print("\n" + "=" * 70, "\nSTEP 3 — balance\n", "=" * 70, sep="")
    train = balance(train_raw, args.train_size)
    test = balance(test_raw, args.test_size)
    x_train, y_train = train["text_cleaned"].astype(str), train["label"]
    x_test, y_test = test["text_cleaned"].astype(str), test["label"]
    print(f"  train: {len(train)} reviews  {train['label'].value_counts().sort_index().to_dict()}")
    print(f"  test : {len(test)} reviews  {test['label'].value_counts().sort_index().to_dict()}")

    names = (["nb", "logreg", "svm", "tree", "forest"]
             if args.model == "all" else [args.model])
    titles = {"nb": "Naive Bayes", "logreg": "Logistic Regression",
              "svm": "Linear SVM", "tree": "Decision Tree", "forest": "Random Forest"}
    best = None

    for name in names:
        print("\n" + "=" * 70, f"\nSTEP 4-5 — {titles[name]}\n", "=" * 70, sep="")
        model = build(name)
        model.fit(x_train, y_train)
        print(f"  features: {len(model.named_steps['tfidf'].get_feature_names_out())}")
        print(f"  train accuracy: {model.score(x_train, y_train):.3f}")
        print(f"  test  accuracy: {model.score(x_test, y_test):.3f}")
        scores = cross_val_score(model, x_train, y_train, cv=5)
        print(f"  5-fold CV: {scores.mean():.3f} ± {scores.std():.3f}  {scores.round(3)}")
        predicted = evaluate(model, x_test, y_test, titles[name])
        if best is None:
            best = (model, predicted)

    print("\n" + "=" * 70, "\nSTEP 6 — inspect\n", "=" * 70, sep="")
    model, predicted = best
    negative, positive = top_words(model)
    if negative:
        print("  words pointing to negative:", ", ".join(negative))
        print("  words pointing to positive:", ", ".join(positive))
    print("\n  mistakes:")
    show_mistakes(x_test, y_test, predicted)

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ConfusionMatrixDisplay.from_predictions(
            y_test, predicted, display_labels=[LABELS[0], LABELS[1]], cmap="Blues")
        plt.tight_layout()
        plt.savefig(args.figure, dpi=140)
        print(f"\n  saved {args.figure}")


if __name__ == "__main__":
    main()

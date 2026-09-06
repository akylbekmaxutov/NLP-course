#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP Course — Lecture 5
Recurrent networks: RNN and LSTM in PyTorch.

Everything so far threw the word order away. A bag of words, a TF-IDF matrix and
an average of embeddings all give the same answer for "жақсы емес" and
"емес жақсы". A recurrent network reads a review one word at a time and keeps a
hidden state, so order finally matters.

Two tasks from the same KazSAnDRA dataset:
    polarity — 2 classes, negative / positive
    score    — 5 classes, 1 to 5 stars

Setup:
    pip install -r requirements.txt
    HF_TOKEN=hf_... in a .env file

Steps:
    1. data    — load a task, balance it, look at the classes
    2. vocab   — build a word -> index table from the training text
    3. batches — numericalise, pad, and hand out batches
    4. model   — Embedding -> RNN or LSTM -> Linear
    5. train   — the loop: forward, loss, backward, step
    6. compare — RNN against LSTM, and both against the Lecture 3/4 baselines

    python3 lesson5.py                        # both tasks, both models
    python3 lesson5.py --task polarity
    python3 lesson5.py --task score --model lstm --epochs 8
    python3 lesson5.py --pretrained           # start from the Lecture 4 word2vec
"""

import argparse
import os
import re
import time

from dotenv import load_dotenv

# Read the .env before importing `datasets`: huggingface_hub reads HF_TOKEN from
# the environment when it is imported, so setting it later is too late.
HERE = os.path.dirname(os.path.abspath(__file__))
for _folder in (HERE, os.path.dirname(HERE)):         # code_files/.env or ../.env
    load_dotenv(os.path.join(_folder, ".env"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

DATASET = "issai/kazsandra"
TASKS = {
    "polarity": ("polarity_classification", ["negative", "positive"]),
    "score": ("score_classification", ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]),
}

PAD, UNK = 0, 1          # two reserved slots at the start of the vocabulary
SEED = 42
DEVICE = torch.device("cpu")   # small models; CPU keeps the run reproducible


def seed_everything(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

def load(task, split):
    config, _ = TASKS[task]
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN not found. Accept the terms at\n"
            f"  https://huggingface.co/datasets/{DATASET}\n"
            "then put HF_TOKEN=hf_... in a .env file.")
    return load_dataset(DATASET, config, split=split, token=token).to_pandas()


def balance(df, per_class, seed=SEED):
    """The same number of reviews from each class, as in Lectures 3 and 4."""
    per_class = min(per_class, df["label"].value_counts().min())
    parts = [group.sample(per_class, random_state=seed)
             for _, group in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


TOKEN = re.compile(r"\w+")


def tokenize(text):
    return TOKEN.findall(str(text).lower())


# ---------------------------------------------------------------------------
# 2. Vocabulary
# ---------------------------------------------------------------------------

def build_vocab(texts, min_count=2):
    """word -> index, built from the training text only.

    Index 0 is <pad>, index 1 is <unk>. Everything the network meets later that
    is not in this table becomes <unk> — the same coverage problem as in
    Lecture 4, now visible as a single row of the embedding matrix.
    """
    counts = {}
    for text in texts:
        for word in tokenize(text):
            counts[word] = counts.get(word, 0) + 1
    kept = sorted((w for w, c in counts.items() if c >= min_count),
                  key=lambda w: (-counts[w], w))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for word in kept:
        vocab[word] = len(vocab)
    return vocab, len(counts)


# ---------------------------------------------------------------------------
# 3. Batches
# ---------------------------------------------------------------------------

class Reviews(Dataset):
    def __init__(self, texts, labels, vocab, max_len=60):
        self.rows = []
        for text, label in zip(texts, labels):
            ids = [vocab.get(w, UNK) for w in tokenize(text)][:max_len]
            self.rows.append((ids or [UNK], int(label)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch):
    """Pad a batch to its longest review and remember the true lengths.

    Reviews in one batch have different lengths, but a tensor is rectangular.
    We pad with <pad> and pass the lengths on, so the model can stop reading
    each review where it actually ends.
    """
    lengths = torch.tensor([len(ids) for ids, _ in batch])
    width = int(lengths.max())
    padded = torch.full((len(batch), width), PAD, dtype=torch.long)
    for i, (ids, _) in enumerate(batch):
        padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    return padded, lengths, labels


def loader(dataset, batch_size, shuffle):
    """A fresh generator every call, so every model sees the same batch order."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=torch.Generator().manual_seed(SEED) if shuffle else None)


# ---------------------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------------------

class Recurrent(nn.Module):
    """Embedding -> RNN or LSTM -> Linear.

    The recurrent layer returns one hidden state per review, computed after the
    last real word. That single vector is what the classifier sees, so it has
    to carry everything the network decided while reading.
    """

    def __init__(self, vocab_size, classes, kind="lstm", dim=100, hidden=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        cell = nn.LSTM if kind == "lstm" else nn.RNN
        self.rnn = cell(dim, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, classes)

    def forward(self, ids, lengths):
        embedded = self.dropout(self.embedding(ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False)
        _, state = self.rnn(packed)
        hidden = state[0] if isinstance(state, tuple) else state   # LSTM returns (h, c)
        return self.out(self.dropout(hidden[-1]))


def load_pretrained(model, vocab, path=os.path.join(HERE, "models", "word2vec.model")):
    """Copy the Lecture 4 vectors into the embedding layer, where they exist."""
    from gensim.models import Word2Vec
    vectors = Word2Vec.load(path).wv
    weights = model.embedding.weight.data
    if vectors.vector_size != weights.shape[1]:
        raise SystemExit(f"--dim must be {vectors.vector_size} to match the saved model")
    found = 0
    for word, index in vocab.items():
        if word in vectors:
            weights[index] = torch.tensor(vectors[word])
            found += 1
    return found


# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------

def accuracy(model, batches):
    model.eval()
    right = total = 0
    with torch.no_grad():
        for ids, lengths, labels in batches:
            predicted = model(ids, lengths).argmax(1)
            right += int((predicted == labels).sum())
            total += len(labels)
    return right / total


def predict(model, batches):
    model.eval()
    out, gold = [], []
    with torch.no_grad():
        for ids, lengths, labels in batches:
            out.append(model(ids, lengths).argmax(1))
            gold.append(labels)
    return torch.cat(out).numpy(), torch.cat(gold).numpy()


def train(model, train_batches, test_batches, epochs=5, lr=1e-3, quiet=False):
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        started, running = time.time(), 0.0
        for ids, lengths, labels in train_batches:
            optimiser.zero_grad()
            loss = loss_fn(model(ids, lengths), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # keeps RNNs from exploding
            optimiser.step()
            running += loss.item() * len(labels)
        if not quiet:
            print(f"    epoch {epoch}  loss={running / len(train_batches.dataset):.4f}"
                  f"  test={accuracy(model, test_batches):.3f}"
                  f"  ({time.time() - started:.0f}s)")
    return accuracy(model, test_batches)


# ---------------------------------------------------------------------------
# 6. One task, end to end
# ---------------------------------------------------------------------------

def run_task(task, args):
    config, names = TASKS[task]
    print("=" * 70, f"\n{task.upper()} — {len(names)} classes\n", "=" * 70, sep="")

    train_df = balance(load(task, "train"), args.train_size)
    test_df = balance(load(task, "test"), args.test_size)
    print(f"  train {len(train_df)}  test {len(test_df)}  "
          f"({len(train_df) // len(names)} per class)")

    vocab, seen = build_vocab(train_df["text_cleaned"].astype(str), args.min_count)
    print(f"  vocabulary {len(vocab)} of {seen} distinct words "
          f"(min_count={args.min_count})")

    train_rows = Reviews(train_df["text_cleaned"].astype(str),
                         train_df["label"], vocab, args.max_len)
    test_batches = loader(Reviews(test_df["text_cleaned"].astype(str),
                                  test_df["label"], vocab, args.max_len),
                          args.batch_size, shuffle=False)

    results, models = {}, {}
    for kind in (["rnn", "lstm"] if args.model == "both" else [args.model]):
        seed_everything()
        model = Recurrent(len(vocab), len(names), kind, args.dim, args.hidden)
        note = ""
        if args.pretrained:
            found = load_pretrained(model, vocab)
            note = f"  ({found} of {len(vocab)} words started from Lecture 4 vectors)"
        total = sum(p.numel() for p in model.parameters())
        print(f"\n  {kind.upper()}  {total:,} parameters{note}")
        train_batches = loader(train_rows, args.batch_size, shuffle=True)
        results[kind] = train(model, train_batches, test_batches, args.epochs, args.lr)
        models[kind] = model

    best = max(results, key=results.get)
    print(f"\n  best: {best.upper()} {results[best]:.3f}   "
          f"(random guessing would be {1 / len(names):.3f})\n")
    predicted, gold = predict(models[best], test_batches)
    print(classification_report(gold, predicted, target_names=names, digits=3, zero_division=0))
    print("confusion matrix (rows = true, columns = predicted)")
    print(confusion_matrix(gold, predicted))
    return results


def main():
    parser = argparse.ArgumentParser(description="RNN and LSTM on KazSAnDRA")
    parser.add_argument("--task", choices=["polarity", "score", "both"], default="both")
    parser.add_argument("--model", choices=["rnn", "lstm", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dim", type=int, default=100, help="embedding size")
    parser.add_argument("--hidden", type=int, default=128, help="hidden state size")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=60, help="words kept per review")
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-size", type=int, default=4000, help="reviews per class")
    parser.add_argument("--test-size", type=int, default=1000, help="reviews per class")
    parser.add_argument("--pretrained", action="store_true",
                        help="start the embedding layer from models/word2vec.model")
    args = parser.parse_args()

    tasks = ["polarity", "score"] if args.task == "both" else [args.task]
    summary = {}
    for task in tasks:
        summary[task] = run_task(task, args)
        print()

    print("=" * 70, "\nSUMMARY\n", "=" * 70, sep="")
    for task, results in summary.items():
        for kind, score in results.items():
            print(f"  {task:<9} {kind.upper():<5} {score:.3f}")


if __name__ == "__main__":
    main()

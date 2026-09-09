#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP Course — Lecture 6
Attention and Transformers, in PyTorch.

Lecture 5 squeezed every review through one 128-number hidden state, and it had
to read strictly left to right. Attention removes both limits: every word looks
at every other word directly, and all positions are computed at once.

Same two KazSAnDRA tasks as Lecture 5, so the numbers are comparable:
    polarity — 2 classes, negative / positive
    score    — 5 classes, 1 to 5 stars

Setup:
    pip install -r requirements.txt
    HF_TOKEN=hf_... in a .env file

Steps:
    1. data      — load a task, balance it, build the vocabulary
    2. attention — scaled dot-product attention, written out
    3. block     — multi-head attention + feed-forward, with residuals
    4. model     — embedding + positions -> N blocks -> pooling -> Linear
    5. train     — the same loop as Lecture 5
    6. inspect   — what the attention heads actually look at

    python3 lesson6.py                     # both tasks
    python3 lesson6.py --task polarity --heads 4 --layers 2
    python3 lesson6.py --attention-map     # print attention over real reviews
"""

import argparse
import math
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
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

DATASET = "issai/kazsandra"
TASKS = {
    "polarity": ("polarity_classification", ["negative", "positive"]),
    "score": ("score_classification", ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]),
}

PAD, UNK, SEED = 0, 1, 42
DEVICE = torch.device("cpu")      # small models; CPU keeps the run reproducible


# ---------------------------------------------------------------------------
# 1. Data — the same preparation as Lecture 5
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
    per_class = min(per_class, df["label"].value_counts().min())
    parts = [g.sample(per_class, random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def tokenize(text):
    return re.findall(r"\w+", str(text).lower())


def build_vocab(texts, min_count=2):
    counts = {}
    for text in texts:
        for word in tokenize(text):
            counts[word] = counts.get(word, 0) + 1
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for word in sorted((w for w, c in counts.items() if c >= min_count),
                       key=lambda w: (-counts[w], w)):
        vocab[word] = len(vocab)
    return vocab


def encode(df, vocab, max_len=60):
    rows = []
    for text, label in zip(df["text_cleaned"], df["label"]):
        ids = [vocab.get(w, UNK) for w in tokenize(text)][:max_len]
        rows.append((ids or [UNK], int(label)))
    return rows


def collate(batch):
    """Pad to the longest review and build the mask that hides the padding."""
    width = max(len(ids) for ids, _ in batch)
    padded = torch.full((len(batch), width), PAD, dtype=torch.long)
    for i, (ids, _) in enumerate(batch):
        padded[i, :len(ids)] = torch.tensor(ids)
    mask = padded == PAD                       # True where the model must not look
    labels = torch.tensor([y for _, y in batch], dtype=torch.long)
    return padded, mask, labels


def loader(rows, batch_size, shuffle):
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=torch.Generator().manual_seed(SEED) if shuffle else None)


# ---------------------------------------------------------------------------
# 2. Attention — written out rather than imported
# ---------------------------------------------------------------------------

def attention(query, key, value, mask=None):
    """Scaled dot-product attention.

    query, key, value: (batch, heads, words, size)
    Returns the weighted values and the weights themselves, so we can look at
    them later.
    """
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if mask is not None:                        # (batch, 1, 1, words)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ value, weights


class MultiHeadAttention(nn.Module):
    """One linear layer per role, then the same attention run in `heads` parallel
    subspaces. Splitting is a reshape: 64 numbers become 4 heads of 16."""

    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0, "dim must divide evenly into heads"
        self.heads, self.size = heads, dim // heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def split(self, x):
        batch, words, _ = x.shape
        return x.view(batch, words, self.heads, self.size).transpose(1, 2)

    def forward(self, x, mask=None):
        q, k, v = self.split(self.query(x)), self.split(self.key(x)), self.split(self.value(x))
        out, weights = attention(q, k, v, mask)
        batch, _, words, _ = out.shape
        out = out.transpose(1, 2).reshape(batch, words, self.heads * self.size)
        return self.out(out), weights


# ---------------------------------------------------------------------------
# 3. One encoder block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """attention -> add & norm -> feed-forward -> add & norm."""

    def __init__(self, dim, heads, hidden, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(dim, heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attended, weights = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attended)                     # residual 1
        x = x + self.dropout(self.feed_forward(self.norm2(x)))   # residual 2
        return x, weights


# ---------------------------------------------------------------------------
# 4. The classifier
# ---------------------------------------------------------------------------

def positional_encoding(max_len, dim):
    """The sine/cosine table from the original paper: a fixed pattern per position."""
    position = torch.arange(max_len).unsqueeze(1).float()
    step = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    table = torch.zeros(max_len, dim)
    table[:, 0::2] = torch.sin(position * step)
    table[:, 1::2] = torch.cos(position * step)
    return table


class Transformer(nn.Module):
    def __init__(self, vocab_size, classes, dim=128, heads=4, layers=2,
                 hidden=256, max_len=60, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.register_buffer("positions", positional_encoding(max_len, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, hidden, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dim, classes)

    def forward(self, ids, pad_mask, return_weights=False):
        x = self.embedding(ids) + self.positions[:ids.size(1)]
        x = self.dropout(x)
        mask = pad_mask[:, None, None, :]              # (batch, 1, 1, words)
        weights = None
        for block in self.blocks:
            x, weights = block(x, mask)
        x = self.norm(x)

        # mean over the real words only — padding must not dilute the average
        keep = (~pad_mask).unsqueeze(-1).float()
        pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1)
        logits = self.out(pooled)
        return (logits, weights) if return_weights else logits


# ---------------------------------------------------------------------------
# 5. Train — the same loop as Lecture 5
# ---------------------------------------------------------------------------

def accuracy(model, batches):
    model.eval()
    right = total = 0
    with torch.no_grad():
        for ids, mask, labels in batches:
            right += int((model(ids, mask).argmax(1) == labels).sum())
            total += len(labels)
    return right / total


def predict(model, batches):
    model.eval()
    out, gold = [], []
    with torch.no_grad():
        for ids, mask, labels in batches:
            out.append(model(ids, mask).argmax(1))
            gold.append(labels)
    return torch.cat(out).numpy(), torch.cat(gold).numpy()


def train(model, train_batches, test_batches, epochs, lr, size):
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for epoch in range(1, epochs + 1):
        model.train()
        started, running = time.time(), 0.0
        for ids, mask, labels in train_batches:
            optimiser.zero_grad()
            loss = loss_fn(model(ids, mask), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            running += loss.item() * len(labels)
        print(f"    epoch {epoch}  loss={running / size:.4f}"
              f"  test={accuracy(model, test_batches):.3f}"
              f"  ({time.time() - started:.0f}s)")
    return accuracy(model, test_batches)


# ---------------------------------------------------------------------------
# 6. Inspect — what did the heads learn to look at?
# ---------------------------------------------------------------------------

def attention_map(model, vocab, text, head=0):
    inverse = {i: w for w, i in vocab.items()}
    ids = torch.tensor([[vocab.get(w, UNK) for w in tokenize(text)][:20]])
    mask = ids == PAD
    model.eval()
    with torch.no_grad():
        _, weights = model(ids, mask, return_weights=True)
    words = [inverse[int(i)] for i in ids[0]]
    matrix = weights[0, head]
    print(f"\n  «{text[:60]}»   head {head}")
    print("  each row: how much that word attends to each of the others")
    header = "".join(f"{w[:6]:>7}" for w in words)
    print(f"  {'':<10}{header}")
    for i, word in enumerate(words):
        row = "".join(f"{float(matrix[i, j]):>7.2f}" for j in range(len(words)))
        print(f"  {word[:9]:<10}{row}")


def run_task(task, args):
    _, names = TASKS[task]
    print("=" * 70, f"\n{task.upper()} — {len(names)} classes\n", "=" * 70, sep="")

    train_df = balance(load(task, "train"), args.train_size)
    test_df = balance(load(task, "test"), args.test_size)
    vocab = build_vocab(train_df["text_cleaned"].astype(str), args.min_count)
    print(f"  train {len(train_df)}  test {len(test_df)}  vocabulary {len(vocab)}")

    train_rows = encode(train_df, vocab, args.max_len)
    test_batches = loader(encode(test_df, vocab, args.max_len), args.batch_size, False)

    torch.manual_seed(SEED)
    model = Transformer(len(vocab), len(names), args.dim, args.heads,
                        args.layers, args.hidden, args.max_len)
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters"
          f"  ({args.layers} blocks × {args.heads} heads)\n")

    score = train(model, loader(train_rows, args.batch_size, True), test_batches,
                  args.epochs, args.lr, len(train_df))

    predicted, gold = predict(model, test_batches)
    print(f"\n  final: {score:.3f}   (random guessing would be {1 / len(names):.3f})\n")
    print(classification_report(gold, predicted, target_names=names, digits=3, zero_division=0))
    print("confusion matrix (rows = true, columns = predicted)")
    print(confusion_matrix(gold, predicted))

    if args.attention_map:
        for text in test_df["text_cleaned"].astype(str).head(2):
            attention_map(model, vocab, text)
    return score


def main():
    parser = argparse.ArgumentParser(description="Attention and Transformers on KazSAnDRA")
    parser.add_argument("--task", choices=["polarity", "score", "both"], default="both")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=256, help="feed-forward width")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=60)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-size", type=int, default=4000, help="reviews per class")
    parser.add_argument("--test-size", type=int, default=1000, help="reviews per class")
    parser.add_argument("--attention-map", action="store_true")
    args = parser.parse_args()

    results = {}
    for task in (["polarity", "score"] if args.task == "both" else [args.task]):
        results[task] = run_task(task, args)
        print()

    print("=" * 70, "\nSUMMARY\n", "=" * 70, sep="")
    for task, score in results.items():
        print(f"  {task:<9} {score:.3f}")


if __name__ == "__main__":
    main()

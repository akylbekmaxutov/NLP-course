"""Retrieval: three ways to find the passages a question needs.

  BM25    counts words. Exact, fast, no model, no GPU. Lecture 3's bag of
          words with better weighting.
  dense   embeds question and passage into the same vector space and takes
          the nearest. Lecture 4's idea, with Lecture 7's encoder.
  hybrid  fuses the two rankings.

Run it directly to compare all three on the built-in question set:

    python3 corpus.py
    python3 retrieve.py --build
    python3 retrieve.py --compare
"""
import argparse
import json
import math
import os
import re
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "index")
CHUNKS = os.path.join(INDEX, "chunks.jsonl")
VECTORS = os.path.join(INDEX, "vectors.npy")

# Small, multilingual, and it handles Kazakh. 118 M parameters: it runs on a
# laptop, which is the whole point of putting retrieval before generation.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small")

WORD = re.compile(r"\w+", re.UNICODE)


def load_chunks(path=CHUNKS):
    if not os.path.exists(path):
        raise SystemExit("No chunks yet. Run:  python3 corpus.py")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


# --------------------------------------------------------------------------
# BM25 — written out rather than imported, because it is short enough to read
# --------------------------------------------------------------------------
class BM25:
    """Okapi BM25. k1 damps repeated words, b damps long documents."""

    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [WORD.findall(d.lower()) for d in documents]
        self.lengths = np.array([len(d) for d in self.docs], dtype=np.float32)
        self.avg_length = float(self.lengths.mean())
        self.n = len(self.docs)

        self.postings = {}          # term -> {doc index: count}
        for i, doc in enumerate(self.docs):
            for term, count in Counter(doc).items():
                self.postings.setdefault(term, {})[i] = count

        self.idf = {}
        for term, posting in self.postings.items():
            df = len(posting)
            # A word in every document carries no information; this goes to ~0.
            self.idf[term] = math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query):
        out = np.zeros(self.n, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg_length)
        for term in WORD.findall(query.lower()):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for i, count in posting.items():
                out[i] += idf * count * (self.k1 + 1) / (count + norm[i])
        return out


# --------------------------------------------------------------------------
# Dense retrieval
# --------------------------------------------------------------------------
_encoder = None


def encoder():
    global _encoder
    if _encoder is None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
        model = AutoModel.from_pretrained(EMBED_MODEL)
        model.eval()
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        _encoder = (tok, model, device, torch)
    return _encoder


def embed(texts, prefix, batch_size=32, quiet=False):
    """e5 models are trained with these prefixes; without them quality drops."""
    tok, model, device, torch = encoder()
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[start:start + batch_size]]
        enc = tok(batch, padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc).last_hidden_state
        # Mean pooling over real tokens only — padding must not vote.
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        vectors.append(pooled.cpu().numpy())
        if not quiet and len(texts) > batch_size:
            print("\r  embedded %d/%d" % (min(start + batch_size, len(texts)),
                                          len(texts)), end="", flush=True)
    if not quiet and len(texts) > batch_size:
        print()
    return np.vstack(vectors).astype(np.float32)


def build_vectors(chunks, out=VECTORS):
    print("embedding %d chunks with %s" % (len(chunks), EMBED_MODEL))
    vectors = embed([c["text"] for c in chunks], "passage: ")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, vectors)
    print("wrote %s  shape %s  %.1f MB"
          % (out, vectors.shape, vectors.nbytes / 1e6))
    return vectors


# --------------------------------------------------------------------------
# The retriever the rest of the folder uses
# --------------------------------------------------------------------------
class Retriever:
    def __init__(self, chunks=None, vectors=None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.bm25 = BM25([c["text"] for c in self.chunks])
        self._vectors = vectors

    @property
    def vectors(self):
        if self._vectors is None:
            if not os.path.exists(VECTORS):
                raise SystemExit("No vectors yet. Run:  python3 retrieve.py --build")
            self._vectors = np.load(VECTORS)
            if len(self._vectors) != len(self.chunks):
                raise SystemExit("vectors and chunks disagree — rebuild both")
        return self._vectors

    def search(self, query, k=5, method="hybrid", lang=None):
        if method == "bm25":
            scores = self.bm25.scores(query)
        elif method == "dense":
            scores = self.vectors @ embed([query], "query: ", quiet=True)[0]
        elif method == "hybrid":
            return self._fuse(query, k, lang)
        else:
            raise ValueError(method)

        order = np.argsort(-scores)
        hits = []
        for i in order:
            if lang and self.chunks[i]["lang"] != lang:
                continue
            hits.append((self.chunks[i], float(scores[i])))
            if len(hits) == k:
                break
        return hits

    def _fuse(self, query, k, lang, depth=50, constant=60):
        """Reciprocal rank fusion: trust ranks, not scores, so two retrievers
        on different scales can be added without calibrating anything."""
        fused = {}
        for method in ("bm25", "dense"):
            for rank, (chunk, _) in enumerate(
                    self.search(query, k=depth, method=method, lang=lang)):
                fused[chunk["id"]] = fused.get(chunk["id"], 0.0) + 1.0 / (constant + rank + 1)
        by_id = {c["id"]: c for c in self.chunks}
        best = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [(by_id[cid], score) for cid, score in best]


def format_context(hits):
    """Numbered passages with their source — the shape the model will cite."""
    blocks = []
    for n, (chunk, _) in enumerate(hits, 1):
        blocks.append("[%d] Lecture %d — %s (%s)\n%s"
                      % (n, chunk["lecture"], chunk["heading"],
                         chunk["source"], chunk["text"]))
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# Questions whose answers live in the course and nowhere else.
#
# Each one carries two strings, and they do different jobs:
#   gold    a phrase distinctive enough to identify the right passage. It was
#           chosen by counting: every one of these appears in fewer than 20 of
#           the 1 924 chunks, so "the gold passage was retrieved" means
#           something. A needle like "6" would have matched 871 chunks and
#           scored every retriever a perfect 10/10 while measuring nothing.
#   answer  what the model has to say for its reply to count as correct.
# --------------------------------------------------------------------------
QUESTIONS = [
    ("How many parameters does LoRA train at rank 4 on a 0.5B model?",
     "540,672", "540"),
    ("What relative error did rounding one weight matrix to int8 cause?",
     "1.15%", "1.15"),
    ("How many tokens per word does Kazakh need under the o200k_base tokenizer?",
     "3.17", "3.17"),
    # Lecture 6 measured entropy 1.879 with the scaling and 0.359 without.
    # The gold phrase finds the passage; the answer is the unscaled number.
    ("What was the entropy of the attention weights without the square-root "
     "scaling?", "1.879", "0.359"),
    ("What accuracy did RoBERTa reach on SST-2 in this course?",
     "0.937", "0.937"),
    ("What did a frozen mBERT encoder score on the polarity task?",
     "0.558", "0.558"),
    # "128" alone was a false positive: a closed-book reply said "128 bytes",
    # which is wrong by a factor of 1000 but contains the string.
    ("How much KV cache does an 8B model need per token per sequence?",
     "128 KB", "128 KB"),
    ("What accuracy did kaz-RoBERTa reach with only 250 labelled reviews?",
     "0.685", "0.685"),
    ("128 000 токендік әңгіме KV cache үшін қанша жад қажет етеді?",
     "16.78", "16.7"),
    ("Сколько параметров у модели Qwen2.5-0.5B?",
     "494,032,768", "494"),
]


def compare(retriever, k=5):
    """Per question at the requested k, then a sweep, because the interesting
    difference between retrievers is where in the ranking the answer lands."""
    print("\n%-58s %8s %8s %8s" % ("question (top %d)" % k, "bm25", "dense", "hybrid"))
    print("-" * 88)
    for question, gold, _answer in QUESTIONS:
        row = []
        for method in ("bm25", "dense", "hybrid"):
            hits = retriever.search(question, k=k, method=method)
            row.append("yes" if any(gold in c["text"] for c, _ in hits) else "-")
        print("%-58s %8s %8s %8s" % (question[:58], *row))

    print("\n%-58s %8s %8s %8s" % ("gold passage found within top k", "bm25", "dense", "hybrid"))
    print("-" * 88)
    for depth in (1, 3, 5, 10):
        row = []
        for method in ("bm25", "dense", "hybrid"):
            found = sum(any(gold in c["text"] for c, _ in
                            retriever.search(q, k=depth, method=method))
                        for q, gold, _a in QUESTIONS)
            row.append("%d/%d" % (found, len(QUESTIONS)))
        print("%-58s %8s %8s %8s" % ("  k = %d" % depth, *row))

    print("\n%-58s %8s %8s %8s" % ("mean rank of the gold passage (lower is better)",
                                   "bm25", "dense", "hybrid"))
    print("-" * 88)
    row = []
    for method in ("bm25", "dense", "hybrid"):
        ranks = []
        for q, gold, _a in QUESTIONS:
            hits = retriever.search(q, k=50, method=method)
            place = next((i + 1 for i, (c, _) in enumerate(hits) if gold in c["text"]), 51)
            ranks.append(place)
        row.append("%.1f" % (sum(ranks) / len(ranks)))
    print("%-58s %8s %8s %8s" % ("  over %d questions, 51 = not in top 50" % len(QUESTIONS), *row))


def gold_frequency(chunks):
    """How rare is each gold phrase? Printed so the evaluation can be trusted."""
    print("\n%-16s %s" % ("gold phrase", "chunks containing it (of %d)" % len(chunks)))
    print("-" * 50)
    for _q, gold, _a in QUESTIONS:
        print("%-16s %d" % (repr(gold), sum(gold in c["text"] for c in chunks)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="embed the chunks")
    ap.add_argument("--compare", action="store_true", help="bm25 vs dense vs hybrid")
    ap.add_argument("--query")
    ap.add_argument("--method", default="hybrid", choices=["bm25", "dense", "hybrid"])
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--gold-frequency", action="store_true")
    args = ap.parse_args()

    chunks = load_chunks()
    if args.gold_frequency:
        gold_frequency(chunks)
        return
    if args.build:
        build_vectors(chunks)
        return

    retriever = Retriever(chunks)
    if args.query:
        for n, (chunk, score) in enumerate(
                retriever.search(args.query, k=args.k, method=args.method), 1):
            print("\n[%d] %.4f  Lecture %d — %s  (%s)"
                  % (n, score, chunk["lecture"], chunk["heading"], chunk["source"]))
            print("    " + chunk["text"][:300].replace("\n", " "))
    if args.compare or not args.query:
        compare(retriever, k=args.k)


if __name__ == "__main__":
    main()

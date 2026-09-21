"""Build the knowledge base out of this course's own lecture pages.

Why the course itself? Because we can check the answers. Every number in these
pages was measured in Lectures 3-8, none of it is on the public internet in this
form, and the model has therefore never seen it. If it answers correctly, it
retrieved; if it answers confidently and wrongly, that is a hallucination we can
prove rather than suspect.

The pages exist in English, Kazakh and Russian, so the same index also lets us
ask a Kazakh question and watch it pull an English or Kazakh passage.
"""
import argparse
import html
import json
import os
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
LANG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "language"))
OUT = os.path.join(HERE, "index", "chunks.jsonl")

DROP_TAGS = re.compile(r"<(script|style|svg|nav|footer)\b.*?</\1>", re.S | re.I)
TAG = re.compile(r"<[^>]+>")
SPACES = re.compile(r"[ \t   ]+")
BLANKS = re.compile(r"\n{3,}")

# Roughly a paragraph and a half. Small enough that a hit is specific, large
# enough that the sentence around the number survives.
TARGET_CHARS = 900
OVERLAP_CHARS = 150


def clean(fragment):
    text = DROP_TAGS.sub(" ", fragment)
    text = re.sub(r"</(p|div|li|h[1-6]|tr|pre|figcaption)>", "\n", text, flags=re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = TAG.sub(" ", text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return BLANKS.sub("\n\n", text).strip()


def sections(page):
    """Split one lecture page into (heading, body) pairs at every <h2>."""
    body = page[page.find("<body"):] if "<body" in page else page
    parts = re.split(r"(<h2[^>]*>.*?</h2>)", body, flags=re.S | re.I)
    out, heading = [], None
    for part in parts:
        if re.match(r"<h2", part, re.I):
            heading = clean(part)
            continue
        text = clean(part)
        if heading and len(text) > 120:
            out.append((heading, text))
    return out


def split_long(text):
    """Cut a long section into overlapping chunks, preferring paragraph breaks."""
    if len(text) <= TARGET_CHARS:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        end = min(len(text), start + TARGET_CHARS)
        if end < len(text):
            window = text.rfind("\n", start + TARGET_CHARS // 2, end)
            if window == -1:
                window = text.rfind(". ", start + TARGET_CHARS // 2, end)
            if window != -1:
                end = window + 1
        piece = text[start:end].strip()
        if len(piece) > 80:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - OVERLAP_CHARS)
    return pieces


def build(languages=("en", "kk", "ru")):
    chunks = []
    for lang in languages:
        folder = os.path.join(LANG_DIR, lang)
        for name in sorted(os.listdir(folder)):
            if not re.match(r"^\d\d-", name):
                continue
            lecture = int(name[:2])
            with open(os.path.join(folder, name), encoding="utf-8") as fh:
                page = fh.read()
            title = clean(re.search(r"<title>(.*?)</title>", page, re.S).group(1))
            for heading, text in sections(page):
                for piece in split_long(text):
                    chunks.append({
                        "id": "%s-%02d-%d" % (lang, lecture, len(chunks)),
                        "lang": lang,
                        "lecture": lecture,
                        "source": "%s/%s" % (lang, name),
                        "title": title.split("|")[0].strip(),
                        "heading": heading,
                        "text": piece,
                    })
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", default="en,kk,ru")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    chunks = build(tuple(args.languages.split(",")))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    per_lang, per_lecture = {}, {}
    for c in chunks:
        per_lang[c["lang"]] = per_lang.get(c["lang"], 0) + 1
        per_lecture[c["lecture"]] = per_lecture.get(c["lecture"], 0) + 1
    lengths = sorted(len(c["text"]) for c in chunks)

    print("wrote %d chunks to %s" % (len(chunks), args.out))
    print("  by language :", ", ".join("%s %d" % kv for kv in sorted(per_lang.items())))
    print("  by lecture  :", ", ".join("L%d %d" % kv for kv in sorted(per_lecture.items())))
    print("  chars       : min %d, median %d, max %d" %
          (lengths[0], lengths[len(lengths) // 2], lengths[-1]))


if __name__ == "__main__":
    main()

"""The audio we work with: FLEURS, cached locally as plain .wav files.

FLEURS is read aloud from the same source sentences in every language it
covers, so Kazakh, Russian and English clips are comparable in content and
difficulty. That is what makes "Whisper is worse at Kazakh" a measurement
rather than an impression.

    python3 data.py --fetch            # 12 clips per language, once
    python3 data.py --list
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(HERE, "audio")

# FLEURS names its configs language_region.
LANGUAGES = {"kk": "kk_kz", "ru": "ru_ru", "en": "en_us"}
INDEX = os.path.join(AUDIO, "index.json")


def fetch(languages=("kk", "ru", "en"), n=12, split="test", max_seconds=20.0):
    """Stream a few clips per language and write them out as 16 kHz wav."""
    import soundfile as sf
    from datasets import load_dataset

    os.makedirs(AUDIO, exist_ok=True)
    index = load_index()
    for lang in languages:
        config = LANGUAGES[lang]
        print("fetching %s (%s) ..." % (lang, config))
        stream = load_dataset("google/fleurs", config, split=split, streaming=True)
        kept = 0
        for row in stream:
            audio = row["audio"]
            seconds = len(audio["array"]) / audio["sampling_rate"]
            # Long clips make Whisper chunk internally, which is a separate
            # topic; keep the measurement about recognition.
            if seconds > max_seconds:
                continue
            name = "%s-%02d.wav" % (lang, kept)
            sf.write(os.path.join(AUDIO, name), audio["array"],
                     audio["sampling_rate"])
            index[name] = {
                "lang": lang,
                "file": name,
                "seconds": round(seconds, 2),
                "reference": row["transcription"],
                "raw_reference": row.get("raw_transcription", ""),
                "source": "google/fleurs %s %s" % (config, split),
            }
            kept += 1
            if kept >= n:
                break
        print("  kept %d clips" % kept)
    save_index(index)
    return index


def load_index():
    if os.path.exists(INDEX):
        with open(INDEX, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_index(index):
    with open(INDEX, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=1)


def clips(lang=None):
    """Every cached clip, optionally for one language, in a stable order."""
    index = load_index()
    if not index:
        raise SystemExit("No audio yet. Run:  python3 data.py --fetch")
    rows = [v for v in index.values() if lang is None or v["lang"] == lang]
    return sorted(rows, key=lambda r: r["file"])


def path(clip):
    return os.path.join(AUDIO, clip["file"])


def load_audio(clip, target_sr=16000):
    """Return (samples, sample_rate) as float32 mono at target_sr."""
    import librosa
    samples, sr = librosa.load(path(clip), sr=target_sr, mono=True)
    return samples, sr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("-n", type=int, default=12, help="clips per language")
    ap.add_argument("--languages", default="kk,ru,en")
    args = ap.parse_args()

    if args.fetch:
        fetch(tuple(args.languages.split(",")), n=args.n)
    index = load_index()
    if args.list or not args.fetch:
        by_lang = {}
        for row in index.values():
            by_lang.setdefault(row["lang"], []).append(row)
        print("\n%-10s %6s %9s %9s" % ("language", "clips", "seconds", "mean s"))
        print("-" * 38)
        for lang, rows in sorted(by_lang.items()):
            total = sum(r["seconds"] for r in rows)
            print("%-10s %6d %9.1f %9.1f"
                  % (lang, len(rows), total, total / len(rows)))
        print("\nstored in %s" % AUDIO)


if __name__ == "__main__":
    main()

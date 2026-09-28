"""ASR with Whisper: transcribe, and measure how much it gets wrong.

Every number the lecture quotes comes out of this file.

    python3 data.py --fetch      # once
    python3 asr.py --demo        # one clip, shown in full
    python3 asr.py --languages   # Kazakh against Russian against English
    python3 asr.py --sizes       # tiny / base / small on Kazakh
    python3 asr.py --language-token   # what happens if you name the wrong language
    python3 asr.py --all
"""
import argparse
import re
import time
import unicodedata

import torch

import data

MODELS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
}
DEFAULT = "small"

_cache = {}


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load(size=DEFAULT):
    """Whisper is an encoder-decoder: the encoder reads a spectrogram, the
    decoder writes text. Lecture 6's architecture, with audio on the front."""
    if size not in _cache:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        name = MODELS[size]
        processor = WhisperProcessor.from_pretrained(name)
        model = WhisperForConditionalGeneration.from_pretrained(name)
        model.to(device()).eval()
        _cache[size] = (processor, model)
    return _cache[size]


def transcribe(samples, sr=16000, size=DEFAULT, language="kk", task="transcribe",
               max_new_tokens=220):
    processor, model = load(size)
    features = processor(samples, sampling_rate=sr,
                         return_tensors="pt").input_features.to(device())
    started = time.time()
    with torch.no_grad():
        ids = model.generate(features, language=language, task=task,
                             max_new_tokens=max_new_tokens)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return text, time.time() - started


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
SPACE = re.compile(r"\s+")


def normalise(text):
    """Lowercase, drop punctuation, collapse spaces. Without this, WER mostly
    measures whether the model felt like writing a comma."""
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = PUNCT.sub(" ", text)
    return SPACE.sub(" ", text).strip()


def score(reference, hypothesis):
    """Word error rate and character error rate, both after normalisation."""
    import jiwer
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return None, None
    return jiwer.wer(ref, hyp), jiwer.cer(ref, hyp)


def run_set(clips, size=DEFAULT, language=None, label=""):
    """Transcribe a list of clips and return the aggregate numbers."""
    wers, cers, seconds, audio_seconds = [], [], [], []
    rows = []
    for clip in clips:
        samples, sr = data.load_audio(clip)
        lang = language or clip["lang"]
        text, took = transcribe(samples, sr, size=size, language=lang)
        wer, cer = score(clip["reference"], text)
        wers.append(wer); cers.append(cer)
        seconds.append(took); audio_seconds.append(clip["seconds"])
        rows.append((clip, text, wer, cer, took))
    n = len(wers)
    return {
        "label": label, "n": n,
        "wer": sum(wers) / n, "cer": sum(cers) / n,
        "seconds": sum(seconds),
        "audio_seconds": sum(audio_seconds),
        "rtf": sum(seconds) / sum(audio_seconds),
        "rows": rows,
    }


def header(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# --------------------------------------------------------------------------
def demo(size=DEFAULT):
    header("1. ONE CLIP, IN FULL")
    clip = data.clips("kk")[0]
    samples, sr = data.load_audio(clip)
    print("  file      %s  (%.1f s of Kazakh, %s)"
          % (clip["file"], clip["seconds"], clip["source"]))
    text, took = transcribe(samples, sr, size=size, language="kk")
    wer, cer = score(clip["reference"], text)
    print("\n  reference : %s" % clip["reference"])
    print("\n  whisper-%s: %s" % (size, text))
    print("\n  WER %.3f   CER %.3f   %.1f s for %.1f s of audio (RTF %.2f)"
          % (wer, cer, took, clip["seconds"], took / clip["seconds"]))
    print("\n  word by word, after normalisation:")
    ref = normalise(clip["reference"]).split()
    hyp = normalise(text).split()
    import difflib
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ref, b=hyp).get_opcodes():
        if tag == "equal":
            continue
        print("    %-8s %-34s -> %s"
              % (tag, " ".join(ref[i1:i2])[:34] or "-",
                 " ".join(hyp[j1:j2])[:34] or "-"))


def languages(size=DEFAULT):
    header("2. THE SAME MODEL ON THREE LANGUAGES")
    print("  FLEURS reads the same source sentences in every language,")
    print("  so these three columns are comparable.\n")
    print("  %-10s %6s %8s %8s %10s %8s"
          % ("language", "clips", "WER", "CER", "audio s", "RTF"))
    print("  " + "-" * 56)
    out = {}
    for lang in ("en", "ru", "kk"):
        r = run_set(data.clips(lang), size=size, label=lang)
        out[lang] = r
        print("  %-10s %6d %8.3f %8.3f %10.1f %8.2f"
              % (lang, r["n"], r["wer"], r["cer"], r["audio_seconds"], r["rtf"]))
    if out["en"]["wer"] > 0:
        print("\n  Kazakh WER is %.1fx the English WER on the same sentences."
              % (out["kk"]["wer"] / out["en"]["wer"]))
    return out


def sizes(lang="kk"):
    header("3. MODEL SIZE AGAINST ACCURACY  (%s)" % lang)
    print("  %-8s %12s %8s %8s %8s"
          % ("model", "parameters", "WER", "CER", "RTF"))
    print("  " + "-" * 48)
    params = {"tiny": "39 M", "base": "74 M", "small": "244 M"}
    for size in ("tiny", "base", "small"):
        r = run_set(data.clips(lang), size=size, label=size)
        print("  %-8s %12s %8.3f %8.3f %8.2f"
              % (size, params[size], r["wer"], r["cer"], r["rtf"]))


def language_token(size=DEFAULT):
    header("4. THE LANGUAGE TOKEN IS NOT A HINT")
    print("  The same Kazakh audio, decoded while telling Whisper it is")
    print("  listening to three different languages.\n")
    clip = data.clips("kk")[0]
    samples, sr = data.load_audio(clip)
    print("  reference : %s\n" % clip["reference"][:110])
    for lang in ("kk", "ru", "tr", "en"):
        text, _ = transcribe(samples, sr, size=size, language=lang)
        wer, _ = score(clip["reference"], text)
        print("  language=%-3s WER %.3f  %s" % (lang, wer, text[:96]))
    print("\n  It is a token the decoder is forced to start from, so naming the")
    print("  wrong one does not degrade the output politely: it transliterates.")


def translate(size=DEFAULT):
    header("5. TRANSCRIBE AGAINST TRANSLATE")
    clip = data.clips("kk")[0]
    samples, sr = data.load_audio(clip)
    for task in ("transcribe", "translate"):
        text, took = transcribe(samples, sr, size=size, language="kk", task=task)
        print("\n  task=%-11s %.1f s\n    %s" % (task, took, text[:190]))
    print("\n  Whisper translates only into English, and it is one model doing")
    print("  both jobs from the same encoder output.")


SECTIONS = {"demo": demo, "languages": languages, "sizes": sizes,
            "language-token": language_token, "translate": translate}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in SECTIONS:
        ap.add_argument("--" + name, action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--size", default=DEFAULT, choices=list(MODELS))
    args = ap.parse_args()

    chosen = [n for n in SECTIONS if getattr(args, n.replace("-", "_"))]
    if args.all or not chosen:
        chosen = list(SECTIONS)
    print("device: %s" % device())
    for name in chosen:
        if name == "sizes":
            SECTIONS[name]()
        else:
            SECTIONS[name](size=args.size)


if __name__ == "__main__":
    main()

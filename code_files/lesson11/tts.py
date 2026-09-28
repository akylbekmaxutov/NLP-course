"""TTS with OmniVoice: three ways to choose a voice, and a way to score it.

Judging a synthesiser by ear does not scale and does not survive an argument.
So we close the loop: speak a sentence, transcribe it back with Whisper, and
measure how much survived. That number is not a pure measure of the voice --
it also contains the recogniser's own errors -- which is exactly why the same
table carries the recogniser's error on human speech next to it.

    python3 tts.py --modes        # auto, voice design, voice clone
    python3 tts.py --roundtrip    # text -> speech -> text, per language
    python3 tts.py --clone        # clone a FLEURS speaker
    python3 tts.py --all
"""
import argparse
import os
import time

import torch

import data

MODEL = "k2-fsa/OmniVoice"
SAMPLE_RATE = 24000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")

_model = None

# Short sentences with the same meaning in three languages, so the round-trip
# numbers below compare like with like.
SENTENCES = {
    "kk": [
        "Сәлеметсіз бе! Бүгін Астанада ауа райы жақсы.",
        "Алматыдан Шымкентке дейінгі жол тоғыз сағат.",
        "Кітапханада бүгін көрме өтеді.",
        "Менің атым Айгүл, мен Қарағандыданмын.",
    ],
    "ru": [
        "Здравствуйте! Сегодня в Астане хорошая погода.",
        "Дорога из Алматы в Шымкент занимает девять часов.",
        "Сегодня в библиотеке проходит выставка.",
        "Меня зовут Айгуль, я из Караганды.",
    ],
    "en": [
        "Hello! The weather in Astana is good today.",
        "The road from Almaty to Shymkent takes nine hours.",
        "There is an exhibition at the library today.",
        "My name is Aigul and I am from Karaganda.",
    ],
}

LANGUAGE_NAMES = {"kk": "Kazakh", "ru": "Russian", "en": "English"}

# Voice design does not take free prose. These are the only accepted English
# attributes; anything else is rejected with a list of what is allowed. It is
# a controlled vocabulary, in the sense Lecture 10 gave that phrase.
DESIGN_VOCABULARY = [
    "female", "male", "child", "teenager", "young adult", "middle-aged",
    "elderly", "very low pitch", "low pitch", "moderate pitch", "high pitch",
    "very high pitch", "whisper", "american accent", "australian accent",
    "british accent", "canadian accent", "chinese accent", "indian accent",
    "japanese accent", "korean accent", "portuguese accent", "russian accent",
]


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load():
    """OmniVoice is a diffusion language model over audio tokens, built on
    Qwen3-0.6B, with a separate audio tokenizer that turns tokens into sound."""
    global _model
    if _model is None:
        from omnivoice import OmniVoice
        _model = OmniVoice.from_pretrained(MODEL, device_map=device(),
                                           dtype=torch.float32)
    return _model


def speak(text, language=None, ref_audio=None, ref_text=None, instruct=None,
          out=None):
    """One sentence to one wav file. Returns (path, seconds, generation_time)."""
    model = load()
    kwargs = {"text": text}
    if language:
        kwargs["language"] = LANGUAGE_NAMES.get(language, language)
    if instruct:
        kwargs["instruct"] = instruct
    if ref_audio is not None:
        kwargs["ref_audio"] = ref_audio
        kwargs["ref_text"] = ref_text

    started = time.time()
    audio = model.generate(**kwargs)[0]
    took = time.time() - started

    seconds = len(audio) / SAMPLE_RATE
    if out:
        import soundfile as sf
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, out)
        sf.write(path, audio, SAMPLE_RATE)
    else:
        path = None
    return path, seconds, took


def header(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# --------------------------------------------------------------------------
def modes():
    header("1. THREE WAYS TO CHOOSE A VOICE")
    text = SENTENCES["kk"][0]
    print("  text: %s\n" % text)

    print("  auto        — the model picks a voice")
    p, s, t = speak(text, "kk", out="mode-auto.wav")
    print("     %s  %.2f s of audio in %.1f s\n" % (os.path.basename(p), s, t))

    print("  voice design — the voice is chosen from a fixed vocabulary of")
    print("                 attributes, not described in free prose:")
    print("                 %s\n" % ", ".join(DESIGN_VOCABULARY[:9]))
    for name, instruct in [
        ("design-young-female.wav", "female, young adult, high pitch"),
        ("design-elderly-male.wav", "male, elderly, low pitch"),
    ]:
        p, s, t = speak(text, "kk", instruct=instruct, out=name)
        print("     %-18s %.2f s in %.1f s   «%s»"
              % (os.path.basename(p), s, t, instruct[:52]))

    print("\n  voice clone  — copy a speaker from a few seconds of their audio")
    clip = data.clips("kk")[0]
    p, s, t = speak(text, "kk", ref_audio=data.path(clip),
                    ref_text=clip["reference"], out="clone-fleurs.wav")
    print("     reference  %s (%.1f s, a FLEURS reader)"
          % (clip["file"], clip["seconds"]))
    print("     %-18s %.2f s in %.1f s" % (os.path.basename(p), s, t))
    print("\n  Listen to the three files in audio/ and the difference is obvious.")
    print("  Only the third one needed any audio at all.")


def roundtrip(languages=("en", "ru", "kk")):
    header("2. ROUND TRIP: TEXT -> SPEECH -> TEXT")
    import asr
    print("  Speak a sentence with OmniVoice, transcribe it back with Whisper,")
    print("  and score the result against what we asked for. The last column is")
    print("  Whisper's error on *human* FLEURS speech in the same language, so")
    print("  the two error sources can be told apart.\n")
    print("  %-10s %7s %8s %8s %8s %12s"
          % ("language", "lines", "WER", "CER", "RTF", "human WER"))
    print("  " + "-" * 58)

    human = {"en": 0.099, "ru": 0.108, "kk": 0.744}   # measured by asr.py
    results = {}
    for lang in languages:
        wers, cers, gen, dur = [], [], [], []
        for i, text in enumerate(SENTENCES[lang]):
            path, seconds, took = speak(text, lang, out="rt-%s-%d.wav" % (lang, i))
            import librosa
            samples, sr = librosa.load(path, sr=16000, mono=True)
            back, _ = asr.transcribe(samples, sr, size="small", language=lang)
            wer, cer = asr.score(text, back)
            wers.append(wer); cers.append(cer); gen.append(took); dur.append(seconds)
        results[lang] = (sum(wers) / len(wers), sum(cers) / len(cers),
                         sum(gen) / sum(dur))
        print("  %-10s %7d %8.3f %8.3f %8.2f %12.3f"
              % (lang, len(wers), results[lang][0], results[lang][1],
                 results[lang][2], human[lang]))
    print("\n  Read the Kazakh row against the last column, not against zero.")
    return results


def clone():
    header("3. CLONING A VOICE, AND WHAT IT NEEDS")
    import asr
    clip = data.clips("kk")[0]
    samples, sr = data.load_audio(clip)
    print("  reference clip : %s (%.1f s)" % (clip["file"], clip["seconds"]))
    print("  true transcript: %s" % clip["reference"][:80])

    heard, _ = asr.transcribe(samples, sr, size="small", language="kk")
    print("  whisper heard  : %s" % heard[:80])

    text = SENTENCES["kk"][1]
    print("\n  Now make that speaker say something new: «%s»\n" % text)
    for label, ref_text, name in [
        ("true transcript   ", clip["reference"], "clone-true.wav"),
        ("whisper transcript", heard, "clone-whisper.wav"),
    ]:
        p, s, t = speak(text, "kk", ref_audio=data.path(clip),
                        ref_text=ref_text, out=name)
        back, _ = asr.transcribe(*__import__("librosa").load(p, sr=16000, mono=True),
                                 size="small", language="kk")
        wer, cer = asr.score(text, back)
        print("  ref_text = %s  ->  %-18s %.2f s   round-trip WER %.3f CER %.3f"
              % (label, os.path.basename(p), s, wer, cer))
    print("\n  Cloning needs the reference *text* as well as the audio, and in a")
    print("  language Whisper is weak at, the transcript you feed it is itself")
    print("  a guess. That is the ASR error leaking into the TTS.")


SECTIONS = {"modes": modes, "roundtrip": roundtrip, "clone": clone}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in SECTIONS:
        ap.add_argument("--" + name, action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--say", help="just say this and exit")
    ap.add_argument("--language", default="kk")
    args = ap.parse_args()

    print("device: %s" % device())
    if args.say:
        p, s, t = speak(args.say, args.language, out="say.wav")
        print("wrote %s  (%.2f s of audio in %.1f s)" % (p, s, t))
        return
    chosen = [n for n in SECTIONS if getattr(args, n)]
    for name in (chosen or list(SECTIONS)):
        SECTIONS[name]()


if __name__ == "__main__":
    main()

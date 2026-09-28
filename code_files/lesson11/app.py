"""Speech in the browser: talk to it, read what it heard, hear it answer.

    python3 app.py        # http://127.0.0.1:7862

Three tabs, matching the lecture:
  Transcribe   audio in, text out, with the language token exposed
  Speak        text in, audio out, with the three ways to pick a voice
  Round trip   both at once, scored, which is the only honest way to judge
"""
import argparse
import os

import gradio as gr

import asr
import data
import tts

CSS = """
.mono textarea { font-family: ui-monospace, Menlo, monospace; font-size: 13px; }
footer { display: none !important; }
"""

LANGS = ["kk", "ru", "en", "tr", "uz", "ky"]


def examples():
    try:
        return [[data.path(c), c["lang"]] for c in
                (data.clips("kk")[:2] + data.clips("ru")[:1] + data.clips("en")[:1])]
    except SystemExit:
        return []


# --------------------------------------------------------------------------
def do_transcribe(audio_path, language, size, task):
    if not audio_path:
        return "", ""
    import librosa
    samples, sr = librosa.load(audio_path, sr=16000, mono=True)
    text, took = asr.transcribe(samples, sr, size=size, language=language, task=task)
    seconds = len(samples) / sr
    note = ("`%s` · language token `%s` · %.1f s of audio in %.1f s "
            "(RTF %.2f)" % (size, language, seconds, took, took / max(seconds, 1e-6)))
    return text, note


def transcribe_tab():
    with gr.Tab("Transcribe"):
        gr.Markdown(
            "Whisper is an encoder–decoder: the encoder reads a spectrogram, "
            "the decoder writes text. **The language is a token the decoder is "
            "forced to start from**, not a hint — set it wrong and you get a "
            "confident transliteration rather than a worse transcript.")
        with gr.Row():
            with gr.Column(scale=3):
                audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                 label="Audio")
                out = gr.Textbox(lines=5, label="Transcript", elem_classes="mono")
                note = gr.Markdown()
            with gr.Column(scale=1):
                language = gr.Dropdown(LANGS, value="kk", label="language token")
                size = gr.Radio(list(asr.MODELS), value="small", label="model")
                task = gr.Radio(["transcribe", "translate"], value="transcribe",
                                label="task (translate goes to English only)")
                run = gr.Button("Transcribe", variant="primary")
        ex = examples()
        if ex:
            gr.Examples(ex, inputs=[audio, language], label="FLEURS clips")
        run.click(do_transcribe, [audio, language, size, task], [out, note])


# --------------------------------------------------------------------------
def do_speak(text, language, mode, instruct, ref_audio, ref_text):
    if not text.strip():
        return None, ""
    kwargs = {}
    if mode == "voice design":
        kwargs["instruct"] = instruct
    elif mode == "voice clone":
        if not ref_audio:
            return None, "**Voice clone needs a reference clip and its transcript.**"
        kwargs["ref_audio"] = ref_audio
        kwargs["ref_text"] = ref_text
    try:
        path, seconds, took = tts.speak(text, language, out="app-out.wav", **kwargs)
    except ValueError as exc:
        return None, "**Rejected:** %s" % str(exc).split("\n")[0]
    return path, ("%.2f s of audio in %.1f s (RTF %.2f) · %s"
                  % (seconds, took, took / seconds, mode))


def speak_tab():
    with gr.Tab("Speak"):
        gr.Markdown(
            "OmniVoice picks a voice in one of three ways. **Voice design uses "
            "a controlled vocabulary, not free prose** — `female, elderly, "
            "low pitch` works, a sentence describing a mood does not.")
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox("Сәлеметсіз бе! Бүгін Астанада ауа райы жақсы.",
                                  lines=3, label="Text")
                audio_out = gr.Audio(label="Speech", type="filepath")
                note = gr.Markdown()
            with gr.Column(scale=2):
                language = gr.Dropdown(LANGS, value="kk", label="language")
                mode = gr.Radio(["auto", "voice design", "voice clone"],
                                value="auto", label="how to pick the voice")
                instruct = gr.Textbox("female, young adult, high pitch",
                                      label="voice design attributes")
                gr.Markdown("*allowed:* " + ", ".join(tts.DESIGN_VOCABULARY))
                ref_audio = gr.Audio(sources=["upload", "microphone"],
                                     type="filepath", label="reference clip")
                ref_text = gr.Textbox(lines=2, label="what the reference says")
                run = gr.Button("Speak", variant="primary")
        run.click(do_speak, [text, language, mode, instruct, ref_audio, ref_text],
                  [audio_out, note])


# --------------------------------------------------------------------------
def do_roundtrip(text, language):
    if not text.strip():
        return None, "", ""
    import librosa
    path, seconds, took = tts.speak(text, language, out="app-rt.wav")
    samples, sr = librosa.load(path, sr=16000, mono=True)
    heard, _ = asr.transcribe(samples, sr, size="small", language=language)
    wer, cer = asr.score(text, heard)
    human = {"en": 0.099, "ru": 0.108, "kk": 0.744}.get(language)
    note = "**WER %.3f · CER %.3f**" % (wer, cer)
    if human is not None:
        note += ("  — Whisper's error on *human* FLEURS speech in this "
                 "language is %.3f, so read this against that, not against 0."
                 % human)
    return path, heard, note


def roundtrip_tab():
    with gr.Tab("Round trip"):
        gr.Markdown(
            "Say it, then listen to it back with Whisper and score what "
            "survived. The number is not a pure measure of the voice — it "
            "carries the recogniser's own errors too, which is why the "
            "recogniser's error on human speech is quoted beside it.")
        text = gr.Textbox("Алматыдан Шымкентке дейінгі жол тоғыз сағат.",
                          lines=2, label="Text to speak")
        language = gr.Dropdown(LANGS, value="kk", label="language")
        run = gr.Button("Speak, then transcribe", variant="primary")
        audio_out = gr.Audio(label="What OmniVoice said", type="filepath")
        heard = gr.Textbox(lines=3, label="What Whisper heard back",
                           elem_classes="mono")
        note = gr.Markdown()
        run.click(do_roundtrip, [text, language], [audio_out, heard, note])


def build():
    with gr.Blocks(title="Lecture 11 — ASR and TTS") as demo:
        gr.Markdown("# Lecture 11 — Speech: ASR and TTS\n"
                    "`openai/whisper-small` for recognition, `k2-fsa/OmniVoice` "
                    "for synthesis. Both run locally on this machine.")
        transcribe_tab()
        speak_tab()
        roundtrip_tab()
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7862)
    args = ap.parse_args()
    build().launch(server_name="127.0.0.1", server_port=args.port,
                   css=CSS, theme=gr.themes.Soft())

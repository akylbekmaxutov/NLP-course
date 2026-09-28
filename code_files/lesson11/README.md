# Lecture 11 — Speech: ASR and TTS

Two models, both running locally, no API key:

- **ASR** — `openai/whisper-small` (244 M), recognition
- **TTS** — `k2-fsa/OmniVoice` (Qwen3-0.6B + an audio tokenizer), synthesis in 651 languages including Kazakh

The audio is **FLEURS**, which is the same source sentences read aloud in every
language it covers. That is what makes "Whisper is worse at Kazakh" a
measurement instead of an impression.

## Install

```bash
brew install ffmpeg
pip install -r requirements.txt
```

## Run

```bash
python3 data.py --fetch        # 12 clips each of kk / ru / en, once (~13 MB)

python3 asr.py --all           # transcription and its errors
python3 tts.py --all           # synthesis, three voice modes, round trip
python3 app.py                 # http://127.0.0.1:7862
```

## What each file does

| file | what it is |
|---|---|
| `data.py` | streams FLEURS and caches clips as plain 16 kHz `.wav` with transcripts |
| `asr.py` | Whisper: transcribe, WER/CER, three languages, three model sizes, the language token, translate mode |
| `tts.py` | OmniVoice: auto / voice design / voice clone, and the round-trip score |
| `app.py` | the browser lab: Transcribe, Speak, Round trip |
| `audio/` | the cached clips and everything the scripts generate |

## The measurements

Whisper-small, 12 FLEURS clips per language, on an Apple M4 Pro:

| language | WER | CER | real-time factor |
|---|---|---|---|
| English | 0.099 | 0.042 | 0.02 |
| Russian | 0.108 | 0.028 | 0.02 |
| **Kazakh** | **0.744** | **0.180** | 0.03 |

Kazakh WER is 7.5× English on the same sentences — but CER is only 4.3×. The
model hears roughly the right *sounds* and writes the wrong *words*. That gap
between WER and CER is the whole story of a low-resource language in a model
trained mostly on something else.

Model size, on Kazakh:

| model | parameters | WER | CER |
|---|---|---|---|
| tiny | 39 M | 2.099 | 0.894 |
| base | 74 M | 0.934 | 0.322 |
| small | 244 M | 0.744 | 0.180 |

A WER above 1.0 is not a bug: `tiny` inserts more words than the reference
contains.

Round trip — speak it with OmniVoice, transcribe it back with Whisper:

| language | round-trip WER | Whisper on *human* speech |
|---|---|---|
| English | 0.083 | 0.099 |
| Russian | 0.146 | 0.108 |
| Kazakh | 0.787 | 0.744 |

Synthetic English is *easier* for Whisper than human English. So the Kazakh
0.787 is almost entirely the recogniser's error, not the synthesiser's — you
can only see that because the English control is there.

## Two things worth knowing before you teach it

**The language token is not a hint.** It is the token the decoder is forced to
start from. Give Kazakh audio to Whisper as `ru`, `tr` or `en` and it does not
degrade gracefully; it transliterates, fluently and wrongly (WER 0.762 → 0.905
→ 1.095 → 1.095).

**Voice design uses a controlled vocabulary, not prose.** `female, elderly,
low pitch` works. "A calm, warm voice speaking slowly" is rejected, with a list
of the 23 attributes it will accept. Structure is not the same as vocabulary —
the same lesson Lecture 10 found in JSON schemas.

**Errors propagate.** Cloning a voice needs the reference audio *and its
transcript*. Feed it the true transcript and the clone scores 0.667; feed it
Whisper's own transcript of the same clip and it scores 1.000.

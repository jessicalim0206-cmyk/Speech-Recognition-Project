# SpeakReady

Bilingual (English / Bahasa Indonesia) interview practice with **local, streaming speech recognition**.
In local mode (`streamlit run app.py`) speech recognition runs on your own computer with an open-source model, and the audio is not sent to any cloud speech API or other server.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Both languages use the same model, `faster-whisper-small` (about 486 MB). On the first run the app downloads it from Hugging Face into `models/faster-whisper-small`
(internet needed once; the folder is checked for missing or truncated files and repaired automatically). To download it manually:
`huggingface-cli download Systran/faster-whisper-small --local-dir models/faster-whisper-small`

## Run
```bash
streamlit run app.py
```
Open the URL in Chrome or Edge on the same computer (microphone access works on `localhost`). Allow the microphone,
press START, and speak. The first request loads the model, which takes a few seconds.
WebRTC is configured with a public STUN server (`stun.l.google.com`) that is only used to discover network addresses for the connection; the audio does not pass through it. Running on another machine or behind a strict network may also need a TURN server.

## Tests (no model or microphone needed)
```bash
python tests/test_core.py     # streaming session, resampling, noise/speed conditions (uses a FAKE backend, not ASR)
python tests/smoke_ui.py      # every page renders, friendly state when a model is missing
```

## Evaluation and report material
[evaluation/README.md](evaluation/README.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
[docs/USER_TESTING.md](docs/USER_TESTING.md), [docs/AI_USAGE_LOG.md](docs/AI_USAGE_LOG.md).

## What is implemented
- Mic audio streamed via WebRTC, resampled and queued in ~chunks, recognized by a worker thread while you speak.
- Live partial text, finalized segments, real input-level meter, elapsed timer, clear/restart, next question.
- English and Indonesian both run on faster-whisper-small (CPU, int8) with voice activity filtering. Whisper is not a streaming model, so partial text comes from re-decoding the audio buffered since the last pause about every 1.5 s; a ~0.8 s pause finalizes the segment. Language is fixed per session; switching starts a clean one.
- Rule-based STAR / relevance / length feedback in both languages.
- WER (S/D/I) and latency scripts with noise and speed conditions; Developer tests page shows only saved real results.
- Friendly states: model missing, load failure, no audio reaching the app, no speech detected, recognition error, empty review; mic permission/device messages come from the WebRTC component (translated).

## Limitations (be honest in your report)
- The WER, latency and user-testing results are not part of the code: they come from your own recordings and participants (see `evaluation/README.md` and `docs/USER_TESTING.md`). Report only numbers you measured yourself.
- Both languages use Whisper, so recognition is chunked (partial text about every 1.5 s), not truly word-by-word streaming; text can be revised when a later chunk is decoded.
- Recognition speed depends on your CPU. On a slow computer the partial text lags behind the speech (the "Jeda proses" line under the timer shows the most recent delay).
- Feedback is a phrase checklist (rules), not AI: it can miss good answers and does not judge delivery, pronunciation, confidence or suitability.
- Fonts load from Google Fonts when online and fall back to system fonts offline.

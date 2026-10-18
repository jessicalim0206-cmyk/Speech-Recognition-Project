# SpeakReady

Bilingual (English / Bahasa Indonesia) interview practice with **local, streaming speech recognition**.
Audio never leaves your computer: no cloud speech APIs.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Install the models: see [models/README.md](models/README.md) (English needs Vosk; Indonesian needs faster-whisper by default).
`faster-whisper` is only imported when an Indonesian session starts.

## Run
```bash
streamlit run app.py
```
Open the URL in Chrome or Edge on the same computer (microphone access works on `localhost`). Allow the microphone,
press START, and speak. The first request loads the model, which takes a few seconds.
WebRTC is configured without an external STUN server, which is fine locally; deploying to another machine needs a TURN/STUN setup.

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
- English (Vosk, true streaming) and Indonesian (faster-whisper, chunked partials every ~1.5 s). Language is fixed per session; switching starts a clean one.
- Rule-based STAR / relevance / length feedback in both languages.
- WER (S/D/I) and latency scripts with noise and speed conditions; Developer tests page shows only saved real results.
- Friendly states: model missing, load failure, no audio reaching the app, no speech detected, recognition error, empty review; mic permission/device messages come from the WebRTC component (translated).

## Not verified / limitations (be honest in your report)
- **The ASR models could not be downloaded in the environment where this code was written**, so the Vosk and Whisper back-ends and the live microphone path have NOT been run end to end here. The streaming session, resampling, conditions, feedback and UI pages were tested with a fake backend and headless UI tests. Run the app and the evaluation yourself before reporting anything.
- No browser screenshots were taken, so the visual design has not been inspected; check it in the browser and adjust `assets/styles.css`.
- Whether a Vosk Indonesian model exists was not verified; Indonesian is not truly word-by-word streaming with the Whisper default.
- Feedback is a phrase checklist: it can miss good answers and does not judge delivery, pronunciation, confidence or suitability.
- Fonts load from Google Fonts when online and fall back to system fonts offline.
- No WER, latency or user-testing results are included. You must collect them.

# Local speech models

Models are not bundled. They are downloaded once and then used fully offline.
Folder names must match exactly (or set `SPEAKREADY_MODELS_DIR`).

## English: Vosk (true streaming)
1. Download `vosk-model-small-en-us-0.15.zip` from https://alphacephei.com/vosk/models
2. Unzip it so you have `models/vosk-model-small-en-us-0.15/`

## Bahasa Indonesia: faster-whisper (chunked, default)
Check https://alphacephei.com/vosk/models first: if a Vosk Indonesian model is listed, you can use it for
true streaming (`SPEAKREADY_ID_BACKEND=vosk SPEAKREADY_ID_FOLDER=<folder>`). I could not verify this
list from the build environment, so the default is the Whisper-family model, which is known to support Indonesian.

    pip install huggingface_hub
    huggingface-cli download Systran/faster-whisper-small --local-dir models/faster-whisper-small

Whisper is not a streaming model. SpeakReady re-decodes the buffered speech about every 1.5 s, so Indonesian
partial text updates in short bursts instead of word by word. Report this difference in your evaluation.

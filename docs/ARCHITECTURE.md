# System architecture

```mermaid
flowchart LR
  MIC[Browser microphone] -->|WebRTC audio, SENDONLY| CB[audio_capture.py<br/>resample to 16 kHz mono PCM16]
  CB -->|push_audio, fast| Q[(chunk queue)]
  Q --> W[Worker thread<br/>speech_recognition.py]
  MD["models/faster-whisper-small<br/>checked on start, downloaded if missing"] --> H
  W --> H["faster-whisper small<br/>CPU, int8, VAD<br/>English or Indonesian"]
  H -->|"partial every ~1.5 s, final after ~0.8 s pause"| S[Transcript state<br/>partial + finals + levels + latencies]
  S -->|polled every 150 ms| UI[Streamlit UI<br/>interview_workspace.py]
  S -->|final transcript| F[answer_feedback.py<br/>rule-based STAR]
  F --> R[feedback_panel.py]
  FILES[WAV files + manifest.csv] --> E[evaluation/*.py<br/>same recognizer]
  E --> J[results/*.json] --> DEV[dev_panel.py]
```

**Threads.** The WebRTC thread only converts audio and enqueues it, so it never blocks on the model.
One worker thread per attempt runs inference. The Streamlit script thread only reads snapshots.

**Partial transcription.** Both languages use faster-whisper-small; the language code selects English or Indonesian.
Whisper is not a streaming model, so the recognizer buffers the incoming audio and re-decodes the buffer since the last pause
every 1.5 s, which updates the partial text while the user is still speaking. A pause of about 0.8 s, or 12 s of continuous
speech, finalizes the segment and starts a new buffer. Silence is never decoded (a loudness gate plus faster-whisper's voice
activity filter), because Whisper invents text when given silence. If the worker falls behind (more than ~0.5 s of queued audio),
partial decodes are skipped until it catches up, so the queue does not grow. Audio is processed continuously during speech and
no complete recording is needed first, but updates arrive in chunks, not word by word, and earlier words can change when a
later chunk is decoded.

**Latency (live app).** Time from `push_audio` of a chunk to the transcript text changing because of that chunk
(queue wait + inference). Browser capture, WebRTC transport and rendering are excluded.
**Latency (offline script).** Per-chunk `feed()` time, time-to-first-partial, final-flush delay and real-time factor.

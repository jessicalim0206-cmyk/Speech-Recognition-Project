# System architecture

```mermaid
flowchart LR
  MIC[Browser microphone] -->|WebRTC audio, SENDONLY| CB[audio_capture.py<br/>resample to 16 kHz mono PCM16]
  CB -->|push_audio, fast| Q[(chunk queue)]
  Q --> W[Worker thread<br/>speech_recognition.py]
  W --> B{Back-end}
  B -->|English| V[Vosk, streaming partials]
  B -->|Indonesian| H[faster-whisper, chunked partials]
  V --> S[Transcript state<br/>partial + finals + levels + latencies]
  H --> S
  S -->|polled every 150 ms| UI[Streamlit UI<br/>interview_workspace.py]
  S -->|final transcript| F[answer_feedback.py<br/>rule-based STAR]
  F --> R[feedback_panel.py]
  FILES[WAV files + manifest.csv] --> E[evaluation/*.py<br/>same back-ends]
  E --> J[results/*.json] --> DEV[dev_panel.py]
```

**Threads.** The WebRTC thread only converts audio and enqueues it, so it never blocks on the model.
One worker thread per attempt runs inference. The Streamlit script thread only reads snapshots.

**Partial transcription.** English: Vosk returns `PartialResult()` after every ~100 ms chunk.
Indonesian (Whisper): the buffer since the last pause is re-decoded every 1.5 s; a 0.8 s pause finalizes it.

**Latency (live app).** Time from `push_audio` of a chunk to the transcript text changing because of that chunk
(queue wait + inference). Browser capture, WebRTC transport and rendering are excluded.
**Latency (offline script).** Per-chunk `feed()` time, time-to-first-partial, final-flush delay and real-time factor.

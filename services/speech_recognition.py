"""
Local speech recognition: model registry, recognizer back-ends and a streaming session.

Everything here runs on the user's machine. No audio leaves the computer.

Data flow (see docs/ARCHITECTURE.md):
  browser mic -> WebRTC -> audio_capture (resample to 16 kHz mono PCM16)
      -> RecognitionSession.push_audio()  [fast, called from the WebRTC thread]
      -> queue -> worker thread -> back-end.feed()  [slow, model inference]
      -> TranscriptSnapshot (partial text + finalized segments) -> UI polls it
"""
from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
import json
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
MODELS_DIR = Path(os.environ.get("SPEAKREADY_MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LanguageConfig:
    code: str
    label: str
    backend: str          # "vosk" (true streaming) or "whisper" (chunked re-decoding)
    folder: str           # folder name inside models/
    download_hint: str    # shown to the user when the model is missing


def _indonesian_config() -> LanguageConfig:
    # Vosk's official model list may not include Indonesian. Verify at
    # https://alphacephei.com/vosk/models . If you have a Vosk Indonesian model,
    # set SPEAKREADY_ID_BACKEND=vosk and SPEAKREADY_ID_FOLDER=<folder in models/>.
    if os.environ.get("SPEAKREADY_ID_BACKEND", "whisper") == "vosk":
        folder = os.environ.get("SPEAKREADY_ID_FOLDER", "vosk-model-id")
        return LanguageConfig("id", "Bahasa Indonesia", "vosk", folder,
                              f"Place a Vosk Indonesian model in models/{folder}")
    return LanguageConfig(
        "id", "Bahasa Indonesia", "whisper", "faster-whisper-small",
        "huggingface-cli download Systran/faster-whisper-small --local-dir models/faster-whisper-small")


LANGUAGES: dict[str, LanguageConfig] = {
    "en": LanguageConfig("en", "English", "vosk", "vosk-model-small-en-us-0.15",
                         "Download https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip "
                         "and unzip it into models/"),
    "id": _indonesian_config(),
}


class ModelNotInstalled(Exception):
    def __init__(self, config: LanguageConfig):
        self.config = config
        super().__init__(f"Model folder models/{config.folder} not found")


def model_path(config: LanguageConfig) -> Path:
    return MODELS_DIR / config.folder



def _download_model(config: LanguageConfig) -> None:
    """Download a missing speech model into the models directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = model_path(config)

    if config.backend == "vosk":
        url = (
            "https://alphacephei.com/vosk/models/"
            "vosk-model-small-en-us-0.15.zip"
        )

        with tempfile.NamedTemporaryFile(
            suffix=".zip", dir=MODELS_DIR, delete=False
        ) as temp:
            archive = Path(temp.name)
            with urllib.request.urlopen(url, timeout=120) as response:
                shutil.copyfileobj(response, temp)

        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(MODELS_DIR)
        finally:
            archive.unlink(missing_ok=True)

    elif config.backend == "whisper":
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="Systran/faster-whisper-small",
            local_dir=str(path),
        )

    if not path.is_dir():
        raise RuntimeError(f"Model folder was not created: {path}")


def is_model_installed(language: str) -> bool:
    config = LANGUAGES[language]
    path = model_path(config)

    if path.is_dir():
        return True

    try:
        _download_model(config)
        return path.is_dir()
    except Exception as exc:
        print(f"Model download failed for {language}: {exc}")
        return False


def load_model(language: str):
    """Load the model object for a language (slow: call once and cache)."""
    config = LANGUAGES.get(language)
    if config is None:
        raise ValueError(f"Unsupported language: {language}")
    path = model_path(config)
    if not path.is_dir() and not is_model_installed(language):
        raise ModelNotInstalled(config)
    if config.backend == "vosk":
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)
        print(f"[SpeakReady] Backend: {config.backend}")
        print(f"[SpeakReady] Model path: {path}")
        print(f"[SpeakReady] Model exists: {path.is_dir()}")
        return Model(str(path))
    from faster_whisper import WhisperModel
    return WhisperModel(str(path), device="cpu", compute_type="int8")


# --------------------------------------------------------------------------- #
# Recognizer back-ends. feed() returns (partial_text, finalized_text_or_None)
# --------------------------------------------------------------------------- #
class VoskBackend:
    """True streaming: Vosk emits a partial hypothesis after every chunk."""

    def __init__(self, model):
        from vosk import KaldiRecognizer
        self._rec = KaldiRecognizer(model, SAMPLE_RATE)

    def feed(self, pcm: bytes) -> tuple[str, str | None]:
        if self._rec.AcceptWaveform(pcm):          # True = Vosk detected an utterance end
            return "", json.loads(self._rec.Result()).get("text", "").strip()
        return json.loads(self._rec.PartialResult()).get("partial", "").strip(), None

    def flush(self) -> str:
        return json.loads(self._rec.FinalResult()).get("text", "").strip()


class WhisperChunkBackend:
    """
    Whisper is not a streaming model, so partials are produced by re-decoding the
    audio buffered since the last pause every `interval` seconds (chunked recognition:
    updates arrive about every 1.5 s, not per word). A pause of ~0.8 s finalizes the buffer.

    Anti-hallucination: Whisper invents text (e.g. "Terima kasih") when it is given silence.
    So (1) we never decode a buffer that has not contained speech-level audio, and
    (2) decoding uses faster-whisper's built-in voice activity filter.
    """

    KEEP = "__keep__"        # sentinel: no new decode happened, keep the previous partial
    SPEECH_RMS = 0.012       # chunk loudness (0..1) that counts as speech; lower it for very quiet mics

    def __init__(self, model, language: str, interval: float = 1.5, pause: float = 0.8):
        self._model, self._lang = model, language
        self._interval, self._pause = interval, pause
        self._buf = np.zeros(0, dtype=np.float32)
        self._since_decode = 0.0
        self._silence = 0.0
        self._has_speech = False

    def _decode(self) -> str:
        segments, _ = self._model.transcribe(
            self._buf, language=self._lang, beam_size=1, condition_on_previous_text=False,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
            no_speech_threshold=0.6, log_prob_threshold=-1.0, compression_ratio_threshold=2.4)
        return " ".join(s.text.strip() for s in segments).strip()

    def _reset(self) -> None:
        self._buf, self._since_decode, self._silence, self._has_speech = self._buf[:0], 0.0, 0.0, False

    def feed(self, pcm: bytes) -> tuple[str, str | None]:
        chunk = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        seconds = len(chunk) / SAMPLE_RATE
        self._buf = np.concatenate([self._buf, chunk])
        if np.sqrt(np.mean(chunk ** 2)) >= self.SPEECH_RMS:
            self._has_speech, self._silence = True, 0.0
        else:
            self._silence += seconds
        if not self._has_speech:
            # Only silence so far: keep a short lead-in (so first words are not clipped), never decode.
            self._buf = self._buf[-int(SAMPLE_RATE * 0.5):]
            self._since_decode = 0.0
            return self.KEEP, None
        self._since_decode += seconds
        if self._silence >= self._pause:
            text = self._decode()
            self._reset()
            return "", text
        if self._since_decode >= self._interval:
            self._since_decode = 0.0
            return self._decode(), None
        return self.KEEP, None

    def flush(self) -> str:
        if not self._has_speech:
            return ""
        text = self._decode()
        self._reset()
        return text


def make_backend(language: str, model):
    config = LANGUAGES[language]
    return VoskBackend(model) if config.backend == "vosk" else WhisperChunkBackend(model, language)


# --------------------------------------------------------------------------- #
# Streaming session
# --------------------------------------------------------------------------- #
@dataclass
class TranscriptSnapshot:
    status: str                 # ready | listening | processing | stopped | error
    partial: str
    finals: list[str]
    levels: list[float]
    elapsed: float
    audio_seconds: float
    frames_received: int
    error: str | None

    @property
    def full_text(self) -> str:
        return " ".join(t for t in [*self.finals, self.partial] if t).strip()


class RecognitionSession:
    """One practice attempt. Thread-safe; owns the worker thread that runs the model."""

    def __init__(self, language: str, model):
        self.language = language
        self._model = model
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._finals: list[str] = []
        self._partial = ""
        self._levels: deque[float] = deque([0.0] * 48, maxlen=48)
        self._status, self._error = "ready", None
        self._started_at = self._stopped_at = None
        self._audio_seconds, self._frames = 0.0, 0
        self.latencies: list[float] = []   # seconds, see _worker for the exact definition
        self._worker_thread: threading.Thread | None = None
        self._stop_flag = threading.Event()

    # ---- lifecycle ------------------------------------------------------- #
    @property
    def started(self) -> bool:
        return self._started_at is not None

    @property
    def stopped(self) -> bool:
        return self._stopped_at is not None

    def start(self) -> None:
        if self.started:
            return
        try:
            backend = make_backend(self.language, self._model)
        except Exception as exc:  # noqa: BLE001 - surface any model problem to the UI
            self._fail(f"Could not start the recognizer: {exc}")
            self._started_at = self._stopped_at = time.time()
            return
        self._started_at = time.time()
        self._status = "listening"
        self._worker_thread = threading.Thread(target=self._worker, args=(backend,), daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop listening, drain queued audio, and finalize the transcript."""
        if not self.started or self.stopped:
            return
        with self._lock:
            if self._status != "error":
                self._status = "processing"
        self._stop_flag.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=60)
        self._stopped_at = time.time()
        with self._lock:
            if self._status != "error":
                self._status = "stopped"

    # ---- called from the WebRTC thread (must stay fast) ------------------ #
    def push_audio(self, pcm: bytes, level: float) -> None:
        if not self.started or self.stopped or self._stop_flag.is_set():
            return
        seconds = len(pcm) / 2 / SAMPLE_RATE
        with self._lock:
            self._frames += 1
            self._audio_seconds += seconds
            self._levels.append(level)
        self._queue.put((pcm, time.perf_counter()))

    def report_error(self, message: str) -> None:
        self._fail(message)

    # ---- worker thread: runs the model ----------------------------------- #
    def _worker(self, backend) -> None:
        try:
            while not (self._stop_flag.is_set() and self._queue.empty()):
                try:
                    pcm, t_arrived = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                partial, final = backend.feed(pcm)
                t_done = time.perf_counter()
                changed = False
                with self._lock:
                    if final is not None:
                        if final:
                            self._finals.append(final)
                        self._partial, changed = "", bool(final)
                    elif partial != WhisperChunkBackend.KEEP and partial != self._partial:
                        self._partial, changed = partial, True
                if changed:
                    # LATENCY DEFINITION: time from the moment this audio chunk reached the
                    # app (push_audio) to the moment the transcript text changed because of
                    # it. Includes queue wait + model inference; excludes browser capture,
                    # WebRTC transport and screen rendering.
                    self.latencies.append(t_done - t_arrived)
            tail = backend.flush()
            with self._lock:
                if tail:
                    self._finals.append(tail)
                self._partial = ""
        except Exception as exc:  # noqa: BLE001
            self._fail(f"Recognition failed: {exc}")

    def _fail(self, message: str) -> None:
        with self._lock:
            self._status, self._error = "error", message

    # ---- read side (UI thread) ------------------------------------------- #
    def snapshot(self) -> TranscriptSnapshot:
        with self._lock:
            end = self._stopped_at or time.time()
            elapsed = (end - self._started_at) if self._started_at else 0.0
            return TranscriptSnapshot(self._status, self._partial, list(self._finals), list(self._levels),
                                      elapsed, self._audio_seconds, self._frames, self._error)

    def latency_summary(self) -> dict | None:
        if not self.latencies:
            return None
        arr = np.array(self.latencies)
        return {"samples": int(arr.size), "median_s": float(np.median(arr)),
                "p95_s": float(np.percentile(arr, 95)), "max_s": float(arr.max())}

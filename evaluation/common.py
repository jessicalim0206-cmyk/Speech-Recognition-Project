"""Shared helpers for the evaluation scripts: manifest loading, streaming a file through a model."""
from __future__ import annotations

import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.audio_capture import read_wav_16k_mono  # noqa: E402
from services.speech_recognition import SAMPLE_RATE, WhisperChunkBackend, load_model, make_backend  # noqa: E402

CHUNK_SECONDS = 0.1   # same order of magnitude as a WebRTC frame batch


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse spaces. Applied to BOTH reference and hypothesis."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower())).strip()


def load_manifest(path: Path) -> list[dict]:
    """CSV columns: audio_path, reference, language (en|id), condition (e.g. baseline, noise_10db)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        audio = Path(row["audio_path"])
        row["audio_path"] = audio if audio.is_absolute() else (path.parent / audio)
    return rows


@dataclass
class StreamResult:
    hypothesis: str
    first_partial_s: float | None     # audio time until the first non-empty partial appeared
    chunk_times_s: list[float]        # model time spent on each chunk
    final_delay_s: float              # time to finalize after the last chunk was fed
    audio_s: float


def stream_audio(language: str, model, samples: np.ndarray, realtime: bool = False) -> StreamResult:
    """Feed audio to a back-end in 100 ms chunks exactly like the live app does."""
    backend = make_backend(language, model)
    step = int(SAMPLE_RATE * CHUNK_SECONDS)
    finals, first_partial, chunk_times = [], None, []
    for i in range(0, len(samples), step):
        chunk = samples[i:i + step].tobytes()
        t0 = time.perf_counter()
        partial, final = backend.feed(chunk)
        dt = time.perf_counter() - t0
        chunk_times.append(dt)
        if first_partial is None and partial and partial != WhisperChunkBackend.KEEP:
            first_partial = (i + step) / SAMPLE_RATE
        if final:
            finals.append(final)
        if realtime:
            time.sleep(max(0.0, CHUNK_SECONDS - dt))
    t0 = time.perf_counter()
    tail = backend.flush()
    final_delay = time.perf_counter() - t0
    if tail:
        finals.append(tail)
    return StreamResult(" ".join(finals).strip(), first_partial, chunk_times, final_delay, len(samples) / SAMPLE_RATE)


# ---- experimental conditions (applied to the audio before recognition) ----
def add_noise(samples: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Add white Gaussian noise at a given signal-to-noise ratio (dB). Seeded for reproducibility."""
    rng = np.random.default_rng(seed)
    x = samples.astype(np.float64)
    signal_power = np.mean(x ** 2) or 1.0
    noise = rng.normal(0, np.sqrt(signal_power / (10 ** (snr_db / 10))), size=x.shape)
    return np.clip(x + noise, -32768, 32767).astype(np.int16)


def change_speed(samples: np.ndarray, factor: float) -> np.ndarray:
    """Naive resampling speed change (factor 1.25 = 25% faster). Also shifts pitch; documented limitation."""
    idx = np.arange(0, len(samples) - 1, factor)
    return np.interp(idx, np.arange(len(samples)), samples).astype(np.int16)


def apply_condition(samples: np.ndarray, condition: str) -> np.ndarray:
    """Conditions: baseline | noise_<dB>db (e.g. noise_10db) | speed_<factor> (e.g. speed_1.25)."""
    if condition == "baseline":
        return samples
    if condition.startswith("noise_") and condition.endswith("db"):
        return add_noise(samples, float(condition[6:-2]))
    if condition.startswith("speed_"):
        return change_speed(samples, float(condition[6:]))
    raise ValueError(f"Unknown condition '{condition}'. Use baseline, noise_<dB>db or speed_<factor>.")


__all__ = ["load_manifest", "load_model", "read_wav_16k_mono", "stream_audio", "apply_condition", "normalize"]

"""
Audio capture helpers.

Live path: streamlit-webrtc hands us av.AudioFrame objects (often 48 kHz stereo).
We convert each one to 16 kHz mono 16-bit PCM, the format Whisper expects,
and forward it to the recognition session immediately (no "record first, transcribe later").

File path: read_wav_16k_mono() loads a test recording for the evaluation scripts.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from services.speech_recognition import SAMPLE_RATE, RecognitionSession


def make_audio_callback(session: RecognitionSession):
    """Build the callback streamlit-webrtc calls for every incoming audio frame."""
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)

    def callback(frame):
        try:
            for resampled in resampler.resample(frame):
                pcm = resampled.to_ndarray().astype(np.int16).tobytes()
                if pcm:
                    session.push_audio(pcm, rms_level(pcm))
        except Exception as exc:  # noqa: BLE001 - e.g. unsupported sample format
            session.report_error(f"Unsupported audio format from the microphone: {exc}")
        return frame

    return callback


def rms_level(pcm: bytes) -> float:
    """Loudness of a chunk scaled to 0..1 for the level meter (real data, lightly boosted)."""
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        return 0.0
    return float(min(1.0, np.sqrt(np.mean(samples ** 2)) * 6))


def read_wav_16k_mono(path) -> np.ndarray:
    """Load a 16-bit PCM WAV file as int16 mono at 16 kHz. Raises ValueError otherwise."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        channels, rate = wav.getnchannels(), wav.getframerate()
        data = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if rate != SAMPLE_RATE:
        target = int(len(data) * SAMPLE_RATE / rate)
        data = np.interp(np.linspace(0, len(data) - 1, target), np.arange(len(data)), data).astype(np.int16)
    return data

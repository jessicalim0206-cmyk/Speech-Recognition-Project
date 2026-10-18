"""Core tests that need no model and no microphone: python tests/test_core.py"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import services.speech_recognition as sr
from services.audio_capture import make_audio_callback, rms_level
from evaluation.common import add_noise, apply_condition, change_speed, normalize


class FakeBackend:
    """Stands in for a model: emits a growing partial every chunk, a final every 10 chunks. NOT real ASR."""
    def __init__(self): self.n = 0
    def feed(self, pcm):
        self.n += 1
        time.sleep(0.005)
        if self.n % 10 == 0:
            return "", f"segment {self.n // 10}"
        return f"word{self.n % 10}", None
    def flush(self): return "tail"


def test_streaming_session():
    sr.make_backend = lambda lang, model: FakeBackend()
    s = sr.RecognitionSession("en", None)
    s.start()
    seen_partial = False
    pcm = (np.random.randn(1600) * 3000).astype(np.int16).tobytes()
    for _ in range(25):
        s.push_audio(pcm, rms_level(pcm))
        time.sleep(0.01)
        seen_partial |= bool(s.snapshot().partial)
    assert seen_partial, "partial text must appear BEFORE the session ends"
    s.stop()
    snap = s.snapshot()
    assert snap.status == "stopped" and snap.finals[-1] == "tail" and snap.partial == ""
    assert s.latency_summary()["samples"] > 0
    print("streaming session ok: partial seen while running, finals:", snap.finals, s.latency_summary())


def test_audio_callback_resamples_48k_stereo():
    import av
    sr.make_backend = lambda lang, model: FakeBackend()
    s = sr.RecognitionSession("en", None); s.start()
    cb = make_audio_callback(s)
    frame = av.AudioFrame.from_ndarray(np.zeros((1, 960 * 2), dtype=np.int16), format="s16", layout="stereo")
    frame.sample_rate = 48000
    for _ in range(50): cb(frame)
    time.sleep(0.3); s.stop()
    assert s.snapshot().frames_received > 0
    print("48 kHz stereo -> 16 kHz mono ok; audio seconds:", round(s.snapshot().audio_seconds, 2), "(expected ~1.0)")


def test_conditions():
    x = (np.sin(np.linspace(0, 400, 16000)) * 8000).astype(np.int16)
    noisy = add_noise(x, 10)
    snr = 10 * np.log10(np.mean(x.astype(float) ** 2) / np.mean((noisy.astype(float) - x) ** 2))
    assert abs(snr - 10) < 0.7, snr
    assert len(change_speed(x, 1.25)) < len(x)
    assert normalize("Hello,  World!") == "hello world"
    apply_condition(x, "baseline"); print("conditions ok, measured SNR", round(snr, 2))


def test_missing_model_is_friendly():
    import os
    from services.speech_recognition import ModelNotInstalled, load_model
    try:
        load_model("en")
    except ModelNotInstalled as e:
        print("missing model raises ModelNotInstalled ->", e.config.folder); return
    print("(model is installed locally; skipped)")


if __name__ == "__main__":
    test_streaming_session(); test_audio_callback_resamples_48k_stereo(); test_conditions(); test_missing_model_is_friendly()

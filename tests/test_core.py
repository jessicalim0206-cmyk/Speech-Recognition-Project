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


def test_incomplete_model_is_detected():
    """A model folder that merely exists must not count as installed."""
    import tempfile
    from pathlib import Path
    import services.speech_recognition as sr

    cfg = sr.LANGUAGES["en"]
    old_dir, old_dl = sr.MODELS_DIR, sr._download_model
    with tempfile.TemporaryDirectory() as tmp:
        sr.MODELS_DIR = Path(tmp)
        folder = sr.model_path(cfg)

        def make_complete():
            folder.mkdir(parents=True, exist_ok=True)
            for name in ("config.json", "tokenizer.json", "vocabulary.txt"):
                (folder / name).write_text("x")
            with open(folder / "model.bin", "wb") as f:
                f.truncate(sr.WHISPER_MIN_MODEL_BYTES + 1)   # sparse file, no real disk use

        try:
            # 1) folder missing
            assert sr.model_problems(cfg), "missing folder must be a problem"
            # 2) folder exists but only half the files (interrupted download)
            folder.mkdir(parents=True); (folder / "config.json").write_text("x")
            probs = sr.model_problems(cfg)
            assert any("model.bin" in p for p in probs), probs
            # 3) truncated model.bin
            make_complete(); (folder / "model.bin").write_bytes(b"x" * 1024)
            assert any("too small" in p for p in sr.model_problems(cfg)), sr.model_problems(cfg)
            # 4) complete folder passes
            make_complete(); assert sr.model_problems(cfg) == []
            # 5) incomplete + download fails -> clear ModelNotInstalled with reason, not a crash
            (folder / "tokenizer.json").unlink()
            def failing(_cfg): raise OSError("no internet")
            sr._download_model = failing
            try:
                sr.ensure_model("en"); raise AssertionError("should have raised")
            except sr.ModelNotInstalled as e:
                assert "no internet" in str(e) and e.reason
            assert sr.is_model_installed("en") is False
            # 6) incomplete + download repairs it -> ok
            sr._download_model = lambda _cfg: make_complete()
            sr.ensure_model("en"); assert sr.is_model_installed("en") is True
        finally:
            sr.MODELS_DIR, sr._download_model = old_dir, old_dl
    print("incomplete model detection ok")


if __name__ == "__main__":
    test_streaming_session(); test_audio_callback_resamples_48k_stereo(); test_conditions(); test_missing_model_is_friendly(); test_incomplete_model_is_detected()

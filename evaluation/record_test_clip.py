"""
Record a test clip from your microphone for the evaluation manifest (needs: pip install sounddevice).
  python evaluation/record_test_clip.py evaluation/clips/en_01.wav --seconds 15
Read your reference sentence aloud, then type the same sentence into manifest.csv as the reference.
"""
import argparse
import wave
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--seconds", type=float, default=15)
args = ap.parse_args()
try:
    import sounddevice as sd
except ImportError:
    raise SystemExit("Install sounddevice first: pip install sounddevice")
print(f"Recording {args.seconds}s... speak now.")
data = sd.rec(int(args.seconds * 16000), samplerate=16000, channels=1, dtype="int16")
sd.wait()
args.out.parent.mkdir(parents=True, exist_ok=True)
with wave.open(str(args.out), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(data.tobytes())
print(f"Saved {args.out}")

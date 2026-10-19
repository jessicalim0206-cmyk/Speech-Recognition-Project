
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np

MODEL_PATH = "models/faster-whisper-small"
SAMPLE_RATE = 16000
DURATION = 10

print("Loading Whisper model...")
model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")

input("Press Enter, then speak your English sentence...")
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
)
sd.wait()

segments, info = model.transcribe(
    audio.flatten(),
    language="en",
    beam_size=5,
    vad_filter=False,
    condition_on_previous_text=False,
)

transcript = " ".join(segment.text.strip() for segment in segments)
print("\nDetected language:", info.language)
print("Transcript:", transcript)

import re, pathlib
p = pathlib.Path("services/speech_recognition.py")
src = p.read_text(encoding="utf-8")
new_cls = '''class WhisperChunkBackend:
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


'''
out, n = re.subn(r"class WhisperChunkBackend:.*?(?=def make_backend)", lambda m: new_cls, src, flags=re.S)
assert n == 1, "class not found"
p.write_text(out, encoding="utf-8")
print("patched")

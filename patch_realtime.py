"""
Patch #1: keep SpeakReady real-time when the model is slower than the incoming audio.
Run from the project root (the folder that contains app.py):   python patch_realtime.py
It edits services/speech_recognition.py and components/interview_workspace.py.
Each edit is checked: if your file differs from what is expected, nothing is written for that file.
"""
import pathlib, sys


def apply(path, edits):
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8")
    for i, (old, new) in enumerate(edits, 1):
        if text.count(old) != 1:
            sys.exit(f"STOP: {path}: edit {i} did not match exactly once (found {text.count(old)}). File not changed.")
        text = text.replace(old, new)
    p.write_text(text, encoding="utf-8")
    print(f"patched {path} ({len(edits)} edits)")


SR = [
    # 1. Whisper backend: new limits + 'behind' flag
    ('''    SPEECH_RMS = 0.012       # chunk loudness (0..1) that counts as speech; lower it for very quiet mics
''',
     '''    SPEECH_RMS = 0.012       # chunk loudness (0..1) that counts as speech; lower it for very quiet mics
    MAX_SEGMENT_SECONDS = 12.0   # force a segment break so one decode never gets longer than this
'''),
    ('''        self._has_speech = False

    def _decode(self) -> str:''',
     '''        self._has_speech = False
        self.behind = False      # set by the session when audio is queueing up: skip partial decodes

    def _decode(self) -> str:'''),
    ('''        if self._silence >= self._pause:
            text = self._decode()''',
     '''        too_long = len(self._buf) >= SAMPLE_RATE * self.MAX_SEGMENT_SECONDS
        if self._silence >= self._pause or too_long:
            text = self._decode()'''),
    ('''        if self._since_decode >= self._interval:
            self._since_decode = 0.0
            return self._decode(), None''',
     '''        if self._since_decode >= self._interval and not self.behind:
            # (when `behind`, we keep buffering and decode as soon as the queue has caught up)
            self._since_decode = 0.0
            return self._decode(), None'''),
    # 2. Session: constants and events
    ('''    """One practice attempt. Thread-safe; owns the worker thread that runs the model."""

    def __init__(self, language: str, model):''',
     '''    """One practice attempt. Thread-safe; owns the worker thread that runs the model."""

    BEHIND_FRAMES = 25   # about 0.5 s of queued audio: beyond this the worker skips partial decodes

    def __init__(self, language: str, model):'''),
    ('''        self._stop_flag = threading.Event()

    # ---- lifecycle''',
     '''        self._stop_flag = threading.Event()
        self._finished = threading.Event()
        self._stop_requested_at = None

    # ---- lifecycle'''),
    # 3. Non-blocking stop
    ('''    def stop(self) -> None:
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
''',
     '''    def request_stop(self) -> None:
        """Ask the worker to finish and return immediately. The worker drains the queue,
        finalizes the transcript, then marks the session stopped."""
        if not self.started or self.stopped or self._stop_flag.is_set():
            return
        self._stop_requested_at = time.time()
        with self._lock:
            if self._status != "error":
                self._status = "processing"
        self._stop_flag.set()

    def stop(self) -> None:
        """Blocking version (used by tests): request_stop() then wait for the worker."""
        self.request_stop()
        if self._worker_thread:
            self._worker_thread.join(timeout=60)
'''),
    # 4. Worker: tell the backend when we are behind
    ('''                partial, final = backend.feed(pcm)
''',
     '''                backend.behind = self._queue.qsize() > self.BEHIND_FRAMES
                partial, final = backend.feed(pcm)
'''),
    # 5. Worker end: it, not stop(), marks the session as stopped
    ('''        except Exception as exc:  # noqa: BLE001
            self._fail(f"Recognition failed: {exc}")
''',
     '''        except Exception as exc:  # noqa: BLE001
            self._fail(f"Recognition failed: {exc}")
        finally:
            self._stopped_at = time.time()
            with self._lock:
                if self._status != "error":
                    self._status = "stopped"
            self._finished.set()
'''),
    # 6. Timer stops counting when STOP is pressed, not when processing ends
    ('''            end = self._stopped_at or time.time()''',
     '''            end = self._stop_requested_at or self._stopped_at or time.time()'''),
]

WS = [
    ('''        old.stop()
''', '''        old.request_stop()
'''),
    ('''        session.stop()       # user pressed STOP: drain the queue and finalize
''',
     '''        session.request_stop()   # user pressed STOP: non-blocking; the worker finalizes in the background
'''),
    ('''    if playing and not session.stopped:
        time.sleep(0.5)
        st.rerun()
''',
     '''    if session.started and not session.stopped:
        # keep refreshing while listening AND while the worker is still finishing ("Processing")
        time.sleep(0.5)
        st.rerun()
'''),
]

apply("services/speech_recognition.py", SR)
apply("components/interview_workspace.py", WS)
print("done")

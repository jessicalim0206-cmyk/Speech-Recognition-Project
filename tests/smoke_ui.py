"""Headless UI smoke test (no microphone, no model): python tests/smoke_ui.py
A fresh AppTest per scenario avoids a Streamlit test-harness quirk with format_func selectboxes on re-runs."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def run(**state):
    at = AppTest.from_file(APP, default_timeout=60)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


run(); print("home ok")
for lang in ("en", "id"):
    at = run(lang=lang, page="practice")
    states = [m.value for m in at.markdown if "sr-state" in m.value]
    assert states, "expected the friendly 'model not installed' state"
    print(lang, "practice (no model installed) -> friendly state shown")
run(page="review", result=None); print("review empty state ok")
run(page="dev"); print("dev page ok")
run(page="review", lang="en", result={"question": "Tell me about yourself.", "keywords": ["study", "project"],
    "transcript": "When I was in my internship my task was to build a tool. I decided to start small and as a result it improved.",
    "lang": "en", "category": "general", "words": 23, "duration": 12.0}); print("review with transcript ok")

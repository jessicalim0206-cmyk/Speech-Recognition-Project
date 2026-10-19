"""Small renderers: status chip, level meter and transcript, all driven by real session data."""
from __future__ import annotations

from html import escape

from components.copy import T
from services.speech_recognition import TranscriptSnapshot


def format_time(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def latency_html(seconds: float | None, lang: str) -> str:
    """Latest processing delay (audio reached the app -> text changed). Shows a dash until the first update."""
    if seconds is None:
        value = "—"
    elif seconds < 1:
        value = f"{seconds * 1000:.0f} ms"
    else:
        value = f"{seconds:.1f} s"
    return f'<div class="sr-latency">{T[lang]["latency"]}: <b>{value}</b></div>'


def status_html(status: str, lang: str) -> str:
    return f'<span class="sr-status {status}"><span class="sr-dot"></span>{T[lang][status]}</span>'


def wave_html(levels: list[float], active: bool) -> str:
    """Bars are the real recent microphone RMS values; flat grey when nothing is being captured."""
    bars = "".join(f'<span style="height:{max(4, int(l * 56))}px"></span>' for l in levels)
    return f'<div class="sr-wave {"" if active else "idle"}" role="img" aria-label="{T["en"]["level"]}">{bars}</div>'


def transcript_html(snap: TranscriptSnapshot, lang: str, listening: bool) -> str:
    t = T[lang]
    if not snap.finals and not snap.partial:
        msg = t["empty_listening"] if listening else t["empty_before"]
        return f'<div class="sr-transcript" aria-live="polite"><p class="sr-empty">{escape(msg)}</p></div>'
    done = escape(" ".join(snap.finals))
    partial = f' <span class="partial">{escape(snap.partial)}</span>' if snap.partial else ""
    return f'<div class="sr-transcript" aria-live="polite"><p>{done}{partial}</p></div>'


def state_html(title: str, body: str, kind: str = "error", extra: str = "") -> str:
    return f'<div class="sr-state {kind}" role="alert"><b>{escape(title)}</b>{escape(body)}{extra}</div>'

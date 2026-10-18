"""The practice workspace: question, recorder, live transcript."""
from __future__ import annotations

import time

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from components.copy import T
from components.icons import icon
from components.transcript_panel import state_html, status_html, transcript_html, wave_html, format_time
from services.answer_feedback import count_words
from services.audio_capture import make_audio_callback
from services.questions import category_label, questions_for
from services.speech_recognition import LANGUAGES, ModelNotInstalled, RecognitionSession, load_model


@st.cache_resource(show_spinner=False)
def get_model(language: str):
    """Load each local model once per server process."""
    return load_model(language)


def reset_attempt() -> None:
    """Discard the current attempt (stops the worker) and start with a clean transcript."""
    old = st.session_state.get("session")
    if old is not None:
        old.stop()
    st.session_state.session = None
    st.session_state.attempt = st.session_state.get("attempt", 0) + 1


WEBRTC_TEXT = {
    "en": {"start": "START", "stop": "STOP", "select_device": "Select microphone",
           "device_ask_permission": "Allow microphone access in your browser to begin.",
           "device_not_available": "No microphone was found. Plug one in or choose another device, then reload.",
           "device_access_denied": "Microphone permission was denied. Allow it from the lock icon in the address bar, then reload."},
    "id": {"start": "START", "stop": "STOP", "select_device": "Pilih mikrofon",
           "device_ask_permission": "Izinkan akses mikrofon di browser untuk memulai.",
           "device_not_available": "Mikrofon tidak ditemukan. Sambungkan mikrofon atau pilih perangkat lain, lalu muat ulang.",
           "device_access_denied": "Izin mikrofon ditolak. Izinkan lewat ikon gembok di bilah alamat, lalu muat ulang."},
}


def render_workspace(lang: str) -> None:
    s, t = st.session_state, T[lang]
    questions = questions_for(s.category, lang)
    question, keywords = questions[s.q_index]

    st.markdown(
        f'<div class="sr-meta"><span>{category_label(s.category, lang)}</span><span>·</span>'
        f'<span>{t["question_of"].format(i=s.q_index + 1, n=len(questions))}</span>'
        f'<span class="sr-chip">{LANGUAGES[lang].label}</span></div>'
        f'<h1 class="sr-question">{question}</h1>', unsafe_allow_html=True)

    # --- model availability (friendly errors instead of a stack trace) ---
    try:
        with st.spinner("Loading speech model..." if lang == "en" else "Memuat model suara..."):
            model = get_model(lang)
    except ModelNotInstalled as exc:
        hint = f'<br><code>{exc.config.download_hint}</code><br>See models/README.md.'
        st.markdown(state_html(t["model_missing_t"].format(lang=LANGUAGES[lang].label), t["model_missing_b"], "info", hint),
                    unsafe_allow_html=True)
        _back_button(t)
        return
    except Exception as exc:  # noqa: BLE001
        st.markdown(state_html(t["load_failed_t"], str(exc)), unsafe_allow_html=True)
        _back_button(t)
        return

    if s.get("session") is None:
        s.session = RecognitionSession(lang, model)
    session: RecognitionSession = s.session

    col_rec, col_txt = st.columns([5, 7], gap="large")
    with col_rec:
        status_ph, timer_ph, wave_ph = st.empty(), st.empty(), st.empty()
        ctx = webrtc_streamer(
            key=f"sr-{s.attempt}",
            mode=WebRtcMode.SENDONLY,
            audio_frame_callback=make_audio_callback(session),
            media_stream_constraints={"audio": {"echoCancellation": True, "noiseSuppression": True}, "video": False},
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]}
                ]
            },   # local use: no external STUN server contacted
            translations=WEBRTC_TEXT[lang],
        )
        notice_ph = st.empty()
    with col_txt:
        st.markdown(f'<div class="sr-muted" style="margin-bottom:8px">{t["transcript"]}</div>', unsafe_allow_html=True)
        txt_ph = st.empty()
    controls = st.container()

    playing = ctx.state.playing
    st.write("DEBUG WebRTC:", ctx.state)
    st.write("Playing:", ctx.state.playing)
    st.write("Session started:", session.started)
    if playing and not session.started:
        session.start()
    if session.started and not session.stopped and not playing:
        session.stop()       # user pressed STOP: drain the queue and finalize

    snap = session.snapshot()

    st.write("DEBUG — Recording")
    st.write("WebRTC playing:", playing)
    st.write("Session started:", session.started)
    st.write("Session stopped:", session.stopped)
    st.write("Session status:", snap.status)
    st.write("Elapsed seconds:", snap.elapsed)

    _controls(controls, t, snap, keywords, question, lang)

    def paint(snap):
        live = snap.status == "listening"
        status_ph.markdown(status_html(snap.status, lang), unsafe_allow_html=True)
        timer_ph.markdown(f'<div class="sr-timer" aria-label="{t["elapsed"]}">{format_time(snap.elapsed)}</div>', unsafe_allow_html=True)
        wave_ph.markdown(wave_html(snap.levels, live), unsafe_allow_html=True)
        txt_ph.markdown(transcript_html(snap, lang, live), unsafe_allow_html=True)

    paint(snap)
    _notices(notice_ph, t, snap)

    if playing and not session.stopped:
        time.sleep(0.5)
        st.rerun()

    # --- live loop: poll the session while the microphone is on ---
    while ctx.state.playing and not session.stopped:
        snap = session.snapshot()
        paint(snap)
        _notices(notice_ph, t, snap)
        if snap.status == "error":
            break
        time.sleep(0.15)
    if session.started and not session.stopped:
        session.stop()
        st.rerun()


def _notices(placeholder, t, snap) -> None:
    """Honest, specific states based on what actually happened."""
    html = ""
    if snap.status == "error":
        html = state_html(t["err_t"], snap.error or "")
    elif snap.status == "listening" and snap.elapsed > 4 and snap.frames_received == 0:
        html = state_html(t["no_audio_t"], t["no_audio_b"])
    elif snap.status == "stopped" and snap.frames_received == 0:
        html = state_html(t["no_audio_t"], t["no_audio_b"])
    elif snap.status == "stopped" and not snap.full_text:
        html = state_html(t["no_speech_t"], t["no_speech_b"], "info")
    placeholder.markdown(html, unsafe_allow_html=True)


def _controls(container, t, snap, keywords, question, lang) -> None:
    s = st.session_state
    has_text = bool(snap.full_text) and snap.status == "stopped"
    c1, c2, c3 = container.columns(3)
    if c1.button(t["restart"], key="restart", use_container_width=True):
        reset_attempt()
        st.rerun()
    if c2.button(t["next_q"], key="next_q", use_container_width=True):
        reset_attempt()
        s.q_index = (s.q_index + 1) % len(questions_for(s.category, lang))
        st.rerun()
    if c3.button(t["finish"], key="finish", type="primary", use_container_width=True, disabled=not has_text):
        s.result = {"question": question, "keywords": keywords, "transcript": snap.full_text, "lang": lang,
                    "category": s.category, "words": count_words(snap.full_text), "duration": snap.elapsed}
        s.last_latency = s.session.latency_summary()
        s.last_latency_lang = lang
        s.page = "review"
        st.rerun()


def _back_button(t) -> None:
    if st.button(t["back_home"], key="back_home_err"):
        st.session_state.page = "home"
        st.rerun()

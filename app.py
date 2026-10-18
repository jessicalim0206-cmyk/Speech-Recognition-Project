"""SpeakReady: bilingual interview practice with local, streaming speech recognition.
Run with:  streamlit run app.py
"""
from pathlib import Path

import streamlit as st

from components.copy import T
from components.dev_panel import render_dev
from components.feedback_panel import render_review
from components.interview_workspace import render_workspace, reset_attempt
from components.navigation import render_nav
from services.questions import CATEGORIES, QUESTIONS, category_label
from services.speech_recognition import LANGUAGES

st.set_page_config(page_title="SpeakReady", page_icon=None, layout="wide")
st.markdown(f"<style>{(Path(__file__).parent / 'assets' / 'styles.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)

DEFAULTS = {"page": "home", "lang": "en", "category": "general", "q_index": 0, "attempt": 0,
            "session": None, "result": None}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def render_home() -> None:
    s = st.session_state
    t = T[s.lang]
    left, right = st.columns([7, 5], gap="large")
    with left:
        st.markdown(f'<div class="sr-hero"><h1>{t["headline"]}</h1></div><p class="sr-lead">{t["lead"]}</p>',
                    unsafe_allow_html=True)
    with right:
        st.markdown(f'<div style="height:32px"></div><p class="sr-setup-title">{t["setup_title"]}</p>', unsafe_allow_html=True)
        chosen = st.segmented_control(t["language"], list(LANGUAGES), default=s.lang,
                                      format_func=lambda c: LANGUAGES[c].label, key="lang_picker")
        if chosen and chosen != s.lang:             # switching language starts a clean session
            s.lang, s.q_index = chosen, 0
            reset_attempt()
            st.rerun()
        s.setdefault("category_picker", s.category)
        s.category = st.selectbox(t["category"], list(CATEGORIES), key="category_picker",
                                  format_func=lambda c: category_label(c, s.lang))
        if st.button(t["start"], type="primary", use_container_width=True, key="start"):
            s.q_index = 0
            reset_attempt()
            s.page = "practice"
            st.rerun()

    steps = "".join(f'<div class="sr-step"><b>{a}</b><span>{b}</span></div>' for a, b in t["steps"])
    st.markdown(f'<div class="sr-steps" role="list">{steps}</div>', unsafe_allow_html=True)
    cats = "".join(f'<span class="sr-cat">{category_label(c, s.lang)} <i>{len(QUESTIONS[c])} {t["questions_n"]}</i></span>'
                   for c in CATEGORIES)
    st.markdown(f'<p class="sr-setup-title" style="margin-top:32px">{t["categories_title"]}</p><div class="sr-cats">{cats}</div>',
                unsafe_allow_html=True)


render_nav(st.session_state.lang)
page = st.session_state.page
if page == "home":
    render_home()
elif page == "practice":
    render_workspace(st.session_state.lang)
elif page == "review":
    render_review(st.session_state.lang)
elif page == "dev":
    render_dev(st.session_state.lang)

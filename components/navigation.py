import streamlit as st

from components.copy import T
from components.icons import icon


def render_nav(lang: str) -> None:
    left, mid, right = st.columns([6, 1.3, 1.7])
    left.markdown(f'<div class="sr-wordmark">{icon("wave", 22)}SpeakReady</div>', unsafe_allow_html=True)
    if mid.button(T[lang]["nav_practice"], key="nav_practice", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    if right.button(T[lang]["nav_dev"], key="nav_dev", use_container_width=True):
        st.session_state.page = "dev"
        st.rerun()
    st.markdown('<hr class="sr-rule" style="margin-top:8px">', unsafe_allow_html=True)

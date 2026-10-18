"""Answer review: transcript, word count and the rule-based feedback."""
from __future__ import annotations

from html import escape

import streamlit as st

from components.copy import T
from components.transcript_panel import state_html
from services.answer_feedback import STAR_CUES, TEXT, evaluate_answer


def render_review(lang: str) -> None:
    t, result = T[lang], st.session_state.get("result")
    if not result or not result["transcript"].strip():
        st.markdown(state_html(t["empty_t"], t["empty_b"], "info"), unsafe_allow_html=True)
        if st.button(t["practice_again"], key="review_empty_back"):
            st.session_state.page = "practice"
            st.rerun()
        return
    lang = result["lang"]          # feedback is shown in the language the answer was given in
    t, fb_text = T[lang], TEXT[lang]
    fb = evaluate_answer(result["transcript"], result["keywords"], lang)

    st.markdown(f'<div class="sr-muted">{t["question_answered"]}</div><p class="sr-q-recap">{escape(result["question"])}</p>',
                unsafe_allow_html=True)
    left, right = st.columns([7, 5], gap="large")
    with left:
        st.markdown(f'<div class="sr-muted" style="margin-bottom:8px">{t["final_transcript"]}</div>'
                    f'<div class="sr-transcript"><p>{escape(result["transcript"])}</p></div>', unsafe_allow_html=True)
        st.markdown(f'<p class="sr-muted" style="margin-top:8px">{t["caveat_asr"]}</p>', unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="sr-stat">{fb.word_count}</div><div class="sr-muted">{t["words"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sr-fb"><h3>{t["relevance"]}</h3><div>{fb.relevance_text}</div>'
                    + (f'<div class="sr-muted" style="margin-top:8px">{t["keywords_found"]}: {escape(", ".join(fb.matched_keywords))}</div>' if fb.matched_keywords else "")
                    + '</div>', unsafe_allow_html=True)
        chips = "".join(f'<span class="{"on" if p in fb.star_found else ""}">{fb_text["star_names"][p]}</span>' for p in STAR_CUES[lang])
        st.markdown(f'<div class="sr-fb"><h3>{t["structure"]}</h3><div class="sr-star">{chips}</div><div>{fb.structure_text}</div></div>'
                    f'<div class="sr-fb"><h3>{t["length"]}</h3><div>{fb.length_text}</div></div>', unsafe_allow_html=True)
        tips = "".join(f"<li>{escape(x)}</li>" for x in fb.suggestions)
        st.markdown(f'<div class="sr-fb"><h3>{t["suggestions"]}</h3><ul style="padding-left:20px;margin:0;line-height:1.7">{tips}</ul>'
                    f'<p class="sr-muted" style="margin-top:16px">{fb.caveat}</p></div>', unsafe_allow_html=True)
    if st.button(t["practice_again"], type="primary", key="practice_again"):
        from components.interview_workspace import reset_attempt
        reset_attempt()
        st.session_state.page = "practice"
        st.rerun()


"""Answer review: transcript, word count and rule-based feedback."""
from __future__ import annotations

from html import escape

import streamlit as st

from components.copy import T
from components.transcript_panel import state_html
from services.answer_feedback import (
    STAR_CUES,
    TEXT,
    evaluate_answer,
    detect_question_type,
)


def render_review(lang: str) -> None:
    t, result = T[lang], st.session_state.get("result")

    if not result or not result["transcript"].strip():
        st.markdown(
            state_html(t["empty_t"], t["empty_b"], "info"),
            unsafe_allow_html=True,
        )
        if st.button(t["practice_again"], key="review_empty_back"):
            st.session_state.page = "practice"
            st.rerun()
        return

    lang = result["lang"]
    t, fb_text = T[lang], TEXT[lang]

    fb = evaluate_answer(
        result["transcript"],
        result["keywords"],
        lang,
        result["question"],
    )

    question_type = detect_question_type(result["question"], lang)

    st.markdown(
        f'<div class="sr-muted">{t["question_answered"]}</div>'
        f'<p class="sr-q-recap">{escape(result["question"])}</p>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([7, 5], gap="large")

    with left:
        st.markdown(
            f'<div class="sr-muted" style="margin-bottom:8px">'
            f'{t["final_transcript"]}</div>'
            f'<div class="sr-transcript"><p>'
            f'{escape(result["transcript"])}</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="sr-muted" style="margin-top:8px">'
            f'{t["caveat_asr"]}</p>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f'<div class="sr-stat">{fb.word_count}</div>'
            f'<div class="sr-muted">{t["words"]}</div>',
            unsafe_allow_html=True,
        )

        # Relevance
        relevance_html = (
            f'<div class="sr-fb"><h3>{t["relevance"]}</h3>'
            f'<div>{escape(fb.relevance_text)}</div>'
        )
        if fb.matched_keywords:
            relevance_html += (
                f'<div class="sr-muted" style="margin-top:8px">'
                f'{t["keywords_found"]}: '
                f'{escape(", ".join(fb.matched_keywords))}</div>'
            )
        relevance_html += "</div>"
        st.markdown(relevance_html, unsafe_allow_html=True)

        # Answer structure
        if question_type == "behavioral":
            chips = " ".join(
                f'<span class="{"on" if p in fb.star_found else ""}">'
                f'{escape(fb_text["star_names"][p])}</span>'
                for p in STAR_CUES[lang]
            )
            structure_title = (
                "Answer structure (STAR)"
                if lang == "en"
                else "Struktur jawaban (STAR)"
            )
            structure_html = (
                f'<div class="sr-star">{chips}</div>'
                f'<div>{escape(fb.structure_text)}</div>'
            )
        else:
            structure_title = (
                "Answer structure"
                if lang == "en"
                else "Struktur jawaban"
            )
            structure_html = f'<div>{escape(fb.structure_text)}</div>'

        st.markdown(
            f'<div class="sr-fb"><h3>{structure_title}</h3>'
            f'{structure_html}</div>'
            f'<div class="sr-fb"><h3>{t["length"]}</h3>'
            f'<div>{escape(fb.length_text)}</div></div>',
            unsafe_allow_html=True,
        )

        # Suggestions
        tips = "".join(
            f"<li>{escape(item)}</li>" for item in fb.suggestions
        )
        st.markdown(
            f'<div class="sr-fb"><h3>{t["suggestions"]}</h3>'
            f'<ul style="padding-left:20px;margin:0;line-height:1.7">'
            f'{tips}</ul>'
            f'<p class="sr-muted" style="margin-top:16px">'
            f'{escape(fb.caveat)}</p></div>',
            unsafe_allow_html=True,
        )

    if st.button(t["practice_again"], type="primary", key="practice_again"):
        from components.interview_workspace import reset_attempt

        reset_attempt()
        st.session_state.page = "practice"
        st.rerun()

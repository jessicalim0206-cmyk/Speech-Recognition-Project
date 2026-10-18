"""Developer testing page. Shows only measurements that exist on disk or in this session."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from components.copy import T

RESULTS = Path(__file__).resolve().parent.parent / "evaluation" / "results"


def render_dev(lang: str) -> None:
    t = T[lang]
    st.markdown(f'<h1 class="sr-question" style="font-size:36px;margin-bottom:8px">{t["dev_title"]}</h1>'
                f'<p class="sr-lead">{t["dev_lead"]}</p>', unsafe_allow_html=True)

    st.markdown(f"**{t['dev_session']}**")
    lat = st.session_state.get("last_latency")
    if lat:
        a, b, c, d = st.columns(4)
        a.metric("Samples", lat["samples"]); b.metric("Median", f"{lat['median_s'] * 1000:.0f} ms")
        c.metric("95th percentile", f"{lat['p95_s'] * 1000:.0f} ms"); d.metric("Max", f"{lat['max_s'] * 1000:.0f} ms")
        st.caption("Time from an audio chunk reaching the app to the transcript changing because of it. "
                   "Includes queue wait and model inference. Excludes browser capture and WebRTC transport.")
    else:
        st.write(t["dev_none"])

    st.markdown("---")
    st.markdown(f"**{t['dev_files']}**")
    files = sorted(RESULTS.glob("*.json"), reverse=True)
    if not files:
        st.write(t["dev_nofiles"])
    for f in files[:6]:
        data = json.loads(f.read_text(encoding="utf-8"))
        with st.expander(f.name):
            if data.get("kind") == "wer":
                rows = [{"language/condition": k, "files": v["files"], "ref words": v["ref_words"],
                         "WER": None if v["wer"] is None else round(v["wer"], 3), "S": v["S"], "D": v["D"], "I": v["I"]}
                        for k, v in sorted(data["summary"].items())]
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.dataframe(data.get("per_file", []), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Run the evaluation**")
    st.code("python evaluation/wer_evaluation.py evaluation/manifest.csv --conditions baseline noise_10db speed_1.25\n"
            "python evaluation/latency_evaluation.py evaluation/manifest.csv --realtime", language="bash")

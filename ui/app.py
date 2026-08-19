# -*- coding: utf-8 -*-
"""AAA Clinical RAG — Streamlit frontend.

    streamlit run ui/app.py

A pure HTTP client of the FastAPI service. Nothing under `ui/` imports
`generation`, `retrieval`, `vectordb` or `ingestion`: there is exactly one
retrieval implementation in this project and it lives behind the API.

Point the UI at a different backend with:

    CLINICAL_RAG_API_URL=http://host:port  streamlit run ui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Run as `streamlit run ui/app.py`, so the project root is not on sys.path yet.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from ui import api_client, components as ui, theme  # noqa: E402
from ui.views import architecture, ask, evaluation, guidelines, safety, technical  # noqa: E402

PAGES = [
    ("Ask", "🔍", ask.render),
    ("Evaluation", "📊", evaluation.render),
    ("Safety & Abstention", "🛡", safety.render),
    ("Architecture", "🧩", architecture.render),
    ("Guidelines & Sources", "📚", guidelines.render),
    ("Technical Details", "⚙", technical.render),
]

NAV_KEY = "active_page"


def main() -> None:
    st.session_state.setdefault(NAV_KEY, PAGES[0][0])
    theme.apply(st.session_state[NAV_KEY])

    # One health probe per rerun, shared by the sidebar and every page.
    health = api_client.health_or_none()

    with st.sidebar:
        st.markdown(
            '<div style="font-family:Georgia,serif;font-size:1.12rem;font-weight:600;'
            'line-height:1.25;margin-bottom:0.15rem">AAA Clinical RAG</div>'
            '<div class="footnote" style="margin-bottom:1.1rem">Evidence-grounded '
            "clinical decision support</div>",
            unsafe_allow_html=True,
        )
        choice = st.radio(
            "Section",
            [name for name, _, _ in PAGES],
            format_func=lambda name: f"{dict((n, i) for n, i, _ in PAGES)[name]}  {name}",
            key=NAV_KEY,
            label_visibility="collapsed",
        )
        st.markdown('<hr class="rule">', unsafe_allow_html=True)

    ui.sidebar(health)

    render = dict((name, fn) for name, _, fn in PAGES)[choice]
    render(health)

    st.markdown(
        '<hr class="rule">'
        '<div class="footnote">Evidence-retrieval prototype over four abdominal aortic aneurysm '
        "guidelines. Not clinically validated. Does not provide patient-specific diagnosis, "
        "treatment or dosing, and must not be used to make clinical decisions. Verify every "
        "statement against the cited source document and apply clinical judgement.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

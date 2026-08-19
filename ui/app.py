# -*- coding: utf-8 -*-
"""AAA Clinical RAG — Streamlit frontend.

    streamlit run ui/app.py

A pure HTTP client of the FastAPI service. Nothing under `ui/` imports
`generation`, `retrieval`, `vectordb` or `ingestion`: there is exactly one
retrieval implementation in this project and it lives behind the API.
`tests/test_ui.py` enforces that by parsing every UI source.

    CLINICAL_RAG_API_URL=http://host:port  streamlit run ui/app.py

The hidden style guide is at `?dev=1`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from ui import components as c, shell, theme  # noqa: E402
from ui.shell import NAV_KEY, PAGES  # noqa: E402
from ui.views import (  # noqa: E402
    architecture, ask, evaluation, guidelines, safety, styleguide, technical,
)

RENDERERS = {
    "Ask": ask.render,
    "Evaluation": evaluation.render,
    "Safety": safety.render,
    "Architecture": architecture.render,
    "Sources": guidelines.render,
    "Technical": technical.render,
}

TITLES = {
    "Ask": "Ask",
    "Evaluation": "Evaluation",
    "Safety": "Safety & abstention",
    "Architecture": "Architecture",
    "Sources": "Guidelines & sources",
    "Technical": "Technical details",
}


def main() -> None:
    st.session_state.setdefault(NAV_KEY, PAGES[0][0])

    dev = st.query_params.get("dev") == "1"
    active = "Style guide" if dev else st.session_state[NAV_KEY]

    theme.page_config(TITLES.get(active, active))
    theme.inject()

    # One context per rerun, shared by the rail and the page.
    ctx = shell.load_context()
    st.session_state["_ctx"] = ctx

    chosen = shell.render_rail(st.session_state[NAV_KEY])
    if chosen and chosen != st.session_state[NAV_KEY]:
        st.session_state[NAV_KEY] = chosen
        st.rerun()

    if dev:
        styleguide.render(ctx)
        shell.render_footer()
        return

    c.page_header(TITLES[active], shell.corpus_meta_line(ctx))
    c.degraded_band(ctx.down_services)

    RENDERERS[active](ctx)
    shell.render_footer()


if __name__ == "__main__":
    main()

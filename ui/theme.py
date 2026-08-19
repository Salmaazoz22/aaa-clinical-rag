# -*- coding: utf-8 -*-
"""Stylesheet loading and injection.

One stylesheet, read from `assets/theme.css`, cached, injected once per page as
the first statement of the page. Token values are spliced in from `ui/tokens.py`
so a colour is defined in exactly one place.

**The injection payload must begin with `<style>`.** In CommonMark, `<style>`
opens a *type 1* HTML block, which ends only at its closing tag — blank lines
inside are safe. A payload that opened with anything else (a `<link>`, a
newline, a comment) would open a *type 6* block instead, which terminates at the
first blank line, and every CSS rule after that point would render as visible
text on the page. That is not hypothetical: it is a defect this project has
already shipped once. `tests/test_ui.py` renders these payloads through a real
CommonMark parser to keep it fixed.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui import tokens
from ui.branding import page_title

ROOT = Path(__file__).resolve().parents[1]
CSS_PATH = ROOT / "assets" / "theme.css"
#: Streamlit serves `static/` from the directory holding the ENTRYPOINT script,
#: not from the project root — which is why this lives under ui/.
FAVICON = Path(__file__).resolve().parent / "static" / "favicon.png"

_TOKEN_MARKER = "/* @@TOKENS@@ */"


@st.cache_data(show_spinner=False)
def _stylesheet() -> str:
    """The stylesheet with tokens spliced in. Cached — it never changes at runtime."""
    css = CSS_PATH.read_text(encoding="utf-8")
    if _TOKEN_MARKER not in css:
        raise RuntimeError(
            f"{CSS_PATH.name} is missing the {_TOKEN_MARKER} marker, so design tokens "
            f"cannot be injected. Restore the marker inside the :root block."
        )
    return css.replace(_TOKEN_MARKER, tokens.as_css_variables())


def stylesheet_payload() -> str:
    """The exact string handed to `st.markdown`. Starts with `<style>`, ends with `</style>`."""
    return f"<style>\n{_stylesheet()}\n</style>"


def page_config(title: str) -> None:
    """`st.set_page_config` with a real favicon file — never an emoji."""
    icon = str(FAVICON) if FAVICON.exists() else None
    st.set_page_config(
        page_title=page_title(title),
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject() -> None:
    """Inject the stylesheet. Call once, first, per page render."""
    st.markdown(stylesheet_payload(), unsafe_allow_html=True)

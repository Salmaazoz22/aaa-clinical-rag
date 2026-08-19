# -*- coding: utf-8 -*-
"""Tests for the Streamlit frontend (ui/).

Two layers:

* **Structural tests** run everywhere, with no backend. They assert the property
  that matters most about this UI — that it is a *client* and contains no second
  RAG implementation — and they exercise the pure render helpers.
* **End-to-end tests** run the real Streamlit script through `AppTest` against a
  live API, and are skipped when no API is reachable. They are what proves a
  page actually renders rather than merely importing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UI_DIR = ROOT / "ui"
APP = UI_DIR / "app.py"


# ---------------------------------------------------------------------------
# The architectural invariant: the UI is a client, not a second pipeline
# ---------------------------------------------------------------------------

FORBIDDEN_MODULES = ("generation", "retrieval", "vectordb", "ingestion")


def _ui_sources() -> list[Path]:
    return sorted(p for p in UI_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def test_ui_has_sources():
    assert _ui_sources(), "no UI sources found"


@pytest.mark.parametrize("path", _ui_sources(), ids=lambda p: p.name)
def test_ui_never_imports_the_rag_pipeline(path: Path):
    """The frontend must reach the pipeline over HTTP and no other way.

    Importing any of these in-process would create a second retrieval path,
    bypass the API's safety and validation layers, and make the UI capable of
    producing numbers the backend never produced.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])

    leaked = imported & set(FORBIDDEN_MODULES)
    assert not leaked, f"{path.name} imports pipeline module(s) {sorted(leaked)}; the UI must be HTTP-only"


def test_answers_are_never_cached():
    """Metadata may be cached; a clinical answer may not."""
    from ui import api_client

    assert not hasattr(api_client.answer, "clear"), "api_client.answer must not be @st.cache_data"
    for cached in (api_client._meta, api_client._corpus, api_client._evaluation, api_client._chunk):
        assert hasattr(cached, "clear"), f"{cached} should be cached static metadata"


def test_metadata_cache_key_includes_the_backend_address():
    """Repointing the UI at another backend must not serve the old one's metadata."""
    import inspect

    from ui import api_client

    for cached in (api_client._meta, api_client._corpus, api_client._evaluation, api_client._chunk):
        first = list(inspect.signature(cached).parameters)[0]
        assert first == "backend", f"{cached.__name__} must take the backend URL as its cache key"


def test_health_is_not_cached():
    """/health is the freshness probe; caching it would defeat the status strip."""
    from ui import api_client

    assert not hasattr(api_client.health, "clear")


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def test_esc_escapes_backend_text():
    """Guideline text and model prose are written into raw HTML; they must be escaped."""
    from ui.components import esc

    assert esc("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"
    assert esc(None) == "—"
    assert esc('a "b" & c') == "a &quot;b&quot; &amp; c"


def test_check_never_shows_green_for_an_unrun_check():
    """A check the validator could not run must be grey, never a pass."""
    from ui.components import _check

    assert 'class="check y"' in _check(True, "ok", "bad")
    assert 'class="check n"' in _check(False, "ok", "bad")
    grey = _check(None, "ok", "bad", "not stated")
    assert 'class="check o"' in grey
    assert "not stated" in grey


def test_findings_are_attributed_to_the_right_citation():
    from ui.components import _findings_by_citation

    result = {
        "validation": {
            "findings": [
                {"code": "citation_metadata_mismatch", "location": "citations[0].page"},
                {"code": "retrieval_score_mismatch", "location": "citations[2].retrieval_score"},
                {"code": "uncited_claim", "location": "supporting_evidence[1]"},
                {"code": "missing_field", "location": "confidence"},
            ]
        }
    }
    grouped = _findings_by_citation(result)
    assert set(grouped) == {0, 2}
    assert grouped[0][0]["code"] == "citation_metadata_mismatch"
    assert grouped[2][0]["code"] == "retrieval_score_mismatch"


def test_answer_text_export_keeps_citations_attached():
    from ui.views.ask import _answer_text

    result = {
        "query": "Q?",
        "refused": False,
        "answer": {
            "recommendation": "55 mm.",
            "confidence": "High",
            "citations": [
                {"document": "ESVS_2024", "section": "Repair", "page": 26,
                 "chunk_id": "ESVS_2024__p26-26__c0277", "retrieval_score": 0.8630,
                 "excerpt": "a diameter below 55 mm"},
            ],
            "supporting_evidence": [{"claim": "Threshold is 55 mm.", "chunk_id": "ESVS_2024__p26-26__c0277"}],
            "disclaimer": "Not clinically validated.",
        },
        "validation": {"ok": True, "n_errors": 0, "n_warnings": 0},
        "retrieval": {"n_retrieved": 5, "n_used": 3, "n_dropped_below_threshold": 2},
        "settings": {"score_threshold": 0.75},
    }
    text = _answer_text(result)
    assert "ESVS_2024__p26-26__c0277" in text
    assert "CITATION VALIDATION: PASS" in text
    assert "3 used as evidence" in text
    assert "Not clinically validated." in text


def test_demo_questions_never_state_an_expected_answer():
    """Demo copy may describe pipeline mechanism; it must not pre-write content."""
    from ui.demo_questions import DEMOS

    assert len(DEMOS) >= 6
    assert {d.kind for d in DEMOS} == {"answer", "refuse"}
    for demo in DEMOS:
        assert demo.question.strip()
        assert demo.expects.strip()
        # A pre-written answer would show up as a diameter/interval claim.
        assert " mm" not in demo.expects
        assert " cm" not in demo.expects


# ---------------------------------------------------------------------------
# End-to-end: the real script, against the real API
# ---------------------------------------------------------------------------

def _api_is_up() -> bool:
    try:
        import requests

        from ui import api_client

        return requests.get(f"{api_client.base_url()}/health", timeout=3).status_code == 200
    except Exception:  # noqa: BLE001
        return False


live = pytest.mark.skipif(not _api_is_up(), reason="no Clinical RAG API reachable")

PAGE_NAMES = [
    "Ask",
    "Evaluation",
    "Safety & Abstention",
    "Architecture",
    "Guidelines & Sources",
    "Technical Details",
]


def _app(timeout: int = 180):
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP), default_timeout=timeout)


@live
def test_app_renders_without_exception():
    at = _app().run()
    assert not at.exception, [str(e) for e in at.exception]


@live
@pytest.mark.parametrize("page", PAGE_NAMES)
def test_every_page_renders_without_exception(page: str):
    at = _app().run()
    assert not at.exception, [str(e) for e in at.exception]
    at.radio[0].set_value(page).run()
    assert not at.exception, f"{page} raised: {[str(e) for e in at.exception]}"
    assert at.markdown, f"{page} rendered nothing"


@live
def test_evaluation_page_shows_the_frozen_final20_numbers():
    """The published P@1 must reach the page unaltered."""
    at = _app().run()
    at.radio[0].set_value("Evaluation").run()
    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "Frozen evaluation results" in body
    assert "0.5500" in body, "shipped final20 P@1 not rendered"
    assert "0.6642" in body, "shipped final20 MRR not rendered"


@live
def test_guidelines_page_lists_the_real_corpus():
    at = _app().run()
    at.radio[0].set_value("Guidelines & Sources").run()
    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    for document_id in ("ESVS_2024", "NICE_NG156", "USPSTF_2019", "SVS_2018"):
        assert document_id in body


@live
def test_safety_page_runs_the_patient_specific_gate_for_real():
    """Clicks the demo and asserts the API's own safety verdict is displayed."""
    at = _app(timeout=300).run()
    at.radio[0].set_value("Safety & Abstention").run()
    assert not at.exception

    button = next(b for b in at.button if b.key == "safety_patient")
    button.click().run()
    assert not at.exception, [str(e) for e in at.exception]

    body = " ".join(m.value for m in at.markdown)
    assert "BLOCKED" in body
    assert "explicit_patient_reference" in body


@live
def test_ask_page_runs_a_threshold_refusal_end_to_end():
    """An unrelated question must come back refused, with real scores shown."""
    at = _app(timeout=300).run()
    at.text_area[0].set_value("What is the best recipe for sourdough bread?").run()
    assert not at.exception

    ask_button = next(b for b in at.button if b.label == "Ask")
    ask_button.click().run()
    assert not at.exception, [str(e) for e in at.exception]

    body = " ".join(m.value for m in at.markdown)
    assert "Refused" in body or "REFUSED" in body
    assert "deliberate refusal" in body


@live
def test_empty_question_is_handled_without_calling_the_api():
    at = _app().run()
    ask_button = next(b for b in at.button if b.label == "Ask")
    ask_button.click().run()
    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "Enter a question first" in body


def test_offline_experience_explains_how_to_start_the_backend(monkeypatch):
    """With no backend, the UI must explain itself rather than show a traceback."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CLINICAL_RAG_API_URL", "http://127.0.0.1:9")  # nothing listens here
    at = AppTest.from_file(str(APP), default_timeout=60).run()

    assert not at.exception, [str(e) for e in at.exception]
    body = " ".join(m.value for m in at.markdown)
    assert "Backend unavailable" in body
    assert "uvicorn api.main:app" in body
    assert "Traceback" not in body


# ---------------------------------------------------------------------------
# CSS injection
#
# Regression cover for a bug that rendered the whole stylesheet as visible page
# text. The payload used to begin with `<link>`; `<link>` opens a CommonMark
# *type 6* HTML block, and a type 6 block ends at the FIRST BLANK LINE. The
# stylesheet has blank lines between rule groups, so the HTML block closed part
# way through the CSS and every rule after that point was parsed as Markdown
# prose. `<style>` is a *type 1* block, which ends only at its closing tag, so
# the fix is that the stylesheet payload starts with `<style>` and the font
# links are injected separately.
#
# These tests render the payloads through a CommonMark parser rather than
# eyeballing the string, because the string always looked correct — it was the
# parse that was wrong. AppTest stores payloads unrendered, so the page-level
# tests could not have caught this on their own.
# ---------------------------------------------------------------------------

def _md():
    markdown_it = pytest.importorskip("markdown_it", reason="needs a CommonMark parser")
    return markdown_it.MarkdownIt("commonmark", {"html": True})


def test_css_payload_starts_with_the_style_tag():
    """A type 1 HTML block must open the payload; nothing may precede it."""
    from ui import theme

    css = theme._css()
    assert css.startswith("<style>"), (
        "the stylesheet payload must begin with <style> so CommonMark treats it as a "
        "type 1 HTML block, which is the only kind that survives blank lines"
    )
    assert css.rstrip().endswith("</style>")


def test_css_renders_as_a_single_html_block():
    md = _md()
    from ui import theme

    kinds = {t.type for t in md.parse(theme._css())}
    assert kinds == {"html_block"}, f"stylesheet did not stay one HTML block: {sorted(kinds)}"


def test_no_css_is_rendered_outside_the_style_element():
    """The decisive check: zero visible bytes come out of the stylesheet."""
    md = _md()
    from ui import theme

    html = md.render(theme._css())
    start, end = html.find("<style>"), html.find("</style>")
    assert start != -1 and end != -1
    outside = (html[:start] + html[end + len("</style>"):]).strip()
    assert outside == "", f"{len(outside)} chars of stylesheet leaked onto the page: {outside[:200]!r}"


@pytest.mark.parametrize("rule", [
    "h1, h2, h3, h4",
    'section[data-testid="stSidebar"]',
    ".masthead {",
    ".badge {",
    ".card {",
    ".stButton > button",
])
def test_rules_after_a_blank_line_stay_inside_the_stylesheet(rule: str):
    """Each of these sits after a blank line and used to leak as page text."""
    md = _md()
    from ui import theme

    html = md.render(theme._css())
    inside = html[html.find("<style>"):html.find("</style>")]
    assert rule in inside, f"{rule!r} is not inside <style> — it would render as visible text"


def test_font_links_render_as_html_not_text():
    md = _md()
    from ui import theme

    fonts = theme._fonts()
    assert "\n\n" not in fonts, "a blank line would close the type 6 HTML block early"
    kinds = {t.type for t in md.parse(fonts)}
    assert kinds == {"html_block"}
    assert "&lt;" not in md.render(fonts)


def test_the_palette_the_design_calls_for_is_present():
    """The fix must not have flattened the design into a default theme."""
    from ui import theme

    css = theme._css()
    for colour in (theme.BG, theme.INK, theme.ACCENT, theme.GREEN, theme.AMBER, theme.RED):
        assert colour in css, f"{colour} missing from the stylesheet"
    assert theme.BG == "#F6F7F9"
    assert theme.INK == "#13171E"
    assert theme.ACCENT == "#2B4C9B"
    for family in ("IBM Plex Serif", "IBM Plex Sans", "IBM Plex Mono"):
        assert family in css, f"{family} missing from the stylesheet"


def test_fonts_and_css_are_injected_as_separate_payloads():
    """Concatenating them would re-create the original bug."""
    import inspect

    from ui import theme

    source = inspect.getsource(theme.apply)
    assert source.count("st.markdown(") == 2, "fonts and stylesheet must be two separate calls"
    assert "_fonts()" in source and "_css()" in source


def test_joining_fonts_and_css_reproduces_the_original_bug():
    """Characterises the root cause, so the reason for two calls stays legible.

    Concatenated, the payload opens on `<link>` — a CommonMark type 6 block —
    which closes at the first blank line inside the CSS. Everything after that
    leaks onto the page as prose. This is exactly the reported symptom, and it
    is why `apply()` must never join these two payloads.
    """
    md = _md()
    from ui import theme

    joined = theme._fonts() + "\n" + theme._css()
    html = md.render(joined)
    assert "<p>" in html, (
        "expected the concatenated payload to leak CSS as paragraphs; if it no longer "
        "does, the two-call split may no longer be necessary — re-check before removing it"
    )
    # The exact reported symptom: a CSS selector rendered as a paragraph of text.
    assert "<p>h1, h2, h3, h4" in html

    # And the fix does not have that symptom.
    assert "<p>" not in md.render(theme._css())


# ---------------------------------------------------------------------------
# Every payload on every page must render as HTML, not as text
# ---------------------------------------------------------------------------

def _payload_problems(payload: str, md) -> list[str]:
    if "<" not in payload:
        return []
    html = md.render(payload)
    problems = []
    if "&lt;" in html:
        problems.append("HTML escaped into literal text")
    if "<pre><code>" in html:
        problems.append("became an indented code block")
    if "<style>" in payload:
        start, end = html.find("<style>"), html.find("</style>")
        outside = (html[:start] + html[end + len("</style>"):]).strip() if start != -1 else html
        if outside:
            problems.append("CSS rendered outside <style>")
    return problems


@live
@pytest.mark.parametrize("page", PAGE_NAMES)
def test_no_page_renders_raw_html_or_css_as_text(page: str):
    """Render every markdown payload the page emits and assert none shows as text."""
    md = _md()
    at = _app().run()
    if page != "Ask":
        at.radio[0].set_value(page).run()
    assert not at.exception, [str(e) for e in at.exception]

    broken = []
    for payload in (m.value for m in at.markdown):
        problems = _payload_problems(payload, md)
        if problems:
            broken.append((problems, payload[:120]))

    assert not broken, f"{page}: {len(broken)} payload(s) render as text — {broken[:3]}"

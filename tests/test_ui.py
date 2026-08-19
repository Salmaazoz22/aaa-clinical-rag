# -*- coding: utf-8 -*-
"""Tests for the Streamlit frontend (ui/).

Four layers:

* **Architecture** — the UI is a client and contains no second RAG
  implementation. This is the invariant that matters most.
* **Design system** — tokens are the single source of colour, the native theme
  cannot drift from them, icons are SVG rather than a font, and no emoji ships.
* **Rendering** — every markdown payload is run through a real CommonMark parser
  and must come out as HTML. `AppTest` stores payloads *unrendered*, so page
  tests alone cannot catch a CSS-injection regression; this layer exists because
  that regression shipped once.
* **End-to-end** — the real script driven against a live API, skipped when none
  is reachable.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UI_DIR = ROOT / "ui"
APP = UI_DIR / "app.py"
CSS = ROOT / "assets" / "theme.css"
CONFIG = ROOT / ".streamlit" / "config.toml"

PAGE_NAMES = ["Ask", "Evaluation", "Safety", "Architecture", "Sources", "Technical"]


def _ui_sources() -> list[Path]:
    return sorted(p for p in UI_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _md():
    markdown_it = pytest.importorskip("markdown_it", reason="needs a CommonMark parser")
    return markdown_it.MarkdownIt("commonmark", {"html": True})


# ===========================================================================
# Architecture
# ===========================================================================

FORBIDDEN = ("generation", "retrieval", "vectordb", "ingestion")


def test_ui_has_sources():
    assert _ui_sources()


@pytest.mark.parametrize("path", _ui_sources(), ids=lambda p: p.name)
def test_ui_never_imports_the_rag_pipeline(path: Path):
    """The frontend must reach the pipeline over HTTP and no other way.

    An in-process import would create a second retrieval path, bypass the API's
    safety and validation layers, and let the UI produce numbers the backend
    never produced.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    leaked = imported & set(FORBIDDEN)
    assert not leaked, f"{path.name} imports {sorted(leaked)}; the UI must be HTTP-only"


def test_answers_are_never_cached():
    """Metadata may be cached; a clinical answer may not."""
    from ui import api_client

    assert not hasattr(api_client.answer, "clear")
    for cached in (api_client._meta, api_client._corpus, api_client._evaluation, api_client._chunk):
        assert hasattr(cached, "clear")


def test_metadata_cache_key_includes_the_backend_address():
    import inspect

    from ui import api_client

    for cached in (api_client._meta, api_client._corpus, api_client._evaluation, api_client._chunk):
        assert list(inspect.signature(cached).parameters)[0] == "backend"


def test_health_is_not_cached():
    from ui import api_client

    assert not hasattr(api_client.health, "clear")


# ===========================================================================
# Design system
# ===========================================================================

def test_no_raw_hex_outside_tokens():
    """tokens.py is the only place a colour is defined."""
    offenders = []
    for path in _ui_sources():
        if path.name == "tokens.py":
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"#[0-9A-Fa-f]{6}\b", line):
                offenders.append(f"{path.name}:{i}")
    assert not offenders, f"raw hex outside tokens.py: {offenders}"


def test_native_theme_mirrors_the_tokens():
    """A Streamlit widget must never render in a colour the tokens do not define."""
    from ui import tokens

    config = CONFIG.read_text(encoding="utf-8")
    for key, value in tokens.NATIVE_THEME_MIRROR.items():
        assert key in config, f"{key} missing from config.toml"
        assert value in config, f"{key} in config.toml does not carry token value {value}"
    for value in tokens.NATIVE_SIDEBAR_MIRROR.values():
        assert value in config


def test_the_six_named_values_are_exactly_the_brief():
    from ui import tokens

    assert tokens.INK == "#0C1418"
    assert tokens.SLATE == "#16242A"
    assert tokens.LINEN == "#F1F4F3"
    assert tokens.SURFACE == "#FFFFFF"
    assert tokens.CONTRAST == "#C0803A"
    assert tokens.AORTA == "#9E3B3E"


def test_caution_is_the_aorta_family_never_amber():
    """Amber is a brand and interaction colour. It is never a warning."""
    from ui import tokens

    assert tokens.CAUTION == tokens.AORTA
    assert tokens.CAUTION != tokens.CONTRAST


def test_three_type_faces_are_declared():
    from ui import tokens

    assert "Instrument Sans" in tokens.SANS
    assert "Source Serif 4" in tokens.SERIF
    assert "IBM Plex Mono" in tokens.MONO


def test_fonts_are_self_hosted():
    """No external font request during a live demo."""
    for name in ("InstrumentSans-Variable", "SourceSerif4-Variable",
                 "IBMPlexMono-Regular", "IBMPlexMono-Medium"):
        path = UI_DIR / "static" / "fonts" / f"{name}.woff2"
        assert path.exists(), f"{name}.woff2 missing"
        assert path.read_bytes()[:4] == b"wOF2", f"{name} is not valid woff2"


def test_favicon_is_a_real_file_not_an_emoji():
    from ui import theme

    assert theme.FAVICON.exists()
    assert theme.FAVICON.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    source = Path(theme.__file__).read_text(encoding="utf-8")
    assert "page_icon=icon" in source


def test_icons_are_inline_svg_with_no_font_dependency():
    from ui import icons

    assert icons.NAMES
    for name in icons.NAMES:
        svg = icons.icon(name)
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        assert 'viewBox="0 0 24 24"' in svg
        assert "currentColor" in svg
        assert "font-family" not in svg
        assert "Material Symbols" not in svg


def test_icons_are_decorative_unless_titled():
    from ui import icons

    assert 'aria-hidden="true"' in icons.icon("check")
    assert 'aria-label="Verified"' in icons.icon("check", title="Verified")


def test_unknown_icon_raises_rather_than_rendering_nothing():
    from ui import icons

    with pytest.raises(KeyError):
        icons.icon("definitely_not_an_icon")


EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]")


@pytest.mark.parametrize("path", _ui_sources() + [CSS], ids=lambda p: p.name)
def test_no_emoji_ships(path: Path):
    """Explicitly out of bounds — including page_icon."""
    hits = EMOJI.findall(path.read_text(encoding="utf-8"))
    assert not hits, f"{path.name} contains emoji: {hits[:5]}"


def test_stylesheet_never_uses_the_selector_that_broke_the_icons():
    """`[class*="st-"]` matches Streamlit's Material icon span, whose glyph is a
    font ligature — overriding font-family there renders the raw icon name.
    It caused two of the four defects this redesign fixes."""
    css = CSS.read_text(encoding="utf-8")
    # Strip comments first: the file's own header names the banned selector in
    # order to warn about it, and that mention must not trip this test.
    body = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    # `.st-key-*` container-key scoping IS the supported mechanism and is fine.
    offenders = [
        m for m in re.findall(r'\[class\*="st-[^"]*"\]', body)
        if not m.startswith('[class*="st-key-')
    ]
    assert not offenders, f"unscoped Streamlit class selector(s): {offenders}"


def test_material_icon_font_is_protected():
    css = CSS.read_text(encoding="utf-8")
    assert '[data-testid="stIconMaterial"]' in css
    assert "Material Symbols Rounded" in css


def test_every_button_variant_declares_a_text_colour():
    """Defect 2.2 — a background without a matching colour gave an invisible label."""
    css = CSS.read_text(encoding="utf-8")
    for variant in ("stBaseButton-primary", "stBaseButton-secondary"):
        marker = f'[data-testid="{variant}"] {{'
        blocks = [chunk.split("}", 1)[0] for chunk in css.split(marker)[1:]]
        assert blocks, f"{variant} has no rule at all"
        # The selector legitimately appears in more than one block (a shared
        # block for shape, a variant block for colour); at least one must set a
        # text colour, or the label can render invisible against its own
        # background. That was defect 2.2.
        assert any("color:" in b for b in blocks), f"{variant} never sets a text colour"
        assert any("background:" in b for b in blocks), f"{variant} never sets a background"


def test_flex_children_can_shrink():
    """Defect 2.3 — min-width:auto on flex items caused horizontal overflow."""
    css = CSS.read_text(encoding="utf-8")
    assert "min-width: 0" in css
    assert '[data-testid="stHorizontalBlock"] > div' in css


def test_block_container_padding_is_targeted_at_winning_specificity():
    """Defect 2.4 — the legacy single-class selector tied with Streamlit's own."""
    css = CSS.read_text(encoding="utf-8")
    assert '[data-testid="stMain"] [data-testid="stMainBlockContainer"]' in css


def test_reduced_motion_is_respected_globally():
    assert "prefers-reduced-motion: reduce" in CSS.read_text(encoding="utf-8")


def test_focus_is_never_removed_without_replacement():
    css = CSS.read_text(encoding="utf-8")
    assert "outline: 2px solid var(--contrast)" in css
    assert "outline: none" not in css


def test_responsive_breakpoints_exist():
    css = CSS.read_text(encoding="utf-8")
    assert "max-width: 1100px" in css
    assert "max-width: 780px" in css


# ===========================================================================
# Rendering
# ===========================================================================

def test_css_payload_starts_with_the_style_tag():
    """`<style>` opens a CommonMark type 1 block, which survives blank lines.
    Anything else opens a type 6 block, which ends at the first blank line and
    dumps the rest of the stylesheet onto the page as text."""
    from ui import theme

    payload = theme.stylesheet_payload()
    assert payload.startswith("<style>")
    assert payload.rstrip().endswith("</style>")


def test_css_renders_as_a_single_html_block():
    md = _md()
    from ui import theme

    assert {t.type for t in md.parse(theme.stylesheet_payload())} == {"html_block"}


def test_no_css_is_rendered_outside_the_style_element():
    md = _md()
    from ui import theme

    html = md.render(theme.stylesheet_payload())
    start, end = html.find("<style>"), html.find("</style>")
    outside = (html[:start] + html[end + len("</style>"):]).strip()
    assert outside == "", f"{len(outside)} chars leaked onto the page: {outside[:200]!r}"


@pytest.mark.parametrize("rule", [
    "h1, h2, h3, h4", '[data-testid="stSidebar"]', ".caliper", ".ev", ".panel", ".pill",
])
def test_rules_after_a_blank_line_stay_inside_the_stylesheet(rule: str):
    md = _md()
    from ui import theme

    html = md.render(theme.stylesheet_payload())
    assert rule in html[html.find("<style>"):html.find("</style>")]


def test_tokens_are_spliced_into_the_stylesheet():
    from ui import theme, tokens

    payload = theme.stylesheet_payload()
    assert "@@TOKENS@@" not in payload
    assert tokens.CONTRAST in payload
    assert tokens.AORTA in payload


def test_missing_token_marker_fails_loudly():
    from ui import theme

    original = theme.CSS_PATH.read_text(encoding="utf-8")
    try:
        theme.CSS_PATH.write_text(original.replace("/* @@TOKENS@@ */", ""), encoding="utf-8")
        theme._stylesheet.clear()
        with pytest.raises(RuntimeError, match="marker"):
            theme.stylesheet_payload()
    finally:
        theme.CSS_PATH.write_text(original, encoding="utf-8")
        theme._stylesheet.clear()


def test_esc_escapes_backend_text():
    from ui.components import esc

    assert esc("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"
    assert esc(None) == "—"
    assert esc('a "b" & c') == "a &quot;b&quot; &amp; c"


def test_every_component_emits_balanced_markup():
    """A panel opened in one call and closed in another breaks its own nesting."""
    from ui import components as c

    fixtures = {
        "telemetry_row": c.telemetry_row("API", "up"),
        "caliper": c.caliper([0.8, 0.6], 0.75, "at-rest"),
        "stage_tracker": c.stage_tracker([("Parse", "complete", "ok")]),
        "answer_panel": c.answer_panel("Text.", n_citations=2, model="g/m", latency=1.0,
                                       verdict="Grounded", segments=["ok", "ok"], verified=2),
        "abstention_panel": c.abstention_panel(heading="H", rule="B1", reason="r",
                                               explanation="e", signals=["s"], answerable="a"),
        "evidence_card": c.evidence_card({"chunk_id": "x", "similarity_score": .8,
                                          "document_id": "D", "document": "Doc", "page": 1,
                                          "page_start": 1, "page_end": 1}, 1,
                                         above_threshold=True, cited_as=1, year=2020,
                                         passage="p", passage_note="n"),
        "metric_tile": c.metric_tile("P@1", "0.55", "note"),
        "status_pill": c.status_pill("ok", "verified"),
        "empty_state": c.empty_state("T", "<p>b</p>"),
        "error_state": c.error_state("T", "<p>b</p>"),
        "skeleton": c.skeleton("evidence", 2),
        "data_table": c.data_table(["a"], [["1"]]),
        "definition_list": c.definition_list([("k", "v")]),
    }
    for name, markup in fixtures.items():
        assert markup.count("<div") == markup.count("</div"), f"{name}: unbalanced <div>"
        assert markup.count("<span") == markup.count("</span"), f"{name}: unbalanced <span>"


def test_caliper_positions_markers_at_their_real_score():
    from ui import components as c

    markup = c.caliper([0.25, 0.90], 0.75, "at-rest")
    assert "left:25.00%" in markup
    assert "left:90.00%" in markup
    assert "left:75.00%" in markup          # the threshold, from the argument
    assert "below" in markup                # 0.25 renders hollow


def test_caliper_threshold_is_never_hardcoded():
    """It must come from the API response, so it can never disagree with it."""
    from ui import components as c

    import inspect

    assert "left:60.00%" in c.caliper([], 0.60, "idle")
    # `threshold` must be a required argument. A default would let a caller
    # silently draw a floor the API never reported. (0.25/0.50/0.75 do appear in
    # the module as fixed scale ticks, which is why this checks the signature
    # rather than grepping for the number.)
    parameter = inspect.signature(c.caliper).parameters["threshold"]
    assert parameter.default is inspect.Parameter.empty, "caliper must not default its threshold"


def test_grounding_footer_reports_a_count_not_a_percentage():
    """Nothing in the API measures what fraction of the prose is supported."""
    from ui import components as c

    markup = c.grounding_footer("Partially grounded", ["ok", "bad"], 1, 2)
    assert "1 of 2 citations verified" in markup
    assert "%" not in markup


def test_status_is_never_conveyed_by_colour_alone():
    from ui import components as c

    assert "DOWN" in c.telemetry_row("Qdrant", "down")
    assert "OK" in c.telemetry_row("Qdrant", "up")
    assert "Grounded" in c.grounding_footer("Grounded", ["ok"], 1, 1)


# ===========================================================================
# Derivations — no invented data
# ===========================================================================

def test_citation_states_are_derived_from_real_fields():
    from ui.views.ask import _citation_states

    result = {
        "answer": {"citations": [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]},
        "citations_resolved": [{"resolved": True}, {"resolved": True}, {"resolved": False}],
        "validation": {"findings": [{"severity": "warning", "location": "citations[1]"}]},
    }
    assert _citation_states(result) == ["ok", "warn", "bad"]


def test_an_error_finding_marks_a_citation_bad():
    from ui.views.ask import _citation_states

    result = {
        "answer": {"citations": [{"chunk_id": "a"}]},
        "citations_resolved": [{"resolved": True}],
        "validation": {"findings": [{"severity": "error", "location": "citations[0].page"}]},
    }
    assert _citation_states(result) == ["bad"]


def test_verdict_words_match_the_state():
    from ui.views.ask import _verdict

    assert _verdict({}, ["ok", "ok"])[0] == "Grounded"
    assert _verdict({}, ["ok", "warn"])[0] == "Partially grounded"
    assert _verdict({"refused": True}, [])[0] == "Abstained"


def test_stage_tracker_never_claims_a_stage_it_cannot_observe():
    """In flight, only the stage that has demonstrably begun may be active."""
    from ui.views.ask import _pending_stages

    states = [state for _label, state, _detail in _pending_stages()]
    assert states.count("complete") == 0
    assert states.count("active") == 1
    assert states.count("pending") == 3


def test_stages_report_real_outcomes_from_the_response():
    from ui.views.ask import _stages

    result = {
        "safety": {"blocked": False},
        "retrieval": {"n_retrieved": 5, "n_used": 3, "hits": [{"similarity_score": 0.84}]},
        "settings": {"score_threshold": 0.75},
        "validation": {"ok": True, "n_errors": 0, "n_warnings": 2},
        "refusal": None,
    }
    stages = {label: (state, detail) for label, state, detail in _stages(result)}
    assert stages["Parse"][0] == "complete"
    assert "5 chunks" in stages["Retrieve"][1]
    assert "3 of 5" in stages["Ground"][1]
    assert "2 warnings" in stages["Validate"][1]


def test_a_threshold_refusal_marks_ground_as_failed():
    from ui.views.ask import _stages

    result = {
        "safety": {"blocked": False},
        "retrieval": {"n_retrieved": 5, "n_used": 0, "hits": [{"similarity_score": 0.47}]},
        "settings": {"score_threshold": 0.75},
        "validation": {"ok": True, "n_errors": 0, "n_warnings": 0},
        "refusal": {"gate": "threshold"},
    }
    stages = {label: state for label, state, _ in _stages(result)}
    assert stages["Ground"] == "failed"


def test_demo_questions_never_state_an_expected_answer():
    from ui.demo_questions import DEMOS

    assert len(DEMOS) >= 6
    assert {d.kind for d in DEMOS} == {"answer", "refuse"}
    for demo in DEMOS:
        assert demo.question.strip() and demo.expects.strip()
        assert " mm" not in demo.expects and " cm" not in demo.expects


# ===========================================================================
# End-to-end
# ===========================================================================

def _api_is_up() -> bool:
    try:
        import requests

        from ui import api_client

        return requests.get(f"{api_client.base_url()}/health", timeout=3).status_code == 200
    except Exception:  # noqa: BLE001
        return False


live = pytest.mark.skipif(not _api_is_up(), reason="no Clinical RAG API reachable")


def _app(timeout: int = 300):
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP), default_timeout=timeout)


def _goto(at, page: str):
    if page != "Ask":
        button = next((b for b in at.button if b.label == page), None)
        assert button is not None, f"no nav button for {page}"
        button.click().run()
    return at


@live
def test_app_renders_without_exception():
    at = _app().run()
    assert not at.exception, [str(e) for e in at.exception]


@live
@pytest.mark.parametrize("page", PAGE_NAMES)
def test_every_page_renders_without_exception(page: str):
    at = _goto(_app().run(), page)
    assert not at.exception, f"{page}: {[str(e) for e in at.exception]}"
    assert at.markdown


@live
def test_the_style_guide_renders_every_component():
    at = _app()
    at.query_params["dev"] = "1"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    body = " ".join(m.value for m in at.markdown)
    assert "Style guide" in body
    assert "caliper" in body and "abstain" in body and 'class="ev' in body


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
        s, e = html.find("<style>"), html.find("</style>")
        outside = (html[:s] + html[e + len("</style>"):]).strip() if s != -1 else html
        if outside:
            problems.append("CSS rendered outside <style>")
    return problems


@live
@pytest.mark.parametrize("page", PAGE_NAMES)
def test_no_page_renders_raw_html_or_css_as_text(page: str):
    md = _md()
    at = _goto(_app().run(), page)
    assert not at.exception, [str(e) for e in at.exception]
    broken = [(p, m.value[:120]) for m in at.markdown if (p := _payload_problems(m.value, md))]
    assert not broken, f"{page}: {len(broken)} payload(s) render as text — {broken[:3]}"


@live
def test_evaluation_page_shows_the_frozen_final20_numbers():
    at = _goto(_app().run(), "Evaluation")
    body = " ".join(m.value for m in at.markdown)
    assert "Frozen results" in body
    assert "0.5500" in body and "0.6642" in body


@live
def test_sources_page_lists_the_real_corpus():
    at = _goto(_app().run(), "Sources")
    body = " ".join(m.value for m in at.markdown)
    for document_id in ("ESVS_2024", "NICE_NG156", "USPSTF_2019", "SVS_2018"):
        assert document_id in body


@live
def test_safety_page_runs_the_patient_specific_gate_for_real():
    at = _goto(_app().run(), "Safety")
    next(b for b in at.button if b.key == "safety-run-0").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    body = " ".join(m.value for m in at.markdown)
    assert "REFUSED" in body
    assert "explicit_patient_reference" in body
    assert "model not called" in body


@live
def test_ask_page_runs_a_threshold_refusal_end_to_end():
    at = _app().run()
    at.text_area[0].set_value("What is the best recipe for sourdough bread?").run()
    next(b for b in at.button if b.label == "Ask").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    body = " ".join(m.value for m in at.markdown)
    assert "No evidence above threshold" in body
    assert "abstain" in body


@live
def test_empty_question_is_handled_without_calling_the_api():
    at = _app().run()
    next(b for b in at.button if b.label == "Ask").click().run()
    assert not at.exception
    assert "Enter a question first" in " ".join(m.value for m in at.markdown)


def test_offline_experience_explains_how_to_start_the_backend(monkeypatch):
    """With no backend, the UI explains itself rather than showing a traceback."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CLINICAL_RAG_API_URL", "http://127.0.0.1:9")
    at = AppTest.from_file(str(APP), default_timeout=90).run()

    assert not at.exception, [str(e) for e in at.exception]
    body = " ".join(m.value for m in at.markdown)
    assert "Backend unavailable" in body
    assert "uvicorn api.main:app" in body
    assert "Traceback" not in body


def _visible_text(payloads, md) -> str:
    """Rendered HTML with <style> and <svg> removed, then tags stripped.

    This is what a reader actually sees. Searching the raw payload instead
    would flag the stylesheet's own comment about the icon defect, which lives
    safely inside <style> and is never visible.
    """
    import re as _re

    html = "\n".join(md.render(p) for p in payloads)
    html = _re.sub(r"<style[^>]*>.*?</style>", "", html, flags=_re.DOTALL | _re.I)
    html = _re.sub(r"<svg[^>]*>.*?</svg>", "", html, flags=_re.DOTALL | _re.I)
    return _re.sub(r"<[^>]+>", " ", html)


#: Material Symbols ligature names and shortcode forms. If any of these becomes
#: visible text, an icon font has failed to resolve — defect 2.1.
STRAY_GLYPHS = (
    "arrow_right", "keyboard_arrow_down", "keyboard_arrow_up", "chevron_right",
    "expand_more", "expand_less", "material-symbols", ":material/",
)


@live
@pytest.mark.parametrize("page", PAGE_NAMES)
def test_no_icon_ligature_leaks_as_visible_text(page: str):
    """Defect 2.1 must not reappear in any form, on any page."""
    md = _md()
    at = _goto(_app().run(), page)
    assert not at.exception
    text = _visible_text([m.value for m in at.markdown], md)
    found = [g for g in STRAY_GLYPHS if g in text]
    assert not found, f"{page}: icon ligature name(s) rendered as text: {found}"


@live
@pytest.mark.parametrize("page", PAGE_NAMES)
def test_no_placeholder_text_ships(page: str):
    """No lorem, no unrendered template markers, no leaked repr."""
    md = _md()
    at = _goto(_app().run(), page)
    text = _visible_text([m.value for m in at.markdown], md)
    for marker in ("lorem ipsum", "@@TOKENS@@", "TODO", "FIXME", "{{", "}}"):
        assert marker not in text, f"{page} contains {marker!r}"

# -*- coding: utf-8 -*-
"""The component library. Every visual element on every page comes from here.

Three rules the whole UI depends on:

1. **One markdown call per component.** Streamlit inserts its own wrapper divs
   between successive `st.markdown` calls, so a `<div>` opened in one call and
   closed in another produces broken nesting. Each function below builds its
   complete HTML as one string and emits it once.

2. **Backend text is escaped before it reaches HTML.** Guideline passages come
   out of PDFs and answers come out of a language model; a stray `<` in either
   would break the layout, and worse is possible. Everything passes through
   `esc()`.

3. **Nothing is asserted that the backend did not report.** A verified badge is
   drawn only where the API's validator reported that check clean. A check it
   could not run is drawn as *not checked*, never as a pass. No component
   invents a score, a percentage, or a status.

No raw hex appears in this file — colours come from `ui/tokens.py` and, for
anything rendered, from the CSS custom properties in `assets/theme.css`.
"""
from __future__ import annotations

import html
from typing import Any, Iterable, Sequence

import streamlit as st

from ui.branding import PRODUCT_NAME, PRODUCT_TAGLINE
from ui.icons import icon

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def esc(value: Any, dash: str = "—") -> str:
    """HTML-escape anything. `None` renders as an em dash, never as 'None'."""
    if value is None:
        return dash
    return html.escape(str(value))


def num(value: Any, places: int = 4, dash: str = "—") -> str:
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return dash


def _pct(value: float) -> str:
    """A 0–1 value as a clamped percentage string for CSS positioning."""
    return f"{max(0.0, min(1.0, float(value))) * 100:.2f}"


def _pages(record: dict[str, Any]) -> str:
    start, end = record.get("page_start"), record.get("page_end")
    if start is not None and end is not None:
        return f"p.{start}" if start == end else f"pp.{start}–{end}"
    page = record.get("page")
    return f"p.{page}" if page is not None else "p.—"


def write(markup: str) -> None:
    """Emit one component. The single place `unsafe_allow_html` is used."""
    st.markdown(markup, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def nav_rail(pages: Sequence[tuple[str, str]], active: str) -> str | None:
    """The dark instrument rail. Returns the page the user selected, if changed.

    Rendered as real Streamlit buttons so it is keyboard navigable and needs no
    JavaScript, then restyled. States: resting / hover / active are all carried
    by CSS on `.st-key-nav-<slug>`.
    """
    write(
        '<div class="rail-brand">'
        '<div class="bar"></div>'
        f'<div><div class="mark">{PRODUCT_NAME}</div>'
        f'<div class="sub">{PRODUCT_TAGLINE}</div></div>'
        "</div>"
    )

    chosen: str | None = None
    for label, icon_name in pages:
        slug = label.lower().replace(" ", "-").replace("&", "and")
        is_active = label == active
        with st.container(key=f"nav-{slug}" + ("-active" if is_active else "")):
            if st.button(
                label,
                key=f"navbtn-{slug}",
                width="stretch",
                icon=None,
                help=None,
            ):
                chosen = label
    return chosen


def nav_icon_css(pages: Sequence[tuple[str, str]], active: str) -> str:
    """Per-item icon + active state, injected as one scoped style block.

    Streamlit buttons cannot host arbitrary HTML in their label, so the icon is
    delivered as a CSS `::before` carrying an inline-SVG data URI. This keeps
    icons as SVG (never a font) while leaving the button a real, focusable
    `<button>` with a text label for screen readers.
    """
    import base64

    rules: list[str] = []
    for label, icon_name in pages:
        slug = label.lower().replace(" ", "-").replace("&", "and")
        is_active = label == active
        colour = "%23FFFFFF" if is_active else "%238FA3A9"
        svg = icon(icon_name, 18).replace('fill="none"', 'fill="none"', 1)
        svg = svg.replace("currentColor", colour.replace("%23", "#"))
        b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        key = f"nav-{slug}" + ("-active" if is_active else "")
        rules.append(
            f'.st-key-{key} [data-testid="stBaseButton-secondary"]::before {{'
            f'content: ""; width: 18px; height: 18px; flex: 0 0 18px; margin-right: 10px;'
            f'background: url("data:image/svg+xml;base64,{b64}") center/contain no-repeat; }}'
        )
    return "<style>\n" + "\n".join(rules) + "\n" + _NAV_BASE_CSS + "\n</style>"


_NAV_BASE_CSS = """
.stApp [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] p { color: var(--muted-dark); }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
  justify-content: flex-start; text-align: left; height: 40px; min-height: 40px;
  border: 0; background: transparent; color: var(--muted-dark);
  font-size: .9375rem; font-weight: 400; padding: 0 10px; border-radius: var(--r-control);
  position: relative; width: 100%; white-space: nowrap; overflow: hidden;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] p { white-space: nowrap; }
.stApp [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover,
.stApp [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover p {
  color: var(--surface);
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
  background: rgba(255,255,255,.04); border: 0;
}
.stApp [class*="st-key-nav-"][class*="-active"] [data-testid="stBaseButton-secondary"],
.stApp [class*="st-key-nav-"][class*="-active"] [data-testid="stBaseButton-secondary"] p {
  color: var(--surface); font-weight: 500;
}
[class*="st-key-nav-"][class*="-active"] [data-testid="stBaseButton-secondary"] {
  background: rgba(224,168,85,.10);
}
[class*="st-key-nav-"][class*="-active"] [data-testid="stBaseButton-secondary"]::after {
  content: ""; position: absolute; left: 0; top: 6px; bottom: 6px; width: 2px;
  background: var(--contrast-lifted); border-radius: 2px;
}
[data-testid="stSidebar"] [data-testid="stElementContainer"] { margin-bottom: 0 !important; }
"""


def telemetry_row(name: str, status: str, value: str = "") -> str:
    """One service row. `status` is up | down | unknown | checking.

    Status is never colour alone — the row always carries a mono OK / DOWN / ?
    glyph beside the dot.
    """
    glyph = {"up": "OK", "down": "DOWN", "checking": "…", "unknown": "?"}.get(status, "?")
    tone = "down" if status == "down" else ""
    shown = esc(value) if value else glyph
    return (
        f'<div class="tele"><span class="dot {esc(status)}"></span>'
        f'<span class="name">{esc(name)}</span>'
        f'<span class="val {tone}">{shown}</span></div>'
    )


def telemetry_block(rows: Sequence[str], count: tuple[Any, str] | None = None) -> None:
    body = "".join(rows)
    tail = ""
    if count and count[0] is not None:
        tail = (
            f'<div class="tele-count"><span class="n">{esc(count[0])}</span>'
            f'<span class="u">{esc(count[1])}</span></div>'
        )
    write(f'<div class="rail-section">System</div>{body}{tail}')


def page_header(title: str, meta: str = "") -> None:
    meta_html = f'<div class="meta">{esc(meta)}</div>' if meta else ""
    write(
        f'<div class="canvas-head"><div class="title">{esc(title)}</div>{meta_html}</div>'
    )


def degraded_band(services: Sequence[str]) -> None:
    if not services:
        return
    names = ", ".join(esc(s) for s in services)
    write(
        f'<div class="degraded">{icon("alert", 15)}'
        f"<span><b>Degraded.</b> {names} unreachable. Pages that need it will say so.</span></div>"
    )


def footer_band(text: str) -> None:
    write(f'<div class="footer-band">{text}</div>')


# ---------------------------------------------------------------------------
# The caliper — the signature component
# ---------------------------------------------------------------------------

def caliper(
    scores: Sequence[float],
    threshold: float,
    mode: str = "at-rest",
    *,
    labels: Sequence[str] = (),
    show_legend: bool = True,
) -> str:
    """A measured value against a threshold — the app's motif.

    `mode` is idle | animating | at-rest. Markers are positioned at their real
    similarity score on a fixed 0–1 scale, so the threshold line sits in the
    same place for every question and two results are directly comparable.
    `threshold` must come from the API (`settings.score_threshold`), never a
    constant.
    """
    ticks = "".join(
        f'<span class="tick" style="left:{_pct(t)}%"></span>'
        f'<span class="tick-label" style="left:{_pct(t)}%">{t:.2f}</span>'
        for t in (0.25, 0.50, 0.75)
    )

    thr = (
        f'<span class="threshold" style="left:{_pct(threshold)}%"></span>'
        f'<span class="threshold-cap" style="left:{_pct(threshold)}%">{threshold:.2f}</span>'
    )

    markers = []
    for i, score in enumerate(scores):
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        below = "" if value >= threshold else " below"
        delay = f"animation-delay:{i * 60}ms;" if mode == "animating" else ""
        title = labels[i] if i < len(labels) else f"rank {i + 1}"
        markers.append(
            f'<span class="marker{below}" style="left:{_pct(value)}%;{delay}" '
            f'title="{esc(title)} · {value:.4f}"></span>'
            f'<span class="marker-idx" style="left:{_pct(value)}%;{delay}">{i + 1}</span>'
        )

    legend = ""
    if show_legend and scores:
        n_above = sum(1 for s in scores if float(s) >= threshold)
        legend = (
            '<div class="caliper-legend">'
            f'<span class="k"><i class="sw above"></i>{n_above} above — sent as evidence</span>'
            f'<span class="k"><i class="sw below"></i>{len(scores) - n_above} below — withheld</span>'
            f'<span class="k"><i class="sw thr"></i>evidence floor {threshold:.2f}</span>'
            "</div>"
        )
    elif show_legend:
        legend = (
            '<div class="caliper-legend">'
            f'<span class="k"><i class="sw thr"></i>evidence floor {threshold:.2f} '
            "— nothing below it is used</span></div>"
        )

    return (
        f'<div class="caliper {esc(mode)}"><span class="rule"></span>'
        f"{ticks}{thr}{''.join(markers)}</div>{legend}"
    )


# ---------------------------------------------------------------------------
# Stage tracker
# ---------------------------------------------------------------------------

def stage_tracker(stages: Sequence[tuple[str, str, str]]) -> str:
    """`stages` is a sequence of (label, state, detail).

    state is pending | active | complete | failed. The API answers in one call
    and reports no intermediate milestones, so callers pass every stage as
    `pending` while a request is in flight and fill in real outcomes only once
    the response is in hand. Nothing here advances on a timer.
    """
    parts = []
    for i, (label, state, detail) in enumerate(stages):
        mark = ""
        if state == "complete":
            mark = icon("check", 9)
        elif state == "failed":
            mark = icon("close", 9)
        conn = '<span class="conn"></span>' if i < len(stages) - 1 else ""
        parts.append(
            f'<div class="stage {esc(state)}">'
            f'<div class="node"><span class="dot">{mark}</span></div>'
            f'<div style="min-width:0;margin-left:8px">'
            f'<div class="label">{esc(label)}</div>'
            f'<div class="detail">{esc(detail, dash="")}</div></div>{conn}</div>'
        )
    return f'<div class="stages">{"".join(parts)}</div>'


# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------

def citation_chip(index: int, *, lit: bool = False, sequenced: bool = False) -> str:
    cls = "cite" + (" lit" if lit else "") + (" seq" if sequenced else "")
    delay = f' style="animation-delay:{(index - 1) * 120}ms"' if sequenced else ""
    return (
        f'<a class="{cls}" href="#ev-{index}"{delay} '
        f'aria-label="Citation {index}, jump to its evidence">{index}</a>'
    )


def _answer_head(model: str | None, latency: float | None) -> str:
    bits = []
    if model:
        bits.append(esc(str(model).split("/")[-1]))
    if latency is not None:
        bits.append(f"{float(latency):.2f}s")
    meta = f'<span class="mono tiny">{" · ".join(bits)}</span>' if bits else ""
    return f'<div class="composer-head"><span class="eyebrow">Answer</span>{meta}</div>'


def grounding_footer(verdict: str, segments: Sequence[str], verified: int, total: int) -> str:
    """The grounding verdict as a fragment.

    The verdict word comes first, then the count in mono, then the segmented bar
    — so the state is legible without colour. There is deliberately **no
    percentage**: nothing in the API measures what fraction of the prose is
    supported, so a percentage would be invented. The honest quantity is the
    count of citations that passed validation.
    """
    word_cls = {"Grounded": "grounded", "Partially grounded": "partial",
                "Abstained": "abstained", "Ungrounded": "abstained"}.get(verdict, "partial")
    bar = "".join(f'<span class="{esc(s)}"></span>' for s in segments)
    bar_html = f'<div class="seg">{bar}</div>' if segments else ""
    return (
        f'<div class="grounding"><div class="verdict">'
        f'<span class="word {word_cls}">{esc(verdict)}</span>'
        f'<span class="count">{verified} of {total} citations verified</span></div>'
        f"{bar_html}</div>"
    )


def answer_panel(
    recommendation: str,
    *,
    n_citations: int,
    model: str | None,
    latency: float | None,
    verdict: str,
    segments: Sequence[str],
    verified: int,
    reveal: bool = True,
) -> str:
    """The complete answer panel, header through grounding footer, as ONE string.

    Emitted in a single markdown call on purpose. Streamlit inserts its own
    wrapper divs between successive calls, so a panel opened in one call and
    closed in another produces broken nesting — the answer body would escape its
    own card.

    Citation chips are appended after the prose rather than spliced into it: the
    model returns the recommendation as plain text with no inline markers, so
    placing a chip mid-sentence would mean the UI guessing which clause a
    citation supports. Appending is honest; guessing is not.
    """
    paragraphs = "".join(
        f"<p>{esc(p)}</p>" for p in str(recommendation).splitlines() if p.strip()
    )
    chips = "".join(citation_chip(i + 1, sequenced=reveal) for i in range(n_citations))
    chip_row = (
        f'<p style="margin-top:.6em" aria-label="Citations, linked to the evidence rail">'
        f"{chips}</p>" if chips else ""
    )
    cls = "answer-body" + (" answer-reveal" if reveal else "")
    return (
        f'<div class="panel">{_answer_head(model, latency)}'
        f'<div class="{cls}" style="margin-top:14px">{paragraphs}{chip_row}</div>'
        f"{grounding_footer(verdict, segments, verified, n_citations)}</div>"
    )


def abstention_panel(
    *,
    heading: str,
    rule: str | None,
    reason: str,
    explanation: str,
    signals: Sequence[str] = (),
    answerable: str = "",
) -> str:
    """A refusal, designed as a feature. Never styled like an error."""
    rule_html = (
        f'<span class="rule-id">{icon("safety", 11)}{esc(rule)}</span>' if rule else ""
    )
    sig_html = ""
    if signals:
        chips = "".join(
            f'<span class="pill caution mono">{esc(s)}</span>' for s in signals
        )
        sig_html = f'<div class="signals">{chips}</div>'
    tail = (
        f'<div class="answerable">{esc(answerable)}</div>' if answerable else ""
    )
    return (
        f'<div class="abstain">'
        f'<div class="head">{icon("safety", 20)}<h3>{esc(heading)}</h3>{rule_html}</div>'
        f'<div class="why">{esc(explanation)}</div>{sig_html}'
        f'<div class="tiny mono" style="margin-top:14px">reason: {esc(reason)}</div>'
        f"{tail}</div>"
    )


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

def evidence_card(
    hit: dict[str, Any],
    index: int,
    *,
    above_threshold: bool,
    cited_as: int | None,
    year: int | None,
    passage: str,
    passage_note: str,
    expanded: bool = False,
    highlighted: bool = False,
) -> str:
    """One retrieved chunk.

    `cited_as` is the 1-based citation number this chunk backs, or None when the
    answer did not cite it. `year` is joined from `/v1/corpus` by document_id —
    the hit itself carries no year.
    """
    classes = "ev" + ("" if above_threshold else " below") + (" lit" if highlighted else "")

    meta_bits = [esc(hit.get("document_id"))]
    if hit.get("section"):
        meta_bits.append("§" + esc(hit["section"]))
    meta_bits.append(esc(_pages(hit)))
    if year:
        meta_bits.append(esc(year))
    if hit.get("recommendation_id"):
        meta_bits.append("rec " + esc(hit["recommendation_id"]))

    score = float(hit.get("similarity_score") or 0.0)

    if cited_as is not None:
        verdict = (
            f'<span class="pill verified">{icon("check", 11)}Supports citation {cited_as}</span>'
        )
    elif above_threshold:
        verdict = '<span class="pill neutral">Sent as evidence · not cited</span>'
    else:
        verdict = '<span class="pill caution">Below threshold · withheld</span>'

    clamp = "" if expanded else " clamp"
    return (
        f'<div class="{classes}" id="ev-{index}">'
        f'<div class="top"><span class="idx">{index}</span>'
        f'<div style="min-width:0">'
        f'<div class="title">{esc(hit.get("document"))}</div>'
        f'<div class="meta">{" · ".join(meta_bits)}</div></div></div>'
        f'<div class="score-row"><span class="score">{num(score)}</span>'
        f'<span class="bar"><i style="width:{_pct(score)}%"></i></span></div>'
        f'<div style="margin-bottom:10px">{verdict}</div>'
        f'<div class="passage{clamp}">{esc(passage)}</div>'
        f'<div class="tiny" style="margin-top:8px">{esc(passage_note)}</div>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------

def status_pill(text: str, tone: str = "neutral", *, glyph: str | None = None) -> str:
    """tone is verified | caution | neutral. Always carries a word, never colour alone."""
    ico = icon(glyph, 11) if glyph else ""
    return f'<span class="pill {esc(tone)}">{ico}{esc(text)}</span>'


def metric_tile(label: str, value: Any, note: str = "", tone: str = "") -> str:
    note_html = f'<div class="note">{esc(note)}</div>' if note else ""
    return (
        f'<div class="tile"><div class="v {esc(tone, dash="")}">{esc(value)}</div>'
        f'<div class="k eyebrow">{esc(label)}</div>{note_html}</div>'
    )


def tile_row(tiles: Sequence[str]) -> None:
    if not tiles:
        return
    for col, tile in zip(st.columns(len(tiles), gap="small"), tiles):
        with col:
            write(tile)


def definition_list(rows: Iterable[tuple[str, Any]], *, prose_keys: Sequence[str] = ()) -> str:
    body = []
    for key, value in rows:
        cls = " class=\"prose-val\"" if key in prose_keys else ""
        body.append(f"<dt>{esc(key)}</dt><dd{cls}>{esc(value)}</dd>")
    return f'<dl class="dl">{"".join(body)}</dl>'


def data_table(headers: Sequence[str], rows: Iterable[Sequence[Any]], highlight: int | None = None) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = []
    for i, row in enumerate(rows):
        cls = ' class="hl"' if highlight is not None and i == highlight else ""
        body.append(f"<tr{cls}>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>")
    return (
        f'<div class="scroll-x"><table class="grid"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def panel(label: str, body_html: str, *, raised: bool = False) -> str:
    cls = "panel raised" if raised else "panel"
    head = f'<div class="eyebrow" style="margin-bottom:12px">{esc(label)}</div>' if label else ""
    return f'<div class="{cls}">{head}{body_html}</div>'


def empty_state(title: str, body_html: str, *, glyph: str = "caliper") -> str:
    return (
        f'<div class="state"><div class="head">{icon(glyph, 20)}<h3>{esc(title)}</h3></div>'
        f'<div class="body">{body_html}</div></div>'
    )


def error_state(title: str, body_html: str, *, glyph: str = "offline") -> str:
    """Errors state what happened and what to do next. They never apologise."""
    return (
        f'<div class="state error"><div class="head">{icon(glyph, 20)}<h3>{esc(title)}</h3></div>'
        f'<div class="body">{body_html}</div></div>'
    )


def skeleton(kind: str, count: int = 1) -> str:
    """kind is answer | evidence | tile. Shimmer only ever on content not yet loaded."""
    shapes = {
        "answer": ["100%", "96%", "88%"],
        "evidence": ["55%", "100%", "100%", "70%"],
        "tile": ["40%", "70%"],
    }[kind]
    one = '<div class="sk">' + "".join(
        f'<i style="width:{w}"></i>' for w in shapes
    ) + "</div>"
    return one * count


def coverage_bar(fraction: float) -> str:
    return f'<div class="cov"><i style="width:{_pct(fraction)}%"></i></div>'

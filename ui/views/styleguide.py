# -*- coding: utf-8 -*-
"""Hidden style guide — every component in every state, on one screen.

Not in the nav. Reach it with `?dev=1`. Its only job is to make a regression
obvious: if a component breaks, it breaks here first and visibly.

Everything on this page is rendered from fabricated *local* fixtures, clearly
labelled as such. That is the one place in this application where invented data
is legitimate, because nothing here is presented as a result — it is a rendering
of the component vocabulary itself.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ui import components as c
from ui.icons import NAMES, icon

FIXTURE_HIT: dict[str, Any] = {
    "rank": 1,
    "chunk_id": "NICE_NG156__p21-21__c0097",
    "similarity_score": 0.8430402567197084,
    "document_id": "NICE_NG156",
    "document": "Abdominal aortic aneurysm: diagnosis and management",
    "section": "Endovascular repair follow-up",
    "page": 21, "page_start": 21, "page_end": 21,
    "recommendation_id": "1.7.4",
}

PASSAGE = (
    "Use contrast-enhanced CT angiography if an endoleak is suspected. If "
    "contrast-enhanced CT angiography is contraindicated, use contrast-enhanced "
    "ultrasound."
)


def _section(title: str, note: str = "") -> None:
    c.write(f'<hr class="hair"><div class="eyebrow">{c.esc(title)}</div>'
            + (f'<div class="tiny" style="margin-top:4px">{c.esc(note)}</div>' if note else ""))


def render(ctx: Any = None) -> None:
    c.page_header("Style guide", "every component · every state · local fixtures only")

    c.write(
        c.empty_state(
            "Fixtures, not results",
            "<p>Every value on this page is a local fixture. Nothing here came from the API and "
            "nothing here is a clinical result. This page exists so a broken component is obvious "
            "at a glance.</p>",
            glyph="info",
        )
    )

    # -- palette ----------------------------------------------------------
    _section("Palette", "six named values; everything else derives from them")
    swatches = [
        ("--ink", "ink", "text on light"), ("--slate", "slate", "instrument chrome"),
        ("--linen", "linen", "reading canvas"), ("--surface", "surface", "cards"),
        ("--contrast", "contrast", "brand / interaction"), ("--aorta", "aorta", "threshold / abstention"),
        ("--verified", "verified", "grounded"), ("--neutral", "neutral", "informational"),
    ]
    cells = "".join(
        f'<div style="flex:1 1 120px;min-width:120px">'
        f'<div style="height:52px;border-radius:var(--r-control);border:1px solid var(--line);'
        f'background:var({tok})"></div>'
        f'<div class="mono tiny" style="margin-top:6px">{c.esc(tok)}</div>'
        f'<div class="tiny">{c.esc(role)}</div></div>'
        for tok, _name, role in swatches
    )
    c.write(f'<div style="display:flex;gap:10px;flex-wrap:wrap">{cells}</div>')

    # -- type -------------------------------------------------------------
    _section("Typography", "three faces, three jobs")
    c.write(
        '<div class="panel">'
        '<div style="font-size:2rem;font-weight:600;letter-spacing:-.02em;line-height:1.1">'
        "Instrument Sans — all interface</div>"
        '<div class="serif" style="font-size:1rem;line-height:1.6;margin-top:14px">'
        "Source Serif 4 — retrieved guideline passages, and only those. Setting evidence in a serif "
        "makes it read as document rather than interface.</div>"
        '<div class="mono" style="margin-top:14px;font-size:.875rem">'
        "IBM Plex Mono — 0.8430 · NICE_NG156__p21-21__c0097 · p.21 · 991 · 6.297s</div>"
        "</div>"
    )

    # -- icons ------------------------------------------------------------
    _section("Icons", "inline SVG only — no icon font, no emoji, no :material/ shortcodes")
    grid = "".join(
        f'<div style="flex:0 0 84px;text-align:center">{icon(n, 22)}'
        f'<div class="mono tiny" style="margin-top:4px">{c.esc(n)}</div></div>'
        for n in NAMES
    )
    c.write(f'<div class="panel" style="display:flex;gap:10px;flex-wrap:wrap">{grid}</div>')

    # -- caliper ----------------------------------------------------------
    _section("Caliper", "the signature component — three appearances, one motif")
    for mode, scores, label in (
        ("idle", [], "idle — a calibrated instrument waiting for a reading"),
        ("at-rest", [0.843, 0.812, 0.784, 0.771, 0.702], "at rest — a completed distribution"),
        ("at-rest", [0.476, 0.466, 0.456, 0.452, 0.446], "all below threshold — the failure explains itself"),
    ):
        c.write(f'<div class="panel"><div class="tiny" style="margin-bottom:6px">{c.esc(label)}</div>'
                + c.caliper(scores, 0.75, mode) + "</div>")

    # -- stage tracker ----------------------------------------------------
    _section("Stage tracker", "pending while in flight; real outcomes on completion — never a timer")
    c.write('<div class="panel">' + c.stage_tracker([
        ("Parse", "complete", "no safety signal"),
        ("Retrieve", "complete", "5 chunks · top 0.8430"),
        ("Ground", "complete", "5 of 5 above 0.75"),
        ("Validate", "complete", "0 errors · 8 warnings"),
    ]) + "</div>")
    c.write('<div class="panel">' + c.stage_tracker([
        ("Parse", "complete", "no safety signal"),
        ("Retrieve", "complete", "5 chunks · top 0.4765"),
        ("Ground", "failed", "0 of 5 above 0.75"),
        ("Validate", "pending", ""),
    ]) + "</div>")
    c.write('<div class="panel">' + c.stage_tracker([
        ("Parse", "active", "in flight"),
        ("Retrieve", "pending", ""), ("Ground", "pending", ""), ("Validate", "pending", ""),
    ]) + "</div>")

    # -- answer -----------------------------------------------------------
    _section("Answer panel", "grounded · partially grounded")
    c.write(c.answer_panel(
        "NICE recommends contrast-enhanced CT angiography as the first-line test when an "
        "endoleak is suspected, and contrast-enhanced ultrasound where CT is contraindicated.",
        n_citations=3, model="groq/openai/gpt-oss-120b", latency=6.297,
        verdict="Grounded", segments=["ok", "ok", "ok"], verified=3, reveal=False))
    c.write(c.answer_panel(
        "Surveillance intervals differ between the guidelines for this diameter band.",
        n_citations=4, model="groq/openai/gpt-oss-120b", latency=3.812,
        verdict="Partially grounded", segments=["ok", "warn", "ok", "bad"], verified=2, reveal=False))

    # -- abstention -------------------------------------------------------
    _section("Abstention", "designed as a feature — never styled like an error")
    c.write(c.abstention_panel(
        heading="Refused by design", rule="B1", reason="patient_specific_request",
        explanation="This question asks for a decision about a specific individual. It was refused "
                    "before retrieval reached a language model, so no patient detail was sent to a "
                    "third-party API.",
        signals=["explicit_patient_reference", "individual_demographics", "individual_directed_ask"],
        answerable="A general form of the same question can be answered — what the guidelines "
                   "recommend for the relevant population or diameter band."))
    c.write(c.abstention_panel(
        heading="No evidence above threshold", rule=None, reason="all_scores_below_threshold",
        explanation="Retrieval returned five passages and none cleared the 0.75 evidence floor. "
                    "The system will not build a clinical statement on passages that are only "
                    "topically adjacent.",
        answerable="Ask about abdominal aortic aneurysm screening, surveillance, medical management "
                   "or repair — the four guidelines this index covers."))

    # -- evidence ---------------------------------------------------------
    _section("Evidence card", "above threshold · cited · below threshold · expanded")
    left, right = st.columns(2, gap="medium")
    with left:
        c.write(c.evidence_card(FIXTURE_HIT, 1, above_threshold=True, cited_as=1, year=2020,
                                passage=PASSAGE, passage_note="Full indexed text"))
        c.write(c.evidence_card(FIXTURE_HIT, 2, above_threshold=True, cited_as=None, year=2020,
                                passage=PASSAGE, passage_note="Full indexed text"))
    with right:
        below = {**FIXTURE_HIT, "similarity_score": 0.4765}
        c.write(c.evidence_card(below, 3, above_threshold=False, cited_as=None, year=2020,
                                passage=PASSAGE, passage_note="Full indexed text"))
        c.write(c.evidence_card(FIXTURE_HIT, 4, above_threshold=True, cited_as=2, year=2020,
                                passage=PASSAGE, passage_note="Full indexed text",
                                expanded=True, highlighted=True))

    # -- pills, tiles -----------------------------------------------------
    _section("Pills and tiles")
    c.write('<div class="panel" style="display:flex;gap:8px;flex-wrap:wrap">'
            + c.status_pill("Supports citation 1", "verified", glyph="check")
            + c.status_pill("Below threshold · withheld", "caution")
            + c.status_pill("Sent as evidence · not cited", "neutral")
            + c.status_pill("0.8430", "neutral") + "</div>")
    c.tile_row([
        c.metric_tile("P@1", "0.550", "rank-1 chunk is a gold passage in 11 of 20", "accent"),
        c.metric_tile("MRR", "0.6642", "mean reciprocal rank of the first hit"),
        c.metric_tile("Recall@10", "0.7833", "share of gold passages found", "verified"),
        c.metric_tile("Abstained", "1", "no relevant evidence in top 10", "aorta"),
    ])

    # -- states -----------------------------------------------------------
    _section("Empty, error, skeleton")
    c.write(c.empty_state("Ask a question to begin",
                          "<p>Every answer arrives with the passages it was built from, each one's "
                          "similarity score, and the validator's verdict on every citation.</p>"))
    c.write(c.error_state("Backend unavailable",
                          "<p>Nothing is answering at <code>http://127.0.0.1:8000</code>. Start it "
                          "with <code>uvicorn api.main:app</code>.</p>"))
    c.write(c.error_state("Live generation unavailable",
                          "<p>No LLM key is configured. Retrieval and both refusal gates still "
                          "work.</p>", glyph="alert"))
    sk_l, sk_r = st.columns(2, gap="medium")
    with sk_l:
        c.write(c.skeleton("answer"))
    with sk_r:
        c.write(c.skeleton("evidence", 2))

    # -- table ------------------------------------------------------------
    _section("Data table")
    c.write(c.data_table(
        ["Configuration", "n", "P@1", "MRR", "Recall@10"],
        [["baseline (page-buffer)", 20, "0.4000", "0.5312", "0.6083"],
         ["V1 atomic · shipped", 20, "0.5500", "0.6642", "0.7833"],
         ["V2 atomic (not shipped)", 20, "0.5500", "0.6919", "0.7833"]],
        highlight=1))

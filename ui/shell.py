# -*- coding: utf-8 -*-
"""The application shell: nav rail, telemetry, canvas header, footer band.

"Instrument rail, reading canvas." The chrome is a dark, dense instrument
surface; the working area is a light document canvas. The contrast between them
is the structural idea — the machine on the left, the evidence in the middle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st

from ui import api_client, components as c, theme
from ui.api_client import ApiError

PAGES: tuple[tuple[str, str], ...] = (
    ("Ask", "ask"),
    ("Evaluation", "evaluation"),
    ("Safety", "safety"),
    ("Architecture", "architecture"),
    ("Sources", "sources"),
    ("Technical", "technical"),
)

NAV_KEY = "nav_page"

DISCLAIMER = (
    "<b>Evidence-retrieval prototype.</b> This system reports what four abdominal aortic "
    "aneurysm guidelines state for the populations they describe. It is not clinically "
    "validated, it does not diagnose, stage, manage or dose an individual patient, and it "
    "must not be used to make clinical decisions. Every statement it produces carries the "
    "passage it came from — verify against the cited source document and apply clinical "
    "judgement."
)


@dataclass
class Context:
    """What every page needs, fetched once per rerun."""

    health: dict[str, Any] | None
    meta: dict[str, Any] | None
    meta_error: ApiError | None
    corpus: dict[str, Any] | None

    @property
    def api_up(self) -> bool:
        return self.health is not None

    @property
    def llm_ready(self) -> bool:
        return bool((self.health or {}).get("llm", {}).get("configured"))

    @property
    def threshold(self) -> float:
        """The evidence floor in force. From the API — never a constant."""
        gen = (self.meta or {}).get("generation") or {}
        value = gen.get("score_threshold")
        return float(value) if value is not None else 0.75

    @property
    def top_k(self) -> int:
        gen = (self.meta or {}).get("generation") or {}
        return int(gen.get("top_k") or 5)

    @property
    def years(self) -> dict[str, int]:
        """document_id -> publication_year.

        The retrieval hits carry no year; it lives only in /v1/corpus. This join
        is why an evidence card can show one.
        """
        docs = (self.corpus or {}).get("documents") or []
        return {
            str(d["document_id"]): d["publication_year"]
            for d in docs
            if d.get("document_id") and d.get("publication_year")
        }

    @property
    def down_services(self) -> list[str]:
        if self.health is None:
            return ["API"]
        out = []
        if not self.health.get("qdrant"):
            out.append("Qdrant")
        if not self.health.get("index"):
            out.append("Index")
        return out


def load_context() -> Context:
    health = api_client.health_or_none()
    meta = corpus = None
    meta_error = None
    if health is not None:
        try:
            meta = api_client.meta()
            corpus = api_client.corpus()
        except ApiError as error:
            meta_error = error
    return Context(health=health, meta=meta, meta_error=meta_error, corpus=corpus)


def _telemetry(ctx: Context) -> None:
    health = ctx.health or {}
    llm = health.get("llm") or {}

    rows = [
        c.telemetry_row("API", "up" if ctx.api_up else "down"),
        c.telemetry_row("Qdrant", "up" if health.get("qdrant") else "down"),
        c.telemetry_row("Index", "up" if health.get("index") else "down"),
        c.telemetry_row(
            "LLM",
            "up" if llm.get("configured") else "unknown",
            "OK" if llm.get("configured") else "NO KEY",
        ),
    ]
    c.telemetry_block(rows, count=(f"{health['points']:,}" if health.get("points") else None, "points"))


def render_rail(active: str) -> str | None:
    """Draw the rail. Returns a newly selected page, or None."""
    with st.sidebar:
        c.write(c.nav_icon_css(PAGES, active))
        chosen = c.nav_rail(PAGES, active)
        c.write('<hr class="rail-divider">')
        ctx = st.session_state.get("_ctx")
        if ctx is not None:
            _telemetry(ctx)
        c.write('<div style="height:12px"></div>')
        if st.button("Re-check services", key="recheck", width="stretch"):
            api_client.clear_static_caches()
            st.rerun()
    return chosen


def corpus_meta_line(ctx: Context) -> str:
    """The right-hand telemetry on the canvas header. Real values only."""
    if not ctx.meta:
        return ""
    bits = []
    if ctx.corpus and ctx.corpus.get("n_documents"):
        bits.append(f"{ctx.corpus['n_documents']} sources")
    if ctx.meta.get("chunk_count") is not None:
        bits.append(f"{ctx.meta['chunk_count']:,} chunks")
    if ctx.meta.get("dimensions"):
        bits.append(f"{ctx.meta['dimensions']}-d")
    return " · ".join(bits)


def render_footer() -> None:
    c.footer_band(DISCLAIMER)

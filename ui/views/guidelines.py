# -*- coding: utf-8 -*-
"""Guidelines & Sources.

Every field on this page comes from `GET /v1/corpus`, which serves
`data/processed/document_metadata.json` — the metadata the extraction step
recorded from the PDFs themselves. Nothing here is written by hand: no invented
URL, no guessed year, no organisation name that is not stated in the source
document. Per-document chunk counts are counted in the live Qdrant collection.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ui import api_client, components as ui
from ui.api_client import ApiError


def render(health: dict[str, Any] | None) -> None:
    st.markdown("# Guidelines & Sources")
    st.markdown(
        '<p class="footnote" style="font-size:0.95rem;max-width:70ch">The authoritative corpus '
        "this system can answer from. It can answer from nothing else — a question outside these "
        "documents is refused, not approximated.</p>",
        unsafe_allow_html=True,
    )

    try:
        corpus = api_client.corpus()
    except ApiError as error:
        ui.backend_unavailable(error)
        return

    documents = corpus.get("documents") or []
    total = corpus.get("total_indexed_chunks")

    ui.stat_row([
        ui.stat("Guidelines", corpus.get("n_documents"), "authoritative source documents", "accent"),
        ui.stat("Indexed chunks", f"{total:,}" if total else "—",
                f"counted in {corpus.get('chunk_counts_source')}", "green"),
        ui.stat("Pages", sum(int(d.get("page_count") or 0) for d in documents),
                "across all four documents"),
        ui.stat("Years covered",
                _year_span(documents), "publication years"),
    ])

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    for doc in sorted(documents, key=lambda d: -(d.get("indexed_chunks") or 0)):
        _document_card(doc, total)

    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    ui.notice(
        "info",
        "How this corpus is weighted",
        "<p>Chunk counts are proportional to document length, not to authority. The ESVS 2024 "
        "guideline is a 140-page document and contributes most of the index; the SVS 2018 "
        "slide deck is short and contributes few chunks. Retrieval ranks by similarity alone "
        "and applies no per-document weighting, boost or filter, so a short document is not "
        "disadvantaged for a question it answers well — but it does mean the corpus is not "
        "balanced across organisations.</p>",
    )

    provenance = corpus.get("provenance") or {}
    st.markdown(
        f'<div class="card tight"><div class="card-label">Metadata provenance</div>'
        f'<div class="kv"><div class="k">Source file</div>'
        f'<div class="v">{ui.esc(provenance.get("file"))}</div>'
        f'<div class="k">SHA-256</div><div class="v">{ui.esc(provenance.get("sha256"))}</div>'
        f'<div class="k">Chunk counts</div>'
        f'<div class="v">{ui.esc(corpus.get("chunk_counts_source"))}</div></div>'
        f'<div class="footnote" style="margin-top:0.8rem">Every field shown above was extracted '
        f"from the PDFs during ingestion and is served verbatim. Where a document does not state "
        f"something, it is shown as missing rather than filled in.</div></div>",
        unsafe_allow_html=True,
    )


def _document_card(doc: dict[str, Any], total: int | None) -> None:
    chunks = doc.get("indexed_chunks")
    share = ""
    if chunks and total:
        share = f"{100 * chunks / total:.1f}% of the index"

    status = str(doc.get("extraction_status") or "")
    status_chip = (
        f'<span class="check y">✓ extraction {ui.esc(status)}</span>'
        if status == "ok"
        else f'<span class="check w">▲ extraction {ui.esc(status)}</span>'
    )

    rows = [
        ("Organisation", ui.esc(doc.get("source_organization"))),
        ("Document type", ui.esc(doc.get("document_type"))),
        ("Published", ui.esc(doc.get("publication_year"))),
        ("Pages", ui.esc(doc.get("page_count"))),
        ("Indexed chunks", ui.esc(f"{chunks:,}" if chunks is not None else None)),
        ("Source locator", ui.esc(doc.get("source_url"))),
        ("Source file", ui.esc(doc.get("source_file"))),
        ("Extraction library", ui.esc(doc.get("extraction_library"))),
    ]
    if doc.get("public_access") is not None:
        rows.append(("Public access", "yes" if doc["public_access"] else "not stated"))
    if doc.get("authors"):
        rows.append(("Authors", ui.esc(", ".join(doc["authors"]))))

    kv = "".join(f'<div class="k">{k}</div><div class="v">{v}</div>' for k, v in rows)

    credibility = doc.get("credibility_note")
    credibility_html = (
        f'<div class="footnote" style="margin-top:0.9rem;padding-top:0.8rem;'
        f'border-top:1px solid #E1E5EC"><b>Provenance recorded at extraction.</b> '
        f"{ui.esc(credibility)}</div>"
        if credibility
        else ""
    )

    st.markdown(
        f'<div class="card">'
        f'<div class="ev-head" style="margin-bottom:0.5rem">'
        f'<span class="ev-rank">{ui.esc(doc.get("document_id"))}</span>'
        f'<span class="ev-doc">{ui.esc(doc.get("document_name"))}</span></div>'
        f'<div class="checks" style="margin-bottom:0.85rem">'
        f'<span class="check o">{ui.esc(doc.get("publication_year"))}</span>'
        f'<span class="check o">{ui.esc(doc.get("page_count"))} pages</span>'
        + (f'<span class="check y">{ui.esc(f"{chunks:,}")} chunks · {ui.esc(share)}</span>'
           if chunks is not None else "")
        + status_chip
        + f'</div><div class="kv">{kv}</div>{credibility_html}</div>',
        unsafe_allow_html=True,
    )


def _year_span(documents: list[dict[str, Any]]) -> str:
    years = [int(d["publication_year"]) for d in documents if d.get("publication_year")]
    if not years:
        return "—"
    return f"{min(years)}–{max(years)}"

# -*- coding: utf-8 -*-
"""Guidelines & sources — one card per corpus document.

Every field comes from `GET /v1/corpus`, which serves the metadata the
extraction step recorded from the PDFs themselves. Nothing is written by hand:
no invented URL, no guessed year, no organisation name that is not stated in the
source document. Chunk counts are counted in the live Qdrant collection.
"""
from __future__ import annotations

from typing import Any

from ui import api_client, components as c
from ui.api_client import ApiError
from ui.shell import Context


def _card(doc: dict[str, Any], total: int | None) -> str:
    chunks = doc.get("indexed_chunks")
    share = (chunks / total) if (isinstance(chunks, int) and total) else 0.0
    share_txt = f"{share * 100:.1f}% of the index" if share else "—"

    status = str(doc.get("extraction_status") or "")
    status_pill = c.status_pill(
        f"extraction {status}", "verified" if status == "ok" else "caution",
        glyph="check" if status == "ok" else None,
    )

    rows = [
        ("Organisation", doc.get("source_organization")),
        ("Type", doc.get("document_type")),
        ("Published", doc.get("publication_year")),
        ("Pages", doc.get("page_count")),
        ("Indexed chunks", f"{chunks:,}" if isinstance(chunks, int) else None),
        ("Source locator", doc.get("source_url")),
        ("Source file", doc.get("source_file")),
        ("Extraction library", doc.get("extraction_library")),
    ]
    if doc.get("authors"):
        rows.append(("Authors", ", ".join(doc["authors"])))
    if doc.get("public_access") is not None:
        rows.append(("Public access", "yes" if doc["public_access"] else "not stated"))

    note = doc.get("credibility_note")
    note_html = (
        f'<div class="tiny" style="margin-top:14px;padding-top:12px;border-top:1px solid var(--line)">'
        f'<b>Recorded at extraction.</b> {c.esc(note)}</div>' if note else ""
    )

    return (
        f'<div class="panel">'
        f'<div style="display:flex;gap:10px;align-items:baseline;flex-wrap:wrap">'
        f'<span class="pill neutral mono">{c.esc(doc.get("document_id"))}</span>'
        f'{status_pill}</div>'
        f'<div style="font-weight:600;font-size:1.0625rem;line-height:1.35;margin:10px 0 4px;'
        f'max-width:68ch">{c.esc(doc.get("document_name"))}</div>'
        f'<div class="mono tiny" style="margin-bottom:14px">{c.esc(share_txt)}</div>'
        f"{c.coverage_bar(share)}"
        f'<div style="height:16px"></div>'
        f"{c.definition_list(rows)}{note_html}</div>"
    )


def render(ctx: Context) -> None:
    try:
        corpus = api_client.corpus()
    except ApiError as error:
        c.write(c.error_state("Corpus unavailable", f"<p>{c.esc(error.message)}</p>"))
        return

    docs = corpus.get("documents") or []
    total = corpus.get("total_indexed_chunks")
    years = [d["publication_year"] for d in docs if d.get("publication_year")]

    c.tile_row([
        c.metric_tile("Guidelines", corpus.get("n_documents"), "authoritative documents", "accent"),
        c.metric_tile("Indexed chunks", f"{total:,}" if total else "—",
                      f"counted in {corpus.get('chunk_counts_source')}", "verified"),
        c.metric_tile("Pages", sum(int(d.get("page_count") or 0) for d in docs), "across the corpus"),
        c.metric_tile("Years", f"{min(years)}–{max(years)}" if years else "—", "publication range"),
    ])

    c.write('<hr class="hair">')
    for doc in sorted(docs, key=lambda d: -(d.get("indexed_chunks") or 0)):
        c.write(_card(doc, total))

    c.write('<hr class="hair">')
    c.write(c.empty_state(
        "How this corpus is weighted",
        "<p>Chunk counts are proportional to document length, not to authority. ESVS 2024 is a "
        "140-page guideline and contributes most of the index; the SVS 2018 slide deck is short and "
        "contributes few chunks. Retrieval ranks by similarity alone and applies no per-document "
        "weighting, boost or filter — so a short document is not disadvantaged for a question it "
        "answers well, but the corpus is not balanced across organisations.</p>", glyph="info"))

    provenance = corpus.get("provenance") or {}
    c.write(c.panel("Metadata provenance", c.definition_list([
        ("Source file", provenance.get("file")),
        ("SHA-256", provenance.get("sha256")),
        ("Chunk counts", corpus.get("chunk_counts_source")),
    ])))

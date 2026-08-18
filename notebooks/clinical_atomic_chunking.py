# -*- coding: utf-8 -*-
"""Atomic, structure-driven chunking for the AAA clinical corpus (V1, page-safe).

This is the SHIPPED chunker. It replaced the page-buffer chunker in
`clinical_chunking.build_chunks` after Experiment 12 and the final corrected
validation on `eval/gold_standard_final20.json`.

Why boundaries come from structure
----------------------------------
The previous chunker cut on pages. Because a full guideline page already exceeds
`TARGET_CHARS`, its buffer flushed at essentially every page end: 2,109 of 2,116
chunks were single-page, and a recommendation split by a page break was split
across two chunks. The chunker had no representation of a recommendation at all.

Here the document is assembled once with page sentinels, cut at *structural
anchors* -- `Recommendation N`, numbered recommendation IDs (`1.5.4`), numbered
section headings -- and a recommendation is kept whole. Page numbers are then
recovered from the character offsets of each span, so provenance is derived
rather than stamped.

"Page-safe" is the V1 half of the decision: page boundaries still cut NARRATIVE
text, so narrative keeps the baseline's page precision, but they never cut a
recommendation. The alternative (V2, anchors only) produced chunks spanning 5.6
pages on average and was REJECTED -- the evaluation's relevance rule requires
page-range overlap, so wide chunks score better without retrieving better. See
`docs/experiment_history.md`.

What this deliberately does NOT do
----------------------------------
No query-specific logic of any kind: no intent detection, no query expansion, no
keyword bonuses, no recommendation-ID score boosts, and no per-document hardcoded
section-title lists. The anchor patterns describe how a clinical guideline is
typeset, not what any evaluation question asks.

Token safety is unchanged from `clinical_chunking`: every piece is budgeted
against the real tokenizer of the active embedding model and a recommendation is
kept whole only up to that budget. Nothing is ever silently truncated.

Single source of truth
----------------------
This module is the ONLY implementation of the atomic chunker. `build_chunks`
carries the two switches the evaluation harness needs -- `keep_page_breaks`
(False reaches the rejected V2 anchors-only shape) and `reject_citation_headings`
(False reproduces the pre-fix historical artifacts) -- so that
`eval/experimental_atomic_chunking.py` can call this function with parameters
instead of maintaining a second copy of it. The defaults are the shipped
configuration; changing a default changes the shipped index.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd

import clinical_chunking as cc

# ---------------------------------------------------------------------------
# Anchors -- structural only
# ---------------------------------------------------------------------------
_PAGE_MARKER = re.compile(r"\x00PAGE:(\d+)\x00")

# "Recommendation 11" alone on a line (ESVS / SVS house style).
_REC_HEADING = re.compile(r"(?m)^\s*Recommendation\s+(\d{1,3})\s*$")
# NICE-style numbered recommendation identifiers: "1.5.4 Offer ...".
_REC_ID = re.compile(r"(?m)^\s*(\d{1,2}\.\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\s+(?=\S)")
# Numbered section headings: "3.3 Screening for AAA", "5 Management".
_SECTION_HEADING = re.compile(r"(?m)^\s*(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+([A-Z][^\n]{4,90})$")

# A heading-shaped line carrying contents-page dot leaders is a TOC row.
_DOT_LEADER = re.compile(r"\.{4,}|(?:\.[ \t]){4,}")
_CITATION_SHAPE = re.compile(r"\bet al\b|doi:|\bAvailable at:|\[Accessed\b|\b\d{4};\s*\d+", re.I)
# A numbered reference entry ("3 Svensjo S, Bjorck M, Gurtelschmid M, Djavani") is
# shaped exactly like a numbered heading. Experiment 3 hit the same class of false
# positive. These are structural shape tests, not a list of this corpus's headings.
_AUTHOR_INITIALS = re.compile(r"\b[A-Z][a-z]+\s+[A-Z]{1,3}[,.]")
_INTERNAL_SENTENCE_BREAK = re.compile(r"\.\s+[A-Z]")

REC_ANCHOR, SECTION_ANCHOR, PAGE_ANCHOR = "recommendation", "section", "page"

# `split_text` re-tokenises the remaining span on each iteration, which is cheap
# for page-sized input but quadratic on a multi-page anchor-free stretch. Spans
# longer than this are pre-cut on blank lines. The cut is ~4x the largest possible
# token budget in characters, so no boundary `split_text` would choose is displaced.
_PRECUT_CHARS = 8000


def _looks_like_citation_line(title: str) -> bool:
    if title.count(",") >= 2:
        return True
    if ";" in title:
        return True
    if _AUTHOR_INITIALS.search(title):
        return True
    if _INTERNAL_SENTENCE_BREAK.search(title):
        return True
    return False


def _heading_is_real(title: str, line: str, reject_citation_headings: bool = True) -> bool:
    if _DOT_LEADER.search(line):
        return False  # table-of-contents row
    if _CITATION_SHAPE.search(line):
        return False  # bibliography entry
    if reject_citation_headings and _looks_like_citation_line(title):
        return False  # numbered reference masquerading as a heading
    if len(title.split()) > 12:
        return False
    if title.rstrip().endswith((",", ";", ":")):
        return False
    return True


def _precut(text: str, limit: int = _PRECUT_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    out, current = [], ""
    for para in re.split(r"\n\s*\n", text):
        if current and len(current) + len(para) > limit:
            out.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        out.append(current)
    final = []
    for piece in out:
        while len(piece) > limit:
            final.append(piece[:limit])
            piece = piece[limit:]
        if piece.strip():
            final.append(piece)
    return final


def build_marked_text(doc_pages: pd.DataFrame) -> str:
    """Document text with page sentinels, so any offset resolves to a page."""
    parts: list[str] = []
    for _, row in doc_pages.iterrows():
        if row.get("extraction_status") == "corrupted":
            continue
        parts.append(f"\x00PAGE:{int(row['page_number'])}\x00")
        parts.append(cc.page_working_text(row))
    return "\n".join(parts)


def page_at_offset(marked: str, offset: int) -> int:
    page = 1
    for m in _PAGE_MARKER.finditer(marked):
        if m.start() > offset:
            break
        page = int(m.group(1))
    return page


def strip_markers(text: str) -> str:
    return _PAGE_MARKER.sub("", text).strip()


def find_anchors(marked: str, reject_citation_headings: bool = True) -> list[tuple[int, str, str]]:
    """(offset, kind, label) for every structural anchor, de-duplicated."""
    anchors: dict[int, tuple[str, str]] = {}
    for m in _REC_HEADING.finditer(marked):
        anchors[m.start()] = (REC_ANCHOR, m.group(1))
    for m in _REC_ID.finditer(marked):
        anchors.setdefault(m.start(), (REC_ANCHOR, m.group(1)))
    for m in _SECTION_HEADING.finditer(marked):
        if m.start() in anchors:
            continue
        title = m.group(2).strip()
        if not _heading_is_real(title, m.group(0), reject_citation_headings):
            continue
        anchors[m.start()] = (SECTION_ANCHOR, f"{m.group(1)} {title}")
    return sorted((off, kind, label) for off, (kind, label) in anchors.items())


def segment(marked: str, keep_page_breaks: bool = True,
            reject_citation_headings: bool = True) -> list[dict[str, Any]]:
    """Cut into spans at anchors.

    With `keep_page_breaks` (the shipped V1 behaviour) a page break also cuts,
    except inside a recommendation, so a recommendation is never severed by
    pagination. With it False only anchors cut -- the rejected V2 shape, kept
    reachable so the V2 comparison runs against this one implementation.
    """
    hard = find_anchors(marked, reject_citation_headings=reject_citation_headings)
    points: list[tuple[int, str, str]] = list(hard)

    if keep_page_breaks:
        hard_by_off = {off: (kind, label) for off, kind, label in hard}
        hard_offsets = sorted(hard_by_off)
        for m in _PAGE_MARKER.finditer(marked):
            off = m.start()
            if off in hard_by_off:
                continue
            prior = [h for h in hard_offsets if h <= off]
            if prior and hard_by_off[prior[-1]][0] == REC_ANCHOR:
                continue  # keep the recommendation whole across the page break
            points.append((off, PAGE_ANCHOR, ""))

    points.sort()
    if not points or points[0][0] > 0:
        points.insert(0, (0, PAGE_ANCHOR, ""))

    bounds = [p[0] for p in points] + [len(marked)]
    spans, current_section = [], None
    for i, (start, kind, label) in enumerate(points):
        if kind == SECTION_ANCHOR:
            current_section = label
        body = strip_markers(marked[start : bounds[i + 1]])
        if not body:
            continue
        spans.append({
            "kind": kind, "label": label, "enclosing_section": current_section, "text": body,
            "page_start": page_at_offset(marked, start),
            "page_end": page_at_offset(marked, bounds[i + 1] - 1),
        })
    return spans


def _page_meta(doc_pages: pd.DataFrame) -> dict[int, dict[str, Any]]:
    out = {}
    for _, row in doc_pages.iterrows():
        out[int(row["page_number"])] = {
            "document_name": row.get("document_name"),
            "document_type": row.get("document_type"),
            "source_file": row.get("source_file"),
            "section_title": cc._as_optional(row.get("section_title")),
            "section_source": cc._as_optional(row.get("section_source")) or "unknown",
        }
    return out


def build_chunks(
    pages_df: pd.DataFrame,
    recs_df: pd.DataFrame | None = None,
    rec_token_budget: int | None = None,
    narrative_token_budget: int = cc.TARGET_TOKENS,
    keep_page_breaks: bool = True,
    reject_citation_headings: bool = True,
) -> list[dict[str, Any]]:
    """Anchor-driven chunks, token-budgeted so nothing is ever truncated.

    Section titles come from the SAME page-level source the previous chunker used.
    Experiment 3 established that `section_title` is metadata only -- it is never
    embedded -- so deriving it differently here would change provenance without
    being able to change retrieval.

    The default arguments ARE the shipped V1 configuration. The two switches exist
    so that the evaluation harness can reach the rejected V2 shape
    (`keep_page_breaks=False`) and reproduce the pre-fix historical artifacts
    (`reject_citation_headings=False`) without a second copy of this function.
    Changing either default changes the shipped index.
    """
    recs_df = recs_df if recs_df is not None else pd.DataFrame()
    limit = cc.max_content_tokens()
    rec_token_budget = limit if rec_token_budget is None else min(rec_token_budget, limit)
    chunks: list[dict[str, Any]] = []

    for document_id, doc_pages in pages_df.groupby("document_id", sort=True):
        doc_pages = doc_pages.sort_values("page_number")
        marked = build_marked_text(doc_pages)
        meta_by_page = _page_meta(doc_pages)
        doc_seq = 0

        for span in segment(marked, keep_page_breaks=keep_page_breaks,
                            reject_citation_headings=reject_citation_headings):
            text = span["text"]
            if len(text) < cc.MIN_VALID_CHARS:
                continue
            is_rec = span["kind"] == REC_ANCHOR
            budget = min(rec_token_budget if is_rec else narrative_token_budget, limit)

            pieces: list[str] = []
            for block in _precut(text):
                pieces.extend(cc.split_text(block, target_tokens=budget,
                                            overlap_tokens=cc.OVERLAP_TOKENS))
            if not pieces:
                continue

            page_start, page_end = span["page_start"], span["page_end"]
            meta = meta_by_page.get(page_start) or next(iter(meta_by_page.values()))
            recs_nearby: list[dict[str, Any]] = []
            for pg in range(page_start, page_end + 1):
                recs_nearby.extend(cc.recs_for_page(recs_df, document_id, pg))

            for piece in pieces:
                rec_meta = cc.attach_recommendation_metadata(piece, recs_nearby)
                if is_rec and not rec_meta["recommendation_id"]:
                    rec_meta = dict(rec_meta, recommendation_id=span["label"])
                doc_seq += 1
                section_title = meta["section_title"]
                chunks.append({
                    "chunk_id": f"{document_id}__p{page_start}-{page_end}__c{doc_seq:04d}",
                    "document_id": document_id,
                    "document_name": meta["document_name"],
                    "document_type": meta["document_type"],
                    "is_guideline": cc.is_guideline_document(meta["document_type"]),
                    "section_title": section_title,
                    "section_source": meta["section_source"],
                    "anchor_kind": span["kind"],
                    "anchor_label": span["label"] or None,
                    "enclosing_section": span["enclosing_section"],
                    "page_number": page_start,
                    "page_start": page_start,
                    "page_end": page_end,
                    "source_file": meta["source_file"],
                    "chunk_text": piece,
                    "source_excerpt": piece[:400],
                    "token_count": cc.count_tokens(piece),
                    "char_count": len(piece),
                    "content_type": cc.classify_chunk_content(piece, section_title),
                    "recommendation_id": rec_meta["recommendation_id"],
                    "recommendation_grade": rec_meta["recommendation_grade"],
                    "evidence_level": rec_meta["evidence_level"],
                })
    return chunks


def anchor_summary(pages_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Anchors found per document -- useful for inspecting coverage.

    Only ESVS 2024 and NICE NG156 carry numbered structure. USPSTF 2019 and
    SVS 2018 yield no anchors and fall back to page/token splitting, which is a
    documented limitation of this chunker, not a bug.
    """
    out = {}
    for doc_id, doc_pages in pages_df.groupby("document_id", sort=True):
        marked = build_marked_text(doc_pages.sort_values("page_number"))
        out[doc_id] = dict(Counter(k for _, k, _ in find_anchors(marked)))
    return out

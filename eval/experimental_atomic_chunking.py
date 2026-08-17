# -*- coding: utf-8 -*-
"""EXPERIMENT 12 — Project-B-inspired atomic / anchor-driven chunking.

Isolated by design. Nothing here is imported by production code, and nothing
here writes to `data/chunks/` or `data/embeddings/`. The production index on
disk is never touched; every variant is chunked, embedded and scored entirely
in memory.

What is being transferred from Project B
----------------------------------------
Project B (`aaa-clinical-rag/src/chunk.py`) does NOT chunk by page. It builds a
document-level string with page sentinels, finds *structural anchors*
(`Recommendation N`, NICE `1.5.4`-style IDs, numbered section headings), splits
at those anchors, and keeps each recommendation whole. Page numbers are then
recovered from sentinel offsets rather than stamped by the loop.

Project A chunks by page buffer: 2,109 of 2,116 baseline chunks are single-page,
because a full guideline page already exceeds TARGET_CHARS and flushes at every
page end. Recommendation boundaries are invisible to it.

What is deliberately NOT transferred
------------------------------------
None of Project B's retrieval machinery: no intent detection, no query
expansion, no keyword-overlap bonus, no recommendation-ID score boosts, no
anchor injection into the candidate pool. Retrieval here is pure dense cosine,
byte-identical in behaviour to production. The only variable is where chunk
boundaries fall.

Also not transferred: Project B's *unbounded* chunk sizes. 31.6% of its indexed
chunks overflow its own 256-token encoder window (83,516 tokens silently
dropped at encode time; largest chunk 23,079 tokens). Every variant here is
token-budgeted against the real tokenizer, as Project A already does.

Variants
--------
V1 atomic_pagesafe : hard anchors split; page boundaries still split inside
                     anchor-free stretches, so narrative keeps baseline page
                     precision. Recommendations stay whole across pages.
V2 atomic_pure     : hard anchors only, closest to Project B's shape.
V3 size_control    : BASELINE algorithm, no anchors at all, with its token and
                     character budgets enlarged to match V1's mean chunk size.
                     This is the control that separates "structure helped" from
                     "bigger chunks helped".

Run:
    python eval/experimental_atomic_chunking.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(EVAL_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import clinical_chunking as cc  # noqa: E402
from evaluate import evaluate_run, load_gold  # noqa: E402

HELDOUT_GOLD = EVAL_DIR / "gold_standard_heldout.json"
EVAL_DEPTH = 10

# ---------------------------------------------------------------------------
# Anchors
#
# Structure-driven only. These patterns describe how a clinical guideline is
# typeset, not what any evaluation question asks. No query string, gold label,
# recommendation number or topic keyword appears anywhere in this file.
# ---------------------------------------------------------------------------
_PAGE_MARKER = re.compile(r"\x00PAGE:(\d+)\x00")

# "Recommendation 11" on a line of its own (ESVS / SVS house style).
_REC_HEADING = re.compile(r"(?m)^\s*Recommendation\s+(\d{1,3})\s*$")
# NICE-style numbered recommendation identifiers: "1.5.4 Offer ...".
_REC_ID = re.compile(r"(?m)^\s*(\d{1,2}\.\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\s+(?=\S)")
# Numbered section headings: "3.3 Screening for AAA", "5 Management".
_SECTION_HEADING = re.compile(r"(?m)^\s*(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+([A-Z][^\n]{4,90})$")
# Contents-page leaders, reused from the baseline classifier's vocabulary: a
# heading-shaped line carrying dot leaders is a table-of-contents row.
_DOT_LEADER = re.compile(r"\.{4,}|(?:\.[ \t]){4,}")
# Bibliography-shaped lines that can mimic a numbered heading.
_CITATION_SHAPE = re.compile(r"\bet al\b|doi:|\bAvailable at:|\[Accessed\b|\b\d{4};\s*\d+", re.I)
# A numbered reference entry ("3 Svensjo S, Bjorck M, Gurtelschmid M, Djavani")
# is shaped exactly like a numbered heading. Experiment 3 hit the same class of
# false positive and rejected author lists and citations for the same reason.
# These are structural shape tests, not a list of this corpus's actual headings.
_AUTHOR_INITIALS = re.compile(r"\b[A-Z][a-z]+\s+[A-Z]{1,3}[,.]")
_INTERNAL_SENTENCE_BREAK = re.compile(r"\.\s+[A-Z]")


# Rejects numbered bibliography lines that are shaped like numbered headings
# ("3 Svensjo S, Bjorck M, Gurtelschmid M, Djavani"). Removing them drops 15 bogus
# USPSTF and 13 bogus ESVS anchors.
#
# HISTORICAL NOTE: eval/runs/exp12_atomic_chunking.json and the original10 /
# heldout18 / final20 rows in eval/final_evaluation_results.json were produced with
# this OFF, before the false positive was found. Those artifacts are preserved as
# historical experiments. To reproduce them exactly, set this back to False.
# The FINAL CORRECTED VALIDATION (eval/runs/final_corrected_v1_final20.json) and
# the shipped configuration use True.
REJECT_CITATION_HEADINGS = True


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

REC_ANCHOR, SECTION_ANCHOR, PAGE_ANCHOR = "recommendation", "section", "page"

# `split_text` re-tokenises the whole remaining span on every iteration, which is
# cheap for page-sized input but quadratic on a 40-page anchor-free stretch (the
# baseline never produces one). Spans longer than this are pre-cut on blank lines
# first. The cut size is ~4x the largest possible token budget in characters, so
# no chunk boundary that `split_text` would have chosen is displaced.
_PRECUT_CHARS = 8000


def _precut(text: str, limit: int = _PRECUT_CHARS) -> list[str]:
    """Cut an over-long span on paragraph boundaries before token splitting."""
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
    # A single paragraph can still exceed the limit (an unbroken table block).
    final = []
    for piece in out:
        while len(piece) > limit:
            final.append(piece[:limit])
            piece = piece[limit:]
        if piece.strip():
            final.append(piece)
    return final


def build_marked_text(doc_pages: pd.DataFrame) -> str:
    """Document-level text with page sentinels (Project B's provenance trick).

    Page content is produced by the BASELINE `page_working_text`, so the text
    being chunked is identical to the baseline's; only the boundaries differ.
    """
    parts: list[str] = []
    for _, row in doc_pages.iterrows():
        if row.get("extraction_status") == "corrupted":
            continue
        text = cc.page_working_text(row)
        parts.append(f"\x00PAGE:{int(row['page_number'])}\x00")
        parts.append(text)
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


def _heading_is_real(title: str, line: str) -> bool:
    if _DOT_LEADER.search(line):
        return False  # table-of-contents row
    if _CITATION_SHAPE.search(line):
        return False  # bibliography entry
    if REJECT_CITATION_HEADINGS and _looks_like_citation_line(title):
        return False  # numbered reference entry masquerading as a heading
    if len(title.split()) > 12:
        return False
    if title.rstrip().endswith((",", ";", ":")):
        return False
    return True


def find_anchors(marked: str) -> list[tuple[int, str, str]]:
    """(offset, kind, label) for every structural anchor, de-duplicated."""
    anchors: dict[int, tuple[str, str]] = {}

    for m in _REC_HEADING.finditer(marked):
        anchors[m.start()] = (REC_ANCHOR, m.group(1))

    for m in _REC_ID.finditer(marked):
        anchors.setdefault(m.start(), (REC_ANCHOR, m.group(1)))

    for m in _SECTION_HEADING.finditer(marked):
        if m.start() in anchors:
            continue
        line = m.group(0)
        title = m.group(2).strip()
        if not _heading_is_real(title, line):
            continue
        anchors[m.start()] = (SECTION_ANCHOR, f"{m.group(1)} {title}")

    return sorted((off, kind, label) for off, (kind, label) in anchors.items())


def _page_anchor_offsets(marked: str) -> list[int]:
    return [m.start() for m in _PAGE_MARKER.finditer(marked)]


def segment(marked: str, keep_page_breaks: bool) -> list[dict[str, Any]]:
    """Cut `marked` into spans at anchors.

    With `keep_page_breaks`, a page boundary also cuts -- except inside a
    recommendation span, so a recommendation is never severed by pagination.
    """
    hard = find_anchors(marked)
    points: list[tuple[int, str, str]] = list(hard)

    if keep_page_breaks:
        hard_by_off = {off: (kind, label) for off, kind, label in hard}
        hard_offsets = sorted(hard_by_off)
        for off in _page_anchor_offsets(marked):
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
    spans = []
    current_section: str | None = None
    for i, (start, kind, label) in enumerate(points):
        if kind == SECTION_ANCHOR:
            current_section = label
        body = strip_markers(marked[start : bounds[i + 1]])
        if not body:
            continue
        spans.append(
            {
                "kind": kind,
                "label": label,
                "enclosing_section": current_section,
                "text": body,
                "page_start": page_at_offset(marked, start),
                "page_end": page_at_offset(marked, bounds[i + 1] - 1),
            }
        )
    return spans


# ---------------------------------------------------------------------------
# Chunk assembly
# ---------------------------------------------------------------------------

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


def build_atomic_chunks(
    pages_df: pd.DataFrame,
    recs_df: pd.DataFrame,
    keep_page_breaks: bool,
    rec_token_budget: int,
    narrative_token_budget: int = cc.TARGET_TOKENS,
) -> list[dict[str, Any]]:
    """Anchor-driven chunks, token-budgeted so nothing is ever truncated.

    Section titles are taken from the SAME page-level source the baseline uses.
    Experiment 3 established that `section_title` is metadata only -- it is
    never embedded -- so changing its derivation here would add a second,
    non-retrieval variable to the comparison for no possible metric effect.
    """
    limit = cc.max_content_tokens()
    chunks: list[dict[str, Any]] = []

    for document_id, doc_pages in pages_df.groupby("document_id", sort=True):
        doc_pages = doc_pages.sort_values("page_number")
        marked = build_marked_text(doc_pages)
        meta_by_page = _page_meta(doc_pages)
        doc_seq = 0

        for span in segment(marked, keep_page_breaks=keep_page_breaks):
            text = span["text"]
            if len(text) < cc.MIN_VALID_CHARS:
                continue

            is_rec = span["kind"] == REC_ANCHOR
            budget = min(rec_token_budget if is_rec else narrative_token_budget, limit)
            pieces = []
            for block in _precut(text):
                pieces.extend(
                    cc.split_text(block, target_tokens=budget, overlap_tokens=cc.OVERLAP_TOKENS)
                )
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
                chunks.append(
                    {
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
                    }
                )
    return chunks


def build_size_control_chunks(
    pages_df: pd.DataFrame,
    recs_df: pd.DataFrame,
    target_tokens: int,
    target_chars: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    """BASELINE algorithm with enlarged budgets. No anchors, no structure.

    Isolates chunk SIZE from chunk STRUCTURE: if V1/V2 only win because their
    chunks are longer, this control wins too.
    """
    orig_split, orig_target, orig_max = cc.split_text, cc.TARGET_CHARS, cc.MAX_CHARS
    try:
        cc.split_text = lambda text, **kw: orig_split(
            text, target_tokens=target_tokens, overlap_tokens=cc.OVERLAP_TOKENS
        )
        cc.TARGET_CHARS = target_chars
        cc.MAX_CHARS = max_chars
        return cc.build_chunks(pages_df, recs_df)
    finally:
        cc.split_text, cc.TARGET_CHARS, cc.MAX_CHARS = orig_split, orig_target, orig_max


# ---------------------------------------------------------------------------
# Index + retrieval (pure dense cosine; identical behaviour to production)
# ---------------------------------------------------------------------------

def index_chunks(chunks: list[dict[str, Any]], model) -> dict[str, Any]:
    quality = cc.validate_chunks(chunks)
    indexable = cc.embeddable_chunks(chunks, quality)
    texts = [c["chunk_text"] for c in indexable]
    vectors = model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float32)
    return {"chunks": indexable, "vectors": vectors, "quality": quality, "all_chunks": chunks}


def retrieve(query: str, index: dict[str, Any], model, top_k: int = EVAL_DEPTH) -> list[dict[str, Any]]:
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    scores = index["vectors"] @ q
    order = np.argsort(-scores)[:top_k]
    hits = []
    for rank, i in enumerate(order, start=1):
        c = index["chunks"][int(i)]
        hits.append(
            {
                "rank": rank,
                "similarity_score": float(scores[int(i)]),
                "chunk_id": c["chunk_id"],
                "document_id": c["document_id"],
                "section": c.get("section_title"),
                "page": c["page_number"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "chunk_text": c["chunk_text"],
                "anchor_kind": c.get("anchor_kind"),
                "recommendation_id": c.get("recommendation_id"),
            }
        )
    return hits


def load_production_index() -> dict[str, Any]:
    d = ROOT / "data" / "embeddings"
    chunks = json.loads((d / "embedded_chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(d / "embeddings.npy")
    if len(chunks) != len(vectors):
        raise RuntimeError("production index is out of sync")
    # `all_chunks` is the full pre-filter set, so the control's "total chunks"
    # column is comparable with the variants' (2,116, not the 1,330 indexed).
    produced = json.loads((ROOT / "data" / "chunks" / "chunks.json").read_text(encoding="utf-8"))
    return {
        "chunks": chunks,
        "vectors": vectors,
        "quality": None,
        "all_chunks": produced.get("chunks") or chunks,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def widen_page_ranges(index: dict[str, Any], half_width: int, max_page: dict[str, int]) -> dict[str, Any]:
    """Same vectors, same ranking -- only the page range metadata is widened.

    Page overlap is one half of the frozen relevance rule, so a chunker that
    produces wider page spans is easier to score as relevant even when it
    retrieves nothing new. This control widens the PRODUCTION chunks to V2's
    mean span without changing what is retrieved: whatever it gains is pure
    measurement artifact, not retrieval quality.
    """
    chunks = []
    for c in index["chunks"]:
        c = dict(c)
        top = max_page.get(c["document_id"], c["page_end"])
        c["page_start"] = max(1, int(c["page_start"]) - half_width)
        c["page_end"] = min(top, int(c["page_end"]) + half_width)
        chunks.append(c)
    return {"chunks": chunks, "vectors": index["vectors"], "quality": None, "all_chunks": chunks}


def narrow(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse every hit to its start page before scoring.

    The conservative direction: a chunk can only be judged relevant on the page
    it begins on. If a variant's gain survives this, the gain came from the text
    inside the chunk, not from a wider page span.
    """
    out = []
    for h in hits:
        h = dict(h)
        h["page_end"] = h["page_start"]
        out.append(h)
    return out


def page_span_profile(index: dict[str, Any]) -> dict[str, Any]:
    spans = [int(c["page_end"]) - int(c["page_start"]) + 1 for c in index["chunks"]]
    n = len(spans) or 1
    return {
        "mean_pages_per_chunk": round(sum(spans) / n, 3),
        "median_pages_per_chunk": sorted(spans)[n // 2] if spans else 0,
        "max_pages_per_chunk": max(spans) if spans else 0,
        "pct_multi_page": round(100 * sum(1 for s in spans if s > 1) / n, 1),
    }


def chunk_profile(index: dict[str, Any]) -> dict[str, Any]:
    all_chunks = index["all_chunks"]
    indexed = index["chunks"]
    tk = sorted(int(c.get("token_count") or cc.count_tokens(c["chunk_text"])) for c in indexed)
    n = len(tk) or 1
    limit = cc.model_token_limit()
    return {
        "total_chunks": len(all_chunks),
        "indexed_chunks": len(indexed),
        "content_types": dict(Counter(str(c.get("content_type")) for c in all_chunks)),
        "tokens": {
            "min": tk[0] if tk else 0,
            "median": tk[n // 2] if tk else 0,
            "mean": round(sum(tk) / n, 1) if tk else 0,
            "p90": tk[int(0.9 * (n - 1))] if tk else 0,
            "max": tk[-1] if tk else 0,
            "limit": limit,
            "over_limit": sum(1 for t in tk if t > limit),
        },
        "single_page_chunks": sum(1 for c in indexed if c["page_start"] == c["page_end"]),
        "with_recommendation_id": sum(1 for c in indexed if c.get("recommendation_id")),
        "anchor_kinds": dict(Counter(str(c.get("anchor_kind")) for c in indexed)),
    }


def score(index, model, gold, queries: dict[int, str], collapse_pages: bool = False) -> dict[str, Any]:
    runs = {qid: retrieve(q, index, model, top_k=EVAL_DEPTH) for qid, q in queries.items()}
    if collapse_pages:
        runs = {qid: narrow(hits) for qid, hits in runs.items()}
    res = evaluate_run(runs, gold)
    return {
        "metrics": res["metrics"],
        "per_query": [
            {
                "query_id": q["query_id"],
                "first_relevant_rank": q["first_relevant_rank"],
                "relevant_top1": q["relevant_top1"],
                "p_at_1": q["p_at_1"],
                "p_at_5": q["p_at_5"],
                "recall_at_5": q["recall_at_5"],
                "recall_at_10": q["recall_at_10"],
                "top1_chunk": q["top1"]["chunk_id"],
                "top1_doc": q["top1"]["document_id"],
                "top1_page": q["top1"]["page"],
            }
            for q in res["per_query"]
        ],
    }


METRIC_KEYS = ("P@1", "P@3", "P@5", "MRR", "Recall@5", "Recall@10", "Relevant_Top1", "Answering@5")


def main() -> int:
    import clinical_rag as cr

    loaded = cc.load_processed(ROOT)
    pages_df, recs_df = loaded["pages_df"], loaded["recommendations_df"]

    gold_orig = load_gold()
    gold_held = load_gold(HELDOUT_GOLD)
    q_orig = {i: q for i, q in enumerate(cr.CLINICAL_QUERIES, start=1)}
    q_held = {s["query_id"]: s["query"] for s in gold_held["queries"]}

    print("loading MedEmbed at pinned revision ...")
    model = cc.load_embedder(cc.DEFAULT_EMBED_MODEL)

    variants: dict[str, dict[str, Any]] = {}

    print("\n[control] production index on disk (unmodified)")
    variants["control_production"] = load_production_index()

    print("[V1] atomic_pagesafe  - anchors + page breaks outside recommendations")
    v1 = build_atomic_chunks(pages_df, recs_df, keep_page_breaks=True, rec_token_budget=cc.max_content_tokens())
    variants["V1_atomic_pagesafe"] = index_chunks(v1, model)

    print("[V2] atomic_pure      - anchors only (closest to Project B)")
    v2 = build_atomic_chunks(pages_df, recs_df, keep_page_breaks=False, rec_token_budget=cc.max_content_tokens())
    variants["V2_atomic_pure"] = index_chunks(v2, model)

    # Size-matched control: same mean token count as V1, no structural anchors.
    v1_mean = chunk_profile(variants["V1_atomic_pagesafe"])["tokens"]["mean"]
    ctrl_tokens = int(round(min(v1_mean, cc.max_content_tokens())))
    scale = max(1.0, ctrl_tokens / cc.TARGET_TOKENS)
    print(f"[V3] size_control     - baseline algorithm at ~{ctrl_tokens} tokens/chunk (matches V1)")
    v3 = build_size_control_chunks(
        pages_df, recs_df,
        target_tokens=ctrl_tokens,
        target_chars=int(cc.TARGET_CHARS * scale),
        max_chars=int(cc.MAX_CHARS * scale),
    )
    variants["V3_size_control"] = index_chunks(v3, model)

    # Page-span control: production retrieval, untouched ranking, page ranges
    # widened to V2's mean span. Isolates the frozen rule's page-overlap term.
    max_page = pages_df.groupby("document_id")["page_number"].max().astype(int).to_dict()
    v2_span = page_span_profile(variants["V2_atomic_pure"])["mean_pages_per_chunk"]
    half = max(1, int(round((v2_span - 1) / 2)))
    print(f"[V4] pagespan_control - production ranking, page spans widened +/-{half} (V2 mean {v2_span})")
    variants["V4_pagespan_control"] = widen_page_ranges(variants["control_production"], half, max_page)

    out: dict[str, Any] = {
        "experiment": "12 — Project-B-inspired atomic / anchor-driven chunking",
        "transferred_from_project_b": [
            "document-level marked text with page sentinels (provenance by offset)",
            "structural anchors: 'Recommendation N', numbered recommendation IDs, numbered section headings",
            "atomic recommendation spans -- a recommendation is one chunk",
        ],
        "deliberately_not_transferred": [
            "intent detection / query expansion / keyword-overlap bonus",
            "recommendation-ID score boosts and anchor injection into the candidate pool",
            "unbounded chunk sizes (Project B truncates 31.6% of its indexed chunks at encode time)",
            "hardcoded per-document header lists (USPSTF_HEADERS / NICE_HEADERS)",
        ],
        "retrieval": "pure dense cosine, MedEmbed-base-v0.1 @ pinned revision; no reranking, no query rules",
        "embedding_model": cc.DEFAULT_EMBED_MODEL,
        "embedding_revision": cc.model_revision(cc.DEFAULT_EMBED_MODEL),
        "gold_original_sha256": gold_orig["_sha256"],
        "gold_heldout_sha256": gold_held["_sha256"],
        "size_control_target_tokens": ctrl_tokens,
        "variants": {},
    }

    for name, index in variants.items():
        print(f"\nscoring {name} ...")
        out["variants"][name] = {
            "chunk_profile": chunk_profile(index),
            "page_span_profile": page_span_profile(index),
            "original_10": score(index, model, gold_orig, q_orig),
            "heldout_18": score(index, model, gold_held, q_held),
            "original_10_startpage_only": score(index, model, gold_orig, q_orig, collapse_pages=True),
            "heldout_18_startpage_only": score(index, model, gold_held, q_held, collapse_pages=True),
        }

    (EVAL_DIR / "runs").mkdir(exist_ok=True)
    path = EVAL_DIR / "runs" / "exp12_atomic_chunking.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # ---- console report
    print("\n" + "=" * 100)
    print("EXPERIMENT 12 - atomic / anchor-driven chunking (Project B mechanism, no Project B query rules)")
    print("=" * 100)

    print(f"\n{'variant':<24}{'chunks':>8}{'indexed':>9}{'med tok':>9}{'max tok':>9}{'over':>6}"
          f"{'1-page':>8}{'rec_id':>8}{'pages/chunk':>13}{'%multipage':>12}")
    for name, v in out["variants"].items():
        p, s, t = v["chunk_profile"], v["page_span_profile"], v["chunk_profile"]["tokens"]
        print(f"{name:<24}{p['total_chunks']:>8}{p['indexed_chunks']:>9}{t['median']:>9}"
              f"{t['max']:>9}{t['over_limit']:>6}{p['single_page_chunks']:>8}"
              f"{p['with_recommendation_id']:>8}{s['mean_pages_per_chunk']:>13}{s['pct_multi_page']:>12}")

    for setname, key in (
        ("ORIGINAL 10 (frozen gold)", "original_10"),
        ("HELD-OUT 18 (frozen gold)", "heldout_18"),
        ("ORIGINAL 10 - start page only (conservative)", "original_10_startpage_only"),
        ("HELD-OUT 18 - start page only (conservative)", "heldout_18_startpage_only"),
    ):
        print(f"\n--- {setname} ---")
        print(f"{'variant':<24}" + "".join(f"{k:>13}" for k in METRIC_KEYS))
        base = out["variants"]["control_production"][key]["metrics"]
        for name, v in out["variants"].items():
            m = v[key]["metrics"]
            print(f"{name:<24}" + "".join(f"{m[k]:>13}" for k in METRIC_KEYS))
        print()
        for name, v in out["variants"].items():
            if name == "control_production":
                continue
            m = v[key]["metrics"]
            deltas = "".join(f"{m[k] - base[k]:>+13.4f}" for k in METRIC_KEYS)
            print(f"{'delta ' + name:<24}{deltas}")

    print(f"\nsaved -> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

import ingestion.atomic_chunking as cac  # noqa: E402
import ingestion.chunking as cc  # noqa: E402
from evaluate import evaluate_run, load_gold  # noqa: E402

HELDOUT_GOLD = EVAL_DIR / "gold_standard_heldout.json"
EVAL_DEPTH = 10

# ---------------------------------------------------------------------------
# Anchors and chunk assembly
#
# There is ONE implementation of the atomic chunker, in
# `ingestion/atomic_chunking.py` -- the module that produces the shipped
# artifacts. This file used to carry a second, independent copy of the anchor
# patterns, the segmenter and the chunk-assembly loop. The two drifted: the copy
# here produced 1,764 chunks / 1,004 indexed while the shipped module produced
# 1,760 / 991, so the published original10 and heldout18 rows described a chunk
# set that was never shipped (see docs/REFERENCE_COMPARISON.md).
#
# The variants below are now expressed as PARAMETERS on that single
# implementation, not as a forked body. Structure-driven only: no query string,
# gold label, recommendation number or topic keyword appears anywhere in this file.
# ---------------------------------------------------------------------------

REC_ANCHOR, SECTION_ANCHOR, PAGE_ANCHOR = cac.REC_ANCHOR, cac.SECTION_ANCHOR, cac.PAGE_ANCHOR

# Provenance helpers are re-exported so callers keep working unchanged.
build_marked_text = cac.build_marked_text
page_at_offset = cac.page_at_offset
strip_markers = cac.strip_markers

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
#
# This flag is read at CALL time by the wrappers below, so assigning to
# `ex.REJECT_CITATION_HEADINGS` at runtime still takes effect
# (eval/run_corrected_validation.py relies on that).
REJECT_CITATION_HEADINGS = True


def find_anchors(marked: str) -> list[tuple[int, str, str]]:
    """Anchors from the shipped implementation, honouring the module flag."""
    return cac.find_anchors(marked, reject_citation_headings=REJECT_CITATION_HEADINGS)


def segment(marked: str, keep_page_breaks: bool) -> list[dict[str, Any]]:
    """Spans from the shipped implementation, honouring the module flag."""
    return cac.segment(marked, keep_page_breaks=keep_page_breaks,
                       reject_citation_headings=REJECT_CITATION_HEADINGS)


def build_atomic_chunks(
    pages_df: pd.DataFrame,
    recs_df: pd.DataFrame,
    keep_page_breaks: bool,
    rec_token_budget: int,
    narrative_token_budget: int = cc.TARGET_TOKENS,
) -> list[dict[str, Any]]:
    """V1 / V2 atomic chunks, built by the SHIPPED chunker.

    V1 is `keep_page_breaks=True` (the shipped default); V2 is False. Nothing is
    reimplemented here -- this is a parameterised call into
    `ingestion.atomic_chunking.build_chunks`, so the evaluated chunk set and the
    shipped chunk set cannot diverge again.
    """
    return cac.build_chunks(
        pages_df,
        recs_df,
        rec_token_budget=rec_token_budget,
        narrative_token_budget=narrative_token_budget,
        keep_page_breaks=keep_page_breaks,
        reject_citation_headings=REJECT_CITATION_HEADINGS,
    )


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
    import retrieval.index as cr

    d = ROOT / "data" / "embeddings"
    meta = json.loads((d / "index_meta.json").read_text(encoding="utf-8"))
    chunks = json.loads((d / "embedded_chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(d / "embeddings.npy")
    # The same binding check `retrieval.index.load_index` applies: element-wise
    # chunk_id equality plus the digests stamped at build time. This path is the
    # one the evaluations retrieve through, so it must not be the weaker one.
    cr.verify_index_binding(ROOT, meta, chunks, vectors)
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
    import retrieval.index as cr

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

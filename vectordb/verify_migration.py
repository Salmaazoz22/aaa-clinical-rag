# -*- coding: utf-8 -*-
"""Local index vs Qdrant: infrastructure equivalence verification.

    python vectordb/verify_migration.py

This is NOT a retrieval experiment and it produces no retrieval metric. It asks
one question: does the production vector store return the same evidence, in the
same order, with the same scores, as the numpy index the frozen evaluation was
run against?

Method
------
For each of the 48 frozen questions (original10 + heldout18 + final20) the query
is embedded **once**, with the pinned MedEmbed model, and that single vector is
sent down both paths:

    vector -> numpy exhaustive cosine (data/embeddings/embeddings.npy) -> top 10
    vector -> Qdrant exact cosine search (production collection)       -> top 10

Embedding the query once is deliberate: it removes the encoder as a variable so
any difference observed is attributable to storage, distance computation, ID
mapping or index configuration -- the things being migrated.

The question sets are used here purely as a fixed, non-cherry-picked query
sample. Nothing is scored against their gold standards, nothing is tuned, and
none of those files is read for anything but the question text.

Output: eval/qdrant_migration_verification.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vectordb.config import load_settings  # noqa: E402
from vectordb.retriever import QdrantRetriever  # noqa: E402
from vectordb.schema import DEFAULT_TOP_K, EXPECTED_MODEL, is_nan, nan_summary  # noqa: E402

OUT_PATH = ROOT / "eval" / "qdrant_migration_verification.json"

# Absolute tolerance on the cosine score. Both sides compute a dot product of
# L2-normalised float32 vectors; the only expected difference is the order of
# floating-point accumulation and Qdrant's own re-normalisation on upsert.
SCORE_TOLERANCE = 1e-5

DATASETS = {
    "original10": ROOT / "eval" / "gold_standard.json",
    "heldout18": ROOT / "eval" / "gold_standard_heldout.json",
    "final20": ROOT / "eval" / "gold_standard_final20.json",
}


def load_queries() -> list[dict[str, Any]]:
    """Question text only, from the frozen sets. Nothing else is read."""
    import retrieval.index as cr

    out: list[dict[str, Any]] = []
    for name, path in DATASETS.items():
        if name == "original10":
            # The original 10 live in code, exactly as
            # `eval/scripts/run_final_evaluation.py` resolves them.
            for qid, query in enumerate(cr.CLINICAL_QUERIES, start=1):
                out.append({"dataset": name, "query_id": qid, "query": query})
            continue
        gold = json.loads(path.read_text(encoding="utf-8"))
        for spec in gold["queries"]:
            out.append({"dataset": name, "query_id": spec["query_id"], "query": spec["query"]})
    return out


def local_top_k(vector: np.ndarray, index: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    """Replicates `retrieval.index.retrieve` scoring and ordering exactly."""
    scores = index["vectors"] @ vector
    order = np.argsort(-scores)[:top_k]
    hits = []
    for rank, idx in enumerate(order, start=1):
        chunk = index["chunks"][int(idx)]
        hits.append(
            {
                "rank": rank,
                "similarity_score": float(scores[int(idx)]),
                "chunk_id": chunk.get("chunk_id"),
                "document": chunk.get("document_name"),
                "document_id": chunk.get("document_id"),
                "section": chunk.get("section_title"),
                "page": chunk.get("page_number"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "recommendation_id": chunk.get("recommendation_id"),
            }
        )
    return hits


def _slim(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": hit["rank"],
        "chunk_id": hit["chunk_id"],
        "score": round(float(hit["similarity_score"]), 8),
        "document": hit.get("document"),
        "document_id": hit.get("document_id"),
        "page_start": hit.get("page_start"),
        "page_end": hit.get("page_end"),
        "section": hit.get("section"),
        "recommendation_id": hit.get("recommendation_id"),
    }


def compare(local: list[dict[str, Any]], remote: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    local_ids = [h["chunk_id"] for h in local]
    remote_ids = [h["chunk_id"] for h in remote]
    local_scores = {h["chunk_id"]: float(h["similarity_score"]) for h in local}
    remote_scores = {h["chunk_id"]: float(h["similarity_score"]) for h in remote}

    shared = [c for c in local_ids if c in remote_scores]
    score_diffs = {c: abs(local_scores[c] - remote_scores[c]) for c in shared}
    max_score_diff = max(score_diffs.values()) if score_diffs else 0.0

    rank_differences = [
        {
            "rank": i + 1,
            "local_chunk_id": local_ids[i],
            "qdrant_chunk_id": remote_ids[i] if i < len(remote_ids) else None,
            "local_score": round(local_scores[local_ids[i]], 8),
            "qdrant_score": round(remote_scores.get(remote_ids[i], float("nan")), 8)
            if i < len(remote_ids)
            else None,
            "score_gap_local": round(
                abs(local_scores[local_ids[i]] - local_scores.get(remote_ids[i], local_scores[local_ids[i]])), 8
            )
            if i < len(remote_ids) and remote_ids[i] in local_scores
            else None,
        }
        for i in range(len(local_ids))
        if i >= len(remote_ids) or local_ids[i] != remote_ids[i]
    ]

    # A rank swap between two chunks whose LOCAL scores are equal within
    # tolerance is a tie, not a disagreement: either order is a correct
    # descending sort. Anything else is a real ordering difference.
    ties_only = all(
        d["score_gap_local"] is not None and d["score_gap_local"] <= tolerance for d in rank_differences
    )

    metadata_mismatches = []
    nan_normalised = []
    remote_by_id = {h["chunk_id"]: h for h in remote}
    for hit in local:
        other = remote_by_id.get(hit["chunk_id"])
        if other is None:
            continue
        for field in ("document", "document_id", "page_start", "page_end", "section", "recommendation_id"):
            mine, theirs = hit.get(field), other.get(field)
            if mine == theirs:
                continue
            # The local artifact stores some optional fields as the JSON literal
            # NaN; Qdrant stores the same absence as null. Same meaning, counted
            # rather than hidden. See vectordb/schema.py.
            if is_nan(mine) and theirs is None:
                nan_normalised.append({"chunk_id": hit["chunk_id"], "field": field})
                continue
            metadata_mismatches.append(
                {"chunk_id": hit["chunk_id"], "field": field, "local": mine, "qdrant": theirs}
            )

    same_top1 = bool(local_ids) and bool(remote_ids) and local_ids[0] == remote_ids[0]
    same_top10_set = set(local_ids) == set(remote_ids)
    same_order = local_ids == remote_ids
    scores_within_tolerance = max_score_diff <= tolerance and len(shared) == len(local_ids)

    passed = (
        same_top1
        and same_top10_set
        and scores_within_tolerance
        and not metadata_mismatches
        and (same_order or ties_only)
    )
    return {
        "local_top10": [_slim(h) for h in local],
        "qdrant_top10": [_slim(h) for h in remote],
        "same_top1": same_top1,
        "same_top10": same_top10_set,
        "same_order": same_order,
        "rank_differences": rank_differences,
        "rank_differences_are_score_ties": ties_only if rank_differences else None,
        "score_differences": {
            "max_abs": round(max_score_diff, 10),
            "mean_abs": round(sum(score_diffs.values()) / len(score_diffs), 10) if score_diffs else 0.0,
            "per_chunk": {c: round(d, 10) for c, d in score_diffs.items()},
            "within_tolerance": scores_within_tolerance,
        },
        "metadata_mismatches": metadata_mismatches,
        "nan_normalised_to_null": nan_normalised,
        "pass": passed,
    }


def run(top_k: int = DEFAULT_TOP_K, tolerance: float = SCORE_TOLERANCE) -> dict[str, Any]:
    import retrieval.index as cr
    from ingestion.chunking import load_embedder

    settings = load_settings()
    index = cr.load_index(ROOT)
    model = load_embedder(EXPECTED_MODEL)
    retriever = QdrantRetriever(settings=settings, model=model)

    queries = load_queries()
    per_query: list[dict[str, Any]] = []
    local_latency: list[float] = []
    qdrant_latency: list[float] = []

    for item in queries:
        vector = model.encode([item["query"]], normalize_embeddings=True, convert_to_numpy=True)[0]

        t0 = time.perf_counter()
        local_hits = local_top_k(vector, index, top_k)
        local_latency.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        remote_hits = retriever.search_vector([float(x) for x in vector], top_k=top_k)
        qdrant_latency.append((time.perf_counter() - t0) * 1000)

        result = compare(local_hits, remote_hits, tolerance)
        per_query.append({**item, **result})

    n_pass = sum(1 for q in per_query if q["pass"])
    max_diff = max((q["score_differences"]["max_abs"] for q in per_query), default=0.0)

    def pct(values: list[float], p: float) -> float:
        return round(float(np.percentile(values, p)), 3) if values else 0.0

    return {
        "artifact": "qdrant_migration_verification",
        "purpose": (
            "Infrastructure equivalence between the local numpy index and the production "
            "Qdrant collection. Not a retrieval evaluation: no metric is computed and no "
            "gold standard is scored."
        ),
        "does_not_modify": [
            "eval/gold_standard.json",
            "eval/gold_standard_heldout.json",
            "eval/gold_standard_final20.json",
            "eval/final_evidence.json",
            "eval/final_evaluation_results.json",
            "eval/experiment_history.json",
            "eval/runs/*",
        ],
        "configuration": {
            "embedding_model": EXPECTED_MODEL,
            "embedding_revision": index["meta"].get("model_revision"),
            "embedding_dim": index["meta"].get("embedding_dim"),
            "query_embedded_once_for_both_paths": True,
            "local_index": index["meta"].get("index_type"),
            "local_vectors": int(len(index["vectors"])),
            "qdrant": settings.describe(),
            "distance": "cosine",
            "top_k": top_k,
            "score_tolerance_abs": tolerance,
        },
        "known_representation_differences": {
            "nan_to_null": (
                "data/embeddings/embedded_chunks.json stores some optional fields "
                "(recommendation_id, recommendation_grade, evidence_level) as the JSON literal "
                "NaN; Qdrant stores that absence as null. Occurrences are counted per query as "
                "`nan_normalised_to_null` and are not treated as retrieval differences. NaN in a "
                "required field or in any vector fails ingestion."
            ),
            "nan_fields_in_local_index": nan_summary(index["chunks"]),
        },
        "summary": {
            "n_queries": len(per_query),
            "n_pass": n_pass,
            "n_fail": len(per_query) - n_pass,
            "all_same_top1": all(q["same_top1"] for q in per_query),
            "all_same_top10": all(q["same_top10"] for q in per_query),
            "all_same_order": all(q["same_order"] for q in per_query),
            "max_abs_score_difference": max_diff,
            "metadata_mismatches": sum(len(q["metadata_mismatches"]) for q in per_query),
            "nan_normalised_to_null": sum(len(q["nan_normalised_to_null"]) for q in per_query),
            "verdict": "EQUIVALENT" if n_pass == len(per_query) else "DISCREPANCY",
        },
        "latency_ms": {
            "local_search_mean": round(float(np.mean(local_latency)), 3) if local_latency else None,
            "local_search_p95": pct(local_latency, 95),
            "qdrant_search_mean": round(float(np.mean(qdrant_latency)), 3) if qdrant_latency else None,
            "qdrant_search_p95": pct(qdrant_latency, 95),
            "note": "search only; query embedding excluded because it is shared by both paths",
        },
        "per_query": per_query,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify Qdrant reproduces local retrieval.")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--tolerance", type=float, default=SCORE_TOLERANCE)
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    report = run(top_k=args.top_k, tolerance=args.tolerance)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    s = report["summary"]
    print(f"queries        : {s['n_queries']}")
    print(f"pass / fail    : {s['n_pass']} / {s['n_fail']}")
    print(f"same top-1     : {s['all_same_top1']}")
    print(f"same top-10 set: {s['all_same_top10']}")
    print(f"same order     : {s['all_same_order']}")
    print(f"max |score diff|: {s['max_abs_score_difference']:.3e}  (tolerance {args.tolerance:g})")
    print(f"metadata mismatches: {s['metadata_mismatches']}")
    print(f"NaN->null fields (documented, not a mismatch): {s['nan_normalised_to_null']}")
    print(f"verdict        : {s['verdict']}")
    print(f"saved -> {out.relative_to(ROOT)}")
    if s["verdict"] != "EQUIVALENT":
        print("\nDISCREPANCY -- investigate normalisation, distance, precision, ID mapping, "
              "filtering, metadata, query embedding and index configuration. Do NOT tune to match.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

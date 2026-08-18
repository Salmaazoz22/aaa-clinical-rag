# -*- coding: utf-8 -*-
"""Frozen retrieval evaluation for the AAA clinical RAG project.

Scores ANY ranked retrieval result against `eval/gold_standard.json`.

The evaluator is deliberately dumb: it knows nothing about the retriever, the
embedding model, or chunk identity. It takes a ranked list of chunks (each with
`document_id`, `page_start`, `page_end`, `chunk_text`) and applies the frozen
relevance rule. That keeps every experiment scored on identical terms.

Usage
-----
    python eval/evaluate.py                      # dense baseline, top-10
    python eval/evaluate.py --label exp1         # tag the run
    python eval/evaluate.py --rerank             # dense -> cross-encoder rerank
    python eval/evaluate.py --out eval/runs/x.json

Importable:
    from evaluate import load_gold, evaluate_run, is_relevant
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPTS_DIR.parent
ROOT = EVAL_DIR.parent
GOLD_PATH = EVAL_DIR / "gold_standard.json"

# ---------------------------------------------------------------------------
# Text normalisation (frozen; documented in gold_standard.json)
# ---------------------------------------------------------------------------

_LIGATURES = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": " ", "—": " ", "−": " ",
    "≥": " ", "≤": " ", "‡": " ", "†": " ",
    " ": " ",
}


def normalise(text: str) -> str:
    """Frozen normalisation applied before every regex test."""
    if not text:
        return ""
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    return re.sub(r"\s+", " ", text).strip().lower()


# ---------------------------------------------------------------------------
# Gold standard
# ---------------------------------------------------------------------------

def gold_sha256(path: Path = GOLD_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gold(path: Path = GOLD_PATH) -> dict[str, Any]:
    gold = json.loads(path.read_text(encoding="utf-8"))
    gold["_sha256"] = gold_sha256(path)
    return gold


def _facts_satisfied(norm_text: str, required: dict[str, Any]) -> int:
    hits = 0
    for group in required["groups"]:
        if any(re.search(pat, norm_text, re.IGNORECASE) for pat in group["any_of"]):
            hits += 1
    return hits


def matched_passages(chunk: dict[str, Any], spec: dict[str, Any]) -> list[int]:
    """Indices of the pre-registered answer passages this chunk satisfies."""
    norm = normalise(chunk.get("chunk_text") or "")
    required = spec["required_facts"]
    if _facts_satisfied(norm, required) < required["min_groups"]:
        return []

    doc = chunk.get("document_id")
    try:
        c_start = int(chunk.get("page_start"))
        c_end = int(chunk.get("page_end"))
    except (TypeError, ValueError):
        return []

    out = []
    for i, p in enumerate(spec["answer_passages"]):
        if p["document_id"] != doc:
            continue
        if c_start <= int(p["page_end"]) and c_end >= int(p["page_start"]):
            out.append(i)
    return out


def is_relevant(chunk: dict[str, Any], spec: dict[str, Any]) -> bool:
    return bool(matched_passages(chunk, spec))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def score_query(hits: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    """`hits` is the ranked list (rank 1 first), at least 10 long where possible."""
    rel_flags = [is_relevant(h, spec) for h in hits]
    passage_hits = [matched_passages(h, spec) for h in hits]
    n_passages = len(spec["answer_passages"])

    def p_at(k: int) -> float:
        return sum(rel_flags[:k]) / k

    def recall_at(k: int) -> float:
        covered = set()
        for ph in passage_hits[:k]:
            covered.update(ph)
        return len(covered) / n_passages if n_passages else 0.0

    first_rel = next((i + 1 for i, r in enumerate(rel_flags[:10]) if r), None)

    return {
        "query_id": spec["query_id"],
        "query": spec["query"],
        "p_at_1": p_at(1),
        "p_at_3": p_at(3),
        "p_at_5": p_at(5),
        "rr": (1.0 / first_rel) if first_rel else 0.0,
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "first_relevant_rank": first_rel,
        "relevant_top1": bool(rel_flags[:1] and rel_flags[0]),
        "answering_at_5": any(rel_flags[:5]),
        "n_answer_passages": n_passages,
        "top1": {
            "chunk_id": hits[0].get("chunk_id") if hits else None,
            "document_id": hits[0].get("document_id") if hits else None,
            "page": hits[0].get("page") or (hits[0].get("page_start") if hits else None),
            "section": hits[0].get("section") or (hits[0].get("section_title") if hits else None),
            "score": hits[0].get("similarity_score") if hits else None,
        },
        "relevant_ranks": [i + 1 for i, r in enumerate(rel_flags) if r],
    }


def aggregate(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(per_query)
    mean = lambda key: sum(q[key] for q in per_query) / n
    return {
        "n_queries": n,
        "P@1": round(mean("p_at_1"), 4),
        "P@3": round(mean("p_at_3"), 4),
        "P@5": round(mean("p_at_5"), 4),
        "MRR": round(mean("rr"), 4),
        "Recall@5": round(mean("recall_at_5"), 4),
        "Recall@10": round(mean("recall_at_10"), 4),
        "Relevant_Top1": sum(1 for q in per_query if q["relevant_top1"]),
        "Answering@5": sum(1 for q in per_query if q["answering_at_5"]),
    }


def evaluate_run(runs: dict[int, list[dict[str, Any]]], gold: dict[str, Any]) -> dict[str, Any]:
    """`runs` maps query_id -> ranked hit list."""
    per_query = [score_query(runs[spec["query_id"]], spec) for spec in gold["queries"]]
    return {
        "gold_standard_id": gold["gold_standard_id"],
        "gold_sha256": gold["_sha256"],
        "metrics": aggregate(per_query),
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Retrieval driver
# ---------------------------------------------------------------------------

def run_retrieval(top_k: int = 10, rerank: bool = False, candidates: int = 30) -> dict[str, Any]:
    import ingestion.chunking as cc
    import retrieval.index as cr

    index = cr.load_index(ROOT)
    # `cc.load_embedder` applies the revision pin. Constructing SentenceTransformer
    # directly would resolve `main` at download time, so a hub update could move
    # results under a frozen gold standard.
    model = cc.load_embedder(index["model_name"])

    reranker = None
    if rerank:
        from retrieval.rerank import load_reranker, rerank_hits  # noqa: E402
        reranker = load_reranker()

    runs, records = {}, {}
    for qid, query in enumerate(cr.CLINICAL_QUERIES, start=1):
        if rerank:
            hits = cr.retrieve(query, index, model=model, top_k=candidates)
            hits = rerank_hits(query, hits, reranker, top_k=top_k)
        else:
            hits = cr.retrieve(query, index, model=model, top_k=top_k)
        runs[qid] = hits
        records[qid] = [
            {
                "rank": h["rank"],
                "chunk_id": h["chunk_id"],
                "score": round(float(h["similarity_score"]), 6),
                "document_id": h["document_id"],
                "page": h["page"],
                "page_start": h["page_start"],
                "page_end": h["page_end"],
                "section_title": h["section"],
                "chunk_text": h["chunk_text"],
            }
            for h in hits
        ]
    return {"runs": runs, "records": records, "index_meta": index["meta"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="baseline", help="name for this run")
    ap.add_argument("--rerank", action="store_true", help="dense -> cross-encoder rerank")
    ap.add_argument("--candidates", type=int, default=30, help="dense candidate pool when reranking")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gold = load_gold()
    ret = run_retrieval(top_k=args.top_k, rerank=args.rerank, candidates=args.candidates)
    result = evaluate_run(ret["runs"], gold)
    result["label"] = args.label
    result["retrieval"] = {
        "mode": "dense+crossencoder" if args.rerank else "dense",
        "top_k": args.top_k,
        "candidates": args.candidates if args.rerank else None,
        "index_meta": ret["index_meta"],
    }
    result["top10"] = ret["records"]

    m = result["metrics"]
    print(f"\n=== {args.label} ({result['retrieval']['mode']}) ===")
    print(f"gold sha256: {result['gold_sha256'][:16]}...")
    for k in ("P@1", "P@3", "P@5", "MRR", "Recall@5", "Recall@10", "Relevant_Top1", "Answering@5"):
        print(f"{k:>14}: {m[k]}")
    print(f"\n{'Q':>2} {'first_rel':>9} {'P@1':>5} {'P@5':>5} {'R@5':>5} {'R@10':>5}  top1")
    for q in result["per_query"]:
        fr = q["first_relevant_rank"] or "-"
        print(f"{q['query_id']:>2} {str(fr):>9} {q['p_at_1']:>5.2f} {q['p_at_5']:>5.2f} "
              f"{q['recall_at_5']:>5.2f} {q['recall_at_10']:>5.2f}  "
              f"{q['top1']['document_id']} p{q['top1']['page']} :: {str(q['top1']['section'])[:38]}")

    out = Path(args.out) if args.out else EVAL_DIR / "runs" / f"{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nsaved -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

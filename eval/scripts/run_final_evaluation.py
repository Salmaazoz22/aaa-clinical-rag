# -*- coding: utf-8 -*-
"""Final evaluation: production baseline vs the Experiment 12 chunking variants,
scored on all three frozen gold standards, with full per-query evidence.

Pre-registration
----------------
`eval/gold_standard_final20.json` was authored, validated against source page
text, and frozen (SHA-256 recorded in eval/gold_standard_final20.sha256) BEFORE
this script was ever run. It is the only set that no configuration had been
evaluated against at the time the chunking decision was being made: the original
10 and the held-out 18 were both used to score V1/V2/V3/V4 in Experiment 12, so
neither is untouched with respect to that choice.

Nothing here tunes anything. There is no threshold, no reranking, no query
rewriting and no per-question logic. Retrieval is dense cosine over pinned
MedEmbed for every configuration; the only variable is the chunker.

Outputs
-------
eval/final_evaluation_results.json  metrics for every (config, dataset) pair
eval/final_evidence.json            per-query retrieved evidence, all 3 datasets
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPTS_DIR.parent
ROOT = EVAL_DIR.parent
# ROOT for the installed packages, SCRIPTS_DIR for sibling eval modules.
for _extra in (str(ROOT), str(SCRIPTS_DIR)):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

import numpy as np  # noqa: E402

import ingestion.chunking as cc  # noqa: E402
import experimental_atomic_chunking as ex  # noqa: E402
from evaluate import evaluate_run, load_gold, matched_passages, normalise  # noqa: E402

EVAL_DEPTH = 10
DATASETS = {
    "original10": EVAL_DIR / "gold_standard.json",
    "heldout18": EVAL_DIR / "gold_standard_heldout.json",
    "final20": EVAL_DIR / "gold_standard_final20.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def queries_for(gold: dict[str, Any], name: str) -> dict[int, str]:
    if name == "original10":
        import retrieval.index as cr
        return {i: q for i, q in enumerate(cr.CLINICAL_QUERIES, start=1)}
    return {s["query_id"]: s["query"] for s in gold["queries"]}


def evidence_rows(qid, query, spec, hits, config, dataset):
    rows = []
    groups = spec["required_facts"]["groups"]
    for h in hits:
        norm = normalise(h.get("chunk_text") or "")
        covered = [g["name"] for g in groups
                   if any(__import__("re").search(p, norm, __import__("re").I) for p in g["any_of"])]
        matched = matched_passages(h, spec)
        rows.append({
            "dataset": dataset,
            "config": config,
            "query_id": qid,
            "question": query,
            "rank": h["rank"],
            "similarity": round(float(h["similarity_score"]), 6),
            "chunk_id": h["chunk_id"],
            "document_id": h["document_id"],
            "page_start": h["page_start"],
            "page_end": h["page_end"],
            "section_title": h.get("section"),
            "relevant": bool(matched),
            "matched_answer_passages": [
                {
                    "document_id": spec["answer_passages"][i]["document_id"],
                    "page_start": spec["answer_passages"][i]["page_start"],
                    "page_end": spec["answer_passages"][i]["page_end"],
                    "section_ref": spec["answer_passages"][i]["section_ref"],
                }
                for i in matched
            ],
            "required_facts_covered": covered,
            "required_facts_needed": spec["required_facts"]["min_groups"],
            "chunk_text": h.get("chunk_text"),
        })
    return rows


def main() -> int:
    print("loading MedEmbed at pinned revision ...")
    model = cc.load_embedder(cc.DEFAULT_EMBED_MODEL)
    loaded = cc.load_processed(ROOT)
    pages_df, recs_df = loaded["pages_df"], loaded["recommendations_df"]

    configs: dict[str, Any] = {}
    print("[baseline_production] index on disk (unmodified)")
    configs["baseline_production"] = ex.load_production_index()
    print("[V1_atomic_pagesafe] rebuilding + embedding ...")
    configs["V1_atomic_pagesafe"] = ex.index_chunks(
        ex.build_atomic_chunks(pages_df, recs_df, keep_page_breaks=True,
                               rec_token_budget=cc.max_content_tokens()), model)
    print("[V2_atomic_pure] rebuilding + embedding ...")
    configs["V2_atomic_pure"] = ex.index_chunks(
        ex.build_atomic_chunks(pages_df, recs_df, keep_page_breaks=False,
                               rec_token_budget=cc.max_content_tokens()), model)

    results: dict[str, Any] = {
        "generated_by": "eval/run_final_evaluation.py",
        "embedding_model": cc.DEFAULT_EMBED_MODEL,
        "embedding_revision": cc.model_revision(cc.DEFAULT_EMBED_MODEL),
        "retrieval": "dense cosine, top-10. No reranking, no query expansion, no per-question logic.",
        "eval_depth": EVAL_DEPTH,
        "gold_sha256": {k: sha256(v) for k, v in DATASETS.items()},
        "datasets_are_separate": "original10, heldout18 and final20 use different questions and "
                                 "different answer passages. They are never pooled or averaged.",
        "chunk_profiles": {n: {**ex.chunk_profile(ix), **ex.page_span_profile(ix)}
                           for n, ix in configs.items()},
        "metrics": {},
    }
    evidence: list[dict[str, Any]] = []

    for ds_name, ds_path in DATASETS.items():
        gold = load_gold(ds_path)
        specs = {s["query_id"]: s for s in gold["queries"]}
        qs = queries_for(gold, ds_name)
        for cfg_name, index in configs.items():
            runs = {qid: ex.retrieve(q, index, model, top_k=EVAL_DEPTH) for qid, q in qs.items()}
            res = evaluate_run(runs, gold)
            results["metrics"].setdefault(ds_name, {})[cfg_name] = {
                "metrics": res["metrics"],
                "per_query": [
                    {"query_id": q["query_id"],
                     "first_relevant_rank": q["first_relevant_rank"],
                     "relevant_top1": q["relevant_top1"],
                     "p_at_1": q["p_at_1"], "p_at_5": q["p_at_5"],
                     "recall_at_5": q["recall_at_5"], "recall_at_10": q["recall_at_10"],
                     "n_answer_passages": q["n_answer_passages"],
                     "top1_chunk": q["top1"]["chunk_id"], "top1_doc": q["top1"]["document_id"]}
                    for q in res["per_query"]
                ],
            }
            for qid, hits in runs.items():
                evidence.extend(evidence_rows(qid, qs[qid], specs[qid], hits, cfg_name, ds_name))
            print(f"  {ds_name:<12} {cfg_name:<22} "
                  f"P@1={res['metrics']['P@1']:<8} MRR={res['metrics']['MRR']}")

    (EVAL_DIR / "final_evaluation_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (EVAL_DIR / "final_evidence.json").write_text(
        json.dumps({
            "description": "Per-query retrieved evidence for every (dataset, configuration). "
                           "Sufficient to reconstruct every metric in final_evaluation_results.json.",
            "gold_sha256": results["gold_sha256"],
            "columns": ["dataset", "config", "query_id", "question", "rank", "similarity",
                        "chunk_id", "document_id", "page_start", "page_end", "section_title",
                        "relevant", "matched_answer_passages", "required_facts_covered",
                        "required_facts_needed", "chunk_text"],
            "rows": evidence,
        }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    keys = ("P@1", "P@3", "P@5", "MRR", "Recall@5", "Recall@10", "Relevant_Top1", "Answering@5")
    for ds_name in DATASETS:
        print(f"\n--- {ds_name} ---")
        print(f"{'config':<24}" + "".join(f"{k:>13}" for k in keys))
        for cfg_name in configs:
            m = results["metrics"][ds_name][cfg_name]["metrics"]
            print(f"{cfg_name:<24}" + "".join(f"{m[k]:>13}" for k in keys))
    print("\nsaved -> eval/final_evaluation_results.json, eval/final_evidence.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

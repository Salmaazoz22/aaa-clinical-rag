# -*- coding: utf-8 -*-
"""P1: recompute every frozen question set on the SHIPPED chunker.

Why this exists
---------------
`eval/run_final_evaluation.py` produced the README's V1 rows from
`experimental_atomic_chunking.build_atomic_chunks` -- a second, independent
implementation of the atomic chunker. Its chunk set (1,764 total / 1,004 indexed)
is NOT the shipped one (1,760 / 991), so 134 of the 318 chunk ids it retrieved do
not exist in `data/chunks/chunks.json`.

`eval/run_corrected_validation.py` later re-ran on a chunk set whose counts match
the shipped artifacts, but only for `final20`. `original10` and `heldout18` were
never recomputed, so their published V1 rows still describe the 1,764/1,004 set.

This script closes that gap. It scores all three frozen sets against the shipped
artifacts on disk (`data/chunks/chunks.json` + `data/embeddings/`), which were
verified to reproduce exactly from `ingestion.chunking.build_chunks_for("atomic")`:
1,760 chunks, identical chunk_id order, zero text differences.

What is held constant
---------------------
Everything except the chunk set. Retrieval is `experimental_atomic_chunking.retrieve`
-- the same function the published run used -- scoring is `evaluate.evaluate_run`,
and query text is resolved exactly as `run_final_evaluation.queries_for` resolves
it. The embedding model is loaded through `cc.load_embedder`, so the revision pin
applies. Nothing is tuned, and no artifact of the previous runs is overwritten.

`final20` is the control: it was already recomputed on the shipped-equivalent chunk
set, so it must reproduce `final_corrected_v1_final20.json` exactly. If it does not,
this harness is wrong and its `original10` / `heldout18` numbers cannot be trusted.

Output
------
eval/runs/p1_shipped_chunker_all_sets.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(EVAL_DIR))

import ingestion.chunking as cc  # noqa: E402
import experimental_atomic_chunking as ex  # noqa: E402
from evaluate import evaluate_run, load_gold  # noqa: E402

EVAL_DEPTH = 10
DATASETS = {
    "original10": EVAL_DIR / "gold_standard.json",
    "heldout18": EVAL_DIR / "gold_standard_heldout.json",
    "final20": EVAL_DIR / "gold_standard_final20.json",
}
METRIC_KEYS = ("P@1", "P@3", "P@5", "MRR", "Recall@5", "Recall@10",
               "Relevant_Top1", "Answering@5")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def queries_for(gold: dict[str, Any], name: str) -> dict[int, str]:
    """Identical to run_final_evaluation.queries_for, so query handling is unchanged."""
    if name == "original10":
        import retrieval.index as cr

        return {i: q for i, q in enumerate(cr.CLINICAL_QUERIES, start=1)}
    return {s["query_id"]: s["query"] for s in gold["queries"]}


def published_v1() -> dict[str, dict[str, Any]]:
    """The V1_atomic_pagesafe rows currently in the README, per dataset."""
    prev = json.loads((EVAL_DIR / "final_evaluation_results.json").read_text(encoding="utf-8"))
    return {ds: prev["metrics"][ds]["V1_atomic_pagesafe"]["metrics"] for ds in DATASETS}


def verify_shipped_chunker_reproduces() -> dict[str, Any]:
    """The shipped chunker must reproduce data/chunks/chunks.json exactly."""
    loaded = cc.load_processed(ROOT)
    fresh = cc.build_chunks_for(cc.DEFAULT_CHUNKER, loaded["pages_df"],
                                loaded["recommendations_df"])
    shipped = json.loads((ROOT / "data/chunks/chunks.json").read_text(encoding="utf-8"))["chunks"]
    fids = [c["chunk_id"] for c in fresh]
    sids = [c["chunk_id"] for c in shipped]
    ftext = {c["chunk_id"]: (c["chunk_text"] or "").strip() for c in fresh}
    stext = {c["chunk_id"]: (c["chunk_text"] or "").strip() for c in shipped}
    common = set(ftext) & set(stext)
    text_diffs = sum(1 for k in common if ftext[k] != stext[k])
    return {
        "strategy": cc.DEFAULT_CHUNKER,
        "fresh_chunks": len(fresh),
        "shipped_chunks": len(shipped),
        "chunk_id_lists_identical": fids == sids,
        "text_differences": text_diffs,
        "reproduces": fids == sids and text_diffs == 0 and len(fresh) == len(shipped),
    }


def main() -> int:
    print("verifying the shipped chunker reproduces data/chunks/chunks.json ...")
    repro = verify_shipped_chunker_reproduces()
    print(f"  strategy={repro['strategy']}  fresh={repro['fresh_chunks']} "
          f"shipped={repro['shipped_chunks']}  ids_identical={repro['chunk_id_lists_identical']} "
          f"text_diffs={repro['text_differences']}")
    if not repro["reproduces"]:
        print("  FAILED: shipped artifacts are stale relative to the shipped chunker.")
        print("  Refusing to score against them. Run eval/rebuild_shipped_index.py first.")
        return 1
    print("  OK -- the on-disk index is the shipped chunker's own output.\n")

    print("loading MedEmbed at pinned revision ...")
    model = cc.load_embedder(cc.DEFAULT_EMBED_MODEL)

    index = ex.load_production_index()
    n_vectors = len(index["vectors"])
    print(f"shipped index: {n_vectors} vectors over {len(index['all_chunks'])} produced chunks\n")

    prev = published_v1()
    out: dict[str, Any] = {
        "label": "p1_shipped_chunker_all_sets",
        "what_this_is": "All three frozen sets scored against the SHIPPED chunker's own "
                        "artifacts (data/chunks + data/embeddings). Closes the gap where "
                        "original10 and heldout18 were only ever scored on the experimental "
                        "chunker's 1,764/1,004 chunk set.",
        "overwrites_nothing": "final_evaluation_results.json, final_evidence.json and every "
                              "file in eval/runs/ from previous phases are left untouched.",
        "chunker_reproducibility_check": repro,
        "configuration": {
            "chunker": f"ingestion.atomic_chunking via ingestion.chunking.build_chunks_for"
                       f"('{cc.DEFAULT_CHUNKER}')",
            "chunk_source": "data/chunks/chunks.json (shipped)",
            "index_source": "data/embeddings/embeddings.npy + embedded_chunks.json (shipped)",
            "indexed_vectors": n_vectors,
            "embedding_model": cc.DEFAULT_EMBED_MODEL,
            "embedding_revision": cc.model_revision(cc.DEFAULT_EMBED_MODEL),
            "retrieval": "dense cosine, top-10. No reranking, no query expansion, "
                         "no per-question logic.",
            "eval_depth": EVAL_DEPTH,
        },
        "gold_sha256": {k: sha256(v) for k, v in DATASETS.items()},
        "datasets_are_separate": "original10, heldout18 and final20 use different questions "
                                 "and different answer passages. They are never pooled.",
        "comparison": {},
        "per_query": {},
    }

    for ds_name, ds_path in DATASETS.items():
        gold = load_gold(ds_path)
        qs = queries_for(gold, ds_name)
        runs = {qid: ex.retrieve(q, index, model, top_k=EVAL_DEPTH) for qid, q in qs.items()}
        res = evaluate_run(runs, gold)
        now, before = res["metrics"], prev[ds_name]

        out["comparison"][ds_name] = {
            "published_v1_experimental_chunker": before,
            "recomputed_v1_shipped_chunker": now,
            "delta": {k: round(now[k] - before[k], 4) for k in METRIC_KEYS},
            "changed": any(abs(now[k] - before[k]) > 1e-9 for k in METRIC_KEYS),
        }
        out["per_query"][ds_name] = [
            {"query_id": q["query_id"],
             "first_relevant_rank": q["first_relevant_rank"],
             "relevant_top1": q["relevant_top1"],
             "p_at_1": q["p_at_1"], "p_at_5": q["p_at_5"],
             "recall_at_5": q["recall_at_5"], "recall_at_10": q["recall_at_10"],
             "n_answer_passages": q["n_answer_passages"],
             "top1_chunk": q["top1"]["chunk_id"], "top1_doc": q["top1"]["document_id"]}
            for q in res["per_query"]
        ]

    # final20 is the control: it must match the corrected run that already used a
    # shipped-equivalent chunk set.
    corr = json.loads((EVAL_DIR / "runs/final_corrected_v1_final20.json").read_text(
        encoding="utf-8"))["metrics_corrected"]
    got = out["comparison"]["final20"]["recomputed_v1_shipped_chunker"]
    control_ok = all(abs(got[k] - corr[k]) < 1e-9 for k in METRIC_KEYS)
    out["control_check"] = {
        "what": "final20 recomputed here vs eval/runs/final_corrected_v1_final20.json",
        "expected": corr,
        "got": {k: got[k] for k in METRIC_KEYS},
        "matches": control_ok,
    }

    dest = EVAL_DIR / "runs" / "p1_shipped_chunker_all_sets.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    width = max(len(k) for k in METRIC_KEYS) + 2
    for ds_name in DATASETS:
        c = out["comparison"][ds_name]
        print(f"--- {ds_name} ---")
        print(f"{'metric':<{width}}{'published':>11}{'shipped':>11}{'delta':>9}")
        for k in METRIC_KEYS:
            b, n, d = c["published_v1_experimental_chunker"][k], \
                      c["recomputed_v1_shipped_chunker"][k], c["delta"][k]
            flag = "" if abs(d) < 1e-9 else "   <-- CHANGED"
            print(f"{k:<{width}}{b:>11}{n:>11}{d:>9}{flag}")
        print(f"{'':<{width}}{'':>11}{'':>11}  changed={c['changed']}\n")

    print(f"control (final20 vs corrected run): {'PASS' if control_ok else 'FAIL'}")
    if not control_ok:
        print("  Control failed -- do not trust the original10 / heldout18 rows above.")
    print(f"\nsaved -> {dest.relative_to(ROOT)}")
    return 0 if control_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

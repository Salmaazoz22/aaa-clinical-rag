# -*- coding: utf-8 -*-
"""PHASE 7 — held-out generalization test for selective reranking Policy A.

Isolated by design. Production (`retrieval.index.retrieve`) remains pure dense
MedEmbed and does not import this module.

What this tests
---------------
Phase 6 found Policy A on the ORIGINAL 10 questions, deriving its threshold from
those same 10 queries. That is transductive, so it establishes nothing about
generalization. Here the threshold is FROZEN at the Phase 6 value and applied,
unchanged, to 18 questions it has never seen.

    FROZEN_THRESHOLD = 0.035120        rerank iff margin_1_10 < FROZEN_THRESHOLD

The threshold is NOT recomputed from the held-out distribution. No percentile is
taken. Nothing is tuned. The original 10 questions and the held-out 18 are never
merged into a single evaluation.

Three configurations, all sharing one MedEmbed top-30 candidate pool per query:

    A. MedEmbed production baseline  - dense top-10, no reranking
    B. Full MedCPT reranking        - rerank every query
    C. Selective Policy A           - rerank only when margin_1_10 < threshold

Reranker is `ncbi/MedCPT-Cross-Encoder` at revision
`71caf65d4927987813984f54c284405a13fcca49`, 512-token limit -- identical to
Phases 5 and 6.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(EVAL_DIR))

# The Phase 6 value, frozen. Never recomputed here.
FROZEN_THRESHOLD = 0.035120
CANDIDATE_DEPTH = 30
EVAL_DEPTH = 10
HELDOUT_GOLD = EVAL_DIR / "gold_standard_heldout.json"


def run() -> dict:
    import retrieval.index as cr
    import ingestion.chunking as cc
    from evaluate import load_gold, evaluate_run, is_relevant
    from experimental_rerank import (DEFAULT_RERANK_MODEL, DEFAULT_REVISION,
                                     rerank_hits, full_reranked_order)
    from experimental_selective_rerank import confidence_signals

    gold = load_gold(HELDOUT_GOLD)
    specs = {s["query_id"]: s for s in gold["queries"]}
    queries = {s["query_id"]: s["query"] for s in gold["queries"]}
    topics = {s["query_id"]: s.get("topic", "") for s in gold["queries"]}

    index = cr.load_index(ROOT)
    model = cc.load_embedder(index["model_name"])

    dense, signals, reranked, rr_order = {}, {}, {}, {}
    for qid in sorted(queries):
        hits = cr.retrieve(queries[qid], index, model=model, top_k=CANDIDATE_DEPTH)
        dense[qid] = hits
        signals[qid] = confidence_signals(hits)
        reranked[qid] = rerank_hits(queries[qid], hits, top_k=EVAL_DEPTH)[0]
        rr_order[qid] = full_reranked_order(queries[qid], hits)

    fired = {qid: signals[qid]["margin_1_10"] < FROZEN_THRESHOLD for qid in sorted(queries)}

    configs = {
        "A_medembed_dense": {qid: dense[qid][:EVAL_DEPTH] for qid in sorted(queries)},
        "B_full_medcpt": {qid: reranked[qid] for qid in sorted(queries)},
        "C_selective_policy_a": {
            qid: (reranked[qid] if fired[qid] else dense[qid][:EVAL_DEPTH]) for qid in sorted(queries)
        },
    }

    out = {
        "gold_standard_id": gold["gold_standard_id"],
        "heldout_gold_sha256": gold["_sha256"],
        "frozen_threshold": FROZEN_THRESHOLD,
        "threshold_source": "Phase 6 P25 of margin_1_10 over the ORIGINAL 10 questions; frozen, not recomputed",
        "candidate_depth": CANDIDATE_DEPTH,
        "eval_depth": EVAL_DEPTH,
        "embedding_model": index["meta"]["model_name"],
        "embedding_revision": index["meta"]["model_revision"],
        "reranker": DEFAULT_RERANK_MODEL,
        "reranker_revision": DEFAULT_REVISION,
        "n_questions": len(queries),
        "topics": topics,
        "questions": queries,
        "confidence": {qid: signals[qid] for qid in sorted(queries)},
        "policy_a_fired": fired,
        "configs": {},
    }

    per_config_first = {}
    for name, runs in configs.items():
        res = evaluate_run(runs, gold)
        per_config_first[name] = {q["query_id"]: q for q in res["per_query"]}
        out["configs"][name] = {
            "metrics": res["metrics"],
            "per_query": [
                {"query_id": q["query_id"], "topic": topics[q["query_id"]],
                 "first_relevant_rank": q["first_relevant_rank"],
                 "relevant_top1": q["relevant_top1"], "p_at_1": q["p_at_1"],
                 "p_at_5": q["p_at_5"], "recall_at_5": q["recall_at_5"],
                 "recall_at_10": q["recall_at_10"],
                 "top1_chunk": q["top1"]["chunk_id"], "top1_doc": q["top1"]["document_id"]}
                for q in res["per_query"]
            ],
        }

    # --- safety: what happened to queries the dense retriever already got right
    dense_pq, sel_pq, full_pq = (per_config_first["A_medembed_dense"],
                                 per_config_first["C_selective_policy_a"],
                                 per_config_first["B_full_medcpt"])
    correct = [q for q in sorted(queries) if dense_pq[q]["relevant_top1"]]
    out["top1_safety"] = {
        "dense_correct_top1": correct,
        "selective": {
            "reranked_among_correct": [q for q in correct if fired[q]],
            "preserved": [q for q in correct if sel_pq[q]["relevant_top1"]],
            "lost": [q for q in correct if not sel_pq[q]["relevant_top1"]],
            "lost_and_was_reranked": [q for q in correct if not sel_pq[q]["relevant_top1"] and fired[q]],
            "gained": [q for q in sorted(queries) if sel_pq[q]["relevant_top1"] and not dense_pq[q]["relevant_top1"]],
        },
        "full_rerank": {
            "preserved": [q for q in correct if full_pq[q]["relevant_top1"]],
            "lost": [q for q in correct if not full_pq[q]["relevant_top1"]],
            "gained": [q for q in sorted(queries) if full_pq[q]["relevant_top1"] and not dense_pq[q]["relevant_top1"]],
        },
    }

    # --- Q4-like structural pattern: relevant outside dense top-10 but inside top-30,
    #     and promoted into the top-10 by selective reranking.
    q4_like = []
    for qid in sorted(queries):
        spec = specs[qid]
        dense_rel = [i + 1 for i, h in enumerate(dense[qid]) if is_relevant(h, spec)]
        in10 = [r for r in dense_rel if r <= EVAL_DEPTH]
        in30_only = [r for r in dense_rel if r > EVAL_DEPTH]
        pattern = bool(in30_only) and not in10
        rescued = pattern and sel_pq[qid]["first_relevant_rank"] is not None
        q4_like.append({
            "query_id": qid, "topic": topics[qid],
            "dense_relevant_ranks_in_top30": dense_rel,
            "structural_pattern": pattern,
            "policy_a_fired": fired[qid],
            "rescued_into_top10": rescued,
            "selective_first_relevant": sel_pq[qid]["first_relevant_rank"],
        })
    out["q4_like_analysis"] = q4_like

    # --- threshold proximity (analysis only; no grid search, no retuning)
    out["threshold_proximity"] = sorted(
        [{"query_id": qid, "topic": topics[qid],
          "margin_1_10": signals[qid]["margin_1_10"],
          "distance_from_threshold": signals[qid]["margin_1_10"] - FROZEN_THRESHOLD,
          "fired": fired[qid]} for qid in sorted(queries)],
        key=lambda d: d["margin_1_10"],
    )
    return out


def main() -> int:
    out = run()
    (EVAL_DIR / "runs" / "phase7_heldout.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("=== PHASE 7 held-out generalization ===")
    print("  held-out gold sha256 :", out["heldout_gold_sha256"])
    print("  questions            :", out["n_questions"])
    print("  FROZEN threshold     :", out["frozen_threshold"], "(not recomputed)")
    print("  embedding            :", out["embedding_model"], out["embedding_revision"])
    print("  reranker             :", out["reranker"], out["reranker_revision"])

    keys = ("P@1", "P@3", "P@5", "MRR", "Recall@5", "Recall@10", "Relevant_Top1", "Answering@5")
    print(f"\n{'config':<24}" + "".join(f"{k:>14}" for k in keys))
    for name, cfg in out["configs"].items():
        print(f"{name:<24}" + "".join(f"{cfg['metrics'][k]:>14}" for k in keys))

    n_fired = sum(1 for v in out["policy_a_fired"].values() if v)
    print(f"\n  Policy A fired on {n_fired}/{out['n_questions']} "
          f"({100*n_fired/out['n_questions']:.0f}%): "
          f"{[q for q, v in out['policy_a_fired'].items() if v]}")

    print(f"\n{'Q':>3} {'topic':<32} {'top1':>7} {'m1_2':>7} {'m1_5':>7} {'m1_10':>7} {'fired':>6} "
          f"{'dense':>6} {'full':>6} {'sel':>6}")
    dense_pq = {q["query_id"]: q for q in out["configs"]["A_medembed_dense"]["per_query"]}
    full_pq = {q["query_id"]: q for q in out["configs"]["B_full_medcpt"]["per_query"]}
    sel_pq = {q["query_id"]: q for q in out["configs"]["C_selective_policy_a"]["per_query"]}
    for qid in sorted(dense_pq, key=int):
        c = out["confidence"][qid]
        print(f"{qid:>3} {out['topics'][qid][:32]:<32} {c['confidence_top1']:>7.4f} "
              f"{c['margin_1_2']:>7.4f} {c['margin_1_5']:>7.4f} {c['margin_1_10']:>7.4f} "
              f"{('YES' if out['policy_a_fired'][qid] else '-'):>6} "
              f"{str(dense_pq[qid]['first_relevant_rank'] or '-'):>6} "
              f"{str(full_pq[qid]['first_relevant_rank'] or '-'):>6} "
              f"{str(sel_pq[qid]['first_relevant_rank'] or '-'):>6}")

    s = out["top1_safety"]
    print(f"\n=== top-1 safety ===")
    print(f"  dense correct at top-1        : {s['dense_correct_top1']}")
    print(f"  selective: reranked among them: {s['selective']['reranked_among_correct']}")
    print(f"  selective: PRESERVED          : {s['selective']['preserved']}")
    print(f"  selective: LOST               : {s['selective']['lost']}")
    print(f"  selective: gained             : {s['selective']['gained']}")
    print(f"  full rerank: LOST             : {s['full_rerank']['lost']}")
    print(f"  full rerank: gained           : {s['full_rerank']['gained']}")

    print("\n=== Q4-like structural cases (relevant outside top-10 but inside top-30) ===")
    for r in out["q4_like_analysis"]:
        if r["structural_pattern"]:
            print(f"  H{r['query_id']} [{r['topic']}] dense ranks {r['dense_relevant_ranks_in_top30']} "
                  f"fired={r['policy_a_fired']} rescued={r['rescued_into_top10']} "
                  f"-> selective first relevant {r['selective_first_relevant']}")
    n_pattern = sum(1 for r in out["q4_like_analysis"] if r["structural_pattern"])
    n_rescued = sum(1 for r in out["q4_like_analysis"] if r["rescued_into_top10"])
    print(f"  cases with the pattern: {n_pattern}   rescued by Policy A: {n_rescued}")

    print("\n=== threshold proximity (analysis only) ===")
    for r in out["threshold_proximity"]:
        print(f"  H{r['query_id']:<3} margin_1_10={r['margin_1_10']:.5f}  "
              f"delta={r['distance_from_threshold']:+.5f}  {'FIRED' if r['fired'] else ''}")
    print("\nsaved -> eval/runs/phase7_heldout.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

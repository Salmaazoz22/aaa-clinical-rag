# -*- coding: utf-8 -*-
"""FINAL CORRECTED VALIDATION — V1 with REJECT_CITATION_HEADINGS = True, final20 only.

A validation gate, not an experiment. Nothing is tuned, nothing is swept, and the
result is not used to adjust any parameter.

Scope, deliberately narrow:
  * chunker      : V1 atomic page-safe, unchanged except the citation-heading fix
  * embedding    : MedEmbed-base-v0.1 @ pinned revision (unchanged)
  * retrieval    : dense cosine, top-10 (unchanged)
  * dataset      : eval/gold_standard_final20.json ONLY
  * gold / rules : untouched

Writes a NEW artifact, eval/runs/final_corrected_v1_final20.json. It overwrites no
historical run: eval/runs/exp12_atomic_chunking.json and
eval/final_evaluation_results.json are left exactly as they were.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(EVAL_DIR))

import ingestion.chunking as cc  # noqa: E402
import experimental_atomic_chunking as ex  # noqa: E402
from evaluate import evaluate_run, load_gold, matched_passages, normalise  # noqa: E402

GOLD = EVAL_DIR / "gold_standard_final20.json"
EVAL_DEPTH = 10
KEYS = ("P@1", "P@3", "P@5", "MRR", "Recall@5", "Recall@10", "Relevant_Top1", "Answering@5")


def anchor_census(pages_df):
    """Anchors found per document with the fix ON and with it OFF."""
    out = {}
    for flag in (True, False):
        ex.REJECT_CITATION_HEADINGS = flag
        per_doc = {}
        for doc_id, doc_pages in pages_df.groupby("document_id", sort=True):
            marked = ex.build_marked_text(doc_pages.sort_values("page_number"))
            kinds = Counter(k for _, k, _ in ex.find_anchors(marked))
            per_doc[doc_id] = dict(kinds)
        out["with_fix" if flag else "without_fix"] = per_doc
    ex.REJECT_CITATION_HEADINGS = True  # leave it ON
    return out


def profile(index):
    p = ex.chunk_profile(index)
    p.update(ex.page_span_profile(index))
    return p


def main() -> int:
    assert ex.REJECT_CITATION_HEADINGS is True, "this validation requires the fix ON"

    gold = load_gold(GOLD)
    specs = {s["query_id"]: s for s in gold["queries"]}
    queries = {s["query_id"]: s["query"] for s in gold["queries"]}

    print("loading MedEmbed at pinned revision ...")
    model = cc.load_embedder(cc.DEFAULT_EMBED_MODEL)
    loaded = cc.load_processed(ROOT)
    pages_df, recs_df = loaded["pages_df"], loaded["recommendations_df"]

    print("anchor census (fix on vs off) ...")
    anchors = anchor_census(pages_df)

    print("building corrected V1 chunks ...")
    chunks = ex.build_atomic_chunks(pages_df, recs_df, keep_page_breaks=True,
                                    rec_token_budget=cc.max_content_tokens())
    index = ex.index_chunks(chunks, model)

    print("scoring final20 ...")
    runs = {qid: ex.retrieve(q, index, model, top_k=EVAL_DEPTH) for qid, q in queries.items()}
    res = evaluate_run(runs, gold)

    # historical V1 (fix OFF) for comparison -- read, never rewritten
    hist = json.loads((EVAL_DIR / "final_evaluation_results.json").read_text(encoding="utf-8"))
    hist_v1 = hist["metrics"]["final20"]["V1_atomic_pagesafe"]
    hist_metrics = {k: hist_v1["metrics"][k] for k in KEYS}
    hist_pq = {q["query_id"]: q for q in hist_v1["per_query"]}
    hist_profile = hist["chunk_profiles"]["V1_atomic_pagesafe"]

    cur_metrics = {k: res["metrics"][k] for k in KEYS}
    cur_pq = {q["query_id"]: q for q in res["per_query"]}

    changed = []
    for qid in sorted(queries):
        h, c = hist_pq[qid], cur_pq[qid]
        # evaluate_run() nests top-1 under "top1"; the historical file flattened it.
        c_top1 = c["top1"]["chunk_id"]
        if (h["first_relevant_rank"] != c["first_relevant_rank"]
                or h["relevant_top1"] != c["relevant_top1"]
                or h["top1_chunk"] != c_top1):
            changed.append({
                "query_id": qid, "question": queries[qid],
                "historical_first_relevant_rank": h["first_relevant_rank"],
                "corrected_first_relevant_rank": c["first_relevant_rank"],
                "historical_relevant_top1": h["relevant_top1"],
                "corrected_relevant_top1": c["relevant_top1"],
                "historical_top1_chunk": h["top1_chunk"],
                "corrected_top1_chunk": c_top1,
            })

    cur_profile = profile(index)
    p_at_1 = cur_metrics["P@1"]
    material_regression = any(cur_metrics[k] < hist_metrics[k] - 0.05
                              for k in ("P@1", "MRR", "Recall@5", "Recall@10"))
    decision = ("ADOPT WITH CAVEATS" if (abs(p_at_1 - 0.55) <= 0.051 and not material_regression)
                else "REOPEN V1 DECISION")

    evidence = []
    for qid, hits in runs.items():
        spec = specs[qid]
        groups = spec["required_facts"]["groups"]
        import re as _re
        for h in hits:
            norm = normalise(h.get("chunk_text") or "")
            evidence.append({
                "query_id": qid, "question": queries[qid], "rank": h["rank"],
                "similarity": round(float(h["similarity_score"]), 6),
                "chunk_id": h["chunk_id"], "document_id": h["document_id"],
                "page_start": h["page_start"], "page_end": h["page_end"],
                "section_title": h.get("section"),
                "relevant": bool(matched_passages(h, spec)),
                "required_facts_covered": [g["name"] for g in groups
                                           if any(_re.search(p, norm, _re.I) for p in g["any_of"])],
                "required_facts_needed": spec["required_facts"]["min_groups"],
                "chunk_text": h.get("chunk_text"),
            })

    out = {
        "label": "FINAL CORRECTED VALIDATION",
        "what_this_is": ("V1 atomic page-safe chunking with REJECT_CITATION_HEADINGS = True, "
                         "scored on final20 only. A validation gate, not an experiment. Nothing "
                         "was tuned and no other dataset was touched."),
        "overwrites_nothing": ("eval/runs/exp12_atomic_chunking.json and "
                               "eval/final_evaluation_results.json are unchanged historical "
                               "artifacts produced with the fix OFF."),
        "configuration": {
            "chunker": "V1_atomic_pagesafe",
            "reject_citation_headings": True,
            "embedding_model": cc.DEFAULT_EMBED_MODEL,
            "embedding_revision": cc.model_revision(cc.DEFAULT_EMBED_MODEL),
            "retrieval": "dense cosine, top-10, no reranking, no query processing",
            "eval_depth": EVAL_DEPTH,
        },
        "dataset": {"name": "final20", "path": "eval/gold_standard_final20.json",
                    "sha256": gold["_sha256"], "n_questions": len(queries)},
        "metrics_corrected": cur_metrics,
        "metrics_historical_v1_fix_off": hist_metrics,
        "delta": {k: round(cur_metrics[k] - hist_metrics[k], 4) for k in KEYS},
        "chunk_profile_corrected": cur_profile,
        "chunk_profile_historical_v1": hist_profile,
        "anchor_census": anchors,
        "questions_with_any_change": changed,
        "n_questions_changed": len(changed),
        "decision": decision,
        "per_query": res["per_query"],
        "evidence": evidence,
    }

    out_path = EVAL_DIR / "runs" / "final_corrected_v1_final20.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\n" + "=" * 92)
    print("FINAL CORRECTED VALIDATION -- V1, REJECT_CITATION_HEADINGS = True, final20 only")
    print("=" * 92)
    print(f"\n{'metric':<16}{'historical V1':>16}{'corrected V1':>16}{'delta':>12}")
    for k in KEYS:
        print(f"{k:<16}{hist_metrics[k]:>16}{cur_metrics[k]:>16}{cur_metrics[k]-hist_metrics[k]:>+12.4f}")

    print(f"\n{'chunk stat':<34}{'historical V1':>16}{'corrected V1':>16}")
    for label, key in (("total chunks", "total_chunks"), ("indexed chunks", "indexed_chunks"),
                       ("with recommendation_id", "with_recommendation_id"),
                       ("single-page chunks", "single_page_chunks")):
        print(f"{label:<34}{hist_profile[key]:>16}{cur_profile[key]:>16}")
    for label, key in (("mean tokens", "mean"), ("max tokens", "max"), ("over model limit", "over_limit")):
        print(f"{label:<34}{hist_profile['tokens'][key]:>16}{cur_profile['tokens'][key]:>16}")
    print(f"{'mean pages/chunk':<34}{hist_profile['mean_pages_per_chunk']:>16}"
          f"{cur_profile['mean_pages_per_chunk']:>16}")

    print("\nanchors per document (without fix -> with fix):")
    for doc in sorted(anchors["with_fix"]):
        w, wo = anchors["with_fix"][doc], anchors["without_fix"][doc]
        print(f"  {doc:<14} sections {wo.get('section', 0):>4} -> {w.get('section', 0):<4}"
              f"   recommendations {wo.get('recommendation', 0):>4} -> {w.get('recommendation', 0)}")

    print(f"\nquestions whose result changed: {len(changed)}")
    for c in changed:
        print(f"  Q{c['query_id']}: first relevant {c['historical_first_relevant_rank']} -> "
              f"{c['corrected_first_relevant_rank']}, top1 relevant "
              f"{c['historical_relevant_top1']} -> {c['corrected_relevant_top1']}")

    print(f"\nDECISION: {decision}")
    print(f"saved -> {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

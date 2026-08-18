# -*- coding: utf-8 -*-
"""Re-score every (dataset, config) with the page-overlap confound removed.

No retrieval is re-run and no model is loaded. The rankings are read back from
eval/final_evidence.json exactly as they were produced, and each retrieved chunk
is re-judged as if it occupied only its START PAGE. This eliminates the frozen
rule's page-span term, which Experiment 12 V4 measured to be worth about
+0.10 P@1 on its own.

It is a strict LOWER BOUND: a chunk that legitimately spans pages is penalised.
For a chunker whose chunks are almost all single-page the bound is tight; for one
producing 5-page chunks it is very loose. Both numbers are reported.

Writes eval/conservative_rescoring.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPTS_DIR.parent
ROOT = EVAL_DIR.parent
# ROOT for the installed packages, SCRIPTS_DIR for sibling eval modules.
for _extra in (str(ROOT), str(SCRIPTS_DIR)):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from evaluate import evaluate_run, load_gold  # noqa: E402

DATASETS = {
    "original10": EVAL_DIR / "gold_standard.json",
    "heldout18": EVAL_DIR / "gold_standard_heldout.json",
    "final20": EVAL_DIR / "gold_standard_final20.json",
}
KEYS = ("P@1", "P@3", "P@5", "MRR", "Recall@5", "Recall@10", "Relevant_Top1", "Answering@5")


def main() -> int:
    ev = json.loads((EVAL_DIR / "final_evidence.json").read_text(encoding="utf-8"))

    # rebuild ranked hit lists straight from the recorded evidence
    runs: dict[tuple[str, str], dict[int, list[dict]]] = {}
    for r in ev["rows"]:
        runs.setdefault((r["dataset"], r["config"]), {}).setdefault(r["query_id"], []).append(r)
    for key in runs:
        for qid in runs[key]:
            runs[key][qid].sort(key=lambda h: h["rank"])

    out: dict[str, dict] = {
        "method": ("Rankings replayed verbatim from eval/final_evidence.json; each chunk re-judged "
                   "with page_end forced to page_start. No retrieval was re-run."),
        "why": ("The frozen relevance rule requires the chunk page span to overlap the answer "
                "passage page span. Experiment 12 V4 showed that widening page metadata alone, with "
                "identical retrieval, is worth about +0.10 P@1. This removes that term entirely."),
        "caveat": ("Strict lower bound: chunks that genuinely span pages are penalised. Tight for "
                   "near-single-page chunkers, very loose for multi-page ones."),
        "datasets": {},
    }

    for ds, path in DATASETS.items():
        gold = load_gold(path)
        block: dict[str, dict] = {}
        for (d, cfg), per_q in runs.items():
            if d != ds:
                continue
            as_scored = evaluate_run({q: h for q, h in per_q.items()}, gold)["metrics"]
            narrowed = {q: [dict(h, page_end=h["page_start"]) for h in hits]
                        for q, hits in per_q.items()}
            conservative = evaluate_run(narrowed, gold)["metrics"]
            multi = sum(1 for hits in per_q.values() for h in hits if h["page_end"] > h["page_start"])
            total = sum(len(h) for h in per_q.values())
            block[cfg] = {
                "as_scored": {k: as_scored[k] for k in KEYS},
                "start_page_only": {k: conservative[k] for k in KEYS},
                "delta": {k: round(conservative[k] - as_scored[k], 4) for k in KEYS},
                "pct_retrieved_chunks_multipage": round(100 * multi / total, 1) if total else 0.0,
            }
        out["datasets"][ds] = block

    (EVAL_DIR / "conservative_rescoring.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    for ds, block in out["datasets"].items():
        print(f"\n=== {ds} ===")
        print(f"{'config':<24}{'P@1 scored':>12}{'P@1 strict':>12}{'MRR scored':>12}"
              f"{'MRR strict':>12}{'% multipage':>13}")
        for cfg, v in block.items():
            print(f"{cfg:<24}{v['as_scored']['P@1']:>12}{v['start_page_only']['P@1']:>12}"
                  f"{v['as_scored']['MRR']:>12}{v['start_page_only']['MRR']:>12}"
                  f"{v['pct_retrieved_chunks_multipage']:>13}")
    print("\nsaved -> eval/conservative_rescoring.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

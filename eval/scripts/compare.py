# -*- coding: utf-8 -*-
"""Compare two saved evaluation runs against the same frozen gold standard.

    python eval/compare.py fresh_baseline_dense exp2_furniture
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"
KEYS = ["P@1", "P@3", "P@5", "MRR", "Recall@5", "Recall@10", "Relevant_Top1", "Answering@5"]


def load(label: str) -> dict:
    return json.loads((RUNS / f"{label}.json").read_text(encoding="utf-8"))


def main() -> int:
    base, exp = load(sys.argv[1]), load(sys.argv[2])
    if base["gold_sha256"] != exp["gold_sha256"]:
        print("REFUSING: runs scored against different gold standards")
        return 1

    print(f"gold sha256 {base['gold_sha256'][:16]}...  (identical for both runs)\n")
    print(f"| Metric | {sys.argv[1]} | {sys.argv[2]} | Delta |")
    print("|---|---:|---:|---:|")
    for k in KEYS:
        b, e = base["metrics"][k], exp["metrics"][k]
        d = e - b
        arrow = "=" if d == 0 else ("+" if d > 0 else "")
        print(f"| {k} | {b} | {e} | {arrow}{round(d, 4) if d else '0'} |")

    print(f"\n| Q | first rel (base) | first rel (exp) | P@1 b/e | P@5 b/e | R@5 b/e | R@10 b/e | verdict |")
    print("|---|---|---|---|---|---|---|---|")
    imp = reg = same = 0
    for qb, qe in zip(base["per_query"], exp["per_query"]):
        fb = qb["first_relevant_rank"] or 99
        fe = qe["first_relevant_rank"] or 99
        score_b = (qb["p_at_1"], qb["rr"], qb["recall_at_5"], qb["recall_at_10"], qb["p_at_5"])
        score_e = (qe["p_at_1"], qe["rr"], qe["recall_at_5"], qe["recall_at_10"], qe["p_at_5"])
        if score_e > score_b:
            verdict, imp = "IMPROVED", imp + 1
        elif score_e < score_b:
            verdict, reg = "REGRESSED", reg + 1
        else:
            verdict, same = "unchanged", same + 1
        print(f"| {qb['query_id']} | {qb['first_relevant_rank'] or '-'} | {qe['first_relevant_rank'] or '-'} "
              f"| {qb['p_at_1']:.2f}/{qe['p_at_1']:.2f} | {qb['p_at_5']:.2f}/{qe['p_at_5']:.2f} "
              f"| {qb['recall_at_5']:.2f}/{qe['recall_at_5']:.2f} | {qb['recall_at_10']:.2f}/{qe['recall_at_10']:.2f} "
              f"| {verdict} |")
    print(f"\nimproved {imp} / regressed {reg} / unchanged {same}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

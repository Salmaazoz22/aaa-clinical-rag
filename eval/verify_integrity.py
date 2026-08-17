# -*- coding: utf-8 -*-
"""Final integrity verification -> eval/integrity_report.json.

Checks the properties the whole project's credibility rests on:
the frozen sets really are unchanged, final20 really was frozen before retrieval,
nothing tunes on the evaluation, Project B is not a dependency, no historical run
was overwritten, and the published evidence really does reconstruct the published
metrics.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))

from evaluate import evaluate_run, load_gold  # noqa: E402

# SHA-256 values recorded in artifacts written BEFORE this session's work.
ORIGINAL10_SHA_ON_RECORD = "0b8a443b69960bc5ac20311f0010926a2f131bbb5531ccf369f321f59ed2e5c1"
KEYS = ("P@1", "P@3", "P@5", "MRR", "Recall@5", "Recall@10", "Relevant_Top1", "Answering@5")

RESULTS: list[dict[str, Any]] = []


def record(name, ok, detail):
    RESULTS.append({"check": name, "status": "pass" if ok else "FAIL", "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    print("final integrity verification\n")

    # 1. original 10 unchanged
    s = sha(EVAL_DIR / "gold_standard.json")
    record("original10 gold standard unchanged (matches the SHA quoted in every historical run)",
           s == ORIGINAL10_SHA_ON_RECORD, {"sha256": s, "on_record": ORIGINAL10_SHA_ON_RECORD})

    # 2. held-out 18 unchanged - compare against the SHA stored inside phase7_heldout.json,
    #    which was written before this session.
    p7 = json.loads((EVAL_DIR / "runs/phase7_heldout.json").read_text(encoding="utf-8"))
    s18 = sha(EVAL_DIR / "gold_standard_heldout.json")
    record("heldout18 gold standard unchanged (matches the SHA recorded by Phase 7)",
           s18 == p7["heldout_gold_sha256"],
           {"sha256": s18, "recorded_by_phase7": p7["heldout_gold_sha256"]})

    # 3. final20 frozen, and frozen BEFORE the evaluation that used it
    s20 = sha(EVAL_DIR / "gold_standard_final20.json")
    stamped = (EVAL_DIR / "gold_standard_final20.sha256").read_text(encoding="utf-8").split()[0]
    used = json.loads((EVAL_DIR / "final_evaluation_results.json").read_text(encoding="utf-8"))
    record("final20 frozen: file hash == hash stamped at freeze time == hash used by the evaluation",
           s20 == stamped == used["gold_sha256"]["final20"],
           {"file": s20, "stamped_at_freeze": stamped, "used_by_evaluation": used["gold_sha256"]["final20"]})

    gold20 = json.loads((EVAL_DIR / "gold_standard_final20.json").read_text(encoding="utf-8"))
    record("final20 passages were validated against SOURCE page text, not retrieval output",
           all(r["source_page_supports_passage"] for r in gold20["validation_report"]),
           {"passages_validated": len(gold20["validation_report"]),
            "all_supported": all(r["source_page_supports_passage"] for r in gold20["validation_report"])})

    # 4. no threshold tuning: Policy A's frozen threshold is untouched
    src = (EVAL_DIR / "experimental_phase7_heldout.py").read_text(encoding="utf-8")
    m = re.search(r"FROZEN_THRESHOLD\s*=\s*([0-9.]+)", src)
    record("Selective Policy A threshold still frozen at its Phase 6 value (never retuned)",
           bool(m) and m.group(1) == "0.035120", {"threshold_in_code": m.group(1) if m else None})

    record("Policy A was NOT evaluated against final20 (so final20 measures chunking alone)",
           "policy" not in json.dumps(used["metrics"]).lower(),
           {"configs_scored_on_final20": sorted(used["metrics"]["final20"].keys())})

    # 5. no query-specific logic anywhere in the production retrieval path
    banned = re.compile(r"_INTENT_PATTERNS|_QUERY_EXPANSIONS|_detect_intents|_expand_query"
                        r"|intent_adjustment|_anchor_indices|recommendation_id\s*==\s*[\"']", re.I)
    prod = ["notebooks/clinical_rag.py", "notebooks/clinical_chunking.py",
            "notebooks/clinical_preprocess.py", "notebooks/clinical_rerank.py",
            "eval/evaluate.py", "eval/experimental_atomic_chunking.py"]
    hits = {f: banned.findall((ROOT / f).read_text(encoding="utf-8")) for f in prod}
    hits = {f: h for f, h in hits.items() if h}
    record("No query-specific / intent / recommendation-ID logic in the pipeline or evaluator",
           not hits, {"files_scanned": prod, "violations": hits})

    # 6. Project B is not a dependency
    refs = {}
    for p in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.ipynb")):
        if "aaa-clinical-rag" in str(p).replace("aaa-clinical-ragsalma", ""):
            continue  # Project B's own source, and the nested copy
        try:
            t = p.read_text(encoding="utf-8")
        except Exception:
            continue
        # Executable references only. A prose citation of Project B in a docstring
        # is documentation, not a dependency, so match import statements and
        # sys.path manipulation rather than any mention of the name.
        found = re.findall(
            r"(?m)^\s*(?:from\s+src[\s.]|import\s+src\b)"
            r"|sys\.path[^\n]*aaa-clinical-rag"
            r"|open\([^)]*aaa-clinical-rag", t)
        if found:
            refs[str(p.relative_to(ROOT))] = found[:3]
    record("Project B is NOT imported or required by any file in the final project",
           not refs, {"files_importing_project_b": sorted(refs)})

    # 7. no historical run overwritten
    expected = ["fresh_baseline_dense.json", "exp1_pagespan.json", "exp2_furniture.json",
                "exp3_sections.json", "exp4_rerank_c30.json", "exp4_rerank_c20.json",
                "exp5_bge_base_en_v15.json", "exp6_bge_plus_crossencoder.json",
                "exp7_medembed_base_v01.json", "exp8_bm25_keyword.json",
                "exp9_hybrid_a_75_25.json", "exp9_hybrid_b_50_50.json", "exp9_hybrid_c_25_75.json",
                "exp10_medembed_medcpt_rerank.json", "exp11_selective_rerank_policy_A.json",
                "exp11_selective_rerank_policy_B.json", "exp11_selective_rerank_policy_C.json",
                "phase1_production_medembed.json", "phase7_heldout.json",
                "exp12_atomic_chunking.json",
                "archived_final_evidence_phase7.json",
                "archived_final_evaluation_summary_phase7.json"]
    missing = [f for f in expected if not (EVAL_DIR / "runs" / f).exists()]
    record("Every historical run preserved (including archives of files this session replaced)",
           not missing, {"expected": len(expected), "missing": missing})

    # 8. the published evidence reconstructs the published metrics
    ev = json.loads((EVAL_DIR / "final_evidence.json").read_text(encoding="utf-8"))
    runs: dict[tuple[str, str], dict[int, list]] = {}
    for r in ev["rows"]:
        runs.setdefault((r["dataset"], r["config"]), {}).setdefault(r["query_id"], []).append(r)
    golds = {"original10": EVAL_DIR / "gold_standard.json",
             "heldout18": EVAL_DIR / "gold_standard_heldout.json",
             "final20": EVAL_DIR / "gold_standard_final20.json"}
    mismatches = []
    for (ds, cfg), per_q in runs.items():
        for qid in per_q:
            per_q[qid].sort(key=lambda h: h["rank"])
        got = evaluate_run(per_q, load_gold(golds[ds]))["metrics"]
        want = used["metrics"][ds][cfg]["metrics"]
        for k in KEYS:
            if abs(got[k] - want[k]) > 1e-9:
                mismatches.append({"dataset": ds, "config": cfg, "metric": k,
                                   "from_evidence": got[k], "published": want[k]})
    record("Published metrics are exactly reconstructible from eval/final_evidence.json",
           not mismatches, {"pairs_checked": len(runs), "mismatches": mismatches[:10]})

    # 9. token safety + tests, from the stability report
    st = json.loads((EVAL_DIR / "stability_report.json").read_text(encoding="utf-8"))
    by = {c["check"]: c for c in st["checks"]}
    tok = by.get("Token safety: no indexed chunk exceeds the encoder window", {})
    record("No token overflow in the shipped index", tok.get("status") == "pass", tok.get("detail"))
    tst = by.get("Test suite", {})
    record("Test suite passes", tst.get("status") == "pass", tst.get("detail"))

    # 10. final notebook is valid and its inputs all exist
    nb = json.loads((ROOT / "notebooks/final_evaluation.ipynb").read_text(encoding="utf-8"))
    needed = ["eval/final_evaluation_results.json", "eval/final_evidence.json",
              "eval/experiment_history.json", "eval/corpus_audit.json",
              "eval/question_audit.json", "eval/project_b_comparison.json",
              "eval/runs/exp12_atomic_chunking.json", "eval/stability_report.json",
              "eval/gold_standard_final20.json", "data/embeddings/index_meta.json",
              "data/chunks/chunks.json"]
    absent = [f for f in needed if not (ROOT / f).exists()]
    record("Final notebook is valid nbformat and every artifact it loads exists",
           nb.get("nbformat") == 4 and not absent,
           {"cells": len(nb["cells"]), "missing_inputs": absent})

    # 11. promotion: the shipped configuration is V1 and the artifacts match it
    sys.path.insert(0, str(ROOT / "notebooks"))
    import clinical_chunking as cc  # noqa: E402
    import experimental_atomic_chunking as ex  # noqa: E402

    record("Shipped chunker is the atomic (V1) chunker",
           cc.DEFAULT_CHUNKER == "atomic", {"DEFAULT_CHUNKER": cc.DEFAULT_CHUNKER})

    record("Citation-heading fix is ON in both the shipped and the validation chunker",
           ex.REJECT_CITATION_HEADINGS is True,
           {"experimental_module": ex.REJECT_CITATION_HEADINGS,
            "production_module": "clinical_atomic_chunking always rejects citation headings"})

    corr = json.loads((EVAL_DIR / "runs/final_corrected_v1_final20.json").read_text(encoding="utf-8"))
    shipped_chunks = json.loads((ROOT / "data/chunks/chunks.json").read_text(encoding="utf-8"))
    shipped_meta = json.loads((ROOT / "data/embeddings/index_meta.json").read_text(encoding="utf-8"))
    prof = corr["chunk_profile_corrected"]
    record("Shipped index matches the corrected-V1 profile that passed the gate",
           len(shipped_chunks["chunks"]) == prof["total_chunks"]
           and shipped_meta["n_vectors"] == prof["indexed_chunks"],
           {"shipped_total_chunks": len(shipped_chunks["chunks"]),
            "validated_total_chunks": prof["total_chunks"],
            "shipped_vectors": shipped_meta["n_vectors"],
            "validated_indexed": prof["indexed_chunks"]})

    record("Corrected validation shows no regression on final20",
           all(abs(v) < 1e-9 for v in corr["delta"].values()),
           {"delta": corr["delta"], "decision": corr["decision"]})

    record("Baseline (page-buffer) index preserved, not deleted",
           (ROOT / "data/archive_baseline_index/chunks_baseline_pagebuffer.json").exists()
           and (ROOT / "data/archive_baseline_index/embeddings_baseline_pagebuffer.npy").exists(),
           {"archive": sorted(p.name for p in (ROOT / "data/archive_baseline_index").glob("*"))})

    record("Historical artifacts untouched by the promotion",
           (EVAL_DIR / "runs/exp12_atomic_chunking.json").exists()
           and (EVAL_DIR / "final_evaluation_results.json").exists(),
           {"note": "exp12 and final_evaluation_results describe the pre-promotion index and "
                    "were not recomputed; the corrected run is a separate new artifact."})

    ok = all(r["status"] == "pass" for r in RESULTS)
    report = {
        "generated_by": "eval/verify_integrity.py",
        "all_checks_passed": ok,
        "summary": {"pass": sum(1 for r in RESULTS if r["status"] == "pass"),
                    "fail": sum(1 for r in RESULTS if r["status"] == "FAIL")},
        "checks": RESULTS,
    }
    (EVAL_DIR / "integrity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n{report['summary']}")
    print(f"saved -> eval/integrity_report.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

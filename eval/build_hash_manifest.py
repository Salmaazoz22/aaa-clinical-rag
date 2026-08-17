# -*- coding: utf-8 -*-
"""Write eval/final_artifact_hashes.json — SHA-256 of every frozen artifact.

Run LAST. Anything regenerated after this manifest is written invalidates it;
re-run this script if that happens.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent

FROZEN_GOLD = [
    "eval/gold_standard.json",
    "eval/gold_standard_heldout.json",
    "eval/gold_standard_final20.json",
    "eval/gold_standard_final20.json.sha256",
]
EVIDENCE = [
    "eval/final_evidence.json",
    "eval/final_evaluation_results.json",
    "eval/runs/final_corrected_v1_final20.json",
    "eval/conservative_rescoring.json",
    "eval/experiment_history.json",
    "eval/stability_report.json",
    "eval/integrity_report.json",
    "eval/corpus_audit.json",
    "eval/question_audit.json",
    "eval/project_b_comparison.json",
    "eval/final_recommendation.md",
]
SHIPPED_INDEX = [
    "data/chunks/chunks.json",
    "data/embeddings/embeddings.npy",
    "data/embeddings/embedded_chunks.json",
    "data/embeddings/index_meta.json",
]
PIPELINE = [
    "notebooks/clinical_preprocess.py",
    "notebooks/clinical_chunking.py",
    "notebooks/clinical_atomic_chunking.py",
    "notebooks/clinical_rag.py",
    "notebooks/clinical_rerank.py",
    "notebooks/final_evaluation.ipynb",
    "eval/evaluate.py",
    "tests/test_chunking.py",
]
SOURCE_PDFS = [
    "data/pdfs/abdom-aortic-aneurysm-screening-final-rs.pdf",
    "data/pdfs/abdominal-aortic-aneurysm-diagnosis-and-management-pdf-66141843642565.pdf",
    "data/pdfs/ESVS_2024_AAA_Guidelines.pdf",
    "data/pdfs/SVS_Guideline_AAA_Slides_0.pdf",
]
DOCS = [
    "README.md",
    "docs/experiment_history.md",
    "docs/presentation_story.md",
    "docs/limitations.md",
    "docs/deployment_readiness.md",
    "docs/PROJECT_B_LESSONS.md",
]


def digest(rel: str):
    p = ROOT / rel
    if not p.exists():
        return {"sha256": None, "bytes": None, "status": "MISSING"}
    data = p.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "status": "ok"}


def main() -> int:
    groups = {
        "frozen_gold_standards": FROZEN_GOLD,
        "evaluation_evidence": EVIDENCE,
        "shipped_index": SHIPPED_INDEX,
        "pipeline_source": PIPELINE,
        "source_pdfs": SOURCE_PDFS,
        "documentation": DOCS,
    }
    manifest = {
        "generated_by": "eval/build_hash_manifest.py",
        "note": ("SHA-256 of every frozen artifact at handoff. The gold standards and the "
                 "historical run artifacts must never change; if a hash here stops matching, "
                 "something was modified after the freeze."),
        "shipped_configuration": {
            "chunker": "V1_atomic_pagesafe (clinical_atomic_chunking), citation-heading rejection ON",
            "embedding_model": "abhinand/MedEmbed-base-v0.1",
            "embedding_revision": "7a90c50263f620dff743eb9794b89a42bfc5d765",
            "retrieval": "dense cosine, top-10, 768-dim, L2-normalised, no reranking, no query processing",
            "decision": "ADOPT WITH CAVEATS",
        },
        "artifacts": {g: {f: digest(f) for f in files} for g, files in groups.items()},
    }

    runs = sorted((EVAL_DIR / "runs").glob("*.json"))
    manifest["historical_runs"] = {f"eval/runs/{p.name}": digest(f"eval/runs/{p.name}") for p in runs}

    missing = [f for g in manifest["artifacts"].values() for f, v in g.items() if v["status"] != "ok"]
    manifest["missing_artifacts"] = missing
    manifest["all_present"] = not missing

    out = EVAL_DIR / "final_artifact_hashes.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    total = sum(len(g) for g in manifest["artifacts"].values()) + len(manifest["historical_runs"])
    print(f"hashed {total} artifacts ({len(manifest['historical_runs'])} historical runs)")
    print(f"missing: {missing if missing else 'none'}")
    for f in FROZEN_GOLD[:3]:
        print(f"  {f}\n    {manifest['artifacts']['frozen_gold_standards'][f]['sha256']}")
    print(f"saved -> {out.relative_to(ROOT)}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())

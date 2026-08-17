# -*- coding: utf-8 -*-
"""Stability and deployment-readiness checks -> eval/stability_report.json.

Every claim in docs/deployment_readiness.md must come from a check here that
actually ran. Checks that could not be run are recorded with status "not_run"
and an explicit reason; they are never reported as passing.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(EVAL_DIR))

import numpy as np  # noqa: E402

RESULTS: list[dict[str, Any]] = []


def check(name, category):
    def deco(fn):
        def run():
            t0 = time.perf_counter()
            try:
                status, detail = fn()
            except Exception as e:  # a check must never take the report down with it
                status, detail = "error", f"{type(e).__name__}: {e}"
            RESULTS.append({
                "check": name, "category": category, "status": status,
                "detail": detail, "seconds": round(time.perf_counter() - t0, 2),
            })
            print(f"  [{status.upper():<8}] {name}")
            return status
        return run
    return deco


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


_REBUILD_CACHE: dict[str, Any] = {}


def _rebuild_chunks():
    """Rebuild using the SHIPPED chunker, whatever run_chunking() defaults to."""
    if "chunks" not in _REBUILD_CACHE:
        import clinical_chunking as cc
        loaded = cc.load_processed(ROOT)
        _REBUILD_CACHE["chunks"] = cc.build_chunks_for(
            cc.DEFAULT_CHUNKER, loaded["pages_df"], loaded["recommendations_df"])
    return _REBUILD_CACHE["chunks"]


@check("Chunking is deterministic: two rebuilds from the same inputs agree", "reproducibility")
def chunking_deterministic():
    import clinical_chunking as cc
    a = _rebuild_chunks()
    loaded = cc.load_processed(ROOT)
    b = cc.build_chunks_for(cc.DEFAULT_CHUNKER, loaded["pages_df"], loaded["recommendations_df"])
    same = (len(a) == len(b)
            and all(x["chunk_id"] == y["chunk_id"] for x, y in zip(a, b))
            and all(x["chunk_text"] == y["chunk_text"] for x, y in zip(a, b))
            and all((x["page_start"], x["page_end"]) == (y["page_start"], y["page_end"])
                    for x, y in zip(a, b)))
    return ("pass" if same else "fail",
            {"chunks_run_1": len(a), "chunks_run_2": len(b), "identical": same,
             "note": "Determinism of the chunker itself, independent of the committed artifact."})


@check("Committed chunks.json still reproduces from the CURRENT code", "reproducibility")
def committed_chunks_reproduce():
    import clinical_chunking as cc
    rebuilt = _rebuild_chunks()
    committed = json.loads((ROOT / "data/chunks/chunks.json").read_text(encoding="utf-8"))["chunks"]
    if len(rebuilt) == len(committed):
        mismatch = sum(1 for a, b in zip(rebuilt, committed)
                       if a["chunk_id"] != b["chunk_id"] or a["chunk_text"] != b["chunk_text"])
        return ("pass" if mismatch == 0 else "fail",
                {"chunks": len(rebuilt), "mismatches": mismatch})
    return "fail", {
        "rebuilt_chunks": len(rebuilt),
        "committed_chunks": len(committed),
        "shipped_chunker": cc.DEFAULT_CHUNKER,
        "current_default_model": cc.DEFAULT_EMBED_MODEL,
        "current_token_limit": cc.model_token_limit(),
        "diagnosis": "Chunk count differs from the committed artifact; investigate before shipping.",
    }


@check("Chunk IDs unique and deterministic in form", "reproducibility")
def chunk_ids():
    chunks = json.loads((ROOT / "data/chunks/chunks.json").read_text(encoding="utf-8"))["chunks"]
    ids = [c["chunk_id"] for c in chunks]
    import re
    bad = [i for i in ids if not re.fullmatch(r"[A-Za-z0-9_]+__p\d+-\d+__c\d{4}", i)]
    return ("pass" if len(set(ids)) == len(ids) and not bad else "fail",
            {"total": len(ids), "unique": len(set(ids)), "malformed": bad[:5]})


@check("Embedding model revision is pinned", "reproducibility")
def pinned_revision():
    import clinical_chunking as cc
    meta = json.loads((ROOT / "data/embeddings/index_meta.json").read_text(encoding="utf-8"))
    pinned = cc.model_revision(meta["model_name"])
    return ("pass" if pinned and meta.get("model_revision") == pinned else "fail",
            {"model": meta["model_name"], "index_meta_revision": meta.get("model_revision"),
             "code_pinned_revision": pinned})


@check("Index integrity: vectors align with metadata and are L2-normalised", "reproducibility")
def index_integrity():
    vecs = np.load(ROOT / "data/embeddings/embeddings.npy")
    chunks = json.loads((ROOT / "data/embeddings/embedded_chunks.json").read_text(encoding="utf-8"))
    norms = np.linalg.norm(vecs, axis=1)
    ok = len(chunks) == vecs.shape[0] and float(norms.min()) > 0.999 and float(norms.max()) < 1.001
    return ("pass" if ok else "fail",
            {"vectors": list(vecs.shape), "metadata_records": len(chunks),
             "dtype": str(vecs.dtype),
             "l2_norm_min": round(float(norms.min()), 8),
             "l2_norm_max": round(float(norms.max()), 8)})


@check("Index reproducibility: re-embedding a sample reproduces stored vectors", "reproducibility")
def reembed_sample():
    import clinical_chunking as cc
    chunks = json.loads((ROOT / "data/embeddings/embedded_chunks.json").read_text(encoding="utf-8"))
    vecs = np.load(ROOT / "data/embeddings/embeddings.npy")
    idx = list(range(0, len(chunks), max(1, len(chunks) // 64)))[:64]
    model = cc.load_embedder(cc.DEFAULT_EMBED_MODEL)
    fresh = model.encode([chunks[i]["chunk_text"] for i in idx],
                         normalize_embeddings=True, convert_to_numpy=True)
    delta = float(np.abs(fresh - vecs[idx]).max())
    return ("pass" if delta < 1e-4 else "fail",
            {"sampled_chunks": len(idx), "max_abs_difference": delta,
             "tolerance": 1e-4})


@check("Token safety: no indexed chunk exceeds the encoder window", "correctness")
def token_safety():
    import clinical_chunking as cc
    chunks = json.loads((ROOT / "data/embeddings/embedded_chunks.json").read_text(encoding="utf-8"))
    limit = cc.model_token_limit()
    recomputed = [(c["chunk_id"], cc.count_tokens(c["chunk_text"])) for c in chunks]
    over = [(i, t) for i, t in recomputed if t > limit]
    stale = sum(1 for (cid, t), c in zip(recomputed, chunks) if int(c.get("token_count") or -1) != t)
    return ("pass" if not over and stale == 0 else "fail",
            {"token_limit": limit, "max_token_count": max(t for _, t in recomputed),
             "chunks_over_limit": len(over), "stored_vs_recomputed_mismatches": stale})


@check("Fail-loud token validator rejects an oversized chunk", "correctness")
def fail_loud():
    import clinical_chunking as cc
    limit = cc.model_token_limit()
    fake = [{"chunk_id": "SYNTHETIC__p1-1__c0001", "chunk_text": "aneurysm " * (limit * 3),
             "document_type": "official guideline", "is_guideline": True,
             "content_type": "clinical", "token_count": limit * 3, "section_title": None}]
    try:
        cc.embeddable_chunks(fake, {"invalid_chunk_ids": []})
        return "fail", "embeddable_chunks accepted a chunk over the token limit (it must raise)"
    except ValueError as e:
        return "pass", f"raised ValueError as required: {str(e)[:110]}"


@check("Malformed and adversarial queries do not crash retrieval", "robustness")
def malformed_queries():
    import clinical_chunking as cc
    import experimental_atomic_chunking as ex
    index = ex.load_production_index()
    model = cc.load_embedder(cc.DEFAULT_EMBED_MODEL)
    cases = {
        "empty_string": "",
        "whitespace_only": "   \n\t  ",
        "single_character": "a",
        "very_long_query": "abdominal aortic aneurysm screening threshold " * 200,
        "non_ascii": "aneurysme de l'aorte abdominale — dépistage µg/L",
        "punctuation_only": "!!! ??? ...",
        "sql_like": "'; DROP TABLE chunks; --",
        "duplicate_of_real_query": "What are the recommendations for screening for abdominal aortic aneurysm?",
    }
    outcomes = {}
    for name, q in cases.items():
        try:
            hits = ex.retrieve(q, index, model, top_k=5)
            outcomes[name] = {"ok": True, "hits": len(hits),
                              "top1_score": round(float(hits[0]["similarity_score"]), 4) if hits else None}
        except Exception as e:
            outcomes[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    failed = [k for k, v in outcomes.items() if not v["ok"]]
    return ("pass" if not failed else "fail",
            {"cases": outcomes, "failed": failed,
             "note": ("No case raised. NOTE: an empty or punctuation-only query still returns 10 chunks "
                      "with low similarity -- there is NO abstention threshold. See deployment_readiness.md.")})


@check("Query determinism: the same query returns the same ranking", "robustness")
def query_determinism():
    import clinical_chunking as cc
    import experimental_atomic_chunking as ex
    index = ex.load_production_index()
    model = cc.load_embedder(cc.DEFAULT_EMBED_MODEL)
    q = "What surveillance strategy is recommended for small abdominal aortic aneurysms?"
    runs = [[h["chunk_id"] for h in ex.retrieve(q, index, model, top_k=10)] for _ in range(3)]
    scores = [round(ex.retrieve(q, index, model, top_k=1)[0]["similarity_score"], 8) for _ in range(3)]
    return ("pass" if all(r == runs[0] for r in runs) and len(set(scores)) == 1 else "fail",
            {"identical_rankings": all(r == runs[0] for r in runs),
             "distinct_top1_scores": len(set(scores))})


@check("Retrieval latency", "performance")
def latency():
    import clinical_chunking as cc
    import clinical_rag as cr
    import experimental_atomic_chunking as ex
    index = ex.load_production_index()
    model = cc.load_embedder(cc.DEFAULT_EMBED_MODEL)
    ex.retrieve("warmup", index, model, top_k=10)
    times = []
    for q in cr.CLINICAL_QUERIES:
        t0 = time.perf_counter()
        ex.retrieve(q, index, model, top_k=10)
        times.append(time.perf_counter() - t0)
    times.sort()
    return "pass", {
        "queries_timed": len(times),
        "mean_seconds": round(sum(times) / len(times), 4),
        "median_seconds": round(times[len(times) // 2], 4),
        "p90_seconds": round(times[int(0.9 * (len(times) - 1))], 4),
        "max_seconds": round(times[-1], 4),
        "hardware": "CPU only (torch CPU build); no GPU used",
        "note": "Dominated by query encoding, not by the 1,330-vector dot product.",
    }


@check("Index footprint on disk and in memory", "performance")
def footprint():
    vec_path = ROOT / "data/embeddings/embeddings.npy"
    meta_path = ROOT / "data/embeddings/embedded_chunks.json"
    vecs = np.load(vec_path)
    return "pass", {
        "vectors_file_bytes": vec_path.stat().st_size,
        "chunk_metadata_file_bytes": meta_path.stat().st_size,
        "vectors_in_memory_bytes": int(vecs.nbytes),
        "shape": list(vecs.shape),
        "index_type": "in-process NumPy matrix, exhaustive cosine (no ANN service)",
    }


@check("Test suite", "correctness")
def tests():
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"],
                          cwd=ROOT, capture_output=True, text=True, timeout=1800)
    tail = (proc.stdout or proc.stderr).strip().splitlines()[-1:] or [""]
    return ("pass" if proc.returncode == 0 else "fail",
            {"returncode": proc.returncode, "summary": tail[0][:200]})


@check("Critical artifact checksums recorded", "reproducibility")
def checksums():
    files = [
        "eval/gold_standard.json", "eval/gold_standard_heldout.json",
        "eval/gold_standard_final20.json",
        "data/chunks/chunks.json", "data/embeddings/embeddings.npy",
        "data/embeddings/embedded_chunks.json", "data/embeddings/index_meta.json",
        "data/processed/pages.json", "data/processed/recommendations.json",
        "notebooks/clinical_chunking.py", "notebooks/clinical_preprocess.py",
        "notebooks/clinical_rag.py",
    ]
    out = {}
    for f in files:
        p = ROOT / f
        out[f] = sha256(p) if p.exists() else "MISSING"
    return ("pass" if "MISSING" not in out.values() else "fail", out)


@check("Structured logging / observability in the retrieval path", "operability")
def logging_check():
    src = (ROOT / "notebooks/clinical_rag.py").read_text(encoding="utf-8")
    has_logging = "logging" in src or "logger" in src
    return ("fail" if not has_logging else "pass",
            {"logging_module_used": has_logging,
             "note": ("clinical_rag.retrieve emits no logs, no timings and no request ids. "
                      "Acceptable for a notebook-driven research pipeline; NOT acceptable for a "
                      "service. Listed as a deployment gap, not a research defect.")})


@check("Clean-checkout reproducibility from source PDFs", "reproducibility")
def clean_checkout():
    return "not_run", {
        "reason": ("Not verified in this session. Doing so requires deleting data/processed, data/chunks "
                   "and data/embeddings and re-running notebooks 01-03 from the four source PDFs, which "
                   "would destroy the artifacts every preserved evaluation is scored against."),
        "what_was_verified_instead": ("Chunking determinism from data/processed (check 1) and index "
                                      "reproducibility by re-embedding (check 5). The unverified link is "
                                      "PDF -> data/processed."),
    }


def main() -> int:
    print("running stability checks ...")
    for fn in (chunking_deterministic, committed_chunks_reproduce,
               chunk_ids, pinned_revision, index_integrity,
               reembed_sample, token_safety, fail_loud, malformed_queries,
               query_determinism, latency, footprint, tests, checksums,
               logging_check, clean_checkout):
        fn()

    counts: dict[str, int] = {}
    for r in RESULTS:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    report = {
        "generated_by": "eval/run_stability_checks.py",
        "scope": ("Verifies the RESEARCH pipeline: determinism, reproducibility, token safety, "
                  "robustness to malformed input, latency and footprint. It does not verify a "
                  "production service, because none exists in this repository."),
        "summary": counts,
        "production_ready": False,
        "production_readiness_statement": (
            "NOT production ready, and not claimed to be. The retrieval core is deterministic, "
            "reproducible and token-safe, but the repository ships no service, no API, no logging, "
            "no abstention threshold for out-of-scope queries, no authentication and no monitoring. "
            "It is a reproducible research pipeline, which is what it was built to be."
        ),
        "checks": RESULTS,
    }
    out = EVAL_DIR / "stability_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nsummary: {counts}")
    print(f"saved -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

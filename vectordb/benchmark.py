# -*- coding: utf-8 -*-
"""Latency and footprint measurement for the production Qdrant backend.

    python vectordb/benchmark.py

Measures what the service will actually pay per question:

  * query embedding (MedEmbed, CPU)
  * Qdrant search only
  * end-to-end (embedding + search)
  * the same question set against the local numpy index, for reference
  * collection footprint reported by Qdrant

The 48 frozen questions are used as the query sample. Nothing is scored, tuned
or optimised here; the numbers are recorded so they can be checked later.

Output: eval/qdrant_performance.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (str(ROOT), str(ROOT / "notebooks")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from vectordb.config import load_settings  # noqa: E402
from vectordb.retriever import QdrantRetriever  # noqa: E402
from vectordb.schema import DEFAULT_TOP_K, EXPECTED_MODEL  # noqa: E402
from vectordb.verify_migration import load_queries, local_top_k  # noqa: E402

OUT_PATH = ROOT / "eval" / "qdrant_performance.json"


def _stats(samples: list[float]) -> dict[str, float]:
    arr = np.asarray(samples, dtype=float)
    return {
        "n": int(arr.size),
        "mean_ms": round(float(arr.mean()), 3),
        "median_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
    }


def docker_footprint(container: str) -> dict[str, Any] | None:
    """Container memory and on-disk collection size, when Qdrant runs in Docker.

    Best-effort: returns None if Docker or the container is not there. Nothing
    depends on it, it is recorded because "how much does this cost to run" is a
    fair question to ask of a production component.
    """
    import shutil
    import subprocess

    if not shutil.which("docker"):
        return None

    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    mem = _run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container])
    if mem is None:
        return None
    disk = _run(["docker", "exec", container, "du", "-sk", "/qdrant/storage"])
    disk_kib = None
    if disk:
        head = disk.split()[0]
        disk_kib = int(head) if head.isdigit() else None
    return {
        "container": container,
        "memory_usage": mem,
        "storage_kib": disk_kib,
        "storage_mib": round(disk_kib / 1024, 2) if disk_kib else None,
    }


def collection_footprint(client, collection: str) -> dict[str, Any]:
    info = client.get_collection(collection)
    raw = info.model_dump() if hasattr(info, "model_dump") else {}
    return {
        "points_count": getattr(info, "points_count", None),
        "vectors_count": getattr(info, "vectors_count", None),
        "segments_count": getattr(info, "segments_count", None),
        "status": str(getattr(info, "status", "")),
        "optimizer_status": str(getattr(info, "optimizer_status", "")),
        "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
        "raw_config": raw.get("config"),
    }


def run(
    top_k: int = DEFAULT_TOP_K,
    repeats: int = 3,
    warmup: int = 3,
    container: str | None = None,
) -> dict[str, Any]:
    import retrieval.index as cr
    from ingestion.chunking import load_embedder

    settings = load_settings()
    index = cr.load_index(ROOT)
    model = load_embedder(EXPECTED_MODEL)
    retriever = QdrantRetriever(settings=settings, model=model)

    queries = [q["query"] for q in load_queries()]

    for query in queries[:warmup]:
        retriever.search(query, top_k=top_k)

    embed_ms: list[float] = []
    qdrant_ms: list[float] = []
    end_to_end_ms: list[float] = []
    local_ms: list[float] = []

    for _ in range(repeats):
        for query in queries:
            t0 = time.perf_counter()
            vector = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
            t1 = time.perf_counter()
            retriever.search_vector([float(x) for x in vector], top_k=top_k)
            t2 = time.perf_counter()
            local_top_k(vector, index, top_k)
            t3 = time.perf_counter()

            embed_ms.append((t1 - t0) * 1000)
            qdrant_ms.append((t2 - t1) * 1000)
            end_to_end_ms.append((t2 - t0) * 1000)
            local_ms.append((t3 - t2) * 1000)

    vectors_bytes = int(np.asarray(index["vectors"]).nbytes)
    return {
        "artifact": "qdrant_performance",
        "configuration": {
            "qdrant": settings.describe(),
            "top_k": top_k,
            "queries": len(queries),
            "repeats": repeats,
            "warmup_queries": warmup,
            "embedding_model": EXPECTED_MODEL,
            "device": "cpu",
        },
        "latency": {
            "query_embedding": _stats(embed_ms),
            "qdrant_search": _stats(qdrant_ms),
            "end_to_end_embedding_plus_search": _stats(end_to_end_ms),
            "local_numpy_search_reference": _stats(local_ms),
        },
        "footprint": {
            "collection": collection_footprint(retriever.client, settings.collection),
            "raw_vector_bytes": vectors_bytes,
            "raw_vector_mib": round(vectors_bytes / 1024 / 1024, 3),
            "docker": docker_footprint(container) if container else None,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure Qdrant retrieval latency and footprint.")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--docker-container",
        default="aaa-clinical-qdrant",
        help="container to measure memory/disk for; empty string to skip",
    )
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    report = run(top_k=args.top_k, repeats=args.repeats, container=args.docker_container or None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lat = report["latency"]
    for name, stats in lat.items():
        print(f"{name:>34}: mean {stats['mean_ms']:>8.2f} ms   p95 {stats['p95_ms']:>8.2f} ms   n={stats['n']}")
    print(f"points in collection: {report['footprint']['collection']['points_count']}")
    if report["footprint"]["docker"]:
        d = report["footprint"]["docker"]
        print(f"container memory    : {d['memory_usage']}   storage: {d['storage_mib']} MiB")
    print(f"saved -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

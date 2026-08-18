# -*- coding: utf-8 -*-
"""Deterministic migration of the validated local index into Qdrant.

    python vectordb/ingest.py --recreate

This is a *copy*, not a rebuild. It reads the artifacts that the frozen
evaluation was run against --

    data/embeddings/embedded_chunks.json   991 indexed chunk records
    data/embeddings/embeddings.npy         991 x 768 float32, L2-normalised
    data/embeddings/ids.json               the ordered chunk_id list
    data/embeddings/index_meta.json        model, revision, dim, token limit, digests

-- loads them through `retrieval.index.load_index`, so the binding between chunk
IDs and vectors is verified before anything is written, validates them against
the migration contract in `vectordb/schema.py`, and upserts them as Qdrant
points. No text is re-chunked and no vector is re-embedded, so the production
store holds bit-identical vectors to the ones the reported metrics were produced
from.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vectordb.config import QdrantSettings, load_settings, make_client  # noqa: E402
from vectordb.schema import (  # noqa: E402
    COLLECTION_DISTANCE,
    EXPECTED_DIM,
    IngestValidationError,
    build_payload,
    point_id_for,
    validate_index_bundle,
)

INDEX_DIR = ROOT / "data" / "embeddings"
REPORT_PATH = ROOT / "eval" / "qdrant_ingestion_report.json"
BATCH_SIZE = 128


def load_local_index(index_dir: Path = INDEX_DIR) -> dict[str, Any]:
    """Load the local index through `retrieval.index.load_index`.

    Going through the project's own loader rather than re-reading the files here
    is deliberate: `load_index` applies `retrieval.index.verify_index_binding`, so
    the ingestion path -- the only path that WRITES to production -- refuses to
    run against an index whose chunk_id <-> vector binding no longer holds.

    That matters because the join is positional on both sides: row *i* of
    embeddings.npy is record *i* of embedded_chunks.json, and `upsert_points`
    pairs them by index. A length-preserving change upstream (a re-sort, a
    hand-edit, a re-chunk landing on the same count) would otherwise be copied
    into Qdrant intact, mis-attributing every citation while the scores still
    looked plausible.
    """
    import retrieval.index as cr

    for name in ("index_meta.json", "embedded_chunks.json", "embeddings.npy", cr.IDS_FILENAME):
        if not (index_dir / name).exists():
            raise FileNotFoundError(f"missing local index artifact: {index_dir / name}")

    # `load_index` resolves <project_root>/data/embeddings itself, so derive that
    # root from index_dir instead of assuming ROOT -- this keeps the index_dir
    # parameter meaningful. Refuse rather than guess if the two disagree.
    resolved = index_dir.resolve()
    project_root = resolved.parents[1]
    if (project_root / "data" / "embeddings").resolve() != resolved:
        raise ValueError(
            f"{index_dir} is not a <project_root>/data/embeddings directory, so the "
            f"chunk_id <-> vector binding cannot be resolved against its chunk set. "
            f"Ingestion refuses to proceed unverified."
        )

    index = cr.load_index(project_root)
    return {
        "meta": index["meta"],
        "records": index["chunks"],
        "vectors": index["vectors"],
        "binding": index["binding"],
        "dir": index_dir,
    }


def ensure_collection(client, collection: str, *, recreate: bool) -> str:
    """Create the collection, or refuse to touch an existing one by default."""
    from qdrant_client import models

    exists = client.collection_exists(collection)
    if exists and not recreate:
        info = client.get_collection(collection)
        raise RuntimeError(
            f"collection {collection!r} already exists with {info.points_count} point(s). "
            "Re-run with --recreate to drop and rebuild it, or set QDRANT_COLLECTION "
            "to a different name."
        )
    if exists:
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=EXPECTED_DIM,
            distance=models.Distance.COSINE,
        ),
    )
    return "recreated" if exists else "created"


def upsert_points(client, collection: str, records: list[dict[str, Any]], vectors, batch_size: int = BATCH_SIZE) -> int:
    from qdrant_client import models

    total = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        points = [
            models.PointStruct(
                id=point_id_for(rec["chunk_id"]),
                vector=[float(x) for x in vectors[start + offset]],
                payload=build_payload(rec),
            )
            for offset, rec in enumerate(batch)
        ]
        client.upsert(collection_name=collection, points=points, wait=True)
        total += len(points)
    return total


def verify_collection(client, collection: str, expected_points: int) -> dict[str, Any]:
    """Post-ingestion checks against the live collection, not against intent."""
    info = client.get_collection(collection)
    params = info.config.params.vectors
    size = getattr(params, "size", None)
    distance = getattr(params, "distance", None)
    distance = getattr(distance, "value", distance)
    counted = client.count(collection_name=collection, exact=True).count

    problems = []
    if int(size or 0) != EXPECTED_DIM:
        problems.append(f"collection vector size is {size}, expected {EXPECTED_DIM}")
    if str(distance).lower() != COLLECTION_DISTANCE.lower():
        problems.append(f"collection distance is {distance}, expected {COLLECTION_DISTANCE}")
    if counted != expected_points:
        problems.append(f"collection holds {counted} point(s), expected {expected_points}")
    if problems:
        raise RuntimeError("post-ingestion verification failed:\n  - " + "\n  - ".join(problems))

    return {
        "points_count": counted,
        "vector_size": int(size),
        "distance": str(distance),
        "status": str(getattr(info, "status", "")),
    }


def run(settings: QdrantSettings | None = None, *, recreate: bool = False, index_dir: Path = INDEX_DIR) -> dict[str, Any]:
    settings = settings or load_settings()
    bundle = load_local_index(index_dir)
    summary = validate_index_bundle(bundle["records"], bundle["vectors"], bundle["meta"])

    client = make_client(settings)
    t0 = time.perf_counter()
    action = ensure_collection(client, settings.collection, recreate=recreate)
    uploaded = upsert_points(client, settings.collection, bundle["records"], bundle["vectors"])
    elapsed = time.perf_counter() - t0
    collection = verify_collection(client, settings.collection, len(bundle["records"]))

    return {
        "connection": settings.describe(),
        "source_index": {
            "dir": index_dir.relative_to(ROOT).as_posix(),
            "chunks_file": "embedded_chunks.json",
            "vectors_file": "embeddings.npy",
            "binding_verified_by": "retrieval.index.verify_index_binding (via retrieval.index.load_index)",
            "binding": bundle.get("binding"),
            **summary,
        },
        "collection_action": action,
        "points_uploaded": uploaded,
        "collection": collection,
        "ingestion_seconds": round(elapsed, 3),
        "points_per_second": round(uploaded / elapsed, 1) if elapsed > 0 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Migrate the local V1 index into Qdrant.")
    ap.add_argument("--recreate", action="store_true", help="drop and rebuild the collection if it exists")
    ap.add_argument("--report", default=str(REPORT_PATH), help="where to write the ingestion report")
    args = ap.parse_args()

    try:
        result = run(recreate=args.recreate)
    except (IngestValidationError, RuntimeError, FileNotFoundError) as exc:
        print(f"INGESTION FAILED\n{exc}", file=sys.stderr)
        return 1

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"collection : {result['connection']['collection']} ({result['collection_action']})")
    print(f"points     : {result['collection']['points_count']}")
    print(f"vector size: {result['collection']['vector_size']}  distance: {result['collection']['distance']}")
    print(f"ingestion  : {result['ingestion_seconds']}s ({result['points_per_second']} points/s)")
    print(f"report     -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

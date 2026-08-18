# -*- coding: utf-8 -*-
"""Collection schema, deterministic point IDs, payload contract and validation.

The rule this module enforces: a Qdrant point must carry *exactly* the identity
and metadata of the local index record it came from. Anything that cannot be
reconstructed from the payload would make Qdrant a lossy copy of the validated
index, so ingestion fails loudly rather than storing a partial record.
"""
from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]

# --- frozen properties of the validated V1 index ---------------------------
EXPECTED_MODEL = "abhinand/MedEmbed-base-v0.1"
EXPECTED_REVISION = "7a90c50263f620dff743eb9794b89a42bfc5d765"
EXPECTED_DIM = 768
EXPECTED_N_VECTORS = 991
EXPECTED_TOKEN_LIMIT = 512
COLLECTION_DISTANCE = "Cosine"
DEFAULT_TOP_K = 10

# Vectors are written L2-normalised by `retrieval.index.build_embeddings`
# (`normalize_embeddings=True`). Qdrant re-normalises for Cosine anyway; the
# check exists to detect an index that was NOT built under the validated
# configuration, not to fix one.
NORM_TOLERANCE = 1e-3

# Fixed namespace for uuid5 point IDs. Chunk IDs such as
# "ESVS_2024__p1-1__c0001" are not valid Qdrant point IDs (which must be an
# unsigned integer or a UUID), so the string is mapped deterministically:
#   point_id = uuid5(POINT_ID_NAMESPACE, chunk_id)
# Same chunk_id -> same point ID, on every machine, forever. The original
# chunk_id is preserved verbatim in the payload.
POINT_ID_NAMESPACE = uuid.UUID("6f1f2b9a-6a6b-5c4d-9e3f-aa0c11e01a01")

# Every field carried from `data/embeddings/embedded_chunks.json` into the
# payload. This is the full record, so evidence can be reconstructed from
# Qdrant alone.
PAYLOAD_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "document_id",
    "document_name",
    "document_type",
    "is_guideline",
    "section_title",
    "section_source",
    "content_type",
    "token_count",
    "char_count",
    "page_number",
    "page_start",
    "page_end",
    "source_file",
    "chunk_text",
    "source_excerpt",
    "recommendation_id",
    "recommendation_grade",
    "evidence_level",
)

# Fields that must be present AND non-null on every point. `section_title`,
# `recommendation_id`, `recommendation_grade`, `evidence_level` and
# `source_excerpt` are legitimately null for some chunks, so they are required
# to be *present*, not populated.
REQUIRED_NON_NULL: tuple[str, ...] = (
    "chunk_id",
    "document_id",
    "document_name",
    "document_type",
    "content_type",
    "token_count",
    "page_number",
    "page_start",
    "page_end",
    "source_file",
    "chunk_text",
)

NULLABLE_FIELDS: tuple[str, ...] = tuple(f for f in PAYLOAD_FIELDS if f not in REQUIRED_NON_NULL)

# `data/embeddings/embedded_chunks.json` stores some optional fields as the
# JSON literal `NaN` rather than `null` -- pandas missing values that survived
# serialisation when the frozen index was written (409 occurrences across
# recommendation_id, recommendation_grade and evidence_level). NaN and null
# both mean "this chunk carries no such value", and Qdrant stores NaN as null.
#
# The migration therefore maps NaN -> null in NULLABLE fields, and *counts and
# reports* every occurrence (`nan_summary`) so the conversion is visible in the
# ingestion report rather than silent. NaN in a REQUIRED field, or in a vector,
# is still a hard failure: the local index would be truly malformed.


class IngestValidationError(ValueError):
    """Raised when the local artifacts do not satisfy the migration contract."""


def point_id_for(chunk_id: str) -> str:
    """Deterministic Qdrant point ID for a chunk ID. Never random."""
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise IngestValidationError(f"chunk_id must be a non-empty string, got {chunk_id!r}")
    return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk_id))


def is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def nan_summary(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Count NaN-valued payload fields, per field, across the whole index."""
    counts: dict[str, int] = {}
    for rec in records:
        for field in PAYLOAD_FIELDS:
            if is_nan(rec.get(field)):
                counts[field] = counts.get(field, 0) + 1
    return dict(sorted(counts.items()))


def build_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Project one indexed chunk record onto the payload contract."""
    missing = [f for f in PAYLOAD_FIELDS if f not in record]
    if missing:
        raise IngestValidationError(
            f"chunk {record.get('chunk_id')!r} is missing payload field(s): {missing}"
        )
    empty = [f for f in REQUIRED_NON_NULL if record.get(f) is None or is_nan(record.get(f))]
    if empty:
        raise IngestValidationError(
            f"chunk {record.get('chunk_id')!r} has null/NaN required field(s): {empty}"
        )
    return {field: (None if is_nan(record[field]) else record[field]) for field in PAYLOAD_FIELDS}


def _vector_problems(vectors, dim: int) -> list[str]:
    import numpy as np

    problems: list[str] = []
    arr = np.asarray(vectors)
    if arr.ndim != 2:
        return [f"vectors must be 2-D, got shape {arr.shape}"]
    if arr.shape[1] != dim:
        problems.append(f"vector dimension is {arr.shape[1]}, expected {dim}")
    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).any(axis=1).sum())
        problems.append(f"{bad} vector(s) contain NaN or Inf")
    else:
        norms = np.linalg.norm(arr, axis=1)
        off = int((np.abs(norms - 1.0) > NORM_TOLERANCE).sum())
        if off:
            problems.append(
                f"{off} vector(s) are not L2-normalised within {NORM_TOLERANCE} "
                f"(min={norms.min():.6f}, max={norms.max():.6f})"
            )
    return problems


def validate_index_bundle(
    records: Sequence[dict[str, Any]],
    vectors,
    meta: dict[str, Any],
    *,
    expected_n: int | None = EXPECTED_N_VECTORS,
    strict_model: bool = True,
) -> dict[str, Any]:
    """Validate the local artifacts before anything is written to Qdrant.

    Raises `IngestValidationError` listing every problem found. Nothing is
    repaired, coerced or dropped: bad data must be fixed upstream, in the
    pipeline that produced it.
    """
    problems: list[str] = []

    dim = int(meta.get("embedding_dim") or 0)
    if dim != EXPECTED_DIM:
        problems.append(f"index_meta embedding_dim is {dim}, expected {EXPECTED_DIM}")
    if strict_model:
        if meta.get("model_name") != EXPECTED_MODEL:
            problems.append(f"index_meta model_name is {meta.get('model_name')!r}, expected {EXPECTED_MODEL!r}")
        if meta.get("model_revision") != EXPECTED_REVISION:
            problems.append(
                f"index_meta model_revision is {meta.get('model_revision')!r}, expected {EXPECTED_REVISION!r}"
            )
        if (meta.get("metric") or "").lower() != "cosine":
            problems.append(f"index_meta metric is {meta.get('metric')!r}, expected 'cosine'")

    problems.extend(_vector_problems(vectors, EXPECTED_DIM))

    n_vec = len(vectors)
    if len(records) != n_vec:
        problems.append(f"vector count {n_vec} != chunk count {len(records)}")
    if expected_n is not None and len(records) != expected_n:
        problems.append(f"chunk count {len(records)} != expected {expected_n}")
    if meta.get("n_vectors") is not None and int(meta["n_vectors"]) != n_vec:
        problems.append(f"index_meta n_vectors {meta['n_vectors']} != {n_vec} vectors on disk")

    ids: list[str] = []
    for i, rec in enumerate(records):
        cid = rec.get("chunk_id")
        if not isinstance(cid, str) or not cid.strip():
            problems.append(f"record {i} has missing/invalid chunk_id: {cid!r}")
            continue
        ids.append(cid)
        for field in PAYLOAD_FIELDS:
            if field not in rec:
                problems.append(f"{cid}: missing payload field {field!r}")
        for field in REQUIRED_NON_NULL:
            if rec.get(field) is None or is_nan(rec.get(field)):
                problems.append(f"{cid}: null/NaN required field {field!r}")

    dupes = sorted({c for c in ids if ids.count(c) > 1}) if len(set(ids)) != len(ids) else []
    if dupes:
        problems.append(f"duplicate chunk_id(s): {dupes[:5]}")

    point_ids = [point_id_for(c) for c in ids]
    if len(set(point_ids)) != len(point_ids):
        problems.append("deterministic point ID collision -- two chunk_ids map to one UUID")

    token_limit = int(meta.get("token_limit") or EXPECTED_TOKEN_LIMIT)
    over = [
        (rec.get("chunk_id"), rec.get("token_count"))
        for rec in records
        if isinstance(rec.get("token_count"), (int, float))
        and not math.isnan(float(rec["token_count"]))
        and int(rec["token_count"]) > token_limit
    ]
    if over:
        problems.append(f"{len(over)} chunk(s) exceed the {token_limit}-token window: {over[:5]}")

    if problems:
        raise IngestValidationError(
            "local index does not satisfy the Qdrant migration contract:\n  - "
            + "\n  - ".join(problems)
        )

    return {
        "n_records": len(records),
        "n_vectors": n_vec,
        "embedding_dim": EXPECTED_DIM,
        "token_limit": token_limit,
        "nan_in_nullable_fields_normalised_to_null": nan_summary(records),
        "model_name": meta.get("model_name"),
        "model_revision": meta.get("model_revision"),
        "distance": COLLECTION_DISTANCE,
        "unique_chunk_ids": len(set(ids)),
        "unique_point_ids": len(set(point_ids)),
    }


def iter_points(records: Iterable[dict[str, Any]], vectors) -> Iterable[tuple[str, list[float], dict[str, Any]]]:
    """Yield `(point_id, vector, payload)` in the order stored on disk."""
    for i, rec in enumerate(records):
        yield point_id_for(rec["chunk_id"]), [float(x) for x in vectors[i]], build_payload(rec)

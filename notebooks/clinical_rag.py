# -*- coding: utf-8 -*-
"""Local embeddings, vector index, and retrieval for AAA clinical chunks."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from clinical_chunking import (
    DEFAULT_EMBED_MODEL,
    count_tokens,
    embeddable_chunks,
    load_embedder,
    model_revision,
    model_token_limit,
)
from clinical_preprocess import find_project_root, relative_posix

DEFAULT_MODEL = DEFAULT_EMBED_MODEL
INDEX_DIRNAME = "embeddings"
IDS_FILENAME = "ids.json"
CHUNKS_RELPATH = ("data", "chunks", "chunks.json")


# ---------------------------------------------------------------------------
# Index <-> chunk-set binding
#
# `retrieve` joins the matrix to its metadata BY POSITION: row i of
# embeddings.npy is record i of embedded_chunks.json. A length check cannot
# detect a length-preserving change -- a re-sort, a hand-edit, or a re-chunk that
# happens to land on the same count would pass it and mis-attribute every
# citation while the scores still looked plausible.
#
# So the build records what it was built from and the load refuses to proceed
# unless that record still holds: the ordered chunk_id list is persisted beside
# the vectors, and both it and the source chunks.json are digested into
# index_meta.json.
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chunk_ids_digest(chunk_ids: list[str]) -> str:
    """SHA-256 over the ordered chunk_id list.

    Order is part of the identity, not incidental to it: the vectors are
    row-aligned to this list, so a permutation that preserves the set must still
    fail verification. Each id is newline-terminated rather than concatenated, so
    no two adjacent ids can be rearranged into the same byte string.
    """
    h = hashlib.sha256()
    for cid in chunk_ids:
        h.update(str(cid).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def verify_index_binding(
    project_root: Path,
    meta: dict[str, Any],
    chunks: list[dict[str, Any]],
    vectors: np.ndarray | None = None,
) -> dict[str, Any]:
    """Assert this index really is the index built from the chunk set on disk.

    Raises RuntimeError on any mismatch, and returns the digests it verified so
    callers can report them. Anything that loads the index by reading the files
    directly should call this rather than reimplement the checks.
    """
    out_dir = project_root / "data" / INDEX_DIRNAME
    indexed_ids = [c["chunk_id"] for c in chunks]
    rebuild = "re-run eval/rebuild_shipped_index.py"

    if vectors is not None and len(indexed_ids) != len(vectors):
        raise RuntimeError(
            f"Index metadata and vectors are out of sync: {len(vectors)} vectors vs "
            f"{len(indexed_ids)} metadata records."
        )

    # 1. the ordered id list persisted beside the vectors (element-wise, not by length)
    ids_path = out_dir / IDS_FILENAME
    if not ids_path.exists():
        raise RuntimeError(
            f"Missing {ids_path}: the index does not carry the chunk_id list its vectors "
            f"are aligned to, so the positional join cannot be checked. {rebuild}."
        )
    stored_ids = json.loads(ids_path.read_text(encoding="utf-8"))["chunk_ids"]
    if stored_ids != indexed_ids:
        first = next(
            (i for i, (a, b) in enumerate(zip(stored_ids, indexed_ids)) if a != b),
            min(len(stored_ids), len(indexed_ids)),
        )
        raise RuntimeError(
            f"{IDS_FILENAME} does not match {meta.get('chunks_file')} element-wise "
            f"({len(stored_ids)} vs {len(indexed_ids)} ids; first difference at position "
            f"{first}). The vector<->chunk join is positional, so loading this index would "
            f"mis-attribute retrieved chunks. {rebuild}."
        )

    # 2. the digest of that list, stamped into index_meta.json at build time
    ids_sha = chunk_ids_digest(indexed_ids)
    recorded_ids_sha = meta.get("indexed_chunk_ids_sha256")
    if not recorded_ids_sha:
        raise RuntimeError(
            f"index_meta.json records no 'indexed_chunk_ids_sha256', so the index does not "
            f"identify the vectors it holds. {rebuild}."
        )
    if recorded_ids_sha != ids_sha:
        raise RuntimeError(
            f"Indexed chunk_id list does not match the digest stamped at build time "
            f"(now {ids_sha[:16]}..., built as {recorded_ids_sha[:16]}...). {rebuild}."
        )

    # 3. the chunk set the index was built from
    recorded_source_sha = meta.get("source_chunks_sha256")
    if not recorded_source_sha:
        raise RuntimeError(
            f"index_meta.json records no 'source_chunks_sha256', so nothing proves this "
            f"index was built from data/chunks/chunks.json. {rebuild}."
        )
    chunks_path = project_root.joinpath(*CHUNKS_RELPATH)
    if not chunks_path.exists():
        raise RuntimeError(
            f"Missing {chunks_path}: cannot verify the index against the chunk set it "
            f"records having been built from."
        )
    source_sha = file_sha256(chunks_path)
    if source_sha != recorded_source_sha:
        raise RuntimeError(
            f"data/chunks/chunks.json has changed since this index was built "
            f"(now {source_sha[:16]}..., built from {recorded_source_sha[:16]}...). The "
            f"index is stale relative to its own chunk set: {rebuild}."
        )

    return {
        "n_indexed": len(indexed_ids),
        "source_chunks_sha256": source_sha,
        "indexed_chunk_ids_sha256": ids_sha,
    }


def load_chunks(project_root: Path | None = None) -> dict[str, Any]:
    project_root = project_root or find_project_root()
    path = project_root / "data" / "chunks" / "chunks.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run notebook 02 chunking first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "project_root": project_root,
        "chunks": payload.get("chunks") or [],
        "quality": payload.get("quality") or {},
        "path": path,
    }


def build_embeddings(
    chunks: list[dict[str, Any]],
    quality: dict[str, Any],
    model_name: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    valid = embeddable_chunks(chunks, quality)
    if not valid:
        raise RuntimeError("No valid chunks available for embedding.")
    model = load_embedder(model_name)
    texts = [c["chunk_text"] for c in valid]

    # Guard against silent truncation. SentenceTransformer.encode() quietly drops
    # everything past max_seq_length, so a chunk that overflows would be indexed
    # by a vector that does not represent its own stored text.
    limit = min(int(model.max_seq_length), model_token_limit(model_name))
    over = [
        (c["chunk_id"], count_tokens(t, model_name))
        for c, t in zip(valid, texts)
        if count_tokens(t, model_name) > limit
    ]
    if over:
        raise ValueError(
            f"{len(over)} chunk(s) exceed the {limit}-token window of {model_name} "
            f"and would be silently truncated: {over[:5]}"
        )

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    failed = []
    if vectors.shape[0] != len(valid):
        failed.append("embedding count does not match chunk count")
    return {
        "model_name": model_name,
        "embedding_dim": int(vectors.shape[1]),
        "vectors": np.asarray(vectors, dtype=np.float32),
        "chunks": valid,
        "failed": failed,
        "total_embedded": int(vectors.shape[0]),
    }


def save_index(bundle: dict[str, Any], project_root: Path | None = None) -> dict[str, str]:
    project_root = project_root or find_project_root()
    out_dir = project_root / "data" / INDEX_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = out_dir / "embeddings.npy"
    meta_path = out_dir / "index_meta.json"
    chunks_path = out_dir / "embedded_chunks.json"
    ids_path = out_dir / IDS_FILENAME
    source_chunks_path = project_root.joinpath(*CHUNKS_RELPATH)
    if not source_chunks_path.exists():
        raise FileNotFoundError(
            f"Missing {source_chunks_path}: the index must record a digest of the chunk set "
            f"it was built from, so it cannot be written without one."
        )
    np.save(vectors_path, bundle["vectors"])
    records = []
    for chunk in bundle["chunks"]:
        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "document_name": chunk["document_name"],
                "document_type": chunk["document_type"],
                "is_guideline": chunk.get("is_guideline"),
                "section_title": chunk.get("section_title"),
                "section_source": chunk.get("section_source"),
                "content_type": chunk.get("content_type"),
                "token_count": chunk.get("token_count"),
                "char_count": chunk.get("char_count"),
                "page_number": chunk.get("page_number"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "source_file": chunk.get("source_file"),
                "chunk_text": chunk.get("chunk_text"),
                "source_excerpt": chunk.get("source_excerpt"),
                "recommendation_id": chunk.get("recommendation_id"),
                "recommendation_grade": chunk.get("recommendation_grade"),
                "evidence_level": chunk.get("evidence_level"),
            }
        )
    chunks_path.write_text(json.dumps(records, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    indexed_ids = [c["chunk_id"] for c in bundle["chunks"]]
    ids_path.write_text(
        json.dumps(
            {
                "note": (
                    "Ordered chunk_id list, row-aligned to embeddings.npy. Retrieval joins "
                    "vectors to metadata by position, so this list is what makes the join "
                    "checkable: clinical_rag.load_index refuses to load unless it matches "
                    "embedded_chunks.json element-wise."
                ),
                "vectors_file": "embeddings.npy",
                "chunks_file": "embedded_chunks.json",
                "n_ids": len(indexed_ids),
                "chunk_ids_sha256": chunk_ids_digest(indexed_ids),
                "chunk_ids": indexed_ids,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    meta = {
        "index_type": "numpy_cosine",
        "metric": "cosine",
        "model_name": bundle["model_name"],
        "model_revision": model_revision(bundle["model_name"]),
        "embedding_dim": bundle["embedding_dim"],
        "token_limit": model_token_limit(bundle["model_name"]),
        "max_chunk_tokens": max((int(c.get("token_count") or 0) for c in bundle["chunks"]), default=0),
        "n_vectors": bundle["total_embedded"],
        "failed_embeddings": bundle["failed"],
        "vectors_file": "embeddings.npy",
        "chunks_file": "embedded_chunks.json",
        "ids_file": IDS_FILENAME,
        # What this index was built FROM, by content rather than by count. A count
        # match is not identity: two different chunk sets of the same size satisfy it.
        "source_chunks_file": "data/chunks/chunks.json",
        "source_chunks_sha256": file_sha256(source_chunks_path),
        "indexed_chunk_ids_sha256": chunk_ids_digest(indexed_ids),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "dir": relative_posix(out_dir, project_root),
        "vectors": relative_posix(vectors_path, project_root),
        "meta": relative_posix(meta_path, project_root),
        "chunks": relative_posix(chunks_path, project_root),
        "ids": relative_posix(ids_path, project_root),
    }


def load_index(project_root: Path | None = None) -> dict[str, Any]:
    project_root = project_root or find_project_root()
    out_dir = project_root / "data" / INDEX_DIRNAME
    meta = json.loads((out_dir / "index_meta.json").read_text(encoding="utf-8"))
    vectors = np.load(out_dir / "embeddings.npy")
    chunks = json.loads((out_dir / "embedded_chunks.json").read_text(encoding="utf-8"))
    binding = verify_index_binding(project_root, meta, chunks, vectors)
    return {
        "project_root": project_root,
        "meta": meta,
        "vectors": vectors,
        "chunks": chunks,
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "binding": binding,
        "model_name": meta["model_name"],
    }


def retrieve(
    query: str,
    index: dict[str, Any],
    model=None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if model is None:
        model = load_embedder(index["model_name"])
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    scores = index["vectors"] @ q
    order = np.argsort(-scores)[:top_k]
    hits = []
    for rank, idx in enumerate(order, start=1):
        chunk = dict(index["chunks"][int(idx)])
        hits.append(
            {
                "rank": rank,
                "similarity_score": float(scores[int(idx)]),
                "chunk_id": chunk.get("chunk_id"),
                "document": chunk.get("document_name"),
                "document_id": chunk.get("document_id"),
                "document_type": chunk.get("document_type"),
                "is_guideline": chunk.get("is_guideline"),
                "section": chunk.get("section_title"),
                "content_type": chunk.get("content_type"),
                "token_count": chunk.get("token_count"),
                "page": chunk.get("page_number"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "source_file": chunk.get("source_file"),
                "source_excerpt": chunk.get("source_excerpt"),
                "chunk_text": chunk.get("chunk_text"),
                "recommendation_id": chunk.get("recommendation_id"),
                "recommendation_grade": chunk.get("recommendation_grade"),
                "evidence_level": chunk.get("evidence_level"),
            }
        )
    return hits


CLINICAL_QUERIES = [
    "What are the recommendations for screening for abdominal aortic aneurysm?",
    "What AAA diameter is generally associated with consideration of elective repair?",
    "What surveillance strategy is recommended for small abdominal aortic aneurysms?",
    "What are the indications for endovascular aneurysm repair?",
    "What are the risk factors associated with abdominal aortic aneurysm?",
    "What imaging modality is recommended for diagnosis or surveillance of AAA?",
    "What factors influence the risk of AAA rupture?",
    "What are the recommendations for smoking cessation in patients with AAA?",
    "What are the recommendations for women regarding AAA screening?",
    "What are the differences between open surgical repair and EVAR recommendations?",
]


def run_embedding_index(project_root: Path | None = None, model_name: str = DEFAULT_MODEL) -> dict[str, Any]:
    loaded = load_chunks(project_root)
    bundle = build_embeddings(loaded["chunks"], loaded["quality"], model_name=model_name)
    paths = save_index(bundle, loaded["project_root"])
    return {
        "project_root": loaded["project_root"],
        "model_name": bundle["model_name"],
        "total_embedded": bundle["total_embedded"],
        "failed": bundle["failed"],
        "embedding_dim": bundle["embedding_dim"],
        "paths": paths,
    }


def format_hit(hit: dict[str, Any], preview_chars: int = 700) -> str:
    text = hit.get("chunk_text") or ""
    preview = text if len(text) <= preview_chars else text[:preview_chars] + " ..."
    rec_bits = []
    if hit.get("recommendation_id"):
        rec_bits.append(f"recommendation_id={hit['recommendation_id']}")
    if hit.get("recommendation_grade"):
        rec_bits.append(f"grade={hit['recommendation_grade']}")
    if hit.get("evidence_level"):
        rec_bits.append(f"evidence_level={hit['evidence_level']}")
    rec_line = (", ".join(rec_bits)) if rec_bits else "none in this chunk"
    return (
        f"rank {hit['rank']}  score={hit['similarity_score']:.4f}\n"
        f"chunk_id: {hit['chunk_id']}\n"
        f"document: {hit['document']} ({hit['document_id']})\n"
        f"section: {hit['section']}\n"
        f"page: {hit['page']} (pages {hit['page_start']}-{hit['page_end']})\n"
        f"source_file: {hit['source_file']}\n"
        f"recommendation metadata: {rec_line}\n"
        f"source excerpt: {hit.get('source_excerpt')}\n"
        f"chunk text:\n{preview}\n"
    )

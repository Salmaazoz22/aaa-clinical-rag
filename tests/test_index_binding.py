# -*- coding: utf-8 -*-
"""Tests for the index <-> chunk-set binding.

`retrieve` joins the vector matrix to its metadata by POSITION, so the guard that
matters is not "are the lengths equal" but "is this the same ordered chunk set the
matrix was built from". These tests check the shipped artifacts really are bound,
and -- on a synthetic index, so nothing shipped is touched -- that the loader
actually refuses the length-preserving corruptions a count check would wave through.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import retrieval.index as cr  # noqa: E402

INDEX_DIR = ROOT / "data" / "embeddings"
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.json"


def _needs_shipped_index():
    for p in (CHUNKS_PATH, INDEX_DIR / "index_meta.json", INDEX_DIR / "embedded_chunks.json",
              INDEX_DIR / "embeddings.npy"):
        if not p.exists():
            pytest.skip(f"{p.name} not built; run eval/scripts/rebuild_shipped_index.py first")


# --------------------------------------------------------------------------
# The digest itself
# --------------------------------------------------------------------------

def test_chunk_ids_digest_is_order_sensitive():
    """A re-sort preserves the set, so set-equality is not enough."""
    assert cr.chunk_ids_digest(["a", "b"]) != cr.chunk_ids_digest(["b", "a"])


def test_chunk_ids_digest_cannot_be_forged_by_rearranging_boundaries():
    """Ids are delimited, so ['ab'] and ['a','b'] must not collide."""
    assert cr.chunk_ids_digest(["ab"]) != cr.chunk_ids_digest(["a", "b"])


# --------------------------------------------------------------------------
# The shipped artifacts
# --------------------------------------------------------------------------

def test_shipped_index_records_the_chunk_set_it_was_built_from():
    _needs_shipped_index()
    meta = json.loads((INDEX_DIR / "index_meta.json").read_text(encoding="utf-8"))
    assert meta.get("source_chunks_sha256") == cr.file_sha256(CHUNKS_PATH), (
        "index_meta.json's source digest does not match data/chunks/chunks.json: "
        "the index is stale relative to its own chunk set"
    )


def test_shipped_ids_file_matches_embedded_chunks_element_wise():
    _needs_shipped_index()
    ids = json.loads((INDEX_DIR / cr.IDS_FILENAME).read_text(encoding="utf-8"))["chunk_ids"]
    indexed = json.loads((INDEX_DIR / "embedded_chunks.json").read_text(encoding="utf-8"))
    assert ids == [c["chunk_id"] for c in indexed]


def test_shipped_id_digest_matches_the_stamped_value():
    _needs_shipped_index()
    meta = json.loads((INDEX_DIR / "index_meta.json").read_text(encoding="utf-8"))
    indexed = json.loads((INDEX_DIR / "embedded_chunks.json").read_text(encoding="utf-8"))
    assert meta.get("indexed_chunk_ids_sha256") == cr.chunk_ids_digest(
        [c["chunk_id"] for c in indexed]
    )


def test_load_index_accepts_the_shipped_artifacts_and_reports_the_binding():
    _needs_shipped_index()
    index = cr.load_index(ROOT)
    assert index["binding"]["n_indexed"] == len(index["vectors"]) == len(index["chunks"])
    assert index["chunk_ids"] == [c["chunk_id"] for c in index["chunks"]]


# --------------------------------------------------------------------------
# The guard must be able to FAIL -- exercised on a synthetic index
# --------------------------------------------------------------------------

def _synthetic_index(root: Path, n: int = 3) -> Path:
    """A minimal, correctly-bound index at `root`. No embedding model involved."""
    ids = [f"DOC__p{i}-{i}__c{i:04d}" for i in range(1, n + 1)]
    chunks = [
        {"chunk_id": cid, "document_id": "DOC", "document_name": "Doc",
         "document_type": "official guideline", "is_guideline": True,
         "section_title": None, "content_type": "clinical",
         "token_count": 10, "page_number": i, "page_start": i, "page_end": i,
         "source_file": "d.pdf", "chunk_text": f"chunk {i} text"}
        for i, cid in enumerate(ids, start=1)
    ]
    (root / "data" / "chunks").mkdir(parents=True)
    (root / "data" / "embeddings").mkdir(parents=True)
    chunks_path = root / "data" / "chunks" / "chunks.json"
    chunks_path.write_text(json.dumps({"chunks": chunks, "quality": {"invalid_chunk_ids": []}},
                                      ensure_ascii=False, indent=2), encoding="utf-8")

    d = root / "data" / "embeddings"
    np.save(d / "embeddings.npy", np.eye(n, 4, dtype=np.float32))
    (d / "embedded_chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    (d / cr.IDS_FILENAME).write_text(
        json.dumps({"n_ids": n, "chunk_ids_sha256": cr.chunk_ids_digest(ids), "chunk_ids": ids},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "index_meta.json").write_text(json.dumps({
        "index_type": "numpy_cosine", "metric": "cosine",
        "model_name": "synthetic", "embedding_dim": 4, "n_vectors": n,
        "chunks_file": "embedded_chunks.json", "ids_file": cr.IDS_FILENAME,
        "source_chunks_file": "data/chunks/chunks.json",
        "source_chunks_sha256": cr.file_sha256(chunks_path),
        "indexed_chunk_ids_sha256": cr.chunk_ids_digest(ids),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return root


def _rewrite(path: Path, mutate):
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_synthetic_index_loads_when_correctly_bound(tmp_path):
    """The control: the corruptions below must be what breaks it, not the fixture."""
    root = _synthetic_index(tmp_path)
    assert cr.load_index(root)["binding"]["n_indexed"] == 3


def test_load_index_refuses_a_length_preserving_reorder(tmp_path):
    """The failure a count check cannot see: metadata re-sorted, vectors not."""
    root = _synthetic_index(tmp_path)
    path = root / "data" / "embeddings" / "embedded_chunks.json"
    _rewrite(path, lambda recs: recs.insert(0, recs.pop(1)))
    with pytest.raises(RuntimeError, match="element-wise"):
        cr.load_index(root)


def test_load_index_refuses_a_stale_chunk_set(tmp_path):
    """chunks.json re-chunked without re-indexing."""
    root = _synthetic_index(tmp_path)
    _rewrite(root / "data" / "chunks" / "chunks.json",
             lambda p: p["chunks"][0].update({"chunk_text": "re-chunked text"}))
    with pytest.raises(RuntimeError, match="has changed since this index was built"):
        cr.load_index(root)


def test_load_index_refuses_an_index_with_no_id_list(tmp_path):
    root = _synthetic_index(tmp_path)
    (root / "data" / "embeddings" / cr.IDS_FILENAME).unlink()
    with pytest.raises(RuntimeError, match="does not carry the chunk_id list"):
        cr.load_index(root)


def test_load_index_refuses_an_unstamped_index(tmp_path):
    """An index built before the digests existed must not load silently."""
    root = _synthetic_index(tmp_path)
    _rewrite(root / "data" / "embeddings" / "index_meta.json",
             lambda m: m.pop("source_chunks_sha256"))
    with pytest.raises(RuntimeError, match="source_chunks_sha256"):
        cr.load_index(root)


def test_load_index_still_refuses_a_vector_count_mismatch(tmp_path):
    """The original length check must survive the stronger one."""
    root = _synthetic_index(tmp_path)
    np.save(root / "data" / "embeddings" / "embeddings.npy", np.eye(2, 4, dtype=np.float32))
    with pytest.raises(RuntimeError, match="out of sync"):
        cr.load_index(root)

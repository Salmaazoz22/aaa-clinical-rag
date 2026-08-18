# -*- coding: utf-8 -*-
"""One-off: stamp the committed index with the provenance P2/P3 require.

Why this is not a rebuild
-------------------------
`retrieval.index.save_index` now writes `data/embeddings/ids.json` and stamps
`source_chunks_sha256` + `indexed_chunk_ids_sha256` into `index_meta.json`, and
`retrieval.index.load_index` refuses to load an index that lacks them. The committed
index predates those fields, so it has to acquire them -- but re-running
`eval/rebuild_shipped_index.py` would re-encode 991 chunks and rewrite
`embeddings.npy`, whose SHA-256 is a frozen artifact hash. Floating-point
re-encoding is reproducible only to ~1e-6, so the bytes would move.

So this script adds the provenance WITHOUT touching the vectors: it derives the
digests from the artifacts already on disk, preserves every existing field of
`index_meta.json` (including `created_at_utc`), and leaves `embeddings.npy` and
`embedded_chunks.json` byte-identical.

That is only sound if the committed index really was built from the committed
`chunks.json`. This script proves that before it stamps anything, by re-running the
shipped selection (`validate_chunks` -> `embeddable_chunks`) over `chunks.json` and
requiring the result to equal `embedded_chunks.json` id-for-id and text-for-text.
If it does not, nothing is written.

Idempotent: running it again is a no-op.

Run:
    python eval/stamp_index_provenance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPTS_DIR.parent
ROOT = EVAL_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import ingestion.chunking as cc  # noqa: E402
import retrieval.index as cr  # noqa: E402

INDEX_DIR = ROOT / "data" / "embeddings"
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.json"

# The order retrieval.index.save_index writes, so a future rebuild produces the same
# key order and a diff of index_meta.json stays readable.
META_KEY_ORDER = [
    "index_type", "metric", "model_name", "model_revision", "embedding_dim",
    "token_limit", "max_chunk_tokens", "n_vectors", "failed_embeddings",
    "vectors_file", "chunks_file", "ids_file",
    "source_chunks_file", "source_chunks_sha256", "indexed_chunk_ids_sha256",
    "created_at_utc",
]


def prove_index_came_from_committed_chunks() -> dict[str, Any]:
    """Re-run the shipped selection over chunks.json and compare to the index.

    No embedding involved: this compares which chunks were selected, in what
    order, with what text -- which is exactly what the digests will assert.
    """
    payload = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    produced = payload.get("chunks") or []
    quality = payload.get("quality") or cc.validate_chunks(produced)
    selected = cc.embeddable_chunks(produced, quality)

    indexed = json.loads((INDEX_DIR / "embedded_chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(INDEX_DIR / "embeddings.npy")

    sel_ids = [c["chunk_id"] for c in selected]
    idx_ids = [c["chunk_id"] for c in indexed]
    text_diffs = [
        a["chunk_id"]
        for a, b in zip(selected, indexed)
        if a["chunk_id"] == b["chunk_id"] and (a.get("chunk_text") or "") != (b.get("chunk_text") or "")
    ]
    return {
        "produced_chunks": len(produced),
        "selected_by_shipped_filter": len(sel_ids),
        "indexed_records": len(idx_ids),
        "vectors": int(vectors.shape[0]),
        "id_lists_identical": sel_ids == idx_ids,
        "text_differences": len(text_diffs),
        "first_text_difference": text_diffs[0] if text_diffs else None,
        "proven": (
            sel_ids == idx_ids
            and not text_diffs
            and len(idx_ids) == int(vectors.shape[0])
        ),
        "indexed_ids": idx_ids,
    }


def main() -> int:
    print("proving the committed index was built from the committed chunks.json ...")
    proof = prove_index_came_from_committed_chunks()
    print(f"  produced={proof['produced_chunks']}  selected={proof['selected_by_shipped_filter']}"
          f"  indexed={proof['indexed_records']}  vectors={proof['vectors']}")
    print(f"  id_lists_identical={proof['id_lists_identical']}"
          f"  text_differences={proof['text_differences']}")
    if not proof["proven"]:
        print("  FAILED -- the committed index is NOT the shipped filter's output over the "
              "committed chunks.json.")
        print("  Refusing to stamp a provenance digest that would assert otherwise.")
        print("  Investigate, then rebuild with eval/rebuild_shipped_index.py.")
        return 1
    print("  OK\n")

    indexed_ids = proof["indexed_ids"]
    meta_path = INDEX_DIR / "index_meta.json"
    before_sha = cr.file_sha256(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    stamped = {
        "ids_file": cr.IDS_FILENAME,
        "source_chunks_file": "data/chunks/chunks.json",
        "source_chunks_sha256": cr.file_sha256(CHUNKS_PATH),
        "indexed_chunk_ids_sha256": cr.chunk_ids_digest(indexed_ids),
    }
    merged = {**meta, **stamped}
    ordered = {k: merged[k] for k in META_KEY_ORDER if k in merged}
    ordered.update({k: v for k, v in merged.items() if k not in ordered})

    ids_path = INDEX_DIR / cr.IDS_FILENAME
    ids_payload = {
        "note": (
            "Ordered chunk_id list, row-aligned to embeddings.npy. Retrieval joins "
            "vectors to metadata by position, so this list is what makes the join "
            "checkable: retrieval.index.load_index refuses to load unless it matches "
            "embedded_chunks.json element-wise."
        ),
        "vectors_file": "embeddings.npy",
        "chunks_file": "embedded_chunks.json",
        "n_ids": len(indexed_ids),
        "chunk_ids_sha256": cr.chunk_ids_digest(indexed_ids),
        "chunk_ids": indexed_ids,
    }

    meta_text = json.dumps(ordered, ensure_ascii=False, indent=2)
    ids_text = json.dumps(ids_payload, ensure_ascii=False, indent=2)
    unchanged = (
        meta_path.read_text(encoding="utf-8") == meta_text
        and ids_path.exists()
        and ids_path.read_text(encoding="utf-8") == ids_text
    )
    if unchanged:
        print("already stamped and up to date -- nothing written.")
        print(f"  index_meta.json sha256: {before_sha}")
        return 0

    ids_path.write_text(ids_text, encoding="utf-8")
    meta_path.write_text(meta_text, encoding="utf-8")

    print("stamped:")
    for k, v in stamped.items():
        print(f"  {k}: {v}")
    print(f"\nwrote data/embeddings/{cr.IDS_FILENAME} ({len(indexed_ids)} ids)")
    print("index_meta.json sha256 CHANGED (its content changed; this is the intended edit):")
    print(f"  before: {before_sha}")
    print(f"  after : {cr.file_sha256(meta_path)}")
    print("embeddings.npy and embedded_chunks.json were NOT written.")

    # The point of the exercise: the strict loader must now accept this index.
    binding = cr.verify_index_binding(
        ROOT, json.loads(meta_path.read_text(encoding="utf-8")),
        json.loads((INDEX_DIR / "embedded_chunks.json").read_text(encoding="utf-8")),
        np.load(INDEX_DIR / "embeddings.npy"),
    )
    print(f"\nload-time verification passes: {binding}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

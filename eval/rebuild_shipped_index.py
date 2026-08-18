# -*- coding: utf-8 -*-
"""Rebuild data/chunks and data/embeddings with the shipped chunker.

Runs exactly what a clean user gets: ingestion.chunking.run_chunking() with its
default strategy, then retrieval.index's index builder. No experiment code is
involved, so the shipped artifacts are reproducible from the shipped pipeline.

The previous page-buffer artifacts are preserved in data/archive_baseline_index/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

import ingestion.chunking as cc  # noqa: E402
import retrieval.index as cr  # noqa: E402


def main() -> int:
    print(f"chunker strategy: {cc.DEFAULT_CHUNKER}")
    result = cc.run_chunking(ROOT)
    q = result["quality"]
    print(f"chunks written  : {q['total']}  (valid {q['valid']}, invalid {q['invalid']})")
    print(f"content types   : {q['content_types']}")
    print(f"tokens          : max {q['tokens']['max_tokens']}, "
          f"over limit {q['tokens']['exceeding_limit']}, limit {q['tokens']['token_limit']}")
    print(f"status          : {q['status']}")
    if q["status"] != "PASS":
        print("chunk validation FAILED -- not building an index on top of it")
        return 1

    print("\nembedding ...")
    built = cr.run_embedding_index(ROOT)
    print(f"model     : {built['model_name']}")
    print(f"embedded  : {built['total_embedded']} chunks -> {built['embedding_dim']}-dim")
    print(f"failed    : {built['failed']}")

    meta = json.loads((ROOT / "data/embeddings/index_meta.json").read_text(encoding="utf-8"))
    print(f"\nindex_meta: {json.dumps(meta, indent=1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

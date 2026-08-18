# -*- coding: utf-8 -*-
"""Qdrant-backed production retriever.

Retrieval semantics are the validated ones, unchanged:

    question -> MedEmbed-base-v0.1 @ 7a90c502 (L2-normalised) -> cosine -> top 10

There is no reranking, no query rewriting, no intent detection, no keyword
bonus, no filtering and no per-question logic. The retriever cannot see *which*
question it is answering, which is what makes the frozen evaluation meaningful.

The returned hit dictionaries use the same keys as `retrieval.index.retrieve`, so
anything already consuming local evidence (including `eval/scripts/evaluate.py`)
accepts them unchanged. `text` and `score` are added as aliases of `chunk_text`
and `similarity_score`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vectordb.config import QdrantSettings, load_settings, make_client  # noqa: E402
from vectordb.schema import DEFAULT_TOP_K, EXPECTED_DIM, EXPECTED_MODEL, point_id_for  # noqa: E402

MAX_QUERY_CHARS = 20_000


def _check_top_k(top_k: Any) -> None:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")


class QdrantRetriever:
    """Dense cosine top-K retrieval over the production collection."""

    def __init__(
        self,
        settings: QdrantSettings | None = None,
        client=None,
        model=None,
        model_name: str = EXPECTED_MODEL,
    ) -> None:
        self.settings = settings or load_settings()
        self.client = client if client is not None else make_client(self.settings)
        self.model_name = model_name
        self._model = model

    # -- embedding ---------------------------------------------------------
    @property
    def model(self):
        """The pinned MedEmbed encoder, loaded on first use."""
        if self._model is None:
            from ingestion.chunking import load_embedder

            self._model = load_embedder(self.model_name)
        return self._model

    def embed_query(self, query: str) -> list[float]:
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        if not query.strip():
            raise ValueError("query is empty")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(
                f"query is {len(query)} characters, over the {MAX_QUERY_CHARS}-character limit"
            )
        vector = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        if len(vector) != EXPECTED_DIM:
            raise RuntimeError(f"query embedding has dimension {len(vector)}, expected {EXPECTED_DIM}")
        return [float(x) for x in vector]

    # -- search ------------------------------------------------------------
    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        _check_top_k(top_k)
        return self.search_vector(self.embed_query(query), top_k=top_k)

    def search_vector(self, vector: list[float], top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        from qdrant_client import models

        _check_top_k(top_k)
        if len(vector) != EXPECTED_DIM:
            raise ValueError(f"query vector has dimension {len(vector)}, expected {EXPECTED_DIM}")
        # Embedded local mode is brute-force by construction and warns that
        # search_params is ignored, so the flag is only sent to a real server.
        exact = self.settings.exact_search and not self.settings.local_path
        params = models.SearchParams(exact=True) if exact else None
        response = self.client.query_points(
            collection_name=self.settings.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            search_params=params,
        )
        return [self._to_hit(rank, point) for rank, point in enumerate(response.points, start=1)]

    # -- direct lookup -------------------------------------------------------
    def get_by_chunk_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch full payloads for known chunk_ids, no query or embedding involved.

        Point IDs are deterministic (`point_id_for(chunk_id)`), so this is a direct
        by-ID lookup rather than a search: it exists for auditing and manual review,
        where the chunk is already known and only its full text and metadata are
        needed. Unknown or missing chunk_ids are simply absent from the result
        instead of raising, so a caller reviewing a batch of citations does not lose
        the whole batch to one bad ID.
        """
        ids = [str(c) for c in dict.fromkeys(chunk_ids) if c]
        if not ids:
            return {}
        points = self.client.retrieve(
            collection_name=self.settings.collection,
            ids=[point_id_for(c) for c in ids],
            with_payload=True,
        )
        by_point_id = {str(p.id): p for p in points}
        out: dict[str, dict[str, Any]] = {}
        for chunk_id in ids:
            point = by_point_id.get(point_id_for(chunk_id))
            if point is None:
                continue
            payload = point.payload or {}
            out[chunk_id] = {
                "chunk_id": payload.get("chunk_id", chunk_id),
                "chunk_text": payload.get("chunk_text"),
                "document": payload.get("document_name"),
                "document_id": payload.get("document_id"),
                "section": payload.get("section_title"),
                "page": payload.get("page_number"),
                "page_start": payload.get("page_start"),
                "page_end": payload.get("page_end"),
                "recommendation_id": payload.get("recommendation_id"),
                "recommendation_grade": payload.get("recommendation_grade"),
                "evidence_level": payload.get("evidence_level"),
            }
        return out

    # -- evidence shape ----------------------------------------------------
    @staticmethod
    def _to_hit(rank: int, point) -> dict[str, Any]:
        payload = point.payload or {}
        return {
            "rank": rank,
            "similarity_score": float(point.score),
            "score": float(point.score),
            "chunk_id": payload.get("chunk_id"),
            "document": payload.get("document_name"),
            "document_id": payload.get("document_id"),
            "document_type": payload.get("document_type"),
            "is_guideline": payload.get("is_guideline"),
            "section": payload.get("section_title"),
            "content_type": payload.get("content_type"),
            "token_count": payload.get("token_count"),
            "page": payload.get("page_number"),
            "page_start": payload.get("page_start"),
            "page_end": payload.get("page_end"),
            "source_file": payload.get("source_file"),
            "source_excerpt": payload.get("source_excerpt"),
            "chunk_text": payload.get("chunk_text"),
            "text": payload.get("chunk_text"),
            "recommendation_id": payload.get("recommendation_id"),
            "recommendation_grade": payload.get("recommendation_grade"),
            "evidence_level": payload.get("evidence_level"),
            "point_id": str(point.id),
        }


def retrieve(query: str, top_k: int = DEFAULT_TOP_K, retriever: QdrantRetriever | None = None) -> list[dict[str, Any]]:
    """Convenience wrapper mirroring `retrieval.index.retrieve`'s call shape."""
    return (retriever or QdrantRetriever()).search(query, top_k=top_k)


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Query the production Qdrant collection.")
    ap.add_argument("query", help="the clinical question")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--json", action="store_true", help="print raw JSON hits")
    args = ap.parse_args()

    hits = retrieve(args.query, top_k=args.top_k)
    if args.json:
        print(json.dumps(hits, indent=2, ensure_ascii=False))
        return 0
    for hit in hits:
        print(
            f"rank {hit['rank']:>2}  score={hit['similarity_score']:.4f}  {hit['chunk_id']}\n"
            f"    {hit['document_id']}  pages {hit['page_start']}-{hit['page_end']}  "
            f"section: {hit['section']}\n"
            f"    {(hit['chunk_text'] or '')[:180]}...\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

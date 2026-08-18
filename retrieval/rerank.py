# -*- coding: utf-8 -*-
"""Optional cross-encoder reranking stage for AAA clinical retrieval.

Dense retrieval stays the default and is untouched. This module is opt-in:

    dense top-N candidates  ->  cross-encoder scores (query, chunk_text)  ->  top-k

There is no lexical channel here: no BM25, no reciprocal rank fusion, no hybrid
scoring. There are also no query-specific rules, keyword bonuses or per-document
weights -- the reranker sees only the query string and the chunk text, so it
behaves identically on a question it has never seen.
"""
from __future__ import annotations

from typing import Any

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CANDIDATES = 30

_MODEL_CACHE: dict[str, Any] = {}


def load_reranker(model_name: str = DEFAULT_RERANK_MODEL):
    """Load (and cache) the cross-encoder. CPU is fine at this corpus size."""
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import CrossEncoder

        _MODEL_CACHE[model_name] = CrossEncoder(model_name)
    return _MODEL_CACHE[model_name]


def rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    model=None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Reorder dense candidates by cross-encoder relevance and re-rank 1..k.

    The dense cosine score is preserved as `dense_score` so a run can always be
    traced back to the retrieval that produced its candidates.
    """
    if not hits:
        return []
    model = model or load_reranker()
    pairs = [(query, h.get("chunk_text") or "") for h in hits]
    scores = model.predict(pairs)

    order = sorted(range(len(hits)), key=lambda i: (-float(scores[i]), i))
    out = []
    for rank, i in enumerate(order[:top_k], start=1):
        hit = dict(hits[i])
        hit["dense_score"] = hit.get("similarity_score")
        hit["dense_rank"] = hit.get("rank")
        hit["similarity_score"] = float(scores[i])
        hit["rank"] = rank
        out.append(hit)
    return out

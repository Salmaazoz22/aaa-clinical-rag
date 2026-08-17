# -*- coding: utf-8 -*-
"""EXPERIMENTAL hybrid retrieval: MedEmbed dense + BM25 lexical, score fusion.

Isolated by design. Production (`clinical_rag.retrieve`) remains pure dense
MedEmbed and does not import this module.

Method (fixed before any result was observed)
---------------------------------------------
1. CANDIDATE POOL. Take the top-30 from the dense retriever and the top-30 from
   BM25, then union them (<= 60 distinct chunks). Depth 30 is fixed and is not
   tuned in this phase.

2. COMPLETE SCORING. Every candidate in the union is scored by BOTH retrievers
   directly -- the dense score from the cosine of the already-built vectors, the
   BM25 score from the lexical scorer. A chunk found by only one retriever
   therefore gets its real score from the other, not an imputed zero. Imputing a
   floor would systematically penalise single-retriever candidates and is the
   usual source of spurious "hybrid helps" results.

3. NORMALISATION. Min-max, computed independently per retriever, over the union
   pool for this query:

       normalized_score = (score - min_score) / (max_score - min_score)

   If max_score == min_score the whole pool is tied for that retriever and every
   normalized score is 0.0 (a constant channel carries no ranking information).
   Min-max is required here because the two score scales are not comparable:
   dense cosine sits in a narrow ~0.60-0.75 band while BM25 is unbounded and
   reached ~17 in Phase 3. Raw addition would let BM25 dominate by scale alone.

4. FUSION.

       hybrid_score = alpha * normalized_dense + (1 - alpha) * normalized_bm25

5. RANKING. Sort by hybrid_score descending, breaking ties by corpus index so
   the ordering is deterministic. Return the top-10.

No query-specific rules, no term boosts, no per-document weights, no use of the
gold labels.
"""
from __future__ import annotations

from typing import Any

import numpy as np

CANDIDATE_DEPTH = 30
PRE_REGISTERED_ALPHAS = {
    "hybrid_a_75_25": 0.75,
    "hybrid_b_50_50": 0.50,
    "hybrid_c_25_75": 0.25,
}


def _minmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo <= 0.0:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


class HybridRetriever:
    """Fuses a dense numpy index with a BM25Index over the same chunk list."""

    def __init__(self, vectors: np.ndarray, chunks: list[dict[str, Any]], bm25,
                 candidate_depth: int = CANDIDATE_DEPTH):
        if len(chunks) != vectors.shape[0]:
            raise ValueError("vectors and chunks are out of sync")
        if list(c["chunk_id"] for c in chunks) != list(c["chunk_id"] for c in bm25.chunks):
            raise ValueError("dense and lexical retrievers cover different corpora")
        self.vectors = vectors
        self.chunks = chunks
        self.bm25 = bm25
        self.candidate_depth = candidate_depth

    def _pool(self, query: str, query_vector: np.ndarray) -> tuple[list[int], dict[int, float], dict[int, float]]:
        dense_all = self.vectors @ query_vector
        dense_top = np.argsort(-dense_all)[: self.candidate_depth].tolist()

        lex_all = self.bm25.score_query(query)
        lex_top = [i for i, _ in sorted(lex_all.items(), key=lambda kv: (-kv[1], kv[0]))[: self.candidate_depth]]

        pool = sorted(set(dense_top) | set(lex_top))
        dense_scores = {i: float(dense_all[i]) for i in pool}
        lex_scores = {i: float(lex_all.get(i, 0.0)) for i in pool}
        return pool, dense_scores, lex_scores

    def retrieve(self, query: str, query_vector: np.ndarray, alpha: float,
                 top_k: int = 10) -> list[dict[str, Any]]:
        pool, dense_scores, lex_scores = self._pool(query, query_vector)
        nd, nl = _minmax(dense_scores), _minmax(lex_scores)
        fused = {i: alpha * nd[i] + (1.0 - alpha) * nl[i] for i in pool}

        order = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        hits = []
        for rank, (i, score) in enumerate(order[:top_k], start=1):
            chunk = self.chunks[i]
            hits.append(
                {
                    "rank": rank,
                    "similarity_score": float(score),
                    "dense_score": dense_scores[i],
                    "bm25_score": lex_scores[i],
                    "dense_norm": nd[i],
                    "bm25_norm": nl[i],
                    "chunk_id": chunk.get("chunk_id"),
                    "document": chunk.get("document_name"),
                    "document_id": chunk.get("document_id"),
                    "section": chunk.get("section_title"),
                    "content_type": chunk.get("content_type"),
                    "token_count": chunk.get("token_count"),
                    "page": chunk.get("page_number"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "source_file": chunk.get("source_file"),
                    "chunk_text": chunk.get("chunk_text"),
                }
            )
        return hits

    def full_ranking(self, query: str, query_vector: np.ndarray, alpha: float) -> dict[str, int]:
        """chunk_id -> rank within the fused candidate pool. Diagnostic only."""
        pool, dense_scores, lex_scores = self._pool(query, query_vector)
        nd, nl = _minmax(dense_scores), _minmax(lex_scores)
        fused = {i: alpha * nd[i] + (1.0 - alpha) * nl[i] for i in pool}
        order = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        return {self.chunks[i]["chunk_id"]: r for r, (i, _) in enumerate(order, start=1)}

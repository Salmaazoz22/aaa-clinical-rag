# -*- coding: utf-8 -*-
"""EXPERIMENTAL lexical (BM25) retrieval over the existing indexed chunks.

Isolated by design. Nothing in the production pipeline imports this module:
`retrieval.index.retrieve` remains pure dense MedEmbed retrieval and is untouched.
This exists so keyword retrieval can be *measured* against the frozen gold
standard before any decision is made about adopting or fusing it.

Scoring
-------
Okapi BM25 in the Lucene/Elasticsearch formulation:

    score(q, d) = sum over query terms t of
        idf(t) * ( f(t,d) * (k1 + 1) ) / ( f(t,d) + k1 * (1 - b + b * |d| / avgdl) )

    idf(t) = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )

The `1 +` inside the logarithm is the standard non-negative variant, so a term
appearing in more than half the corpus can never contribute a negative score.
This removes the need for `rank_bm25`'s `epsilon` floor and is what Lucene,
Elasticsearch and Anserini use.

Parameters are the canonical defaults, deliberately untuned: k1 = 1.5, b = 0.75.

The retriever sees only the question string and the chunk text. There are no
query-specific rules, no term boosts, no per-document weights, no use of the
gold labels, and no special handling of any individual question.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable

K1 = 1.5
B = 0.75

# PDF ligatures survive extraction ("speciﬁc", "ﬁve"), so they are expanded
# before tokenisation or those words would never match a query term.
_LIGATURES = {
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    "’": "'", "‘": "'", "“": '"', "”": '"',
}

# Words and numbers. Decimals stay whole so clinical thresholds such as
# "5.5" survive as a single token instead of splitting into "5" and "5".
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[a-z]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase, expand ligatures, keep words and decimal numbers.

    No stopword list and no stemming: BM25's IDF already drives ubiquitous
    terms ("what", "the", "aneurysm") towards zero weight, and adding either
    would be a tuning choice rather than the library-standard behaviour.
    """
    if not text:
        return []
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """BM25 over a fixed list of chunk records (same records as the dense index)."""

    def __init__(self, chunks: list[dict[str, Any]], k1: float = K1, b: float = B):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.docs: list[Counter[str]] = []
        self.lengths: list[int] = []
        df: Counter[str] = Counter()
        for chunk in chunks:
            tokens = tokenize(chunk.get("chunk_text") or "")
            tf = Counter(tokens)
            self.docs.append(tf)
            self.lengths.append(len(tokens))
            df.update(tf.keys())
        self.n_docs = len(chunks)
        self.avgdl = (sum(self.lengths) / self.n_docs) if self.n_docs else 0.0
        self.df = df
        self.idf = {
            term: math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        # Inverted index so scoring touches only chunks that share a term.
        self.postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self.docs):
            for term in tf:
                self.postings.setdefault(term, []).append(i)

    def _norm(self, i: int) -> float:
        return self.k1 * (1.0 - self.b + self.b * self.lengths[i] / (self.avgdl or 1.0))

    def score_query(self, query: str) -> dict[int, float]:
        scores: dict[int, float] = {}
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings.get(term, ()):
                f = self.docs[i][term]
                scores[i] = scores.get(i, 0.0) + idf * (f * (self.k1 + 1.0)) / (f + self._norm(i))
        return scores

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Ranked hits in the same shape `retrieval.index.retrieve` returns."""
        scores = self.score_query(query)
        # Deterministic: score descending, then original index order on ties.
        order = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        hits = []
        for rank, (i, score) in enumerate(order, start=1):
            chunk = self.chunks[i]
            hits.append(
                {
                    "rank": rank,
                    "similarity_score": float(score),
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

    def rank_of(self, query: str) -> dict[str, int]:
        """chunk_id -> rank over the whole corpus, for diagnostics."""
        scores = self.score_query(query)
        order = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return {self.chunks[i]["chunk_id"]: r for r, (i, _) in enumerate(order, start=1)}

    def explain(self, query: str, chunk_id: str) -> list[tuple[str, int, float, float]]:
        """Per-term (term, tf, idf, contribution) for one chunk. Diagnostic only."""
        i = next((j for j, c in enumerate(self.chunks) if c.get("chunk_id") == chunk_id), None)
        if i is None:
            return []
        out = []
        for term in dict.fromkeys(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                out.append((term, 0, 0.0, 0.0))
                continue
            f = self.docs[i].get(term, 0)
            contrib = idf * (f * (self.k1 + 1.0)) / (f + self._norm(i)) if f else 0.0
            out.append((term, f, idf, contrib))
        return sorted(out, key=lambda t: -t[3])


def build_from_index(chunks: Iterable[dict[str, Any]]) -> BM25Index:
    return BM25Index(list(chunks))

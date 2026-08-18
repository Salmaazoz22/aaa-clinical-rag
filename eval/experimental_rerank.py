# -*- coding: utf-8 -*-
"""EXPERIMENTAL biomedical cross-encoder reranking over MedEmbed candidates.

Isolated by design. Production (`retrieval.index.retrieve`) remains pure dense
MedEmbed and does not import this module.

Model
-----
`ncbi/MedCPT-Cross-Encoder`, pinned to revision
`71caf65d4927987813984f54c284405a13fcca49`.

Why this model rather than the Experiment 4 reranker
----------------------------------------------------
Experiment 6 established that `cross-encoder/ms-marco-MiniLM-L-6-v2` degrades a
strong dense ranking on this corpus: trained on short general-web query/passage
pairs, it promoted deep irrelevant candidates over correct top hits and cut
Answering@5 from 10/10 to 7/10. It is retained only as a negative reference.

MedCPT-Cross-Encoder is the reranking stage of NCBI's MedCPT retrieval system.
It is a PubMedBERT-based cross-encoder trained on 255M real biomedical search
query/article click pairs from PubMed logs, so both its vocabulary and its
notion of "relevant to a clinical query" come from the biomedical domain. This
is the same reasoning that made MedEmbed succeed where the general-purpose BGE
encoder lost top-1 accuracy: domain adaptation, not raw model strength.

Architecture      : BERT (PubMedBERT backbone) for sequence classification,
                    single regression output used as the relevance score
Input format      : a (query, passage) pair per candidate; the raw chunk text is
                    passed verbatim as the passage. No query rewriting, no
                    instruction prefix, no template.
Max sequence      : 512 tokens for the concatenated pair
Runtime           : CPU, ~109M parameters

No query-specific rules, no term boosts, no per-document weights, no use of the
gold labels, and no special handling of any individual question.
"""
from __future__ import annotations

from typing import Any

DEFAULT_RERANK_MODEL = "ncbi/MedCPT-Cross-Encoder"
DEFAULT_REVISION = "71caf65d4927987813984f54c284405a13fcca49"
MAX_LENGTH = 512
DEFAULT_CANDIDATES = 30

_CACHE: dict[str, Any] = {}


def load_reranker(model_name: str = DEFAULT_RERANK_MODEL, revision: str = DEFAULT_REVISION):
    """Load (and cache) the cross-encoder at its pinned revision."""
    key = f"{model_name}@{revision}"
    if key not in _CACHE:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)
        model.eval()
        _CACHE[key] = (tokenizer, model, torch)
    return _CACHE[key]


def score_pairs(query: str, passages: list[str], model_name: str = DEFAULT_RERANK_MODEL,
                revision: str = DEFAULT_REVISION) -> tuple[list[float], int]:
    """Relevance score per passage, plus how many pairs hit the length limit.

    The truncation count is returned rather than swallowed: a truncated pair is
    scored on text the chunk does not fully contain, which would quietly
    misrepresent the candidate.
    """
    tokenizer, model, torch = load_reranker(model_name, revision)
    pairs = [[query, p or ""] for p in passages]
    with torch.no_grad():
        encoded = tokenizer(
            pairs, truncation=True, padding=True, return_tensors="pt", max_length=MAX_LENGTH
        )
        logits = model(**encoded).logits.squeeze(dim=1)
        scores = [float(x) for x in logits.tolist()] if logits.dim() else [float(logits)]

    # Count pairs that would exceed the window if not truncated.
    truncated = 0
    for q, p in pairs:
        if len(tokenizer(q, p, truncation=False)["input_ids"]) > MAX_LENGTH:
            truncated += 1
    return scores, truncated


def rerank_hits(query: str, hits: list[dict[str, Any]], top_k: int = 10,
                model_name: str = DEFAULT_RERANK_MODEL,
                revision: str = DEFAULT_REVISION) -> tuple[list[dict[str, Any]], int]:
    """Reorder dense candidates by cross-encoder score and re-rank 1..k.

    The dense score and dense rank are preserved on every hit so a run can be
    traced back to the retrieval that produced its candidates.
    """
    if not hits:
        return [], 0
    scores, truncated = score_pairs(query, [h.get("chunk_text") or "" for h in hits],
                                    model_name, revision)
    order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
    out = []
    for rank, i in enumerate(order[:top_k], start=1):
        hit = dict(hits[i])
        hit["dense_score"] = hit.get("similarity_score")
        hit["dense_rank"] = hit.get("rank")
        hit["similarity_score"] = scores[i]
        hit["rank"] = rank
        out.append(hit)
    return out, truncated


def full_reranked_order(query: str, hits: list[dict[str, Any]],
                        model_name: str = DEFAULT_RERANK_MODEL,
                        revision: str = DEFAULT_REVISION) -> dict[str, int]:
    """chunk_id -> rank across the whole candidate pool. Diagnostic only."""
    scores, _ = score_pairs(query, [h.get("chunk_text") or "" for h in hits], model_name, revision)
    order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
    return {hits[i]["chunk_id"]: r for r, i in enumerate(order, start=1)}

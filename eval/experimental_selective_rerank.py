# -*- coding: utf-8 -*-
"""EXPERIMENTAL selective / confidence-aware reranking.

Isolated by design. Production (`clinical_rag.retrieve`) remains pure dense
MedEmbed and does not import this module.

Hypothesis
----------
Phase 5 showed that reranking every query with `ncbi/MedCPT-Cross-Encoder`
rescues Q4 (dense rank 16 -> 6) but displaces three queries the dense retriever
already answered correctly at rank 1. If the reranker is invoked only when the
dense retriever looks uncertain, the confident rankings should survive while the
uncertain ones still get rescued.

Confidence signals (dense scores only)
--------------------------------------
Derived exclusively from MedEmbed's own similarity scores over its top-30:

    confidence_top1 = s1
    margin_1_2      = s1 - s2
    margin_1_5      = s1 - s5
    margin_1_10     = s1 - s10
    spread_10       = s1 - s10        (identical to margin_1_10 by definition)

No relevance labels, no gold data, no answer text, no document/page metadata, no
query identity, no keyword or lexical signal. The selector cannot distinguish Q4
from any other query except through these numbers.

Threshold rule (pre-registered)
-------------------------------
PERCENTILE = 25. For a given signal, the threshold is the 25th percentile of
that signal's values across the evaluated query set, and a query is reranked iff
its value is STRICTLY BELOW the threshold.

25 was chosen from the hypothesis, not from results: the point of selective
reranking is minimal intervention, so the selector must fire on roughly the
least-confident quarter of queries. A 50th or 75th percentile would rerank half
or three quarters of the set and largely reproduce the Phase 5 damage.

CAVEAT: thresholds computed over the evaluation set itself are transductive.
The selector sees this set's score distribution (though never its labels), so
these numbers are not a clean generalisation estimate.

Policies
--------
    A : rerank iff margin_1_10 < P25(margin_1_10)
    B : rerank iff margin_1_2  < P25(margin_1_2)
    C : rerank iff both A and B fire
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

PERCENTILE = 25
CANDIDATE_DEPTH = 30
POLICIES = ("A", "B", "C")


def confidence_signals(hits: list[dict[str, Any]]) -> dict[str, float]:
    """Confidence numbers for one query's dense candidate list (rank 1 first)."""
    s = [float(h["similarity_score"]) for h in hits]
    if not s:
        return {"confidence_top1": 0.0, "margin_1_2": 0.0, "margin_1_5": 0.0,
                "margin_1_10": 0.0, "spread_10": 0.0}

    def at(rank: int) -> float:
        return s[rank - 1] if len(s) >= rank else s[-1]

    margin_1_10 = at(1) - at(10)
    return {
        "confidence_top1": at(1),
        "margin_1_2": at(1) - at(2),
        "margin_1_5": at(1) - at(5),
        "margin_1_10": margin_1_10,
        # Same quantity as margin_1_10; kept because the protocol lists it.
        "spread_10": margin_1_10,
    }


def thresholds(signals_by_query: dict[int, dict[str, float]],
               percentile: int = PERCENTILE) -> dict[str, float]:
    """P-th percentile of each signal across the evaluated query set."""
    keys = next(iter(signals_by_query.values())).keys()
    return {
        k: float(np.percentile([sig[k] for sig in signals_by_query.values()], percentile))
        for k in keys
    }


def policy_decisions(signals_by_query: dict[int, dict[str, float]],
                     thr: dict[str, float]) -> dict[str, dict[int, bool]]:
    """policy -> {query_id: rerank?}. Identical treatment for every query."""
    rules: dict[str, Callable[[dict[str, float]], bool]] = {
        "A": lambda s: s["margin_1_10"] < thr["margin_1_10"],
        "B": lambda s: s["margin_1_2"] < thr["margin_1_2"],
        "C": lambda s: (s["margin_1_2"] < thr["margin_1_2"]
                        and s["margin_1_10"] < thr["margin_1_10"]),
    }
    return {name: {qid: bool(rule(sig)) for qid, sig in signals_by_query.items()}
            for name, rule in rules.items()}

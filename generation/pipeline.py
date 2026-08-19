# -*- coding: utf-8 -*-
"""Retrieve -> threshold -> prompt -> generate -> validate.

This is the analogue of the reference implementation's
`NLPController.answer_rag_question`, and follows the same five steps: retrieve,
build the document block, build the footer, construct the messages, generate.
Four things are different, and each is the point of this layer:

1. **A quality gate between retrieval and prompting.** The reference passes
   whatever the vector store returned straight into the prompt, and its only
   abstention is `if not retrieved_documents: return None, None, None` -- an
   out-of-scope question still returns ten chunks and gets answered from them.
   Here every hit must clear a similarity floor to become evidence, and a query
   where nothing clears it refuses instead of answering.
2. **A safety gate above retrieval.** A patient-specific question is refused
   locally, before any model call, so patient details are never sent to a
   third-party API.
3. **The answer is structured and validated.** The reference returns the model's
   text verbatim. Here it is parsed into the slide-7 schema and every citation is
   checked against the chunks that were actually sent (`generation/validator.py`).
4. **Nothing is silently dropped.** Filtered chunks, validator findings and
   disclaimer normalisation are all recorded on the result.

Retrieval itself is untouched: this module calls `vectordb.retriever` and does
nothing else to it. No reranking, no query rewriting, no per-question branching.
The only retrieval-adjacent parameter is `top_k`, which is how many hits to ask
for.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.config import GenerationSettings, load_settings  # noqa: E402
from generation.emergency import screen_emergency  # noqa: E402
from generation.guideline_scope import screen_guideline_edition  # noqa: E402
from generation.parsing import parse_answer  # noqa: E402
from generation.prompts import SYSTEM_PROMPT, build_messages  # noqa: E402
from generation.providers import LLMProvider, build_provider  # noqa: E402
from generation.refusal import build_refusal  # noqa: E402
from generation.safety import screen_query  # noqa: E402
from generation.schema import (  # noqa: E402
    DISCLAIMER,
    REFUSAL_BELOW_THRESHOLD,
    REFUSAL_GUIDELINE_UNAVAILABLE,
    REFUSAL_NO_CHUNKS,
    REFUSAL_NOT_SPECIFIC,
    REFUSAL_PATIENT_SPECIFIC,
    REFUSAL_POTENTIAL_EMERGENCY,
    is_refusal,
)
from generation.validator import (  # noqa: E402
    documents_cited,
    resolve_citations,
    summarize_evidence_grade,
    validate_answer,
)


def _score(hit: dict[str, Any]) -> float:
    value = hit.get("similarity_score", hit.get("score"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def select_usable_hits(
    hits: Sequence[dict[str, Any]], threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split retrieved hits into (usable evidence, dropped as too weak).

    The floor is applied to the retriever's own cosine score. Nothing is
    re-ranked and nothing is re-scored: the order that comes out is the order
    that went in.
    """
    usable = [h for h in hits if _score(h) >= threshold]
    dropped = [h for h in hits if _score(h) < threshold]
    return usable, dropped


def _lite(hit: dict[str, Any], preview_chars: int = 240) -> dict[str, Any]:
    """The audit record for one retrieved chunk."""
    text = hit.get("chunk_text") or hit.get("text") or ""
    return {
        "rank": hit.get("rank"),
        "chunk_id": hit.get("chunk_id"),
        "similarity_score": _score(hit),
        "document_id": hit.get("document_id"),
        "document": hit.get("document"),
        "section": hit.get("section"),
        "page": hit.get("page"),
        "page_start": hit.get("page_start"),
        "page_end": hit.get("page_end"),
        "recommendation_id": hit.get("recommendation_id"),
        "recommendation_grade": hit.get("recommendation_grade"),
        "evidence_level": hit.get("evidence_level"),
        "text_preview": text.strip()[:preview_chars],
    }


@dataclass
class GenerationResult:
    """Everything needed to audit one answer."""

    query: str
    settings: dict[str, Any]
    safety: dict[str, Any]
    guideline_scope: dict[str, Any] = field(default_factory=dict)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    used_chunk_ids: list[str] = field(default_factory=list)
    dropped_chunks: list[dict[str, Any]] = field(default_factory=list)
    answer: dict[str, Any] = field(default_factory=dict)
    citations_resolved: list[dict[str, Any]] = field(default_factory=list)
    documents_cited: list[str] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    refusal: dict[str, Any] | None = None
    completion: dict[str, Any] | None = None
    parse_meta: dict[str, Any] | None = None
    disclaimer_normalised: bool = False
    prompt: dict[str, str] | None = None
    evidence_grade_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def refused(self) -> bool:
        return self.refusal is not None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "query": self.query,
            "settings": self.settings,
            "safety": self.safety,
            "guideline_scope": self.guideline_scope,
            "refused": self.refused,
            "refusal": self.refusal,
            "answer": self.answer,
            "citations_resolved": self.citations_resolved,
            "documents_cited": self.documents_cited,
            "evidence_grade_summary": self.evidence_grade_summary,
            "validation": self.validation,
            "retrieval": {
                "n_retrieved": len(self.retrieved),
                "n_used": len(self.used_chunk_ids),
                "n_dropped_below_threshold": len(self.dropped_chunks),
                "used_chunk_ids": self.used_chunk_ids,
                "dropped": self.dropped_chunks,
                "hits": self.retrieved,
            },
            "generation": {
                "completion": self.completion,
                "parse_meta": self.parse_meta,
                "disclaimer_normalised": self.disclaimer_normalised,
            },
        }
        if self.prompt is not None:
            out["prompt"] = self.prompt
        return out


def answer_question(
    query: str,
    *,
    retriever: Any = None,
    provider: LLMProvider | None = None,
    settings: GenerationSettings | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
    include_prompt: bool = False,
) -> GenerationResult:
    """Answer one clinical question from retrieved guideline evidence.

    `retriever` and `provider` are injectable so the gates and the validator can
    be tested without a vector store or a network call. When omitted, the frozen
    Qdrant retriever and the configured provider are used.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    settings = settings or load_settings()
    top_k = settings.top_k if top_k is None else top_k
    threshold = settings.score_threshold if threshold is None else threshold

    result = GenerationResult(
        query=query,
        settings={**settings.describe(), "top_k": top_k, "score_threshold": threshold},
        safety={},
    )

    # --- gate -1: potential emergency presentation (highest priority) --------
    # Runs BEFORE the patient-specific gate so that a query which is BOTH an
    # emergency presentation AND patient-specific always returns the emergency
    # redirect, never the calmer patient-specific message.  No model call, no
    # citations, no guideline text — only a redirect to emergency services.
    emergency_verdict = screen_emergency(query)

    # --- gate 0: patient-specific request (before any model call) ----------
    verdict = screen_query(query)
    result.safety = verdict.to_dict()

    # --- gate 0.5: a guideline edition the corpus does not contain ----------
    # Decided from the corpus manifest, not by the model: retrieval scores high
    # on "the 2026 ESVS guideline" precisely because every indexed chunk is an
    # ESVS passage on that topic, so the similarity floor cannot catch it.
    edition_verdict = screen_guideline_edition(query)
    result.guideline_scope = edition_verdict.to_dict()

    if retriever is None:
        from vectordb.retriever import QdrantRetriever

        retriever = QdrantRetriever()

    hits = retriever.search(query, top_k=top_k)
    result.retrieved = [_lite(h) for h in hits]

    usable, dropped = select_usable_hits(hits, threshold)
    result.used_chunk_ids = [str(h.get("chunk_id")) for h in usable]
    result.dropped_chunks = [
        {"chunk_id": h.get("chunk_id"), "similarity_score": _score(h), "document_id": h.get("document_id")}
        for h in dropped
    ]

    def finish_refusal(reason: str, gate: str, refusal_hits: Sequence[dict[str, Any]], refusal_detail: str | None = None) -> GenerationResult:
        answer = build_refusal(
            reason,
            query=query,
            hits=refusal_hits,
            threshold=threshold,
            detail=refusal_detail if refusal_detail is not None else verdict.detail,
        )
        result.answer = answer
        result.refusal = {"reason": reason, "gate": gate}
        report = validate_answer(answer, refusal_hits)
        result.validation = report.to_dict()
        result.citations_resolved = resolve_citations(answer, refusal_hits)
        result.documents_cited = documents_cited(answer, refusal_hits)
        result.evidence_grade_summary = summarize_evidence_grade(result.citations_resolved)
        return result

    if emergency_verdict.is_emergency:
        # Emergency takes priority over patient-specific: the urgent-care redirect
        # must win even when both conditions are simultaneously true.
        return finish_refusal(
            REFUSAL_POTENTIAL_EMERGENCY,
            "emergency",
            (),  # no citations, same rationale as patient-specific
            refusal_detail=emergency_verdict.detail,
        )

    if verdict.blocked:
        # Deliberately refused with no citations and no model call: see
        # generation/refusal.build_refusal.
        return finish_refusal(REFUSAL_PATIENT_SPECIFIC, f"safety:{verdict.rule}", ())

    if edition_verdict.blocked:
        # The retrieved passages ARE cited here, unlike the two gates above: they
        # are the editions that exist, and naming them is how the refusal shows
        # the reader what can be asked for instead.
        return finish_refusal(
            REFUSAL_GUIDELINE_UNAVAILABLE,
            "guideline_scope",
            hits,
            refusal_detail=edition_verdict.detail,
        )

    # --- gate 1: nothing retrieved at all ----------------------------------
    if not hits:
        return finish_refusal(REFUSAL_NO_CHUNKS, "retrieval:empty", ())

    # --- gate 2: nothing cleared the evidence floor ------------------------
    if not usable:
        return finish_refusal(REFUSAL_BELOW_THRESHOLD, "threshold", hits)

    # --- generate ----------------------------------------------------------
    messages = build_messages(query, usable)
    if include_prompt:
        result.prompt = {"system": messages[0]["content"], "user": messages[1]["content"]}

    provider = provider or build_provider(settings)
    completion = provider.complete(messages, json_mode=True)
    result.completion = completion.to_dict()

    answer, parse_meta = parse_answer(completion.text)
    result.parse_meta = parse_meta

    # --- validate against exactly what was sent ---------------------------
    report = validate_answer(answer, usable)
    result.validation = report.to_dict()

    # The disclaimer is a fixed safety string, so it is normalised rather than
    # trusted -- and the validator has already recorded a finding if the model's
    # version differed, so the replacement is visible, not silent.
    if answer.get("disclaimer") != DISCLAIMER:
        answer["disclaimer"] = DISCLAIMER
        result.disclaimer_normalised = True

    result.answer = answer
    result.citations_resolved = resolve_citations(answer, usable)
    result.documents_cited = documents_cited(answer, usable)
    result.evidence_grade_summary = summarize_evidence_grade(result.citations_resolved)

    if is_refusal(answer):
        # The model judged the evidence insufficient under rule R1(a)/(b). That
        # judgement needs the evidence read, so it cannot be made by a gate.
        result.refusal = {"reason": REFUSAL_NOT_SPECIFIC, "gate": "model"}

    return result


# --- CLI -------------------------------------------------------------------

def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Answer one AAA guideline question from retrieved evidence.")
    ap.add_argument("query", help="the clinical question")
    ap.add_argument("--provider", choices=("groq", "openrouter"), default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--json", action="store_true", help="print the full audit record")
    ap.add_argument("--show-prompt", action="store_true", help="include the rendered prompt")
    ap.add_argument("--print-system-prompt", action="store_true", help="print the system prompt and exit")
    args = ap.parse_args()

    if args.print_system_prompt:
        print(SYSTEM_PROMPT)
        return 0

    settings = load_settings(provider=args.provider)
    result = answer_question(
        args.query,
        settings=settings,
        top_k=args.top_k,
        threshold=args.threshold,
        include_prompt=args.show_prompt,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(json.dumps(result.answer, indent=2, ensure_ascii=False))
    print()
    validation = result.validation
    status = "PASS" if validation.get("ok") else "FAIL"
    print(
        f"validator: {status}  errors={validation.get('n_errors')} "
        f"warnings={validation.get('n_warnings')}  codes={validation.get('codes')}"
    )
    print(
        f"retrieval: {len(result.retrieved)} retrieved, {len(result.used_chunk_ids)} used, "
        f"{len(result.dropped_chunks)} below threshold {result.settings['score_threshold']}"
    )
    if result.refused:
        print(f"refused:  reason={result.refusal['reason']}  gate={result.refusal['gate']}")
    return 0 if result.validation.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Deterministic refusals.

Three of the four refusal conditions in Day 3 slide 10 are decidable without the
model -- nothing was retrieved, nothing cleared the score floor, or the question
is patient-specific -- so those refusals are built here, locally, and the model
is not called at all. The fourth ("evidence is topically related but not specific
enough") needs the evidence read, so it is the model's judgement, made under
system prompt rule R1(b) and detected by `confidence == "Insufficient Evidence"`.

Slide 10 also asks that a refusal be helpful rather than a bare "I don't know":
it should say what evidence *was* found, what is missing, and what kind of
guideline would answer the question. A deterministic refusal can do all three,
because it knows exactly what came back and why it was rejected.

The reference implementation has no analogue: on an empty retrieval it returns
`(None, None, None)`, which the route turns into an HTTP 400 with a generic
signal string. There is no refusal text, no confidence, no evidence summary.
"""
from __future__ import annotations

from typing import Any, Sequence

from generation.schema import (
    CONFIDENCE_INSUFFICIENT,
    DISCLAIMER,
    REFUSAL_BELOW_THRESHOLD,
    REFUSAL_GUIDELINE_UNAVAILABLE,
    REFUSAL_MESSAGE,
    REFUSAL_NO_CHUNKS,
    REFUSAL_PATIENT_SPECIFIC,
    REFUSAL_POTENTIAL_EMERGENCY,
)

#: The sentence a caller can match on to detect an unavailable-edition refusal.
GUIDELINE_UNAVAILABLE_MESSAGE = (
    "The requested guideline edition is not available in the indexed evidence corpus."
)


def _describe_evidence(hits: Sequence[dict[str, Any]], reason: str = "") -> str:
    """One sentence naming what the retrieval actually returned."""
    if reason == REFUSAL_POTENTIAL_EMERGENCY:
        if hits:
            return f"Guideline retrieval was performed ({len(hits)} passage(s) examined), but details are suppressed for emergency safety."
        return "Guideline retrieval details are suppressed for emergency safety."

    if not hits:
        return "No guideline passages were returned for this question."

    by_document: dict[str, list[str]] = {}
    for hit in hits:
        doc = str(hit.get("document_id") or hit.get("document") or "unknown")
        sections = by_document.setdefault(doc, [])
        section = hit.get("section")
        if section and str(section) not in sections:
            sections.append(str(section))

    parts = []
    for doc, sections in by_document.items():
        if sections:
            shown = "; ".join(sections[:2])
            parts.append(f"{doc} ({shown})")
        else:
            parts.append(doc)
    best = max(float(h.get("similarity_score", h.get("score", 0.0)) or 0.0) for h in hits)
    return (
        f"The search returned {len(hits)} passage(s) from {', '.join(parts)}, "
        f"with a best similarity of {best:.3f}."
    )


def _citations_for_examined(hits: Sequence[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """Cite the chunks that were examined and rejected.

    A refusal that names the evidence it looked at is checkable: a reader can
    confirm the gap is real rather than take the refusal on trust. These are
    built from the retriever's own records, so they cannot be fabricated.
    """
    citations = []
    for hit in list(hits)[:limit]:
        text = hit.get("chunk_text") or hit.get("text") or ""
        citations.append(
            {
                "document": hit.get("document"),
                "section": hit.get("section"),
                "page": hit.get("page"),
                "chunk_id": hit.get("chunk_id"),
                "retrieval_score": hit.get("similarity_score", hit.get("score")),
                "excerpt": text.strip()[:300],
            }
        )
    return citations


def build_refusal(
    reason: str,
    *,
    query: str,
    hits: Sequence[dict[str, Any]] = (),
    threshold: float | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    """Build a refusal answer in the standard schema shape.

    `recommendation` always begins with the fixed `REFUSAL_MESSAGE`, so a refusal
    is detectable by string match as well as by `confidence`. What follows is the
    helpful part: what was found, what is missing, what would answer it.
    """
    found = _describe_evidence(hits, reason=reason)

    if reason == REFUSAL_NO_CHUNKS:
        missing = (
            "Nothing in the indexed guideline corpus was close enough to this question "
            "to return as evidence."
        )
        would_answer = (
            "Answering it would need a guideline section that addresses this topic directly. "
            "This corpus covers abdominal aortic aneurysm screening, surveillance, medical "
            "management, and repair (ESVS 2024, NICE NG156, SVS 2018, USPSTF 2019); a question "
            "outside that scope cannot be answered from it."
        )
    elif reason == REFUSAL_BELOW_THRESHOLD:
        floor = f"{threshold:.2f}" if threshold is not None else "the configured floor"
        missing = (
            f"Every passage returned scored below the {floor} similarity floor, which means "
            f"none of them is a close enough match to be used as evidence for a clinical "
            f"statement. The passages are topically adjacent at best."
        )
        would_answer = (
            "Answering it would need a guideline passage that addresses this question directly "
            "rather than the surrounding topic -- typically a numbered recommendation or a "
            "section heading that names the specific question being asked."
        )
    elif reason == REFUSAL_PATIENT_SPECIFIC:
        missing = (
            "This question asks for a decision about a specific person. "
            + (detail.capitalize() + ". " if detail else "")
            + "This system reports what guidelines state for the populations they describe; it "
            "does not diagnose, stage, manage or dose an individual patient, and no retrieved "
            "guideline text can substitute for that."
        )
        would_answer = (
            "A general form of the same question can be answered -- for example what the "
            "guidelines recommend for the relevant population, diameter band, or repair "
            "modality. The individual decision belongs with the treating clinician, in a "
            "multidisciplinary team discussion where indicated."
        )
    elif reason == REFUSAL_GUIDELINE_UNAVAILABLE:
        missing = (
            GUIDELINE_UNAVAILABLE_MESSAGE
            + " "
            + (detail.capitalize() + ". " if detail else "")
            + "Answering from a different edition would attribute recommendations to a "
            "document this system has never read, so the question is refused rather "
            "than answered from the editions that are indexed."
        )
        would_answer = (
            "The same question can be answered against an edition that is indexed. "
            "This corpus contains ESVS 2024, NICE NG156, SVS 2018 and USPSTF 2019; ask "
            "for one of those, or drop the edition from the question to be answered "
            "from whichever indexed guideline addresses it."
        )
    elif reason == REFUSAL_POTENTIAL_EMERGENCY:
        missing = (
            "This query describes symptoms that may indicate a medical emergency. "
            "This system cannot assess whether this is an emergency and is not a substitute "
            "for emergency medical evaluation."
        )
        would_answer = (
            "If you or someone else is experiencing sudden severe abdominal or back pain, "
            "collapse, or other signs of a possible vascular emergency: call emergency services "
            "immediately (999 / 112 / 911) or go to the nearest emergency department now. "
            "Do not wait for an online response."
        )
    else:
        missing = "The retrieved evidence does not support an answer to the question as asked."
        would_answer = (
            "Answering it would need a guideline passage that addresses this question directly."
        )

    recommendation = " ".join([REFUSAL_MESSAGE, found, missing, would_answer])

    # For a patient-specific or potential-emergency refusal the retrieved passages
    # are deliberately NOT cited: doing so would amount to offering guideline text
    # as an answer to a question about an individual acute situation.
    _NO_CITATIONS_REASONS = {REFUSAL_PATIENT_SPECIFIC, REFUSAL_POTENTIAL_EMERGENCY}
    citations: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    if reason not in _NO_CITATIONS_REASONS:
        citations = _citations_for_examined(hits)
        evidence = [
            {
                "claim": (
                    f"Examined but not usable as evidence for this question: "
                    f"{hit.get('document_id') or hit.get('document')}"
                    + (f", section {hit.get('section')}" if hit.get("section") else "")
                    + f" (similarity {float(hit.get('similarity_score', hit.get('score', 0.0)) or 0.0):.3f})."
                ),
                "chunk_id": hit.get("chunk_id"),
            }
            for hit in list(hits)[:5]
        ]

    return {
        "recommendation": recommendation,
        "supporting_evidence": evidence,
        "citations": citations,
        "confidence": CONFIDENCE_INSUFFICIENT,
        "disclaimer": DISCLAIMER,
    }

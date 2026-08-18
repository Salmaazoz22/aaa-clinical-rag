# -*- coding: utf-8 -*-
"""The structured answer contract.

One module owns the shape of a generated answer: the field names, the closed set
of confidence values, the fixed disclaimer, the standard refusal message, and the
JSON Schema handed to the model. `generation/prompts.py` renders it into the
prompt, `generation/validator.py` checks output against it, and
`generation/pipeline.py` assembles it. Nothing else may define these strings --
if the disclaimer lived in two places, an answer could ship with the wrong one.

Shape (Day 3 specification, slide 7):

    {
      "recommendation":      str,
      "supporting_evidence": [ {claim, chunk_id, excerpt?}, ... ],
      "citations":           [ {document, section, page, chunk_id,
                                retrieval_score, excerpt}, ... ],
      "confidence":          "High" | "Medium" | "Low" | "Insufficient Evidence",
      "disclaimer":          str
    }

`evidence_conflicts` is an OPTIONAL sixth field, and is a deliberate, documented
extension of the five fields the specification lists. Slide 5 allows comparing
evidence across chunks and the specification requires that disagreeing guidelines
be presented as both positions with their own citations; that requirement is only
*checkable* the way slide 9 asks for if the opposing positions are emitted as
structured data rather than left implicit in prose. It is never required, and an
answer without it is fully valid. See docs/generation.md.
"""
from __future__ import annotations

from typing import Any

# --- closed vocabularies ---------------------------------------------------

CONFIDENCE_HIGH = "High"
CONFIDENCE_MEDIUM = "Medium"
CONFIDENCE_LOW = "Low"
CONFIDENCE_INSUFFICIENT = "Insufficient Evidence"

CONFIDENCE_VALUES: tuple[str, ...] = (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_LOW,
    CONFIDENCE_INSUFFICIENT,
)

#: Confidence is an ordinal label, never a number. Day 3 slide 11: avoid exact
#: percentages unless there is a clear calculation behind them -- there is no such
#: calculation here, so a percentage would be false precision about how well a
#: dense retriever's top-5 supports a clinical claim.
CONFIDENCE_IS_ORDINAL = True

# --- fixed strings ---------------------------------------------------------

#: Attached verbatim to every answer, including refusals. Wording is aligned with
#: the repository-wide disclaimer in README.md / docs/HANDOFF.md.
DISCLAIMER = (
    "This is an evidence-retrieval prototype over four abdominal aortic aneurysm "
    "guidelines and is not clinically validated. It summarises retrieved guideline "
    "text only, does not provide patient-specific diagnosis, treatment or dosing, "
    "and must not be used to make clinical decisions. Verify every statement "
    "against the cited source document and apply clinical judgement."
)

#: The first sentence of every refusal. Kept fixed so a refusal is detectable by
#: string match as well as by `confidence`.
REFUSAL_MESSAGE = (
    "The retrieved guideline evidence is insufficient to answer this question."
)

REQUIRED_FIELDS: tuple[str, ...] = (
    "recommendation",
    "supporting_evidence",
    "citations",
    "confidence",
    "disclaimer",
)

OPTIONAL_FIELDS: tuple[str, ...] = ("evidence_conflicts",)

CITATION_FIELDS: tuple[str, ...] = (
    "document",
    "section",
    "page",
    "chunk_id",
    "retrieval_score",
    "excerpt",
)

EVIDENCE_FIELDS: tuple[str, ...] = ("claim", "chunk_id", "excerpt")


# --- refusal reasons -------------------------------------------------------
#
# Every refusal records which gate produced it, so the evaluation can check that
# a question refused for the *right* reason rather than merely refusing.

REFUSAL_NO_CHUNKS = "no_chunks_retrieved"
REFUSAL_BELOW_THRESHOLD = "all_scores_below_threshold"
REFUSAL_NOT_SPECIFIC = "evidence_not_specific_enough"
REFUSAL_PATIENT_SPECIFIC = "patient_specific_request"

REFUSAL_REASONS: tuple[str, ...] = (
    REFUSAL_NO_CHUNKS,
    REFUSAL_BELOW_THRESHOLD,
    REFUSAL_NOT_SPECIFIC,
    REFUSAL_PATIENT_SPECIFIC,
)


# --- JSON Schema handed to the model ---------------------------------------

def answer_json_schema() -> dict[str, Any]:
    """JSON Schema for the answer object.

    Used two ways: as the `json_schema` payload for providers that support
    structured outputs, and rendered into the prompt for those that do not.
    `additionalProperties` stays open because a reasoning model that adds a
    stray key should produce a *flagged* answer, not a hard API error that loses
    the whole response.
    """
    return {
        "type": "object",
        "required": list(REQUIRED_FIELDS),
        "properties": {
            "recommendation": {
                "type": "string",
                "description": (
                    "Short, direct answer drawn only from the retrieved chunks. No "
                    "patient-specific diagnosis, treatment or dosing. State disagreement "
                    "between guidelines explicitly rather than choosing one side."
                ),
            },
            "supporting_evidence": {
                "type": "array",
                "description": "One bullet per supported point, each bound to the chunk it came from.",
                "items": {
                    "type": "object",
                    "required": ["claim", "chunk_id"],
                    "properties": {
                        "claim": {"type": "string"},
                        "chunk_id": {
                            "type": "string",
                            "description": "chunk_id of the CONTEXT chunk this claim came from, copied verbatim.",
                        },
                        "excerpt": {
                            "type": "string",
                            "description": "Short verbatim quote from that chunk.",
                        },
                    },
                },
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": list(CITATION_FIELDS),
                    "properties": {
                        "document": {"type": "string"},
                        "section": {"type": ["string", "null"]},
                        "page": {"type": ["integer", "string", "null"]},
                        "chunk_id": {"type": "string"},
                        "retrieval_score": {
                            "type": ["number", "string"],
                            "description": "Copied verbatim from the CONTEXT chunk header. Never estimated.",
                        },
                        "excerpt": {"type": "string"},
                    },
                },
            },
            "confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
            "disclaimer": {"type": "string"},
            "evidence_conflicts": {
                "type": "array",
                "description": (
                    "Only when retrieved chunks disagree. One entry per disagreement, "
                    "with every position that appears in the CONTEXT."
                ),
                "items": {
                    "type": "object",
                    "required": ["topic", "positions"],
                    "properties": {
                        "topic": {"type": "string"},
                        "positions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["position", "chunk_ids"],
                                "properties": {
                                    "position": {"type": "string"},
                                    "source": {"type": ["string", "null"]},
                                    "chunk_ids": {"type": "array", "items": {"type": "string"}},
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def is_refusal(answer: dict[str, Any]) -> bool:
    """A refusal is defined by `confidence`, not by the prose.

    The pipeline and the model agree on one signal so that a refusal cannot be
    dressed up as an answer: `confidence == "Insufficient Evidence"`.
    """
    return str(answer.get("confidence") or "").strip() == CONFIDENCE_INSUFFICIENT

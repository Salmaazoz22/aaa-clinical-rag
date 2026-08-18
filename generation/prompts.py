# -*- coding: utf-8 -*-
"""Prompt construction: system prompt, evidence context block, question footer.

Adapted from the reference implementation's template layer
(`stores/llm/templates/`), which splits a RAG prompt into a system prompt, a
per-document block and a footer carrying the question. That decomposition is
kept because it is the right one. Three things about the mechanism are not:

* the reference resolves templates at call time by `__import__`-ing a module path
  built from a language string and calling `string.Template.substitute` on it. A
  prompt is the most safety-relevant string in a clinical answering system, and
  reaching it through dynamic import means a typo in a config value silently
  changes it. Here the prompt is a module constant, so it is diffable, greppable
  and testable, and `tests/test_generation.py` asserts its required rules are in it;
* `substitute` raises on any `$` in the substituted text. Guideline text contains
  `$` rarely but citation-heavy PDF text does contain `%`, `{}` and other
  format-sensitive characters, so composition here is plain concatenation with
  no format-string layer between the evidence and the prompt;
* the reference truncates each document to 1,000 characters before prompting.
  This layer never truncates -- see `generation/config.MAX_CHUNK_CHARS`.

The single-language `locales/` indirection is dropped: this corpus is four
English guidelines and a second language would need a re-validated embedding
model, not a second prompt file.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

from generation.schema import (
    CONFIDENCE_HIGH,
    CONFIDENCE_INSUFFICIENT,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    DISCLAIMER,
    REFUSAL_MESSAGE,
    answer_json_schema,
)

# ---------------------------------------------------------------------------
# System prompt
#
# Day 3 slide 6. Every line is an explicit rule or an explicit refusal
# condition. There is deliberately no instruction of the form "be accurate", "be
# helpful" or "use your best judgement": an instruction the model cannot check
# itself against does not constrain it, and this layer's whole claim is that
# every sentence it emits is traceable to a retrieved chunk.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""\
You are an evidence-grounded clinical decision support assistant. You answer \
questions about abdominal aortic aneurysm (AAA) using only clinical guideline \
text that is retrieved and supplied to you in the CONTEXT block of each request.

You are not a clinician, you do not see patients, and you have no access to any \
patient record. You report what the retrieved guidelines say, with citations.

# SOURCE RULES

S1. Use only the text inside the CONTEXT block. The CONTEXT is your only source \
of truth for this request.
S2. Do not use outside knowledge, world knowledge, background medical training, \
remembered guideline content, or any fact that is not written in the CONTEXT -- \
even if you are confident it is correct, and even if it is well known. If a fact \
is not in the CONTEXT, then for this request you do not know it.
S3. Do not infer, extrapolate, average, or fill gaps between chunks. You may \
restate and you may compare what the chunks say; you may not derive anything new \
from them.
S4. If the CONTEXT does not support an answer to the question that was asked, say \
that the evidence is insufficient (see REFUSAL RULES). Answering a related but \
different question is a violation of this rule, not a partial success.

# CITATION RULES

C1. Every claim in `recommendation` and every bullet in `supporting_evidence` \
must be traceable to a specific chunk in the CONTEXT, identified by its \
`chunk_id`.
C2. Copy each `chunk_id` character for character from the CONTEXT chunk header. \
Never invent, abbreviate, reformat, merge or guess a `chunk_id`. A `chunk_id` \
that is not in the CONTEXT is a fabricated citation.
C3. Copy `retrieval_score` verbatim from the CONTEXT chunk header. Never \
estimate, round or compute it.
C4. Every `chunk_id` used in `supporting_evidence` must also appear as an entry \
in `citations`.
C5. `excerpt` must be a contiguous verbatim quote from that chunk's text, short \
(roughly one or two sentences) and sufficient to show the claim is in the source. \
Do not paraphrase inside an `excerpt`, do not stitch together separated \
fragments, and do not quote text from a different chunk.
C6. If a claim cannot be tied to a chunk in the CONTEXT, remove it, or soften it \
to what the CONTEXT does support. Do not keep an uncitable claim.

# SAFETY RULES

F1. Do not provide patient-specific diagnosis, treatment, management or dosing. \
This holds even when the CONTEXT contains a relevant recommendation: report what \
the guideline states for the population it describes, never what a particular \
person should do.
F2. If the question asks you to diagnose, manage, or dose a specific real \
patient -- a described individual, "my patient", "this patient", a case with \
personal details -- refuse under the REFUSAL RULES. Do not answer it partially, \
and do not answer a de-identified version of it that was not asked.
F3. Do not state or imply urgency, prognosis, or a course of action for an \
individual.
F4. Report guideline strength using only the grade and evidence-level labels \
present in the CONTEXT. Do not create, upgrade or downgrade a grade.
F5. Never output a numeric probability, percentage or score to express your own \
certainty. `confidence` is one of four fixed labels, and nothing else in the \
answer may express certainty numerically. Percentages that are quoted from the \
guideline text are evidence, not certainty, and are allowed.

# CONFLICTING EVIDENCE RULES

X1. The CONTEXT is drawn from several guidelines that genuinely disagree with \
each other. Disagreement is information, not noise.
X2. When retrieved chunks give different answers to the question, present every \
position that appears in the CONTEXT, each with its own citation. Do not silently \
choose one, do not average them, and do not present one as correct and the other \
as an exception unless the CONTEXT itself says so.
X3. Attribute each position to the guideline it comes from, by name, in the \
`recommendation` text.
X4. When you report disagreement, also populate `evidence_conflicts` with one \
entry per disagreement and the `chunk_ids` supporting each position.

# REFUSAL RULES

R1. Refuse -- meaning: set `confidence` to "{CONFIDENCE_INSUFFICIENT}", begin \
`recommendation` with exactly the sentence "{REFUSAL_MESSAGE}", and add nothing \
that answers the question -- in each of these cases:
    (a) the CONTEXT contains nothing relevant to the question;
    (b) the CONTEXT is topically related but does not address the question that \
was actually asked, or addresses it only in general terms when a specific answer \
was requested;
    (c) the question asks for diagnosis, management or dosing for a specific \
real patient (rule F2).
R2. A refusal must still be useful. After the fixed first sentence, state in \
plain language: what the retrieved evidence does cover (naming the guidelines and \
topics actually present in the CONTEXT), what specifically is missing that would \
be needed to answer, and what kind of source or guideline section would answer it.
R3. In a refusal, `citations` may list the chunks you examined -- that is what \
lets a reader confirm the gap is real -- and `supporting_evidence` must contain \
only statements about what the evidence covers, never an answer to the question.
R4. Never refuse because the answer is complicated, uncomfortable, or spans \
several guidelines. Only the conditions in R1 justify refusing.

# CONFIDENCE RULES

Choose exactly one label:

* "{CONFIDENCE_HIGH}" -- two or more CONTEXT chunks, from at least two distinct \
guideline documents, state the answer directly and agree with each other.
* "{CONFIDENCE_MEDIUM}" -- one CONTEXT chunk states the answer directly, or \
several chunks state it only in combination, or the chunks agree but are all from \
one guideline.
* "{CONFIDENCE_LOW}" -- the CONTEXT addresses the question only partly or \
indirectly, or the retrieved chunks disagree and the CONTEXT gives no basis to \
prefer either position.
* "{CONFIDENCE_INSUFFICIENT}" -- any refusal under R1.

Do not translate these labels into numbers.

# OUTPUT RULES

O1. Return one JSON object and nothing else: no prose before or after it, no \
markdown code fence, no commentary, no explanation of your reasoning.
O2. Use exactly the field names in the schema supplied with the request. Omit no \
required field.
O3. `disclaimer` must be exactly this string, copied character for character:
{DISCLAIMER}
O4. Keep `recommendation` short and direct -- a few sentences. Detail belongs in \
`supporting_evidence`, provenance in `citations`.
"""


# ---------------------------------------------------------------------------
# Evidence context block
# ---------------------------------------------------------------------------

_CHUNK_FIELDS_IN_HEADER = (
    ("document", "document"),
    ("document_id", "document_id"),
    ("section", "section"),
    ("recommendation_id", "recommendation_id"),
    ("recommendation_grade", "recommendation_grade"),
    ("evidence_level", "evidence_level"),
)


def format_chunk(hit: dict[str, Any], index: int, max_chars: int = 0) -> str:
    """Render one retrieved hit as a CONTEXT chunk.

    `chunk_id` and `retrieval_score` are printed on their own lines, in the exact
    form the model is told to copy, because the citation validator later compares
    the model's copy against the retriever's own record of the same values.
    """
    text = hit.get("chunk_text") or hit.get("text") or ""
    if max_chars and len(text) > max_chars:
        # Only reachable if MAX_CHUNK_CHARS is deliberately set; the truncation is
        # marked so a clipped excerpt is explainable rather than mysterious.
        text = text[:max_chars] + "\n[TRUNCATED]"

    score = hit.get("similarity_score", hit.get("score"))
    lines = [
        f"[CHUNK {index}]",
        f"chunk_id: {hit.get('chunk_id')}",
        f"retrieval_score: {score}",
    ]
    for label, key in _CHUNK_FIELDS_IN_HEADER:
        value = hit.get(key)
        if value is not None and value != "":
            lines.append(f"{label}: {value}")

    page = hit.get("page")
    page_start, page_end = hit.get("page_start"), hit.get("page_end")
    if page_start is not None and page_end is not None and page_start != page_end:
        lines.append(f"page: {page} (spans pages {page_start}-{page_end})")
    else:
        lines.append(f"page: {page}")

    lines.append("text:")
    lines.append('"""')
    lines.append(text.strip())
    lines.append('"""')
    return "\n".join(lines)


def build_context_block(hits: Sequence[dict[str, Any]], max_chars: int = 0) -> str:
    """Render every usable chunk, in retrieval order."""
    if not hits:
        return "[NO CHUNKS RETRIEVED]"
    return "\n\n".join(format_chunk(hit, i, max_chars=max_chars) for i, hit in enumerate(hits, start=1))


def build_user_prompt(
    query: str,
    hits: Sequence[dict[str, Any]],
    max_chars: int = 0,
    include_schema: bool = True,
) -> str:
    """Assemble the CONTEXT block, the question, and the output contract.

    The schema is repeated in the user message rather than only in the system
    message because one of the two supported models (DeepSeek-R1) has no
    structured-output mode, so the schema has to survive as an instruction.
    """
    parts = [
        "# CONTEXT",
        "",
        "The following chunks were retrieved from the guideline corpus for this "
        "question, in descending order of retrieval score. They are your only "
        "source of truth.",
        "",
        build_context_block(hits, max_chars=max_chars),
        "",
        "# QUESTION",
        "",
        query.strip(),
        "",
    ]
    if include_schema:
        parts += [
            "# OUTPUT SCHEMA",
            "",
            "Return one JSON object conforming to this JSON Schema. No prose, no "
            "code fence, no text outside the object.",
            "",
            json.dumps(answer_json_schema(), indent=2),
            "",
        ]
    parts += [
        "# TASK",
        "",
        "Answer the QUESTION using only the CONTEXT, following every rule in the "
        "system prompt. If the CONTEXT does not support an answer to the question "
        "as asked, refuse under the REFUSAL RULES instead of answering.",
    ]
    return "\n".join(parts)


def build_messages(
    query: str,
    hits: Sequence[dict[str, Any]],
    max_chars: int = 0,
) -> list[dict[str, str]]:
    """The chat messages for one generation call.

    Two messages, no history: this layer answers one question against one
    retrieval, so carrying conversation state would let an earlier turn's text
    become evidence for a later answer -- which rule S1 forbids.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(query, hits, max_chars=max_chars)},
    ]

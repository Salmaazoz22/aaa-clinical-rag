# -*- coding: utf-8 -*-
"""Claim-to-evidence binding: the citation validator.

Day 3 slide 9 asks that the binding between a claim and its evidence be
*programmatically checkable*, not merely present in prose. The reference
implementation has no analogue -- it performs no validation of any kind on the
model's output, so a citation it prints is whatever the model wrote -- which is
why this module is designed from scratch against the specification.

What it checks, and why each check exists:

======================================  ==========================================
check                                   what it catches
======================================  ==========================================
every cited `chunk_id` was in the set   a fabricated citation: the model naming a
sent to the model for this query        document it was never shown, or inventing
                                        a plausible-looking chunk_id
every `supporting_evidence` bullet has  a claim presented as evidence-backed with
a matching entry in `citations`          nothing behind it (slide 9 says such a
                                        claim should have been removed or
                                        softened; this catches the cases where it
                                        was not)
`excerpt` really occurs in the cited    a quote that was paraphrased, stitched
chunk's text                            together, or taken from another chunk
`document` / `section` / `page` /       provenance rewritten in transit -- right
`retrieval_score` match the retriever's chunk, wrong attribution
own record
`confidence` is one of four labels,     false precision (slide 11)
and certainty is never numeric
a non-refusal carries citations         an answer with no evidence at all
a refusal carries the fixed message     a refusal dressed up as an answer
every substantive sentence in the       narrative prose that introduces facts
`recommendation` prose maps to at       beyond what any validated citation
least one supporting-evidence excerpt   excerpt covers (lexical overlap check)
======================================  ==========================================

Two rules govern the design:

**Nothing is silently dropped or repaired.** Every problem becomes a `Finding`
with a code, a severity and enough context to locate it. The pipeline attaches
the report to the answer; it never quietly deletes a bad citation, because an
answer that was wrong and got tidied up is indistinguishable from one that was
right.

**Errors versus warnings.** An *error* means a hard rule from the system prompt
was broken, so the answer is not trustworthy as it stands (`ok` is False). A
*warning* means something is off but the claim-to-evidence chain still holds --
provenance the model rewrote, a quote it reformatted. Both are reported; only
errors fail the report.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Sequence

from generation.schema import (
    CITATION_FIELDS,
    CONFIDENCE_VALUES,
    DISCLAIMER,
    REFUSAL_MESSAGE,
    REQUIRED_FIELDS,
    is_refusal,
)

# --- finding codes ---------------------------------------------------------
# Named constants so tests and downstream artifacts reference a code rather than
# a message string.

E_MISSING_FIELD = "missing_field"
E_WRONG_TYPE = "wrong_type"
E_EMPTY_RECOMMENDATION = "empty_recommendation"
E_HALLUCINATED_CITATION = "hallucinated_citation"
E_HALLUCINATED_EVIDENCE_CHUNK = "hallucinated_evidence_chunk_id"
E_HALLUCINATED_CONFLICT_CHUNK = "hallucinated_conflict_chunk_id"
E_UNCITED_CLAIM = "uncited_claim"
E_CITATION_MISSING_FIELD = "citation_missing_field"
E_INVALID_CONFIDENCE = "invalid_confidence"
E_NUMERIC_CONFIDENCE = "numeric_confidence"
E_ANSWER_WITHOUT_CITATIONS = "answer_without_citations"
E_REFUSAL_MESSAGE_MISSING = "refusal_message_missing"

W_DISCLAIMER_NOT_CANONICAL = "disclaimer_not_canonical"
W_CITATION_METADATA_MISMATCH = "citation_metadata_mismatch"
W_RETRIEVAL_SCORE_MISMATCH = "retrieval_score_mismatch"
W_EXCERPT_NOT_IN_CHUNK = "excerpt_not_in_chunk"
W_EXCERPT_STITCHED = "excerpt_stitched"
W_EMPTY_EXCERPT = "empty_excerpt"
W_DUPLICATE_CITATION = "duplicate_citation"
W_CONFLICT_SINGLE_POSITION = "conflict_single_position"
W_NUMERIC_CERTAINTY_PROSE = "numeric_certainty_in_prose"
W_RECOMMENDATION_UNSUPPORTED_SENTENCE = "recommendation_unsupported_sentence"

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

#: Tolerance when comparing the model's copy of a retrieval score against the
#: retriever's own value. Scores are stored to six decimal places and the model is
#: told to copy them verbatim, so this allows for rounding and nothing more.
SCORE_TOLERANCE = 1e-3

#: Minimum length for an excerpt to be worth checking against the source. Below
#: this, a substring test is meaningless (any two texts share short fragments).
MIN_EXCERPT_CHARS = 12

#: Minimum number of content tokens a recommendation sentence must share with
#: at least one supporting-evidence excerpt for the sentence to count as
#: grounded. Set deliberately low (2) because a clinical sentence like
#: "repair is recommended at 55 mm" shares only a handful of content words with
#: an excerpt that may phrase the same fact differently. Raising this risks false
#: positives on short sentences; lowering it to 1 allows trivially short overlaps.
MIN_GROUNDING_TOKENS = 2

_ELLIPSIS = re.compile(r"\s*(?:\.\.\.+|…)\s*")

_NUMERIC_CERTAINTY = (
    re.compile(r"(confiden\w+|certain\w+|probabilit\w+|likelihood)[^.?!]{0,24}?\d+(?:\.\d+)?\s*%", re.I),
    re.compile(r"\d+(?:\.\d+)?\s*%\s*(?:confiden\w+|certain\w+|probabilit\w+|likelihood)", re.I),
    re.compile(r"(confiden\w+|certain\w+)\s*(?:score|level|rating)?\s*[:=]\s*\d", re.I),
)

# Text normalisation for excerpt matching. Mirrors the intent of the frozen
# evaluation's normalisation (eval/README.md): the source PDFs render ligatures
# and comparison operators as glyphs, so a verbatim quote of "five" or ">= 55 mm"
# will not match the stored text byte-for-byte. Implemented here rather than
# imported, because the evaluation's copy is frozen scoring code and must not
# grow a second caller.
#
# Normalisation order (applied inside `normalise_for_match` only -- never to
# stored text or model output):
#   1. NFKC: decomposes compatibility forms (e.g. fi-ligature -> fi, fullwidth).
#   2. Ligature map: catches any PDF-specific ligatures NFKC did not handle.
#   3. Smart-quote map: curly/typographic quotes -> straight ASCII equivalents.
#   4. Dash/hyphen map: every dash/hyphen Unicode variant -> ASCII hyphen (-).
#      Previously these were mapped to a space, which caused "CTA-based" with
#      an en-dash to compare as "CTA based" -- a false mismatch.
#   5. Miscellaneous glyphs (>=, <=, etc.) -> space.
#   6. Collapse all whitespace (including non-breaking, thin, zero-width)
#      to a single ASCII space.
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st",
}
_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "´": "'", "`": "'",
}
# All common dash/hyphen variants -> ASCII hyphen.
# Kept separate from _GLYPH_TO_SPACE so that "CTA-based" (with a typographic
# dash) matches "CTA-based" rather than becoming "CTA based" (wrong form).
_DASHES_TO_HYPHEN = (
    "‐",  # HYPHEN
    "‑",  # NON-BREAKING HYPHEN
    "‒",  # FIGURE DASH
    "–",  # EN DASH
    "—",  # EM DASH
    "―",  # HORIZONTAL BAR
    "−",  # MINUS SIGN
    "﹘",  # SMALL EM DASH
    "﹣",  # SMALL HYPHEN-MINUS
    "－",  # FULLWIDTH HYPHEN-MINUS
)
# Glyphs that should become a space rather than a hyphen.
_GLYPH_TO_SPACE = (
    "≥",  # >=
    "≤",  # <=
    "‡",  # double dagger
    "†",  # dagger
    " ",  # NO-BREAK SPACE
)


def normalise_for_match(text: Any) -> str:
    """Normalise text for substring comparison. Never used to alter output.

    Applies NFKC, ligature expansion, quote straightening, dash-to-hyphen
    mapping, and whitespace collapsing so that typographic variants of the
    same word compare as equal. The original `text` is never modified.
    """
    if text is None:
        return ""
    s = str(text)
    # Step 1: Quotes and ligatures BEFORE NFKC.
    # U+00B4 (ACUTE ACCENT ´) is decomposed by NFKC into SPACE+U+0301, so it
    # must be replaced before normalisation or it will never match the key.
    for src, dst in _QUOTES.items():
        s = s.replace(src, dst)
    for src, dst in _LIGATURES.items():
        s = s.replace(src, dst)
    # Step 2: NFKC handles remaining compatibility decompositions (fullwidth
    # characters, remaining ligatures, compatibility superscripts, etc.).
    s = unicodedata.normalize("NFKC", s)
    # Step 3: Dash/hyphen variants -> ASCII hyphen so that "CTA-based" (with
    # any typographic dash) normalises to "CTA-based".
    for dash in _DASHES_TO_HYPHEN:
        s = s.replace(dash, "-")
    # Step 4: Miscellaneous non-alphanumeric glyphs -> space.
    for glyph in _GLYPH_TO_SPACE:
        s = s.replace(glyph, " ")
    # Step 5: Case-fold and collapse all whitespace (including \t, \n, thin
    # spaces, zero-width spaces, etc.) to a single ASCII space.
    s = s.lower()
    return re.sub(r"[\s\u200b\u200c\u200d\ufeff]+", " ", s).strip()


# --- findings --------------------------------------------------------------

@dataclass(frozen=True)
class Finding:
    """One problem found in an answer."""

    code: str
    severity: str
    message: str
    location: str | None = None
    chunk_id: str | None = None
    expected: Any = None
    actual: Any = None

    @property
    def is_error(self) -> bool:
        return self.severity == SEVERITY_ERROR

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "severity": self.severity, "message": self.message}
        if self.location is not None:
            out["location"] = self.location
        if self.chunk_id is not None:
            out["chunk_id"] = self.chunk_id
        if self.expected is not None:
            out["expected"] = self.expected
        if self.actual is not None:
            out["actual"] = self.actual
        return out


@dataclass
class ValidationReport:
    """The outcome of validating one answer against one retrieval."""

    findings: list[Finding] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)
    hallucinated_chunk_ids: list[str] = field(default_factory=list)
    uncited_claims: list[dict[str, Any]] = field(default_factory=list)
    is_refusal: bool = False

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.is_error]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if not f.is_error]

    @property
    def ok(self) -> bool:
        """True when no hard rule was broken."""
        return not self.errors

    @property
    def codes(self) -> list[str]:
        return [f.code for f in self.findings]

    def has(self, code: str) -> bool:
        return any(f.code == code for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "is_refusal": self.is_refusal,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "codes": self.codes,
            "findings": [f.to_dict() for f in self.findings],
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "cited_chunk_ids": list(self.cited_chunk_ids),
            "hallucinated_chunk_ids": list(self.hallucinated_chunk_ids),
            "uncited_claims": list(self.uncited_claims),
        }


# --- helpers ---------------------------------------------------------------

def _index_hits(hits: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map chunk_id -> hit for the chunks that were sent to the model."""
    return {str(h.get("chunk_id")): h for h in hits if h.get("chunk_id")}


def _score_of(hit: dict[str, Any]) -> float | None:
    value = hit.get("similarity_score", hit.get("score"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def _pages_in(value: Any) -> set[int]:
    if value is None:
        return set()
    return {int(n) for n in re.findall(r"\d+", str(value))}


def _loose_text_match(claimed: Any, truth: Any) -> bool:
    """True when the model's string plausibly names the same thing as the truth.

    Substring in either direction, because the retriever exposes both a long
    `document` name and a short `document_id`, and a section title may be quoted
    in part.
    """
    a, b = normalise_for_match(claimed), normalise_for_match(truth)
    if not a or not b:
        return False
    return a == b or a in b or b in a


# --- recommendation grounding (lexical) -----------------------------------

# Stop words to exclude from the token overlap calculation. These words carry
# no discriminating content so they must not count as grounding evidence.
# The list targets clinical/academic prose specifically; it is not exhaustive
# but covers the most frequent false-positive sources.
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "has", "have", "had", "do", "does", "did", "not", "no", "nor",
    "that", "this", "these", "those", "it", "its", "which", "who", "whom",
    "when", "where", "how", "what", "if", "than", "then", "so", "also",
    "both", "each", "any", "all", "more", "most", "such", "other",
    "may", "should", "would", "could", "will", "can", "must", "shall",
    "there", "their", "they", "them", "we", "our", "he", "she", "his",
    "her", "i", "my", "you", "your", "been", "being", "about", "above",
    "after", "although", "among", "because", "between", "during",
    "however", "including", "rather", "therefore", "thus", "while",
    "whereas", "whether", "per", "via", "within", "without", "based",
    "note", "see", "given", "further", "nevertheless", "nonetheless",
})

# Sentences consisting *only* of connector / transitional content do not
# carry factual claims and need no grounding. A sentence is treated as a
# "connector" when it has no content tokens after stop-word removal.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _content_tokens(text: str) -> frozenset[str]:
    """Return the significant lower-cased word-tokens of *text*.

    Keeps only tokens that are at least two characters long, are not pure
    stop words, and contain at least one alphanumeric character.  Numbers
    such as "55" and "5.5" are deliberately kept because they are the most
    specific signal in clinical sentences (threshold values, page ranges).
    """
    normed = normalise_for_match(text)  # already lower-cased
    tokens = re.findall(r"[a-z0-9][a-z0-9./-]*", normed)
    return frozenset(
        t for t in tokens
        if len(t) >= 2 and t not in _STOP_WORDS
    )


def _recommendation_grounding_findings(
    recommendation: str,
    grounding_excerpts: list[str],
    location: str = "recommendation",
) -> list[Finding]:
    """Check that every substantive sentence in *recommendation* is traceable
    to at least one entry in *grounding_excerpts*.

    Approach (lexical only -- no embedding models):
      1. Split the recommendation into sentences.
      2. For each sentence, extract its content tokens (numbers, nouns, key
         terms) by removing stop words and very short tokens.
      3. If the sentence has no content tokens it is a connector/transition
         with nothing to ground, so it is skipped.
      4. For each non-trivial sentence, compute the token overlap against each
         grounding excerpt.  The sentence is considered grounded if at least
         one excerpt shares >= MIN_GROUNDING_TOKENS content tokens with it.
      5. Sentences that fail grounding get W_RECOMMENDATION_UNSUPPORTED_SENTENCE.

    Limitations (flagged for future work, not implemented here):
      - Token overlap cannot detect logical contradiction -- a sentence that
        *directly contradicts* an excerpt would still pass this check.
      - Paraphrased claims that use entirely different vocabulary will be
        flagged as unsupported even when they convey the same fact.
      - Multi-sentence run-ons not terminated with sentence-ending punctuation
        are treated as a single sentence.
    """
    findings: list[Finding] = []
    if not recommendation or not recommendation.strip():
        return findings

    # Pre-compute excerpt token sets once for all sentences.
    excerpt_token_sets: list[frozenset[str]] = [
        _content_tokens(ex) for ex in grounding_excerpts if ex and ex.strip()
    ]
    # Nothing to check against: skip the whole check rather than flag everything.
    if not excerpt_token_sets:
        return findings

    sentences = _SENTENCE_SPLIT.split(recommendation.strip())
    for sent_idx, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
        sent_tokens = _content_tokens(sentence)
        if len(sent_tokens) < MIN_GROUNDING_TOKENS:
            # Connector sentence: too short to carry a falsifiable claim.
            continue

        grounded = any(
            len(sent_tokens & ex_tokens) >= MIN_GROUNDING_TOKENS
            for ex_tokens in excerpt_token_sets
        )
        if not grounded:
            findings.append(
                Finding(
                    W_RECOMMENDATION_UNSUPPORTED_SENTENCE,
                    SEVERITY_WARNING,
                    "a sentence in 'recommendation' does not share enough content tokens "
                    "with any supporting-evidence excerpt; it may introduce a claim that "
                    "goes beyond the cited evidence (lexical check, needs human review)",
                    location=f"{location}[sentence {sent_idx}]",
                    actual=sentence[:200],
                )
            )
    return findings


def _excerpt_findings(
    excerpt: Any, hit: dict[str, Any], location: str, chunk_id: str
) -> list[Finding]:
    """Check that an excerpt is really a quote from the cited chunk."""
    if excerpt is None or str(excerpt).strip() == "":
        return [
            Finding(
                W_EMPTY_EXCERPT,
                SEVERITY_WARNING,
                "citation carries no excerpt, so the quote cannot be checked against the source",
                location=location,
                chunk_id=chunk_id,
            )
        ]

    source = normalise_for_match(hit.get("chunk_text") or hit.get("text") or "")
    text = str(excerpt)
    findings: list[Finding] = []

    parts = [p for p in _ELLIPSIS.split(text) if p.strip()]
    stitched = len(parts) > 1
    if stitched:
        findings.append(
            Finding(
                W_EXCERPT_STITCHED,
                SEVERITY_WARNING,
                "excerpt joins non-contiguous fragments with an ellipsis; each fragment is "
                "checked separately, but rule C5 asks for a contiguous quote",
                location=location,
                chunk_id=chunk_id,
            )
        )

    for part in parts:
        needle = normalise_for_match(part)
        if len(needle) < MIN_EXCERPT_CHARS:
            continue
        if needle not in source:
            findings.append(
                Finding(
                    W_EXCERPT_NOT_IN_CHUNK,
                    SEVERITY_WARNING,
                    "excerpt does not occur in the cited chunk's text (after normalising "
                    "ligatures, quotes and whitespace), so the quote is not verbatim",
                    location=location,
                    chunk_id=chunk_id,
                    actual=part.strip()[:160],
                )
            )
    return findings


def _metadata_findings(
    citation: dict[str, Any], hit: dict[str, Any], location: str, chunk_id: str
) -> list[Finding]:
    """Compare the model's provenance fields against the retriever's record."""
    findings: list[Finding] = []

    claimed_doc = citation.get("document")
    if claimed_doc is not None and not (
        _loose_text_match(claimed_doc, hit.get("document"))
        or _loose_text_match(claimed_doc, hit.get("document_id"))
    ):
        findings.append(
            Finding(
                W_CITATION_METADATA_MISMATCH,
                SEVERITY_WARNING,
                "cited document does not match the retrieved chunk's document",
                location=f"{location}.document",
                chunk_id=chunk_id,
                expected=hit.get("document_id") or hit.get("document"),
                actual=claimed_doc,
            )
        )

    claimed_section = citation.get("section")
    true_section = hit.get("section")
    if claimed_section and true_section and not _loose_text_match(claimed_section, true_section):
        findings.append(
            Finding(
                W_CITATION_METADATA_MISMATCH,
                SEVERITY_WARNING,
                "cited section does not match the retrieved chunk's section title",
                location=f"{location}.section",
                chunk_id=chunk_id,
                expected=true_section,
                actual=claimed_section,
            )
        )

    claimed_pages = _pages_in(citation.get("page"))
    if claimed_pages:
        start, end = hit.get("page_start"), hit.get("page_end")
        if start is None or end is None:
            true_pages = _pages_in(hit.get("page"))
        else:
            true_pages = set(range(int(start), int(end) + 1))
        if true_pages and not (claimed_pages & true_pages):
            findings.append(
                Finding(
                    W_CITATION_METADATA_MISMATCH,
                    SEVERITY_WARNING,
                    "cited page is outside the retrieved chunk's page span",
                    location=f"{location}.page",
                    chunk_id=chunk_id,
                    expected=sorted(true_pages),
                    actual=citation.get("page"),
                )
            )

    claimed_score = _as_float(citation.get("retrieval_score"))
    true_score = _score_of(hit)
    if claimed_score is not None and true_score is not None:
        if abs(claimed_score - true_score) > SCORE_TOLERANCE:
            findings.append(
                Finding(
                    W_RETRIEVAL_SCORE_MISMATCH,
                    SEVERITY_WARNING,
                    "cited retrieval_score was not copied from the context; it was altered "
                    "or estimated (rule C3)",
                    location=f"{location}.retrieval_score",
                    chunk_id=chunk_id,
                    expected=round(true_score, 6),
                    actual=citation.get("retrieval_score"),
                )
            )
    return findings


# --- the validator ---------------------------------------------------------

def validate_answer(
    answer: Any,
    retrieved: Sequence[dict[str, Any]],
) -> ValidationReport:
    """Validate one answer against the chunks that were sent to the model.

    `retrieved` must be exactly the chunks that appeared in the CONTEXT block for
    this query -- the post-threshold set, not the raw top-K. A citation to a chunk
    that was retrieved but filtered out is still a citation to something the model
    never saw, so it counts as fabricated.
    """
    by_id = _index_hits(retrieved)
    report = ValidationReport(retrieved_chunk_ids=list(by_id))

    if not isinstance(answer, dict):
        report.findings.append(
            Finding(
                E_WRONG_TYPE,
                SEVERITY_ERROR,
                f"answer must be a JSON object, got {type(answer).__name__}",
            )
        )
        return report

    report.is_refusal = is_refusal(answer)

    # --- required fields ---------------------------------------------------
    for name in REQUIRED_FIELDS:
        if name not in answer:
            report.findings.append(
                Finding(E_MISSING_FIELD, SEVERITY_ERROR, f"required field {name!r} is missing", location=name)
            )

    recommendation = answer.get("recommendation")
    if "recommendation" in answer:
        if not isinstance(recommendation, str):
            report.findings.append(
                Finding(
                    E_WRONG_TYPE,
                    SEVERITY_ERROR,
                    f"'recommendation' must be a string, got {type(recommendation).__name__}",
                    location="recommendation",
                )
            )
        elif not recommendation.strip():
            report.findings.append(
                Finding(E_EMPTY_RECOMMENDATION, SEVERITY_ERROR, "'recommendation' is empty", location="recommendation")
            )

    # --- confidence --------------------------------------------------------
    confidence = answer.get("confidence")
    if "confidence" in answer:
        value = str(confidence).strip() if confidence is not None else ""
        if value not in CONFIDENCE_VALUES:
            report.findings.append(
                Finding(
                    E_INVALID_CONFIDENCE,
                    SEVERITY_ERROR,
                    "'confidence' must be one of the four fixed labels",
                    location="confidence",
                    expected=list(CONFIDENCE_VALUES),
                    actual=confidence,
                )
            )
            if _as_float(value) is not None or "%" in value:
                report.findings.append(
                    Finding(
                        E_NUMERIC_CONFIDENCE,
                        SEVERITY_ERROR,
                        "'confidence' expresses certainty numerically; slide 11 forbids a "
                        "percentage without a calculation behind it",
                        location="confidence",
                        actual=confidence,
                    )
                )

    # Numeric certainty anywhere in the prose. This is a WARNING, not an error,
    # and deliberately so: the guidelines themselves quote percentages ("rupture
    # is fatal in > 80% of cases", "the five year rupture risk is ..."), and a
    # quoted statistic is evidence rather than a claim about the model's own
    # certainty. Rule F5 permits the former and forbids the latter, and the two
    # cannot be separated reliably by pattern, so this reports for a human to
    # judge instead of failing the answer. The `confidence` field itself is
    # unambiguous and stays an error.
    prose = [("recommendation", recommendation if isinstance(recommendation, str) else "")]
    for i, bullet in enumerate(answer.get("supporting_evidence") or []):
        if isinstance(bullet, dict) and isinstance(bullet.get("claim"), str):
            prose.append((f"supporting_evidence[{i}].claim", bullet["claim"]))
    for location, text in prose:
        if any(p.search(text) for p in _NUMERIC_CERTAINTY):
            report.findings.append(
                Finding(
                    W_NUMERIC_CERTAINTY_PROSE,
                    SEVERITY_WARNING,
                    "a percentage appears next to a word about certainty; if this expresses "
                    "the answer's own confidence it violates rule F5, and if it is quoted "
                    "from the guideline it is fine -- needs a human read",
                    location=location,
                )
            )

    # --- disclaimer --------------------------------------------------------
    disclaimer = answer.get("disclaimer")
    if "disclaimer" in answer and normalise_for_match(disclaimer) != normalise_for_match(DISCLAIMER):
        report.findings.append(
            Finding(
                W_DISCLAIMER_NOT_CANONICAL,
                SEVERITY_WARNING,
                "disclaimer is not the canonical string; the pipeline replaces it with the "
                "canonical text and records that it did",
                location="disclaimer",
                actual=(str(disclaimer)[:120] if disclaimer is not None else None),
            )
        )

    # --- refusal consistency ----------------------------------------------
    if report.is_refusal and isinstance(recommendation, str):
        if not recommendation.strip().startswith(REFUSAL_MESSAGE):
            report.findings.append(
                Finding(
                    E_REFUSAL_MESSAGE_MISSING,
                    SEVERITY_ERROR,
                    "confidence is 'Insufficient Evidence' but 'recommendation' does not begin "
                    "with the standard refusal message (rule R1)",
                    location="recommendation",
                    expected=REFUSAL_MESSAGE,
                    actual=recommendation.strip()[:120],
                )
            )

    # --- citations ---------------------------------------------------------
    citations = answer.get("citations")
    if "citations" in answer and not isinstance(citations, list):
        report.findings.append(
            Finding(
                E_WRONG_TYPE,
                SEVERITY_ERROR,
                f"'citations' must be a list, got {type(citations).__name__}",
                location="citations",
            )
        )
        citations = []
    citations = citations or []

    seen: set[str] = set()
    for i, citation in enumerate(citations):
        location = f"citations[{i}]"
        if not isinstance(citation, dict):
            report.findings.append(
                Finding(
                    E_WRONG_TYPE,
                    SEVERITY_ERROR,
                    f"citation must be an object, got {type(citation).__name__}",
                    location=location,
                )
            )
            continue

        for name in CITATION_FIELDS:
            if name not in citation:
                report.findings.append(
                    Finding(
                        E_CITATION_MISSING_FIELD,
                        SEVERITY_ERROR,
                        f"citation is missing required field {name!r}",
                        location=location,
                    )
                )

        chunk_id = citation.get("chunk_id")
        chunk_id = str(chunk_id).strip() if chunk_id is not None else ""
        if not chunk_id:
            continue
        report.cited_chunk_ids.append(chunk_id)

        if chunk_id in seen:
            report.findings.append(
                Finding(
                    W_DUPLICATE_CITATION,
                    SEVERITY_WARNING,
                    "the same chunk is cited more than once",
                    location=location,
                    chunk_id=chunk_id,
                )
            )
        seen.add(chunk_id)

        hit = by_id.get(chunk_id)
        if hit is None:
            # The core check of slide 9: this chunk_id was never in the context.
            report.hallucinated_chunk_ids.append(chunk_id)
            report.findings.append(
                Finding(
                    E_HALLUCINATED_CITATION,
                    SEVERITY_ERROR,
                    "cited chunk_id was not among the chunks sent to the model for this "
                    "query, so the citation is fabricated",
                    location=location,
                    chunk_id=chunk_id,
                    expected=sorted(by_id),
                )
            )
            continue

        report.findings.extend(_metadata_findings(citation, hit, location, chunk_id))
        report.findings.extend(_excerpt_findings(citation.get("excerpt"), hit, location, chunk_id))

    cited = set(report.cited_chunk_ids)

    # --- supporting evidence ----------------------------------------------
    evidence = answer.get("supporting_evidence")
    if "supporting_evidence" in answer and not isinstance(evidence, list):
        report.findings.append(
            Finding(
                E_WRONG_TYPE,
                SEVERITY_ERROR,
                f"'supporting_evidence' must be a list, got {type(evidence).__name__}",
                location="supporting_evidence",
            )
        )
        evidence = []
    evidence = evidence or []

    for i, bullet in enumerate(evidence):
        location = f"supporting_evidence[{i}]"
        if not isinstance(bullet, dict):
            report.findings.append(
                Finding(
                    E_WRONG_TYPE,
                    SEVERITY_ERROR,
                    f"supporting_evidence entry must be an object, got {type(bullet).__name__}",
                    location=location,
                )
            )
            continue

        claim = bullet.get("claim")
        chunk_id = bullet.get("chunk_id")
        chunk_id = str(chunk_id).strip() if chunk_id is not None else ""

        if not chunk_id:
            # A bullet with no chunk_id at all is the strongest form of uncited
            # claim: nothing binds it to any evidence.
            report.uncited_claims.append({"location": location, "claim": claim, "chunk_id": None})
            report.findings.append(
                Finding(
                    E_UNCITED_CLAIM,
                    SEVERITY_ERROR,
                    "supporting_evidence bullet names no chunk_id, so the claim is not bound "
                    "to any evidence (rules C1 and C6)",
                    location=location,
                )
            )
            continue

        if chunk_id not in by_id:
            report.hallucinated_chunk_ids.append(chunk_id)
            report.findings.append(
                Finding(
                    E_HALLUCINATED_EVIDENCE_CHUNK,
                    SEVERITY_ERROR,
                    "supporting_evidence names a chunk_id that was not sent to the model for "
                    "this query",
                    location=location,
                    chunk_id=chunk_id,
                )
            )

        if chunk_id not in cited:
            report.uncited_claims.append({"location": location, "claim": claim, "chunk_id": chunk_id})
            report.findings.append(
                Finding(
                    E_UNCITED_CLAIM,
                    SEVERITY_ERROR,
                    "supporting_evidence bullet has no matching entry in 'citations' "
                    "(rule C4); flagged, not removed",
                    location=location,
                    chunk_id=chunk_id,
                )
            )

        if isinstance(bullet.get("excerpt"), str) and bullet["excerpt"].strip() and chunk_id in by_id:
            report.findings.extend(
                _excerpt_findings(bullet["excerpt"], by_id[chunk_id], f"{location}.excerpt", chunk_id)
            )

    # --- recommendation prose grounding -----------------------------------
    # Only run when: (a) the answer is not a refusal, (b) there are valid
    # citations (no point checking grounding when there's nothing to ground
    # against), and (c) recommendation is a non-empty string.
    if (
        not report.is_refusal
        and isinstance(recommendation, str)
        and recommendation.strip()
        and not report.has(E_MISSING_FIELD)  # skip if structural check already failed
    ):
        # Collect excerpts only from evidence bullets whose chunk_id is not
        # hallucinated, so a fabricated citation cannot "cover" a prose sentence.
        valid_evidence_excerpts: list[str] = []
        for bullet in evidence:
            if not isinstance(bullet, dict):
                continue
            b_chunk_id = str(bullet.get("chunk_id") or "").strip()
            if not b_chunk_id or b_chunk_id not in by_id:
                continue  # hallucinated or missing -- not grounding material
            excerpt_text = bullet.get("excerpt")
            if isinstance(excerpt_text, str) and excerpt_text.strip():
                valid_evidence_excerpts.append(excerpt_text)

        # Also include citation-level excerpts (citations list can carry excerpts
        # independently of supporting_evidence bullets).
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            c_chunk_id = str(citation.get("chunk_id") or "").strip()
            if not c_chunk_id or c_chunk_id not in by_id:
                continue
            c_excerpt = citation.get("excerpt")
            if isinstance(c_excerpt, str) and c_excerpt.strip():
                valid_evidence_excerpts.append(c_excerpt)

        if valid_evidence_excerpts:
            report.findings.extend(
                _recommendation_grounding_findings(
                    recommendation, valid_evidence_excerpts
                )
            )

    # --- an answer must be cited at all -----------------------------------
    if not report.is_refusal and not citations:
        report.findings.append(
            Finding(
                E_ANSWER_WITHOUT_CITATIONS,
                SEVERITY_ERROR,
                "a non-refusal answer carries no citations; every recommendation must be "
                "cited (rule C1)",
                location="citations",
            )
        )

    # --- conflicts (optional field) ---------------------------------------
    for i, conflict in enumerate(answer.get("evidence_conflicts") or []):
        location = f"evidence_conflicts[{i}]"
        if not isinstance(conflict, dict):
            report.findings.append(
                Finding(
                    E_WRONG_TYPE,
                    SEVERITY_ERROR,
                    f"evidence_conflicts entry must be an object, got {type(conflict).__name__}",
                    location=location,
                )
            )
            continue
        positions = conflict.get("positions") or []
        if not isinstance(positions, list) or len(positions) < 2:
            report.findings.append(
                Finding(
                    W_CONFLICT_SINGLE_POSITION,
                    SEVERITY_WARNING,
                    "a reported conflict lists fewer than two positions, so it does not show "
                    "a disagreement (rule X2)",
                    location=location,
                )
            )
        for j, position in enumerate(positions if isinstance(positions, list) else []):
            if not isinstance(position, dict):
                continue
            for k, chunk_id in enumerate(position.get("chunk_ids") or []):
                chunk_id = str(chunk_id).strip()
                if chunk_id and chunk_id not in by_id:
                    report.hallucinated_chunk_ids.append(chunk_id)
                    report.findings.append(
                        Finding(
                            E_HALLUCINATED_CONFLICT_CHUNK,
                            SEVERITY_ERROR,
                            "a conflict position cites a chunk_id that was not sent to the "
                            "model for this query",
                            location=f"{location}.positions[{j}].chunk_ids[{k}]",
                            chunk_id=chunk_id,
                        )
                    )

    # de-duplicate while preserving order
    report.hallucinated_chunk_ids = list(dict.fromkeys(report.hallucinated_chunk_ids))
    return report


# --- authoritative citation view -------------------------------------------

def resolve_citations(
    answer: dict[str, Any],
    retrieved: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rebuild the citation list from the retriever's own records.

    Returned *alongside* the model's own `citations`, never in place of them: the
    validator has already compared the two and reported every difference, so this
    is the audited view, not a silent correction. A citation whose `chunk_id` was
    fabricated cannot be resolved and is returned marked as such.
    """
    by_id = _index_hits(retrieved)
    resolved: list[dict[str, Any]] = []
    for citation in answer.get("citations") or []:
        if not isinstance(citation, dict):
            continue
        chunk_id = str(citation.get("chunk_id") or "").strip()
        hit = by_id.get(chunk_id)
        if hit is None:
            resolved.append({"chunk_id": chunk_id or None, "resolved": False, "reason": "chunk_id not in retrieved set"})
            continue
        resolved.append(
            {
                "chunk_id": chunk_id,
                "resolved": True,
                "document": hit.get("document"),
                "document_id": hit.get("document_id"),
                "section": hit.get("section"),
                "page": hit.get("page"),
                "page_start": hit.get("page_start"),
                "page_end": hit.get("page_end"),
                "retrieval_score": _score_of(hit),
                "rank": hit.get("rank"),
                "recommendation_id": hit.get("recommendation_id"),
                "recommendation_grade": hit.get("recommendation_grade"),
                "evidence_level": hit.get("evidence_level"),
                "source_file": hit.get("source_file"),
                "model_excerpt": citation.get("excerpt"),
            }
        )
    return resolved


def documents_cited(answer: dict[str, Any], retrieved: Sequence[dict[str, Any]]) -> list[str]:
    """Distinct guideline `document_id`s actually backing this answer.

    Read from the retrieved chunks rather than from the model's `document`
    strings, so it reports which guidelines the answer is really grounded in.
    """
    by_id = _index_hits(retrieved)
    docs: list[str] = []
    for citation in answer.get("citations") or []:
        if not isinstance(citation, dict):
            continue
        hit = by_id.get(str(citation.get("chunk_id") or "").strip())
        if hit is None:
            continue
        doc = hit.get("document_id") or hit.get("document")
        if doc and str(doc) not in docs:
            docs.append(str(doc))
    return docs


# --- evidence-grade summary ------------------------------------------------

import math as _math


def _is_usable_grade(value: Any) -> bool:
    """True when *value* is a non-empty string that is not a NaN sentinel.

    The corpus stores grades as strings (e.g. ``'A'``, ``'1'``,
    ``'C recommendation'``) or as ``None`` when the guideline does not use a
    formal grading system (ESVS 2024, NICE NG156, USPSTF 2019 in this
    corpus).  Some ingestion pipelines also leave ``float('nan')`` in the
    payload field when a cell in the source spreadsheet was blank -- that is
    treated as absent, not as a meaningful grade string.
    """
    if value is None:
        return False
    if isinstance(value, float) and _math.isnan(value):
        return False
    s = str(value).strip()
    return bool(s) and s.lower() not in {"nan", "none", "null", "n/a", ""}


def summarize_evidence_grade(citations_resolved: list[dict[str, Any]]) -> dict[str, Any]:
    """Produce an answer-level evidence-grade summary from already-resolved citations.

    Reads *only* the ``recommendation_grade`` and ``evidence_level`` fields
    that ``resolve_citations()`` populates from the retriever's own records.
    Never invents, infers, upgrades or downgrades a value.

    Parameters
    ----------
    citations_resolved:
        The list returned by ``resolve_citations(answer, retrieved)``.
        Unresolved citations (``resolved == False``) are excluded from the
        summary because their metadata fields are unknown.

    Returns
    -------
    A dict with the following guaranteed keys:

    ``available``
        ``True`` if at least one resolved citation carries a non-null,
        non-NaN grade or level string.  ``False`` otherwise (e.g. a refusal
        with no citations, or all cited chunks from guidelines that do not
        publish formal grades in this corpus).

    ``recommendation_grades``
        Sorted list of distinct non-null ``recommendation_grade`` strings
        present across all resolved citations.  Empty list when none are
        available.  **Do not attempt to order these across guidelines**:
        ``'1'`` (SVS) and ``'C recommendation'`` (SVS/USPSTF) come from
        different scales.

    ``evidence_levels``
        Sorted list of distinct non-null ``evidence_level`` strings (e.g.
        ``['A', 'B']``).  Empty list when none are available.

    ``n_citations_with_grade``
        Integer count of resolved citations that carry at least one of
        ``recommendation_grade`` or ``evidence_level``.

    ``n_citations_total``
        Integer count of *resolved* citations (unresolved / hallucinated
        citations are excluded).

    Shape is stable: all four keys are always present regardless of whether
    metadata is available, so callers need not guard against ``KeyError``.

    Corpus note (as of index V1_atomic_pagesafe):
        - SVS 2018 chunks carry ``recommendation_grade`` (``'1'``, ``'2'``,
          ``'I statement'``) and ``evidence_level`` (``'A'``, ``'B'``,
          ``'C'``).
        - ESVS 2024, NICE NG156 and USPSTF 2019 chunks have ``None`` for
          both fields.  This is expected; the summary returns
          ``available: false`` for answers grounded exclusively in those
          three guidelines.
    """
    resolved_only = [c for c in (citations_resolved or []) if isinstance(c, dict) and c.get("resolved")]
    n_total = len(resolved_only)

    grades: list[str] = []
    levels: list[str] = []
    n_with_grade = 0

    for citation in resolved_only:
        g = citation.get("recommendation_grade")
        l = citation.get("evidence_level")
        has_either = False
        if _is_usable_grade(g):
            s = str(g).strip()
            if s not in grades:
                grades.append(s)
            has_either = True
        if _is_usable_grade(l):
            s = str(l).strip()
            if s not in levels:
                levels.append(s)
            has_either = True
        if has_either:
            n_with_grade += 1

    grades.sort()
    levels.sort()
    available = bool(grades or levels)

    return {
        "available": available,
        "recommendation_grades": grades,
        "evidence_levels": levels,
        "n_citations_with_grade": n_with_grade,
        "n_citations_total": n_total,
    }


# --- views used by the evaluation ------------------------------------------

def answer_prose(answer: dict[str, Any]) -> str:
    """The answer's own words: recommendation, claims, conflict positions.

    Quoted excerpts are deliberately excluded. A fact appearing only inside a
    quoted excerpt was not *stated* by the answer -- it was quoted from the
    corpus -- and the distinction matters when checking whether an answer
    actually said something.
    """
    parts: list[str] = [str(answer.get("recommendation") or "")]
    for bullet in answer.get("supporting_evidence") or []:
        if isinstance(bullet, dict):
            parts.append(str(bullet.get("claim") or ""))
    for conflict in answer.get("evidence_conflicts") or []:
        if not isinstance(conflict, dict):
            continue
        parts.append(str(conflict.get("topic") or ""))
        for position in conflict.get("positions") or []:
            if isinstance(position, dict):
                parts.append(str(position.get("position") or ""))
                parts.append(str(position.get("source") or ""))
    return "\n".join(p for p in parts if p)


def conflict_position_count(answer: dict[str, Any]) -> int:
    """The largest number of distinct positions reported for any one conflict.

    Two or more means the answer presented a disagreement rather than resolving
    it silently, which is what slide 5's "compare evidence across chunks" asks
    for and what the conflicting-evidence test asserts.
    """
    best = 0
    for conflict in answer.get("evidence_conflicts") or []:
        if isinstance(conflict, dict) and isinstance(conflict.get("positions"), list):
            best = max(best, len(conflict["positions"]))
    return best

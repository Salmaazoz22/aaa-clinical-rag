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

# --- claim-to-excerpt binding (errors) -------------------------------------
# The three checks below close the gap the lexical warning above cannot: a
# `supporting_evidence` bullet whose *claim* says the opposite of, or a
# different number from, the excerpt it is attached to. Those are hard errors,
# not warnings: a claim that reverses its own evidence is not "off", it is
# wrong, and an answer carrying one must not report `ok`.
E_CLAIM_CONTRADICTS_EXCERPT = "claim_contradicts_excerpt"
E_CLAIM_NUMERIC_MISMATCH = "claim_numeric_mismatch"
E_CLAIM_UNSUPPORTED_TERMS = "claim_unsupported_terms"
E_RECOMMENDATION_UNSUPPORTED_FACT = "recommendation_unsupported_fact"

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

#: How many *novel* content terms a claim or a recommendation sentence may
#: introduce before it is treated as carrying a fact the evidence does not.
#: A term is novel when its stem appears nowhere in the chunk text it is bound
#: to and it is not ordinary clinical/discourse vocabulary (`_GENERIC_STEMS`).
#: One novel term is normal paraphrase ("modality", "declines"); two or more
#: specific ones ("lifelong", "antiplatelet") is a fact that was added.
MIN_NOVEL_TERMS = 2

#: Terms shorter than this are not considered when looking for novel facts:
#: short words are overwhelmingly function words and abbreviations, and the
#: numeric side of a claim is covered by the measurement check instead.
MIN_NOVEL_TERM_CHARS = 5

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
    mapping, range 'e' artifact normalization, and whitespace collapsing so that
    typographic variants of the same word compare as equal. The original `text` is
    never modified.
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
    # Step 3.5: Convert '30 e 39' (PDF extraction artifact for range en-dash) to '30-39'
    s = re.sub(r"\b(\d+)\s*e\s*(\d+)\b", r"\1-\2", s)
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


_NEGATION_TERMS = frozenset({"not", "no", "never", "against", "contraindicated", "unrecommended", "unsupported", "discouraged"})


# --- claim vs excerpt: polarity -------------------------------------------
#
# Deterministic, no model. The question asked is narrow on purpose: does the
# claim take the OPPOSITE stance to the excerpt it cites? Not "is the claim
# true", not "does it follow" -- only whether one says do and the other says
# don't. That is the failure mode a citation validator can actually decide.
#
# Two rules keep the false-positive rate down, because a paraphrase that is
# merely *worded* differently must not be rejected (task requirement 6):
#
#   1. A negation word only counts when it is attached to a clinical stance
#      verb within a few words ("not recommended", "never be offered", "no
#      benefit"). A trailing caveat like "...but not in patients unfit for
#      surgery" carries "not" with no stance verb behind it, and is ignored.
#   2. Polarity is compared only between texts that are demonstrably about the
#      same thing (>= MIN_GROUNDING_TOKENS shared content tokens). Two
#      sentences on different subjects cannot contradict each other here.

#: Verbs/nouns that carry a clinical stance. A negation is only polarity-bearing
#: when one of these follows it closely.
_STANCE_WORDS = (
    r"recommend\w*|indicat\w*|advis\w*|consider\w*|offer\w*|warrant\w*|"
    r"benefi\w*|justifi\w*|appropriate|support\w*|perform\w*|use[ds]?|"
    r"repair\w*|treat\w*|screen\w*|operat\w*|requir\w*|need\w*|necessary|"
    r"suitab\w*|eligib\w*|prefer\w*"
)

#: Negation attached to a stance word within four intervening words.
_NEG_SCOPED = re.compile(
    rf"\b(?:not|no|never|cannot|nor|without|nolonger)\b(?:\s+\S+){{0,4}}?\s+(?:{_STANCE_WORDS})\b",
    re.I,
)
#: Words that are negative on their own, wherever they appear.
_NEG_ABSOLUTE = re.compile(
    r"\b(?:contraindicat\w*|inadvisab\w*|unnecessar\w*|unsuitab\w*|ineligib\w*|"
    r"discourag\w*|harmful|avoid\w*|refrain\w*|withhold\w*|forbidden|"
    r"unrecommended|not\s+recommended|shouldn'?t|mustn'?t|can'?t|don'?t|doesn'?t)\b",
    re.I,
)
#: Positive stance, checked only when nothing negative was found.
_POS_STANCE = re.compile(
    r"\b(?:recommend\w*|indicated|advis\w*|consider\w*|offer\w*|warrant\w*|"
    r"benefic\w*|benefit\w*|appropriate|should|shall|must|can|may\s+be\s+(?:used|offered|considered))\b",
    re.I,
)


def _polarity(text: Any) -> str | None:
    """`"negative"`, `"positive"`, or None when the text takes no clear stance.

    Negation dominates: a scoped negation or an absolute negative word makes the
    whole text negative regardless of any positive word it also contains, because
    "must never be offered" is a prohibition, not a recommendation.
    """
    s = normalise_for_match(text)
    if not s:
        return None
    if _NEG_ABSOLUTE.search(s) or _NEG_SCOPED.search(s):
        return "negative"
    if _POS_STANCE.search(s):
        return "positive"
    return None


# --- claim vs excerpt: measurements ---------------------------------------

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

#: Unit -> (canonical unit, multiplier). Comparable quantities share a canonical
#: unit, so "5.5 cm" and "55 mm" compare equal and are never reported as a
#: numeric contradiction.
_UNIT_CANON: dict[str, tuple[str, float]] = {
    "mm": ("mm", 1.0), "millimetre": ("mm", 1.0), "millimeter": ("mm", 1.0),
    "millimetres": ("mm", 1.0), "millimeters": ("mm", 1.0),
    "cm": ("mm", 10.0), "centimetre": ("mm", 10.0), "centimeter": ("mm", 10.0),
    "centimetres": ("mm", 10.0), "centimeters": ("mm", 10.0),
    "month": ("month", 1.0), "months": ("month", 1.0),
    "year": ("month", 12.0), "years": ("month", 12.0),
    "week": ("week", 1.0), "weeks": ("week", 1.0),
    "day": ("day", 1.0), "days": ("day", 1.0),
    "%": ("%", 1.0), "percent": ("%", 1.0),
    "mg": ("mg", 1.0), "g": ("mg", 1000.0), "mcg": ("mg", 0.001),
}

_UNIT_ALTERNATION = "|".join(sorted((re.escape(u) for u in _UNIT_CANON), key=len, reverse=True))
_NUM = r"\d+(?:\.\d+)?"
#: "40-49 mm" -> both endpoints carry the unit.
_RANGE_WITH_UNIT = re.compile(rf"({_NUM})\s*-\s*({_NUM})\s*({_UNIT_ALTERNATION})\b", re.I)
_VALUE_WITH_UNIT = re.compile(rf"({_NUM})\s*({_UNIT_ALTERNATION})\b", re.I)
#: Interval adverbs the guidelines use instead of a number.
_INTERVAL_WORDS = (
    (re.compile(r"\bannual\w*\b|\bevery\s+year\b|\byearly\b|\bper\s+year\b", re.I), 12.0),
    (re.compile(r"\bsix\s+month\w*\b|\bbi[-\s]?annual\w*\b|\bhalf[-\s]?yearly\b", re.I), 6.0),
)


def _measurements(text: Any) -> dict[str, set[float]]:
    """Every quantity in *text*, keyed by canonical unit.

    Only unit-bearing numbers are collected. A bare number ("Recommendation 13",
    "reference 106") carries no comparable meaning and is deliberately ignored.
    """
    out: dict[str, set[float]] = {}
    s = normalise_for_match(text)
    if not s:
        return out
    for word, value in _NUMBER_WORDS.items():
        s = re.sub(rf"\b{word}\b", str(value), s)

    def add(unit_token: str, raw: float) -> None:
        canon = _UNIT_CANON.get(unit_token.lower())
        if canon is None:
            return
        name, factor = canon
        out.setdefault(name, set()).add(round(raw * factor, 4))

    for match in _RANGE_WITH_UNIT.finditer(s):
        add(match.group(3), float(match.group(1)))
        add(match.group(3), float(match.group(2)))
    for match in _VALUE_WITH_UNIT.finditer(s):
        add(match.group(2), float(match.group(1)))
    for pattern, months in _INTERVAL_WORDS:
        if pattern.search(s):
            out.setdefault("month", set()).add(months)
    return out


def _numeric_conflicts(claim: str, excerpt: str) -> list[tuple[str, list[float], list[float]]]:
    """Units where the claim states a quantity the excerpt does not carry.

    A unit the excerpt never mentions is not a conflict -- the excerpt simply
    does not speak to it, which the grounding checks handle. A conflict is a
    unit BOTH texts use, where the claim's values are not among the excerpt's.
    """
    claim_units, excerpt_units = _measurements(claim), _measurements(excerpt)
    conflicts = []
    for unit, values in claim_units.items():
        source = excerpt_units.get(unit)
        if not source:
            continue
        if not values.issubset(source):
            conflicts.append((unit, sorted(values - source), sorted(source)))
    return conflicts


# --- claim vs chunk: novel facts ------------------------------------------

_SUFFIXES = ("ations", "ation", "ements", "ement", "ingly", "edly", "ing", "ies", "ied", "ed", "es", "ly", "s")

#: Words a faithful paraphrase is expected to introduce, in two groups.
#:
#: `_EVIDENCE_VOCABULARY` is how one *talks about* guideline evidence, plus the
#: domain nouns this corpus is entirely made of. `_QUALIFIER_VOCABULARY` is
#: prepositions, adverbs and hedges: they change how a sentence reads, never
#: what it asserts. Novelty in either carries no new clinical fact, so neither
#: counts towards MIN_NOVEL_TERMS.
_EVIDENCE_VOCABULARY: frozenset[str] = frozenset({
    "guideline", "guidelines", "recommend", "recommended", "recommendation",
    "suggest", "suggested", "state", "stated", "report", "reported", "indicate",
    "indicated", "advise", "advised", "consider", "considered", "decline",
    "declines", "according", "evidence", "clinical", "patient", "patients",
    "population", "populations", "manage", "management", "threshold",
    "thresholds", "diameter", "diameters", "aneurysm", "aneurysms", "aortic",
    "aorta", "abdominal", "repair", "repairs", "surveillance", "imaging",
    "image", "screen", "screening", "modality", "modalities", "interval",
    "intervals", "elective", "electively", "surgical", "surgery",
    "asymptomatic", "symptomatic", "unruptured", "ruptured", "rupture",
    "maximum", "minimum", "larger", "smaller", "greater", "measure",
    "measured", "measurement", "follow", "outcome", "outcomes", "risk",
    "benefit", "source", "sources", "document", "documents", "section",
    "passage", "passages", "context", "corpus", "answer", "question",
    "proposal", "proposals", "propose", "proposed", "record", "records",
    "recorded", "note", "noted", "notes", "describe", "described", "mention",
    "mentioned", "adopt", "adopted", "agree", "agreement", "disagree",
    "disagreement", "conclude", "conclusion", "finding", "findings",
    "statement", "position", "positions", "apply", "applies", "based",
    "insufficient", "sufficient", "examine", "examined", "usable", "similar",
    "similarity", "retrieval", "retrieved", "specific", "general", "provide",
    "require", "required", "offer", "offered", "perform", "performed",
    "treat", "treatment", "therapy", "prefer", "preference", "committee",
    "issue", "issued", "raise", "raised", "write", "writing", "strong",
    "negative", "positive", "believe", "support", "supported", "level",
    "grade", "class", "women", "male", "female", "adult", "adults", "year",
    "years", "month", "months", "week", "weeks", "size", "growth", "rate",
    "case", "cases", "study", "studies", "trial", "trials", "review", "data",
    "result", "results", "practice", "people", "person", "cannot", "should",
    "diagnosis", "diagnostic", "trigger", "triggers", "value", "values",
})
_QUALIFIER_VOCABULARY: frozenset[str] = frozenset({
    "above", "below", "against", "across", "beyond", "before", "after",
    "during", "unless", "until", "since", "among", "around", "under", "over",
    "through", "toward", "towards", "upon", "versus", "regarding",
    "concerning", "explicitly", "implicitly", "specifically", "particularly",
    "especially", "notably", "typically", "usually", "generally", "commonly",
    "frequently", "rarely", "always", "often", "sometimes", "approximately",
    "roughly", "nearly", "almost", "merely", "simply", "clearly", "directly",
    "indirectly", "currently", "previously", "subsequently", "respectively",
    "accordingly", "additionally", "moreover", "furthermore", "instead",
    "otherwise", "likewise", "similarly", "together", "overall", "entirely",
    "itself", "themselves", "onwards", "alone", "least", "prior", "later",
    "earlier", "first", "second", "third", "third-party", "every", "explicit",
})


def _stem(token: str) -> str:
    """A crude suffix strip, enough for `recommended`/`recommendation` to agree.

    Not a linguistic stemmer and not trying to be: its only job is to stop a
    faithful paraphrase from looking novel because it inflected a word the
    source also uses.
    """
    for suffix in _SUFFIXES:
        if len(token) - len(suffix) >= 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


#: The two vocabularies above, stemmed once, so membership is checked the same
#: way novelty is. Built here rather than written by hand: a hand-stemmed list
#: drifts from `_stem` the moment either changes.
_IGNORED_STEMS: frozenset[str] = frozenset(
    _stem(word) for word in (_EVIDENCE_VOCABULARY | _QUALIFIER_VOCABULARY)
)


def _novel_terms(text: str, sources: Sequence[str]) -> list[str]:
    """Substantive words in *text* whose stem appears in none of *sources*."""
    known = {_stem(t) for source in sources for t in _content_tokens(source)}
    seen: set[str] = set()
    novel: list[str] = []
    for token in sorted(_content_tokens(text)):
        if len(token) < MIN_NOVEL_TERM_CHARS or not token.isalpha():
            continue
        stem = _stem(token)
        if stem in known or stem in _IGNORED_STEMS or stem in seen:
            continue
        seen.add(stem)
        novel.append(token)
    return novel


def _claim_binding_findings(
    claim: Any, excerpt: Any, chunk_text: str, location: str, chunk_id: str
) -> list[Finding]:
    """Check one `supporting_evidence` bullet: claim <-> excerpt <-> chunk.

    Three independent failures, each an error:

      * the claim takes the opposite stance to the excerpt it cites;
      * the claim states a quantity in a unit the excerpt uses, with a value the
        excerpt does not carry ("5 mm" against "55 mm", "every year" against
        "every three years");
      * the claim introduces two or more substantive terms that appear nowhere in
        the chunk it is bound to.

    Nothing here is triggered by wording alone: a paraphrase that keeps the
    stance, the numbers and the vocabulary of its source passes all three.
    """
    findings: list[Finding] = []
    if not isinstance(claim, str) or not claim.strip():
        return findings

    quote = excerpt if isinstance(excerpt, str) and excerpt.strip() else ""

    if quote:
        claim_polarity, excerpt_polarity = _polarity(claim), _polarity(quote)
        shared = _content_tokens(claim) & _content_tokens(quote)
        if (
            claim_polarity is not None
            and excerpt_polarity is not None
            and claim_polarity != excerpt_polarity
            and len(shared) >= MIN_GROUNDING_TOKENS
        ):
            findings.append(
                Finding(
                    E_CLAIM_CONTRADICTS_EXCERPT,
                    SEVERITY_ERROR,
                    "the claim takes the opposite stance to the excerpt it cites: the "
                    "excerpt is "
                    f"{excerpt_polarity} and the claim is {claim_polarity}",
                    location=f"{location}.claim",
                    chunk_id=chunk_id,
                    expected=quote.strip()[:160],
                    actual=claim.strip()[:160],
                )
            )

        for unit, claimed, in_excerpt in _numeric_conflicts(claim, quote):
            findings.append(
                Finding(
                    E_CLAIM_NUMERIC_MISMATCH,
                    SEVERITY_ERROR,
                    f"the claim states {claimed} {unit} but the excerpt it cites carries "
                    f"{in_excerpt} {unit}",
                    location=f"{location}.claim",
                    chunk_id=chunk_id,
                    expected=in_excerpt,
                    actual=claimed,
                )
            )

    sources = [s for s in (chunk_text, quote) if s]
    if sources:
        novel = _novel_terms(claim, sources)
        if len(novel) >= MIN_NOVEL_TERMS:
            findings.append(
                Finding(
                    E_CLAIM_UNSUPPORTED_TERMS,
                    SEVERITY_ERROR,
                    "the claim introduces terms that appear nowhere in the chunk it cites, "
                    "so it asserts more than the evidence it is bound to",
                    location=f"{location}.claim",
                    chunk_id=chunk_id,
                    expected=None,
                    actual=novel[:8],
                )
            )
    return findings


def _recommendation_fact_findings(
    recommendation: str, chunk_texts: Sequence[str], location: str = "recommendation"
) -> list[Finding]:
    """Recommendation sentences that assert facts absent from every cited chunk.

    Checked against the FULL text of the cited chunks rather than the quoted
    excerpts, so a correct sentence supported by a part of the chunk the model
    chose not to quote is not punished. What survives that is a sentence built
    from vocabulary the evidence never uses -- an invented fact.
    """
    findings: list[Finding] = []
    sources = [t for t in chunk_texts if t and t.strip()]
    if not recommendation or not recommendation.strip() or not sources:
        return findings

    for index, sentence in enumerate(_SENTENCE_SPLIT.split(recommendation.strip())):
        sentence = sentence.strip()
        if not sentence:
            continue
        novel = _novel_terms(sentence, sources)
        if len(novel) >= MIN_NOVEL_TERMS:
            findings.append(
                Finding(
                    E_RECOMMENDATION_UNSUPPORTED_FACT,
                    SEVERITY_ERROR,
                    "a sentence in 'recommendation' asserts terms that appear in none of "
                    "the cited chunks, so it states a fact the retrieved evidence does "
                    "not carry",
                    location=f"{location}[sentence {index}]",
                    expected=novel[:8],
                    actual=sentence[:200],
                )
            )
    return findings


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
      5. Also check for polarity/negation contradictions between sentence and excerpt.
      6. Sentences that fail grounding get W_RECOMMENDATION_UNSUPPORTED_SENTENCE.
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

        matching_excerpts = [
            ex_tokens for ex_tokens in excerpt_token_sets
            if len(sent_tokens & ex_tokens) >= MIN_GROUNDING_TOKENS
        ]

        if not matching_excerpts:
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
        else:
            # Polarity contradiction check: sentence has negation but excerpt does not (or vice versa)
            sent_has_negation = bool(sent_tokens & _NEGATION_TERMS)
            any_polarity_match = any(
                bool(ex_tokens & _NEGATION_TERMS) == sent_has_negation
                for ex_tokens in matching_excerpts
            )
            if not any_polarity_match:
                findings.append(
                    Finding(
                        W_RECOMMENDATION_UNSUPPORTED_SENTENCE,
                        SEVERITY_WARNING,
                        "a sentence in 'recommendation' has a polarity/negation mismatch "
                        "with its cited excerpt; claim may contradict evidence",
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
        if claim is None or not isinstance(claim, str) or not claim.strip():
            report.findings.append(
                Finding(
                    E_MISSING_FIELD,
                    SEVERITY_ERROR,
                    "supporting_evidence bullet is missing required non-empty string 'claim'",
                    location=f"{location}.claim",
                )
            )

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

        # Claim <-> excerpt <-> chunk. Skipped on a refusal, whose evidence
        # bullets are built by `generation/refusal.py` to describe what was
        # examined rather than to assert a clinical fact.
        if not report.is_refusal and chunk_id in by_id:
            hit = by_id[chunk_id]
            report.findings.extend(
                _claim_binding_findings(
                    claim,
                    bullet.get("excerpt"),
                    str(hit.get("chunk_text") or hit.get("text") or ""),
                    location,
                    chunk_id,
                )
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

        # The prose is additionally checked against the FULL text of every chunk
        # it cites, which is what the model was actually shown. A sentence that
        # uses vocabulary appearing in none of them is asserting something the
        # evidence does not carry, and that is an error rather than a warning.
        cited_chunk_texts = [
            str(by_id[cid].get("chunk_text") or by_id[cid].get("text") or "")
            for cid in dict.fromkeys(report.cited_chunk_ids)
            if cid in by_id
        ]
        report.findings.extend(
            _recommendation_fact_findings(recommendation, cited_chunk_texts)
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

# -*- coding: utf-8 -*-
"""Pre-retrieval safety gate for patient-specific requests.

Day 3 slide 10 requires a refusal when the question asks for diagnosis or dosing
for a real, specific patient. The reference implementation has no analogue for
this -- it has no safety or refusal layer of any kind -- so the design here is
from scratch.

Two properties are deliberate:

* the gate runs **before** the model call, so a question containing patient
  details is refused without those details being sent to a third-party API. The
  refusal is built locally and deterministically;
* the gate runs **above** retrieval and never alters it. Retrieval semantics are
  frozen: nothing here changes ranking, scoring, or which chunks come back, and
  the retrieval path still cannot see which question it is answering. The gate
  only decides whether the *model* is called at all.

What this is not: a classifier. It is a conservative pattern gate that catches
the unambiguous cases, and it will miss paraphrases. The real backstop is system
prompt rule F2, which instructs the model to refuse the same class of request;
the gate exists so that the common cases never reach the model in the first
place. Both layers are needed and neither is sufficient alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- signals ---------------------------------------------------------------
#
# Each signal is one observable feature of the question. Blocking is a rule over
# combinations of signals, not any single regex, because most single signals have
# legitimate general-question uses:
#
#   "What dose of statin do the guidelines recommend?"  -- dosage, but general
#   "Should I screen men aged 65-75?"                   -- directed, but general
#
# Both of those must be answered (or refused for lack of evidence), not refused
# on safety grounds.

_SIGNALS: dict[str, tuple[re.Pattern[str], ...]] = {
    # An individual is explicitly named as the subject. Unambiguous on its own.
    "explicit_patient_reference": (
        re.compile(r"\b(my|our|this|the)\s+(patient|case)\b", re.I),
        re.compile(r"\bmy\s+(father|mother|dad|mum|mom|husband|wife|son|daughter|brother|sister|uncle|aunt)\b", re.I),
        re.compile(r"\bpatient\s+(is|has|was|presents|presented)\b", re.I),
        re.compile(r"\bI\s+have\s+(a|an)\b[^.?!]{0,40}\b(aneurysm|aaa)\b", re.I),
        re.compile(r"\bmr\.?\s+[a-z]+\b|\bmrs\.?\s+[a-z]+\b|\bms\.?\s+[a-z]+\b", re.I),
    ),
    # A specific individual's demographics. Age-in-years only: "45 mm" and
    # "aged 65 to 75" describe populations, not a person.
    "individual_demographics": (
        re.compile(r"\b\d{1,3}[\s-]*(?:year|yr)s?[\s-]*old\b", re.I),
        re.compile(r"\b\d{1,3}\s*y[/\s]?o\b", re.I),
        re.compile(r"\b(?:a|an)\s+\d{1,3}[\s-]*(?:year|yr)s?[\s-]*old\s+(?:man|woman|male|female|patient)\b", re.I),
        re.compile(r"\b(i\s+am|i'm|aged?)\s+\d{1,3}\b", re.I),
        re.compile(r"\b\d{1,3}\s+years?\s+of\s+age\b", re.I),
    ),
    # The question asks what to do about someone, rather than what a guideline says.
    "individual_directed_ask": (
        re.compile(r"\b(should|shall|do|can|must)\s+(i|we)\b", re.I),
        re.compile(r"\bwhat\s+(should|would|do)\s+(i|we)\b", re.I),
        re.compile(r"\b(he|she|they|him|her|them)\s+(should|needs?|requires?|qualif\w+)\b", re.I),
        re.compile(r"\b(operate|refer|admit|discharge|treat)\s+(him|her|them|this\s+patient|my\s+patient)\b", re.I),
    ),
    # A dosing request.
    "dosage_request": (
        re.compile(r"\b(dose|doses|dosage|dosing|posology)\b", re.I),
        re.compile(r"\b\d+\s*(mg|mcg|µg|g|units?)\b(?:/\s*(kg|day|d|hr|h))?", re.I),
        re.compile(r"\bhow\s+much\s+\w+\s+(should|do|to)\b", re.I),
        re.compile(r"\b(prescribe|titrate|start\s+\w+\s+on)\b", re.I),
    ),
    # A diagnosis request about an individual.
    "individual_diagnosis_request": (
        re.compile(r"\bdiagnose\s+(this|my|him|her|them|the\s+patient)\b", re.I),
        re.compile(r"\bdoes\s+(he|she|they|my\s+patient|this\s+patient)\s+have\b", re.I),
        re.compile(r"\bis\s+(this|my|his|her)\s+\w*\s*(an?\s+)?(aaa|aneurysm|rupture)\b", re.I),
        re.compile(r"\bwhat('s|\s+is)\s+(the\s+)?diagnosis\b", re.I),
    ),
    # First-person possessive reference to own medical condition.
    # Fires when a speaker uses "my" + a vascular / aneurysm term to describe
    # their own body or diagnosis.
    "self_reported_condition": (
        re.compile(r"\bmy\s+(aaa|aneurysm|aortic\s+aneurysm|abdominal\s+(aortic\s+)?aneurysm)\b", re.I),
        re.compile(r"\bmy\s+(aorta|aortic\s+dilation|aortic\s+dilatation|aortic\s+expansion)\b", re.I),
        re.compile(r"\bmy\s+(condition|disease|diagnosis|symptoms?)\b.{0,40}\b(aaa|aneurysm|aortic)\b", re.I),
        re.compile(r"\b(aaa|aneurysm|aortic)\b.{0,40}\bmy\s+(condition|disease|diagnosis|symptoms?)\b", re.I),
        re.compile(r"\b(have|has|got|diagnosed\s+with)\s+a?\s*[\d.]*\s*(cm|mm)\s*(aaa|aneurysm|aortic\s+aneurysm)\b", re.I),
        re.compile(r"\b[\d.]+\s*(cm|mm)\s*(aaa|aneurysm|aortic\s+aneurysm)\b", re.I),
    ),
}


@dataclass(frozen=True)
class SafetyVerdict:
    """Outcome of screening one question."""

    blocked: bool
    signals: list[str] = field(default_factory=list)
    rule: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "blocked": self.blocked,
            "signals": list(self.signals),
            "rule": self.rule,
            "detail": self.detail,
        }


def detect_signals(query: str) -> list[str]:
    """Which safety signals the question exhibits, in a stable order."""
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}")
    return [name for name, patterns in _SIGNALS.items() if any(p.search(query) for p in patterns)]


def screen_query(query: str) -> SafetyVerdict:
    """Decide whether this question is a patient-specific request.

    Blocking rules, in the order they are tried:

    B1  an individual is explicitly named as the subject ("my patient", "this
        patient", a relative, a title + surname);
    B2  the question asks for a diagnosis of an individual;
    B3  an individual's demographics appear together with an
        individual-directed ask or a dosing request;
    B4  a dosing request appears together with an individual-directed ask;
    B5  a first-person directed ask co-occurs with a first-person possessive
        reference to the speaker's own vascular condition ("Should I have
        surgery for my aneurysm?").  B1–B4 all miss this because there is no
        explicit patient token, no demographics, and no dosing language; the
        bypass is closed here by recognising that "my aneurysm" + "should I"
        is unambiguously personal.

    A dosing or directed question with no individual attached is NOT blocked
    here. "What statin dose do the guidelines recommend?" is a general question
    about the corpus; it gets answered if the evidence supports it, and refused
    for insufficient evidence if it does not. Refusing it on safety grounds
    would be wrong twice over: it is not patient-specific, and it would hide the
    fact that the corpus carries no dosing guidance.
    """
    signals = detect_signals(query)
    have = set(signals)

    def verdict(rule: str, detail: str) -> SafetyVerdict:
        return SafetyVerdict(blocked=True, signals=signals, rule=rule, detail=detail)

    if "explicit_patient_reference" in have:
        return verdict(
            "B1",
            "the question is about a specific individual (an explicit patient or family-member reference)",
        )
    if "individual_diagnosis_request" in have:
        return verdict("B2", "the question asks for a diagnosis of a specific individual")
    if "individual_demographics" in have and have & {"individual_directed_ask", "dosage_request"}:
        return verdict(
            "B3",
            "the question describes a specific individual and asks what to do for them",
        )
    if "dosage_request" in have and "individual_directed_ask" in have:
        return verdict("B4", "the question asks what to prescribe or dose for a specific individual")
    if "individual_directed_ask" in have and "self_reported_condition" in have:
        return verdict(
            "B5",
            "the question uses a first-person directed ask together with a possessive reference "
            "to the speaker's own vascular condition",
        )

    return SafetyVerdict(blocked=False, signals=signals)

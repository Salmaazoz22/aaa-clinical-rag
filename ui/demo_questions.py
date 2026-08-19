# -*- coding: utf-8 -*-
"""Pre-defined demo questions.

Each entry says which capability it is meant to exercise and, where the outcome
is determined by a gate rather than by a model, what the system will do with it.
Those `expects` strings describe *mechanism*, not content: "refused by the
pre-retrieval safety gate" is a property of the pipeline that holds regardless
of what is in the corpus. Nothing here predicts, or pre-writes, an answer.

The guideline questions are drawn verbatim from the frozen evaluation set
(`eval/gold_standard_final20.json`), so a demo shows the same questions the
published retrieval metrics were measured on — not a hand-picked set chosen
because it happens to look good.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoQuestion:
    label: str          # short button label
    question: str       # the text sent to the API
    capability: str     # what this question demonstrates
    expects: str        # what the pipeline does with it, and why
    kind: str           # "answer" | "refuse"


DEMOS: tuple[DemoQuestion, ...] = (
    DemoQuestion(
        label="Diameter threshold for repair",
        question=(
            "At what maximum diameter is elective repair recommended for a man with an "
            "asymptomatic fusiform abdominal aortic aneurysm?"
        ),
        capability="Direct guideline lookup",
        expects=(
            "A single numeric recommendation that appears explicitly in the guideline text, so "
            "the citation can be checked against the exact passage it came from."
        ),
        kind="answer",
    ),
    DemoQuestion(
        label="Surveillance interval",
        question="How often should imaging surveillance be repeated for an aneurysm measuring 4.0 to 4.9 cm?",
        capability="Screening / surveillance recommendation",
        expects=(
            "A band-specific interval. The guidelines in this corpus do not all use the same "
            "bands, so this is a good test of whether sources are separated rather than merged."
        ),
        kind="answer",
    ),
    DemoQuestion(
        label="Familial screening",
        question="Should relatives of a patient with an aneurysm be screened, and at what age?",
        capability="Multi-source evidence",
        expects=(
            "Screening guidance appears in more than one document here, so the answer should "
            "cite more than one guideline rather than picking whichever ranked first."
        ),
        kind="answer",
    ),
    DemoQuestion(
        label="Endoleak management",
        question="How should a type 2 endoleak be managed after endovascular aneurysm repair?",
        capability="Multi-step management recommendation",
        expects=(
            "A conditional, staged recommendation rather than a single number — the answer has "
            "to hold several retrieved passages together without inventing the joins."
        ),
        kind="answer",
    ),
    DemoQuestion(
        label="Risk-factor management",
        question=(
            "What cardiovascular risk factor management should every patient with an "
            "abdominal aortic aneurysm receive?"
        ),
        capability="Medical management",
        expects="A multi-part recommendation, each part expected to carry its own citation.",
        kind="answer",
    ),
    DemoQuestion(
        label="Patient-specific decision → refused",
        question="My patient is a 72-year-old man with a 5.2 cm aneurysm. Should I operate on him?",
        capability="Safety gate — patient-specific request",
        expects=(
            "Refused by the pre-retrieval safety gate. The refusal is built locally and the "
            "model is never called, so the patient detail is never sent to a third-party API. "
            "The refusal deliberately carries no citations."
        ),
        kind="refuse",
    ),
    DemoQuestion(
        label="Out-of-corpus clinical → refused",
        question="What is the recommended insulin dose for type 2 diabetes?",
        capability="Evidence threshold — question outside the corpus",
        expects=(
            "Retrieval runs and returns the nearest passages, but none clears the similarity "
            "floor, so the system refuses and names the passages it examined and rejected."
        ),
        kind="refuse",
    ),
    DemoQuestion(
        label="Unrelated question → refused",
        question="What is the best recipe for sourdough bread?",
        capability="Evidence threshold — unrelated question",
        expects=(
            "The same threshold gate, with much lower similarity scores. Demonstrates that the "
            "system does not answer from whatever the vector store happened to return."
        ),
        kind="refuse",
    ),
)

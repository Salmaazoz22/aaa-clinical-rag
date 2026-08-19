# -*- coding: utf-8 -*-
"""The deterministic gate for guideline editions the corpus does not contain.

Why this cannot be left to the similarity floor: "What does the 2026 ESVS
guideline recommend for AAA management?" retrieves ESVS 2024 passages at 0.85 --
comfortably above the 0.75 floor -- because every chunk in the index is an ESVS
passage on exactly that topic. The retrieval is good; it is simply the wrong
edition. Before this gate the question reached the model, and the refusal
depended on the model noticing the year and being reachable at all.

The two halves of the requirement are tested together, because a gate that
refuses too much is as broken as one that refuses nothing:

    refused   2026 ESVS, 2019 ESVS, NG999, "the 2030 guidelines"
    allowed   2024 ESVS, NICE NG156, SVS 2018, USPSTF 2019, every question with
              no edition in it, and a year attached to a study rather than a
              guideline
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.guideline_scope import (  # noqa: E402
    CORPUS_METADATA_PATH,
    CorpusEditions,
    load_corpus_editions,
    screen_guideline_edition,
)
from generation.pipeline import answer_question  # noqa: E402
from generation.config import GenerationSettings  # noqa: E402
from generation.schema import REFUSAL_GUIDELINE_UNAVAILABLE, REFUSAL_REASONS  # noqa: E402
from generation.refusal import GUIDELINE_UNAVAILABLE_MESSAGE  # noqa: E402


# ---------------------------------------------------------------------------
# The corpus manifest is the source of truth — nothing here is hardcoded
# ---------------------------------------------------------------------------

class TestCorpusEditions:
    def test_editions_come_from_the_manifest(self):
        editions = load_corpus_editions()
        documents = json.loads(CORPUS_METADATA_PATH.read_text(encoding="utf-8"))
        assert editions.all_years == frozenset(
            d["publication_year"] for d in documents if isinstance(d.get("publication_year"), int)
        ) | {2024, 2018, 2019}

    def test_the_four_indexed_guidelines_are_known(self):
        editions = load_corpus_editions()
        assert {"esvs", "nice", "svs", "uspstf"} <= editions.known_aliases
        assert editions.years_for(["esvs"]) == frozenset({2024})
        assert editions.years_for(["svs"]) == frozenset({2018})
        assert editions.years_for(["uspstf"]) == frozenset({2019})
        assert "ng156" in editions.identifiers

    def test_an_unreadable_manifest_disables_the_gate(self):
        """A gate that cannot see the corpus must refuse nothing, not guess."""
        empty = load_corpus_editions(str(ROOT / "data" / "processed" / "does-not-exist.json"))
        assert empty.all_years == frozenset()
        assert not screen_guideline_edition("What does the 2026 ESVS guideline say?", empty).blocked


# ---------------------------------------------------------------------------
# Refused: an edition the corpus does not have
# ---------------------------------------------------------------------------

class TestUnavailableEditionIsRefused:
    def test_2026_esvs_is_refused(self):
        verdict = screen_guideline_edition(
            "What does the 2026 ESVS guideline recommend for AAA management?"
        )
        assert verdict.blocked
        assert verdict.requested == ("ESVS 2026",)

    @pytest.mark.parametrize(
        "question",
        [
            "What does the 2026 ESVS guideline recommend?",
            "What does the ESVS 2027 update recommend for surveillance?",
            "What does the 2019 ESVS guideline recommend?",
            "What does the updated 2025 NICE guideline say about repair?",
            "What do the 2030 guidelines recommend for screening?",
            "What does NICE NG999 say about surveillance?",
        ],
    )
    def test_editions_outside_the_corpus_are_refused(self, question):
        assert screen_guideline_edition(question).blocked

    def test_verdict_names_what_is_available(self):
        verdict = screen_guideline_edition("What does the 2026 ESVS guideline recommend?")
        assert any("ESVS" in label for label in verdict.available)
        assert verdict.detail and "not among the guideline editions" in verdict.detail


# ---------------------------------------------------------------------------
# Allowed: everything else
# ---------------------------------------------------------------------------

class TestAvailableEditionsAreAllowed:
    @pytest.mark.parametrize(
        "question",
        [
            # the four indexed guidelines, named with their edition
            "What does the 2024 ESVS guideline recommend for AAA management?",
            "What is the 2024 ESVS recommendation on surveillance intervals?",
            "What does NICE NG156 say about surveillance?",
            "What are the SVS 2018 recommendations for screening?",
            "What does the USPSTF 2019 recommendation statement say about screening?",
            "Compare the 2024 ESVS and 2019 USPSTF recommendations.",
            # ordinary questions with no edition named at all
            "Which imaging modality is recommended for abdominal aortic aneurysm surveillance?",
            "At what diameter is elective repair recommended for men?",
            "Should men aged 65 to 75 be screened for abdominal aortic aneurysm?",
            # a year that belongs to a study, not to a guideline edition
            "What did the 2013 Bown study report about AAA growth rates?",
            "A trial published in 2021 enrolled 500 patients; what does the guidance say about growth?",
        ],
    )
    def test_allowed(self, question):
        assert not screen_guideline_edition(question).blocked

    def test_a_year_alone_does_not_refuse(self):
        """Requirement: do not reject a question merely because it contains a year."""
        assert not screen_guideline_edition("Is 2019 mortality data included?").blocked

    def test_empty_and_non_string_queries_are_safe(self):
        assert not screen_guideline_edition("").blocked
        assert not screen_guideline_edition(None).blocked  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The gate fires in the pipeline, before any model call
# ---------------------------------------------------------------------------

class _ExplodingProvider:
    """Any call to this is a test failure: the gate must run before generation."""

    name = "exploding"
    model = "exploding"

    def complete(self, messages, *, json_mode: bool = True):  # pragma: no cover
        raise AssertionError("the LLM was called for an unavailable guideline edition")


class _FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, top_k=5):
        return list(self.hits)[:top_k]


def _esvs_hit() -> dict:
    text = (
        "Elective repair should be considered for men with an abdominal aortic aneurysm "
        "with a maximum diameter of 55 mm or larger."
    )
    return {
        "rank": 1,
        "chunk_id": "ESVS_2024__p26-26__c0093",
        "chunk_text": text,
        "text": text,
        "document": "ESVS 2024 Clinical Practice Guidelines",
        "document_id": "ESVS_2024",
        "section": "Elective repair",
        "page": 26,
        "page_start": 26,
        "page_end": 26,
        "similarity_score": 0.85,
        "score": 0.85,
    }


def _settings() -> GenerationSettings:
    return GenerationSettings(
        provider="groq",
        model="test-model",
        base_url="https://example.invalid/v1",
        api_key="not-used",
        top_k=5,
        score_threshold=0.75,
        temperature=0.0,
        max_output_tokens=1000,
        timeout=30.0,
    )


class TestPipelineGate:
    def test_2026_esvs_refuses_without_calling_the_model(self):
        result = answer_question(
            "What does the 2026 ESVS guideline recommend for AAA management?",
            retriever=_FakeRetriever([_esvs_hit()]),
            provider=_ExplodingProvider(),
            settings=_settings(),
        )
        assert result.refused
        assert result.refusal == {
            "reason": REFUSAL_GUIDELINE_UNAVAILABLE,
            "gate": "guideline_scope",
        }
        assert result.completion is None, "no completion means no model call"

    def test_the_refusal_says_what_is_wrong(self):
        result = answer_question(
            "What does the 2026 ESVS guideline recommend for AAA management?",
            retriever=_FakeRetriever([_esvs_hit()]),
            provider=_ExplodingProvider(),
            settings=_settings(),
        )
        assert GUIDELINE_UNAVAILABLE_MESSAGE in result.answer["recommendation"]
        assert result.answer["confidence"] == "Insufficient Evidence"
        assert result.validation["ok"], result.validation["findings"]

    def test_the_refusal_is_recorded_on_the_result(self):
        result = answer_question(
            "What does the 2026 ESVS guideline recommend?",
            retriever=_FakeRetriever([_esvs_hit()]),
            provider=_ExplodingProvider(),
            settings=_settings(),
        )
        assert result.guideline_scope["blocked"] is True
        assert result.to_dict()["guideline_scope"]["requested"] == ["ESVS 2026"]

    def test_2024_esvs_reaches_the_model(self):
        """The available edition must still be answered, not gated."""
        with pytest.raises(AssertionError, match="the LLM was called"):
            answer_question(
                "What does the 2024 ESVS guideline recommend for AAA management?",
                retriever=_FakeRetriever([_esvs_hit()]),
                provider=_ExplodingProvider(),
                settings=_settings(),
            )

    def test_an_ordinary_question_reaches_the_model(self):
        with pytest.raises(AssertionError, match="the LLM was called"):
            answer_question(
                "At what diameter is elective repair recommended for men?",
                retriever=_FakeRetriever([_esvs_hit()]),
                provider=_ExplodingProvider(),
                settings=_settings(),
            )

    def test_the_new_reason_is_registered(self):
        assert REFUSAL_GUIDELINE_UNAVAILABLE in REFUSAL_REASONS

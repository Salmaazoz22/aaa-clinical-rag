# -*- coding: utf-8 -*-
"""Tests for the grounded-generation layer.

No network and no vector store: the retriever and the model provider are both
injected, so the gates, the parser and the citation validator are tested against
fixed evidence. The fixture chunks mirror the shape `vectordb.retriever` returns
and carry text drawn from the real corpus, including the documented 55 mm / 60 mm
disagreement.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.config import DEFAULT_SCORE_THRESHOLD, GenerationSettings  # noqa: E402
from generation.parsing import AnswerParseError, parse_answer  # noqa: E402
from generation.pipeline import answer_question, select_usable_hits  # noqa: E402
from generation.prompts import SYSTEM_PROMPT, build_context_block, build_messages  # noqa: E402
from generation.providers import (  # noqa: E402
    PROVIDER_SPECS,
    Completion,
    MissingAPIKeyError,
    OpenAICompatibleProvider,
    UnknownProviderError,
    resolve_provider_spec,
)
from generation.safety import screen_query  # noqa: E402
from generation.schema import (  # noqa: E402
    CONFIDENCE_INSUFFICIENT,
    DISCLAIMER,
    REFUSAL_BELOW_THRESHOLD,
    REFUSAL_MESSAGE,
    REFUSAL_NO_CHUNKS,
    REFUSAL_PATIENT_SPECIFIC,
    is_refusal,
)
from generation.validator import (  # noqa: E402
    E_ANSWER_WITHOUT_CITATIONS,
    E_HALLUCINATED_CITATION,
    E_HALLUCINATED_EVIDENCE_CHUNK,
    E_INVALID_CONFIDENCE,
    E_MISSING_FIELD,
    E_NUMERIC_CONFIDENCE,
    E_REFUSAL_MESSAGE_MISSING,
    E_UNCITED_CLAIM,
    W_CITATION_METADATA_MISMATCH,
    W_DISCLAIMER_NOT_CANONICAL,
    W_EXCERPT_NOT_IN_CHUNK,
    W_NUMERIC_CERTAINTY_PROSE,
    W_RETRIEVAL_SCORE_MISMATCH,
    answer_prose,
    conflict_position_count,
    documents_cited,
    resolve_citations,
    validate_answer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# The ESVS passage that carries the disagreement: it names the NAAASP-derived
# proposal to raise the threshold to 60 mm and the writing committee's decision
# not to adopt it. The ligature in "suf<fi>cient" is present in the real PDF text
# and is what the excerpt normaliser has to see through.
ESVS_CONFLICT = {
    "rank": 1,
    "similarity_score": 0.843058,
    "score": 0.843058,
    "chunk_id": "ESVS_2024__p26-26__c0093",
    "document": "ESVS 2024 Clinical Practice Guidelines on the Management of Abdominal Aorto-Iliac Artery Aneurysms",
    "document_id": "ESVS_2024",
    "section": "ABDOMINAL AORTIC ANEURYSM",
    "page": 26,
    "page_start": 26,
    "page_end": 26,
    "recommendation_id": None,
    "recommendation_grade": None,
    "evidence_level": None,
    "source_file": "data/pdfs/ESVS_2024.pdf",
    "chunk_text": (
        "On the contrary, based on the NAAASP data it has been suggested to raise the diameter "
        "threshold to 60 mm when based on CTA. Although it is possible that the threshold should "
        "be raised in the future, the GWC does not believe there is sufﬁcient support at this "
        "time. Nevertheless, the GWC has chosen to issue a new strong negative recommendation of "
        "elective repair of AAA < 55 mm, and to downgrade the recommendation on the threshold for "
        "considering repair in men."
    ),
}
ESVS_CONFLICT["text"] = ESVS_CONFLICT["chunk_text"]

NICE_THRESHOLD = {
    "rank": 2,
    "similarity_score": 0.801899,
    "score": 0.801899,
    "chunk_id": "NICE_NG156__p16-16__c0050",
    "document": "Abdominal aortic aneurysm: diagnosis and management (NG156)",
    "document_id": "NICE_NG156",
    "section": "1.5.1 When to consider repair",
    "page": 16,
    "page_start": 16,
    "page_end": 16,
    "recommendation_id": "1.5.1",
    "recommendation_grade": None,
    "evidence_level": None,
    "source_file": "data/pdfs/NICE_NG156.pdf",
    "chunk_text": (
        "Consider elective surgical repair for people with an unruptured abdominal aortic aneurysm "
        "if it is symptomatic, or it is asymptomatic, larger than 4.0 cm and has grown by more "
        "than 1 cm in a year, or it is asymptomatic and 5.5 cm or larger."
    ),
}
NICE_THRESHOLD["text"] = NICE_THRESHOLD["chunk_text"]

SVS_SCREENING = {
    "rank": 3,
    "similarity_score": 0.838520,
    "score": 0.838520,
    "chunk_id": "SVS_2018__p15-15__c0016",
    "document": "SVS 2018 Practice Guidelines on the Care of Patients with an Abdominal Aortic Aneurysm",
    "document_id": "SVS_2018",
    "section": "Aneurysm imaging",
    "page": 15,
    "page_start": 15,
    "page_end": 15,
    "recommendation_id": None,
    "recommendation_grade": "1",
    "evidence_level": "A",
    "source_file": "data/pdfs/SVS_2018.pdf",
    "chunk_text": (
        "We recommend using ultrasound, when feasible, as the preferred imaging modality for "
        "aneurysm screening and surveillance. We recommend a one-time ultrasound screening for "
        "AAAs in men or women 65 to 75 years of age with a history of tobacco use."
    ),
}
SVS_SCREENING["text"] = SVS_SCREENING["chunk_text"]

WEAK_HIT = {**SVS_SCREENING, "chunk_id": "SVS_2018__p99-99__c0999", "similarity_score": 0.61, "score": 0.61, "rank": 4}


def settings_for(threshold: float = 0.75, top_k: int = 5) -> GenerationSettings:
    return GenerationSettings(
        provider="groq",
        model="openai/gpt-oss-120b",
        base_url="https://api.groq.com/openai/v1",
        api_key="test-key-not-real",
        top_k=top_k,
        score_threshold=threshold,
        temperature=0.0,
        max_output_tokens=4000,
        timeout=30.0,
    )


class FakeRetriever:
    """Returns fixed hits and records how it was called."""

    def __init__(self, hits):
        self.hits = list(hits)
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.hits[:top_k]


class FakeProvider:
    """Returns fixed text and counts calls, so 'never called' is assertable."""

    name = "fake"
    model = "fake-model"

    def __init__(self, payload):
        self.payload = payload if isinstance(payload, str) else json.dumps(payload)
        self.calls = []

    def complete(self, messages, *, json_mode=True):
        self.calls.append({"messages": list(messages), "json_mode": json_mode})
        return Completion(text=self.payload, provider=self.name, model=self.model, finish_reason="stop")


class ExplodingProvider:
    """Fails the test if the pipeline calls the model when it should not."""

    name = "exploding"
    model = "none"

    def complete(self, messages, *, json_mode=True):
        raise AssertionError("the model must not be called on this path")


def good_answer(chunks=(ESVS_CONFLICT, NICE_THRESHOLD)) -> dict:
    """A well-formed, fully-cited answer over the given chunks."""
    return {
        "recommendation": (
            "Elective repair is considered at a maximum aortic diameter of 55 mm (5.5 cm) in the "
            "retrieved guidelines, and ESVS 2024 explicitly declines to raise that to 60 mm."
        ),
        "supporting_evidence": [
            {
                "claim": "ESVS 2024 issues a strong negative recommendation against elective repair below 55 mm.",
                "chunk_id": chunks[0]["chunk_id"],
                "excerpt": "a new strong negative recommendation of elective repair of AAA < 55 mm",
            },
            {
                "claim": "NICE NG156 sets the asymptomatic threshold at 5.5 cm or larger.",
                "chunk_id": chunks[1]["chunk_id"],
                "excerpt": "it is asymptomatic and 5.5 cm or larger",
            },
        ],
        "citations": [
            {
                "document": chunks[0]["document_id"],
                "section": chunks[0]["section"],
                "page": chunks[0]["page"],
                "chunk_id": chunks[0]["chunk_id"],
                "retrieval_score": chunks[0]["similarity_score"],
                "excerpt": "a new strong negative recommendation of elective repair of AAA < 55 mm",
            },
            {
                "document": chunks[1]["document_id"],
                "section": chunks[1]["section"],
                "page": chunks[1]["page"],
                "chunk_id": chunks[1]["chunk_id"],
                "retrieval_score": chunks[1]["similarity_score"],
                "excerpt": "it is asymptomatic and 5.5 cm or larger",
            },
        ],
        "confidence": "High",
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Citation validator -- the three cases the specification names
# ---------------------------------------------------------------------------

class TestValidatorValidCase:
    def test_well_formed_answer_passes(self):
        report = validate_answer(good_answer(), [ESVS_CONFLICT, NICE_THRESHOLD])
        assert report.ok, [f.to_dict() for f in report.errors]
        assert report.errors == []
        assert set(report.cited_chunk_ids) == {ESVS_CONFLICT["chunk_id"], NICE_THRESHOLD["chunk_id"]}
        assert report.hallucinated_chunk_ids == []
        assert report.uncited_claims == []

    def test_excerpt_matches_through_pdf_ligature(self):
        """The stored text contains "suf<fi>cient"; a plain-ASCII quote must match."""
        answer = good_answer()
        answer["citations"][0]["excerpt"] = "the GWC does not believe there is sufficient support at this time"
        answer["supporting_evidence"][0]["excerpt"] = answer["citations"][0]["excerpt"]
        report = validate_answer(answer, [ESVS_CONFLICT, NICE_THRESHOLD])
        assert not report.has(W_EXCERPT_NOT_IN_CHUNK)
        assert report.ok

    def test_report_is_serialisable(self):
        report = validate_answer(good_answer(), [ESVS_CONFLICT, NICE_THRESHOLD])
        assert json.loads(json.dumps(report.to_dict()))["ok"] is True


class TestValidatorHallucinatedCitation:
    def test_fabricated_chunk_id_in_citations_is_an_error(self):
        answer = good_answer()
        answer["citations"][1]["chunk_id"] = "ESVS_2024__p99-99__c9999"
        report = validate_answer(answer, [ESVS_CONFLICT, NICE_THRESHOLD])
        assert not report.ok
        assert report.has(E_HALLUCINATED_CITATION)
        assert "ESVS_2024__p99-99__c9999" in report.hallucinated_chunk_ids

    def test_chunk_retrieved_but_not_sent_still_counts_as_fabricated(self):
        """A chunk dropped by the score floor was never shown to the model."""
        answer = good_answer()
        answer["citations"][1]["chunk_id"] = WEAK_HIT["chunk_id"]
        report = validate_answer(answer, [ESVS_CONFLICT, NICE_THRESHOLD])
        assert report.has(E_HALLUCINATED_CITATION)

    def test_fabricated_chunk_id_in_supporting_evidence_is_an_error(self):
        answer = good_answer()
        answer["supporting_evidence"][0]["chunk_id"] = "NICE_NG156__p01-01__c0001"
        report = validate_answer(answer, [ESVS_CONFLICT, NICE_THRESHOLD])
        assert not report.ok
        assert report.has(E_HALLUCINATED_EVIDENCE_CHUNK)

    def test_validator_does_not_mutate_the_answer(self):
        answer = good_answer()
        answer["citations"][0]["chunk_id"] = "totally-made-up"
        before = copy.deepcopy(answer)
        validate_answer(answer, [ESVS_CONFLICT, NICE_THRESHOLD])
        assert answer == before, "findings are reported, never repaired"

    def test_resolve_citations_marks_the_unresolvable_one(self):
        answer = good_answer()
        answer["citations"][0]["chunk_id"] = "made-up"
        resolved = resolve_citations(answer, [ESVS_CONFLICT, NICE_THRESHOLD])
        assert resolved[0]["resolved"] is False
        assert resolved[1]["resolved"] is True
        assert resolved[1]["document_id"] == "NICE_NG156"


class TestValidatorUncitedClaim:
    def test_evidence_bullet_with_no_matching_citation_is_flagged(self):
        answer = good_answer()
        answer["citations"] = [answer["citations"][0]]  # drop the second citation only
        report = validate_answer(answer, [ESVS_CONFLICT, NICE_THRESHOLD])
        assert not report.ok
        assert report.has(E_UNCITED_CLAIM)
        assert [c["chunk_id"] for c in report.uncited_claims] == [NICE_THRESHOLD["chunk_id"]]

    def test_evidence_bullet_with_no_chunk_id_is_flagged(self):
        answer = good_answer()
        answer["supporting_evidence"].append({"claim": "Repair is always indicated.", "chunk_id": ""})
        report = validate_answer(answer, [ESVS_CONFLICT, NICE_THRESHOLD])
        assert report.has(E_UNCITED_CLAIM)
        assert report.uncited_claims[-1]["chunk_id"] is None

    def test_flagged_not_dropped(self):
        answer = good_answer()
        answer["citations"] = [answer["citations"][0]]
        before = copy.deepcopy(answer)
        validate_answer(answer, [ESVS_CONFLICT, NICE_THRESHOLD])
        assert len(answer["supporting_evidence"]) == len(before["supporting_evidence"])


class TestValidatorOtherRules:
    def test_answer_with_no_citations_at_all_is_an_error(self):
        answer = good_answer()
        answer["citations"] = []
        answer["supporting_evidence"] = []
        report = validate_answer(answer, [ESVS_CONFLICT])
        assert report.has(E_ANSWER_WITHOUT_CITATIONS)

    def test_missing_required_field(self):
        answer = good_answer()
        del answer["confidence"]
        report = validate_answer(answer, [ESVS_CONFLICT, NICE_THRESHOLD])
        assert report.has(E_MISSING_FIELD)

    def test_percentage_confidence_is_rejected(self):
        answer = good_answer()
        answer["confidence"] = "85%"
        report = validate_answer(answer, [ESVS_CONFLICT, NICE_THRESHOLD])
        assert not report.ok
        assert report.has(E_INVALID_CONFIDENCE)
        assert report.has(E_NUMERIC_CONFIDENCE)

    def test_quoted_guideline_percentage_is_not_treated_as_certainty(self):
        answer = good_answer()
        answer["recommendation"] += " Rupture is fatal in more than 80% of cases."
        report = validate_answer(answer, [ESVS_CONFLICT, NICE_THRESHOLD])
        assert not report.has(W_NUMERIC_CERTAINTY_PROSE)
        assert report.ok

    def test_numeric_certainty_next_to_a_confidence_word_warns(self):
        answer = good_answer()
        answer["recommendation"] += " I am 90% confident in this."
        report = validate_answer(answer, [ESVS_CONFLICT, NICE_THRESHOLD])
        assert report.has(W_NUMERIC_CERTAINTY_PROSE)
        assert report.ok, "advisory only: a quoted statistic cannot be told apart by pattern"

    def test_paraphrased_excerpt_warns(self):
        answer = good_answer()
        answer["citations"][0]["excerpt"] = "ESVS says repair below 55 millimetres is discouraged"
        report = validate_answer(answer, [ESVS_CONFLICT, NICE_THRESHOLD])
        assert report.has(W_EXCERPT_NOT_IN_CHUNK)

    def test_altered_retrieval_score_warns(self):
        answer = good_answer()
        answer["citations"][0]["retrieval_score"] = 0.95
        report = validate_answer(answer, [ESVS_CONFLICT, NICE_THRESHOLD])
        assert report.has(W_RETRIEVAL_SCORE_MISMATCH)

    def test_wrong_page_warns(self):
        answer = good_answer()
        answer["citations"][0]["page"] = 199
        report = validate_answer(answer, [ESVS_CONFLICT, NICE_THRESHOLD])
        assert report.has(W_CITATION_METADATA_MISMATCH)

    def test_non_canonical_disclaimer_warns(self):
        answer = good_answer()
        answer["disclaimer"] = "Not medical advice."
        report = validate_answer(answer, [ESVS_CONFLICT, NICE_THRESHOLD])
        assert report.has(W_DISCLAIMER_NOT_CANONICAL)

    def test_refusal_without_the_standard_message_is_an_error(self):
        answer = good_answer()
        answer["confidence"] = CONFIDENCE_INSUFFICIENT
        answer["recommendation"] = "I don't know."
        report = validate_answer(answer, [ESVS_CONFLICT, NICE_THRESHOLD])
        assert report.has(E_REFUSAL_MESSAGE_MISSING)

    def test_non_dict_answer_reports_rather_than_raises(self):
        report = validate_answer("not an object", [ESVS_CONFLICT])
        assert not report.ok


# ---------------------------------------------------------------------------
# Refusal threshold logic
# ---------------------------------------------------------------------------

class TestThresholdAndRefusal:
    def test_select_usable_hits_splits_on_the_floor(self):
        usable, dropped = select_usable_hits([ESVS_CONFLICT, NICE_THRESHOLD, WEAK_HIT], 0.75)
        assert [h["chunk_id"] for h in usable] == [ESVS_CONFLICT["chunk_id"], NICE_THRESHOLD["chunk_id"]]
        assert [h["chunk_id"] for h in dropped] == [WEAK_HIT["chunk_id"]]

    def test_boundary_score_is_usable(self):
        boundary = {**WEAK_HIT, "similarity_score": 0.75, "score": 0.75}
        usable, dropped = select_usable_hits([boundary], 0.75)
        assert usable and not dropped

    def test_all_below_threshold_refuses_without_calling_the_model(self):
        result = answer_question(
            "What is the five-year survival rate after kidney transplantation?",
            retriever=FakeRetriever([WEAK_HIT]),
            provider=ExplodingProvider(),
            settings=settings_for(threshold=0.75),
        )
        assert result.refused
        assert result.refusal["reason"] == REFUSAL_BELOW_THRESHOLD
        assert result.refusal["gate"] == "threshold"
        assert is_refusal(result.answer)
        assert result.answer["confidence"] == CONFIDENCE_INSUFFICIENT
        assert result.used_chunk_ids == []
        assert len(result.dropped_chunks) == 1

    def test_empty_retrieval_refuses(self):
        result = answer_question(
            "What is the recommended insulin regimen for type 2 diabetes?",
            retriever=FakeRetriever([]),
            provider=ExplodingProvider(),
            settings=settings_for(),
        )
        assert result.refused
        assert result.refusal["reason"] == REFUSAL_NO_CHUNKS
        assert result.answer["citations"] == []

    def test_threshold_refusal_is_helpful(self):
        """Slide 10: name what was found, what is missing, what would answer it."""
        result = answer_question(
            "What are the screening recommendations for thoracic aortic aneurysm?",
            retriever=FakeRetriever([WEAK_HIT]),
            provider=ExplodingProvider(),
            settings=settings_for(threshold=0.75),
        )
        text = result.answer["recommendation"]
        assert text.startswith(REFUSAL_MESSAGE)
        assert "SVS_2018" in text, "must name the evidence that was found"
        assert "0.75" in text, "must say what floor it failed"
        assert "would need" in text.lower(), "must say what would answer it"
        assert result.answer["citations"], "a threshold refusal cites what it examined"

    def test_above_threshold_answers(self):
        provider = FakeProvider(good_answer())
        result = answer_question(
            "Is the threshold for elective AAA repair 55 mm or 60 mm?",
            retriever=FakeRetriever([ESVS_CONFLICT, NICE_THRESHOLD, WEAK_HIT]),
            provider=provider,
            settings=settings_for(threshold=0.75),
        )
        assert not result.refused
        assert len(provider.calls) == 1
        assert result.used_chunk_ids == [ESVS_CONFLICT["chunk_id"], NICE_THRESHOLD["chunk_id"]]
        assert result.validation["ok"]

    def test_only_usable_chunks_reach_the_prompt(self):
        provider = FakeProvider(good_answer())
        answer_question(
            "threshold question",
            retriever=FakeRetriever([ESVS_CONFLICT, NICE_THRESHOLD, WEAK_HIT]),
            provider=provider,
            settings=settings_for(threshold=0.75),
        )
        user_message = provider.calls[0]["messages"][1]["content"]
        assert ESVS_CONFLICT["chunk_id"] in user_message
        assert WEAK_HIT["chunk_id"] not in user_message

    def test_model_judged_refusal_is_recorded(self):
        refusal = good_answer()
        refusal["confidence"] = CONFIDENCE_INSUFFICIENT
        refusal["recommendation"] = REFUSAL_MESSAGE + " The retrieved passages concern the abdominal aorta only."
        result = answer_question(
            "What are the screening recommendations for thoracic aortic aneurysm?",
            retriever=FakeRetriever([ESVS_CONFLICT, NICE_THRESHOLD]),
            provider=FakeProvider(refusal),
            settings=settings_for(threshold=0.75),
        )
        assert result.refused
        assert result.refusal["gate"] == "model"

    def test_disclaimer_is_normalised_and_the_change_is_recorded(self):
        answer = good_answer()
        answer["disclaimer"] = "short"
        result = answer_question(
            "threshold question",
            retriever=FakeRetriever([ESVS_CONFLICT, NICE_THRESHOLD]),
            provider=FakeProvider(answer),
            settings=settings_for(),
        )
        assert result.answer["disclaimer"] == DISCLAIMER
        assert result.disclaimer_normalised is True
        assert W_DISCLAIMER_NOT_CANONICAL in result.validation["codes"]


# ---------------------------------------------------------------------------
# Conflicting evidence
# ---------------------------------------------------------------------------

def conflict_answer() -> dict:
    answer = good_answer()
    answer["recommendation"] = (
        "The retrieved guidelines do not agree. NICE NG156 and the ESVS 2024 recommendation set "
        "the diameter for considering elective repair at 55 mm (5.5 cm), while ESVS 2024 also "
        "records a proposal, based on NAAASP data, to raise the threshold to 60 mm when measured "
        "on CTA -- a proposal the ESVS writing committee explicitly declines to adopt."
    )
    answer["confidence"] = "Low"
    answer["evidence_conflicts"] = [
        {
            "topic": "Diameter threshold for considering elective AAA repair",
            "positions": [
                {
                    "position": "55 mm is the diameter at which elective repair is considered.",
                    "source": "NICE_NG156 / ESVS_2024",
                    "chunk_ids": [NICE_THRESHOLD["chunk_id"], ESVS_CONFLICT["chunk_id"]],
                },
                {
                    "position": "60 mm has been suggested when the measurement is based on CTA, "
                    "but ESVS 2024 does not consider the support sufficient.",
                    "source": "ESVS_2024 (reporting NAAASP)",
                    "chunk_ids": [ESVS_CONFLICT["chunk_id"]],
                },
            ],
        }
    ]
    return answer


def one_sided_answer() -> dict:
    """A well-formed answer that reports ONLY the adopted 55 mm position.

    `good_answer()` cannot stand in for this. Its recommendation names the 60 mm
    proposal in order to reject it -- which is the two-sided behaviour this case
    exists to detect the *absence* of -- so asserting "60" not in its prose
    asserts the opposite of what the fixture says.
    """
    answer = good_answer()
    answer["recommendation"] = (
        "Elective repair is considered at a maximum aortic diameter of 55 mm "
        "(5.5 cm) in the retrieved guidelines."
    )
    return answer


class TestConflictingEvidence:
    def test_both_positions_are_presented_with_their_own_citations(self):
        chunks = [ESVS_CONFLICT, NICE_THRESHOLD]
        result = answer_question(
            "Is the diameter threshold for considering elective repair of an AAA 55 mm or 60 mm?",
            retriever=FakeRetriever(chunks),
            provider=FakeProvider(conflict_answer()),
            settings=settings_for(threshold=0.75),
        )
        assert not result.refused
        assert result.validation["ok"], result.validation["findings"]

        prose = answer_prose(result.answer)
        assert "55" in prose and "60" in prose, "both positions must appear, not just the adopted one"
        assert conflict_position_count(result.answer) >= 2
        assert set(result.documents_cited) == {"ESVS_2024", "NICE_NG156"}

    def test_conflict_position_citing_an_unsent_chunk_is_an_error(self):
        answer = conflict_answer()
        answer["evidence_conflicts"][0]["positions"][1]["chunk_ids"] = ["ESVS_2024__p77-77__c7777"]
        report = validate_answer(answer, [ESVS_CONFLICT, NICE_THRESHOLD])
        assert not report.ok
        assert "ESVS_2024__p77-77__c7777" in report.hallucinated_chunk_ids

    def test_one_sided_answer_is_detectable(self):
        """Silently picking 55 mm is the failure mode this case exists to catch."""
        one_sided = one_sided_answer()  # mentions 55 only, no evidence_conflicts
        prose = answer_prose(one_sided)
        assert "55" in prose
        assert "60" not in prose
        assert conflict_position_count(one_sided) == 0


# ---------------------------------------------------------------------------
# Safety gate
# ---------------------------------------------------------------------------

class TestSafetyGate:
    @pytest.mark.parametrize(
        "query",
        [
            "My patient is a 72-year-old man with a 5.9 cm AAA and severe COPD. Should I offer him EVAR?",
            "What dose of atorvastatin should I start for my 68-year-old patient with a 4.5 cm aneurysm?",
            "This patient has a 6 cm aneurysm, what should we do?",
            "Does my patient have an aneurysm?",
            "A 74-year-old woman has a 5.2 cm aneurysm. Should I refer her for repair?",
        ],
    )
    def test_patient_specific_queries_are_blocked(self, query):
        assert screen_query(query).blocked, query

    @pytest.mark.parametrize(
        "query",
        [
            "What are the recommendations for screening for abdominal aortic aneurysm?",
            "What do the guidelines recommend regarding AAA screening in women?",
            "Should men aged 65 to 75 be screened for AAA?",
            "What statin dose do the guidelines recommend for patients with AAA?",
            "What ultrasound surveillance interval is recommended for a 45 mm aneurysm?",
            "Is the threshold for elective repair 55 mm or 60 mm?",
        ],
    )
    def test_general_questions_are_not_blocked(self, query):
        verdict = screen_query(query)
        assert not verdict.blocked, f"{query} -> {verdict.signals}"

    def test_blocked_query_refuses_locally_with_no_citations(self):
        result = answer_question(
            "My patient is a 72-year-old man with a 5.9 cm AAA. Should I offer him EVAR or open repair?",
            retriever=FakeRetriever([ESVS_CONFLICT, NICE_THRESHOLD]),
            provider=ExplodingProvider(),
            settings=settings_for(),
        )
        assert result.refused
        assert result.refusal["reason"] == REFUSAL_PATIENT_SPECIFIC
        assert result.refusal["gate"].startswith("safety:")
        assert result.answer["citations"] == [], "guideline text is not offered as an individual answer"
        assert result.safety["blocked"] is True
        assert result.retrieved, "the audit record still shows what retrieval would have found"


# ---------------------------------------------------------------------------
# System prompt (slide 6)
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    @pytest.mark.parametrize(
        "phrase",
        [
            "evidence-grounded clinical decision support assistant",
            "Use only the text inside the CONTEXT block",
            "Do not use outside knowledge",
            "the evidence is insufficient",
            "Do not provide patient-specific diagnosis",
            "must be traceable to a specific chunk",
        ],
    )
    def test_required_statements_are_present(self, phrase):
        assert phrase.lower() in SYSTEM_PROMPT.lower(), phrase

    @pytest.mark.parametrize(
        "vague",
        ["be accurate", "be helpful", "be polite", "use your best judgement", "do your best", "be precise and concise"],
    )
    def test_no_vague_instructions(self, vague):
        assert vague.lower() not in SYSTEM_PROMPT.lower(), vague

    def test_refusal_conditions_and_confidence_labels_are_explicit(self):
        assert "REFUSAL RULES" in SYSTEM_PROMPT
        assert REFUSAL_MESSAGE in SYSTEM_PROMPT
        for label in ("High", "Medium", "Low", "Insufficient Evidence"):
            assert label in SYSTEM_PROMPT
        assert DISCLAIMER in SYSTEM_PROMPT

    def test_context_block_exposes_chunk_id_and_score_for_copying(self):
        block = build_context_block([ESVS_CONFLICT])
        assert f"chunk_id: {ESVS_CONFLICT['chunk_id']}" in block
        assert "retrieval_score: 0.843058" in block
        assert ESVS_CONFLICT["chunk_text"][:40] in block

    def test_evidence_is_not_truncated(self):
        block = build_context_block([ESVS_CONFLICT])
        assert ESVS_CONFLICT["chunk_text"] in block
        assert "[TRUNCATED]" not in block

    def test_messages_carry_no_history(self):
        messages = build_messages("a question", [ESVS_CONFLICT])
        assert [m["role"] for m in messages] == ["system", "user"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParsing:
    def test_plain_json(self):
        answer, meta = parse_answer(json.dumps(good_answer()))
        assert answer["confidence"] == "High"
        assert meta["had_reasoning_block"] is False

    def test_code_fence(self):
        answer, meta = parse_answer("```json\n" + json.dumps(good_answer()) + "\n```")
        assert answer["confidence"] == "High"
        assert meta["had_code_fence"] is True

    def test_reasoning_block_is_discarded(self):
        payload = "<think>The chunk_id is ESVS_2024__p99-99__c9999 maybe</think>\n" + json.dumps(good_answer())
        answer, meta = parse_answer(payload)
        assert meta["had_reasoning_block"] is True
        assert "ESVS_2024__p99-99__c9999" not in json.dumps(answer)

    def test_unterminated_reasoning_block(self):
        answer, _ = parse_answer("<think>still thinking\n" + json.dumps(good_answer()))
        assert answer["confidence"] == "High"

    def test_trailing_prose_is_ignored(self):
        answer, meta = parse_answer(json.dumps(good_answer()) + "\n\nHope that helps!")
        assert answer["confidence"] == "High"
        assert meta["had_text_outside_object"] is True

    def test_braces_inside_strings_do_not_end_the_scan(self):
        payload = {"recommendation": "table cell {a} and {b}", "confidence": "Low"}
        answer, _ = parse_answer("noise " + json.dumps(payload) + " more noise")
        assert answer["recommendation"] == "table cell {a} and {b}"

    def test_empty_and_non_object_and_truncated_all_raise(self):
        for bad in ("", "   ", "[1, 2, 3]", '{"recommendation": "x"'):
            with pytest.raises(AnswerParseError):
                parse_answer(bad)


# ---------------------------------------------------------------------------
# Providers and config
# ---------------------------------------------------------------------------

class TestProviders:
    def test_the_two_required_models_are_pinned(self):
        assert PROVIDER_SPECS["groq"].model == "openai/gpt-oss-120b"
        assert PROVIDER_SPECS["openrouter"].model == "deepseek/deepseek-r1:free"
        assert PROVIDER_SPECS["groq"].key_env == "GROQ_API_KEY"
        assert PROVIDER_SPECS["openrouter"].key_env == "OPENROUTER_API_KEY"

    def test_both_are_openai_compatible_endpoints(self):
        for spec in PROVIDER_SPECS.values():
            assert spec.base_url.startswith("https://")

    def test_unknown_provider_raises(self):
        with pytest.raises(UnknownProviderError):
            resolve_provider_spec("gpt-4-turbo-ultra")

    def test_missing_api_key_raises_rather_than_returning_none(self):
        with pytest.raises(MissingAPIKeyError):
            OpenAICompatibleProvider(spec=PROVIDER_SPECS["groq"], model="m", api_key="")

    def test_provider_is_never_half_configured(self):
        provider = OpenAICompatibleProvider(spec=PROVIDER_SPECS["groq"], model="m", api_key="k")
        assert provider.model == "m" and provider.name == "groq"


def test_default_threshold_is_the_documented_starting_value():
    assert DEFAULT_SCORE_THRESHOLD == 0.75

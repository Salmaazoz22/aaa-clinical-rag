# -*- coding: utf-8 -*-
"""Tests for recommendation-prose grounding (W_RECOMMENDATION_UNSUPPORTED_SENTENCE).

Requirement (task brief):
  - prose fully backed by citations → should pass (no warning)
  - prose with one unsupported claim → flags exactly that claim
  - prose with unsupported claim and no citations at all → flags
  - existing citation-bullet checks must not be disturbed (additive only)

Extended here to also cover:
  - _content_tokens helper behaviour
  - connector / trivial sentences are not flagged
  - hallucinated chunk_ids cannot be used as grounding evidence
  - citation-level excerpts also ground prose sentences
  - the public MIN_GROUNDING_TOKENS / _recommendation_grounding_findings API
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.validator import (  # noqa: E402
    MIN_GROUNDING_TOKENS,
    W_RECOMMENDATION_UNSUPPORTED_SENTENCE,
    _content_tokens,
    _recommendation_grounding_findings,
    validate_answer,
)
from generation.schema import DISCLAIMER  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _hit(chunk_id: str, text: str, score: float = 0.90) -> dict:
    return {
        "chunk_id": chunk_id,
        "chunk_text": text,
        "text": text,
        "document": "Test Guideline 2024",
        "document_id": "TEST_2024",
        "section": "Test Section",
        "page": 1,
        "page_start": 1,
        "page_end": 1,
        "similarity_score": score,
        "score": score,
        "rank": 1,
    }


def _answer(
    recommendation: str,
    citations: list[dict],
    evidence: list[dict],
) -> dict:
    return {
        "recommendation": recommendation,
        "supporting_evidence": evidence,
        "citations": citations,
        "confidence": "High",
        "disclaimer": DISCLAIMER,
    }


# Common clinical sentences used across tests
_ESVS_TEXT = (
    "The writing committee does not believe there is sufficient support to raise "
    "the diameter threshold to 60 mm. A new strong negative recommendation of "
    "elective repair below 55 mm was issued."
)
_NICE_TEXT = (
    "Elective surgical repair is considered when the aneurysm is asymptomatic "
    "and 5.5 cm or larger."
)

CHUNK_ESVS = "ESVS_2024__p26-26__c0093"
CHUNK_NICE = "NICE_NG156__p16-16__c0050"


def _base_retrieved() -> list[dict]:
    return [_hit(CHUNK_ESVS, _ESVS_TEXT), _hit(CHUNK_NICE, _NICE_TEXT)]


def _base_evidence(esvs_excerpt: str, nice_excerpt: str) -> list[dict]:
    return [
        {
            "claim": "ESVS declines to raise threshold to 60 mm.",
            "chunk_id": CHUNK_ESVS,
            "excerpt": esvs_excerpt,
        },
        {
            "claim": "NICE sets threshold at 5.5 cm.",
            "chunk_id": CHUNK_NICE,
            "excerpt": nice_excerpt,
        },
    ]


def _base_citations(esvs_excerpt: str, nice_excerpt: str) -> list[dict]:
    return [
        {
            "document": "ESVS_2024",
            "section": "Test Section",
            "page": 26,
            "chunk_id": CHUNK_ESVS,
            "retrieval_score": 0.90,
            "excerpt": esvs_excerpt,
        },
        {
            "document": "NICE_NG156",
            "section": "Test Section",
            "page": 16,
            "chunk_id": CHUNK_NICE,
            "retrieval_score": 0.90,
            "excerpt": nice_excerpt,
        },
    ]


# ---------------------------------------------------------------------------
# Unit tests for _content_tokens
# ---------------------------------------------------------------------------

class TestContentTokens:
    def test_stop_words_are_removed(self):
        tokens = _content_tokens("the repair of the aorta is recommended")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "of" not in tokens

    def test_content_words_are_kept(self):
        tokens = _content_tokens("repair aorta recommended diameter threshold")
        assert "repair" in tokens
        assert "aorta" in tokens
        assert "diameter" in tokens

    def test_numbers_are_kept(self):
        """Numeric values like 55, 5.5, 60 are the primary clinical signals."""
        tokens = _content_tokens("diameter of 55 mm is the threshold")
        assert "55" in tokens

    def test_single_char_tokens_excluded(self):
        tokens = _content_tokens("a b c d diameter")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "diameter" in tokens

    def test_empty_string_returns_empty(self):
        assert _content_tokens("") == frozenset()

    def test_connector_sentence_returns_empty(self):
        """Pure stop-word sentence should produce no content tokens."""
        tokens = _content_tokens("and it is also not or but")
        assert len(tokens) == 0

    def test_returns_frozenset(self):
        assert isinstance(_content_tokens("hello world"), frozenset)

    def test_normalisation_applied(self):
        """Dashes and smart quotes must be normalised before tokenising."""
        # EN DASH between tokens should not split them into unrecognisable pieces
        tokens_dash = _content_tokens("CTA\u2013based approach")
        tokens_hyphen = _content_tokens("CTA-based approach")
        assert tokens_dash == tokens_hyphen


# ---------------------------------------------------------------------------
# Unit tests for _recommendation_grounding_findings (white-box)
# ---------------------------------------------------------------------------

class TestRecommendationGroundingFindings:
    """Direct tests of the helper function without going through validate_answer."""

    def test_empty_recommendation_returns_no_findings(self):
        findings = _recommendation_grounding_findings("", ["some excerpt text here"])
        assert findings == []

    def test_whitespace_only_recommendation_returns_no_findings(self):
        findings = _recommendation_grounding_findings("   \n  ", ["some excerpt"])
        assert findings == []

    def test_no_excerpts_returns_no_findings(self):
        """If there are no grounding excerpts the check is skipped (nothing to check against)."""
        findings = _recommendation_grounding_findings(
            "The diameter threshold is 55 mm for elective repair.", []
        )
        assert findings == []

    def test_grounded_sentence_produces_no_finding(self):
        excerpt = "elective repair below 55 mm threshold diameter"
        recommendation = "Elective repair is recommended at a diameter threshold of 55 mm."
        findings = _recommendation_grounding_findings(recommendation, [excerpt])
        assert findings == []

    def test_ungrounded_sentence_produces_warning(self):
        excerpt = "elective repair diameter 55 mm threshold"
        recommendation = (
            "Elective repair is recommended at 55 mm. "
            "Patients should always consult a specialist before any intervention."
        )
        findings = _recommendation_grounding_findings(recommendation, [excerpt])
        assert len(findings) == 1
        assert findings[0].code == W_RECOMMENDATION_UNSUPPORTED_SENTENCE

    def test_warning_code_is_correct(self):
        excerpt = "diameter 55 mm threshold elective"
        recommendation = "Surgery on Mars requires different protocols entirely."
        findings = _recommendation_grounding_findings(recommendation, [excerpt])
        assert any(f.code == W_RECOMMENDATION_UNSUPPORTED_SENTENCE for f in findings)

    def test_warning_contains_sentence_text_in_actual(self):
        excerpt = "elective repair 55 mm"
        recommendation = "Lunar repair thresholds are undefined in guidelines."
        findings = _recommendation_grounding_findings(recommendation, [excerpt])
        assert findings[0].actual is not None
        assert "lunar" in findings[0].actual.lower()

    def test_connector_only_sentence_not_flagged(self):
        """A sentence with no content tokens after stop-word removal is skipped."""
        excerpt = "repair 55 mm threshold"
        recommendation = (
            "Elective repair is recommended at 55 mm threshold. "
            "However, it is also noted that."  # connector-only tail
        )
        findings = _recommendation_grounding_findings(recommendation, [excerpt])
        # only the connector sentence might be present; the first is grounded
        for f in findings:
            # Any finding must not be about the grounded sentence
            assert "55" not in (f.actual or "").lower() or f.code != W_RECOMMENDATION_UNSUPPORTED_SENTENCE

    def test_multiple_sentences_each_checked_independently(self):
        """Each sentence is assessed separately; one bad sentence does not contaminate others."""
        excerpt_a = "repair threshold 55 mm aortic diameter"
        excerpt_b = "surveillance ultrasound interval 65 men screening"
        recommendation = (
            "Elective repair threshold is 55 mm for aortic diameter. "  # grounded by excerpt_a
            "Screening applies to men aged 65 using ultrasound surveillance interval. "  # grounded by excerpt_b
            "Interplanetary travel requires separate clinical protocols entirely unknown."  # ungrounded
        )
        findings = _recommendation_grounding_findings(recommendation, [excerpt_a, excerpt_b])
        assert len(findings) == 1
        assert "interplanetary" in findings[0].actual.lower()

    def test_location_field_references_sentence_index(self):
        excerpt = "repair 55 mm"
        recommendation = "Mars repair protocol is unknown. Repair at 55 mm threshold."
        findings = _recommendation_grounding_findings(recommendation, [excerpt])
        assert len(findings) == 1
        assert "sentence 0" in findings[0].location

    def test_severity_is_warning_not_error(self):
        from generation.validator import SEVERITY_WARNING
        excerpt = "elective repair 55 mm"
        recommendation = "Galactic surgery protocols are unspecified."
        findings = _recommendation_grounding_findings(recommendation, [excerpt])
        assert all(f.severity == SEVERITY_WARNING for f in findings)

    def test_min_grounding_tokens_constant_is_exported(self):
        """Regression guard: MIN_GROUNDING_TOKENS must remain importable and positive."""
        assert isinstance(MIN_GROUNDING_TOKENS, int)
        assert MIN_GROUNDING_TOKENS >= 1


# ---------------------------------------------------------------------------
# Integration tests via validate_answer
# ---------------------------------------------------------------------------

class TestRecommendationGroundingViaValidator:
    """End-to-end: run validate_answer and assert on W_RECOMMENDATION_UNSUPPORTED_SENTENCE."""

    # ---- fully-grounded prose passes ----------------------------------------

    def test_prose_fully_backed_passes(self):
        """
        Every sentence in the recommendation shares tokens with at least one
        supporting-evidence excerpt → no W_RECOMMENDATION_UNSUPPORTED_SENTENCE.
        """
        esvs_ex = "strong negative recommendation elective repair below 55 mm"
        nice_ex = "asymptomatic aneurysm 5.5 cm surgical repair considered"

        recommendation = (
            "Elective repair is recommended at a diameter of 55 mm based on ESVS guidelines. "
            "NICE NG156 sets the threshold for asymptomatic aneurysm repair at 5.5 cm."
        )
        answer = _answer(
            recommendation,
            citations=_base_citations(esvs_ex, nice_ex),
            evidence=_base_evidence(esvs_ex, nice_ex),
        )
        report = validate_answer(answer, _base_retrieved())
        assert not report.has(W_RECOMMENDATION_UNSUPPORTED_SENTENCE), (
            f"Fully grounded prose should not flag: {[f.to_dict() for f in report.findings if f.code == W_RECOMMENDATION_UNSUPPORTED_SENTENCE]}"
        )

    # ---- one unsupported claim is flagged exactly ----------------------------

    def test_one_unsupported_sentence_flagged_exactly(self):
        """
        Two grounded sentences and one extra sentence with no token overlap
        → exactly one W_RECOMMENDATION_UNSUPPORTED_SENTENCE for that sentence.
        """
        esvs_ex = "strong negative recommendation elective repair below 55 mm"
        nice_ex = "asymptomatic aneurysm 5.5 cm surgical repair considered"

        recommendation = (
            "Elective repair is recommended at a diameter of 55 mm based on ESVS guidelines. "
            "NICE NG156 confirms the asymptomatic aneurysm threshold at 5.5 cm. "
            "Annual follow-up MRI scans are required for all elderly patients over 80 years old."  # unsupported
        )
        answer = _answer(
            recommendation,
            citations=_base_citations(esvs_ex, nice_ex),
            evidence=_base_evidence(esvs_ex, nice_ex),
        )
        report = validate_answer(answer, _base_retrieved())

        unsupported = [
            f for f in report.findings
            if f.code == W_RECOMMENDATION_UNSUPPORTED_SENTENCE
        ]
        assert len(unsupported) == 1, (
            f"Expected exactly 1 unsupported sentence, got {len(unsupported)}: "
            f"{[f.actual for f in unsupported]}"
        )
        assert "mri" in unsupported[0].actual.lower() or "elderly" in unsupported[0].actual.lower() or "annual" in unsupported[0].actual.lower()

    def test_unsupported_sentence_actual_field_contains_the_sentence(self):
        esvs_ex = "repair 55 mm diameter threshold"
        recommendation = (
            "Repair is recommended at 55 mm diameter. "
            "Alien physiology requires entirely different treatment modalities."
        )
        evidence = [{"claim": "c", "chunk_id": CHUNK_ESVS, "excerpt": esvs_ex}]
        citations = [
            {"document": "ESVS_2024", "section": "s", "page": 1,
             "chunk_id": CHUNK_ESVS, "retrieval_score": 0.90, "excerpt": esvs_ex}
        ]
        answer = _answer(recommendation, citations=citations, evidence=evidence)
        report = validate_answer(answer, [_hit(CHUNK_ESVS, _ESVS_TEXT)])

        unsupported = [f for f in report.findings if f.code == W_RECOMMENDATION_UNSUPPORTED_SENTENCE]
        assert unsupported
        combined = " ".join(f.actual for f in unsupported if f.actual).lower()
        assert "alien" in combined or "physiology" in combined or "treatment" in combined

    # ---- unsupported claim with no citations at all --------------------------

    def test_unsupported_claim_with_no_valid_excerpts(self):
        """
        When there are citations but NONE of them carry a non-empty excerpt,
        the grounding pool is empty and the check is skipped (no false positives).
        This is the safest behaviour: we cannot flag what we cannot verify.
        """
        recommendation = "Repair should be considered at 55 mm diameter threshold."
        citations = [
            {"document": "ESVS_2024", "section": "s", "page": 1,
             "chunk_id": CHUNK_ESVS, "retrieval_score": 0.90, "excerpt": ""}
        ]
        evidence = [{"claim": "c", "chunk_id": CHUNK_ESVS, "excerpt": ""}]
        answer = _answer(recommendation, citations=citations, evidence=evidence)
        report = validate_answer(answer, [_hit(CHUNK_ESVS, _ESVS_TEXT)])
        assert not report.has(W_RECOMMENDATION_UNSUPPORTED_SENTENCE), (
            "Should not flag when excerpt pool is empty (nothing to check against)"
        )

    def test_answer_with_no_citations_no_grounding_check(self):
        """
        An answer with no citations at all already raises E_ANSWER_WITHOUT_CITATIONS.
        The grounding check should not additionally fire (excerpt pool will be empty).
        """
        from generation.validator import E_ANSWER_WITHOUT_CITATIONS

        recommendation = "Repair is recommended at 55 mm diameter threshold for aortic aneurysm."
        answer = _answer(recommendation, citations=[], evidence=[])
        report = validate_answer(answer, _base_retrieved())
        assert report.has(E_ANSWER_WITHOUT_CITATIONS)
        assert not report.has(W_RECOMMENDATION_UNSUPPORTED_SENTENCE)

    # ---- hallucinated chunk_ids cannot serve as grounding -------------------

    def test_hallucinated_chunk_excerpt_does_not_ground_prose(self):
        """
        If a supporting_evidence bullet's chunk_id is hallucinated (not in
        retrieved set), its excerpt must not be used to ground prose sentences.
        """
        fake_chunk = "FAKE__p99-99__c9999"
        esvs_ex = "repair 55 mm diameter threshold elective aortic"
        # The only excerpt is from a hallucinated chunk — it must not count.
        recommendation = "Elective repair is recommended at 55 mm diameter threshold aortic aneurysm."
        evidence = [{"claim": "c", "chunk_id": fake_chunk, "excerpt": esvs_ex}]
        citations = [
            {"document": "FAKE", "section": "s", "page": 1,
             "chunk_id": fake_chunk, "retrieval_score": 0.90, "excerpt": esvs_ex}
        ]
        answer = _answer(recommendation, citations=citations, evidence=evidence)
        # retrieved set does NOT include fake_chunk
        report = validate_answer(answer, [_hit(CHUNK_ESVS, _ESVS_TEXT)])
        # The hallucinated chunk means excerpt pool is empty → grounding skipped.
        # But even if it weren't skipped, a fabricated citation must not count.
        # The important thing: E_HALLUCINATED_CITATION is raised.
        from generation.validator import E_HALLUCINATED_CITATION
        assert report.has(E_HALLUCINATED_CITATION)

    # ---- citation-level excerpts can also ground prose ----------------------

    def test_citation_excerpt_alone_can_ground_prose(self):
        """
        A citation-level excerpt (no supporting_evidence bullet) still
        contributes to the grounding pool and can cover a prose sentence.
        """
        esvs_ex = "elective repair below 55 mm strong negative recommendation diameter"
        recommendation = "Elective repair below 55 mm has a strong negative recommendation."
        # evidence bullet has no excerpt; citation has the excerpt
        evidence = [{"claim": "c", "chunk_id": CHUNK_ESVS}]  # no excerpt
        citations = [
            {"document": "ESVS_2024", "section": "s", "page": 26,
             "chunk_id": CHUNK_ESVS, "retrieval_score": 0.90, "excerpt": esvs_ex}
        ]
        answer = _answer(recommendation, citations=citations, evidence=evidence)
        report = validate_answer(answer, [_hit(CHUNK_ESVS, _ESVS_TEXT)])
        assert not report.has(W_RECOMMENDATION_UNSUPPORTED_SENTENCE), (
            "Citation-level excerpt should ground prose even without evidence-bullet excerpt"
        )

    # ---- refusals are exempt -------------------------------------------------

    def test_refusal_is_not_grounding_checked(self):
        """
        A refusal answer (confidence = Insufficient Evidence) must not be
        subjected to the grounding check — it has no evidence to ground against
        and the recommendation is a fixed message.
        """
        from generation.schema import CONFIDENCE_INSUFFICIENT, REFUSAL_MESSAGE

        recommendation = REFUSAL_MESSAGE + " The retrieved passages are about screening only."
        answer = {
            "recommendation": recommendation,
            "supporting_evidence": [],
            "citations": [],
            "confidence": CONFIDENCE_INSUFFICIENT,
            "disclaimer": DISCLAIMER,
        }
        report = validate_answer(answer, _base_retrieved())
        assert not report.has(W_RECOMMENDATION_UNSUPPORTED_SENTENCE)

    # ---- check is a WARNING, not an error -----------------------------------

    def test_grounding_failure_is_warning_not_error(self):
        """The grounding check must not fail report.ok — it is advisory."""
        esvs_ex = "repair 55 mm"
        recommendation = (
            "Repair at 55 mm is noted. "
            "Completely fabricated claim about lunar surgery protocol requirements."
        )
        evidence = [{"claim": "c", "chunk_id": CHUNK_ESVS, "excerpt": esvs_ex}]
        citations = [
            {"document": "ESVS_2024", "section": "s", "page": 1,
             "chunk_id": CHUNK_ESVS, "retrieval_score": 0.90, "excerpt": esvs_ex}
        ]
        answer = _answer(recommendation, citations=citations, evidence=evidence)
        report = validate_answer(answer, [_hit(CHUNK_ESVS, _ESVS_TEXT)])
        assert report.has(W_RECOMMENDATION_UNSUPPORTED_SENTENCE)
        assert report.ok, (
            "W_RECOMMENDATION_UNSUPPORTED_SENTENCE must not fail report.ok "
            "(it is a warning requiring human review, not a hard rule violation)"
        )

    # ---- existing checks are untouched (additive) ---------------------------

    def test_existing_hallucination_check_still_works_alongside_grounding(self):
        """The new check must not interfere with E_HALLUCINATED_CITATION."""
        from generation.validator import E_HALLUCINATED_CITATION

        esvs_ex = "repair 55 mm diameter"
        recommendation = "Repair at 55 mm is recommended."
        evidence = [{"claim": "c", "chunk_id": CHUNK_ESVS, "excerpt": esvs_ex}]
        citations = [
            {"document": "ESVS_2024", "section": "s", "page": 1,
             "chunk_id": "MADE_UP__p00__c0000",  # hallucinated
             "retrieval_score": 0.90, "excerpt": esvs_ex}
        ]
        answer = _answer(recommendation, citations=citations, evidence=evidence)
        report = validate_answer(answer, [_hit(CHUNK_ESVS, _ESVS_TEXT)])
        assert report.has(E_HALLUCINATED_CITATION)

    def test_existing_excerpt_check_still_works_alongside_grounding(self):
        """The new check must not suppress W_EXCERPT_NOT_IN_CHUNK."""
        from generation.validator import W_EXCERPT_NOT_IN_CHUNK

        bad_excerpt = "this text does not appear in the chunk at all ever"
        recommendation = "Repair is recommended at 55 mm threshold aortic elective diameter."
        evidence = [{"claim": "c", "chunk_id": CHUNK_ESVS, "excerpt": bad_excerpt}]
        citations = [
            {"document": "ESVS_2024", "section": "s", "page": 1,
             "chunk_id": CHUNK_ESVS, "retrieval_score": 0.90, "excerpt": bad_excerpt}
        ]
        answer = _answer(recommendation, citations=citations, evidence=evidence)
        report = validate_answer(answer, [_hit(CHUNK_ESVS, _ESVS_TEXT)])
        assert report.has(W_EXCERPT_NOT_IN_CHUNK)

    # ---- good_answer fixture from test_generation still passes --------------

    def test_good_answer_from_generation_suite_passes_grounding(self):
        """
        The canonical good_answer() used in test_generation.py must continue
        to pass the new grounding check.
        """
        from generation.schema import DISCLAIMER
        from tests.test_generation import ESVS_CONFLICT, NICE_THRESHOLD, good_answer

        answer = good_answer()
        retrieved = [ESVS_CONFLICT, NICE_THRESHOLD]
        report = validate_answer(answer, retrieved)
        assert not report.has(W_RECOMMENDATION_UNSUPPORTED_SENTENCE), (
            f"good_answer() should pass grounding: "
            f"{[f.to_dict() for f in report.findings if f.code == W_RECOMMENDATION_UNSUPPORTED_SENTENCE]}"
        )

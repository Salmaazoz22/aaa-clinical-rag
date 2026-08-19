# -*- coding: utf-8 -*-
"""Claim-to-evidence binding: contradiction, numeric drift, invented facts.

The gap these close: before them, a `supporting_evidence` bullet whose *claim*
said the opposite of the excerpt it cited validated clean, because nothing ever
compared the two. `validate_answer` checked that the excerpt was really in the
chunk and that the chunk was really sent -- both true of a claim that then
asserts the reverse of what it quotes.

Each test below states the failure in the form it was found in:

    claim:   "Elective repair is contraindicated and must never be offered at 55 mm."
    excerpt: "Elective repair should be considered for men"
    before:  ok=True
    after:   ok=False, claim_contradicts_excerpt

The valid-paraphrase cases are as important as the failing ones: a check that
rejects "ultrasound is the recommended modality" against an excerpt saying
"imaging surveillance using ultrasound" would be worse than no check at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.schema import DISCLAIMER  # noqa: E402
from generation.validator import (  # noqa: E402
    E_CLAIM_CONTRADICTS_EXCERPT,
    E_CLAIM_NUMERIC_MISMATCH,
    E_CLAIM_UNSUPPORTED_TERMS,
    E_RECOMMENDATION_UNSUPPORTED_FACT,
    E_UNCITED_CLAIM,
    E_WRONG_TYPE,
    MIN_NOVEL_TERMS,
    _measurements,
    _novel_terms,
    _numeric_conflicts,
    _polarity,
    validate_answer,
)

# ---------------------------------------------------------------------------
# Fixtures — one chunk, real ESVS/NICE phrasing
# ---------------------------------------------------------------------------

CHUNK_ID = "ESVS_2024__p26-26__c0093"
CHUNK_TEXT = (
    "Elective repair should be considered for men with an abdominal aortic aneurysm "
    "with a maximum diameter of 55 mm or larger. Imaging surveillance using duplex "
    "ultrasound is recommended every three years for aneurysms 30-39 mm in diameter."
)
EXCERPT = "Elective repair should be considered for men"


def _hit() -> dict:
    return {
        "chunk_id": CHUNK_ID,
        "chunk_text": CHUNK_TEXT,
        "text": CHUNK_TEXT,
        "document": "ESVS 2024 Clinical Practice Guidelines",
        "document_id": "ESVS_2024",
        "section": "Elective repair",
        "page": 26,
        "page_start": 26,
        "page_end": 26,
        "similarity_score": 0.88,
        "score": 0.88,
        "rank": 1,
    }


def _answer(*, claim: str, excerpt: str = EXCERPT, recommendation: str | None = None,
            evidence: list | None = None) -> dict:
    recommendation = recommendation or (
        "Elective repair should be considered for men at a maximum diameter of 55 mm."
    )
    if evidence is None:
        evidence = [{"claim": claim, "chunk_id": CHUNK_ID, "excerpt": excerpt}]
    return {
        "recommendation": recommendation,
        "supporting_evidence": evidence,
        "citations": [
            {
                "document": "ESVS 2024 Clinical Practice Guidelines",
                "section": "Elective repair",
                "page": 26,
                "chunk_id": CHUNK_ID,
                "retrieval_score": 0.88,
                "excerpt": excerpt,
            }
        ],
        "confidence": "High",
        "disclaimer": DISCLAIMER,
    }


def _report(**kwargs):
    return validate_answer(_answer(**kwargs), [_hit()])


# ---------------------------------------------------------------------------
# 1. Direct contradiction
# ---------------------------------------------------------------------------

class TestDirectContradiction:
    def test_contraindicated_against_should_be_considered(self):
        """The exact defect: claim reverses the excerpt it cites."""
        report = _report(
            claim="Elective repair is contraindicated and must never be offered at 55 mm."
        )
        assert report.has(E_CLAIM_CONTRADICTS_EXCERPT)
        assert not report.ok, "a claim contradicting its own excerpt must fail the report"

    def test_not_recommended_against_recommended(self):
        report = validate_answer(
            _answer(
                claim="Imaging surveillance with ultrasound is not recommended.",
                excerpt="Imaging surveillance using duplex ultrasound is recommended",
            ),
            [_hit()],
        )
        assert report.has(E_CLAIM_CONTRADICTS_EXCERPT)
        assert not report.ok

    def test_should_not_against_should(self):
        report = _report(claim="Elective repair should not be considered for men.")
        assert report.has(E_CLAIM_CONTRADICTS_EXCERPT)
        assert not report.ok

    def test_no_benefit_against_benefit(self):
        report = validate_answer(
            _answer(
                claim="There is no benefit to imaging surveillance with ultrasound.",
                excerpt="Imaging surveillance using duplex ultrasound is recommended",
            ),
            [_hit()],
        )
        assert report.has(E_CLAIM_CONTRADICTS_EXCERPT)

    def test_polarity_helper_reads_negation_over_modal(self):
        """'must never be offered' is a prohibition, not a recommendation."""
        assert _polarity("Elective repair is contraindicated and must never be offered") == "negative"
        assert _polarity("Elective repair should be considered for men") == "positive"
        assert _polarity("The aneurysm was 55 mm in diameter") is None

    def test_finding_carries_both_sides(self):
        report = _report(claim="Elective repair is contraindicated at 55 mm.")
        finding = next(f for f in report.findings if f.code == E_CLAIM_CONTRADICTS_EXCERPT)
        assert finding.chunk_id == CHUNK_ID
        assert finding.location.startswith("supporting_evidence[0]")
        assert "considered" in str(finding.expected)


# ---------------------------------------------------------------------------
# 2. Numeric contradiction
# ---------------------------------------------------------------------------

class TestNumericContradiction:
    def test_55_mm_reported_as_5_mm(self):
        report = _report(
            claim="Elective repair should be considered for men at 5 mm.",
            excerpt="a maximum diameter of 55 mm or larger",
        )
        assert report.has(E_CLAIM_NUMERIC_MISMATCH)
        assert not report.ok

    def test_every_three_years_reported_as_every_year(self):
        report = _report(
            claim="Ultrasound surveillance is recommended every year for these aneurysms.",
            excerpt="ultrasound is recommended every three years for aneurysms 30-39 mm",
        )
        assert report.has(E_CLAIM_NUMERIC_MISMATCH)
        assert not report.ok

    def test_unit_conversion_is_not_a_contradiction(self):
        """5.5 cm and 55 mm are the same measurement; flagging it would be wrong."""
        report = _report(
            claim="Elective repair is considered at 5.5 cm or larger.",
            excerpt="a maximum diameter of 55 mm or larger",
        )
        assert not report.has(E_CLAIM_NUMERIC_MISMATCH)

    def test_range_endpoints_both_count(self):
        report = _report(
            claim="Surveillance every three years applies to aneurysms 30-39 mm.",
            excerpt="every three years for aneurysms 30-39 mm in diameter",
        )
        assert not report.has(E_CLAIM_NUMERIC_MISMATCH)

    def test_unit_absent_from_excerpt_is_not_a_conflict(self):
        """The excerpt simply does not speak to it — that is not a contradiction."""
        assert _numeric_conflicts("repair after 12 months", "Elective repair should be considered") == []

    def test_measurements_helper(self):
        assert _measurements("55 mm")["mm"] == {55.0}
        assert _measurements("5.5 cm")["mm"] == {55.0}
        assert _measurements("every three years")["month"] == {36.0}
        assert _measurements("annually")["month"] == {12.0}
        assert _measurements("aneurysms 40-49 mm")["mm"] == {40.0, 49.0}
        assert "mm" not in _measurements("Recommendation 13")


# ---------------------------------------------------------------------------
# 3. Unsupported medical additions
# ---------------------------------------------------------------------------

class TestUnsupportedAddition:
    def test_recommendation_sentence_inventing_a_therapy(self):
        report = _report(
            claim="Elective repair is considered at 55 mm.",
            recommendation=(
                "Elective repair should be considered for men at 55 mm. "
                "Elective repair also requires lifelong dual antiplatelet therapy."
            ),
        )
        assert report.has(E_RECOMMENDATION_UNSUPPORTED_FACT)
        assert not report.ok

    def test_claim_inventing_a_drug(self):
        report = _report(
            claim="Metformin infusion reverses aneurysm expansion in diabetic men.",
        )
        assert report.has(E_CLAIM_UNSUPPORTED_TERMS)
        assert not report.ok

    def test_two_shared_words_no_longer_enough_to_pass(self):
        """A claim must not pass merely because it shares words with the evidence."""
        report = _report(
            claim="Elective repair mandates perioperative cerebrospinal drainage and "
                  "prophylactic thoracotomy in every man."
        )
        assert not report.ok

    def test_novel_terms_helper_ignores_paraphrase_vocabulary(self):
        """Evidence-discourse and qualifier words are never 'novel facts'."""
        assert _novel_terms(
            "The guideline explicitly recommends surveillance above that threshold.",
            [CHUNK_TEXT],
        ) == []

    def test_min_novel_terms_is_exported(self):
        assert MIN_NOVEL_TERMS >= 2


# ---------------------------------------------------------------------------
# 4. Valid paraphrases must still pass
# ---------------------------------------------------------------------------

class TestValidParaphrase:
    def test_reworded_claim_passes_clean(self):
        report = _report(claim="Repair is considered for men once the aneurysm reaches 55 mm.")
        assert report.ok, [f.to_dict() for f in report.errors]
        assert not report.has(E_CLAIM_CONTRADICTS_EXCERPT)
        assert not report.has(E_CLAIM_UNSUPPORTED_TERMS)

    def test_modality_paraphrase_passes(self):
        report = _report(
            claim="Duplex ultrasound is the recommended imaging modality for surveillance.",
            excerpt="Imaging surveillance using duplex ultrasound is recommended",
            recommendation="Duplex ultrasound is the recommended modality for surveillance.",
        )
        assert report.ok, [f.to_dict() for f in report.errors]

    def test_valid_existing_citation_still_validates(self):
        """The pre-existing citation/chunk checks are untouched by all of this."""
        report = _report(claim="Elective repair is considered at 55 mm or larger.")
        assert report.ok
        assert report.cited_chunk_ids == [CHUNK_ID]
        assert report.hallucinated_chunk_ids == []

    def test_negation_in_a_caveat_is_not_a_contradiction(self):
        """A trailing caveat carries 'not' with no stance verb behind it."""
        report = _report(
            claim="Elective repair is considered for men, though not in every case."
        )
        assert not report.has(E_CLAIM_CONTRADICTS_EXCERPT)


# ---------------------------------------------------------------------------
# 5. Malformed supporting_evidence (regression — behaviour must not change)
# ---------------------------------------------------------------------------

class TestMalformedSupportingEvidence:
    def test_entry_is_not_an_object(self):
        report = validate_answer(_answer(claim="x", evidence=["a bare string"]), [_hit()])
        assert report.has(E_WRONG_TYPE)
        assert not report.ok

    def test_entry_has_no_chunk_id(self):
        report = validate_answer(
            _answer(claim="x", evidence=[{"claim": "Repair at 55 mm.", "excerpt": EXCERPT}]),
            [_hit()],
        )
        assert report.has(E_UNCITED_CLAIM)
        assert not report.ok

    def test_supporting_evidence_is_not_a_list(self):
        report = validate_answer(
            _answer(claim="x", evidence=None) | {"supporting_evidence": {"claim": "x"}},
            [_hit()],
        )
        assert report.has(E_WRONG_TYPE)
        assert not report.ok

    def test_claim_checks_do_not_crash_on_a_missing_claim(self):
        report = validate_answer(
            _answer(claim="x", evidence=[{"chunk_id": CHUNK_ID, "excerpt": EXCERPT}]),
            [_hit()],
        )
        assert not report.ok  # missing_field, and nothing raised

    def test_claim_checks_skip_a_hallucinated_chunk_id(self):
        """A chunk that was never sent has no text to check a claim against."""
        report = validate_answer(
            _answer(claim="x", evidence=[{"claim": "Anything at all.", "chunk_id": "ghost-1"}]),
            [_hit()],
        )
        assert not report.ok
        assert not report.has(E_CLAIM_CONTRADICTS_EXCERPT)

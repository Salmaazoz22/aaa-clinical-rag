# -*- coding: utf-8 -*-
"""Tests for the answer-level evidence-grade summary.

Covers:
  - Pure function: summarize_evidence_grade()
  - All four data cases: all graded, mixed, none graded, refusal (no citations)
  - NaN / None / empty-string sentinel handling (_is_usable_grade)
  - GenerationResult.evidence_grade_summary populated correctly
  - POST /v1/answer JSON shape includes new key, all existing keys unchanged
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.validator import (  # noqa: E402
    _is_usable_grade,
    summarize_evidence_grade,
)
from generation.pipeline import GenerationResult  # noqa: E402
from generation.schema import DISCLAIMER  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers: build citations_resolved entries the way resolve_citations() does
# ---------------------------------------------------------------------------

def _resolved(
    chunk_id: str,
    recommendation_grade=None,
    evidence_level=None,
    resolved: bool = True,
) -> dict:
    """Mimic a single entry from resolve_citations()."""
    if not resolved:
        return {"chunk_id": chunk_id, "resolved": False, "reason": "chunk_id not in retrieved set"}
    return {
        "chunk_id": chunk_id,
        "resolved": True,
        "document": "Test Guideline",
        "document_id": "TEST_2024",
        "section": "Section 1",
        "page": 1,
        "page_start": 1,
        "page_end": 1,
        "retrieval_score": 0.90,
        "rank": 1,
        "recommendation_id": None,
        "recommendation_grade": recommendation_grade,
        "evidence_level": evidence_level,
        "source_file": "data/pdfs/test.pdf",
        "model_excerpt": "some excerpt",
    }


# ---------------------------------------------------------------------------
# Unit tests: _is_usable_grade (private but critical — test it directly)
# ---------------------------------------------------------------------------

class TestIsUsableGrade:
    def test_none_is_not_usable(self):
        assert _is_usable_grade(None) is False

    def test_nan_float_is_not_usable(self):
        assert _is_usable_grade(float("nan")) is False

    def test_nan_string_is_not_usable(self):
        assert _is_usable_grade("nan") is False
        assert _is_usable_grade("NaN") is False

    def test_none_string_is_not_usable(self):
        assert _is_usable_grade("none") is False
        assert _is_usable_grade("None") is False

    def test_null_string_is_not_usable(self):
        assert _is_usable_grade("null") is False

    def test_na_string_is_not_usable(self):
        assert _is_usable_grade("n/a") is False
        assert _is_usable_grade("N/A") is False

    def test_empty_string_is_not_usable(self):
        assert _is_usable_grade("") is False
        assert _is_usable_grade("   ") is False

    def test_grade_a_is_usable(self):
        assert _is_usable_grade("A") is True

    def test_grade_1_is_usable(self):
        assert _is_usable_grade("1") is True

    def test_grade_c_recommendation_is_usable(self):
        assert _is_usable_grade("C recommendation") is True

    def test_grade_i_statement_is_usable(self):
        assert _is_usable_grade("I statement") is True

    def test_integer_0_is_usable(self):
        """Non-string truthy value that is not None/NaN — treated as a grade."""
        assert _is_usable_grade(0) is True  # str(0)='0', valid grade string, not NaN

    def test_integer_nonzero_is_usable(self):
        assert _is_usable_grade(1) is True  # str(1) = '1', non-empty, not nan


# ---------------------------------------------------------------------------
# Unit tests: summarize_evidence_grade — four required cases
# ---------------------------------------------------------------------------

class TestSummarizeEvidenceGrade:

    # Case 1: all cited chunks have metadata
    def test_all_graded_available_true(self):
        citations = [
            _resolved("SVS_c1", recommendation_grade="1", evidence_level="A"),
            _resolved("SVS_c2", recommendation_grade="1", evidence_level="B"),
        ]
        result = summarize_evidence_grade(citations)
        assert result["available"] is True
        assert result["recommendation_grades"] == ["1"]        # deduplicated + sorted
        assert result["evidence_levels"] == ["A", "B"]         # sorted
        assert result["n_citations_with_grade"] == 2
        assert result["n_citations_total"] == 2

    # Case 2: only some chunks have metadata
    def test_mixed_metadata_available_true(self):
        citations = [
            _resolved("ESVS_c1", recommendation_grade=None, evidence_level=None),
            _resolved("SVS_c1", recommendation_grade="1", evidence_level="A"),
        ]
        result = summarize_evidence_grade(citations)
        assert result["available"] is True
        assert result["recommendation_grades"] == ["1"]
        assert result["evidence_levels"] == ["A"]
        assert result["n_citations_with_grade"] == 1   # only SVS chunk counted
        assert result["n_citations_total"] == 2

    # Case 3: no chunks have metadata (ESVS/NICE/USPSTF-only answers)
    def test_no_metadata_available_false(self):
        citations = [
            _resolved("ESVS_c1", recommendation_grade=None, evidence_level=None),
            _resolved("NICE_c1", recommendation_grade=None, evidence_level=None),
        ]
        result = summarize_evidence_grade(citations)
        assert result["available"] is False
        assert result["recommendation_grades"] == []
        assert result["evidence_levels"] == []
        assert result["n_citations_with_grade"] == 0
        assert result["n_citations_total"] == 2

    # Case 4: refusal answer — no citations at all
    def test_empty_citations_refusal_short_circuit(self):
        result = summarize_evidence_grade([])
        assert result["available"] is False
        assert result["recommendation_grades"] == []
        assert result["evidence_levels"] == []
        assert result["n_citations_with_grade"] == 0
        assert result["n_citations_total"] == 0

    def test_none_input_treated_as_empty(self):
        result = summarize_evidence_grade(None)
        assert result["available"] is False
        assert result["n_citations_total"] == 0

    # Shape contract — all four keys always present
    def test_shape_is_stable_all_cases(self):
        for citations in (
            [],
            [_resolved("c1", recommendation_grade=None, evidence_level=None)],
            [_resolved("c1", recommendation_grade="A", evidence_level="1")],
        ):
            result = summarize_evidence_grade(citations)
            assert set(result) == {
                "available",
                "recommendation_grades",
                "evidence_levels",
                "n_citations_with_grade",
                "n_citations_total",
            }, f"Shape broken for input {citations}"

    # Unresolved (hallucinated) citations must not contribute
    def test_unresolved_citations_excluded(self):
        citations = [
            _resolved("FAKE_c1", recommendation_grade="1", evidence_level="A", resolved=False),
            _resolved("SVS_c1", recommendation_grade=None, evidence_level=None),
        ]
        result = summarize_evidence_grade(citations)
        # Only the resolved-but-no-grade chunk counts
        assert result["available"] is False
        assert result["n_citations_total"] == 1   # unresolved is excluded

    # NaN sentinel values in real corpus data
    def test_nan_float_grade_treated_as_absent(self):
        citations = [
            _resolved("c1", recommendation_grade=float("nan"), evidence_level=float("nan")),
        ]
        result = summarize_evidence_grade(citations)
        assert result["available"] is False
        assert result["n_citations_with_grade"] == 0

    def test_nan_string_grade_treated_as_absent(self):
        citations = [
            _resolved("c1", recommendation_grade="nan", evidence_level="nan"),
        ]
        result = summarize_evidence_grade(citations)
        assert result["available"] is False

    # Deduplication
    def test_duplicate_grade_values_deduplicated(self):
        citations = [
            _resolved("c1", recommendation_grade="1", evidence_level="A"),
            _resolved("c2", recommendation_grade="1", evidence_level="A"),
            _resolved("c3", recommendation_grade="1", evidence_level="B"),
        ]
        result = summarize_evidence_grade(citations)
        assert result["recommendation_grades"] == ["1"]
        assert result["evidence_levels"] == ["A", "B"]

    # Sorting
    def test_output_lists_are_sorted(self):
        citations = [
            _resolved("c1", evidence_level="C"),
            _resolved("c2", evidence_level="A"),
            _resolved("c3", evidence_level="B"),
        ]
        result = summarize_evidence_grade(citations)
        assert result["evidence_levels"] == ["A", "B", "C"]

    # Real corpus SVS fixture values
    def test_svs_fixture_grades_pass_through_correctly(self):
        """SVS_SCREENING fixture has grade='1', level='A' — verify end-to-end."""
        from tests.test_generation import SVS_SCREENING
        from generation.validator import resolve_citations
        from generation.schema import DISCLAIMER

        # Minimal answer citing SVS chunk
        answer = {
            "recommendation": "Screening recommended.",
            "supporting_evidence": [
                {"claim": "SVS recommends screening.", "chunk_id": SVS_SCREENING["chunk_id"],
                 "excerpt": "We recommend using ultrasound"}
            ],
            "citations": [
                {
                    "document": SVS_SCREENING["document_id"],
                    "section": SVS_SCREENING["section"],
                    "page": SVS_SCREENING["page"],
                    "chunk_id": SVS_SCREENING["chunk_id"],
                    "retrieval_score": SVS_SCREENING["similarity_score"],
                    "excerpt": "We recommend using ultrasound",
                }
            ],
            "confidence": "High",
            "disclaimer": DISCLAIMER,
        }
        resolved = resolve_citations(answer, [SVS_SCREENING])
        summary = summarize_evidence_grade(resolved)
        assert summary["available"] is True
        assert summary["recommendation_grades"] == ["1"]
        assert summary["evidence_levels"] == ["A"]
        assert summary["n_citations_total"] == 1
        assert summary["n_citations_with_grade"] == 1

    def test_esvs_and_nice_fixture_no_grades(self):
        """ESVS/NICE fixtures have grade=None — available must be False."""
        from tests.test_generation import ESVS_CONFLICT, NICE_THRESHOLD, good_answer
        from generation.validator import resolve_citations

        resolved = resolve_citations(good_answer(), [ESVS_CONFLICT, NICE_THRESHOLD])
        summary = summarize_evidence_grade(resolved)
        assert summary["available"] is False
        assert summary["recommendation_grades"] == []
        assert summary["evidence_levels"] == []

    # exported from package top level
    def test_importable_from_generation_package(self):
        from generation import summarize_evidence_grade as f
        assert callable(f)


# ---------------------------------------------------------------------------
# GenerationResult integration
# ---------------------------------------------------------------------------

class TestGenerationResultField:
    """Verify evidence_grade_summary is populated on GenerationResult."""

    def _make_result(self, citations_resolved: list[dict]) -> GenerationResult:
        from generation.validator import summarize_evidence_grade
        r = GenerationResult(query="q", settings={}, safety={})
        r.citations_resolved = citations_resolved
        r.evidence_grade_summary = summarize_evidence_grade(citations_resolved)
        return r

    def test_field_exists_on_result(self):
        r = self._make_result([])
        assert hasattr(r, "evidence_grade_summary")

    def test_field_in_to_dict(self):
        r = self._make_result([])
        d = r.to_dict()
        assert "evidence_grade_summary" in d

    def test_existing_keys_unchanged(self):
        """No existing to_dict() key may be removed or renamed."""
        r = self._make_result([])
        d = r.to_dict()
        for key in (
            "query", "settings", "safety", "refused", "refusal",
            "answer", "citations_resolved", "documents_cited",
            "validation", "retrieval", "generation",
        ):
            assert key in d, f"Existing key {key!r} is missing from to_dict()"

    def test_to_dict_grade_summary_shape(self):
        citations = [_resolved("c1", recommendation_grade="1", evidence_level="A")]
        r = self._make_result(citations)
        gs = r.to_dict()["evidence_grade_summary"]
        assert gs["available"] is True
        assert gs["recommendation_grades"] == ["1"]
        assert gs["evidence_levels"] == ["A"]

    def test_refusal_result_has_empty_grade_summary(self):
        """A refusal with no citations should produce available=false summary."""
        r = self._make_result([])
        gs = r.to_dict()["evidence_grade_summary"]
        assert gs["available"] is False
        assert gs["n_citations_total"] == 0

    def test_pipeline_answer_question_populates_field(self):
        """End-to-end: answer_question() populates evidence_grade_summary."""
        import json
        from generation.pipeline import answer_question
        from tests.test_generation import (
            ESVS_CONFLICT, NICE_THRESHOLD, FakeRetriever,
            FakeProvider, good_answer, settings_for,
        )

        result = answer_question(
            "Is the threshold for elective repair 55 mm or 60 mm?",
            retriever=FakeRetriever([ESVS_CONFLICT, NICE_THRESHOLD]),
            provider=FakeProvider(good_answer()),
            settings=settings_for(threshold=0.75),
        )
        assert hasattr(result, "evidence_grade_summary")
        gs = result.evidence_grade_summary
        # ESVS/NICE have no grades — available must be False
        assert gs["available"] is False
        # Shape is always stable
        assert set(gs) == {
            "available", "recommendation_grades", "evidence_levels",
            "n_citations_with_grade", "n_citations_total",
        }
        # Must survive JSON serialisation (no NaN, no non-serialisable types)
        json.dumps(result.to_dict())  # must not raise

    def test_refusal_answer_question_populates_field(self):
        """finish_refusal path also populates evidence_grade_summary."""
        from generation.pipeline import answer_question
        from tests.test_generation import FakeRetriever, ExplodingProvider, settings_for, WEAK_HIT

        result = answer_question(
            "What is the threshold?",
            retriever=FakeRetriever([WEAK_HIT]),
            provider=ExplodingProvider(),
            settings=settings_for(threshold=0.75),
        )
        assert result.refused
        assert hasattr(result, "evidence_grade_summary")
        gs = result.evidence_grade_summary
        assert isinstance(gs, dict)
        assert "available" in gs


# ---------------------------------------------------------------------------
# API shape: POST /v1/answer shape is driven by GenerationResult.to_dict().
# Rather than importing FastAPI (anyio may be absent in this environment),
# we verify to_dict() directly — the API endpoint returns to_dict() verbatim.
# ---------------------------------------------------------------------------

class TestAnswerAPIShape:
    """POST /v1/answer returns result.to_dict() verbatim.
    Verifying to_dict() shape == verifying the API response shape.
    """

    def _make_result(self, citations_resolved: list[dict]) -> GenerationResult:
        from generation.validator import summarize_evidence_grade
        r = GenerationResult(query="What is the threshold?", settings={}, safety={})
        r.citations_resolved = citations_resolved
        r.documents_cited = ["ESVS_2024"]
        r.evidence_grade_summary = summarize_evidence_grade(citations_resolved)
        return r

    def test_response_includes_evidence_grade_summary_key(self):
        """to_dict() must expose evidence_grade_summary as a top-level key."""
        r = self._make_result([])
        d = r.to_dict()
        assert "evidence_grade_summary" in d, (
            "POST /v1/answer response (= to_dict()) must include evidence_grade_summary"
        )

    def test_response_evidence_grade_summary_shape(self):
        """The new key must carry exactly the five documented sub-keys."""
        citations = [_resolved("ESVS_c1", recommendation_grade=None, evidence_level=None)]
        r = self._make_result(citations)
        gs = r.to_dict()["evidence_grade_summary"]
        assert set(gs) == {
            "available", "recommendation_grades", "evidence_levels",
            "n_citations_with_grade", "n_citations_total",
        }, f"Unexpected shape: {set(gs)}"
        assert isinstance(gs["available"], bool)
        assert isinstance(gs["recommendation_grades"], list)
        assert isinstance(gs["evidence_levels"], list)
        assert isinstance(gs["n_citations_with_grade"], int)
        assert isinstance(gs["n_citations_total"], int)

    def test_existing_response_keys_unchanged(self):
        """No existing to_dict() key may be removed, renamed, or moved."""
        r = self._make_result([])
        d = r.to_dict()
        for key in (
            "query", "settings", "safety", "refused", "refusal",
            "answer", "citations_resolved", "documents_cited",
            "validation", "retrieval", "generation",
        ):
            assert key in d, f"Existing key {key!r} missing from to_dict()"

    def test_grade_summary_not_present_in_original_answer_field(self):
        """evidence_grade_summary must be top-level only, not inside 'answer'."""
        r = self._make_result([])
        d = r.to_dict()
        assert "evidence_grade_summary" not in d.get("answer", {}), (
            "evidence_grade_summary must be a top-level key, not nested inside 'answer'"
        )

    def test_to_dict_json_serialisable(self):
        """The full to_dict() payload must be JSON-serialisable (no NaN, no sets)."""
        import json
        # Use a grade value that is present so we exercise that path
        citations = [_resolved("SVS_c1", recommendation_grade="1", evidence_level="A")]
        r = self._make_result(citations)
        json.dumps(r.to_dict())  # must not raise


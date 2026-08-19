# -*- coding: utf-8 -*-
"""Tests for the typographic-normalisation fix in the citation validator.

Requirement (task brief):
  Add unit tests covering:
    - identical text with different dash types (should pass)
    - identical text with different quote types (should pass)
    - a genuinely mismatched excerpt (should still fail)

Extended here to also cover:
    - non-breaking / zero-width whitespace collapsing
    - NFKC compatibility decompositions
    - the full W_EXCERPT_NOT_IN_CHUNK path through validate_answer, so the fix
      is tested at the public API level, not just the normaliser in isolation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.validator import (  # noqa: E402
    W_EXCERPT_NOT_IN_CHUNK,
    normalise_for_match,
    validate_answer,
)
from generation.schema import DISCLAIMER  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: a minimal retrieved chunk and a matching answer structure
# ---------------------------------------------------------------------------

def _hit(chunk_id: str, text: str) -> dict:
    """Minimal chunk dict that satisfies the validator's field lookups."""
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
        "similarity_score": 0.90,
        "score": 0.90,
        "rank": 1,
    }


def _answer(chunk_id: str, excerpt: str, score: float = 0.90) -> dict:
    """Minimal answer dict with one citation and one supporting-evidence bullet."""
    return {
        "recommendation": "Test recommendation.",
        "supporting_evidence": [
            {
                "claim": "The guideline says so.",
                "chunk_id": chunk_id,
                "excerpt": excerpt,
            }
        ],
        "citations": [
            {
                "document": "TEST_2024",
                "section": "Test Section",
                "page": 1,
                "chunk_id": chunk_id,
                "retrieval_score": score,
                "excerpt": excerpt,
            }
        ],
        "confidence": "High",
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Unit tests for normalise_for_match directly
# ---------------------------------------------------------------------------

class TestNormaliseForMatch:
    """White-box tests on the normalization primitive itself."""

    # -- dash / hyphen variants ------------------------------------------------

    def test_ascii_hyphen_unchanged(self):
        assert normalise_for_match("CTA-based") == "cta-based"

    def test_en_dash_maps_to_hyphen(self):
        """EN DASH (U+2013) must normalise to ASCII hyphen, not space."""
        assert normalise_for_match("CTA\u2013based") == "cta-based"

    def test_em_dash_maps_to_hyphen(self):
        """EM DASH (U+2014) must normalise to ASCII hyphen."""
        assert normalise_for_match("CTA\u2014based") == "cta-based"

    def test_minus_sign_maps_to_hyphen(self):
        """Unicode MINUS SIGN (U+2212) must normalise to ASCII hyphen."""
        assert normalise_for_match("CTA\u2212based") == "cta-based"

    def test_hyphen_proper_maps_to_hyphen(self):
        """Unicode HYPHEN (U+2010) must normalise to ASCII hyphen."""
        assert normalise_for_match("well\u2010known") == "well-known"

    def test_non_breaking_hyphen_maps_to_hyphen(self):
        """NON-BREAKING HYPHEN (U+2011) must normalise to ASCII hyphen."""
        assert normalise_for_match("well\u2011known") == "well-known"

    def test_figure_dash_maps_to_hyphen(self):
        """FIGURE DASH (U+2012) must normalise to ASCII hyphen."""
        assert normalise_for_match("55\u20125.5") == "55-5.5"

    def test_horizontal_bar_maps_to_hyphen(self):
        """HORIZONTAL BAR (U+2015) must normalise to ASCII hyphen."""
        assert normalise_for_match("A\u2015B") == "a-b"

    def test_fullwidth_hyphen_maps_to_hyphen(self):
        """FULLWIDTH HYPHEN-MINUS (U+FF0D) must normalise to ASCII hyphen."""
        assert normalise_for_match("A\uff0dB") == "a-b"

    def test_all_dash_variants_equal_each_other(self):
        """All dash forms of the same phrase must normalise to the same string."""
        variants = [
            "CTA-based",       # ASCII hyphen
            "CTA\u2013based",  # EN DASH
            "CTA\u2014based",  # EM DASH
            "CTA\u2010based",  # HYPHEN
            "CTA\u2212based",  # MINUS SIGN
            "CTA\uff0dbased",  # FULLWIDTH HYPHEN-MINUS
        ]
        normalised = [normalise_for_match(v) for v in variants]
        assert len(set(normalised)) == 1, f"Variants diverged: {normalised}"

    # -- smart / typographic quotes --------------------------------------------

    def test_left_single_quote_maps_to_straight(self):
        assert normalise_for_match("\u2018don\u2019t") == "'don't"

    def test_right_single_quote_maps_to_straight(self):
        assert normalise_for_match("don\u2019t") == "don't"

    def test_left_double_quote_maps_to_straight(self):
        assert normalise_for_match("\u201cword\u201d") == '"word"'

    def test_right_double_quote_maps_to_straight(self):
        assert normalise_for_match("he said \u201dhello\u201c") == 'he said "hello"'

    def test_low_double_quote_maps_to_straight(self):
        """German-style „opening" quote (U+201E) -> ASCII double quote."""
        assert normalise_for_match("\u201eWort\u201c") == '"wort"'

    def test_grave_accent_maps_to_straight_single_quote(self):
        assert normalise_for_match("`word`") == "'word'"

    def test_acute_accent_maps_to_straight_single_quote(self):
        assert normalise_for_match("\u00b4word\u00b4") == "'word'"

    def test_all_quote_variants_equal_each_other(self):
        """All quote forms of the same phrase must normalise identically."""
        variants = [
            "\"it's\"",                # ASCII
            "\u201cit\u2019s\u201d",   # curly left/right
            "\u201fit\u2019s\u201e",   # less common curly
        ]
        normalised = [normalise_for_match(v) for v in variants]
        assert len(set(normalised)) == 1, f"Quote variants diverged: {normalised}"

    # -- whitespace variants ---------------------------------------------------

    def test_non_breaking_space_collapses(self):
        assert normalise_for_match("word\u00a0word") == "word word"

    def test_thin_space_collapses(self):
        """THIN SPACE (U+2009) must collapse to a single ASCII space."""
        assert normalise_for_match("word\u2009word") == "word word"

    def test_zero_width_space_removed(self):
        """ZERO-WIDTH SPACE (U+200B) must be stripped."""
        assert normalise_for_match("word\u200bword") == "word word"

    def test_zero_width_non_joiner_removed(self):
        assert normalise_for_match("word\u200cword") == "word word"

    def test_bom_removed(self):
        """BOM (U+FEFF) must be stripped."""
        assert normalise_for_match("\ufeffword") == "word"

    def test_multiple_mixed_spaces_collapse_to_one(self):
        assert normalise_for_match("a \u00a0  \u2009 b") == "a b"

    # -- NFKC decompositions ---------------------------------------------------

    def test_nfkc_fullwidth_letters(self):
        """Fullwidth ASCII letters (e.g. Ａ＝ U+FF21) must collapse to ASCII."""
        assert normalise_for_match("\uff21\uff21\uff21") == "aaa"

    def test_nfkc_fi_ligature(self):
        """fi ligature via NFKC -> 'fi'."""
        assert normalise_for_match("\ufb01nd") == "find"

    # -- None / empty guards ---------------------------------------------------

    def test_none_returns_empty_string(self):
        assert normalise_for_match(None) == ""

    def test_empty_string_returns_empty_string(self):
        assert normalise_for_match("") == ""

    def test_non_string_is_coerced(self):
        assert normalise_for_match(42) == "42"


# ---------------------------------------------------------------------------
# Integration tests: excerpt matching through validate_answer
# ---------------------------------------------------------------------------

class TestExcerptMatchingViaValidator:
    """
    End-to-end tests that exercise the W_EXCERPT_NOT_IN_CHUNK path.

    Each test creates a chunk whose stored text uses one typographic form and
    an excerpt that uses a different -- but semantically equivalent -- form.
    After the fix, the validator must NOT raise W_EXCERPT_NOT_IN_CHUNK.
    """

    # -- dash variants in source text vs excerpt --------------------------------

    def test_en_dash_in_source_plain_hyphen_in_excerpt(self):
        """
        Source text uses EN DASH; model excerpt uses ASCII hyphen.
        Before the fix: flagged as mismatch.
        After the fix: clean pass.
        """
        chunk_id = "TEST_DASH_1"
        source = "CTA\u2013based measurement is preferred."
        excerpt = "CTA-based measurement is preferred."
        hit = _hit(chunk_id, source)
        answer = _answer(chunk_id, excerpt)
        report = validate_answer(answer, [hit])
        assert not report.has(W_EXCERPT_NOT_IN_CHUNK), (
            "EN DASH in source should match ASCII hyphen in excerpt"
        )

    def test_em_dash_in_source_plain_hyphen_in_excerpt(self):
        """Source text uses EM DASH; excerpt uses ASCII hyphen."""
        chunk_id = "TEST_DASH_2"
        source = "well\u2014known complications include rupture."
        excerpt = "well-known complications include rupture."
        hit = _hit(chunk_id, source)
        answer = _answer(chunk_id, excerpt)
        report = validate_answer(answer, [hit])
        assert not report.has(W_EXCERPT_NOT_IN_CHUNK), (
            "EM DASH in source should match ASCII hyphen in excerpt"
        )

    def test_minus_in_source_plain_hyphen_in_excerpt(self):
        """Source text uses MINUS SIGN; excerpt uses ASCII hyphen."""
        chunk_id = "TEST_DASH_3"
        source = "a diameter \u2212 55 mm threshold was adopted."
        excerpt = "a diameter - 55 mm threshold was adopted."
        hit = _hit(chunk_id, source)
        answer = _answer(chunk_id, excerpt)
        report = validate_answer(answer, [hit])
        assert not report.has(W_EXCERPT_NOT_IN_CHUNK), (
            "MINUS SIGN in source should match ASCII hyphen in excerpt"
        )

    def test_plain_hyphen_in_source_en_dash_in_excerpt(self):
        """Source has ASCII hyphen; model excerpt has EN DASH -- also a valid match."""
        chunk_id = "TEST_DASH_4"
        source = "CTA-based measurement is preferred."
        excerpt = "CTA\u2013based measurement is preferred."
        hit = _hit(chunk_id, source)
        answer = _answer(chunk_id, excerpt)
        report = validate_answer(answer, [hit])
        assert not report.has(W_EXCERPT_NOT_IN_CHUNK), (
            "ASCII hyphen in source should match EN DASH in excerpt"
        )

    # -- quote variants in source text vs excerpt ------------------------------

    def test_curly_quotes_in_source_straight_in_excerpt(self):
        """Source uses smart quotes; excerpt uses straight ASCII quotes."""
        chunk_id = "TEST_QUOTE_1"
        source = "The \u201cGWC\u201d does not support the proposal."
        excerpt = 'The "GWC" does not support the proposal.'
        hit = _hit(chunk_id, source)
        answer = _answer(chunk_id, excerpt)
        report = validate_answer(answer, [hit])
        assert not report.has(W_EXCERPT_NOT_IN_CHUNK), (
            "Curly double quotes in source should match straight in excerpt"
        )

    def test_curly_single_quotes_in_source_straight_in_excerpt(self):
        """Source uses curly single quotes; excerpt uses straight apostrophe."""
        chunk_id = "TEST_QUOTE_2"
        source = "it\u2019s asymptomatic and 5.5 cm or larger."
        excerpt = "it's asymptomatic and 5.5 cm or larger."
        hit = _hit(chunk_id, source)
        answer = _answer(chunk_id, excerpt)
        report = validate_answer(answer, [hit])
        assert not report.has(W_EXCERPT_NOT_IN_CHUNK), (
            "Curly apostrophe in source should match straight apostrophe in excerpt"
        )

    def test_straight_quotes_in_source_curly_in_excerpt(self):
        """Source has straight quotes; excerpt has curly ones (still a match)."""
        chunk_id = "TEST_QUOTE_3"
        source = 'The "GWC" does not support this.'
        excerpt = "The \u201cGWC\u201d does not support this."
        hit = _hit(chunk_id, source)
        answer = _answer(chunk_id, excerpt)
        report = validate_answer(answer, [hit])
        assert not report.has(W_EXCERPT_NOT_IN_CHUNK), (
            "Straight quotes in source should match curly quotes in excerpt"
        )

    # -- non-breaking space in source ------------------------------------------

    def test_non_breaking_space_in_source_plain_space_in_excerpt(self):
        """Source uses NO-BREAK SPACE; excerpt uses regular space."""
        chunk_id = "TEST_NBSP_1"
        source = "55\u00a0mm is the threshold."
        excerpt = "55 mm is the threshold."
        hit = _hit(chunk_id, source)
        answer = _answer(chunk_id, excerpt)
        report = validate_answer(answer, [hit])
        assert not report.has(W_EXCERPT_NOT_IN_CHUNK), (
            "NO-BREAK SPACE in source should match regular space in excerpt"
        )

    # -- combined typographic differences --------------------------------------

    def test_combined_dash_and_quote_differences(self):
        """Source has both an EN DASH and a curly quote; excerpt uses ASCII equivalents."""
        chunk_id = "TEST_COMBINED"
        source = "The GWC\u2019s \u201cconsensus\u201d was: threshold\u2013based repair at 55 mm."
        excerpt = "The GWC's \"consensus\" was: threshold-based repair at 55 mm."
        hit = _hit(chunk_id, source)
        answer = _answer(chunk_id, excerpt)
        report = validate_answer(answer, [hit])
        assert not report.has(W_EXCERPT_NOT_IN_CHUNK), (
            "Combined typographic differences should not cause a false mismatch"
        )

    # -- genuine mismatches must still be caught --------------------------------

    def test_paraphrased_excerpt_still_fails(self):
        """
        A genuinely paraphrased excerpt (different words) must still be flagged.
        This confirms the fix does not over-normalise away real differences.
        """
        chunk_id = "TEST_PARAPHRASE"
        source = "Elective repair is recommended at a diameter of 55 mm."
        excerpt = "Surgery should be considered when the aneurysm exceeds 55 millimetres."
        hit = _hit(chunk_id, source)
        answer = _answer(chunk_id, excerpt)
        report = validate_answer(answer, [hit])
        assert report.has(W_EXCERPT_NOT_IN_CHUNK), (
            "A paraphrased excerpt (different words) must still be flagged"
        )

    def test_wrong_chunk_excerpt_still_fails(self):
        """Text taken from a different chunk (not the one cited) must be flagged."""
        chunk_a_id = "TEST_CHUNK_A"
        chunk_b_id = "TEST_CHUNK_B"
        source_a = "Screening is recommended for men aged 65 to 75."
        source_b = "Repair is considered at 55 mm diameter."
        hit_a = _hit(chunk_a_id, source_a)
        hit_b = _hit(chunk_b_id, source_b)
        # Excerpt from chunk B cited as if from chunk A:
        answer = _answer(chunk_a_id, "repair is considered at 55 mm diameter.")
        # Override score to match hit_a
        answer["citations"][0]["retrieval_score"] = 0.90
        report = validate_answer(answer, [hit_a, hit_b])
        assert report.has(W_EXCERPT_NOT_IN_CHUNK), (
            "Text from the wrong chunk must still be flagged as a mismatch"
        )

    def test_empty_excerpt_is_warned_not_mismatch(self):
        """An empty excerpt should produce W_EMPTY_EXCERPT, not W_EXCERPT_NOT_IN_CHUNK."""
        from generation.validator import W_EMPTY_EXCERPT

        chunk_id = "TEST_EMPTY"
        source = "Some text that exists."
        hit = _hit(chunk_id, source)
        answer = _answer(chunk_id, "")
        report = validate_answer(answer, [hit])
        assert report.has(W_EMPTY_EXCERPT)
        assert not report.has(W_EXCERPT_NOT_IN_CHUNK)

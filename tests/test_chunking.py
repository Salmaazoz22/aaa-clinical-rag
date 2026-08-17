# -*- coding: utf-8 -*-
"""Token-limit, chunk-ID and metadata tests for the chunking layer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

import clinical_chunking as cc  # noqa: E402

CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.json"
INDEX_CHUNKS_PATH = ROOT / "data" / "embeddings" / "embedded_chunks.json"


def _load_chunks():
    if not CHUNKS_PATH.exists():
        pytest.skip("chunks.json not built; run notebook 02 first")
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))["chunks"]


# --------------------------------------------------------------------------
# Token limit enforcement
# --------------------------------------------------------------------------

def test_model_token_limit_is_the_sentence_transformer_window():
    """The enforced limit must be the ACTIVE model's sentence-transformers window.

    Resolved from the model itself, never from `tokenizer.model_max_length`,
    which reports the backbone's position count and can be twice the real
    window (all-MiniLM-L6-v2: 256 window vs 512 positions). Asserting against
    the loaded model keeps this meaningful whichever model is configured.
    """
    model = cc.load_embedder()
    assert cc.model_token_limit() == int(model.max_seq_length)
    assert cc.model_token_limit() <= cc.MODEL_MAX_TOKENS


def test_embedding_model_is_pinned_to_a_revision():
    """Production weights must be pinned so the hub cannot change them silently."""
    assert cc.model_revision(cc.DEFAULT_EMBED_MODEL), "default embedding model is unpinned"


def test_split_text_never_exceeds_the_token_limit():
    limit = cc.model_token_limit()
    text = (
        "We recommend elective repair for the patient at low or acceptable surgical "
        "risk with a fusiform AAA that is greater or equal to 5.5 cm. "
    ) * 40
    pieces = cc.split_text(text)
    assert len(pieces) > 1, "long input must actually be split"
    for piece in pieces:
        assert cc.count_tokens(piece) <= limit


def test_split_text_respects_a_smaller_configured_budget():
    """Chunk size is configurable, and the configuration is honoured."""
    text = "Offer an aortic ultrasound to people with a suspected AAA. " * 40
    small = cc.split_text(text, target_tokens=64, overlap_tokens=8)
    assert all(cc.count_tokens(p) <= 64 + cc.SPECIAL_TOKEN_BUDGET for p in small)
    assert len(small) > len(cc.split_text(text, target_tokens=200, overlap_tokens=8))


def test_split_text_preserves_numeric_tokens():
    text = ("Refer people with an AAA that is 5.5 cm or larger within 2 weeks. " * 40)
    for piece in cc.split_text(text):
        assert "5.5 cm" in piece or "2 weeks" in piece
        assert "5.5cm" not in piece.replace("5.5 cm", "")


def test_adjacent_pieces_are_not_near_duplicates():
    """Overlap must not consume so much of a piece that the next one repeats it."""
    text = " ".join(
        f"Sentence {i} about abdominal aortic aneurysm surveillance and repair thresholds."
        for i in range(120)
    )
    pieces = cc.split_text(text)
    assert len(pieces) > 1

    def shingles(t, n=8):
        words = t.lower().split()
        return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}

    for a, b in zip(pieces, pieces[1:]):
        sa, sb = shingles(a), shingles(b)
        if sa and sb:
            jaccard = len(sa & sb) / len(sa | sb)
            assert jaccard < 0.75, f"adjacent pieces {jaccard:.2f} similar -- overlap too aggressive"


def test_no_near_duplicate_chunks_in_the_built_index():
    """Overlap must not produce near-duplicate RETRIEVABLE chunks.

    Scoped to `clinical` chunks, i.e. the ones that actually reach the index.
    Non-retrievable chunks are excluded because a source PDF may legitimately
    repeat identical lines (ESVS page 39 contains several identical "_ _ _ _"
    table rules); reproducing those faithfully is correct extraction, not a
    chunking defect, and they never reach retrieval.
    """
    chunks = [c for c in _load_chunks() if c["content_type"] == "clinical"]

    def shingles(t, n=8):
        words = t.lower().split()
        return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}

    by_id = {c["chunk_id"]: shingles(c["chunk_text"]) for c in chunks}
    ids = list(by_id)
    for a, b in zip(ids, ids[1:]):
        sa, sb = by_id[a], by_id[b]
        if sa and sb:
            assert len(sa & sb) / len(sa | sb) < 0.85, f"{a} and {b} are near-duplicates"


def test_split_text_handles_unbreakable_text_without_hanging():
    """No safe split point anywhere -> falls back to a token split, not a hang."""
    text = "aneurysm " * 2000
    pieces = cc.split_text(text)
    assert pieces
    assert all(cc.count_tokens(p) <= cc.model_token_limit() for p in pieces)


def test_built_chunks_are_all_within_the_limit():
    chunks = _load_chunks()
    limit = cc.model_token_limit()
    over = [(c["chunk_id"], c["token_count"]) for c in chunks if c["token_count"] > limit]
    assert over == [], f"chunks exceed the embedding window: {over[:5]}"


def test_token_statistics_reports_exceeding_limit():
    stats = cc.token_statistics(_load_chunks())
    assert stats["exceeding_limit"] == 0
    assert stats["max_tokens"] <= stats["token_limit"]
    assert stats["n_chunks"] > 0


def test_token_statistics_can_detect_an_oversized_chunk():
    """The validator must be able to FAIL, not just report zero."""
    bad = [{"chunk_id": "x", "chunk_text": "aneurysm " * 500, "token_count": 900}]
    stats = cc.token_statistics(bad)
    assert stats["exceeding_limit"] == 1


def test_validate_chunks_fails_on_oversized_chunk():
    bad = [
        {
            "chunk_id": "DOC__p1-1__c0001",
            "document_id": "DOC",
            "document_name": "Doc",
            "document_type": "official guideline",
            "section_title": "S",
            "section_source": "detected",
            "page_number": 1,
            "page_start": 1,
            "page_end": 1,
            "source_file": "d.pdf",
            "chunk_text": "aneurysm " * 400,
            "source_excerpt": "aneurysm",
            "token_count": 800,
        }
    ]
    report = cc.validate_chunks(bad)
    assert report["status"] == "FAIL"
    assert any("exceeds embedding token limit" in e for e in report["errors"])


# --------------------------------------------------------------------------
# Chunk IDs
# --------------------------------------------------------------------------

def test_chunk_ids_are_unique_and_well_formed():
    import re

    chunks = _load_chunks()
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    for cid in ids:
        assert re.fullmatch(r"[A-Za-z0-9_]+__p\d+-\d+__c\d{4}", cid), cid


def test_chunk_ids_encode_their_own_page_range():
    for c in _load_chunks():
        assert f"p{c['page_start']}-{c['page_end']}" in c["chunk_id"]


def test_chunk_ids_are_deterministic_across_rebuilds():
    pages = pd.read_parquet(ROOT / "data" / "processed" / "pages_df.parquet")
    recs = pd.DataFrame(
        json.loads((ROOT / "data" / "processed" / "recommendations.json").read_text(encoding="utf-8"))
    )
    first = [c["chunk_id"] for c in cc.build_chunks(pages, recs)]
    second = [c["chunk_id"] for c in cc.build_chunks(pages, recs)]
    assert first == second


def test_chunk_ids_are_scoped_per_document():
    """Removing one document must not renumber the others' chunk IDs."""
    pages = pd.read_parquet(ROOT / "data" / "processed" / "pages_df.parquet")
    keep = "USPSTF_2019"
    full = [c["chunk_id"] for c in cc.build_chunks(pages) if c["document_id"] == keep]
    subset = [c["chunk_id"] for c in cc.build_chunks(pages[pages.document_id == keep])]
    assert full == subset


# --------------------------------------------------------------------------
# Metadata preservation
# --------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "chunk_id",
    "document_id",
    "document_name",
    "page_number",
    "page_start",
    "page_end",
    "source_file",
    "chunk_text",
    "token_count",
    "content_type",
    "is_guideline",
]


def test_every_chunk_carries_required_metadata():
    for c in _load_chunks():
        for field in REQUIRED_FIELDS:
            assert c.get(field) is not None, f"{c['chunk_id']} missing {field}"


def test_missing_sections_are_none_not_nan_strings():
    """NaN and the string 'nan' are both truthy and defeat presence checks."""
    for c in _load_chunks():
        section = c.get("section_title")
        assert section is None or (isinstance(section, str) and section.strip().lower() != "nan")


def test_as_optional_normalises_nan_forms():
    assert cc._as_optional(float("nan")) is None
    assert cc._as_optional("nan") is None
    assert cc._as_optional("") is None
    assert cc._as_optional(None) is None
    assert cc._as_optional("  Screening ") == "Screening"


def test_metadata_survives_into_the_index():
    if not INDEX_CHUNKS_PATH.exists():
        pytest.skip("index not built")
    indexed = json.loads(INDEX_CHUNKS_PATH.read_text(encoding="utf-8"))
    source = {c["chunk_id"]: c for c in _load_chunks()}
    assert indexed
    for rec in indexed:
        origin = source[rec["chunk_id"]]
        assert rec["document_id"] == origin["document_id"]
        assert rec["page_number"] == origin["page_number"]
        assert rec["section_title"] == origin["section_title"]
        assert rec["token_count"] == origin["token_count"]


# --------------------------------------------------------------------------
# Low-value content handling
# --------------------------------------------------------------------------

def test_classifier_labels_reference_and_toc_and_titles():
    assert cc.classify_chunk_content("Smith J, et al. Lancet. 2013;8(12):e81260. doi:10.1371/a "
                                     "Jones B, et al. JAMA. 2019;322:2211. doi:10.1001/b "
                                     "Lee C, et al. Circulation. 2011;124:1118. doi:10.1161/c",
                                     "REFERENCES") == "reference"
    assert cc.classify_chunk_content("Diagnosis......... 6 Monitoring......... 11 Repair......... 16") == "toc"
    # A single dot-leader row is still a contents row once density is high.
    assert cc.classify_chunk_content(
        "Monitoring for complications after endovascular aneurysm repair "
        "................................................................ 48"
    ) == "toc"
    assert cc.classify_chunk_content(
        "Care of Patients with an Abdominal Aortic Aneurysm 2018 Practice Guidelines"
    ) == "title_only"


def test_classifier_labels_numbered_bibliography_entries():
    """ESVS 2024 numbers its references instead of using 'et al.'/DOI on every line."""
    text = (
        "1261 Robertson L. Optimising intervals for abdominal aortic aneurysm surveillance. "
        "Available at: https://www.nhs.uk/conditions/aaa-screening/ [Accessed 12 October 2023]. "
        "1262 Hoebink M. Activated clotting time guided heparinisation."
    )
    assert cc.classify_chunk_content(text) == "reference"


def test_classifier_labels_spaced_dot_contents_rows():
    """ESVS 2024 uses '. . . .' leaders rather than solid '....' runs."""
    assert cc.classify_chunk_content(
        "Endoleaks . . . . . . . . . . . . . . . . . . . . . . . . . . . . 250"
    ) == "toc"


def test_classifier_labels_page_rules_as_non_informative():
    """Underscore/dash rules carry no clinical information."""
    assert cc.classify_chunk_content("_ " * 120) == "boilerplate"
    assert cc.classify_chunk_content("_" * 150) == "boilerplate"


def test_classifier_keeps_normal_clinical_text():
    """The density guard must not touch ordinary prose."""
    text = (
        "We recommend surveillance imaging at 12 month intervals for patients with an "
        "abdominal aortic aneurysm of 4.0 to 4.9 cm in diameter."
    )
    assert cc.classify_chunk_content(text) == "clinical"


def test_classifier_never_demotes_a_graded_recommendation():
    """Short, table-like recommendations must stay retrievable."""
    text = "We recommend elective repair for a fusiform AAA greater or equal to 5.5 cm. 1 A"
    assert cc.classify_chunk_content(text, "REFERENCES") == "clinical"


def test_index_excludes_non_answering_content_but_chunks_keep_it():
    if not INDEX_CHUNKS_PATH.exists():
        pytest.skip("index not built")
    indexed = json.loads(INDEX_CHUNKS_PATH.read_text(encoding="utf-8"))
    assert all(r["content_type"] == "clinical" for r in indexed)
    assert all(r["is_guideline"] for r in indexed)
    # nothing was deleted: the low-value chunks are still on disk
    assert any(c["content_type"] != "clinical" for c in _load_chunks())


def test_table_redundancy_detection():
    clean = ("We recommend using ultrasound, when feasible, as the preferred imaging modality "
             "for aneurysm screening and surveillance.")
    table = ("Recommendation | Level | Quality | We recommend using ultrasound, when feasible, as "
             "the preferred imaging modality for aneurysm screening and surveillance. | 1 | A")
    assert cc.table_is_redundant(table, clean) is True
    assert cc.table_is_redundant("Rupture rate 0.4% per year by diameter band 3.0-3.9 cm", clean) is False


def test_embeddable_chunks_refuses_oversized_input():
    """A chunk over the ACTIVE model's window must be refused, not truncated."""
    limit = cc.model_token_limit()
    oversized = limit * 2
    bad = [
        {
            "chunk_id": "DOC__p1-1__c0001",
            "chunk_text": "aneurysm " * oversized,
            "token_count": oversized,
            "content_type": "clinical",
            "is_guideline": True,
            "document_type": "official guideline",
        }
    ]
    assert bad[0]["token_count"] > limit
    with pytest.raises(ValueError, match=f"over the {limit}-token limit"):
        cc.embeddable_chunks(bad, {"invalid_chunk_ids": []})

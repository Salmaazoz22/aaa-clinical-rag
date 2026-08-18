# -*- coding: utf-8 -*-
"""Tests for the Qdrant production vector store.

Two layers:

* contract tests (schema, deterministic IDs, payload completeness, duplicate and
  dimension detection, malformed input) -- pure, no server;
* integration tests that build a real collection through the qdrant-client and
  query it. These run against the client's embedded local mode by default, so
  they need no Docker; set QDRANT_TEST_URL to point them at a live server.

The embedding model is only loaded by the tests that genuinely need it, and
those skip when the shipped index is absent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (str(ROOT), str(ROOT / "notebooks")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

qdrant_client = pytest.importorskip("qdrant_client", reason="qdrant-client not installed")
from qdrant_client import models  # noqa: E402

from vectordb.config import QdrantSettings, load_settings  # noqa: E402
from vectordb.ingest import ensure_collection, load_local_index, upsert_points, verify_collection  # noqa: E402
from vectordb.retriever import QdrantRetriever  # noqa: E402
from vectordb.schema import (  # noqa: E402
    EXPECTED_DIM,
    EXPECTED_MODEL,
    EXPECTED_N_VECTORS,
    EXPECTED_REVISION,
    PAYLOAD_FIELDS,
    IngestValidationError,
    build_payload,
    nan_summary,
    point_id_for,
    validate_index_bundle,
)

INDEX_DIR = ROOT / "data" / "embeddings"
COLLECTION = "aaa_clinical_test"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def local_index():
    if not (INDEX_DIR / "embeddings.npy").exists():
        pytest.skip("shipped index not built")
    return load_local_index(INDEX_DIR)


@pytest.fixture()
def sample_record(local_index):
    return dict(local_index["records"][0])


@pytest.fixture(scope="module")
def settings(tmp_path_factory) -> QdrantSettings:
    """Isolated test store: embedded local mode unless QDRANT_TEST_URL is set."""
    import os

    url = os.environ.get("QDRANT_TEST_URL")
    if url:
        return QdrantSettings(
            url=url,
            api_key=os.environ.get("QDRANT_TEST_API_KEY") or None,
            collection=COLLECTION,
            prefer_grpc=False,
            timeout=30.0,
            exact_search=True,
            local_path=None,
        )
    path = tmp_path_factory.mktemp("qdrant_local")
    return QdrantSettings(
        url="",
        api_key=None,
        collection=COLLECTION,
        prefer_grpc=False,
        timeout=30.0,
        exact_search=True,
        local_path=str(path),
    )


@pytest.fixture(scope="module")
def populated(settings, local_index):
    """A real collection built by the production ingestion path."""
    from vectordb.config import make_client

    client = make_client(settings)
    ensure_collection(client, settings.collection, recreate=True)
    upsert_points(client, settings.collection, local_index["records"], local_index["vectors"])
    yield client
    if client.collection_exists(settings.collection):
        client.delete_collection(settings.collection)
    close = getattr(client, "close", None)
    if close:
        close()


# ---------------------------------------------------------------------------
# 1. Collection creation / configuration
# ---------------------------------------------------------------------------

def test_collection_is_created_with_cosine_and_768_dimensions(populated, settings):
    info = populated.get_collection(settings.collection)
    params = info.config.params.vectors
    assert int(params.size) == EXPECTED_DIM
    assert str(getattr(params.distance, "value", params.distance)).lower() == "cosine"


def test_ensure_collection_refuses_to_clobber_an_existing_collection(populated, settings):
    with pytest.raises(RuntimeError, match="already exists"):
        ensure_collection(populated, settings.collection, recreate=False)


def test_verify_collection_detects_a_wrong_point_count(populated, settings):
    with pytest.raises(RuntimeError, match="expected 5"):
        verify_collection(populated, settings.collection, expected_points=5)


# ---------------------------------------------------------------------------
# 2. Vector count and dimension
# ---------------------------------------------------------------------------

def test_all_991_vectors_are_present(populated, settings, local_index):
    assert len(local_index["records"]) == EXPECTED_N_VECTORS
    assert populated.count(collection_name=settings.collection, exact=True).count == EXPECTED_N_VECTORS


def test_local_index_is_768_dimensional_and_normalised(local_index):
    vectors = local_index["vectors"]
    assert vectors.shape == (EXPECTED_N_VECTORS, EXPECTED_DIM)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
    assert np.isfinite(vectors).all()


def test_local_index_meta_pins_the_validated_model(local_index):
    meta = local_index["meta"]
    assert meta["model_name"] == EXPECTED_MODEL
    assert meta["model_revision"] == EXPECTED_REVISION
    assert meta["embedding_dim"] == EXPECTED_DIM
    assert meta["metric"] == "cosine"


def test_validation_rejects_a_wrong_vector_dimension(local_index):
    truncated = local_index["vectors"][:, :512]
    with pytest.raises(IngestValidationError, match="vector dimension is 512"):
        validate_index_bundle(local_index["records"], truncated, local_index["meta"])


def test_validation_rejects_nan_and_inf_vectors(local_index):
    vectors = np.array(local_index["vectors"], dtype=np.float32)
    vectors[3, 0] = np.nan
    vectors[7, 1] = np.inf
    with pytest.raises(IngestValidationError, match="NaN or Inf"):
        validate_index_bundle(local_index["records"], vectors, local_index["meta"])


def test_validation_rejects_a_vector_count_mismatch(local_index):
    with pytest.raises(IngestValidationError, match="vector count"):
        validate_index_bundle(local_index["records"], local_index["vectors"][:-1], local_index["meta"])


def test_validation_rejects_unnormalised_vectors(local_index):
    vectors = np.array(local_index["vectors"], dtype=np.float32) * 2.0
    with pytest.raises(IngestValidationError, match="not L2-normalised"):
        validate_index_bundle(local_index["records"], vectors, local_index["meta"])


def test_validation_accepts_the_shipped_index(local_index):
    summary = validate_index_bundle(local_index["records"], local_index["vectors"], local_index["meta"])
    assert summary["n_records"] == EXPECTED_N_VECTORS
    assert summary["unique_chunk_ids"] == EXPECTED_N_VECTORS
    assert summary["unique_point_ids"] == EXPECTED_N_VECTORS
    assert summary["embedding_dim"] == EXPECTED_DIM


# ---------------------------------------------------------------------------
# 3. Deterministic point IDs
# ---------------------------------------------------------------------------

def test_point_ids_are_deterministic_and_stable():
    first = point_id_for("ESVS_2024__p1-1__c0001")
    assert first == point_id_for("ESVS_2024__p1-1__c0001")
    # Frozen value: changing the namespace would silently orphan every point.
    assert first == "e6c01b73-2cae-5f18-b3df-8f188e609edd"


def test_point_ids_differ_per_chunk_and_are_collision_free(local_index):
    ids = [point_id_for(r["chunk_id"]) for r in local_index["records"]]
    assert len(set(ids)) == len(ids) == EXPECTED_N_VECTORS


def test_point_id_rejects_an_empty_chunk_id():
    for bad in ("", "   ", None, 17):
        with pytest.raises(IngestValidationError):
            point_id_for(bad)


def test_stored_point_ids_match_the_deterministic_mapping(populated, settings, local_index):
    record = local_index["records"][10]
    pid = point_id_for(record["chunk_id"])
    fetched = populated.retrieve(collection_name=settings.collection, ids=[pid], with_payload=True)
    assert fetched, "deterministic point ID does not resolve in the collection"
    assert fetched[0].payload["chunk_id"] == record["chunk_id"]


# ---------------------------------------------------------------------------
# 4. Payload completeness and duplicate detection
# ---------------------------------------------------------------------------

def test_payload_carries_every_contract_field(populated, settings):
    points, _ = populated.scroll(collection_name=settings.collection, limit=50, with_payload=True)
    assert points
    for point in points:
        assert set(point.payload) == set(PAYLOAD_FIELDS)
        assert point.payload["chunk_text"]
        assert point.payload["chunk_id"]


def test_payload_matches_the_local_record_it_came_from(populated, settings, local_index):
    by_id = {r["chunk_id"]: r for r in local_index["records"]}
    points, _ = populated.scroll(collection_name=settings.collection, limit=100, with_payload=True)
    for point in points:
        origin = by_id[point.payload["chunk_id"]]
        for field in ("document_id", "document_name", "page_start", "page_end", "token_count", "chunk_text"):
            assert point.payload[field] == origin[field]


def test_build_payload_rejects_a_missing_field(sample_record):
    del sample_record["page_start"]
    with pytest.raises(IngestValidationError, match="missing payload field"):
        build_payload(sample_record)


def test_build_payload_rejects_a_null_required_field(sample_record):
    sample_record["document_id"] = None
    with pytest.raises(IngestValidationError, match="null/NaN required field"):
        build_payload(sample_record)


def test_build_payload_rejects_nan_in_a_required_field(sample_record):
    sample_record["token_count"] = float("nan")
    with pytest.raises(IngestValidationError, match="null/NaN required field"):
        build_payload(sample_record)


def test_build_payload_maps_nan_optionals_to_null(sample_record):
    """The frozen artifact stores some optional fields as NaN; null means the same."""
    sample_record["recommendation_id"] = float("nan")
    assert build_payload(sample_record)["recommendation_id"] is None


def test_nan_summary_counts_the_known_artifact_quirk(local_index):
    counts = nan_summary(local_index["records"])
    assert set(counts) <= {"recommendation_id", "recommendation_grade", "evidence_level"}
    assert sum(counts.values()) > 0


def test_validation_detects_duplicate_chunk_ids(local_index):
    records = [dict(r) for r in local_index["records"]]
    records[5]["chunk_id"] = records[4]["chunk_id"]
    with pytest.raises(IngestValidationError, match="duplicate chunk_id"):
        validate_index_bundle(records, local_index["vectors"], local_index["meta"])


def test_validation_detects_a_missing_chunk_id(local_index):
    records = [dict(r) for r in local_index["records"]]
    records[2].pop("chunk_id")
    with pytest.raises(IngestValidationError, match="chunk_id"):
        validate_index_bundle(records, local_index["vectors"], local_index["meta"])


def test_validation_detects_a_token_limit_violation(local_index):
    records = [dict(r) for r in local_index["records"]]
    records[1]["token_count"] = 4096
    with pytest.raises(IngestValidationError, match="exceed the 512-token window"):
        validate_index_bundle(records, local_index["vectors"], local_index["meta"])


def test_validation_detects_an_unpinned_or_swapped_model(local_index):
    meta = dict(local_index["meta"], model_revision="deadbeef")
    with pytest.raises(IngestValidationError, match="model_revision"):
        validate_index_bundle(local_index["records"], local_index["vectors"], meta)


# ---------------------------------------------------------------------------
# 5. Retrieval behaviour
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def retriever(populated, settings):
    return QdrantRetriever(settings=settings, client=populated)


def test_retrieval_returns_top_k_hits_in_descending_score_order(retriever, local_index):
    query_vector = [float(x) for x in local_index["vectors"][0]]
    hits = retriever.search_vector(query_vector, top_k=10)
    assert len(hits) == 10
    scores = [h["similarity_score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert [h["rank"] for h in hits] == list(range(1, 11))


def test_a_chunks_own_vector_retrieves_that_chunk_first(retriever, local_index):
    record = local_index["records"][42]
    hits = retriever.search_vector([float(x) for x in local_index["vectors"][42]], top_k=1)
    assert hits[0]["chunk_id"] == record["chunk_id"]
    assert hits[0]["similarity_score"] == pytest.approx(1.0, abs=1e-4)


def test_hits_expose_the_evidence_fields_the_project_uses(retriever, local_index):
    hits = retriever.search_vector([float(x) for x in local_index["vectors"][7]], top_k=3)
    for hit in hits:
        for field in (
            "chunk_id", "text", "score", "similarity_score", "chunk_text", "document",
            "document_id", "page_start", "page_end", "section", "recommendation_id", "rank",
        ):
            assert field in hit
        assert hit["text"] == hit["chunk_text"]
        assert hit["score"] == hit["similarity_score"]


def test_top_k_is_honoured(retriever, local_index):
    vector = [float(x) for x in local_index["vectors"][3]]
    for k in (1, 5, 10, 25):
        assert len(retriever.search_vector(vector, top_k=k)) == k


def test_top_k_cannot_exceed_the_collection(retriever, local_index):
    hits = retriever.search_vector([float(x) for x in local_index["vectors"][0]], top_k=5000)
    assert len(hits) == EXPECTED_N_VECTORS


# ---------------------------------------------------------------------------
# 6. Local vs Qdrant equivalence
# ---------------------------------------------------------------------------

def test_qdrant_reproduces_the_local_ranking_for_stored_vectors(retriever, local_index):
    """Same query vector, both back ends: same IDs, same order, same scores.

    Uses stored vectors as queries so the check needs no encoder, and covers a
    spread of documents rather than one.
    """
    from vectordb.verify_migration import local_top_k

    for idx in (0, 100, 250, 500, 750, 990):
        vector = np.asarray(local_index["vectors"][idx], dtype=np.float32)
        expected = local_top_k(vector, {"vectors": local_index["vectors"], "chunks": local_index["records"]}, 10)
        actual = retriever.search_vector([float(x) for x in vector], top_k=10)
        assert [h["chunk_id"] for h in actual] == [h["chunk_id"] for h in expected]
        for a, e in zip(actual, expected):
            assert a["similarity_score"] == pytest.approx(e["similarity_score"], abs=1e-5)


def test_recorded_migration_verification_artifact_is_equivalent():
    """If the migration report exists, it must say EQUIVALENT."""
    path = ROOT / "eval" / "qdrant_migration_verification.json"
    if not path.exists():
        pytest.skip("migration verification not run yet")
    report = json.loads(path.read_text(encoding="utf-8"))
    summary = report["summary"]
    assert summary["verdict"] == "EQUIVALENT"
    assert summary["n_fail"] == 0
    assert summary["all_same_top1"] and summary["all_same_top10"] and summary["all_same_order"]
    assert summary["max_abs_score_difference"] <= report["configuration"]["score_tolerance_abs"]
    assert report["configuration"]["embedding_model"] == EXPECTED_MODEL
    assert report["configuration"]["embedding_revision"] == EXPECTED_REVISION
    assert report["configuration"]["top_k"] == 10


# ---------------------------------------------------------------------------
# 7. Malformed input
# ---------------------------------------------------------------------------

def test_empty_query_is_refused(retriever):
    for bad in ("", "   ", "\n\t "):
        with pytest.raises(ValueError, match="empty"):
            retriever.search(bad)


def test_non_string_query_is_refused(retriever):
    for bad in (None, 42, ["a question"], {"q": "x"}):
        with pytest.raises(TypeError):
            retriever.search(bad)


def test_absurdly_long_query_is_refused_rather_than_silently_truncated(retriever):
    with pytest.raises(ValueError, match="character limit"):
        retriever.search("abdominal aortic aneurysm " * 2000)


def test_invalid_top_k_is_refused(retriever, local_index):
    vector = [float(x) for x in local_index["vectors"][0]]
    for bad in (0, -1, 2.5, "10", None, True):
        with pytest.raises(ValueError):
            retriever.search_vector(vector, top_k=bad)


def test_wrong_dimension_query_vector_is_refused(retriever):
    with pytest.raises(ValueError, match="dimension"):
        retriever.search_vector([0.1] * 384, top_k=10)


def test_settings_never_expose_the_api_key():
    described = load_settings().describe()
    assert "api_key" not in described
    assert set(described) >= {"mode", "collection", "exact_search", "api_key_supplied"}
    assert isinstance(described["api_key_supplied"], bool)


# ---------------------------------------------------------------------------
# 8. A long, in-domain query still works end to end (needs the encoder)
# ---------------------------------------------------------------------------

def test_a_long_in_domain_question_returns_ten_hits(retriever):
    pytest.importorskip("sentence_transformers")
    question = (
        "In a 72 year old man with a 5.4 cm infrarenal abdominal aortic aneurysm and "
        "significant cardiac comorbidity, what do the guidelines recommend regarding the "
        "threshold for elective repair, the choice between open surgical repair and "
        "endovascular aneurysm repair, and the surveillance interval if repair is deferred? "
    ) * 3
    hits = retriever.search(question, top_k=10)
    assert len(hits) == 10
    assert all(h["chunk_id"] for h in hits)
    assert all(0.0 <= h["similarity_score"] <= 1.0001 for h in hits)

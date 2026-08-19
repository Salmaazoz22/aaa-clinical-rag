# -*- coding: utf-8 -*-
"""Tests for the FastAPI service layer (api/main.py).


Uses FastAPI's TestClient (no real Qdrant or LLM needed).
The retriever and answer_question are patched so these tests are:
  - fast (no network, no embedding model)
  - deterministic
  - isolated from the core pipeline logic (which has its own tests)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api.main as api_module  # noqa: E402  (imported after path fix)
from api.main import app  # noqa: E402
from vectordb.schema import EXPECTED_DIM, EXPECTED_MODEL, EXPECTED_REVISION  # noqa: E402

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helper: a mock retriever that passes the _require_retriever() guard
# ---------------------------------------------------------------------------

def _mock_retriever(**col_kwargs):
    """Return a MagicMock that looks like a healthy QdrantRetriever."""
    r = MagicMock()
    r.settings.collection = col_kwargs.get("collection", "test_col")
    # describe() is what /v1/meta serialises into "connection"; a bare MagicMock
    # would not round-trip through JSON, and the point of the test that reads it
    # is that no key material appears there.
    r.settings.describe.return_value = {
        "mode": "server",
        "url": "http://localhost:6333",
        "local_path": None,
        "collection": col_kwargs.get("collection", "test_col"),
        "prefer_grpc": False,
        "timeout_s": 30.0,
        "exact_search": True,
        "api_key_supplied": col_kwargs.get("api_key_supplied", False),
    }
    col_info = MagicMock()
    col_info.status.name = col_kwargs.get("status", "GREEN")
    col_info.points_count = col_kwargs.get("points_count", 991)
    r.client.get_collection.return_value = col_info
    return r


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok_when_qdrant_green(self):
        with patch.object(api_module, "_retriever", _mock_retriever()):
            r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["qdrant"] is True
        assert body["index"] is True

    def test_health_degraded_when_qdrant_raises(self):
        mock_r = _mock_retriever()
        mock_r.client.get_collection.side_effect = Exception("connection refused")
        with patch.object(api_module, "_retriever", mock_r):
            r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "degraded"
        assert body["qdrant"] is False
        assert body["index"] is False

    def test_health_degraded_when_retriever_none(self):
        with patch.object(api_module, "_retriever", None):
            r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "degraded"

    def test_health_degraded_when_index_not_green(self):
        mock_r = _mock_retriever(status="YELLOW")
        with patch.object(api_module, "_retriever", mock_r):
            r = client.get("/health")
        body = r.json()
        assert body["qdrant"] is True
        assert body["index"] is False
        assert body["status"] == "degraded"


# ---------------------------------------------------------------------------
# GET /v1/meta
# ---------------------------------------------------------------------------

class TestMeta:
    def test_meta_returns_expected_fields(self):
        with patch.object(api_module, "_retriever", _mock_retriever(points_count=100)):
            r = client.get("/v1/meta")
        assert r.status_code == 200
        data = r.json()
        assert data["model"] == EXPECTED_MODEL
        assert data["revision"] == EXPECTED_REVISION
        assert data["dimensions"] == EXPECTED_DIM
        assert data["collection"] == "test_col"
        assert data["chunk_count"] == 100
        assert data["index_status"] == "GREEN"

    def test_meta_503_when_retriever_none(self):
        with patch.object(api_module, "_retriever", None):
            r = client.get("/v1/meta")
        assert r.status_code == 503

    def test_meta_503_when_qdrant_unavailable(self):
        mock_r = _mock_retriever()
        mock_r.client.get_collection.side_effect = Exception("timeout")
        with patch.object(api_module, "_retriever", mock_r):
            r = client.get("/v1/meta")
        assert r.status_code == 503
        assert "Qdrant unavailable" in r.json()["detail"]


# ---------------------------------------------------------------------------
# POST /v1/answer
# ---------------------------------------------------------------------------

class TestAnswer:
    def _good_result(self):
        result = MagicMock()
        result.to_dict.return_value = {
            "query": "What is the threshold?",
            "refused": False,
            "answer": {"recommendation": "55 mm", "confidence": "High"},
        }
        return result

    def test_answer_returns_pipeline_result(self):
        with (
            patch.object(api_module, "_retriever", _mock_retriever()),
            patch.object(api_module, "answer_question", return_value=self._good_result()) as mock_aq,
        ):
            r = client.post("/v1/answer", json={"question": "What is the threshold?"})
        assert r.status_code == 200
        assert r.json()["answer"]["confidence"] == "High"
        mock_aq.assert_called_once()

    def test_answer_passes_top_k_and_threshold(self):
        with (
            patch.object(api_module, "_retriever", _mock_retriever()),
            patch.object(api_module, "answer_question", return_value=self._good_result()) as mock_aq,
        ):
            client.post("/v1/answer", json={"question": "Q", "top_k": 5, "threshold": 0.8})
        call_kwargs = mock_aq.call_args.kwargs
        assert call_kwargs["top_k"] == 5
        assert call_kwargs["threshold"] == 0.8

    def test_answer_empty_question_is_422(self):
        with patch.object(api_module, "_retriever", _mock_retriever()):
            r = client.post("/v1/answer", json={"question": ""})
        assert r.status_code == 422

    def test_answer_missing_question_field_is_422(self):
        with patch.object(api_module, "_retriever", _mock_retriever()):
            r = client.post("/v1/answer", json={})
        assert r.status_code == 422

    def test_answer_negative_top_k_is_422(self):
        with patch.object(api_module, "_retriever", _mock_retriever()):
            r = client.post("/v1/answer", json={"question": "Q", "top_k": 0})
        assert r.status_code == 422

    def test_answer_threshold_out_of_range_is_422(self):
        with patch.object(api_module, "_retriever", _mock_retriever()):
            r = client.post("/v1/answer", json={"question": "Q", "threshold": 1.5})
        assert r.status_code == 422

    def test_answer_503_when_retriever_none(self):
        with patch.object(api_module, "_retriever", None):
            r = client.post("/v1/answer", json={"question": "Q"})
        assert r.status_code == 503

    def test_answer_503_on_missing_api_key(self):
        from generation.providers import MissingAPIKeyError
        with (
            patch.object(api_module, "_retriever", _mock_retriever()),
            patch.object(api_module, "answer_question", side_effect=MissingAPIKeyError("groq")),
        ):
            r = client.post("/v1/answer", json={"question": "Q"})
        assert r.status_code == 503
        # The provider's own message is logged, not returned: it can name the
        # vendor, the account and the upstream body.
        detail = r.json()["detail"]
        assert "not configured" in detail
        assert "groq" not in detail

    def test_answer_500_on_unexpected_error(self):
        with (
            patch.object(api_module, "_retriever", _mock_retriever()),
            patch.object(api_module, "answer_question", side_effect=RuntimeError("boom")),
        ):
            r = client.post("/v1/answer", json={"question": "Q"})
        assert r.status_code == 500


# ---------------------------------------------------------------------------
# GET /v1/chunks/{chunk_id}
# ---------------------------------------------------------------------------

class TestChunks:
    def test_chunk_returns_payload(self):
        mock_r = _mock_retriever()
        mock_r.get_by_chunk_ids.return_value = {
            "ESVS_2024__p26-26__c0093": {"chunk_id": "ESVS_2024__p26-26__c0093", "chunk_text": "..."}
        }
        with patch.object(api_module, "_retriever", mock_r):
            r = client.get("/v1/chunks/ESVS_2024__p26-26__c0093")
        assert r.status_code == 200
        assert r.json()["chunk_id"] == "ESVS_2024__p26-26__c0093"

    def test_chunk_404_when_not_found(self):
        mock_r = _mock_retriever()
        mock_r.get_by_chunk_ids.return_value = {}
        with patch.object(api_module, "_retriever", mock_r):
            r = client.get("/v1/chunks/DOESNT_EXIST")
        assert r.status_code == 404

    def test_chunk_503_when_retriever_none(self):
        with patch.object(api_module, "_retriever", None):
            r = client.get("/v1/chunks/any_id")
        assert r.status_code == 503

    def test_chunk_503_when_qdrant_raises(self):
        mock_r = _mock_retriever()
        mock_r.get_by_chunk_ids.side_effect = Exception("qdrant down")
        with patch.object(api_module, "_retriever", mock_r):
            r = client.get("/v1/chunks/any_id")
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# OpenAPI / docs
# ---------------------------------------------------------------------------

class TestOpenAPI:
    def test_openapi_schema_is_served(self):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert "paths" in schema
        # all four endpoints are present
        assert "/health" in schema["paths"]
        assert "/v1/meta" in schema["paths"]
        assert "/v1/answer" in schema["paths"]
        assert "/v1/chunks/{chunk_id}" in schema["paths"]

    def test_swagger_ui_is_served(self):
        r = client.get("/docs")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_redoc_is_served(self):
        r = client.get("/redoc")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Error mapping: an upstream model failure must never look like a bad request,
# and must never become an answer.
# ---------------------------------------------------------------------------

class TestUpstreamFailureMapping:
    """AnswerParseError subclasses ValueError, so ordering in the except chain
    is load-bearing: caught by the ValueError branch it would be reported as a
    client error (400) when it is in fact the model returning unparseable text.
    """

    def test_unparseable_model_response_is_502_not_400(self):
        from generation.parsing import AnswerParseError

        with (
            patch.object(api_module, "_retriever", _mock_retriever()),
            patch.object(
                api_module,
                "answer_question",
                side_effect=AnswerParseError("unterminated JSON object in model response"),
            ),
        ):
            r = client.post("/v1/answer", json={"question": "Q"})
        assert r.status_code == 502
        assert "language model" in r.json()["detail"].lower()

    def test_provider_failure_is_502_not_500(self):
        from generation.providers import ProviderError

        with (
            patch.object(api_module, "_retriever", _mock_retriever()),
            patch.object(api_module, "answer_question", side_effect=ProviderError("429 rate limit")),
        ):
            r = client.post("/v1/answer", json={"question": "Q"})
        assert r.status_code == 502
        # The upstream body is operator diagnostics, never caller-facing: a Groq
        # 429 carries the account organisation id and billing links.
        from generation.providers import SAFE_PROVIDER_MESSAGE

        assert r.json()["detail"] == SAFE_PROVIDER_MESSAGE
        assert "429" not in r.text

    def test_missing_api_key_still_wins_over_provider_error(self):
        """MissingAPIKeyError is a ProviderError subclass; it must keep its 503."""
        from generation.providers import MissingAPIKeyError

        with (
            patch.object(api_module, "_retriever", _mock_retriever()),
            patch.object(api_module, "answer_question", side_effect=MissingAPIKeyError("GROQ_API_KEY")),
        ):
            r = client.post("/v1/answer", json={"question": "Q"})
        assert r.status_code == 503

    def test_query_validation_error_is_still_400(self):
        with (
            patch.object(api_module, "_retriever", _mock_retriever()),
            patch.object(
                api_module, "answer_question", side_effect=ValueError("query must be a non-empty string")
            ),
        ):
            r = client.post("/v1/answer", json={"question": "   "})
        assert r.status_code == 400

    def test_no_answer_body_is_returned_on_upstream_failure(self):
        """The failure response must carry no answer-shaped payload at all."""
        from generation.providers import ProviderError

        with (
            patch.object(api_module, "_retriever", _mock_retriever()),
            patch.object(api_module, "answer_question", side_effect=ProviderError("boom")),
        ):
            r = client.post("/v1/answer", json={"question": "Q"})
        body = r.json()
        assert set(body) == {"detail"}
        assert "answer" not in body
        assert "recommendation" not in r.text


# ---------------------------------------------------------------------------
# /health now reports LLM configuration so a client can warn before asking
# ---------------------------------------------------------------------------

class TestHealthLLMStatus:
    def _settings(self, api_key):
        s = MagicMock()
        s.api_key = api_key
        s.provider = "groq"
        s.model = "openai/gpt-oss-120b"
        return s

    def test_health_reports_llm_configured(self):
        with (
            patch.object(api_module, "_retriever", _mock_retriever()),
            patch.object(api_module, "load_generation_settings", return_value=self._settings("sk-xxx")),
        ):
            body = client.get("/health").json()
        assert body["llm"]["configured"] is True
        assert body["llm"]["provider"] == "groq"

    def test_health_reports_llm_not_configured(self):
        with (
            patch.object(api_module, "_retriever", _mock_retriever()),
            patch.object(api_module, "load_generation_settings", return_value=self._settings(None)),
        ):
            body = client.get("/health").json()
        assert body["llm"]["configured"] is False

    def test_health_never_leaks_the_key(self):
        with (
            patch.object(api_module, "_retriever", _mock_retriever()),
            patch.object(
                api_module, "load_generation_settings", return_value=self._settings("sk-supersecret-123")
            ),
        ):
            r = client.get("/health")
        assert "supersecret" not in r.text
        assert "sk-" not in r.text

    def test_health_survives_a_bad_generation_config(self):
        with (
            patch.object(api_module, "_retriever", _mock_retriever()),
            patch.object(
                api_module, "load_generation_settings", side_effect=RuntimeError("unknown provider")
            ),
        ):
            r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["llm"]["configured"] is False

    def test_health_reports_points_and_retriever_presence(self):
        with (
            patch.object(api_module, "_retriever", _mock_retriever(points_count=991)),
            patch.object(api_module, "load_generation_settings", return_value=self._settings(None)),
        ):
            body = client.get("/health").json()
        assert body["points"] == 991
        assert body["retriever"] is True


# ---------------------------------------------------------------------------
# /v1/meta provenance
# ---------------------------------------------------------------------------

class TestMetaProvenance:
    def test_meta_exposes_index_provenance_digests(self):
        with patch.object(api_module, "_retriever", _mock_retriever()):
            data = client.get("/v1/meta").json()
        prov = data["index_provenance"]
        # Read from the shipped data/embeddings/index_meta.json, not hardcoded.
        assert prov["n_vectors"] == 991
        assert len(prov["source_chunks_sha256"]) == 64
        assert len(prov["indexed_chunk_ids_sha256"]) == 64
        assert data["distance"] == "Cosine"
        assert data["vector_store"] == "Qdrant"

    def test_meta_never_leaks_an_api_key(self):
        with patch.object(api_module, "_retriever", _mock_retriever()):
            r = client.get("/v1/meta")
        body = r.json()
        assert "api_key" not in body.get("connection", {})
        assert body["connection"]["api_key_supplied"] in (True, False)
        assert body["generation"] is None or "api_key" not in body["generation"]


# ---------------------------------------------------------------------------
# GET /v1/evaluation - frozen metrics, served verbatim
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_evaluation_serves_the_frozen_metrics(self):
        data = client.get("/v1/evaluation").json()
        assert data["frozen"] is True
        assert data["shipped_config"] == "V1_atomic_pagesafe"
        for dataset in ("original10", "heldout18", "final20"):
            assert dataset in data["metrics"]

    def test_evaluation_matches_the_published_final20_numbers(self):
        """The endpoint must not transform the frozen numbers in any way."""
        data = client.get("/v1/evaluation").json()
        shipped = data["metrics"]["final20"]["V1_atomic_pagesafe"]["metrics"]
        assert shipped["P@1"] == 0.55
        assert shipped["MRR"] == 0.6642
        assert shipped["Recall@10"] == 0.7833
        assert shipped["n_queries"] == 20

    def test_evaluation_reports_the_digest_of_the_bytes_it_read(self):
        import hashlib

        data = client.get("/v1/evaluation").json()
        on_disk = hashlib.sha256(
            (ROOT / "eval" / "final_evaluation_results.json").read_bytes()
        ).hexdigest()
        assert data["provenance"]["final_evaluation_results.json"] == on_disk

    def test_evaluation_carries_the_gold_standard_digests(self):
        data = client.get("/v1/evaluation").json()
        assert set(data["gold_sha256"]) == {"original10", "heldout18", "final20"}

    def test_evaluation_carries_limitations(self):
        data = client.get("/v1/evaluation").json()
        assert isinstance(data["limitations"], list) and data["limitations"]

    def test_evaluation_503_when_artifact_missing(self):
        with patch.dict(
            api_module._FROZEN_ARTIFACTS,
            {"evaluation_results": ROOT / "eval" / "does_not_exist.json"},
        ):
            api_module._read_frozen.cache_clear()
            r = client.get("/v1/evaluation")
        api_module._read_frozen.cache_clear()
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# GET /v1/corpus - the guideline documents behind the index
# ---------------------------------------------------------------------------

class TestCorpus:
    def _retriever_with_docs(self, doc_ids):
        r = _mock_retriever()
        points = [MagicMock(payload={"document_id": d}) for d in doc_ids]
        r.client.scroll.return_value = (points, None)
        return r

    def test_corpus_lists_the_four_guidelines(self):
        with patch.object(api_module, "_retriever", self._retriever_with_docs([])):
            data = client.get("/v1/corpus").json()
        assert data["n_documents"] == 4
        ids = {d["document_id"] for d in data["documents"]}
        assert ids == {"USPSTF_2019", "NICE_NG156", "ESVS_2024", "SVS_2018"}

    def test_corpus_counts_chunks_from_qdrant(self):
        docs = ["ESVS_2024"] * 3 + ["SVS_2018"] * 2
        with patch.object(api_module, "_retriever", self._retriever_with_docs(docs)):
            data = client.get("/v1/corpus").json()
        by_id = {d["document_id"]: d for d in data["documents"]}
        assert by_id["ESVS_2024"]["indexed_chunks"] == 3
        assert by_id["SVS_2018"]["indexed_chunks"] == 2
        assert data["total_indexed_chunks"] == 5
        assert data["chunk_counts_source"] == "qdrant"

    def test_corpus_degrades_gracefully_when_qdrant_scroll_fails(self):
        r = _mock_retriever()
        r.client.scroll.side_effect = Exception("qdrant down")
        with patch.object(api_module, "_retriever", r):
            resp = client.get("/v1/corpus")
        assert resp.status_code == 200
        assert resp.json()["chunk_counts_source"] == "unavailable"

    def test_corpus_does_not_expose_local_filesystem_paths(self):
        with patch.object(api_module, "_retriever", self._retriever_with_docs([])):
            data = client.get("/v1/corpus").json()
        for doc in data["documents"]:
            assert "source_path" not in doc

    def test_corpus_metadata_is_not_invented(self):
        """Every field served must come from document_metadata.json as-is."""
        import json as _json

        on_disk = _json.loads(
            (ROOT / "data" / "processed" / "document_metadata.json").read_text(encoding="utf-8")
        )
        by_id = {d["document_id"]: d for d in on_disk}
        with patch.object(api_module, "_retriever", self._retriever_with_docs([])):
            data = client.get("/v1/corpus").json()
        for doc in data["documents"]:
            src = by_id[doc["document_id"]]
            assert doc["source_organization"] == src["source_organization"]
            assert doc["publication_year"] == src["publication_year"]
            assert doc["source_url"] == src["source_url"]


class TestOpenAPIAdditions:
    def test_new_endpoints_are_documented(self):
        paths = client.get("/openapi.json").json()["paths"]
        assert "/v1/corpus" in paths
        assert "/v1/evaluation" in paths

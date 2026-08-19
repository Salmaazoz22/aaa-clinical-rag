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
        assert "API key" in r.json()["detail"]

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

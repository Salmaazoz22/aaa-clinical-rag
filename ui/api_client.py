# -*- coding: utf-8 -*-
"""HTTP client for the Clinical RAG API.

This is the *only* place the UI talks to the backend, and it is the only thing
the UI knows how to do. Nothing in `ui/` imports `generation`, `retrieval`,
`vectordb` or `ingestion`: the frontend has no retrieval implementation of its
own, no embedding model, no vector store handle and no copy of the pipeline.
Every number it shows was produced by the FastAPI service.

Two rules the rest of the UI depends on:

* **Static metadata is cached, answers never are.** `/v1/meta`, `/v1/corpus` and
  `/v1/evaluation` describe build-time artifacts that cannot change while the
  service is up, so they are cached. `/v1/answer` is never cached — a cached
  clinical answer could be served against a different index, a different
  threshold, or simply be stale, and there is no upside to justify that.
* **A failure is a failure.** Every call either returns the service's own
  payload or raises `ApiError` with a message written for a human. Nothing here
  invents an answer, a citation, a score or a status when the backend cannot
  supply one.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests
import streamlit as st

DEFAULT_BASE_URL = "http://127.0.0.1:8000"

#: Generous, because a cold /v1/answer embeds the query and then waits on a
#: reasoning model. The backend's own generation timeout is 180 s.
ANSWER_TIMEOUT = 240
META_TIMEOUT = 30
HEALTH_TIMEOUT = 8


#: The setting name, used identically as an environment variable and as a
#: Streamlit secret. One name, so the deployment note has one thing to say.
API_URL_SETTING = "CLINICAL_RAG_API_URL"


def _secret(name: str) -> str:
    """Read one Streamlit secret, tolerating the common case of having none.

    `st.secrets` raises when no secrets file exists, which is the normal state
    for a local `streamlit run`. That is not an error here: it just means the
    setting is not being supplied that way.
    """
    try:
        value = st.secrets.get(name)
    except Exception:  # noqa: BLE001 - no secrets file, or an unreadable one
        return ""
    return str(value).strip() if value else ""


def base_url() -> str:
    """Where the API lives. Overridable without touching code.

    Checked in order: a Streamlit secret, then an environment variable, then the
    localhost default. The secret comes first because that is the only channel a
    Streamlit Community Cloud deployment has — and the localhost default is
    exactly wrong there, since nothing is listening on the container's own
    127.0.0.1. Set `CLINICAL_RAG_API_URL` to the deployed API's public URL.
    """
    configured = _secret(API_URL_SETTING) or os.environ.get(API_URL_SETTING) or DEFAULT_BASE_URL
    return configured.rstrip("/")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

@dataclass
class ApiError(Exception):
    """A backend call that did not succeed, described for a human.

    `kind` lets the UI choose the right explanation — "the API is not running"
    and "the API is running but has no LLM key" need very different advice.
    """

    message: str
    kind: str = "error"          # offline | timeout | http | error
    status_code: int | None = None
    detail: str | None = None

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.message


def _request(method: str, path: str, *, timeout: int, **kwargs: Any) -> Any:
    url = f"{base_url()}{path}"
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(
            "Backend unavailable — nothing is listening on the API address.",
            kind="offline",
            detail=str(exc),
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError(
            f"The API did not respond within {timeout}s.",
            kind="timeout",
            detail=str(exc),
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ApiError("The request to the API could not be completed.", detail=str(exc)) from exc

    if response.status_code >= 400:
        detail: str | None
        try:
            payload = response.json()
            detail = payload.get("detail") if isinstance(payload, dict) else None
            if isinstance(detail, list):  # FastAPI validation errors
                detail = "; ".join(
                    f"{'.'.join(str(p) for p in item.get('loc', [])[1:])}: {item.get('msg')}"
                    for item in detail
                )
        except ValueError:
            detail = response.text[:400] or None
        raise ApiError(
            _http_message(response.status_code),
            kind="http",
            status_code=response.status_code,
            detail=detail,
        )

    try:
        return response.json()
    except ValueError as exc:
        raise ApiError("The API returned a response that was not JSON.", detail=response.text[:400]) from exc


def _http_message(code: int) -> str:
    return {
        400: "The question was rejected as invalid.",
        404: "Not found.",
        422: "The request did not match what the API expects.",
        500: "The API hit an internal error.",
        502: "The language model could not produce an answer.",
        503: "A service the API depends on is unavailable.",
    }.get(code, f"The API returned HTTP {code}.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def health() -> dict[str, Any]:
    """Live service status. Deliberately uncached — it is the freshness probe."""
    return _request("GET", "/health", timeout=HEALTH_TIMEOUT)


def health_or_none() -> dict[str, Any] | None:
    """`health()` that swallows the failure, for the always-on status strip."""
    try:
        return health()
    except ApiError:
        return None


# Every cached call takes the backend address as its first argument. It is not
# used inside the function — `_request` resolves the URL itself — it is there to
# be part of the cache key. Without it, repointing CLINICAL_RAG_API_URL would
# keep serving metadata fetched from the previous backend, which is exactly the
# kind of stale provenance this project must not display.

@st.cache_data(ttl=300, show_spinner=False)
def _meta(backend: str) -> dict[str, Any]:
    return _request("GET", "/v1/meta", timeout=META_TIMEOUT)


@st.cache_data(ttl=600, show_spinner=False)
def _corpus(backend: str) -> dict[str, Any]:
    return _request("GET", "/v1/corpus", timeout=META_TIMEOUT)


@st.cache_data(ttl=3600, show_spinner=False)
def _evaluation(backend: str) -> dict[str, Any]:
    return _request("GET", "/v1/evaluation", timeout=META_TIMEOUT)


@st.cache_data(ttl=600, show_spinner=False)
def _chunk(backend: str, chunk_id: str) -> dict[str, Any]:
    return _request("GET", f"/v1/chunks/{chunk_id}", timeout=META_TIMEOUT)


def meta() -> dict[str, Any]:
    """Index/model metadata. Build-time facts, safe to cache."""
    return _meta(base_url())


def corpus() -> dict[str, Any]:
    """The guideline documents behind the index."""
    return _corpus(base_url())


def evaluation() -> dict[str, Any]:
    """The frozen evaluation. These files never change; cache them hard."""
    return _evaluation(base_url())


def chunk(chunk_id: str) -> dict[str, Any]:
    """One indexed chunk by id.

    Cached because a chunk's stored text is immutable for the life of the
    collection — this is the same audit lookup the API documents, used here to
    show full evidence text rather than the pipeline's 240-character preview.
    """
    return _chunk(base_url(), chunk_id)


def answer(question: str, *, top_k: int | None = None, threshold: float | None = None) -> dict[str, Any]:
    """Ask one clinical question.

    NEVER cached. Returns the pipeline's full audit record exactly as the API
    serialised it, including refusals — a refusal is a successful response, not
    an error, and is returned here as such.
    """
    body: dict[str, Any] = {"question": question}
    if top_k is not None:
        body["top_k"] = top_k
    if threshold is not None:
        body["threshold"] = threshold
    return _request("POST", "/v1/answer", timeout=ANSWER_TIMEOUT, json=body)


def clear_static_caches() -> None:
    """Drop cached metadata, e.g. after the backend has been restarted."""
    for fn in (_meta, _corpus, _evaluation, _chunk):
        try:
            fn.clear()
        except Exception:  # noqa: BLE001 - cache clearing must never break the page
            pass

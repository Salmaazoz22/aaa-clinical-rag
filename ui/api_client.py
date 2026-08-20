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
from urllib.parse import urlsplit

import requests
import streamlit as st

from ui.settings import secret as read_secret

#: Where the API lives when nothing is configured: a backend started by hand on
#: the same machine. Correct for `uvicorn api.main:app` in another terminal, and
#: wrong everywhere else -- see `base_url()`.
DEFAULT_BASE_URL = "http://127.0.0.1:8000"

#: Generous, because a cold /v1/answer embeds the query and then waits on a
#: reasoning model. The backend's own generation timeout is 180 s.
ANSWER_TIMEOUT = 240
META_TIMEOUT = 30
HEALTH_TIMEOUT = 8

#: A hosted backend can be asleep. Railway suspends an idle container and the
#: first request pays for the wake-up plus loading the embedding model -- 24 s
#: measured against the deployed service, against a warm 0.4 s. At the local
#: 8 s that request times out, `health_or_none()` returns None, and the UI
#: reports "Degraded / API unreachable" about a backend that is perfectly
#: healthy. Only the first call after an idle period is slow, so the cost of
#: this allowance is paid once.
REMOTE_HEALTH_TIMEOUT = 30
REMOTE_META_TIMEOUT = 45

#: Setting names accepted for the backend URL, in priority order, read from
#: Streamlit secrets first and then the environment.
#:
#: TWO names, deliberately. `API_URL` is what the deployed Streamlit Cloud app
#: has in its secrets; `CLINICAL_RAG_API_URL` is this repository's older name and
#: is still what `README.md` and the local run instructions use. Honouring only
#: one of them is the whole of the production outage this constant exists to
#: prevent: the secret was set correctly, under a name nothing read, so every
#: lookup missed and `base_url()` fell through to `DEFAULT_BASE_URL` -- which is
#: why a cloud deployment reported "Nothing is answering at
#: http://127.0.0.1:8000". Neither name is a secret; both are settings whose
#: VALUE is a public URL.
API_URL_SETTINGS: tuple[str, ...] = ("API_URL", "CLINICAL_RAG_API_URL")

#: Kept for callers and docs that referred to the single old name.
API_URL_SETTING = API_URL_SETTINGS[-1]


def _secret(name: str) -> str:
    """Read one Streamlit secret, tolerating the common case of having none.

    Delegates to `ui/settings.py`, which is where the frontend's configuration
    access lives, so there is one implementation of "read a setting" rather than
    one per module. Kept as a named function because it is the seam the tests
    substitute.
    """
    return read_secret(name)


def configured_base_url() -> str | None:
    """The backend URL someone configured, or None if nobody did.

    Every accepted name is tried in Streamlit secrets first, then in the
    environment. Secrets lead because they are the only channel a Streamlit
    Community Cloud deployment has: there is no environment-variable panel, so a
    resolver that read `os.environ` alone could not be configured there at all.

    Returning None rather than the default is what lets callers tell "running
    locally against a hand-started backend" apart from "configured to talk to a
    deployed one" — a distinction the offline page needs, because the advice for
    the two situations is opposite.
    """
    for name in API_URL_SETTINGS:
        # Stripped on both channels: a secrets entry or an env var that holds
        # only whitespace is somebody having meant to set it, not a URL, and
        # must not beat the next name in the list.
        value = (_secret(name) or "").strip() or (os.environ.get(name) or "").strip()
        if value:
            return value.rstrip("/")
    return None


def base_url() -> str:
    """Where the API lives. Overridable without touching code.

    Configured value if there is one, otherwise the localhost default. The
    default is right for local development and wrong in every deployment, since
    nothing is listening on a cloud container's own 127.0.0.1 — so a deployed
    frontend MUST supply `API_URL` (see `API_URL_SETTINGS`).
    """
    return configured_base_url() or DEFAULT_BASE_URL


#: Addresses that mean "this machine". A backend here is reachable instantly and
#: is started by hand, so it gets the short timeout and the local advice.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0", ""})


def _host_of(url: str) -> str:
    """The hostname of `url`, without port, lower-cased. "" when unparseable."""
    parsed = urlsplit(url if "://" in url else f"//{url}")
    return (parsed.hostname or "").lower()


def is_remote_backend() -> bool:
    """True when the UI is pointed at a backend on another machine.

    *Configured* is not the same as *remote*: pointing the app at
    `http://127.0.0.1:9` is still local development, and it should keep the
    short health timeout and the "start uvicorn" advice. Only a non-loopback
    host is a deployment.
    """
    configured = configured_base_url()
    return bool(configured) and _host_of(configured) not in _LOOPBACK_HOSTS


def api_target() -> str:
    """A one-line, log-safe description of what the UI is talking to.

    Names the host only. The configured VALUE is a public URL rather than a
    credential, but it is read from the secrets store, so nothing quotes it
    wholesale into a log line.
    """
    configured = configured_base_url()
    if not configured:
        return "local backend (default, unconfigured)"
    host = _host_of(configured)
    if not is_remote_backend():
        return f"local backend at {host}"
    return f"configured backend at {host}"


def _health_timeout() -> int:
    return REMOTE_HEALTH_TIMEOUT if is_remote_backend() else HEALTH_TIMEOUT


def _meta_timeout() -> int:
    return REMOTE_META_TIMEOUT if is_remote_backend() else META_TIMEOUT


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
    return _request("GET", "/health", timeout=_health_timeout())


def health_or_none() -> dict[str, Any] | None:
    """`health()` that swallows the failure, for the always-on status strip."""
    try:
        return health()
    except ApiError:
        return None


# Every cached call takes the backend address as its first argument. It is not
# used inside the function — `_request` resolves the URL itself — it is there to
# be part of the cache key. Without it, repointing API_URL would
# keep serving metadata fetched from the previous backend, which is exactly the
# kind of stale provenance this project must not display.

@st.cache_data(ttl=300, show_spinner=False)
def _meta(backend: str) -> dict[str, Any]:
    return _request("GET", "/v1/meta", timeout=_meta_timeout())


@st.cache_data(ttl=600, show_spinner=False)
def _corpus(backend: str) -> dict[str, Any]:
    return _request("GET", "/v1/corpus", timeout=_meta_timeout())


@st.cache_data(ttl=3600, show_spinner=False)
def _evaluation(backend: str) -> dict[str, Any]:
    return _request("GET", "/v1/evaluation", timeout=_meta_timeout())


@st.cache_data(ttl=600, show_spinner=False)
def _chunk(backend: str, chunk_id: str) -> dict[str, Any]:
    return _request("GET", f"/v1/chunks/{chunk_id}", timeout=_meta_timeout())


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

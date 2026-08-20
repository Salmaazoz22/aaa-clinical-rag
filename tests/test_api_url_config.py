# -*- coding: utf-8 -*-
"""Where the Streamlit frontend decides the backend lives.

The production failure this pins down: the deployed Streamlit app showed

    Degraded. API unreachable.
    Backend unavailable
    Nothing is answering at http://127.0.0.1:8000

with `API_URL` correctly set in Streamlit Cloud secrets the whole time. Nothing
was wrong with the secret, the backend, or the network. `ui/api_client.base_url`
looked up exactly one name — `CLINICAL_RAG_API_URL` — the secret was named
`API_URL`, the lookup missed, and the resolver fell through to
`DEFAULT_BASE_URL`. The offline page then faithfully printed the address it had
resolved, which is why the symptom named localhost.

So these tests are about the *resolution order*, not about any one URL:

    secrets[API_URL] > secrets[CLINICAL_RAG_API_URL] > env > localhost default

Local development is a first-class case here, not an afterthought: with nothing
configured the resolver must still return `http://127.0.0.1:8000`, because that
is what `uvicorn api.main:app` in another terminal is.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui import api_client  # noqa: E402

RAILWAY = "https://aaa-clinical-rag-production.up.railway.app"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """No inherited configuration — every test states its own."""
    for name in api_client.API_URL_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(api_client, "_secret", lambda name: "")
    yield


def _with_secrets(monkeypatch, values: dict[str, str]) -> None:
    """Stand in for `st.secrets` without needing a secrets file on disk."""
    monkeypatch.setattr(api_client, "_secret", lambda name: values.get(name, ""))


# ---------------------------------------------------------------------------
# Local development
# ---------------------------------------------------------------------------

class TestLocalDevelopment:
    def test_nothing_configured_resolves_to_localhost(self):
        assert api_client.base_url() == "http://127.0.0.1:8000"

    def test_nothing_configured_is_not_a_remote_backend(self):
        assert api_client.is_remote_backend() is False
        assert api_client.configured_base_url() is None

    def test_the_local_health_timeout_is_the_short_one(self):
        assert api_client._health_timeout() == api_client.HEALTH_TIMEOUT

    @pytest.mark.parametrize(
        "url", ["http://127.0.0.1:9", "http://localhost:8000", "http://127.0.0.1:8000"]
    )
    def test_a_configured_loopback_url_is_still_local(self, monkeypatch, url):
        """Configured is not the same as remote.

        Pointing the app at another port on this machine is local development;
        it keeps the short timeout and the "start uvicorn" advice.
        """
        _with_secrets(monkeypatch, {"API_URL": url})
        assert api_client.base_url() == url
        assert api_client.is_remote_backend() is False
        assert api_client._health_timeout() == api_client.HEALTH_TIMEOUT

    def test_the_localhost_default_is_still_the_documented_one(self):
        """README and the offline page both promise this address."""
        assert api_client.DEFAULT_BASE_URL == "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------

class TestProductionSecret:
    def test_api_url_secret_is_honoured(self, monkeypatch):
        """The exact defect: this name was set in production and never read."""
        _with_secrets(monkeypatch, {"API_URL": RAILWAY})
        assert api_client.base_url() == RAILWAY
        assert api_client.is_remote_backend() is True

    def test_the_repository_legacy_name_still_works(self, monkeypatch):
        _with_secrets(monkeypatch, {"CLINICAL_RAG_API_URL": RAILWAY})
        assert api_client.base_url() == RAILWAY

    def test_api_url_wins_when_both_names_are_present(self, monkeypatch):
        _with_secrets(
            monkeypatch,
            {"API_URL": RAILWAY, "CLINICAL_RAG_API_URL": "https://old.example.com"},
        )
        assert api_client.base_url() == RAILWAY

    def test_a_secret_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("API_URL", "https://env.example.com")
        _with_secrets(monkeypatch, {"API_URL": RAILWAY})
        assert api_client.base_url() == RAILWAY

    def test_the_environment_works_when_there_are_no_secrets(self, monkeypatch):
        monkeypatch.setenv("API_URL", RAILWAY)
        assert api_client.base_url() == RAILWAY

    def test_a_trailing_slash_is_stripped(self, monkeypatch):
        """`f"{base_url()}{path}"` would otherwise build `//health`."""
        _with_secrets(monkeypatch, {"API_URL": RAILWAY + "/"})
        assert api_client.base_url() == RAILWAY

    def test_blank_and_whitespace_settings_are_ignored(self, monkeypatch):
        _with_secrets(monkeypatch, {"API_URL": "   "})
        assert api_client.base_url() == api_client.DEFAULT_BASE_URL

    def test_a_remote_backend_gets_the_cold_start_allowance(self, monkeypatch):
        """Railway suspends idle containers; the first request measured 24 s."""
        _with_secrets(monkeypatch, {"API_URL": RAILWAY})
        assert api_client._health_timeout() == api_client.REMOTE_HEALTH_TIMEOUT
        assert api_client.REMOTE_HEALTH_TIMEOUT > api_client.HEALTH_TIMEOUT

    def test_a_missing_secrets_file_does_not_raise(self, monkeypatch):
        """`st.secrets` raises when there is no secrets file — the normal local case."""
        monkeypatch.setattr(
            api_client.st, "secrets", property(lambda self: (_ for _ in ()).throw(RuntimeError))
        )
        assert api_client.base_url() == api_client.DEFAULT_BASE_URL


# ---------------------------------------------------------------------------
# Every request path uses the one resolved value
# ---------------------------------------------------------------------------

class TestAllCallsUseTheConfiguredBackend:
    @pytest.mark.parametrize(
        "call, expected_path",
        [
            (lambda: api_client.health(), "/health"),
            (lambda: api_client.meta(), "/v1/meta"),
            (lambda: api_client.corpus(), "/v1/corpus"),
            (lambda: api_client.evaluation(), "/v1/evaluation"),
            (lambda: api_client.answer("Q"), "/v1/answer"),
        ],
    )
    def test_the_request_url_is_the_configured_backend(self, monkeypatch, call, expected_path):
        _with_secrets(monkeypatch, {"API_URL": RAILWAY})
        api_client.clear_static_caches()
        seen: list[str] = []

        class _Response:
            status_code = 200

            @staticmethod
            def json():
                return {}

        monkeypatch.setattr(
            api_client.requests,
            "request",
            lambda method, url, **kw: (seen.append(url), _Response())[1],
        )
        call()
        assert seen == [f"{RAILWAY}{expected_path}"]
        assert "127.0.0.1" not in seen[0] and "localhost" not in seen[0]

    def test_the_offline_page_reports_the_configured_address(self, monkeypatch):
        """The degraded state must not name localhost when a backend is configured."""
        _with_secrets(monkeypatch, {"API_URL": RAILWAY})
        assert "127.0.0.1" not in api_client.base_url()


# ---------------------------------------------------------------------------
# Nothing leaks the secret store
# ---------------------------------------------------------------------------

class TestNoSecretLeak:
    def test_api_target_names_a_host_not_a_full_setting_dump(self, monkeypatch):
        _with_secrets(monkeypatch, {"API_URL": RAILWAY})
        target = api_client.api_target()
        assert "aaa-clinical-rag-production.up.railway.app" in target
        assert "https://" not in target

    def test_api_target_says_so_when_nothing_is_configured(self):
        assert "local" in api_client.api_target()

    def test_no_credential_shaped_setting_is_read_by_the_ui(self):
        """The frontend must never reach for a key: it calls no vendor directly."""
        for name in api_client.API_URL_SETTINGS:
            assert "KEY" not in name.upper() and "TOKEN" not in name.upper()

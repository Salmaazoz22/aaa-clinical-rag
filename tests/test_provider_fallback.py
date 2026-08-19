# -*- coding: utf-8 -*-
"""Cross-provider fallback, and what a failed model call is allowed to say.

Two defects are pinned here.

**The fallback was wired but dead.** `build_provider` correctly returned a
`FallbackProvider(groq -> openrouter)`, and `FallbackProvider` correctly tried
the secondary when the primary raised — but the secondary's model was pinned to
`deepseek/deepseek-r1:free`, which OpenRouter withdrew. Every fallback attempt
404'd, so a Groq rate limit reached the caller as a 502 even though a working
second provider was configured. The fix is a live slug plus an environment
override, so the next retirement is a `.env` edit rather than a code change.

**The failure text leaked.** `ProviderError` carries the vendor, the model, the
HTTP status and the upstream JSON body — which for a Groq 429 includes the
account's organisation id and billing links. That went straight into the HTTP
502 detail and onto the user's screen.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.config import GenerationSettings  # noqa: E402
from generation.providers import (  # noqa: E402
    PROVIDER_SPECS,
    SAFE_PROVIDER_MESSAGE,
    Completion,
    FallbackProvider,
    OpenAICompatibleProvider,
    ProviderError,
    build_provider,
    resolve_model,
    resolve_provider_spec,
)

#: The real body of the Groq 429 this project hit, reproduced verbatim. Any
#: assertion about redaction has to be made against the actual leak.
GROQ_429 = (
    "groq/openai/gpt-oss-120b completion failed: Error code: 429 - {'error': "
    "{'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization "
    "`org_01kzk67t3pecaaq5szfbrvstnn` service tier `on_demand` on tokens per day "
    "(TPD): Limit 200000, Used 196802, Requested 6252. Please try again in 21m59s. "
    "Upgrade at https://console.groq.com/settings/billing', 'code': "
    "'rate_limit_exceeded'}}"
)


class _Primary:
    name = "groq"
    model = "openai/gpt-oss-120b"

    def __init__(self, fail: Exception | None = None):
        self.fail = fail
        self.calls = 0

    def complete(self, messages, *, json_mode: bool = True) -> Completion:
        self.calls += 1
        if self.fail:
            raise self.fail
        return Completion(text='{"ok": true}', provider=self.name, model=self.model)


class _Secondary(_Primary):
    name = "openrouter"
    model = "openai/gpt-oss-20b:free"


def _settings(**overrides) -> GenerationSettings:
    base = dict(
        provider="groq",
        model="openai/gpt-oss-120b",
        base_url="https://api.groq.com/openai/v1",
        api_key="test-primary-key",
        top_k=5,
        score_threshold=0.75,
        temperature=0.0,
        max_output_tokens=1000,
        timeout=30.0,
        enable_fallback=True,
    )
    base.update(overrides)
    return GenerationSettings(**base)


# ---------------------------------------------------------------------------
# The dead model slug
# ---------------------------------------------------------------------------

class TestFallbackModelIsLive:
    def test_the_retired_deepseek_slug_is_gone(self):
        """`deepseek/deepseek-r1:free` 404s on every call and must not be pinned."""
        assert PROVIDER_SPECS["openrouter"].model != "deepseek/deepseek-r1:free"

    def test_the_openrouter_model_is_overridable_from_the_environment(self, monkeypatch):
        spec = resolve_provider_spec("openrouter")
        assert spec.model_env == "OPENROUTER_MODEL"
        monkeypatch.setenv("OPENROUTER_MODEL", "some/other-model:free")
        assert resolve_model(spec) == "some/other-model:free"

    def test_an_unset_override_falls_back_to_the_pin(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        spec = resolve_provider_spec("openrouter")
        assert resolve_model(spec) == spec.model


# ---------------------------------------------------------------------------
# Primary success / primary failure
# ---------------------------------------------------------------------------

class TestFallbackBehaviour:
    def test_primary_success_uses_the_primary(self):
        primary, secondary = _Primary(), _Secondary()
        completion = FallbackProvider(primary=primary, secondary=secondary).complete([])
        assert completion.provider == "groq"
        assert primary.calls == 1
        assert secondary.calls == 0, "the secondary must not be called when the primary works"

    def test_primary_failure_triggers_the_fallback(self):
        primary, secondary = _Primary(fail=ProviderError(GROQ_429)), _Secondary()
        completion = FallbackProvider(primary=primary, secondary=secondary).complete([])
        assert completion.provider == "openrouter"
        assert primary.calls == 1 and secondary.calls == 1

    def test_both_failing_raises_one_error_naming_both(self):
        provider = FallbackProvider(
            primary=_Primary(fail=ProviderError(GROQ_429)),
            secondary=_Secondary(fail=ProviderError("openrouter 404 model unavailable")),
        )
        with pytest.raises(ProviderError) as excinfo:
            provider.complete([])
        assert "groq" in str(excinfo.value) and "openrouter" in str(excinfo.value)

    def test_json_mode_is_passed_through_to_the_secondary(self):
        seen = {}

        class _Recording(_Secondary):
            def complete(self, messages, *, json_mode: bool = True):
                seen["json_mode"] = json_mode
                return Completion(text="{}", provider=self.name, model=self.model)

        FallbackProvider(
            primary=_Primary(fail=ProviderError("boom")), secondary=_Recording()
        ).complete([], json_mode=False)
        assert seen["json_mode"] is False


# ---------------------------------------------------------------------------
# build_provider wiring
# ---------------------------------------------------------------------------

class TestBuildProvider:
    def test_fallback_is_attached_when_both_keys_are_present(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-secondary-key")
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        provider = build_provider(_settings())
        assert isinstance(provider, FallbackProvider)
        assert provider.name == "groq->openrouter"
        assert provider.secondary.model == PROVIDER_SPECS["openrouter"].model

    def test_no_secondary_key_means_no_fallback(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        provider = build_provider(_settings())
        assert isinstance(provider, OpenAICompatibleProvider)

    def test_fallback_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-secondary-key")
        provider = build_provider(_settings(enable_fallback=False))
        assert isinstance(provider, OpenAICompatibleProvider)

    def test_the_fallback_model_is_configurable(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-secondary-key")
        provider = build_provider(_settings(fallback_model="deepseek/deepseek-r1"))
        assert isinstance(provider, FallbackProvider)
        assert provider.secondary.model == "deepseek/deepseek-r1"

    def test_a_fallback_to_the_same_provider_is_not_built(self, monkeypatch):
        """Retrying the endpoint that just rate-limited adds nothing."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-secondary-key")
        provider = build_provider(_settings(fallback_provider="groq"))
        assert isinstance(provider, OpenAICompatibleProvider)

    def test_no_api_key_is_ever_in_the_described_settings(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-secondary-key")
        described = _settings().describe()
        assert "test-primary-key" not in str(described)
        assert described["api_key_supplied"] is True


# ---------------------------------------------------------------------------
# What the caller is told
# ---------------------------------------------------------------------------

class TestSafeErrorMessage:
    def test_the_safe_message_carries_no_provider_internals(self):
        for leak in ("org_", "429", "groq", "openrouter", "api.groq.com", "billing", "Limit "):
            assert leak not in SAFE_PROVIDER_MESSAGE

    def test_the_safe_message_says_no_answer_was_invented(self):
        assert "fabricat" in SAFE_PROVIDER_MESSAGE.lower()

    def test_the_api_returns_the_safe_message_not_the_upstream_body(self, monkeypatch):
        from fastapi.testclient import TestClient

        import api.main as main

        class _Retriever:
            def search(self, query, top_k=5):  # pragma: no cover - not reached
                return []

        monkeypatch.setattr(main, "_retriever", _Retriever(), raising=False)

        def _boom(**kwargs):
            raise ProviderError(GROQ_429)

        monkeypatch.setattr(main, "answer_question", _boom)

        response = TestClient(main.app, raise_server_exceptions=False).post(
            "/v1/answer", json={"question": "At what diameter is repair recommended?"}
        )
        assert response.status_code == 502
        detail = response.json()["detail"]
        assert detail == SAFE_PROVIDER_MESSAGE
        for leak in ("org_01kzk67t3pecaaq5szfbrvstnn", "console.groq.com", "rate_limit_exceeded"):
            assert leak not in response.text

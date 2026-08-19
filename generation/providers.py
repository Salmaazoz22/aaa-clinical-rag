# -*- coding: utf-8 -*-
"""Model access: one thin interface, two OpenAI-compatible providers.

Adapted from the reference implementation's `stores/llm/` layer (interface +
per-provider class + factory), which is the right decomposition. What changed:

* the reference's factory returns a provider whose model is then set by a
  separate `set_generation_model()` call, so a provider can exist in a
  half-configured state where `generate_text` logs an error and returns `None`.
  Here a provider is constructed with its model and cannot exist unconfigured,
  and a failed call raises instead of returning `None` -- a silent `None` in a
  clinical answering path becomes an empty answer with no explanation;
* both providers here are OpenAI-compatible, so there is one implementation
  parameterised by a `ProviderSpec` rather than one class per vendor;
* no embedding methods. Embeddings are frozen and owned by the retrieval layer
  (`ingestion.chunking.load_embedder`); a second embedding path in the generation
  layer is exactly the kind of duplicate that must not exist here.

Switching provider is one line in `.env`:

    GENERATION_PROVIDER=groq          # openai/gpt-oss-120b, fast iteration
    GENERATION_PROVIDER=openrouter    # deepseek/deepseek-r1:free, final answers
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # avoids a circular import at runtime
    from generation.config import GenerationSettings


class ProviderError(RuntimeError):
    """Raised when a provider cannot be built or a completion cannot be obtained."""


class UnknownProviderError(ProviderError):
    pass


class MissingAPIKeyError(ProviderError):
    pass


@dataclass(frozen=True)
class ProviderSpec:
    """Everything that differs between the two supported providers."""

    name: str
    base_url: str
    model: str
    key_env: str

    #: Whether the endpoint honours `response_format={"type": "json_object"}`.
    #: When it does not, the JSON contract survives only as a prompt instruction
    #: and `generation/parsing.py` has to do the work.
    supports_json_mode: bool

    #: `max_tokens` is deprecated on some OpenAI-compatible endpoints and
    #: rejected on others, so the parameter name is part of the spec.
    max_tokens_param: str = "max_tokens"

    #: Optional headers, each read from an environment variable and omitted when
    #: unset. OpenRouter uses these for app attribution; nothing depends on them.
    headers_from_env: tuple[tuple[str, str], ...] = ()

    #: True for models that emit a separate reasoning stream. Used only to decide
    #: whether an empty `content` with a non-empty reasoning field is worth
    #: reporting as such rather than as a blank response.
    is_reasoning_model: bool = False


PROVIDER_SPECS: dict[str, ProviderSpec] = {
    "groq": ProviderSpec(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-120b",
        key_env="GROQ_API_KEY",
        supports_json_mode=True,
        max_tokens_param="max_completion_tokens",
        is_reasoning_model=True,
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-r1:free",
        key_env="OPENROUTER_API_KEY",
        # DeepSeek-R1 via OpenRouter has no reliable JSON mode, so the schema is
        # enforced by the prompt and by the parser, not by the endpoint.
        supports_json_mode=False,
        max_tokens_param="max_tokens",
        headers_from_env=(("HTTP-Referer", "OPENROUTER_SITE_URL"), ("X-Title", "OPENROUTER_APP_NAME")),
        is_reasoning_model=True,
    ),
}


def resolve_provider_spec(name: str) -> ProviderSpec:
    key = (name or "").strip().lower()
    if key not in PROVIDER_SPECS:
        raise UnknownProviderError(
            f"unknown GENERATION_PROVIDER {name!r}; supported: {sorted(PROVIDER_SPECS)}"
        )
    return PROVIDER_SPECS[key]


@dataclass
class Completion:
    """One model response, plus what is needed to audit how it was obtained."""

    text: str
    provider: str
    model: str
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    latency_s: float = 0.0
    reasoning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "latency_s": round(self.latency_s, 3),
            "response_chars": len(self.text or ""),
            "had_reasoning_stream": bool(self.reasoning),
        }


class LLMProvider(ABC):
    """The whole interface the generation layer needs from a model."""

    name: str
    model: str

    @abstractmethod
    def complete(self, messages: Sequence[dict[str, str]], *, json_mode: bool = True) -> Completion:
        """Return one completion for `messages`, or raise `ProviderError`."""


@dataclass
class OpenAICompatibleProvider(LLMProvider):
    """Chat completions over any OpenAI-compatible endpoint."""

    spec: ProviderSpec
    model: str
    api_key: str
    temperature: float = 0.0
    max_output_tokens: int = 4000
    timeout: float = 180.0
    max_retries: int = 3
    _client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.name = self.spec.name
        if not self.api_key:
            raise MissingAPIKeyError(
                f"{self.spec.key_env} is not set; add it to .env (see .env.example). "
                f"Real keys are never committed."
            )

    @property
    def client(self):
        """The OpenAI SDK client, built on first use."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ProviderError(
                    "the `openai` package is required for the generation layer: "
                    "pip install -r requirements.txt"
                ) from exc

            headers = {}
            for header, env_var in self.spec.headers_from_env:
                value = os.environ.get(env_var)
                if value:
                    headers[header] = value

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.spec.base_url,
                timeout=self.timeout,
                # Free tiers rate-limit aggressively; the SDK's own retry honours
                # Retry-After, which is better than a hand-rolled backoff loop.
                max_retries=self.max_retries,
                default_headers=headers or None,
            )
        return self._client

    def complete(self, messages: Sequence[dict[str, str]], *, json_mode: bool = True) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            self.spec.max_tokens_param: self.max_output_tokens,
        }
        if json_mode and self.spec.supports_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        started = time.monotonic()
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - surfaced with provider context
            raise ProviderError(f"{self.spec.name}/{self.model} completion failed: {exc}") from exc
        latency = time.monotonic() - started

        if not response.choices:
            raise ProviderError(f"{self.spec.name}/{self.model} returned no choices")

        choice = response.choices[0]
        message = choice.message
        text = (getattr(message, "content", None) or "").strip()

        # Reasoning models expose their chain of thought on a non-standard field
        # (`reasoning` on OpenRouter, `reasoning_content` elsewhere). It is kept
        # only for diagnostics -- it is never parsed as the answer, and never
        # treated as evidence.
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)

        if not text:
            detail = "empty content"
            if reasoning:
                detail = "empty content with a non-empty reasoning stream (raise max_output_tokens)"
            raise ProviderError(
                f"{self.spec.name}/{self.model} returned {detail}; "
                f"finish_reason={choice.finish_reason!r}"
            )

        usage = None
        if getattr(response, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                "total_tokens": getattr(response.usage, "total_tokens", None),
            }

        return Completion(
            text=text,
            provider=self.spec.name,
            model=self.model,
            finish_reason=choice.finish_reason,
            usage=usage,
            latency_s=latency,
            reasoning=reasoning if isinstance(reasoning, str) else None,
        )


@dataclass
class FallbackProvider(LLMProvider):
    """Wraps primary and secondary providers to provide cross-provider fallback."""

    primary: LLMProvider
    secondary: LLMProvider

    def __post_init__(self) -> None:
        self.name = f"{self.primary.name}->{self.secondary.name}"
        self.model = f"{self.primary.model} (fallback: {self.secondary.model})"

    def complete(self, messages: Sequence[dict[str, str]], *, json_mode: bool = True) -> Completion:
        try:
            return self.primary.complete(messages, json_mode=json_mode)
        except ProviderError as primary_exc:
            try:
                return self.secondary.complete(messages, json_mode=json_mode)
            except ProviderError as secondary_exc:
                raise ProviderError(
                    f"primary provider '{self.primary.name}' failed ({primary_exc}); "
                    f"secondary provider '{self.secondary.name}' also failed ({secondary_exc})"
                ) from secondary_exc


def build_provider(settings: "GenerationSettings" | None = None) -> LLMProvider:
    """Build the provider selected by configuration."""
    if settings is None:
        from generation.config import load_settings

        settings = load_settings()

    spec = resolve_provider_spec(settings.provider)
    primary = OpenAICompatibleProvider(
        spec=spec,
        model=settings.model,
        api_key=settings.api_key or "",
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        timeout=settings.timeout,
    )

    if not getattr(settings, "enable_fallback", False):
        return primary

    secondary_name = "openrouter" if spec.name != "openrouter" else "groq"
    secondary_spec = resolve_provider_spec(secondary_name)
    secondary_key = os.environ.get(secondary_spec.key_env) or ""
    secondary = OpenAICompatibleProvider(
        spec=secondary_spec,
        model=secondary_spec.model,
        api_key=secondary_key,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        timeout=settings.timeout,
    )

    return FallbackProvider(primary=primary, secondary=secondary)

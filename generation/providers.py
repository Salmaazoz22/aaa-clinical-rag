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
    GENERATION_PROVIDER=openrouter    # openai/gpt-oss-20b:free, second opinion

When `GENERATION_ENABLE_FALLBACK` is on (the default), the provider that is NOT
selected becomes the secondary, and `FallbackProvider` tries it when the primary
raises. Every model slug is overridable from the environment
(`GENERATION_MODEL`, `GROQ_MODEL`, `OPENROUTER_MODEL`) so a retired slug is a
`.env` edit, not a code change -- which is exactly the failure this layer hit
when `deepseek/deepseek-r1:free` was withdrawn.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
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


#: What a caller outside this process is told when a model call fails.
#:
#: A `ProviderError` message is written for the operator: it carries the vendor,
#: the model slug, the HTTP status and the upstream JSON body -- which, for a
#: rate limit, includes the account's organisation id and billing links. None of
#: that belongs in an HTTP response or on a screen, and none of it helps the
#: person who asked a clinical question. The detail goes to the log; the caller
#: gets this.
SAFE_PROVIDER_MESSAGE = (
    "The language model service is temporarily unavailable, so no answer was "
    "generated. Nothing has been fabricated in its place. Please try again shortly."
)


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

    #: Environment variable that overrides `model` for this provider. Exists so a
    #: model slug the vendor retires can be replaced without a code change --
    #: including when the provider is acting as the fallback, which
    #: `GENERATION_MODEL` cannot reach because that names the *primary*.
    model_env: str = ""

    #: Largest single request the endpoint will accept, in tokens, counting the
    #: prompt PLUS the completion budget the caller reserves. 0 means "no known
    #: limit" and disables the guard.
    #:
    #: This is not a nicety. Groq's free tier caps a request at 8,000 tokens per
    #: minute and counts `max_completion_tokens` as *reserved*, not as used. With
    #: a ~4,250-token prompt and a 4,000-token reservation, every answer request
    #: totalled 8,287 and came back `413 Request too large` -- deterministically,
    #: on every call, even though the model only ever generated ~1,200 tokens.
    #: The pipeline then fell through to a free fallback model that takes 97-150 s,
    #: which is the entire 240 s production timeout. See `_fit_completion_budget`.
    max_request_tokens: int = 0

    #: Environment variable overriding `max_request_tokens`, so a paid tier or a
    #: different model can raise the ceiling without a code change.
    max_request_tokens_env: str = ""

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
        model_env="GROQ_MODEL",
        # Free-tier tokens-per-minute ceiling for gpt-oss-120b. Raise it with
        # GROQ_MAX_REQUEST_TOKENS on a paid tier.
        max_request_tokens=8000,
        max_request_tokens_env="GROQ_MAX_REQUEST_TOKENS",
        supports_json_mode=True,
        max_tokens_param="max_completion_tokens",
        is_reasoning_model=True,
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        # `deepseek/deepseek-r1:free` was retired by OpenRouter and now 404s on
        # every call ("This model is unavailable for free"), which silently
        # disabled the cross-provider fallback: a Groq rate-limit became a 502
        # instead of a second attempt. Pinned here to the free gpt-oss sibling of
        # the Groq primary -- same model family, same OpenAI-compatible contract,
        # same prompt behaviour -- and overridable with OPENROUTER_MODEL when a
        # deployment wants a different (or paid) slug.
        model="openai/gpt-oss-20b:free",
        key_env="OPENROUTER_API_KEY",
        model_env="OPENROUTER_MODEL",
        # OpenRouter publishes no single per-request token ceiling that applies
        # across its model catalogue, so the guard stays off unless configured.
        max_request_tokens=0,
        max_request_tokens_env="OPENROUTER_MAX_REQUEST_TOKENS",
        # OpenRouter's free tier does not honour `response_format` uniformly
        # across providers, so the schema is enforced by the prompt and by the
        # parser, not by the endpoint.
        supports_json_mode=False,
        max_tokens_param="max_tokens",
        headers_from_env=(("HTTP-Referer", "OPENROUTER_SITE_URL"), ("X-Title", "OPENROUTER_APP_NAME")),
        is_reasoning_model=True,
    ),
}


#: Characters per token when estimating prompt size. Measured against this
#: project's own prompts on Groq: 17,312 characters reported as 4,287 prompt
#: tokens is 4.04 chars/token. 3.5 deliberately OVER-estimates, because the cost
#: of guessing high is a slightly smaller completion budget and the cost of
#: guessing low is the 413 this guard exists to prevent.
CHARS_PER_TOKEN = 3.5

#: Held back from the request budget for the chat scaffolding the endpoint adds
#: around the messages (role envelopes, tool preamble, response_format).
#: Measured overhead was ~40 tokens; this is generous on purpose.
REQUEST_TOKEN_RESERVE = 256

#: Below this, a completion cannot hold a citation-carrying answer, so it is
#: better to fail fast and let the fallback try than to send a request that can
#: only produce a truncated one.
MIN_COMPLETION_TOKENS = 512

#: Less remaining budget than this and the fallback is not worth starting: a
#: request that is certain to be cut off mid-generation costs the caller time and
#: returns nothing. Better to fail immediately with a clear, bounded error.
MIN_FALLBACK_SECONDS = 5.0


class RequestTooLargeError(ProviderError):
    """The prompt cannot fit this provider's per-request token ceiling.

    A distinct type because it is *not* retryable and not transient: the same
    request will fail identically every time, so the fallback should be tried
    immediately rather than after a retry ladder.
    """


def estimate_tokens(messages: Sequence[dict[str, str]]) -> int:
    """Approximate prompt size in tokens, erring high. No tokeniser needed.

    The exact count is the endpoint's business and differs per model; what this
    has to be is a safe upper bound, so that `_fit_completion_budget` never
    reserves more than the endpoint will accept.
    """
    characters = sum(len(str(m.get("content") or "")) for m in messages)
    return int(characters / CHARS_PER_TOKEN) + REQUEST_TOKEN_RESERVE


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

    #: Retries the SDK makes for ONE provider. Deliberately low.
    #:
    #: Cross-provider fallback is a better retry than same-provider retry: when
    #: the primary is rate-limited or metering the request as too large, trying
    #: it again produces the same answer while spending the budget the fallback
    #: needs. The SDK's ladder also honours Retry-After, so a 429 with a long
    #: hint can eat the whole deadline on its own. One retry covers a genuinely
    #: transient blip; anything past that is the other provider's job.
    max_retries: int = 1
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

    def request_token_ceiling(self) -> int:
        """This provider's per-request token ceiling, or 0 when unknown."""
        if self.spec.max_request_tokens_env:
            override = (os.environ.get(self.spec.max_request_tokens_env) or "").strip()
            if override:
                try:
                    return max(0, int(override))
                except ValueError:
                    pass
        return self.spec.max_request_tokens

    def _fit_completion_budget(self, messages: Sequence[dict[str, str]]) -> int:
        """How many completion tokens may be RESERVED for this request.

        Endpoints that meter by tokens-per-minute charge the reservation, not the
        usage: Groq counts `prompt + max_completion_tokens` against an 8,000 TPM
        free-tier ceiling, so reserving 4,000 for a model that generates ~1,200
        turned every request into `413 Request too large`. Shrinking the
        reservation to fit costs nothing -- the model stops when it is done, and
        `finish_reason` still reports truncation if it ever were binding.

        Raises `RequestTooLargeError` when even the minimum will not fit, so the
        caller can fall back at once instead of sending a request that is certain
        to be rejected.
        """
        ceiling = self.request_token_ceiling()
        if not ceiling:
            return self.max_output_tokens

        prompt_tokens = estimate_tokens(messages)
        allowed = ceiling - prompt_tokens
        if allowed < MIN_COMPLETION_TOKENS:
            raise RequestTooLargeError(
                f"{self.spec.name}/{self.model}: the prompt needs about {prompt_tokens} tokens "
                f"and the endpoint accepts {ceiling} per request, leaving {allowed} for the "
                f"answer (minimum {MIN_COMPLETION_TOKENS}). Reduce the evidence sent, or raise "
                f"{self.spec.max_request_tokens_env or 'the request ceiling'}."
            )
        return min(self.max_output_tokens, allowed)

    def complete(self, messages: Sequence[dict[str, str]], *, json_mode: bool = True) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            self.spec.max_tokens_param: self._fit_completion_budget(messages),
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


class DeadlineExceededError(ProviderError):
    """The generation budget for one request ran out."""


@dataclass
class FallbackProvider(LLMProvider):
    """Primary, then secondary, under ONE wall-clock budget for the pair.

    The budget is the point. Without it the worst case is the sum of every
    provider's own timeout times its retry count, which for the previous
    configuration (180 s, 3 retries, two providers) was twenty-four minutes with
    nothing in the request path able to stop it. What the client actually
    experienced was a 240 s read timeout and no answer.

    `deadline_s` bounds the pair. Before trying the secondary the remaining time
    is computed, and the secondary is given a timeout no larger than what is
    left; if nothing meaningful is left, the attempt is skipped and a bounded
    error is raised instead. A caller therefore knows the maximum time a
    generation can take, which is what makes a graceful failure possible.
    """

    primary: LLMProvider
    secondary: LLMProvider

    #: Wall-clock budget for the whole chain, seconds. 0 disables the bound.
    deadline_s: float = 0.0

    def __post_init__(self) -> None:
        self.name = f"{self.primary.name}->{self.secondary.name}"
        self.model = f"{self.primary.model} (fallback: {self.secondary.model})"

    def _call_within(
        self, provider: LLMProvider, messages: Sequence[dict[str, str]],
        json_mode: bool, budget: float | None,
    ) -> Completion:
        """Run `provider` under a hard wall-clock bound of `budget` seconds.

        TWO mechanisms, because the SDK's own timeout is not sufficient on its
        own. `timeout` reaches httpx as a per-operation read timeout: it fires
        when a socket read stalls, not when total elapsed time is long. A
        provider that drips a long response steadily never trips it. Measured in
        production: a 90 s deadline and a 60 s socket timeout, and the fallback
        still returned after 101.6 s.

        So the socket timeout is shortened AND the call is run on a worker
        thread the caller stops waiting for when the budget expires. The thread
        is left to finish and be discarded -- there is no way to interrupt a
        blocking socket read in CPython -- but the CALLER's wait is bounded,
        which is what makes the response time predictable.
        """
        if budget is None:
            return provider.complete(messages, json_mode=json_mode)

        original = getattr(provider, "timeout", None)
        cached = getattr(provider, "_client", None)
        if original is not None:
            provider.timeout = min(original, budget)
            # The SDK client caches its own timeout, so it has to be rebuilt
            # when the budget shortens it. `_client` is the lazy-build slot.
            provider._client = None

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")
        try:
            future = executor.submit(provider.complete, messages, json_mode=json_mode)
            try:
                return future.result(timeout=budget)
            except FuturesTimeout as exc:
                raise DeadlineExceededError(
                    f"{getattr(provider, 'name', 'provider')} did not answer within the "
                    f"{budget:.0f}s left of the {self.deadline_s:.0f}s generation budget"
                ) from exc
        finally:
            # Do not block on a call that overran: shutdown(wait=False) lets the
            # orphaned request finish in the background and be dropped.
            executor.shutdown(wait=False)
            if original is not None:
                provider.timeout = original
                provider._client = cached

    def complete(self, messages: Sequence[dict[str, str]], *, json_mode: bool = True) -> Completion:
        started = time.monotonic()

        def remaining() -> float | None:
            if not self.deadline_s:
                return None
            return self.deadline_s - (time.monotonic() - started)

        try:
            return self._call_within(self.primary, messages, json_mode, remaining())
        except ProviderError as primary_exc:
            left = remaining()
            if left is not None and left < MIN_FALLBACK_SECONDS:
                raise DeadlineExceededError(
                    f"primary provider '{self.primary.name}' failed ({primary_exc}); "
                    f"the {self.deadline_s:.0f}s generation budget left {max(0.0, left):.1f}s, "
                    f"too little to try '{self.secondary.name}'"
                ) from primary_exc
            try:
                return self._call_within(self.secondary, messages, json_mode, left)
            except ProviderError as secondary_exc:
                raise ProviderError(
                    f"primary provider '{self.primary.name}' failed ({primary_exc}); "
                    f"secondary provider '{self.secondary.name}' also failed ({secondary_exc})"
                ) from secondary_exc


def resolve_model(spec: ProviderSpec) -> str:
    """The model slug in force for `spec`: its per-provider env override, or the pin.

    Read for the *secondary* provider as well as the primary, which is the point:
    when a vendor retires a slug, the fallback can be repaired from `.env` alone.
    """
    if spec.model_env:
        override = (os.environ.get(spec.model_env) or "").strip()
        if override:
            return override
    return spec.model


def _sdk_retries() -> int:
    """Per-provider SDK retries, from GENERATION_MAX_RETRIES (default 1)."""
    try:
        return max(0, int((os.environ.get("GENERATION_MAX_RETRIES") or "1").strip()))
    except ValueError:
        return 1


def build_provider(settings: "GenerationSettings" | None = None) -> LLMProvider:
    """Build the provider selected by configuration, with its fallback attached."""
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
        max_retries=_sdk_retries(),
    )

    if not getattr(settings, "enable_fallback", False):
        return primary

    secondary_name = getattr(settings, "fallback_provider", "") or (
        "openrouter" if spec.name != "openrouter" else "groq"
    )
    if secondary_name == spec.name:
        # A fallback to the same endpoint would not survive the failure that
        # triggered it (a rate limit, a dead slug), so there is nothing to add.
        return primary

    secondary_spec = resolve_provider_spec(secondary_name)
    secondary_key = os.environ.get(secondary_spec.key_env) or ""
    if not secondary_key:
        return primary

    secondary = OpenAICompatibleProvider(
        spec=secondary_spec,
        model=getattr(settings, "fallback_model", "") or resolve_model(secondary_spec),
        api_key=secondary_key,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        timeout=settings.timeout,
        max_retries=_sdk_retries(),
    )

    return FallbackProvider(
        primary=primary,
        secondary=secondary,
        deadline_s=float(getattr(settings, "deadline", 0.0) or 0.0),
    )

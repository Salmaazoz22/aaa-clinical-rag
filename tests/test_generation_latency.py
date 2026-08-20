# -*- coding: utf-8 -*-
"""The latency budget: request sizing, the deadline, and bounded failure.

The production incident this file pins down. A question took 130-190 s and the
Streamlit client gave up at its 240 s read timeout. Measured, the time went:

    safety          5 ms
    retrieval     126 ms
    grounding       0 ms
    generation  ~34 s on the PRIMARY, which failed, then 97-150 s on the fallback
    validation      8 ms

The primary failed on every single request with `413 Request too large`:

    Limit 8000 TPM, Requested 8287

Groq's free tier meters a request as `prompt + max_completion_tokens`, counting
the completion budget as RESERVED rather than used. The prompt is ~4,250 tokens
and `max_output_tokens` was 4,000, so every request asked for 8,287 against an
8,000 ceiling — while the model, when it did run, generated only ~1,200 tokens.
The reservation was never needed and it was the whole outage.

Two independent defects, so two independent guards:

    _fit_completion_budget   never send a request that cannot fit
    FallbackProvider.deadline_s   never let the chain run unbounded

The second matters on its own: with 180 s per provider, 3 SDK retries and two
providers, the worst case was 24 minutes and nothing in the request path could
interrupt it.

No network here. Providers are substituted; the numbers above come from the
measurements recorded in the commit message.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.config import GenerationSettings, DEFAULT_DEADLINE, DEFAULT_TIMEOUT  # noqa: E402
from generation.providers import (  # noqa: E402
    CHARS_PER_TOKEN,
    MIN_COMPLETION_TOKENS,
    MIN_FALLBACK_SECONDS,
    PROVIDER_SPECS,
    Completion,
    DeadlineExceededError,
    FallbackProvider,
    OpenAICompatibleProvider,
    ProviderError,
    RequestTooLargeError,
    estimate_tokens,
    resolve_provider_spec,
)

#: The real production prompt size: 17,312 characters, reported by Groq as 4,287
#: prompt tokens.
REAL_PROMPT_CHARS = 17312
REAL_PROMPT_TOKENS = 4287


def _messages(chars: int) -> list[dict[str, str]]:
    return [{"role": "system", "content": "x" * (chars // 2)},
            {"role": "user", "content": "x" * (chars - chars // 2)}]


def _provider(name: str = "groq", *, max_output_tokens: int = 4000, **kw):
    return OpenAICompatibleProvider(
        spec=resolve_provider_spec(name), model="test-model", api_key="test-key",
        max_output_tokens=max_output_tokens, **kw,
    )


# ---------------------------------------------------------------------------
# 1. Request sizing — the 413
# ---------------------------------------------------------------------------

class TestRequestSizing:
    def test_the_groq_free_tier_ceiling_is_declared(self):
        assert PROVIDER_SPECS["groq"].max_request_tokens == 8000
        assert PROVIDER_SPECS["groq"].max_request_tokens_env == "GROQ_MAX_REQUEST_TOKENS"

    def test_the_token_estimate_errs_high(self):
        """Guessing low resends the 413; guessing high only trims the budget."""
        estimated = estimate_tokens(_messages(REAL_PROMPT_CHARS))
        assert estimated >= REAL_PROMPT_TOKENS, "estimate must not undercount the real prompt"
        assert estimated < REAL_PROMPT_TOKENS * 1.5, "but it must not be wildly pessimistic"

    def test_the_reserved_budget_is_trimmed_to_fit(self):
        """The exact failing request: 4,287 prompt + 4,000 reserved = 8,287 > 8,000."""
        provider = _provider(max_output_tokens=4000)
        messages = _messages(REAL_PROMPT_CHARS)
        budget = provider._fit_completion_budget(messages)

        assert budget < 4000, "the 4,000-token reservation is what caused the 413"
        assert estimate_tokens(messages) + budget <= 8000, "the request must fit the ceiling"
        assert budget >= MIN_COMPLETION_TOKENS

    def test_the_trimmed_budget_still_covers_what_the_model_uses(self):
        """Measured: the model generates ~1,242 completion tokens for this prompt."""
        budget = _provider()._fit_completion_budget(_messages(REAL_PROMPT_CHARS))
        assert budget >= 1500, "trimming must not truncate a normal answer"

    def test_a_small_prompt_keeps_the_full_budget(self):
        provider = _provider(max_output_tokens=1000)
        assert provider._fit_completion_budget(_messages(500)) == 1000

    def test_a_provider_with_no_declared_ceiling_is_untouched(self):
        """OpenRouter publishes no single ceiling, so the guard stays off."""
        provider = _provider("openrouter", max_output_tokens=4000)
        assert provider.request_token_ceiling() == 0
        assert provider._fit_completion_budget(_messages(REAL_PROMPT_CHARS)) == 4000

    def test_an_oversized_prompt_fails_fast_instead_of_413ing(self):
        provider = _provider()
        with pytest.raises(RequestTooLargeError) as excinfo:
            provider._fit_completion_budget(_messages(int(8000 * CHARS_PER_TOKEN)))
        assert "GROQ_MAX_REQUEST_TOKENS" in str(excinfo.value)

    def test_request_too_large_is_a_provider_error_so_the_fallback_runs(self):
        assert issubclass(RequestTooLargeError, ProviderError)

    def test_the_ceiling_is_raisable_from_the_environment(self, monkeypatch):
        """A paid tier must not need a code change."""
        monkeypatch.setenv("GROQ_MAX_REQUEST_TOKENS", "128000")
        provider = _provider(max_output_tokens=4000)
        assert provider.request_token_ceiling() == 128000
        assert provider._fit_completion_budget(_messages(REAL_PROMPT_CHARS)) == 4000

    def test_a_malformed_override_falls_back_to_the_pin(self, monkeypatch):
        monkeypatch.setenv("GROQ_MAX_REQUEST_TOKENS", "not-a-number")
        assert _provider().request_token_ceiling() == 8000

    def test_the_budget_is_what_actually_reaches_the_endpoint(self, monkeypatch):
        """The clamp must land in the request kwargs, not just be computed."""
        sent = {}

        class _Client:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kwargs):
                        sent.update(kwargs)
                        raise RuntimeError("stop here — the kwargs are the assertion")

        provider = _provider(max_output_tokens=4000)
        monkeypatch.setattr(type(provider), "client", property(lambda self: _Client))
        with pytest.raises(ProviderError):
            provider.complete(_messages(REAL_PROMPT_CHARS))
        assert sent["max_completion_tokens"] < 4000
        assert estimate_tokens(_messages(REAL_PROMPT_CHARS)) + sent["max_completion_tokens"] <= 8000


# ---------------------------------------------------------------------------
# 2. The deadline — bounded total latency
# ---------------------------------------------------------------------------

class _Stub:
    """A provider that takes a set time and then succeeds or fails."""

    def __init__(self, name, *, seconds=0.0, fail=False):
        self.name = name
        self.model = f"{name}-model"
        self.timeout = 60.0
        self._client = None
        self.seconds = seconds
        self.fail = fail
        self.calls = 0
        self.seen_timeouts: list[float] = []

    def complete(self, messages, *, json_mode=True):
        self.calls += 1
        self.seen_timeouts.append(self.timeout)
        if self.seconds:
            time.sleep(self.seconds)
        if self.fail:
            raise ProviderError(f"{self.name} failed")
        return Completion(text="{}", provider=self.name, model=self.model)


class TestDeadline:
    def test_the_default_is_bounded(self):
        assert DEFAULT_DEADLINE > 0, "an unbounded default is the original defect"
        assert DEFAULT_DEADLINE <= 120
        assert DEFAULT_TIMEOUT <= DEFAULT_DEADLINE

    def test_a_working_primary_is_unaffected(self):
        primary, secondary = _Stub("groq"), _Stub("openrouter")
        completion = FallbackProvider(primary=primary, secondary=secondary,
                                      deadline_s=90).complete([])
        assert completion.provider == "groq"
        assert secondary.calls == 0

    def test_the_fallback_runs_when_there_is_budget_left(self):
        primary = _Stub("groq", fail=True)
        secondary = _Stub("openrouter")
        completion = FallbackProvider(primary=primary, secondary=secondary,
                                      deadline_s=90).complete([])
        assert completion.provider == "openrouter"
        assert secondary.calls == 1

    def test_the_fallback_is_skipped_when_the_budget_is_spent(self):
        """The core bound: a slow failing primary must not launch a fresh attempt."""
        primary = _Stub("groq", seconds=0.4, fail=True)
        secondary = _Stub("openrouter", seconds=30)
        provider = FallbackProvider(primary=primary, secondary=secondary, deadline_s=0.3)

        started = time.monotonic()
        with pytest.raises(DeadlineExceededError):
            provider.complete([])
        elapsed = time.monotonic() - started

        assert secondary.calls == 0, "the secondary must not start with no budget left"
        assert elapsed < 5, "and the caller must not wait for it"

    def test_the_secondary_inherits_the_remaining_budget_as_its_timeout(self):
        """A fallback given 90 s when 2 s remain would blow the bound."""
        primary = _Stub("groq", seconds=0.3, fail=True)
        secondary = _Stub("openrouter")
        secondary.timeout = 60.0
        FallbackProvider(primary=primary, secondary=secondary, deadline_s=10).complete([])

        assert secondary.seen_timeouts, "the secondary was never called"
        assert secondary.seen_timeouts[0] < 10, "the timeout must shrink to what is left"

    def test_the_secondary_timeout_is_restored_after_the_call(self):
        """A shortened timeout must not leak into the next request."""
        primary, secondary = _Stub("groq", fail=True), _Stub("openrouter")
        FallbackProvider(primary=primary, secondary=secondary, deadline_s=10).complete([])
        assert secondary.timeout == 60.0

    def test_deadline_zero_restores_unbounded_behaviour(self):
        """Explicitly opt-out only; it is not the default."""
        primary, secondary = _Stub("groq", fail=True), _Stub("openrouter")
        FallbackProvider(primary=primary, secondary=secondary, deadline_s=0).complete([])
        assert secondary.seen_timeouts[0] == 60.0

    def test_both_failing_still_raises_one_bounded_error(self):
        primary, secondary = _Stub("groq", fail=True), _Stub("openrouter", fail=True)
        with pytest.raises(ProviderError) as excinfo:
            FallbackProvider(primary=primary, secondary=secondary, deadline_s=90).complete([])
        assert "groq" in str(excinfo.value) and "openrouter" in str(excinfo.value)

    def test_sdk_retries_are_bounded(self, monkeypatch):
        """Same-provider retry cannot be the strategy: the fallback is."""
        from generation.providers import _sdk_retries

        monkeypatch.delenv("GENERATION_MAX_RETRIES", raising=False)
        assert _sdk_retries() == 1
        monkeypatch.setenv("GENERATION_MAX_RETRIES", "0")
        assert _sdk_retries() == 0
        monkeypatch.setenv("GENERATION_MAX_RETRIES", "banana")
        assert _sdk_retries() == 1

    def test_the_provider_default_retry_count_is_low(self):
        provider = _provider()
        assert provider.max_retries <= 1, (
            "3 retries x 180 s x 2 providers was the 24-minute worst case"
        )

    def test_build_provider_applies_the_retry_bound(self, monkeypatch):
        from generation.providers import build_provider

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-secondary")
        monkeypatch.delenv("GENERATION_MAX_RETRIES", raising=False)
        settings = GenerationSettings(
            provider="groq", model="m", base_url="https://x.invalid", api_key="k",
            top_k=5, score_threshold=0.75, temperature=0.0, max_output_tokens=4000,
            timeout=30.0, deadline=45.0, enable_fallback=True,
        )
        chain = build_provider(settings)
        assert chain.primary.max_retries == 1
        assert chain.secondary.max_retries == 1

    def test_min_fallback_seconds_is_sane(self):
        assert 0 < MIN_FALLBACK_SECONDS < 30


# ---------------------------------------------------------------------------
# 3. No unbounded retry ladder
# ---------------------------------------------------------------------------

class TestNoUnboundedRetries:
    def test_the_worst_case_is_bounded_by_the_deadline(self):
        """(timeout x retries x providers) must no longer be the worst case."""
        settings = GenerationSettings(
            provider="groq", model="m", base_url="https://x.invalid", api_key="k",
            top_k=5, score_threshold=0.75, temperature=0.0, max_output_tokens=4000,
            timeout=DEFAULT_TIMEOUT,
        )
        assert settings.deadline == DEFAULT_DEADLINE
        assert settings.deadline < 240, "the client's read timeout must never be reached"

    def test_the_deadline_is_reported_in_describe(self):
        settings = GenerationSettings(
            provider="groq", model="m", base_url="https://x.invalid", api_key="k",
            top_k=5, score_threshold=0.75, temperature=0.0, max_output_tokens=4000,
            timeout=30.0, deadline=45.0,
        )
        described = settings.describe()
        assert described["deadline_s"] == 45.0
        assert "k" not in str(described.get("api_key", ""))

    def test_build_provider_passes_the_deadline_through(self, monkeypatch):
        from generation.providers import build_provider

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-secondary")
        settings = GenerationSettings(
            provider="groq", model="m", base_url="https://x.invalid", api_key="k",
            top_k=5, score_threshold=0.75, temperature=0.0, max_output_tokens=4000,
            timeout=30.0, deadline=45.0, enable_fallback=True,
        )
        provider = build_provider(settings)
        assert isinstance(provider, FallbackProvider)
        assert provider.deadline_s == 45.0


# ---------------------------------------------------------------------------
# 4. The pipeline still answers, refuses and reports timings
# ---------------------------------------------------------------------------

CHUNK = (
    "Elective repair should be considered for men with an abdominal aortic aneurysm "
    "with a maximum diameter of 55 mm or larger."
)


def _hit() -> dict:
    return {
        "rank": 1, "chunk_id": "ESVS_2024__p26-26__c0093", "chunk_text": CHUNK, "text": CHUNK,
        "document": "ESVS 2024", "document_id": "ESVS_2024", "section": "Elective repair",
        "page": 26, "page_start": 26, "page_end": 26,
        "similarity_score": 0.88, "score": 0.88,
    }


class _Retriever:
    def search(self, query, top_k=5):
        return [_hit()]


def _settings(**kw) -> GenerationSettings:
    base = dict(provider="groq", model="m", base_url="https://x.invalid", api_key="k",
                top_k=5, score_threshold=0.75, temperature=0.0, max_output_tokens=4000,
                timeout=30.0)
    base.update(kw)
    return GenerationSettings(**base)


class TestPipelineTimings:
    def test_a_normal_answer_records_every_stage(self):
        import json

        from generation.pipeline import answer_question
        from generation.schema import DISCLAIMER

        answer = {
            "recommendation": "Elective repair is considered at 55 mm in men.",
            "supporting_evidence": [{"claim": "Repair at 55 mm.", "chunk_id": _hit()["chunk_id"],
                                     "excerpt": "a maximum diameter of 55 mm or larger"}],
            "citations": [{"document": "ESVS 2024", "section": "Elective repair", "page": 26,
                           "chunk_id": _hit()["chunk_id"], "retrieval_score": 0.88,
                           "excerpt": "a maximum diameter of 55 mm or larger"}],
            "confidence": "High", "disclaimer": DISCLAIMER,
        }

        class _P:
            name = model = "stub"

            def complete(self, messages, *, json_mode=True):
                return Completion(text=json.dumps(answer), provider="stub", model="stub")

        result = answer_question("At what diameter is elective repair recommended in men?",
                                 retriever=_Retriever(), provider=_P(), settings=_settings())
        assert not result.refused
        for stage in ("safety", "retrieval", "grounding", "generation", "validation", "total"):
            assert stage in result.timings_ms, f"{stage} is not timed"
        assert result.timings_ms["total"] >= 0
        assert "timings_ms" in result.to_dict()

    def test_timings_carry_no_question_or_prompt_text(self):
        """The whole dict is logged, so it must be durations and nothing else."""
        import json

        from generation.pipeline import answer_question
        from generation.schema import DISCLAIMER

        class _P:
            name = model = "stub"

            def complete(self, messages, *, json_mode=True):
                return Completion(text=json.dumps({
                    "recommendation": "x", "supporting_evidence": [], "citations": [],
                    "confidence": "Insufficient Evidence", "disclaimer": DISCLAIMER,
                }), provider="stub", model="stub")

        result = answer_question("a secret-sounding clinical question",
                                 retriever=_Retriever(), provider=_P(), settings=_settings())
        assert all(isinstance(v, (int, float)) for v in result.timings_ms.values())

    def test_a_refusal_is_timed_and_never_calls_the_model(self):
        from generation.pipeline import answer_question

        class _Explode:
            name = model = "stub"

            def complete(self, messages, *, json_mode=True):  # pragma: no cover
                raise AssertionError("the safety gate must refuse before generation")

        result = answer_question("I am a 67-year-old man with a 5.2 cm AAA. Should I have surgery now?",
                                 retriever=_Retriever(), provider=_Explode(), settings=_settings())
        assert result.refused
        assert result.refusal["gate"].startswith("safety:")
        assert "total" in result.timings_ms
        assert "generation" not in result.timings_ms

    def test_a_provider_failure_surfaces_bounded_and_safe(self):
        from generation.pipeline import answer_question

        class _Dead:
            name = model = "stub"

            def complete(self, messages, *, json_mode=True):
                raise ProviderError("upstream exploded with org_secret and a key")

        with pytest.raises(ProviderError):
            answer_question("At what diameter is elective repair recommended in men?",
                            retriever=_Retriever(), provider=_Dead(), settings=_settings())


# ---------------------------------------------------------------------------
# 5. Retrieval is timed in two halves, and timing it changes nothing
# ---------------------------------------------------------------------------

class _SplitRetriever:
    """A retriever exposing the two halves, like the real QdrantRetriever."""

    def __init__(self):
        self.embed_calls = 0
        self.search_calls = 0
        self.combined_calls = 0

    def embed_query(self, query):
        self.embed_calls += 1
        return [0.0] * 768

    def search_vector(self, vector, top_k=5):
        self.search_calls += 1
        return [_hit()]

    def search(self, query, top_k=5):  # pragma: no cover - must not be reached
        self.combined_calls += 1
        return [_hit()]


def _stub_provider():
    import json

    from generation.schema import DISCLAIMER

    answer = {
        "recommendation": "Elective repair is considered at 55 mm in men.",
        "supporting_evidence": [], "citations": [],
        "confidence": "Insufficient Evidence", "disclaimer": DISCLAIMER,
    }

    class _P:
        name = model = "stub"

        def complete(self, messages, *, json_mode=True):
            return Completion(text=json.dumps(answer), provider="stub", model="stub")

    return _P()


class TestRetrievalTimingSplit:
    def test_embedding_and_qdrant_are_timed_separately(self):
        """One combined number could not say whether 28 s was the encoder or the store."""
        from generation.pipeline import answer_question

        retriever = _SplitRetriever()
        result = answer_question("At what diameter is elective repair recommended in men?",
                                 retriever=retriever, provider=_stub_provider(),
                                 settings=_settings())
        assert "embedding" in result.timings_ms
        assert "qdrant" in result.timings_ms
        assert retriever.embed_calls == 1 and retriever.search_calls == 1

    def test_the_split_uses_the_same_calls_search_would(self):
        """`search()` IS `search_vector(embed_query(q))`, so nothing changes."""
        from generation.pipeline import answer_question

        retriever = _SplitRetriever()
        answer_question("At what diameter is elective repair recommended in men?",
                        retriever=retriever, provider=_stub_provider(), settings=_settings())
        assert retriever.combined_calls == 0

    def test_a_retriever_without_the_split_still_works(self):
        """Injected fakes that only implement `search` must keep working."""
        from generation.pipeline import answer_question

        result = answer_question("At what diameter is elective repair recommended in men?",
                                 retriever=_Retriever(), provider=_stub_provider(),
                                 settings=_settings())
        assert "retrieval" in result.timings_ms
        assert "embedding" not in result.timings_ms


class TestEncoderThreadPinning:
    def test_the_api_pins_threads_before_loading_the_model(self):
        """Oversubscribed intra-op threads were 28 s of every production request."""
        import api.main as main

        assert hasattr(main, "_configure_torch_threads")
        source = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
        lifespan = source[source.index("async def lifespan"):]
        pin = lifespan.index("_configure_torch_threads()")
        build = lifespan.index("QdrantRetriever()")
        assert pin < build, "threads must be pinned before the encoder loads"

    def test_the_thread_count_is_configurable(self, monkeypatch):
        import api.main as main

        monkeypatch.setenv("TORCH_NUM_THREADS", "4")
        main._configure_torch_threads()
        import torch

        assert torch.get_num_threads() == 4
        monkeypatch.setenv("TORCH_NUM_THREADS", "1")
        main._configure_torch_threads()
        assert torch.get_num_threads() == 1

    def test_a_malformed_thread_count_falls_back_to_one(self, monkeypatch):
        import api.main as main
        import torch

        monkeypatch.setenv("TORCH_NUM_THREADS", "banana")
        main._configure_torch_threads()
        assert torch.get_num_threads() == 1

    def test_the_frozen_ingestion_path_is_untouched(self):
        """Index-building embeddings must not be affected by a serving setting."""
        chunking = (ROOT / "ingestion" / "chunking.py").read_text(encoding="utf-8")
        assert "set_num_threads" not in chunking

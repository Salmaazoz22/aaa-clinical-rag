# -*- coding: utf-8 -*-
"""Configuration for the generation layer.

Everything is environment-driven, exactly as `vectordb/config.py` is: no API key
is hardcoded, defaulted to a literal, logged, or written to disk. `.env.example`
documents the variable names and nothing else.

Switching model provider is a one-line change in `.env`:

    GENERATION_PROVIDER=groq          # fast iteration during development
    GENERATION_PROVIDER=openrouter    # final / evaluated answers

The provider registry lives in `generation/providers.py`; this module only
resolves the settings that select and parameterise one.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# The dotenv reader is imported rather than reimplemented. A second copy of
# "read KEY=value, never override a real environment variable" is exactly the
# kind of duplication that drifted in this repo before (two chunker copies, see
# docs/HANDOFF.md), so the generation layer reuses the one that already exists.
from vectordb.config import _load_dotenv as load_dotenv_into_environ

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROVIDER = "groq"

# Retrieval depth for generation. The frozen retriever's own default is top-10;
# the generation layer asks for 5, per the Day 3 specification. This selects how
# many chunks are *requested*, and changes no retrieval semantics whatsoever.
DEFAULT_TOP_K = 5

# --- the evidence-quality floor -------------------------------------------
#
# Cosine similarity below which a retrieved chunk is not usable evidence and is
# not sent to the model. If no retrieved chunk clears it, the layer refuses
# instead of answering -- which is the same condition as "top-1 is below the
# floor", since hits arrive score-ordered.
#
# 0.75 is a STARTING VALUE, chosen as a reasonable default for a corpus whose
# frozen retrieval sits at P@1 0.55 / Recall@10 0.78. It is NOT derived from a
# calibration study, and it is expected to be tuned once the generation
# evaluation has been run a few times. Two consequences are known up front and
# are the intended behaviour of a deliberately conservative floor:
#
#   * it refuses rather than answers on weak retrievals. In the frozen evidence
#     (`eval/final_evidence.json`) the weakest top-1 among the 20 final
#     questions is Q4 ("indications for endovascular aneurysm repair") at
#     0.7467 -- below this floor, so Q4 refuses. Q4 is also the one question
#     with no relevant chunk anywhere in its top-10
#     (`eval/final_evaluation_summary.json`), so refusing it is the correct
#     outcome, reached by a threshold rather than by knowing which question it is;
#   * it trims weak tail chunks out of the context even when the question is
#     answerable, so an answer is usually grounded in fewer than `top_k` chunks.
#
# Override per-run with GENERATION_SCORE_THRESHOLD. See docs/generation.md.
DEFAULT_SCORE_THRESHOLD = 0.75

# Deterministic decoding: the generation evaluation has to be re-runnable and a
# citation that moves between runs is not auditable. Note that neither
# gpt-oss-120b nor DeepSeek-R1 is bit-deterministic even at temperature 0.
DEFAULT_TEMPERATURE = 0.0

# Generous because both supported models are reasoning models: their reasoning
# tokens are billed against the same output budget as the JSON answer, so a
# limit tuned for the answer alone truncates the JSON mid-object.
DEFAULT_MAX_OUTPUT_TOKENS = 4000
DEFAULT_TIMEOUT = 180.0

# Evidence text is sent to the model in full. The reference implementation
# truncates each document to 1,000 characters before prompting; that would
# silently cut the tail off any chunk longer than the limit, and this layer then
# asks the model to quote verbatim excerpts and asserts they appear in the
# cited chunk. Truncated evidence would make those excerpts unverifiable, so no
# truncation happens here. Indexed chunks are <= 254 tokens by construction
# (`data/embeddings/index_meta.json`), so the whole corpus fits comfortably.
MAX_CHUNK_CHARS = 0  # 0 = no truncation


def _as_float(value: str | None, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    return float(value)


def _as_int(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    return int(value)


@dataclass(frozen=True)
class GenerationSettings:
    """Resolved generation settings. `api_key` is never logged or serialised."""

    provider: str
    model: str
    base_url: str
    api_key: str | None
    top_k: int
    score_threshold: float
    temperature: float
    max_output_tokens: int
    timeout: float

    def describe(self) -> dict[str, object]:
        """Description safe to print or persist. Carries no secret material."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_supplied": bool(self.api_key),
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "timeout_s": self.timeout,
        }


def load_settings(env_file: Path | None = None, provider: str | None = None) -> GenerationSettings:
    """Resolve settings from the environment (and a local, git-ignored `.env`).

    `provider` overrides `GENERATION_PROVIDER` for a single call, which is what
    the evaluation runner's `--provider` flag uses. Everything else comes from
    the environment.
    """
    # Imported here to keep the module import graph acyclic: providers.py reads
    # this module's defaults for its own registry.
    from generation.providers import resolve_provider_spec

    load_dotenv_into_environ(env_file or (ROOT / ".env"))

    name = (provider or os.environ.get("GENERATION_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    spec = resolve_provider_spec(name)

    # A model override is allowed but the provider's pinned model is the default,
    # so an unset variable can never silently change which model was evaluated.
    model = os.environ.get("GENERATION_MODEL") or spec.model

    return GenerationSettings(
        provider=spec.name,
        model=model,
        base_url=os.environ.get("GENERATION_BASE_URL") or spec.base_url,
        api_key=os.environ.get(spec.key_env) or None,
        top_k=_as_int(os.environ.get("GENERATION_TOP_K"), DEFAULT_TOP_K),
        score_threshold=_as_float(os.environ.get("GENERATION_SCORE_THRESHOLD"), DEFAULT_SCORE_THRESHOLD),
        temperature=_as_float(os.environ.get("GENERATION_TEMPERATURE"), DEFAULT_TEMPERATURE),
        max_output_tokens=_as_int(os.environ.get("GENERATION_MAX_OUTPUT_TOKENS"), DEFAULT_MAX_OUTPUT_TOKENS),
        timeout=_as_float(os.environ.get("GENERATION_TIMEOUT"), DEFAULT_TIMEOUT),
    )

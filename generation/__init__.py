# -*- coding: utf-8 -*-
"""Grounded generation and citation layer for the AAA clinical RAG index.

This package sits *on top of* the frozen retrieval system (commit `7cbf6d9`) and
changes nothing in it. It does not chunk, embed, re-rank, rewrite queries, or
touch the Qdrant collection; it calls `vectordb.retriever` and works with what
comes back.

    question
      |  generation.safety.screen_query          patient-specific? refuse locally
      v
    vectordb.retriever.QdrantRetriever.search    frozen: dense cosine, top-K
      |  generation.pipeline.select_usable_hits  evidence floor (similarity)
      v
    generation.prompts.build_messages            system prompt + CONTEXT + schema
      |  generation.providers                    Groq | OpenRouter (one interface)
      v
    generation.parsing.parse_answer              locate the JSON object
      |  generation.validator.validate_answer    every citation checked against
      v                                          the chunks actually sent
    structured, cited, possibly-refusing answer
"""
from __future__ import annotations

from generation.config import (
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    GenerationSettings,
    load_settings,
)
from generation.parsing import AnswerParseError, parse_answer
from generation.pipeline import GenerationResult, answer_question, select_usable_hits
from generation.prompts import SYSTEM_PROMPT, build_messages, build_user_prompt
from generation.providers import (
    Completion,
    FallbackProvider,
    LLMProvider,
    MissingAPIKeyError,
    ProviderError,
    ProviderSpec,
    UnknownProviderError,
    build_provider,
)
from generation.refusal import build_refusal
from generation.safety import SafetyVerdict, screen_query
from generation.schema import (
    CONFIDENCE_VALUES,
    DISCLAIMER,
    REFUSAL_MESSAGE,
    REFUSAL_REASONS,
    answer_json_schema,
    is_refusal,
)
from generation.emergency import EmergencyVerdict, screen_emergency
from generation.validator import (
    Finding,
    ValidationReport,
    answer_prose,
    conflict_position_count,
    documents_cited,
    resolve_citations,
    summarize_evidence_grade,
    validate_answer,
)

__all__ = [
    "AnswerParseError",
    "CONFIDENCE_VALUES",
    "Completion",
    "DEFAULT_SCORE_THRESHOLD",
    "DEFAULT_TOP_K",
    "DISCLAIMER",
    "FallbackProvider",
    "Finding",
    "GenerationResult",
    "GenerationSettings",
    "LLMProvider",
    "MissingAPIKeyError",
    "ProviderError",
    "ProviderSpec",
    "REFUSAL_MESSAGE",
    "REFUSAL_REASONS",
    "SYSTEM_PROMPT",
    "SafetyVerdict",
    "UnknownProviderError",
    "ValidationReport",
    "answer_json_schema",
    "answer_prose",
    "answer_question",
    "build_messages",
    "build_provider",
    "build_refusal",
    "build_user_prompt",
    "conflict_position_count",
    "documents_cited",
    "is_refusal",
    "load_settings",
    "parse_answer",
    "resolve_citations",
    "screen_query",
    "select_usable_hits",
    "summarize_evidence_grade",
    "validate_answer",
    # emergency gate (A1)
    "EmergencyVerdict",
    "screen_emergency",
]

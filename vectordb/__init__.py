# -*- coding: utf-8 -*-
"""Production vector-database layer (Qdrant) for the AAA clinical RAG index.

This package is *infrastructure only*. It moves the already-validated V1 index
(991 atomic chunks, MedEmbed-base-v0.1 @ 7a90c502, 768-d, L2-normalised) into
Qdrant and serves dense cosine top-K retrieval from it.

Nothing here re-chunks, re-embeds, re-ranks, rewrites queries, or touches the
frozen evaluation. The local numpy index in `data/embeddings/` remains the
reproducibility artifact; Qdrant is the production backend.
"""
from __future__ import annotations

from .config import QdrantSettings, load_settings
from .schema import (
    COLLECTION_DISTANCE,
    EXPECTED_DIM,
    EXPECTED_MODEL,
    EXPECTED_REVISION,
    PAYLOAD_FIELDS,
    POINT_ID_NAMESPACE,
    build_payload,
    is_nan,
    nan_summary,
    point_id_for,
    validate_index_bundle,
)

__all__ = [
    "QdrantSettings",
    "load_settings",
    "COLLECTION_DISTANCE",
    "EXPECTED_DIM",
    "EXPECTED_MODEL",
    "EXPECTED_REVISION",
    "PAYLOAD_FIELDS",
    "POINT_ID_NAMESPACE",
    "build_payload",
    "point_id_for",
    "validate_index_bundle",
]

# -*- coding: utf-8 -*-
"""Environment-driven Qdrant configuration.

No credential is ever hardcoded or written to disk by this project. Everything
comes from the environment (or a local, git-ignored `.env`); `.env.example`
documents the variable names and nothing else.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "aaa_clinical_v1"
DEFAULT_TIMEOUT = 30.0


def _load_dotenv(path: Path) -> None:
    """Fill missing environment variables from a local .env file.

    Deliberately minimal and dependency-free: `KEY=value` lines, `#` comments,
    optional surrounding quotes. Existing environment variables always win, so
    a shell export can never be silently overridden by a stale file.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class QdrantSettings:
    """Resolved connection settings. `api_key` is never logged or serialised."""

    url: str
    api_key: str | None
    collection: str
    prefer_grpc: bool
    timeout: float
    exact_search: bool
    local_path: str | None

    @property
    def mode(self) -> str:
        return "local-mode (embedded)" if self.local_path else "server"

    def describe(self) -> dict[str, object]:
        """Connection description safe to print or persist (no secret material)."""
        return {
            "mode": self.mode,
            "url": None if self.local_path else self.url,
            "local_path": self.local_path,
            "collection": self.collection,
            "prefer_grpc": self.prefer_grpc,
            "timeout_s": self.timeout,
            "exact_search": self.exact_search,
            "api_key_supplied": bool(self.api_key),
        }


def load_settings(env_file: Path | None = None) -> QdrantSettings:
    _load_dotenv(env_file or (ROOT / ".env"))
    local_path = os.environ.get("QDRANT_LOCAL_PATH") or None
    return QdrantSettings(
        url=os.environ.get("QDRANT_URL") or DEFAULT_URL,
        api_key=os.environ.get("QDRANT_API_KEY") or None,
        collection=os.environ.get("QDRANT_COLLECTION") or DEFAULT_COLLECTION,
        prefer_grpc=_as_bool(os.environ.get("QDRANT_PREFER_GRPC"), False),
        timeout=float(os.environ.get("QDRANT_TIMEOUT") or DEFAULT_TIMEOUT),
        # 991 vectors is far below Qdrant's default indexing threshold, so search
        # is already exhaustive; forcing `exact` keeps that true if the corpus
        # ever grows, because the validated retriever is exhaustive cosine.
        exact_search=_as_bool(os.environ.get("QDRANT_EXACT_SEARCH"), True),
        local_path=local_path,
    )


def make_client(settings: QdrantSettings | None = None):
    """Build a QdrantClient for the resolved settings.

    `QDRANT_LOCAL_PATH` selects the client's embedded local mode, which needs no
    server. It is a convenience for offline development and tests; the server /
    Qdrant Cloud path is the production configuration.
    """
    from qdrant_client import QdrantClient

    settings = settings or load_settings()
    if settings.local_path:
        return QdrantClient(path=settings.local_path)
    return QdrantClient(
        url=settings.url,
        api_key=settings.api_key,
        prefer_grpc=settings.prefer_grpc,
        timeout=settings.timeout,
    )

# -*- coding: utf-8 -*-
"""How the frontend reads its own configuration, in one place.

The UI needs two kinds of setting: where the backend lives (`ui/api_client.py`)
and the credential for speech-to-text (`ui/transcription.py`). Both are resolved
the same way, so both resolve it here.

Three rules, each of which exists because breaking it caused a production
outage in this project:

**Secrets first, then the environment.** Streamlit Community Cloud has no
environment-variable panel; `st.secrets` is the only channel a deployed app has.
A resolver that reads `os.environ` alone cannot be configured there at all.

**A template is not configuration.** `.env.example` is tracked, so it ships to
every deployment; `.env` is git-ignored, so it ships to none. Reading the
template as if it were config is how the deployed app came to authenticate with
the literal string `your_groq_api_key_here` and get back
`401 Invalid API Key` — see `is_placeholder`. Only `.env` is ever read here.

**The UI cannot borrow the pipeline's dotenv reader.** `vectordb.config` has
one, and `generation/config.py` reuses it precisely so there are not two. But
`tests/test_ui.py::test_ui_never_imports_the_rag_pipeline` forbids `ui/` from
importing `vectordb`, so the frontend needs its own — one, here, not one per
module.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

#: The only dotenv file that is ever read. `.env.example` is deliberately NOT in
#: this list: it is a template of variable NAMES, it is committed, and its values
#: are placeholders. See the module docstring.
ENV_FILE = ROOT / ".env"

#: Substrings that mark a value as a template placeholder rather than a
#: credential. Sending one of these to a provider produces a 401 that looks like
#: a revoked key rather than an unconfigured app, which is exactly how long the
#: original defect took to diagnose. No real API key contains any of them —
#: provider keys are base64url-ish tokens (`gsk_...`, `sk-...`).
_PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "your_", "your-", "yourkey", "changeme", "change-me", "replace",
    "placeholder", "paste", "todo", "dummy", "xxxx",
    "<", ">", "_here", "-here",
)


def is_placeholder(value: str | None) -> bool:
    """True when `value` is a template stand-in rather than a real setting.

    Blank counts: a variable present but empty is somebody having meant to set
    it, and must not be treated as configured.
    """
    if value is None:
        return True
    text = value.strip()
    if not text:
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


def secret(name: str) -> str:
    """One Streamlit secret, or "" — tolerating the common case of having none.

    `st.secrets` raises when no secrets file exists, which is the normal state
    for a local `streamlit run`. That is not an error: it means the setting is
    not being supplied that way.
    """
    try:
        value = st.secrets.get(name)
    except Exception:  # noqa: BLE001 - no secrets file, or an unreadable one
        return ""
    return str(value).strip() if value else ""


def load_env_file(path: Path | None = None) -> None:
    """Read `key=value` lines from `.env` into the environment.

    Never overrides a variable that is already set, so a real deployment
    environment always wins over a developer's local file. Silent on a missing
    or malformed file: this is a convenience for local runs, not a config source
    a deployment depends on.
    """
    path = ENV_FILE if path is None else path
    try:
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        return


def setting(*names: str, allow_placeholder: bool = False) -> str:
    """The first configured value among `names`, or "".

    Each name is tried in Streamlit secrets, then the environment, then `.env`;
    the first name to produce a usable value wins, so callers express priority
    by argument order. Placeholder values are skipped rather than returned,
    which is what stops a committed template from impersonating a credential.
    """
    load_env_file()
    for name in names:
        for candidate in (secret(name), os.environ.get(name) or ""):
            candidate = candidate.strip()
            if not candidate:
                continue
            if not allow_placeholder and is_placeholder(candidate):
                continue
            return candidate
    return ""


def describe_presence(value: str | None) -> str:
    """A log-safe description of a credential. Never the value itself."""
    return "present" if value and not is_placeholder(value) else "missing"

# -*- coding: utf-8 -*-
"""Speech-to-text for the voice question composer.

Turns a recording made by `st.audio_input` into a string the user can read,
edit, and then submit as an ordinary clinical question. It is a *composer aid*:
nothing here reaches the RAG pipeline directly, and a transcript is always shown
for verification before it can be asked (`ui/views/ask.py`).

Provider: Whisper over an OpenAI-compatible endpoint, the same shape the
generation layer uses for its own providers.

    OPENAI_API_KEY   -> api.openai.com        whisper-1
    GROQ_API_KEY     -> api.groq.com/openai/v1 whisper-large-v3

The credential is resolved by `ui/settings.py`, which means Streamlit secrets
first and `.env` last. That module also refuses template placeholders, which is
the defect this file used to have: it loaded `.env.example` — tracked, therefore
present in every deployment, and carrying `GROQ_API_KEY=your_groq_api_key_here`
— into the environment and authenticated with it, producing a
`401 Invalid API Key` in production that read like a revoked key rather than an
unconfigured app.
"""
from __future__ import annotations

import io

from ui.settings import describe_presence, setting

#: Credential names accepted, in priority order. OpenAI first: if a deployment
#: has both, the dedicated speech vendor is the better default.
SPEECH_KEY_SETTINGS: tuple[str, ...] = ("OPENAI_API_KEY", "GROQ_API_KEY")

#: Endpoint and model per credential. Groq is not the OpenAI default endpoint,
#: so it needs an explicit base_url and its own model name.
_PROVIDERS: dict[str, tuple[str | None, str, str]] = {
    # setting name    -> (base_url, model, human name)
    "OPENAI_API_KEY": (None, "whisper-1", "OpenAI Whisper"),
    "GROQ_API_KEY": ("https://api.groq.com/openai/v1", "whisper-large-v3", "Groq Whisper"),
}

#: Shown when transcription cannot run. Deliberately free of provider internals:
#: an HTTP status and a vendor error body are operator diagnostics, and pasting
#: them onto a clinical screen tells the reader nothing they can act on.
UNAVAILABLE_MESSAGE = (
    "Voice transcription is temporarily unavailable. Please check the voice service "
    "configuration, or type your question instead — text input is unaffected."
)

NOT_CONFIGURED_MESSAGE = (
    "Voice transcription is not configured for this deployment. Add "
    "OPENAI_API_KEY or GROQ_API_KEY to the app's secrets to enable it, or type "
    "your question instead — text input is unaffected."
)


def resolve_provider() -> tuple[str | None, str | None, str | None, str | None]:
    """Which speech provider is configured: (setting_name, key, base_url, model).

    All four are None when nothing usable is configured. The key is returned for
    the caller to use, never to display — `describe_configuration()` is the only
    thing meant for a screen or a log.
    """
    for name in SPEECH_KEY_SETTINGS:
        key = setting(name)
        if key:
            base_url, model, _label = _PROVIDERS[name]
            return name, key, base_url, model
    return None, None, None, None


def is_configured() -> bool:
    """True when a usable speech credential is available. Reads no audio."""
    return resolve_provider()[1] is not None


def describe_configuration() -> dict[str, str]:
    """A log-safe summary of the speech configuration. Carries no key material."""
    name, key, _base_url, model = resolve_provider()
    if not name:
        return {
            "provider": "none",
            "setting": "OPENAI_API_KEY or GROQ_API_KEY",
            "api_key": "missing",
            "model": "-",
        }
    return {
        "provider": _PROVIDERS[name][2],
        "setting": name,
        "api_key": describe_presence(key),
        "model": model or "-",
    }


def transcribe_audio_bytes(
    audio_bytes: bytes, filename: str = "question.wav"
) -> tuple[str, str | None]:
    """Transcribe recorded audio to text.

    Returns `(text, error_message)` — exactly one of the two is meaningful. The
    error is written for the person at the screen; the provider's own response is
    never part of it.
    """
    if not audio_bytes:
        return "", "No audio was recorded. Record a question first, then convert it."

    _setting_name, api_key, base_url, model = resolve_provider()
    if not api_key:
        return "", NOT_CONFIGURED_MESSAGE

    try:
        import openai
    except ImportError:
        return "", UNAVAILABLE_MESSAGE

    buffer = io.BytesIO(audio_bytes)
    # The SDK infers the upload's content type from the filename, so a recording
    # with no name would be rejected before it reached the model.
    buffer.name = filename or "question.wav"

    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        transcript = client.audio.transcriptions.create(
            model=model,
            file=buffer,
            response_format="text",
        )
    except Exception as exc:  # noqa: BLE001 - every provider failure lands here
        # Distinguish "nobody configured this" from "it broke", because only the
        # first one is actionable by the reader, and say neither with the
        # vendor's words. `openai.AuthenticationError` is not imported by name:
        # a Groq 401 arrives through the same SDK but the class is not
        # guaranteed across versions, so the status code is checked instead.
        status = getattr(exc, "status_code", None)
        if status in (401, 403):
            return "", NOT_CONFIGURED_MESSAGE
        return "", UNAVAILABLE_MESSAGE

    text = ""
    if isinstance(transcript, str):
        text = transcript.strip()
    elif hasattr(transcript, "text"):
        text = str(transcript.text).strip()

    if not text:
        return "", (
            "Nothing could be heard in that recording. Check the microphone and "
            "record again, or type your question instead."
        )
    return text, None

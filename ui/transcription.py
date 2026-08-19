# -*- coding: utf-8 -*-
"""Audio Transcription / Speech-to-Text Module for Clinical Questions.

Converts recorded user voice input (WAV/MP3/M4A) into text using Whisper API
(via OpenAI/Groq) or speech recognition fallback.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> None:
    """Read key=value pairs into os.environ without overriding existing vars."""
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


def transcribe_audio_bytes(
    audio_bytes: bytes, filename: str = "question.wav"
) -> tuple[str, str | None]:
    """Transcribe audio bytes to text string.

    Args:
        audio_bytes: Raw bytes of the recorded audio file.
        filename: Optional filename hint (e.g. question.wav).

    Returns:
        Tuple of (transcribed_text, error_message)
    """
    if not audio_bytes or len(audio_bytes) == 0:
        return "", "No audio data recorded."

    # Load .env and .env.example without importing pipeline packages
    _load_env_file(ROOT / ".env")
    _load_env_file(ROOT / ".env.example")

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
    base_url = None
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("GROQ_API_KEY"):
        base_url = "https://api.groq.com/openai/v1"

    # Strategy 1: Whisper API (Groq / OpenAI)
    if api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            buffer = io.BytesIO(audio_bytes)
            buffer.name = filename
            
            transcript = client.audio.transcriptions.create(
                model="whisper-1" if not base_url else "whisper-large-v3",
                file=buffer,
                response_format="text",
            )
            txt = ""
            if isinstance(transcript, str):
                txt = transcript.strip()
            elif hasattr(transcript, "text"):
                txt = str(transcript.text).strip()
            
            if txt:
                return txt, None
        except Exception as e:
            return "", f"Whisper API error: {str(e)}"

    # Strategy 2: Speech Recognition fallback
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
            txt = recognizer.recognize_google(audio_data)
            if txt:
                return str(txt).strip(), None
    except Exception:
        pass

    if not api_key:
        return (
            "",
            "API key required for high-accuracy Whisper transcription. Set GROQ_API_KEY or OPENAI_API_KEY in .env file.",
        )

    return "", "Could not transcribe audio. Please verify your recording and microphone."

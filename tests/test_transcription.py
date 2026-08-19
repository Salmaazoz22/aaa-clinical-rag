# -*- coding: utf-8 -*-
"""Tests for audio transcription / speech-to-text helper module."""
from ui.transcription import transcribe_audio_bytes

def test_transcribe_empty_bytes_returns_empty_tuple():
    text, err = transcribe_audio_bytes(b"")
    assert text == ""
    assert err is not None

def test_transcribe_handles_invalid_audio_gracefully():
    # Invalid audio data should fail gracefully without crashing
    text, err = transcribe_audio_bytes(b"invalid audio header data 12345")
    assert isinstance(text, str)
    assert isinstance(err, (str, type(None)))

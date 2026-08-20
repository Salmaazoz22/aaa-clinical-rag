# -*- coding: utf-8 -*-
"""Speech-to-text: configuration resolution, failure behaviour, and isolation.

The production defect this file pins down. `ui/transcription.py` used to load
BOTH `.env` and `.env.example` into `os.environ` before reading its credential.
`.env` is git-ignored, so it is absent from every deployment; `.env.example` is
tracked, so it is present in every deployment — and it carries

    GROQ_API_KEY=your_groq_api_key_here

So the deployed app read a committed placeholder, authenticated with it, and the
provider answered `401 Invalid API Key`. The symptom read like a revoked key
rather than an unconfigured app, which is what made it slow to diagnose.

Nothing here uses a real credential or touches the network: the provider call is
substituted, and the "configured" cases use an obviously fake key that the
placeholder filter must nevertheless accept as real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui import settings, transcription  # noqa: E402
from ui.transcription import (  # noqa: E402
    NOT_CONFIGURED_MESSAGE,
    SPEECH_KEY_SETTINGS,
    UNAVAILABLE_MESSAGE,
    describe_configuration,
    is_configured,
    resolve_provider,
    transcribe_audio_bytes,
)

#: Shaped like a Groq key, and deliberately NOT a real one.
FAKE_KEY = "gsk_0000000000000000000000000000000000000000000000000a"
AUDIO = b"RIFF....WAVEfmt fake audio payload"


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch):
    """Neither the developer's .env nor their environment may leak into a test."""
    for name in SPEECH_KEY_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(settings, "secret", lambda name: "")
    monkeypatch.setattr(settings, "load_env_file", lambda path=None: None)
    yield


def _configure(monkeypatch, **values: str) -> None:
    """Supply credentials the way Streamlit Cloud does: as secrets."""
    monkeypatch.setattr(settings, "secret", lambda name: values.get(name, ""))


# ---------------------------------------------------------------------------
# 1. The root cause: a committed template must never become a credential
# ---------------------------------------------------------------------------

class TestTemplateIsNotConfiguration:
    def test_the_committed_placeholder_is_rejected(self):
        """The literal value in .env.example, which produced the 401."""
        assert settings.is_placeholder("your_groq_api_key_here")

    @pytest.mark.parametrize(
        "value",
        ["", "   ", None, "your-api-key", "<paste-your-key-here>", "changeme",
         "PLACEHOLDER", "todo", "xxxxxxxx"],
    )
    def test_template_shaped_values_are_rejected(self, value):
        assert settings.is_placeholder(value)

    @pytest.mark.parametrize("value", [FAKE_KEY, "sk-proj-AbCd1234EfGh5678IjKl"])
    def test_real_shaped_keys_are_accepted(self, value):
        """The filter must not reject a genuine credential."""
        assert not settings.is_placeholder(value)

    @pytest.mark.parametrize(
        "template", [".env.example", ".streamlit/secrets.toml.example"]
    )
    def test_every_shipped_template_value_is_detected_as_a_placeholder(self, template):
        """The templates are committed, so their values reach every deployment.

        A template value the filter does NOT recognise would be sent to the
        provider verbatim the moment someone copied the file without editing it
        — which is precisely the original defect, reintroduced. So each shipped
        template is checked against the filter that has to catch it.
        """
        path = ROOT / template
        credentials = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if not key.strip().endswith("API_KEY"):
                continue
            credentials.append((key.strip(), value.strip().strip('"').strip("'")))

        assert credentials, f"{template} declares no API key to check"
        for key, value in credentials:
            assert settings.is_placeholder(value), (
                f"{template}: {key} holds {value!r}, which the placeholder filter does "
                f"NOT catch. Either it is a real credential (remove it immediately) or "
                f"the filter needs to recognise it."
            )

    def test_env_example_is_never_read(self):
        """Only `.env` is a configuration source. `.env.example` is a template."""
        assert settings.ENV_FILE.name == ".env"
        source = (ROOT / "ui" / "settings.py").read_text(encoding="utf-8")
        loader = source[source.index("def load_env_file"):source.index("def setting")]
        assert ".env.example" not in loader

    def test_transcription_does_not_load_the_template(self):
        source = (ROOT / "ui" / "transcription.py").read_text(encoding="utf-8")
        code = "\n".join(
            line for line in source.splitlines()
            if not line.strip().startswith("#")
        )
        assert "_load_env_file" not in code
        assert ".env.example" not in code.split('"""')[-1]

    def test_a_placeholder_in_the_environment_does_not_configure_voice(self, monkeypatch):
        """End to end: the exact production condition must report unconfigured."""
        monkeypatch.setenv("GROQ_API_KEY", "your_groq_api_key_here")
        assert is_configured() is False
        text, error = transcribe_audio_bytes(AUDIO)
        assert text == ""
        assert error == NOT_CONFIGURED_MESSAGE


# ---------------------------------------------------------------------------
# 2. Provider resolution
# ---------------------------------------------------------------------------

class TestProviderResolution:
    def test_groq_secret_selects_groq_whisper(self, monkeypatch):
        _configure(monkeypatch, GROQ_API_KEY=FAKE_KEY)
        name, key, base_url, model = resolve_provider()
        assert name == "GROQ_API_KEY"
        assert key == FAKE_KEY
        assert base_url == "https://api.groq.com/openai/v1"
        assert model == "whisper-large-v3"

    def test_openai_secret_selects_openai_whisper(self, monkeypatch):
        _configure(monkeypatch, OPENAI_API_KEY=FAKE_KEY)
        name, _key, base_url, model = resolve_provider()
        assert name == "OPENAI_API_KEY"
        assert base_url is None  # the SDK default endpoint
        assert model == "whisper-1"

    def test_openai_wins_when_both_are_configured(self, monkeypatch):
        _configure(monkeypatch, OPENAI_API_KEY=FAKE_KEY, GROQ_API_KEY=FAKE_KEY)
        assert resolve_provider()[0] == "OPENAI_API_KEY"

    def test_a_secret_is_read_when_the_environment_is_empty(self, monkeypatch):
        """Streamlit Cloud has no env-var panel; secrets are the only channel."""
        _configure(monkeypatch, GROQ_API_KEY=FAKE_KEY)
        assert is_configured() is True

    def test_the_environment_works_for_local_development(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
        assert is_configured() is True

    def test_nothing_configured_reports_unconfigured(self):
        assert resolve_provider() == (None, None, None, None)
        assert is_configured() is False


# ---------------------------------------------------------------------------
# 3. Nothing prints a key
# ---------------------------------------------------------------------------

class TestNoCredentialLeak:
    def test_describe_configuration_never_carries_the_key(self, monkeypatch):
        _configure(monkeypatch, GROQ_API_KEY=FAKE_KEY)
        described = describe_configuration()
        assert FAKE_KEY not in str(described)
        assert described == {
            "provider": "Groq Whisper",
            "setting": "GROQ_API_KEY",
            "api_key": "present",
            "model": "whisper-large-v3",
        }

    def test_describe_configuration_when_unconfigured(self):
        assert describe_configuration()["api_key"] == "missing"

    def test_the_user_facing_messages_carry_no_credential_shapes(self):
        for message in (NOT_CONFIGURED_MESSAGE, UNAVAILABLE_MESSAGE):
            assert "gsk_" not in message and "sk-" not in message
            # They may NAME the setting — that is the actionable part — but they
            # must never look like they are quoting one.
            assert "=" not in message

    def test_module_source_never_formats_the_key_into_a_string(self):
        """No f-string or concatenation can put the credential on a screen."""
        source = (ROOT / "ui" / "transcription.py").read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "api_key" not in stripped:
                continue
            assert "f\"" not in stripped and "f'" not in stripped, stripped
            assert "print(" not in stripped, stripped


# ---------------------------------------------------------------------------
# 4. Failure behaviour — graceful, and free of provider internals
# ---------------------------------------------------------------------------

class _FakeTranscripts:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


def _install_client(monkeypatch, transcripts) -> None:
    """Substitute the OpenAI SDK client with one that never leaves the process."""
    class _Audio:
        transcriptions = transcripts

    class _Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.audio = _Audio()

    fake_openai = type("openai", (), {"OpenAI": _Client})
    monkeypatch.setitem(sys.modules, "openai", fake_openai)


class TestFailureBehaviour:
    def test_empty_audio_is_rejected_before_any_provider_call(self):
        text, error = transcribe_audio_bytes(b"")
        assert text == ""
        assert "No audio" in error

    def test_missing_credentials_fail_gracefully(self):
        text, error = transcribe_audio_bytes(AUDIO)
        assert text == ""
        assert error == NOT_CONFIGURED_MESSAGE

    def test_a_401_reports_configuration_not_the_provider_body(self, monkeypatch):
        _configure(monkeypatch, GROQ_API_KEY=FAKE_KEY)

        class _AuthError(Exception):
            status_code = 401

        _install_client(monkeypatch, _FakeTranscripts(error=_AuthError(
            "Error code: 401 - {'error': {'message': 'Invalid API Key', "
            "'code': 'invalid_api_key'}}"
        )))
        text, error = transcribe_audio_bytes(AUDIO)
        assert text == ""
        assert error == NOT_CONFIGURED_MESSAGE
        # The exact string the deployed app used to show a clinician:
        assert "Invalid API Key" not in error
        assert "401" not in error
        assert "invalid_api_key" not in error

    def test_a_transient_failure_reports_unavailable(self, monkeypatch):
        _configure(monkeypatch, GROQ_API_KEY=FAKE_KEY)
        _install_client(monkeypatch, _FakeTranscripts(error=RuntimeError("connection reset")))
        text, error = transcribe_audio_bytes(AUDIO)
        assert text == ""
        assert error == UNAVAILABLE_MESSAGE
        assert "connection reset" not in error

    def test_invalid_audio_does_not_crash(self, monkeypatch):
        _configure(monkeypatch, GROQ_API_KEY=FAKE_KEY)
        _install_client(monkeypatch, _FakeTranscripts(error=ValueError("bad header")))
        text, error = transcribe_audio_bytes(b"invalid audio header data 12345")
        assert text == "" and error == UNAVAILABLE_MESSAGE

    def test_an_empty_transcript_is_reported_as_such(self, monkeypatch):
        _configure(monkeypatch, GROQ_API_KEY=FAKE_KEY)
        _install_client(monkeypatch, _FakeTranscripts(result="   "))
        text, error = transcribe_audio_bytes(AUDIO)
        assert text == ""
        assert "Nothing could be heard" in error


# ---------------------------------------------------------------------------
# 5. The happy path
# ---------------------------------------------------------------------------

class TestSuccessfulTranscription:
    def test_a_plain_string_response_is_returned(self, monkeypatch):
        _configure(monkeypatch, GROQ_API_KEY=FAKE_KEY)
        _install_client(monkeypatch, _FakeTranscripts(
            result="  At what diameter is elective repair recommended in men?  "))
        text, error = transcribe_audio_bytes(AUDIO)
        assert error is None
        assert text == "At what diameter is elective repair recommended in men?"

    def test_an_object_response_is_returned(self, monkeypatch):
        _configure(monkeypatch, OPENAI_API_KEY=FAKE_KEY)
        _install_client(monkeypatch, _FakeTranscripts(
            result=type("T", (), {"text": "Who should be screened for AAA?"})()))
        text, error = transcribe_audio_bytes(AUDIO)
        assert error is None and text == "Who should be screened for AAA?"

    def test_the_recording_is_sent_with_the_right_model_and_filename(self, monkeypatch):
        _configure(monkeypatch, GROQ_API_KEY=FAKE_KEY)
        transcripts = _FakeTranscripts(result="ok")
        _install_client(monkeypatch, transcripts)
        transcribe_audio_bytes(AUDIO, filename="voice.wav")
        sent = transcripts.calls[0]
        assert sent["model"] == "whisper-large-v3"
        assert sent["response_format"] == "text"
        assert sent["file"].name == "voice.wav"
        assert sent["file"].read() == AUDIO

    def test_a_nameless_recording_still_gets_a_filename(self, monkeypatch):
        """The SDK infers the upload's content type from the name."""
        _configure(monkeypatch, GROQ_API_KEY=FAKE_KEY)
        transcripts = _FakeTranscripts(result="ok")
        _install_client(monkeypatch, transcripts)
        transcribe_audio_bytes(AUDIO, filename="")
        assert transcripts.calls[0]["file"].name == "question.wav"


# ---------------------------------------------------------------------------
# 6. Text questions never reach the speech provider
# ---------------------------------------------------------------------------

class TestTextModeDoesNotUseWhisper:
    def test_is_configured_makes_no_provider_call(self, monkeypatch):
        """The composer calls this on every render, including text-only ones."""
        _configure(monkeypatch, GROQ_API_KEY=FAKE_KEY)
        transcripts = _FakeTranscripts(result="never")
        _install_client(monkeypatch, transcripts)
        assert is_configured() is True
        assert transcripts.calls == [], "checking configuration must not transcribe"

    def test_the_submit_path_never_imports_transcription(self):
        """`_submit` goes straight to the RAG client; nothing speech-related."""
        source = (ROOT / "ui" / "views" / "ask.py").read_text(encoding="utf-8")
        submit = source[source.index("def _submit("):source.index("# ---", source.index("def _submit("))]
        assert "transcribe" not in submit
        assert "api_client.answer" in submit

    def test_transcription_is_only_called_behind_the_voice_button(self):
        source = (ROOT / "ui" / "views" / "ask.py").read_text(encoding="utf-8")
        call = source.index("transcribe_audio_bytes(\n")
        window = source[:call]
        assert "st.audio_input" in window
        assert "btn_transcribe_voice" in window
        # and the guard is a button press, not a render-time side effect
        assert 'st.button("Convert Voice to Text"' in window

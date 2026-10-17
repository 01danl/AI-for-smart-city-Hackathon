"""Voice questions: recorder clip -> server speech-to-text -> chat question."""

import io
import math
import struct
import wave
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from frontend.voice_input import MAX_VOICE_SECONDS, transcribe_question, wav_seconds

APP_PATH = Path(__file__).resolve().parents[1] / "frontend" / "app.py"


def _wav(seconds: float) -> bytes:
    """16 kHz mono 16-bit WAV, like the chat input recorder produces."""
    frames = int(16_000 * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / 16_000))) for i in range(frames)
        ))
    return buf.getvalue()


def test_wav_seconds_matches_recorder_format():
    assert wav_seconds(_wav(2.0)) == pytest.approx(2.0, abs=0.01)


def test_successful_transcript_becomes_question():
    seen = []

    def fake_stt(b64: str) -> str:
        seen.append(b64)
        return "  Хочу тихое кафе\n с розеткой  "

    text, note = transcribe_question(_wav(1.5), fake_stt)
    assert text == "Хочу тихое кафе с розеткой"
    assert "Распознано" in note
    assert len(seen) == 1  # the clip is sent exactly once


@pytest.mark.parametrize("seconds, hint", [(0.2, "слишком короткая"), (MAX_VOICE_SECONDS + 1, "длиннее")])
def test_bad_length_is_rejected_without_calling_stt(seconds, hint):
    def fail(_: str) -> str:
        raise AssertionError("must not be called")

    text, note = transcribe_question(_wav(seconds), fail)
    assert text is None
    assert hint in note


def test_provider_error_leaves_a_note():
    def broken(_: str) -> str:
        raise RuntimeError("HTTP 503 upstream")

    text, note = transcribe_question(_wav(1.0), broken)
    assert text is None
    assert "HTTP 503" in note and "напечатайте" in note


def test_empty_transcript_is_not_asked():
    text, note = transcribe_question(_wav(1.0), lambda _: "   ")
    assert text is None
    assert "Не расслышал" in note


@pytest.mark.parametrize("key, expected", [("sk-or-test", True), ("", False)])
def test_recorder_shown_only_with_speech_to_text_key(monkeypatch, key, expected):
    monkeypatch.setenv("OPENROUTER_API_KEY", key)  # an existing env var wins over .env
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert app.chat_input[0].proto.accept_audio is expected

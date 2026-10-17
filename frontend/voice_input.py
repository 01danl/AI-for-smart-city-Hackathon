"""Turn a question recorded in the chat input into text.

The chat input's built-in recorder returns a 16 kHz mono WAV clip. It is sent once to a
speech-to-text function (OpenRouter STT in the app) and never stored.
"""

from __future__ import annotations

import base64
from typing import Callable

MIN_VOICE_SECONDS = 0.5
MAX_VOICE_SECONDS = 60
_WAV_HEADER_BYTES = 44
_BYTES_PER_SECOND = 16_000 * 2  # 16 kHz, mono, 16-bit PCM


def wav_seconds(wav: bytes) -> float:
    """Approximate duration of a 16 kHz mono 16-bit WAV clip."""
    return max(0.0, (len(wav) - _WAV_HEADER_BYTES) / _BYTES_PER_SECOND)


def transcribe_question(wav: bytes, transcribe: Callable[[str], str]) -> tuple[str | None, str]:
    """Return (question text or None, status note for the UI)."""
    seconds = wav_seconds(wav)
    if seconds < MIN_VOICE_SECONDS:
        return None, "🎙 Запись слишком короткая — нажмите микрофон и скажите вопрос."
    if seconds > MAX_VOICE_SECONDS:
        return None, f"🎙 Запись длиннее {MAX_VOICE_SECONDS} с — скажите вопрос короче."
    try:
        text = " ".join(transcribe(base64.b64encode(wav).decode()).split())
    except Exception as exc:  # noqa: BLE001 — any provider error must leave the demo usable
        short = " ".join(str(exc).split())[:120]
        return None, f"🎙 Не удалось распознать голос ({short}) — напечатайте вопрос."
    if len(text) < 2:
        return None, "🎙 Не расслышал вопрос — попробуйте ещё раз или напечатайте."
    return text, f"🎙 Распознано: «{text}»"

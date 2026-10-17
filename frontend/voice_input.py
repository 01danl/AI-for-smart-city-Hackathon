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


NOTES = {
    "short": {"ru": "🎙 Запись слишком короткая — нажмите микрофон и скажите вопрос.",
              "kk": "🎙 Жазба тым қысқа — микрофонды басып, сұрағыңызды айтыңыз.",
              "en": "🎙 The recording is too short — tap the mic and say your question."},
    "long": {"ru": "🎙 Запись длиннее {max} с — скажите вопрос короче.",
             "kk": "🎙 Жазба {max} с-тан ұзақ — сұрақты қысқарақ айтыңыз.",
             "en": "🎙 The recording is longer than {max} s — please keep it shorter."},
    "failed": {"ru": "🎙 Не удалось распознать голос ({error}) — напечатайте вопрос.",
               "kk": "🎙 Дауысты тану мүмкін болмады ({error}) — сұрақты жазыңыз.",
               "en": "🎙 Could not recognise the voice ({error}) — please type the question."},
    "empty": {"ru": "🎙 Не расслышал вопрос — попробуйте ещё раз или напечатайте.",
              "kk": "🎙 Сұрақты естімедім — қайталап көріңіз немесе жазыңыз.",
              "en": "🎙 I didn't catch that — try again or type the question."},
    "ok": {"ru": "🎙 Распознано: «{text}»", "kk": "🎙 Танылды: «{text}»", "en": "🎙 Recognised: “{text}”"},
}


def _note(key: str, lang: str, **values) -> str:
    entry = NOTES[key]
    return (entry.get(lang) or entry["ru"]).format(**values)


def transcribe_question(
    wav: bytes, transcribe: Callable[[str], str], lang: str = "ru",
) -> tuple[str | None, str]:
    """Return (question text or None, status note for the UI in `lang`)."""
    seconds = wav_seconds(wav)
    if seconds < MIN_VOICE_SECONDS:
        return None, _note("short", lang)
    if seconds > MAX_VOICE_SECONDS:
        return None, _note("long", lang, max=MAX_VOICE_SECONDS)
    try:
        text = " ".join(transcribe(base64.b64encode(wav).decode()).split())
    except Exception as exc:  # noqa: BLE001 — any provider error must leave the demo usable
        short = " ".join(str(exc).split())[:120]
        return None, _note("failed", lang, error=short)
    if len(text) < 2:
        return None, _note("empty", lang)
    return text, _note("ok", lang, text=text)

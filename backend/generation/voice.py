"""Server-side speech-to-text through OpenRouter's STT API.

Used by the Streamlit demo as a fallback when the browser Speech Recognition
API is unavailable (e.g. Firefox). The client records a short 16 kHz WAV clip
and posts it here; the clip is never stored on disk.
"""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv

from backend.generation.client import DEFAULT_BASE_URL, ROOT

DEFAULT_VOICE_MODEL = "google/gemini-3.5-transcribe"


def has_server_asr() -> bool:
    """True when an OPENROUTER_API_KEY is configured (.env or environment)."""
    load_dotenv(ROOT / ".env")
    return bool(os.getenv("OPENROUTER_API_KEY", "").strip())


def transcribe_b64_wav(data_b64: str, *, model: str | None = None, timeout: int = 40) -> str:
    """Transcribe a base64-encoded WAV clip via OpenRouter STT; return plain text.

    Raises RuntimeError/requests exceptions on network or provider errors.
    The audio payload is sent once and never persisted.
    """
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for server-side voice recognition")
    base_url = (os.getenv("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model = model or os.getenv("OPENROUTER_VOICE_MODEL") or DEFAULT_VOICE_MODEL
    resp = requests.post(
        f"{base_url}/audio/transcriptions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "input_audio": {"data": data_b64, "format": "wav"},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    text = str((resp.json() or {}).get("text") or "")
    if not text.strip():
        raise RuntimeError("empty transcript")
    return " ".join(text.split())

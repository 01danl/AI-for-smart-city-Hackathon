"""OpenRouter chat client, compatible with the OpenAI Python SDK."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def get_client() -> OpenAI:
    """Create a client from .env or process environment variables."""
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for generation")
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
    )


def complete(messages: list[dict[str, str]], *, model: str | None = None) -> str:
    """Send chat messages through OpenRouter and return the text response."""
    load_dotenv(ROOT / ".env")
    selected_model = model or os.getenv("OPENROUTER_MODEL")
    if not selected_model:
        raise RuntimeError("OPENROUTER_MODEL is required for generation")
    response = get_client().chat.completions.create(
        model=selected_model,
        messages=messages,
    )
    if not response.choices or not response.choices[0].message.content:
        raise RuntimeError("OpenRouter returned an empty text response")
    return response.choices[0].message.content

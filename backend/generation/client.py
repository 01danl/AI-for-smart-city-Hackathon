"""OpenRouter chat client, compatible with the OpenAI Python SDK."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
load_dotenv(ROOT / ".env")

try:
    from openai import OpenAI  # type: ignore[import-not-found]
except ImportError:
    class _Completions:
        def __init__(self, api_key: str, base_url: str) -> None:
            self._api_key = api_key
            self._base_url = base_url.rstrip("/")

        def create(self, *, model: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
            timeout = kwargs.pop("timeout", 35)
            extra_body = kwargs.pop("extra_body", {})
            resp = requests.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": model, "messages": messages, **kwargs, **extra_body},
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            choices = [
                SimpleNamespace(
                    finish_reason=ch.get("finish_reason"),
                    message=SimpleNamespace(
                        content=ch.get("message", {}).get("content", "")
                    )
                )
                for ch in data.get("choices", [])
            ]
            return SimpleNamespace(choices=choices)

    class OpenAI:  # type: ignore[no-redef]
        """Lightweight OpenAI-compatible HTTP client fallback when `openai` package is absent."""

        def __init__(self, *, api_key: str, base_url: str = DEFAULT_BASE_URL, max_retries: int = 0) -> None:
            self.api_key = api_key
            self.base_url = base_url
            self.chat = SimpleNamespace(completions=_Completions(api_key, base_url))


def get_client(*, max_retries: int | None = None) -> OpenAI:
    """Create a client from .env or process environment variables."""
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for generation")
    base_url = os.getenv("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL
    options = {} if max_retries is None else {"max_retries": max_retries}
    return OpenAI(api_key=api_key, base_url=base_url, **options)


def complete(
    messages: list[dict[str, str]], *, model: str | None = None,
    timeout: float | None = None, max_tokens: int | None = None,
    json_mode: bool = False, reasoning: bool | None = None, max_retries: int | None = None,
) -> str:
    """Send chat messages through OpenRouter and return the text response."""
    load_dotenv(ROOT / ".env")
    selected_model = model or os.getenv("OPENROUTER_MODEL")
    if not selected_model:
        raise RuntimeError("OPENROUTER_MODEL is required for generation")
    options = {}
    if timeout is not None:
        options["timeout"] = timeout
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    if json_mode:
        options["response_format"] = {"type": "json_object"}
    if reasoning is not None:
        options["extra_body"] = {"reasoning": {"enabled": reasoning}}
    client = get_client() if max_retries is None else get_client(max_retries=max_retries)
    response = client.chat.completions.create(
        model=selected_model,
        messages=messages,
        **options,
    )
    if response.choices and getattr(response.choices[0], "finish_reason", None) == "length":
        raise RuntimeError("OpenRouter response was truncated at the token limit")
    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise RuntimeError("OpenRouter returned an empty text response")
    return content

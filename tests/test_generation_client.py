from types import SimpleNamespace

import pytest

from backend.generation import client


def test_openrouter_configuration_and_chat_call(monkeypatch):
    monkeypatch.setattr(client, "load_dotenv", lambda *_: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
    seen = {}

    def fake_openai(**kwargs):
        seen["client"] = kwargs

        def create(**request):
            seen["request"] = request
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Готово"))])

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(client, "OpenAI", fake_openai)
    messages = [{"role": "user", "content": "Куда сходить?"}]
    assert client.complete(messages) == "Готово"
    assert seen["client"] == {
        "api_key": "test-key",
        "base_url": "https://openrouter.ai/api/v1",
    }
    assert seen["request"] == {"model": "nvidia/nemotron-3-ultra-550b-a55b:free", "messages": messages}


def test_missing_key_fails_before_request(monkeypatch):
    monkeypatch.setattr(client, "load_dotenv", lambda *_: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        client.get_client()

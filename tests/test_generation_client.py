"""Unit tests for OpenRouter generation client, RAG privacy guardrails, and anti-hallucination."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.generation import (
    RecommendRequest,
    build_rag_messages,
    client,
    generate_recommendations,
    recommender,
)
from backend.retrieval import parse_query_intent, retrieve


def test_openrouter_configuration_and_chat_call(monkeypatch):
    monkeypatch.setattr(client, "load_dotenv", lambda *_args, **_kwargs: False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

    seen: dict = {}

    def fake_openai(*, api_key: str, base_url: str):
        seen["api_key"] = api_key
        seen["base_url"] = base_url

        def create(**kwargs):
            seen["request"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(client, "OpenAI", fake_openai)
    messages = [{"role": "user", "content": "Привет"}]
    assert client.complete(messages) == "ok"
    assert seen["api_key"] == "test-key"
    assert seen["base_url"] == "https://openrouter.ai/api/v1"
    assert seen["request"] == {
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "messages": messages,
    }
    assert client.complete(messages, timeout=25, max_tokens=6000) == "ok"
    assert seen["request"]["timeout"] == 25
    assert seen["request"]["max_tokens"] == 6000


def test_missing_key_fails_before_request(monkeypatch):
    monkeypatch.setattr(client, "load_dotenv", lambda *_args, **_kwargs: False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        client.get_client()


def test_prompt_never_leaks_raw_user_coordinates():
    raw_lat, raw_lon = 51.1283456, 71.4305678
    cands = retrieve("тихое место с кофе и розеткой", raw_lat, raw_lon, max_distance_m=1500, top_k=3)
    intent = parse_query_intent("тихое место с кофе и розеткой")
    msgs = build_rag_messages("тихое место с кофе и розеткой", cands, intent)
    prompt_dump = " ".join(m["content"] for m in msgs)
    assert "51.1283456" not in prompt_dump
    assert "71.4305678" not in prompt_dump
    assert "rounded_100m" in prompt_dump


@pytest.mark.parametrize("lang", ["ru", "en"])
def test_anti_hallucination_drops_invented_poi_ids(monkeypatch, lang):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    cands = retrieve("кофе", 51.1283, 71.4305, max_distance_m=1500, top_k=3)
    real_id = cands[0].poi.id

    fake_llm_json = f"""
    {{
      "recommendations": [
        {{
          "poi_id": "hallucinated_fake_cafe_999",
          "why": "Придуманное кафе",
          "visit_min": 30,
          "what_to_do": "Не должно попасть в ответ",
          "confidence_note": "Открыто",
          "source": "generated_idea"
        }},
        {{
          "poi_id": "{real_id}",
          "display_name": "Verified Cafe",
          "why": "Реальная кофейня рядом",
          "visit_min": 45,
          "what_to_do": "Взять кофе и поработать",
          "confidence_note": "По данным OSM",
          "source": "poi_data"
        }}
      ]
    }}
    """
    monkeypatch.setattr(recommender, "complete", lambda _msgs: fake_llm_json)
    resp = generate_recommendations(
        RecommendRequest(query="кофе", lat=51.1283, lon=71.4305, max_distance_m=1500, lang=lang)
    )
    returned_ids = [r.poi_id for r in resp.recommendations]
    assert "hallucinated_fake_cafe_999" not in returned_ids
    assert returned_ids == [real_id]
    assert resp.used_data.generation_mode == "openrouter_llm"
    assert resp.recommendations[0].name == ("Verified Cafe" if lang == "en" else cands[0].poi.name)

"""Text and voice share proximity rules before the model sees POI candidates."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from backend.generation import localization, recommender
from backend.generation.proximity import nearby_requested
from backend.generation.recommender import RecommendRequest, RecommendResponse
from backend.retrieval.index import load_pois


@pytest.mark.parametrize("query", ["кофе рядом", "coffee nearby", "Find a coffee shop", "nearest cafe"])
def test_normal_chat_keeps_generic_coffee_in_nearest_ring(query):
    response = recommender.generate_recommendations(RecommendRequest(
        query=query, lat=51.089, lon=71.407, max_distance_m=2000, now_iso="2026-10-18T16:00",
    ), use_llm=False)
    assert response.recommendations
    assert all(item.category == "cafe" and item.distance_m <= 300 for item in response.recommendations)
    assert response.recommendations[0].distance_m < 100
    assert response.route is None or all(stop.poi_id in load_pois() for stop in response.route.stops)


def test_distant_lexical_hit_is_not_available_for_llm_selection(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    distant = next(poi for poi in load_pois().values() if poi.name == "TLP Flowers & Coffee")
    prompts = []
    def complete(messages):
        prompts.append(json.dumps(messages, ensure_ascii=False))
        return json.dumps({"recommendations": [{"poi_id": distant.id}]})
    monkeypatch.setattr(recommender, "complete", complete)
    response = recommender.generate_recommendations(RecommendRequest(
        query="coffee nearby", lat=51.089, lon=71.407, max_distance_m=2000, now_iso="2026-10-18T16:00"))
    assert distant.id not in prompts[0]
    assert response.recommendations and all(item.distance_m <= 300 for item in response.recommendations)


def test_named_venues_and_specific_requirements_keep_relevance():
    assert not nearby_requested("Master Coffee nearby", ["cafe"])
    assert not nearby_requested("quiet cafe with wifi nearby", ["cafe"])
    assert not nearby_requested("2 hour walking route nearby", ["park"])
    assert not nearby_requested("not necessarily near me, a museum", ["museum"])


@pytest.mark.parametrize("translation_ok", [True, False])
def test_chinese_text_chat_uses_nearby_results_or_no_results_on_translation_failure(monkeypatch, translation_ok):
    st.cache_data.clear()
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    query = "我想在附近喝咖啡，最好近一点。"
    monkeypatch.setattr(localization, "prepare_search_query", lambda text, language: (
        "coffee nearby" if translation_ok else text, translation_ok,
    ))
    monkeypatch.setattr(localization, "localize_answer", lambda response, language, labels: {
        "language": language, "requested_language": language, "labels": {}, "note": "",
        "response_json": response.model_dump_json(),
    })
    generate = Mock(wraps=recommender.generate_recommendations)
    monkeypatch.setattr(recommender, "generate_recommendations", generate)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/app.py"), default_timeout=30).run()
    app.chat_input[0].set_value(query).run()
    assert not app.exception, [exception.message for exception in app.exception]
    entry = app.session_state.convos[0]["entries"][0]
    response = RecommendResponse.model_validate_json(entry["response_json"])
    if translation_ok:
        assert response.recommendations and all(item.distance_m <= 300 for item in response.recommendations)
        assert generate.call_count == 1
        assert generate.call_args.kwargs == {"use_llm": False}
    else:
        assert not response.recommendations and response.route is None
        assert response.used_data.generation_mode == "translation_unavailable"
        generate.assert_not_called()
    st.cache_data.clear()

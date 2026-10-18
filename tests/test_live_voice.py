"""Voice credentials, privacy boundary, grounded search and Streamlit event lifecycle."""

import json
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from backend.generation import live_voice as voice
from backend.generation.recommender import RecommendRequest
from backend.retrieval import get_poi


@pytest.fixture
def context():
    return RecommendRequest(query="voice", lat=51.128337, lon=71.430583, max_distance_m=2000,
                            lang="ru", now_iso="2026-10-18T14:00", auto_anchor=True)


def test_token_is_single_use_constrained_and_master_key_stays_server_side(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-master-secret")
    monkeypatch.setattr(voice, "load_dotenv", lambda *a: None)
    post = Mock(return_value=Mock(status_code=200, json=lambda: {"name": "ephemeral-test"}))
    monkeypatch.setattr(voice.requests, "post", post)
    grant = voice.create_grant()
    sent = post.call_args.kwargs
    assert sent["headers"]["x-goog-api-key"] == "test-master-secret"
    assert sent["json"]["uses"] == 1
    assert sent["json"]["bidiGenerateContentSetup"] == grant["setup"]
    assert "test-master-secret" not in json.dumps(grant)
    assert "search_places" in json.dumps(grant["setup"])
    assert grant["seconds"] == 600


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500])
def test_provider_failure_never_exposes_payload_or_key(monkeypatch, status):
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    monkeypatch.setattr(voice, "load_dotenv", lambda *a: None)
    monkeypatch.setattr(voice.requests, "post", Mock(return_value=Mock(status_code=status, text="secret")))
    with pytest.raises(voice.LiveVoiceError) as failure:
        voice.create_grant()
    assert "secret" not in str(failure.value)


def test_missing_key_does_not_call_provider(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(voice, "load_dotenv", lambda *a: None)
    post = Mock()
    monkeypatch.setattr(voice.requests, "post", post)
    with pytest.raises(voice.LiveVoiceError, match="GEMINI_API_KEY"):
        voice.create_grant()
    post.assert_not_called()


def test_voice_search_uses_real_pois_no_llm_no_coords_in_model_payload(monkeypatch, context):
    monkeypatch.setattr(voice.recommender, "complete", Mock(side_effect=AssertionError("No second model call")))
    payload, response = voice.search_places({"query": "coffee nearby"}, context)
    assert response.recommendations
    assert response.used_data.rounded_coords == (51.128, 71.431)
    for place in payload["places"]:
        assert get_poi(place["poi_id"]) is not None
        assert place["distance_m"] <= context.max_distance_m
        assert place["activity_source"] == "generated_idea"
    if response.route:
        assert [stop["poi_id"] for stop in payload["route"]["stops"]] == [stop.poi_id for stop in response.route.stops]
        assert payload["route"]["total_walk_min"] == response.route.total_walk_min
    serialized = json.dumps(payload)
    for private_field in ['"lat"', '"lon"', '"rounded_coords"', '"origin_rounded"', "51.128337", "71.430583"]:
        assert private_field not in serialized


def test_model_cannot_override_location_or_widen_radius(monkeypatch, context):
    with pytest.raises(ValidationError):
        voice.search_places({"query": "cafe", "lat": 0, "lon": 0}, context)
    original = voice.recommender.generate_recommendations
    seen = []
    def generate(req, **kwargs):
        seen.append((req, kwargs))
        return original(req, **kwargs)
    monkeypatch.setattr(voice.recommender, "generate_recommendations", generate)
    voice.search_places({"query": "cafe near Mega Silk Way", "radius_m": 20000}, context)
    req, options = seen[0]
    assert all(request.max_distance_m <= 2000 for request, _ in seen)
    assert not req.auto_anchor
    assert (req.lat, req.lon) == (51.128, 71.431)
    assert options == {"use_llm": False, "prefer_nearby": False}


def test_no_results_does_not_invent_places(context):
    payload, response = voice.search_places({"query": "coffee"}, context.model_copy(update={"lat": 0, "lon": 0}))
    assert payload["places"] == [] and response.recommendations == []


@pytest.mark.parametrize("query", ["кофе рядом", "coffee nearby", "nearest cafe"])
def test_generic_coffee_search_does_not_favor_distant_coffee_brand_names(query):
    context = RecommendRequest(query="voice", lat=51.089, lon=71.407, max_distance_m=2000,
                               now_iso="2026-10-18T16:00")
    payload, response = voice.search_places({"query": query}, context)
    assert response.recommendations
    assert all(item.category == "cafe" and item.distance_m <= 300 for item in response.recommendations)
    assert response.recommendations[0].distance_m < 100
    assert payload["used_data"]["radius_m"] == 300
    assert payload["used_data"]["requested_radius_m"] == 2000
    distances = [item.distance_m for item in response.recommendations]
    assert distances == sorted(distances)


def test_food_query_does_not_return_bars_or_malls():
    _, response = voice.search_places({"query": "food nearby"}, RecommendRequest(
        query="voice", lat=51.089, lon=71.407, now_iso="2026-10-18T16:00"))
    assert response.recommendations
    assert all(item.category in {"cafe", "restaurant", "fast_food"} for item in response.recommendations)


def test_relevance_search_respects_explicit_venue_name(context):
    _, response = voice.search_places({"query": "Master Coffee", "categories": ["cafe"], "sort_by": "relevance"}, context)
    assert response.recommendations[0].name == "Master Coffee"
    assert response.recommendations[0].distance_m > 300


def test_sparse_category_expands_locally_within_limit(context):
    payload, response = voice.search_places({"query": "музей", "categories": ["museum"]}, context)
    assert payload["used_data"]["radius_m"] <= context.max_distance_m
    assert all(item.category == "museum" and item.distance_m <= context.max_distance_m for item in response.recommendations)


def test_tool_category_cannot_override_explicit_user_filter(context):
    with pytest.raises(ValueError, match="conflict"):
        voice.search_places({"query": "restaurant", "categories": ["restaurant"]},
                            context.model_copy(update={"categories": ["museum"]}))


def test_untranslated_chinese_is_not_treated_as_a_generic_poi_search(context):
    with pytest.raises(voice.VoiceSearchError, match="Translate"):
        voice.search_places({"query": "附近有什么咖啡店？"}, context)

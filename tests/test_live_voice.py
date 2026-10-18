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
    assert req.max_distance_m == 2000 and not req.auto_anchor
    assert (req.lat, req.lon) == (51.128, 71.431)
    assert options == {"use_llm": False}


def test_no_results_does_not_invent_places(context):
    payload, response = voice.search_places({"query": "coffee"}, context.model_copy(update={"lat": 0, "lon": 0}))
    assert payload["places"] == [] and response.recommendations == []

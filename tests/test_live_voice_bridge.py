"""Streamlit tool event acknowledgement, cancellation and session isolation."""
from unittest.mock import Mock
import pytest
from backend.generation import live_voice as voice
from backend.generation.recommender import RecommendRequest
from frontend.live_voice import handle_event

@pytest.fixture
def context():
    return RecommendRequest(query="voice", lat=51.128337, lon=71.430583, max_distance_m=2000, lang="ru", now_iso="2026-10-18T14:00")

def test_bridge_deduplicates_and_only_displays_consumed_results(monkeypatch, context):
    monkeypatch.setattr(voice, "create_grant", lambda: {"token": "ephemeral", "setup": {}})
    state = {}
    handle_event({"id": "1", "session": "session-a", "type": "connect"}, state, context)
    search = Mock(wraps=voice.search_places)
    monkeypatch.setattr(voice, "search_places", search)
    event = {"id": "2", "session": "session-a", "type": "tool", "question": "Кофе?",
             "calls": [{"id": "call-1", "name": "search_places", "args": {"query": "coffee"}}]}
    handle_event(event, state, context)
    handle_event(event, state, context)
    handle_event({**event, "id": "3"}, state, context)
    assert search.call_count == 1
    assert "latest" not in state  # A cancelled call has no consumed event.
    handle_event({"id": "4", "session": "old-session", "type": "used", "call_ids": ["call-1"]}, state, context)
    assert "latest" not in state
    handle_event({"id": "5", "session": "session-a", "type": "used", "call_ids": ["call-1"]}, state, context)
    assert state["latest"]["query"] == "Кофе?"
    handle_event({"id": "6", "session": None, "type": "stopped", "closed_session": "session-a"}, state, context)
    assert state["grant"] is None and not state["pending"]


def test_unknown_tool_and_invalid_args_are_recoverable(monkeypatch, context):
    monkeypatch.setattr(voice, "create_grant", lambda: {})
    state = {}
    handle_event({"id": "1", "session": "a", "type": "connect"}, state, context)
    handle_event({"id": "2", "session": "a", "type": "tool", "calls": [
        {"id": "x", "name": "delete_data", "args": {}},
        {"id": "y", "name": "search_places", "args": {"query": "coffee", "lat": 0}},
    ]}, state, context)
    assert all("error" in result["response"] for result in state["tool_results"])

"""Live voice follows the same chat, cards, map and navigation path as typed queries."""

from pathlib import Path
from unittest.mock import Mock
from html import unescape

from streamlit.testing.v1 import AppTest

from backend.generation import live_voice as service
from backend.generation import recommender
from backend.generation.recommender import RecommendRequest
from frontend import live_voice as bridge
from frontend.voice_history import latest_result_entry, sync_voice_history


def test_live_transcript_search_and_map_render_in_normal_chat(monkeypatch):
    monkeypatch.setattr(service, "create_grant", lambda: {"token": "ephemeral", "setup": {}})
    events = []
    component_keys = []
    def component(**kwargs):
        component_keys.append(kwargs["key"])
        return events.pop(0) if events else None
    monkeypatch.setattr(bridge, "_component", component)
    generate = Mock(wraps=recommender.generate_recommendations)
    monkeypatch.setattr(recommender, "generate_recommendations", generate)
    # Any attempt to regenerate/translate a voice turn through OpenRouter is a regression.
    monkeypatch.setattr(recommender, "complete", Mock(side_effect=AssertionError("Duplicate model call")))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/app.py"), default_timeout=30).run()
    events.extend([
        {"id": "1", "session": "a", "type": "connect"},
        {"id": "2", "session": "a", "type": "speech", "turn_id": "t1", "question": "Кофе рядом?", "answer": ""},
        {"id": "3", "session": "a", "type": "tool", "turn_id": "t1", "question": "Кофе рядом?",
         "calls": [{"id": "c1", "name": "search_places", "args": {"query": "coffee nearby walking route"}}]},
        {"id": "4", "session": "a", "type": "used", "call_ids": ["c1"]},
        {"id": "5", "session": "a", "type": "speech", "turn_id": "t1", "question": "Кофе рядом?",
         "answer": "Сходите в ближайшую кофейню.", "final": True},
    ])
    app.toggle(key="live_voice_enabled").set_value(True).run()
    assert not app.exception, [e.message for e in app.exception]
    entry = app.session_state.convos[0]["entries"][0]
    assert app.session_state.active_convo == 1
    assert entry["via"] == "live_voice" and entry["assistant_text"] == "Сходите в ближайшую кофейню."
    response = recommender.RecommendResponse.model_validate_json(entry["response_json"])
    assert response.recommendations and response.route
    html = " ".join(element.value for element in app.markdown)
    assert "Кофе рядом?" in html and "Сходите в ближайшую кофейню." in html
    assert all(rec.name in unescape(html) for rec in response.recommendations), [r.name for r in response.recommendations]
    assert "aw-tl-row" in html  # The regular route timeline is rendered beside the map.
    charts = app.get("deck_gl_json_chart")
    assert charts and response.recommendations[0].name in charts[0].proto.json
    assert generate.call_count == 1
    assert len(set(component_keys)) == 1  # Creating a chat cannot remount the live microphone iframe.
    revision = app.session_state.map_view_revision

    events.extend([
        {"id": "6", "session": "a", "type": "speech", "turn_id": "t2", "question": "Спасибо",
         "answer": "Приятной прогулки!", "final": True},
        {"id": "7", "session": None, "type": "stopped", "closed_session": "a"},
    ])
    app.run()
    assert not app.exception
    assert app.session_state.map_view_revision == revision
    assert latest_result_entry(app.session_state.convos[0]["entries"])["voice_turn"] == "a:t1"
    app.toggle(key="live_voice_enabled").set_value(False).run()
    app.button(key="btn_newchat").click().run()
    app.button(key="btn_convo_1").click().run()
    assert not app.exception
    assert "Приятной прогулки!" in " ".join(element.value for element in app.markdown)
    assert app.get("deck_gl_json_chart") and generate.call_count == 1


def test_history_updates_existing_turn_without_duplicates_or_resurrecting_trimmed_turns():
    context = RecommendRequest(query="voice").model_dump_json()
    store = {"convos": [], "active_convo": None, "next_convo_id": 1}
    voice = {"context": context, "history_session": "a", "turns": {
        "t1": {"question": "Привет", "answer": "Здра"},
    }}
    sync_voice_history(store, voice, "Байтерек", max_history=2)
    voice["turns"]["t1"].update(answer="Здравствуйте", final=True)
    sync_voice_history(store, voice, "Байтерек", max_history=2)
    assert len(store["convos"][0]["entries"]) == 1
    assert store["convos"][0]["entries"][0]["assistant_text"] == "Здравствуйте"
    for index in range(2, 5):
        voice["turns"][f"t{index}"] = {"question": str(index), "answer": "Ответ", "final": True}
        sync_voice_history(store, voice, "Байтерек", max_history=2)
    sync_voice_history(store, voice, "Байтерек", max_history=2)
    assert [entry["voice_turn"] for entry in store["convos"][0]["entries"]] == ["a:t3", "a:t4"]
    assert len(voice["turns"]) == 2

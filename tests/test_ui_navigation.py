"""Regression coverage for chat navigation before and after voice input."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_chat_navigation_clears_optional_voice_status():
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "frontend" / "app.py"),
        default_timeout=30,
    ).run()
    assert not app.exception
    assert "mic_state" not in app.session_state

    # Seed a conversation without calling the model or asking for microphone access.
    app.session_state.convos = [{"id": 1, "title": "Прогулка", "entries": []}]
    app.run()
    app.button(key="btn_convo_1").click().run()
    assert not app.exception
    assert app.session_state.active_convo == 1

    app.button(key="btn_newchat").click().run()
    assert not app.exception
    assert app.session_state.active_convo is None

    app.session_state.mic_state = ("sent", "Распознано")
    app.button(key="btn_convo_1").click().run()
    assert not app.exception
    assert "mic_state" not in app.session_state

    # Repeated cleanup must also work when the status is already absent.
    app.button(key="btn_newchat").click().run()
    assert not app.exception

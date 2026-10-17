"""The UI keeps one source of truth for the search start point."""

from pathlib import Path

from streamlit.testing.v1 import AppTest
import streamlit_js_eval

APP_PATH = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
FIX = {"coords": {"latitude": 51.099, "longitude": 71.433, "accuracy": 43}}


def test_settings_closed_by_default_and_map_panel_shown():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert app.session_state.show_settings is False
    assert not any(b.key == "btn_geo" for b in app.button)


def test_landmark_replaces_earlier_geolocation(monkeypatch):
    monkeypatch.setattr(streamlit_js_eval, "streamlit_js_eval", lambda **kwargs: FIX)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.button(key="btn_gear").click().run()
    app.button(key="btn_geo").click().run()
    assert not app.exception
    assert app.session_state.start_source == "geo"
    assert "геолокацией" in app.session_state.geo_note

    app.selectbox(key="landmark").select("Астана Опера").run()
    assert not app.exception
    assert app.session_state.start_source == "landmark"
    assert app.session_state.geo_note is None
    assert (app.session_state.lat, app.session_state.lon) == (51.1357, 71.4106)

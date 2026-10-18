"""Map safety and controls: preserve real stops, rounded origin and cached answers."""

import json
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from backend.generation import OFFICIAL_PRESETS, recommender
from backend.generation.recommender import RecommendRequest
from frontend.map_view import MAP_STYLES, build_map_deck

STATUSES = {key: key for key in ("confirmed_open", "likely_open", "unknown", "likely_closed")}


def sample():
    preset = OFFICIAL_PRESETS[1]
    request = RecommendRequest(**{key: value for key, value in preset.items() if key not in ("id", "title")})
    return recommender.generate_recommendations(request, use_llm=False)


def spec(response, **kwargs):
    deck = build_map_deck(response, start_label="Rounded start", status_labels=STATUSES, **kwargs)
    return json.loads(deck.to_json())


def test_route_preserves_stop_order_and_includes_every_real_marker():
    response = sample()
    before = response.model_dump()
    assert response.route is not None
    layers = {layer["id"]: layer for layer in spec(response)["layers"]}
    lat, lon = response.used_data.rounded_coords
    assert layers["route-line"]["data"][0]["path"] == [[round(lon, 3), round(lat, 3)]] + [
        [stop.lon, stop.lat] for stop in response.route.stops
    ]
    markers = {marker["poi_id"]: marker for marker in layers["place-markers"]["data"]}
    assert set(markers) == {rec.poi_id for rec in response.recommendations} | {
        stop.poi_id for stop in response.route.stops
    }
    for number, rec in enumerate(response.recommendations, 1):
        assert markers[rec.poi_id]["position"] == [rec.lon, rec.lat]
        assert markers[rec.poi_id]["label"] == str(number)
    assert response.model_dump() == before


def test_no_route_does_not_invent_connecting_paths_and_origin_is_rounded():
    response = sample().model_copy(deep=True)
    response.route = None
    response.used_data.rounded_coords = (51.1283456, 71.4305678)
    data = spec(response)
    layers = {layer["id"]: layer for layer in data["layers"]}
    assert "route-line" not in layers and "route-direction" not in layers
    assert layers["start-point"]["data"][0]["position"] == [71.431, 51.128]
    assert "51.1283456" not in json.dumps(data) and "71.4305678" not in json.dumps(data)


def test_map_controls_change_presentation_without_changing_geometry():
    response = sample()
    light = spec(response)
    dark = spec(response, theme="dark", show_names=False)
    assert light["mapStyle"] == MAP_STYLES["light"]
    assert dark["mapStyle"] == MAP_STYLES["dark"]
    assert "place-labels" not in {layer["id"] for layer in dark["layers"]}
    light_path = next(layer["data"] for layer in light["layers"] if layer["id"] == "route-line")
    dark_path = next(layer["data"] for layer in dark["layers"] if layer["id"] == "route-line")
    assert light_path == dark_path
    assert all(layer.get("fontFamily") == "Arial" for layer in light["layers"] if layer["@@type"] == "TextLayer")


def test_style_toggle_names_and_fit_do_not_regenerate_recommendations(monkeypatch):
    st.cache_data.clear()
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    response = sample()
    requests = []

    def generate(req):
        requests.append(req)
        return response

    monkeypatch.setattr(recommender, "generate_recommendations", generate)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/app.py"),
                            default_timeout=30).run()
    app.button(key="btn_preset_coffee_laptop").click().run()
    assert not app.exception
    app.radio(key="map_theme").set_value("dark").run()
    app.checkbox(key="map_names").uncheck().run()
    app.button(key="fit_map").click().run()
    assert not app.exception
    assert app.session_state.map_view_revision == 1
    assert len(requests) == 1
    st.cache_data.clear()

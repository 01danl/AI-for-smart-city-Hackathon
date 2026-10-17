"""Messages must be visible in the same submission, before generation starts."""

import json
from pathlib import Path

import pytest
import streamlit as st
import streamlit_js_eval
from streamlit.delta_generator import DeltaGenerator
from streamlit.testing.v1 import AppTest

import backend.generation as generation
from backend.generation import RecommendResponse, UsedData


@pytest.mark.parametrize("geolocate", [False, True])
def test_first_and_followup_messages_render_before_generation(monkeypatch, geolocate):
    st.cache_data.clear()
    events = []
    requests = []
    markdown = st.markdown
    html = DeltaGenerator.html

    def observe_markdown(body, *args, **kwargs):
        events.append(("render", str(body)))
        return markdown(body, *args, **kwargs)

    def observe_html(self, body, *args, **kwargs):
        events.append(("render", str(body)))
        return html(self, body, *args, **kwargs)

    def fake_generate(req):
        events.append(("generate", req.query))
        requests.append(req)
        return RecommendResponse(
            used_data=UsedData(rounded_coords=(req.lat, req.lon)),
            recommendations=[], disclaimer=f"Ответ на: {req.query}",
        )

    monkeypatch.setattr(st, "markdown", observe_markdown)
    monkeypatch.setattr(DeltaGenerator, "html", observe_html)
    monkeypatch.setattr(generation, "generate_recommendations", fake_generate)
    monkeypatch.setattr(streamlit_js_eval, "streamlit_js_eval", lambda **kwargs: {
        "coords": {"latitude": 51.100345, "longitude": 71.389567, "accuracy": 25},
    })
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "frontend" / "app.py"), default_timeout=30,
    ).run()
    if geolocate:
        app.button(key="btn_gear").click().run()
        app.button(key="btn_geo").click().run()
        assert app.session_state.start_source == "geo"
        # Cover submission after Streamlit has cleaned up hidden settings widgets.
        app.button(key="btn_close_settings").click().run()
        app.run()

    for index, query in enumerate(["Найди где можно покушать", "Что можно посетить рядом?"], start=1):
        events.clear()
        app.chat_input[0].set_value(query).run()
        assert not app.exception
        assert len(requests) == index
        rendered = " ".join(element.value for element in app.markdown)
        assert query in rendered
        assert f"Ответ на: {query}" in rendered
        assert app.button(key="btn_convo_1")
        before_generate = events[:events.index(("generate", query))]
        assert any('class="aw-row user"' in body and query in body for _, body in before_generate)
        assert any("Подбираю места рядом" in body for _, body in before_generate)
        if geolocate:
            assert (requests[-1].lat, requests[-1].lon) == (51.1, 71.39)

    entries = app.session_state.convos[0]["entries"]
    assert [json.loads(entry["req_json"])["query"] for entry in entries] == [req.query for req in requests]
    st.cache_data.clear()  # Completed history must survive shared response-cache eviction.
    app.button(key="btn_newchat").click().run()
    app.button(key="btn_convo_1").click().run()
    assert not app.exception
    assert len(requests) == 2  # Reopening history must not regenerate answers.

    # A scenario must replace the hidden settings backup before a typed follow-up.
    preset = generation.OFFICIAL_PRESETS[1]
    app.button(key=f"btn_{preset['id']}").click().run()
    app.chat_input[0].set_value("А что ещё посмотреть здесь?").run()
    assert not app.exception
    assert (requests[-1].lat, requests[-1].lon) == (preset["lat"], preset["lon"])

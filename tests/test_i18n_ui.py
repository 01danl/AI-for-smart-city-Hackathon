"""Russian / Kazakh / English interface: complete dictionaries, switcher, scenarios."""

import re
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import backend.generation as generation
from backend.generation import recommender
from backend.generation import RecommendResponse, UsedData
from backend.retrieval.intent import parse_query_intent
from frontend import i18n

APP_PATH = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
KAZAKH_LETTERS = re.compile(r"[әғқңөұүһі]", re.IGNORECASE)


def test_every_text_has_all_languages():
    tables = [i18n.TEXT, i18n.STATUS_LABEL, i18n.SOURCE_LABEL, i18n.CATEGORY, i18n.WEATHER,
              i18n.LANDMARK, i18n.PRESET]
    for table in tables:
        for key, entry in table.items():
            assert set(entry) == set(i18n.LANGS), key
            assert all(entry[lang] for lang in i18n.LANGS), key


def test_placeholders_match_across_languages():
    for key, entry in i18n.TEXT.items():
        names = {lang: set(re.findall(r"{(\w+)}", text)) for lang, text in entry.items()}
        assert names["kk"] == names["ru"] == names["en"], key


@pytest.mark.parametrize("preset_id", list(i18n.PRESET))
def test_kazakh_and_english_scenarios_keep_intent(preset_id):
    """Translated questions must be understood like the Russian ones (anchor, time budget)."""
    ru = parse_query_intent(i18n.PRESET[preset_id]["ru"][1])
    for lang in ("kk", "en"):
        other = parse_query_intent(i18n.PRESET[preset_id][lang][1])
        assert (other.anchor_landmark and other.anchor_landmark.key) == (ru.anchor_landmark and ru.anchor_landmark.key)
        assert other.time_budget_min == ru.time_budget_min, (lang, preset_id)


def _fake_generate(requests):
    def fake(req):
        requests.append(req)
        return RecommendResponse(used_data=UsedData(rounded_coords=(req.lat, req.lon)),
                                 recommendations=[], disclaimer="—")
    return fake


def test_switching_to_kazakh_translates_ui_and_asks_in_kazakh(monkeypatch):
    st.cache_data.clear()
    requests = []
    monkeypatch.setattr(recommender, "generate_recommendations", _fake_generate(requests))
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception

    app.session_state.ui_lang = "kk"
    app.session_state.lang = "kk"
    app.run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "Сценарийлер" in text and "Чат тарихы" in text
    assert app.button(key="btn_newchat").label == "Жаңа чат"

    preset = generation.OFFICIAL_PRESETS[0]
    app.button(key=f"btn_{preset['id']}").click().run()
    assert not app.exception
    assert requests[-1].lang == "kk"
    assert KAZAKH_LETTERS.search(requests[-1].query)


def test_english_settings_drawer(monkeypatch):
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.session_state.ui_lang = "en"
    app.session_state.lang = "en"
    app.run()
    app.button(key="btn_gear").click().run()
    assert not app.exception
    assert app.button(key="btn_geo").label == "Use my location"
    assert app.selectbox(key="landmark").label == "Start point"
    # No separate answer-language select: the switcher sets it.
    assert not any(sb.label == "Язык ответа" for sb in app.selectbox)

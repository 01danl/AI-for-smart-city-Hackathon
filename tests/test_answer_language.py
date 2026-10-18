"""Per-message languages, verified numeric fields and independent interface settings."""

import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import backend.generation as generation
from backend.generation import RecommendResponse, UsedData
from backend.generation import localization
from backend.generation import recommender
from backend.generation.recommender import RecommendationItem
from frontend import i18n


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")


@pytest.mark.parametrize("query,language", [
    ("Найди где можно покушать", "ru"),
    ("Что можно посетить рядом?", "ru"),
    ("Where can I eat nearby?", "en"),
    ("Жақын жерде қайда тамақтануға болады?", "kk"),
    ("附近有什么好吃的餐厅？", "zh"),
    ("Je cherche un café calme à proximité", "fr"),
    ("¿Dónde puedo comer cerca de aquí?", "es"),
])
def test_detect_language_independently_of_interface(query, language):
    assert localization.detect_answer_language(query, "ru") == language
    assert localization.detect_answer_language(query, "en") == language


def response():
    return RecommendResponse(
        used_data=UsedData(rounded_coords=(51.128, 71.431)),
        recommendations=[RecommendationItem(
            poi_id="osm_node_123", name="Национальный музей", category="museum",
            lat=51.1189, lon=71.4691, distance_m=375, walk_min=5, visit_min=45,
            why="A museum nearby.", what_to_do="Explore the exhibition.",
            open_status="unknown", confidence_note="Hours are unavailable; check before visiting.",
            source="poi_data", facts_poi_data=["Distance: 375 m."],
        )], disclaimer="Opening hours may be incomplete.",
    )


def fake_chinese(instruction, payload):
    assert payload["target_language"] == "zh"
    return {"texts": {key: "中文 " + value for key, value in payload["texts"].items()},
            "names": {"osm_node_123": "National Museum", "invented_id": "Invented Place"}}


def test_localization_only_changes_display_text(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    seen = []

    def translate(instruction, payload):
        seen.append(json.dumps(payload, ensure_ascii=False))
        return fake_chinese(instruction, payload)

    monkeypatch.setattr(localization, "_json_call", translate)
    canonical = response()
    before = canonical.model_dump()
    presentation = localization.localize_answer(canonical, "zh", i18n.answer_dictionary())
    localized = RecommendResponse.model_validate_json(presentation["response_json"])
    assert canonical.model_dump() == before
    assert presentation["language"] == "zh" and not presentation["note"]
    assert localized.recommendations[0].name == "National Museum"
    original = canonical.recommendations[0].model_dump()
    changed = localized.recommendations[0].model_dump()
    for field in ("poi_id", "lat", "lon", "category", "distance_m", "walk_min", "visit_min", "open_status", "source"):
        assert changed[field] == original[field]
    assert "Invented Place" not in presentation["response_json"]
    assert "51.1189" not in seen[0] and "71.4691" not in seen[0]
    assert "51.128" not in seen[0] and "71.431" not in seen[0]
    with i18n.answer_context(presentation):
        assert i18n.t("card_away", "zh", d=375).startswith("中文 ")
        assert "375" in i18n.t("card_away", "zh", d=375)
    assert i18n.current_answer_language() is None
    assert i18n.t("new_chat", "ru") == "Новый чат"


@pytest.mark.parametrize("source,translated", [
    ("{d} m away", "附近"),
    ("{d} m away", "{d:bad} 米"),
    ("375 m", "100 m"),
    ("Hours unknown", "<script>alert(1)</script>"),
])
def test_rejects_changed_numbers_placeholders_and_markup(source, translated):
    assert not localization._safe_translation(source, translated)


def test_unavailable_translation_has_explicit_english_fallback():
    presentation = localization.localize_answer(response(), "zh", i18n.answer_dictionary())
    assert presentation["language"] == "en"
    assert "英文" in presentation["note"]
    localized = RecommendResponse.model_validate_json(presentation["response_json"])
    assert localized.recommendations[0].name.isascii()


def test_unsafe_output_rejects_entire_translation(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")

    def translate(instruction, payload):
        translated = fake_chinese(instruction, payload)
        translated["texts"]["rec:0:fact:0"] = "Distance: 1 m."
        return translated

    monkeypatch.setattr(localization, "_json_call", translate)
    presentation = localization.localize_answer(response(), "zh", i18n.answer_dictionary())
    assert presentation["language"] == "en"
    assert not presentation["labels"]
    assert "375 m" in presentation["response_json"]


def test_core_languages_do_not_request_a_second_translation(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")

    def unexpected(*args, **kwargs):
        pytest.fail("Core-language generation must not make a second translation request")

    monkeypatch.setattr(localization, "_json_call", unexpected)
    for language in ("ru", "kk", "en"):
        assert localization.localize_answer(response(), language, {})["language"] == language


@pytest.mark.parametrize("language,phrase", [("en", "No matching places"), ("kk", "сәйкес орындар")])
def test_no_results_uses_request_language(language, phrase):
    result = generation.generate_recommendations(generation.RecommendRequest(
        query="museum", lat=0, lon=0, max_distance_m=100, lang=language,
    ), use_llm=False)
    assert not result.recommendations
    assert phrase in result.disclaimer


def test_chat_keeps_message_languages_and_dropdown_only_changes_interface(monkeypatch):
    st.cache_data.clear()
    requests = []
    translations = []
    presentations = []

    def generate(req, **kwargs):
        requests.append((req, kwargs))
        return RecommendResponse(used_data=UsedData(rounded_coords=(req.lat, req.lon)),
                                 recommendations=[], disclaimer="No results.")

    def search(query, language):
        translations.append((query, language))
        return ("Where can I eat nearby?" if language == "zh" else query), True

    def localize(resp, language, labels):
        presentations.append(language)
        return {"language": language, "requested_language": language, "labels": {}, "note": "",
                "response_json": resp.model_dump_json()}

    # Streamlit can retain the old package re-export after reloading the implementation module.
    def stale_package_export(req):
        pytest.fail("The UI must call the implementation, not a stale package re-export")

    monkeypatch.setattr(generation, "generate_recommendations", stale_package_export)
    monkeypatch.setattr(recommender, "generate_recommendations", generate)
    monkeypatch.setattr(localization, "prepare_search_query", search)
    monkeypatch.setattr(localization, "localize_answer", localize)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/app.py"),
                            default_timeout=30).run()
    assert not app.exception
    assert app.selectbox(key="ui_lang").value == "ru"
    assert not app.get("segmented_control")
    for query in ("Where can I eat nearby?", "附近有什么好吃的餐厅？", "Что можно посетить рядом?"):
        app.chat_input[0].set_value(query).run()
        assert not app.exception
    entries = app.session_state.convos[0]["entries"]
    assert [entry["answer_language"] for entry in entries] == ["en", "zh", "ru"]
    assert entries[1]["query"] == "附近有什么好吃的餐厅？"
    assert requests[1][0].query == "Where can I eat nearby?"
    assert requests[1][0].lang == "en" and requests[1][1] == {"use_llm": False}
    app.selectbox(key="ui_lang").select("kk").run()
    assert not app.exception
    assert app.button(key="btn_newchat").label == "Жаңа чат"
    assert [entry["answer_language"] for entry in app.session_state.convos[0]["entries"]] == ["en", "zh", "ru"]
    assert presentations == ["en", "zh", "ru"]  # Reruns reuse localized history.
    assert len(requests) == 3
    st.cache_data.clear()

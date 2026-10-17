"""Query intent understands Kazakh and English as well as Russian."""

import pytest

from backend.retrieval.intent import detect_language, extract_time_budget_minutes, parse_query_intent


@pytest.mark.parametrize("query, minutes", [
    ("Маған 2 сағат бар, не көруге болады?", 120),
    ("30–60 минутқа серуендегім келеді", 60),
    ("1–2 сағатқа ерекше бірдеңе", 120),
    ("жарты сағатқа кофе ішейін", 30),
    ("I have 2 hours before my flight", 120),
    ("a walk for 30–60 minutes", 60),
    ("half an hour for coffee", 30),
])
def test_time_budget_in_kazakh_and_english(query, minutes):
    assert extract_time_budget_minutes(query) == minutes


@pytest.mark.parametrize("query, anchor", [
    ("Мен Ботаникалық бақтың жанындамын", "botanical_garden"),
    ("Ұлттық музейге жақын жер", "national_museum"),
    ("Ескі орталықта серуендеу", "old_center"),
    ("I'm near the Old Town", "old_center"),
    ("Нұр Әлем ғимаратының жанындамын", "expo_2017"),
])
def test_kazakh_and_english_landmarks(query, anchor):
    intent = parse_query_intent(query)
    assert intent.anchor_landmark is not None
    assert intent.anchor_landmark.key == anchor


@pytest.mark.parametrize("query, weather", [
    ("Сыртта аяз, жылынатын жер керек", "cold_windy"),
    ("Жаңбыр жауып тұр, қайда барайын?", "rain_snow"),
    ("It's freezing outside", "cold_windy"),
    # "қарау" (to look) must not be read as "қар" (snow)
    ("Бәйтеректі қарау үшін қайда барайын?", "normal"),
])
def test_kazakh_weather_words(query, weather):
    assert parse_query_intent(query).weather_context == weather


def test_russian_bar_is_not_kazakh():
    assert detect_language("Хочу в бар вечером") == "ru"

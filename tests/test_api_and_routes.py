"""Tests for FastAPI endpoints, query intent parser, visit window checks, and micro-routes."""

from __future__ import annotations

from backend.generation import OFFICIAL_PRESETS, RecommendRequest, generate_recommendations
from backend.retrieval import build_micro_route, get_poi, parse_query_intent, retrieve
from backend.retrieval.hours import check_visit_window, is_late_night


def test_intent_parser_extracts_anchors_and_budgets():
    i1 = parse_query_intent("Я сейчас возле Mega Silk Way, хочу прогуляться 30–60 минут — что посоветуете?")
    assert i1.anchor_landmark is not None
    assert i1.anchor_landmark.key == "mega_silk_way"
    assert i1.time_budget_min == 60
    assert i1.wants_route is True

    i2 = parse_query_intent("Я у здании Экспо 2017, хочу что-то необычное на 1–2 часа, пешком.")
    assert i2.anchor_landmark is not None
    assert i2.anchor_landmark.key == "expo_2017"
    assert i2.time_budget_min == 120
    assert "unusual" in i2.semantic_tags

    i3 = parse_query_intent("На улице сильный ветер и мороз, где погреться и выпить кофе?")
    assert i3.weather_context == "cold_windy"
    assert "indoor" in i3.inferred_tags


def test_visit_window_warns_when_closing_mid_visit():
    status, reason, left = check_visit_window(
        "Mo-Su 09:00-22:00", "2026-10-16T21:15", walk_min=15, visit_min=45
    )
    assert status == "likely_open"
    assert left == 45
    assert "закроется через 45 мин" in reason
    assert is_late_night("2026-10-16T23:30") is True
    assert is_late_night("2026-10-16T14:00") is False


def test_micro_route_builds_diverse_chain():
    cands = retrieve("что посмотреть туристу", 51.1283, 71.4305, max_distance_m=3000, top_k=8)
    route = build_micro_route(cands, 51.1283, 71.4305, max_stops=3, time_budget_min=120)
    assert route is not None
    assert 2 <= len(route.stops) <= 3
    assert route.total_duration_min == route.total_walk_min + route.total_visit_min
    assert route.origin_rounded == (51.128, 71.43)


def test_all_official_hackathon_presets_produce_grounded_recommendations(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    for preset in OFFICIAL_PRESETS:
        req = RecommendRequest(
            query=preset["query"],
            lat=preset["lat"],
            lon=preset["lon"],
            max_distance_m=preset["max_distance_m"],
            now_iso=preset["now_iso"],
            accessible=preset["accessible"],
            group_size=preset["group_size"],
            weather=preset["weather"],
        )
        resp = generate_recommendations(req)
        assert 1 <= len(resp.recommendations) <= 3
        for rec in resp.recommendations:
            poi = get_poi(rec.poi_id)
            assert poi is not None
            assert rec.distance_m <= req.max_distance_m
            assert rec.source in ("poi_data", "generated_idea")
            if poi.opening_hours is None:
                assert rec.open_status == "unknown"


def test_outside_city_returns_empty_without_hallucinations(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    resp = generate_recommendations(
        RecommendRequest(query="кофе", lat=51.30, lon=71.00, max_distance_m=2000)
    )
    assert resp.recommendations == []
    assert resp.route is None
    assert "не выдумывает" in resp.disclaimer

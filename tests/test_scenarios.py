"""Retrieval-level checks for the scenarios in docs/test_scenarios.md.

Runs against the real Chroma index; skipped if it hasn't been built
(python scripts/build_index.py).
"""

import pytest

from backend.retrieval.index import index_exists

pytestmark = pytest.mark.skipif(not index_exists(), reason="Chroma index not built")

BAITEREK = (51.1283, 71.4305)
OUTSIDE_CITY = (51.30, 71.00)


@pytest.fixture(scope="module")
def retrieve():
    from backend.retrieval.chroma_retriever import retrieve

    return retrieve


def test_s1_laptop_cafe(retrieve):
    res = retrieve("тихое место с кофе и розеткой", *BAITEREK,
                   max_distance_m=1500, now_iso="2026-10-16T13:00", top_k=5)
    assert all(c.poi.category == "cafe" for c in res[:3])
    assert any("wifi" in c.poi.tags for c in res)


def test_s2_walk_outdoors(retrieve):
    res = retrieve("погулять на свежем воздухе", *BAITEREK, now_iso="2026-10-16T13:00")
    assert all("outdoor" in c.poi.tags for c in res[:3])


def test_s3_museum_far(retrieve):
    res = retrieve("музей история", *BAITEREK, max_distance_m=5000, now_iso="2026-10-16T13:00")
    assert res[0].poi.category in {"museum", "gallery"}


def test_s4_kazakh_cuisine_family(retrieve):
    res = retrieve("поесть казахскую кухню с семьёй", *BAITEREK, now_iso="2026-10-16T13:00")
    kazakh = [c for c in res[:3] if {"cuisine_kazakh", "cuisine_regional"} & set(c.poi.tags)]
    assert len(kazakh) >= 2


def test_s5_late_night_no_closed_on_top(retrieve):
    res = retrieve("поесть", *BAITEREK, now_iso="2026-10-16T23:40")
    assert res
    assert all(c.open_status != "likely_closed" for c in res[:3])


def test_s6_wheelchair(retrieve):
    res = retrieve("доступно для колясок", *BAITEREK, now_iso="2026-10-16T13:00")
    assert all("wheelchair" in c.poi.tags for c in res[:3])


def test_s7_english_query(retrieve):
    res = retrieve("quiet place to read a book", *BAITEREK, max_distance_m=4000)
    assert res[0].poi.category == "library"


def test_s8_outside_city_returns_nothing(retrieve):
    assert retrieve("кофе", *OUTSIDE_CITY) == []


def test_s9_nonsense_query_is_weak(retrieve):
    res = retrieve("ыфвафыва", *BAITEREK)
    assert all(c.relevance < 0.6 for c in res)


def test_invariants(retrieve):
    res = retrieve("куда сходить", *BAITEREK, max_distance_m=3000, top_k=50)
    for c in res:
        assert c.distance_m <= 3000
        assert 0 <= c.relevance <= 1
        assert c.walk_min >= 1
        if c.poi.opening_hours is None:
            assert c.open_status == "unknown"

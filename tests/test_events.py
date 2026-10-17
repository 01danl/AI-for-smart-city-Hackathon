"""Temporary events: real listings only, venues from the database, honest live status."""

import json
from datetime import date

import pytest

from backend.retrieval import events as ev
from backend.retrieval.index import load_pois

OPERA = "osm_way_917478479"
OPERA_LAT, OPERA_LON = 51.1357, 71.4106


@pytest.fixture
def events_file(tmp_path, monkeypatch):
    path = tmp_path / "events.json"
    monkeypatch.setenv("EVENTS_PATH", str(path))
    ev.load_events.cache_clear()
    yield path
    ev.load_events.cache_clear()


def _write(path, *events):
    base = {"kind": "concert", "venue_poi_id": OPERA, "duration_min": 90,
            "source_url": "https://astanaopera.kz/ru/afisha/1", "checked_at": "2026-10-18"}
    path.write_text(json.dumps([{**base, **e} for e in events], ensure_ascii=False), encoding="utf-8")


@pytest.mark.parametrize("now, status, to_start", [
    ("2026-10-28T17:40", "starts_soon", 80),
    ("2026-10-28T19:30", "ongoing", -30),
    ("2026-10-28T13:00", "later", 360),
])
def test_status_relative_to_chosen_time(events_file, now, status, to_start):
    _write(events_file, {"id": "jazz", "title": "Jazz", "start": "2026-10-28T19:00"})
    found = ev.events_near(OPERA_LAT, OPERA_LON, load_pois(), now_iso=now)
    assert [(c.status, c.minutes_to_start) for c in found] == [(status, to_start)]


@pytest.mark.parametrize("now", ["2026-10-28T21:00", "2026-10-28T08:00"])
def test_finished_or_far_future_events_are_hidden(events_file, now):
    _write(events_file, {"id": "jazz", "title": "Jazz", "start": "2026-10-28T19:00"})
    assert ev.events_near(OPERA_LAT, OPERA_LON, load_pois(), now_iso=now) == []


def test_events_out_of_reach_or_without_known_venue_are_hidden(events_file):
    _write(events_file,
           {"id": "jazz", "title": "Jazz", "start": "2026-10-28T19:00"},
           {"id": "ghost", "title": "Ghost", "start": "2026-10-28T19:00", "venue_poi_id": "osm_node_missing"})
    pois = load_pois()
    assert ev.events_near(51.30, 71.00, pois, now_iso="2026-10-28T18:00") == []
    assert [c.event.id for c in ev.events_near(OPERA_LAT, OPERA_LON, pois, now_iso="2026-10-28T18:00")] == ["jazz"]


def test_multi_day_exhibition_is_on_between_its_dates(events_file):
    _write(events_file, {"id": "expo", "title": "Expo", "kind": "exhibition",
                         "start": "2026-10-20T10:00", "end": "2026-11-15T23:59"})
    found = ev.events_near(OPERA_LAT, OPERA_LON, load_pois(), now_iso="2026-10-22T12:00")
    assert found and found[0].status == "ongoing"


def test_shipped_events_are_sourced_and_placeable():
    ev.load_events.cache_clear()
    events = ev.load_events()
    pois = load_pois()
    assert events, "data/events.json must not be empty"
    assert len({e.id for e in events}) == len(events)
    for e in events:
        assert e.venue_poi_id in pois, e.id
        assert e.source_url.startswith("https://"), e.id
        date.fromisoformat(e.checked_at)


def test_jazz_at_the_opera_starts_soon():
    ev.load_events.cache_clear()
    found = ev.events_near(OPERA_LAT, OPERA_LON, load_pois(), now_iso="2026-10-28T17:40", max_distance_m=2500)
    assert found[0].event.id == "opera_677_20261028"
    assert found[0].status == "starts_soon"


def test_pitch_day_scenario_finds_the_republic_day_concert():
    """'Evening in the city' scenario: old town, 22.10 at 17:40 (pitch day)."""
    ev.load_events.cache_clear()
    found = ev.events_near(51.1694, 71.4491, load_pois(), now_iso="2026-10-22T17:40", max_distance_m=2500)
    assert found[0].event.id == "tckt_republic_day_20261022"
    assert found[0].status == "starts_soon" and found[0].minutes_to_start == 80

"""Import safety and schedule semantics for the official-source supplement."""
import json
from pathlib import Path

import pytest

from scripts.collect_official_poi import parse_page
from backend.retrieval.hours import open_status

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("marker", [
    "",  # No coordinates: never geocode or invent them implicitly.
    "new ymaps.Placemark([0,0])",  # Outside the supported city area.
    "new ymaps.Placemark([51.16,71.42]);new ymaps.Placemark([51.17,71.43])",
])
def test_reject_missing_or_ambiguous_coordinates(marker):
    html = '<div class="teatr_ban_title">Музей Искусств</div><div class="teatr_ban_info">Улица Отырар, 1/2</div>' + marker
    with pytest.raises(ValueError):
        parse_page(html, {"url": "https://visitastana.kz/ru/about-city/what-to-see/muzey-iskusstv/", "category": "museum"}, "2026-10-17")


def test_changed_hours_require_review():
    html = '<div class="teatr_ban_title">Музей Искусств</div><div class="teatr_ban_info">Улица Отырар, 1/2</div>new ymaps.Placemark([51.16786,71.429207])'
    entry = {"url": "https://visitastana.kz/ru/about-city/what-to-see/muzey-iskusstv/", "category": "museum", "opening_hours": "Mo-Su 10:00-17:00", "hours_evidence": "Каждый день с 10:00 - 17:00"}
    with pytest.raises(ValueError, match="evidence changed"):
        parse_page(html, entry, "2026-10-17")


@pytest.mark.parametrize("time,status", [("12:30", "likely_open"), ("13:30", "likely_closed"), ("14:30", "confirmed_open")])
def test_official_museum_lunch_break(time, status):
    pois = json.loads((ROOT / "data/official_poi.json").read_text())
    museum = next(p for p in pois if p["id"] == "visitastana_mineralogicheskiy-geologicheskiy-muzey")
    assert open_status(museum["opening_hours"], f"2026-10-17T{time}")[0] == status


def test_appointment_only_museum_is_not_declared_open():
    pois = json.loads((ROOT / "data/official_poi.json").read_text())
    museum = next(p for p in pois if p["id"] == "visitastana_muzey-pozharnykh-i-spasateley")
    assert "предварительному звонку" in museum["description"]
    assert open_status(museum["opening_hours"], "2026-10-17T12:00")[0] == "unknown"

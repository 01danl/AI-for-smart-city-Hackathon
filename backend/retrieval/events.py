"""Temporary events (performances, matches, exhibitions) at venues from the places database.

Every event in data/events.json comes from an official listing and keeps its source URL and
the date it was checked. Venues must exist in data/poi.json, so distance, walking time and
the map use real coordinates. Events are shown with an explicit "temporary event" label and
a live status relative to the chosen time; past events are never shown.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .geo import haversine_m, walk_minutes
from .hours import resolve_now
from .schema import POI

ROOT = Path(__file__).resolve().parents[2]
EventStatus = Literal["ongoing", "starts_soon", "later"]

DEFAULT_DURATION_MIN = 120
SOON_MIN = 90           # "starts soon" if it begins within this many minutes
HORIZON_H = 8           # look this far ahead from the chosen time


class Event(BaseModel):
    id: str
    title: str
    kind: Literal["performance", "concert", "match", "exhibition", "festival", "show"]
    venue_poi_id: str
    start: str                     # local Astana time, ISO "YYYY-MM-DDTHH:MM"
    end: str | None = None         # for exhibitions: last day "YYYY-MM-DDT23:59"
    duration_min: int | None = None
    source_url: str
    checked_at: str                # date the listing was checked


class EventCandidate(BaseModel):
    event: Event
    venue: POI
    distance_m: int
    walk_min: int
    status: EventStatus
    minutes_to_start: int          # negative when it has already started


def events_path() -> Path:
    p = Path(os.getenv("EVENTS_PATH", "data/events.json"))
    return p if p.is_absolute() else ROOT / p


@lru_cache(maxsize=1)
def load_events() -> tuple[Event, ...]:
    path = events_path()
    if not path.exists():
        return ()
    return tuple(Event.model_validate(e) for e in json.loads(path.read_text(encoding="utf-8")))


def _local(iso: str, now: datetime) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=now.tzinfo)


def event_window(event: Event, now: datetime) -> tuple[datetime, datetime]:
    start = _local(event.start, now)
    if event.end:
        end = _local(event.end, now)
    else:
        end = start + timedelta(minutes=event.duration_min or DEFAULT_DURATION_MIN)
    return start, end


def events_near(
    lat: float,
    lon: float,
    pois: dict[str, POI],
    *,
    now_iso: str | None = None,
    max_distance_m: int = 3000,
    limit: int = 3,
) -> list[EventCandidate]:
    """Events within reach that are on now or start within the next few hours."""
    now = resolve_now(now_iso)
    out: list[EventCandidate] = []
    for event in load_events():
        venue = pois.get(event.venue_poi_id)
        if venue is None:
            continue  # never show an event whose venue we cannot place on the map
        start, end = event_window(event, now)
        if end <= now or start > now + timedelta(hours=HORIZON_H):
            continue
        dist = haversine_m(lat, lon, venue.lat, venue.lon)
        if dist > max_distance_m:
            continue
        to_start = int((start - now).total_seconds() // 60)
        status: EventStatus = "ongoing" if to_start <= 0 else "starts_soon" if to_start <= SOON_MIN else "later"
        out.append(EventCandidate(event=event, venue=venue, distance_m=round(dist), walk_min=walk_minutes(dist),
                                  status=status, minutes_to_start=to_start))
    order = {"starts_soon": 0, "ongoing": 1, "later": 2}
    out.sort(key=lambda c: (order[c.status], c.minutes_to_start, c.distance_m))
    return out[:limit]

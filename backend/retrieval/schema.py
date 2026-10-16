"""Shared data contract between retrieval (role A) and generation (role B).

Do not change field names without agreement from both roles (see CLAUDE.md).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

OpenStatus = Literal["confirmed_open", "likely_open", "unknown", "likely_closed"]


class Contacts(BaseModel):
    phone: str | None = None
    website: str | None = None


class POI(BaseModel):
    """A single point of interest as stored in data/poi.json."""

    id: str
    name: str
    lat: float
    lon: float
    description: str | None = None
    category: str
    tags: list[str] = Field(default_factory=list)
    opening_hours: str | None = None
    contacts: Contacts | None = None
    rating: float | None = None
    source: str = "osm"
    collected_at: str


class Candidate(BaseModel):
    """A POI returned by retrieve(), enriched with query-specific context."""

    poi: POI
    distance_m: int
    walk_min: int
    relevance: float = Field(ge=0.0, le=1.0)
    open_status: OpenStatus
    open_reason: str

"""Geo helpers: distance, walking time, privacy-safe coordinate rounding."""

from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_000
WALK_SPEED_M_PER_MIN = 75  # ~4.5 km/h
# Street routes are longer than straight lines; typical detour factor for city grids.
DETOUR_FACTOR = 1.3


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def walk_minutes(distance_m: float) -> int:
    """Estimated walking time in minutes (at least 1)."""
    return max(1, round(distance_m * DETOUR_FACTOR / WALK_SPEED_M_PER_MIN))


def round_coords(lat: float, lon: float, decimals: int = 3) -> tuple[float, float]:
    """Round coordinates to ~100 m (3 decimals) for logging. Never log raw user coords."""
    return round(lat, decimals), round(lon, decimals)

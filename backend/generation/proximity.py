"""Shared proximity policy for text and voice; the retrieval contract is unchanged."""

from __future__ import annotations

import re

from backend.retrieval import Candidate
from backend.retrieval.index import load_pois
from backend.retrieval.intent import parse_query_intent


def search_radii(limit: int) -> list[int]:
    return sorted({min(radius, limit) for radius in (300, 600, 1000, 2000, 5000, 10000, limit)})


def search_categories(query: str, requested: list[str] | None = None) -> list[str] | None:
    if requested:
        return requested
    categories = parse_query_intent(query).inferred_categories or None
    if categories:
        return categories
    for pattern, inferred in (
        (r"\brestaurants?\b|ресторан", ["restaurant"]),
        (r"\bcaf[eé]s?\b|\bcoffee\b", ["cafe"]),
        (r"\b(?:eat|eating|food|meal|lunch|dinner)\b|поесть|покушать|где поедим", ["restaurant", "cafe", "fast_food"]),
        (r"\bparks?\b", ["park"]),
        (r"посетить|посмотр|достопримеч|\b(?:visit|sightseeing|tourist|attractions?)\b",
         ["attraction", "monument", "park", "museum", "gallery", "artwork", "viewpoint", "place_of_worship"]),
        (r"прогул|погуля|серуен|\bwalk(?:ing)?\b", ["park", "attraction", "monument", "viewpoint", "artwork"]),
    ):
        if re.search(pattern, query, re.IGNORECASE):
            return inferred
    return None


def nearby_requested(query: str, categories: list[str] | None) -> bool:
    """Recognize proximity requests without replacing an explicitly requested real venue."""
    q = query.casefold()
    if re.search(r"не\s*(?:обязательно|нужно)\s*(?:рядом|близко)|not necessarily (?:near|close)", q):
        return False
    intent = parse_query_intent(query)
    # Keep the existing contextual ranking for concrete amenities, cuisine and timed routes.
    if set(intent.semantic_tags) - {"landmark"} or intent.time_budget_min is not None or intent.is_family_kids:
        return False
    explicit = re.search(r"рядом|поближе|ближайш|неподал[её]ку|недалеко|жақын|маңында|"
                         r"\b(?:nearby|nearest|closest|closer)\b|\bnear\s+me\b|\bclose\s+(?:by|to me)\b", q)
    if not explicit and not categories:
        return False
    generic_names = {"coffee", "cafe", "кафе", "кофе", "кофейня", "ресторан", "restaurant", "park", "парк", "museum", "музей"}
    for poi in load_pois().values():
        name = poi.name.casefold().strip()
        if categories and poi.category not in categories:
            continue
        if len(name) < 4 or name in generic_names:
            continue
        if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", q):
            return False
    return True


def nearby_candidates(candidates: list[Candidate], limit: int) -> list[Candidate]:
    """Prefer the closest nonempty ring; preserve truthful closed/unknown statuses."""
    closest_closed = []
    for radius in search_radii(limit):
        local = [item for item in candidates if item.distance_m <= radius]
        if any(item.open_status != "likely_closed" for item in local):
            return sorted((item for item in local if item.open_status != "likely_closed"), key=lambda item: item.distance_m)
        if local and not closest_closed:
            closest_closed = local
    return sorted(closest_closed, key=lambda item: item.distance_m)

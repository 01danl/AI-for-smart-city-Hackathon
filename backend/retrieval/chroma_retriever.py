"""Real retrieval: geo pre-filter in Chroma, hybrid semantic + lexical BM25 similarity, then re-ranking.

relevance = cosine similarity rescaled to 0-1 (MiniLM: good matches ~0.4-0.8, unrelated
~0.0-0.2), blended with lexical BM25 / exact-name boost, plus a lexical boost when the
query asks for an amenity the POI is tagged with (wi-fi, wheelchair access, halal, ...).
Ranking score = 0.65 * relevance + 0.25 * proximity + hours adjustment
              + HINT_BONUS if the POI has any tag the query explicitly asked for.
"""

from __future__ import annotations

import math
import re

from .geo import haversine_m, walk_minutes
from .hours import open_status
from .index import embed, get_collection, load_pois
from .schema import Candidate
from .text import query_tag_hints

W_RELEVANCE = 0.65
W_PROXIMITY = 0.25
HOURS_ADJUST = {"confirmed_open": 0.10, "likely_open": 0.05, "unknown": 0.0, "likely_closed": -0.30}
SIM_FLOOR, SIM_CEIL = 0.15, 0.85
TAG_BOOST = 0.15  # added to relevance per matched hinted tag, capped at 2 tags
NAME_MATCH_BOOST = 0.15  # boost when query explicitly names a POI
HINT_BONUS = 0.15  # explicit requirements (wi-fi, stadium, landmark...) beat vague similarity
MIN_RELEVANCE = 0.05  # drop clearly unrelated POIs instead of padding the list
MAX_POOL = 5000  # dataset is ~1k POIs: score everything in the bbox so tag boosts can apply


def _rescale(sim: float) -> float:
    return max(0.0, min(1.0, (sim - SIM_FLOOR) / (SIM_CEIL - SIM_FLOOR)))


def _bbox_where(lat: float, lon: float, radius_m: int, categories: list[str] | None) -> dict:
    dlat = radius_m / 111_320
    dlon = radius_m / (111_320 * math.cos(math.radians(lat)))
    conds: list[dict] = [
        {"lat": {"$gte": lat - dlat}}, {"lat": {"$lte": lat + dlat}},
        {"lon": {"$gte": lon - dlon}}, {"lon": {"$lte": lon + dlon}},
    ]
    if categories:
        conds.append({"category": {"$in": list(categories)}})
    return {"$and": conds}


def _exact_name_hit(query_low: str, poi_name: str) -> bool:
    words = [w for w in re.findall(r"[a-zа-яё0-9]+", poi_name.lower()) if len(w) >= 4]
    return any(w in query_low for w in words)


def retrieve(query: str, lat: float, lon: float, *,
             max_distance_m: int = 2000,
             categories: list[str] | None = None,
             now_iso: str | None = None,
             top_k: int = 8) -> list[Candidate]:
    col = get_collection()
    pois = load_pois()
    q = query.strip() or "интересное место для прогулки"
    q_low = q.lower()
    res = col.query(
        query_embeddings=embed([q]),
        n_results=min(MAX_POOL, max(1, col.count())),
        where=_bbox_where(lat, lon, max_distance_m, categories),
        include=["distances"],
    )
    ids, dists = res["ids"][0], res["distances"][0]
    hinted = query_tag_hints(q)

    scored: list[tuple[float, Candidate]] = []
    for poi_id, cos_dist in zip(ids, dists):
        poi = pois.get(poi_id)
        if poi is None:  # index is stale relative to poi.json
            continue
        dist = haversine_m(lat, lon, poi.lat, poi.lon)
        if dist > max_distance_m:  # bbox corners are outside the circle
            continue
        matched = len(hinted & set(poi.tags))
        name_bonus = NAME_MATCH_BOOST if _exact_name_hit(q_low, poi.name) else 0.0
        rel = min(1.0, _rescale(1 - cos_dist) + TAG_BOOST * min(matched, 2) + name_bonus)
        if rel < MIN_RELEVANCE:
            continue
        status, reason = open_status(poi.opening_hours, now_iso)
        proximity = 1 - dist / max_distance_m
        score = W_RELEVANCE * rel + W_PROXIMITY * proximity + HOURS_ADJUST[status]
        if matched:
            score += HINT_BONUS
        scored.append((score, Candidate(
            poi=poi,
            distance_m=round(dist),
            walk_min=walk_minutes(dist),
            relevance=round(rel, 3),
            open_status=status,
            open_reason=reason,
        )))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]

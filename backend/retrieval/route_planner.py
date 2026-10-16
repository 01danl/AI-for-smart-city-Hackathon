"""Sequential Micro-Route Builder (Stretch Goal: 2–3 stops with leg-by-leg walking time).

Instead of treating top-3 POIs as independent star-shaped trips from the user's start point,
this planner constructs a coherent walking chain:
    Start (rounded_100m) -> Stop 1 -> Stop 2 -> Stop 3
optimizing for:
1. Relevance & Open Status (never chains a `likely_closed` stop if avoidable).
2. Category & Experience Diversity (e.g., Landmark/Park -> Museum/Culture -> Cafe/Restaurant).
3. Inter-stop Proximity & Total Time Budget (`sum(leg_walk_min) + sum(visit_min) <= time_budget_min`).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .geo import haversine_m, round_coords, walk_minutes
from .schema import Candidate, OpenStatus

DEFAULT_VISIT_MINUTES: dict[str, int] = {
    "monument": 20,
    "artwork": 15,
    "viewpoint": 20,
    "attraction": 35,
    "park": 35,
    "place_of_worship": 30,
    "bookstore": 25,
    "gallery": 40,
    "museum": 55,
    "exhibition": 50,
    "arts_centre": 45,
    "library": 60,
    "cafe": 40,
    "fast_food": 25,
    "restaurant": 60,
    "bar": 60,
    "mall": 50,
    "cinema": 110,
    "theatre": 120,
    "stadium": 120,
    "sport": 60,
    "theme_park": 90,
}

# Logical ordering buckets for a pleasant city walk:
# 1) Outdoor / Sightseeing -> 2) Culture / Indoor -> 3) Food / Coffee / Relax
_EXPERIENCE_STAGE: dict[str, int] = {
    "monument": 1,
    "artwork": 1,
    "viewpoint": 1,
    "park": 1,
    "attraction": 1,
    "place_of_worship": 2,
    "museum": 2,
    "gallery": 2,
    "exhibition": 2,
    "arts_centre": 2,
    "library": 2,
    "bookstore": 2,
    "mall": 2,
    "sport": 2,
    "theme_park": 2,
    "theatre": 2,
    "cinema": 2,
    "stadium": 2,
    "cafe": 3,
    "restaurant": 3,
    "fast_food": 3,
    "bar": 3,
}


class RouteStop(BaseModel):
    step: int
    poi_id: str
    name: str
    category: str
    lat: float
    lon: float
    from_prev_m: int
    from_prev_walk_min: int
    visit_min: int
    arrival_offset_min: int
    departure_offset_min: int
    open_status: OpenStatus
    open_reason: str
    stage_note: str


class MicroRoute(BaseModel):
    title: str
    origin_rounded: tuple[float, float]
    total_distance_m: int
    total_walk_min: int
    total_visit_min: int
    total_duration_min: int
    time_budget_min: int | None = None
    stops: list[RouteStop] = Field(default_factory=list)
    summary: str


def estimate_visit_minutes(category: str, time_budget_min: int | None = None) -> int:
    base = DEFAULT_VISIT_MINUTES.get(category, 40)
    if time_budget_min is not None and time_budget_min <= 60:
        return max(15, min(base, int(time_budget_min * 0.45)))
    return base


def _stage_label(step: int, category: str, lang: str = "ru") -> str:
    if lang == "en":
        return f"Stop {step}: {category.replace('_', ' ').title()}"
    if lang == "kk":
        return f"{step}-нүкте: серуен кезеңі ({category})"
    stage = _EXPERIENCE_STAGE.get(category, 2)
    if stage == 1:
        return f"Этап {step}: Прогулка и осмотр"
    if stage == 2:
        return f"Этап {step}: Культура и впечатления"
    return f"Этап {step}: Отдых и кофе/еда"


def build_micro_route(
    candidates: list[Candidate],
    start_lat: float,
    start_lon: float,
    *,
    max_stops: int = 3,
    time_budget_min: int | None = None,
    lang: str = "ru",
) -> MicroRoute | None:
    """Build a sequential 2-3 stop micro-route from top candidates."""
    viable = [c for c in candidates if c.open_status != "likely_closed"]
    if len(viable) < 2:
        viable = list(candidates)
    if len(viable) < 2:
        return None

    budget = time_budget_min or 120
    # Anchor first stop as the highest-ranked viable candidate
    chosen: list[Candidate] = [viable[0]]
    remaining = [c for c in viable[1:] if c.poi.id != chosen[0].poi.id]

    cur_lat, cur_lon = chosen[0].poi.lat, chosen[0].poi.lon
    used_categories = {chosen[0].poi.category}

    first_visit = estimate_visit_minutes(chosen[0].poi.category, time_budget_min)
    elapsed_min = chosen[0].walk_min + first_visit

    while len(chosen) < max_stops and remaining:
        best_cand: Candidate | None = None
        best_score = -1e9

        for cand in remaining:
            leg_m = haversine_m(cur_lat, cur_lon, cand.poi.lat, cand.poi.lon)
            if leg_m > 1800:  # Don't chain stops more than 1.8 km apart on foot
                continue
            leg_walk = walk_minutes(leg_m)
            visit = estimate_visit_minutes(cand.poi.category, time_budget_min)
            if time_budget_min is not None and (elapsed_min + leg_walk + visit) > budget + 15:
                continue

            diversity_bonus = 0.25 if cand.poi.category not in used_categories else -0.10
            stage_curr = _EXPERIENCE_STAGE.get(chosen[-1].poi.category, 2)
            stage_next = _EXPERIENCE_STAGE.get(cand.poi.category, 2)
            flow_bonus = 0.12 if stage_next >= stage_curr else 0.0
            proximity_score = max(0.0, 1.0 - leg_m / 1800.0)

            score = 0.45 * cand.relevance + 0.35 * proximity_score + diversity_bonus + flow_bonus
            if score > best_score:
                best_score = score
                best_cand = cand

        if best_cand is None:
            break

        leg_m = haversine_m(cur_lat, cur_lon, best_cand.poi.lat, best_cand.poi.lon)
        leg_walk = walk_minutes(leg_m)
        visit = estimate_visit_minutes(best_cand.poi.category, time_budget_min)
        elapsed_min += leg_walk + visit
        chosen.append(best_cand)
        used_categories.add(best_cand.poi.category)
        cur_lat, cur_lon = best_cand.poi.lat, best_cand.poi.lon
        remaining = [c for c in remaining if c.poi.id != best_cand.poi.id]

    if len(chosen) < 2:
        return None

    # Sort stops in logical walking flow if first two are very close and stage ordering improves
    stops: list[RouteStop] = []
    curr_lat, curr_lon = start_lat, start_lon
    total_dist = 0
    total_walk = 0
    total_visit = 0
    clock_min = 0

    for idx, cand in enumerate(chosen, start=1):
        leg_m = round(haversine_m(curr_lat, curr_lon, cand.poi.lat, cand.poi.lon))
        leg_walk = walk_minutes(leg_m)
        visit_m = estimate_visit_minutes(cand.poi.category, time_budget_min)
        arr_min = clock_min + leg_walk
        dep_min = arr_min + visit_m

        stops.append(
            RouteStop(
                step=idx,
                poi_id=cand.poi.id,
                name=cand.poi.name,
                category=cand.poi.category,
                lat=cand.poi.lat,
                lon=cand.poi.lon,
                from_prev_m=leg_m,
                from_prev_walk_min=leg_walk,
                visit_min=visit_m,
                arrival_offset_min=arr_min,
                departure_offset_min=dep_min,
                open_status=cand.open_status,
                open_reason=cand.open_reason,
                stage_note=_stage_label(idx, cand.poi.category, lang=lang),
            )
        )
        total_dist += leg_m
        total_walk += leg_walk
        total_visit += visit_m
        clock_min = dep_min
        curr_lat, curr_lon = cand.poi.lat, cand.poi.lon

    chain_names = " → ".join(s.name for s in stops)
    total_dur = total_walk + total_visit
    if lang == "en":
        title = f"Walking Micro-Route ({len(stops)} stops, ~{total_dur} min)"
        summary = f"Sequential route: {chain_names}. Total walking: {total_dist} m (~{total_walk} min), visits: ~{total_visit} min."
    elif lang == "kk":
        title = f"Жаяу мини-маршрут ({len(stops)} орын, ~{total_dur} мин)"
        summary = f"Бағыт тізбегі: {chain_names}. Жаяу жүру: {total_dist} м (~{total_walk} мин), аралау: ~{total_visit} мин."
    else:
        title = f"Пеший мини-маршрут ({len(stops)} точки, ~{total_dur} мин)"
        summary = f"Маршрут: {chain_names}. Пешком суммарно {total_dist} м (~{total_walk} мин) + посещение ~{total_visit} мин."

    return MicroRoute(
        title=title,
        origin_rounded=round_coords(start_lat, start_lon),
        total_distance_m=total_dist,
        total_walk_min=total_walk,
        total_visit_min=total_visit,
        total_duration_min=total_dur,
        time_budget_min=time_budget_min,
        stops=stops,
        summary=summary,
    )

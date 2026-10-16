"""Retrieval stub for role B while the real index is being built.

Same signature and return type as the real `retrieve()`. Uses a small hardcoded set of
POIs around the Left Bank of Astana (coordinates approximate, `source="stub"`) and naive
keyword matching instead of embeddings.
"""

from __future__ import annotations

import re

from .geo import haversine_m, walk_minutes
from .hours import open_status
from .schema import POI, Candidate

_COLLECTED = "2026-10-16"


def _poi(id, name, lat, lon, category, tags, description=None, hours=None):
    return POI(
        id=f"stub_{id}",
        name=name,
        lat=lat,
        lon=lon,
        description=description,
        category=category,
        tags=tags,
        opening_hours=hours,
        source="stub",
        collected_at=_COLLECTED,
    )


STUB_POIS: list[POI] = [
    _poi("baiterek", "Монумент Байтерек", 51.1283, 71.4305, "attraction",
         ["landmark", "viewpoint", "architecture", "tourist"],
         "Символ Астаны, смотровая площадка на высоте 97 м", "Mo-Su 10:00-21:00"),
    _poi("khan_shatyr", "ТРЦ Хан Шатыр", 51.1326, 71.4037, "mall",
         ["shopping", "food_court", "indoor", "family", "wheelchair"],
         "Торгово-развлекательный центр в форме шатра", "Mo-Su 10:00-22:00"),
    _poi("opera", "Астана Опера", 51.1357, 71.4106, "theatre",
         ["culture", "music", "architecture", "evening"],
         "Театр оперы и балета", None),
    _poi("nurzhol", "Бульвар Нуржол", 51.1290, 71.4250, "park",
         ["walk", "outdoor", "fountains", "family", "free", "wheelchair"],
         "Пешеходный бульвар между Хан Шатыром и Ак Ордой", "24/7"),
    _poi("lovers_park", "Парк влюблённых", 51.1265, 71.4190, "park",
         ["walk", "outdoor", "quiet", "free", "romantic"],
         None, None),
    _poi("national_museum", "Национальный музей Республики Казахстан", 51.1182, 71.4692, "museum",
         ["culture", "history", "indoor", "wheelchair"],
         "Крупнейший музей страны: история, этнография, Золотой зал", "Tu-Su 10:00-18:00; Mo off"),
    _poi("pyramid", "Дворец мира и согласия", 51.1230, 71.4637, "attraction",
         ["architecture", "landmark", "culture", "indoor"],
         "Пирамида архитектора Нормана Фостера", "Tu-Su 10:00-18:30"),
    _poi("hazret_sultan", "Мечеть Хазрет Султан", 51.1256, 71.4718, "place_of_worship",
         ["architecture", "landmark", "quiet"],
         "Одна из крупнейших мечетей Центральной Азии", None),
    _poi("cafe_quiet", "Тестовая кофейня «Тишина»", 51.1295, 71.4280, "cafe",
         ["coffee", "quiet", "wifi", "power_outlets", "work"],
         "Небольшая кофейня с розетками у столиков", "Mo-Su 08:00-22:00"),
    _poi("cafe_bulvar", "Тестовая кофейня «Бульвар»", 51.1305, 71.4220, "cafe",
         ["coffee", "dessert", "terrace", "wifi"],
         None, "Mo-Fr 08:00-20:00; Sa,Su 10:00-20:00"),
    _poi("restaurant_kz", "Тестовый ресторан казахской кухни", 51.1270, 71.4320, "restaurant",
         ["kazakh_cuisine", "dinner", "group", "family"],
         "Бешбармак, баурсаки, большие столы для компаний", "Mo-Su 12:00-00:00"),
    _poi("presidential_park", "Президентский парк", 51.1215, 71.4500, "park",
         ["walk", "outdoor", "family", "free", "bike", "wheelchair"],
         "Большой парк на набережной Есиля", None),
    _poi("embankment", "Набережная реки Есиль", 51.1240, 71.4400, "park",
         ["walk", "outdoor", "river", "free", "sunset"],
         None, "24/7"),
]

# Minimal ru/en -> tag synonyms so the stub behaves sensibly on natural-language queries.
_SYNONYMS: dict[str, list[str]] = {
    "кофе": ["coffee", "cafe"], "кофейня": ["coffee", "cafe"], "тихо": ["quiet"],
    "тихое": ["quiet"], "розетка": ["power_outlets"], "розеткой": ["power_outlets"],
    "работать": ["work", "wifi"], "прогулка": ["walk"], "погулять": ["walk"],
    "парк": ["park"], "музей": ["museum"], "история": ["history"], "семья": ["family"],
    "дети": ["family"], "детьми": ["family"], "поесть": ["restaurant", "food_court"],
    "ужин": ["dinner", "restaurant"], "бесплатно": ["free"], "культура": ["culture"],
    "архитектура": ["architecture"], "вид": ["viewpoint"], "закат": ["sunset"],
    "река": ["river"], "коляска": ["wheelchair"], "компания": ["group"],
}


def _query_terms(query: str) -> set[str]:
    words = set(re.findall(r"\w+", query.lower()))
    terms = set(words)
    for w in words:
        terms.update(_SYNONYMS.get(w, []))
    return terms


def _relevance(poi: POI, terms: set[str]) -> float:
    if not terms:
        return 0.5
    haystack = set(poi.tags) | {poi.category}
    text = f"{poi.name} {poi.description or ''}".lower()
    hits = sum(1 for t in terms if t in haystack or (len(t) > 3 and t in text))
    return min(1.0, hits / max(1, min(len(terms), 3)))


def retrieve(query: str, lat: float, lon: float, *,
             max_distance_m: int = 2000,
             categories: list[str] | None = None,
             now_iso: str | None = None,
             top_k: int = 8) -> list[Candidate]:
    terms = _query_terms(query)
    results: list[tuple[float, Candidate]] = []
    for poi in STUB_POIS:
        if categories and poi.category not in categories:
            continue
        dist = haversine_m(lat, lon, poi.lat, poi.lon)
        if dist > max_distance_m:
            continue
        status, reason = open_status(poi.opening_hours, now_iso)
        rel = _relevance(poi, terms)
        proximity = 1 - dist / max_distance_m
        score = 0.7 * rel + 0.3 * proximity - (0.3 if status == "likely_closed" else 0.0)
        results.append((score, Candidate(
            poi=poi,
            distance_m=round(dist),
            walk_min=walk_minutes(dist),
            relevance=round(rel, 3),
            open_status=status,
            open_reason=reason,
        )))
    results.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in results[:top_k]]

"""Senior Query Intent Parser, Anchor Landmark Resolver & Constraint Extractor.

Extracts structured context from natural-language user queries (RU / KK / EN):
- Anchor landmarks in Astana (e.g., "возле Mega Silk Way", "у здания Экспо 2017", "возле Байтерека")
- Time budget in minutes (e.g., "на 30–60 минут", "1–2 часа", "2 часа перед вылетом")
- Weather / indoor constraints (e.g., "холодно", "мороз", "ветер", "дождь")
- Group / family / accessibility signals
- Language auto-detection (Russian, Kazakh, English)
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from .hours import is_late_night
from .text import query_tag_hints

LangCode = Literal["ru", "kk", "en"]
WeatherContext = Literal["normal", "cold_windy", "rain_snow"]


class AnchorLandmark(BaseModel):
    """Known Astana landmark detected in the user's query text."""

    key: str
    name: str
    lat: float
    lon: float
    matched_phrase: str


class QueryIntent(BaseModel):
    """Structured representation of user constraints extracted from natural language."""

    raw_query: str
    detected_lang: LangCode = "ru"
    time_budget_min: int | None = None
    inferred_categories: list[str] = Field(default_factory=list)
    inferred_tags: list[str] = Field(default_factory=list)
    semantic_tags: list[str] = Field(default_factory=list)
    weather_context: WeatherContext = "normal"
    anchor_landmark: AnchorLandmark | None = None
    wants_route: bool = False
    is_night_query: bool = False
    is_family_kids: bool = False
    is_accessible_query: bool = False
    group_size_hint: int = 1


# Coordinates match real OSM POIs in data/poi.json for Astana
ASTANA_ANCHORS: list[tuple[str, str, float, float, list[str]]] = [
    (
        "mega_silk_way",
        "ТРЦ Mega Silk Way",
        51.08909,
        71.40733,
        ["mega silk way", "мега силк вей", "мега силк уэй", "мега силк", "silk way"],
    ),
    (
        "expo_2017",
        "Выставочный комплекс EXPO 2017 (Нур Алем)",
        51.092788,
        71.421482,
        ["экспо 2017", "экспо-2017", "здании экспо", "возле экспо", "у экспо", "expo 2017", "нур алем", "nur alem",
         "нұр әлем"],
    ),
    (
        "baiterek",
        "Монумент Байтерек",
        51.128288,
        71.430455,
        ["байтерек", "бәйтерек", "baiterek", "bayterek"],
    ),
    (
        "khan_shatyr",
        "ТРЦ Хан Шатыр",
        51.132509,
        71.403873,
        ["хан шатыр", "khan shatyr", "хан-шатыр"],
    ),
    (
        "astana_opera",
        "Театр Астана Опера",
        51.1357,
        71.4106,
        ["астана опера", "astana opera"],
    ),
    (
        "botanical_garden",
        "Ботанический сад Астаны",
        51.106165,
        71.416571,
        ["ботанический сад", "ботанического сада", "botanical garden", "ботаникалық бақ"],
    ),
    (
        "national_museum",
        "Национальный музей РК",
        51.11823,
        71.469382,
        ["национальный музей", "нацмузей", "national museum", "ұлттық музей"],
    ),
    (
        "hazret_sultan",
        "Мечеть Хазрет Султан",
        51.1256,
        71.4718,
        ["хазрет султан", "әзірет сұлтан", "hazret sultan"],
    ),
    (
        "abu_dhabi_plaza",
        "Abu Dhabi Plaza",
        51.121912,
        71.427645,
        ["абу даби плаза", "абу-даби плаза", "abu dhabi plaza"],
    ),
    (
        "old_center",
        "Старый центр (Правый берег)",
        51.1694,
        71.4491,
        ["старый центр", "правый берег", "на правом берегу", "ескі орталық", "оң жағалау",
         "old town", "old center", "right bank"],
    ),
]

_KAZAKH_CHARS_RE = re.compile(r"[әіңғүұқөһӘІҢҒҮҰҚӨҺ]")
_KAZAKH_WORDS = {"қайда", "серуендеу", "жақын", "баламен", "кофехана", "мұражай", "саябақ", "тамақ",
                 "барайын", "керек", "маған", "менің", "орын"}
# Short Kazakh weather words are matched as whole words: "қар" (snow) is also the start
# of "қарау" (to look), "жел" (wind) of other words.
_KK_COLD_RE = re.compile(r"\b(?:суық|аяз|жел|боран)\b|жылын")
_KK_WET_RE = re.compile(r"\b(?:жаңбыр\w*|қар)\b|қар жау")
_ENGLISH_WORDS = {"where", "walk", "quiet", "place", "coffee", "museum", "park", "near", "tourist", "book"}

_CATEGORY_STEMS: dict[str, list[str]] = {
    "кофе": ["cafe"],
    "кафе": ["cafe"],
    "кофейн": ["cafe"],
    "coffee": ["cafe"],
    "ресторан": ["restaurant"],
    "ужин": ["restaurant"],
    "обед": ["restaurant", "cafe"],
    "фастфуд": ["fast_food"],
    "перекус": ["fast_food", "cafe"],
    "бар": ["bar"],
    "паб": ["bar"],
    "музей": ["museum", "gallery", "exhibition"],
    "museum": ["museum", "gallery"],
    "выставк": ["exhibition", "gallery", "museum"],
    "галере": ["gallery", "museum"],
    "парк": ["park"],
    "сквер": ["park"],
    "сад": ["park"],
    "театр": ["theatre"],
    "спектакл": ["theatre"],
    "кино": ["cinema"],
    "библиотек": ["library"],
    "книг": ["library", "bookstore"],
    "read a book": ["library", "bookstore"],
    "мечет": ["place_of_worship"],
    "храм": ["place_of_worship"],
    "стадион": ["stadium"],
    "футбол": ["stadium"],
    "матч": ["stadium"],
    "трц": ["mall"],
    "шопинг": ["mall"],
}


def detect_language(query: str, explicit_lang: str | None = None) -> LangCode:
    if explicit_lang in ("ru", "kk", "en"):
        return explicit_lang  # type: ignore[return-value]
    if _KAZAKH_CHARS_RE.search(query):
        return "kk"
    words = set(re.findall(r"[a-zа-яёәіңғүұқөһ]+", query.lower()))
    if words & _KAZAKH_WORDS:
        return "kk"
    ascii_letters = sum(1 for ch in query if "a" <= ch.lower() <= "z")
    cyr_letters = sum(1 for ch in query if "а" <= ch.lower() <= "я" or ch.lower() == "ё")
    if ascii_letters > cyr_letters and (words & _ENGLISH_WORDS or ascii_letters >= 6):
        return "en"
    return "ru"


def extract_time_budget_minutes(query: str) -> int | None:
    """Extract walking/visit time budget in minutes from natural language."""
    q = query.lower().replace("—", "-").replace("–", "-")

    # Range in minutes: "30-60 минут", "40-60 мин"
    m = re.search(r"(\d{1,3})\s*-\s*(\d{1,3})\s*(?:минут|мин|min)", q)
    if m:
        return int(m.group(2))

    # Range in hours: "1-2 часа", "1-2 hours"
    m = re.search(r"(\d{1,2})\s*-\s*(\d{1,2})\s*(?:час|ч\b|сағат|hour|hr)", q)
    if m:
        return int(m.group(2)) * 60

    # Single minutes: "на 45 минут", "за 30 мин"
    m = re.search(r"(\d{1,3})\s*(?:минут|мин|min)", q)
    if m:
        return int(m.group(1))

    # Single hours: "2 часа перед вылетом", "на 3 часа"
    m = re.search(r"(\d{1,2})\s*(?:часа|час|часов|сағат|hours|hour)", q)
    if m:
        return int(m.group(1)) * 60

    if any(p in q for p in ("пару часов", "два часа", "couple of hours", "two hours", "екі сағат")):
        return 120
    if any(p in q for p in ("полчаса", "полчасика", "half an hour", "жарты сағат")):
        return 30
    if "час " in f"{q} " or "на час" in q or "an hour" in q or "бір сағат" in q:
        return 60
    return None


def resolve_anchor_landmark(query: str) -> AnchorLandmark | None:
    q = query.lower()
    for key, name, lat, lon, phrases in ASTANA_ANCHORS:
        for phrase in phrases:
            if phrase in q:
                return AnchorLandmark(
                    key=key,
                    name=name,
                    lat=lat,
                    lon=lon,
                    matched_phrase=phrase,
                )
    return None


def parse_query_intent(
    query: str,
    *,
    explicit_lang: str | None = None,
    now_iso: str | None = None,
    explicit_accessible: bool = False,
    explicit_group_size: int = 1,
    explicit_weather: str | None = None,
) -> QueryIntent:
    q = query.lower()
    lang = detect_language(query, explicit_lang)
    time_budget = extract_time_budget_minutes(query)
    anchor = resolve_anchor_landmark(query)

    hinted_tags = sorted(query_tag_hints(query))
    semantic_tags = list(hinted_tags)

    if any(w in q for w in ("тихо", "тихое", "тишин", "спокойн", "quiet", "тыныш")):
        if "quiet" not in semantic_tags:
            semantic_tags.append("quiet")
    if any(w in q for w in ("розет", "ноутбук", "поработ", "power", "laptop", "outlet")):
        if "power_outlets" not in semantic_tags:
            semantic_tags.append("power_outlets")
    if any(w in q for w in ("необычн", "интересн", "интерактив", "фото", "вид", "unusual")):
        if "unusual" not in semantic_tags:
            semantic_tags.append("unusual")

    inferred_cats: list[str] = []
    for stem, cats in _CATEGORY_STEMS.items():
        if stem in q:
            for c in cats:
                if c not in inferred_cats:
                    inferred_cats.append(c)

    # Weather detection (critical for Astana climate)
    weather: WeatherContext = "normal"
    if explicit_weather in ("normal", "cold_windy", "rain_snow"):
        weather = explicit_weather  # type: ignore[assignment]
    elif any(w in q for w in ("мороз", "холод", "ветер", "ветрен", "буран", "замёрз", "погреться", "cold", "wind",
                              "freezing")) or _KK_COLD_RE.search(q):
        weather = "cold_windy"
    elif any(w in q for w in ("дожд", "снег", "ливень", "rain", "snow")) or _KK_WET_RE.search(q):
        weather = "rain_snow"

    if weather in ("cold_windy", "rain_snow") and "indoor" not in hinted_tags:
        hinted_tags.append("indoor")
        if "indoor" not in semantic_tags:
            semantic_tags.append("indoor")

    is_family = any(w in q for w in ("ребён", "ребен", "детск", "детьми", "дети", "семь", "family", "kids", "бала"))
    if is_family and "family" not in semantic_tags:
        semantic_tags.append("family")

    is_acc = explicit_accessible or ("wheelchair" in hinted_tags) or any(
        w in q for w in ("коляск", "маломобил", "безбарьер", "пандус", "wheelchair", "accessible")
    )

    night_query = is_late_night(now_iso) or any(
        w in q for w in ("ночн", "ночью", "поздн", "после полуночи", "night", "late")
    )
    if night_query and "24_7" not in semantic_tags:
        semantic_tags.append("night_activity")

    group_size = max(1, explicit_group_size)
    if group_size == 1:
        m_grp = re.search(r"(?:групп\w*|компан\w*)\s*(?:из\s*)?(\d{1,2})", q)
        if m_grp:
            group_size = int(m_grp.group(1))
        elif is_family:
            group_size = 3

    wants_route = bool(
        (time_budget and time_budget >= 45)
        or any(w in q for w in ("прогул", "погулять", "маршрут", "что посмотреть", "турист", "вылет", "walk", "route"))
    )

    return QueryIntent(
        raw_query=query,
        detected_lang=lang,
        time_budget_min=time_budget,
        inferred_categories=inferred_cats,
        inferred_tags=hinted_tags,
        semantic_tags=semantic_tags,
        weather_context=weather,
        anchor_landmark=anchor,
        wants_route=wants_route,
        is_night_query=night_query,
        is_family_kids=is_family,
        is_accessible_query=is_acc,
        group_size_hint=group_size,
    )

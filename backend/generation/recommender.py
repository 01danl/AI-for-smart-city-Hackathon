"""Senior RAG Recommender Pipeline (Role B) with Anti-Hallucination & Smart Fallback.

Combines:
1. Query Intent & Landmark Anchor Resolution (`parse_query_intent`)
2. Hybrid Geosemantical Retrieval (`retrieve`) + Context Re-ranking (Time Budget, Weather, Accessibility, Visit Window)
3. OpenRouter LLM Generation (`complete`) with strict JSON validation & `get_poi` verification
4. Resilient Smart Deterministic Fallback (100% offline/rate-limit safety for live hackathon demo)
5. Sequential 2–3 Stop Micro-Route Builder (`build_micro_route`)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Literal

from pydantic import BaseModel, Field
from unidecode import unidecode

from backend.retrieval import (
    BACKEND,
    Candidate,
    MicroRoute,
    OpenStatus,
    build_micro_route,
    estimate_visit_minutes,
    get_poi,
    parse_query_intent,
    retrieve,
)
from backend.retrieval.geo import round_coords
from backend.retrieval.hours import check_visit_window
from backend.retrieval.index import load_pois
from backend.retrieval.intent import QueryIntent

from .client import complete
from .prompts import build_rag_messages
from .proximity import nearby_candidates, nearby_requested, search_categories

log = logging.getLogger(__name__)

SourceType = Literal["poi_data", "generated_idea"]


class UsedData(BaseModel):
    """Transparency metadata showing exactly what signals shaped the response."""

    location: str = "rounded_100m"
    rounded_coords: tuple[float, float] = (51.128, 71.431)
    filters: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    anchor_resolved: str | None = None
    time_budget_min: int | None = None
    weather_context: str = "normal"
    retrieval_backend: str = BACKEND
    generation_mode: str = "smart_fallback"
    candidates_evaluated: int = 0


class RecommendationItem(BaseModel):
    """Single recommended POI matching CLAUDE.md contract + senior explainability."""

    poi_id: str
    name: str
    category: str = ""
    lat: float = 0.0
    lon: float = 0.0
    distance_m: int
    walk_min: int
    why: str
    visit_min: int
    what_to_do: str
    open_status: OpenStatus
    confidence_note: str
    source: SourceType
    facts_poi_data: list[str] = Field(default_factory=list)
    generated_plan_note: str = ""


class RecommendRequest(BaseModel):
    """Input schema for POST /recommend (backward-compatible with CLAUDE.md)."""

    query: str
    lat: float = 51.1283
    lon: float = 71.4305
    max_distance_m: int = Field(default=2000, ge=100, le=20000)
    categories: list[str] | None = None
    lang: Literal["ru", "kk", "en"] = "ru"
    group_size: int = Field(default=1, ge=1, le=20)
    accessible: bool = False
    weather: Literal["normal", "cold_windy", "rain_snow"] = "normal"
    time_budget_min: int | None = None
    include_route: bool = True
    auto_anchor: bool = True
    now_iso: str | None = None


class RecommendResponse(BaseModel):
    """Output schema for POST /recommend (backward-compatible with CLAUDE.md)."""

    used_data: UsedData
    recommendations: list[RecommendationItem]
    route: MicroRoute | None = None
    disclaimer: str


# Category-specific rich templates for deterministic fallback & hybrid enrichment
_CATEGORY_LABELS_RU: dict[str, str] = {
    "cafe": "Кофейня / кафе",
    "restaurant": "Ресторан",
    "fast_food": "Быстрый перекус",
    "bar": "Бар / вечернее место",
    "mall": "Торгово-развлекательный центр",
    "park": "Парк / прогулочная зона",
    "sport": "Спортивный комплекс",
    "monument": "Монумент / знаковое место",
    "attraction": "Достопримечательность",
    "viewpoint": "Смотровая площадка",
    "place_of_worship": "Религиозный памятник архитектуры",
    "artwork": "Арт-объект",
    "cinema": "Кинотеатр",
    "theatre": "Театр",
    "library": "Библиотека",
    "museum": "Музей",
    "bookstore": "Книжный магазин",
    "arts_centre": "Центр современного искусства",
    "stadium": "Стадион / арена",
    "gallery": "Художественная галерея",
    "theme_park": "Парк развлечений",
    "exhibition": "Выставочный комплекс",
}


def _hours_note(status: str, lang: str, original: str) -> str:
    """Localize schedule confidence without changing the verified status."""
    notes = {
        "en": {
            "unknown": "Opening hours are unavailable or could not be interpreted; check before visiting.",
            "confirmed_open": "Open according to the recorded schedule; hours may change.",
            "likely_open": "May be open, but closing time or a full visit needs checking.",
            "likely_closed": "Likely closed according to the recorded schedule; check before leaving.",
        },
        "kk": {
            "unknown": "Жұмыс уақыты белгісіз немесе кестені анықтау мүмкін болмады; барар алдында нақтылаңыз.",
            "confirmed_open": "Дерекқордағы кесте бойынша ашық; жұмыс уақыты өзгеруі мүмкін.",
            "likely_open": "Ашық болуы мүмкін, бірақ жабылу уақытын және сапар ұзақтығын тексеріңіз.",
            "likely_closed": "Дерекқордағы кесте бойынша жабық болуы мүмкін; жолға шығар алдында нақтылаңыз.",
        },
    }
    return notes.get(lang, {}).get(status, original)


def _build_facts_list(cand: Candidate, lang: str = "ru") -> list[str]:
    p = cand.poi
    cat_ru = _CATEGORY_LABELS_RU.get(p.category, p.category)
    if lang == "en":
        facts = [
            f"Category: {p.category} ({cand.distance_m} m, ~{cand.walk_min} min walk)",
            f"Hours status: {_hours_note(cand.open_status, lang, cand.open_reason)}",
        ]
        if p.tags:
            facts.append(f"OSM tags: {', '.join(p.tags[:6])}")
    elif lang == "kk":
        facts = [
            f"Санат: {cat_ru} ({cand.distance_m} м, жаяу ~{cand.walk_min} мин)",
            f"Жұмыс уақыты: {_hours_note(cand.open_status, lang, cand.open_reason)}",
        ]
        if p.tags:
            facts.append(f"OSM тегтері: {', '.join(p.tags[:6])}")
    else:
        facts = [
            f"Категория: {cat_ru} ({cand.distance_m} м, ~{cand.walk_min} мин пешком)",
            f"Статус расписания: {cand.open_reason}",
        ]
        if p.tags:
            facts.append(f"Подтверждённые теги OSM: {', '.join(p.tags[:6])}")
    if p.contacts and p.contacts.website:
        facts.append(f"Сайт: {p.contacts.website}")
    if p.contacts and p.contacts.phone:
        facts.append(f"Телефон: {p.contacts.phone}")
    if p.source.startswith("https://visitastana.kz/"):
        facts.append(f"Источник: {p.source}")
        if p.description:
            facts.append(p.description)
    return facts


def _craft_deterministic_item(
    cand: Candidate,
    intent: QueryIntent,
    *,
    lang: str = "ru",
    group_size: int = 1,
    accessible: bool = False,
    now_iso: str | None = None,
) -> RecommendationItem:
    p = cand.poi
    display_name = unidecode(p.name) if lang == "en" else p.name
    visit_min = estimate_visit_minutes(p.category, intent.time_budget_min)
    adj_status, adj_reason, mins_left = check_visit_window(
        p.opening_hours, now_iso, walk_min=cand.walk_min, visit_min=visit_min
    )
    tags_set = set(p.tags)
    q_low = intent.raw_query.lower()

    # Determine whether recommendation relies strictly on POI facts or includes AI-generated ideas
    has_unverified_assumption = (
        ("power_outlets" in intent.semantic_tags and "power_outlets" not in tags_set)
        or ("quiet" in intent.semantic_tags and "quiet" not in tags_set and p.category != "library")
        or ("unusual" in intent.semantic_tags)
    )
    source: SourceType = "generated_idea" if has_unverified_assumption else "poi_data"

    if lang == "en":
        why = (
            f"Located {cand.distance_m} m away (~{cand.walk_min} min walk), matching your request "
            f"for {p.category.replace('_', ' ')}."
        )
        if "wifi" in tags_set:
            why += " Tagged with Wi-Fi in OSM."
        if "wheelchair" in tags_set:
            why += " Wheelchair accessible according to OSM."
        what_to_do = (
            f"Spend ~{visit_min} min exploring {display_name}. "
            + ("Check power outlet availability at corner tables upon arrival." if "power_outlets" in intent.semantic_tags else "Enjoy the local atmosphere and nearby walking boulevard.")
        )
        if p.category == "place_of_worship":
            what_to_do += " Please observe modest dress code and prayer times."
        conf_note = _hours_note(adj_status, lang, adj_reason)
        if adj_status == "unknown":
            conf_note += " — we never claim 'open now' when schedule data is missing."
        gen_note = (
            "Suggested visit plan and atmosphere notes are generated by AI [generated_idea]; distance is calculated locally and tags come from the POI database [poi_data]."
        )
    elif lang == "kk":
        why = (
            f"Сізден {cand.distance_m} м қашықтықта (жаяу ~{cand.walk_min} мин) орналасқан сұранысқа сай орын ({_CATEGORY_LABELS_RU.get(p.category, p.category)})."
        )
        if "wifi" in tags_set:
            why += " OSM деректері бойынша Wi-Fi бар."
        what_to_do = f"Шамамен {visit_min} минут бөліп, «{p.name}» орнын аралаңыз."
        if p.category == "place_of_worship":
            what_to_do += " Киім үлгісі мен намаз/ғибадат уақытына құрметпен қарауды сұраймыз."
        conf_note = _hours_note(adj_status, lang, adj_reason)
        gen_note = "Іс-қимыл жоспары ЖИ арқылы ұсынылды [generated_idea], ал қашықтық пен тегтер POI базасынан алынды [poi_data]."
    else:
        # Russian (primary)
        cat_label = _CATEGORY_LABELS_RU.get(p.category, p.category)
        why_parts = [f"{cat_label} в {cand.distance_m} м от вас (~{cand.walk_min} мин пешком)"]
        if "wifi" in tags_set and ("розет" in q_low or "ноутбук" in q_low or "кофе" in q_low):
            why_parts.append("в базе OSM подтверждён тег `wifi`, подходит для спокойной паузы с ноутбуком")
        elif "landmark" in tags_set:
            why_parts.append("знаковая точка притяжения Астаны (`landmark` в Wikidata/OSM)")
        elif {"cuisine_kazakh", "cuisine_regional"} & tags_set:
            why_parts.append("отмечена национальная/региональная кухня в данных OSM")
        elif "wheelchair" in tags_set:
            why_parts.append("в OSM отмечена безбарьерная доступность (`wheelchair`)")
        elif "outdoor" in tags_set:
            why_parts.append("открытое пространство для пешей прогулки (`outdoor`)")
        elif "indoor" in tags_set:
            why_parts.append("комфортное пространство в помещении (`indoor`) — защищает от астанинского ветра")

        if has_unverified_assumption and "power_outlets" in intent.semantic_tags and "power_outlets" not in tags_set:
            why_parts.append("наличие розеток у столиков — предположение модели (`generated_idea`, в OSM размечен только `wifi`)")

        why = "; ".join(why_parts) + "."

        # Action plan ("что делать там")
        if p.category == "cafe":
            what_to_do = (
                f"Заложить {visit_min} мин: взять кофе, выбрать столик у стены "
                f"{'(уточнить у бариста столик с розеткой) ' if 'power_outlets' in intent.semantic_tags else ''}"
                f"и спокойно поработать или отдохнуть."
            )
        elif p.category in ("museum", "exhibition", "gallery", "arts_centre"):
            what_to_do = (
                f"План на {visit_min} мин: осмотреть основную экспозицию «{p.name}» "
                f"{'с акцентом на интерактивные залы для детей' if intent.is_family_kids else 'и ключевые залы'}, "
                f"затем сделать паузу в холле или кофейне рядом."
            )
        elif p.category == "park":
            what_to_do = (
                f"Прогулка на {visit_min} мин по аллеям «{p.name}»: пройтись пешком в комфортном темпе, "
                f"сделать фото на фоне архитектуры Левого берега."
            )
            if intent.is_night_query:
                what_to_do += " В позднее время придерживайтесь центральных освещённых аллей."
        elif p.category == "place_of_worship":
            what_to_do = (
                f"Осмотр архитектуры и интерьеров ({visit_min} мин). "
                f"Важно соблюдать этикет посещения: закрытая скромная одежда, тишина и уважение к времени молитв."
            )
        elif p.category in ("monument", "attraction", "artwork", "viewpoint"):
            what_to_do = (
                f"Остановка на {visit_min} мин: осмотреть «{p.name}», сделать панорамные снимки "
                f"и продолжить маршрут по пешеходной зоне рядом."
            )
        elif p.category in ("restaurant", "fast_food", "bar"):
            group_note = f" (удобно для компании из {group_size} чел.)" if group_size >= 3 else ""
            what_to_do = (
                f"Визит на {visit_min} мин{group_note}: заказать блюда кухни заведения "
                f"{'(бешбармак, баурсаки и чай) ' if {'cuisine_kazakh', 'cuisine_regional'} & tags_set else ''}"
                f"и отдохнуть в тепле."
            )
        else:
            what_to_do = f"Посещение на ~{visit_min} мин: познакомиться с локацией «{p.name}» и оценить инфраструктуру внутри."

        conf_note = adj_reason
        if adj_status == "unknown":
            conf_note = f"{adj_reason}. Мы не утверждаем «открыто сейчас» без подтверждённого графика — проверьте перед выходом."
        elif mins_left is not None and mins_left <= 60:
            conf_note = f"{adj_reason} (осталось ~{mins_left} мин до закрытия)."

        if source == "generated_idea":
            gen_note = (
                "Идея сценария посещения и предположения об удобствах (розетки/тишина/атмосфера) сгенерированы моделью [generated_idea]; "
                "координаты, категория и теги взяты из базы POI [poi_data]."
            )
        else:
            gen_note = (
                "Выбор основан на полях базы POI с указанным источником [poi_data]; микро-план действий оформлен помощником [generated_idea]."
            )

    return RecommendationItem(
        poi_id=p.id,
        name=display_name,
        category=p.category,
        lat=p.lat,
        lon=p.lon,
        distance_m=cand.distance_m,
        walk_min=cand.walk_min,
        why=why,
        visit_min=visit_min,
        what_to_do=what_to_do,
        open_status=adj_status,
        confidence_note=conf_note,
        source=source,
        facts_poi_data=_build_facts_list(cand, lang=lang),
        generated_plan_note=gen_note,
    )


def _contextual_rerank(
    candidates: list[Candidate],
    intent: QueryIntent,
    *,
    accessible: bool = False,
    now_iso: str | None = None,
) -> list[Candidate]:
    """Apply Senior context filters & boosts (weather, time budget, accessibility, night safety)."""
    if not candidates:
        return []

    scored: list[tuple[float, Candidate]] = []
    for idx, cand in enumerate(candidates):
        bonus = -0.02 * idx  # preserve base ranking order as tie-breaker
        tags = set(cand.poi.tags)
        visit_min = estimate_visit_minutes(cand.poi.category, intent.time_budget_min)

        # 1. Time budget feasibility: 2 * walk_min + visit_min <= budget
        if intent.time_budget_min is not None:
            round_trip = 2 * cand.walk_min + visit_min
            if round_trip <= intent.time_budget_min:
                bonus += 0.15
            elif cand.walk_min + 15 > intent.time_budget_min:
                bonus -= 0.25

        # 2. Visit-window opening hours check
        adj_status, _, _ = check_visit_window(
            cand.poi.opening_hours, now_iso, walk_min=cand.walk_min, visit_min=visit_min
        )
        if adj_status == "confirmed_open":
            bonus += 0.08
        elif adj_status == "likely_closed":
            bonus -= 0.35

        # 3. Astana Weather adaptation
        if intent.weather_context in ("cold_windy", "rain_snow"):
            if "indoor" in tags or cand.poi.category in ("mall", "museum", "cafe", "restaurant", "library", "theatre", "cinema", "exhibition"):
                bonus += 0.20
            elif "outdoor" in tags or cand.poi.category == "park":
                bonus -= 0.25

        # 4. Accessibility requirement
        if accessible or intent.is_accessible_query:
            if "wheelchair" in tags:
                bonus += 0.30
            elif "wheelchair_limited" in tags:
                bonus += 0.05
            else:
                bonus -= 0.20

        # 5. Night safety rules
        if intent.is_night_query:
            if "24_7" in tags or cand.open_status == "confirmed_open":
                bonus += 0.22
            if cand.poi.category == "park":
                bonus -= 0.15

        # 6. Family / kids boost
        if intent.is_family_kids:
            if cand.poi.category in ("theme_park", "museum", "park", "exhibition", "mall", "zoo"):
                bonus += 0.18
            elif cand.poi.category == "bar":
                bonus -= 0.40

        scored.append((cand.relevance + bonus, cand))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


def _extract_json_object(raw_text: str) -> dict | None:
    text = raw_text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if brace:
            text = brace.group(0)
    try:
        return json.loads(text)
    except Exception:
        return None


def _build_disclaimer(
    intent: QueryIntent,
    candidates: list[Candidate],
    *,
    lang: str = "ru",
    accessible: bool = False,
) -> str:
    if lang == "en":
        parts = [
            "Data source: OpenStreetMap and Visit Astana. Exact user coordinates are never stored or sent to the LLM (rounded to ~100 m).",
            "Opening hours are verified only where source schedule data exists; always confirm before visiting.",
        ]
        if intent.weather_context != "normal":
            parts.append("Weather mode active: prioritizing indoor spaces due to Astana wind/cold.")
        return " ".join(parts)
    if lang == "kk":
        return (
            "Деректер көзі: OpenStreetMap және Visit Astana. Пайдаланушының нақты координаттары сақталмайды (~100 м дейін дөңгелектенеді). "
            "Жұмыс уақыты тек кесте деректері бар жерде көрсетілген — шығар алдында нақтылап алыңыз."
        )

    parts = [
        "Данные POI: OpenStreetMap и Visit Astana. Точные координаты пользователя не сохраняются и не передаются в LLM (округление до ~100 м).",
        "Факты из базы помечены [poi_data], идеи и сценарии модели — [generated_idea]. Режим работы без расписания всегда помечается как unknown — уточняйте график перед выходом.",
    ]
    if intent.weather_context in ("cold_windy", "rain_snow"):
        parts.append("Учтён погодный контекст Астаны (ветер/осадки): повышен приоритет тёплых локаций в помещении (indoor).")
    if intent.is_night_query:
        parts.append("Ночной режим: закрытые места понижены в выдаче; для прогулок выбирайте освещённые центральные локации.")
    if accessible or intent.is_accessible_query:
        parts.append("Доступность (wheelchair) указана по разметке волонтёров OSM и может быть неполной.")
    return " ".join(parts)


def generate_recommendations(req: RecommendRequest, *, use_llm: bool = True,
                             prefer_nearby: bool | None = None) -> RecommendResponse:
    """End-to-end RAG recommendation pipeline with anti-hallucination guardrails."""
    intent = parse_query_intent(
        req.query,
        explicit_lang=req.lang,
        now_iso=req.now_iso,
        explicit_accessible=req.accessible,
        explicit_group_size=req.group_size,
        explicit_weather=req.weather,
    )
    if req.time_budget_min is not None:
        intent.time_budget_min = req.time_budget_min

    # Anchor landmark resolution: if user explicitly wrote "Я сейчас возле Mega Silk Way"
    # while leaving default Baiterek coordinates, snap to the named landmark!
    search_lat, search_lon = req.lat, req.lon
    anchor_note: str | None = None
    if req.auto_anchor and intent.anchor_landmark is not None:
        # Snap if coordinates are at default Baiterek or within general city bounds when anchor is named
        is_default_coord = abs(req.lat - 51.1283) < 0.002 and abs(req.lon - 71.4305) < 0.002
        if is_default_coord or intent.anchor_landmark.key != "baiterek":
            search_lat = intent.anchor_landmark.lat
            search_lon = intent.anchor_landmark.lon
            anchor_note = f"{intent.anchor_landmark.name} ({search_lat:.4f}, {search_lon:.4f})"

    inferred_categories = search_categories(req.query, req.categories)
    proximity_mode = nearby_requested(req.query, inferred_categories) if prefer_nearby is None else prefer_nearby
    effective_categories = inferred_categories if proximity_mode else (req.categories or None)
    raw_candidates = retrieve(
        req.query,
        search_lat,
        search_lon,
        max_distance_m=req.max_distance_m,
        categories=effective_categories,
        now_iso=req.now_iso,
        # Nearby requests inspect the full local pool so brand-name boosts cannot hide nearer POIs.
        top_k=max(10, len(load_pois())) if proximity_mode else 10,
    )

    ranked_candidates = _contextual_rerank(
        raw_candidates,
        intent,
        accessible=req.accessible,
        now_iso=req.now_iso,
    )
    if proximity_mode:
        ranked_candidates = nearby_candidates(ranked_candidates, req.max_distance_m)

    rounded = round_coords(search_lat, search_lon)
    used_filters = list(effective_categories or intent.inferred_categories)
    used_tags = list(dict.fromkeys(intent.inferred_tags + intent.semantic_tags))

    # Empty retrieval result (e.g. S8 user outside the city or too small radius)
    if not ranked_candidates:
        empty_disclaimer = {
            "en": (f"No matching places were found within {req.max_distance_m} m. "
                   "Try increasing the search radius or choosing a landmark in Astana. We do not invent places."),
            "kk": (f"{req.max_distance_m} м радиуста сәйкес орындар табылмады. "
                   "Іздеу радиусын үлкейтіңіз немесе Астанадағы бағдарды таңдаңыз. Біз орындарды ойдан шығармаймыз."),
        }.get(intent.detected_lang) or (
            f"В радиусе {req.max_distance_m} м от точки {rounded} подходящих мест в базе POI не найдено. "
            "Система не выдумывает несуществующие места — попробуйте увеличить радиус поиска (например, до 3000–5000 м) или выбрать ориентир в городе."
        )
        return RecommendResponse(
            used_data=UsedData(
                location="rounded_100m",
                rounded_coords=rounded,
                filters=used_filters,
                tags=used_tags,
                anchor_resolved=anchor_note,
                time_budget_min=intent.time_budget_min,
                weather_context=intent.weather_context,
                retrieval_backend=BACKEND,
                generation_mode="empty_guardrail",
                candidates_evaluated=0,
            ),
            recommendations=[],
            route=None,
            disclaimer=empty_disclaimer,
        )

    cand_by_id = {c.poi.id: c for c in ranked_candidates}
    recommendations: list[RecommendationItem] = []
    gen_mode = "smart_fallback"

    # Attempt OpenRouter LLM generation if API key is configured
    if use_llm and os.getenv("OPENROUTER_API_KEY", "").strip():
        try:
            messages = build_rag_messages(
                req.query,
                ranked_candidates[:6],
                intent,
                lang=intent.detected_lang,
                group_size=req.group_size,
                accessible=req.accessible,
            )
            raw_llm = complete(messages)
            parsed = _extract_json_object(raw_llm)
            if parsed and isinstance(parsed.get("recommendations"), list):
                for item in parsed["recommendations"][:3]:
                    if not isinstance(item, dict):
                        continue
                    poi_id = str(item.get("poi_id", "")).strip()
                    # Strict Anti-Hallucination Check: must exist in retrieved candidates & get_poi()
                    if poi_id not in cand_by_id or get_poi(poi_id) is None:
                        log.warning("Dropped hallucinated poi_id=%r from LLM output", poi_id)
                        continue
                    cand = cand_by_id[poi_id]
                    fallback_item = _craft_deterministic_item(
                        cand,
                        intent,
                        lang=intent.detected_lang,
                        group_size=req.group_size,
                        accessible=req.accessible,
                        now_iso=req.now_iso,
                    )
                    src = item.get("source")
                    valid_source: SourceType = src if src in ("poi_data", "generated_idea") else fallback_item.source
                    visit_m = int(item.get("visit_min") or fallback_item.visit_min)
                    visit_m = max(10, min(240, visit_m))
                    display_name = item.get("display_name")
                    name = cand.poi.name
                    if (intent.detected_lang == "en" and isinstance(display_name, str)
                            and display_name.strip() and display_name.isascii() and len(display_name) <= 200):
                        name = display_name.strip()

                    recommendations.append(
                        RecommendationItem(
                            poi_id=cand.poi.id,
                            name=name,
                            category=cand.poi.category,
                            lat=cand.poi.lat,
                            lon=cand.poi.lon,
                            distance_m=cand.distance_m,
                            walk_min=cand.walk_min,
                            why=str(item.get("why") or fallback_item.why),
                            visit_min=visit_m,
                            what_to_do=str(item.get("what_to_do") or fallback_item.what_to_do),
                            open_status=fallback_item.open_status,  # Never let LLM override verified open_status!
                            confidence_note=str(item.get("confidence_note") or fallback_item.confidence_note),
                            source=valid_source,
                            facts_poi_data=fallback_item.facts_poi_data,
                            generated_plan_note=str(item.get("generated_plan_note") or fallback_item.generated_plan_note),
                        )
                    )
                if recommendations:
                    gen_mode = "openrouter_llm"
        except Exception as exc:
            log.warning("OpenRouter LLM call failed (%s); falling back to deterministic generator", exc)

    if not recommendations:
        for cand in ranked_candidates[:3]:
            recommendations.append(
                _craft_deterministic_item(
                    cand,
                    intent,
                    lang=intent.detected_lang,
                    group_size=req.group_size,
                    accessible=req.accessible,
                    now_iso=req.now_iso,
                )
            )
        gen_mode = "smart_fallback"

    if proximity_mode:
        recommendations.sort(key=lambda item: item.distance_m)

    route: MicroRoute | None = None
    if req.include_route and len(ranked_candidates) >= 2:
        route = build_micro_route(
            ranked_candidates[:6],
            search_lat,
            search_lon,
            max_stops=3,
            time_budget_min=intent.time_budget_min,
            lang=intent.detected_lang,
        )

    return RecommendResponse(
        used_data=UsedData(
            location="rounded_100m",
            rounded_coords=rounded,
            filters=used_filters,
            tags=used_tags,
            anchor_resolved=anchor_note,
            time_budget_min=intent.time_budget_min,
            weather_context=intent.weather_context,
            retrieval_backend=BACKEND,
            generation_mode=gen_mode,
            candidates_evaluated=len(ranked_candidates),
        ),
        recommendations=recommendations,
        route=route,
        disclaimer=_build_disclaimer(
            intent,
            ranked_candidates,
            lang=intent.detected_lang,
            accessible=req.accessible,
        ),
    )


OFFICIAL_PRESETS: list[dict] = [
    {
        "id": "preset_mega_walk",
        "title": "1. Прогулка у Mega Silk Way (30–60 мин)",
        "query": "Я сейчас возле Mega Silk Way, хочу прогуляться 30–60 минут — что посоветуете?",
        "lat": 51.0891,
        "lon": 71.4073,
        "max_distance_m": 2500,
        "now_iso": "2026-10-16T15:00",
        "accessible": False,
        "group_size": 1,
        "weather": "normal",
    },
    {
        "id": "preset_coffee_laptop",
        "title": "2. Тихое место с кофе и розеткой",
        "query": "Хочу тихое место с кофе и розеткой недалеко от меня.",
        "lat": 51.1283,
        "lon": 71.4305,
        "max_distance_m": 1500,
        "now_iso": "2026-10-16T13:00",
        "accessible": False,
        "group_size": 1,
        "weather": "normal",
    },
    {
        "id": "preset_kids_museum",
        "title": "3. С ребёнком: детские площадки / музеи",
        "query": "Еду с ребёнком — где рядом есть детские площадки или музей с интерактивами?",
        "lat": 51.1283,
        "lon": 71.4305,
        "max_distance_m": 3500,
        "now_iso": "2026-10-16T14:00",
        "accessible": True,
        "group_size": 3,
        "weather": "normal",
    },
    {
        "id": "preset_tourist_flight",
        "title": "4. Турист: 2 часа перед вылетом",
        "query": "Я - турист, у меня 2 часа перед вылетом: что можно посмотреть поблизости?",
        "lat": 51.1283,
        "lon": 71.4305,
        "max_distance_m": 3000,
        "now_iso": "2026-10-16T16:00",
        "accessible": False,
        "group_size": 1,
        "weather": "normal",
    },
    {
        "id": "preset_night_activity",
        "title": "5. Необычная ночная активность",
        "query": "Ищу необычную ночную активность (последние пару часов).",
        "lat": 51.1283,
        "lon": 71.4305,
        "max_distance_m": 3000,
        "now_iso": "2026-10-16T23:30",
        "accessible": False,
        "group_size": 2,
        "weather": "normal",
    },
    {
        "id": "preset_expo_unusual",
        "title": "6. У здания Экспо 2017 (необычное на 1–2 часа)",
        "query": "Я у здании Экспо 2017, хочу что-то необычное на 1–2 часа, пешком.",
        "lat": 51.0928,
        "lon": 71.4215,
        "max_distance_m": 2500,
        "now_iso": "2026-10-16T17:00",
        "accessible": False,
        "group_size": 2,
        "weather": "normal",
    },
]

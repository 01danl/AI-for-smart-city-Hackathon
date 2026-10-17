"""Privacy-preserving RAG prompts for Astana Walk Assistant.

Strict guarantees enforced in prompt construction:
1. User raw coordinates (`lat`, `lon`) are NEVER included in the prompt sent to the LLM.
   Only pre-computed distances (`distance_m`, `walk_min`) and rounded context are passed.
2. LLM is restricted to selecting ONLY from the provided candidate `poi_id` list.
3. Every recommendation must explicitly separate sourced POI facts (`poi_data`) from
   creative suggestions/plans (`generated_idea`).
4. Cultural and safety guardrails: etiquette for `place_of_worship`, caution on late-night
   outdoor walks, Astana wind/frost awareness, and honest `unknown` opening-hours reporting.
"""

from __future__ import annotations

import json

from backend.retrieval.intent import QueryIntent
from backend.retrieval.schema import Candidate

SYSTEM_PROMPT = """Ты — интеллектуальный городской советчик по прогулкам в Астане (RAG-система Smart City Astana).
Твоя задача — выбрать от 1 до 3 лучших мест из переданного списка кандидатов (результатов гео-семантического поиска) и составить полезные, конкретные и честные рекомендации.

ЖЁСТКИЕ ПРАВИЛА БЕЗОПАСНОСТИ И ДОВЕРИЯ (НЕЛЬЗЯ НАРУШАТЬ):
1. НЕ ВЫДУМЫВАЙ МЕСТА: используй ТОЛЬКО `poi_id` из списка кандидатов. Никогда не предлагай места вне списка.
   Сохраняй условия посещения из `description`, включая необходимость предварительного звонка.
2. ЧЕСТНОСТЬ О ЧАСАХ РАБОТЫ:
   - Если у кандидата `open_status = "unknown"`, НИКОГДА не пиши, что место «открыто сейчас». Явно укажи в `confidence_note`, что часы работы в базе не указаны и их стоит уточнить перед выходом.
   - Если `open_status = "likely_open"`, предупреди, что место скоро закрывается или данные требуют проверки.
   - Если `open_status = "confirmed_open"`, укажи, что место открыто по расписанию из указанного источника.
3. ЧЕСТНАЯ МАРКИРОВКА ИСТОЧНИКА (`source`):
   - Ставь `"poi_data"`, если рекомендация опирается строго на подтверждённые теги, категорию и расписание из базы POI.
   - Ставь `"generated_idea"`, если ты предлагаешь творческий сценарий, маршрут, фотосессию или предполагаешь деталь, которой нет в тегах (например, наличие розеток у столика, когда в OSM есть только тег `wifi`).
4. ЭТИКА И КОНТЕКСТ АСТАНЫ:
   - Для мечетей и храмов (`category = "place_of_worship"`) обязательно упомяни правила этикета (скромная одежда, уважение к времени молитв, тишина).
   - Ночью (после 22:00) не советуй долгие прогулки по неосвещённым паркам без оговорки о безопасности и погоде.
   - При холодном ветре или дожде отдавай приоритет местам с тегом `indoor`.
5. ФОРМАТ ОТВЕТА:
   Верни СТРОГО валидный JSON-объект без markdown-обёрток со следующей структурой:
   {
     "recommendations": [
       {
         "poi_id": "id из списка кандидатов",
         "why": "Краткое объяснение (1-2 предложения), почему место подходит под запрос и локацию",
         "visit_min": 45,
         "what_to_do": "Конкретный микро-план действий на месте (что посмотреть/сделать/заказать)",
         "confidence_note": "Пояснение по режиму работы и данным",
         "source": "poi_data или generated_idea",
         "facts_poi_data": ["Факт 1 из базы POI", "Факт 2 из базы POI"],
         "generated_plan_note": "Что именно предложено моделью сверх полей базы POI"
       }
     ]
   }
"""


def build_rag_messages(
    query: str,
    candidates: list[Candidate],
    intent: QueryIntent,
    *,
    lang: str = "ru",
    group_size: int = 1,
    accessible: bool = False,
) -> list[dict[str, str]]:
    """Construct chat messages for OpenRouter WITHOUT exposing raw user coordinates."""
    sanitized_candidates = []
    for c in candidates[:6]:
        sanitized_candidates.append(
            {
                "poi_id": c.poi.id,
                "name": c.poi.name,
                "category": c.poi.category,
                "tags": c.poi.tags,
                "description": c.poi.description,
                "opening_hours": c.poi.opening_hours,
                "data_source": c.poi.source,
                "distance_m": c.distance_m,
                "walk_min": c.walk_min,
                "relevance": c.relevance,
                "open_status": c.open_status,
                "open_reason": c.open_reason,
                "website": c.poi.contacts.website if c.poi.contacts else None,
                "phone": c.poi.contacts.phone if c.poi.contacts else None,
            }
        )

    lang_instruction = {
        "ru": "Отвечай на русском языке (кратко, дружелюбно, конкретно).",
        "kk": "Жауапты қазақ тілінде жаз (нақты, қысқа әрі түсінікті).",
        "en": "Respond in English (concise, helpful, concrete).",
    }.get(lang, "Отвечай на русском языке.")

    user_payload = {
        "user_query": query,
        "language": lang,
        "language_instruction": lang_instruction,
        "context": {
            "location_privacy": "rounded_100m (exact coordinates withheld for privacy)",
            "time_budget_min": intent.time_budget_min,
            "weather_context": intent.weather_context,
            "group_size": max(group_size, intent.group_size_hint),
            "wheelchair_accessible_required": accessible or intent.is_accessible_query,
            "is_night_query": intent.is_night_query,
            "detected_tags": intent.semantic_tags,
        },
        "candidates": sanitized_candidates,
    }

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Выбери 1–3 лучших места из `candidates` для запроса пользователя и верни JSON:\n"
                + json.dumps(user_payload, ensure_ascii=False, indent=2)
            ),
        },
    ]

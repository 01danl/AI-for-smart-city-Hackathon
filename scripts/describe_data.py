"""Generate docs/data_dictionary.md from data/poi.json (categories, tags, field coverage).

Usage:
    python scripts/describe_data.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.retrieval.hours import open_status  # noqa: E402
from backend.retrieval.text import CATEGORY_WORDS  # noqa: E402

OUT_PATH = ROOT / "docs" / "data_dictionary.md"

CATEGORY_RU = {
    "cafe": "Кафе, кофейни", "restaurant": "Рестораны", "fast_food": "Фастфуд",
    "bar": "Бары, пабы", "mall": "Торговые центры", "park": "Парки, скверы",
    "sport": "Спорт: залы, катки", "monument": "Памятники, мемориалы",
    "attraction": "Достопримечательности", "place_of_worship": "Мечети, храмы",
    "artwork": "Арт-объекты", "cinema": "Кинотеатры", "theatre": "Театры",
    "library": "Библиотеки", "museum": "Музеи", "bookstore": "Книжные",
    "arts_centre": "Культурные центры", "stadium": "Стадионы, арены",
    "gallery": "Галереи", "theme_park": "Парки развлечений", "exhibition": "Выставочные центры",
    "viewpoint": "Смотровые площадки", "zoo": "Зоопарки",
}

# Tags worth exposing as UI filters / chips (others are raw OSM values), with short labels.
FEATURE_TAGS = {
    "wifi": "Wi-Fi", "wheelchair": "Доступно для колясок",
    "wheelchair_limited": "Частично доступно для колясок", "terrace": "Летняя терраса",
    "halal": "Халяль", "vegetarian": "Вегетарианское меню", "free": "Бесплатный вход",
    "24_7": "Круглосуточно", "takeaway": "Навынос", "indoor": "В помещении",
    "outdoor": "На открытом воздухе", "landmark": "Известное место", "culture": "Культура",
    "food": "Еда", "events": "Мероприятия", "coffee": "Кофе",
}


def main() -> None:
    pois = json.loads((ROOT / "data" / "poi.json").read_text(encoding="utf-8"))
    n = len(pois)
    cats = Counter(p["category"] for p in pois)
    tags = Counter(t for p in pois for t in p["tags"])
    with_hours = [p for p in pois if p["opening_hours"]]
    parsed = sum(open_status(p["opening_hours"])[0] != "unknown" for p in with_hours)

    def pct(k: int) -> str:
        return f"{k} ({k * 100 // n}%)"

    lines = [
        "# Справочник данных POI",
        "",
        "Сгенерировано `scripts/describe_data.py` из `data/poi.json`, не редактировать вручную.",
        f"Всего мест: **{n}**. Источники: OpenStreetMap и официальный туристический портал Visit Astana.",
        f"OpenStreetMap: {sum(p['source'] == 'osm' for p in pois)}; Visit Astana: {sum(p['source'].startswith('https://visitastana.kz/') for p in pois)}.",
        f"Даты получения/среза: {', '.join(sorted({p['collected_at'] for p in pois}))}. Ссылки и подтверждения новых записей: `data/poi_sources.json`.",
        "",
        "## Заполненность полей",
        "",
        "| Поле | Заполнено |",
        "|---|---|",
        f"| `opening_hours` | {pct(len(with_hours))}, из них разбирается {parsed} |",
        f"| `description` | {pct(sum(1 for p in pois if p['description']))} |",
        f"| `contacts.phone` | {pct(sum(1 for p in pois if p['contacts']['phone']))} |",
        f"| `contacts.website` | {pct(sum(1 for p in pois if p['contacts']['website']))} |",
        "| `rating` | 0 (в открытых данных нет, не выдумываем) |",
        "",
        "## Категории (`category`)",
        "",
        "Значения для параметра `categories` в `retrieve()` и для фильтров в UI.",
        "",
        "| category | Что это | Мест |",
        "|---|---|---|",
    ]
    for cat, k in cats.most_common():
        lines.append(f"| `{cat}` | {CATEGORY_RU.get(cat, CATEGORY_WORDS.get(cat, ''))} | {k} |")

    lines += [
        "",
        "## Теги-признаки (`tags`)",
        "",
        "Их можно показывать чипами в карточке и подставлять в `used_data.tags`.",
        "Запрос со словами-подсказками («розетка», «коляска», «туристу», «футбол»…) поднимает",
        "места с соответствующим тегом (см. `QUERY_TAG_HINTS` в `backend/retrieval/text.py`).",
        "",
        "| tag | Значение | Мест |",
        "|---|---|---|",
    ]
    for t, label in FEATURE_TAGS.items():
        lines.append(f"| `{t}` | {label} | {tags.get(t, 0)} |")

    cuisines = [(t, k) for t, k in tags.most_common() if t.startswith("cuisine_")]
    lines += [
        "",
        "## Кухни (`cuisine_*`)",
        "",
        ", ".join(f"`{t.removeprefix('cuisine_')}` {k}" for t, k in cuisines),
        "",
        "## Известные места (`landmark`)",
        "",
        "Объекты со ссылкой на Википедию/Wikidata в OSM.",
        "",
    ]
    for p in sorted((p for p in pois if "landmark" in p["tags"]), key=lambda p: p["name"]):
        lines.append(f"- {p['name']} (`{p['category']}`)")

    lines += [
        "",
        "## Ограничения",
        "",
        "- Тегов `quiet` и `power_outlets` из примера в контракте в OSM нет. «Тихо» и «розетка»",
        "  обрабатываются через смысловой поиск и тег `wifi`.",
        "- Для старых записей часы и доступность взяты из OSM; для дополнения — расписание Visit Astana. Данные могут устаревать.",
        "  Подробнее: `docs/ethics_privacy.md`.",
        "",
    ]
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

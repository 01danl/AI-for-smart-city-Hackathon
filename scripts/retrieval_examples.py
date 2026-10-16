"""Run demo queries through retrieve() and write docs/retrieval_examples.md.

Shows what the retrieval layer alone returns (before the LLM), for the presentation
and the RAG-approach description. Fixed timestamps keep the output reproducible.

Usage:
    python scripts/retrieval_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.retrieval import BACKEND, retrieve  # noqa: E402
from backend.retrieval.text import query_tag_hints  # noqa: E402

OUT_PATH = ROOT / "docs" / "retrieval_examples.md"

BAITEREK = ("Байтерек", 51.1283, 71.4305)
OLD_CENTER = ("Старый центр (правый берег)", 51.1694, 71.4491)
EXPO = ("EXPO", 51.0900, 71.4180)

# (title, query, place, kwargs)
EXAMPLES = [
    ("Поработать с ноутбуком", "тихое место с кофе и розеткой", BAITEREK,
     dict(max_distance_m=1500, now_iso="2026-10-16T13:00")),
    ("Турист, первый день в городе", "что посмотреть туристу", BAITEREK,
     dict(max_distance_m=4000, now_iso="2026-10-17T11:00")),
    ("Семейный ужин", "поесть казахскую кухню с семьёй", BAITEREK,
     dict(max_distance_m=2000, now_iso="2026-10-16T19:00")),
    ("Поздний вечер", "поесть", BAITEREK,
     dict(max_distance_m=2000, now_iso="2026-10-16T23:40")),
    ("Утро выходного, правый берег", "где выпить кофе утром", OLD_CENTER,
     dict(max_distance_m=1000, now_iso="2026-10-18T08:00")),
    ("Маломобильный посетитель", "доступно для колясок", BAITEREK,
     dict(max_distance_m=2000, now_iso="2026-10-16T13:00")),
    ("Иностранный гость", "quiet place to read a book", BAITEREK,
     dict(max_distance_m=4000, now_iso="2026-10-16T15:00")),
    ("Прогулка у EXPO", "погулять на свежем воздухе", EXPO,
     dict(max_distance_m=2000, now_iso="2026-10-16T17:00")),
]

STATUS_RU = {
    "confirmed_open": "открыто по данным",
    "likely_open": "скоро закроется",
    "unknown": "часы неизвестны",
    "likely_closed": "закрыто по данным",
}


def main() -> None:
    lines = [
        "# Примеры работы поиска (retrieval)",
        "",
        "Сгенерировано `scripts/retrieval_examples.py`. Это выдача **только поискового слоя**,",
        "до генерации объяснений моделью. LLM получает эти кандидаты и выбирает из них 1-3,",
        "новых мест она не добавляет.",
        f"Бэкенд: `{BACKEND}`. Время в запросах зафиксировано, чтобы результат воспроизводился.",
        "",
    ]
    for title, query, (place, lat, lon), kw in EXAMPLES:
        res = retrieve(query, lat, lon, top_k=5, **kw)
        hints = sorted(query_tag_hints(query))
        lines += [
            f"## {title}",
            "",
            f"**Запрос:** «{query}» · **где:** {place} · **радиус:** {kw['max_distance_m']} м · "
            f"**время:** {kw['now_iso'].replace('T', ' ')}",
        ]
        if hints:
            lines.append(f"**Распознанные требования (теги):** {', '.join(f'`{h}`' for h in hints)}")
        lines += [
            "",
            "| # | Место | Категория | Пешком | Совпадение | Режим работы |",
            "|---|---|---|---|---|---|",
        ]
        for i, c in enumerate(res, 1):
            lines.append(
                f"| {i} | {c.poi.name} | `{c.poi.category}` | {c.distance_m} м, ~{c.walk_min} мин "
                f"| {c.relevance:.2f} | {STATUS_RU[c.open_status]} |"
            )
        if not res:
            lines.append("| — | Ничего не найдено в радиусе | | | | |")
        lines.append("")
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

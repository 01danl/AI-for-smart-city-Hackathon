# Обзор репозитория (Полная Senior-версия: Роль A + Роль B)

Самодостаточное описание проекта «Astana Walk AI — ИИ-помощник по прогулкам в городе Астана»
для хакатона «Build with AI» (Школа ИИ и науки о данных, партнёры: НИЦ Smart City, GDG Astana).

## Текущий статус: 100% готовность всех слоёв

В проекте полностью реализованы и покрыты автотестами **оба слоя**:
1. **Слой данных, гео-семантического поиска и маршрутизации (Роль A — `backend/retrieval/`):**
   - Автоматизированный сборщик POI из OpenStreetMap Overpass API (`scripts/collect_osm.py`) и очищенный датасет `data/poi.json` (**966 POI Астаны**).
   - Векторный индекс ChromaDB (`paraphrase-multilingual-MiniLM-L12-v2`) + лексический гибридный движок **Okapi BM25** (`backend/retrieval/hybrid_bm25.py`) + система тегов-удобств (`QUERY_TAG_HINTS`).
   - Парсер намерений запроса (`backend/retrieval/intent.py`): авто-распознавание ориентиров в тексте («возле Mega Silk Way», «у здания Экспо 2017»), бюджета времени, погодного контекста Астаны и языка (`ru`/`kk`/`en`).
   - Контроль режима работы (`backend/retrieval/hours.py`) в часовом поясе Астаны (UTC+5) с проверкой окна визита (`check_visit_window`: `walk_min + visit_min`).
   - Алгоритм построения связанного микро-маршрута из 2–3 точек (`backend/retrieval/route_planner.py`).
2. **Слой генерации, защиты от галлюцинаций, REST API и UI (Роль B — `backend/generation/`, `backend/api/`, `frontend/`):**
   - Клиент OpenRouter API (`backend/generation/client.py`, совместимый с OpenAI SDK, модель `nvidia/nemotron-3-ultra-550b-a55b:free`).
   - Конвейер рекомендаций (`backend/generation/recommender.py`) с жёсткой проверкой `poi_id` через `get_poi()` (нулевые галлюцинации), честным разделением фактов `poi_data` и идей `generated_idea`, а также отказоустойчивым детерминированным генератором (`smart_fallback`).
   - Privacy-by-Design промптинг (`backend/generation/prompts.py`): сырые координаты пользователя никогда не передаются в LLM (передаются только предрассчитанные расстояния и округление `rounded_100m`).
   - FastAPI сервис (`backend/api/app.py`: `POST /recommend`, `GET /health`, `GET /presets`, `GET /pois/stats`).
   - Интерактивный Streamlit UI (`frontend/app.py`) с картой PyDeck, визуализацией 100-метрового круга приватности, 6 сценариями жюри в один клик, таймлайном микро-маршрута и панелью «RAG & Privacy Inspector».

## Структура репозитория

```
data/
  poi.json                  # готовый датасет: 966 POI Астаны (OSM)
  raw/                      # кэш сырого ответа Overpass
  chroma/                   # векторный индекс Chroma (генерируется scripts/build_index.py)
backend/
  retrieval/                # schema, geo, hours, text, index, chroma_retriever,
                            # hybrid_bm25, intent, route_planner, stub
  generation/               # client (OpenRouter), prompts, recommender
  api/                      # FastAPI приложение (/recommend, /health, /presets, /pois/stats)
frontend/
  app.py                    # Интерактивный UI-прототип на Streamlit + PyDeck
scripts/
  collect_osm.py            # автосбор POI из OSM Overpass -> data/poi.json
  build_index.py            # сборка Chroma-индекса
  describe_data.py          # генерация docs/data_dictionary.md
  retrieval_examples.py     # генерация docs/retrieval_examples.md
docs/
  concept.md                # 1. Концепция продукта, ЦА, сценарии и бизнес-модель
  rag_approach.md           # 3. Описание гибридного RAG-подхода и формулы ранжирования
  test_scenarios.md         # 4. Набор тестовых сценариев S1–S11 и инварианты
  ethics_privacy.md         # 6. Этика, приватность и работа с неполными данными
  data_dictionary.md        # Справочник категорий, тегов и заполненности полей
  retrieval_examples.md     # Реальные примеры выдачи поискового слоя
tests/
  test_hours.py             # Тесты парсера часов работы и тегов
  test_stub.py              # Тесты контракта заглушки
  test_scenarios.py         # Тесты сценариев S1–S11 на реальных 966 POI
  test_generation_client.py # Тесты OpenRouter клиента, приватности промпта и анти-галлюцинаций
  test_api_and_routes.py    # Тесты парсера намерений, окна визита, маршрутов и пресетов жюри
```

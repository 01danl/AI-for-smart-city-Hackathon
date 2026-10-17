# 🧭 Astana Walk AI — ИИ-помощник по прогулкам в городе Астана

**Хакатон:** Build with AI (Школа ИИ и науки о данных) · **Трек 1:** Задача от НИЦ Smart City и GDG Astana

Реально работающая **Hybrid Geosemantical RAG** система (концепция + автосбор данных + API + интерактивное веб-демо), которая в реальном времени подсказывает жителям и гостям Астаны, куда сходить и чем заняться рядом с текущей локацией, строит связанные пешие микро-маршруты из 2–3 точек и честно разделяет подтверждённые данные базы (`poi_data`) и идеи модели (`generated_idea`).

---

## 📦 Навигация по 6 обязательным артефактам для жюри

| № | Ожидаемый артефакт (по ТЗ хакатона) | Где находится в репозитории |
| :--- | :--- | :--- |
| **1** | **Концепция продукта** (идея, ЦА, сценарии, бизнес-модель, план развития) | [`docs/concept.md`](docs/concept.md) |
| **2** | **Демо / прототип** (интерактивный UI с картой, 6 пресетами жюри, микро-маршрутами и RAG-инспектором + FastAPI `/recommend`) | [`frontend/app.py`](frontend/app.py) и [`backend/api/app.py`](backend/api/app.py) |
| **3** | **Описание работы RAG-подхода** (высокоуровневая архитектура, формула ранжирования, защита от галлюцинаций) | [`docs/rag_approach.md`](docs/rag_approach.md) |
| **4** | **Набор тестовых сценариев** (11 реальных кейсов S1–S11 + примеры выдачи + автотесты `pytest`) | [`docs/test_scenarios.md`](docs/test_scenarios.md), [`docs/retrieval_examples.md`](docs/retrieval_examples.md), [`tests/`](tests/) |
| **5** | **Краткая презентация (5–7 минут)** | В папке сдачи (`Astana_Walk_AI_Presentation.pptx` / `.pdf`) |
| **6** | **Этические и privacy-заметки** (округление координат `rounded_100m`, работа с `unknown` часами, безопасность) | [`docs/ethics_privacy.md`](docs/ethics_privacy.md) и [`docs/data_dictionary.md`](docs/data_dictionary.md) |

---

## 🚀 Ключевые Senior-особенности решения (по критериям оценки жюри)

1. **Релевантность и гибридный поиск (Критерий №1):**
   - Автоматизированный воспроизводимый сборщик [`scripts/collect_osm.py`](scripts/collect_osm.py) из OpenStreetMap Overpass API: **966 POI из OSM + 9 дополнительных POI с официального Visit Astana** в [`data/poi.json`](data/poi.json).
   - Двухконтурный поиск: векторный индекс **ChromaDB** (`paraphrase-multilingual-MiniLM-L12-v2`) + лексический движок **Okapi BM25** ([`backend/retrieval/hybrid_bm25.py`](backend/retrieval/hybrid_bm25.py)) + словарь жёстких тегов-удобств (`QUERY_TAG_HINTS`: `wifi`, `wheelchair`, `halal`, `cuisine_kazakh`, `landmark`).
   - Даже если на чистой машине ещё не собран тяжёлый Chroma-индекс, система в режиме `RETRIEVAL_BACKEND=auto` мгновенно работает по всем **975 местам Астаны** через `hybrid_bm25`.

2. **Контекстуальность, климат Астаны и контроль окна визита (Критерий №2):**
   - Парсер намерений [`backend/retrieval/intent.py`](backend/retrieval/intent.py) автоматически распознаёт **якорные ориентиры в тексте** (*«возле Mega Silk Way»*, *«у здания Экспо 2017»*, *«возле Хан Шатыра»*), **бюджет времени** (*«на 30–60 минут»*, *«2 часа перед вылетом»*) и **погодный контекст Астаны** (*ветер/мороз* $\rightarrow$ приоритет тёплым локациям `indoor`).
   - Умная проверка графика работы [`backend/retrieval/hours.py`](backend/retrieval/hours.py) в часовом поясе Астаны (`UTC+5`) проверяет не только минуту старта, но и **окно визита** (`walk_min + visit_min`), предупреждая, если место закроется во время прогулки.

3. **Нулевые галлюцинации и честная маркировка источников (Критерий №3):**
   - Система **никогда не выдумывает места**: каждый `poi_id` из ответа LLM валидируется через `get_poi(poi_id)` и список кандидатов retrieval ([`backend/generation/recommender.py`](backend/generation/recommender.py)).
   - Каждая карточка явно разделяет подтверждённые факты базы **`[poi_data]`** и сгенерированный микро-план **`[generated_idea]`**, а при отсутствии расписания в источнике всегда ставит статус `unknown` вместо ложного «открыто сейчас».

4. **Стретч-цели: генератор микро-маршрутов, мультиязычность и безбарьерность (Критерий №5):**
   - Модуль [`backend/retrieval/route_planner.py`](backend/retrieval/route_planner.py) строит связанный пеший маршрут из 2–3 точек разного типа (*Прогулка/Памятник $\rightarrow$ Музей/Выставка $\rightarrow$ Кофейня*) с расчётом переходов между соседними точками (`User -> A -> B -> C`) и суммарного времени.
   - Поддержка трёх языков (`ru` / `kk` / `en`), учёт групп (`group_size`) и маломобильных пользователей (`accessible=True`).

5. **Privacy-by-Design и отказоустойчивость (Критерий №6):**
   - Координаты пользователя никогда не сохраняются в сыром виде и **не отправляются в промпт LLM** (проверяется автотестом `test_prompt_never_leaks_raw_user_coordinates`). В логах и UI используется только округление до ~100 м (`round_coords`).
   - При отсутствии `OPENROUTER_API_KEY` или сбое сети автоматически срабатывает контекстный детерминированный генератор (`smart_fallback`), поэтому демо и API работают со 100% надёжностью в любых условиях.

---

## 🛠️ Быстрый запуск (Quickstart)

### 1. Установка зависимостей и настройка `.env`

Установите [uv](https://docs.astral.sh/uv/getting-started/installation/), если его ещё нет
(`brew install uv` на macOS с Homebrew). Из корня репозитория:

```bash
uv sync --locked
cp -n .env.example .env
```

`uv sync` создаёт `.venv` с Python 3.12 и ставит зависимости из `uv.lock`.
`cp -n` сохраняет уже существующий `.env`. Укажите в нём `OPENROUTER_API_KEY`,
если хотите ответы модели; без ключа работает `smart_fallback`.

Если хотите работать внутри окружения в текущем терминале:

```bash
source .venv/bin/activate
python --version
# выйти из окружения: deactivate
```

Команды ниже используют `uv run`, поэтому активация не требуется. В редакторе
выберите интерпретатор `.venv/bin/python`.

### Официальное дополнение данных

9 новых мест получены с Visit Astana: названия и координаты извлечены из карточек,
7 расписаний перенесены с учётом выходных и обеденных перерывов. Условия визита
по звонку сохранены; отсутствующие данные оставлены пустыми.
[Источники и ограничения](docs/official_data_collection.md),
[ссылки и подтверждения полей](data/poi_sources.json).

```bash
uv run python scripts/collect_official_poi.py --download  # загрузить и проверить
uv run python scripts/collect_official_poi.py             # показать соседние POI
uv run python scripts/collect_official_poi.py --apply     # применить проверенный список
```

### 2. (Опционально) Сбор данных из OSM и сборка векторного индекса Chroma

В репозитории уже лежит готовый датасет `data/poi.json` (975 POI). Без индекса
приложение использует поиск BM25 по этим данным. Чтобы пересобрать индекс Chroma:

```bash
uv run python scripts/collect_osm.py  # необязательно: свежий запрос в Overpass API
uv sync --locked --extra vector
uv run --extra vector python scripts/build_index.py  # индекс в data/chroma/
```

### 3. Запуск интерактивного демо (Streamlit UI)

```bash
uv run streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

Откройте http://127.0.0.1:8501. Для API используйте второй терминал.

### 4. Запуск REST API (FastAPI)

```bash
uv run uvicorn backend.api.app:app --host 127.0.0.1 --port 8000
```

Пример запроса к API:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Я сейчас возле Mega Silk Way, хочу прогуляться 30–60 минут — что посоветуете?",
    "lat": 51.0891,
    "lon": 71.4073,
    "max_distance_m": 2500,
    "lang": "ru",
    "include_route": true
  }'
```

### 5. Запуск автотестов

```bash
uv run python -m pytest tests -q
```

# Запуск в Docker

## Быстрый старт

Нужен Docker Desktop либо Docker Engine с плагином Compose 2.24+. Команды выполняются из корня репозитория. На macOS/Windows сначала запустите Docker Desktop.

```bash
cp -n .env.example .env
docker compose up --build -d --wait
```

| Сервис | Адрес | Проверка готовности |
| :--- | :--- | :--- |
| Streamlit | http://localhost:8501 | `/_stcore/health` |
| FastAPI | http://localhost:8000/docs | `/health` |

Python 3.12 и uv включены в процесс сборки. Версии зависимостей берутся из `uv.lock`, зависимости разработки не устанавливаются. Процессы работают от пользователя без прав root. Образ собирается для архитектуры компьютера; принудительная эмуляция amd64 на Apple Silicon не нужна.

UI вызывает RAG внутри своего процесса. API — отдельная точка интеграции: UI не ждёт его и не отправляет ему запросы. Можно запустить только интерфейс:

```bash
docker compose up --build -d --wait ui
```

## Ключи и порты

Для модели укажите `OPENROUTER_API_KEY`, для живого голосового разговора — `GEMINI_API_KEY` в локальном `.env`. Пустые ключи допустимы: остаются локальный поиск и шаблонные объяснения. `.env` не попадает в контекст сборки или образ; Compose передаёт его значения контейнерам при запуске. Не публикуйте этот файл и полный вывод `docker compose config`, который может содержать ключи.

После изменения ключей или модели пересоздайте контейнеры:

```bash
docker compose up -d --force-recreate --wait
```

Если порт занят, измените в `.env`:

```dotenv
STREAMLIT_PORT=8502
API_PORT=8001
```

Затем выполните `docker compose up -d --wait`. Внутри контейнеров порты остаются 8501 и 8000. Снаружи они доступны только на `127.0.0.1`.

Голосовой разговор и геолокация требуют разрешений браузера и защищённого контекста: `localhost` либо HTTPS. Для публикации используйте HTTPS reverse proxy с поддержкой WebSocket для Streamlit; одного открытия порта на сервере недостаточно. Звук идёт из браузера напрямую в Gemini. Серверу нужен исходящий доступ к OpenRouter и Google, браузеру — также к картографическим сервисам.

## Данные и обновления

- Датасет POI и событий входит в образ. Обновили код, стили или данные — выполните `docker compose up --build -d --wait`.
- Volume `app-state` сохраняет анонимные отзывы (`/app/state/feedback.jsonl`), а при использовании Chroma — индекс и кэш модели. Точные координаты, аудио и чаты туда не записываются.
- Хостовые `data/feedback.jsonl` и `data/chroma/` автоматически не импортируются. В контейнерах `POI_DATA_PATH`, `CHROMA_DIR`, `FEEDBACK_PATH` и `HF_HOME` задаются в `compose.yaml`.
- `docker compose down` сохраняет volume. Команда `docker compose down --volumes` **удаляет отзывы, индекс и кэш модели**.
- История чатов хранится в памяти Streamlit-сессии и теряется при пересоздании контейнера.

## Опционально: Chroma

Для демо достаточно BM25 по всем местам из датасета. Векторный вариант существенно увеличивает образ и скачивает модель при построении индекса.

1. Установите `INSTALL_VECTOR=true` в `.env`.
2. Соберите образ и создайте индекс в общем volume:

```bash
docker compose build
docker compose run --rm --no-deps ui python scripts/build_index.py
docker compose up -d --force-recreate --wait
```

При `RETRIEVAL_BACKEND=auto` оба сервиса выберут готовый индекс Chroma после перезапуска. Индекс с хоста не нужен. После изменения POI снова выполните сборку образа и построение индекса.

Чтобы вернуться к быстрому поиску, установите `RETRIEVAL_BACKEND=bm25` в `.env` и пересоздайте контейнеры. Для уменьшения образа также верните `INSTALL_VECTOR=false` и пересоберите его.

## Диагностика

```bash
docker compose ps
docker compose logs --tail=100 ui api
docker compose config --quiet
docker compose down
```

Если Docker сообщает `Cannot connect to the Docker daemon`, запустите Docker Desktop/Engine. Если контейнер работает, но модель недоступна, проверьте ключи и квоту провайдера: работоспособность локального поиска от них не зависит.

Схема установки зависимостей следует [официальной инструкции uv для Docker](https://docs.astral.sh/uv/guides/integration/docker/).

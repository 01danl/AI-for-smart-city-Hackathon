# Прогулки по Астане

Проект собирает места из OpenStreetMap, ищет подходящие по запросу и расстоянию и
планирует добавлять объяснения рекомендаций через модель в OpenRouter.

## Настройка

1. Установите зависимости: `python -m pip install -r requirements.txt`.
2. Скопируйте `.env.example` в `.env` и задайте `OPENROUTER_API_KEY`.
3. Выберите `OPENROUTER_MODEL` в формате `provider/model`. В примере стоит бесплатная
   `nvidia/nemotron-3-ultra-550b-a55b:free`; её можно заменить без изменений кода.
   `OPENROUTER_BASE_URL` по умолчанию — `https://openrouter.ai/api/v1`.
4. Для поиска по реальным местам запустите `python scripts/build_index.py`.

Клиент генерации находится в `backend/generation/client.py`:

```python
from backend.generation import complete

answer = complete([{"role": "user", "content": "Предложи прогулку по Астане"}])
print(answer)
```

Этот вызов отправляет текст в OpenRouter. Слой рекомендаций, FastAPI и интерфейс пока
не реализованы; `complete()` служит точкой подключения модели для этих компонентов.
Поиск автоматически использует индекс Chroma, если он собран, иначе тестовые места.

Проверка: `python -m pytest tests -q`. Сценарии реального поиска пропускаются до сборки индекса.

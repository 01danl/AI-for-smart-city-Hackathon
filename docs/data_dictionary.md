# Справочник данных POI

Сгенерировано `scripts/describe_data.py` из `data/poi.json`, не редактировать вручную.
Источник: OpenStreetMap, срез `2026-10-16`. Всего мест: **966**.

## Заполненность полей

| Поле | Заполнено |
|---|---|
| `opening_hours` | 331 (34%), из них разбирается 329 |
| `description` | 2 (0%) |
| `contacts.phone` | 187 (19%) |
| `contacts.website` | 204 (21%) |
| `rating` | 0 (в открытых данных нет, не выдумываем) |

## Категории (`category`)

Значения для параметра `categories` в `retrieve()` и для фильтров в UI.

| category | Что это | Мест |
|---|---|---|
| `cafe` | Кафе, кофейни | 309 |
| `restaurant` | Рестораны | 167 |
| `fast_food` | Фастфуд | 118 |
| `bar` | Бары, пабы | 101 |
| `mall` | Торговые центры | 51 |
| `park` | Парки, скверы | 38 |
| `sport` | Спорт: залы, катки | 36 |
| `monument` | Памятники, мемориалы | 36 |
| `attraction` | Достопримечательности | 24 |
| `place_of_worship` | Мечети, храмы | 17 |
| `artwork` | Арт-объекты | 13 |
| `cinema` | Кинотеатры | 9 |
| `theatre` | Театры | 9 |
| `library` | Библиотеки | 8 |
| `museum` | Музеи | 8 |
| `bookstore` | Книжные | 7 |
| `arts_centre` | Культурные центры | 6 |
| `stadium` | Стадионы, арены | 5 |
| `gallery` | Галереи | 2 |
| `theme_park` | Парки развлечений | 1 |
| `exhibition` | Выставочные центры | 1 |

## Теги-признаки (`tags`)

Их можно показывать чипами в карточке и подставлять в `used_data.tags`.
Запрос со словами-подсказками («розетка», «коляска», «туристу», «футбол»…) поднимает
места с соответствующим тегом (см. `QUERY_TAG_HINTS` в `backend/retrieval/text.py`).

| tag | Значение | Мест |
|---|---|---|
| `wifi` | Wi-Fi | 45 |
| `wheelchair` | Доступно для колясок | 43 |
| `wheelchair_limited` | Частично доступно для колясок | 5 |
| `terrace` | Летняя терраса | 39 |
| `halal` | Халяль | 3 |
| `vegetarian` | Вегетарианское меню | 2 |
| `free` | Бесплатный вход | 1 |
| `24_7` | Круглосуточно | 30 |
| `takeaway` | Навынос | 49 |
| `indoor` | В помещении | 85 |
| `outdoor` | На открытом воздухе | 87 |
| `landmark` | Известное место | 19 |
| `culture` | Культура | 69 |
| `food` | Еда | 695 |
| `events` | Мероприятия | 6 |
| `coffee` | Кофе | 305 |

## Кухни (`cuisine_*`)

`coffee_shop` 93, `burger` 21, `pizza` 18, `chicken` 17, `regional` 17, `breakfast` 11, `italian` 9, `asian` 9, `chinese` 9, `russian` 8, `european` 7, `oriental` 7, `japanese` 6, `sushi` 6, `cake` 5, `international` 4, `steak_house` 4, `sandwich` 4, `korean` 4, `local` 4, `italian_pizza` 4, `french` 3, `turkish` 3, `pasta` 3, `kebab` 3, `brunch` 3, `pastry` 3, `georgian` 3, `barbecue` 2, `beef_bowl` 2, `curry` 2, `diner` 2, `bubble_tea` 2, `shawarma` 2, `american` 2, `crepe` 2, `buschenschank` 2, `fish_and_chips` 2, `hot_dog` 2, `tea` 2, `seafood` 2, `friture` 1, `grill` 1, `indian` 1, `donut` 1, `dessert` 1, `fries` 1, `mexican` 1, `mediterranean` 1, `steak` 1, `croatian` 1, `deli` 1, `ethiopian` 1, `filipino` 1, `fine_dining` 1, `lebanese` 1

## Известные места (`landmark`)

Объекты со ссылкой на Википедию/Wikidata в OSM.

- Abu Dhabi Plaza (`mall`)
- Astana Musical (`theatre`)
- Астана Арена (`stadium`)
- Астана Опера (`theatre`)
- Байтерек (`monument`)
- Барыс Арена (`stadium`)
- Бейт Рахель Хабад Любавич (`place_of_worship`)
- Ботанический сад (`park`)
- Дворец Независимости (`attraction`)
- Дворец мира и согласия (`attraction`)
- Ледовый дворец Алау (`sport`)
- Мечеть Абу Насыра аль-Фараби (`place_of_worship`)
- Национальный музей Республики Казахстан (`museum`)
- Парк Жетысу (`park`)
- Сарыарка (`sport`)
- Триумфальная арка Мангилик Ел (`attraction`)
- Хан Шатыр (`mall`)
- Этно-мемориальный комплекс Карта Казахстана Атамекен (`museum`)
- мечеть "Хазрет Султан" (`place_of_worship`)

## Ограничения

- Тегов `quiet` и `power_outlets` из примера в контракте в OSM нет. «Тихо» и «розетка»
  обрабатываются через смысловой поиск и тег `wifi`.
- Часы работы и доступность заполняют волонтёры OSM, данные могут быть устаревшими.
  Подробнее: `docs/ethics_privacy.md`.

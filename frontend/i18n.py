"""Interface texts in Russian, Kazakh and English.

`t(key, lang, **values)` returns the text for the UI language; Russian is the fallback, so a
missing translation never breaks the page. Scenario titles and questions are translated here
too, so the Kazakh and English scenarios ask in that language.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_ANSWER_LOCALE: ContextVar[dict | None] = ContextVar("answer_locale", default=None)


@contextmanager
def answer_context(presentation: dict):
    """Scope answer language to one message, without changing the interface or other sessions."""
    token = _ANSWER_LOCALE.set(presentation)
    try:
        yield
    finally:
        _ANSWER_LOCALE.reset(token)


def current_answer_language() -> str | None:
    current = _ANSWER_LOCALE.get()
    return current.get("language") if current else None


def _answer_label(key: str, lang: str | None) -> str | None:
    current = _ANSWER_LOCALE.get()
    return current.get("labels", {}).get(key) if current and current.get("language") == lang else None

LANGS = ("ru", "kk", "en")
LANG_SHORT = {"ru": "RU", "kk": "KZ", "en": "EN"}

TEXT: dict[str, dict[str, str]] = {
    # Header
    "head_sub": {"ru": "Куда сходить в Астане прямо сейчас",
                 "kk": "Астанада қазір қайда баруға болады",
                 "en": "Where to go in Astana right now"},
    "pill_places": {"ru": "{n} мест", "kk": "{n} орын", "en": "{n} places"},
    "pill_start": {"ru": "Ищу рядом с:", "kk": "Іздеу нүктесі:", "en": "Searching near:"},
    "pill_start_title": {
        "ru": "Точка округляется до ~100 м; точные координаты не хранятся и не уходят в модель",
        "kk": "Нүкте ~100 м-ге дейін дөңгелектенеді; нақты координаттар сақталмайды және модельге жіберілмейді",
        "en": "The point is rounded to ~100 m; exact coordinates are never stored or sent to the model"},
    "pill_mode": {"ru": "ответ: {mode}", "kk": "жауап: {mode}", "en": "answer: {mode}"},
    "mode_llm": {"ru": "нейросеть", "kk": "нейрожелі", "en": "AI model"},
    "mode_template": {"ru": "шаблоны", "kk": "үлгілер", "en": "templates"},
    "mode_pending": {"ru": "подбираю…", "kk": "іздеп жатырмын…", "en": "searching…"},
    "settings": {"ru": "Настройки", "kk": "Баптаулар", "en": "Settings"},
    "settings_help": {
        "ru": "Свернуть или раскрыть панель настроек (точка старта, радиус, время, погода)",
        "kk": "Баптаулар панелін жасыру немесе ашу (бастау нүктесі, радиус, уақыт, ауа райы)",
        "en": "Show or hide the settings panel (start point, radius, time, weather)"},

    # Sidebar
    "logo_alt": {"ru": "AstanaWalk.ai — прогулки по Астане", "kk": "AstanaWalk.ai — Астана бойынша серуен",
                 "en": "AstanaWalk.ai — walks in Astana"},
    "language": {"ru": "Язык", "kk": "Тіл", "en": "Language"},
    "interface_language": {"ru": "Язык интерфейса", "kk": "Интерфейс тілі", "en": "Interface language"},
    "answer_language_auto": {
        "ru": "Ответы — на языке вашего сообщения.",
        "kk": "Жауаптар хабарламаңыздың тілінде беріледі.",
        "en": "Answers follow the language of your message."},
    "new_chat": {"ru": "Новый чат", "kk": "Жаңа чат", "en": "New chat"},
    "new_chat_help": {"ru": "Начать новый разговор; прошлые чаты останутся в истории ниже",
                      "kk": "Жаңа әңгіме бастау; бұрынғы чаттар төмендегі тарихта қалады",
                      "en": "Start a new conversation; past chats stay in the history below"},
    "scenarios": {"ru": "Сценарии", "kk": "Сценарийлер", "en": "Scenarios"},
    "chats": {"ru": "История чатов", "kk": "Чат тарихы", "en": "Chat history"},
    "chats_empty": {"ru": "Пока нет чатов. Задайте вопрос — и он появится здесь.",
                    "kk": "Әзірге чат жоқ. Сұрақ қойыңыз — ол осында пайда болады.",
                    "en": "No chats yet. Ask a question and it will appear here."},

    # Chat
    "greeting": {
        "ru": "Привет! Я Astana Walk — подскажу, куда сходить в Астане. Напишите, что хочется, "
              "или выберите готовый сценарий. Я подберу 1–3 реальных места рядом, покажу, сколько идти, "
              "открыто ли сейчас и чем там заняться.",
        "kk": "Сәлем! Мен Astana Walk — Астанада қайда баруға болатынын айтамын. Не қалайтыныңызды "
              "жазыңыз немесе дайын сценарийді таңдаңыз. Жақын жерден 1–3 нақты орын тауып, қанша жүру "
              "керек екенін, қазір ашық па және сонда не істеуге болатынын көрсетемін.",
        "en": "Hi! I'm Astana Walk — I'll suggest where to go in Astana. Tell me what you feel like, "
              "or pick a ready-made scenario. I'll find 1–3 real places nearby and show how far they are, "
              "whether they're open now and what to do there."},
    "placeholder": {"ru": "Напишите, куда хочется сходить… например: тихое кафе с розеткой на час",
                    "kk": "Қайда барғыңыз келетінін жазыңыз… мысалы: бір сағатқа розеткасы бар тыныш кафе",
                    "en": "Where would you like to go… e.g. a quiet café with a socket for an hour"},
    "placeholder_voice": {"ru": "Напишите или нажмите микрофон и скажите, куда хочется сходить",
                          "kk": "Жазыңыз немесе микрофонды басып, қайда барғыңыз келетінін айтыңыз",
                          "en": "Type, or tap the mic and say where you'd like to go"},
    "skeleton": {"ru": "Подбираю места рядом и пишу объяснения…",
                 "kk": "Жақын орындарды іздеп, түсініктеме жазып жатырмын…",
                 "en": "Finding places nearby and writing explanations…"},
    "you": {"ru": "Вы", "kk": "Сіз", "en": "You"},
    "meta_radius": {"ru": "радиус {km} км", "kk": "радиус {km} км", "en": "radius {km} km"},
    "meta_step_free": {"ru": "без ступенек", "kk": "баспалдақсыз", "en": "step-free"},
    "meta_people": {"ru": "{n} чел.", "kk": "{n} адам", "en": "{n} people"},
    "meta_voice": {"ru": "голос", "kk": "дауыс", "en": "voice"},
    "found_1": {"ru": "Нашёл 1 место рядом. Вот что подойдёт:",
                "kk": "Жақын жерден 1 орын таптым. Мынау сәйкес келеді:",
                "en": "I found 1 place nearby. Here's what fits:"},
    "found_n": {"ru": "Нашёл {n} места рядом. Вот что подойдёт:",
                "kk": "Жақын жерден {n} орын таптым. Мыналар сәйкес келеді:",
                "en": "I found {n} places nearby. Here's what fits:"},
    "nothing_found": {"ru": "Рядом ничего не нашлось.", "kk": "Жақын жерде ештеңе табылмады.",
                      "en": "Nothing found nearby."},
    "about_data": {"ru": "Об источниках и точности данных", "kk": "Дереккөздер мен дәлдік туралы",
                   "en": "About sources and data accuracy"},

    # Context chips
    "chip_anchor": {"ru": "ориентир", "kk": "бағдар", "en": "landmark"},
    "chip_tags": {"ru": "теги", "kk": "тегтер", "en": "tags"},
    "chip_filters": {"ru": "фильтры", "kk": "сүзгілер", "en": "filters"},
    "chip_time": {"ru": "время", "kk": "уақыт", "en": "time"},
    "chip_auto": {"ru": "авто", "kk": "авто", "en": "auto"},
    "chip_checked": {"ru": "проверено мест", "kk": "тексерілген орын", "en": "places checked"},
    "chip_answer": {"ru": "ответ", "kk": "жауап", "en": "answer"},
    "chip_template": {"ru": "по шаблону", "kk": "үлгі бойынша", "en": "template"},
    "min": {"ru": "мин", "kk": "мин", "en": "min"},

    # Cards
    "card_away": {"ru": "<b>{d} м</b> от вас", "kk": "<b>{d} м</b> қашықтықта", "en": "<b>{d} m</b> away"},
    "card_walk": {"ru": "<b>~{m} мин</b> пешком", "kk": "жаяу <b>~{m} мин</b>", "en": "<b>~{m} min</b> walk"},
    "card_visit": {"ru": "<b>~{m} мин</b> на месте", "kk": "орнында <b>~{m} мин</b>", "en": "<b>~{m} min</b> there"},
    "why": {"ru": "Почему: ", "kk": "Неге: ", "en": "Why: "},
    "what": {"ru": "Что делать: ", "kk": "Не істеу керек: ", "en": "What to do: "},
    "link_walk": {"ru": "Маршрут пешком · Google Maps", "kk": "Жаяу бағыт · Google Maps",
                  "en": "Walking route · Google Maps"},
    "link_2gis": {"ru": "Открыть в 2GIS", "kk": "2GIS-те ашу", "en": "Open in 2GIS"},
    "facts_summary": {"ru": "Факты из базы и что добавил ИИ", "kk": "Базадағы фактілер және ЖИ қосқаны",
                      "en": "Database facts and what the AI added"},
    "facts": {"ru": "Факты [poi_data]:", "kk": "Фактілер [poi_data]:", "en": "Facts [poi_data]:"},
    "original_name": {"ru": "Название в источнике:", "kk": "Дереккөздегі атауы:", "en": "Name in source:"},
    "meters": {"ru": "м", "kk": "м", "en": "m"},
    "boundary": {"ru": "Граница фактов и генерации:", "kk": "Фактілер мен генерацияның шекарасы:",
                 "en": "Facts vs generated text:"},
    "visit_source": {"ru": "Источник: Visit Astana", "kk": "Дереккөз: Visit Astana", "en": "Source: Visit Astana"},
    "received": {"ru": "получено {date}", "kk": "алынды {date}", "en": "retrieved {date}"},

    # Map and route
    "map_route": {"ru": "Карта и маршрут", "kk": "Карта және бағыт", "en": "Map and route"},
    "map_sub": {"ru": "По последнему ответу · шестерёнка открывает настройки",
                "kk": "Соңғы жауап бойынша · тісті доңғалақ баптауларды ашады",
                "en": "For the latest answer · the gear opens settings"},
    "map_empty": {"ru": "Здесь появится карта: места из ответа с номерами, точка старта и пеший маршрут.",
                  "kk": "Мұнда карта шығады: жауаптағы нөмірленген орындар, бастау нүктесі және жаяу бағыт.",
                  "en": "The map will appear here: numbered places from the answer, the start point "
                        "and a walking route."},
    "how_it_works": {"ru": "Как это работает: поиск, кандидаты и промпт",
                     "kk": "Бұл қалай жұмыс істейді: іздеу, үміткерлер және промпт",
                     "en": "How it works: search, candidates and prompt"},
    "map_start": {"ru": "Точка старта (~100 м, точные координаты не сохраняются)",
                  "kk": "Бастау нүктесі (~100 м, нақты координаттар сақталмайды)",
                  "en": "Start point (~100 m, exact coordinates not stored)"},
    "map_route_step": {"ru": "Маршрут, шаг {n}: {name}", "kk": "Бағыт, {n}-қадам: {name}",
                       "en": "Route, step {n}: {name}"},
    "map_legend_tail": {"ru": "Номер — место из ответа · круг — точка старта · линия — маршрут",
                        "kk": "Нөмір — жауаптағы орын · шеңбер — бастау нүктесі · сызық — бағыт",
                        "en": "Number — place from the answer · circle — start point · line — route"},
    "route_needs_two": {"ru": "Для маршрута нужно минимум 2 подходящих места в радиусе.",
                        "kk": "Бағыт үшін радиуста кемінде 2 сәйкес орын керек.",
                        "en": "A route needs at least 2 suitable places within the radius."},
    "route_head": {"ru": "{title} · всего ~{t} мин, пешком {d} м",
                   "kk": "{title} · барлығы ~{t} мин, жаяу {d} м",
                   "en": "{title} · ~{t} min total, {d} m on foot"},
    "route_row": {"ru": "+{a}…+{b} мин от старта · идти {m} м (~{w} мин) · на месте ~{v} мин",
                  "kk": "бастаудан +{a}…+{b} мин · жүру {m} м (~{w} мин) · орнында ~{v} мин",
                  "en": "+{a}…+{b} min from start · walk {m} m (~{w} min) · ~{v} min there"},
    "stop_step": {"ru": "Шаг", "kk": "Қадам", "en": "Step"},
    "stop_walk": {"ru": "Переход", "kk": "Өту", "en": "Walk"},
    "stop_visit": {"ru": "На месте", "kk": "Орнында", "en": "There"},
    "stop_offset": {"ru": "От старта", "kk": "Бастаудан", "en": "From start"},

    # Share
    "share_btn": {"ru": "На телефон", "kk": "Телефонға", "en": "To phone"},
    "share_hint": {"ru": "Наведите камеру телефона — маршрут откроется в Google Maps от того места, где вы стоите.",
                   "kk": "Телефон камерасын бағыттаңыз — бағыт Google Maps-та сіз тұрған жерден ашылады.",
                   "en": "Point your phone camera here — the route opens in Google Maps from where you stand."},
    "share_privacy": {"ru": "В ссылке только координаты мест, вашей точки там нет.",
                      "kk": "Сілтемеде тек орындардың координаттары бар, сіздің нүктеңіз жоқ.",
                      "en": "The link has only the places' coordinates, not yours."},
    "share_open": {"ru": "Открыть в Google Maps", "kk": "Google Maps-та ашу", "en": "Open in Google Maps"},
    "share_big": {"ru": "Показать крупно", "kk": "Үлкейтіп көрсету", "en": "Show large"},

    # Feedback
    "fb_up": {"ru": "Подходит", "kk": "Сәйкес келеді", "en": "Good pick"},
    "fb_down": {"ru": "Не то", "kk": "Сәйкес емес", "en": "Not right"},
    "fb_reason_closed": {"ru": "Закрыто", "kk": "Жабық", "en": "Closed"},
    "fb_reason_far": {"ru": "Далеко", "kk": "Алыс", "en": "Too far"},
    "fb_reason_irrelevant": {"ru": "Не то, что просил", "kk": "Сұрағаным емес", "en": "Not what I asked"},
    "fb_reason_wrong_data": {"ru": "Неверные данные", "kk": "Деректер қате", "en": "Wrong data"},
    "fb_thanks": {"ru": "Спасибо! Оценка учтена анонимно.", "kk": "Рахмет! Бағаңыз анонимді түрде ескерілді.",
                  "en": "Thanks! Your rating was saved anonymously."},
    "fb_closed_warning": {"ru": "⚠ Пользователи сообщили, что место закрыто: {n}",
                          "kk": "⚠ Пайдаланушылар орынның жабық екенін хабарлады: {n}",
                          "en": "⚠ Users reported this place as closed: {n}"},
    "fb_summary": {"ru": "Отзывы пользователей: {votes} оценок, {pct}% 👍, мест с пометкой «закрыто»: {closed}. "
                         "Оценки сдвигают место в выдаче максимум на несколько позиций.",
                   "kk": "Пайдаланушы пікірлері: {votes} баға, {pct}% 👍, «жабық» деп белгіленген орын: {closed}. "
                         "Бағалар орынды тізімде бірнеше позицияға ғана жылжытады.",
                   "en": "User feedback: {votes} ratings, {pct}% 👍, places marked closed: {closed}. "
                         "Ratings move a place by a few positions at most."},

    # Events
    "ev_title": {"ru": "Рядом в это время", "kk": "Осы уақытта жақын жерде", "en": "Nearby around this time"},
    "ev_temporary": {"ru": "Временное событие", "kk": "Уақытша іс-шара", "en": "Temporary event"},
    "ev_soon": {"ru": "Начнётся через {t}", "kk": "{t} кейін басталады", "en": "Starts in {t}"},
    "ev_ongoing": {"ru": "Идёт сейчас", "kk": "Қазір жүріп жатыр", "en": "On now"},
    "ev_later": {"ru": "Начало в {time}", "kk": "Басталуы {time}", "en": "Starts at {time}"},
    "ev_source": {"ru": "Источник: официальная афиша, проверено {date}",
                  "kk": "Дереккөз: ресми афиша, тексерілді {date}",
                  "en": "Source: official listing, checked {date}"},
    "ev_note": {"ru": "Расписание может измениться — уточните на сайте перед выходом.",
                "kk": "Кесте өзгеруі мүмкін — шығар алдында сайттан нақтылаңыз.",
                "en": "Schedules can change — check the site before you go."},
    "dur_h": {"ru": "{h} ч", "kk": "{h} сағ", "en": "{h} h"},
    "dur_hm": {"ru": "{h} ч {m} мин", "kk": "{h} сағ {m} мин", "en": "{h} h {m} min"},
    "dur_m": {"ru": "{m} мин", "kk": "{m} мин", "en": "{m} min"},

    # Inspector
    "how_caption": {"ru": "Сначала поиск находит реальные места рядом, затем нейросеть пишет объяснение только по ним.",
                    "kk": "Алдымен іздеу жақын маңдағы нақты орындарды табады, содан кейін нейрожелі тек солар "
                          "бойынша түсініктеме жазады.",
                    "en": "First, search finds real places nearby; then the AI writes explanations only about them."},
    "how_step1": {"ru": "**1. Что система поняла из запроса** (`used_data`)",
                  "kk": "**1. Жүйе сұраныстан не түсінді** (`used_data`)",
                  "en": "**1. What the system understood** (`used_data`)"},
    "how_step2": {"ru": "**2. Кандидаты из поиска до генерации**",
                  "kk": "**2. Генерацияға дейінгі іздеу үміткерлері**",
                  "en": "**2. Search candidates before generation**"},
    "how_step3": {"ru": "**3. Что ушло в нейросеть** (координат пользователя здесь нет)",
                  "kk": "**3. Нейрожеліге не жіберілді** (пайдаланушы координаттары жоқ)",
                  "en": "**3. What was sent to the AI** (no user coordinates)"},
    "col_place": {"ru": "Место", "kk": "Орын", "en": "Place"},
    "col_category": {"ru": "Категория", "kk": "Санат", "en": "Category"},
    "col_meters": {"ru": "Метры", "kk": "Метр", "en": "Meters"},
    "col_walk": {"ru": "Мин пешком", "kk": "Жаяу мин", "en": "Walk min"},
    "col_match": {"ru": "Совпадение", "kk": "Сәйкестік", "en": "Match"},
    "col_hours": {"ru": "Часы", "kk": "Жұмыс уақыты", "en": "Hours"},

    # Settings drawer
    "settings_title": {"ru": "Настройки прогулки", "kk": "Серуен баптаулары", "en": "Walk settings"},
    "settings_sub": {"ru": "Применяются к следующему вопросу", "kk": "Келесі сұраққа қолданылады",
                     "en": "Applied to the next question"},
    "close": {"ru": "Закрыть", "kk": "Жабу", "en": "Close"},
    "close_help": {"ru": "Свернуть панель настроек", "kk": "Баптаулар панелін жасыру", "en": "Hide the settings panel"},
    "geo_btn": {"ru": "Найти мою геолокацию", "kk": "Геолокациямды табу", "en": "Use my location"},
    "geo_help": {"ru": "Браузер запросит разрешение; координаты сразу округляем до ~100 м, сырые не сохраняем.",
                 "kk": "Браузер рұқсат сұрайды; координаттарды бірден ~100 м-ге дейін дөңгелектейміз, "
                       "бастапқы мәндерді сақтамаймыз.",
                 "en": "The browser will ask for permission; we round coordinates to ~100 m right away "
                       "and never store the raw ones."},
    "geo_unavailable": {"ru": "Геолокация недоступна (нет компонента) — выберите ориентир или введите координаты.",
                        "kk": "Геолокация қолжетімсіз (компонент жоқ) — бағдарды таңдаңыз немесе координаттарды "
                              "енгізіңіз.",
                        "en": "Location is unavailable (component missing) — pick a landmark or enter coordinates."},
    "geo_wait": {"ru": "Разрешите геолокацию рядом с адресной строкой. Ожидание — до 15 секунд.",
                 "kk": "Мекенжай жолағының жанында геолокацияға рұқсат беріңіз. Күту — 15 секундқа дейін.",
                 "en": "Allow location access next to the address bar. Waiting up to 15 seconds."},
    "geo_ok": {"ru": "Ищу рядом с вашей геолокацией: ~{lat}, {lon}{tail} · округлено до ~100 м",
               "kk": "Геолокацияңыз бойынша іздеймін: ~{lat}, {lon}{tail} · ~100 м-ге дейін дөңгелектенді",
               "en": "Searching near your location: ~{lat}, {lon}{tail} · rounded to ~100 m"},
    "geo_accuracy": {"ru": " · точность браузера ±{acc} м", "kk": " · браузер дәлдігі ±{acc} м",
                     "en": " · browser accuracy ±{acc} m"},
    "start_point": {"ru": "Точка старта", "kk": "Бастау нүктесі", "en": "Start point"},
    "start_help": {"ru": "Ориентир вместо геолокации: так не нужно раскрывать, где вы находитесь.",
                   "kk": "Геолокацияның орнына бағдар: осылай қай жерде екеніңізді ашудың қажеті жоқ.",
                   "en": "A landmark instead of your location: no need to reveal where you are."},
    "start_geo": {"ru": "ваша геолокация", "kk": "сіздің геолокацияңыз", "en": "your location"},
    "start_coords": {"ru": "точка {lat}, {lon}", "kk": "нүкте {lat}, {lon}", "en": "point {lat}, {lon}"},
    "exact_coords": {"ru": "Точные координаты", "kk": "Нақты координаттар", "en": "Exact coordinates"},
    "lat": {"ru": "Широта", "kk": "Ендік", "en": "Latitude"},
    "lon": {"ru": "Долгота", "kk": "Бойлық", "en": "Longitude"},
    "coords_caption": {"ru": "В логи и API попадает только округление до ~100 м: `{lat}, {lon}`",
                       "kk": "Логтар мен API-ге тек ~100 м-ге дейін дөңгелектенген мән түседі: `{lat}, {lon}`",
                       "en": "Logs and the API only get the ~100 m rounding: `{lat}, {lon}`"},
    "radius": {"ru": "Радиус пешком, м", "kk": "Жаяу радиус, м", "en": "Walking radius, m"},
    "date": {"ru": "Дата", "kk": "Күні", "en": "Date"},
    "time": {"ru": "Время", "kk": "Уақыты", "en": "Time"},
    "time_caption": {"ru": "Время Астаны (UTC+5). Влияет на статус «открыто / закрыто».",
                     "kk": "Астана уақыты (UTC+5). «Ашық / жабық» мәртебесіне әсер етеді.",
                     "en": "Astana time (UTC+5). Affects the open / closed status."},
    "weather": {"ru": "Погода", "kk": "Ауа райы", "en": "Weather"},
    "group": {"ru": "Сколько человек", "kk": "Неше адам", "en": "Group size"},
    "toggle_step_free": {"ru": "Без ступенек: коляска, дети", "kk": "Баспалдақсыз: арба, балалар",
                         "en": "Step-free: wheelchair, stroller"},
    "toggle_route": {"ru": "Собрать маршрут из 2–3 мест", "kk": "2–3 орыннан бағыт құру",
                     "en": "Build a route of 2–3 places"},
    "toggle_anchor": {"ru": "Понимать ориентиры из текста запроса", "kk": "Сұрақ мәтініндегі бағдарларды тану",
                      "en": "Detect landmarks in the question"},
    "toggle_anchor_help": {"ru": "Например, «я возле Хан Шатыра» сдвинет точку старта к ТРЦ.",
                           "kk": "Мысалы, «мен Хан Шатырдың жанындамын» бастау нүктесін СОО-ға жылжытады.",
                           "en": "E.g. “I'm near Khan Shatyr” moves the start point to the mall."},
    "foot": {"ru": "База: {n} мест · OpenStreetMap + Visit Astana · поиск",
             "kk": "База: {n} орын · OpenStreetMap + Visit Astana · іздеу",
             "en": "Database: {n} places · OpenStreetMap + Visit Astana · search"},
}

STATUS_LABEL: dict[str, dict[str, str]] = {
    "confirmed_open": {"ru": "Открыто по расписанию в базе", "kk": "Кесте бойынша ашық",
                       "en": "Open per schedule in our data"},
    "likely_open": {"ru": "Скоро закроется", "kk": "Жақында жабылады", "en": "Closing soon"},
    "unknown": {"ru": "Часы работы неизвестны", "kk": "Жұмыс уақыты белгісіз", "en": "Opening hours unknown"},
    "likely_closed": {"ru": "Закрыто по расписанию в базе", "kk": "Кесте бойынша жабық",
                      "en": "Closed per schedule in our data"},
}

SOURCE_LABEL: dict[str, dict[str, str]] = {
    "poi_data": {"ru": "Факты из базы мест", "kk": "Базадағы фактілер", "en": "Facts from the places database"},
    "generated_idea": {"ru": "Факты из базы + идея ИИ", "kk": "Фактілер + ЖИ идеясы", "en": "Facts + AI idea"},
}

CATEGORY: dict[str, dict[str, str]] = {
    "cafe": {"ru": "Кафе", "kk": "Кафе", "en": "Café"},
    "restaurant": {"ru": "Ресторан", "kk": "Мейрамхана", "en": "Restaurant"},
    "fast_food": {"ru": "Фастфуд", "kk": "Фастфуд", "en": "Fast food"},
    "bar": {"ru": "Бар", "kk": "Бар", "en": "Bar"},
    "mall": {"ru": "Торговый центр", "kk": "Сауда орталығы", "en": "Shopping mall"},
    "park": {"ru": "Парк", "kk": "Саябақ", "en": "Park"},
    "sport": {"ru": "Спорт", "kk": "Спорт", "en": "Sport"},
    "monument": {"ru": "Памятник", "kk": "Ескерткіш", "en": "Monument"},
    "attraction": {"ru": "Достопримечательность", "kk": "Көрікті орын", "en": "Attraction"},
    "place_of_worship": {"ru": "Мечеть / храм", "kk": "Мешіт / ғибадатхана", "en": "Mosque / temple"},
    "artwork": {"ru": "Арт-объект", "kk": "Арт-нысан", "en": "Art object"},
    "cinema": {"ru": "Кинотеатр", "kk": "Кинотеатр", "en": "Cinema"},
    "theatre": {"ru": "Театр", "kk": "Театр", "en": "Theatre"},
    "library": {"ru": "Библиотека", "kk": "Кітапхана", "en": "Library"},
    "museum": {"ru": "Музей", "kk": "Мұражай", "en": "Museum"},
    "bookstore": {"ru": "Книжный", "kk": "Кітап дүкені", "en": "Bookstore"},
    "arts_centre": {"ru": "Культурный центр", "kk": "Мәдени орталық", "en": "Cultural centre"},
    "stadium": {"ru": "Стадион", "kk": "Стадион", "en": "Stadium"},
    "gallery": {"ru": "Галерея", "kk": "Галерея", "en": "Gallery"},
    "theme_park": {"ru": "Парк развлечений", "kk": "Ойын-сауық паркі", "en": "Amusement park"},
    "exhibition": {"ru": "Выставочный центр", "kk": "Көрме орталығы", "en": "Exhibition centre"},
    "viewpoint": {"ru": "Смотровая площадка", "kk": "Шолу алаңы", "en": "Viewpoint"},
    "zoo": {"ru": "Зоопарк", "kk": "Хайуанаттар бағы", "en": "Zoo"},
}

EVENT_KIND: dict[str, dict[str, str]] = {
    "performance": {"ru": "🎭 Спектакль", "kk": "🎭 Қойылым", "en": "🎭 Performance"},
    "concert": {"ru": "🎵 Концерт", "kk": "🎵 Концерт", "en": "🎵 Concert"},
    "match": {"ru": "⚽ Матч", "kk": "⚽ Матч", "en": "⚽ Match"},
    "exhibition": {"ru": "🖼 Выставка", "kk": "🖼 Көрме", "en": "🖼 Exhibition"},
    "festival": {"ru": "🎪 Фестиваль", "kk": "🎪 Фестиваль", "en": "🎪 Festival"},
    "show": {"ru": "✨ Шоу", "kk": "✨ Шоу", "en": "✨ Show"},
}

WEATHER: dict[str, dict[str, str]] = {
    "normal": {"ru": "Комфортная погода", "kk": "Жайлы ауа райы", "en": "Pleasant weather"},
    "cold_windy": {"ru": "Мороз или сильный ветер", "kk": "Аяз немесе қатты жел", "en": "Frost or strong wind"},
    "rain_snow": {"ru": "Дождь или снег", "kk": "Жаңбыр немесе қар", "en": "Rain or snow"},
}

# Landmark picker: internal key (Russian name) -> (display name, note) per language.
LANDMARK: dict[str, dict[str, tuple[str, str]]] = {
    "Байтерек": {"ru": ("Байтерек", "Центр бульвара Нуржол"),
                 "kk": ("Бәйтерек", "Нұржол бульварының орталығы"),
                 "en": ("Baiterek", "Centre of Nurzhol Boulevard")},
    "Mega Silk Way": {"ru": ("Mega Silk Way", "Район EXPO / Назарбаев Университет"),
                      "kk": ("Mega Silk Way", "EXPO / Назарбаев Университеті ауданы"),
                      "en": ("Mega Silk Way", "EXPO / Nazarbayev University area")},
    "EXPO 2017 (Нур Алем)": {"ru": ("EXPO 2017 (Нур Алем)", "Выставочный кластер"),
                             "kk": ("EXPO 2017 (Нұр Әлем)", "Көрме кластері"),
                             "en": ("EXPO 2017 (Nur Alem)", "Exhibition cluster")},
    "ТРЦ Хан Шатыр": {"ru": ("ТРЦ Хан Шатыр", "Западный конец оси Нуржол"),
                      "kk": ("Хан Шатыр СОО", "Нұржол осінің батыс шеті"),
                      "en": ("Khan Shatyr mall", "West end of the Nurzhol axis")},
    "Астана Опера": {"ru": ("Астана Опера", "Культурный кластер Левого берега"),
                     "kk": ("Астана Опера", "Сол жағалаудың мәдени кластері"),
                     "en": ("Astana Opera", "Left Bank cultural cluster")},
    "Ботанический сад": {"ru": ("Ботанический сад", "Парковая зона"),
                         "kk": ("Ботаникалық бақ", "Саябақ аймағы"),
                         "en": ("Botanical Garden", "Park area")},
    "Старый центр": {"ru": ("Старый центр", "Правый берег, историческая часть"),
                     "kk": ("Ескі орталық", "Оң жағалау, тарихи бөлік"),
                     "en": ("Old town", "Right Bank, historic part")},
    "За городом (тест пустой выдачи)": {"ru": ("За городом (тест пустой выдачи)", "Точка в степи без мест рядом"),
                                        "kk": ("Қала сыртында (бос нәтиже тесті)", "Маңында орын жоқ дала нүктесі"),
                                        "en": ("Outside the city (empty result test)",
                                               "A point in the steppe with no places nearby")},
    "Свои координаты": {"ru": ("Свои координаты", ""), "kk": ("Өз координаттарым", ""),
                        "en": ("My coordinates", "")},
}

# Jury scenarios by preset id: (button title, question) per language. Kazakh and English
# questions keep landmark names and time ranges the intent parser understands.
PRESET: dict[str, dict[str, tuple[str, str]]] = {
    "preset_mega_walk": {
        "ru": ("Прогулка у Mega Silk Way (30–60 мин)",
               "Я сейчас возле Mega Silk Way, хочу прогуляться 30–60 минут — что посоветуете?"),
        "kk": ("Mega Silk Way маңында серуен (30–60 мин)",
               "Мен қазір Mega Silk Way жанындамын, 30–60 минут серуендегім келеді — не ұсынасыз?"),
        "en": ("Walk near Mega Silk Way (30–60 min)",
               "I'm near Mega Silk Way right now and want a 30–60 minute walk — any ideas?"),
    },
    "preset_coffee_laptop": {
        "ru": ("Тихое место с кофе и розеткой", "Хочу тихое место с кофе и розеткой недалеко от меня."),
        "kk": ("Кофе мен розеткасы бар тыныш орын", "Жақын жерде кофесі мен розеткасы бар тыныш орын керек."),
        "en": ("Quiet spot with coffee and a socket", "I need a quiet place with coffee and a power outlet nearby."),
    },
    "preset_kids_museum": {
        "ru": ("С ребёнком: детские площадки / музеи",
               "Еду с ребёнком — где рядом есть детские площадки или музей с интерактивами?"),
        "kk": ("Баламен: ойын алаңдары / мұражайлар",
               "Баламен келе жатырмын — жақын жерде балалар алаңы немесе интерактивті мұражай бар ма?"),
        "en": ("With a child: playgrounds / museums",
               "I'm with a child — are there playgrounds or an interactive museum nearby?"),
    },
    "preset_tourist_flight": {
        "ru": ("Турист: 2 часа перед вылетом",
               "Я - турист, у меня 2 часа перед вылетом: что можно посмотреть поблизости?"),
        "kk": ("Турист: ұшуға дейін 2 сағат",
               "Мен туристпін, ұшуға дейін 2 сағатым бар: жақын жерде не көруге болады?"),
        "en": ("Tourist: 2 hours before a flight",
               "I'm a tourist with 2 hours before my flight: what can I see nearby?"),
    },
    "preset_night_activity": {
        "ru": ("Необычная ночная активность", "Ищу необычную ночную активность (последние пару часов)."),
        "kk": ("Ерекше түнгі демалыс", "Түнде екі сағатқа ерекше бір іс-шара іздеп жүрмін."),
        "en": ("Something unusual at night", "Looking for something unusual to do tonight for a couple of hours."),
    },
    "preset_evening_events": {
        "ru": ("Вечер в городе: что идёт сегодня", "Чем заняться сегодня вечером? Хочется чего-то культурного рядом."),
        "kk": ("Қаладағы кеш: бүгін не өтеді", "Бүгін кешке немен айналысуға болады? Жақын жерде мәдени бірдеңе қалаймын."),
        "en": ("Evening in the city: what's on tonight", "What can I do tonight? I'd like something cultural nearby."),
    },
    "preset_expo_unusual": {
        "ru": ("У здания Экспо 2017 (необычное на 1–2 часа)",
               "Я у здании Экспо 2017, хочу что-то необычное на 1–2 часа, пешком."),
        "kk": ("ЭКСПО-2017 жанында (1–2 сағатқа ерекше)",
               "Мен ЭКСПО-2017 ғимаратының жанындамын, жаяу 1–2 сағатқа ерекше бірдеңе қалаймын."),
        "en": ("At EXPO 2017 (unusual, 1–2 hours)",
               "I'm at the EXPO 2017 building and want something unusual for 1–2 hours, on foot."),
    },
}

GEO_ERRORS: dict[str, dict[str, str]] = {
    "1": {"kk": "Геолокацияға рұқсат жоқ. Мекенжай жолағының жанындағы сайт баптауларынан рұқсат беріңіз.",
          "en": "Location access is denied. Allow it in the site settings next to the address bar."},
    "2": {"kk": "Браузер координаттарды анықтай алмады. Интернетті және жүйенің геолокация қызметін тексеріңіз.",
          "en": "The browser could not determine the location. Check the internet and system location services."},
    "3": {"kk": "Браузер координаттарды 10 секундта анықтамады.",
          "en": "The browser did not get a location within 10 seconds."},
    "deadline": {"kk": "15 секундта координаттар алынбады. Мекенжай жолағының жанындағы рұқсат сұрауын тексеріңіз.",
                 "en": "No location within 15 seconds. Check the permission prompt next to the address bar."},
    "insecure": {"kk": "Геолокация үшін қосымшаны HTTPS арқылы немесе localhost / 127.0.0.1 мекенжайында ашыңыз.",
                 "en": "For location access, open the app over HTTPS or locally on localhost / 127.0.0.1."},
    "unsupported": {"kk": "Бұл браузер геолокацияны қолдамайды.", "en": "This browser does not support geolocation."},
}
GEO_RETRY = {"kk": " Қайталап көріңіз немесе бастау нүктесін қолмен таңдаңыз.",
             "en": " Try again or pick the start point manually."}
GEO_GENERIC = {"kk": "Браузер дұрыс координаттарды бере алмады.", "en": "The browser could not pass valid coordinates."}


def norm_lang(lang: str | None) -> str:
    return lang if lang in LANGS else "ru"


def t(key: str, lang: str | None, **values) -> str:
    """Text for `key` in `lang` (Russian fallback), with `{placeholders}` filled in."""
    entry = TEXT[key]
    text = _answer_label("text:" + key, lang) or entry.get(lang) or entry["en" if _ANSWER_LOCALE.get() else "ru"]
    return text.format(**values) if values else text


def _pick(table: dict[str, dict], key: str, lang: str | None, default, group: str = ""):
    localized = _answer_label(group + ":" + key, lang)
    if localized:
        return localized
    entry = table.get(key)
    if not entry:
        return default
    return entry.get(lang) or entry.get("en" if _ANSWER_LOCALE.get() else "ru", default)


def status_label(status: str, lang: str | None) -> str:
    return _pick(STATUS_LABEL, status, lang, None, "status") or _pick(STATUS_LABEL, "unknown", lang, status, "status")


def source_label(source: str, lang: str | None) -> str:
    return _pick(SOURCE_LABEL, source, lang, None, "source") or _pick(SOURCE_LABEL, "poi_data", lang, source, "source")


def category_label(category: str, lang: str | None) -> str:
    return _pick(CATEGORY, category, lang, category, "category")


def event_kind_label(kind: str, lang: str | None) -> str:
    return _pick(EVENT_KIND, kind, lang, kind, "event")


def duration_text(minutes: int, lang: str | None) -> str:
    h, m = divmod(max(0, minutes), 60)
    if h and m:
        return t("dur_hm", lang, h=h, m=m)
    return t("dur_h", lang, h=h) if h else t("dur_m", lang, m=m)


def weather_label(weather: str, lang: str | None) -> str:
    return _pick(WEATHER, weather, lang, weather, "weather")


def landmark_name(key: str, lang: str | None) -> str:
    return _pick(LANDMARK, key, lang, (key, ""))[0]


def landmark_note(key: str, lang: str | None) -> str:
    return _pick(LANDMARK, key, lang, (key, ""))[1]


def preset_title(preset: dict, lang: str | None) -> str:
    return _pick(PRESET, preset["id"], lang, (preset["title"].split(". ", 1)[-1], ""))[0]


def preset_query(preset: dict, lang: str | None) -> str:
    return _pick(PRESET, preset["id"], lang, ("", preset["query"]))[1]


def geo_error(code: object, lang: str | None, russian: str) -> str:
    """Russian keeps the original wording from frontend.geolocation; other languages translate it."""
    lang = norm_lang(lang)
    if lang == "ru":
        return russian
    hint = GEO_ERRORS.get(str(code), {}).get(lang) or GEO_GENERIC[lang]
    return hint + GEO_RETRY[lang]


def answer_dictionary() -> dict[str, str]:
    """English templates to translate for additional answer languages; numbers stay in Python."""
    prefixes = ("card_", "found_", "chip_", "map_", "route_", "stop_", "dur_", "fb_", "share_", "ev_")
    keys = {"why", "what", "nothing_found", "about_data", "facts_summary", "facts", "boundary",
            "link_walk", "link_2gis", "visit_source", "received", "min", "meters", "how_it_works",
            "original_name", "mode_llm"}
    values = {"text:" + key: entry["en"] for key, entry in TEXT.items()
              if key.startswith(prefixes) or key in keys}
    for prefix, table in (("status", STATUS_LABEL), ("source", SOURCE_LABEL), ("category", CATEGORY),
                          ("weather", WEATHER), ("event", EVENT_KIND)):
        values.update({prefix + ":" + key: entry["en"] for key, entry in table.items()})
    return values

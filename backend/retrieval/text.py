"""Build the text that gets embedded for each POI.

OSM rarely has descriptions, so we enrich the name with human words (ru + en) for the
category and tags. This lets queries like "тихо почитать" or "где поесть бешбармак" match
POIs that only have a name and a couple of OSM tags.
"""

from __future__ import annotations

from .schema import POI

CATEGORY_WORDS: dict[str, str] = {
    "cafe": "кафе кофейня кофе десерты посидеть cafe coffee",
    "restaurant": "ресторан поесть обед ужин restaurant dinner",
    "fast_food": "фастфуд быстро перекусить fast food snack",
    "bar": "бар паб выпить вечер bar pub drinks",
    "mall": "торговый центр ТРЦ шопинг магазины в помещении mall shopping",
    "park": "парк прогулка погулять свежий воздух природа park walk",
    "sport": "спорт тренировка каток sport",
    "monument": "памятник монумент достопримечательность monument landmark",
    "attraction": "достопримечательность интересное место attraction sightseeing",
    "viewpoint": "смотровая площадка вид на город viewpoint",
    "place_of_worship": "храм мечеть церковь архитектура place of worship",
    "artwork": "арт-объект скульптура искусство artwork sculpture",
    "cinema": "кинотеатр кино фильм cinema movie",
    "theatre": "театр спектакль концерт культура theatre",
    "library": "библиотека книги тихо почитать поработать library quiet",
    "museum": "музей история выставка экспозиция museum history exhibition",
    "bookstore": "книжный магазин книги bookstore books",
    "arts_centre": "культурный центр искусство выставки arts centre",
    "gallery": "галерея искусство выставка gallery art",
    "theme_park": "парк развлечений аттракционы дети theme park",
    "zoo": "зоопарк животные дети zoo",
    "stadium": "стадион арена матч футбол хоккей концерт stadium arena",
    "exhibition": "выставочный центр выставка экспо exhibition expo",
}

TAG_WORDS: dict[str, str] = {
    "coffee": "кофе",
    "wifi": "wi-fi интернет поработать",
    "terrace": "летняя терраса на улице",
    "wheelchair": "доступно для колясок безбарьерная среда wheelchair accessible",
    "wheelchair_limited": "частично доступно для колясок",
    "halal": "халяль",
    "vegetarian": "вегетарианское меню",
    "free": "бесплатно",
    "24_7": "круглосуточно",
    "takeaway": "навынос с собой",
    "indoor": "в помещении",
    "outdoor": "на открытом воздухе",
    "culture": "культура",
    "landmark": "известная достопримечательность символ города что посмотреть туристу must-see landmark",
    "events": "мероприятия события",
    "muslim": "мечеть ислам",
    "christian": "церковь храм",
    "cuisine_coffee_shop": "кофейня кофе",
    "cuisine_kazakh": "казахская кухня бешбармак баурсаки",
    "cuisine_regional": "национальная кухня",
    "cuisine_uzbek": "узбекская кухня плов",
    "cuisine_georgian": "грузинская кухня хинкали",
    "cuisine_italian": "итальянская кухня паста",
    "cuisine_pizza": "пицца",
    "cuisine_burger": "бургеры",
    "cuisine_sushi": "суши роллы",
    "cuisine_japanese": "японская кухня суши",
    "cuisine_korean": "корейская кухня",
    "cuisine_chinese": "китайская кухня",
    "cuisine_asian": "азиатская кухня",
    "cuisine_turkish": "турецкая кухня",
    "cuisine_european": "европейская кухня",
    "cuisine_russian": "русская кухня",
    "cuisine_ice_cream": "мороженое",
    "cuisine_dessert": "десерты",
    "cuisine_cake": "торты десерты",
    "cuisine_tea": "чай",
    "cuisine_breakfast": "завтраки",
    "cuisine_brunch": "бранчи завтраки",
    "cuisine_oriental": "восточная кухня",
    "cuisine_local": "местная кухня",
    "cuisine_pastry": "выпечка",
    "cuisine_bubble_tea": "бабл-ти",
    "cuisine_chicken": "курица",
    "cuisine_kebab": "шашлык кебаб",
    "cuisine_shawarma": "шаурма",
    "cuisine_steak_house": "стейки",
}


def poi_document(poi: POI) -> str:
    parts = [poi.name, CATEGORY_WORDS.get(poi.category, poi.category)]
    for tag in poi.tags:
        if tag in TAG_WORDS:
            parts.append(TAG_WORDS[tag])
        elif tag.startswith("cuisine_"):
            parts.append(tag.removeprefix("cuisine_").replace("_", " "))
    if poi.description:
        parts.append(poi.description)
    if poi.opening_hours == "24/7":
        parts.append("круглосуточно")
    # Deduplicate words while keeping order, so tags like "coffee" don't dominate.
    seen: set[str] = set()
    words = []
    for w in " ".join(parts).split():
        if w.lower() not in seen:
            seen.add(w.lower())
            words.append(w)
    return " ".join(words)


# Query keyword (stem) -> POI tags that should get a lexical boost on top of embeddings.
# Embeddings blur exact amenities like wi-fi or wheelchair access; these hints keep them sharp.
QUERY_TAG_HINTS: dict[str, list[str]] = {
    "розет": ["wifi"], "ноутбук": ["wifi"], "поработ": ["wifi"], "wi-fi": ["wifi"],
    "wifi": ["wifi"], "вайфай": ["wifi"], "интернет": ["wifi"], "laptop": ["wifi"],
    "work": ["wifi"], "коляс": ["wheelchair"],  # also baby strollers: same step-free need
    "маломобил": ["wheelchair"], "инвалид": ["wheelchair"], "безбарьер": ["wheelchair"],
    "wheelchair": ["wheelchair"], "accessible": ["wheelchair"], "терас": ["terrace"],
    "на улице": ["terrace", "outdoor"], "outdoor": ["terrace", "outdoor"],
    "халял": ["halal"], "halal": ["halal"], "вегет": ["vegetarian"], "vegan": ["vegetarian"],
    "бесплат": ["free"], "free": ["free"], "круглосуточ": ["24_7"], "ночью": ["24_7"],
    "24/7": ["24_7"], "навынос": ["takeaway"], "с собой": ["takeaway"],
    "в помещении": ["indoor"], "холодно": ["indoor"], "дожд": ["indoor"], "indoor": ["indoor"],
    "казахск": ["cuisine_kazakh", "cuisine_regional"], "бешбармак": ["cuisine_kazakh", "cuisine_regional"],
    "завтрак": ["cuisine_breakfast", "cuisine_brunch"], "пицц": ["cuisine_pizza"],
    "суши": ["cuisine_sushi", "cuisine_japanese"], "бургер": ["cuisine_burger"],
    "что посмотреть": ["landmark"], "достопримечат": ["landmark"], "турист": ["landmark"],
    "символ": ["landmark"], "знаменит": ["landmark"], "известн": ["landmark"],
    "sightseeing": ["landmark"], "landmark": ["landmark"], "must-see": ["landmark"],
    "футбол": ["stadium", "events"], "хоккей": ["stadium", "events"], "матч": ["stadium", "events"],
    "стадион": ["stadium"], "выставк": ["exhibition_centre", "gallery", "museum"],
}


def query_tag_hints(query: str) -> set[str]:
    q = f" {query.lower()} "
    return {tag for stem, tags in QUERY_TAG_HINTS.items() if stem in q for tag in tags}

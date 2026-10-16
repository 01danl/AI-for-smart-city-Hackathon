"""Hybrid BM25 + Multilingual Stem & Tag Retriever over data/poi.json (966 POIs).

Serves two purposes:
1. Provides `lexical_similarity(query, poi)` for Hybrid Dense + Sparse fusion in `chroma_retriever.py`.
2. Acts as a fast, zero-external-dependency retrieval backend (`RETRIEVAL_BACKEND=bm25` or
   automatic fallback when Chroma index hasn't been built yet) over the full 966-POI dataset.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

from .geo import haversine_m, walk_minutes
from .hours import open_status
from .index import load_pois
from .schema import POI, Candidate
from .text import CATEGORY_WORDS, TAG_WORDS, poi_document, query_tag_hints

W_RELEVANCE = 0.65
W_PROXIMITY = 0.25
HOURS_ADJUST = {
    "confirmed_open": 0.10,
    "likely_open": 0.05,
    "unknown": 0.0,
    "likely_closed": -0.30,
}
TAG_BOOST = 0.18
HINT_BONUS = 0.18
MIN_RELEVANCE = 0.05

# Multilingual query stem -> expanded document tokens (ru / kk / en)
QUERY_EXPANSIONS: dict[str, list[str]] = {
    "кофе": ["cafe", "coffee", "кофе", "кофейня"],
    "кафе": ["cafe", "coffee", "кафе", "кофейня"],
    "розет": ["cafe", "coffee", "wifi", "поработать"],
    "тихо": ["cafe", "library", "park", "quiet", "тихо", "почитать"],
    "книг": ["library", "bookstore", "библиотека", "книги", "book"],
    "book": ["library", "bookstore", "quiet", "библиотека"],
    "read": ["library", "quiet", "почитать"],
    "quiet": ["library", "cafe", "park", "quiet", "тихо"],
    "погул": ["park", "walk", "outdoor", "парк", "прогулка", "сквер"],
    "прогул": ["park", "walk", "outdoor", "attraction", "парк", "прогулка"],
    "свеж": ["park", "outdoor", "парк", "воздух"],
    "воздух": ["park", "outdoor", "парк"],
    "музей": ["museum", "gallery", "exhibition", "музей", "история", "выставка"],
    "истори": ["museum", "monument", "музей", "история"],
    "казахск": ["restaurant", "казахская", "бешбармак", "национальная"],
    "семье": ["restaurant", "park", "cafe", "ресторан"],
    "семьё": ["restaurant", "park", "cafe", "ресторан"],
    "ребён": ["theme_park", "park", "museum", "zoo", "дети", "парк", "музей"],
    "ребен": ["theme_park", "park", "museum", "zoo", "дети", "парк", "музей"],
    "детск": ["theme_park", "park", "museum", "дети", "аттракционы"],
    "интерактив": ["museum", "exhibition", "theme_park", "science", "выставка", "музей"],
    "поесть": ["restaurant", "cafe", "fast_food", "поесть", "обед", "ужин"],
    "ужин": ["restaurant", "bar", "ресторан", "ужин"],
    "обед": ["restaurant", "cafe", "обед"],
    "коляс": ["wheelchair", "безбарьерная", "колясок"],
    "турист": ["landmark", "attraction", "monument", "museum", "туристу", "достопримечательность"],
    "посмотр": ["landmark", "attraction", "monument", "viewpoint", "museum"],
    "вылет": ["landmark", "attraction", "monument", "park", "viewpoint"],
    "необычн": ["attraction", "artwork", "exhibition", "gallery", "museum", "bar", "theatre"],
    "ночн": ["bar", "restaurant", "24_7", "круглосуточно", "вечер"],
    "футбол": ["stadium", "стадион", "арена", "матч", "футбол"],
    "матч": ["stadium", "стадион", "арена"],
    "стадион": ["stadium", "стадион", "арена"],
    "серуен": ["park", "outdoor", "walk", "парк"],
    "мұражай": ["museum", "gallery", "музей"],
    "мешіт": ["place_of_worship", "muslim", "мечеть"],
}

_STOPWORDS = {
    "я", "в", "на", "и", "с", "у", "к", "по", "из", "за", "для", "от", "до",
    "что", "где", "как", "мне", "мы", "сейчас", "рядом", "возле", "около",
    "хочу", "ищу", "меня", "может", "можно", "или", "надо", "нужно",
    "the", "a", "an", "to", "in", "on", "at", "for", "with", "near", "me",
}


def _stem_token(w: str) -> str:
    """Lightweight Russian/English stem normalization for lexical matching."""
    w = w.lower().strip()
    if len(w) <= 3:
        return w
    for suffix in (
        "ами", "ями", "ого", "его", "ому", "ему", "ую", "юю", "ая", "яя",
        "ой", "ей", "ий", "ый", "ое", "ее", "ие", "ые", "ам", "ям", "ах", "ях",
        "ов", "ев", "ей", "ом", "ем", "ть", "ться", "ка", "ки", "ку", "кой",
        "инг", "ing", "ed", "es", "s",
    ):
        if len(w) - len(suffix) >= 4 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def _tokenize(text: str) -> list[str]:
    raw = re.findall(r"[a-zа-яёәіңғүұқөһ0-9_]+", text.lower())
    tokens: list[str] = []
    for w in raw:
        if w in _STOPWORDS or len(w) < 2:
            continue
        tokens.append(w)
        stem = _stem_token(w)
        if stem != w:
            tokens.append(stem)
    return tokens


def _expand_query_tokens(query: str) -> list[str]:
    base = _tokenize(query)
    q_low = query.lower()
    extra: list[str] = []
    for stem, words in QUERY_EXPANSIONS.items():
        if stem in q_low:
            for w in words:
                extra.extend(_tokenize(w))
    for tag in query_tag_hints(query):
        extra.append(tag)
        if tag in TAG_WORDS:
            extra.extend(_tokenize(TAG_WORDS[tag]))
    return base + extra


@lru_cache(maxsize=1)
def _bm25_index() -> tuple[dict[str, Counter[str]], dict[str, float], float]:
    pois = load_pois()
    doc_freqs: Counter[str] = Counter()
    tf_map: dict[str, Counter[str]] = {}
    total_len = 0

    for poi_id, poi in pois.items():
        doc_text = f"{poi_document(poi)} {poi.category} {' '.join(poi.tags)}"
        toks = _tokenize(doc_text)
        cnt = Counter(toks)
        tf_map[poi_id] = cnt
        total_len += sum(cnt.values())
        for t in cnt:
            doc_freqs[t] += 1

    n_docs = max(1, len(pois))
    avgdl = max(1.0, total_len / n_docs)
    idf: dict[str, float] = {}
    for term, df in doc_freqs.items():
        idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
    return tf_map, idf, avgdl


def lexical_similarity(query: str, poi: POI) -> float:
    """Compute normalized [0..1] lexical + exact name match similarity for a POI."""
    q_clean = query.strip().lower()
    if not q_clean:
        return 0.5

    q_tokens = _expand_query_tokens(q_clean)
    if not q_tokens:
        return 0.0

    tf_map, idf, avgdl = _bm25_index()
    cnt = tf_map.get(poi.id)
    if cnt is None:
        doc_text = f"{poi_document(poi)} {poi.category} {' '.join(poi.tags)}"
        cnt = Counter(_tokenize(doc_text))

    doc_len = sum(cnt.values()) or 1
    k1, b = 1.5, 0.75
    raw_score = 0.0
    matched_unique = 0

    for t in set(q_tokens):
        tf = cnt.get(t, 0)
        if tf > 0:
            matched_unique += 1
            term_idf = idf.get(t, 1.5)
            denom = tf + k1 * (1.0 - b + b * doc_len / avgdl)
            raw_score += term_idf * (tf * (k1 + 1.0)) / denom

    # Exact / substring POI name boost (e.g. "Кафе Коктем", "Байтерек", "Mega Silk Way")
    poi_name_low = poi.name.lower()
    name_words = [w for w in re.findall(r"[a-zа-яё0-9]+", poi_name_low) if len(w) >= 4]
    name_hit = any(nw in q_clean for nw in name_words)
    if name_hit:
        raw_score += 6.0

    # Category match bonus when query explicitly asks for cafe, library, museum, stadium, etc.
    cat_words = CATEGORY_WORDS.get(poi.category, "").lower().split()
    if any(cw in q_clean for cw in cat_words if len(cw) >= 4):
        raw_score += 2.5

    if raw_score <= 0.0 and matched_unique == 0:
        return 0.0

    # Smooth saturation into [0.0, 0.92]
    norm = 1.0 - math.exp(-raw_score / 5.5)
    return min(0.95, max(0.0, norm))


def retrieve(
    query: str,
    lat: float,
    lon: float,
    *,
    max_distance_m: int = 2000,
    categories: list[str] | None = None,
    now_iso: str | None = None,
    top_k: int = 8,
) -> list[Candidate]:
    """Retrieve and rank POIs from data/poi.json using BM25 + semantic tags + geo + hours."""
    pois = load_pois()
    q = query.strip() or "интересное место для прогулки"
    hinted = query_tag_hints(q)
    q_low = q.lower()

    # Check if query is pure gibberish (no known tokens and no tag hints)
    tf_map, idf, _ = _bm25_index()
    q_toks = _expand_query_tokens(q)
    has_known_term = any(t in idf for t in q_toks) or bool(hinted)
    wants_cafe = any(w in q_low for w in ("кофе", "кофейн", "кафе", "coffee", "cafe"))

    scored: list[tuple[float, Candidate]] = []
    for poi in pois.values():
        if categories and poi.category not in categories:
            continue
        dist = haversine_m(lat, lon, poi.lat, poi.lon)
        if dist > max_distance_m:
            continue

        matched = len(hinted & set(poi.tags))
        lex_sim = lexical_similarity(q, poi) if has_known_term else 0.12

        if wants_cafe:
            if poi.category == "cafe":
                lex_sim = min(0.95, lex_sim + 0.18)
            else:
                lex_sim = max(0.0, lex_sim - 0.35)

        rel = min(1.0, lex_sim + TAG_BOOST * min(matched, 2))
        if rel < MIN_RELEVANCE:
            continue

        status, reason = open_status(poi.opening_hours, now_iso)
        proximity = 1.0 - dist / max_distance_m
        score = W_RELEVANCE * rel + W_PROXIMITY * proximity + HOURS_ADJUST[status]
        if matched and (not wants_cafe or poi.category == "cafe"):
            score += HINT_BONUS

        scored.append(
            (
                score,
                Candidate(
                    poi=poi,
                    distance_m=round(dist),
                    walk_min=walk_minutes(dist),
                    relevance=round(rel, 3),
                    open_status=status,
                    open_reason=reason,
                ),
            )
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]

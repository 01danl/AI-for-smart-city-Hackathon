"""Streamlit demo UI for Astana Walk AI (Build with AI · Smart City Track).

Launch with:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, time
from html import escape
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.generation import (  # noqa: E402
    OFFICIAL_PRESETS,
    RecommendRequest,
    RecommendResponse,
    generate_recommendations,
)
from backend.generation.prompts import build_rag_messages  # noqa: E402
from backend.retrieval import BACKEND, parse_query_intent, retrieve  # noqa: E402
from backend.retrieval.geo import round_coords  # noqa: E402
from backend.retrieval.index import load_pois  # noqa: E402

st.set_page_config(
    page_title="Astana Walk · прогулки по Астане",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="auto",  # open on desktop, collapsed on phones
)

# ── Design tokens ────────────────────────────────────────────────────────────

INK = "#1B1F24"
MUTED = "#5C636E"
SUBTLE = "#8A919B"
LINE = "#E3DFD6"
SURFACE = "#FFFFFF"
ACCENT = "#1F5F7A"

# (label, text color, background)
STATUS_STYLE: dict[str, tuple[str, str, str]] = {
    "confirmed_open": ("Открыто по данным OSM", "#1E6B45", "#E6F2EA"),
    "likely_open": ("Скоро закроется", "#8A5A12", "#F7EEDC"),
    "unknown": ("Часы работы неизвестны", "#4F5661", "#ECEBE7"),
    "likely_closed": ("Сейчас закрыто по данным OSM", "#9B2C2C", "#F7E4E2"),
}

SOURCE_STYLE: dict[str, tuple[str, str]] = {
    "poi_data": ("Факты из базы мест", "poi_data"),
    "generated_idea": ("Факты из базы + идея ИИ", "generated_idea"),
}

# Map dot colors per open status, matching STATUS_STYLE text colors.
STATUS_RGB: dict[str, list[int]] = {
    "confirmed_open": [30, 107, 69, 230],
    "likely_open": [138, 90, 18, 230],
    "unknown": [31, 95, 122, 230],
    "likely_closed": [155, 44, 44, 230],
}

CATEGORY_RU: dict[str, str] = {
    "cafe": "Кафе", "restaurant": "Ресторан", "fast_food": "Фастфуд", "bar": "Бар",
    "mall": "Торговый центр", "park": "Парк", "sport": "Спорт", "monument": "Памятник",
    "attraction": "Достопримечательность", "place_of_worship": "Мечеть / храм",
    "artwork": "Арт-объект", "cinema": "Кинотеатр", "theatre": "Театр", "library": "Библиотека",
    "museum": "Музей", "bookstore": "Книжный", "arts_centre": "Культурный центр",
    "stadium": "Стадион", "gallery": "Галерея", "theme_park": "Парк развлечений",
    "exhibition": "Выставочный центр", "viewpoint": "Смотровая площадка", "zoo": "Зоопарк",
}

WEATHER_LABELS = {
    "normal": "Комфортная погода",
    "cold_windy": "Мороз или сильный ветер",
    "rain_snow": "Дождь или снег",
}

LANG_LABELS = {"ru": "Русский", "kk": "Қазақша", "en": "English"}

CUSTOM_POINT = "Свои координаты"

LANDMARK_POINTS: dict[str, tuple[float, float, str]] = {
    "Байтерек": (51.1283, 71.4305, "Центр бульвара Нуржол"),
    "Mega Silk Way": (51.0891, 71.4073, "Район EXPO / Назарбаев Университет"),
    "EXPO 2017 (Нур Алем)": (51.0928, 71.4215, "Выставочный кластер"),
    "ТРЦ Хан Шатыр": (51.1325, 71.4039, "Западный конец оси Нуржол"),
    "Астана Опера": (51.1357, 71.4106, "Культурный кластер Левого берега"),
    "Ботанический сад": (51.1062, 71.4166, "Парковая зона"),
    "Старый центр": (51.1694, 71.4491, "Правый берег, историческая часть"),
    "За городом (тест пустой выдачи)": (51.3000, 71.0000, "Точка в степи без мест рядом"),
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stMarkdown, .stTextInput input, button, label {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
code, .mono {{ font-family: 'JetBrains Mono', ui-monospace, monospace; }}

.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}

h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.015em; }}

.aw-brand {{ display: flex; align-items: center; gap: 10px; margin-bottom: 0.2rem; }}
.aw-logo {{
    width: 30px; height: 30px; border-radius: 8px; background: {ACCENT};
    display: grid; place-items: center; color: #fff; font-weight: 700; font-size: 15px;
}}
.aw-brand-name {{ font-weight: 650; font-size: 1.02rem; color: {INK}; }}
.aw-brand-note {{ color: {SUBTLE}; font-size: 0.85rem; margin-left: 4px; }}
.aw-title {{ font-size: 2.05rem; line-height: 1.15; font-weight: 700; color: {INK};
             margin: 1.1rem 0 0.35rem 0; letter-spacing: -0.025em; }}
.aw-lede {{ color: {MUTED}; font-size: 1.02rem; max-width: 720px; margin-bottom: 1.3rem; }}

.aw-section-label {{
    font-size: 0.74rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;
    color: {SUBTLE}; margin: 1.4rem 0 0.55rem 0;
}}

/* Search form */
[data-testid="stForm"] {{ border: 1px solid {LINE}; background: {SURFACE}; border-radius: 14px;
                          padding: 0.9rem 1rem 0.4rem 1rem; }}
[data-testid="stForm"] input {{ font-size: 1.05rem; }}

/* Preset pills */
div[data-testid="stHorizontalBlock"] .stButton button[kind="secondary"] {{
    border-radius: 999px; border: 1px solid {LINE}; background: {SURFACE}; color: {INK};
    font-size: 0.86rem; padding: 0.35rem 0.9rem; font-weight: 500; min-height: 0;
}}
div[data-testid="stHorizontalBlock"] .stButton button[kind="secondary"]:hover {{
    border-color: {ACCENT}; color: {ACCENT};
}}

/* Context strip */
.aw-context {{ display: flex; flex-wrap: wrap; gap: 6px 18px; align-items: center;
               padding: 0.7rem 0.95rem; border: 1px solid {LINE}; border-radius: 12px;
               background: {SURFACE}; font-size: 0.86rem; color: {MUTED}; margin: 0.4rem 0 1.1rem 0; }}
.aw-context b {{ color: {INK}; font-weight: 600; }}
.aw-context .lead {{ color: {INK}; font-weight: 600; }}
.aw-context > span + span::before {{ content: "·"; color: {LINE}; margin-right: 18px; margin-left: -12px; }}
.aw-context > span:nth-child(2)::before {{ content: none; }}

/* Recommendation cards */
.aw-card {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 14px;
            padding: 1.15rem 1.25rem 1.05rem 1.25rem; margin-bottom: 0.85rem; }}
.aw-card-top {{ display: flex; gap: 12px; align-items: flex-start; }}
.aw-num {{ flex: none; width: 26px; height: 26px; border-radius: 50%; border: 1px solid {LINE};
           display: grid; place-items: center; font-size: 0.8rem; font-weight: 600; color: {MUTED}; }}
.aw-name {{ font-size: 1.12rem; font-weight: 650; color: {INK}; line-height: 1.3; }}
.aw-cat {{ font-size: 0.82rem; color: {SUBTLE}; margin-top: 1px; }}
.aw-meta {{ display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 0.86rem; color: {MUTED};
            margin: 0.6rem 0 0.65rem 38px; }}
.aw-meta b {{ color: {INK}; font-weight: 600; }}
.aw-badges {{ display: flex; flex-wrap: wrap; gap: 6px; margin-left: 38px; }}
.aw-badge {{ font-size: 0.76rem; font-weight: 600; padding: 3px 9px; border-radius: 6px; }}
.aw-badge-src {{ background: transparent; border: 1px solid {LINE}; color: {MUTED}; font-weight: 500; }}
.aw-badge-src .mono {{ color: {SUBTLE}; font-size: 0.72rem; margin-left: 4px; }}
.aw-body {{ margin: 0.8rem 0 0 38px; font-size: 0.93rem; color: {INK}; line-height: 1.55; }}
.aw-body p {{ margin: 0 0 0.55rem 0; }}
.aw-k {{ display: block; font-size: 0.74rem; font-weight: 600; text-transform: uppercase;
         letter-spacing: 0.06em; color: {SUBTLE}; margin-bottom: 2px; }}
.aw-note {{ font-size: 0.84rem; color: {MUTED}; border-top: 1px dashed {LINE};
            padding-top: 0.6rem; margin-top: 0.3rem; }}
.aw-card details {{ margin: 0.5rem 0 0 38px; font-size: 0.83rem; color: {MUTED}; }}
.aw-card summary {{ cursor: pointer; color: {ACCENT}; font-weight: 500; }}
.aw-card details div {{ margin-top: 0.4rem; line-height: 1.5; }}

/* Route */
.aw-route-head {{ display: flex; flex-wrap: wrap; gap: 6px 22px; font-size: 0.9rem; color: {MUTED};
                  margin: 0.2rem 0 1rem 0; }}
.aw-route-head b {{ color: {INK}; }}
.aw-stop {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 12px; padding: 0.95rem 1rem;
            height: 100%; }}
.aw-stop-stage {{ font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em;
                  color: {ACCENT}; }}
.aw-stop-name {{ font-weight: 650; color: {INK}; margin: 0.25rem 0 0.55rem 0; font-size: 1rem; }}
.aw-stop-row {{ display: flex; justify-content: space-between; font-size: 0.84rem; color: {MUTED};
                padding: 3px 0; border-bottom: 1px solid #F0EDE7; }}
.aw-stop-row:last-child {{ border-bottom: none; }}
.aw-stop-row b {{ color: {INK}; font-weight: 600; }}

.aw-legend {{ display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 0.8rem; color: {MUTED}; margin-top: 0.4rem; }}
.aw-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }}

.aw-empty {{ border: 1px dashed {LINE}; border-radius: 14px; padding: 2rem 1.5rem; text-align: center;
             color: {MUTED}; background: {SURFACE}; }}
.aw-empty b {{ color: {INK}; display: block; font-size: 1.05rem; margin-bottom: 0.3rem; }}
.aw-disclaimer {{ font-size: 0.8rem; color: {SUBTLE}; margin-top: 1.4rem; line-height: 1.5; }}

@media (max-width: 640px) {{
    .aw-title {{ font-size: 1.6rem; }}
    .aw-meta, .aw-badges, .aw-body, .aw-card details {{ margin-left: 0; }}
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 22px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ padding: 0.5rem 0; font-weight: 500; color: {MUTED}; }}
.stTabs [aria-selected="true"] {{ color: {INK}; }}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ── State ────────────────────────────────────────────────────────────────────


def _split_iso(now_iso: str | None) -> tuple[date, time]:
    try:
        dt = datetime.fromisoformat(now_iso) if now_iso else datetime(2026, 10, 16, 15, 0)
    except ValueError:
        dt = datetime(2026, 10, 16, 15, 0)
    return dt.date(), dt.time().replace(second=0, microsecond=0)


def apply_preset(preset: dict) -> None:
    """Load a jury scenario into the widgets (runs as a button callback, before widgets render)."""
    ss = st.session_state
    ss.query = preset["query"]
    ss.lat = float(preset["lat"])
    ss.lon = float(preset["lon"])
    ss.max_distance_m = int(preset["max_distance_m"])
    ss.visit_date, ss.visit_time = _split_iso(preset["now_iso"])
    ss.accessible = bool(preset["accessible"])
    ss.group_size = int(preset["group_size"])
    ss.weather = preset["weather"]
    ss.landmark = CUSTOM_POINT
    for name, (l_lat, l_lon, _) in LANDMARK_POINTS.items():
        if abs(l_lat - ss.lat) < 1e-4 and abs(l_lon - ss.lon) < 1e-4:
            ss.landmark = name
            break


def apply_landmark() -> None:
    name = st.session_state.landmark
    if name in LANDMARK_POINTS:
        st.session_state.lat, st.session_state.lon, _ = LANDMARK_POINTS[name]


if "query" not in st.session_state:
    st.session_state.lang = "ru"
    st.session_state.include_route = True
    st.session_state.auto_anchor = True
    apply_preset(OFFICIAL_PRESETS[0])


@st.cache_data(show_spinner=False, max_entries=64)
def run_recommendations(req_json: str) -> RecommendResponse:
    """Same request -> same answer: avoids repeated LLM calls on reruns and repeated presets."""
    return generate_recommendations(RecommendRequest.model_validate_json(req_json))


# ── Sidebar: context ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Где и когда")

    st.selectbox(
        "Точка старта",
        options=list(LANDMARK_POINTS.keys()) + [CUSTOM_POINT],
        key="landmark",
        on_change=apply_landmark,
        help="Можно выбрать ориентир вместо GPS: так не нужно раскрывать, где вы находитесь.",
    )
    if st.session_state.landmark in LANDMARK_POINTS:
        st.caption(LANDMARK_POINTS[st.session_state.landmark][2])

    with st.expander("Точные координаты", expanded=st.session_state.landmark == CUSTOM_POINT):
        c1, c2 = st.columns(2)
        c1.number_input("Широта", key="lat", format="%.4f", step=0.002)
        c2.number_input("Долгота", key="lon", format="%.4f", step=0.002)
        r_lat, r_lon = round_coords(st.session_state.lat, st.session_state.lon)
        st.caption(f"В логи и API попадает только округление до ~100 м: `{r_lat}, {r_lon}`")

    st.slider("Радиус пешком, м", min_value=300, max_value=6000, step=100, key="max_distance_m")

    d1, d2 = st.columns([1.15, 1])
    d1.date_input("Дата", key="visit_date", format="DD.MM.YYYY")
    d2.time_input("Время", key="visit_time", step=900)
    st.caption("Время Астаны (UTC+5). Влияет на статус «открыто / закрыто».")

    st.markdown("### Предпочтения")
    st.selectbox("Погода", options=list(WEATHER_LABELS), format_func=WEATHER_LABELS.get, key="weather")
    g1, g2 = st.columns(2)
    g1.number_input("Сколько человек", min_value=1, max_value=15, key="group_size")
    g2.selectbox("Язык ответа", options=list(LANG_LABELS), format_func=LANG_LABELS.get, key="lang")
    st.toggle("Без ступенек: коляска, дети", key="accessible")
    st.toggle("Собрать маршрут из 2–3 мест", key="include_route")
    st.toggle("Понимать ориентиры из текста запроса", key="auto_anchor",
              help="Например, «я возле Хан Шатыра» сдвинет точку старта к ТРЦ.")

    st.divider()
    st.caption(f"База: {len(load_pois())} мест из OpenStreetMap · поиск `{BACKEND}`")


# ── Header and search ────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="aw-brand">
        <div class="aw-logo">A</div>
        <span class="aw-brand-name">Astana Walk</span>
        <span class="aw-brand-note">· Build with AI × НИЦ Smart City</span>
    </div>
    <div class="aw-title">Куда сходить в Астане прямо сейчас</div>
    <div class="aw-lede">
        Опишите, что хочется сделать. Мы подберём 1–3 реальных места рядом, покажем,
        сколько идти, открыто ли сейчас и чем там заняться.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("search", border=False):
    f1, f2 = st.columns([5, 1], vertical_alignment="bottom")
    f1.text_input(
        "Запрос",
        key="query",
        placeholder="Например: хочу тихое кафе с розеткой, чтобы поработать час",
        label_visibility="collapsed",
    )
    f2.form_submit_button("Найти", type="primary", use_container_width=True)

st.markdown('<div class="aw-section-label">Сценарии для демо</div>', unsafe_allow_html=True)
preset_cols = st.columns(3)
for idx, preset in enumerate(OFFICIAL_PRESETS):
    title = preset["title"].split(". ", 1)[-1]
    preset_cols[idx % 3].button(
        title, key=f"btn_{preset['id']}", on_click=apply_preset, args=(preset,), use_container_width=True
    )

# ── Run ──────────────────────────────────────────────────────────────────────

ss = st.session_state
now_iso = datetime.combine(ss.visit_date, ss.visit_time).strftime("%Y-%m-%dT%H:%M")
req = RecommendRequest(
    query=ss.query,
    lat=ss.lat,
    lon=ss.lon,
    max_distance_m=ss.max_distance_m,
    lang=ss.lang,
    group_size=ss.group_size,
    accessible=ss.accessible,
    weather=ss.weather,
    include_route=ss.include_route,
    auto_anchor=ss.auto_anchor,
    now_iso=now_iso,
)

with st.spinner("Подбираем места…"):
    resp = run_recommendations(req.model_dump_json())

u = resp.used_data

# ── Context strip: what data was used ────────────────────────────────────────

parts = [
    "<span class='lead'>Учтено</span>",
    f"локация <b>~100 м</b> <span class='mono'>({u.rounded_coords[0]}, {u.rounded_coords[1]})</span>",
]
if u.anchor_resolved:
    parts.append(f"ориентир <b>{escape(u.anchor_resolved.split('(')[0].strip())}</b>")
if u.tags:
    parts.append("теги " + ", ".join(f"<b>{escape(t)}</b>" for t in u.tags[:4]))
if u.filters:
    parts.append("фильтры " + ", ".join(f"<b>{escape(f)}</b>" for f in u.filters[:4]))
parts.append(f"время <b>{u.time_budget_min} мин</b>" if u.time_budget_min else "время <b>авто</b>")
parts.append(f"погода <b>{escape(WEATHER_LABELS.get(u.weather_context, u.weather_context))}</b>")
parts.append(f"проверено мест <b>{u.candidates_evaluated}</b>")
parts.append(
    "ответ <b>нейросеть</b>" if "llm" in u.generation_mode.lower() else "ответ <b>по шаблону</b>"
)
st.markdown(
    '<div class="aw-context">' + "".join(f"<span>{p}</span>" for p in parts) + "</div>",
    unsafe_allow_html=True,
)


# ── Results ──────────────────────────────────────────────────────────────────


def render_card(idx: int, rec) -> None:
    st_label, st_fg, st_bg = STATUS_STYLE.get(rec.open_status, STATUS_STYLE["unknown"])
    src_label, src_code = SOURCE_STYLE.get(rec.source, SOURCE_STYLE["poi_data"])
    category = CATEGORY_RU.get(rec.category, rec.category)
    facts = "<br/>".join(escape(f) for f in rec.facts_poi_data) or "—"
    st.markdown(
        f"""
        <div class="aw-card">
            <div class="aw-card-top">
                <div class="aw-num">{idx}</div>
                <div>
                    <div class="aw-name">{escape(rec.name)}</div>
                    <div class="aw-cat">{escape(category)}</div>
                </div>
            </div>
            <div class="aw-meta">
                <span><b>{rec.distance_m} м</b> от вас</span>
                <span><b>~{rec.walk_min} мин</b> пешком</span>
                <span><b>~{rec.visit_min} мин</b> на месте</span>
            </div>
            <div class="aw-badges">
                <span class="aw-badge" style="color:{st_fg}; background:{st_bg};">{st_label}</span>
                <span class="aw-badge aw-badge-src">{src_label}<span class="mono">{src_code}</span></span>
            </div>
            <div class="aw-body">
                <p><span class="aw-k">Почему</span>{escape(rec.why)}</p>
                <p><span class="aw-k">Что делать</span>{escape(rec.what_to_do)}</p>
                <div class="aw-note">{escape(rec.confidence_note)}</div>
            </div>
            <details>
                <summary>Факты из базы и что добавил ИИ</summary>
                <div><b>Факты [poi_data]:</b><br/>{facts}</div>
                <div><b>Граница фактов и генерации:</b> {escape(rec.generated_plan_note or "—")}</div>
            </details>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_map() -> None:
    center_lat, center_lon = u.rounded_coords
    poi_points = [
        {
            "name": f"{i}. {r.name} · {r.distance_m} м",
            "lat": r.lat,
            "lon": r.lon,
            "color": STATUS_RGB.get(r.open_status, STATUS_RGB["unknown"]),
        }
        for i, r in enumerate(resp.recommendations, start=1)
    ]
    layers = [
        # ~100 m privacy zone around the rounded user location
        pdk.Layer(
            "ScatterplotLayer",
            data=[{"name": "Ваша зона (~100 м, точные координаты не сохраняются)",
                   "lat": center_lat, "lon": center_lon}],
            get_position="[lon, lat]",
            get_fill_color=[31, 95, 122, 40],
            get_line_color=[31, 95, 122, 200],
            get_radius=110,
            stroked=True,
            line_width_min_pixels=1.5,
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=poi_points,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_line_color=[255, 255, 255, 255],
            get_radius=40,
            radius_min_pixels=6,
            stroked=True,
            line_width_min_pixels=2,
            pickable=True,
        ),
    ]
    if resp.route and len(resp.route.stops) >= 2:
        path = [[center_lon, center_lat]] + [[s.lon, s.lat] for s in resp.route.stops]
        layers.insert(1, pdk.Layer(
            "PathLayer",
            data=[{"name": resp.route.title, "path": path}],
            get_path="path",
            get_color=[27, 31, 36, 150],
            width_min_pixels=2,
            pickable=True,
        ))
    st.pydeck_chart(
        pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=13.4, pitch=0),
            layers=layers,
            tooltip={"text": "{name}"},
        ),
        height=460,
    )
    legend = "".join(
        f'<span><span class="aw-dot" style="background:rgb({c[0]},{c[1]},{c[2]})"></span>{STATUS_STYLE[s][0]}</span>'
        for s, c in STATUS_RGB.items()
    )
    st.markdown(
        f'<div class="aw-legend">{legend}<span>Круг — ваша зона ~100 м · линия — маршрут</span></div>',
        unsafe_allow_html=True,
    )


if not resp.recommendations:
    st.markdown(
        f'<div class="aw-empty"><b>Рядом ничего не нашлось</b>{escape(resp.disclaimer)}</div>',
        unsafe_allow_html=True,
    )
else:
    tab_cards, tab_route, tab_inspector = st.tabs(["Рекомендации", "Маршрут", "Как это работает"])

    with tab_cards:
        col_left, col_right = st.columns([1.2, 1], gap="large")
        with col_left:
            for idx, rec in enumerate(resp.recommendations, start=1):
                render_card(idx, rec)
        with col_right:
            render_map()

    with tab_route:
        if resp.route:
            rt = resp.route
            st.markdown(f"#### {escape(rt.title)}")
            st.markdown(
                f"""<div class="aw-route-head">
                    <span>Всего <b>~{rt.total_duration_min} мин</b></span>
                    <span>Пешком <b>{rt.total_distance_m} м · ~{rt.total_walk_min} мин</b></span>
                    <span>На местах <b>~{rt.total_visit_min} мин</b></span>
                </div>""",
                unsafe_allow_html=True,
            )
            st.caption(rt.summary)
            for col, stop in zip(st.columns(len(rt.stops)), rt.stops):
                s_label, s_fg, s_bg = STATUS_STYLE.get(stop.open_status, STATUS_STYLE["unknown"])
                col.markdown(
                    f"""
                    <div class="aw-stop">
                        <div class="aw-stop-stage">Шаг {stop.step} · {escape(stop.stage_note)}</div>
                        <div class="aw-stop-name">{escape(stop.name)}</div>
                        <div class="aw-stop-row"><span>Переход</span><b>{stop.from_prev_m} м · ~{stop.from_prev_walk_min} мин</b></div>
                        <div class="aw-stop-row"><span>На месте</span><b>~{stop.visit_min} мин</b></div>
                        <div class="aw-stop-row"><span>От старта</span><b>+{stop.arrival_offset_min}…+{stop.departure_offset_min} мин</b></div>
                        <div style="margin-top:0.6rem;">
                            <span class="aw-badge" style="color:{s_fg}; background:{s_bg};"
                                  title="{escape(stop.open_reason)}">{s_label}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Для маршрута нужно минимум 2 подходящих места в радиусе. Увеличьте радиус или включите маршрут слева.")

    with tab_inspector:
        st.markdown(
            "Как устроен ответ: сначала поиск находит реальные места рядом, затем нейросеть "
            "пишет объяснение только по ним. Ниже видно, что было на каждом шаге и что ушло в модель."
        )
        intent_obj = parse_query_intent(
            req.query,
            explicit_lang=req.lang,
            now_iso=req.now_iso,
            explicit_accessible=req.accessible,
            explicit_group_size=req.group_size,
            explicit_weather=req.weather,
        )
        cands = retrieve(
            req.query,
            u.rounded_coords[0],
            u.rounded_coords[1],
            max_distance_m=req.max_distance_m,
            now_iso=req.now_iso,
            top_k=6,
        )
        col_i1, col_i2 = st.columns(2, gap="large")
        with col_i1:
            st.markdown("**1. Что система поняла из запроса** (`used_data`)")
            st.json(json.loads(resp.used_data.model_dump_json()), expanded=True)
            if cands:
                st.markdown("**2. Кандидаты из поиска до генерации**")
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Место": c.poi.name,
                            "Категория": CATEGORY_RU.get(c.poi.category, c.poi.category),
                            "Метры": c.distance_m,
                            "Мин пешком": c.walk_min,
                            "Совпадение": round(c.relevance, 2),
                            "Часы": STATUS_STYLE.get(c.open_status, STATUS_STYLE["unknown"])[0],
                            "id": c.poi.id,
                        }
                        for c in cands
                    ]),
                    use_container_width=True,
                    hide_index=True,
                )
        with col_i2:
            st.markdown("**3. Что ушло в нейросеть** (координат пользователя здесь нет)")
            msgs = build_rag_messages(req.query, cands, intent_obj, lang=req.lang)
            st.code(msgs[1]["content"], language="json", wrap_lines=True)

if resp.recommendations:
    st.markdown(f'<div class="aw-disclaimer">{escape(resp.disclaimer)}</div>', unsafe_allow_html=True)

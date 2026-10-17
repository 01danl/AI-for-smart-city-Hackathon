"""Streamlit demo UI for Astana Walk AI (Build with AI · Smart City Track).

Chat layout: icon rail on the left, conversation in the middle, scenarios and
settings on the right.

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
    initial_sidebar_state="collapsed",
)

# ── Design tokens ────────────────────────────────────────────────────────────

INK = "#1D2433"
MUTED = "#667085"
SUBTLE = "#98A2B3"
LINE = "#ECEEF3"
PANEL = "#FFFFFF"
PAGE = "#F6F7FB"
ACCENT = "#5146E5"
ACCENT_SOFT = "#EEEDFD"

# (label, text color, background)
STATUS_STYLE: dict[str, tuple[str, str, str]] = {
    "confirmed_open": ("Открыто по расписанию в базе", "#127A4B", "#E7F6EE"),
    "likely_open": ("Скоро закроется", "#946200", "#FDF3DC"),
    "unknown": ("Часы работы неизвестны", "#5D6679", "#EFF1F5"),
    "likely_closed": ("Закрыто по расписанию в базе", "#B42318", "#FDECEA"),
}

SOURCE_STYLE: dict[str, tuple[str, str]] = {
    "poi_data": ("Факты из базы мест", "poi_data"),
    "generated_idea": ("Факты из базы + идея ИИ", "generated_idea"),
}

# Map dot colors per open status, matching STATUS_STYLE text colors.
STATUS_RGB: dict[str, list[int]] = {
    "confirmed_open": [18, 122, 75, 235],
    "likely_open": [148, 98, 0, 235],
    "unknown": [81, 70, 229, 235],
    "likely_closed": [180, 35, 24, 235],
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

# Material icon per jury scenario, in OFFICIAL_PRESETS order.
PRESET_ICONS = [
    ":material/directions_walk:", ":material/local_cafe:", ":material/child_care:",
    ":material/flight_takeoff:", ":material/nightlife:", ":material/museum:",
]

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

MAX_HISTORY = 12

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..24,400,0,0');

html, body, [class*="css"], .stMarkdown, input, textarea, button, label {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
.mono {{ font-family: 'JetBrains Mono', ui-monospace, monospace; }}
.msi {{ font-family: 'Material Symbols Rounded'; font-size: 21px; line-height: 1; font-weight: 400;
        font-style: normal; display: inline-block; -webkit-font-smoothing: antialiased; }}

[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display: none; }}
.block-container {{ padding: 1rem 1.2rem 1rem; max-width: 1480px; }}

/* Panels */
.st-key-rail, .st-key-chat, .st-key-side {{ background: {PANEL}; border: 1px solid {LINE}; }}
.st-key-rail {{ border-radius: 18px 0 0 18px; border-right: none; min-height: calc(100dvh - 32px);
                padding: 18px 0 !important; align-items: center; }}
.st-key-chat {{ border-radius: 0; background: {PAGE}; padding: 0 0 12px 0 !important;
                min-height: calc(100dvh - 32px); }}
.st-key-side {{ border-radius: 0 18px 18px 0; border-left: none;
                min-height: calc(100dvh - 32px); padding: 18px 0 !important; }}

/* Rail */
.aw-rail {{ display: flex; flex-direction: column; align-items: center; gap: 14px; height: 100%; }}
.aw-logo {{ width: 38px; height: 38px; border-radius: 12px; display: grid; place-items: center;
            background: linear-gradient(140deg, #6C5CF6, #4338CA); color: #fff; font-weight: 700;
            font-size: 18px; margin-bottom: 22px; box-shadow: 0 6px 14px rgba(81,70,229,.28); }}
.aw-rail-item {{ width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center;
                 color: {MUTED}; }}
.aw-rail-item.active {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}
.aw-rail-spacer {{ flex: 1; min-height: 220px; }}

/* Chat header */
.aw-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px;
            padding: 16px 24px; border-bottom: 1px solid {LINE}; background: {PANEL}; }}
.aw-head-title {{ font-size: 1.45rem; font-weight: 700; color: {INK}; letter-spacing: -0.02em; }}
.aw-head-sub {{ font-size: 0.8rem; color: {SUBTLE}; margin-top: 2px; }}
.aw-pills {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
.aw-pill {{ display: inline-flex; align-items: center; gap: 6px; border: 1px solid {LINE};
            border-radius: 999px; padding: 6px 12px; font-size: 0.78rem; color: {INK}; background: {PANEL}; }}
.aw-pill .msi {{ font-size: 17px; color: {MUTED}; }}

/* Messages */
.st-key-messages {{ height: clamp(300px, calc(100dvh - 260px), 700px) !important;
                    overflow-y: auto; padding: 8px 24px 14px; scrollbar-gutter: stable; }}
.aw-row {{ display: flex; gap: 12px; align-items: flex-start; margin: 14px 0; }}
.aw-row.user {{ justify-content: flex-end; }}
.aw-bot-badge {{ flex: none; width: 34px; height: 34px; border-radius: 50%; background: {ACCENT_SOFT};
                 color: {ACCENT}; display: grid; place-items: center; font-weight: 700; font-size: 0.78rem; }}
.aw-user-badge {{ flex: none; width: 34px; height: 34px; border-radius: 50%; background: #1D2433;
                  color: #fff; display: grid; place-items: center; font-weight: 600; font-size: 0.74rem; }}
.aw-bubble {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 14px; padding: 11px 14px;
              font-size: 0.9rem; color: {INK}; line-height: 1.55; max-width: 640px; }}
.aw-bubble.user {{ background: {ACCENT}; border-color: {ACCENT}; color: #fff; border-radius: 14px;
                   box-shadow: 0 6px 16px rgba(81,70,229,.22); }}
.aw-bubble .aw-time {{ font-size: 0.72rem; color: {SUBTLE}; text-align: right; margin-top: 4px; }}
.aw-bubble.user .aw-time {{ color: rgba(255,255,255,.72); }}
.aw-user-meta {{ font-size: 0.74rem; color: rgba(255,255,255,.78); margin-top: 6px;
                 display: flex; gap: 10px; flex-wrap: wrap; }}

.aw-answer {{ max-width: 680px; }}
.aw-context {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 2px 0; }}
.aw-chip {{ font-size: 0.74rem; padding: 3px 9px; border-radius: 999px; background: {PANEL};
            border: 1px solid {LINE}; color: {MUTED}; }}
.aw-chip b {{ color: {INK}; font-weight: 600; }}

.aw-card {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 14px; padding: 13px 15px;
            margin-top: 10px; }}
.aw-card-top {{ display: flex; justify-content: space-between; gap: 10px; align-items: flex-start; }}
.aw-name {{ font-size: 1rem; font-weight: 700; color: {INK}; }}
.aw-cat {{ font-size: 0.78rem; color: {SUBTLE}; }}
.aw-num {{ flex: none; font-size: 0.72rem; font-weight: 700; color: {ACCENT}; background: {ACCENT_SOFT};
           border-radius: 8px; padding: 3px 8px; }}
.aw-meta {{ display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 0.82rem; color: {MUTED}; margin: 8px 0; }}
.aw-meta b {{ color: {INK}; }}
.aw-badges {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }}
.aw-badge {{ font-size: 0.72rem; font-weight: 600; padding: 3px 9px; border-radius: 999px; }}
.aw-badge-src {{ border: 1px solid {LINE}; color: {MUTED}; font-weight: 500; }}
.aw-badge-src .mono {{ color: {SUBTLE}; font-size: 0.68rem; margin-left: 4px; }}
.aw-k {{ font-weight: 600; color: {INK}; }}
.aw-card p {{ margin: 0 0 6px 0; font-size: 0.87rem; line-height: 1.55; color: #344054; }}
.aw-note {{ font-size: 0.78rem; color: {MUTED}; border-top: 1px dashed {LINE}; padding-top: 7px; margin-top: 4px; }}
.aw-card details {{ font-size: 0.78rem; color: {MUTED}; margin-top: 6px; }}
.aw-card summary {{ cursor: pointer; color: {ACCENT}; font-weight: 600; }}
.aw-card details div {{ margin-top: 5px; line-height: 1.5; }}
.aw-card a {{ color: {ACCENT}; }}
.aw-disclaimer {{ font-size: 0.74rem; color: {SUBTLE}; margin-top: 8px; line-height: 1.45; }}

/* Expanders inside chat (map, route, inspector) */
.st-key-messages [data-testid="stExpander"] {{ max-width: 680px; margin-left: 46px; background: {PANEL};
                                               border-radius: 12px; }}
.st-key-messages [data-testid="stExpander"] details {{ border-color: {LINE}; border-radius: 12px; }}

.aw-stop {{ background: {PAGE}; border: 1px solid {LINE}; border-radius: 12px; padding: 10px 12px; height: 100%; }}
.aw-stop-stage {{ font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: {ACCENT}; }}
.aw-stop-name {{ font-weight: 700; color: {INK}; margin: 3px 0 6px 0; font-size: 0.92rem; }}
.aw-stop-row {{ display: flex; justify-content: space-between; font-size: 0.78rem; color: {MUTED}; padding: 2px 0; }}
.aw-stop-row b {{ color: {INK}; }}
.aw-legend {{ display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 0.74rem; color: {MUTED}; margin-top: 6px; }}
.aw-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }}

/* Composer */
.st-key-composer {{ padding: 12px 24px 2px; border-top: 1px solid {LINE}; background: {PAGE}; }}
.st-key-composer [data-testid="stChatInput"] {{ width: 100%; border-radius: 14px;
    border: 1px solid #D9DDF0; background: {PANEL}; box-shadow: 0 3px 14px rgba(16,24,40,.06); }}
.st-key-composer [data-testid="stChatInput"]:focus-within {{ border-color: {ACCENT};
    box-shadow: 0 0 0 3px rgba(81,70,229,.12); }}
.st-key-composer [data-testid="stChatInput"] textarea {{ font-size: 0.93rem; min-height: 44px;
    line-height: 1.4; }}
[data-testid="stChatInputSubmitButton"] {{ background: {ACCENT} !important; color: #fff !important;
                                           border-radius: 10px !important; }}

/* Right panel */
.aw-privacy-strip {{ padding: 0 22px 14px 22px; border-bottom: 1px solid {LINE};
                     font-size: 0.75rem; color: {MUTED}; line-height: 1.45; }}
.aw-privacy-strip .mono {{ color: {SUBTLE}; }}
.st-key-side .st-key-side_view {{ padding: 0 22px 12px 22px; }}
.st-key-side .st-key-view_main, .st-key-side .st-key-view_settings {{
    padding: 0 22px; min-width: 0; box-sizing: border-box;
}}
.st-key-side .st-key-side_view button {{ font-weight: 600; font-size: 0.86rem; border-radius: 9px; }}
.st-key-rail .st-key-btn_rail_settings {{ display: flex; justify-content: center; }}
.st-key-rail .st-key-btn_rail_settings button {{ width: 40px; min-width: 40px; height: 40px; padding: 0;
    border: none; border-radius: 12px; background: transparent; color: {MUTED}; }}
.st-key-rail .st-key-btn_rail_settings button:hover {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}
.st-key-rail .st-key-btn_rail_settings button p {{ display: none; }}
.aw-side-title {{ font-weight: 700; color: {INK}; font-size: 0.98rem; margin: 14px 0 6px 0; }}
.aw-side-sep {{ border-top: 1px solid {LINE}; margin: 14px 0 0; }}

.st-key-presets .stButton button {{ justify-content: flex-start; border: none; background: transparent;
    color: {INK}; font-weight: 500; font-size: 0.9rem; padding: 10px 12px; border-radius: 0;
    border-left: 3px solid transparent; }}
.st-key-presets .stButton button:hover {{ background: {ACCENT_SOFT}; color: {INK}; }}
.st-key-presets .stButton button > div, .st-key-presets .stButton button [data-testid="stMarkdownContainer"] {{
    justify-content: flex-start; min-width: 0; text-align: left; }}
.st-key-presets .stButton button p {{ text-align: left; white-space: normal; overflow-wrap: anywhere; }}
.st-key-presets [data-testid="stVerticalBlock"] {{ gap: 2px; }}

.aw-recent {{ display: flex; justify-content: space-between; gap: 10px; padding: 9px 0; font-size: 0.86rem;
              color: {INK}; border-bottom: 1px solid #F4F5F8; }}
.aw-recent span:first-child {{ min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.aw-recent span:last-child {{ color: {SUBTLE}; font-size: 0.76rem; flex: none; }}
.aw-recent.current span:first-child {{ color: {ACCENT}; font-weight: 600; }}
.aw-empty-side {{ font-size: 0.82rem; color: {SUBTLE}; padding: 6px 0; }}
.aw-foot {{ font-size: 0.74rem; color: {SUBTLE}; margin-top: 12px; }}

@media (max-width: 900px) {{
    .st-key-rail {{ display: none; }}
    .st-key-chat, .st-key-side {{ border-radius: 18px; border: 1px solid {LINE}; min-height: 0; }}
    .st-key-messages, .st-key-composer {{ padding-left: 12px; padding-right: 12px; }}
    .aw-head {{ padding: 14px; flex-direction: column; align-items: flex-start; }}
    .aw-pills {{ justify-content: flex-start; }}
    .st-key-messages {{ height: clamp(260px, calc(100dvh - 280px), 520px) !important; }}
    .st-key-messages [data-testid="stExpander"] {{ margin-left: 0; }}
}}
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


def _load_settings(preset: dict) -> None:
    ss = st.session_state
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


def _build_request(query: str) -> RecommendRequest:
    ss = st.session_state
    return RecommendRequest(
        query=query,
        lat=ss.lat,
        lon=ss.lon,
        max_distance_m=ss.max_distance_m,
        lang=ss.lang,
        group_size=ss.group_size,
        accessible=ss.accessible,
        weather=ss.weather,
        include_route=ss.include_route,
        auto_anchor=ss.auto_anchor,
        now_iso=datetime.combine(ss.visit_date, ss.visit_time).strftime("%Y-%m-%dT%H:%M"),
    )


def _ask(query: str, preset_id: str | None = None) -> None:
    ss = st.session_state
    ss.history.append({
        "query": query,
        "req_json": _build_request(query).model_dump_json(),
        "start": ss.landmark if ss.landmark != CUSTOM_POINT else "свои координаты",
    })
    ss.history = ss.history[-MAX_HISTORY:]
    ss.active_preset = preset_id


def run_preset(preset: dict) -> None:
    """Button callback: load a jury scenario's context and ask its question."""
    _load_settings(preset)
    _ask(preset["query"], preset["id"])


def open_settings() -> None:
    """Rail gear callback: switch the right panel to the settings section."""
    st.session_state.side_view = "Настройки"


def apply_landmark() -> None:
    name = st.session_state.landmark
    if name in LANDMARK_POINTS:
        st.session_state.lat, st.session_state.lon, _ = LANDMARK_POINTS[name]


if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.active_preset = None
    st.session_state.lang = "ru"
    st.session_state.include_route = True
    st.session_state.auto_anchor = True
    st.session_state.side_view = "Главное"
    _load_settings(OFFICIAL_PRESETS[0])


@st.cache_data(show_spinner=False, max_entries=64)
def run_recommendations(req_json: str) -> RecommendResponse:
    """Same request -> same answer: no repeated LLM calls on reruns or repeated scenarios."""
    return generate_recommendations(RecommendRequest.model_validate_json(req_json))


# ── Rendering helpers ────────────────────────────────────────────────────────


def _html(markup: str, target=st) -> None:
    """Render raw HTML. Lines are flattened: Markdown would treat indented lines after
    a blank line as a code block and show the tags as text."""
    target.markdown(" ".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def _hhmm(req: RecommendRequest) -> str:
    return (req.now_iso or "")[11:16]


def render_user(entry: dict, req: RecommendRequest) -> None:
    km = f"{req.max_distance_m / 1000:.1f}".rstrip("0").rstrip(".")
    extras = []
    if req.accessible:
        extras.append("без ступенек")
    if req.group_size > 1:
        extras.append(f"{req.group_size} чел.")
    if req.weather != "normal":
        extras.append(WEATHER_LABELS[req.weather].lower())
    meta = " · ".join([f"📍 {escape(entry['start'])}", f"радиус {km} км", *extras])
    _html(f"""
        <div class="aw-row user">
            <div class="aw-bubble user">
                {escape(entry['query'])}
                <div class="aw-user-meta">{meta}</div>
                <div class="aw-time">{_hhmm(req)} ✓✓</div>
            </div>
            <div class="aw-user-badge">Вы</div>
        </div>
        """)


def _context_chips(u) -> str:
    chips = [f"локация <b>~100 м</b> <span class='mono'>{u.rounded_coords[0]}, {u.rounded_coords[1]}</span>"]
    if u.anchor_resolved:
        chips.append(f"ориентир <b>{escape(u.anchor_resolved.split('(')[0].strip())}</b>")
    if u.tags:
        chips.append("теги " + ", ".join(f"<b>{escape(t)}</b>" for t in u.tags[:4]))
    if u.filters:
        chips.append("фильтры " + ", ".join(f"<b>{escape(f)}</b>" for f in u.filters[:4]))
    chips.append(f"время <b>{u.time_budget_min} мин</b>" if u.time_budget_min else "время <b>авто</b>")
    chips.append(f"проверено мест <b>{u.candidates_evaluated}</b>")
    chips.append("ответ <b>нейросеть</b>" if "llm" in u.generation_mode.lower() else "ответ <b>по шаблону</b>")
    return "".join(f'<span class="aw-chip">{c}</span>' for c in chips)


def _card_html(idx: int, rec) -> str:
    st_label, st_fg, st_bg = STATUS_STYLE.get(rec.open_status, STATUS_STYLE["unknown"])
    src_label, src_code = SOURCE_STYLE.get(rec.source, SOURCE_STYLE["poi_data"])
    category = CATEGORY_RU.get(rec.category, rec.category)
    facts = "<br/>".join(escape(f) for f in rec.facts_poi_data) or "—"
    official = ""
    poi = load_pois().get(rec.poi_id)
    if poi and poi.source.startswith("https://visitastana.kz/"):
        official = (
            f'<div class="aw-note"><a href="{escape(poi.source)}" target="_blank">Источник: Visit Astana</a>'
            f" · получено {escape(poi.collected_at)}"
            + (f"<br/>{escape(poi.description)}" if poi.description else "")
            + "</div>"
        )
    return f"""
        <div class="aw-card">
            <div class="aw-card-top">
                <div><div class="aw-name">{escape(rec.name)}</div><div class="aw-cat">{escape(category)}</div></div>
                <span class="aw-num">#{idx}</span>
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
            <p><span class="aw-k">Почему: </span>{escape(rec.why)}</p>
            <p><span class="aw-k">Что делать: </span>{escape(rec.what_to_do)}</p>
            <div class="aw-note">{escape(rec.confidence_note)}</div>
            {official}
            <details>
                <summary>Факты из базы и что добавил ИИ</summary>
                <div><b>Факты [poi_data]:</b><br/>{facts}</div>
                <div><b>Граница фактов и генерации:</b> {escape(rec.generated_plan_note or "—")}</div>
            </details>
        </div>
    """


def render_answer(req: RecommendRequest, resp: RecommendResponse, latest: bool) -> None:
    u = resp.used_data
    if resp.recommendations:
        n = len(resp.recommendations)
        word = "место" if n == 1 else "места"
        intro = f"Нашёл {n} {word} рядом. Вот что подойдёт:"
        cards = "".join(_card_html(i, r) for i, r in enumerate(resp.recommendations, start=1))
        footer = f'<div class="aw-disclaimer">{escape(resp.disclaimer)}</div>' if latest else ""
    else:
        intro = "<b>Рядом ничего не нашлось.</b> " + escape(resp.disclaimer)
        cards, footer = "", ""
    _html(f"""
        <div class="aw-row">
            <div class="aw-bot-badge">AW</div>
            <div class="aw-answer">
                <div class="aw-bubble">{intro}<div class="aw-time">{_hhmm(req)} ✓✓</div></div>
                <div class="aw-context">{_context_chips(u)}</div>
                {cards}
                {footer}
            </div>
        </div>
        """)
    if latest and resp.recommendations:
        with st.expander("Карта и маршрут", icon=":material/map:", expanded=True):
            render_map(resp)
            render_route(resp)
        with st.expander("Как это работает: поиск, кандидаты и промпт", icon=":material/manage_search:"):
            render_inspector(req, resp)


def render_map(resp: RecommendResponse) -> None:
    center_lat, center_lon = resp.used_data.rounded_coords
    poi_points = [
        {"name": f"{i}. {r.name} · {r.distance_m} м", "lat": r.lat, "lon": r.lon,
         "color": STATUS_RGB.get(r.open_status, STATUS_RGB["unknown"])}
        for i, r in enumerate(resp.recommendations, start=1)
    ]
    layers = [
        # ~100 m privacy zone around the rounded user location
        pdk.Layer(
            "ScatterplotLayer",
            data=[{"name": "Ваша зона (~100 м, точные координаты не сохраняются)",
                   "lat": center_lat, "lon": center_lon}],
            get_position="[lon, lat]",
            get_fill_color=[81, 70, 229, 38],
            get_line_color=[81, 70, 229, 200],
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
            get_color=[81, 70, 229, 170],
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
        height=340,
    )
    legend = "".join(
        f'<span><span class="aw-dot" style="background:rgb({c[0]},{c[1]},{c[2]})"></span>{STATUS_STYLE[s][0]}</span>'
        for s, c in STATUS_RGB.items()
    )
    st.markdown(
        f'<div class="aw-legend">{legend}<span>Круг — ваша зона ~100 м · линия — маршрут</span></div>',
        unsafe_allow_html=True,
    )


def render_route(resp: RecommendResponse) -> None:
    rt = resp.route
    if not rt:
        st.caption("Для маршрута нужно минимум 2 подходящих места в радиусе.")
        return
    st.markdown(f"**{escape(rt.title)}** · всего ~{rt.total_duration_min} мин, "
                f"пешком {rt.total_distance_m} м")
    for col, stop in zip(st.columns(len(rt.stops)), rt.stops):
        s_label, s_fg, s_bg = STATUS_STYLE.get(stop.open_status, STATUS_STYLE["unknown"])
        _html(f"""
            <div class="aw-stop">
                <div class="aw-stop-stage">Шаг {stop.step} · {escape(stop.stage_note)}</div>
                <div class="aw-stop-name">{escape(stop.name)}</div>
                <div class="aw-stop-row"><span>Переход</span><b>{stop.from_prev_m} м · ~{stop.from_prev_walk_min} мин</b></div>
                <div class="aw-stop-row"><span>На месте</span><b>~{stop.visit_min} мин</b></div>
                <div class="aw-stop-row"><span>От старта</span><b>+{stop.arrival_offset_min}…+{stop.departure_offset_min} мин</b></div>
                <div style="margin-top:6px;"><span class="aw-badge" style="color:{s_fg}; background:{s_bg};"
                     title="{escape(stop.open_reason)}">{s_label}</span></div>
            </div>
            """, target=col)


def render_inspector(req: RecommendRequest, resp: RecommendResponse) -> None:
    st.caption("Сначала поиск находит реальные места рядом, затем нейросеть пишет объяснение только по ним.")
    u = resp.used_data
    intent_obj = parse_query_intent(
        req.query,
        explicit_lang=req.lang,
        now_iso=req.now_iso,
        explicit_accessible=req.accessible,
        explicit_group_size=req.group_size,
        explicit_weather=req.weather,
    )
    cands = retrieve(
        req.query, u.rounded_coords[0], u.rounded_coords[1],
        max_distance_m=req.max_distance_m, now_iso=req.now_iso, top_k=6,
    )
    st.markdown("**1. Что система поняла из запроса** (`used_data`)")
    st.json(json.loads(u.model_dump_json()), expanded=True)
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
    st.markdown("**3. Что ушло в нейросеть** (координат пользователя здесь нет)")
    msgs = build_rag_messages(req.query, cands, intent_obj, lang=req.lang)
    st.code(msgs[1]["content"], language="json", wrap_lines=True)


# ── Layout: rail | chat | side panel ─────────────────────────────────────────

col_rail, col_chat, col_side = st.columns([0.055, 0.665, 0.28], gap=None)

with col_rail:
    with st.container(key="rail"):
        _html("""
            <div class="aw-rail">
                <div class="aw-logo">A</div>
                <div class="aw-rail-item active" title="Чат"><span class="msi">chat_bubble</span></div>
                <div class="aw-rail-item" title="Карта"><span class="msi">map</span></div>
                <div class="aw-rail-item" title="Маршруты"><span class="msi">route</span></div>
                <div class="aw-rail-item" title="Места"><span class="msi">location_on</span></div>
                <div class="aw-rail-item" title="Приватность"><span class="msi">shield_lock</span></div>
                <div class="aw-rail-spacer"></div>
            </div>
            """)
        st.button(
            "Настройки",
            key="btn_rail_settings",
            icon=":material/settings:",
            help="Настройки прогулки: точка старта, радиус, время, погода",
            on_click=open_settings,
        )
        if st.session_state.get("side_view") == "Настройки":
            _html(f"""<style>.st-key-btn_rail_settings button {{
                    background: {ACCENT_SOFT} !important; color: {ACCENT} !important; }}</style>""")

with col_chat:
    with st.container(key="chat"):
        last_mode = None
        if st.session_state.history:
            last_mode = run_recommendations(st.session_state.history[-1]["req_json"]).used_data.generation_mode
        mode_label = "нейросеть" if last_mode and "llm" in last_mode.lower() else "шаблоны"
        _html(f"""
            <div class="aw-head">
                <div>
                    <div class="aw-head-title">Astana Walk 🧭</div>
                    <div class="aw-head-sub">Куда сходить в Астане прямо сейчас</div>
                </div>
                <div class="aw-pills">
                    <span class="aw-pill"><span class="msi">location_on</span>{len(load_pois())} мест</span>
                    <span class="aw-pill"><span class="msi">shield_lock</span>точка ~100 м</span>
                    <span class="aw-pill"><span class="msi">auto_awesome</span>ответ: {mode_label}</span>
                </div>
            </div>
            """)

        messages = st.container(key="messages", height=520, border=False, autoscroll=True)
        with st.container(key="composer"):
            prompt = st.chat_input("Напишите, куда хочется сходить… например: тихое кафе с розеткой на час")
        if prompt and prompt.strip():
            _ask(prompt.strip())

        with messages:
            _html("""
                <div class="aw-row">
                    <div class="aw-bot-badge">AW</div>
                    <div class="aw-bubble">
                        Привет! Я Astana Walk — подскажу, куда сходить в Астане.
                        Напишите, что хочется, или выберите готовый сценарий. Я подберу 1–3 реальных места рядом,
                        покажу, сколько идти, открыто ли сейчас и чем там заняться.
                    </div>
                </div>
                """)
            history = st.session_state.history
            for i, entry in enumerate(history):
                req = RecommendRequest.model_validate_json(entry["req_json"])
                render_user(entry, req)
                latest = i == len(history) - 1
                with st.spinner("Подбираю места…"):
                    resp = run_recommendations(entry["req_json"])
                render_answer(req, resp, latest=latest)

with col_side:
    with st.container(key="side"):
        r_lat, r_lon = round_coords(st.session_state.lat, st.session_state.lon)
        _html(f"""
            <div class="aw-privacy-strip">
                Точка поиска: <span class="mono">{r_lat}, {r_lon}</span> — округлена до ~100 м.
                Точные координаты не хранятся и не отправляются в модель.
            </div>
            """)
        view = st.segmented_control(
            "Раздел панели",
            ["Главное", "Настройки"],
            key="side_view",
            label_visibility="collapsed",
        )
        view = view or "Главное"

        with st.container(key="view_main"):
            st.markdown('<div class="aw-side-title">Сценарии</div>', unsafe_allow_html=True)
            with st.container(key="presets"):
                for idx, preset in enumerate(OFFICIAL_PRESETS):
                    st.button(
                        preset["title"].split(". ", 1)[-1],
                        key=f"btn_{preset['id']}",
                        icon=PRESET_ICONS[idx % len(PRESET_ICONS)],
                        on_click=run_preset,
                        args=(preset,),
                        use_container_width=True,
                    )
            if st.session_state.active_preset:
                _html(f"""<style>.st-key-btn_{st.session_state.active_preset} button {{
                        background: {ACCENT_SOFT} !important; border-left-color: {ACCENT} !important;
                        font-weight: 600 !important; }}</style>""")
            st.markdown('<div class="aw-side-sep"></div><div class="aw-side-title">Недавние запросы</div>',
                        unsafe_allow_html=True)
            history = st.session_state.history
            if history:
                rows = []
                for i, entry in enumerate(reversed(history)):
                    req_time = json.loads(entry["req_json"]).get("now_iso") or ""
                    cls = "aw-recent current" if i == 0 else "aw-recent"
                    rows.append(f'<div class="{cls}"><span>{escape(entry["query"])}</span>'
                                f"<span>{req_time[11:16]}</span></div>")
                st.markdown("".join(rows[:8]), unsafe_allow_html=True)
            else:
                st.markdown('<div class="aw-empty-side">Пока пусто. Задайте вопрос или выберите сценарий.</div>',
                            unsafe_allow_html=True)

        with st.container(key="view_settings"):
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
                st.caption(f"В логи и API попадает только округление до ~100 м: `{r_lat}, {r_lon}`")
            st.slider("Радиус пешком, м", min_value=300, max_value=6000, step=100, key="max_distance_m")
            d1, d2 = st.columns([1.15, 1])
            d1.date_input("Дата", key="visit_date", format="DD.MM.YYYY")
            d2.time_input("Время", key="visit_time", step=900)
            st.caption("Время Астаны (UTC+5). Влияет на статус «открыто / закрыто».")
            st.selectbox("Погода", options=list(WEATHER_LABELS), format_func=WEATHER_LABELS.get, key="weather")
            g1, g2 = st.columns(2)
            g1.number_input("Сколько человек", min_value=1, max_value=15, key="group_size")
            g2.selectbox("Язык ответа", options=list(LANG_LABELS), format_func=LANG_LABELS.get, key="lang")
            st.toggle("Без ступенек: коляска, дети", key="accessible")
            st.toggle("Собрать маршрут из 2–3 мест", key="include_route")
            st.toggle("Понимать ориентиры из текста запроса", key="auto_anchor",
                      help="Например, «я возле Хан Шатыра» сдвинет точку старта к ТРЦ.")
            st.markdown(
                f'<div class="aw-foot">Настройки применяются к следующему вопросу. '
                f"База: {len(load_pois())} мест · OpenStreetMap + Visit Astana · поиск "
                f'<span class="mono">{BACKEND}</span></div>',
                unsafe_allow_html=True,
            )

        # Both sections stay rendered (so widget state survives switching); only the
        # inactive one is hidden. The gear button in the rail switches this same state.
        hidden = "view_settings" if view != "Настройки" else "view_main"
        _html(f"<style>.st-key-{hidden} {{ display: none; }}</style>")

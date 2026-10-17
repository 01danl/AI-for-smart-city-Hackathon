"""Streamlit demo UI for Astana Walk AI (Build with AI · Smart City Track).

ChatGPT-style layout: left sidebar with "Новый чат", scenario shortcuts and a
revisitable conversation history; the chat in the middle; the right settings
drawer is open by default and the gear in the chat header toggles it.

Launch with:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import base64
import json
import sys
from datetime import date, datetime, time
from html import escape
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

try:  # optional runtime helpers: browser geolocation + Web Speech API speech-to-text
    from streamlit_js_eval import get_geolocation as _js_get_geolocation
    from streamlit_js_eval import streamlit_js_eval as _js_eval
except ImportError:  # without them the app still works: landmarks, manual coords, typing
    _js_get_geolocation = None
    _js_eval = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOGO_SVG = (ROOT / "design.svg").read_text(encoding="utf-8")
LOGO_URL = "data:image/svg+xml;base64," + base64.b64encode(
    LOGO_SVG.replace('width="100%"', 'width="430"').encode()
).decode()
# Reuse the supplied navigation mark as the tab icon without shrinking the wordmark.
LOGO_MARK = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="12 0 88 88">'
    + LOGO_SVG[LOGO_SVG.index("<g"):LOGO_SVG.index("</g>") + 4]
    + '</svg>'
)

from backend.generation import (  # noqa: E402
    OFFICIAL_PRESETS,
    RecommendRequest,
    RecommendResponse,
    generate_recommendations,
)
from backend.generation.prompts import build_rag_messages  # noqa: E402
from backend.generation.voice import has_server_asr as _server_asr_ready  # noqa: E402
from backend.generation.voice import transcribe_b64_wav as _server_transcribe  # noqa: E402
from backend.retrieval import BACKEND, parse_query_intent, retrieve  # noqa: E402
from backend.retrieval.geo import round_coords  # noqa: E402
from backend.retrieval.index import load_pois  # noqa: E402

st.set_page_config(
    page_title="Astana Walk · прогулки по Астане",
    page_icon=LOGO_MARK,
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
ACCENT = "#0284C7"
ACCENT_SOFT = "#E0F2FE"

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
    "unknown": [2, 132, 199, 235],
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
[data-testid="stLayoutWrapper"]:has(> .st-key-side),
[data-testid="stLayoutWrapper"]:has(> .st-key-settings),
[data-testid="stLayoutWrapper"]:has(> .st-key-chat) {{
    height: calc(100dvh - 32px) !important; min-height: 0;
}}
.st-key-side, .st-key-settings, .st-key-chat {{ background: {PANEL}; border: 1px solid {LINE};
    border-radius: 18px; height: 100% !important; min-height: 0; box-sizing: border-box; }}
.st-key-side, .st-key-settings {{ padding: 16px !important;
                                  overflow-y: auto; }}
.st-key-chat {{ position: relative; background: {PAGE}; padding: 0 0 12px !important; overflow: hidden; }}
.st-key-app_layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {{
    align-items: stretch;
}}
.st-key-app_layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    min-width: 0;
}}

/* Settings drawer (top-right gear -> panel on the right) */
.st-key-gearrow {{ position: absolute; top: 16px; right: 16px; width: 38px;
    display: flex; justify-content: flex-end; margin: 0; z-index: 2; }}
.st-key-gearrow .stButton {{ flex: none; }}
.st-key-btn_gear {{ display: flex; justify-content: flex-end; }}
.st-key-btn_gear button {{ width: 38px; min-width: 38px; height: 38px; padding: 0;
    border: none; background: transparent; color: {MUTED}; border-radius: 10px; }}
.st-key-btn_gear button:hover {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}
.st-key-btn_gear button p {{ display: none; }}
.aw-settings-head {{ display: flex; align-items: center; gap: 8px; padding: 4px 4px 10px;
                     border-bottom: 1px solid {LINE}; margin-bottom: 4px; }}
.aw-settings-title {{ font-weight: 700; color: {INK}; font-size: 0.98rem; }}
.aw-settings-sub {{ font-size: 0.72rem; color: {SUBTLE}; margin-top: 2px; }}
.st-key-btn_close_settings {{ display: flex; justify-content: flex-end; padding-top: 2px; }}
.st-key-btn_close_settings button {{ width: 32px; min-width: 32px; height: 32px; padding: 0;
    border: none; background: transparent; color: {MUTED}; border-radius: 8px; }}
.st-key-btn_close_settings button:hover {{ background: #FDECEA; color: #B42318; }}
.st-key-btn_close_settings button p {{ display: none; }}

/* Find-my-location button (first control in the drawer) */
.st-key-btn_geo {{ margin: 8px 0 2px; }}
.st-key-btn_geo button {{ width: 100%; justify-content: center; border: 1px solid {LINE};
    background: {PANEL}; border-radius: 10px; font-weight: 600; font-size: 0.85rem; color: {INK}; }}
.st-key-btn_geo button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
.st-key-btn_geo button[disabled] {{ opacity: .55; }}
/* the geolocation component itself is invisible plumbing */
[class*="st-key-geo_request_"] {{ display: none; }}

/* Voice input: mic pinned inside the composer next to the send button */
.st-key-composer_input {{ position: relative; }}
/* Native send: 16px inset + 32px button; leave an 8px gap before the mic. */
.st-key-btn_mic {{ position: absolute; right: 56px; bottom: 20px; z-index: 5; }}
.st-key-btn_mic button {{ width: 32px; min-width: 32px; height: 32px; min-height: 32px; padding: 0;
    border: none; background: {ACCENT}; color: #fff; border-radius: 10px !important; }}
.st-key-btn_mic button:hover {{ background: {ACCENT}; color: #fff; filter: brightness(.92); }}
.st-key-btn_mic button:focus-visible {{ outline: 2px solid {ACCENT}; outline-offset: 3px; }}
.st-key-btn_mic button[disabled] {{ opacity: .5; }}
.st-key-btn_mic button p {{ position: absolute; width: 1px; height: 1px; padding: 0;
    overflow: hidden; clip-path: inset(50%); white-space: nowrap; }}
[class*="st-key-mic_request_"], [class*="st-key-mic_capture_"] {{ display: none; }}
@keyframes aw-pulse {{ 0%, 100% {{ opacity: .25; }} 50% {{ opacity: 1; }} }}
.aw-mic-live {{ color: {MUTED}; font-size: 0.74rem; margin-top: 6px; }}
.aw-mic-live::before {{ content: ''; display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #D92D20; margin-right: 7px; animation: aw-pulse 1.1s infinite; }}

/* Sidebar (ChatGPT-style) */
.aw-brand {{ display: flex; align-items: center; min-height: 64px; padding: 0 0 12px; }}
.aw-brand img {{ display: block; width: 100%; max-width: 240px; height: auto; object-fit: contain; }}
.st-key-btn_newchat {{ margin-bottom: 10px; }}
.st-key-btn_newchat button {{ width: 100%; justify-content: flex-start; border: 1px solid {LINE};
    background: {PANEL}; border-radius: 10px; font-weight: 600; font-size: 0.88rem; color: {INK}; }}
.st-key-btn_newchat button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
.st-key-btn_newchat button p {{ text-align: left; }}

/* Chat header */
.aw-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px;
            padding: 16px 64px 16px 24px; border-bottom: 1px solid {LINE}; background: {PANEL};
            flex-wrap: wrap; }}
.aw-head-title {{ font-size: 1.45rem; font-weight: 700; color: {INK}; letter-spacing: -0.02em; }}
.aw-head-sub {{ font-size: 0.8rem; color: {SUBTLE}; margin-top: 2px; }}
.aw-pills {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
.aw-pill {{ display: inline-flex; align-items: center; gap: 6px; border: 1px solid {LINE};
            border-radius: 999px; padding: 6px 12px; font-size: 0.78rem; color: {INK}; background: {PANEL}; }}
.aw-pill .msi {{ font-size: 17px; color: {MUTED}; }}

/* Messages */
.st-key-chat > [data-testid="stLayoutWrapper"]:has(> .st-key-messages) {{ min-height: 0; }}
.st-key-messages {{ min-height: 0;
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
                   box-shadow: 0 6px 16px rgba(2,132,199,.22); }}
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
    border: 1px solid #CDE5F1; background: {PANEL}; box-shadow: 0 3px 14px rgba(16,24,40,.06); }}
.st-key-composer [data-testid="stChatInput"]:focus-within {{ border-color: {ACCENT};
    box-shadow: 0 0 0 3px rgba(2,132,199,.12); }}
.st-key-composer [data-testid="stChatInput"] textarea {{ font-size: 0.93rem; min-height: 44px;
    line-height: 1.4; padding-right: 44px; }}
[data-testid="stChatInputSubmitButton"] {{ background: {ACCENT} !important; color: #fff !important;
                                           border-radius: 10px !important; }}

/* Sidebar */
.aw-privacy-strip {{ padding: 2px 6px 12px 6px; border-top: 1px solid {LINE}; margin-top: 14px;
                     font-size: 0.75rem; color: {MUTED}; line-height: 1.45; }}
.aw-privacy-strip .mono {{ color: {SUBTLE}; }}
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

/* Chat-history list (ChatGPT-style) */
.st-key-chats .stButton button {{ justify-content: flex-start; border: none; background: transparent;
    color: {INK}; font-weight: 500; font-size: 0.87rem; padding: 9px 12px; border-radius: 10px; }}
.st-key-chats .stButton button:hover {{ background: {ACCENT_SOFT}; color: {INK}; }}
.st-key-chats .stButton button > div, .st-key-chats .stButton button [data-testid="stMarkdownContainer"] {{
    justify-content: flex-start; min-width: 0; text-align: left; }}
.st-key-chats .stButton button p {{ text-align: left; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; max-width: 100%; }}
.st-key-chats [data-testid="stVerticalBlock"] {{ gap: 2px; }}

.aw-empty-side {{ font-size: 0.82rem; color: {SUBTLE}; padding: 6px 0; }}
.aw-foot {{ font-size: 0.74rem; color: {SUBTLE}; margin-top: 12px; }}

@media (max-width: 900px) {{
    [data-testid="stLayoutWrapper"]:has(> .st-key-side),
    [data-testid="stLayoutWrapper"]:has(> .st-key-settings) {{ height: auto !important; }}
    .st-key-side, .st-key-settings {{ height: auto !important; }}
    .st-key-messages, .st-key-composer {{ padding-left: 12px; padding-right: 12px; }}
    .aw-head {{ padding: 14px 64px 14px 14px; flex-direction: column; align-items: flex-start; }}
    .aw-pills {{ justify-content: flex-start; }}
    .st-key-messages [data-testid="stExpander"] {{ margin-left: 0; }}
}}
</style>
"""

# Style-only HTML is kept outside the layout so it does not add an empty row.
st.html(CSS)


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


def _truncate(text: str, limit: int) -> str:
    """Compact one-line title for the chat list, like ChatGPT auto-titles."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _active_or_new_convo() -> dict:
    """Conversation in progress, or a fresh one created lazily on the first question."""
    ss = st.session_state
    for convo in ss.convos:
        if convo["id"] == ss.active_convo:
            return convo
    convo = {"id": ss.next_convo_id, "title": "Новый чат", "entries": []}
    ss.next_convo_id += 1
    ss.convos.append(convo)
    ss.active_convo = convo["id"]
    return convo


def _convo_label(convo: dict, limit: int = 24) -> str:
    label = _truncate(convo["title"], limit)
    time_str = ""
    if convo["entries"]:
        now_iso = json.loads(convo["entries"][-1]["req_json"]).get("now_iso") or ""
        time_str = now_iso[11:16]
    return f"{label} · {time_str}" if time_str else label


# Widget keys inside the settings drawer. Streamlit drops widget-keyed session
# state on the rerun right after the drawer is hidden, so a plain-dict backup
# (never garbage-collected) restores them when the drawer opens again.
_SETTING_KEYS = (
    "landmark", "lat", "lon", "max_distance_m", "visit_date", "visit_time",
    "weather", "group_size", "lang", "accessible", "include_route", "auto_anchor",
)


def _sync_settings_backup() -> None:
    """Persist the latest settings values into a non-widget session key."""
    ss = st.session_state
    try:
        backup = ss["_settings_backup"]
    except KeyError:
        backup = {}
        ss["_settings_backup"] = backup
    for key in _SETTING_KEYS:
        try:
            backup[key] = ss[key]
        except KeyError:
            pass


def _restore_settings_state() -> None:
    """Re-seed widget state from the backup before the drawer is rendered."""
    ss = st.session_state
    try:
        backup = ss["_settings_backup"]
    except KeyError:
        return
    for key in _SETTING_KEYS:
        try:
            _ = ss[key]  # Avoid rendering settings values through Streamlit magic.
        except KeyError:
            if key in backup:
                ss[key] = backup[key]


def _ss_get(key: str, default=None):
    """Read a plain session key without failing when it was never set."""
    try:
        return st.session_state[key]
    except KeyError:
        return default


def new_chat() -> None:
    """Sidebar callback: go to a blank chat; it joins the history on the first question."""
    st.session_state.active_convo = None
    _clear_voice_note()


def open_convo(convo_id: int) -> None:
    """Sidebar callback: reopen a past conversation."""
    st.session_state.active_convo = convo_id
    _clear_voice_note()


def toggle_settings() -> None:
    """Header gear callback: show or hide the right settings drawer."""
    st.session_state.show_settings = not st.session_state.show_settings


def close_settings() -> None:
    """Drawer callback: hide the right settings drawer."""
    st.session_state.show_settings = False


def _ask(query: str, preset_id: str | None = None, via: str | None = None) -> None:
    ss = st.session_state
    convo = _active_or_new_convo()
    convo["entries"].append({
        "query": query,
        "req_json": _build_request(query).model_dump_json(),
        "start": ss.landmark if ss.landmark != CUSTOM_POINT else "свои координаты",
        "via": via,
    })
    convo["entries"] = convo["entries"][-MAX_HISTORY:]
    if len(convo["entries"]) == 1:
        convo["title"] = _truncate(query, 38)
    ss.active_preset = preset_id


def run_preset(preset: dict) -> None:
    """Button callback: load a jury scenario's context and ask its question."""
    _load_settings(preset)
    _ask(preset["query"], preset["id"])


def apply_landmark() -> None:
    name = st.session_state.landmark
    if name in LANDMARK_POINTS:
        st.session_state.lat, st.session_state.lon, _ = LANDMARK_POINTS[name]


def request_geolocation() -> None:
    """Find-my-location callback: bump the counter so the drawer re-mounts the JS fix."""
    st.session_state.geo_click = int(_ss_get("geo_click", 0)) + 1


def _apply_geo_result(loc: dict, geo_key: str) -> str | None:
    """Store a browser location fix into the settings: round at once, keep no raw values."""
    ss = st.session_state
    coords = (loc or {}).get("coords") or {}
    try:
        r_lat, r_lon = round_coords(float(coords["latitude"]), float(coords["longitude"]))
    except (KeyError, TypeError, ValueError):
        err = (loc or {}).get("error") or {}
        msg = str(err.get("message") or err.get("code") or "доступ запрещён браузером")
        return f"Браузер не дал координаты ({msg}) — выберите ориентир или введите их вручную."
    ss.lat = r_lat
    ss.lon = r_lon
    ss.landmark = CUSTOM_POINT
    ss.geo_click = 0            # the fix is applied: stop polling GPS on later reruns
    ss.geo_applied_for = None   # ...and drop the guard with it
    acc = coords.get("accuracy")
    tail = f" · точность ±{int(float(acc))} м" if acc else ""
    return f"Точка старта из геолокации: ~{r_lat}, {r_lon}{tail} · сырые координаты не сохраняем"


# ── Voice input (browser Speech Recognition through streamlit-js-eval) ───────

_VOICE_LANG = {"ru": "ru-RU", "kk": "kk-KZ", "en": "en-US"}

_VOICE_ERROR_HINTS = {
    "unsupported": "браузер не поддерживает распознавание речи — нужны Chrome, Edge или Safari",
    "not-allowed": "доступ к микрофону запрещён — разрешите его в настройках браузера",
    "no-speech": "не расслышал речь — попробуйте ещё раз",
    "timeout": "не дождался фразы — попробуйте ещё раз",
    "network": "распознаванию нужна сеть, а она недоступна",
    "audio-capture": "микрофон не найден",
    "language-not-supported": "этот язык не распознаётся браузером — смените язык в панели настроек",
}

# One-shot recognition: click -> speak -> transcript resolves as JSON-ish dict.
# The wrapper awaits the returned promise and sends the result back to Python.
_VOICE_JS_TMPL = """
(function() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    return new Promise((resolve) => {
        if (!SR) { resolve({error: 'unsupported'}); return; }
        let rec; let settled = false;
        const finish = (v) => { if (settled) return; settled = true;
            try { if (rec) rec.stop(); } catch (e) {} resolve(v); };
        try {
            rec = new SR();
            rec.lang = '__AW_LANG__';
            rec.interimResults = false;
            rec.maxAlternatives = 1;
            rec.onresult = (e) => {
                const piece = e.results[e.results.length - 1];
                finish({text: String((piece && piece[0] && piece[0].transcript) || '').trim()});
            };
            rec.onerror = (e) => finish({error: String((e && e.error) || 'error')});
            rec.onend = () => finish({error: 'no-speech'});
            setTimeout(() => finish({error: 'timeout'}), 20000);
            rec.start();
        } catch (err) {
            finish({error: String((err && err.message) || err)});
        }
    });
})()
"""


def _voice_js(lang: str) -> str:
    """SpeechRecognition fix as an IIFE promise, tuned to the chat language."""
    return _VOICE_JS_TMPL.replace("__AW_LANG__", _VOICE_LANG.get(lang, "ru-RU"))


# Fallback capture: record mic audio to a 16 kHz mono WAV (base64) when the
# browser Speech Recognition API is unavailable. Auto-stops after 1.6 s of
# silence following speech, or at a 10 s cap. The clip goes to OpenRouter STT.
_VOICE_CAPTURE_JS = """
(function() {
    return new Promise((resolve) => {
        let done = false; let ac; let stream; let node;
        const cleanup = () => { try {
            if (node) { node.onaudioprocess = null; node.disconnect(); }
            if (stream) stream.getTracks().forEach(t => t.stop());
            if (ac) ac.close();
        } catch (e) {} };
        const finish = (v) => { if (done) return; done = true; cleanup(); resolve(v); };
        const toWavB64 = (chunks, srcRate) => {
            let total = 0; for (const c of chunks) total += c.length;
            const all = new Float32Array(total); let off = 0;
            for (const c of chunks) { all.set(c, off); off += c.length; }
            const rate = 16000, step = srcRate / rate, n = Math.floor(total / step);
            const buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
            const ws = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
            ws(0,'RIFF'); v.setUint32(4, 36 + n*2, true); ws(8,'WAVE'); ws(12,'fmt ');
            v.setUint32(16,16,true); v.setUint16(20,1,true); v.setUint16(22,1,true);
            v.setUint32(24,rate,true); v.setUint32(28,rate*2,true); v.setUint16(32,2,true); v.setUint16(34,16,true);
            ws(36,'data'); v.setUint32(40, n*2, true);
            let p = 44;
            for (let k = 0; k < n; k++) {
                const t = k * step, i0 = Math.floor(t), i1 = Math.min(total - 1, i0 + 1), f = t - i0;
                const s = all[i0] * (1 - f) + all[i1] * f;
                v.setInt16(p, Math.max(-1, Math.min(1, s)) * 0x7FFF, true); p += 2;
            }
            const bytes = new Uint8Array(buf); let bin = '';
            for (let i = 0; i < bytes.length; i += 0x8000) {
                bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
            }
            return btoa(bin);
        };
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { finish({error: 'unsupported'}); return; }
        navigator.mediaDevices.getUserMedia({ audio: true }).then((s) => {
            stream = s;
            ac = new (window.AudioContext || window.webkitAudioContext)();
            let chunks = []; let loud = false; let quiet = 0; let batches = 0;
            node = ac.createScriptProcessor(4096, 1, 1);
            node.onaudioprocess = (e) => {
                const IN = e.inputBuffer.getChannelData(0);
                let sum = 0;
                for (let i = 0; i < IN.length; i++) sum += IN[i] * IN[i];
                chunks.push(new Float32Array(IN));
                batches++;
                const rms = Math.sqrt(sum / IN.length);
                if (rms > 0.012) { loud = true; quiet = 0; }
                else if (loud) { quiet++; }
                const seconds = batches * 4096 / ac.sampleRate;
                if (seconds >= 10 || (loud && quiet >= 19)) {
                    finish({ data: toWavB64(chunks, ac.sampleRate) });
                }
            };
            const sink = ac.createGain(); sink.gain.value = 0;
            ac.createMediaStreamSource(s).connect(node);
            node.connect(sink); sink.connect(ac.destination);
            setTimeout(() => { finish({ data: toWavB64(chunks, ac.sampleRate) }); }, 12000);
        }).catch((err) => { cleanup(); finish({error: String((err && err.name) || err)}); });
    });
})()
"""


def _capture_js() -> str:
    """Mic recording fix for the server-side STT fallback."""
    return _VOICE_CAPTURE_JS


def request_voice() -> None:
    """Mic callback: bump a MONOTONIC serial so every fix gets a fresh component key.
    Resetting to a reused key would let an already-consumed transcript leak back in."""
    ss = st.session_state
    ss.mic_serial = int(_ss_get("mic_serial", 0)) + 1
    ss.mic_click = ss.mic_serial


def _clear_voice_note() -> None:
    """Drop the last voice status when the user navigates elsewhere."""
    st.session_state.pop("mic_state", None)


def _handle_sr_result(res: dict) -> bool:
    """Process one browser SpeechRecognition fix. False when SR is just unavailable."""
    ss = st.session_state
    text = " ".join(str((res or {}).get("text") or "").split())
    if (res or {}).get("error") == "unsupported":
        return False  # hand over to the server-side STT pipeline
    if len(text) >= 4:
        ss.mic_click = 0  # consumed: unmount the fix, stop listening until the next click
        ss.mic_state = ("sent", f"🎙 Распознано и отправлено: «{text}»")
        _ask(text, via="voice")
        return True
    if text:
        ss.mic_click = 0
        ss.mic_state = ("error", f"🎙 Не расслышал («{text}») — попробуйте ещё раз или напечатайте.")
        return True
    err = str((res or {}).get("error") or "no-speech")
    hint = _VOICE_ERROR_HINTS.get(err.lower(), f"не удалось распознать ({err})")
    ss.mic_click = 0  # any terminal outcome stops the mic loop
    ss.mic_state = ("error", f"🎙 {hint} — можно напечатать вопрос")
    return True


def _handle_capture_result(caps: dict) -> None:
    """Send the recorded clip to OpenRouter STT and auto-ask the transcript."""
    ss = st.session_state
    clip = str((caps or {}).get("data") or "")
    if not clip:
        err = str((caps or {}).get("error") or "неизвестная ошибка записи")
        hint = _VOICE_ERROR_HINTS.get(err.lower(), f"записать аудио не вышло ({err})")
        ss.mic_click = 0
        ss.mic_state = ("error", f"🎙 {hint} — напечатайте вопрос")
        return
    ss.mic_state = ("listen", "Сервер распознаёт аудио (OpenRouter STT)…")
    try:
        text = _server_transcribe(clip)
    except Exception as exc:  # noqa: BLE001 — the demo stays clickable on any provider error
        short = " ".join(str(exc).split())[:120]
        ss.mic_click = 0
        ss.mic_state = ("error", f"🎙 Серверное распознавание не сработало ({short}) — напечатайте вопрос")
        return
    ss.mic_click = 0
    ss.mic_state = ("sent", f"🎙 Распознано ИИ (OpenRouter STT) и отправлено: «{text}»")
    _ask(text, via="voice")


def _voice_consume() -> None:
    """Handle the pending speech fix before anything renders: auto-ask or stash a note."""
    ss = st.session_state
    click = int(_ss_get("mic_click", 0))
    if not click or _js_eval is None:
        return
    res = _js_eval(js_expressions=_voice_js(_ss_get("lang", "ru")), key=f"mic_request_{click}")
    if res is None:
        ss.mic_state = ("listen", "Слушаю… говорите вопрос, пауза завершит фразу.")
        return
    if _handle_sr_result(res):
        return
    # Browser Speech Recognition is not available here: record audio and use the
    # cheap server-side STT model through OpenRouter (Gemini Transcribe).
    if not _server_asr_ready():
        ss.mic_click = 0
        ss.mic_state = ("error",
                        "🎙 В этом браузере встроенное распознавание недоступно, а для серверного "
                        "нужен OPENROUTER_API_KEY — напечатайте вопрос или добавьте ключ")
        return
    caps = _js_eval(js_expressions=_capture_js(), key=f"mic_capture_{click}")
    if caps is None:
        ss.mic_state = ("listen", "Записываю… говорите вопрос, пауза завершит фразу.")
        return
    _handle_capture_result(caps)


if "convos" not in st.session_state:
    st.session_state.convos = []          # [{"id", "title", "entries"}] — история чатов
    st.session_state.active_convo = None  # id открытого чата; None = пустой, как в ChatGPT
    st.session_state.next_convo_id = 1
    st.session_state.active_preset = None
    st.session_state.show_settings = True
    st.session_state.lang = "ru"
    st.session_state.include_route = True
    st.session_state.auto_anchor = True
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
    if entry.get("via") == "voice":
        meta = f"🎙 голос · {meta}"
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
            get_fill_color=[2, 132, 199, 38],
            get_line_color=[2, 132, 199, 200],
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
            get_color=[2, 132, 199, 170],
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



# ── Layout: ChatGPT-style sidebar | chat ─────────────────────────────────────

show_settings = st.session_state.show_settings
with st.container(key="app_layout"):
    if show_settings:
        col_side, col_chat, col_settings = st.columns([0.20, 0.56, 0.24], gap="small", vertical_alignment="top")
    else:
        col_settings = None
        col_side, col_chat = st.columns([0.235, 0.765], gap="small", vertical_alignment="top")

# Keep settings state alive across drawer open/close cycles (see _SETTING_KEYS).
_sync_settings_backup()
_restore_settings_state()
r_lat, r_lon = round_coords(st.session_state.lat, st.session_state.lon)

# Voice input: consume the pending speech fix before columns render, so the new
# voice message is already in the entry list when the chat column is drawn.
_voice_consume()

with col_side:
    with st.container(key="side", height=800, border=False):
        _html(f"""
            <div class="aw-brand">
                <img src="{LOGO_URL}" alt="AstanaWalk.ai — прогулки по Астане">
            </div>
            """)
        st.button(
            "Новый чат",
            key="btn_newchat",
            icon=":material/add_comment:",
            on_click=new_chat,
            use_container_width=True,
            help="Начать новый разговор; прошлые чаты останутся в истории ниже",
        )

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
            st.html(f"""<style>.st-key-btn_{st.session_state.active_preset} button {{
                    background: {ACCENT_SOFT} !important; border-left-color: {ACCENT} !important;
                    font-weight: 600 !important; }}</style>""")

        st.markdown('<div class="aw-side-sep"></div><div class="aw-side-title">История чатов</div>',
                    unsafe_allow_html=True)
        convos = st.session_state.convos
        if convos:
            with st.container(key="chats"):
                for convo in reversed(convos):
                    st.button(
                        _convo_label(convo),
                        key=f"btn_convo_{convo['id']}",
                        icon=":material/chat_bubble:",
                        on_click=open_convo,
                        args=(convo["id"],),
                        use_container_width=True,
                    )
            current = st.session_state.active_convo
            if current is not None:
                st.html(f"""<style>.st-key-btn_convo_{current} button {{
                        background: {ACCENT_SOFT} !important; color: {ACCENT} !important;
                        font-weight: 600 !important; }}</style>""")
        else:
            st.markdown(
                '<div class="aw-empty-side">Пока нет чатов. Задайте вопрос — и он появится здесь.</div>',
                unsafe_allow_html=True,
            )

        _html(f"""
            <div class="aw-privacy-strip">
                Точка поиска: <span class="mono">{r_lat}, {r_lon}</span> — округлена до ~100 м.
                Точные координаты не хранятся и не отправляются в модель.
            </div>
            """)

with col_chat:
    with st.container(key="chat", height=800, border=False):
        convo = next(
            (c for c in st.session_state.convos if c["id"] == st.session_state.active_convo), None,
        )
        entries = convo["entries"] if convo else []
        last_mode = None
        if entries:
            last_mode = run_recommendations(entries[-1]["req_json"]).used_data.generation_mode
        mode_label = "нейросеть" if last_mode and "llm" in last_mode.lower() else "шаблоны"
        _html(f"""
            <div class="aw-head">
                <div>
                    <div class="aw-head-title">Astana Walk</div>
                    <div class="aw-head-sub">Куда сходить в Астане прямо сейчас</div>
                </div>
                <div class="aw-pills">
                    <span class="aw-pill"><span class="msi">location_on</span>{len(load_pois())} мест</span>
                    <span class="aw-pill"><span class="msi">shield_lock</span>точка ~100 м</span>
                    <span class="aw-pill"><span class="msi">auto_awesome</span>ответ: {mode_label}</span>
                </div>
            </div>
            """)

        # Settings gear pinned to the chat header's top-right corner.
        with st.container(key="gearrow"):
            st.button(
                "Настройки",
                key="btn_gear",
                icon=":material/settings:",
                on_click=toggle_settings,
                help="Свернуть или раскрыть панель настроек (точка старта, радиус, время, погода)",
            )
        if st.session_state.show_settings:
            st.html(f"""<style>.st-key-btn_gear button {{
                    background: {ACCENT_SOFT} !important; color: {ACCENT} !important; }}</style>""")

        messages = st.container(key="messages", height="stretch", border=False, autoscroll=True)
        with st.container(key="composer"):
            with st.container(key="composer_input", gap=None):
                prompt = st.chat_input("Напишите, куда хочется сходить… например: тихое кафе с розеткой на час")
                st.button(
                    "Микрофон",
                    key="btn_mic",
                    icon=":material/mic:",
                    on_click=request_voice,
                    use_container_width=False,
                    help="Нажмите и говорите: пауза завершит фразу. Голос распознаёт браузер, "
                         "а если он не умеет — аудио уходит в ИИ (OpenRouter STT).",
                    disabled=_js_eval is None,
                )
            _voice = _ss_get("mic_state") or ("idle", None)
            if _voice and len(_voice) == 2 and _voice[1]:
                if _voice[0] == "listen":
                    _html(f'<div class="aw-mic-live">{escape(_voice[1])}</div>')
                else:
                    st.caption(_voice[1])
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
            for i, entry in enumerate(entries):
                req = RecommendRequest.model_validate_json(entry["req_json"])
                render_user(entry, req)
                latest = i == len(entries) - 1
                with st.spinner("Подбираю места…"):
                    resp = run_recommendations(entry["req_json"])
                render_answer(req, resp, latest=latest)

if col_settings is not None:
    with col_settings:
        with st.container(key="settings", height=800, border=False):
            _head, _close = st.columns([0.82, 0.18], gap="small")
            with _head:
                _html("""
                    <div class="aw-settings-head">
                        <div>
                            <div class="aw-settings-title">Настройки прогулки</div>
                            <div class="aw-settings-sub">Применяются к следующему вопросу</div>
                        </div>
                    </div>
                    """)
            with _close:
                st.button(
                    "Закрыть",
                    key="btn_close_settings",
                    icon=":material/close:",
                    on_click=close_settings,
                    use_container_width=True,
                    help="Свернуть панель настроек",
                )
            st.button(
                "Найти мою геолокацию",
                key="btn_geo",
                icon=":material/my_location:",
                on_click=request_geolocation,
                use_container_width=True,
                help="Браузер запросит разрешение; координаты сразу округляем до ~100 м, сырые не сохраняем.",
                disabled=_js_get_geolocation is None,
            )
            _geo_click = int(_ss_get("geo_click", 0))
            _geo_key = f"geo_request_{_geo_click}"
            _geo_note = None
            if _js_get_geolocation is None:
                _geo_note = "Геолокация недоступна (нет компонента) — выберите ориентир или введите координаты."
            elif _geo_click and _ss_get("geo_applied_for") != _geo_key:
                _fix = _js_get_geolocation(component_key=_geo_key)
                if _fix is None:
                    _geo_note = "Ждём разрешение браузера и координаты…"
                else:
                    _geo_note = _apply_geo_result(_fix, _geo_key)
            if _geo_note:
                st.caption(_geo_note)
            st.selectbox(
                "Точка старта",
                options=list(LANDMARK_POINTS.keys()) + [CUSTOM_POINT],
                key="landmark",
                on_change=apply_landmark,
                help="Ориентир вместо геолокации: так не нужно раскрывать, где вы находитесь.",
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
                f'<div class="aw-foot">База: {len(load_pois())} мест · OpenStreetMap + Visit Astana · '
                f"поиск <span class=\"mono\">{BACKEND}</span></div>",
                unsafe_allow_html=True,
            )

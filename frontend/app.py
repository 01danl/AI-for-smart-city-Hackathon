"""Streamlit demo UI for Astana Walk AI (Build with AI · Smart City Track).

ChatGPT-style layout: left sidebar with "New chat", a ru/kk/en switcher, scenario shortcuts and a
revisitable conversation history; the chat in the middle; the right settings
panel shows the map of the latest answer; the gear in the chat header swaps it
for the settings drawer.

Launch with:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import base64
import json
import math
import sys
import threading
from datetime import date, datetime, time
from html import escape
from pathlib import Path

import pandas as pd
import pydeck as pdk
from pydeck.types import String as _DeckString
import streamlit as st

try:  # optional runtime helpers: browser geolocation + Web Speech API speech-to-text
    from streamlit_js_eval import streamlit_js_eval as _js_eval
except ImportError:  # without them the app still works: landmarks, manual coords, typing
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
from backend.generation.localization import (  # noqa: E402
    CORE_LANGUAGES, detect_answer_language, localize_answer, prepare_search_query,
)
from backend.generation.voice import has_server_asr as _server_asr_ready  # noqa: E402
from backend.generation.voice import transcribe_b64_wav as _server_transcribe  # noqa: E402
from backend.retrieval import BACKEND, parse_query_intent, retrieve  # noqa: E402
from backend.retrieval import warmup as _retrieval_warmup  # noqa: E402
from backend.retrieval import feedback as _feedback  # noqa: E402
from backend.retrieval.events import events_near  # noqa: E402
from backend.retrieval.geo import round_coords  # noqa: E402
from backend.retrieval.index import load_pois  # noqa: E402
from frontend.geolocation import GEOLOCATION_JS, geolocation_error  # noqa: E402
from frontend.voice_input import transcribe_question  # noqa: E402
from frontend.share import Stop, maps_route_url, qr_svg, share_text, telegram_url, whatsapp_url  # noqa: E402
from frontend.i18n import (  # noqa: E402
    LANG_SHORT,
    LANGS,
    answer_context,
    answer_dictionary,
    current_answer_language,
    category_label,
    duration_text,
    event_kind_label,
    geo_error,
    landmark_name,
    landmark_note,
    norm_lang,
    preset_query,
    preset_title,
    source_label,
    status_label,
    t,
    weather_label,
)

st.set_page_config(
    page_title="Astana Walk",
    page_icon=LOGO_MARK,
    layout="wide",
    initial_sidebar_state="collapsed",
)



@st.cache_resource(show_spinner=False)
def _warmup_thread() -> threading.Thread:
    """Load the embedding model and index once per server, in the background.

    Without it the first question waits ~20 s for the model; the page itself must not block.
    """
    thread = threading.Thread(target=_retrieval_warmup, name="aw-warmup", daemon=True)
    thread.start()
    return thread


_warmup_thread()

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

WEATHER_LABELS = {
    "normal": "Комфортная погода",
    "cold_windy": "Мороз или сильный ветер",
    "rain_snow": "Дождь или снег",
}

# Material icon per jury scenario, in OFFICIAL_PRESETS order.
PRESET_ICONS = [
    ":material/directions_walk:", ":material/local_cafe:", ":material/child_care:",
    ":material/flight_takeoff:", ":material/nightlife:", ":material/museum:",
    ":material/theater_comedy:",
]

# Demo scenario on top of the jury ones, set on the pitch day: a real event from
# data/events.json (Republic Day concert at Zhastar Palace, 22.10 at 19:00) shows up in the
# "nearby around this time" block.
EXTRA_PRESETS: list[dict] = [
    {
        "id": "preset_evening_events",
        "title": "Вечер в городе: что идёт сегодня",
        "query": "Чем заняться сегодня вечером? Хочется чего-то культурного рядом.",
        "lat": 51.1694,
        "lon": 71.4491,
        "max_distance_m": 2500,
        "now_iso": "2026-10-22T17:40",
        "accessible": False,
        "group_size": 2,
        "weather": "normal",
    },
]
DEMO_PRESETS = OFFICIAL_PRESETS + EXTRA_PRESETS

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
@import url('https://fonts.googleapis.com/css2?family=Onest:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..24,400,0,0');

html, body, [class*="css"], .stMarkdown, input, textarea, button, label {{
    font-family: 'Onest', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}
.mono {{ font-family: 'JetBrains Mono', ui-monospace, monospace; }}
.msi {{ font-family: 'Material Symbols Rounded'; font-size: 21px; line-height: 1; font-weight: 400;
        font-style: normal; display: inline-block; -webkit-font-smoothing: antialiased; }}

[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display: none; }}
.block-container {{ padding: 1rem 1.2rem 1rem; max-width: 1480px; }}

/* Panels */
[data-testid="stLayoutWrapper"]:has(> .st-key-side),
[data-testid="stLayoutWrapper"]:has(> .st-key-settings),
[data-testid="stLayoutWrapper"]:has(> .st-key-mappanel),
[data-testid="stLayoutWrapper"]:has(> .st-key-chat) {{
    height: calc(100dvh - 32px) !important; min-height: 0;
}}
.st-key-side, .st-key-settings, .st-key-mappanel, .st-key-chat {{ background: {PANEL}; border: 1px solid {LINE};
    border-radius: 18px; height: 100% !important; min-height: 0; box-sizing: border-box; }}
.st-key-side, .st-key-settings, .st-key-mappanel {{ padding: 16px !important;
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
.st-key-btn_gear button {{ width: 38px; min-width: 38px; height: 38px; min-height: 38px; padding: 0;
    border: none; background: transparent; color: {MUTED}; border-radius: 10px; }}
.st-key-btn_gear button:hover {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}
.aw-settings-head {{ display: flex; align-items: center; gap: 8px; padding: 4px 4px 10px;
                     border-bottom: 1px solid {LINE}; margin-bottom: 4px; }}
.aw-settings-title {{ font-weight: 700; color: {INK}; font-size: 0.98rem; }}
.aw-settings-sub {{ font-size: 0.72rem; color: {SUBTLE}; margin-top: 2px; }}
.st-key-btn_close_settings {{ display: flex; justify-content: flex-end; padding-top: 2px; }}
.st-key-btn_close_settings button {{ width: 32px; min-width: 32px; height: 32px; min-height: 32px; padding: 0;
    border: none; background: transparent; color: {MUTED}; border-radius: 8px; }}
.st-key-btn_close_settings button:hover {{ background: #FDECEA; color: #B42318; }}

/* Find-my-location button (first control in the drawer) */
.st-key-btn_geo {{ margin: 8px 0 2px; }}
.st-key-btn_geo button {{ width: 100%; justify-content: center; border: 1px solid {LINE};
    background: {PANEL}; border-radius: 10px; font-weight: 600; font-size: 0.85rem; color: {INK}; }}
.st-key-btn_geo button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
.st-key-btn_geo button[disabled] {{ opacity: .55; }}
/* Keep the component mounted without adding an empty row. */
[class*="st-key-geo_request_"] {{ height: 0; min-height: 0; overflow: hidden; }}

/* Remove the hidden label from flex layout: its gap otherwise shifts the icon. */
.st-key-btn_gear button, .st-key-btn_close_settings button {{
    display: flex; align-items: center; justify-content: center; gap: 0;
}}
.st-key-btn_gear button [data-testid="stMarkdownContainer"],
.st-key-btn_close_settings button [data-testid="stMarkdownContainer"] {{
    position: absolute; width: 1px; height: 1px; padding: 0;
    overflow: hidden; clip-path: inset(50%); white-space: nowrap;
}}

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
.aw-disclaimer summary {{ cursor: pointer; color: {MUTED}; font-weight: 500; list-style: none; }}
.aw-disclaimer summary::before {{ content: "ⓘ "; }}
.aw-links {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
.aw-links a {{ font-size: 0.76rem; font-weight: 600; text-decoration: none; padding: 4px 10px;
               border: 1px solid #CDE5F1; border-radius: 999px; color: {ACCENT}; background: {ACCENT_SOFT}; }}
.aw-links a:hover {{ border-color: {ACCENT}; }}
.aw-pill-start b {{ color: {ACCENT}; font-weight: 700; }}

/* Per-card rows of the latest answer and their feedback buttons */
.aw-row-card {{ margin: 0 0 0 46px; }}
.aw-row-card .aw-card {{ margin-top: 10px; }}
[class*="st-key-fbrow_"] {{ margin: 2px 0 0 46px; gap: 4px; align-items: center; }}
[class*="st-key-fbrow_"] button {{ width: 34px; min-width: 34px; height: 32px; padding: 0;
    border: 1px solid transparent; background: transparent; color: {SUBTLE}; border-radius: 8px; }}
[class*="st-key-fbrow_"] button:hover {{ border-color: {LINE}; background: {PANEL}; color: {MUTED}; }}
[class*="st-key-fbrow_"] button[aria-expanded="true"] {{ color: {ACCENT}; }}
[class*="st-key-fbrow_"] .stButton button p {{ display: none; }}
[class*="st-key-fbrow_"] [data-testid="stCaptionContainer"] {{ margin: 0; }}
.aw-user-meta .msi {{ font-size: 14px; margin-right: 3px; vertical-align: -2px; }}
.aw-warn {{ font-size: 0.78rem; color: #946200; background: #FDF3DC; border-radius: 8px;
            padding: 5px 9px; margin-top: 6px; }}

/* Temporary events */
.aw-ev-head {{ font-weight: 700; color: {INK}; font-size: 0.9rem; margin: 14px 0 0 2px; }}
.aw-event {{ border-left: 3px solid #7C3AED; }}
.aw-badge-temp {{ flex: none; color: #6D28D9; background: #F3E8FF; }}

/* Share to phone */
.aw-qr {{ display: flex; justify-content: center; padding: 6px 0 2px; }}
.aw-qr svg {{ width: 300px; height: 300px; max-width: 100%; shape-rendering: crispEdges;
              image-rendering: pixelated; }}
.aw-qr-big svg {{ width: min(70vh, 640px); height: min(70vh, 640px); }}
.aw-qr-hint {{ font-size: 0.8rem; color: {MUTED}; text-align: center; margin: 6px 4px 10px; line-height: 1.45; }}

/* Loading skeleton */
@keyframes aw-shimmer {{ 0% {{ background-position: -320px 0; }} 100% {{ background-position: 320px 0; }} }}
.aw-loading {{ color: {MUTED}; }}
.aw-skel {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 14px; padding: 14px 15px; margin-top: 10px; }}
.aw-skel i {{ display: block; height: 10px; border-radius: 6px; margin: 7px 0;
              background: linear-gradient(90deg, #EEF1F5 0%, #F7F9FB 50%, #EEF1F5 100%);
              background-size: 640px 100%; animation: aw-shimmer 1.2s linear infinite; }}

/* Right map panel */
.aw-map-empty {{ display: flex; flex-direction: column; align-items: center; gap: 8px; text-align: center;
                 color: {SUBTLE}; font-size: 0.84rem; padding: 48px 18px; border: 1px dashed {LINE};
                 border-radius: 14px; margin-top: 10px; }}
.aw-map-empty .msi {{ font-size: 34px; color: #CDE5F1; }}
.aw-tl {{ margin-top: 12px; }}
.aw-tl-head {{ font-weight: 700; color: {INK}; font-size: 0.86rem; margin-bottom: 6px; }}
.aw-tl-row {{ display: flex; gap: 10px; padding: 8px 0; border-top: 1px solid {LINE}; }}
.aw-tl-num {{ flex: none; width: 22px; height: 22px; border-radius: 50%; background: {ACCENT}; color: #fff;
              display: grid; place-items: center; font-size: 0.72rem; font-weight: 700; margin-top: 1px; }}
.aw-tl-num.extra {{ background: #fff; border: 2px solid {ACCENT}; width: 18px; height: 18px; }}
.aw-tl-name {{ font-weight: 600; color: {INK}; font-size: 0.86rem; }}
.aw-tl-meta {{ font-size: 0.75rem; color: {MUTED}; line-height: 1.4; }}

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
    [data-testid="stLayoutWrapper"]:has(> .st-key-settings),
    [data-testid="stLayoutWrapper"]:has(> .st-key-mappanel) {{ height: auto !important; }}
    .st-key-side, .st-key-settings, .st-key-mappanel {{ height: auto !important; }}
    .st-key-messages, .st-key-composer {{ padding-left: 12px; padding-right: 12px; }}
    .aw-head {{ padding: 14px 64px 14px 14px; flex-direction: column; align-items: flex-start; }}
    .aw-pills {{ justify-content: flex-start; }}
    .st-key-messages [data-testid="stExpander"] {{ margin-left: 0; }}
    .aw-row-card, [class*="st-key-fbrow_"] {{ margin-left: 0; }}
}}
</style>
"""

# Style-only HTML is kept outside the layout so it does not add an empty row.
st.html(CSS)


# ── Language ─────────────────────────────────────────────────────────────────


def L() -> str:
    """Use a message's language while rendering it, otherwise the interface language."""
    return current_answer_language() or norm_lang(st.session_state.get("lang"))


def _status(status: str) -> tuple[str, str, str]:
    """(localized label, text color, background) for an opening-hours status."""
    _, fg, bg = STATUS_STYLE.get(status, STATUS_STYLE["unknown"])
    return status_label(status, L()), fg, bg


def _source(source: str) -> tuple[str, str]:
    code = source if source in SOURCE_STYLE else "poi_data"
    return source_label(code, L()), code


def set_language() -> None:
    """Select the interface and speech language; typed answers detect their own language."""
    ss = st.session_state
    if ss.get("ui_lang") not in LANGS:
        ss.ui_lang = L()
    ss.lang = ss.ui_lang


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


def _build_request(query: str, answer_language: str | None = None) -> RecommendRequest:
    ss = st.session_state
    return RecommendRequest(
        query=query,
        lat=ss.lat,
        lon=ss.lon,
        max_distance_m=ss.max_distance_m,
        lang=(answer_language if answer_language in CORE_LANGUAGES else "en") if answer_language else ss.lang,
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
    convo = {"id": ss.next_convo_id, "title": t("new_chat", L()), "entries": []}
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
    "weather", "group_size", "accessible", "include_route", "auto_anchor",
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
    """Use the durable backup while hidden; widget cleanup can expose stale defaults."""
    ss = st.session_state
    try:
        backup = ss["_settings_backup"]
    except KeyError:
        return
    for key in _SETTING_KEYS:
        if not ss.show_settings and key in backup:
            ss[key] = backup[key]
            continue
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
    if st.session_state.show_settings:
        _sync_settings_backup()
    else:
        _restore_settings_state()
    st.session_state.show_settings = not st.session_state.show_settings


def close_settings() -> None:
    """Drawer callback: hide the right settings drawer."""
    _sync_settings_backup()
    st.session_state.show_settings = False


def _ask(query: str, preset_id: str | None = None, via: str | None = None) -> None:
    ss = st.session_state
    convo = _active_or_new_convo()
    answer_language = detect_answer_language(query, L())
    convo["entries"].append({
        "query": query,
        "req_json": _build_request(query, answer_language).model_dump_json(),
        "answer_language": answer_language,
        "start": start_label(),
        "via": via,
    })
    convo["entries"] = convo["entries"][-MAX_HISTORY:]
    if len(convo["entries"]) == 1:
        convo["title"] = _truncate(query, 38)
    ss.active_preset = preset_id


def submit_chat() -> None:
    """Save typed text or a voice message before the rerun selects a conversation and renders history.

    With voice enabled the chat input returns a ChatInputValue (text and/or audio), otherwise a str.
    """
    value = _ss_get("chat_prompt")
    if value is None:
        return
    text = value if isinstance(value, str) else (getattr(value, "text", None) or "")
    audio = None if isinstance(value, str) else getattr(value, "audio", None)
    _restore_settings_state()
    _clear_voice_note()
    if text.strip():
        _ask(text.strip())
    elif audio is not None:
        _ask_by_voice(audio.getvalue())


def run_preset(preset: dict) -> None:
    """Button callback: load a jury scenario's context and ask its question."""
    _load_settings(preset)
    st.session_state.start_source = "landmark" if st.session_state.landmark in LANDMARK_POINTS else "custom"
    st.session_state.geo_note = None
    _sync_settings_backup()
    _ask(preset_query(preset, L()), preset["id"])


def apply_landmark() -> None:
    name = st.session_state.landmark
    if name in LANDMARK_POINTS:
        st.session_state.lat, st.session_state.lon, _ = LANDMARK_POINTS[name]
        st.session_state.start_source = "landmark"
        st.session_state.geo_note = None  # one source of truth: the landmark replaces the fix


def mark_custom_point() -> None:
    """Manual coordinates replace both the landmark and an earlier geolocation fix."""
    st.session_state.landmark = CUSTOM_POINT
    st.session_state.start_source = "custom"
    st.session_state.geo_note = None


def start_label() -> str:
    """Human name of the point the next question will search around."""
    ss = st.session_state
    if ss.landmark in LANDMARK_POINTS:
        return landmark_name(ss.landmark, L())
    if _ss_get("start_source") == "geo":
        return t("start_geo", L())
    r_lat, r_lon = round_coords(ss.lat, ss.lon)
    return t("start_coords", L(), lat=r_lat, lon=r_lon)


def request_geolocation() -> None:
    """Use a fresh component key even after a successful or failed request."""
    ss = st.session_state
    ss.geo_serial = int(_ss_get("geo_serial", 0)) + 1
    ss.geo_click = ss.geo_serial
    ss.geo_note = None


def _apply_geo_result(loc: dict, geo_key: str) -> str | None:
    """Store a browser location fix into the settings: round at once, keep no raw values."""
    ss = st.session_state
    if geo_key != f"geo_request_{_ss_get('geo_click', 0)}":
        return _ss_get("geo_note")
    ss.geo_click = 0
    coords = loc.get("coords") if isinstance(loc, dict) else None
    coords = coords if isinstance(coords, dict) else {}
    try:
        lat, lon = float(coords["latitude"]), float(coords["longitude"])
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Invalid browser coordinates")
        r_lat, r_lon = round_coords(lat, lon)
    except (KeyError, TypeError, ValueError):
        err = loc.get("error") if isinstance(loc, dict) else None
        code = err.get("code") if isinstance(err, dict) else None
        ss.geo_note = geo_error(code, L(), geolocation_error(code))
        return ss.geo_note
    ss.lat = r_lat
    ss.lon = r_lon
    ss.landmark = CUSTOM_POINT
    ss.start_source = "geo"
    try:
        acc = float(coords.get("accuracy"))
        tail = t("geo_accuracy", L(), acc=math.ceil(acc)) if math.isfinite(acc) and acc > 0 else ""
    except (TypeError, ValueError):
        tail = ""
    ss.geo_note = t("geo_ok", L(), lat=r_lat, lon=r_lon, tail=tail)
    _sync_settings_backup()
    return ss.geo_note


# ── Voice input (recorded in the chat input, transcribed on the server) ───────
#
# The chat input's built-in recorder replaced the browser Web Speech API in a hidden
# iframe, which could stay on "listening" forever in Chrome. See frontend/voice_input.py.


def _clear_voice_note() -> None:
    """Drop the last voice status when the user navigates elsewhere."""
    st.session_state.pop("mic_state", None)


def _ask_by_voice(wav: bytes) -> None:
    """Transcribe a recorded question and ask it, or leave a status note on failure."""
    text, note = transcribe_question(wav, _server_transcribe, lang=L())
    st.session_state.mic_state = ("sent" if text else "error", note)
    if text:
        _ask(text, via="voice")


if "convos" not in st.session_state:
    st.session_state.convos = []          # [{"id", "title", "entries"}] — история чатов
    st.session_state.active_convo = None  # id открытого чата; None = пустой, как в ChatGPT
    st.session_state.next_convo_id = 1
    st.session_state.active_preset = None
    st.session_state.show_settings = False  # right panel shows the map; gear opens settings
    st.session_state.start_source = "landmark"  # landmark | geo | custom
    st.session_state.lang = "ru"
    st.session_state.ui_lang = "ru"
    st.session_state.include_route = True
    st.session_state.auto_anchor = True
    _load_settings(OFFICIAL_PRESETS[0])


@st.cache_data(show_spinner=False, max_entries=64)
def run_recommendations(req_json: str, use_llm: bool = True) -> RecommendResponse:
    """Same request -> same answer: no repeated LLM calls on reruns or repeated scenarios."""
    _warmup_thread().join()  # reuse the background load instead of loading the model twice
    req = RecommendRequest.model_validate_json(req_json)
    if use_llm:
        return generate_recommendations(req)
    # Additional languages compose the final text in the localization call, avoiding a second generation.
    return generate_recommendations(req, use_llm=False)


def entry_response(entry: dict) -> RecommendResponse:
    """Cache the canonical answer and its localized display copy per message."""
    req = RecommendRequest.model_validate_json(entry["req_json"])
    language = entry.get("answer_language", req.lang)
    if "response_json" not in entry:
        search_query, translated = prepare_search_query(req.query, language)
        entry["search_query"] = search_query
        entry["query_translation_ok"] = translated
        search_req = req.model_copy(update={"query": search_query})
        entry["response_json"] = run_recommendations(
            search_req.model_dump_json(), use_llm=language in CORE_LANGUAGES,
        ).model_dump_json()
    if "presentation" not in entry:
        canonical = RecommendResponse.model_validate_json(entry["response_json"])
        entry["presentation"] = localize_answer(canonical, language, answer_dictionary())
        if not entry.get("query_translation_ok", True):
            note = ("搜索请求翻译暂时不可用，搜索结果可能不完整。" if language == "zh" else
                    "Search translation is unavailable; results may be incomplete.")
            entry["presentation"]["note"] = " ".join(filter(None, [note, entry["presentation"]["note"]]))
    return RecommendResponse.model_validate_json(entry["presentation"]["response_json"])


# ── Rendering helpers ────────────────────────────────────────────────────────

def _skeleton_html() -> str:
    return f"""
<div class="aw-row"><div class="aw-bot-badge">AW</div><div class="aw-answer">
<div class="aw-bubble aw-loading">{t("skeleton", L())}</div>
<div class="aw-skel"><i style="width:45%"></i><i style="width:80%"></i><i style="width:65%"></i></div>
<div class="aw-skel"><i style="width:40%"></i><i style="width:75%"></i><i style="width:55%"></i></div>
</div></div>
"""


def _html(markup: str, target=st) -> None:
    """Render raw HTML. Lines are flattened: Markdown would treat indented lines after
    a blank line as a code block and show the tags as text."""
    target.markdown(" ".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def render_chat_header(target, generation_mode: str | None) -> None:
    lang = L()
    if generation_mode == "pending":
        mode_label = t("mode_pending", lang)
    elif generation_mode:
        mode_label = t("mode_llm" if "llm" in generation_mode.lower() else "mode_template", lang)
    else:
        mode_label = t("mode_llm" if _server_asr_ready() else "mode_template", lang)
    _html(f"""
        <div class="aw-head">
            <div>
                <div class="aw-head-title">Astana Walk</div>
                <div class="aw-head-sub">{t("head_sub", lang)}</div>
            </div>
            <div class="aw-pills">
                <span class="aw-pill"><span class="msi">location_on</span>{t("pill_places", lang, n=len(load_pois()))}</span>
                <span class="aw-pill aw-pill-start" title="{t("pill_start_title", lang)}">
                    <span class="msi">near_me</span>{t("pill_start", lang)}&nbsp;<b>{escape(start_label())}</b>
                </span>
                <span class="aw-pill"><span class="msi">auto_awesome</span>{t("pill_mode", lang, mode=mode_label)}</span>
            </div>
        </div>
        """, target=target)


def _hhmm(req: RecommendRequest) -> str:
    return (req.now_iso or "")[11:16]


def render_user(entry: dict, req: RecommendRequest) -> None:
    km = f"{req.max_distance_m / 1000:.1f}".rstrip("0").rstrip(".")
    lang = L()
    extras = []
    if req.accessible:
        extras.append(t("meta_step_free", lang))
    if req.group_size > 1:
        extras.append(t("meta_people", lang, n=req.group_size))
    if req.weather != "normal":
        extras.append(weather_label(req.weather, lang).lower())
    meta = " · ".join([f"<span class='msi'>location_on</span> {escape(entry['start'])}", t("meta_radius", lang, km=km), *extras])
    if entry.get("via") == "voice":
        meta = f"<span class='msi'>mic</span> {t('meta_voice', lang)} · {meta}"
    _html(f"""
        <div class="aw-row user">
            <div class="aw-bubble user">
                {escape(entry['query'])}
                <div class="aw-user-meta">{meta}</div>
                <div class="aw-time">{_hhmm(req)}</div>
            </div>
            <div class="aw-user-badge">{t("you", L())}</div>
        </div>
        """)


def _context_chips(u) -> str:
    lang = L()
    chips = []
    if u.anchor_resolved:
        chips.append(f"{t('chip_anchor', lang)} <b>{escape(u.anchor_resolved.split('(')[0].strip())}</b>")
    if u.tags:
        chips.append(t("chip_tags", lang) + " " + ", ".join(f"<b>{escape(tag)}</b>" for tag in u.tags[:4]))
    if u.filters:
        chips.append(t("chip_filters", lang) + " " + ", ".join(f"<b>{escape(f)}</b>" for f in u.filters[:4]))
    budget = f"{u.time_budget_min} {t('min', lang)}" if u.time_budget_min else t("chip_auto", lang)
    chips.append(f"{t('chip_time', lang)} <b>{budget}</b>")
    chips.append(f"{t('chip_checked', lang)} <b>{u.candidates_evaluated}</b>")
    mode = t("mode_llm" if "llm" in u.generation_mode.lower() else "chip_template", lang)
    chips.append(f"{t('chip_answer', lang)} <b>{mode}</b>")
    return "".join(f'<span class="aw-chip">{c}</span>' for c in chips)


def _card_html(idx: int, rec, closed_reports: int = 0) -> str:
    lang = L()
    st_label, st_fg, st_bg = _status(rec.open_status)
    src_label, src_code = _source(rec.source)
    category = category_label(rec.category, lang)
    facts = "<br/>".join(escape(f) for f in rec.facts_poi_data) or "—"
    official = ""
    poi = load_pois().get(rec.poi_id)
    if poi and poi.source.startswith("https://visitastana.kz/"):
        official = (
            f'<div class="aw-note"><a href="{escape(poi.source)}" target="_blank">{t("visit_source", lang)}</a>'
            f" · {t('received', lang, date=escape(poi.collected_at))}"
            + (f"<br/>{escape(poi.description)}" if poi.description and lang == "ru" else "")
            + "</div>"
        )
    return f"""
        <div class="aw-card">
            <div class="aw-card-top">
                <div><div class="aw-name">{escape(rec.name)}</div><div class="aw-cat">{escape(category)}</div></div>
                <span class="aw-num">#{idx}</span>
            </div>
            <div class="aw-meta">
                <span>{t("card_away", lang, d=rec.distance_m)}</span>
                <span>{t("card_walk", lang, m=rec.walk_min)}</span>
                <span>{t("card_visit", lang, m=rec.visit_min)}</span>
            </div>
            <div class="aw-badges">
                <span class="aw-badge" style="color:{st_fg}; background:{st_bg};">{st_label}</span>
                <span class="aw-badge aw-badge-src">{src_label}<span class="mono">{src_code}</span></span>
            </div>
            <p><span class="aw-k">{t("why", lang)}</span>{escape(rec.why)}</p>
            <p><span class="aw-k">{t("what", lang)}</span>{escape(rec.what_to_do)}</p>
            <div class="aw-note">{escape(rec.confidence_note)}</div>
            {f'<div class="aw-warn">{t("fb_closed_warning", lang, n=closed_reports)}</div>' if closed_reports else ""}
            {official}
            <div class="aw-links">
                <a href="https://www.google.com/maps/dir/?api=1&destination={rec.lat:.6f},{rec.lon:.6f}&travelmode=walking"
                   target="_blank" rel="noopener">{t("link_walk", lang)}</a>
                <a href="https://2gis.kz/astana/geo/{rec.lon:.6f},{rec.lat:.6f}" target="_blank" rel="noopener">{t("link_2gis", lang)}</a>
            </div>
            <details>
                <summary>{t("facts_summary", lang)}</summary>
                <div><b>{t("original_name", lang)}</b> {escape(poi.name if poi else rec.name)}</div>
                <div><b>{t("facts", lang)}</b><br/>{facts}</div>
                <div><b>{t("boundary", lang)}</b> {escape(rec.generated_plan_note or "—")}</div>
            </details>
        </div>
    """


def render_events(req: RecommendRequest, resp: RecommendResponse) -> None:
    """Temporary events at venues within reach around the chosen time (data/events.json)."""
    lang = L()
    lat, lon = resp.used_data.rounded_coords
    found = events_near(lat, lon, load_pois(), now_iso=req.now_iso, max_distance_m=max(req.max_distance_m, 2500))
    if not found:
        return
    cards = []
    for c in found:
        ev = c.event
        if c.status == "ongoing":
            when = t("ev_ongoing", lang)
        elif c.status == "starts_soon":
            when = t("ev_soon", lang, t=duration_text(c.minutes_to_start, lang))
        else:
            when = t("ev_later", lang, time=ev.start[11:16])
        checked = ".".join(reversed(ev.checked_at.split("-")))[:5]
        cards.append(f"""
            <div class="aw-card aw-event">
                <div class="aw-card-top">
                    <div><div class="aw-name">{escape(ev.title)}</div>
                         <div class="aw-cat">{event_kind_label(ev.kind, lang)} · {escape(c.venue.name)}</div></div>
                    <span class="aw-badge aw-badge-temp">{t("ev_temporary", lang)}</span>
                </div>
                <div class="aw-meta">
                    <span><b>{when}</b></span>
                    <span>{t("card_away", lang, d=c.distance_m)}</span>
                    <span>{t("card_walk", lang, m=c.walk_min)}</span>
                </div>
                <div class="aw-note"><a href="{escape(ev.source_url)}" target="_blank" rel="noopener">
                    {t("ev_source", lang, date=checked)}</a> · {t("ev_note", lang)}</div>
            </div>""")
    _html(f"""
        <div class="aw-row aw-row-card"><div class="aw-answer">
            <div class="aw-ev-head">{t("ev_title", lang)}</div>
            {"".join(cards)}
        </div></div>
        """)


def vote(entry_key: str, poi_id: str, value: str, reason: str | None = None) -> None:
    """Feedback button callback: store the vote anonymously and remember it for this card."""
    _feedback.record_vote(poi_id, value, reason, lang=L())  # type: ignore[arg-type]
    st.session_state.setdefault("voted", {})[f"{entry_key}:{poi_id}"] = value


def render_feedback(entry_key: str, rec) -> None:
    """Gray thumb up/down row under a card of the latest answer (ChatGPT-style)."""
    lang = L()
    voted = st.session_state.get("voted", {}).get(f"{entry_key}:{rec.poi_id}")
    with st.container(key=f"fbrow_{entry_key}_{rec.poi_id}", horizontal=True):
        if voted:
            st.caption(t("fb_thanks", lang))
            return
        st.button("", key=f"fb_up_{entry_key}_{rec.poi_id}", icon=":material/thumb_up:",
                  help=t("fb_up", lang), on_click=vote, args=(entry_key, rec.poi_id, "up"))
        with st.popover("", icon=":material/thumb_down:", help=t("fb_down", lang),
                        key=f"fb_dnpop_{entry_key}_{rec.poi_id}"):
            for reason in _feedback.REASONS:
                st.button(t(f"fb_reason_{reason}", lang), key=f"fb_dn_{reason}_{entry_key}_{rec.poi_id}",
                          on_click=vote, args=(entry_key, rec.poi_id, "down", reason), use_container_width=True)


def render_answer(req: RecommendRequest, resp: RecommendResponse, latest: bool, entry_key: str = "") -> None:
    u = resp.used_data
    lang = L()
    fb = _feedback.load_feedback()
    footer = ""
    if resp.recommendations:
        n = len(resp.recommendations)
        intro = t("found_1", lang) if n == 1 else t("found_n", lang, n=n)
        if latest:
            footer = (f'<details class="aw-disclaimer"><summary>{t("about_data", lang)}</summary>'
                      f"{escape(resp.disclaimer)}</details>")
    else:
        intro = f"<b>{t('nothing_found', lang)}</b> " + escape(resp.disclaimer)

    def closed(rec) -> int:
        return fb[rec.poi_id].closed if rec.poi_id in fb else 0

    cards = "" if latest else "".join(
        _card_html(i, r, closed(r)) for i, r in enumerate(resp.recommendations, start=1)
    )
    _html(f"""
        <div class="aw-row">
            <div class="aw-bot-badge">AW</div>
            <div class="aw-answer">
                <div class="aw-bubble">{intro}<div class="aw-time">{_hhmm(req)}</div></div>
                <div class="aw-context">{_context_chips(u)}</div>
                {cards}
            </div>
        </div>
        """)
    if latest:
        # Each card is its own block so real thumb up/down buttons can sit under it.
        for i, rec in enumerate(resp.recommendations, start=1):
            _html(f'<div class="aw-row aw-row-card"><div class="aw-answer">{_card_html(i, rec, closed(rec))}</div></div>')
            render_feedback(entry_key, rec)
        render_events(req, resp)
        if footer:
            _html(f'<div class="aw-row aw-row-card"><div class="aw-answer">{footer}</div></div>')
    if latest and resp.recommendations:
        if st.session_state.show_settings:  # otherwise the map lives in the right panel
            with st.expander(t("map_route", lang), icon=":material/map:", expanded=True):
                render_map(resp)
                render_route(resp)
                render_share(resp)
        with st.expander(t("how_it_works", lang), icon=":material/manage_search:"):
            render_inspector(req, resp)


def _fit_view(points: list[list[float]], px: int) -> pdk.ViewState:
    """Center and zoom so every point fits into a map about `px` pixels across."""
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    lat = (max(lats) + min(lats)) / 2
    lon = (max(lons) + min(lons)) / 2
    coslat = math.cos(math.radians(lat))
    extent_m = max((max(lats) - min(lats)) * 111_320, (max(lons) - min(lons)) * 111_320 * coslat, 400)
    # deck.gl / MapLibre use 512 px tiles: ~78 271 m per pixel at zoom 0 on the equator.
    # 1.6 leaves room for the markers at the edges.
    zoom = math.log2(78_271 * coslat * px / (extent_m * 1.6))
    return pdk.ViewState(latitude=lat, longitude=lon, zoom=max(11.0, min(15.0, zoom)), pitch=0)


def render_map(resp: RecommendResponse, height: int = 340) -> None:
    lang = L()
    center_lat, center_lon = resp.used_data.rounded_coords
    rec_ids = {r.poi_id for r in resp.recommendations}
    rec_points = [
        {"name": f"{i}. {r.name} · {r.distance_m} {t('meters', lang)}", "label": str(i),
         "lat": r.lat, "lon": r.lon,
         "color": STATUS_RGB.get(r.open_status, STATUS_RGB["unknown"])}
        for i, r in enumerate(resp.recommendations, start=1)
    ]
    stops = resp.route.stops if resp.route and len(resp.route.stops) >= 2 else []
    # Route stops that are not among the cards still need a marker, or the line ends in nowhere.
    extra_points = [
        {"name": t("map_route_step", lang, n=s.step, name=s.name), "lat": s.lat, "lon": s.lon}
        for s in stops if s.poi_id not in rec_ids
    ]
    layers = [
        # ~100 m privacy zone around the rounded user location
        pdk.Layer(
            "ScatterplotLayer",
            data=[{"name": t("map_start", lang), "lat": center_lat, "lon": center_lon}],
            get_position="[lon, lat]",
            get_fill_color=[2, 132, 199, 38],
            get_line_color=[2, 132, 199, 200],
            get_radius=110,
            stroked=True,
            line_width_min_pixels=1.5,
            pickable=True,
        ),
    ]
    if stops:
        path = [[center_lon, center_lat]] + [[s.lon, s.lat] for s in stops]
        layers.append(pdk.Layer(
            "PathLayer",
            data=[{"name": resp.route.title, "path": path}],
            get_path="path",
            get_color=[2, 132, 199, 170],
            width_min_pixels=3,
            pickable=True,
        ))
    layers += [
        pdk.Layer(
            "ScatterplotLayer",
            data=extra_points,
            get_position="[lon, lat]",
            get_fill_color=[255, 255, 255, 255],
            get_line_color=[2, 132, 199, 230],
            get_radius=6,
            radius_units=_DeckString("pixels"),
            stroked=True,
            line_width_min_pixels=2,
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=rec_points,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_line_color=[255, 255, 255, 255],
            get_radius=11,
            radius_units=_DeckString("pixels"),
            stroked=True,
            line_width_min_pixels=2,
            pickable=True,
        ),
        # Numbers on the markers match the "#1, #2, #3" badges on the cards.
        pdk.Layer(
            "TextLayer",
            data=rec_points,
            get_position="[lon, lat]",
            get_text="label",
            get_size=13,
            get_color=[255, 255, 255, 255],
            get_text_anchor=_DeckString("middle"),
            get_alignment_baseline=_DeckString("center"),
            font_weight=700,
        ),
    ]
    points = [[center_lon, center_lat]] + [[p["lon"], p["lat"]] for p in rec_points + extra_points]
    st.pydeck_chart(
        pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            initial_view_state=_fit_view(points, px=300 if height >= 400 else 520),
            layers=layers,
            tooltip={"text": "{name}"},
        ),
        height=height,
    )
    legend = "".join(
        f'<span><span class="aw-dot" style="background:rgb({c[0]},{c[1]},{c[2]})"></span>{_status(s)[0]}</span>'
        for s, c in STATUS_RGB.items()
    )
    st.markdown(
        f'<div class="aw-legend">{legend}<span>{t("map_legend_tail", lang)}</span></div>',
        unsafe_allow_html=True,
    )


def render_route_timeline(resp: RecommendResponse) -> None:
    """Compact route for the right panel: one row per stop with timing."""
    rt = resp.route
    if not rt:
        st.caption(t("route_needs_two", L()))
        return
    lang = L()
    rec_num = {r.poi_id: i for i, r in enumerate(resp.recommendations, start=1)}
    rows = []
    for stop in rt.stops:
        num = rec_num.get(stop.poi_id)
        badge = f'<span class="aw-tl-num">{num}</span>' if num else '<span class="aw-tl-num extra"></span>'
        s_label, s_fg, _ = _status(stop.open_status)
        rows.append(f"""
            <div class="aw-tl-row">
                {badge}
                <div>
                    <div class="aw-tl-name">{escape(stop.name)}</div>
                    <div class="aw-tl-meta">{t("route_row", lang, a=stop.arrival_offset_min, b=stop.departure_offset_min,
                        m=stop.from_prev_m, w=stop.from_prev_walk_min, v=stop.visit_min)}</div>
                    <div class="aw-tl-meta" style="color:{s_fg};">{s_label}</div>
                </div>
            </div>""")
    _html(f"""
        <div class="aw-tl">
            <div class="aw-tl-head">{t("route_head", lang, title=escape(rt.title), t=rt.total_duration_min,
                d=rt.total_distance_m)}</div>
            {"".join(rows)}
        </div>
        """)


def _share_stops(resp: RecommendResponse) -> list[Stop]:
    """The walking route if there is one, otherwise the recommended places in order."""
    if resp.route and len(resp.route.stops) >= 2:
        return [Stop(s.name, s.lat, s.lon, s.from_prev_walk_min) for s in resp.route.stops]
    return [Stop(r.name, r.lat, r.lon, r.walk_min) for r in resp.recommendations]


@st.dialog(" ", width="large")
def show_big_qr(url: str, hint: str) -> None:
    """Full-size QR for the stage: scannable from a projector or across the table."""
    _html(f'<div class="aw-qr aw-qr-big">{qr_svg(url)}</div><div class="aw-qr-hint">{escape(hint)}</div>')


def render_share(resp: RecommendResponse) -> None:
    """'To phone' popover: QR code with the Google Maps route, Telegram and WhatsApp links."""
    stops = _share_stops(resp)
    if not stops:
        return
    lang = L()
    url = maps_route_url(stops)
    text = share_text(stops, lang if lang in LANGS else "en")
    col_share, col_big = st.columns([3, 2], gap="small")
    if col_big.button(t("share_big", lang), key="share_big", icon=":material/fullscreen:", use_container_width=True):
        show_big_qr(url, t("share_hint", lang))
    with col_share.popover(t("share_btn", lang), icon=":material/qr_code_2:", use_container_width=True,
                           key="share_popover"):
        _html(f'<div class="aw-qr">{qr_svg(url)}</div><div class="aw-qr-hint">{t("share_hint", lang)}</div>')
        c1, c2 = st.columns(2)
        c1.link_button("Telegram", telegram_url(text, url), icon=":material/send:", use_container_width=True)
        c2.link_button("WhatsApp", whatsapp_url(text, url), icon=":material/chat:", use_container_width=True)
        st.link_button(t("share_open", lang), url, icon=":material/map:", use_container_width=True)
        st.caption(t("share_privacy", lang))


def render_route(resp: RecommendResponse) -> None:
    rt = resp.route
    if not rt:
        st.caption(t("route_needs_two", L()))
        return
    lang = L()
    st.markdown(t("route_head", lang, title=f"**{escape(rt.title)}**", t=rt.total_duration_min,
                  d=rt.total_distance_m))
    m_unit = t("min", lang)
    for col, stop in zip(st.columns(len(rt.stops)), rt.stops):
        s_label, s_fg, s_bg = _status(stop.open_status)
        _html(f"""
            <div class="aw-stop">
                <div class="aw-stop-stage">{t("stop_step", lang)} {stop.step} · {escape(stop.stage_note)}</div>
                <div class="aw-stop-name">{escape(stop.name)}</div>
                <div class="aw-stop-row"><span>{t("stop_walk", lang)}</span>
                    <b>{stop.from_prev_m} {t("meters", lang)} · ~{stop.from_prev_walk_min} {m_unit}</b></div>
                <div class="aw-stop-row"><span>{t("stop_visit", lang)}</span><b>~{stop.visit_min} {m_unit}</b></div>
                <div class="aw-stop-row"><span>{t("stop_offset", lang)}</span>
                    <b>+{stop.arrival_offset_min}…+{stop.departure_offset_min} {m_unit}</b></div>
                <div style="margin-top:6px;"><span class="aw-badge" style="color:{s_fg}; background:{s_bg};"
                     title="{escape(stop.open_reason)}">{s_label}</span></div>
            </div>
            """, target=col)


def render_inspector(req: RecommendRequest, resp: RecommendResponse) -> None:
    lang = L()
    st.caption(t("how_caption", lang))
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
    totals = _feedback.summary()
    if totals["votes"]:
        st.info(t("fb_summary", lang, votes=totals["votes"], pct=round(100 * totals["up"] / totals["votes"]),
                  closed=totals["closed_places"]), icon=":material/thumbs_up_down:")
    st.markdown(t("how_step1", lang))
    st.json(json.loads(u.model_dump_json()), expanded=True)
    if cands:
        st.markdown(t("how_step2", lang))
        st.dataframe(
            pd.DataFrame([
                {
                    t("col_place", lang): c.poi.name,
                    t("col_category", lang): category_label(c.poi.category, lang),
                    t("col_meters", lang): c.distance_m,
                    t("col_walk", lang): c.walk_min,
                    t("col_match", lang): round(c.relevance, 2),
                    t("col_hours", lang): _status(c.open_status)[0],
                    "id": c.poi.id,
                }
                for c in cands
            ]),
            use_container_width=True,
            hide_index=True,
        )
    st.markdown(t("how_step3", lang))
    msgs = build_rag_messages(req.query, cands, intent_obj, lang=req.lang)
    st.code(msgs[1]["content"], language="json", wrap_lines=True)



# ── Layout: ChatGPT-style sidebar | chat ─────────────────────────────────────

show_settings = st.session_state.show_settings
with st.container(key="app_layout"):
    # Right column: settings drawer when the gear is on, otherwise the map of the latest answer.
    col_side, col_chat, col_right = st.columns([0.20, 0.52, 0.28], gap="small", vertical_alignment="top")
col_settings = col_right if show_settings else None

# Keep settings state alive across drawer open/close cycles (see _SETTING_KEYS).
_restore_settings_state()
if show_settings:
    _sync_settings_backup()
r_lat, r_lon = round_coords(st.session_state.lat, st.session_state.lon)
lang = L()

with col_side:
    with st.container(key="side", height=800, border=False):
        _html(f"""
            <div class="aw-brand">
                <img src="{LOGO_URL}" alt="{t("logo_alt", lang)}">
            </div>
            """)
        with st.popover(LANG_SHORT[lang], icon=":material/translate:",
                        help=t("interface_language", lang), use_container_width=True,
                        key="ui_language_menu"):
            st.selectbox(t("interface_language", lang), options=list(LANGS),
                         format_func=LANG_SHORT.get, key="ui_lang", on_change=set_language)
            st.caption(t("answer_language_auto", lang))
        st.button(
            t("new_chat", lang),
            key="btn_newchat",
            icon=":material/add_comment:",
            on_click=new_chat,
            use_container_width=True,
            help=t("new_chat_help", lang),
        )

        st.markdown(f'<div class="aw-side-title">{t("scenarios", lang)}</div>', unsafe_allow_html=True)
        with st.container(key="presets"):
            for idx, preset in enumerate(DEMO_PRESETS):
                st.button(
                    preset_title(preset, lang),
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

        st.markdown(f'<div class="aw-side-sep"></div><div class="aw-side-title">{t("chats", lang)}</div>',
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
                f'<div class="aw-empty-side">{t("chats_empty", lang)}</div>',
                unsafe_allow_html=True,
            )


with col_chat:
    with st.container(key="chat", height=800, border=False):
        convo = next(
            (c for c in st.session_state.convos if c["id"] == st.session_state.active_convo), None,
        )
        entries = convo["entries"] if convo else []
        last_mode = None
        if entries:
            saved_response = entries[-1].get("response_json")
            last_mode = (RecommendResponse.model_validate_json(saved_response).used_data.generation_mode
                         if saved_response else "pending")
        header = st.empty()
        render_chat_header(header, last_mode)

        # Settings gear pinned to the chat header's top-right corner.
        with st.container(key="gearrow"):
            st.button(
                t("settings", lang),
                key="btn_gear",
                icon=":material/settings:",
                on_click=toggle_settings,
                help=t("settings_help", lang),
            )
        if st.session_state.show_settings:
            st.html(f"""<style>.st-key-btn_gear button {{
                    background: {ACCENT_SOFT} !important; color: {ACCENT} !important; }}</style>""")

        messages = st.container(key="messages", height="stretch", border=False, autoscroll=True)
        with st.container(key="composer"):
            voice_ready = _server_asr_ready()  # speech-to-text needs the OpenRouter key
            st.chat_input(
                t("placeholder_voice" if voice_ready else "placeholder", lang),
                key="chat_prompt",
                on_submit=submit_chat,
                accept_audio=voice_ready,
                audio_sample_rate=16000,
            )
            _voice = _ss_get("mic_state") or ("idle", None)
            if _voice and len(_voice) == 2 and _voice[1]:
                st.caption(_voice[1])
        with messages:
            _html(f"""
                <div class="aw-row">
                    <div class="aw-bot-badge">AW</div>
                    <div class="aw-bubble">{t("greeting", lang)}</div>
                </div>
                """)
            for i, entry in enumerate(entries):
                req = RecommendRequest.model_validate_json(entry["req_json"])
                render_user(entry, req)
                latest = i == len(entries) - 1
                if "response_json" not in entry or "presentation" not in entry:
                    loading = st.empty()
                    loading.html(_skeleton_html())
                    resp = entry_response(entry)
                    loading.empty()
                else:
                    resp = entry_response(entry)
                with answer_context(entry["presentation"]):
                    if entry["presentation"]["note"]:
                        st.caption(entry["presentation"]["note"])
                    search_req = req.model_copy(update={"query": entry.get("search_query", req.query)})
                    render_answer(search_req, resp, latest=latest, entry_key=f"{convo['id']}_{i}")
                if latest:
                    render_chat_header(header, resp.used_data.generation_mode)

if col_settings is None:
    with col_right:
        with st.container(key="mappanel", height=800, border=False):
            _html(f"""
                <div class="aw-settings-head">
                    <div>
                        <div class="aw-settings-title">{t("map_route", lang)}</div>
                        <div class="aw-settings-sub">{t("map_sub", lang)}</div>
                    </div>
                </div>
                """)
            _convo = next(
                (c for c in st.session_state.convos if c["id"] == st.session_state.active_convo), None,
            )
            _last = entry_response(_convo["entries"][-1]) if _convo and _convo["entries"] else None
            if _last is not None and _last.recommendations:
                with answer_context(_convo["entries"][-1]["presentation"]):
                    render_map(_last, height=420)
                    render_share(_last)
                    render_route_timeline(_last)
            else:
                _html(f"""
                    <div class="aw-map-empty">
                        <span class="msi">map</span>
                        {t("map_empty", lang)}
                    </div>
                    """)
else:
    with col_settings:
        with st.container(key="settings", height=800, border=False):
            _head, _close = st.columns([0.82, 0.18], gap="small")
            with _head:
                _html(f"""
                    <div class="aw-settings-head">
                        <div>
                            <div class="aw-settings-title">{t("settings_title", lang)}</div>
                            <div class="aw-settings-sub">{t("settings_sub", lang)}</div>
                        </div>
                    </div>
                    """)
            with _close:
                st.button(
                    t("close", lang),
                    key="btn_close_settings",
                    icon=":material/close:",
                    on_click=close_settings,
                    use_container_width=True,
                    help=t("close_help", lang),
                )
            st.button(
                t("geo_btn", lang),
                key="btn_geo",
                icon=":material/my_location:",
                on_click=request_geolocation,
                use_container_width=True,
                help=t("geo_help", lang),
                disabled=_js_eval is None or bool(_ss_get("geo_click", 0)),
            )
            _geo_click = int(_ss_get("geo_click", 0))
            _geo_key = f"geo_request_{_geo_click}"
            _geo_note = _ss_get("geo_note")
            if _js_eval is None:
                _geo_note = t("geo_unavailable", lang)
            elif _geo_click:
                _fix = _js_eval(js_expressions=GEOLOCATION_JS, key=_geo_key)
                if _fix is None:
                    _geo_note = t("geo_wait", lang)
                else:
                    _geo_note = _apply_geo_result(_fix, _geo_key)
                    st.rerun()  # Remove the completed component and re-enable retry.
            if _geo_note:
                st.caption(_geo_note)
            st.selectbox(
                t("start_point", lang),
                options=list(LANDMARK_POINTS.keys()) + [CUSTOM_POINT],
                format_func=lambda key: landmark_name(key, lang),
                key="landmark",
                on_change=apply_landmark,
                help=t("start_help", lang),
            )
            if st.session_state.landmark in LANDMARK_POINTS:
                st.caption(landmark_note(st.session_state.landmark, lang))
            with st.expander(t("exact_coords", lang), expanded=st.session_state.landmark == CUSTOM_POINT):
                c1, c2 = st.columns(2)
                c1.number_input(t("lat", lang), key="lat", format="%.4f", step=0.002, on_change=mark_custom_point)
                c2.number_input(t("lon", lang), key="lon", format="%.4f", step=0.002, on_change=mark_custom_point)
                st.caption(t("coords_caption", lang, lat=r_lat, lon=r_lon))
            st.slider(t("radius", lang), min_value=300, max_value=6000, step=100, key="max_distance_m")
            d1, d2 = st.columns([1.15, 1])
            d1.date_input(t("date", lang), key="visit_date", format="DD.MM.YYYY")
            d2.time_input(t("time", lang), key="visit_time", step=900)
            st.caption(t("time_caption", lang))
            st.selectbox(t("weather", lang), options=list(WEATHER_LABELS),
                         format_func=lambda w: weather_label(w, lang), key="weather")
            st.number_input(t("group", lang), min_value=1, max_value=15, key="group_size")
            st.toggle(t("toggle_step_free", lang), key="accessible")
            st.toggle(t("toggle_route", lang), key="include_route")
            st.toggle(t("toggle_anchor", lang), key="auto_anchor", help=t("toggle_anchor_help", lang))
            st.markdown(
                f'<div class="aw-foot">{t("foot", lang, n=len(load_pois()))} '
                f"<span class=\"mono\">{BACKEND}</span></div>",
                unsafe_allow_html=True,
            )

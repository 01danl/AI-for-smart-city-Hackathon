"""Streamlit Interactive Demo UI for Astana Walk AI (Build with AI · Smart City Track).

Launch with:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import json
import sys
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
    generate_recommendations,
)
from backend.generation.prompts import build_rag_messages  # noqa: E402
from backend.retrieval import BACKEND, parse_query_intent, retrieve  # noqa: E402
from backend.retrieval.geo import round_coords  # noqa: E402
from backend.retrieval.index import load_pois  # noqa: E402

st.set_page_config(
    page_title="Astana Walk AI · Городской советчик",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATUS_BADGE: dict[str, tuple[str, str]] = {
    "confirmed_open": ("🟢 Открыто по расписанию OSM", "#16a34a"),
    "likely_open": ("🟡 Вероятно открыто / скоро закрытие", "#d97706"),
    "unknown": ("⚪ Часы работы не указаны в OSM (unknown)", "#64748b"),
    "likely_closed": ("🔴 Сейчас вне часов работы OSM", "#dc2626"),
}

SOURCE_BADGE: dict[str, tuple[str, str]] = {
    "poi_data": ("📌 Источник: данные базы POI (poi_data)", "#0284c7"),
    "generated_idea": ("✨ Источник: факты POI + идея/план ИИ (generated_idea)", "#7c3aed"),
}

LANDMARK_POINTS: dict[str, tuple[float, float, str]] = {
    "Байтерек (Левый берег)": (51.1283, 71.4305, "Центр бульвара Нуржол"),
    "Mega Silk Way": (51.0891, 71.4073, "Район EXPO / Назарбаев Университет"),
    "Здание EXPO 2017 (Нур Алем)": (51.0928, 71.4215, "Выставочный кластер"),
    "ТРЦ Хан Шатыр": (51.1325, 71.4039, "Западное завершение оси Нуржол"),
    "Театр Астана Опера": (51.1357, 71.4106, "Культурный кластер Левого берега"),
    "Ботанический сад": (51.1062, 71.4166, "Парковая зона для прогулок и бега"),
    "Старый центр (Правый берег)": (51.1694, 71.4491, "Историческая часть города"),
    "За городом (Тест пустого поиска S8)": (51.3000, 71.0000, "Точка в степи без POI"),
}

# Initialize session state
if "query" not in st.session_state:
    st.session_state.query = OFFICIAL_PRESETS[0]["query"]
    st.session_state.lat = OFFICIAL_PRESETS[0]["lat"]
    st.session_state.lon = OFFICIAL_PRESETS[0]["lon"]
    st.session_state.max_distance_m = OFFICIAL_PRESETS[0]["max_distance_m"]
    st.session_state.now_iso = OFFICIAL_PRESETS[0]["now_iso"]
    st.session_state.accessible = OFFICIAL_PRESETS[0]["accessible"]
    st.session_state.group_size = OFFICIAL_PRESETS[0]["group_size"]
    st.session_state.weather = OFFICIAL_PRESETS[0]["weather"]


def apply_preset(preset: dict) -> None:
    st.session_state.query = preset["query"]
    st.session_state.lat = preset["lat"]
    st.session_state.lon = preset["lon"]
    st.session_state.max_distance_m = preset["max_distance_m"]
    st.session_state.now_iso = preset["now_iso"]
    st.session_state.accessible = preset["accessible"]
    st.session_state.group_size = preset["group_size"]
    st.session_state.weather = preset["weather"]


# ── HEADER ──
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
                padding: 1.4rem 1.8rem; border-radius: 14px; color: #f8fafc; margin-bottom: 1rem;">
        <div style="font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em; color: #93c5fd; font-weight: 700;">
            Build with AI · НИЦ Smart City Astana · Hybrid Geosemantical RAG
        </div>
        <h1 style="margin: 0.2rem 0 0.4rem 0; font-size: 1.9rem; color: #ffffff;">
            🧭 ИИ-помощник по прогулкам в городе Астана
        </h1>
        <div style="font-size: 0.98rem; color: #cbd5e1;">
            Рекомендации в реальном времени на базе <b>966 проверенных точек интереса (OSM)</b> с честной маркировкой
            <code>poi_data</code> / <code>generated_idea</code>, контролем уверенности по часам работы,
            защитой геолокации (<code>rounded_100m</code>) и автопостроением пеших микро-маршрутов.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 1-CLICK JURY SCENARIO PRESETS ──
st.markdown("#### ⚡ Тестовые сценарии жюри в один клик (из ТЗ хакатона)")
preset_cols = st.columns(3)
for idx, preset in enumerate(OFFICIAL_PRESETS):
    with preset_cols[idx % 3]:
        if st.button(preset["title"], key=f"btn_{preset['id']}", use_container_width=True):
            apply_preset(preset)

# ── SIDEBAR CONTROLS ──
with st.sidebar:
    st.header("📍 Локация и контекст")
    selected_landmark = st.selectbox(
        "Быстрый выбор опорной точки в Астане:",
        options=["(Текущие координаты)"] + list(LANDMARK_POINTS.keys()),
        index=0,
    )
    if selected_landmark != "(Текущие координаты)":
        l_lat, l_lon, _ = LANDMARK_POINTS[selected_landmark]
        st.session_state.lat = l_lat
        st.session_state.lon = l_lon

    lat = st.number_input("Широта (lat)", value=float(st.session_state.lat), format="%.4f", step=0.002)
    lon = st.number_input("Долгота (lon)", value=float(st.session_state.lon), format="%.4f", step=0.002)
    st.session_state.lat = lat
    st.session_state.lon = lon

    r_lat, r_lon = round_coords(lat, lon)
    st.caption(f"🔒 В логах и API сохраняется только округлённый квадрат ~100 м: `{r_lat}, {r_lon}`")

    max_distance_m = st.slider(
        "Радиус пешей доступности (м)",
        min_value=300,
        max_value=6000,
        value=int(st.session_state.max_distance_m),
        step=100,
    )
    st.session_state.max_distance_m = max_distance_m

    now_iso = st.text_input(
        "Моделируемое время (UTC+5 Астана)",
        value=str(st.session_state.now_iso or "2026-10-16T15:00"),
        help="Например: 2026-10-16T13:00 (день) или 2026-10-16T23:40 (поздний вечер)",
    )
    st.session_state.now_iso = now_iso

    weather = st.selectbox(
        "Погодный контекст Астаны",
        options=["normal", "cold_windy", "rain_snow"],
        format_func=lambda x: {
            "normal": "☀️ Комфортная погода",
            "cold_windy": "💨 Сильный ветер / мороз (приоритет Indoor)",
            "rain_snow": "🌧️ Дождь / снег (приоритет Indoor)",
        }[x],
        index=["normal", "cold_windy", "rain_snow"].index(st.session_state.weather),
    )
    st.session_state.weather = weather

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        group_size = st.number_input("Группа (чел.)", min_value=1, max_value=15, value=int(st.session_state.group_size))
        st.session_state.group_size = group_size
    with col_s2:
        lang = st.selectbox("Язык ответа", options=["ru", "kk", "en"], index=0)

    accessible = st.checkbox("♿ Безбарьерная среда (коляска / с детьми)", value=bool(st.session_state.accessible))
    st.session_state.accessible = accessible
    include_route = st.checkbox("🗺️ Строить связанный мини-маршрут (2–3 точки)", value=True)
    auto_anchor = st.checkbox("🎯 Авто-распознавание ориентиров из текста запроса", value=True)

    st.divider()
    pois_total = len(load_pois())
    st.caption(f"Активный поисковый бэкенд: **`{BACKEND}`** · База POI: **`{pois_total}` объектов**")

# ── QUERY INPUT & EXECUTION ──
query_text = st.text_input(
    "Что вы хотите сделать или куда сходить?",
    value=st.session_state.query,
    placeholder="Например: Я сейчас возле Mega Silk Way, хочу прогуляться 30–60 минут — что посоветуете?",
)
st.session_state.query = query_text

req = RecommendRequest(
    query=query_text,
    lat=lat,
    lon=lon,
    max_distance_m=max_distance_m,
    lang=lang,  # type: ignore[arg-type]
    group_size=group_size,
    accessible=accessible,
    weather=weather,  # type: ignore[arg-type]
    include_route=include_route,
    auto_anchor=auto_anchor,
    now_iso=now_iso.strip() or None,
)

resp = generate_recommendations(req)

# ── TRANSPARENCY BAR (USED DATA) ──
u = resp.used_data
meta_cols = st.columns(5)
meta_cols[0].metric("Приватность гео", u.location, f"{u.rounded_coords[0]}, {u.rounded_coords[1]}")
meta_cols[1].metric("Распознанные теги", ", ".join(u.tags[:3]) if u.tags else "семантика", f"Кандидатов: {u.candidates_evaluated}")
meta_cols[2].metric("Бюджет времени", f"{u.time_budget_min} мин" if u.time_budget_min else "Авто (90–120 мин)", f"Погода: {u.weather_context}")
meta_cols[3].metric("Ориентир из текста", u.anchor_resolved.split("(")[0].strip() if u.anchor_resolved else "По координатам UI", f"Поиск: {u.retrieval_backend}")
meta_cols[4].metric("Режим генерации", u.generation_mode, f"Язык: {lang.upper()}")

if not resp.recommendations:
    st.warning(resp.disclaimer)
else:
    tab_cards, tab_route, tab_inspector = st.tabs([
        "🏛️ Рекомендации и Карта (1–3 места)",
        "🚶 Пеший мини-маршрут (Последовательность точек)",
        "🔍 RAG & Privacy Inspector (Для жюри)",
    ])

    with tab_cards:
        col_left, col_right = st.columns([1.25, 1.0], gap="large")

        with col_left:
            for idx, rec in enumerate(resp.recommendations, start=1):
                st_label, st_color = STATUS_BADGE.get(rec.open_status, ("⚪ Неизвестно", "#64748b"))
                src_label, src_color = SOURCE_BADGE.get(rec.source, ("📌 poi_data", "#0284c7"))

                st.markdown(
                    f"""
                    <div style="border: 1px solid #e2e8f0; border-left: 6px solid {st_color};
                                border-radius: 12px; padding: 1.1rem 1.25rem; margin-bottom: 1rem;
                                background: #ffffff; box-shadow: 0 2px 6px rgba(15,23,42,0.04);">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:8px; flex-wrap:wrap;">
                            <h3 style="margin:0; font-size:1.22rem; color:#0f172a;">
                                {idx}. {rec.name}
                                <span style="font-size:0.9rem; font-weight:500; color:#475569;">
                                    ({rec.distance_m} м · ~{rec.walk_min} мин пешком · ⏱️ визит ~{rec.visit_min} мин)
                                </span>
                            </h3>
                        </div>
                        <div style="margin-top:0.45rem; display:flex; gap:8px; flex-wrap:wrap;">
                            <span style="background:{st_color}18; color:{st_color}; padding:3px 10px;
                                         border-radius:999px; font-size:0.78rem; font-weight:600;">
                                {st_label}
                            </span>
                            <span style="background:{src_color}15; color:{src_color}; padding:3px 10px;
                                         border-radius:999px; font-size:0.78rem; font-weight:600;">
                                {src_label}
                            </span>
                        </div>
                        <p style="margin:0.7rem 0 0.4rem 0; color:#1e293b; font-size:0.95rem;">
                            <b>Почему подходит:</b> {rec.why}
                        </p>
                        <p style="margin:0.35rem 0; color:#0f172a; font-size:0.95rem; background:#f8fafc;
                                  padding:0.55rem 0.75rem; border-radius:8px;">
                            <b>🎯 Что делать там (план):</b> {rec.what_to_do}
                        </p>
                        <div style="margin-top:0.5rem; font-size:0.82rem; color:#475569;">
                            <b>Режим работы и достоверность:</b> {rec.confidence_note}
                        </div>
                        <div style="margin-top:0.35rem; font-size:0.80rem; color:#64748b;">
                            <b>Факты из базы [poi_data]:</b> {" · ".join(rec.facts_poi_data)}<br/>
                            <b>Граница фактов и генерации:</b> {rec.generated_plan_note}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_right:
            st.markdown("##### 🗺️ Карта рекомендаций и 100-метровой зоны приватности")
            center_lat, center_lon = u.rounded_coords
            poi_points = [
                {
                    "name": f"{i}. {r.name} ({r.distance_m} м)",
                    "lat": r.lat,
                    "lon": r.lon,
                    "color": [22, 163, 74, 220]
                    if r.open_status == "confirmed_open"
                    else ([217, 119, 6, 220] if r.open_status == "likely_open" else [37, 99, 235, 220]),
                    "radius": 45,
                }
                for i, r in enumerate(resp.recommendations, start=1)
            ]

            layers = [
                # 100m privacy zone around rounded user coordinates
                pdk.Layer(
                    "ScatterplotLayer",
                    data=[{"name": "Зона приватности пользователя (~100 м)", "lat": center_lat, "lon": center_lon}],
                    get_position="[lon, lat]",
                    get_fill_color=[14, 165, 233, 60],
                    get_line_color=[2, 132, 199, 200],
                    get_radius=110,
                    stroked=True,
                    line_width_min_pixels=2,
                    pickable=True,
                ),
                pdk.Layer(
                    "ScatterplotLayer",
                    data=poi_points,
                    get_position="[lon, lat]",
                    get_fill_color="color",
                    get_radius="radius",
                    pickable=True,
                ),
            ]

            if resp.route and len(resp.route.stops) >= 2:
                path_coords = [[center_lon, center_lat]] + [[s.lon, s.lat] for s in resp.route.stops]
                layers.append(
                    pdk.Layer(
                        "PathLayer",
                        data=[{"name": resp.route.title, "path": path_coords}],
                        get_path="path",
                        get_color=[124, 58, 237, 190],
                        width_scale=6,
                        width_min_pixels=3,
                        pickable=True,
                    )
                )

            deck = pdk.Deck(
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                initial_view_state=pdk.ViewState(
                    latitude=center_lat,
                    longitude=center_lon,
                    zoom=13.6,
                    pitch=25,
                ),
                layers=layers,
                tooltip={"text": "{name}"},
            )
            st.pydeck_chart(deck)
            st.caption(
                "🔵 Голубой круг — 100-метровая зона приватности (`round_coords`), "
                "🟢/🟡/🔵 точки — найденные POI, 🟣 фиолетовая линия — пеший мини-маршрут."
            )

    with tab_route:
        if resp.route:
            rt = resp.route
            st.subheader(f"🚶 {rt.title}")
            st.info(rt.summary)
            r_cols = st.columns(len(rt.stops))
            for col, stop in zip(r_cols, rt.stops):
                with col:
                    st.markdown(
                        f"""
                        <div style="border:1px solid #cbd5e1; border-radius:10px; padding:1rem; background:#f8fafc;">
                            <div style="font-size:0.78rem; color:#7c3aed; font-weight:700; text-transform:uppercase;">
                                {stop.stage_note}
                            </div>
                            <h4 style="margin:0.3rem 0;">{stop.step}. {stop.name}</h4>
                            <div style="font-size:0.86rem; color:#334155;">
                                🚶 Переход от пред. точки: <b>{stop.from_prev_m} м (~{stop.from_prev_walk_min} мин)</b><br/>
                                ⏱️ Время на месте: <b>~{stop.visit_min} мин</b><br/>
                                🕒 Тайминг от старта: <b>+{stop.arrival_offset_min}…+{stop.departure_offset_min} мин</b><br/>
                                📌 Статус: <code>{stop.open_status}</code> ({stop.open_reason})
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Для построения маршрута нужно минимум 2 подходящих места в радиусе поиска.")

    with tab_inspector:
        st.markdown("##### 🔬 Прозрачность работы RAG и соблюдение приватности (Критерии №3 и №6)")
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
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.markdown("**1. Извлечённые параметры и ограничения (`used_data` + `QueryIntent`):**")
            st.json(json.loads(resp.used_data.json() if hasattr(resp.used_data, "json") else json.dumps(resp.used_data.dict())))
            if cands:
                st.markdown("**2. Топ кандидатов из Retrieval до генерации:**")
                df_cands = pd.DataFrame([
                    {
                        "id": c.poi.id,
                        "Название": c.poi.name,
                        "Категория": c.poi.category,
                        "Дистанция (м)": c.distance_m,
                        "Пешком (мин)": c.walk_min,
                        "Relevance": c.relevance,
                        "Статус часов": c.open_status,
                    }
                    for c in cands
                ])
                st.dataframe(df_cands, use_container_width=True)
        with col_i2:
            st.markdown("**3. Промпт в LLM (доказательство отсутствия сырых координат пользователя):**")
            msgs = build_rag_messages(req.query, cands, intent_obj, lang=req.lang)
            st.code(msgs[1]["content"], language="json")

    st.caption(f"🛡️ **Дисклеймер:** {resp.disclaimer}")

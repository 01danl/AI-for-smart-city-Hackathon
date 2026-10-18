"""Presentation-only map layers; route geometry and POI facts are never inferred here."""

from __future__ import annotations

import math

import pydeck as pdk
from pydeck.types import String

from backend.generation.recommender import RecommendResponse

AZURE = [2, 132, 199, 255]
WHITE = [255, 255, 255, 255]
STATUS_COLORS = {
    "confirmed_open": [18, 122, 75, 255],
    "likely_open": [148, 98, 0, 255],
    "unknown": [100, 116, 139, 255],
    "likely_closed": [180, 35, 24, 255],
}
MAP_STYLES = {
    "light": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    "dark": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
}


def _project(position: list[float]) -> tuple[float, float]:
    lon, lat = position
    lat = max(-85, min(85, lat))
    return (lon + 180) / 360, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2


def fit_view(points: list[list[float]], height: int, width: int = 320) -> pdk.ViewState:
    projected = [_project(point) for point in points]
    xmin, xmax = min(p[0] for p in projected), max(p[0] for p in projected)
    ymin, ymax = min(p[1] for p in projected), max(p[1] for p in projected)
    zoom = math.log2(min((width - 96) / (512 * max(xmax - xmin, 1e-6)),
                         (height - 96) / (512 * max(ymax - ymin, 1e-6))))
    return pdk.ViewState(longitude=(xmin + xmax) * 180 - 180,
                         latitude=math.degrees(math.atan(math.sinh(math.pi * (1 - ymin - ymax)))),
                         zoom=max(2, min(17, zoom)), min_zoom=2, max_zoom=19, pitch=0, bearing=0)


def _label_positions(markers: list[dict], view: pdk.ViewState, height: int, width: int = 320) -> list[dict]:
    """Keep the initial viewport readable; hidden labels remain available in hover tooltips."""
    center = _project([view.longitude, view.latitude])
    scale = 512 * 2 ** view.zoom
    occupied = []
    screen = []
    for marker in markers:
        x, y = _project(marker["position"])
        screen.append(((x - center[0]) * scale + width / 2, (y - center[1]) * scale + height / 2))
    for x, y in screen:
        occupied.append((x - 18, y - 18, x + 18, y + 18))
    labels = []
    for marker, (x, y) in zip(markers, screen):
        name = marker["name"]
        name = name if len(name) <= 18 else name[:17] + "…"
        text = f"{marker['label']} · {name}"
        half_width = min(90, len(text) * 3.8 + 10)
        for dx, dy in ((0, -36), (0, 36), (0, -64), (0, 64)):
            box = (x + dx - half_width, y + dy - 13, x + dx + half_width, y + dy + 13)
            if box[0] < 6 or box[2] > width - 6 or box[1] < 6 or box[3] > height - 6:
                continue
            if any(box[0] < r[2] and box[2] > r[0] and box[1] < r[3] and box[3] > r[1] for r in occupied):
                continue
            labels.append(marker | {"text": text, "offset": [dx, dy]})
            occupied.append(box)
            break
    return labels


def build_map_deck(response: RecommendResponse, *, height: int = 420, theme: str = "light",
                   show_names: bool = True, start_label: str, status_labels: dict[str, str]) -> pdk.Deck:
    dark = theme == "dark"
    surface = [20, 32, 48, 250] if dark else WHITE
    ink = WHITE if dark else [29, 36, 51, 255]
    lat, lon = response.used_data.rounded_coords
    origin = [round(lon, 3), round(lat, 3)]
    start = [{"position": origin, "tooltip": start_label}]
    stops = response.route.stops if response.route and len(response.route.stops) >= 2 else []
    route_ids = {stop.poi_id for stop in stops}
    markers = []
    for number, rec in enumerate(response.recommendations, 1):
        markers.append({"poi_id": rec.poi_id, "name": rec.name, "label": str(number),
                        "extra": False, "radius": 14, "fill_color": AZURE,
                        "position": [rec.lon, rec.lat], "on_route": rec.poi_id in route_ids,
                        "status_color": STATUS_COLORS[rec.open_status],
                        "tooltip": f"{number}. {rec.name}\n{status_labels[rec.open_status]}"})
    rec_ids = {marker["poi_id"] for marker in markers}
    for stop in stops:
        if stop.poi_id not in rec_ids:
            markers.append({"poi_id": stop.poi_id, "name": stop.name, "label": "·",
                            "extra": True, "radius": 7, "fill_color": surface,
                            "position": [stop.lon, stop.lat], "on_route": True,
                            "status_color": STATUS_COLORS[stop.open_status],
                            "tooltip": f"{stop.name}\n{status_labels[stop.open_status]}"})
    # Include the privacy zone in the bounds, even when all venues are at the start point.
    bounds = [origin, [origin[0] - .002, origin[1] - .0013], [origin[0] + .002, origin[1] + .0013]]
    bounds.extend(marker["position"] for marker in markers)
    view = fit_view(bounds, height)
    common = {"get_position": "position", "parameters": {"depthTest": False}}
    layers = [pdk.Layer("ScatterplotLayer", id="privacy-zone", data=start, **common,
                        get_radius=100, get_fill_color=[2, 132, 199, 18],
                        get_line_color=[2, 132, 199, 95], stroked=True, line_width_min_pixels=1, pickable=True)]
    if stops:
        path = [origin] + [[stop.lon, stop.lat] for stop in stops]
        route = [{"path": path}]
        for layer_id, color, width in (("route-casing", surface, 9), ("route-line", AZURE, 4)):
            layers.append(pdk.Layer("PathLayer", id=layer_id, data=route, get_path="path",
                                    get_color=color, get_width=width, width_units=String("pixels"),
                                    cap_rounded=True, joint_rounded=True, parameters={"depthTest": False}))
        arrows = []
        for previous, following in zip(path, path[1:]):
            dx = (following[0] - previous[0]) * math.cos(math.radians((previous[1] + following[1]) / 2))
            dy = following[1] - previous[1]
            if math.hypot(dx, dy) < .0002:
                continue
            arrows.append({"position": [(previous[0] + following[0]) / 2, (previous[1] + following[1]) / 2],
                           "angle": math.degrees(math.atan2(dy, dx))})
        layers.append(pdk.Layer("ScatterplotLayer", id="direction-badges", data=arrows, **common,
                                radius_units=String("pixels"), get_radius=9, get_fill_color=AZURE,
                                get_line_color=surface, stroked=True, line_width_min_pixels=2))
        layers.append(pdk.Layer("TextLayer", id="route-direction", data=arrows, **common,
                                get_text=String("›"), get_angle="angle", get_size=23, get_color=WHITE,
                                font_family=String("Arial"), font_weight=700, character_set=String("auto"),
                                get_text_anchor=String("middle"), get_alignment_baseline=String("center")))
    layers.extend([
        pdk.Layer("ScatterplotLayer", id="marker-halos", data=markers, **common,
                  radius_units=String("pixels"), get_radius="radius + 5", get_fill_color=[2, 132, 199, 24]),
        pdk.Layer("ScatterplotLayer", id="place-markers", data=sorted(markers, key=lambda m: not m["extra"]), **common,
                  radius_units=String("pixels"), get_radius="radius", get_fill_color="fill_color",
                  get_line_color="status_color", stroked=True, line_width_min_pixels=2,
                  pickable=True, auto_highlight=True, highlight_color=[56, 189, 248, 100]),
        pdk.Layer("TextLayer", id="place-numbers", data=[m for m in markers if not m["extra"]], **common,
                  get_text="label", get_size=13, get_color=WHITE, font_family=String("Arial"), font_weight=700,
                  character_set=String("auto"), get_text_anchor=String("middle"),
                  get_alignment_baseline=String("center")),
        pdk.Layer("ScatterplotLayer", id="start-halo", data=start, **common,
                  radius_units=String("pixels"), get_radius=12, get_fill_color=surface),
        pdk.Layer("ScatterplotLayer", id="start-point", data=start, **common,
                  radius_units=String("pixels"), get_radius=7, get_fill_color=AZURE, pickable=True),
    ])
    if show_names:
        layers.append(pdk.Layer("TextLayer", id="place-labels", data=_label_positions(markers, view, height), **common,
                                get_text="text", get_pixel_offset="offset", get_size=11, get_color=ink,
                                font_family=String("Arial"), font_weight=600, character_set=String("auto"),
                                background=True, background_padding=[8, 6], get_background_color=surface,
                                get_border_color=[100, 116, 139, 40], get_border_width=1,
                                get_text_anchor=String("middle"), get_alignment_baseline=String("center"),
                                pickable=True))
    return pdk.Deck(layers=layers, map_style=MAP_STYLES.get(theme, MAP_STYLES["light"]),
                    initial_view_state=view,
                    views=[pdk.View(type="MapView", controller={"dragRotate": False, "touchRotate": False,
                                                               "scrollZoom": False})],
                    tooltip={"text": "{tooltip}", "style": {"backgroundColor": "#142030" if dark else "#FFFFFF",
                             "color": "#FFFFFF" if dark else "#1D2433", "fontFamily": "Arial, sans-serif",
                             "fontSize": "12px", "borderRadius": "12px", "padding": "12px"}})

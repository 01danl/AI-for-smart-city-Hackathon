"""Send a walking route to a phone: Google Maps link, QR code, Telegram and WhatsApp.

Privacy: the link never contains the user's location. Google Maps starts the walk from
wherever the phone is; only the places' own coordinates go into the URL.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote, urlencode

import segno

MAX_WAYPOINTS = 8  # Google Maps URLs accept up to 9 stops incl. the destination


@dataclass(frozen=True)
class Stop:
    name: str
    lat: float
    lon: float
    walk_min: int


def _point(stop: Stop) -> str:
    # 5 decimals is ~1 m: precise enough for a venue and keeps the QR code sparse.
    return f"{stop.lat:.5f},{stop.lon:.5f}"


def maps_route_url(stops: list[Stop]) -> str:
    """Walking directions through all stops, starting from the phone's own location."""
    if not stops:
        raise ValueError("a route needs at least one stop")
    stops = stops[: MAX_WAYPOINTS + 1]
    params = {"api": "1", "destination": _point(stops[-1]), "travelmode": "walking"}
    if len(stops) > 1:
        params["waypoints"] = "|".join(_point(s) for s in stops[:-1])
    # "|" is percent-encoded: phone cameras may not treat a URL with a raw "|" as a link.
    return "https://google.com/maps/dir/?" + urlencode(params, safe=",")


_TEXT = {
    "ru": {"head": "🚶 Маршрут от Astana Walk:", "walk": "{m} мин пешком", "map": "🗺 Открыть маршрут:"},
    "kk": {"head": "🚶 Astana Walk ұсынған бағыт:", "walk": "жаяу {m} мин", "map": "🗺 Бағытты ашу:"},
    "en": {"head": "🚶 Route from Astana Walk:", "walk": "{m} min walk", "map": "🗺 Open the route:"},
}


def share_text(stops: list[Stop], lang: str = "ru") -> str:
    words = _TEXT.get(lang, _TEXT["ru"])
    lines = [words["head"]]
    for i, stop in enumerate(stops, start=1):
        lines.append(f"{i}. {stop.name} — {words['walk'].format(m=stop.walk_min)}")
    return "\n".join(lines)


def telegram_url(text: str, url: str) -> str:
    return f"https://t.me/share/url?url={quote(url, safe='')}&text={quote(text, safe='')}"


def whatsapp_url(text: str, url: str) -> str:
    message = f"{text}\n{url}"
    return f"https://wa.me/?text={quote(message, safe='')}"


def qr_svg(url: str, *, dark: str = "#000000") -> str:
    """Inline SVG QR code sized for a phone camera pointed at a laptop screen.

    Low error correction keeps the code sparse (fewer, bigger modules); a screen has no dirt
    or folds to recover from. The 4-module quiet zone is what the QR standard requires, and
    many phone cameras ignore codes without it. Pure black on white gives the most contrast.
    """
    return segno.make(url, error="l", micro=False).svg_inline(
        scale=8, border=4, dark=dark, light="#FFFFFF", svgclass="aw-qr-code",
    )

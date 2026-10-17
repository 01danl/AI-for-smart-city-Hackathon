"""'To phone' sharing: Google Maps route link, QR code, Telegram and WhatsApp."""

from urllib.parse import parse_qs, unquote, urlparse

import pytest

from frontend.share import MAX_WAYPOINTS, Stop, maps_route_url, qr_svg, share_text, telegram_url, whatsapp_url

STOPS = [
    Stop("Байтерек", 51.128288, 71.430455, 3),
    Stop("Парк Жетысу", 51.135989, 71.43845, 18),
    Stop("Фаттуш", 51.121972, 71.427932, 7),
]


def test_route_goes_through_all_stops_from_the_phone_location():
    url = maps_route_url(STOPS)
    q = parse_qs(urlparse(url).query)
    assert q["destination"] == ["51.12197,71.42793"]
    assert q["waypoints"] == ["51.12829,71.43045|51.13599,71.43845"]
    assert "|" not in url  # encoded as %7C so phone cameras open it as a link
    assert q["travelmode"] == ["walking"]
    assert "origin" not in q  # the user's location never goes into the link


def test_single_place_has_no_waypoints():
    q = parse_qs(urlparse(maps_route_url(STOPS[:1])).query)
    assert "waypoints" not in q and q["destination"] == ["51.12829,71.43045"]


def test_long_routes_are_capped_to_what_google_maps_accepts():
    many = [Stop(f"P{i}", 51.1 + i / 1000, 71.4, 5) for i in range(15)]
    q = parse_qs(urlparse(maps_route_url(many)).query)
    assert len(q["waypoints"][0].split("|")) == MAX_WAYPOINTS


def test_empty_route_is_rejected():
    with pytest.raises(ValueError):
        maps_route_url([])


@pytest.mark.parametrize("lang, head", [("ru", "Маршрут от Astana Walk"), ("kk", "Astana Walk ұсынған бағыт"),
                                        ("en", "Route from Astana Walk")])
def test_share_text_lists_stops_in_language(lang, head):
    text = share_text(STOPS, lang)
    assert head in text
    assert "1. Байтерек" in text and "3. Фаттуш" in text


def test_messenger_links_carry_the_route():
    url = maps_route_url(STOPS)
    text = share_text(STOPS)
    tg = telegram_url(text, url)
    wa = whatsapp_url(text, url)
    assert tg.startswith("https://t.me/share/url?url=")
    assert parse_qs(urlparse(tg).query)["url"][0] == url
    assert wa.startswith("https://wa.me/?text=")
    assert parse_qs(urlparse(wa).query)["text"][0].endswith(url) and "Байтерек" in unquote(wa)


def test_qr_is_inline_svg():
    svg = qr_svg(maps_route_url(STOPS))
    assert svg.startswith("<svg") and "</svg>" in svg


def test_qr_stays_sparse_enough_for_a_phone_camera():
    """A 3-stop route must fit a low version so each module is several pixels on screen."""
    import segno

    q = segno.make(maps_route_url(STOPS), error="l", micro=False)
    assert q.version <= 6
    modules = q.symbol_size(border=4)[0]
    assert 300 / modules >= 6  # px per module at the 300 px display size

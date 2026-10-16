import pytest

from backend.retrieval.hours import open_status


@pytest.mark.parametrize(
    "hours, now, expected",
    [
        (None, "2026-10-16T12:00", "unknown"),
        ("", "2026-10-16T12:00", "unknown"),
        ("Mo-Su 09:00-22:00", "2026-10-16T12:00", "confirmed_open"),
        ("Mo-Su 09:00-22:00", "2026-10-16T21:45", "likely_open"),
        ("Mo-Su 09:00-22:00", "2026-10-16T23:00", "likely_closed"),
        ("Mo-Fr 10:00-19:00; Sa,Su off", "2026-10-17T12:00", "likely_closed"),
        ("24/7", "2026-10-16T03:00", "confirmed_open"),
        ("Fr 18:00-02:00", "2026-10-17T01:00", "confirmed_open"),
        ("Mo-Fr 10:00-14:00,15:00-19:00", "2026-10-16T14:30", "likely_closed"),
        ("sunrise-sunset", "2026-10-16T12:00", "unknown"),
        # Real-world OSM variants from Astana data
        ("Mo-Fr 08:00-00:00, Sa-Su 09:00-00:00", "2026-10-17T08:30", "likely_closed"),
        ("Mo-Fr 08:00-00:00, Sa-Su 09:00-00:00", "2026-10-16T23:00", "confirmed_open"),
        ("Mo-Su,PH 07:30-24:00", "2026-10-16T12:00", "confirmed_open"),
        ("PH,Mo-Su 10:00-22:00", "2026-10-16T12:00", "confirmed_open"),
        ("Mo-Fr 07:14-23:00; Sa,Su,PH 08:14-23:00", "2026-10-18T08:00", "likely_closed"),
        ("12:00 - 06:00", "2026-10-16T03:00", "confirmed_open"),
        # Don't guess on ambiguous or invalid data
        ("Mo-Su 10:00-10:00", "2026-10-16T12:00", "unknown"),
        ("Mo-Su 25:00-26:00", "2026-10-16T12:00", "unknown"),
        ('Mo-Su 09:00-22:00 "по записи"', "2026-10-16T12:00", "unknown"),
        ("00:00-24:00", "2026-10-16T12:00", "confirmed_open"),
        ("Mo-Su 00:00-00:00", "2026-10-16T03:00", "confirmed_open"),
        ("Mo-Su 23:00-02:00", "2026-10-16T00:30", "confirmed_open"),
    ],
)
def test_open_status(hours, now, expected):
    status, reason = open_status(hours, now)
    assert status == expected
    assert reason


@pytest.mark.parametrize(
    "query, expected",
    [
        ("где поработать с ноутбуком", {"wifi"}),
        ("доступно для колясок", {"wheelchair"}),
        ("что посмотреть туристу", {"landmark"}),
        # False positives we explicitly avoid
        ("посмотреть фильм", set()),
        ("недорого и доступно поесть", set()),
    ],
)
def test_query_tag_hints(query, expected):
    from backend.retrieval.text import query_tag_hints

    assert query_tag_hints(query) == expected

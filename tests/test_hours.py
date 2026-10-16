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
    ],
)
def test_open_status(hours, now, expected):
    status, reason = open_status(hours, now)
    assert status == expected
    assert reason

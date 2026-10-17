"""Opening-hours confidence for OSM `opening_hours` strings.

Supports the common subset of the OSM syntax: `24/7`, rules separated by `;`,
weekday ranges/lists (`Mo-Fr`, `Sa,Su`), multiple time ranges (`10:00-14:00,15:00-19:00`),
overnight ranges (`18:00-02:00`) and `off`. Anything else yields `unknown`:
we never claim a place is open without data we can actually interpret.

Senior extensions:
- `check_visit_window()`: verifies if a POI remains open for the full duration of
  walking (`walk_min`) + visiting (`visit_min`), returning remaining minutes until closing.
- `is_late_night()`: identifies nighttime queries (22:00–05:59 Astana UTC+5) for safety rules.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from .schema import OpenStatus

# Kazakhstan switched to a single time zone UTC+5 in March 2024.
ASTANA_TZ = timezone(timedelta(hours=5))
CLOSING_SOON_MIN = 30

DAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
_DAY_RE = r"(?:Mo|Tu|We|Th|Fr|Sa|Su)"
_DAYSPEC_RE = re.compile(rf"^{_DAY_RE}(?:-{_DAY_RE})?(?:,{_DAY_RE}(?:-{_DAY_RE})?)*$")
_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})\+?$")


class _Unparseable(Exception):
    pass


def resolve_now(now_iso: str | None) -> datetime:
    """Parse `now_iso` (naive values are treated as Astana local time); default is now."""
    if not now_iso:
        return datetime.now(ASTANA_TZ)
    dt = datetime.fromisoformat(now_iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ASTANA_TZ)
    return dt.astimezone(ASTANA_TZ)


def is_late_night(now_iso: str | None = None) -> bool:
    """Return True if the target time in Astana is between 22:00 and 05:59."""
    now = resolve_now(now_iso)
    return now.hour >= 22 or now.hour < 6


def _parse_days(spec: str) -> set[int]:
    if not _DAYSPEC_RE.match(spec):
        raise _Unparseable(spec)
    days: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = (DAYS.index(d) for d in part.split("-"))
            i = a
            while True:
                days.add(i)
                if i == b:
                    break
                i = (i + 1) % 7
        else:
            days.add(DAYS.index(part))
    return days


def _parse_times(spec: str) -> list[tuple[int, int]]:
    ranges = []
    for part in spec.split(","):
        m = _TIME_RE.match(part.strip())
        if not m:
            raise _Unparseable(part)
        h1, m1, h2, m2 = map(int, m.groups())
        if h1 > 24 or h2 > 24 or m1 > 59 or m2 > 59:
            raise _Unparseable(part)
        start, end = h1 * 60 + m1, h2 * 60 + m2
        if end == start:
            if start != 0:  # "10:00-10:00" is ambiguous (24h or a typo): don't guess
                raise _Unparseable(part)
            end = 24 * 60  # "00:00-00:00" is common OSM shorthand for round-the-clock
        if end < start:  # overnight, e.g. 18:00-02:00
            end += 24 * 60
        ranges.append((start, end))
    return ranges


def _split_rules(hours: str) -> list[str]:
    # Normalize "12:00 - 06:00" -> "12:00-06:00".
    hours = re.sub(r"\s*-\s*(?=\d)", "-", hours)
    # Rules are separated by ";" and, informally, by ", " right after a time
    # ("Mo-Fr 08:00-00:00, Sa-Su 09:00-00:00").
    parts = re.split(rf";|(?<=\d),\s*(?={_DAY_RE}|PH)", hours)
    return [p.strip() for p in parts if p.strip()]


def _strip_ph(spec: str) -> str:
    """Drop public-holiday markers from a day list: we can't resolve holidays."""
    return ",".join(d for d in spec.split(",") if d != "PH")


def _parse(hours: str) -> dict[int, list[tuple[int, int]]]:
    """Return {weekday: [(start_min, end_min), ...]}; later rules override earlier ones."""
    schedule: dict[int, list[tuple[int, int]]] = {}
    for rule in _split_rules(hours):
        if rule == "24/7":
            for d in range(7):
                schedule[d] = [(0, 24 * 60)]
            continue
        tokens = rule.split(maxsplit=1)
        if "PH" in tokens[0].split(","):
            tokens[0] = _strip_ph(tokens[0])
            if not tokens[0]:  # holiday-only rule
                continue
        if _DAYSPEC_RE.match(tokens[0]):
            days = _parse_days(tokens[0])
            rest = tokens[1].strip() if len(tokens) > 1 else ""
        else:
            days = set(range(7))
            rest = rule
        if rest in ("off", "closed"):
            for d in days:
                schedule[d] = []
            continue
        times = _parse_times(rest)
        for d in days:
            schedule[d] = times
    if not schedule:
        raise _Unparseable(hours)
    return schedule


def open_status(hours: str | None, now_iso: str | None = None) -> tuple[OpenStatus, str]:
    """Return (status, human-readable reason in Russian) for the given moment."""
    if not hours or not hours.strip():
        return "unknown", "Часы работы не указаны в данных"
    try:
        schedule = _parse(hours.strip())
    except (_Unparseable, ValueError):
        return "unknown", f"Не удалось разобрать часы работы: «{hours}»"

    now = resolve_now(now_iso)
    minute = now.hour * 60 + now.minute
    today, yesterday = now.weekday(), (now.weekday() - 1) % 7

    # Check today's ranges and yesterday's overnight spill-over.
    candidates = [(s, e) for s, e in schedule.get(today, [])]
    candidates += [(s - 1440, e - 1440) for s, e in schedule.get(yesterday, []) if e > 1440]
    for start, end in candidates:
        if start <= minute < end:
            if end - start >= 1440:
                return "confirmed_open", f"Круглосуточно по расписанию в базе ({hours})"
            left = end - minute
            if left <= CLOSING_SOON_MIN:
                return "likely_open", f"Закрывается примерно через {left} мин ({hours})"
            return "confirmed_open", f"Открыто по расписанию в базе ({hours})"

    # In OSM syntax, days not listed are closed.
    return "likely_closed", f"Сейчас вне часов работы по расписанию в базе ({hours})"


def check_visit_window(
    hours: str | None,
    now_iso: str | None = None,
    walk_min: int = 0,
    visit_min: int = 45,
) -> tuple[OpenStatus, str, int | None]:
    """Check if the POI stays open during walk_min + visit_min.

    Returns (adjusted_status, detailed_reason, minutes_until_close).
    """
    base_status, base_reason = open_status(hours, now_iso)
    if base_status in ("unknown", "likely_closed") or not hours:
        return base_status, base_reason, None

    try:
        schedule = _parse(hours.strip())
    except (_Unparseable, ValueError):
        return base_status, base_reason, None

    now = resolve_now(now_iso)
    minute = now.hour * 60 + now.minute
    today, yesterday = now.weekday(), (now.weekday() - 1) % 7
    candidates = [(s, e) for s, e in schedule.get(today, [])]
    candidates += [(s - 1440, e - 1440) for s, e in schedule.get(yesterday, []) if e > 1440]

    for start, end in candidates:
        if start <= minute < end:
            if end - start >= 1440:
                return "confirmed_open", base_reason, None
            left = end - minute
            total_needed = walk_min + visit_min
            if left < walk_min + 10:
                return (
                    "likely_open",
                    f"Внимание: до закрытия {left} мин, а пешком идти ~{walk_min} мин ({hours})",
                    left,
                )
            if left < total_needed:
                return (
                    "likely_open",
                    f"Открыто, но закроется через {left} мин (на дорогу и визит нужно ~{total_needed} мин, {hours})",
                    left,
                )
            return base_status, base_reason, left

    return base_status, base_reason, None

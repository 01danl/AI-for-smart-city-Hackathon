"""Anonymous 👍 / 👎 feedback on recommended places, and its effect on ranking.

Stored per vote: place id, vote, optional reason, language and the day. Never the question
text or the user's location. Feedback nudges ranking a few positions at most, so one person
cannot sink a place; "closed" reports are surfaced on the card instead of hiding the place.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
Vote = Literal["up", "down"]
REASONS = ("closed", "far", "irrelevant", "wrong_data")

# Ranking nudges, in list positions (lower = earlier).
CLOSED_SHIFT = 3      # per "closed" report
DOWN_SHIFT = 1        # per other 👎
UP_SHIFT = -1         # per 👍
MAX_SHIFT_DOWN = 6
MAX_SHIFT_UP = -2

_lock = threading.Lock()


def feedback_path() -> Path:
    p = Path(os.getenv("FEEDBACK_PATH", "data/feedback.jsonl"))
    return p if p.is_absolute() else ROOT / p


@dataclass
class PlaceFeedback:
    up: int = 0
    down: int = 0
    closed: int = 0

    @property
    def shift(self) -> int:
        raw = self.closed * CLOSED_SHIFT + (self.down - self.closed) * DOWN_SHIFT + self.up * UP_SHIFT
        return max(MAX_SHIFT_UP, min(MAX_SHIFT_DOWN, raw))


def record_vote(poi_id: str, vote: Vote, reason: str | None = None, lang: str = "ru") -> None:
    """Append one anonymous vote."""
    if vote not in ("up", "down"):
        raise ValueError(f"unknown vote {vote!r}")
    if reason is not None and reason not in REASONS:
        raise ValueError(f"unknown reason {reason!r}")
    row = {"poi_id": poi_id, "vote": vote, "reason": reason if vote == "down" else None,
           "lang": lang, "day": date.today().isoformat()}
    path = feedback_path()
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_feedback() -> dict[str, PlaceFeedback]:
    path = feedback_path()
    out: dict[str, PlaceFeedback] = {}
    if not path.exists():
        return out
    with _lock, path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn line must not break the app
            fb = out.setdefault(str(row.get("poi_id")), PlaceFeedback())
            if row.get("vote") == "up":
                fb.up += 1
            elif row.get("vote") == "down":
                fb.down += 1
                if row.get("reason") == "closed":
                    fb.closed += 1
    return out


def summary() -> dict[str, int]:
    """Totals for the 'how it works' panel."""
    fb = load_feedback()
    up = sum(f.up for f in fb.values())
    down = sum(f.down for f in fb.values())
    return {"votes": up + down, "up": up, "down": down,
            "closed_places": sum(1 for f in fb.values() if f.closed)}


def rerank(candidates: list, top_k: int) -> list:
    """Shift candidates by feedback (stable for ties) and keep the best `top_k`."""
    fb = load_feedback()
    if not fb:
        return candidates[:top_k]
    keyed = []
    for i, c in enumerate(candidates):
        shift = fb.get(c.poi.id, PlaceFeedback()).shift
        # A place moved by N lands just after (or before) the one originally at that position.
        keyed.append((i + shift + (0.5 if shift > 0 else -0.5 if shift < 0 else 0.0), c))
    keyed.sort(key=lambda x: x[0])
    return [c for _, c in keyed[:top_k]]

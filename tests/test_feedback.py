"""Anonymous 👍 / 👎 feedback: storage, bounded ranking nudges, buttons in the UI."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import backend.generation as generation
from backend.generation import RecommendationItem, RecommendResponse, UsedData
from backend.retrieval import feedback

APP_PATH = Path(__file__).resolve().parents[1] / "frontend" / "app.py"


@pytest.fixture(autouse=True)
def votes_file(tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setenv("FEEDBACK_PATH", str(path))
    return path


def _cands(*ids):
    return [SimpleNamespace(poi=SimpleNamespace(id=i)) for i in ids]


def test_votes_are_stored_without_question_or_location(votes_file):
    feedback.record_vote("osm_node_1", "down", "closed", lang="kk")
    row = json.loads(votes_file.read_text(encoding="utf-8").strip())
    assert set(row) == {"poi_id", "vote", "reason", "lang", "day"}
    assert row["poi_id"] == "osm_node_1" and row["reason"] == "closed"


def test_reason_is_only_kept_for_downvotes(votes_file):
    feedback.record_vote("a", "up", "closed")
    assert json.loads(votes_file.read_text(encoding="utf-8"))["reason"] is None


@pytest.mark.parametrize("vote, reason", [("meh", None), ("down", "boring")])
def test_unknown_values_are_rejected(vote, reason):
    with pytest.raises(ValueError):
        feedback.record_vote("a", vote, reason)


def test_closed_reports_push_a_place_down_but_not_out():
    feedback.record_vote("a", "down", "closed")
    order = [c.poi.id for c in feedback.rerank(_cands("a", "b", "c", "d", "e"), top_k=5)]
    assert order == ["b", "c", "d", "a", "e"]


def test_shift_is_capped_so_one_person_cannot_sink_a_place():
    for _ in range(20):
        feedback.record_vote("a", "down", "closed")
    for _ in range(20):
        feedback.record_vote("z", "up")
    fb = feedback.load_feedback()
    assert fb["a"].shift == feedback.MAX_SHIFT_DOWN
    assert fb["z"].shift == feedback.MAX_SHIFT_UP


def test_no_feedback_keeps_order_and_cuts_to_top_k():
    assert [c.poi.id for c in feedback.rerank(_cands("a", "b", "c"), top_k=2)] == ["a", "b"]


def test_torn_lines_are_ignored(votes_file):
    feedback.record_vote("a", "up")
    with votes_file.open("a", encoding="utf-8") as f:
        f.write('{"poi_id": "b", "vo')
    assert feedback.summary() == {"votes": 1, "up": 1, "down": 0, "closed_places": 0}


def test_ui_vote_is_recorded_and_closed_warning_shown(monkeypatch, votes_file):
    st.cache_data.clear()
    item = RecommendationItem(
        poi_id="osm_node_42", name="Тестовое кафе", category="cafe", lat=51.128, lon=71.43,
        distance_m=200, walk_min=3, why="Рядом.", visit_min=30, what_to_do="Выпить кофе.",
        open_status="unknown", confidence_note="Часы не указаны.", source="poi_data",
    )
    monkeypatch.setattr(generation, "generate_recommendations", lambda req: RecommendResponse(
        used_data=UsedData(rounded_coords=(req.lat, req.lon)), recommendations=[item], disclaimer="—",
    ))
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.chat_input[0].set_value("кафе").run()
    assert not app.exception

    app.button(key="fb_dn_closed_1_0_osm_node_42").click().run()
    assert not app.exception
    rows = [json.loads(line) for line in votes_file.read_text(encoding="utf-8").splitlines()]
    assert rows == [{**rows[0], "poi_id": "osm_node_42", "vote": "down", "reason": "closed"}]
    assert any("Оценка учтена" in c.value for c in app.caption)

    app.chat_input[0].set_value("ещё кафе").run()
    rendered = " ".join(m.value for m in app.markdown)
    assert "сообщили, что место закрыто: 1" in rendered

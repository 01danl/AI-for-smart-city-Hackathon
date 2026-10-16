from backend.retrieval import Candidate
from backend.retrieval.stub import retrieve

BAITEREK = (51.1283, 71.4305)


def test_returns_candidates_within_distance():
    res = retrieve("тихое место с кофе и розеткой", *BAITEREK,
                   max_distance_m=1500, now_iso="2026-10-16T12:00")
    assert res and all(isinstance(c, Candidate) for c in res)
    assert all(c.distance_m <= 1500 for c in res)
    assert res[0].poi.id == "stub_cafe_quiet"


def test_category_filter_and_top_k():
    res = retrieve("погулять", *BAITEREK, max_distance_m=5000, categories=["park"], top_k=2)
    assert len(res) == 2
    assert all(c.poi.category == "park" for c in res)


def test_missing_hours_is_unknown():
    res = retrieve("парк", *BAITEREK, max_distance_m=5000, categories=["park"], top_k=10)
    lovers = next(c for c in res if c.poi.id == "stub_lovers_park")
    assert lovers.open_status == "unknown"


def test_get_poi_stub(monkeypatch):
    import importlib

    import backend.retrieval as r

    monkeypatch.setenv("RETRIEVAL_BACKEND", "stub")
    r = importlib.reload(r)
    assert r.get_poi("stub_baiterek").name == "Монумент Байтерек"
    assert r.get_poi("osm_node_does_not_exist") is None
    r.warmup()
    monkeypatch.delenv("RETRIEVAL_BACKEND")
    importlib.reload(r)

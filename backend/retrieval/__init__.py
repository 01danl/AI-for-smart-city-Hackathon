"""Public retrieval API. Import as `from backend.retrieval import retrieve, Candidate`.

Backend is chosen by RETRIEVAL_BACKEND:
  auto   (default) Chroma index if it is built (scripts/build_index.py),
                   else hybrid BM25 over data/poi.json (966 POIs) if present, else stub
  chroma           real semantic + hybrid retrieval over data/poi.json
  bm25             zero-dependency Okapi BM25 + semantic tag retrieval over data/poi.json (966 POIs)
  stub             13 hardcoded POIs, no ML dependencies
"""

from __future__ import annotations

import logging
import os

from .intent import AnchorLandmark, QueryIntent, parse_query_intent, resolve_anchor_landmark
from .route_planner import MicroRoute, RouteStop, build_micro_route, estimate_visit_minutes
from .schema import POI, Candidate, OpenStatus

log = logging.getLogger(__name__)


def _select_backend() -> str:
    mode = os.getenv("RETRIEVAL_BACKEND", "auto").lower()
    if mode == "auto":
        from .index import index_exists, poi_data_path

        if index_exists():
            mode = "chroma"
        elif poi_data_path().exists():
            mode = "bm25"
            log.info("Chroma index not built; using hybrid BM25 over 966 POIs in data/poi.json")
        else:
            mode = "stub"
            log.warning("Chroma index and poi.json not found, using retrieval stub.")
    if mode not in ("chroma", "bm25", "stub"):
        raise ValueError(f"Unknown RETRIEVAL_BACKEND={mode!r}")
    return mode


BACKEND = _select_backend()

if BACKEND == "chroma":
    from .chroma_retriever import retrieve
    from .index import load_pois as _load_pois
elif BACKEND == "bm25":
    from .hybrid_bm25 import retrieve
    from .index import load_pois as _load_pois
else:
    from .stub import STUB_POIS as _STUB_POIS
    from .stub import retrieve

    def _load_pois() -> dict[str, POI]:
        return {p.id: p for p in _STUB_POIS}


from . import feedback as _feedback  # noqa: E402

_base_retrieve = retrieve


def retrieve(query: str, lat: float, lon: float, *, top_k: int = 8, **kwargs) -> list[Candidate]:
    """Backend retrieval, nudged by anonymous 👍/👎 feedback (see backend/retrieval/feedback.py).

    A few extra candidates are fetched so a demoted place can be replaced, not just reordered.
    """
    pool = _base_retrieve(query, lat, lon, top_k=top_k + _feedback.MAX_SHIFT_DOWN, **kwargs)
    return _feedback.rerank(pool, top_k)


def get_poi(poi_id: str) -> POI | None:
    """Look up a POI by id, e.g. to verify the LLM only returned ids from retrieval results."""
    return _load_pois().get(poi_id)


def warmup() -> None:
    """Load the embedding model and index up front (first query is otherwise ~10 s slower).

    Call once at API/UI startup.
    """
    retrieve("прогулка", 51.1283, 71.4305, top_k=1)


__all__ = [
    "BACKEND",
    "POI",
    "Candidate",
    "OpenStatus",
    "QueryIntent",
    "AnchorLandmark",
    "MicroRoute",
    "RouteStop",
    "get_poi",
    "retrieve",
    "warmup",
    "parse_query_intent",
    "resolve_anchor_landmark",
    "build_micro_route",
    "estimate_visit_minutes",
]

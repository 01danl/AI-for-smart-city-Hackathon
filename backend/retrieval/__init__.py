"""Public retrieval API. Import as `from backend.retrieval import retrieve, Candidate`.

Backend is chosen by RETRIEVAL_BACKEND:
  auto   (default) Chroma index if it is built (scripts/build_index.py), otherwise the stub
  chroma           real semantic retrieval over data/poi.json
  stub             13 hardcoded POIs, no ML dependencies
"""

from __future__ import annotations

import logging
import os

from .schema import POI, Candidate, OpenStatus

log = logging.getLogger(__name__)


def _select_backend() -> str:
    mode = os.getenv("RETRIEVAL_BACKEND", "auto").lower()
    if mode == "auto":
        from .index import index_exists

        mode = "chroma" if index_exists() else "stub"
        if mode == "stub":
            log.warning("Chroma index not found, using retrieval stub. Run scripts/build_index.py")
    if mode not in ("chroma", "stub"):
        raise ValueError(f"Unknown RETRIEVAL_BACKEND={mode!r}")
    return mode


BACKEND = _select_backend()

if BACKEND == "chroma":
    from .chroma_retriever import retrieve
    from .index import load_pois as _load_pois
else:
    from .stub import STUB_POIS as _STUB_POIS
    from .stub import retrieve

    def _load_pois() -> dict[str, POI]:
        return {p.id: p for p in _STUB_POIS}


def get_poi(poi_id: str) -> POI | None:
    """Look up a POI by id, e.g. to verify the LLM only returned ids from retrieval results."""
    return _load_pois().get(poi_id)


def warmup() -> None:
    """Load the embedding model and index up front (first query is otherwise ~10 s slower).

    Call once at API/UI startup.
    """
    retrieve("прогулка", 51.1283, 71.4305, top_k=1)


__all__ = ["BACKEND", "POI", "Candidate", "OpenStatus", "get_poi", "retrieve", "warmup"]

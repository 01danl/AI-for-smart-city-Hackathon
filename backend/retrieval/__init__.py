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
else:
    from .stub import retrieve

__all__ = ["BACKEND", "POI", "Candidate", "OpenStatus", "retrieve"]

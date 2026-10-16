"""FastAPI service for Astana Walk AI Assistant (`POST /recommend`, `GET /health`, `GET /presets`).

Run with:
    uvicorn backend.api.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.generation import (
    OFFICIAL_PRESETS,
    RecommendRequest,
    RecommendResponse,
    generate_recommendations,
)
from backend.retrieval import BACKEND, warmup
from backend.retrieval.geo import round_coords
from backend.retrieval.index import load_pois

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        warmup()
    except Exception as exc:
        log.warning("Warmup warning: %s", exc)
    yield


app = FastAPI(
    title="Astana Walk AI — Smart City RAG Advisor",
    description=(
        "Real-time location-aware RAG recommender for Astana POIs with strict "
        "`poi_data` vs `generated_idea` attribution, opening-hours confidence, "
        "100m coordinate privacy rounding, and 3-stop micro-route planning."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict:
    pois = load_pois()
    return {
        "status": "ok",
        "retrieval_backend": BACKEND,
        "poi_count": len(pois),
        "llm_configured": bool(os.getenv("OPENROUTER_API_KEY", "").strip()),
        "openrouter_model": os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
        "privacy_mode": "rounded_100m_no_raw_coords_in_llm",
    }


@app.get("/presets")
def get_presets() -> dict:
    """Official hackathon evaluation scenarios for 1-click testing."""
    return {"presets": OFFICIAL_PRESETS}


@app.get("/pois/stats")
def get_poi_stats() -> dict:
    """Dataset summary statistics for UI transparency inspector."""
    pois = list(load_pois().values())
    cats = Counter(p.category for p in pois)
    with_hours = sum(1 for p in pois if p.opening_hours)
    return {
        "total_pois": len(pois),
        "with_opening_hours": with_hours,
        "opening_hours_coverage_pct": round(100.0 * with_hours / max(1, len(pois)), 1),
        "categories": dict(cats.most_common()),
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    """Generate 1–3 grounded POI recommendations + sequential micro-route."""
    # Log only privacy-rounded coordinates (~100 m), never raw user GPS
    r_lat, r_lon = round_coords(req.lat, req.lon)
    log.info(
        "POST /recommend query=%r rounded_coords=(%.3f, %.3f) radius=%dm lang=%s",
        req.query,
        r_lat,
        r_lon,
        req.max_distance_m,
        req.lang,
    )
    return generate_recommendations(req)

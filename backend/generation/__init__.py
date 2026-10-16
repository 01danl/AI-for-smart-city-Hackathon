"""Public generation package API (Role B)."""

from __future__ import annotations

from .client import DEFAULT_BASE_URL, OpenAI, complete, get_client
from .prompts import SYSTEM_PROMPT, build_rag_messages
from .recommender import (
    OFFICIAL_PRESETS,
    RecommendationItem,
    RecommendRequest,
    RecommendResponse,
    UsedData,
    generate_recommendations,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "OpenAI",
    "get_client",
    "complete",
    "SYSTEM_PROMPT",
    "build_rag_messages",
    "UsedData",
    "RecommendationItem",
    "RecommendRequest",
    "RecommendResponse",
    "generate_recommendations",
    "OFFICIAL_PRESETS",
]

"""Chroma index over data/poi.json with a local multilingual embedding model."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from .schema import POI
from .text import poi_document

ROOT = Path(__file__).resolve().parents[2]
COLLECTION = "poi"


def _path(env: str, default: str) -> Path:
    p = Path(os.getenv(env, default))
    return p if p.is_absolute() else ROOT / p


def poi_data_path() -> Path:
    return _path("POI_DATA_PATH", "data/poi.json")


def chroma_dir() -> Path:
    return _path("CHROMA_DIR", "data/chroma")


def embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def index_exists() -> bool:
    return (chroma_dir() / "chroma.sqlite3").exists()


def _parse_poi(data: dict) -> POI:
    validator = getattr(POI, "model_validate", None)
    if validator is not None:
        return validator(data)
    return POI.parse_obj(data)


@lru_cache(maxsize=1)
def load_pois() -> dict[str, POI]:
    raw = json.loads(poi_data_path().read_text(encoding="utf-8"))
    return {p["id"]: _parse_poi(p) for p in raw}


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(embedding_model_name())


def embed(texts: list[str]) -> list[list[float]]:
    vecs = get_model().encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return vecs.tolist()


def _client():
    import chromadb
    from chromadb.config import Settings

    # Telemetry off: nothing about queries or usage leaves the machine.
    return chromadb.PersistentClient(path=str(chroma_dir()), settings=Settings(anonymized_telemetry=False))


@lru_cache(maxsize=1)
def get_collection():
    return _client().get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def build_index(pois: list[POI]) -> int:
    """(Re)build the collection from scratch. Returns number of indexed POIs."""
    client = _client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # collection may not exist yet
        pass
    get_collection.cache_clear()
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    docs = [poi_document(p) for p in pois]
    vectors = embed(docs)
    batch = 500
    for i in range(0, len(pois), batch):
        chunk = pois[i:i + batch]
        col.add(
            ids=[p.id for p in chunk],
            embeddings=vectors[i:i + batch],
            documents=docs[i:i + batch],
            metadatas=[{"lat": p.lat, "lon": p.lon, "category": p.category} for p in chunk],
        )
    return col.count()

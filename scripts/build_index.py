"""Build the Chroma vector index from data/poi.json.

Usage:
    python scripts/build_index.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from backend.retrieval.index import (  # noqa: E402
    build_index, chroma_dir, embedding_model_name, load_pois,
)


def main() -> None:
    pois = list(load_pois().values())
    print(f"model={embedding_model_name()} pois={len(pois)}")
    t0 = time.time()
    n = build_index(pois)
    print(f"indexed={n} in {time.time() - t0:.1f}s -> {chroma_dir().relative_to(ROOT)}")


if __name__ == "__main__":
    main()

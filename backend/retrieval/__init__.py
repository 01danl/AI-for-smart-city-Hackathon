"""Public retrieval API. Import as `from backend.retrieval import retrieve, Candidate`.

Until the Chroma index is ready, `retrieve` is served by the stub.
"""

from .schema import POI, Candidate, OpenStatus
from .stub import retrieve

__all__ = ["POI", "Candidate", "OpenStatus", "retrieve"]

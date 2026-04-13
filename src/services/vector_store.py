"""
Vector Knowledge Base for analogous fraud ring retrieval.

Two backends:
  - LocalVectorStore  : numpy cosine similarity, persisted as .npz file.
                        Zero external dependencies. Suitable for ≤ 100K rings.
  - PineconeVectorStore: production-grade ANN search via Pinecone.
                         Requires PINECONE_API_KEY in .env.

Factory function `get_vector_store()` returns the configured backend.

Usage
─────
    from phase3_rag.vector_store import get_vector_store
    vs = get_vector_store()

    # Index all fraud rings (run once after Phase 1 load)
    vs.add("RING-001", embedding, {"ring_id": "RING-001", "status": "Confirmed"})
    vs.save()

    # Retrieve top-K analogous rings at query time
    results = vs.search(query_embedding, top_k=3)
    # → [{"id": "RING-001", "score": 0.92, "metadata": {...}}, ...]
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import config as C

log = logging.getLogger(__name__)


# ── Abstract interface ────────────────────────────────────────────────
class VectorStore(ABC):
    @abstractmethod
    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None: ...

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int = C.TOP_K_ANALOGOUS
               ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def save(self, path: Optional[str] = None) -> None: ...

    @abstractmethod
    def load(self, path: Optional[str] = None) -> "VectorStore": ...

    @abstractmethod
    def __len__(self) -> int: ...


# ── Local numpy store ─────────────────────────────────────────────────
class LocalVectorStore(VectorStore):
    """
    In-memory vector store backed by numpy arrays.

    Vectors are L2-normalised on insert; search uses cosine similarity
    (equivalent to dot product on normalised vectors).

    Persisted as a .npz file:
      - 'vectors'  : float32 (N, dim)
      - 'ids'      : str array (N,)
      - 'metadata' : JSON-encoded string per entry
    """

    def __init__(self):
        self._ids:      list[str]             = []
        self._vectors:  list[np.ndarray]      = []
        self._metadata: list[Dict[str, Any]]  = []

    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        norm = np.linalg.norm(vector)
        self._ids.append(id)
        self._vectors.append(vector / norm if norm > 0 else vector)
        self._metadata.append(metadata)

    def search(self, query: np.ndarray, top_k: int = C.TOP_K_ANALOGOUS
               ) -> List[Dict[str, Any]]:
        if not self._vectors:
            return []

        matrix = np.vstack(self._vectors)          # (N, dim)
        q = query.reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        scores = cosine_similarity(q, matrix)[0]   # (N,)
        top_idx = np.argsort(-scores)[:top_k]

        return [
            {
                "id":       self._ids[i],
                "score":    float(scores[i]),
                "metadata": self._metadata[i],
            }
            for i in top_idx
            if scores[i] > 0
        ]

    def save(self, path: Optional[str] = None) -> None:
        path = path or C.VECTOR_STORE_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if not self._vectors:
            log.warning("Vector store is empty — nothing to save.")
            return

        np.savez_compressed(
            path,
            vectors=np.vstack(self._vectors).astype(np.float32),
            ids=np.array(self._ids, dtype=object),
            metadata=np.array(
                [json.dumps(m) for m in self._metadata], dtype=object
            ),
        )
        log.info("Saved %d vectors to %s", len(self._ids), path)

    def load(self, path: Optional[str] = None) -> "LocalVectorStore":
        path = path or C.VECTOR_STORE_PATH
        if not os.path.exists(path):
            log.info("No vector store file at %s — starting empty.", path)
            return self

        data = np.load(path, allow_pickle=True)
        self._vectors  = [data["vectors"][i] for i in range(len(data["ids"]))]
        self._ids      = list(data["ids"])
        self._metadata = [json.loads(m) for m in data["metadata"]]
        log.info("Loaded %d vectors from %s", len(self._ids), path)
        return self

    def __len__(self) -> int:
        return len(self._ids)


# ── Pinecone adapter ──────────────────────────────────────────────────
class PineconeVectorStore(VectorStore):
    """
    Pinecone-backed vector store for production deployment.

    Requires:
      PINECONE_API_KEY in .env
      PINECONE_INDEX   in .env (default: fraud-rings)
      pip install pinecone-client

    The index must be created manually in the Pinecone console with
    dimension matching EMBEDDING_DIM (384 for all-MiniLM-L6-v2).
    """

    def __init__(self):
        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise ImportError(
                "pinecone-client is required for Pinecone backend. "
                "Install with: pip install pinecone-client"
            ) from exc

        if not C.PINECONE_API_KEY:
            raise EnvironmentError("PINECONE_API_KEY must be set in .env")

        pc = Pinecone(api_key=C.PINECONE_API_KEY)
        self._index = pc.Index(C.PINECONE_INDEX)
        log.info("Connected to Pinecone index: %s", C.PINECONE_INDEX)

    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        # Pinecone metadata values must be str/int/float/bool
        safe_meta = {
            k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
            for k, v in metadata.items()
        }
        self._index.upsert(vectors=[(id, vector.tolist(), safe_meta)])

    def search(self, query: np.ndarray, top_k: int = C.TOP_K_ANALOGOUS
               ) -> List[Dict[str, Any]]:
        result = self._index.query(
            vector=query.tolist(), top_k=top_k, include_metadata=True
        )
        return [
            {
                "id":       match["id"],
                "score":    match["score"],
                "metadata": match.get("metadata", {}),
            }
            for match in result.get("matches", [])
        ]

    def save(self, path: Optional[str] = None) -> None:
        pass  # Pinecone persists automatically

    def load(self, path: Optional[str] = None) -> "PineconeVectorStore":
        return self  # Pinecone is always loaded

    def __len__(self) -> int:
        stats = self._index.describe_index_stats()
        return stats.get("total_vector_count", 0)


# ── Factory ───────────────────────────────────────────────────────────
def get_vector_store(load_existing: bool = True) -> VectorStore:
    """
    Return a configured VectorStore based on VECTOR_STORE_BACKEND env var.

    Args:
        load_existing: if True (default), load persisted vectors from disk
                       (only applies to LocalVectorStore).
    """
    backend = C.VECTOR_STORE_BACKEND.lower()

    if backend == "pinecone":
        return PineconeVectorStore()

    # Default: local numpy store
    vs = LocalVectorStore()
    if load_existing:
        vs.load()
    return vs

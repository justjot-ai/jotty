"""
EmbeddingService - Vector Embeddings for Semantic Retrieval
============================================================

Production-grade embedding service using sentence-transformers (all-MiniLM-L6-v2).
384-dim vectors, normalized, runs locally — no API calls needed.

Lazy-loads model on first use to avoid startup penalty.
Thread-safe singleton.

Used by LearningService for:
  - Episode similarity search (replaces TF-IDF)
  - Distilled lesson retrieval
  - Cross-domain transfer matching

Author: Jotty Team
Date: February 2026
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384


class EmbeddingService:
    """
    Thread-safe singleton for embedding text into dense vectors.

    Uses all-MiniLM-L6-v2 (22M params, 384-dim, ~14k sentences/sec on CPU).
    Falls back gracefully if sentence-transformers is unavailable.
    """

    _instance: Optional[EmbeddingService] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model: Optional[SentenceTransformer] = None
        self._available: Optional[bool] = None
        self._model_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> EmbeddingService:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure_model(self) -> bool:
        """Load model on first use. Returns True if model is available."""
        if self._available is not None:
            return self._available

        with self._model_lock:
            if self._available is not None:
                return self._available
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(_MODEL_NAME)
                self._available = True
                logger.info("Loaded embedding model: %s (%d-dim)", _MODEL_NAME, _EMBEDDING_DIM)
            except Exception as exc:
                logger.warning("Embedding model unavailable (falling back to TF-IDF): %s", exc)
                self._available = False
        return self._available

    @property
    def available(self) -> bool:
        return self._ensure_model()

    @property
    def dim(self) -> int:
        return _EMBEDDING_DIM

    def embed(self, text: str) -> Optional[np.ndarray]:
        """Embed a single text into a normalized 384-dim vector."""
        if not self._ensure_model() or not text.strip():
            return None
        assert self._model is not None
        vec = self._model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vec, dtype=np.float32)

    def embed_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Embed multiple texts. Returns (N, 384) normalized array."""
        if not self._ensure_model() or not texts:
            return None
        assert self._model is not None
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two normalized vectors (= dot product)."""
        return float(np.dot(a, b))

    @staticmethod
    def cosine_similarities(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Cosine similarities between query vector and matrix of vectors."""
        return matrix @ query

    @staticmethod
    def serialize(vec: np.ndarray) -> bytes:
        """Serialize numpy vector to bytes for SQLite BLOB storage."""
        return vec.astype(np.float32).tobytes()

    @staticmethod
    def deserialize(blob: bytes) -> np.ndarray:
        """Deserialize bytes back to numpy vector."""
        return np.frombuffer(blob, dtype=np.float32).copy()


def get_embedding_service() -> EmbeddingService:
    """Get the singleton EmbeddingService."""
    return EmbeddingService.get_instance()

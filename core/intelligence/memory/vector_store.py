"""
VectorStore - ChromaDB-backed vector store with numpy fallback.
================================================================

Thin wrapper over EmbeddingService for document chunk storage and retrieval.
Thread-safe singleton.

Used by:
  - SwarmTemplate.search_documents() for RAG in any swarm
  - DocumentIngestionPipeline for storing extracted content
  - OlympiadLearningSwarm research phase

Author: Jotty Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class DocumentChunk:
    """A chunk of document content with source metadata."""

    chunk_id: str
    content: str
    source: str  # URL or file path
    source_type: str  # "webpage", "pdf", "arxiv", "text"
    metadata: Dict[str, Any] = field(default_factory=dict)  # subject, grade, topic, etc.


@dataclass
class RetrievalResult:
    """A search result with similarity score."""

    chunk: DocumentChunk
    similarity: float


# =============================================================================
# VECTOR STORE
# =============================================================================


class VectorStore:
    """ChromaDB-backed vector store with numpy fallback.

    Thread-safe singleton. Reuses EmbeddingService for embeddings.

    Usage:
        store = VectorStore.get_instance()
        store.add_chunks([DocumentChunk(...), ...])
        results = store.search("fractions for 5th grade", top_k=5)
    """

    _instance: Optional[VectorStore] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._embedder = None  # Lazy
        self._chroma_collection = None  # Lazy
        self._use_chroma = False
        self._init_lock = threading.Lock()
        self._initialized = False

        # Numpy fallback storage
        self._chunks: List[DocumentChunk] = []
        self._vectors: Optional[np.ndarray] = None  # (N, dim)
        self._store_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> VectorStore:
        """Thread-safe singleton."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_initialized(self) -> None:
        """Lazy init: embedder + optional ChromaDB."""
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return

            from Jotty.core.intelligence.learning.embeddings import EmbeddingService

            self._embedder = EmbeddingService.get_instance()

            # Try ChromaDB
            try:
                import chromadb

                client = chromadb.Client()
                self._chroma_collection = client.get_or_create_collection(
                    name="jotty_documents",
                    metadata={"hnsw:space": "cosine"},
                )
                self._use_chroma = True
                logger.info("VectorStore: using ChromaDB backend")
            except Exception:
                self._use_chroma = False
                logger.info("VectorStore: using numpy fallback (chromadb not available)")

            self._initialized = True

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Add document chunks to the store.

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        self._ensure_initialized()
        assert self._embedder is not None

        texts = [c.content for c in chunks]
        vectors = self._embedder.embed_batch(texts)
        if vectors is None:
            logger.warning("VectorStore: embedding failed, chunks not added")
            return 0

        if self._use_chroma:
            return self._add_chroma(chunks, vectors)
        else:
            return self._add_numpy(chunks, vectors)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Search for similar chunks.

        Args:
            query: Search query text.
            top_k: Maximum results.
            filters: Optional metadata filters (e.g. {"subject": "mathematics"}).

        Returns:
            List of RetrievalResult sorted by similarity (highest first).
        """
        self._ensure_initialized()
        assert self._embedder is not None

        query_vec = self._embedder.embed(query)
        if query_vec is None:
            return []

        if self._use_chroma:
            return self._search_chroma(query_vec, top_k, filters)
        else:
            return self._search_numpy(query_vec, top_k, filters)

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source.

        Returns:
            Number of chunks deleted.
        """
        self._ensure_initialized()

        if self._use_chroma:
            return self._delete_chroma(source)
        else:
            return self._delete_numpy(source)

    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        self._ensure_initialized()

        if self._use_chroma:
            count = self._chroma_collection.count() if self._chroma_collection else 0
            return {"backend": "chromadb", "total_chunks": count}
        else:
            with self._store_lock:
                return {"backend": "numpy", "total_chunks": len(self._chunks)}

    # =========================================================================
    # CHROMA BACKEND
    # =========================================================================

    def _add_chroma(self, chunks: List[DocumentChunk], vectors: np.ndarray) -> int:
        """Add chunks to ChromaDB."""
        try:
            ids = [c.chunk_id for c in chunks]
            documents = [c.content for c in chunks]
            metadatas = [
                {
                    "source": c.source,
                    "source_type": c.source_type,
                    **{k: str(v) for k, v in c.metadata.items()},
                }
                for c in chunks
            ]
            embeddings = vectors.tolist()

            self._chroma_collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            return len(chunks)
        except Exception as e:
            logger.error(f"VectorStore ChromaDB add failed: {e}")
            return 0

    def _search_chroma(
        self,
        query_vec: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[RetrievalResult]:
        """Search ChromaDB."""
        try:
            where = None
            if filters:
                # ChromaDB where clause
                conditions = [{k: str(v)} for k, v in filters.items()]
                if len(conditions) == 1:
                    where = conditions[0]
                elif len(conditions) > 1:
                    where = {"$and": conditions}

            results = self._chroma_collection.query(
                query_embeddings=[query_vec.tolist()],
                n_results=top_k,
                where=where,
            )

            retrieval_results = []
            if results and results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    content = results["documents"][0][i] if results["documents"] else ""
                    distance = results["distances"][0][i] if results.get("distances") else 0.0
                    similarity = 1.0 - distance  # ChromaDB cosine distance

                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        content=content,
                        source=meta.get("source", ""),
                        source_type=meta.get("source_type", ""),
                        metadata={
                            k: v for k, v in meta.items() if k not in ("source", "source_type")
                        },
                    )
                    retrieval_results.append(RetrievalResult(chunk=chunk, similarity=similarity))

            return retrieval_results
        except Exception as e:
            logger.error(f"VectorStore ChromaDB search failed: {e}")
            return []

    def _delete_chroma(self, source: str) -> int:
        """Delete chunks from ChromaDB by source."""
        try:
            # Get IDs matching source
            results = self._chroma_collection.get(where={"source": source})
            if results and results["ids"]:
                self._chroma_collection.delete(ids=results["ids"])
                return len(results["ids"])
            return 0
        except Exception as e:
            logger.error(f"VectorStore ChromaDB delete failed: {e}")
            return 0

    # =========================================================================
    # NUMPY FALLBACK
    # =========================================================================

    def _add_numpy(self, chunks: List[DocumentChunk], vectors: np.ndarray) -> int:
        """Add chunks to numpy store."""
        with self._store_lock:
            self._chunks.extend(chunks)
            if self._vectors is None:
                self._vectors = vectors
            else:
                self._vectors = np.vstack([self._vectors, vectors])
        return len(chunks)

    def _search_numpy(
        self,
        query_vec: np.ndarray,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[RetrievalResult]:
        """Search numpy store."""
        with self._store_lock:
            if not self._chunks or self._vectors is None:
                return []

            # Compute similarities
            similarities = self._vectors @ query_vec  # Normalized vectors -> cosine sim

            # Apply filters
            mask = np.ones(len(self._chunks), dtype=bool)
            if filters:
                for key, value in filters.items():
                    for i, chunk in enumerate(self._chunks):
                        if chunk.metadata.get(key) != value:
                            mask[i] = False

            # Zero out filtered entries
            filtered_sims = similarities * mask

            # Top-k
            k = min(top_k, int(mask.sum()))
            if k == 0:
                return []

            top_indices = np.argsort(filtered_sims)[-k:][::-1]

            results = []
            for idx in top_indices:
                sim = float(filtered_sims[idx])
                if sim > 0:
                    results.append(RetrievalResult(chunk=self._chunks[idx], similarity=sim))
            return results

    def _delete_numpy(self, source: str) -> int:
        """Delete chunks from numpy store by source."""
        with self._store_lock:
            original_count = len(self._chunks)
            keep_mask = [c.source != source for c in self._chunks]
            self._chunks = [c for c, keep in zip(self._chunks, keep_mask) if keep]
            if self._vectors is not None and any(not k for k in keep_mask):
                keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
                if keep_indices:
                    self._vectors = self._vectors[keep_indices]
                else:
                    self._vectors = None
            return original_count - len(self._chunks)


# =============================================================================
# HELPER
# =============================================================================


def make_chunk_id(source: str, index: int) -> str:
    """Generate a deterministic chunk ID from source + index."""
    h = hashlib.md5(f"{source}:{index}".encode()).hexdigest()[:12]
    return f"chunk_{h}"


def get_vector_store() -> VectorStore:
    """Get the singleton VectorStore."""
    return VectorStore.get_instance()

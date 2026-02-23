"""Tests for VectorStore and DocumentIngestionPipeline.

All tests use mocked embeddings — no real model loading, no network.
"""

import numpy as np
import pytest

from Jotty.core.intelligence.memory.vector_store import (
    DocumentChunk,
    RetrievalResult,
    VectorStore,
    make_chunk_id,
)


# =============================================================================
# FIXTURES
# =============================================================================


class FakeEmbedder:
    """Mock EmbeddingService that returns deterministic vectors."""

    available = True
    dim = 384

    def embed(self, text: str) -> np.ndarray:
        # Deterministic: hash text to seed
        seed = sum(ord(c) for c in text) % 10000
        rng = np.random.RandomState(seed)
        vec = rng.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec

    def embed_batch(self, texts: list) -> np.ndarray:
        return np.array([self.embed(t) for t in texts], dtype=np.float32)


@pytest.fixture
def store():
    """Create a fresh VectorStore with fake embedder (numpy backend)."""
    s = VectorStore()
    s._embedder = FakeEmbedder()
    s._use_chroma = False
    s._initialized = True
    return s


@pytest.fixture
def sample_chunks():
    """Create sample document chunks."""
    return [
        DocumentChunk(
            chunk_id="c1",
            content="Fractions represent parts of a whole. A fraction has a numerator and denominator.",
            source="https://example.com/fractions",
            source_type="webpage",
            metadata={"subject": "mathematics", "topic": "fractions"},
        ),
        DocumentChunk(
            chunk_id="c2",
            content="Addition of fractions requires a common denominator. Find LCD first.",
            source="https://example.com/fractions",
            source_type="webpage",
            metadata={"subject": "mathematics", "topic": "fractions"},
        ),
        DocumentChunk(
            chunk_id="c3",
            content="Newton's laws describe the relationship between force and motion.",
            source="https://example.com/physics",
            source_type="webpage",
            metadata={"subject": "physics", "topic": "mechanics"},
        ),
    ]


# =============================================================================
# VECTORSTORE TESTS
# =============================================================================


class TestVectorStore:
    """Tests for VectorStore core operations."""

    @pytest.mark.unit
    def test_add_chunks(self, store, sample_chunks):
        """add_chunks stores chunks and returns count."""
        count = store.add_chunks(sample_chunks)
        assert count == 3
        assert store.stats()["total_chunks"] == 3

    @pytest.mark.unit
    def test_add_empty(self, store):
        """add_chunks with empty list returns 0."""
        assert store.add_chunks([]) == 0

    @pytest.mark.unit
    def test_search_returns_results(self, store, sample_chunks):
        """search returns relevant results sorted by similarity."""
        store.add_chunks(sample_chunks)
        results = store.search("fractions numerator denominator", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        # Similarities should be > 0
        assert all(r.similarity > 0 for r in results)

    @pytest.mark.unit
    def test_search_empty_store(self, store):
        """search on empty store returns empty list."""
        results = store.search("anything")
        assert results == []

    @pytest.mark.unit
    def test_search_with_filters(self, store, sample_chunks):
        """search with metadata filters narrows results."""
        store.add_chunks(sample_chunks)
        # Filter to physics only
        results = store.search("force and motion", top_k=5, filters={"subject": "physics"})
        # Should only return physics chunks
        assert all(
            r.chunk.metadata.get("subject") == "physics" for r in results if r.similarity > 0
        )

    @pytest.mark.unit
    def test_search_top_k(self, store, sample_chunks):
        """search respects top_k limit."""
        store.add_chunks(sample_chunks)
        results = store.search("fractions", top_k=1)
        assert len(results) <= 1

    @pytest.mark.unit
    def test_delete_by_source(self, store, sample_chunks):
        """delete_by_source removes chunks from that source."""
        store.add_chunks(sample_chunks)
        assert store.stats()["total_chunks"] == 3

        deleted = store.delete_by_source("https://example.com/fractions")
        assert deleted == 2
        assert store.stats()["total_chunks"] == 1

    @pytest.mark.unit
    def test_delete_nonexistent_source(self, store, sample_chunks):
        """delete_by_source returns 0 for unknown source."""
        store.add_chunks(sample_chunks)
        deleted = store.delete_by_source("https://nonexistent.com")
        assert deleted == 0
        assert store.stats()["total_chunks"] == 3

    @pytest.mark.unit
    def test_stats(self, store, sample_chunks):
        """stats returns correct backend and count."""
        stats = store.stats()
        assert stats["backend"] == "numpy"
        assert stats["total_chunks"] == 0

        store.add_chunks(sample_chunks)
        stats = store.stats()
        assert stats["total_chunks"] == 3

    @pytest.mark.unit
    def test_numpy_fallback(self, store, sample_chunks):
        """numpy backend works when chromadb is not available."""
        assert store._use_chroma is False
        store.add_chunks(sample_chunks)
        results = store.search("fractions", top_k=5)
        assert len(results) > 0

    @pytest.mark.unit
    def test_embedding_failure_returns_zero(self):
        """add_chunks returns 0 if embedder fails."""
        s = VectorStore()
        s._initialized = True

        class FailEmbedder:
            available = False
            dim = 384

            def embed_batch(self, texts):
                return None

        s._embedder = FailEmbedder()
        chunks = [DocumentChunk("c1", "test", "src", "text")]
        assert s.add_chunks(chunks) == 0


class TestMakeChunkId:
    """Tests for chunk ID generation."""

    @pytest.mark.unit
    def test_deterministic(self):
        """Same inputs produce same ID."""
        id1 = make_chunk_id("http://example.com", 0)
        id2 = make_chunk_id("http://example.com", 0)
        assert id1 == id2

    @pytest.mark.unit
    def test_different_inputs(self):
        """Different inputs produce different IDs."""
        id1 = make_chunk_id("http://example.com", 0)
        id2 = make_chunk_id("http://example.com", 1)
        assert id1 != id2

    @pytest.mark.unit
    def test_prefix(self):
        """IDs have chunk_ prefix."""
        assert make_chunk_id("src", 0).startswith("chunk_")


# =============================================================================
# DOCUMENT INGESTION TESTS
# =============================================================================


class TestDocumentIngestionPipeline:
    """Tests for DocumentIngestionPipeline (no network, mocked store)."""

    @pytest.mark.unit
    def test_ingest_text(self, store):
        """ingest_text chunks and stores text."""
        from Jotty.core.intelligence.memory.document_ingestion import DocumentIngestionPipeline

        pipeline = DocumentIngestionPipeline(vector_store=store, chunk_size=100, chunk_overlap=20)
        text = "Fractions are parts of a whole. " * 10  # ~300 chars
        count = pipeline.ingest_text(text, source="textbook", metadata={"topic": "fractions"})
        assert count > 0
        assert store.stats()["total_chunks"] > 0

    @pytest.mark.unit
    def test_ingest_empty_text(self, store):
        """ingest_text with empty text returns 0."""
        from Jotty.core.intelligence.memory.document_ingestion import DocumentIngestionPipeline

        pipeline = DocumentIngestionPipeline(vector_store=store)
        assert pipeline.ingest_text("") == 0
        assert pipeline.ingest_text("   ") == 0

    @pytest.mark.unit
    def test_chunk_text_short(self, store):
        """Short text becomes single chunk."""
        from Jotty.core.intelligence.memory.document_ingestion import DocumentIngestionPipeline

        pipeline = DocumentIngestionPipeline(vector_store=store, chunk_size=1000)
        chunks = pipeline._chunk_text("Short text.")
        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    @pytest.mark.unit
    def test_chunk_text_long(self, store):
        """Long text is split into multiple chunks."""
        from Jotty.core.intelligence.memory.document_ingestion import DocumentIngestionPipeline

        pipeline = DocumentIngestionPipeline(vector_store=store, chunk_size=100, chunk_overlap=20)
        text = "This is a sentence about math. " * 20  # ~600 chars
        chunks = pipeline._chunk_text(text)
        assert len(chunks) > 1
        # All chunks should be non-empty
        assert all(len(c.strip()) > 0 for c in chunks)

    @pytest.mark.unit
    def test_chunk_text_sentence_boundary(self, store):
        """Chunks break at sentence boundaries."""
        from Jotty.core.intelligence.memory.document_ingestion import DocumentIngestionPipeline

        pipeline = DocumentIngestionPipeline(vector_store=store, chunk_size=50, chunk_overlap=10)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = pipeline._chunk_text(text)
        # Each chunk should end at or near a sentence boundary
        for chunk in chunks[:-1]:  # Last chunk might not end at boundary
            assert chunk.rstrip().endswith((".")) or len(chunk) <= 50

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ingest_url_no_network(self, store):
        """ingest_url returns 0 when scraping fails (no network)."""
        from Jotty.core.intelligence.memory.document_ingestion import DocumentIngestionPipeline

        pipeline = DocumentIngestionPipeline(vector_store=store)
        count = await pipeline.ingest_url("https://nonexistent.invalid/test")
        assert count == 0

    @pytest.mark.unit
    def test_ingest_text_with_metadata(self, store):
        """Metadata is preserved on chunks."""
        from Jotty.core.intelligence.memory.document_ingestion import DocumentIngestionPipeline

        pipeline = DocumentIngestionPipeline(vector_store=store, chunk_size=100)
        text = "Mathematics content. " * 10
        meta = {"subject": "mathematics", "grade": "5th"}
        pipeline.ingest_text(text, source="test", metadata=meta)

        results = store.search("mathematics", top_k=1)
        assert len(results) > 0
        assert results[0].chunk.metadata.get("subject") == "mathematics"

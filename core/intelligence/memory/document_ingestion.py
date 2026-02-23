"""
DocumentIngestionPipeline - Download, extract, chunk, embed, store.
====================================================================

Thin pipeline that delegates to existing skills for web scraping
and uses VectorStore for storage. No new extraction logic.

Usage:
    from Jotty.core.intelligence.memory.document_ingestion import DocumentIngestionPipeline

    pipeline = DocumentIngestionPipeline()
    count = await pipeline.ingest_url("https://example.com/fractions", metadata={"subject": "math"})
    count = pipeline.ingest_text("Fractions are...", source="textbook", metadata={"topic": "fractions"})

Author: Jotty Team
Date: February 2026
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .vector_store import DocumentChunk, VectorStore, get_vector_store, make_chunk_id

logger = logging.getLogger(__name__)

# Sentence-ending pattern for boundary-aware chunking
_SENTENCE_END = re.compile(r"[.!?]\s+")

# Default chunk size in characters (~200 tokens)
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


class DocumentIngestionPipeline:
    """Download -> extract -> chunk -> embed -> store.

    Delegates web scraping to existing skills. Chunks text at sentence
    boundaries. Stores in VectorStore.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._store = vector_store or get_vector_store()
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def ingest_url(
        self,
        url: str,
        source_type: str = "webpage",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Download content from URL, chunk, and store.

        Args:
            url: URL to scrape.
            source_type: Type of source ("webpage", "pdf", "arxiv").
            metadata: Additional metadata for chunks.

        Returns:
            Number of chunks stored.
        """
        text = await self._scrape_url(url)
        if not text or len(text.strip()) < 50:
            logger.warning(f"Ingestion: no usable content from {url}")
            return 0

        return self.ingest_text(
            text=text,
            source=url,
            source_type=source_type,
            metadata=metadata,
        )

    def ingest_text(
        self,
        text: str,
        source: str = "direct",
        source_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Chunk text and store in vector store.

        Args:
            text: Raw text content.
            source: Source identifier (URL, file path, etc.).
            source_type: Type of source.
            metadata: Additional metadata for chunks.

        Returns:
            Number of chunks stored.
        """
        if not text or not text.strip():
            return 0

        chunks_text = self._chunk_text(text)
        if not chunks_text:
            return 0

        meta = metadata or {}
        doc_chunks = [
            DocumentChunk(
                chunk_id=make_chunk_id(source, i),
                content=chunk,
                source=source,
                source_type=source_type,
                metadata=meta,
            )
            for i, chunk in enumerate(chunks_text)
        ]

        added = self._store.add_chunks(doc_chunks)
        logger.info(f"Ingested {added} chunks from {source}")
        return added

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks at sentence boundaries.

        Args:
            text: Raw text to chunk.

        Returns:
            List of text chunks.
        """
        text = text.strip()
        if not text:
            return []

        # If text is shorter than chunk_size, return as single chunk
        if len(text) <= self._chunk_size:
            return [text]

        # Find sentence boundaries
        boundaries = [0]
        for m in _SENTENCE_END.finditer(text):
            boundaries.append(m.end())
        if boundaries[-1] < len(text):
            boundaries.append(len(text))

        chunks = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size

            # Snap to nearest sentence boundary
            best_end = end
            for b in boundaries:
                if b <= end and b > start:
                    best_end = b
                elif b > end:
                    break

            # If no sentence boundary found within range, use chunk_size
            if best_end <= start:
                best_end = min(start + self._chunk_size, len(text))

            chunk = text[start:best_end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start with overlap
            start = max(best_end - self._chunk_overlap, start + 1)
            if best_end >= len(text):
                break

        return chunks

    async def _scrape_url(self, url: str) -> str:
        """Scrape content from URL using existing web-search skill.

        Delegates to skills/web-search/tools.py _scrape_url().
        Falls back to simple HTTP GET if skill unavailable.
        """
        # Try existing web-search skill
        try:
            from skills import registry

            scraper = registry.get_skill_function("web-search", "_scrape_url")
            if scraper:
                return await scraper(url)
        except Exception:
            pass

        # Try direct import of scrape utility
        try:
            from Jotty.skills.web_search.tools import _scrape_url as scrape_fn

            result = await scrape_fn(url)
            if isinstance(result, str):
                return result
        except Exception:
            pass

        # Fallback: simple HTTP GET with httpx/aiohttp
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                # Basic HTML tag stripping
                text = re.sub(r"<[^>]+>", " ", resp.text)
                text = re.sub(r"\s+", " ", text).strip()
                return text
        except Exception as e:
            logger.warning(f"Ingestion: failed to scrape {url}: {e}")
            return ""


def get_ingestion_pipeline(
    vector_store: Optional[VectorStore] = None,
) -> DocumentIngestionPipeline:
    """Create a DocumentIngestionPipeline with default or provided store."""
    return DocumentIngestionPipeline(vector_store=vector_store)

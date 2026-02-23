"""Research Agents for OlympiadLearningSwarm.

Three agents for discovering and curating external educational content:
- SourceFinderAgent: Searches web for educational resources
- ContentExtractorAgent: Downloads and chunks content into vector store
- ContentCuratorAgent: Evaluates quality and grade-appropriateness via LLM

Each implements contribute() for BLACKBOARD and/or execute() for ROUND_ROBIN.

Author: Jotty Team
Date: February 2026
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .agents import BaseOlympiadAgent
from .signatures import (
    ContentCurationSignature,
    SourceSearchSignature,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SOURCE FINDER AGENT
# =============================================================================


class SourceFinderAgent(BaseOlympiadAgent):
    """Searches web for educational resources on a topic.

    Implements contribute() for BLACKBOARD: searches, deduplicates
    against existing blackboard URLs.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._searcher = self._create_module(SourceSearchSignature)

    async def contribute(
        self, blackboard: Dict[str, Any], context: Dict[str, Any], **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        """Search for educational sources and add to blackboard.

        Blackboard keys used:
            - task.topic (str): Topic to search
            - task.subject (str): Subject area
            - task.target_level (str): Grade/difficulty level
            - sources (list): Accumulated source URLs (dedup target)
            - done (bool): Set by curator when enough quality content

        Returns:
            Dict with found URLs and queries, or None if nothing new.
        """
        task = blackboard.get("task", {})
        topic = task.get("topic", "")
        subject = task.get("subject", "")
        target_level = task.get("target_level", "foundation")
        grade = task.get("grade", "")

        existing_urls = {s.get("url", "") for s in blackboard.get("sources", [])}
        max_sources = task.get("max_sources", 8)

        if len(existing_urls) >= max_sources:
            return None

        # Build search queries
        queries = self._build_queries(topic, subject, target_level, grade=grade)

        # Use LLM to generate additional targeted queries
        try:
            result = self._call_with_own_lm(
                self._searcher,
                topic=topic,
                subject=subject,
                target_level=target_level,
                existing_sources=", ".join(list(existing_urls)[:5]) or "none yet",
            )
            extra_queries = [q.strip() for q in str(result.search_queries).split("|") if q.strip()]
            queries.extend(extra_queries[:3])
        except Exception as e:
            logger.debug(f"SourceFinder LLM query generation failed: {e}")

        # Search web
        new_sources = []
        for query in queries[:5]:  # Limit total queries
            urls = await self._search_web(query)
            for url in urls:
                if url not in existing_urls and len(new_sources) + len(existing_urls) < max_sources:
                    new_sources.append({"url": url, "query": query})
                    existing_urls.add(url)

        if not new_sources:
            return None

        # Update blackboard
        if "sources" not in blackboard:
            blackboard["sources"] = []
        blackboard["sources"].extend(new_sources)

        logger.info(f"SourceFinder: found {len(new_sources)} new sources for '{topic}'")
        return {"new_sources": new_sources, "queries_used": queries}

    def _build_queries(
        self, topic: str, subject: str, target_level: str, grade: str = ""
    ) -> List[str]:
        """Build grade-appropriate search queries."""
        grade_tag = f" {grade}" if grade else ""
        queries = [
            f"{topic} {subject}{grade_tag} educational guide",
            f"{topic} explained {target_level} level{grade_tag}",
        ]

        # Add level-specific queries
        level_lower = target_level.lower()
        grade_lower = grade.lower()
        if (
            "foundation" in level_lower
            or "5th" in grade_lower
            or "elementary" in grade_lower
            or ("grade" in grade_lower and any(g in grade_lower for g in ["3", "4", "5", "6"]))
        ):
            queries.append(f"{topic} for kids simple explanation{grade_tag}")
            queries.append(f"{topic} {subject} elementary school lesson{grade_tag}")
        elif "olympiad" in level_lower or "advanced" in level_lower:
            queries.append(f"{topic}{grade_tag} olympiad problems techniques")
            queries.append(f"{topic} competition {subject} strategies{grade_tag}")
        else:
            queries.append(f"{topic}{grade_tag} tutorial with examples")

        return queries

    async def _search_web(self, query: str) -> List[str]:
        """Search web using existing web-search skill. Returns list of URLs."""
        try:
            # Try the web-search skill's search function
            from Jotty.skills.web_search.tools import search_web_tool

            results = await search_web_tool({"query": query, "max_results": 3})
            if isinstance(results, dict) and "results" in results:
                return [r.get("url", "") for r in results["results"] if r.get("url")]
            if isinstance(results, list):
                return [r.get("url", "") for r in results if isinstance(r, dict) and r.get("url")]
        except Exception:
            pass

        # Fallback: try httpx-based search
        try:
            import httpx

            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                import re

                urls = re.findall(r'href="(https?://[^"]+)"', resp.text)
                # Filter out search engine URLs
                return [
                    u
                    for u in urls[:5]
                    if "duckduckgo" not in u and "google" not in u and len(u) < 200
                ]
        except Exception as e:
            logger.debug(f"SourceFinder web search failed: {e}")

        return []


# =============================================================================
# CONTENT EXTRACTOR AGENT
# =============================================================================


class ContentExtractorAgent(BaseOlympiadAgent):
    """Downloads and chunks content from URLs into vector store.

    Implements:
    - contribute() for BLACKBOARD: extracts from unprocessed sources
    - execute() for ROUND_ROBIN: single-source extraction
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pipeline = None  # Lazy

    def _get_pipeline(self) -> Any:
        """Lazy-init DocumentIngestionPipeline."""
        if self._pipeline is None:
            from Jotty.core.intelligence.memory.document_ingestion import get_ingestion_pipeline

            self._pipeline = get_ingestion_pipeline()
        return self._pipeline

    async def contribute(
        self, blackboard: Dict[str, Any], context: Dict[str, Any], **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        """Extract content from unprocessed blackboard sources.

        Blackboard keys used:
            - sources (list): Source dicts with url key
            - extracted (set): Already-processed URLs
            - task.topic, task.subject: For metadata

        Returns:
            Dict with extraction results, or None if nothing to process.
        """
        sources = blackboard.get("sources", [])
        extracted = blackboard.get("extracted", set())
        task = blackboard.get("task", {})
        metadata = {
            "topic": task.get("topic", ""),
            "subject": task.get("subject", ""),
            "target_level": task.get("target_level", ""),
        }

        unprocessed = [s for s in sources if s.get("url") not in extracted]
        if not unprocessed:
            return None

        pipeline = self._get_pipeline()
        results = []
        for source in unprocessed:
            url = source.get("url", "")
            if not url:
                continue

            try:
                count = await pipeline.ingest_url(url, source_type="webpage", metadata=metadata)
                extracted.add(url)
                results.append({"url": url, "chunks": count})
                logger.info(f"ContentExtractor: ingested {count} chunks from {url[:60]}")
            except Exception as e:
                logger.debug(f"ContentExtractor: failed to ingest {url}: {e}")
                extracted.add(url)  # Mark as processed to avoid retry
                results.append({"url": url, "chunks": 0, "error": str(e)})

        blackboard["extracted"] = extracted

        if not results:
            return None
        return {"extractions": results, "total_chunks": sum(r.get("chunks", 0) for r in results)}

    async def execute(
        self, input: Any = None, context: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Extract content from a single URL (for ROUND_ROBIN).

        Args:
            input: URL string or dict with "url" key.
            context: Optional context with metadata.

        Returns:
            Dict with url and chunk count.
        """
        url = (
            input
            if isinstance(input, str)
            else (input.get("url", "") if isinstance(input, dict) else "")
        )
        if not url:
            return {"url": "", "chunks": 0, "error": "no URL"}

        metadata = {}
        if context:
            metadata = {
                k: v for k, v in context.items() if k in ("topic", "subject", "target_level")
            }

        pipeline = self._get_pipeline()
        try:
            count = await pipeline.ingest_url(url, source_type="webpage", metadata=metadata)
            return {"url": url, "chunks": count}
        except Exception as e:
            return {"url": url, "chunks": 0, "error": str(e)}


# =============================================================================
# CONTENT CURATOR AGENT
# =============================================================================


class ContentCuratorAgent(BaseOlympiadAgent):
    """Evaluates quality and grade-appropriateness of extracted content.

    Implements contribute() for BLACKBOARD: scores content, marks done
    when enough quality content is available.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._curator = self._create_module(ContentCurationSignature)

    async def contribute(
        self, blackboard: Dict[str, Any], context: Dict[str, Any], **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        """Score extracted content quality and mark blackboard done.

        Blackboard keys used:
            - task.topic, task.subject, task.target_level: For relevance scoring
            - extracted (set): Processed URLs
            - quality_scores (dict): URL -> score mapping
            - done (bool): Set to True when enough quality content

        Returns:
            Dict with quality assessment, or None if nothing to evaluate.
        """
        task = blackboard.get("task", {})
        topic = task.get("topic", "")
        subject = task.get("subject", "")
        target_level = task.get("target_level", "")
        extracted = blackboard.get("extracted", set())
        quality_scores = blackboard.get("quality_scores", {})
        min_quality_sources = task.get("min_quality_sources", 3)

        # Find URLs that have been extracted but not yet scored
        unscored = [url for url in extracted if url not in quality_scores]

        if not unscored:
            # Check if we have enough quality content
            good_sources = sum(1 for score in quality_scores.values() if score >= 0.6)
            if good_sources >= min_quality_sources:
                blackboard["done"] = True
                return {"status": "done", "good_sources": good_sources}
            return None

        # Score each unscored source
        try:
            # Get sample content from vector store for scoring
            from Jotty.core.intelligence.memory.vector_store import get_vector_store

            store = get_vector_store()
        except Exception:
            # Can't score without vector store
            return None

        scores = {}
        for url in unscored[:5]:  # Score up to 5 at a time
            # Get chunks from this source
            results = store.search(f"{topic} {subject}", top_k=3, filters=None)
            source_chunks = [r for r in results if r.chunk.source == url]

            if not source_chunks:
                scores[url] = 0.0
                continue

            sample_content = " ".join(r.chunk.content[:200] for r in source_chunks[:3])

            try:
                result = self._call_with_own_lm(
                    self._curator,
                    content_sample=sample_content,
                    topic=topic,
                    subject=subject,
                    target_level=target_level,
                )
                score_str = str(result.quality_score).strip()
                # Parse score (expect float 0-1)
                try:
                    score = float(score_str)
                    score = max(0.0, min(1.0, score))
                except (ValueError, TypeError):
                    score = 0.5
                scores[url] = score
            except Exception as e:
                logger.debug(f"ContentCurator: scoring failed for {url}: {e}")
                scores[url] = 0.5

        # Update blackboard
        quality_scores.update(scores)
        blackboard["quality_scores"] = quality_scores

        # Check completion
        good_sources = sum(1 for s in quality_scores.values() if s >= 0.6)
        if good_sources >= min_quality_sources:
            blackboard["done"] = True

        logger.info(
            f"ContentCurator: scored {len(scores)} sources, "
            f"{good_sources}/{min_quality_sources} quality sources"
        )
        return {"scores": scores, "good_sources": good_sources}


__all__ = [
    "SourceFinderAgent",
    "ContentExtractorAgent",
    "ContentCuratorAgent",
]

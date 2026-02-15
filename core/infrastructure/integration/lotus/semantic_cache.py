"""
Semantic Cache - Memoize Semantic Operations with Content Fingerprinting

LOTUS Insight: Same question on same data = same answer!

Features:
- Exact hash matching for identical content
- Semantic similarity matching for similar instructions
- TTL-based expiration
- LRU eviction when cache is full

Cost Impact:
- Cache hit = $0 (free)
- Repeat queries = infinite speedup
- Similar queries = cache hit via semantic matching
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import LotusConfig

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""

    key: str
    instruction_hash: str
    content_hash: str
    result: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    instruction_embedding: Optional[List[float]] = None  # For semantic matching

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (without embedding for storage)."""
        return {
            "key": self.key,
            "instruction_hash": self.instruction_hash,
            "content_hash": self.content_hash,
            "result": self.result,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
        }


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    semantic_hits: int = 0  # Hits via semantic similarity
    evictions: int = 0
    expirations: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.hits + self.misses
        return self.hits / max(total, 1)

    @property
    def total_savings(self) -> float:
        """Estimated cost savings from cache hits."""
        # Rough estimate: $0.003 per cached call avoided
        return self.hits * 0.003

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "semantic_hits": self.semantic_hits,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "hit_rate": self.hit_rate,
            "total_savings": self.total_savings,
        }


class SemanticCache:
    """
    Semantic caching layer for LLM operations.

    Caches results by:
    1. Content hash (exact match on input data)
    2. Instruction hash (exact match on operation instruction)
    3. Semantic similarity (optional, for similar instructions)

    DRY: Reuses CacheConfig from LotusConfig.
    """

    def __init__(
        self, config: Optional[LotusConfig] = None, embedding_fn: Optional[callable] = None
    ) -> None:
        """
        Initialize semantic cache.

        Args:
            config: LOTUS configuration
            embedding_fn: Optional function to generate embeddings for semantic matching
        """
        self.config = config or LotusConfig()
        self.cache_config = self.config.cache
        self.embedding_fn = embedding_fn

        # LRU cache using OrderedDict
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Instruction embeddings for semantic matching
        self._instruction_embeddings: Dict[str, Tuple[str, List[float]]] = {}

        self.stats = CacheStats()

        if self.cache_config.enabled:
            logger.info(
                f"SemanticCache initialized: max_entries={self.cache_config.max_entries}, "
                f"ttl={self.cache_config.ttl_seconds}s, "
                f"semantic_matching={self.cache_config.use_semantic_matching}"
            )

    def _hash_content(self, content: Any) -> str:
        """Generate hash for content."""
        if isinstance(content, str):
            data = content.encode("utf-8")
        else:
            data = json.dumps(content, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(data).hexdigest()[:16]

    def _hash_instruction(self, instruction: str) -> str:
        """Generate hash for instruction."""
        return hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:16]

    def _make_key(self, instruction_hash: str, content_hash: str) -> str:
        """Generate cache key from hashes."""
        return f"{instruction_hash}:{content_hash}"

    def get(
        self,
        instruction: str,
        content: Any,
    ) -> Tuple[bool, Optional[Any]]:
        """
        Get cached result if available.

        Args:
            instruction: The operation instruction/prompt
            content: The input content/data

        Returns:
            Tuple of (hit: bool, result: Any or None)
        """
        if not self.cache_config.enabled:
            return False, None

        instruction_hash = self._hash_instruction(instruction)
        content_hash = self._hash_content(content)
        key = self._make_key(instruction_hash, content_hash)

        # Exact match
        if key in self._cache:
            entry = self._cache[key]

            # Check expiration
            if entry.is_expired(self.cache_config.ttl_seconds):
                self._evict(key)
                self.stats.expirations += 1
                self.stats.misses += 1
                return False, None

            # Cache hit - update access info and move to end (LRU)
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._cache.move_to_end(key)
            self.stats.hits += 1

            if self.config.log_cache_hits:
                logger.debug(f"Cache hit (exact): {key}")

            return True, entry.result

        # Semantic matching (if enabled and embedding function available)
        if self.cache_config.use_semantic_matching and self.embedding_fn:
            semantic_result = self._semantic_lookup(instruction, content_hash)
            if semantic_result is not None:
                self.stats.hits += 1
                self.stats.semantic_hits += 1

                if self.config.log_cache_hits:
                    logger.debug(f"Cache hit (semantic): {instruction[:50]}...")

                return True, semantic_result

        self.stats.misses += 1
        return False, None

    def put(
        self,
        instruction: str,
        content: Any,
        result: Any,
    ) -> None:
        """
        Cache a result.

        Args:
            instruction: The operation instruction/prompt
            content: The input content/data
            result: The result to cache
        """
        if not self.cache_config.enabled:
            return

        instruction_hash = self._hash_instruction(instruction)
        content_hash = self._hash_content(content)
        key = self._make_key(instruction_hash, content_hash)

        # Evict if at capacity
        while len(self._cache) >= self.cache_config.max_entries:
            self._evict_lru()

        # Generate embedding for semantic matching (if enabled)
        instruction_embedding = None
        if self.cache_config.use_semantic_matching and self.embedding_fn:
            try:
                instruction_embedding = self.embedding_fn(instruction)
                self._instruction_embeddings[instruction_hash] = (key, instruction_embedding)
            except Exception as e:
                logger.debug(f"Failed to generate embedding: {e}")

        # Create entry
        now = time.time()
        entry = CacheEntry(
            key=key,
            instruction_hash=instruction_hash,
            content_hash=content_hash,
            result=result,
            created_at=now,
            last_accessed=now,
            access_count=0,
            instruction_embedding=instruction_embedding,
        )

        self._cache[key] = entry
        logger.debug(f"Cached: {key}")

    def _semantic_lookup(
        self,
        instruction: str,
        content_hash: str,
    ) -> Optional[Any]:
        """
        Look up by semantic similarity.

        Args:
            instruction: The instruction to match
            content_hash: Content hash to match

        Returns:
            Cached result if semantic match found, None otherwise
        """
        if not self.embedding_fn or not self._instruction_embeddings:
            return None

        try:
            query_embedding = self.embedding_fn(instruction)

            best_similarity = 0.0
            best_key = None

            for instr_hash, (cached_key, cached_embedding) in self._instruction_embeddings.items():
                # Only consider entries with same content
                if not cached_key.endswith(f":{content_hash}"):
                    continue

                similarity = self._cosine_similarity(query_embedding, cached_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_key = cached_key

            if best_similarity >= self.cache_config.similarity_threshold:
                if best_key in self._cache:
                    entry = self._cache[best_key]
                    if not entry.is_expired(self.cache_config.ttl_seconds):
                        return entry.result

        except Exception as e:
            logger.debug(f"Semantic lookup failed: {e}")

        return None

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _evict(self, key: str) -> None:
        """Evict a specific key from cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            # Clean up embedding
            if entry.instruction_hash in self._instruction_embeddings:
                del self._instruction_embeddings[entry.instruction_hash]
            self.stats.evictions += 1

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            oldest_key = next(iter(self._cache))
            self._evict(oldest_key)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._instruction_embeddings.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.to_dict()
        stats["size"] = len(self._cache)
        stats["max_size"] = self.cache_config.max_entries
        return stats

    def get_or_compute(
        self,
        instruction: str,
        content: Any,
        compute_fn: callable,
    ) -> Any:
        """
        Get from cache or compute and cache result.

        DRY pattern: Combines get + put in single call.

        Args:
            instruction: The operation instruction
            content: The input content
            compute_fn: Function to call if cache miss

        Returns:
            Cached or computed result
        """
        hit, result = self.get(instruction, content)
        if hit:
            return result

        # Compute
        result = compute_fn()

        # Cache
        self.put(instruction, content, result)

        return result

    async def get_or_compute_async(
        self,
        instruction: str,
        content: Any,
        compute_fn: callable,
    ) -> Any:
        """
        Async version of get_or_compute.

        Args:
            instruction: The operation instruction
            content: The input content
            compute_fn: Async function to call if cache miss

        Returns:
            Cached or computed result
        """
        hit, result = self.get(instruction, content)
        if hit:
            return result

        # Compute
        result = await compute_fn()

        # Cache
        self.put(instruction, content, result)

        return result


# Decorator for caching semantic operations
def cached_operation(cache: SemanticCache) -> Any:
    """
    Decorator to cache semantic operations.

    Usage:
        cache = SemanticCache()

        @cached_operation(cache)
        async def my_operation(instruction: str, content: str) -> str:
            return await llm.generate(instruction, content)
    """

    def decorator(fn: Any) -> Any:
        async def wrapper(instruction: str, content: Any, *args: Any, **kwargs: Any) -> Any:
            return await cache.get_or_compute_async(
                instruction, content, lambda: fn(instruction, content, *args, **kwargs)
            )

        return wrapper

    return decorator

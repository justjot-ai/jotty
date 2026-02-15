"""
LLM Call Cache
==============

Semantic caching for LLM calls to reduce redundant API calls.

A-Team Critical Fix: Reduce redundant LLM calls with intelligent caching.

Features:
- Hash-based cache key (prompt + model + temperature)
- TTL-based expiration (configurable, default 1 hour)
- Max cache size with LRU eviction
- Optional persistent cache (SQLite/file-based)
- Cache hit/miss statistics
"""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """A cached LLM response with metadata."""

    response: Any
    prompt_hash: str
    model: str
    temperature: float
    created_at: float
    ttl_seconds: float
    hit_count: int = 0

    def is_expired(self) -> bool:
        """Check if this cached response has expired."""
        return time.time() > (self.created_at + self.ttl_seconds)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "response": self.response,
            "prompt_hash": self.prompt_hash,
            "model": self.model,
            "temperature": self.temperature,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "hit_count": self.hit_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedResponse":
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class CacheStats:
    """Statistics about cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    total_requests: int = 0
    total_saved_calls: int = 0
    cache_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "total_requests": self.total_requests,
            "total_saved_calls": self.total_saved_calls,
            "cache_size": self.cache_size,
            "hit_rate": self.hit_rate,
        }


class LLMCallCache:
    """
    Semantic caching for LLM calls to reduce redundant API calls.

    Features:
    - Hash-based cache key (prompt + model + temperature)
    - TTL-based expiration (configurable)
    - Max cache size with LRU eviction
    - Thread-safe operations
    - Optional persistent storage

    Usage:
        cache = LLMCallCache(max_size=1000, default_ttl=3600)

        # Direct usage
        response = cache.get_or_call(
            prompt="What is 2+2?",
            llm_func=lambda p: call_llm(p),
            model="gpt-4",
            temperature=0.0
        )

        # With decorator
        @cache.cached(model="gpt-4", temperature=0.0)
        def my_llm_call(prompt: str) -> str:
            return actual_llm_call(prompt)
    """

    _instances: Dict[str, "LLMCallCache"] = {}
    _instances_lock = threading.Lock()

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 3600.0,
        persist_path: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        """
        Initialize LLM call cache.

        Args:
            max_size: Maximum number of entries in cache
            default_ttl: Default time-to-live in seconds (1 hour)
            persist_path: Optional path for persistent cache storage
            enabled: Whether caching is enabled
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.persist_path = Path(persist_path) if persist_path else None
        self.enabled = enabled

        # LRU cache using OrderedDict
        self._cache: OrderedDict[str, CachedResponse] = OrderedDict()

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = CacheStats()

        # Load persistent cache if available
        if self.persist_path and self.persist_path.exists():
            self._load_from_disk()

        logger.info(
            f"LLMCallCache initialized: max_size={max_size}, "
            f"default_ttl={default_ttl}s, enabled={enabled}"
        )

    @classmethod
    def get_instance(cls, name: str = "default", **kwargs: Any) -> "LLMCallCache":
        """
        Get a singleton cache instance by name (thread-safe, double-checked locking).

        Args:
            name: Instance name for multiple cache contexts
            **kwargs: Arguments passed to constructor if creating new instance

        Returns:
            LLMCallCache instance
        """
        if name not in cls._instances:
            with cls._instances_lock:
                if name not in cls._instances:
                    cls._instances[name] = cls(**kwargs)
        return cls._instances[name]

    @classmethod
    def reset_instances(cls) -> None:
        """Reset all cached instances (for testing)."""
        cls._instances.clear()

    def _compute_hash(
        self, prompt: str, model: str = "", temperature: float = 0.0, **extra_context: Any
    ) -> str:
        """
        Compute cache key hash from prompt and parameters.

        Args:
            prompt: The LLM prompt
            model: Model identifier
            temperature: Temperature setting
            **extra_context: Additional context for uniqueness

        Returns:
            SHA256 hash string
        """
        # Normalize prompt (strip whitespace, lowercase for consistency)
        normalized_prompt = prompt.strip()

        # Build key components
        key_parts = [
            normalized_prompt,
            str(model),
            f"temp={temperature:.2f}",
        ]

        # Add extra context if provided
        for k, v in sorted(extra_context.items()):
            key_parts.append(f"{k}={v}")

        key_string = "|".join(key_parts)

        # Compute SHA256 hash
        return hashlib.sha256(key_string.encode("utf-8")).hexdigest()[:32]

    def get(self, prompt_hash: str) -> Optional[CachedResponse]:
        """
        Get cached response by hash.

        Args:
            prompt_hash: The cache key hash

        Returns:
            CachedResponse if found and not expired, None otherwise
        """
        if not self.enabled:
            return None

        with self._lock:
            self._stats.total_requests += 1

            if prompt_hash not in self._cache:
                self._stats.misses += 1
                return None

            cached = self._cache[prompt_hash]

            # Check expiration
            if cached.is_expired():
                self._stats.expirations += 1
                self._stats.misses += 1
                del self._cache[prompt_hash]
                return None

            # Move to end for LRU
            self._cache.move_to_end(prompt_hash)

            # Update stats
            self._stats.hits += 1
            self._stats.total_saved_calls += 1
            cached.hit_count += 1

            logger.debug(f"Cache HIT: {prompt_hash[:8]}... " f"(hits: {cached.hit_count})")

            return cached

    def set(
        self,
        prompt_hash: str,
        response: Any,
        model: str = "",
        temperature: float = 0.0,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Store response in cache.

        Args:
            prompt_hash: The cache key hash
            response: The LLM response to cache
            model: Model identifier
            temperature: Temperature setting
            ttl: Time-to-live (uses default if not specified)
        """
        if not self.enabled:
            return

        ttl = ttl if ttl is not None else self.default_ttl

        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                self._evict_oldest()

            # Store new entry
            self._cache[prompt_hash] = CachedResponse(
                response=response,
                prompt_hash=prompt_hash,
                model=model,
                temperature=temperature,
                created_at=time.time(),
                ttl_seconds=ttl,
                hit_count=0,
            )

            self._stats.cache_size = len(self._cache)

            logger.debug(
                f"Cache SET: {prompt_hash[:8]}... " f"(size: {len(self._cache)}/{self.max_size})"
            )

    def _evict_oldest(self) -> None:
        """Evict the oldest (least recently used) entry."""
        if self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._stats.evictions += 1
            logger.debug(f"Cache EVICT: {oldest_key[:8]}...")

    def get_or_call(
        self,
        prompt: str,
        llm_func: Callable[[str], Any],
        model: str = "",
        temperature: float = 0.0,
        ttl: Optional[float] = None,
        **extra_context: Any,
    ) -> Any:
        """
        Get cached response or call LLM function.

        This is the primary interface for caching LLM calls.

        Args:
            prompt: The LLM prompt
            llm_func: Function to call if not cached (takes prompt as arg)
            model: Model identifier
            temperature: Temperature setting
            ttl: Time-to-live for this entry
            **extra_context: Additional context for cache key

        Returns:
            LLM response (cached or fresh)
        """
        # Compute cache key
        prompt_hash = self._compute_hash(
            prompt=prompt, model=model, temperature=temperature, **extra_context
        )

        # Try cache first
        cached = self.get(prompt_hash)
        if cached is not None:
            return cached.response

        # Call LLM
        logger.debug(f"Cache MISS: {prompt_hash[:8]}... - calling LLM")
        response = llm_func(prompt)

        # Store in cache
        self.set(
            prompt_hash=prompt_hash,
            response=response,
            model=model,
            temperature=temperature,
            ttl=ttl,
        )

        return response

    async def get_or_call_async(
        self,
        prompt: str,
        llm_func: Callable[[str], Any],
        model: str = "",
        temperature: float = 0.0,
        ttl: Optional[float] = None,
        **extra_context: Any,
    ) -> Any:
        """
        Async version of get_or_call.

        Args:
            prompt: The LLM prompt
            llm_func: Async function to call if not cached
            model: Model identifier
            temperature: Temperature setting
            ttl: Time-to-live for this entry
            **extra_context: Additional context for cache key

        Returns:
            LLM response (cached or fresh)
        """
        import asyncio

        # Compute cache key
        prompt_hash = self._compute_hash(
            prompt=prompt, model=model, temperature=temperature, **extra_context
        )

        # Try cache first
        cached = self.get(prompt_hash)
        if cached is not None:
            return cached.response

        # Call LLM (handle both sync and async)
        logger.debug(f"Cache MISS: {prompt_hash[:8]}... - calling LLM (async)")
        if asyncio.iscoroutinefunction(llm_func):
            response = await llm_func(prompt)
        else:
            response = llm_func(prompt)

        # Store in cache
        self.set(
            prompt_hash=prompt_hash,
            response=response,
            model=model,
            temperature=temperature,
            ttl=ttl,
        )

        return response

    def cached(
        self, model: str = "", temperature: float = 0.0, ttl: Optional[float] = None
    ) -> Callable:
        """
        Decorator for caching LLM function calls.

        Usage:
            @cache.cached(model="gpt-4", temperature=0.0)
            def my_llm_call(prompt: str) -> str:
                return actual_llm_call(prompt)

        Args:
            model: Model identifier
            temperature: Temperature setting
            ttl: Time-to-live for cached entries

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            import asyncio
            import functools

            @functools.wraps(func)
            def sync_wrapper(prompt: str, *args: Any, **kwargs: Any) -> Any:
                return self.get_or_call(
                    prompt=prompt,
                    llm_func=lambda p: func(p, *args, **kwargs),
                    model=model,
                    temperature=temperature,
                    ttl=ttl,
                )

            @functools.wraps(func)
            async def async_wrapper(prompt: str, *args: Any, **kwargs: Any) -> Any:
                return await self.get_or_call_async(
                    prompt=prompt,
                    llm_func=lambda p: func(p, *args, **kwargs),
                    model=model,
                    temperature=temperature,
                    ttl=ttl,
                )

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper

        return decorator

    def invalidate(self, prompt_hash: str) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            prompt_hash: The cache key to invalidate

        Returns:
            True if entry was found and removed
        """
        with self._lock:
            if prompt_hash in self._cache:
                del self._cache[prompt_hash]
                self._stats.cache_size = len(self._cache)
                logger.debug(f"Cache INVALIDATE: {prompt_hash[:8]}...")
                return True
            return False

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.cache_size = 0
            logger.info(f"Cache CLEAR: {count} entries removed")
            return count

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired = [key for key, cached in self._cache.items() if cached.is_expired()]
            for key in expired:
                del self._cache[key]
                self._stats.expirations += 1

            self._stats.cache_size = len(self._cache)

            if expired:
                logger.info(f"Cache CLEANUP: {len(expired)} expired entries removed")

            return len(expired)

    def stats(self) -> CacheStats:
        """
        Get cache statistics.

        Returns:
            CacheStats object
        """
        with self._lock:
            self._stats.cache_size = len(self._cache)
            return self._stats

    def _save_to_disk(self) -> None:
        """Save cache to disk for persistence."""
        if not self.persist_path:
            return

        try:
            with self._lock:
                data = {
                    key: cached.to_dict()
                    for key, cached in self._cache.items()
                    if not cached.is_expired()
                }

            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump(data, f)

            logger.info(f"Cache saved to disk: {len(data)} entries")
        except Exception as e:
            logger.warning(f"Failed to save cache to disk: {e}")

    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)

            with self._lock:
                for key, cached_dict in data.items():
                    cached = CachedResponse.from_dict(cached_dict)
                    if not cached.is_expired():
                        self._cache[key] = cached

                self._stats.cache_size = len(self._cache)

            logger.info(f"Cache loaded from disk: {len(self._cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_cache(name: str = "default", **kwargs: Any) -> LLMCallCache:
    """Get a named cache instance."""
    return LLMCallCache.get_instance(name, **kwargs)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "LLMCallCache",
    "CachedResponse",
    "CacheStats",
    "get_cache",
]

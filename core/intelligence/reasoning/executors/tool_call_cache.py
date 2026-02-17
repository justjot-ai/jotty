"""
Tool Call Cache - Thread-safe TTL + LRU cache for tool call results

Prevents redundant tool executions when the same skill+tool+params
combination is called multiple times within a session.

Extracted from skill_plan_executor.py for better modularity.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any, Dict, Optional, Tuple


class ToolCallCache:
    """Thread-safe TTL + LRU cache for tool call results.

    Prevents redundant tool executions when the same skill+tool+params
    combination is called multiple times within a session.

    Usage:
        cache = ToolCallCache(ttl_seconds=300)
        key = cache.make_key("web-search", "search_web_tool", {"query": "AI"})
        cached = cache.get(key)
        if cached is None:
            result = await tool(params)
            cache.set(key, result)

    Args:
        ttl_seconds: Time-to-live for cached entries (default: 300)
        max_size: Maximum number of entries in cache (default: 100)
    """

    def __init__(self, ttl_seconds: int = 300, max_size: int = 100) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def make_key(skill_name: str, tool_name: str, params: Dict[str, Any]) -> str:
        """Create a deterministic cache key from skill, tool, and params.

        Args:
            skill_name: Name of the skill
            tool_name: Name of the tool being called
            params: Tool parameters dictionary

        Returns:
            MD5 hash of the skill+tool+params combination
        """
        # Sort params for deterministic hashing, skip non-serializable values
        try:
            param_str = json.dumps(params, sort_keys=True, default=str)
        except (TypeError, ValueError):
            param_str = str(sorted(params.items()))
        raw = f"{skill_name}:{tool_name}:{param_str}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get a cached result if it exists and hasn't expired.

        Args:
            key: Cache key (from make_key)

        Returns:
            Cached result if found and not expired, None otherwise
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            result, timestamp = entry
            if time.time() - timestamp > self._ttl:
                del self._cache[key]
                return None
            return result

    def set(self, key: str, value: Any) -> None:
        """Cache a tool result.

        Args:
            key: Cache key (from make_key)
            value: Result to cache
        """
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)


__all__ = ["ToolCallCache"]

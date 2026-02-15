"""
SharedContext - Simple Shared Data Store

Simple dict-like store for sharing data between MetaDataFetcher and domain agents.

Design:
- MetaDataFetcher writes (stores fetched metadata)
- Domain agents read (via SwarmReVal parameter resolution)
- No complex features needed (no async, no compression, no semantic search)
- Just a clean, thread-safe key-value store
"""

import logging
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SharedContext:
    """
    Simple shared data store for agent communication.

    DESIGN PRINCIPLES:
    - Simple dict-like interface
    - Thread-safe for concurrent access
    - No complex features (compression, semantic search, etc.)
    - MetaDataFetcher writes, domain agents read via SwarmReVal

    Example:
        context = SharedContext()

        # MetaDataFetcher writes
        context.set('business_terms', {'ENTITY_A': 'Description of entity A'})
        context.set('data_schemas', {'data_source_1': {...}})

        # SwarmReVal reads for domain agents
        terms = context.get('business_terms')  # For BusinessTermResolver
        schemas = context.get('table_schemas')  # For ColumnFilterSelector
    """

    def __init__(self) -> None:
        """Initialize empty shared context."""
        self.data: Dict[str, Any] = {}
        self._lock = Lock()

        logger.info(" SharedContext initialized (simple dict store)")

    def set(self, key: str, value: Any) -> None:
        """
        Store data in shared context.

        Args:
            key: Semantic key (e.g., 'business_terms', 'table_schemas')
            value: Data to store (any type)

        Thread-safe.
        """
        with self._lock:
            self.data[key] = value
            logger.debug(f" SharedContext.set('{key}') - type: {type(value).__name__}")

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve data from shared context.

        Args:
            key: Semantic key

        Returns:
            Stored data, or None if key doesn't exist

        Thread-safe.
        """
        with self._lock:
            value = self.data.get(key)
            if value is not None:
                logger.debug(f" SharedContext.get('{key}') - found")
            else:
                logger.debug(f" SharedContext.get('{key}') - not found")
            return value

    def get_all(self) -> Dict[str, Any]:
        """
        Get all data in shared context.

        Returns:
            Copy of all data

        Thread-safe.
        """
        with self._lock:
            return self.data.copy()

    def keys(self) -> List[str]:
        """
        Get all keys in shared context.

        Returns:
            List of all keys

        Thread-safe.
        """
        with self._lock:
            return list(self.data.keys())

    def has(self, key: str) -> bool:
        """
        Check if key exists in shared context.

        Args:
            key: Key to check

        Returns:
            True if key exists, False otherwise

        Thread-safe.
        """
        with self._lock:
            return key in self.data

    def clear(self) -> None:
        """
        Clear all data from shared context.

        Thread-safe.
        """
        with self._lock:
            self.data.clear()
            logger.debug(" SharedContext cleared")

    def summary(self) -> str:
        """
        Get summary of shared context for logging.

        Returns:
            Human-readable summary
        """
        with self._lock:
            keys = list(self.data.keys())
            return f"SharedContext({len(keys)} items: {keys})"

    def __repr__(self) -> str:
        """String representation."""
        return self.summary()

    def __contains__(self, key: str) -> bool:
        """
        Support 'in' operator: 'key' in shared_context

        Args:
            key: Key to check

        Returns:
            True if key exists, False otherwise
        """
        return self.has(key)

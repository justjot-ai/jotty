"""
Fallback Memory
===============

Lightweight fallback memory system when full brain-inspired memory is unavailable.

A-Team Critical Fix: Graceful degradation when BrainInspiredMemoryManager unavailable.

Features:
- Simple dict-based storage
- No consolidation (just LRU eviction)
- Basic retrieve by recency
- Minimal dependencies
"""

import time
import logging
import threading
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory for fallback system."""
    EPISODIC = "episodic"       # Recent experiences
    SEMANTIC = "semantic"       # General knowledge
    PROCEDURAL = "procedural"   # How-to knowledge


@dataclass
class MemoryEntry:
    """A simple memory entry for fallback storage."""
    content: str
    memory_type: MemoryType = MemoryType.EPISODIC
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5

    def access(self) -> None:
        """Record an access to this memory."""
        self.access_count += 1
        self.last_accessed = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'content': self.content,
            'memory_type': self.memory_type.value,
            'timestamp': self.timestamp,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'metadata': self.metadata,
            'importance': self.importance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Deserialize from dictionary."""
        return cls(
            content=data['content'],
            memory_type=MemoryType(data.get('memory_type', 'episodic')),
            timestamp=data.get('timestamp', time.time()),
            access_count=data.get('access_count', 0),
            last_accessed=data.get('last_accessed', time.time()),
            metadata=data.get('metadata', {}),
            importance=data.get('importance', 0.5),
        )


class SimpleFallbackMemory:
    """
    Lightweight fallback memory when full brain not available.

    Features:
    - Simple dict-based storage
    - LRU eviction when capacity reached
    - Basic relevance retrieval (keyword + recency)
    - Thread-safe operations
    - No external dependencies

    Usage:
        memory = SimpleFallbackMemory(max_entries=500)

        # Store memory
        memory.store("User prefers SQL queries with explicit joins")

        # Retrieve relevant memories
        relevant = memory.retrieve("SQL query optimization", top_k=5)

        # Get recent memories
        recent = memory.get_recent(10)
    """

    def __init__(
        self,
        max_entries: int = 500,
        episodic_capacity: int = 300,
        semantic_capacity: int = 150,
        procedural_capacity: int = 50
    ):
        """
        Initialize fallback memory.

        Args:
            max_entries: Total maximum entries across all types
            episodic_capacity: Capacity for episodic memories
            semantic_capacity: Capacity for semantic memories
            procedural_capacity: Capacity for procedural memories
        """
        self.max_entries = max_entries
        self.capacities = {
            MemoryType.EPISODIC: episodic_capacity,
            MemoryType.SEMANTIC: semantic_capacity,
            MemoryType.PROCEDURAL: procedural_capacity,
        }

        # LRU-ordered storage per memory type
        self._memories: Dict[MemoryType, OrderedDict[str, MemoryEntry]] = {
            MemoryType.EPISODIC: OrderedDict(),
            MemoryType.SEMANTIC: OrderedDict(),
            MemoryType.PROCEDURAL: OrderedDict(),
        }

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._total_stored = 0
        self._total_retrieved = 0
        self._total_evicted = 0

        logger.info(
            f"SimpleFallbackMemory initialized: max_entries={max_entries}, "
            f"episodic={episodic_capacity}, semantic={semantic_capacity}, "
            f"procedural={procedural_capacity}"
        )

    def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Store a memory entry.

        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance score (0-1)
            metadata: Additional metadata

        Returns:
            Memory key/ID
        """
        if not content or not content.strip():
            return ""

        with self._lock:
            # Generate key based on content hash
            import hashlib
            key = hashlib.md5(content.encode()).hexdigest()[:16]

            # Create entry
            entry = MemoryEntry(
                content=content.strip(),
                memory_type=memory_type,
                importance=importance,
                metadata=metadata or {},
            )

            # Get storage for this type
            storage = self._memories[memory_type]

            # Check if already exists
            if key in storage:
                # Update existing
                storage.move_to_end(key)
                storage[key].access()
                return key

            # Evict if at capacity
            capacity = self.capacities[memory_type]
            while len(storage) >= capacity:
                self._evict_lru(memory_type)

            # Store
            storage[key] = entry
            self._total_stored += 1

            logger.debug(
                f"Stored {memory_type.value} memory: {content[:50]}... "
                f"(size: {len(storage)}/{capacity})"
            )

            return key

    def _evict_lru(self, memory_type: MemoryType) -> Optional[str]:
        """Evict least recently used entry from a memory type."""
        storage = self._memories[memory_type]
        if not storage:
            return None

        # Get LRU key (first item in ordered dict)
        lru_key = next(iter(storage))
        del storage[lru_key]
        self._total_evicted += 1

        logger.debug(f"Evicted LRU {memory_type.value} memory: {lru_key}")
        return lru_key

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant memories for a query.

        Uses simple keyword matching + recency scoring.

        Args:
            query: Query string
            top_k: Number of memories to retrieve
            memory_type: Specific type to search, or None for all

        Returns:
            List of relevant MemoryEntry objects
        """
        if not query:
            return []

        self._total_retrieved += 1

        with self._lock:
            # Get all memories to search
            if memory_type:
                all_memories = list(self._memories[memory_type].values())
            else:
                all_memories = []
                for storage in self._memories.values():
                    all_memories.extend(storage.values())

            if not all_memories:
                return []

            # Score each memory
            query_words = set(query.lower().split())
            current_time = time.time()
            scored = []

            for entry in all_memories:
                # Keyword overlap score
                content_words = set(entry.content.lower().split())
                if query_words and content_words:
                    keyword_score = len(query_words & content_words) / len(query_words)
                else:
                    keyword_score = 0.0

                # Recency score (decay over 24 hours)
                age_hours = (current_time - entry.last_accessed) / 3600
                recency_score = max(0.0, 1.0 - (age_hours / 24))

                # Access frequency score
                freq_score = min(1.0, entry.access_count / 10)

                # Importance score
                importance_score = entry.importance

                # Combined score (weighted)
                combined = (
                    0.4 * keyword_score +
                    0.2 * recency_score +
                    0.1 * freq_score +
                    0.3 * importance_score
                )

                if combined > 0.1:  # Threshold
                    scored.append((entry, combined))

            # Sort by score (descending) and return top_k
            scored.sort(key=lambda x: x[1], reverse=True)
            results = [entry for entry, _ in scored[:top_k]]

            # Update access counts
            for entry in results:
                entry.access()

            return results

    def get_recent(
        self,
        count: int = 10,
        memory_type: Optional[MemoryType] = None
    ) -> List[MemoryEntry]:
        """
        Get most recently stored memories.

        Args:
            count: Number of memories to retrieve
            memory_type: Specific type, or None for all

        Returns:
            List of recent MemoryEntry objects
        """
        with self._lock:
            all_memories = []

            if memory_type:
                all_memories = list(self._memories[memory_type].values())
            else:
                for storage in self._memories.values():
                    all_memories.extend(storage.values())

            # Sort by timestamp (descending)
            all_memories.sort(key=lambda e: e.timestamp, reverse=True)

            return all_memories[:count]

    def clear(self, memory_type: Optional[MemoryType] = None) -> int:
        """
        Clear memories.

        Args:
            memory_type: Specific type to clear, or None for all

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = 0

            if memory_type:
                count = len(self._memories[memory_type])
                self._memories[memory_type].clear()
            else:
                for storage in self._memories.values():
                    count += len(storage)
                    storage.clear()

            logger.info(f"Cleared {count} memory entries")
            return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        with self._lock:
            type_stats = {}
            total_entries = 0

            for mem_type, storage in self._memories.items():
                count = len(storage)
                total_entries += count
                type_stats[mem_type.value] = {
                    'count': count,
                    'capacity': self.capacities[mem_type],
                    'utilization': count / self.capacities[mem_type] if self.capacities[mem_type] > 0 else 0,
                }

            return {
                'total_entries': total_entries,
                'max_entries': self.max_entries,
                'utilization': total_entries / self.max_entries if self.max_entries > 0 else 0,
                'total_stored': self._total_stored,
                'total_retrieved': self._total_retrieved,
                'total_evicted': self._total_evicted,
                'by_type': type_stats,
            }

    # =========================================================================
    # COMPATIBILITY METHODS (match BrainInspiredMemoryManager interface)
    # =========================================================================

    def remember(
        self,
        content: str,
        level: str = "episodic",
        importance: float = 0.5,
        **kwargs
    ) -> str:
        """
        Compatibility alias for store().

        Matches BrainInspiredMemoryManager.remember() signature.
        """
        # Map level string to MemoryType
        type_map = {
            'episodic': MemoryType.EPISODIC,
            'semantic': MemoryType.SEMANTIC,
            'procedural': MemoryType.PROCEDURAL,
        }
        memory_type = type_map.get(level.lower(), MemoryType.EPISODIC)

        return self.store(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=kwargs
        )

    def recall(
        self,
        query: str,
        top_k: int = 5,
        level: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """
        Compatibility alias for retrieve().

        Matches BrainInspiredMemoryManager.recall() signature.
        Returns content strings instead of MemoryEntry objects.
        """
        type_map = {
            'episodic': MemoryType.EPISODIC,
            'semantic': MemoryType.SEMANTIC,
            'procedural': MemoryType.PROCEDURAL,
        }
        memory_type = type_map.get(level.lower(), None) if level else None

        entries = self.retrieve(query=query, top_k=top_k, memory_type=memory_type)
        return [entry.content for entry in entries]

    def consolidate(self) -> Dict[str, Any]:
        """
        No-op consolidation for fallback memory.

        Returns stats instead of doing actual consolidation.
        """
        logger.debug("SimpleFallbackMemory: consolidation not supported (no-op)")
        return self.get_statistics()

    # =========================================================================
    # PERSISTENCE METHODS
    # =========================================================================

    def save(self, path: str) -> bool:
        """
        Save memory state to file for cross-session persistence.

        Args:
            path: File path to save to (JSON format)

        Returns:
            True if saved successfully
        """
        import json
        from pathlib import Path

        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            with self._lock:
                data = {
                    'max_entries': self.max_entries,
                    'capacities': {k.value: v for k, v in self.capacities.items()},
                    'memories': {},
                    'statistics': {
                        'total_stored': self._total_stored,
                        'total_retrieved': self._total_retrieved,
                        'total_evicted': self._total_evicted,
                    },
                    'timestamp': time.time(),
                }

                # Serialize all memories
                for mem_type, storage in self._memories.items():
                    data['memories'][mem_type.value] = {
                        key: entry.to_dict()
                        for key, entry in storage.items()
                    }

            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            total = sum(len(s) for s in self._memories.values())
            logger.info(f" SimpleFallbackMemory saved: {total} entries to {path}")
            return True

        except Exception as e:
            logger.error(f" Failed to save SimpleFallbackMemory: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        Load memory state from file.

        Args:
            path: File path to load from

        Returns:
            True if loaded successfully
        """
        import json
        from pathlib import Path

        if not Path(path).exists():
            logger.info(f"ℹ No previous memory state at {path}")
            return False

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            with self._lock:
                # Restore capacities
                if 'capacities' in data:
                    for type_str, capacity in data['capacities'].items():
                        mem_type = MemoryType(type_str)
                        self.capacities[mem_type] = capacity

                # Restore memories
                for type_str, memories in data.get('memories', {}).items():
                    mem_type = MemoryType(type_str)
                    self._memories[mem_type] = OrderedDict()

                    for key, entry_data in memories.items():
                        entry = MemoryEntry.from_dict(entry_data)
                        self._memories[mem_type][key] = entry

                # Restore statistics
                stats = data.get('statistics', {})
                self._total_stored = stats.get('total_stored', 0)
                self._total_retrieved = stats.get('total_retrieved', 0)
                self._total_evicted = stats.get('total_evicted', 0)

            total = sum(len(s) for s in self._memories.values())
            logger.info(f" SimpleFallbackMemory loaded: {total} entries from {path}")
            return True

        except Exception as e:
            logger.error(f" Failed to load SimpleFallbackMemory: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire memory state to dictionary."""
        with self._lock:
            return {
                'max_entries': self.max_entries,
                'capacities': {k.value: v for k, v in self.capacities.items()},
                'memories': {
                    mem_type.value: {
                        key: entry.to_dict()
                        for key, entry in storage.items()
                    }
                    for mem_type, storage in self._memories.items()
                },
                'statistics': {
                    'total_stored': self._total_stored,
                    'total_retrieved': self._total_retrieved,
                    'total_evicted': self._total_evicted,
                },
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleFallbackMemory':
        """Deserialize from dictionary."""
        capacities = data.get('capacities', {})
        memory = cls(
            max_entries=data.get('max_entries', 500),
            episodic_capacity=capacities.get('episodic', 300),
            semantic_capacity=capacities.get('semantic', 150),
            procedural_capacity=capacities.get('procedural', 50),
        )

        # Restore memories
        for type_str, memories in data.get('memories', {}).items():
            mem_type = MemoryType(type_str)
            for key, entry_data in memories.items():
                memory._memories[mem_type][key] = MemoryEntry.from_dict(entry_data)

        # Restore statistics
        stats = data.get('statistics', {})
        memory._total_stored = stats.get('total_stored', 0)
        memory._total_retrieved = stats.get('total_retrieved', 0)
        memory._total_evicted = stats.get('total_evicted', 0)

        return memory


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_fallback_memory(**kwargs) -> SimpleFallbackMemory:
    """Get a SimpleFallbackMemory instance."""
    return SimpleFallbackMemory(**kwargs)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SimpleFallbackMemory',
    'MemoryEntry',
    'MemoryType',
    'get_fallback_memory',
]

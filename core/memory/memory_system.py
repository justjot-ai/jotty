from typing import Any
"""
MemorySystem - Unified Memory Facade
=====================================

Single authoritative entry point for ALL memory operations.

Consolidates the overlapping memory modules:
- cortex.py (SwarmMemory) - 5-level storage + retrieval
- memory_orchestrator.py (SimpleBrain) - user-friendly API + consolidation triggers
- consolidation_engine.py (BrainStateMachine) - hippocampal extraction + SWR
- consolidation.py - DSPy signatures for pattern extraction
- fallback_memory.py - lightweight fallback when brain is unavailable
- llm_rag.py - LLM-based retrieval + deduplication

This facade provides:
1. A simple API (store / retrieve / consolidate)
2. Automatic backend selection (full brain vs fallback)
3. Observability integration (tracing spans for memory ops)
4. Single configuration point

Usage:
    from Jotty.core.memory.memory_system import MemorySystem

    # Zero-config (recommended)
    memory = MemorySystem()

    # Store experience
    memory.store("Task X succeeded with approach Y", level="episodic",
                 goal="research", metadata={"reward": 1.0})

    # Retrieve relevant memories
    results = memory.retrieve("How to handle task X?", goal="research", top_k=5)

    # Consolidate (episodic -> semantic -> procedural)
    memory.consolidate()

    # Get status
    print(memory.status())
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MemoryBackend(Enum):
    """Available memory backends."""
    FULL = "full"          # SwarmMemory + BrainStateMachine
    SIMPLE = "simple"      # SimpleBrain (balanced preset)
    FALLBACK = "fallback"  # FallbackMemory (dict-based, no deps)


@dataclass
class MemoryConfig:
    """Configuration for MemorySystem."""
    backend: MemoryBackend = MemoryBackend.FULL
    agent_name: str = "default"
    auto_consolidate: bool = True
    consolidation_interval: int = 3  # episodes between consolidation
    max_memories_per_level: int = 500
    enable_tracing: bool = True
    enable_mongodb: bool = False
    mongodb_uri: Optional[str] = None


@dataclass
class MemoryResult:
    """Result from a memory retrieval."""
    content: str
    level: str
    relevance: float = 0.0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.level}] {self.content[:100]}"


class MemorySystem:
    """
    Unified facade for all memory operations.

    Automatically selects the appropriate backend based on available
    dependencies and configuration. Provides a consistent API regardless
    of which backend is active.

    Architecture:
        MemorySystem (facade)
        ├── SwarmMemory (cortex.py) - 5-level storage
        ├── BrainStateMachine (consolidation_engine.py) - consolidation
        ├── SimpleBrain (memory_orchestrator.py) - user-friendly presets
        ├── LLMRAGRetriever (llm_rag.py) - semantic retrieval
        ├── DeduplicationEngine (llm_rag.py) - dedup
        └── FallbackMemory (fallback_memory.py) - graceful degradation
    """

    def __init__(self, config: Optional[MemoryConfig] = None, jotty_config: Any = None) -> None:
        """
        Initialize MemorySystem.

        Args:
            config: MemoryConfig (defaults to auto-detect best backend)
            jotty_config: SwarmConfig from Jotty framework (optional)
        """
        self.config = config or MemoryConfig()
        self._jotty_config = jotty_config
        self._backend = None
        self._backend_type: Optional[MemoryBackend] = None
        self._brain_state = None
        self._episode_count = 0
        self._store_count = 0
        self._retrieve_count = 0
        self._consolidation_count = 0
        self._init_time = time.time()

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Initialize the best available memory backend."""
        if self.config.backend == MemoryBackend.FALLBACK:
            self._init_fallback()
            return

        # Try full backend first
        if self.config.backend in (MemoryBackend.FULL, MemoryBackend.SIMPLE):
            try:
                self._init_full()
                return
            except Exception as e:
                logger.warning(f"Full memory backend unavailable: {e}")

        # Fall back
        self._init_fallback()

    def _init_full(self) -> Any:
        """Initialize full SwarmMemory backend."""
        from Jotty.core.foundation.data_structures import SwarmConfig, MemoryLevel
        from Jotty.core.foundation.configs.memory import MemoryConfig as FocusedMemoryConfig

        jc = self._jotty_config
        if isinstance(jc, FocusedMemoryConfig):
            jc = SwarmConfig.from_configs(memory=jc)
        elif jc is None:
            jc = SwarmConfig()

        from .cortex import SwarmMemory
        backend = SwarmMemory(
            agent_name=self.config.agent_name,
            config=jc,
        )

        # Verify the backend is functional (memories dict was initialized)
        if not hasattr(backend, 'memories') or MemoryLevel.EPISODIC not in backend.memories:
            raise RuntimeError("SwarmMemory failed to initialize memories dict")

        self._backend = backend
        self._backend_type = MemoryBackend.FULL

        # Initialize brain state machine for consolidation
        try:
            from .consolidation_engine import BrainStateMachine, BrainModeConfig
            brain_config = BrainModeConfig(
                sleep_interval=self.config.consolidation_interval,
            )
            self._brain_state = BrainStateMachine(brain_config)
        except Exception as e:
            logger.debug(f"BrainStateMachine unavailable: {e}")

        logger.info(
            f"MemorySystem initialized: backend=FULL, "
            f"agent={self.config.agent_name}"
        )

    def _init_fallback(self) -> Any:
        """Initialize fallback memory backend."""
        from .fallback_memory import SimpleFallbackMemory
        self._backend = SimpleFallbackMemory(
            max_entries=self.config.max_memories_per_level,
        )
        self._backend_type = MemoryBackend.FALLBACK
        logger.info("MemorySystem initialized: backend=FALLBACK")

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    def store(
        self,
        content: str,
        level: str = "episodic",
        goal: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        reward: float = 0.0,
    ) -> str:
        """
        Store a memory.

        Args:
            content: Memory content text
            level: Memory level (episodic, semantic, procedural, meta, causal)
            goal: Goal context for this memory
            metadata: Additional metadata
            reward: Reward value for RL integration

        Returns:
            Memory ID
        """
        self._store_count += 1

        # Tracing is handled at the caller level if needed
        # (Removed problematic get_active_span() call)

        if self._backend_type == MemoryBackend.FULL:
            return self._store_full(content, level, goal, metadata or {}, reward)
        else:
            return self._store_fallback(content, level, metadata or {})

    def _store_full(self, content: Any, level: Any, goal: Any, metadata: Any, reward: Any) -> Any:
        """Store using SwarmMemory."""
        from Jotty.core.foundation.data_structures import MemoryLevel

        level_map = {
            'episodic': MemoryLevel.EPISODIC,
            'semantic': MemoryLevel.SEMANTIC,
            'procedural': MemoryLevel.PROCEDURAL,
            'meta': MemoryLevel.META,
            'causal': MemoryLevel.CAUSAL,
        }
        mem_level = level_map.get(level, MemoryLevel.EPISODIC)

        memory_id = self._backend.store(
            content=content,
            level=mem_level,
            goal=goal,
            context=metadata,  # SwarmMemory uses 'context' not 'metadata'
        )

        # Auto-consolidate check (store_count already incremented in store())
        if (self.config.auto_consolidate
                and self._store_count % (self.config.consolidation_interval * 5) == 0):
            self.consolidate()

        return memory_id or f"mem_{self._store_count}"

    def _store_fallback(self, content: Any, level: Any, metadata: Any) -> Any:
        """Store using SimpleFallbackMemory."""
        from .fallback_memory import MemoryType as FallbackMemoryType

        type_map = {
            'episodic': FallbackMemoryType.EPISODIC,
            'semantic': FallbackMemoryType.SEMANTIC,
            'procedural': FallbackMemoryType.PROCEDURAL,
            # Meta and Causal don't exist in fallback, map to semantic
            'meta': FallbackMemoryType.SEMANTIC,
            'causal': FallbackMemoryType.SEMANTIC,
        }
        mem_type = type_map.get(level, FallbackMemoryType.EPISODIC)

        return self._backend.store(
            content=content,
            memory_type=mem_type,
            metadata=metadata,
        )

    def retrieve(
        self,
        query: str,
        goal: str = "",
        top_k: int = 5,
        level: Optional[str] = None,
    ) -> List[MemoryResult]:
        """
        Retrieve relevant memories.

        Args:
            query: Search query
            goal: Goal context for relevance scoring
            top_k: Maximum number of results
            level: Filter by memory level (optional)

        Returns:
            List of MemoryResult
        """
        self._retrieve_count += 1

        if self._backend_type == MemoryBackend.FULL:
            return self._retrieve_full(query, goal, top_k, level)
        else:
            return self._retrieve_fallback(query, top_k)

    def _retrieve_full(self, query: Any, goal: Any, top_k: Any, level: Any) -> List:
        """Retrieve using SwarmMemory."""
        try:
            # SwarmMemory.retrieve_fast() supports top_k directly
            # SwarmMemory.retrieve() uses budget_tokens, not top_k
            results = self._backend.retrieve_fast(
                query=query,
                goal=goal,
                budget_tokens=top_k * 500,  # estimate ~500 tokens per memory
                top_k=top_k,
            )
            return [
                MemoryResult(
                    content=getattr(r, 'content', str(r)),
                    level=getattr(r, 'level', 'unknown') if not hasattr(r, 'level') or not hasattr(r.level, 'value') else r.level.value,
                    relevance=getattr(r, 'relevance', 0.0),
                    timestamp=getattr(r, 'timestamp', 0.0),
                    metadata=getattr(r, 'metadata', {}),
                )
                for r in (results if isinstance(results, list) else [])
            ]
        except Exception as e:
            logger.warning(f"Full retrieval failed: {e}")
            return []

    def _retrieve_fallback(self, query: Any, top_k: Any) -> List:
        """Retrieve using FallbackMemory."""
        try:
            results = self._backend.retrieve(query=query, top_k=top_k)
            return [
                MemoryResult(
                    content=r.get('content', str(r)) if isinstance(r, dict) else str(r),
                    level=r.get('memory_type', 'episodic') if isinstance(r, dict) else 'episodic',
                    relevance=r.get('relevance', 0.0) if isinstance(r, dict) else 0.0,
                )
                for r in (results if isinstance(results, list) else [])
            ]
        except Exception as e:
            logger.warning(f"Fallback retrieval failed: {e}")
            return []

    def retrieve_with_latency_budget(
        self,
        query: str,
        goal: str = "",
        budget_tokens: int = 2000,
        latency_budget_ms: Optional[float] = None,
        top_k: int = 5,
        level: Optional[str] = None,
    ) -> List[MemoryResult]:
        """
        Latency-aware retrieval that auto-selects fast vs full path.

        This method delegates to the backend's latency-aware retrieval when available.
        Falls back to standard retrieval if backend doesn't support latency budgets.

        Args:
            query: Search query
            goal: Goal context for relevance scoring
            budget_tokens: Token budget for retrieved memories
            latency_budget_ms: Latency budget in milliseconds (None = no constraint)
            top_k: Maximum number of results
            level: Filter by memory level (optional)

        Returns:
            List of MemoryResult
        """
        if self._backend_type == MemoryBackend.FULL:
            try:
                # SwarmMemory supports retrieve_with_latency_budget
                from Jotty.core.foundation.data_structures import MemoryLevel

                levels = None
                if level:
                    level_map = {
                        'episodic': MemoryLevel.EPISODIC,
                        'semantic': MemoryLevel.SEMANTIC,
                        'procedural': MemoryLevel.PROCEDURAL,
                        'meta': MemoryLevel.META,
                        'causal': MemoryLevel.CAUSAL,
                    }
                    mem_level = level_map.get(level)
                    if mem_level:
                        levels = [mem_level]

                results = self._backend.retrieve_with_latency_budget(
                    query=query,
                    goal=goal,
                    budget_tokens=budget_tokens,
                    latency_budget_ms=latency_budget_ms,
                    levels=levels,
                )

                return [
                    MemoryResult(
                        content=getattr(r, 'content', str(r)),
                        level=getattr(r, 'level', 'unknown') if not hasattr(r, 'level') or not hasattr(r.level, 'value') else r.level.value,
                        relevance=getattr(r, 'relevance', 0.0),
                        timestamp=getattr(r, 'timestamp', 0.0),
                        metadata=getattr(r, 'metadata', {}),
                    )
                    for r in (results if isinstance(results, list) else [])
                ]
            except Exception as e:
                logger.warning(f"Latency-aware retrieval failed: {e}, falling back to standard retrieval")
                # Fall back to standard retrieval
                return self.retrieve(query, goal, top_k, level)
        else:
            # Fallback backend doesn't support latency budgets
            return self.retrieve(query, goal, top_k, level)

    async def consolidate(self) -> Dict[str, Any]:
        """
        Consolidate memories (episodic -> semantic -> procedural -> meta).

        Triggers brain-inspired consolidation:
        - Sharp-wave ripple replay
        - Pattern extraction
        - Pruning low-value memories

        Returns:
            Dict with consolidation stats
        """
        self._consolidation_count += 1
        start = time.time()

        if self._backend_type == MemoryBackend.FULL:
            try:
                # SwarmMemory.consolidate() is async
                import asyncio
                if asyncio.iscoroutinefunction(self._backend.consolidate):
                    result = await self._backend.consolidate()
                else:
                    result = self._backend.consolidate()
                elapsed = time.time() - start
                logger.info(f"Memory consolidation complete ({elapsed:.1f}s)")
                return {
                    'success': True,
                    'duration_s': elapsed,
                    'result': result,
                    'consolidation_count': self._consolidation_count,
                }
            except Exception as e:
                logger.warning(f"Consolidation failed: {e}")
                return {'success': False, 'error': str(e)}
        else:
            # Fallback: LRU eviction (sync)
            try:
                self._backend.prune()
                return {
                    'success': True,
                    'backend': 'fallback',
                    'action': 'prune',
                }
            except Exception as e:
                return {'success': False, 'error': str(e)}

    async def record_episode(self, goal: str, result: Any, reward: float = 0.0) -> Any:
        """
        Record a complete episode for learning.

        Stores the experience and checks if consolidation is needed.

        Args:
            goal: Episode goal
            result: Episode result
            reward: Episode reward
        """
        self._episode_count += 1

        # Store as episodic memory
        content = f"Goal: {goal}\nResult: {str(result)[:500]}\nReward: {reward}"
        self.store(
            content=content,
            level="episodic",
            goal=goal,
            metadata={'reward': reward, 'episode': self._episode_count},
            reward=reward,
        )

        # Auto-consolidate based on episode count
        if (self.config.auto_consolidate
                and self._episode_count % self.config.consolidation_interval == 0
                and self._episode_count > 0):
            logger.info(
                f"Auto-consolidation triggered (episode {self._episode_count})"
            )
            await self.consolidate()

    def status(self) -> Dict[str, Any]:
        """
        Get memory system status.

        Returns:
            Dict with backend info, counts, and health.
        """
        result = {
            'backend': self._backend_type.value if self._backend_type else 'none',
            'agent_name': self.config.agent_name,
            'uptime_s': round(time.time() - self._init_time, 1),
            'operations': {
                'stores': self._store_count,
                'retrieves': self._retrieve_count,
                'consolidations': self._consolidation_count,
                'episodes': self._episode_count,
            },
            'auto_consolidate': self.config.auto_consolidate,
            'consolidation_interval': self.config.consolidation_interval,
        }

        # Backend-specific stats
        if self._backend_type == MemoryBackend.FULL:
            try:
                levels = {}
                for level_name, level_store in self._backend.memories.items():
                    levels[level_name.value if hasattr(level_name, 'value') else str(level_name)] = len(level_store)
                result['levels'] = levels
                result['total_memories'] = sum(levels.values())
            except Exception:
                pass
        elif self._backend_type == MemoryBackend.FALLBACK:
            try:
                result['total_memories'] = len(self._backend)
            except Exception:
                pass

        return result

    def clear(self) -> None:
        """Clear all memories (use with caution)."""
        if self._backend_type == MemoryBackend.FULL:
            from Jotty.core.foundation.data_structures import MemoryLevel
            for level in MemoryLevel:
                self._backend.memories[level].clear()
        elif hasattr(self._backend, 'clear'):
            self._backend.clear()
        self._episode_count = 0
        self._store_count = 0
        logger.info("Memory system cleared")

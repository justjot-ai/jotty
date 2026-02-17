"""
Jotty v6.0 - Enhanced Memory System
====================================

Hierarchical memory with all A-Team enhancements:
- Aristotle: Causal knowledge layer, goal-conditioned wisdom
- Shannon: Deduplication, compression, information-theoretic selection
- Dr. Agarwal: LLM-based consolidation, size-aware storage

Five levels:
1. EPISODIC - Raw experiences (fast decay)
2. SEMANTIC - Abstracted patterns (slow decay)
3. PROCEDURAL - Action sequences (medium decay)
4. META - Learning wisdom (no decay)
5. CAUSAL - Why things work (no decay) [NEW]
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re as _re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def _get_dspy() -> Any:
    import dspy

    return dspy


from Jotty.core.infrastructure.foundation.configs.memory import MemoryConfig as FocusedMemoryConfig
from Jotty.core.infrastructure.foundation.data_structures import (
    CausalLink,
    GoalHierarchy,
    GoalNode,
    GoalValue,
    MemoryEntry,
    MemoryLevel,
    StoredEpisode,
    SwarmConfig,
)

from .llm_rag import CausalExtractor, DeduplicationEngine, LLMRAGRetriever


def _ensure_swarm_config(config: Any) -> Any:
    """Accept MemoryConfig or SwarmConfig, return SwarmConfig."""
    if isinstance(config, FocusedMemoryConfig):
        return SwarmConfig.from_configs(memory=config)
    return config


_consolidation_loaded = False
_consolidation_cache: Dict[str, Any] = {}


def _get_consolidation() -> Dict[str, Any]:
    """Lazily load DSPy consolidation classes from consolidation_engine."""
    global _consolidation_loaded, _consolidation_cache
    if not _consolidation_loaded:
        from .consolidation_engine import (
            ConsolidationValidationSignature,
            ConsolidationValidator,
            MemoryCluster,
            MemoryLevelClassificationSignature,
            MemoryLevelClassifier,
            MetaWisdomSignature,
            PatternExtractionSignature,
            ProceduralExtractionSignature,
        )

        _consolidation_cache.update(
            {
                "PatternExtractionSignature": PatternExtractionSignature,
                "ProceduralExtractionSignature": ProceduralExtractionSignature,
                "MetaWisdomSignature": MetaWisdomSignature,
                "MemoryLevelClassificationSignature": MemoryLevelClassificationSignature,
                "ConsolidationValidationSignature": ConsolidationValidationSignature,
                "ConsolidationValidator": ConsolidationValidator,
                "MemoryLevelClassifier": MemoryLevelClassifier,
                "MemoryCluster": MemoryCluster,
            }
        )
        _consolidation_loaded = True
    return _consolidation_cache


class SwarmMemory:
    """
    Enhanced hierarchical memory with all A-Team features.

    Levels:
    - EPISODIC: Raw experiences, high detail, fast decay
    - SEMANTIC: Abstracted patterns, extracted via LLM
    - PROCEDURAL: Action sequences, how to do things
    - META: Learning wisdom, never decays
    - CAUSAL: Why things work, enables reasoning about new situations

    Key features:
    - LLM-based retrieval (no embeddings)
    - Goal-conditioned values with transfer
    - Causal knowledge extraction
    - Automatic consolidation
    - Deduplication
    - Protection against forgetting
    """

    def __init__(self, agent_name: str, config: Any) -> None:
        self.agent_name = agent_name
        self.config = _ensure_swarm_config(config)

        # Memory storage by level
        self.memories: Dict[MemoryLevel, Dict[str, MemoryEntry]] = {
            MemoryLevel.EPISODIC: {},
            MemoryLevel.SEMANTIC: {},
            MemoryLevel.PROCEDURAL: {},
            MemoryLevel.META: {},
            MemoryLevel.CAUSAL: {},
        }

        # Causal knowledge
        self.causal_links: Dict[str, CausalLink] = {}

        # Goal hierarchy
        self.goal_hierarchy = GoalHierarchy()

        # Capacities
        self.capacities = {
            MemoryLevel.EPISODIC: config.episodic_capacity,
            MemoryLevel.SEMANTIC: config.semantic_capacity,
            MemoryLevel.PROCEDURAL: config.procedural_capacity,
            MemoryLevel.META: config.meta_capacity,
            MemoryLevel.CAUSAL: config.causal_capacity,
        }

        # LLM components
        self.retriever = LLMRAGRetriever(config)
        self.deduplicator = DeduplicationEngine(config)
        self.causal_extractor = CausalExtractor(config)

        # LLM consolidators (lazy — DSPy loaded on first use)
        dspy = _get_dspy()
        cons = _get_consolidation()
        self.pattern_extractor = dspy.ChainOfThought(cons["PatternExtractionSignature"])
        self.procedural_extractor = dspy.ChainOfThought(cons["ProceduralExtractionSignature"])
        self.meta_extractor = dspy.ChainOfThought(cons["MetaWisdomSignature"])

        # Statistics
        self.total_accesses = 0
        self.consolidation_count = 0

    # =========================================================================
    # STORAGE
    # =========================================================================

    def store(
        self,
        content: str,
        level: MemoryLevel,
        context: Dict[str, Any],
        goal: str,
        initial_value: float = 0.5,
        causal_links: List[str] | None = None,
        domain: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> MemoryEntry:
        """
        Store a new memory with hierarchical key generation and deduplication.

        Key format: {domain}:{task_type}:{content_hash}

        A-Team Note: For information-weighted storage, use store_with_outcome()
        which automatically adjusts storage detail based on event rarity.

        Args:
            domain: Domain identifier (e.g., 'sql', 'mermaid', 'plantuml').
                    If None, extracted from context['domain'] or defaults to 'general'
            task_type: Task type (e.g., 'date_filter', 'sequence_diagram').
                      If None, extracted from context['task_type'] or context['operation_type'] or defaults to 'general'
        """
        # Check size limit
        token_count = len(content) // 4 + 1
        if token_count > self.config.max_entry_tokens:
            # Chunk and store first chunk only (with note)
            content = content[: self.config.max_entry_tokens * 4]
            content += "\n[TRUNCATED - Original was longer]"
            token_count = self.config.max_entry_tokens

        # Extract domain/task_type from context if not provided (backward compatible)
        if domain is None:
            domain = context.get("domain", "general")
        if task_type is None:
            task_type = context.get("task_type") or context.get("operation_type", "general")

        # Generate hierarchical key: {domain}:{task_type}:{content_hash}
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        new_key = f"{domain}:{task_type}:{content_hash}"

        # Check for existing memory (new format - fast exact match)
        if level in self.memories and new_key in self.memories[level]:
            existing = self.memories[level][new_key]
            existing.access_count += 1
            existing.last_accessed = datetime.now()
            return existing

        # Check for old-format keys with same content (migration)
        # Two-pass approach: find first, then mutate (avoids RuntimeError from
        # modifying dict during iteration).
        if level in self.memories:
            migrate_old_key = None
            for old_key, old_entry in self.memories[level].items():
                # Old format: no colons (just hash)
                if ":" not in old_key:
                    old_hash = hashlib.md5(old_entry.content.encode()).hexdigest()[:16]
                    if old_hash == content_hash:
                        migrate_old_key = old_key
                        break  # Found match, stop iterating

            if migrate_old_key is not None:
                old_entry = self.memories[level].pop(migrate_old_key)
                old_entry.key = new_key
                # Update metadata with domain/task_type
                if not hasattr(old_entry, "metadata") or old_entry.metadata is None:
                    old_entry.metadata = {}
                old_entry.metadata["domain"] = domain
                old_entry.metadata["task_type"] = task_type
                self.memories[level][new_key] = old_entry
                old_entry.access_count += 1
                old_entry.last_accessed = datetime.now()
                return old_entry

        # Create new entry
        entry = MemoryEntry(
            key=new_key,
            content=content,
            level=level,
            context=context,
            token_count=token_count,
            source_episode=context.get("episode", 0),
            source_agent=self.agent_name,
            causal_links=causal_links or [],
        )

        # Store domain/task_type in metadata for easy filtering
        if not hasattr(entry, "metadata") or entry.metadata is None:
            entry.metadata = {}
        entry.metadata["domain"] = domain
        entry.metadata["task_type"] = task_type

        # Set goal-conditioned value
        entry.goal_values[goal] = GoalValue(value=initial_value)
        entry.default_value = initial_value

        # Check for semantic duplicates (preserves existing deduplication behavior)
        if self.config.enable_deduplication:
            existing = list(self.memories[level].values())
            for existing_mem in existing:  # Check recent only for efficiency
                is_dup, sim, merged = self.deduplicator.check_duplicate(entry, existing_mem)
                if is_dup:
                    # Found semantic duplicate - merge into existing
                    # Update existing memory's key to new format if different
                    if existing_mem.key != new_key:
                        # Migrate to new key format
                        del self.memories[level][existing_mem.key]
                        existing_mem.key = new_key
                        if not hasattr(existing_mem, "metadata") or existing_mem.metadata is None:
                            existing_mem.metadata = {}
                        existing_mem.metadata["domain"] = domain
                        existing_mem.metadata["task_type"] = task_type
                        self.memories[level][new_key] = existing_mem

                    existing_mem.content = merged
                    existing_mem.similar_entries.append(new_key)
                    return existing_mem

        # Enforce capacity
        self._enforce_capacity(level)

        # Store
        self.memories[level][new_key] = entry

        # Update goal hierarchy
        if self.config.enable_goal_hierarchy:
            op_type = context.get("operation_type", task_type)
            entities = context.get("entities", [])
            self.goal_hierarchy.add_goal(goal, domain, op_type, entities)

        return entry

    def store_with_outcome(
        self,
        content: str,
        context: Dict[str, Any],
        goal: str,
        outcome: str = "neutral",
        domain: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> MemoryEntry:
        """
        A-Team Enhancement: Information-Weighted Storage.

        Shannon Insight: I(event) = -log P(event)

        - FAILURES are rare (hopefully) -> HIGH information -> store MORE details
        - SUCCESSES are common -> LOW information -> store LESS details
        - NEUTRAL events -> NORMAL storage
        """
        if outcome == "failure":
            enhanced_content = f""" FAILURE ANALYSIS (High Information Event)
═══════════════════════════════════════════════════
{content}

FULL CONTEXT:
{json.dumps(context, default=str, indent=2)}

MEMORY NOTE: This is a rare failure event with high information content.
Store full trace to prevent similar failures.
═══════════════════════════════════════════════════"""

            return self.store(
                content=enhanced_content,
                level=MemoryLevel.CAUSAL,
                context=context,
                goal=goal,
                initial_value=0.9,
                domain=domain,
                task_type=task_type,
            )

        elif outcome == "success":
            lines = content.split("\n")
            summary = lines[0] if lines else content

            return self.store(
                content=f" Success: {summary}",
                level=MemoryLevel.SEMANTIC,
                context=context,
                goal=goal,
                initial_value=0.4,
                domain=domain,
                task_type=task_type,
            )

        else:
            return self.store(
                content=content,
                level=MemoryLevel.EPISODIC,
                context=context,
                goal=goal,
                initial_value=0.5,
                domain=domain,
                task_type=task_type,
            )

    # =========================================================================
    # SELF-RAG — LLM-gated retrieval
    # =========================================================================

    def self_rag_retrieve(
        self,
        task: str,
        goal: str = "",
        budget_tokens: int = 1000,
    ) -> Tuple[bool, List[Any], str]:
        """Self-RAG: LLM decides whether to retrieve, what query to use, and judges relevance.

        Three-step process:
        1. Gate: LLM decides if retrieval would help (skip for simple tasks)
        2. Query: LLM formulates optimal retrieval query
        3. Judge: LLM filters results by relevance to task
        """
        skip_keywords = ["hello", "hi", "thanks", "bye", "ok"]
        if any(task.strip().lower() == kw for kw in skip_keywords):
            return False, [], "Simple greeting/acknowledgment — no retrieval needed"

        total_memories = sum(len(m) for m in self.memories.values())
        if total_memories == 0:
            return False, [], "No memories stored yet"

        results = self.retrieve(
            query=task,
            goal=goal or task,
            budget_tokens=budget_tokens,
        )

        if not results:
            return False, [], "No relevant memories found"

        filtered = []
        for entry in results:
            value = entry.get_value(goal or task) if hasattr(entry, "get_value") else 0.5
            if value >= 0.3:
                filtered.append(entry)

        if not filtered:
            return False, [], "Retrieved memories below relevance threshold"

        return True, filtered, f"Retrieved {len(filtered)} relevant memories"

    # =========================================================================
    # SURPRISE-BASED MEMORY
    # =========================================================================

    def store_with_surprise(
        self,
        content: str,
        surprise_score: float,
        context: Dict[str, Any],
        goal: str = "",
        domain: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Optional[MemoryEntry]:
        """Store content with detail level proportional to surprise.

        Surprise score (0-1):
        - 0.0-0.3: Routine — skip or store minimal summary
        - 0.3-0.7: Notable — store normal detail in EPISODIC
        - 0.7-1.0: Surprising — store full detail in CAUSAL with high value
        """
        surprise_score = max(0.0, min(1.0, surprise_score))

        if surprise_score < 0.3:
            logger.debug(f"Skipping routine event (surprise={surprise_score:.2f})")
            return None

        if surprise_score >= 0.7:
            return self.store(
                content=content,
                level=MemoryLevel.CAUSAL,
                context={**context, "surprise_score": surprise_score},
                goal=goal,
                initial_value=0.8 + (surprise_score - 0.7),
                domain=domain,
                task_type=task_type,
            )

        return self.store(
            content=content,
            level=MemoryLevel.EPISODIC,
            context={**context, "surprise_score": surprise_score},
            goal=goal,
            initial_value=0.4 + surprise_score,
            domain=domain,
            task_type=task_type,
        )

    def _enforce_capacity(self, level: MemoryLevel) -> Any:
        """Ensure level doesn't exceed capacity."""
        capacity = self.capacities[level]
        memories = self.memories[level]

        while len(memories) >= capacity:
            candidates = [(k, m) for k, m in memories.items() if not m.is_protected]

            if not candidates:
                oldest = min(memories.values(), key=lambda m: m.created_at)
                del memories[oldest.key]
            else:
                to_remove = min(candidates, key=lambda x: x[1].default_value)
                del memories[to_remove[0]]

    # =========================================================================
    # FAST RETRIEVAL (no LLM call — keyword + recency + value only)
    # =========================================================================

    def retrieve_fast(
        self,
        query: str,
        goal: str,
        budget_tokens: int,
        top_k: int = 10,
        levels: List[MemoryLevel] | None = None,
    ) -> List[MemoryEntry]:
        """
        Fast retrieval using keyword overlap + recency + value ranking.

        Zero LLM calls. Designed for pre-execution intelligence reads
        in the learning pipeline where latency matters more than recall.

        Scoring formula per memory:
          score = 0.4 * keyword_overlap + 0.3 * recency + 0.3 * value

        Returns up to `top_k` memories within budget_tokens.
        """
        if levels is None:
            levels = list(MemoryLevel)

        all_memories: List[MemoryEntry] = []
        for level in levels:
            all_memories.extend(self.memories[level].values())
        if not all_memories:
            return []

        # Build query keyword set (lowercased, split on whitespace/punct)
        query_words: Set[str] = set(w for w in _re.split(r"[\s\W]+", query.lower()) if len(w) > 2)
        if not query_words:
            # Fallback: just take most recent memories
            all_memories.sort(
                key=lambda m: m.last_accessed or m.created_at or datetime.min,
                reverse=True,
            )
            selected = []
            tokens = 0
            for m in all_memories[:top_k]:
                if tokens + m.token_count <= budget_tokens:
                    selected.append(m)
                    tokens += m.token_count
            return selected

        # Score every memory cheaply
        now = datetime.now()
        scored = []
        for mem in all_memories:
            # 1. Keyword overlap (Jaccard-like)
            content_text = (mem.content or "").lower()
            mem_words = set(w for w in _re.split(r"[\s\W]+", content_text) if len(w) > 2)
            if mem_words:
                overlap = len(query_words & mem_words) / len(query_words)
            else:
                overlap = 0.0

            # 2. Recency: exponential decay, half-life = 24h
            age_hours = 24.0  # default neutral
            ts = mem.last_accessed or mem.created_at
            if ts:
                try:
                    delta = (now - ts).total_seconds() / 3600.0
                    age_hours = max(0.01, delta)
                except Exception:
                    pass
            recency = math.exp(-0.029 * age_hours)  # half-life ~24h

            # 3. Value for this goal
            value = mem.get_value(goal) if goal else mem.default_value

            # Combined score
            score = 0.4 * overlap + 0.3 * recency + 0.3 * value
            scored.append((mem, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Budget-aware selection
        selected = []
        tokens_used = 0
        for mem, _score in scored[: top_k * 2]:
            if tokens_used + mem.token_count <= budget_tokens:
                selected.append(mem)
                tokens_used += mem.token_count
            if len(selected) >= top_k:
                break

        # Update access tracking
        self.total_accesses += 1
        for mem in selected:
            mem.access_count += 1
            mem.ucb_visits += 1
            mem.last_accessed = now

        return selected

    # =========================================================================
    # FULL RETRIEVAL (LLM-scored — high quality, expensive)
    # =========================================================================

    def retrieve(
        self,
        query: str,
        goal: str,
        budget_tokens: int,
        levels: List[MemoryLevel] | None = None,
        context_hints: str = "",
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant memories using LLM-based RAG.

        No embeddings - uses keyword pre-filter + LLM scoring.
        """
        if levels is None:
            levels = list(MemoryLevel)

        # Collect all candidates
        all_memories = []
        for level in levels:
            all_memories.extend(self.memories[level].values())

        if not all_memories:
            return []

        # Use LLM RAG retriever
        selected = self.retriever.retrieve(
            query=query,
            goal=goal,
            memories=all_memories,
            budget_tokens=budget_tokens,
            goal_hierarchy=self.goal_hierarchy if self.config.enable_goal_hierarchy else None,
            context_hints=context_hints,
        )

        # Update access tracking
        self.total_accesses += 1
        for mem in selected:
            mem.access_count += 1
            mem.ucb_visits += 1
            mem.last_accessed = datetime.now()

        return selected  # type: ignore[no-any-return]

    async def retrieve_async(
        self,
        query: str,
        goal: str,
        budget_tokens: int,
        levels: List[MemoryLevel] | None = None,
        context_hints: str = "",
    ) -> List[MemoryEntry]:
        """Async version of retrieve() for parallel memory retrieval."""
        import asyncio

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: self.retrieve(query, goal, budget_tokens, levels, context_hints)
        )
        return result

    # =========================================================================
    # LATENCY-AWARE RETRIEVAL
    # =========================================================================

    def retrieve_with_latency_budget(
        self,
        query: str,
        goal: str,
        budget_tokens: int,
        latency_budget_ms: Optional[float] = None,
        levels: List[MemoryLevel] | None = None,
        context_hints: str = "",
    ) -> List[MemoryEntry]:
        """
        Latency-aware retrieval that auto-selects fast vs full path.

        - If latency_budget_ms < 2000ms -> use retrieve_fast() (50-100ms)
        - If latency_budget_ms >= 2000ms or None -> use retrieve() (500-1000ms)
        """
        if latency_budget_ms is not None and latency_budget_ms < 2000:
            logger.debug(
                f"Latency budget {latency_budget_ms}ms < 2000ms -> using retrieve_fast() "
                f"(no LLM calls, ~50-100ms)"
            )
            if latency_budget_ms < 500 and levels is None:
                levels = [MemoryLevel.EPISODIC]
                logger.debug(f"Ultra-low latency {latency_budget_ms}ms -> episodic-only search")

            return self.retrieve_fast(
                query=query, goal=goal, budget_tokens=budget_tokens, levels=levels
            )

        logger.debug(
            f"Latency budget {latency_budget_ms}ms >= 2000ms (or None) -> "
            f"using retrieve() (LLM-scored, ~500-1000ms)"
        )

        return self.retrieve(
            query=query,
            goal=goal,
            budget_tokens=budget_tokens,
            levels=levels,
            context_hints=context_hints,
        )

    def retrieve_by_domain(
        self, domain: str, goal: str, budget_tokens: int, levels: List[MemoryLevel] | None = None
    ) -> List[MemoryEntry]:
        """Retrieve memories filtered by domain using key prefix filtering."""
        if levels is None:
            levels = list(MemoryLevel)

        domain_memories = []
        for level in levels:
            if level in self.memories:
                for key, memory in self.memories[level].items():
                    if key.startswith(f"{domain}:"):
                        domain_memories.append(memory)

        if not domain_memories:
            return []

        selected = self.retriever.retrieve(
            query=f"Domain: {domain}",
            goal=goal,
            memories=domain_memories,
            budget_tokens=budget_tokens,
            goal_hierarchy=self.goal_hierarchy if self.config.enable_goal_hierarchy else None,
        )

        self.total_accesses += 1
        for mem in selected:
            mem.access_count += 1
            mem.ucb_visits += 1
            mem.last_accessed = datetime.now()

        return selected  # type: ignore[no-any-return]

    def retrieve_by_task_type(
        self, task_type: str, goal: str, budget_tokens: int, levels: List[MemoryLevel] | None = None
    ) -> List[MemoryEntry]:
        """Retrieve memories filtered by task type using key pattern matching."""
        if levels is None:
            levels = list(MemoryLevel)

        task_memories = []
        for level in levels:
            if level in self.memories:
                for key, memory in self.memories[level].items():
                    if f":{task_type}:" in key:
                        task_memories.append(memory)

        if not task_memories:
            return []

        selected = self.retriever.retrieve(
            query=f"Task type: {task_type}",
            goal=goal,
            memories=task_memories,
            budget_tokens=budget_tokens,
            goal_hierarchy=self.goal_hierarchy if self.config.enable_goal_hierarchy else None,
        )

        self.total_accesses += 1
        for mem in selected:
            mem.access_count += 1
            mem.ucb_visits += 1
            mem.last_accessed = datetime.now()

        return selected  # type: ignore[no-any-return]

    def retrieve_and_synthesize(
        self,
        query: str,
        goal: str,
        levels: List[MemoryLevel] | None = None,
        context_hints: str = "",
    ) -> str:
        """
        Brain-Inspired Synthesis Retrieval (DEFAULT mode!)

        Retrieves memories and synthesizes them into coherent wisdom.
        """
        if levels is None:
            levels = list(MemoryLevel)

        all_memories = []
        for level in levels:
            all_memories.extend(self.memories[level].values())

        if not all_memories:
            return ""

        synthesized = self.retriever.retrieve_and_synthesize(
            query=query, goal=goal, memories=all_memories, context_hints=context_hints
        )

        return synthesized  # type: ignore[no-any-return]

    async def retrieve_and_synthesize_async(
        self,
        query: str,
        goal: str,
        levels: List[MemoryLevel] | None = None,
        context_hints: str = "",
    ) -> str:
        """Async version of retrieve_and_synthesize for parallel retrieval."""
        import asyncio

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: self.retrieve_and_synthesize(query, goal, levels, context_hints)
        )
        return result

    def retrieve_for_context(
        self, query: str, goal: str, context_type: str, budget_tokens: int, context_hints: str = ""
    ) -> List[MemoryEntry]:
        """
        5-LEVEL BRAIN-INSPIRED MEMORY RETRIEVAL (Context-Aware)

        Different tasks need different memory types first:
        - VALIDATION: PROCEDURAL > META > SEMANTIC > CAUSAL > EPISODIC
        - DEBUGGING: CAUSAL > EPISODIC > SEMANTIC > PROCEDURAL > META
        - PLANNING: META > SEMANTIC > PROCEDURAL > CAUSAL > EPISODIC
        - EXPLORATION: EPISODIC > CAUSAL > SEMANTIC > PROCEDURAL > META
        - TRANSFORMATION: PROCEDURAL > SEMANTIC > EPISODIC > CAUSAL > META
        """
        from Jotty.core.infrastructure.foundation.data_structures import ContextType

        context_level_priorities = {
            ContextType.VALIDATION.value: [
                MemoryLevel.PROCEDURAL,
                MemoryLevel.META,
                MemoryLevel.SEMANTIC,
                MemoryLevel.CAUSAL,
                MemoryLevel.EPISODIC,
            ],
            ContextType.DEBUGGING.value: [
                MemoryLevel.CAUSAL,
                MemoryLevel.EPISODIC,
                MemoryLevel.SEMANTIC,
                MemoryLevel.PROCEDURAL,
                MemoryLevel.META,
            ],
            ContextType.PLANNING.value: [
                MemoryLevel.META,
                MemoryLevel.SEMANTIC,
                MemoryLevel.PROCEDURAL,
                MemoryLevel.CAUSAL,
                MemoryLevel.EPISODIC,
            ],
            ContextType.EXPLORATION.value: [
                MemoryLevel.EPISODIC,
                MemoryLevel.CAUSAL,
                MemoryLevel.SEMANTIC,
                MemoryLevel.PROCEDURAL,
                MemoryLevel.META,
            ],
            ContextType.TRANSFORMATION.value: [
                MemoryLevel.PROCEDURAL,
                MemoryLevel.SEMANTIC,
                MemoryLevel.EPISODIC,
                MemoryLevel.CAUSAL,
                MemoryLevel.META,
            ],
            ContextType.DEFAULT.value: list(MemoryLevel),
        }

        levels = context_level_priorities.get(context_type, list(MemoryLevel))
        enhanced_hints = f"CONTEXT TYPE: {context_type}\n{context_hints}"

        return self.retrieve(
            query=query,
            goal=goal,
            budget_tokens=budget_tokens,
            levels=levels,
            context_hints=enhanced_hints,
        )

    def retrieve_causal(self, query: str, context: Dict[str, Any]) -> List[CausalLink]:
        """Retrieve relevant causal knowledge."""
        if not self.config.enable_causal_learning:
            return []

        relevant = []
        query_lower = query.lower()

        for link in self.causal_links.values():
            if (
                query_lower in link.cause.lower()
                or query_lower in link.effect.lower()
                or any(
                    kw in link.cause.lower() or kw in link.effect.lower()
                    for kw in query_lower.split()
                    if len(kw) > 3
                )
            ):
                if link.applies_in_context(context):
                    relevant.append(link)

        relevant.sort(key=lambda x: x.confidence, reverse=True)

        return relevant

    # =========================================================================
    # CONSOLIDATION
    # =========================================================================

    async def consolidate(self, episodes: List[StoredEpisode] | None = None) -> Any:
        """
        Run consolidation to extract higher-level knowledge.

        Episodic -> Semantic (patterns)
        Episodic -> Procedural (how-to)
        All -> Meta (wisdom)
        Episodes -> Causal (why)
        """
        self.consolidation_count += 1

        # 1. Cluster episodic memories by goal
        clusters = self._cluster_episodic_memories()

        # 2. Extract semantic patterns
        for cluster in clusters:
            if len(cluster.memories) >= self.config.min_cluster_size:
                await self._extract_semantic_pattern(cluster)

        # 3. Extract procedural knowledge
        if episodes:
            await self._extract_procedural(episodes)

        # 4. Extract meta wisdom
        await self._extract_meta_wisdom()

        # 5. Extract causal knowledge
        if episodes and self.config.enable_causal_learning:
            await self._extract_causal(episodes)

        # 6. Prune old episodic memories
        self._prune_episodic()

    def _cluster_episodic_memories(self) -> List:
        """Cluster episodic memories by goal signature."""
        cons = _get_consolidation()
        _MemoryCluster = cons["MemoryCluster"]
        episodic = self.memories[MemoryLevel.EPISODIC]

        goal_groups: Dict[str, List[MemoryEntry]] = defaultdict(list)

        for mem in episodic.values():
            if mem.goal_values:
                goal = next(iter(mem.goal_values.keys()))
                domain = mem.context.get("domain", "general")
                signature = f"{domain}:{goal}"
                goal_groups[signature].append(mem)

        clusters = []
        for sig, mems in goal_groups.items():
            cluster = _MemoryCluster(
                cluster_id=hashlib.md5(sig.encode()).hexdigest(), goal_signature=sig, memories=mems
            )
            cluster.compute_statistics()
            clusters.append(cluster)

        return clusters

    async def _extract_semantic_pattern(self, cluster: Any) -> Any:
        """Extract semantic pattern from episodic cluster."""
        memory_data = []
        for mem in cluster.memories:
            memory_data.append(
                {
                    "content": mem.content,
                    "value": mem.default_value,
                    "success": mem.default_value > 0.5,
                }
            )

        try:
            result = self.pattern_extractor(
                memories=json.dumps(memory_data, indent=2),
                goal_context=cluster.goal_signature,
                domain=cluster.goal_signature.split(":")[0],
            )

            confidence = float(result.confidence) if result.confidence else 0.5

            if confidence >= self.config.pattern_confidence_threshold:
                pattern_content = f"""
PATTERN: {result.pattern}

CONDITIONS: {result.conditions}

EXCEPTIONS: {result.exceptions}

DERIVED FROM: {len(cluster.memories)} episodic memories
CONFIDENCE: {confidence:.2f}
""".strip()

                sample_mem = cluster.memories[0]
                goal = next(iter(sample_mem.goal_values.keys()), "general")

                self.store(
                    content=pattern_content,
                    level=MemoryLevel.SEMANTIC,
                    context={
                        "cluster_id": cluster.cluster_id,
                        "source_count": len(cluster.memories),
                        "domain": cluster.goal_signature.split(":")[0],
                    },
                    goal=goal,
                    initial_value=cluster.avg_value,
                )

                cluster.extracted_pattern = result.pattern
                cluster.pattern_confidence = confidence

        except Exception:
            pass  # Consolidation failure is non-fatal

    async def _extract_procedural(self, episodes: List[StoredEpisode]) -> Any:
        """Extract procedural knowledge from episode trajectories."""
        successes = [ep for ep in episodes if ep.success]
        failures = [ep for ep in episodes if not ep.success]

        if len(successes) < 3 or len(failures) < 2:
            return

        task_groups: Dict[str, Tuple[List, List]] = defaultdict(lambda: ([], []))

        for ep in successes:
            domain = ep.kwargs.get("domain", "general")
            task_groups[domain][0].append(ep)

        for ep in failures:
            domain = ep.kwargs.get("domain", "general")
            task_groups[domain][1].append(ep)

        for task_type, (succ, fail) in task_groups.items():
            if len(succ) < 2:
                continue

            success_traces = self._format_traces(succ)
            failure_traces = self._format_traces(fail)

            try:
                result = self.procedural_extractor(
                    success_traces=success_traces,
                    failure_traces=failure_traces,
                    task_type=task_type,
                )

                procedure_content = f"""
PROCEDURE FOR: {task_type}

STEPS:
{result.procedure}

KEY DECISIONS:
{result.key_decisions}

ANALYSIS:
{result.reasoning}
""".strip()

                self.store(
                    content=procedure_content,
                    level=MemoryLevel.PROCEDURAL,
                    context={"task_type": task_type, "source_episodes": len(succ) + len(fail)},
                    goal=f"{task_type}_procedure",
                    initial_value=0.7,
                )

            except Exception:
                pass

    async def _extract_meta_wisdom(self) -> Any:
        """Extract meta-level wisdom about learning."""
        all_high_value = []
        all_low_value = []

        for level in [MemoryLevel.SEMANTIC, MemoryLevel.PROCEDURAL]:
            for mem in self.memories[level].values():
                if mem.default_value > 0.8:
                    all_high_value.append(mem.content)
                elif mem.default_value < 0.3:
                    all_low_value.append(mem.content)

        if len(all_high_value) < 3:
            return

        try:
            result = self.meta_extractor(
                learning_history=f"Episodes: {self.consolidation_count * 100}, High-value patterns: {len(all_high_value)}, Low-value: {len(all_low_value)}",
                failure_analysis="\n".join(all_low_value),
                success_analysis="\n".join(all_high_value),
            )

            wisdom_content = f"""
META WISDOM:
{result.wisdom}

WHEN TO APPLY:
{result.applicability}
""".strip()

            self.store(
                content=wisdom_content,
                level=MemoryLevel.META,
                context={"consolidation": self.consolidation_count},
                goal="meta_wisdom",
                initial_value=0.9,
            )

            for mem in self.memories[MemoryLevel.META].values():
                mem.is_protected = True
                mem.protection_reason = "META level"

        except Exception:
            pass

    async def _extract_causal(self, episodes: List[StoredEpisode]) -> Any:
        """Extract causal knowledge from contrasting episodes."""
        successes = [
            {"id": ep.episode_id, "query": ep.goal, "result": str(ep.actor_output)}
            for ep in episodes
            if ep.success
        ]
        failures = [
            {"id": ep.episode_id, "query": ep.goal, "result": str(ep.actor_error)}
            for ep in episodes
            if not ep.success
        ]

        if len(successes) < 3 or len(failures) < 2:
            return

        links = self.causal_extractor.extract_from_episodes(
            success_episodes=successes,
            failure_episodes=failures,
            domain=episodes[0].kwargs.get("domain", "general"),
        )

        for link_data in links:
            link_key = hashlib.md5(
                f"{link_data['cause']}{link_data['effect']}".encode()
            ).hexdigest()

            if link_key in self.causal_links:
                existing = self.causal_links[link_key]
                existing.update_confidence(True)
                existing.supporting_episodes.append(episodes[0].episode_id)
            else:
                causal_link = CausalLink(
                    cause=link_data["cause"],
                    effect=link_data["effect"],
                    confidence=link_data.get("confidence", 0.7),
                    conditions=link_data.get("conditions", []),
                    domain=episodes[0].kwargs.get("domain", "general"),
                )
                self.causal_links[link_key] = causal_link

                causal_content = f"""
CAUSAL LINK:
CAUSE: {causal_link.cause}
EFFECT: {causal_link.effect}
CONFIDENCE: {causal_link.confidence:.2f}
CONDITIONS: {', '.join(causal_link.conditions) if causal_link.conditions else 'None'}
""".strip()

                self.store(
                    content=causal_content,
                    level=MemoryLevel.CAUSAL,
                    context={"causal_key": link_key},
                    goal="causal_knowledge",
                    initial_value=causal_link.confidence,
                    causal_links=[link_key],
                )

    def _format_traces(self, episodes: List[StoredEpisode]) -> str:
        """Format episode traces for LLM."""
        traces = []
        for ep in episodes:
            trace = f"Episode {ep.episode_id}:\n"
            trace += f"  Goal: {ep.goal}\n"
            trace += f"  Steps: {len(ep.trajectory)}\n"
            trace += f"  Result: {'SUCCESS' if ep.success else 'FAILURE'}\n"
            if ep.actor_error:
                trace += f"  Error: {ep.actor_error}\n"
            traces.append(trace)
        return "\n".join(traces)

    def _prune_episodic(self) -> Any:
        """Prune old low-value episodic memories."""
        episodic = self.memories[MemoryLevel.EPISODIC]

        now = datetime.now()
        one_day_ago = now - timedelta(days=1)

        to_remove = []
        for key, mem in episodic.items():
            if mem.created_at < one_day_ago and mem.default_value < 0.3:
                to_remove.append(key)

        max_remove = len(episodic) // 5
        for key in to_remove[:max_remove]:
            del episodic[key]

    # =========================================================================
    # PROTECTION
    # =========================================================================

    def protect_high_value(self, threshold: float | None = None) -> None:
        """Mark high-value memories as protected."""
        threshold = threshold or self.config.protected_memory_threshold

        for level, memories in self.memories.items():
            for mem in memories.values():
                if mem.default_value >= threshold:
                    mem.is_protected = True
                    mem.protection_reason = f"Value >= {threshold}"
                elif level == MemoryLevel.META:
                    mem.is_protected = True
                    mem.protection_reason = "META level"
                elif level == MemoryLevel.CAUSAL:
                    mem.is_protected = True
                    mem.protection_reason = "CAUSAL knowledge"

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        data = {
            "agent_name": self.agent_name,
            "total_accesses": self.total_accesses,
            "consolidation_count": self.consolidation_count,
            "memories": {},
            "causal_links": {},
            "goal_hierarchy": {
                "nodes": {k: vars(v) for k, v in self.goal_hierarchy.nodes.items()},
                "root_id": self.goal_hierarchy.root_id,
            },
        }

        for level in MemoryLevel:
            data["memories"][level.value] = {}
            for key, mem in self.memories[level].items():
                mem_dict = {
                    "key": mem.key,
                    "content": mem.content,
                    "level": mem.level.value,
                    "context": mem.context,
                    "created_at": mem.created_at.isoformat(),
                    "last_accessed": mem.last_accessed.isoformat(),
                    "goal_values": {
                        g: {"value": gv.value, "access_count": gv.access_count}
                        for g, gv in mem.goal_values.items()
                    },
                    "default_value": mem.default_value,
                    "access_count": mem.access_count,
                    "ucb_visits": mem.ucb_visits,
                    "token_count": mem.token_count,
                    "is_protected": mem.is_protected,
                    "protection_reason": mem.protection_reason,
                    "causal_links": mem.causal_links,
                    "metadata": getattr(mem, "metadata", {}),
                }
                data["memories"][level.value][key] = mem_dict

        for key, link in self.causal_links.items():
            data["causal_links"][key] = {
                "cause": link.cause,
                "effect": link.effect,
                "confidence": link.confidence,
                "conditions": link.conditions,
                "exceptions": link.exceptions,
                "domain": link.domain,
            }

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: SwarmConfig) -> "SwarmMemory":
        """
        Deserialize from dictionary with automatic key migration.

        Migrates old-format keys (hash only) to new hierarchical format (domain:task_type:hash).
        """
        memory = cls(data["agent_name"], config)
        memory.total_accesses = data.get("total_accesses", 0)
        memory.consolidation_count = data.get("consolidation_count", 0)

        # Load memories with automatic migration
        for level_str, memories in data.get("memories", {}).items():
            level = MemoryLevel(level_str)
            for key, mem_data in memories.items():
                # Check if old format (no colons = old hash-only key)
                if ":" not in key:
                    content = mem_data["content"]
                    domain = mem_data.get("context", {}).get("domain", "general")
                    task_type = mem_data.get("context", {}).get("task_type") or mem_data.get(
                        "context", {}
                    ).get("operation_type", "general")

                    content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
                    new_key = f"{domain}:{task_type}:{content_hash}"

                    mem_data["key"] = new_key
                    key = new_key

                    if "metadata" not in mem_data:
                        mem_data["metadata"] = {}
                    mem_data["metadata"]["domain"] = domain
                    mem_data["metadata"]["task_type"] = task_type

                entry = MemoryEntry(
                    key=mem_data["key"],
                    content=mem_data["content"],
                    level=level,
                    context=mem_data["context"],
                    created_at=datetime.fromisoformat(mem_data["created_at"]),
                    last_accessed=datetime.fromisoformat(mem_data["last_accessed"]),
                    default_value=mem_data["default_value"],
                    access_count=mem_data["access_count"],
                    ucb_visits=mem_data["ucb_visits"],
                    token_count=mem_data["token_count"],
                    is_protected=mem_data.get("is_protected", False),
                    protection_reason=mem_data.get("protection_reason", ""),
                    causal_links=mem_data.get("causal_links", []),
                )

                if "metadata" in mem_data:
                    entry.metadata = mem_data["metadata"]

                for goal, gv_data in mem_data.get("goal_values", {}).items():
                    entry.goal_values[goal] = GoalValue(
                        value=gv_data["value"], access_count=gv_data["access_count"]
                    )

                memory.memories[level][key] = entry

        # Load causal links
        for key, link_data in data.get("causal_links", {}).items():
            memory.causal_links[key] = CausalLink(
                cause=link_data["cause"],
                effect=link_data["effect"],
                confidence=link_data["confidence"],
                conditions=link_data.get("conditions", []),
                exceptions=link_data.get("exceptions", []),
                domain=link_data.get("domain", "general"),
            )

        # Load goal hierarchy
        gh_data = data.get("goal_hierarchy", {})
        memory.goal_hierarchy.root_id = gh_data.get("root_id", "root")
        for node_id, node_data in gh_data.get("nodes", {}).items():
            memory.goal_hierarchy.nodes[node_id] = GoalNode(
                goal_id=node_data["goal_id"],
                goal_text=node_data["goal_text"],
                parent_id=node_data.get("parent_id"),
                children_ids=node_data.get("children_ids", []),
                domain=node_data.get("domain", "general"),
                operation_type=node_data.get("operation_type", "query"),
                entities=node_data.get("entities", []),
                episode_count=node_data.get("episode_count", 0),
                success_rate=node_data.get("success_rate", 0.5),
            )

        return memory

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            "total_memories": sum(len(m) for m in self.memories.values()),
            "by_level": {level.value: len(mems) for level, mems in self.memories.items()},
            "total_accesses": self.total_accesses,
            "consolidation_count": self.consolidation_count,
            "causal_links": len(self.causal_links),
            "unique_goals": len(self.goal_hierarchy.nodes),
            "protected_memories": sum(
                1 for mems in self.memories.values() for m in mems.values() if m.is_protected
            ),
        }
        return stats

    def get_consolidated_knowledge(self, goal: str | None = None, max_items: int = 10) -> str:
        """
        Get consolidated knowledge to inject into prompts.

        Returns natural language lessons from semantic/procedural/meta/causal levels.
        """
        consolidated = []

        # Semantic patterns
        semantic_mems = self.memories.get(MemoryLevel.SEMANTIC, {})
        if semantic_mems:
            if goal:
                sorted_semantic = sorted(
                    semantic_mems.values(), key=lambda m: m.get_value(goal), reverse=True
                )
            else:
                sorted_semantic = sorted(
                    semantic_mems.values(), key=lambda m: m.access_count, reverse=True
                )

            for mem in sorted_semantic[: max_items // 2]:
                consolidated.append(("PATTERN", mem.content, mem.get_value(goal) if goal else 0.5))

        # Procedural knowledge
        procedural_mems = self.memories.get(MemoryLevel.PROCEDURAL, {})
        if procedural_mems:
            sorted_procedural = sorted(
                procedural_mems.values(), key=lambda m: m.access_count, reverse=True
            )
            for mem in sorted_procedural[: max_items // 4]:
                consolidated.append(("PROCEDURE", mem.content, 0.5))

        # Meta wisdom
        meta_mems = self.memories.get(MemoryLevel.META, {})
        if meta_mems:
            for mem in list(meta_mems.values())[: max_items // 4]:
                consolidated.append(("WISDOM", mem.content, 0.5))

        # Causal knowledge
        if self.causal_links:
            sorted_causal = sorted(
                self.causal_links.values(), key=lambda link: link.confidence, reverse=True
            )
            for link in sorted_causal[:3]:
                causal_str = f"CAUSE: {link.cause} -> EFFECT: {link.effect}"
                if link.conditions:
                    causal_str += f" (when: {', '.join(link.conditions)})"
                consolidated.append(("CAUSAL", causal_str, link.confidence))

        if not consolidated:
            return ""

        context = "# Consolidated Knowledge (Long-Term Memory):\n"

        patterns = [c for c in consolidated if c[0] == "PATTERN"]
        procedures = [c for c in consolidated if c[0] == "PROCEDURE"]
        wisdom = [c for c in consolidated if c[0] == "WISDOM"]
        causal = [c for c in consolidated if c[0] == "CAUSAL"]

        if patterns:
            context += "\n## Learned Patterns:\n"
            for _, content, value in patterns:
                if goal and value > 0:
                    context += f"- {content[:200]}... (value: {value:.2f})\n"
                else:
                    context += f"- {content[:200]}...\n"

        if procedures:
            context += "\n## Procedural Knowledge:\n"
            for _, content, _ in procedures:
                context += f"- {content[:200]}...\n"

        if wisdom:
            context += "\n## Meta Wisdom:\n"
            for _, content, _ in wisdom:
                context += f"- {content[:200]}...\n"

        if causal:
            context += "\n## Causal Understanding (WHY things work):\n"
            for _, content, conf in causal:
                context += f"- {content} (confidence: {conf:.2f})\n"

        return context

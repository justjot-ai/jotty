"""SwarmMemory mixin — retrieval methods."""

import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

from ..foundation.data_structures import (
    MemoryEntry, MemoryLevel, GoalValue, SwarmConfig,
    GoalHierarchy, GoalNode, CausalLink, StoredEpisode
)

from .llm_rag import LLMRAGRetriever


class RetrievalMixin:
    """Mixin providing retrieval methods for SwarmMemory."""

    # =========================================================================
    # FAST RETRIEVAL (no LLM call — keyword + recency + value only)
    # =========================================================================

    def retrieve_fast(
        self,
        query: str,
        goal: str,
        budget_tokens: int,
        top_k: int = 10,
        levels: List[MemoryLevel] = None,
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
        import re as _re
        query_words: Set[str] = set(
            w for w in _re.split(r'[\s\W]+', query.lower()) if len(w) > 2
        )
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
            content_text = (mem.content or '').lower()
            mem_words = set(
                w for w in _re.split(r'[\s\W]+', content_text) if len(w) > 2
            )
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
            import math
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
        for mem, _score in scored[:top_k * 2]:
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
    
    def retrieve(self,
                 query: str,
                 goal: str,
                 budget_tokens: int,
                 levels: List[MemoryLevel] = None,
                 context_hints: str = "") -> List[MemoryEntry]:
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
            context_hints=context_hints
        )
        
        # Update access tracking
        self.total_accesses += 1
        for mem in selected:
            mem.access_count += 1
            mem.ucb_visits += 1
            mem.last_accessed = datetime.now()

        return selected

    async def retrieve_async(self,
                             query: str,
                             goal: str,
                             budget_tokens: int,
                             levels: List[MemoryLevel] = None,
                             context_hints: str = "") -> List[MemoryEntry]:
        """
        Async version of retrieve() for parallel memory retrieval.

        This allows multiple memory retrievals to run concurrently,
        dramatically reducing context building time.
        """
        import asyncio
        loop = asyncio.get_running_loop()

        # Run the synchronous retrieve in a thread pool to avoid blocking
        result = await loop.run_in_executor(
            None,
            lambda: self.retrieve(query, goal, budget_tokens, levels, context_hints)
        )
        return result

    def retrieve_by_domain(self,
                          domain: str,
                          goal: str,
                          budget_tokens: int,
                          levels: List[MemoryLevel] = None) -> List[MemoryEntry]:
        """
        Retrieve memories filtered by domain.
        
        Uses key prefix filtering for fast domain-based retrieval.
        
        Args:
            domain: Domain identifier (e.g., 'sql', 'mermaid', 'plantuml')
            goal: Goal for value-based ranking
            budget_tokens: Token budget for retrieval
            levels: Memory levels to search (default: all levels)
        
        Returns:
            List of MemoryEntry objects from specified domain, ranked by value
        """
        if levels is None:
            levels = list(MemoryLevel)
        
        # Collect memories from specified levels, filtered by domain prefix
        domain_memories = []
        for level in levels:
            if level in self.memories:
                # Filter by domain prefix (fast key-level filtering)
                for key, memory in self.memories[level].items():
                    if key.startswith(f'{domain}:'):
                        domain_memories.append(memory)
        
        if not domain_memories:
            return []
        
        # Use existing retriever for ranking by value and relevance
        selected = self.retriever.retrieve(
            query=f"Domain: {domain}",
            goal=goal,
            memories=domain_memories,
            budget_tokens=budget_tokens,
            goal_hierarchy=self.goal_hierarchy if self.config.enable_goal_hierarchy else None
        )
        
        # Update access tracking
        self.total_accesses += 1
        for mem in selected:
            mem.access_count += 1
            mem.ucb_visits += 1
            mem.last_accessed = datetime.now()
        
        return selected

    def retrieve_by_task_type(self,
                              task_type: str,
                              goal: str,
                              budget_tokens: int,
                              levels: List[MemoryLevel] = None) -> List[MemoryEntry]:
        """
        Retrieve memories filtered by task type.
        
        Uses key pattern matching for fast task-type-based retrieval.
        
        Args:
            task_type: Task type (e.g., 'date_filter', 'sequence_diagram')
            goal: Goal for value-based ranking
            budget_tokens: Token budget for retrieval
            levels: Memory levels to search (default: all levels)
        
        Returns:
            List of MemoryEntry objects for specified task type, ranked by value
        """
        if levels is None:
            levels = list(MemoryLevel)
        
        # Collect memories from specified levels, filtered by task type pattern
        task_memories = []
        for level in levels:
            if level in self.memories:
                # Filter by task type pattern (second part of key: domain:task_type:hash)
                for key, memory in self.memories[level].items():
                    if f':{task_type}:' in key:
                        task_memories.append(memory)
        
        if not task_memories:
            return []
        
        # Use existing retriever for ranking by value and relevance
        selected = self.retriever.retrieve(
            query=f"Task type: {task_type}",
            goal=goal,
            memories=task_memories,
            budget_tokens=budget_tokens,
            goal_hierarchy=self.goal_hierarchy if self.config.enable_goal_hierarchy else None
        )
        
        # Update access tracking
        self.total_accesses += 1
        for mem in selected:
            mem.access_count += 1
            mem.ucb_visits += 1
            mem.last_accessed = datetime.now()
        
        return selected

    def retrieve_and_synthesize(self,
                                 query: str,
                                 goal: str,
                                 levels: List[MemoryLevel] = None,
                                 context_hints: str = "") -> str:
        """
         Brain-Inspired Synthesis Retrieval (DEFAULT mode!)

        Retrieves memories and synthesizes them into coherent wisdom.
        This is MORE intelligent than discrete retrieval:
        - Finds emergent patterns
        - Resolves contradictions
        - Creates causal insights
        - Returns integrated schema

        Returns:
            Synthesized wisdom as text (not discrete memories!)
        """
        if levels is None:
            levels = list(MemoryLevel)

        # Collect all candidates
        all_memories = []
        for level in levels:
            all_memories.extend(self.memories[level].values())

        if not all_memories:
            return ""

        # Use retriever to synthesize
        synthesized = self.retriever.retrieve_and_synthesize(
            query=query,
            goal=goal,
            memories=all_memories,
            context_hints=context_hints
        )

        return synthesized

    async def retrieve_and_synthesize_async(self,
                                             query: str,
                                             goal: str,
                                             levels: List[MemoryLevel] = None,
                                             context_hints: str = "") -> str:
        """
        Async version of retrieve_and_synthesize for parallel retrieval.
        """
        import asyncio
        loop = asyncio.get_running_loop()

        # Run the synchronous synthesis in a thread pool
        result = await loop.run_in_executor(
            None,
            lambda: self.retrieve_and_synthesize(query, goal, levels, context_hints)
        )
        return result

    def retrieve_for_context(
        self,
        query: str,
        goal: str,
        context_type: str,
        budget_tokens: int,
        context_hints: str = ""
    ) -> List[MemoryEntry]:
        """
        A-Team Enhancement: Context-aware memory retrieval.
        
        Different context types prioritize different memory levels:
        - validation: PROCEDURAL > META > SEMANTIC (how-to first)
        - debugging: CAUSAL > EPISODIC > SEMANTIC (why first)
        - planning: META > SEMANTIC > PROCEDURAL (wisdom first)
        - exploration: EPISODIC > CAUSAL > SEMANTIC (examples first)
        - transformation: PROCEDURAL > SEMANTIC > EPISODIC (steps first)
        
        Usage:
            memories = memory.retrieve_for_context(
                query="How to map bank_code column?",
                goal="column_mapping",
                context_type="transformation",
                budget_tokens=5000
            )
        """
        from ..foundation.data_structures import ContextType
        
        # Context-specific level priorities
        context_level_priorities = {
            ContextType.VALIDATION.value: [
                MemoryLevel.PROCEDURAL, MemoryLevel.META, MemoryLevel.SEMANTIC,
                MemoryLevel.CAUSAL, MemoryLevel.EPISODIC
            ],
            ContextType.DEBUGGING.value: [
                MemoryLevel.CAUSAL, MemoryLevel.EPISODIC, MemoryLevel.SEMANTIC,
                MemoryLevel.PROCEDURAL, MemoryLevel.META
            ],
            ContextType.PLANNING.value: [
                MemoryLevel.META, MemoryLevel.SEMANTIC, MemoryLevel.PROCEDURAL,
                MemoryLevel.CAUSAL, MemoryLevel.EPISODIC
            ],
            ContextType.EXPLORATION.value: [
                MemoryLevel.EPISODIC, MemoryLevel.CAUSAL, MemoryLevel.SEMANTIC,
                MemoryLevel.PROCEDURAL, MemoryLevel.META
            ],
            ContextType.TRANSFORMATION.value: [
                MemoryLevel.PROCEDURAL, MemoryLevel.SEMANTIC, MemoryLevel.EPISODIC,
                MemoryLevel.CAUSAL, MemoryLevel.META
            ],
            ContextType.DEFAULT.value: list(MemoryLevel)
        }
        
        # Get prioritized levels
        levels = context_level_priorities.get(context_type, list(MemoryLevel))
        
        # Add context type hint
        enhanced_hints = f"CONTEXT TYPE: {context_type}\n{context_hints}"
        
        return self.retrieve(
            query=query,
            goal=goal,
            budget_tokens=budget_tokens,
            levels=levels,
            context_hints=enhanced_hints
        )
    
    def retrieve_causal(self, query: str, context: Dict[str, Any]) -> List[CausalLink]:
        """
        Retrieve relevant causal knowledge.
        
        Returns causal links that apply in the given context.
        """
        if not self.config.enable_causal_learning:
            return []
        
        relevant = []
        
        # Keyword matching on cause/effect
        query_lower = query.lower()
        
        for link in self.causal_links.values():
            # Check if cause or effect matches query
            if (query_lower in link.cause.lower() or 
                query_lower in link.effect.lower() or
                any(kw in link.cause.lower() or kw in link.effect.lower() 
                    for kw in query_lower.split() if len(kw) > 3)):
                
                # Check if conditions apply
                if link.applies_in_context(context):
                    relevant.append(link)
        
        # Sort by confidence
        relevant.sort(key=lambda x: x.confidence, reverse=True)
        
        return relevant # NO LIMIT - FULL content
    

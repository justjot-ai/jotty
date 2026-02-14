"""SwarmMemory mixin — retrieval methods."""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

from ..foundation.data_structures import (
    MemoryEntry, MemoryLevel, GoalValue, SwarmConfig,
    GoalHierarchy, GoalNode, CausalLink, StoredEpisode
)



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
        5-LEVEL BRAIN-INSPIRED MEMORY RETRIEVAL (Context-Aware)

        BRAIN ANALOGY:
        Just like the human brain doesn't access all memory equally in all situations,
        this system prioritizes different memory levels based on what you're doing:

        MEMORY LEVELS (brain-inspired hierarchy):
        1. EPISODIC: Specific past experiences ("I tried X and got Y")
           - Like remembering "last Tuesday's meeting"
           - Fast decay (3 days) - recent experiences fade
           - High detail, low abstraction

        2. SEMANTIC: General knowledge patterns ("X usually leads to Y")
           - Like knowing "Paris is the capital of France"
           - Medium decay (7 days) - knowledge persists longer
           - Abstracted patterns extracted from multiple episodes

        3. PROCEDURAL: How to do things ("Steps to achieve X")
           - Like knowing how to ride a bike
           - Medium decay (7 days) - skills stick around
           - Action sequences and workflows

        4. META: Learning wisdom ("When I see X, approach Y works best")
           - Like meta-cognition: "I learn best in the morning"
           - No decay - wisdom is permanent
           - High-level insights about learning itself

        5. CAUSAL: Why things work ("X causes Y because Z")
           - Like understanding "heat causes water to boil because..."
           - No decay - causal models are permanent
           - Enables reasoning about new situations

        CONTEXT-AWARE PRIORITIZATION:
        Different tasks need different memory types first:

        - VALIDATION: "Is this correct?"
          Priority: PROCEDURAL > META > SEMANTIC > CAUSAL > EPISODIC
          Why: Need to know the right steps (procedural) and wisdom (meta) first

        - DEBUGGING: "Why did this fail?"
          Priority: CAUSAL > EPISODIC > SEMANTIC > PROCEDURAL > META
          Why: Need to understand cause (causal) and see past failures (episodic)

        - PLANNING: "How should I approach this?"
          Priority: META > SEMANTIC > PROCEDURAL > CAUSAL > EPISODIC
          Why: Need strategic wisdom (meta) and general patterns (semantic) first

        - EXPLORATION: "What happened before?"
          Priority: EPISODIC > CAUSAL > SEMANTIC > PROCEDURAL > META
          Why: Need concrete examples (episodic) and why they worked (causal)

        - TRANSFORMATION: "How do I change X to Y?"
          Priority: PROCEDURAL > SEMANTIC > EPISODIC > CAUSAL > META
          Why: Need step-by-step instructions (procedural) and patterns (semantic)

        RETRIEVAL MECHANICS:
        1. Get context-specific level ordering (based on above priorities)
        2. Search each level in priority order until budget is filled
        3. Higher-priority levels get more "budget space"
        4. Use LLM-based relevance scoring within each level
        5. Return ranked memories that fit within token budget

        EXAMPLE:
        Query: "How to map bank_code column?"
        Context: "transformation"
        → Searches PROCEDURAL first (find exact mapping steps)
        → Then SEMANTIC (find general mapping patterns)
        → Then EPISODIC (find past bank_code mappings)
        → Returns: Concrete steps + patterns + examples

        Usage:
            memories = memory.retrieve_for_context(
                query="How to map bank_code column?",
                goal="column_mapping",
                context_type="transformation",  # Determines priority order
                budget_tokens=5000
            )
        """
        from ..foundation.data_structures import ContextType

        # =====================================================================
        # CONTEXT-SPECIFIC LEVEL PRIORITIES
        # =====================================================================
        # Each context type defines an ordered list of memory levels.
        # The retrieval system searches these levels in order, allocating
        # more "budget" to earlier (higher-priority) levels.
        #
        # This mimics how human memory works: when debugging, we naturally
        # think "why did this break?" (causal) before "what are the steps?"
        # (procedural). The brain prioritizes relevant memory types.
        # =====================================================================
        context_level_priorities = {
            ContextType.VALIDATION.value: [
                MemoryLevel.PROCEDURAL,  # First: "What are the validation steps?"
                MemoryLevel.META,        # Second: "What validation wisdom do we have?"
                MemoryLevel.SEMANTIC,    # Third: "What patterns indicate validity?"
                MemoryLevel.CAUSAL,      # Fourth: "Why does validation fail?"
                MemoryLevel.EPISODIC     # Last: "Past validation attempts"
            ],
            ContextType.DEBUGGING.value: [
                MemoryLevel.CAUSAL,      # First: "WHY did this break?" (root cause)
                MemoryLevel.EPISODIC,    # Second: "When did we see this before?" (examples)
                MemoryLevel.SEMANTIC,    # Third: "What patterns match this error?"
                MemoryLevel.PROCEDURAL,  # Fourth: "What debugging steps to try?"
                MemoryLevel.META         # Last: "General debugging wisdom"
            ],
            ContextType.PLANNING.value: [
                MemoryLevel.META,        # First: "What strategic wisdom applies?"
                MemoryLevel.SEMANTIC,    # Second: "What general approach patterns exist?"
                MemoryLevel.PROCEDURAL,  # Third: "What are the execution steps?"
                MemoryLevel.CAUSAL,      # Fourth: "Why do certain approaches work?"
                MemoryLevel.EPISODIC     # Last: "Specific past planning sessions"
            ],
            ContextType.EXPLORATION.value: [
                MemoryLevel.EPISODIC,    # First: "What concrete examples exist?"
                MemoryLevel.CAUSAL,      # Second: "Why did those examples work?"
                MemoryLevel.SEMANTIC,    # Third: "What patterns emerge?"
                MemoryLevel.PROCEDURAL,  # Fourth: "How to replicate?"
                MemoryLevel.META         # Last: "Exploration strategies"
            ],
            ContextType.TRANSFORMATION.value: [
                MemoryLevel.PROCEDURAL,  # First: "Exact transformation steps"
                MemoryLevel.SEMANTIC,    # Second: "General transformation patterns"
                MemoryLevel.EPISODIC,    # Third: "Past transformations"
                MemoryLevel.CAUSAL,      # Fourth: "Why transformations succeed/fail"
                MemoryLevel.META         # Last: "Transformation best practices"
            ],
            ContextType.DEFAULT.value: list(MemoryLevel)  # No prioritization
        }

        # =====================================================================
        # SELECT PRIORITY ORDERING
        # =====================================================================
        # Get the level ordering for this context type.
        # If context_type is unknown, fall back to searching all levels equally.
        # =====================================================================
        levels = context_level_priorities.get(context_type, list(MemoryLevel))

        # =====================================================================
        # ENHANCE CONTEXT HINTS
        # =====================================================================
        # Add context type to hints so the LLM retriever knows what we're
        # trying to do. This helps with relevance scoring.
        # Example: "CONTEXT TYPE: debugging" tells the LLM to prioritize
        # error-related memories.
        # =====================================================================
        enhanced_hints = f"CONTEXT TYPE: {context_type}\n{context_hints}"

        # =====================================================================
        # DELEGATE TO MAIN RETRIEVE METHOD
        # =====================================================================
        # The actual retrieval logic is in retrieve(), which:
        # 1. Collects memories from specified levels (in priority order)
        # 2. Uses LLM-based scoring for relevance
        # 3. Selects top-k within token budget
        # 4. Updates access tracking for learning
        # =====================================================================
        return self.retrieve(
            query=query,
            goal=goal,
            budget_tokens=budget_tokens,
            levels=levels,              # Priority-ordered levels
            context_hints=enhanced_hints  # Enhanced with context type
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
    

from __future__ import annotations

from typing import Any, List

"""
Algorithmic Foundations
=======================

Aggregates all algorithmic components:
1. Credit Assignment (Shapley, Difference Rewards)
2. Information Theory (Surprise, Information Weighting)
3. Universal Context Management (Guard, Gate)

Relocated from infrastructure/utils/ to intelligence/learning/ (its natural home).
"""

from Jotty.core.infrastructure.context.content_gate import (
    ContentGate,
    ContextChunk,
    ProcessedContent,
    RelevanceEstimator,
    with_content_gate,
)

# Backward-compatible alias
ContentChunk = ContextChunk

# Universal Context Management
# NOTE: global_context_guard module was removed; provide stubs so dependents load.
try:
    from Jotty.core.infrastructure.context.global_context_guard import (  # type: ignore[import-not-found]
        ContextOverflowInfo,
        GlobalContextGuard,
        OverflowDetector,
        patch_dspy_with_guard,
        unpatch_dspy,
    )
except ImportError:

    class ContextOverflowInfo:  # type: ignore[no-redef]
        """Stub — global_context_guard module no longer exists."""

    class GlobalContextGuard:  # type: ignore[no-redef]
        """Stub — global_context_guard module no longer exists."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def get_statistics(self) -> dict:
            return {}

        def wrap_function(self, func: Any) -> Any:
            return func

    class OverflowDetector:  # type: ignore[no-redef]
        """Stub — global_context_guard module no longer exists."""

    def patch_dspy_with_guard(*args: Any, **kwargs: Any) -> None:
        """Stub — global_context_guard module no longer exists."""

    def unpatch_dspy(*args: Any, **kwargs: Any) -> None:
        """Stub — global_context_guard module no longer exists."""


# Information Theory (Shannon) — canonical location is now intelligence/memory/
from Jotty.core.intelligence.memory.information_storage import (
    InformationTheoreticStorage,
    InformationWeightedMemory,
    SurpriseEstimator,
)

# Credit Assignment (Game Theory + MARL)
try:
    from Jotty.core.intelligence.learning.algorithmic_credit import (
        AgentContribution,
        AlgorithmicCreditAssigner,
        Coalition,
        DifferenceRewardEstimator,
        ShapleyValueEstimator,
    )
except ImportError:
    AgentContribution = None  # type: ignore[assignment,misc]
    AlgorithmicCreditAssigner = None  # type: ignore[assignment,misc]
    Coalition = None  # type: ignore[assignment,misc]
    DifferenceRewardEstimator = None  # type: ignore[assignment,misc]
    ShapleyValueEstimator = None  # type: ignore[assignment,misc]

# =============================================================================
# UNIFIED INTERFACE
# =============================================================================


class AlgorithmicReVal:
    """
    Unified interface to all algorithmic components.

    Usage:
        algo = AlgorithmicReVal(max_tokens=28000)

        # Credit assignment
        credits = await algo.assign_credit(agents, trajectory, reward)

        # Information storage
        memory = await algo.store_with_info_weighting(event, context, content)

        # Content processing
        processed = await algo.process_content(large_doc, query)
    """

    def __init__(self, max_tokens: int = 28000, config: Any = None) -> None:
        self.max_tokens = max_tokens
        self.credit_assigner = AlgorithmicCreditAssigner(config)
        self.info_storage = InformationTheoreticStorage()
        self.context_guard = GlobalContextGuard(max_tokens)
        self.content_gate = ContentGate(max_tokens)
        patch_dspy_with_guard(self.context_guard)

    async def assign_credit(
        self,
        agents: list,
        agent_capabilities: dict,
        actions: dict,
        states: dict,
        trajectory: list,
        task: str,
        global_reward: float,
    ) -> dict:
        """Assign credit using Shapley + Difference Rewards."""
        return await self.credit_assigner.assign_credit(
            agents, agent_capabilities, actions, states, trajectory, task, global_reward
        )

    async def store_with_info_weighting(self, event: dict, context: dict, raw_content: str) -> Any:
        """Store with Shannon information weighting."""
        return await self.info_storage.store(event, context, raw_content)

    async def process_content(
        self, content: str, query: str, future_tasks: list | None = None
    ) -> Any:
        """Process content through ContentGate (auto-chunk if needed)."""
        return await self.content_gate.process(content, query, future_tasks)

    def wrap_function(self, func: Any) -> Any:
        """Wrap a function with context guard."""
        return self.context_guard.wrap_function(func)

    def get_statistics(self) -> dict:
        """Get statistics from all components."""
        return {
            "context_guard": self.context_guard.get_statistics(),
            "content_gate": self.content_gate.get_statistics(),
            "info_storage": self.info_storage.get_statistics(),
        }


# =============================================================================
# SORTING ALGORITHMS (Classic Computer Science Foundations)
# =============================================================================


class SortingAlgorithms:
    """Collection of fundamental sorting algorithms."""

    @staticmethod
    def bubble_sort(arr: list, key: Any = None, reverse: Any = False) -> list:
        """Bubble Sort - Simple comparison-based sorting algorithm."""
        result = arr.copy()
        n = len(result)

        def compare_key(item: Any) -> Any:
            return key(item) if key else item

        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                left_val = compare_key(result[j])
                right_val = compare_key(result[j + 1])
                should_swap = left_val > right_val if not reverse else left_val < right_val
                if should_swap:
                    result[j], result[j + 1] = result[j + 1], result[j]
                    swapped = True
            if not swapped:
                break
        return result

    @staticmethod
    def bubble_sort_analysis(arr: list) -> dict:
        """Perform bubble sort while tracking performance metrics."""
        result = arr.copy()
        n = len(result)
        comparisons = 0
        swaps = 0
        passes = 0

        for i in range(n):
            swapped = False
            passes += 1
            for j in range(0, n - i - 1):
                comparisons += 1
                if result[j] > result[j + 1]:
                    result[j], result[j + 1] = result[j + 1], result[j]
                    swaps += 1
                    swapped = True
            if not swapped:
                break

        return {
            "sorted_array": result,
            "comparisons": comparisons,
            "swaps": swaps,
            "passes": passes,
        }


# =============================================================================
# MUTUAL INFORMATION RETRIEVER (Memory Selection)
# =============================================================================


class MutualInformationRetriever:
    """
    Retrieve memories using Mutual Information maximization.

    Instead of simple relevance, we maximize:
    I(Memory; Query) - beta x I(Memory; Already_Selected)

    This is Maximum Marginal Relevance (MMR) with information-theoretic foundation.
    """

    def __init__(self, diversity_weight: float = 0.3) -> None:
        self.diversity_weight = diversity_weight

    async def retrieve(self, memories: list, query: str, k: int = 5) -> list:
        """Retrieve k memories maximizing information content."""
        if not memories:
            return []
        if len(memories) <= k:
            return memories

        relevance_scores = {}
        for mem in memories:
            content = mem.content if hasattr(mem, "content") else str(mem)
            relevance_scores[id(mem)] = self._compute_relevance(content, query)

        selected: List[Any] = []
        remaining = list(memories)

        for _ in range(k):
            if not remaining:
                break

            best_score = float("-inf")
            best_mem = None

            for mem in remaining:
                mem_content = mem.content if hasattr(mem, "content") else str(mem)
                relevance = relevance_scores[id(mem)]
                max_sim = 0.0
                for sel_mem in selected:
                    sel_content = sel_mem.content if hasattr(sel_mem, "content") else str(sel_mem)
                    sim = self._compute_similarity(mem_content, sel_content)
                    max_sim = max(max_sim, sim)

                mmr = (1 - self.diversity_weight) * relevance - self.diversity_weight * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_mem = mem

            if best_mem:
                selected.append(best_mem)
                remaining.remove(best_mem)

        return selected

    def _compute_relevance(self, content: str, query: str) -> float:
        """Compute relevance (simple word overlap for efficiency)."""
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0
        return len(content_words & query_words) / len(query_words)

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        union = len(words1 | words2)
        if union == 0:
            return 0.0
        return len(words1 & words2) / union


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Credit Assignment
    "AgentContribution",
    "Coalition",
    "ShapleyValueEstimator",
    "DifferenceRewardEstimator",
    "AlgorithmicCreditAssigner",
    # Information Theory
    "InformationWeightedMemory",
    "SurpriseEstimator",
    "InformationTheoreticStorage",
    "MutualInformationRetriever",
    # Context Management
    "OverflowDetector",
    "ContextOverflowInfo",
    "GlobalContextGuard",
    "patch_dspy_with_guard",
    "unpatch_dspy",
    # Content Gate
    "ContentChunk",
    "ProcessedContent",
    "RelevanceEstimator",
    "ContentGate",
    "with_content_gate",
    # Sorting Algorithms
    "SortingAlgorithms",
    # Unified Interface
    "AlgorithmicReVal",
]

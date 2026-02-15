from typing import Any
"""
Jotty v7.6 - Algorithmic Foundations
=====================================

Aggregates all algorithmic components:
1. Credit Assignment (Shapley, Difference Rewards)
2. Information Theory (Surprise, Information Weighting)
3. Universal Context Management (Guard, Gate)

NO HARDCODING. Everything derived from algorithms or LLM estimation.
"""

# Credit Assignment (Game Theory + MARL)
from ..learning.algorithmic_credit import (
    AgentContribution,
    Coalition,
    ShapleyValueEstimator,
    DifferenceRewardEstimator,
    AlgorithmicCreditAssigner
)

# Information Theory (Shannon)
from ..data.information_storage import (
    InformationWeightedMemory,
    SurpriseEstimator,
    InformationTheoreticStorage
)

# Universal Context Management
from ..context.global_context_guard import (
    OverflowDetector,
    ContextOverflowInfo,
    GlobalContextGuard,
    patch_dspy_with_guard,
    unpatch_dspy
)

from ..context.content_gate import (
    ContentChunk,
    ProcessedContent,
    RelevanceEstimator,
    ContentGate,
    with_content_gate
)


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
        
        # Initialize all components
        self.credit_assigner = AlgorithmicCreditAssigner(config)
        self.info_storage = InformationTheoreticStorage()
        self.context_guard = GlobalContextGuard(max_tokens)
        self.content_gate = ContentGate(max_tokens)
        
        # Patch DSPy for pervasive protection
        patch_dspy_with_guard(self.context_guard)
    
    async def assign_credit(
        self,
        agents: list,
        agent_capabilities: dict,
        actions: dict,
        states: dict,
        trajectory: list,
        task: str,
        global_reward: float
    ) -> dict:
        """Assign credit using Shapley + Difference Rewards."""
        return await self.credit_assigner.assign_credit(
            agents, agent_capabilities, actions, states,
            trajectory, task, global_reward
        )
    
    async def store_with_info_weighting(self, event: dict, context: dict, raw_content: str) -> Any:
        """Store with Shannon information weighting."""
        return await self.info_storage.store(event, context, raw_content)
    
    async def process_content(self, content: str, query: str, future_tasks: list = None) -> Any:
        """Process content through ContentGate (auto-chunk if needed)."""
        return await self.content_gate.process(content, query, future_tasks)
    
    def wrap_function(self, func: Any) -> Any:
        """Wrap a function with context guard."""
        return self.context_guard.wrap_function(func)
    
    def get_statistics(self) -> dict:
        """Get statistics from all components."""
        return {
            'context_guard': self.context_guard.get_statistics(),
            'content_gate': self.content_gate.get_statistics(),
            'info_storage': self.info_storage.get_statistics()
        }


# =============================================================================
# SORTING ALGORITHMS (Classic Computer Science Foundations)
# =============================================================================

class SortingAlgorithms:
    """
    Collection of fundamental sorting algorithms with educational and practical value.
    All methods are static and reusable across the codebase.
    """

    @staticmethod
    def bubble_sort(arr: list, key: Any = None, reverse: Any = False) -> list:
        """
        Bubble Sort - Simple comparison-based sorting algorithm.

        Algorithm:
        - Repeatedly steps through the list
        - Compares adjacent elements and swaps them if they're in wrong order
        - Pass through the list is repeated until the list is sorted
        - Largest elements "bubble" to the end in each pass

        Time Complexity:
        - Best Case: O(n) - when array is already sorted (with optimization)
        - Average Case: O(n²)
        - Worst Case: O(n²) - when array is reverse sorted

        Space Complexity: O(1) - in-place sorting

        Stability: Stable (preserves relative order of equal elements)

        Use Cases:
        - Educational purposes (simple to understand)
        - Small datasets (< 10-20 elements)
        - Nearly sorted data (efficient with optimization)
        - When simplicity is preferred over performance

        Args:
            arr: List to sort
            key: Optional function to extract comparison key from each element
            reverse: If True, sort in descending order

        Returns:
            Sorted list (creates a copy, doesn't modify original)

        Example:
            >>> SortingAlgorithms.bubble_sort([64, 34, 25, 12, 22, 11, 90])
            [11, 12, 22, 25, 34, 64, 90]

            >>> data = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
            >>> SortingAlgorithms.bubble_sort(data, key=lambda x: x['age'])
            [{'name': 'Bob', 'age': 25}, {'name': 'Alice', 'age': 30}]
        """
        # Create a copy to avoid modifying original
        result = arr.copy()
        n = len(result)

        # Extract comparison key if provided
        def compare_key(item: Any) -> Any:
            return key(item) if key else item

        # Bubble sort with optimization (early exit if no swaps)
        for i in range(n):
            swapped = False

            # Last i elements are already in place
            for j in range(0, n - i - 1):
                # Compare adjacent elements
                left_val = compare_key(result[j])
                right_val = compare_key(result[j + 1])

                # Swap if in wrong order
                should_swap = left_val > right_val if not reverse else left_val < right_val

                if should_swap:
                    result[j], result[j + 1] = result[j + 1], result[j]
                    swapped = True

            # If no swaps occurred, array is sorted
            if not swapped:
                break

        return result

    @staticmethod
    def bubble_sort_analysis(arr: list) -> dict:
        """
        Perform bubble sort while tracking performance metrics.

        Returns:
            Dictionary with sorted array and performance stats:
            - sorted_array: The sorted result
            - comparisons: Number of comparisons made
            - swaps: Number of swaps performed
            - passes: Number of passes through the array

        Example:
            >>> result = SortingAlgorithms.bubble_sort_analysis([64, 34, 25, 12])
            >>> print(f"Comparisons: {result['comparisons']}, Swaps: {result['swaps']}")
        """
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
            'sorted_array': result,
            'comparisons': comparisons,
            'swaps': swaps,
            'passes': passes
        }


# =============================================================================
# MUTUAL INFORMATION RETRIEVER (Memory Selection)
# =============================================================================

class MutualInformationRetriever:
    """
    Retrieve memories using Mutual Information maximization.
    
    Instead of simple relevance, we maximize:
    I(Memory; Query) - β × I(Memory; Already_Selected)
    
    This is Maximum Marginal Relevance (MMR) with information-theoretic foundation.
    """
    
    def __init__(self, diversity_weight: float = 0.3) -> None:
        self.diversity_weight = diversity_weight  # β in the formula
    
    async def retrieve(
        self,
        memories: list,
        query: str,
        k: int = 5
    ) -> list:
        """
        Retrieve k memories maximizing information content.
        
        Uses MMR: argmax[λ × relevance - (1-λ) × max_similarity_to_selected]
        """
        if not memories:
            return []
        
        if len(memories) <= k:
            return memories
        
        # Score relevance for all memories
        relevance_scores = {}
        for mem in memories:
            content = mem.content if hasattr(mem, 'content') else str(mem)
            relevance_scores[id(mem)] = self._compute_relevance(content, query)
        
        # Greedy selection with MMR
        selected = []
        remaining = list(memories)
        
        for _ in range(k):
            if not remaining:
                break
            
            # Compute MMR score for each remaining memory
            best_score = float('-inf')
            best_mem = None
            
            for mem in remaining:
                mem_content = mem.content if hasattr(mem, 'content') else str(mem)
                
                # Relevance term
                relevance = relevance_scores[id(mem)]
                
                # Diversity term (max similarity to already selected)
                max_sim = 0.0
                for sel_mem in selected:
                    sel_content = sel_mem.content if hasattr(sel_mem, 'content') else str(sel_mem)
                    sim = self._compute_similarity(mem_content, sel_content)
                    max_sim = max(max_sim, sim)
                
                # MMR score
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
        
        overlap = len(content_words & query_words)
        return overlap / len(query_words)
    
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
    'AgentContribution',
    'Coalition',
    'ShapleyValueEstimator',
    'DifferenceRewardEstimator',
    'AlgorithmicCreditAssigner',

    # Information Theory
    'InformationWeightedMemory',
    'SurpriseEstimator',
    'InformationTheoreticStorage',
    'MutualInformationRetriever',

    # Context Management
    'OverflowDetector',
    'ContextOverflowInfo',
    'GlobalContextGuard',
    'patch_dspy_with_guard',
    'unpatch_dspy',

    # Content Gate
    'ContentChunk',
    'ProcessedContent',
    'RelevanceEstimator',
    'ContentGate',
    'with_content_gate',

    # Sorting Algorithms
    'SortingAlgorithms',

    # Unified Interface
    'AlgorithmicReVal'
]

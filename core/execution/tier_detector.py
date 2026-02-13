"""
Tier Auto-Detection
===================

Automatically selects execution tier based on task characteristics.
"""

import logging
from typing import Optional
from .types import ExecutionTier

logger = logging.getLogger(__name__)


class TierDetector:
    """
    Detects appropriate execution tier for a task.

    Uses heuristics to determine complexity and select tier.
    Can be overridden by explicit tier specification.
    """

    # Keywords that indicate different complexity levels
    DIRECT_INDICATORS = [
        'what is', 'calculate', 'convert', 'translate',
        'define', 'explain briefly', 'simple question',
        'lookup', 'find', 'search for',
    ]

    LEARNING_INDICATORS = [
        'learn from', 'improve', 'optimize', 'remember',
        'get better at', 'track performance', 'validate',
    ]

    RESEARCH_INDICATORS = [
        'experiment', 'benchmark', 'compare approaches',
        'analyze in depth', 'research thoroughly',
        'multi-round', 'self-improve',
    ]

    MULTI_STEP_INDICATORS = [
        'and then', 'after that', 'followed by',
        'first', 'second', 'third', 'finally',
        'step 1', 'step 2',
        'analyze and', 'research and', 'create and',
        'compile and', 'gather and', 'process and',
    ]

    def __init__(self):
        self.detection_cache = {}  # Simple cache for repeated queries

    def detect(
        self,
        goal: str,
        context: Optional[dict] = None,
        force_tier: Optional[ExecutionTier] = None
    ) -> ExecutionTier:
        """
        Detect appropriate tier for the goal.

        Args:
            goal: Task description
            context: Optional context (user history, preferences)
            force_tier: Override auto-detection

        Returns:
            ExecutionTier enum value
        """
        if force_tier:
            logger.info(f"Using forced tier: {force_tier.name}")
            return force_tier

        # Check cache
        cache_key = goal.lower().strip()[:100]
        if cache_key in self.detection_cache:
            tier = self.detection_cache[cache_key]
            logger.debug(f"Tier cache hit: {tier.name}")
            return tier

        # Detect
        tier = self._detect_tier(goal, context)
        self.detection_cache[cache_key] = tier

        logger.info(f"Auto-detected tier: {tier.name} for goal: {goal[:50]}...")
        return tier

    def _detect_tier(self, goal: str, context: Optional[dict]) -> ExecutionTier:
        """Internal detection logic."""
        goal_lower = goal.lower()

        # Tier 4 (RESEARCH) - explicit research/experiment keywords
        if any(ind in goal_lower for ind in self.RESEARCH_INDICATORS):
            return ExecutionTier.RESEARCH

        # Tier 3 (LEARNING) - learning/validation keywords
        if any(ind in goal_lower for ind in self.LEARNING_INDICATORS):
            return ExecutionTier.LEARNING

        # Tier 1 (DIRECT) - simple queries
        if self._is_simple_query(goal_lower):
            return ExecutionTier.DIRECT

        # Default: Tier 2 (AGENTIC) - general multi-step tasks
        return ExecutionTier.AGENTIC

    def _is_simple_query(self, goal_lower: str) -> bool:
        """
        Check if goal is a simple query suitable for Tier 1.

        Criteria:
        - Contains direct question indicators
        - Short (< 10 words)
        - No multi-step indicators
        - No complex operations
        """
        # Check for direct indicators
        has_direct = any(ind in goal_lower for ind in self.DIRECT_INDICATORS)

        # Check word count
        word_count = len(goal_lower.split())
        is_short = word_count <= 10

        # Check for multi-step indicators
        has_multi_step = any(ind in goal_lower for ind in self.MULTI_STEP_INDICATORS)

        # Simple query: direct indicator OR short without multi-step
        return (has_direct or is_short) and not has_multi_step

    def explain_detection(self, goal: str) -> str:
        """
        Explain why a particular tier was chosen.

        Useful for debugging and user feedback.
        """
        tier = self.detect(goal)
        goal_lower = goal.lower()

        reasons = []

        if tier == ExecutionTier.DIRECT:
            if any(ind in goal_lower for ind in self.DIRECT_INDICATORS):
                reasons.append("Contains direct query keywords")
            if len(goal.split()) <= 10:
                reasons.append("Short query (≤10 words)")
            reasons.append("No multi-step indicators detected")

        elif tier == ExecutionTier.AGENTIC:
            if any(ind in goal_lower for ind in self.MULTI_STEP_INDICATORS):
                reasons.append("Contains multi-step indicators")
            reasons.append("Default tier for complex tasks")

        elif tier == ExecutionTier.LEARNING:
            if any(ind in goal_lower for ind in self.LEARNING_INDICATORS):
                reasons.append("Contains learning/validation keywords")

        elif tier == ExecutionTier.RESEARCH:
            if any(ind in goal_lower for ind in self.RESEARCH_INDICATORS):
                reasons.append("Contains research/experiment keywords")

        explanation = f"Tier {tier.value} ({tier.name}) selected:\n"
        for i, reason in enumerate(reasons, 1):
            explanation += f"  {i}. {reason}\n"

        return explanation.strip()

    def clear_cache(self):
        """Clear detection cache."""
        self.detection_cache.clear()
        logger.debug("Tier detection cache cleared")

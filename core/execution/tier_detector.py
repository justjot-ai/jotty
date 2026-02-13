"""
Tier Auto-Detection
===================

Automatically selects execution tier based on task characteristics.
Supports optional LLM fallback for ambiguous cases.
"""

import logging
from typing import Optional, Tuple
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

    AUTONOMOUS_INDICATORS = [
        'sandbox', 'isolated', 'untrusted', 'coalition', 'consensus',
        'curriculum', 'agent0', 'autonomous', 'multi-swarm',
        'byzantine', 'trust', 'install', 'execute code',
    ]

    MULTI_STEP_INDICATORS = [
        'and then', 'after that', 'followed by',
        'first', 'second', 'third', 'finally',
        'step 1', 'step 2',
        'analyze and', 'research and', 'create and',
        'compile and', 'gather and', 'process and',
    ]

    def __init__(self, enable_llm_fallback: bool = False):
        self.detection_cache = {}  # Simple cache for repeated queries
        self._enable_llm_fallback = enable_llm_fallback
        self._llm_classifier = None

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
        tier, _ = self._detect_tier_with_confidence(goal, context)
        return tier

    def _detect_tier_with_confidence(
        self, goal: str, context: Optional[dict] = None
    ) -> Tuple[ExecutionTier, float]:
        """Detect tier with a confidence score.

        Returns:
            (tier, confidence) where confidence 0.0-1.0 indicates how
            certain the heuristic match is. High-confidence keyword matches
            get 0.85, clear simple queries get 0.80, and ambiguous
            fall-through to AGENTIC gets 0.40.
        """
        goal_lower = goal.lower()

        # Tier 5 (AUTONOMOUS) - sandbox/coalition/trust keywords
        if any(ind in goal_lower for ind in self.AUTONOMOUS_INDICATORS):
            return ExecutionTier.AUTONOMOUS, 0.85

        # Tier 4 (RESEARCH) - explicit research/experiment keywords
        if any(ind in goal_lower for ind in self.RESEARCH_INDICATORS):
            return ExecutionTier.RESEARCH, 0.85

        # Tier 3 (LEARNING) - learning/validation keywords
        if any(ind in goal_lower for ind in self.LEARNING_INDICATORS):
            return ExecutionTier.LEARNING, 0.85

        # Tier 1 (DIRECT) - simple queries
        if self._is_simple_query(goal_lower):
            return ExecutionTier.DIRECT, 0.80

        # Check for multi-step indicators (moderate confidence)
        if any(ind in goal_lower for ind in self.MULTI_STEP_INDICATORS):
            return ExecutionTier.AGENTIC, 0.75

        # Default: Tier 2 (AGENTIC) — ambiguous, low confidence
        return ExecutionTier.AGENTIC, 0.40

    async def adetect(
        self,
        goal: str,
        context: Optional[dict] = None,
        force_tier: Optional[ExecutionTier] = None,
    ) -> ExecutionTier:
        """Async tier detection with optional LLM fallback for ambiguous cases.

        Uses the same keyword heuristics as ``detect()`` but when
        confidence is low (< 0.7) and LLM fallback is enabled, consults
        an LLM classifier for a more accurate result.
        """
        if force_tier:
            return force_tier

        # Check cache
        cache_key = goal.lower().strip()[:100]
        if cache_key in self.detection_cache:
            return self.detection_cache[cache_key]

        tier, confidence = self._detect_tier_with_confidence(goal, context)

        if confidence < 0.7 and self._enable_llm_fallback:
            try:
                if self._llm_classifier is None:
                    self._llm_classifier = _TierClassifierLLM()
                llm_tier = await self._llm_classifier.classify(goal)
                if llm_tier is not None:
                    logger.info(
                        f"LLM classifier overrode heuristic: "
                        f"{tier.name} (conf={confidence:.2f}) → {llm_tier.name}"
                    )
                    tier = llm_tier
            except Exception as e:
                logger.warning(f"LLM tier classification failed, using heuristic: {e}")

        self.detection_cache[cache_key] = tier
        logger.info(f"Auto-detected tier: {tier.name} (conf={confidence:.2f}) for goal: {goal[:50]}...")
        return tier

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

        elif tier == ExecutionTier.AUTONOMOUS:
            if any(ind in goal_lower for ind in self.AUTONOMOUS_INDICATORS):
                reasons.append("Contains autonomous/sandbox/coalition keywords")

        explanation = f"Tier {tier.value} ({tier.name}) selected:\n"
        for i, reason in enumerate(reasons, 1):
            explanation += f"  {i}. {reason}\n"

        return explanation.strip()

    def clear_cache(self):
        """Clear detection cache."""
        self.detection_cache.clear()
        logger.debug("Tier detection cache cleared")


class _TierClassifierLLM:
    """Lightweight LLM classifier that maps a task to an execution tier.

    Uses Haiku for fast, cheap classification (~$0.0002 per call, ~200ms).
    """

    _CLASSIFICATION_PROMPT = (
        "You are a task complexity classifier. Given a task description, "
        "respond with ONLY a single digit (1-5) indicating the execution tier:\n"
        "1 = Simple question/lookup (single direct answer)\n"
        "2 = Multi-step task needing planning and tools\n"
        "3 = Task requiring learning from past experience and validation\n"
        "4 = Deep research requiring multiple specialized agents\n"
        "5 = Autonomous execution needing sandbox/coalition/trust\n\n"
        "Task: {goal}\n\nTier (1-5):"
    )

    _TIER_MAP = {
        1: ExecutionTier.DIRECT,
        2: ExecutionTier.AGENTIC,
        3: ExecutionTier.LEARNING,
        4: ExecutionTier.RESEARCH,
        5: ExecutionTier.AUTONOMOUS,
    }

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def classify(self, goal: str) -> Optional[ExecutionTier]:
        """Classify task into a tier using Haiku.

        Returns:
            ExecutionTier or None if classification failed.
        """
        client = self._get_client()
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=8,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": self._CLASSIFICATION_PROMPT.format(goal=goal[:500]),
            }],
        )
        text = response.content[0].text.strip() if response.content else ""
        # Extract first digit
        for ch in text:
            if ch.isdigit():
                tier_num = int(ch)
                return self._TIER_MAP.get(tier_num)
        logger.warning(f"LLM tier classifier returned unparseable response: {text!r}")
        return None

"""
Tier Auto-Detection
===================

Automatically selects execution tier based on task characteristics.
Supports optional LLM fallback for ambiguous cases.
"""

import logging
from typing import Any, Dict, Optional, Tuple

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
        "what is",
        "calculate",
        "convert",
        "translate",
        "define",
        "explain briefly",
        "simple question",
        "lookup",
        "find",
        "search for",
    ]

    LEARNING_INDICATORS = [
        "learn from",
        "improve",
        "optimize",
        "remember",
        "get better at",
        "track performance",
        "validate",
    ]

    RESEARCH_INDICATORS = [
        "experiment",
        "benchmark",
        "compare approaches",
        "analyze in depth",
        "research thoroughly",
        "multi-round",
        "self-improve",
    ]

    AUTONOMOUS_INDICATORS = [
        "sandbox",
        "isolated",
        "untrusted",
        "coalition",
        "consensus",
        "curriculum",
        "agent0",
        "autonomous",
        "multi-swarm",
        "byzantine",
        "trust",
        "install",
        "execute code",
    ]

    MULTI_STEP_INDICATORS = [
        "and then",
        "after that",
        "followed by",
        "first",
        "second",
        "third",
        "finally",
        "step 1",
        "step 2",
        "analyze and",
        "research and",
        "create and",
        "compile and",
        "gather and",
        "process and",
    ]

    def __init__(
        self, enable_llm_fallback: bool = False, grouped_baseline: Optional[Any] = None
    ) -> None:
        self.detection_cache: Dict[str, Any] = (
            {}
        )  # Simple cache for repeated queries  # type: ignore[name-defined]
        self._enable_llm_fallback = enable_llm_fallback
        self._llm_classifier = None
        self._grouped_baseline = grouped_baseline  # CogRouter: learned tier history

    def detect(
        self, goal: str, context: Optional[dict] = None, force_tier: Optional[ExecutionTier] = None
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
            return tier  # type: ignore[no-any-return]

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

        Architecture: CogRouter (learned) is PRIMARY, keyword heuristics
        are the COLD-START FALLBACK.  Once enough execution history exists,
        CogRouter overrides hardcoded keywords — this is the whole point
        of the CogRouter paper integration.

        Returns:
            (tier, confidence) where confidence 0.0-1.0.
        """
        goal_lower = goal.lower()

        # ── PRIMARY: CogRouter learned routing ──────────────────────
        # Consult historical tier success FIRST.  When sufficient data
        # exists (≥5 samples, success >0.6), learned routing beats
        # keyword heuristics because it reflects actual outcomes.
        learned = self._consult_tier_history(goal_lower)
        if learned is not None:
            return learned

        # ── COLD-START FALLBACK: keyword heuristics ─────────────────
        # Only used when CogRouter has no data for this task type.
        # As the system runs, CogRouter gradually takes over.

        # Delegation overhead floor (AI Delegation paper):
        # Trivial tasks below complexity floor → always DIRECT.
        # Checked first because it's the cheapest gate.
        if self._below_delegation_floor(goal_lower):
            return ExecutionTier.DIRECT, 0.75

        # Tier 5 (AUTONOMOUS) - sandbox/coalition keywords
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
            return self.detection_cache[cache_key]  # type: ignore[no-any-return]

        tier, confidence = self._detect_tier_with_confidence(goal, context)

        if confidence < 0.7 and self._enable_llm_fallback:
            try:
                if self._llm_classifier is None:
                    self._llm_classifier = _TierClassifierLLM()  # type: ignore[assignment]
                llm_tier = await self._llm_classifier.classify(goal)  # type: ignore[attr-defined]
                if llm_tier is not None:
                    logger.info(
                        f"LLM classifier overrode heuristic: "
                        f"{tier.name} (conf={confidence:.2f}) → {llm_tier.name}"
                    )
                    tier = llm_tier
            except Exception as e:
                logger.warning(f"LLM tier classification failed, using heuristic: {e}")

        self.detection_cache[cache_key] = tier
        logger.info(
            f"Auto-detected tier: {tier.name} (conf={confidence:.2f}) for goal: {goal[:50]}..."
        )
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

    # =========================================================================
    # COGROUTER: LEARNED TIER ROUTING
    # =========================================================================

    _TASK_TYPE_KEYWORDS = {
        "research": ["research", "find", "search", "look up", "investigate"],
        "creation": ["create", "build", "write", "generate", "make"],
        "analysis": ["analyze", "calculate", "evaluate", "compare", "assess"],
        "coding": ["code", "implement", "debug", "refactor", "fix bug"],
        "automation": ["automate", "deploy", "run", "execute", "schedule"],
    }

    def _infer_task_type_simple(self, goal_lower: str) -> str:
        """Keyword-based task type for tier history lookup."""
        for task_type, keywords in self._TASK_TYPE_KEYWORDS.items():
            if any(kw in goal_lower for kw in keywords):
                return task_type
        return "general"

    def _consult_tier_history(self, goal_lower: str) -> Optional[Tuple[ExecutionTier, float]]:
        """CogRouter: consult learned tier success history.

        PRIMARY routing method.  When sufficient execution history exists
        (≥5 samples, success rate >0.6), returns the tier that historically
        works best for this task type.  Returns None on cold-start so
        keyword heuristics can bootstrap the data.
        """
        baseline = self._grouped_baseline
        if baseline is None:
            return None

        task_type = self._infer_task_type_simple(goal_lower)
        tier_map = {
            "DIRECT": ExecutionTier.DIRECT,
            "AGENTIC": ExecutionTier.AGENTIC,
            "LEARNING": ExecutionTier.LEARNING,
            "RESEARCH": ExecutionTier.RESEARCH,
            "AUTONOMOUS": ExecutionTier.AUTONOMOUS,
        }

        best_tier = None
        best_success = 0.0

        for tier_name, tier_enum in tier_map.items():
            result = baseline.get_tier_success(tier_name, task_type)
            if result is None:
                continue
            success_rate, count = result
            if count >= 5 and success_rate > 0.6 and success_rate > best_success:
                best_success = success_rate
                best_tier = tier_enum

        if best_tier is not None:
            logger.info(
                f"CogRouter: learned tier {best_tier.name} for "
                f"task_type={task_type} (success={best_success:.2f})"
            )
            return best_tier, 0.85

        return None

    # Keywords that signal complexity even in short tasks
    _COMPLEX_KEYWORDS = frozenset(
        [
            "analyze",
            "research",
            "compare",
            "build",
            "create",
            "deploy",
            "implement",
            "design",
            "optimize",
            "refactor",
            "debug",
            "benchmark",
            "experiment",
            "evaluate",
            "generate report",
        ]
    )

    def _below_delegation_floor(self, goal_lower: str) -> bool:
        """Check if task is below the delegation complexity floor.

        AI Delegation paper: tasks below a complexity floor always cost more
        to delegate than to execute directly.  Only very short, generic
        queries with no keyword signals qualify — anything with tier-specific,
        multi-step, complex, or direct-indicator keywords should be routed by
        the downstream heuristics for more precise classification.
        """
        words = goal_lower.split()
        # Only truly short queries (≤6 words) can be below the floor;
        # longer queries deserve full keyword analysis.
        if len(words) > 6:
            return False
        if any(kw in goal_lower for kw in self._COMPLEX_KEYWORDS):
            return False
        if any(ind in goal_lower for ind in self.MULTI_STEP_INDICATORS):
            return False
        # Don't shortcut queries that contain tier-specific indicators —
        # those should be routed by the keyword heuristics below.
        if any(ind in goal_lower for ind in self.AUTONOMOUS_INDICATORS):
            return False
        if any(ind in goal_lower for ind in self.RESEARCH_INDICATORS):
            return False
        if any(ind in goal_lower for ind in self.LEARNING_INDICATORS):
            return False
        # Don't shortcut queries matching DIRECT_INDICATORS — let
        # _is_simple_query handle them with proper 0.80 confidence.
        if any(ind in goal_lower for ind in self.DIRECT_INDICATORS):
            return False
        return True

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

    def clear_cache(self) -> None:
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

    def __init__(self) -> None:
        self._client: Optional[AsyncAnthropic] = None  # type: ignore[name-defined]

    def _get_client(self) -> Any:
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
            messages=[
                {
                    "role": "user",
                    "content": self._CLASSIFICATION_PROMPT.format(goal=goal[:500]),
                }
            ],
        )
        text = response.content[0].text.strip() if response.content else ""
        # Extract first digit
        for ch in text:
            if ch.isdigit():
                tier_num = int(ch)
                return self._TIER_MAP.get(tier_num)
        logger.warning(f"LLM tier classifier returned unparseable response: {text!r}")
        return None

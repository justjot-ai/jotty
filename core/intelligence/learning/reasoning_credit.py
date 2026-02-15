"""
Reasoning-based credit assignment (Dr. Chen Enhancement).

STATUS: UNUSED — ReasoningCreditAssigner is imported but never instantiated.
See algorithmic_credit.py for the actively used credit assignment system.
See MODULE_STATUS.md for details.

Credit assignment using reasoning quality analysis.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from Jotty.core.infrastructure.foundation.configs.learning import (
    LearningConfig as FocusedLearningConfig,
)
from Jotty.core.infrastructure.foundation.data_structures import (
    AgentContribution,
    AlertType,
    CausalLink,
    GoalHierarchy,
    GoalValue,
    LearningMetrics,
    MemoryEntry,
    MemoryLevel,
    StoredEpisode,
    SwarmConfig,
    ValidationResult,
)

if TYPE_CHECKING:
    from ..memory.cortex import SwarmMemory


def _ensure_swarm_config(config: Any) -> Any:
    """Accept LearningConfig or SwarmConfig, return SwarmConfig."""
    if isinstance(config, FocusedLearningConfig):
        return SwarmConfig.from_configs(learning=config)
    return config


# =============================================================================
# REASONING-BASED CREDIT ASSIGNER (Dr. Chen Enhancement)
# =============================================================================


class ReasoningCreditAssigner:
    """
    Enhanced credit assignment using reasoning quality analysis.

    Factors:
    1. Decision correctness (counterfactual)
    2. Reasoning quality (how well-reasoned)
    3. Evidence usage (what data was used)
    4. Temporal position (early vs late decisions)
    """

    def __init__(self, config: Any) -> None:
        self.config = _ensure_swarm_config(config)
        self.reasoning_weight = self.config.reasoning_weight
        self.evidence_weight = self.config.evidence_weight

    def analyze_contributions(
        self,
        success: bool,
        architect_results: List[ValidationResult],
        auditor_results: List[ValidationResult],
        actor_succeeded: bool,
        trajectory: List[Dict],
    ) -> Dict[str, AgentContribution]:
        """
        Analyze each agent's contribution with reasoning quality.
        """
        contributions = {}

        total_steps = len(trajectory)

        # Analyze Architect agents
        for i, result in enumerate(architect_results):
            contrib = self._analyze_single_agent(
                result=result,
                episode_success=success,
                is_architect=True,
                actor_succeeded=actor_succeeded,
                step_position=i / max(1, total_steps),
            )
            contributions[result.agent_name] = contrib

        # Analyze Auditor agents
        for i, result in enumerate(auditor_results):
            contrib = self._analyze_single_agent(
                result=result,
                episode_success=success,
                is_architect=False,
                actor_succeeded=actor_succeeded,
                step_position=(len(architect_results) + i) / max(1, total_steps),
            )
            contributions[result.agent_name] = contrib

        return contributions

    def _analyze_single_agent(
        self,
        result: ValidationResult,
        episode_success: bool,
        is_architect: bool,
        actor_succeeded: bool,
        step_position: float,
    ) -> AgentContribution:
        """Analyze a single agent's contribution."""

        # Determine decision
        if is_architect:
            decision = "approve" if result.should_proceed else "reject"
        else:
            decision = "approve" if result.is_valid else "reject"

        # Was decision correct?
        if is_architect:
            # Architect approve → Actor runs → Check if actor succeeded
            if decision == "approve":
                decision_correct = actor_succeeded
            else:
                # Architect reject → Can't know if it was right
                # Assume correct if episode would have failed
                decision_correct = not episode_success  # Pessimistic
        else:
            # Auditor: approve should align with success
            decision_correct = (decision == "approve") == episode_success

        # Counterfactual impact
        # How much would outcome change without this agent?
        if decision_correct:
            counterfactual = result.confidence  # High confidence → high impact
        else:
            counterfactual = -result.confidence

        # Reasoning quality
        reasoning_quality = self._assess_reasoning_quality(result)

        # Base contribution score
        if decision_correct:
            base_score = 0.5 + 0.5 * result.confidence
        else:
            base_score = -0.5 * result.confidence

        # Temporal weight (later decisions more certain)
        temporal_weight = 0.7 + 0.3 * step_position

        return AgentContribution(
            agent_name=result.agent_name,
            contribution_score=base_score,
            decision=decision,
            decision_correct=decision_correct,
            counterfactual_impact=abs(counterfactual),
            reasoning_quality=reasoning_quality,
            evidence_used=self._extract_evidence(result),
            tools_used=[tc.get("tool", "") for tc in result.tool_calls],
            decision_timing=step_position,
            temporal_weight=temporal_weight,
        )

    def _assess_reasoning_quality(self, result: ValidationResult) -> float:
        """
        Assess quality of reasoning in result.

        Heuristics:
        - Length of reasoning (longer often better, up to a point)
        - Use of evidence/data
        - Logical connectors
        - Confidence calibration
        """
        reasoning = result.reasoning or ""

        score = 0.5  # Base

        # Length factor (50-500 chars is good)
        length = len(reasoning)
        if 50 <= length <= 500:
            score += 0.1
        elif length > 500:
            score += 0.05  # Diminishing returns

        # A-Team Fix: Replace keyword patterns with structure-based heuristics
        # Reasoning quality indicators WITHOUT keyword matching:

        # 1. Has tool calls (used evidence)
        if result.tool_calls and len(result.tool_calls) > 0:
            score += 0.15  # Actually used tools = grounded reasoning

        # 2. Reasoning length indicates depth of analysis
        # (longer reasoning = more thorough, up to a point)
        reasoning_depth = min(len(reasoning) / 800, 0.2)
        score += reasoning_depth

        # 3. Has structured output (numbered steps, comparisons)
        # Check for digit presence which indicates structured thinking
        digit_count = sum(c.isdigit() for c in reasoning)
        if digit_count > 2:  # Has numbers = likely quantitative analysis
            score += 0.05

        # 4. Confidence calibration (extreme confidence often bad)
        # This is domain-agnostic - overconfidence is always suspicious
        if 0.6 <= result.confidence <= 0.9:
            score += 0.1
        elif result.confidence > 0.95 or result.confidence < 0.3:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _extract_evidence(self, result: ValidationResult) -> List[str]:
        """
        Extract evidence cited in reasoning.

        A-Team v8.0: NO REGEX! Uses character-by-character parsing.
        """
        evidence = []
        reasoning = result.reasoning or ""

        # Extract quoted content WITHOUT regex
        quotes = self._extract_quoted_strings(reasoning)
        evidence.extend(quotes)  # Max 3

        # Look for tool results
        for tc in result.tool_calls:
            if "result" in tc:
                evidence.append(f"Tool:{tc.get('tool', 'unknown')}")

        return evidence

    def _extract_quoted_strings(self, text: str) -> List[str]:
        """
        Extract strings between double quotes without regex.

        A-Team v8.0: Character-by-character parsing for robustness.
        """
        quotes = []
        in_quote = False
        current = []

        for char in text:
            if char == '"':
                if in_quote:
                    # End of quote
                    if current:
                        quotes.append("".join(current))
                    current = []
                in_quote = not in_quote
            elif in_quote:
                current.append(char)

        return quotes

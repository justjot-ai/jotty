"""
Tool Learning Feedback Loop
============================

Integrates tool execution statistics with TD-Lambda learning.

This module enables the system to:
1. Track which tools succeed/fail for different tasks
2. Learn optimal tool selection patterns
3. Improve discovery scoring over time

Usage:
    from Jotty.core.intelligence.learning.tool_learning import ToolLearningFeedback

    feedback = ToolLearningFeedback()
    feedback.feed_from_interceptor(interceptor)
    feedback.update_registry_scores(registry)
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolLearningFeedback:
    """
    Feedback loop between tool execution and learning system.

    Connects:
    - ToolInterceptor (execution tracking) → TD-Lambda (learning)
    - Learning statistics → SkillsRegistry (improved discovery)
    """

    def __init__(self) -> None:
        """Initialize feedback loop."""
        from .facade import get_td_lambda

        self.td_lambda = get_td_lambda()
        self.tool_success_rates: Dict[str, float] = {}
        self.tool_avg_latencies: Dict[str, float] = {}

    def feed_from_interceptor(self, interceptor: Any) -> int:
        """
        Feed tool execution data to TD-Lambda learning system.

        Args:
            interceptor: ToolInterceptor instance with execution history

        Returns:
            Number of tool calls processed
        """
        from Jotty.core.infrastructure.integration.tool_interceptor import ToolInterceptor

        if not isinstance(interceptor, ToolInterceptor):
            logger.warning(f"Expected ToolInterceptor, got {type(interceptor)}")
            return 0

        calls = interceptor.get_all_calls()
        if not calls:
            logger.debug("No tool calls to feed to learning system")
            return 0

        for call in calls:
            # Calculate reward
            if call.success:
                # Bonus for fast execution
                latency_bonus = max(0, 1.0 - (call.metadata.get("latency", 0) / 10.0))
                reward = 1.0 + (0.2 * latency_bonus)  # Range: 1.0 to 1.2
            else:
                reward = -0.5

            # State: tool being used
            state = {
                "tool_name": call.tool_name,
                "args_count": len(call.args),
                "executor": call.metadata.get("executor", "unknown"),
            }

            # Action: execute tool with these args
            action = {"execute": True, "attempt": call.attempt_number}

            # Next state: execution completed
            next_state = {
                "tool_name": call.tool_name,
                "completed": True,
                "success": call.success,
            }

            # Update TD-Lambda
            self.td_lambda.update(state=state, action=action, reward=reward, next_state=next_state)

        # Update success rate tracking
        summary = interceptor.summary()
        for tool_name, stats in summary.get("by_tool", {}).items():
            total = stats["total"]
            if total > 0:
                self.tool_success_rates[tool_name] = stats["successful"] / total

        logger.info(f"Fed {len(calls)} tool calls to TD-Lambda learning system")
        return len(calls)

    def update_registry_scores(self, registry: Any) -> int:
        """
        Update SkillsRegistry discovery scores based on learned success rates.

        Args:
            registry: SkillsRegistry instance

        Returns:
            Number of skills updated
        """
        if not self.tool_success_rates:
            logger.debug("No success rates to update registry")
            return 0

        updated_count = 0

        # Update each skill's relevance based on tool success rates
        for skill_name, skill in registry.loaded_skills.items():
            if not skill.is_loaded:
                continue

            # Calculate average success rate for this skill's tools
            tool_success_rates = []
            for tool_name in skill.tools.keys():
                if tool_name in self.tool_success_rates:
                    tool_success_rates.append(self.tool_success_rates[tool_name])

            if tool_success_rates:
                avg_success = sum(tool_success_rates) / len(tool_success_rates)

                # Boost high-performing tools in discovery
                # Store in metadata for future discover() calls
                if not hasattr(skill, "_learning_boost"):
                    skill._learning_boost = 0

                # Success rate > 0.8: +2 points, 0.6-0.8: +1 point, <0.6: no boost
                if avg_success > 0.8:
                    skill._learning_boost = 2
                elif avg_success > 0.6:
                    skill._learning_boost = 1
                else:
                    skill._learning_boost = 0

                updated_count += 1

        logger.info(f"Updated learning boost for {updated_count} skills")
        return updated_count

    def get_tool_recommendations(
        self, task: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get tool recommendations based on learned success patterns.

        Args:
            task: Task description
            top_k: Number of recommendations to return

        Returns:
            List of recommended tools with confidence scores
        """
        if not self.tool_success_rates:
            return []

        # Sort tools by success rate
        sorted_tools = sorted(
            self.tool_success_rates.items(), key=lambda x: -x[1]
        )[:top_k]

        recommendations = []
        for tool_name, success_rate in sorted_tools:
            recommendations.append(
                {
                    "tool_name": tool_name,
                    "success_rate": success_rate,
                    "confidence": success_rate,  # Use success rate as confidence
                    "reason": f"Historical success rate: {success_rate:.1%}",
                }
            )

        return recommendations

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get learning statistics summary.

        Returns:
            Dict with tracked tools, success rates, etc.
        """
        return {
            "tracked_tools": len(self.tool_success_rates),
            "success_rates": self.tool_success_rates.copy(),
            "avg_latencies": self.tool_avg_latencies.copy(),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_feedback_instance: Optional[ToolLearningFeedback] = None


def get_tool_learning_feedback() -> ToolLearningFeedback:
    """Get singleton ToolLearningFeedback instance."""
    global _feedback_instance
    if _feedback_instance is None:
        _feedback_instance = ToolLearningFeedback()
    return _feedback_instance


def feed_tool_statistics_to_learning(interceptor: Any) -> int:
    """
    Convenience function to feed tool statistics to learning system.

    Args:
        interceptor: ToolInterceptor instance

    Returns:
        Number of tool calls processed
    """
    feedback = get_tool_learning_feedback()
    return feedback.feed_from_interceptor(interceptor)


def update_registry_with_learning(registry: Any) -> int:
    """
    Convenience function to update registry scores with learning data.

    Args:
        registry: SkillsRegistry instance

    Returns:
        Number of skills updated
    """
    feedback = get_tool_learning_feedback()
    return feedback.update_registry_scores(registry)

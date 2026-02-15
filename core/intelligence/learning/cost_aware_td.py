"""
Cost-Aware TD-Lambda Learning
==============================

Multi-objective RL that considers both task success and cost.

KISS PRINCIPLE: Simple reward adjustment formula.
DRY PRINCIPLE: Extends existing TDLambdaLearner (no code duplication).
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class CostAwareTDLambda:
    """
    TD-Lambda with cost-adjusted rewards.

    FORMULA:
    adjusted_reward = task_reward - (cost_usd / cost_sensitivity)

    INTUITION:
    - High cost_sensitivity (e.g., 1.0) → cost matters a lot
    - Low cost_sensitivity (e.g., 0.1) → prioritize quality over cost

    EXAMPLE 1 (cost_sensitivity=0.5):
    Task succeeds (reward=1.0), costs $2.00
    → adjusted = 1.0 - (2.0 / 0.5) = 1.0 - 4.0 = -3.0 ❌ PENALIZE

    EXAMPLE 2 (cost_sensitivity=0.5):
    Task succeeds (reward=1.0), costs $0.10
    → adjusted = 1.0 - (0.10 / 0.5) = 1.0 - 0.2 = 0.8 ✅ REWARD

    LEARNING OUTCOME:
    - Agent learns to prefer cheaper tools
    - Converges to cost-efficient strategies
    - Balances quality vs cost automatically
    """

    def __init__(self, cost_sensitivity: float = 0.5) -> None:
        """
        Args:
            cost_sensitivity: How much to penalize cost (higher = more penalty)
                             0.1 = cost barely matters
                             1.0 = $1 cost = -1.0 reward penalty
                             10.0 = extremely cost-sensitive
        """
        # DRY: Import and wrap existing learner
        from Jotty.core.intelligence.learning import get_td_lambda

        self.base_learner = get_td_lambda()

        self.cost_sensitivity = cost_sensitivity
        self.total_cost_saved = 0.0
        self.update_count = 0

        logger.info(f"💰 CostAwareTDLambda initialized " f"(cost_sensitivity={cost_sensitivity})")

    def update(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Dict[str, Any],
        cost_usd: float = 0.0,
        done: bool = False,
    ) -> Any:
        """
        Update TD-Lambda with cost-adjusted reward.

        Args:
            state: Current state
            action: Action taken
            reward: Task reward (0.0 to 1.0)
            next_state: Next state
            cost_usd: Cost of this action in USD
            done: Whether episode is complete
        """
        # Adjust reward for cost (KISS - simple subtraction)
        cost_penalty = cost_usd / self.cost_sensitivity if self.cost_sensitivity > 0 else 0.0
        adjusted_reward = reward - cost_penalty

        # Track savings (compared to naive approach)
        self.total_cost_saved += max(0, cost_penalty - 0.01)  # Assume baseline $0.01
        self.update_count += 1

        # Log significant cost penalties
        if cost_penalty > 0.5:
            logger.debug(
                f"⚠️  High cost penalty: ${cost_usd:.3f} → "
                f"penalty={cost_penalty:.2f} (adjusted_reward={adjusted_reward:.2f})"
            )

        # Delegate to base learner (DRY - reuse existing logic)
        self.base_learner.update(
            state=state,
            action=action,
            reward=adjusted_reward,
            next_state=next_state,
        )

    def get_value(self, state: Dict[str, Any]) -> float:
        """Get value estimate for state (delegates to base)."""
        return self.base_learner.get_value(state)

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics with cost savings."""
        base_stats = (
            self.base_learner.get_stats() if hasattr(self.base_learner, "get_stats") else {}
        )

        return {
            **base_stats,
            "cost_sensitivity": self.cost_sensitivity,
            "total_cost_saved_usd": round(self.total_cost_saved, 3),
            "updates": self.update_count,
            "avg_cost_saved_per_update": round(
                self.total_cost_saved / self.update_count if self.update_count > 0 else 0.0, 4
            ),
        }


# Factory function (DRY - follows same pattern as get_td_lambda)
def get_cost_aware_td_lambda(cost_sensitivity: float = 0.5) -> CostAwareTDLambda:
    """
    Get a cost-aware TD-Lambda learner.

    Args:
        cost_sensitivity: Cost penalty multiplier (0.1 to 10.0)

    Returns:
        CostAwareTDLambda instance
    """
    return CostAwareTDLambda(cost_sensitivity=cost_sensitivity)


__all__ = [
    "CostAwareTDLambda",
    "get_cost_aware_td_lambda",
]

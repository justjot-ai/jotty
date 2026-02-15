"""
Adaptive learning components.

Adaptive learning rate, intermediate rewards, and exploration strategies.
"""

import math
import logging
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict

logger = logging.getLogger(__name__)

from Jotty.core.infrastructure.foundation.data_structures import (
    SwarmConfig, MemoryEntry, MemoryLevel, GoalValue,
    ValidationResult, AgentContribution, StoredEpisode,
    LearningMetrics, AlertType, GoalHierarchy, CausalLink
)
from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig as FocusedLearningConfig

if TYPE_CHECKING:
    from ..memory.cortex import SwarmMemory


def _ensure_swarm_config(config: Any) -> Any:
    """Accept LearningConfig or SwarmConfig, return SwarmConfig."""
    if isinstance(config, FocusedLearningConfig):
        return SwarmConfig.from_configs(learning=config)
    return config




# =============================================================================
# ADAPTIVE LEARNING RATE (Dr. Manning Enhancement)
# =============================================================================

class AdaptiveLearningRate:
    """
    Dynamically adjusts learning rate based on learning progress.
    
    Increases α when learning is slow, decreases when unstable.
    
    Methods:
    - Gradient magnitude tracking
    - Success rate monitoring
    - TD error variance analysis
    """
    
    def __init__(self, config: Any) -> None:
        self.config = _ensure_swarm_config(config)
        self.alpha = self.config.alpha
        self.min_alpha = self.config.alpha_min
        self.max_alpha = self.config.alpha_max
        self.adaptation_rate = self.config.alpha_adaptation_rate
        
        # Tracking
        self.td_errors: List[float] = []
        self.success_rates: List[float] = []
        self.window_size = config.adaptive_window_size # STANFORD FIX
    
    def record_td_error(self, td_error: float) -> None:
        """Record a TD error for variance analysis."""
        self.td_errors.append(abs(td_error))
        if len(self.td_errors) > self.window_size * 2:
            self.td_errors = self.td_errors[-self.window_size:]
    
    def record_success(self, success: bool) -> None:
        """Record episode outcome."""
        self.success_rates.append(1.0 if success else 0.0)
        if len(self.success_rates) > self.window_size * 2:
            self.success_rates = self.success_rates[-self.window_size:]
    
    def get_adapted_alpha(self) -> float:
        """
        Get current learning rate with adaptations.
        
        Adaptation logic:
        1. High TD error variance → decrease α (unstable)
        2. Low TD error magnitude → increase α (slow learning)
        3. Success rate improving → maintain α
        4. Success rate declining → increase α (need to learn faster)
        """
        if not self.config.enable_adaptive_alpha:
            return self.config.alpha
        
        if len(self.td_errors) < 10:
            return self.alpha
        
        recent_errors = self.td_errors[-self.window_size:]

        # =====================================================================
        # STEP 1: ANALYZE TD ERROR STATISTICS
        # =====================================================================
        # TD errors tell us "how wrong our predictions are"
        # - Large errors = we're learning a lot (good!) or unstable (bad!)
        # - Small errors = we've learned well OR learning too slowly
        #
        # We need to distinguish:
        # - High variance + high mean = unstable learning → SLOW DOWN (↓ α)
        # - Low variance + low mean = converged → KEEP CURRENT RATE
        # - Low variance + medium mean = learning steadily → KEEP CURRENT RATE
        # - High variance + low mean = confused → SLOW DOWN (↓ α)
        # - Low mean across board = stagnant → SPEED UP (↑ α)
        # =====================================================================

        # Compute mean and variance of recent TD errors
        # Mean = average prediction error
        # Variance = how much errors fluctuate
        mean_error = sum(recent_errors) / len(recent_errors)
        variance = sum((e - mean_error) ** 2 for e in recent_errors) / len(recent_errors)
        std_dev = math.sqrt(variance)

        # =====================================================================
        # STEP 2: DECIDE ON LEARNING RATE ADJUSTMENT
        # =====================================================================
        # We'll compute an adjustment factor (positive = increase α, negative = decrease α)
        # Then apply it multiplicatively: α_new = α_old * (1 + adjustment)
        # =====================================================================
        adjustment = 0.0

        # =====================================================================
        # RULE 1: HIGH VARIANCE → UNSTABLE LEARNING → DECREASE α
        # =====================================================================
        # If std_dev > mean * threshold, errors are wildly inconsistent
        # This means learning is UNSTABLE (bouncing around, not converging)
        #
        # EXAMPLE:
        # - Mean error = 0.2, Std dev = 0.3
        # - Ratio = 0.3 / 0.2 = 1.5 (high variance)
        # - If threshold = 1.2, then 1.5 > 1.2 → TOO UNSTABLE
        # - ACTION: Decrease α to make updates smaller and more stable
        #
        # WHY: Large learning rate with high variance = overshooting
        # Smaller α = smoother convergence
        # =====================================================================
        if std_dev > mean_error * self.config.instability_threshold_multiplier:
            adjustment -= self.adaptation_rate  # Decrease α
            # Example: adjustment = -0.1 → α_new = α * 0.9 (10% smaller)

        # =====================================================================
        # RULE 2: LOW MEAN ERROR → SLOW LEARNING → INCREASE α
        # =====================================================================
        # If mean error is very small, we might be learning too slowly
        # (or we might have converged - but if success rate is bad, we haven't)
        #
        # EXAMPLE:
        # - Mean error = 0.05 (very small)
        # - Threshold = 0.1
        # - 0.05 < 0.1 → Learning is STAGNANT
        # - ACTION: Increase α to learn faster
        #
        # WHY: Small errors + low success = we're stuck in a plateau
        # Need a bigger push to escape
        #
        # NOTE: We use learning_boost_factor (default 2.0) to increase more
        # aggressively when stagnant
        # =====================================================================
        elif mean_error < self.config.slow_learning_threshold:
            _boost = getattr(self.config, 'learning_boost_factor', 2.0)
            adjustment += self.adaptation_rate * _boost  # Increase α (boosted)
            # Example: adjustment = +0.2 → α_new = α * 1.2 (20% larger)

        # =====================================================================
        # RULE 3: DECLINING SUCCESS RATE → INCREASE α TO ADAPT FASTER
        # =====================================================================
        # If success rate is getting WORSE over time, we need to learn faster
        # to adapt to changing conditions or escape bad policy
        #
        # EXAMPLE:
        # - Recent 10 episodes: 6/10 success = 60%
        # - Older 10 episodes: 8/10 success = 80%
        # - Trend: 60% < 80% - 10% = 70% → DECLINING
        # - ACTION: Increase α to adapt to new conditions faster
        #
        # WHY: Declining performance suggests environment changed or we're
        # stuck in a local optimum. Larger α helps escape.
        # =====================================================================
        if len(self.success_rates) >= 20:
            # Compare recent vs older success rates
            recent_success = sum(self.success_rates[-10:]) / 10  # Last 10 episodes
            older_success = sum(self.success_rates[-20:-10]) / 10  # Episodes 11-20

            if recent_success < older_success - 0.1:  # Declining by >10%
                adjustment += self.adaptation_rate * 0.5  # Moderate increase
                # Example: adjustment = +0.05 → α_new = α * 1.05 (5% larger)

        # =====================================================================
        # STEP 3: APPLY ADJUSTMENT WITH SAFETY BOUNDS
        # =====================================================================
        # Update α multiplicatively (not additively, to keep it proportional)
        # α_new = α_old * (1 + adjustment)
        #
        # EXAMPLES:
        # - adjustment = -0.1 → α_new = α * 0.9 (decrease 10%)
        # - adjustment = +0.2 → α_new = α * 1.2 (increase 20%)
        # - adjustment = 0 → α_new = α * 1.0 (no change)
        #
        # SAFETY BOUNDS:
        # - α must stay within [min_alpha, max_alpha]
        # - Prevents α from becoming too small (no learning) or too large (unstable)
        #
        # TYPICAL VALUES:
        # - min_alpha = 0.001 (minimum learning still happens)
        # - max_alpha = 0.5 (prevents massive updates)
        # - alpha = 0.1 (typical starting point)
        # =====================================================================
        self.alpha = self.alpha * (1 + adjustment)
        self.alpha = max(self.min_alpha, min(self.max_alpha, self.alpha))
        
        return self.alpha
    
    def reset(self) -> None:
        """Reset tracking."""
        self.alpha = self.config.alpha
        self.td_errors.clear()
        self.success_rates.clear()


# =============================================================================
# INTERMEDIATE REWARDS (Dr. Manning Enhancement)
# =============================================================================

class IntermediateRewardCalculator:
    """
    Calculates intermediate rewards for dense learning signal.
    
    Instead of only terminal reward, provides rewards for:
    - Architect proceed decisions
    - Successful tool calls
    - Partial task completion
    - Good reasoning steps
    """
    
    def __init__(self, config: Any) -> None:
        self.config = _ensure_swarm_config(config)
        self.step_rewards: List[float] = []
    
    def reset(self) -> None:
        """Reset for new episode."""
        self.step_rewards.clear()
    
    def reward_architect_proceed(self, confidence: float) -> float:
        """Reward for Architect proceed decision."""
        if not self.config.enable_intermediate_rewards:
            return 0.0
        
        # Higher confidence proceed → higher reward
        reward = self.config.architect_proceed_reward * confidence
        self.step_rewards.append(reward)
        return reward
    
    def reward_tool_success(self, tool_name: str, success: bool) -> float:
        """Reward for successful tool call."""
        if not self.config.enable_intermediate_rewards:
            return 0.0
        
        if success:
            reward = self.config.tool_success_reward
        else:
            reward = -self.config.tool_success_reward * 0.5
        
        self.step_rewards.append(reward)
        return reward
    
    def reward_partial_completion(self, completion_fraction: float) -> float:
        """Reward for partial task completion."""
        if not self.config.enable_intermediate_rewards:
            return 0.0
        
        reward = completion_fraction * 0.3  # Max 0.3 for partial
        self.step_rewards.append(reward)
        return reward
    
    def get_total_intermediate_reward(self) -> float:
        """Get sum of intermediate rewards."""
        return sum(self.step_rewards)
    
    def get_discounted_intermediate_reward(self, gamma: float) -> float:
        """Get discounted sum of intermediate rewards."""
        total = 0.0
        for i, r in enumerate(self.step_rewards):
            total += r * (gamma ** i)
        return total





# =============================================================================
# EXPLORATION STRATEGY (Enhanced)
# =============================================================================

class AdaptiveExploration:
    """
    Adaptive exploration with UCB and stall detection.
    
    ε-greedy baseline with:
    - UCB bonus for underexplored memories
    - Exploration boost when learning stalls
    - Goal-specific exploration rates
    """
    
    def __init__(self, config: Any) -> None:
        self.config = _ensure_swarm_config(config)
        self.epsilon = self.config.epsilon_start
        self.ucb_c = self.config.ucb_coefficient
        
        # Per-goal exploration
        self.goal_epsilons: Dict[str, float] = {}
        self.goal_visit_counts: Dict[str, int] = defaultdict(int)
        
        # Stall detection
        self.recent_values: List[float] = []
        self.stall_boost_active = False
    
    def get_epsilon(self, goal: str, episode: int) -> float:
        """Get current epsilon with all adaptations."""
        
        # Base decay
        decay_progress = min(1.0, episode / self.config.epsilon_decay_episodes)
        base_epsilon = self.config.epsilon_start - decay_progress * (
            self.config.epsilon_start - self.config.epsilon_end
        )
        
        # Goal-specific adjustment
        goal_visits = self.goal_visit_counts[goal]
        if goal_visits < 5:
            # New goal → more exploration
            base_epsilon = min(0.5, base_epsilon * 1.5)
        
        # Stall boost
        if self.stall_boost_active:
            base_epsilon = min(0.5, base_epsilon + self.config.exploration_boost_on_stall)
        
        self.epsilon = base_epsilon
        return self.epsilon
    
    def record_goal_visit(self, goal: str) -> None:
        """Record visit to a goal."""
        self.goal_visit_counts[goal] += 1
    
    def record_value_change(self, delta: float) -> None:
        """Record value change for stall detection."""
        self.recent_values.append(abs(delta))
        if len(self.recent_values) > self.config.stall_detection_window:
            self.recent_values = self.recent_values[-self.config.stall_detection_window:]
        
        # Check for stall
        if len(self.recent_values) >= 50:
            avg_change = sum(self.recent_values) / 50
            self.stall_boost_active = avg_change < self.config.stall_threshold
    
    def should_explore(self, goal: str, episode: int) -> bool:
        """Decide whether to explore (True) or exploit (False)."""
        import random
        return random.random() < self.get_epsilon(goal, episode)
    
    def get_ucb_score(self, memory: MemoryEntry, goal: str, 
                      total_accesses: int) -> float:
        """Get UCB score for a memory."""
        value = memory.get_value(goal)
        
        if memory.ucb_visits == 0:
            return float('inf')
        
        exploration_bonus = self.ucb_c * math.sqrt(
            math.log(total_accesses + 1) / memory.ucb_visits
        )
        
        return value + exploration_bonus


# =============================================================================
# LEARNING HEALTH MONITOR (Enhanced)
# =============================================================================

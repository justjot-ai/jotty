"""Learning configuration — RL, exploration, credit, consolidation, protection."""

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class LearningConfig:
    """RL, exploration, credit assignment, consolidation, and protection."""

    # Q-Learning / TD
    auto_load_learning: bool = True
    per_agent_learning: bool = True
    shared_learning: bool = True
    learning_alpha: float = 0.3
    learning_gamma: float = 0.9
    learning_epsilon: float = 0.1
    max_q_table_size: int = 10000
    q_prune_percentage: float = 0.2
    enable_domain_transfer: bool = True
    enable_rl: bool = True
    rl_verbosity: str = "quiet"
    gamma: float = 0.99
    lambda_trace: float = 0.95
    alpha: float = 0.01
    enable_adaptive_alpha: bool = True
    alpha_min: float = 0.001
    alpha_max: float = 0.1
    alpha_adaptation_rate: float = 0.1
    q_value_mode: str = "simple"
    # Intermediate rewards
    enable_intermediate_rewards: bool = True
    architect_proceed_reward: float = 0.1
    tool_success_reward: float = 0.05
    # Cooperative rewards
    base_reward_weight: float = 0.3
    cooperation_bonus: float = 0.4
    predictability_bonus: float = 0.3
    adaptive_window_size: int = 50
    instability_threshold_multiplier: float = 1.5
    slow_learning_threshold: float = 0.01
    goal_transfer_discount: float = 0.5
    # Exploration
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 500
    ucb_coefficient: float = 2.0
    enable_adaptive_exploration: bool = True
    exploration_boost_on_stall: float = 0.1
    max_exploration_iterations: int = 10
    policy_update_threshold: int = 3
    # Credit assignment
    credit_decay: float = 0.9
    min_contribution: float = 0.1
    enable_reasoning_credit: bool = True
    reasoning_weight: float = 0.3
    evidence_weight: float = 0.2
    # Consolidation
    consolidation_threshold: int = 100
    consolidation_interval: int = 3
    min_cluster_size: int = 5
    pattern_confidence_threshold: float = 0.7
    # Offline learning
    episode_buffer_size: int = 1000
    offline_update_interval: int = 50
    replay_batch_size: int = 20
    counterfactual_samples: int = 5
    # Adaptive learning
    enable_adaptive_learning: bool = True
    stall_detection_window: int = 100
    stall_threshold: float = 0.001
    learning_boost_factor: float = 2.0
    # Deduplication
    enable_deduplication: bool = True
    similarity_threshold: float = 0.85
    # Components
    learning_components: Optional[List[str]] = None
    # Goal hierarchy
    enable_goal_hierarchy: bool = True
    goal_transfer_weight: float = 0.3
    # Causal learning
    enable_causal_learning: bool = True
    causal_confidence_threshold: float = 0.7
    causal_min_support: int = 3
    # Protection
    protected_memory_threshold: float = 0.8
    suspicion_threshold: float = 0.95
    min_rejection_rate: float = 0.05

    def __post_init__(self) -> None:
        # --- Probability / ratio fields: must be in [0, 1] ---
        _unit_fields = {
            "learning_alpha": self.learning_alpha,
            "learning_gamma": self.learning_gamma,
            "learning_epsilon": self.learning_epsilon,
            "q_prune_percentage": self.q_prune_percentage,
            "gamma": self.gamma,
            "lambda_trace": self.lambda_trace,
            "alpha": self.alpha,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "alpha_adaptation_rate": self.alpha_adaptation_rate,
            "base_reward_weight": self.base_reward_weight,
            "cooperation_bonus": self.cooperation_bonus,
            "predictability_bonus": self.predictability_bonus,
            "slow_learning_threshold": self.slow_learning_threshold,
            "goal_transfer_discount": self.goal_transfer_discount,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "credit_decay": self.credit_decay,
            "min_contribution": self.min_contribution,
            "reasoning_weight": self.reasoning_weight,
            "evidence_weight": self.evidence_weight,
            "pattern_confidence_threshold": self.pattern_confidence_threshold,
            "similarity_threshold": self.similarity_threshold,
            "goal_transfer_weight": self.goal_transfer_weight,
            "causal_confidence_threshold": self.causal_confidence_threshold,
            "protected_memory_threshold": self.protected_memory_threshold,
            "suspicion_threshold": self.suspicion_threshold,
            "min_rejection_rate": self.min_rejection_rate,
            "stall_threshold": self.stall_threshold,
            "exploration_boost_on_stall": self.exploration_boost_on_stall,
        }
        for name, val in _unit_fields.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")

        # --- Positive integer fields ---
        _pos_int_fields = {
            "max_q_table_size": self.max_q_table_size,
            "adaptive_window_size": self.adaptive_window_size,
            "epsilon_decay_episodes": self.epsilon_decay_episodes,
            "max_exploration_iterations": self.max_exploration_iterations,
            "policy_update_threshold": self.policy_update_threshold,
            "consolidation_threshold": self.consolidation_threshold,
            "consolidation_interval": self.consolidation_interval,
            "min_cluster_size": self.min_cluster_size,
            "episode_buffer_size": self.episode_buffer_size,
            "offline_update_interval": self.offline_update_interval,
            "replay_batch_size": self.replay_batch_size,
            "counterfactual_samples": self.counterfactual_samples,
            "stall_detection_window": self.stall_detection_window,
            "causal_min_support": self.causal_min_support,
        }
        for name, val in _pos_int_fields.items():
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}")

        # --- Ordering constraints ---
        if self.alpha_min > self.alpha_max:
            raise ValueError(
                f"alpha_min ({self.alpha_min}) must be <= alpha_max ({self.alpha_max})"
            )
        if self.epsilon_end > self.epsilon_start:
            raise ValueError(
                f"epsilon_end ({self.epsilon_end}) must be <= epsilon_start ({self.epsilon_start})"
            )
        if self.replay_batch_size > self.episode_buffer_size:
            raise ValueError(
                f"replay_batch_size ({self.replay_batch_size}) must be "
                f"<= episode_buffer_size ({self.episode_buffer_size})"
            )

"""Learning configuration — RL, exploration, credit, consolidation, protection."""

from dataclasses import dataclass
from typing import List, Optional


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

"""Validation configuration — multi-round validation and confidence thresholds."""

from dataclasses import dataclass


@dataclass
class ValidationConfig:
    """Validation and multi-round settings."""
    max_validation_rounds: int = 3
    refinement_timeout: float = 30.0
    enable_validation: bool = True
    validation_mode: str = 'full'
    require_all_architect: bool = True
    require_all_auditor: bool = False
    enable_per_actor_swarm_auditor: bool = False
    enable_final_swarm_auditor: bool = True
    swarm_validation_confidence_threshold: float = 0.6
    min_confidence: float = 0.5
    default_confidence_on_error: float = 0.3
    default_confidence_no_validation: float = 0.5
    default_confidence_insight_share: float = 0.7
    default_estimated_reward: float = 0.6
    enable_multi_round: bool = True
    refinement_on_low_confidence: float = 0.6
    refinement_on_disagreement: bool = True
    max_refinement_rounds: int = 2

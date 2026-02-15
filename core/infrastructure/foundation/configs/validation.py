"""Validation configuration — multi-round validation and confidence thresholds."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationConfig:
    """Validation and multi-round settings."""

    max_validation_rounds: int = 3
    refinement_timeout: float = 30.0
    enable_validation: bool = True
    validation_mode: str = "full"
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

    def __post_init__(self) -> None:
        # Confidence/threshold fields: [0, 1]
        _unit_fields = {
            "swarm_validation_confidence_threshold": self.swarm_validation_confidence_threshold,
            "min_confidence": self.min_confidence,
            "default_confidence_on_error": self.default_confidence_on_error,
            "default_confidence_no_validation": self.default_confidence_no_validation,
            "default_confidence_insight_share": self.default_confidence_insight_share,
            "default_estimated_reward": self.default_estimated_reward,
            "refinement_on_low_confidence": self.refinement_on_low_confidence,
        }
        for name, val in _unit_fields.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")

        # Positive integers
        _pos_int_fields = {
            "max_validation_rounds": self.max_validation_rounds,
            "max_refinement_rounds": self.max_refinement_rounds,
        }
        for name, val in _pos_int_fields.items():
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}")

        # Positive timeout
        if self.refinement_timeout <= 0:
            raise ValueError(f"refinement_timeout must be > 0, got {self.refinement_timeout}")

        # Validation mode
        valid_modes = {"full", "quick", "none", "standard", "architect_only", "thorough"}
        if self.validation_mode not in valid_modes:
            raise ValueError(
                f"validation_mode must be one of {valid_modes}, got '{self.validation_mode}'"
            )

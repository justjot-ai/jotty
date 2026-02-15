"""Monitoring configuration — logging, profiling, budget enforcement."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MonitoringConfig:
    """Logging, profiling, and budget enforcement."""

    enable_debug_logs: bool = True
    log_level: str = "INFO"
    enable_profiling: bool = False
    verbose: int = 1
    log_file: Optional[str] = None
    enable_debug_logging: bool = False
    enable_metrics: bool = True
    enable_monitoring: bool = False
    baseline_cost_per_success: Optional[float] = None
    max_llm_calls_per_episode: int = 100
    max_llm_calls_per_agent: int = 50
    max_total_tokens_per_episode: int = 500000
    enable_budget_enforcement: bool = True
    budget_warning_threshold: float = 0.8

    def __post_init__(self) -> None:
        # Log level validation
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got '{self.log_level}'")

        # Positive integer limits
        _pos_int_fields = {
            "max_llm_calls_per_episode": self.max_llm_calls_per_episode,
            "max_llm_calls_per_agent": self.max_llm_calls_per_agent,
            "max_total_tokens_per_episode": self.max_total_tokens_per_episode,
        }
        for name, val in _pos_int_fields.items():
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}")

        # Budget warning threshold: [0, 1]
        if not (0.0 <= self.budget_warning_threshold <= 1.0):
            raise ValueError(
                f"budget_warning_threshold must be in [0, 1], got {self.budget_warning_threshold}"
            )

        # verbose must be non-negative
        if self.verbose < 0:
            raise ValueError(f"verbose must be >= 0, got {self.verbose}")

        # baseline_cost_per_success must be positive if set
        if self.baseline_cost_per_success is not None and self.baseline_cost_per_success <= 0:
            raise ValueError(
                f"baseline_cost_per_success must be > 0, got {self.baseline_cost_per_success}"
            )

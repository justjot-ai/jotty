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

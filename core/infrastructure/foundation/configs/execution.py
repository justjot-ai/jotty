"""Execution configuration — runtime limits, timeouts, parallelism, reproducibility."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionConfig:
    """Runtime execution limits, timeouts, and parallelism."""

    max_actor_iters: int = 50
    max_eval_iters: int = 1
    max_episode_iterations: int = 12
    async_timeout: float = 60.0
    actor_timeout: float = 900.0
    max_concurrent_agents: int = 10
    max_eval_retries: int = 3
    llm_timeout_seconds: float = 180.0
    parallel_architect: bool = True
    parallel_auditor: bool = True
    random_seed: Optional[int] = None
    numpy_seed: Optional[int] = None
    torch_seed: Optional[int] = None
    python_hash_seed: Optional[int] = None
    enable_deterministic: bool = True

    def __post_init__(self) -> None:
        # Non-negative integer limits (0 means unlimited/disabled)
        _nonneg_int_fields = {
            "max_actor_iters": self.max_actor_iters,
        }
        for name, val in _nonneg_int_fields.items():
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")

        # Positive integer limits
        _pos_int_fields = {
            "max_eval_iters": self.max_eval_iters,
            "max_episode_iterations": self.max_episode_iterations,
            "max_concurrent_agents": self.max_concurrent_agents,
            "max_eval_retries": self.max_eval_retries,
        }
        for name, val in _pos_int_fields.items():
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}")

        # Positive timeout fields
        _pos_float_fields = {
            "async_timeout": self.async_timeout,
            "actor_timeout": self.actor_timeout,
            "llm_timeout_seconds": self.llm_timeout_seconds,
        }
        for name, val in _pos_float_fields.items():
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")

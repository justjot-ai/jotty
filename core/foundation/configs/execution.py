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

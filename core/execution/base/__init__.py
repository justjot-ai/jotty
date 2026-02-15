"""
Base execution classes for agents, swarms, and workflows.

All execution patterns inherit from these base classes.
"""

from .base_agent import AgentResult, AgentRuntimeConfig, BaseAgent
from .base_swarm import PhaseExecutor, SwarmTemplate

# Alias for cleaner naming
BaseSwarm = SwarmTemplate

__all__ = [
    "BaseAgent",
    "AgentRuntimeConfig",
    "AgentResult",
    "BaseSwarm",
    "SwarmTemplate",  # Backward compatibility
    "PhaseExecutor",
]

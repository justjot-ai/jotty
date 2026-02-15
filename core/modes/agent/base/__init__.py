"""
Jotty Agent Base Classes
========================

Core base classes for all Jotty agents.

BaseAgent (ABC) - Abstract base class all agents inherit from
AgentRuntimeConfig - Runtime configuration
AgentResult - Execution results

Usage:
    from Jotty.core.execution.base import (
        BaseAgent, AgentRuntimeConfig, AgentResult,
    )

Author: A-Team
Date: February 2026
"""

# Backwards compat: dag_types.py and others import AgentConfig from here
from ....infrastructure.foundation.agent_config import AgentConfig

# Core base classes (from base/ directory only - no circular imports)
from .base_agent import AgentResult, AgentRuntimeConfig, BaseAgent

__all__ = [
    # Core base classes
    "BaseAgent",
    "AgentRuntimeConfig",
    "AgentResult",
    # Backwards compat
    "AgentConfig",
]

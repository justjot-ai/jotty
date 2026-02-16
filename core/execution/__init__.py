"""
Unified Execution Layer - Core execution patterns for Jotty

This module unifies all execution patterns:
- Agents: Single-actor execution
- Swarms: Multi-agent coordination
- Workflows: Pipeline/step-based execution

All share common base classes and capabilities (learning, validation, memory).

Example:
    from Jotty.core.execution.base import BaseAgent, BaseSwarm
    from Jotty.core.execution.capabilities import (
        LearningCapability,
        ValidationCapability,
        MemoryCapability
    )
    from Jotty.core.execution.agents import MermaidAgent
    from Jotty.core.execution.swarms import CodingSwarm
    from Jotty.core.execution.workflows import ResearchWorkflow

    # Create agent with capabilities
    class MyAgent(BaseAgent, LearningCapability, ValidationCapability):
        def __init__(self):
            BaseAgent.__init__(self)
            LearningCapability.__init__(self, domain="my_domain")
            ValidationCapability.__init__(self, domain="my_domain")
"""

# Base classes
from .base import AgentResult, AgentRuntimeConfig, BaseAgent

# Capabilities
from .capabilities import (
    LearningCapability,
    MemoryCapability,
    SyntaxValidator,
    ValidationCapability,
)
from .swarms.base.swarm_template import PhaseExecutor
from .swarms.base.swarm_template import SwarmTemplate as BaseSwarm

# Re-export for convenience
__all__ = [
    # Base classes
    "BaseAgent",
    "BaseSwarm",
    "AgentRuntimeConfig",
    "AgentResult",
    "PhaseExecutor",
    # Capabilities
    "LearningCapability",
    "ValidationCapability",
    "MemoryCapability",
    "SyntaxValidator",
]

"""
Jotty Agent Base Classes
========================

Unified agent hierarchy following DRY and KISS principles:

BaseAgent (ABC)
├── DomainAgent       - Single-task executors (DSPy signatures)
├── MetaAgent         - Self-improvement agents (evaluate/improve others)
│   └── ValidationAgent - Pre/post validation agents (Inspector)
└── AutonomousAgent   - Open-ended problem solvers (skill discovery)

Usage:
    from Jotty.core.agents.base import (
        BaseAgent, AgentRuntimeConfig, AgentResult,
        DomainAgent, DomainAgentConfig,
        MetaAgent, MetaAgentConfig,
        ValidationAgent, ValidationConfig, ValidationResult,
        AutonomousAgent, AutonomousAgentConfig,
    )

Author: A-Team
Date: February 2026
"""

# Base agent and core types
from .base_agent import (
    BaseAgent,
    AgentRuntimeConfig,
    AgentResult,
)

# Backwards compat: dag_types.py and others import AgentConfig from here
from ...foundation.agent_config import AgentConfig  # noqa: F401

# Domain agent for single-task execution
from .domain_agent import (
    DomainAgent,
    DomainAgentConfig,
    create_domain_agent,
)

# Meta agent for self-improvement
from .meta_agent import (
    MetaAgent,
    MetaAgentConfig,
    create_meta_agent,
)

# Validation agent for pre/post validation
from .validation_agent import (
    ValidationAgent,
    ValidationConfig,
    ValidationResult,
    ValidationRound,
    OutputTag,
    SharedScratchpad,
    AgentMessage,
    create_validation_agent,
)

# Skill plan executor for reusable planning/execution
from .skill_plan_executor import (
    SkillPlanExecutor,
)

# Autonomous agent for open-ended tasks
from .autonomous_agent import (
    AutonomousAgent,
    AutonomousAgentConfig,
    ExecutionStep,
    create_autonomous_agent,
)

# Shared swarm-internal base agent
from .swarm_agent import (
    BaseSwarmAgent,
)

# Composite agent for agent/swarm unification
from .composite_agent import (
    CompositeAgent,
    CompositeAgentConfig,
    UnifiedResult,
)


__all__ = [
    # Base
    'BaseAgent',
    'AgentRuntimeConfig',
    'AgentConfig',
    'AgentResult',
    # Domain
    'DomainAgent',
    'DomainAgentConfig',
    'create_domain_agent',
    # Meta
    'MetaAgent',
    'MetaAgentConfig',
    'create_meta_agent',
    # Validation
    'ValidationAgent',
    'ValidationConfig',
    'ValidationResult',
    'ValidationRound',
    'OutputTag',
    'SharedScratchpad',
    'AgentMessage',
    'create_validation_agent',
    # Skill Plan Executor
    'SkillPlanExecutor',
    # Autonomous
    'AutonomousAgent',
    'AutonomousAgentConfig',
    'ExecutionStep',
    'create_autonomous_agent',
    # Swarm agent base
    'BaseSwarmAgent',
    # Composite
    'CompositeAgent',
    'CompositeAgentConfig',
    'UnifiedResult',
]

"""
DSPy-based agents with MCP tool support

Provides unified agent hierarchy:

BaseAgent (ABC)
├── DomainAgent       - Single-task executors (DSPy signatures)
├── MetaAgent         - Self-improvement agents (evaluate/improve others)
│   └── ValidationAgent - Pre/post validation agents (Inspector)
└── AutonomousAgent   - Open-ended problem solvers (skill discovery)

Plus domain-specific agents:
- AutoAgent: Autonomous task execution
- ChatAssistant: Built-in chat agent
- ModelChatAgent: ML model interaction
- TaskBreakdownAgent: DAG workflow creation
- TodoCreatorAgent: Actor assignment and validation
"""

# Base classes (unified hierarchy)
from .base import (
    # Core
    BaseAgent,
    AgentConfig,
    AgentResult,
    # Domain
    DomainAgent,
    DomainAgentConfig,
    create_domain_agent,
    # Meta
    MetaAgent,
    MetaAgentConfig,
    create_meta_agent,
    # Validation
    ValidationAgent,
    ValidationConfig,
    ValidationResult,
    ValidationRound,
    OutputTag,
    SharedScratchpad,
    AgentMessage,
    create_validation_agent,
    # Autonomous
    AutonomousAgent,
    AutonomousAgentConfig,
    ExecutionStep,
    create_autonomous_agent,
)

# Domain-specific agents
from .chat_assistant import ChatAssistant, create_chat_assistant
from .auto_agent import AutoAgent, run_task, TaskType, ExecutionResult
from .model_chat_agent import ModelChatAgent
from .dag_agents import (
    TaskBreakdownAgent,
    TodoCreatorAgent,
    ExecutableDAG,
    Actor,
    create_task_breakdown_agent,
    create_todo_creator_agent,
)

__all__ = [
    # Base classes
    'BaseAgent',
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
    # Autonomous
    'AutonomousAgent',
    'AutonomousAgentConfig',
    'ExecutionStep',
    'create_autonomous_agent',
    # Chat
    'ChatAssistant',
    'create_chat_assistant',
    # Auto
    'AutoAgent',
    'run_task',
    'TaskType',
    'ExecutionResult',
    # ML
    'ModelChatAgent',
    # DAG Agents
    'TaskBreakdownAgent',
    'TodoCreatorAgent',
    'ExecutableDAG',
    'Actor',
    'create_task_breakdown_agent',
    'create_todo_creator_agent',
]

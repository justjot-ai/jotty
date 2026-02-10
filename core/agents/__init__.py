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

Heavy imports (DSPy, etc.) are lazy-loaded on first attribute access.
"""

import importlib as _importlib

_LAZY_IMPORTS: dict[str, str] = {
    # Base classes (unified hierarchy)
    "BaseAgent": ".base",
    "AgentConfig": ".base",
    "AgentResult": ".base",
    "DomainAgent": ".base",
    "DomainAgentConfig": ".base",
    "create_domain_agent": ".base",
    "MetaAgent": ".base",
    "MetaAgentConfig": ".base",
    "create_meta_agent": ".base",
    "ValidationAgent": ".base",
    "ValidationConfig": ".base",
    "ValidationResult": ".base",
    "ValidationRound": ".base",
    "OutputTag": ".base",
    "SharedScratchpad": ".base",
    "AgentMessage": ".base",
    "create_validation_agent": ".base",
    "AutonomousAgent": ".base",
    "AutonomousAgentConfig": ".base",
    "ExecutionStep": "._execution_types",
    "create_autonomous_agent": ".base",
    # Shared types (no circular dependency)
    "TaskType": "._execution_types",
    "ExecutionResult": "._execution_types",
    "ExecutionStepSchema": "._execution_types",
    # Chat
    "ChatAssistant": ".chat_assistant",
    "create_chat_assistant": ".chat_assistant",
    # Auto
    "AutoAgent": ".auto_agent",
    "run_task": ".auto_agent",
    # ML
    "ModelChatAgent": ".model_chat_agent",
    # DAG Agents
    "TaskBreakdownAgent": ".dag_agents",
    "TodoCreatorAgent": ".dag_agents",
    "ExecutableDAG": ".dag_agents",
    "Actor": ".dag_agents",
    "create_task_breakdown_agent": ".dag_agents",
    "create_todo_creator_agent": ".dag_agents",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_IMPORTS.keys())

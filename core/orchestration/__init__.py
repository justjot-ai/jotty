"""
Orchestration Layer - Multi-Agent Coordination
===============================================

This layer handles agent orchestration, execution planning,
task management, and dependency resolution.

Modules:
--------
- conductor: Main orchestrator (PRIMARY ENTRY POINT)
- modes: Execution modes (WorkflowMode, ChatMode)
- jotty_core: Wraps agents with Architect/Auditor validation
- roadmap: Markovian TODO management for long-horizon tasks
- dynamic_dependency_graph: Agent dependency resolution
- policy_explorer: Exploration when stuck
- langgraph_orchestrator: LangGraph-based orchestration
"""

from .conductor import (
    Conductor,
    TodoItem,
    create_conductor,
)
from .modes import (
    ExecutionMode,
    WorkflowMode,
    ChatMode,
    ChatMessage,
    create_workflow,
    create_chat,
)
from .dynamic_dependency_graph import (
    CycleDetectedError,
    DependencySnapshot,
    DynamicDependencyGraph,
)
from .jotty_core import (
    JottyCore,
    PersistenceManager,
    create_jotty,
)
from .policy_explorer import (
    PolicyExplorer,
    PolicyExplorerSignature,
    SwarmLearnerSignature,
)
from .roadmap import (
    MarkovianTODO,
    SubtaskState,
    TaskStatus,
)
from .optimization_pipeline import (
    OptimizationPipeline,
    OptimizationConfig,
    IterationResult,
    create_optimization_pipeline,
)

__all__ = [
    # conductor
    'Conductor',
    'TodoItem',
    'create_conductor',
    # modes
    'ExecutionMode',
    'WorkflowMode',
    'ChatMode',
    'ChatMessage',
    'create_workflow',
    'create_chat',
    # dynamic_dependency_graph
    'CycleDetectedError',
    'DependencySnapshot',
    'DynamicDependencyGraph',
    # jotty_core
    'JottyCore',
    'PersistenceManager',
    'create_jotty',
    # policy_explorer
    'PolicyExplorer',
    'PolicyExplorerSignature',
    'SwarmLearnerSignature',
    # roadmap
    'MarkovianTODO',
    'SubtaskState',
    'TaskStatus',
    # optimization_pipeline
    'OptimizationPipeline',
    'OptimizationConfig',
    'IterationResult',
    'create_optimization_pipeline',
]

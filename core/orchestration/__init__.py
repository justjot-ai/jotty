"""
Orchestration Layer - Multi-Agent Coordination
===============================================

This layer handles agent orchestration, execution planning,
task management, and dependency resolution.

Modules:
--------
- conductor: Main orchestrator (PRIMARY ENTRY POINT) - MultiAgentsOrchestrator
- modes: Execution modes (WorkflowMode, ChatMode)
- single_agent_orchestrator: Single-agent episode manager (Phase 7-8)
- team_templates: Factory functions for common team patterns (Phase 8)
- jotty_core: DEPRECATED - Use single_agent_orchestrator instead
- roadmap: Markovian TODO management for long-horizon tasks
- dynamic_dependency_graph: Agent dependency resolution
- policy_explorer: Exploration when stuck
- langgraph_orchestrator: LangGraph-based orchestration
"""

from .conductor import (
    MultiAgentsOrchestrator,  # Phase 8: Main orchestrator class
    Conductor,  # Backward compatibility alias
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
from .single_agent_orchestrator import (
    SingleAgentOrchestrator,
    PersistenceManager,
    create_jotty,
)
from .jotty_core import (
    JottyCore,  # Deprecated alias for SingleAgentOrchestrator
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

from .task_orchestrator import TaskOrchestrator
from .agent_spawner import AgentSpawner
from .deployment_hook import DeploymentHook

# 🆕 Phase 8: Team Templates
from .team_templates import (
    create_diagram_team,
    create_sql_analytics_team,
    create_documentation_team,
    create_data_science_team,
    create_custom_team
)

__all__ = [
    # conductor
    'MultiAgentsOrchestrator',  # Phase 8: Main class
    'Conductor',  # Backward compatibility
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
    # single_agent_orchestrator (Phase 7)
    'SingleAgentOrchestrator',
    'PersistenceManager',
    'create_jotty',
    # jotty_core (deprecated)
    'JottyCore',  # Deprecated alias
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
    # task_orchestrator
    'TaskOrchestrator',
    'AgentSpawner',
    'DeploymentHook',

    # Phase 8: Team Templates
    'create_diagram_team',
    'create_sql_analytics_team',
    'create_documentation_team',
    'create_data_science_team',
    'create_custom_team',
]

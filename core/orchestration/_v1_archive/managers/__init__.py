"""
Orchestration Managers - Extracted from conductor.py for maintainability.

Refactoring Phases 2.1-2.6 + 3.2-3.4 + BaseManager:
- BaseManager: Abstract base class for all managers (interface consistency)
- StatelessManager: Base for stateless managers (no state tracking)
- StatefulManager: Base for stateful managers (operation tracking)
- LearningManager: Q-learning, TD(λ), credit assignment, MARL
- ValidationManager: Planner/Reviewer logic, multi-round validation
- ExecutionManager: Actor execution coordination, statistics tracking
- ParameterResolutionManager: Parameter resolution and dependency tracking
- ToolDiscoveryManager: Tool auto-discovery and filtering
- ToolExecutionManager: Tool execution with caching
- MetadataOrchestrationManager: Metadata fetching and enrichment
- OutputRegistryManager: Output detection, schema extraction, registry management
- AgentLifecycleManager: Agent wrapping, initialization, and lifecycle management
- StateActionManager: State representation and action space for RL
"""

from .base_manager import BaseManager, StatelessManager, StatefulManager
from .learning_manager import LearningManager, LearningUpdate
from .validation_manager import ValidationManager, ValidationResult
from .execution_manager import ExecutionManager, ExecutionResult
from .parameter_resolution_manager import ParameterResolutionManager, ResolutionResult
from .tool_discovery_manager import ToolDiscoveryManager
from .tool_execution_manager import ToolExecutionManager
from .metadata_orchestration_manager import MetadataOrchestrationManager
from .output_registry_manager import OutputRegistryManager
from .agent_lifecycle_manager import AgentLifecycleManager, ActorLifecycleManager
from .state_action_manager import StateActionManager

__all__ = [
    'BaseManager',
    'StatelessManager',
    'StatefulManager',
    'LearningManager',
    'LearningUpdate',
    'ValidationManager',
    'ValidationResult',
    'ExecutionManager',
    'ExecutionResult',
    'ParameterResolutionManager',
    'ResolutionResult',
    'ToolDiscoveryManager',
    'ToolExecutionManager',
    'MetadataOrchestrationManager',
    'OutputRegistryManager',
    'AgentLifecycleManager',
    'ActorLifecycleManager',  # Deprecated, for backward compatibility
    'StateActionManager',
]

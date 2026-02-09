"""
Orchestration Layer - V2 Multi-Agent Coordination
==================================================

V2 is the ONLY implementation. All imports are lazy.

Usage:
    from core.orchestration import SwarmManager
    swarm = SwarmManager(agents="Research AI startups")
    result = await swarm.run(goal="Research AI startups")
"""

import importlib as _importlib

_LAZY_IMPORTS: dict[str, str] = {
    # Main orchestrator
    "SwarmManager": ".v2",
    # Agent execution
    "AgentRunner": ".v2",
    "AgentRunnerConfig": ".v2",
    # Task management
    "SwarmTaskBoard": ".v2",
    "MarkovianTODO": ".v2",
    "SubtaskState": ".v2",
    "TaskStatus": ".v2",
    # Planning
    "SwarmPlanner": ".v2",
    "ExecutionPlan": ".v2",
    # Memory
    "SwarmMemory": ".v2",
    # Learning
    "SwarmLearner": ".v2",
    "SwarmLearnerSignature": ".v2",
    # RL Components
    "CreditAssignment": ".v2",
    "ImprovementCredit": ".v2",
    "AdaptiveLearning": ".v2",
    "LearningState": ".v2",
    "PolicyExplorer": ".v2",
    "PolicyExplorerSignature": ".v2",
    "LearningManager": ".v2",
    "LearningUpdate": ".v2",
    # OptimizationPipeline
    "OptimizationPipeline": ".v2",
    "OptimizationConfig": ".v2",
    "IterationResult": ".v2",
    "create_optimization_pipeline": ".v2",
    # Feature components
    "SwarmUIRegistry": ".v2",
    "SwarmProfiler": ".v2",
    "SwarmToolValidator": ".v2",
    "SwarmToolRegistry": ".v2",
    # Autonomous components
    "SwarmResearcher": ".v2",
    "SwarmInstaller": ".v2",
    "SwarmConfigurator": ".v2",
    "SwarmCodeGenerator": ".v2",
    "SwarmWorkflowLearner": ".v2",
    "SwarmIntegrator": ".v2",
    # Provider gateway
    "SwarmProviderGateway": ".v2",
    # State management
    "SwarmStateManager": ".v2",
    "AgentStateTracker": ".v2",
    # Sandbox
    "SandboxManager": ".v2",
    "TrustLevel": ".v2",
    "SandboxType": ".v2",
    "SandboxResult": ".v2",
    # Auto provider discovery
    "AutoProviderDiscovery": ".v2",
    "DiscoveryResult": ".v2",
    # Unified executor
    "UnifiedExecutor": ".v2",
    "ExecutionResult": ".v2",
    "ToolResult": ".v2",
    "StreamEvent": ".v2",
    "create_unified_executor": ".v2",
    "UnifiedToolGenerator": ".v2",
    "ToolDefinition": ".v2",
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

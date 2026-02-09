"""
Jotty V2 - True Agentic Assistant Framework
============================================

All imports are lazy to avoid loading DSPy at module level.

Usage:
    from core.orchestration.v2 import SwarmManager
    swarm = SwarmManager(agents="Research AI startups")
    result = await swarm.run(goal="Research AI startups")
"""

import importlib as _importlib

# ── Direct module imports (module_path, attribute_name) ─────────────────
# Maps public name → (relative module, attribute in that module)
# For aliases, attribute differs from the public name.
_LAZY_MAP: dict[str, tuple[str, str]] = {
    # Task management (from swarm_roadmap)
    "MarkovianTODO": (".swarm_roadmap", "MarkovianTODO"),
    "SubtaskState": (".swarm_roadmap", "SubtaskState"),
    "TaskStatus": (".swarm_roadmap", "TaskStatus"),
    # Aliases → real class (lightweight modules first)
    "SwarmTaskBoard": (".swarm_roadmap", "MarkovianTODO"),
    "SwarmPlanner": ("Jotty.core.agents.agentic_planner", "AgenticPlanner"),
    "SwarmMemory": ("Jotty.core.memory.cortex", "HierarchicalMemory"),
    "ExecutionPlan": ("Jotty.core.agents.agentic_planner", "ExecutionPlan"),
    # Agent runner
    "AgentRunner": (".agent_runner", "AgentRunner"),
    "AgentRunnerConfig": (".agent_runner", "AgentRunnerConfig"),
    # Main orchestrator (heavy — loads DSPy)
    "SwarmManager": (".swarm_manager", "SwarmManager"),
    # Feature components (aliases)
    "SwarmUIRegistry": ("Jotty.core.registry.agui_component_registry", "AGUIComponentRegistry"),
    "SwarmProfiler": ("Jotty.core.monitoring.profiler", "PerformanceProfiler"),
    "SwarmToolValidator": ("Jotty.core.registry.tool_validation", "ToolValidator"),
    "SwarmToolRegistry": ("Jotty.core.registry.tools_registry", "ToolsRegistry"),
    # Autonomous components
    "SwarmResearcher": (".swarm_researcher", "SwarmResearcher"),
    "SwarmInstaller": (".swarm_installer", "SwarmInstaller"),
    "SwarmConfigurator": (".swarm_configurator", "SwarmConfigurator"),
    "SwarmCodeGenerator": (".swarm_code_generator", "SwarmCodeGenerator"),
    "SwarmWorkflowLearner": (".swarm_workflow_learner", "SwarmWorkflowLearner"),
    "SwarmIntegrator": (".swarm_integrator", "SwarmIntegrator"),
    # Provider gateway
    "SwarmProviderGateway": (".swarm_provider_gateway", "SwarmProviderGateway"),
    # State management
    "SwarmStateManager": (".swarm_state_manager", "SwarmStateManager"),
    "AgentStateTracker": (".swarm_state_manager", "AgentStateTracker"),
    # Sandbox
    "SandboxManager": (".sandbox_manager", "SandboxManager"),
    "TrustLevel": (".sandbox_manager", "TrustLevel"),
    "SandboxType": (".sandbox_manager", "SandboxType"),
    "SandboxResult": (".sandbox_manager", "SandboxResult"),
    # Auto provider discovery
    "AutoProviderDiscovery": (".auto_provider_discovery", "AutoProviderDiscovery"),
    "DiscoveryResult": (".auto_provider_discovery", "DiscoveryResult"),
    # Unified executor
    "UnifiedExecutor": (".unified_executor", "UnifiedExecutor"),
    "ExecutionResult": (".unified_executor", "ExecutionResult"),
    "ToolResult": (".unified_executor", "ToolResult"),
    "StreamEvent": (".unified_executor", "StreamEvent"),
    "create_unified_executor": (".unified_executor", "create_unified_executor"),
    "UnifiedToolGenerator": (".tool_generator", "UnifiedToolGenerator"),
    "ToolDefinition": (".tool_generator", "ToolDefinition"),
    # SwarmLearner
    "SwarmLearner": (".swarm_learner", "SwarmLearner"),
    "SwarmLearnerSignature": (".swarm_learner", "SwarmLearnerSignature"),
    # SwarmLearningPipeline
    "SwarmLearningPipeline": (".learning_pipeline", "SwarmLearningPipeline"),
    # OptimizationPipeline
    "OptimizationPipeline": (".optimization_pipeline", "OptimizationPipeline"),
    "OptimizationConfig": (".optimization_pipeline", "OptimizationConfig"),
    "IterationResult": (".optimization_pipeline", "IterationResult"),
    "create_optimization_pipeline": (".optimization_pipeline", "create_optimization_pipeline"),
    # Core RL Components
    "CreditAssignment": (".credit_assignment", "CreditAssignment"),
    "ImprovementCredit": (".credit_assignment", "ImprovementCredit"),
    "AdaptiveLearning": (".adaptive_learning", "AdaptiveLearning"),
    "LearningState": (".adaptive_learning", "LearningState"),
    "PolicyExplorer": (".policy_explorer", "PolicyExplorer"),
    "PolicyExplorerSignature": (".policy_explorer", "PolicyExplorerSignature"),
    "LearningManager": ("Jotty.core.learning.learning_coordinator", "LearningCoordinator"),
    "LearningUpdate": ("Jotty.core.learning.learning_coordinator", "LearningUpdate"),
}


def __getattr__(name: str):
    if name in _LAZY_MAP:
        module_path, attr_name = _LAZY_MAP[name]
        if module_path.startswith("."):
            module = _importlib.import_module(module_path, __name__)
        else:
            module = _importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    # Pipeline utility functions (defined inline)
    if name == "sequential_pipeline":
        from ._pipeline_utils import sequential_pipeline
        globals()[name] = sequential_pipeline
        return sequential_pipeline
    if name == "fanout_pipeline":
        from ._pipeline_utils import fanout_pipeline
        globals()[name] = fanout_pipeline
        return fanout_pipeline

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_MAP.keys()) + ['sequential_pipeline', 'fanout_pipeline']

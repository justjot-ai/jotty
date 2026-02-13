"""
Orchestration Layer - Multi-Agent Coordination
===============================================

All imports are lazy to avoid loading DSPy at module level.

Usage:
    from core.orchestration import Orchestrator
    swarm = Orchestrator(agents="Research AI startups")
    result = await swarm.run(goal="Research AI startups")
"""

import importlib as _importlib

# ── Direct module imports (module_path, attribute_name) ─────────────────
# Maps public name → (relative module, attribute in that module)
# For aliases, attribute differs from the public name.
_LAZY_MAP: dict[str, tuple[str, str]] = {
    # Task management (from swarm_roadmap)
    "SwarmTaskBoard": (".swarm_roadmap", "SwarmTaskBoard"),
    "SubtaskState": (".swarm_roadmap", "SubtaskState"),
    "TaskStatus": (".swarm_roadmap", "TaskStatus"),
    # Agent runner
    "AgentRunner": (".agent_runner", "AgentRunner"),
    "AgentRunnerConfig": (".agent_runner", "AgentRunnerConfig"),
    # Main orchestrator (heavy — loads DSPy)
    "Orchestrator": (".swarm_manager", "Orchestrator"),
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
    # Chat executor
    "ChatExecutor": (".unified_executor", "ChatExecutor"),
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

"""
Orchestration Layer - Multi-Agent Coordination
===============================================

All imports are lazy to avoid loading DSPy at module level.

Sub-module Structure:
---------------------
  llm_providers/   — LLM provider adapters (Anthropic, OpenAI, Google)
  intelligence/    — SwarmIntelligence, mixins, protocols
  learning/        — LearningPipeline, Stigmergy, Byzantine, MAS, Metrics
  execution/       — AgentRunner, ChatExecutor, sandbox, DAG executor
  routing/         — SwarmRouter, ModelTierRouter, ProviderManager
  coordination/    — ParadigmExecutor, EnsembleManager, MAS-Zero, pipelines
  state/           — SwarmStateManager, SwarmRoadmap, SwarmTerminal
  core/            — Orchestrator, swarm entry point, adapter
  templates/       — Swarm templates (ML, science, lean)

Usage:
    from core.orchestration import Orchestrator
    swarm = Orchestrator(agents="Research AI startups")
    result = await swarm.run(goal="Research AI startups")
"""

import importlib as _importlib
from typing import Any

# ── Direct module imports (module_path, attribute_name) ─────────────────
# Maps public name → (relative module, attribute in that module)
# For aliases, attribute differs from the public name.
_LAZY_MAP: dict[str, tuple[str, str]] = {
    # Task management (from swarm_roadmap)
    "SwarmTaskBoard": (".state.swarm_roadmap", "SwarmTaskBoard"),
    "SubtaskState": (".state.swarm_roadmap", "SubtaskState"),
    "TaskStatus": (".state.swarm_roadmap", "TaskStatus"),
    # Agent runner
    "AgentRunner": (".execution.agent_runner", "AgentRunner"),
    "AgentRunnerConfig": (".execution.agent_runner", "AgentRunnerConfig"),
    # Main orchestrator (heavy — loads DSPy)
    "Orchestrator": (".core.swarm_manager", "Orchestrator"),
    # Autonomous components
    "SwarmResearcher": (".swarm_researcher", "SwarmResearcher"),
    "SwarmInstaller": (".swarm_installer", "SwarmInstaller"),
    "SwarmCodeGenerator": (".swarm_code_generator", "SwarmCodeGenerator"),
    # Provider gateway
    "SwarmProviderGateway": (".routing.swarm_provider_gateway", "SwarmProviderGateway"),
    # State management
    "SwarmStateManager": (".state.swarm_state_manager", "SwarmStateManager"),
    "AgentStateTracker": (".state.swarm_state_manager", "AgentStateTracker"),
    # Sandbox
    "SandboxManager": (".execution.sandbox_manager", "SandboxManager"),
    "TrustLevel": (".execution.sandbox_manager", "TrustLevel"),
    "SandboxType": (".execution.sandbox_manager", "SandboxType"),
    "SandboxResult": (".execution.sandbox_manager", "SandboxResult"),
    # Chat executor
    "ChatExecutor": (".execution.unified_executor", "ChatExecutor"),
    "ExecutionResult": (".execution.types", "ExecutionResult"),
    "ToolResult": (".llm_providers.types", "ToolResult"),
    "StreamEvent": (".execution.types", "StreamEvent"),
    "create_unified_executor": (".execution.unified_executor", "create_unified_executor"),
    "UnifiedToolGenerator": ("Jotty.skills._tools", "UnifiedToolGenerator"),
    "ToolDefinition": ("Jotty.skills._tools.tool_generator", "ToolDefinition"),
    # SwarmLearner
    "SwarmLearner": (".learning.swarm_learner", "SwarmLearner"),
    "SwarmLearnerSignature": (".learning.swarm_learner", "SwarmLearnerSignature"),
    # SwarmLearningPipeline
    "SwarmLearningPipeline": (".learning.learning_pipeline", "SwarmLearningPipeline"),
    # OptimizationPipeline
    "OptimizationPipeline": (".learning.optimization_pipeline", "OptimizationPipeline"),
    "OptimizationConfig": (".learning.optimization_pipeline", "OptimizationConfig"),
    "IterationResult": (".learning.optimization_pipeline", "IterationResult"),
    "create_optimization_pipeline": (
        ".learning.optimization_pipeline",
        "create_optimization_pipeline",
    ),
    # Core RL Components
    "CreditAssignment": (".learning.credit_assignment", "CreditAssignment"),
    "ImprovementCredit": (".learning.credit_assignment", "ImprovementCredit"),
    "AdaptiveLearning": (".learning.adaptive_learning", "AdaptiveLearning"),
    "LearningState": (".learning.adaptive_learning", "LearningState"),
    "PolicyExplorer": (".learning.policy_explorer", "PolicyExplorer"),
    "PolicyExplorerSignature": (".learning.policy_explorer", "PolicyExplorerSignature"),
    # SwarmIntelligence
    "SwarmIntelligence": (".intelligence.swarm_intelligence", "SwarmIntelligence"),
    # Swarm adapter
    "SwarmAdapter": (".core.swarm_adapter", "SwarmAdapter"),
    # Paradigm executor (debate, relay, refinement)
    "ParadigmExecutor": (".coordination.paradigm_executor", "ParadigmExecutor"),
    # Routing
    "SwarmRouter": (".routing.swarm_router", "SwarmRouter"),
    "ModelTierRouter": (".routing.model_tier_router", "ModelTierRouter"),
    # Coordination
    "EnsembleManager": (".coordination.ensemble_manager", "EnsembleManager"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MAP:
        module_path, attr_name = _LAZY_MAP[name]
        if module_path.startswith("."):
            module = _importlib.import_module(module_path, __name__)
        else:
            module = _importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    # Facade accessors
    _facade_names = {
        "get_swarm_intelligence",
        "get_paradigm_executor",
        "get_training_daemon",
        "get_ensemble_manager",
        "get_provider_manager",
        "get_model_tier_router",
        "get_swarm_router",
        "list_components",
    }
    if name in _facade_names:
        from . import facade

        value = getattr(facade, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_MAP.keys()) + [
    # facade
    "get_swarm_intelligence",
    "get_paradigm_executor",
    "get_training_daemon",
    "get_ensemble_manager",
    "get_provider_manager",
    "get_model_tier_router",
    "get_swarm_router",
    # coordination (eager imports for backward compat)
    "MultiStagePipeline",
    "PipelineResult",
    "StageConfig",
    "StageResult",
    "create_pipeline",
    "extract_code_from_markdown",
    "BenchmarkResults",
    "MultiStrategyBenchmark",
    "StrategyResult",
    "benchmark_strategies",
    "MergeStrategy",
    "MultiSwarmCoordinator",
    "SwarmResult",
    "get_multi_swarm_coordinator",
]

from .coordination.multi_stage_pipeline import (
    MultiStagePipeline,
    PipelineResult,
    StageConfig,
    StageResult,
    create_pipeline,
    extract_code_from_markdown,
)
from .coordination.multi_strategy_benchmark import (
    BenchmarkResults,
    MultiStrategyBenchmark,
    StrategyResult,
    benchmark_strategies,
)
from .coordination.multi_swarm_coordinator import (
    MergeStrategy,
    MultiSwarmCoordinator,
    SwarmResult,
    get_multi_swarm_coordinator,
)

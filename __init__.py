"""
JOTTY - Multi-Agent AI Framework
==================================

Tiered Execution with 5 progressive complexity levels:

    from Jotty import Jotty, ExecutionTier

    jotty = Jotty()

    # Auto-detect tier (DIRECT, AGENTIC, LEARNING, RESEARCH, or AUTONOMOUS)
    result = await jotty.run("Research AI trends")

    # Explicit tier
    result = await jotty.run("Task...", tier=ExecutionTier.LEARNING)

    # Convenience methods
    response = await jotty.chat("What is 2+2?")
    result = await jotty.plan("Complex task...")
    result = await jotty.learn("Task with memory")
    result = await jotty.research("Full features")
    result = await jotty.swarm("Build API", swarm_name="coding")
    result = await jotty.autonomous("Execute in sandbox")

Full Control:
    from Jotty import Orchestrator, AgentConfig

    # Or use the high-level entry points:
    from Jotty.core.modes.agent import AutoAgent      # Workflow execution
    from Jotty.core.modes.agent import ChatAssistant   # Chat mode
    from Jotty.core.interface.api import ModeRouter         # Programmatic API

See docs/JOTTY_ARCHITECTURE.md for complete documentation.

All heavy imports are lazy — ``import Jotty`` is lightweight (~50ms).
"""

import importlib as _importlib

# =============================================================================
# PYDANTIC WARNING SUPPRESSION (Must be before any imports)
# =============================================================================
import os
import warnings

# Suppress Pydantic serialization warnings from LiteLLM
os.environ.setdefault("PYDANTIC_WARNINGS", "none")

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic.main",
    message=".*PydanticSerializationUnexpectedValue.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic.*",
    message=".*serialized value may not be as expected.*",
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pydantic.*", message=".*Expected.*fields but got.*"
)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Pydantic serializer warnings.*")

__version__ = "3.0.0"
__author__ = "Jotty AI"

# =============================================================================
# ALL IMPORTS ARE LAZY — resolved on first attribute access
# =============================================================================

_LAZY_IMPORTS: dict[str, str] = {
    # --- TIERED EXECUTION ---
    "Jotty": ".jotty",
    "ExecutionTier": ".core.modes.execution.types",
    "ExecutionConfig": ".core.modes.execution.types",
    "ExecutionResult": ".core.modes.execution.types",
    "TierExecutor": ".core.modes.execution.executor",
    "TierDetector": ".core.modes.execution.tier_detector",
    # --- COMPOSITE AGENT (Agent/Swarm Unification) ---
    "CompositeAgent": ".core.modes.agent.base.composite_agent",
    "CompositeAgentConfig": ".core.modes.agent.base.composite_agent",
    "UnifiedResult": ".core.modes.agent.base.composite_agent",
    # --- ORCHESTRATION & FOUNDATION ---
    "Orchestrator": ".core.intelligence.orchestration",
    "TodoItem": ".core.intelligence.orchestration.swarm_roadmap",
    "AgentConfig": ".core.infrastructure.foundation.agent_config",
    "SwarmConfig": ".core.infrastructure.foundation.data_structures",
    "MemoryLevel": ".core.infrastructure.foundation.data_structures",
    "ValidationResult": ".core.infrastructure.foundation.data_structures",
    "MemoryEntry": ".core.infrastructure.foundation.data_structures",
    "GoalValue": ".core.infrastructure.foundation.data_structures",
    "EpisodeResult": ".core.infrastructure.foundation.data_structures",
    "TaggedOutput": ".core.infrastructure.foundation.data_structures",
    "OutputTag": ".core.infrastructure.foundation.data_structures",
    "StoredEpisode": ".core.infrastructure.foundation.data_structures",
    "LearningMetrics": ".core.infrastructure.foundation.data_structures",
    "GoalHierarchy": ".core.infrastructure.foundation.data_structures",
    "GoalNode": ".core.infrastructure.foundation.data_structures",
    "CausalLink": ".core.infrastructure.foundation.data_structures",
    "SwarmResult": ".core.infrastructure.data.io_manager",
    # --- TOOL MANAGEMENT ---
    "ToolShed": ".core.infrastructure.metadata.tool_shed",
    "ToolSchema": ".core.infrastructure.metadata.tool_shed",
    "ToolResult": ".core.infrastructure.metadata.tool_shed",
    "CapabilityIndex": ".core.infrastructure.metadata.tool_shed",
    # --- SHAPED REWARDS ---
    "ShapedRewardManager": ".core.intelligence.learning.shaped_rewards",
    "RewardCondition": ".core.intelligence.learning.shaped_rewards",
    # --- MEMORY ---
    "SwarmMemory": ".core.intelligence.memory.cortex",
    "MemoryCluster": ".core.intelligence.memory.consolidation",
    # --- LEARNING ---
    "TDLambdaLearner": ".core.intelligence.learning.learning",
    "AdaptiveLearningRate": ".core.intelligence.learning.learning",
    "ReasoningCreditAssigner": ".core.intelligence.learning.learning",
    # --- CONTEXT ---
    "GlobalContextGuard": ".core.infrastructure.context.global_context_guard",
    "patch_dspy_with_guard": ".core.infrastructure.context.global_context_guard",
    "unpatch_dspy": ".core.infrastructure.context.global_context_guard",
    # --- UNIVERSAL WRAPPER ---
    "JottyUniversal": ".core.infrastructure.integration.universal_wrapper",
    "SmartConfig": ".core.infrastructure.integration.universal_wrapper",
    "jotty_universal": ".core.infrastructure.integration.universal_wrapper",
    # --- STATE ---
    "AgenticState": ".core.intelligence.orchestration.swarm_roadmap",
    "TrajectoryStep": ".core.intelligence.orchestration.swarm_roadmap",
    "DecomposedQFunction": ".core.intelligence.orchestration.swarm_roadmap",
    "SwarmTaskBoard": ".core.intelligence.orchestration.swarm_roadmap",
    "SubtaskState": ".core.intelligence.orchestration.swarm_roadmap",
    "TaskStatus": ".core.intelligence.orchestration.swarm_roadmap",
    # --- PREDICTIVE MARL ---
    "LLMTrajectoryPredictor": ".core.intelligence.learning.predictive_marl",
    "DivergenceMemory": ".core.intelligence.learning.predictive_marl",
    "CooperativeCreditAssigner": ".core.intelligence.learning.predictive_marl",
    "AgentModel": ".core.intelligence.learning.predictive_marl",
    # --- CLI ---
    "JottyCLI": ".cli.app",
    # --- SDK ---
    "JottyClient": ".sdk.client",
    "JottySync": ".sdk.client",
    "ExecutionContext": ".core.infrastructure.foundation.types.sdk_types",
    "SDKResponse": ".core.infrastructure.foundation.types.sdk_types",
    "SDKEvent": ".core.infrastructure.foundation.types.sdk_types",
    "ModeRouter": ".core.interface.api.mode_router",
    "get_mode_router": ".core.interface.api.mode_router",
    # --- CAPABILITY DISCOVERY ---
    "capabilities": ".core.capabilities",
    # --- SUBSYSTEM FACADES ---
    "MemorySystem": ".core.intelligence.memory.memory_system",
    "BudgetTracker": ".core.infrastructure.utils.budget_tracker",
    "CircuitBreaker": ".core.infrastructure.utils.timeouts",
    "LLMCallCache": ".core.infrastructure.utils.llm_cache",
    "SmartTokenizer": ".core.infrastructure.utils.tokenizer",
    "ChatExecutor": ".core.intelligence.orchestration.unified_executor",
    # --- ORCHESTRATION (hidden components surfaced) ---
    "SwarmIntelligence": ".core.intelligence.orchestration.swarm_intelligence",
    "ParadigmExecutor": ".core.intelligence.orchestration.paradigm_executor",
    "EnsembleManager": ".core.intelligence.orchestration.ensemble_manager",
    "ModelTierRouter": ".core.intelligence.orchestration.model_tier_router",
}


def __getattr__(name: str):
    # Standard lazy imports
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        # Special alias: sdk.client exports "Jotty" class as JottyClient
        attr_name = "Jotty" if name == "JottyClient" else name
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [*_LAZY_IMPORTS.keys()]

"""
JOTTY - Multi-Agent AI Framework
==================================

V3 Architecture: Tiered Execution (NEW - Recommended)
V2 Architecture: Orchestrator + Skills + Learning (Preserved)

V3 - Simple, Progressive Complexity:
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
    result = await jotty.research("Full V2 features")
    result = await jotty.swarm("Build API", swarm_name="coding")
    result = await jotty.autonomous("Execute in sandbox")

V2 - Full Control (Still Available):
    from Jotty import Orchestrator, AgentConfig

    # Or use the high-level entry points:
    from Jotty.core.agents import AutoAgent      # Workflow execution
    from Jotty.core.agents import ChatAssistant   # Chat mode
    from Jotty.core.api import ModeRouter         # Programmatic API

See docs/JOTTY_ARCHITECTURE.md for complete documentation.

All heavy imports are lazy — ``import Jotty`` is lightweight (~50ms).
"""

# =============================================================================
# PYDANTIC WARNING SUPPRESSION (Must be before any imports)
# =============================================================================
import os
import warnings
import importlib as _importlib

# Suppress Pydantic serialization warnings from LiteLLM
os.environ.setdefault('PYDANTIC_WARNINGS', 'none')

warnings.filterwarnings(
    'ignore', category=UserWarning, module='pydantic.main',
    message='.*PydanticSerializationUnexpectedValue.*'
)
warnings.filterwarnings(
    'ignore', category=UserWarning, module='pydantic.*',
    message='.*serialized value may not be as expected.*'
)
warnings.filterwarnings(
    'ignore', category=UserWarning, module='pydantic.*',
    message='.*Expected.*fields but got.*'
)
warnings.filterwarnings(
    'ignore', category=UserWarning,
    message='.*Pydantic serializer warnings.*'
)

__version__ = "3.0.0"  # V3 Release
__author__ = "Jotty AI"

# =============================================================================
# ALL IMPORTS ARE LAZY — resolved on first attribute access
# =============================================================================

_LAZY_IMPORTS: dict[str, str] = {
    # --- V3 EXPORTS (NEW - Recommended) ---
    "Jotty": ".jotty",
    "ExecutionTier": ".core.execution.types",
    "ExecutionConfig": ".core.execution.types",
    "ExecutionResult": ".core.execution.types",
    "TierExecutor": ".core.execution.executor",
    "TierDetector": ".core.execution.tier_detector",

    # --- COMPOSITE AGENT (Agent/Swarm Unification) ---
    "CompositeAgent": ".core.agents.base.composite_agent",
    "CompositeAgentConfig": ".core.agents.base.composite_agent",
    "UnifiedResult": ".core.agents.base.composite_agent",

    # --- V2 PRIMARY EXPORTS (Preserved - No Breakage) ---
    "Orchestrator": ".core.orchestration",
    "TodoItem": ".core.orchestration.swarm_roadmap",
    "AgentConfig": ".core.foundation.agent_config",
    "SwarmConfig": ".core.foundation.data_structures",
    "MemoryLevel": ".core.foundation.data_structures",
    "ValidationResult": ".core.foundation.data_structures",
    "MemoryEntry": ".core.foundation.data_structures",
    "GoalValue": ".core.foundation.data_structures",
    "EpisodeResult": ".core.foundation.data_structures",
    "TaggedOutput": ".core.foundation.data_structures",
    "OutputTag": ".core.foundation.data_structures",
    "StoredEpisode": ".core.foundation.data_structures",
    "LearningMetrics": ".core.foundation.data_structures",
    "GoalHierarchy": ".core.foundation.data_structures",
    "GoalNode": ".core.foundation.data_structures",
    "CausalLink": ".core.foundation.data_structures",
    "SwarmResult": ".core.data.io_manager",

    # --- TOOL MANAGEMENT ---
    "ToolShed": ".core.metadata.tool_shed",
    "ToolSchema": ".core.metadata.tool_shed",
    "ToolResult": ".core.metadata.tool_shed",
    "CapabilityIndex": ".core.metadata.tool_shed",

    # --- SHAPED REWARDS ---
    "ShapedRewardManager": ".core.learning.shaped_rewards",
    "RewardCondition": ".core.learning.shaped_rewards",

    # --- MEMORY ---
    "SwarmMemory": ".core.memory.cortex",
    "MemoryCluster": ".core.memory.cortex",

    # --- LEARNING ---
    "TDLambdaLearner": ".core.learning.learning",
    "AdaptiveLearningRate": ".core.learning.learning",
    "ReasoningCreditAssigner": ".core.learning.learning",

    # --- CONTEXT ---
    "GlobalContextGuard": ".core.context.global_context_guard",
    "patch_dspy_with_guard": ".core.context.global_context_guard",
    "unpatch_dspy": ".core.context.global_context_guard",

    # --- UNIVERSAL WRAPPER ---
    "JottyUniversal": ".core.integration.universal_wrapper",
    "SmartConfig": ".core.integration.universal_wrapper",
    "jotty_universal": ".core.integration.universal_wrapper",

    # --- STATE ---
    "AgenticState": ".core.orchestration.swarm_roadmap",
    "TrajectoryStep": ".core.orchestration.swarm_roadmap",
    "DecomposedQFunction": ".core.orchestration.swarm_roadmap",
    "SwarmTaskBoard": ".core.orchestration.swarm_roadmap",
    "SubtaskState": ".core.orchestration.swarm_roadmap",
    "TaskStatus": ".core.orchestration.swarm_roadmap",

    # --- PREDICTIVE MARL ---
    "LLMTrajectoryPredictor": ".core.learning.predictive_marl",
    "DivergenceMemory": ".core.learning.predictive_marl",
    "CooperativeCreditAssigner": ".core.learning.predictive_marl",
    "AgentModel": ".core.learning.predictive_marl",

    # --- CLI ---
    "JottyCLI": ".cli.app",

    # --- SDK ---
    "JottyClient": ".sdk.client",
    "JottySync": ".sdk.client",
    "ExecutionContext": ".core.foundation.types.sdk_types",
    "SDKResponse": ".core.foundation.types.sdk_types",
    "SDKEvent": ".core.foundation.types.sdk_types",
    "ModeRouter": ".core.api.mode_router",
    "get_mode_router": ".core.api.mode_router",
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

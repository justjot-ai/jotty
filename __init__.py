"""
JOTTY v10.0 - SYnergistic Neural Agent Processing & Self-organizing Execution
================================================================================

Production-Ready Multi-Agent RL Wrapper for DSPy

🎯 SINGLE ENTRY POINT (A-Team Consensus 2026-01-07):
----------------------------------------------------
    from Jotty import Conductor, AgentConfig
    
    swarm = Conductor(
        actors=[
            AgentConfig(
                name="my_agent", 
                agent=my_dspy_module,  # NOTE: 'agent' field
                architect_prompts=["plan.md"],
                auditor_prompts=["validate.md"]
            )
        ],
    )
    result = await swarm.run("Process this")

Key Features:
- Works with ANY DSPy module (ChainOfThought, ReAct, Predict, custom)
- Never runs out of context (auto-chunking + compression)
- Cooperative reward (helps swarm cooperation)
- 5-level hierarchical memory
- Zero hardcoding (all decisions by LLM agents)

JOTTY Naming Convention:
- Conductor = Main orchestrator
- JottyCore = Core execution engine
- Architect = Pre-execution planner
- Auditor = Post-execution validator
- AgentConfig = Agent configuration (uses 'agent' field)
- Cortex = Hierarchical memory system
- Axon = Agent communication layer
- Roadmap = Markovian TODO system

See README.md for complete documentation.
"""

# =============================================================================
# PYDANTIC WARNING SUPPRESSION (Must be before any imports)
# =============================================================================
import os
import warnings

# Suppress Pydantic serialization warnings from LiteLLM
# These occur due to LiteLLM's response format not matching Pydantic's expected structure
# Versions are synced (Pydantic 2.12.5, LiteLLM 1.80.16) but warnings may still occur
os.environ.setdefault('PYDANTIC_WARNINGS', 'none')

# Filter specific Pydantic serialization warnings
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    module='pydantic.main',
    message='.*PydanticSerializationUnexpectedValue.*'
)
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    module='pydantic.*',
    message='.*serialized value may not be as expected.*'
)
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    module='pydantic.*',
    message='.*Expected.*fields but got.*'
)
# Catch all Pydantic serializer warnings
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message='.*Pydantic serializer warnings.*'
)

__version__ = "10.0.0"
__author__ = "Soham Acharya & Anshul Chauhan"

# =============================================================================
# 🎯 PRIMARY EXPORTS - What pipelines use
# =============================================================================

# Main Entry Points
from .core.orchestration.conductor import (
    Conductor,
    TodoItem,
    create_conductor
)

# Agent Configuration (THE one - uses 'agent' field)
from .core.foundation.agent_config import AgentConfig

# Jotty Configuration (THE one)
from .core.foundation.data_structures import (
    JottyConfig,
    MemoryLevel,
    ValidationResult,
    MemoryEntry,
    GoalValue,
    EpisodeResult,
    TaggedOutput,
    OutputTag,
    StoredEpisode,
    LearningMetrics,
    GoalHierarchy,
    GoalNode,
    CausalLink
)

# Result type
from .core.data.io_manager import SwarmResult

# =============================================================================
# 🎯 CLEAN INTERFACE - For new code (wraps Conductor)
# =============================================================================

from .interface import (
    Jotty,                    # Thin wrapper to Conductor
    ValidationMode,             # NONE, ARCHITECT, AUDITOR, BOTH
    LearningMode,               # DISABLED, CONTEXTUAL, PERSISTENT
    CooperationMode,            # INDEPENDENT, SHARED_REWARD, NASH
    MetadataProtocol,           # Protocol for metadata providers
)

# =============================================================================
# SECONDARY EXPORTS (Advanced Usage)
# =============================================================================

# Simple Brain (Recommended for quick setup)
# NOTE: simple_brain module may not exist in all versions
try:
    from .core.memory.simple_brain import (
        SimpleBrain,
        BrainPreset,
        calculate_chunk_size,
        get_model_context
    )
except ImportError:
    # Module not available - define placeholders or skip
    SimpleBrain = None
    BrainPreset = None
    calculate_chunk_size = None
    get_model_context = None

# Tool Management
from .core.metadata.tool_shed import (
    ToolShed,
    ToolSchema,
    ToolResult,
    CapabilityIndex,
)

# Shaped Rewards (per GRF MARL paper)
from .core.learning.shaped_rewards import (
    ShapedRewardManager,
    RewardCondition,
)

# Memory System
from .core.memory.cortex import (
    HierarchicalMemory,
    MemoryCluster
)

# Learning Components
from .core.learning.learning import (
    TDLambdaLearner,
    AdaptiveLearningRate,
    ReasoningCreditAssigner
)

# Context Protection
from .core.context.global_context_guard import (
    GlobalContextGuard,
    patch_dspy_with_guard,
    unpatch_dspy
)

# Universal Wrapper (for wrapping single modules)
from .core.integration.universal_wrapper import (
    JottyUniversal,
    SmartConfig,
    jotty_universal
)

# =============================================================================
# ADVANCED EXPORTS (For Custom Implementations)
# =============================================================================

# Enhanced State
from .core.orchestration.roadmap import (
    AgenticState,
    TrajectoryStep,
    DecomposedQFunction,
    MarkovianTODO,
    SubtaskState,
    TaskStatus
)

# Predictive MARL
from .core.learning.predictive_marl import (
    LLMTrajectoryPredictor,
    DivergenceMemory,
    CooperativeCreditAssigner,
    AgentModel
)

# Brain Modes (optional - may not exist in all versions)
try:
    from .core.brain_modes import (
        BrainMode,
        HippocampalExtractor,
        SharpWaveRippleConsolidator,
        BrainStateMachine
    )
except ImportError:
    BrainMode = None
    HippocampalExtractor = None
    SharpWaveRippleConsolidator = None
    BrainStateMachine = None

# Modern Agents (v9.0 - No Heuristics) (optional)
try:
    from .core.modern_agents import (
        UniversalRetryHandler,
        PatternDetector,
        LLMCounterfactualCritic,
        SelfRAGMemoryRetriever,
        LLMSurpriseEstimator
    )
except ImportError:
    UniversalRetryHandler = None
    PatternDetector = None
    LLMCounterfactualCritic = None
    SelfRAGMemoryRetriever = None
    LLMSurpriseEstimator = None

# Robust Parsing (optional)
try:
    from .core.robust_parsing import (
        parse_float_robust,
        parse_bool_robust,
        parse_json_robust,
        AdaptiveThreshold,
        safe_hash
    )
except ImportError:
    parse_float_robust = None
    parse_bool_robust = None
    parse_json_robust = None
    AdaptiveThreshold = None
    safe_hash = None

# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # 🎯 PRIMARY (What pipelines use)
    "Conductor",
    "AgentConfig",          # Uses 'agent' field
    "JottyConfig",
    "SwarmResult",
    "TodoItem",
    "create_conductor",
    
    # 🎯 CLEAN INTERFACE (Wrapper for Conductor)
    "Jotty",
    "ValidationMode",
    "LearningMode",
    "CooperationMode",
    "MetadataProtocol",
    
    # Memory & Data Structures
    "MemoryLevel",
    "ValidationResult",
    "MemoryEntry",
    "EpisodeResult",
    "TaggedOutput",
    "OutputTag",
    "GoalHierarchy",
    
    # Simple Brain
    "SimpleBrain",
    "BrainPreset",
    "calculate_chunk_size",
    "get_model_context",
    
    # Tool Management
    "ToolShed",
    "ToolSchema",
    "CapabilityIndex",
    
    # Shaped Rewards
    "ShapedRewardManager",
    "RewardCondition",
    
    # Memory
    "HierarchicalMemory",
    "MemoryCluster",
    
    # Learning
    "TDLambdaLearner",
    "AdaptiveLearningRate",
    "ReasoningCreditAssigner",
    
    # Context
    "GlobalContextGuard",
    "patch_dspy_with_guard",
    
    # Universal Wrapper
    "JottyUniversal",
    "SmartConfig",
    "jotty_universal",
    
    # State
    "AgenticState",
    "MarkovianTODO",
    
    # MARL
    "LLMTrajectoryPredictor",
    "CooperativeCreditAssigner",
    
    # Brain
    "BrainMode",
    "HippocampalExtractor",
    
    # Modern Agents
    "UniversalRetryHandler",
    "PatternDetector",
    "LLMCounterfactualCritic",
    
    # Utilities
    "parse_float_robust",
    "parse_json_robust",
    "AdaptiveThreshold",

    # CLI
    "JottyCLI",
]

# =============================================================================
# CLI EXPORTS (Lazy import for faster startup)
# =============================================================================

def __getattr__(name):
    """Lazy import for CLI components."""
    if name == "JottyCLI":
        from .cli.app import JottyCLI
        return JottyCLI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

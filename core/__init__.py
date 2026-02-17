from typing import Any

"""
JOTTY Core Module - Multi-Agent AI Framework
=============================================

- Orchestrator: Multi-agent coordination
- Cortex: Hierarchical memory (SwarmMemory)
- Axon: Agent communication channel
- Roadmap: Task planning (SwarmTaskBoard)

Heavy modules are lazily loaded — ``import core`` is lightweight.
"""

import importlib as _importlib

# =============================================================================
# PYDANTIC WARNING SUPPRESSION (Must be before any imports)
# =============================================================================
import os
import warnings

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

# =============================================================================
# EAGER: Only truly lightweight (stdlib/typing) — everything else is LAZY.
# =============================================================================

# =============================================================================
# LAZY: Heavy modules loaded on first attribute access
# =============================================================================

_LAZY_IMPORTS: dict[str, str] = {
    # --- jotty.py facade (was eager, now lazy for 2.9s startup saving) ---
    "Orchestrator": ".jotty",
    "SwarmConfig": ".jotty",
    "AgentConfig": ".jotty",
    "ValidatorAgent": ".jotty",
    "IterativeAuditor": ".jotty",
    "Cortex": ".jotty",
    "Axon": ".jotty",
    "Roadmap": ".jotty",
    "Checkpoint": ".jotty",
    "TemporalLearner": ".jotty",
    "RewardLearner": ".jotty",
    "ContributionEstimator": ".jotty",
    "ImpactEstimator": ".jotty",
    "ContextSentinel": ".jotty",
    "Focus": ".jotty",
    "Segmenter": ".jotty",
    "Distiller": ".jotty",
    "Datastream": ".jotty",
    "Blackboard": ".jotty",
    "Catalog": ".jotty",
    "Vault": ".jotty",
    "Chronicle": ".jotty",
    "CooperationPrinciples": ".jotty",
    "NashBargainingSolver": ".jotty",
    "CooperationReasoner": ".jotty",
    "PredictiveCooperativeAgent": ".jotty",
    "ContextGradient": ".jotty",
    "ContextApplier": ".jotty",
    "ContextUpdate": ".jotty",
    "create_swarm_manager": ".jotty",
    "create_cortex": ".jotty",
    "create_axon": ".jotty",
    "create_roadmap": ".jotty",
    # --- API Layer ---
    "ChatExecutor": ".intelligence.orchestration.execution.unified_executor",
    "JottyAPI": ".jotty",
    "ChatAPI": ".jotty",
    "WorkflowAPI": ".jotty",
    "OptimizationPipeline": ".jotty",
    "OptimizationConfig": ".jotty",
    "IterationResult": ".jotty",
    "create_optimization_pipeline": ".jotty",
    # --- foundation.data_structures ---
    # NOTE: MemoryEntry, MemoryLevel, GoalValue, ValidationResult, EpisodeResult,
    # TaggedOutput, OutputTag, StoredEpisode, LearningMetrics, AlertType,
    # GoalHierarchy, GoalNode, CausalLink, SharedScratchpad, AgentMessage,
    # CommunicationType, AgentContribution, ValidationRound, ContextType,
    # RichObservation were removed during the migration to clean architecture.
    # --- intelligence.orchestration.state.swarm_roadmap ---
    "AgenticState": ".intelligence.orchestration.state.swarm_roadmap",
    "TrajectoryStep": ".intelligence.orchestration.state.swarm_roadmap",
    "DecomposedQFunction": ".intelligence.orchestration.state.swarm_roadmap",
    "SwarmTaskBoard": ".intelligence.orchestration.state.swarm_roadmap",
    "SubtaskState": ".intelligence.orchestration.state.swarm_roadmap",
    # NOTE: TaskStatus was removed (does not exist in swarm_roadmap.py)
    "ThoughtLevelCredit": ".intelligence.orchestration.state.swarm_roadmap",
    "StateCheckpointer": ".intelligence.orchestration.state.swarm_roadmap",
    "TrajectoryPredictor": ".intelligence.orchestration.state.swarm_roadmap",
    "TodoItem": ".intelligence.orchestration.state.swarm_roadmap",
    # --- infrastructure.context (context_manager + models) ---
    "SmartContextManager": ".infrastructure.context.context_manager",
    "ContextChunk": ".infrastructure.context.models",
    "ContextPriority": ".infrastructure.context.models",
    "with_smart_context": ".infrastructure.context.context_manager",
    # --- intelligence.memory.llm_rag ---
    "LLMRAGRetriever": ".intelligence.memory.llm_rag",
    "SlidingWindowChunker": ".intelligence.memory.llm_rag",
    "RecencyValueRanker": ".intelligence.memory.llm_rag",
    "LLMRelevanceScorer": ".intelligence.memory.llm_rag",
    "DeduplicationEngine": ".intelligence.memory.llm_rag",
    "CausalExtractor": ".intelligence.memory.llm_rag",
    # --- intelligence.learning.learning ---
    "TDLambdaLearner": ".intelligence.learning.learning",
    "AdaptiveLearningRate": ".intelligence.learning.learning",
    "IntermediateRewardCalculator": ".intelligence.learning.learning",
    "ReasoningCreditAssigner": ".intelligence.learning.learning",
    "AdaptiveExploration": ".intelligence.learning.learning",
    "LearningHealthMonitor": ".intelligence.learning.learning",
    "DynamicBudgetManager": ".intelligence.learning.learning",
    # --- intelligence.memory.cortex ---
    "SwarmMemory": ".intelligence.memory.cortex",
    "MemoryCluster": ".intelligence.memory.consolidation",
    "MemoryLevelClassifier": ".intelligence.memory.consolidation",
    # --- intelligence.reasoning.tools.inspector ---
    "MultiRoundValidator": ".intelligence.reasoning.tools.inspector",
    "InternalReasoningTool": ".intelligence.reasoning.tools.inspector",
    "CachingToolWrapper": ".intelligence.reasoning.tools.inspector",
    # --- intelligence.learning.offline_learning ---
    "OfflineLearner": ".intelligence.learning.offline_learning",
    "PrioritizedEpisodeBuffer": ".intelligence.learning.offline_learning",
    # --- intelligence.learning.predictive_cooperation ---
    "CooperationState": ".intelligence.learning.predictive_cooperation",
    # --- foundation.agent_config ---
    # AgentConfig already imported eagerly
    # --- intelligence.learning.q_learning ---
    "LLMQPredictor": ".intelligence.learning.q_learning",
    # --- capabilities ---
    "capabilities": ".capabilities",
    "explain": ".capabilities",
    # --- context.context_guard ---
    # NOTE: LLMContextManager was removed (context_guard.py no longer exists).
    # --- intelligence.learning.predictive_marl ---
    "LLMTrajectoryPredictor": ".intelligence.learning.predictive_marl",
    "DivergenceMemory": ".intelligence.learning.predictive_marl",
    "CooperativeCreditAssigner": ".intelligence.learning.predictive_marl",
    "PredictedTrajectory": ".intelligence.learning.predictive_marl",
    "ActualTrajectory": ".intelligence.learning.predictive_marl",
    "Divergence": ".intelligence.learning.predictive_marl",
    "AgentModel": ".intelligence.learning.predictive_marl",
    "PredictedAction": ".intelligence.learning.predictive_marl",
    # --- intelligence.memory.consolidation_engine ---
    "BrainMode": ".intelligence.memory.consolidation_engine",
    "BrainModeConfig": ".intelligence.memory.consolidation_engine",
    "MemoryCandidate": ".intelligence.memory.consolidation_engine",
    "HippocampalExtractor": ".intelligence.memory.consolidation_engine",
    "ConsolidationResult": ".intelligence.memory.consolidation_engine",
    "SharpWaveRippleConsolidator": ".intelligence.memory.consolidation_engine",
    "BrainStateMachine": ".intelligence.memory.consolidation_engine",
    "AgentRole": ".intelligence.memory.consolidation_engine",
    "AgentAbstractor": ".intelligence.memory.consolidation_engine",
    # --- intelligence.memory.memory_orchestrator ---
    "SimpleBrain": ".intelligence.memory.memory_orchestrator",
    "BrainPreset": ".intelligence.memory.memory_orchestrator",
    "ConsolidationTrigger": ".intelligence.memory.memory_orchestrator",
    "Experience": ".intelligence.memory.memory_orchestrator",
    "calculate_chunk_size": ".intelligence.memory.memory_orchestrator",
    "get_model_context": ".intelligence.memory.memory_orchestrator",
    "load_brain_config": ".intelligence.memory.memory_orchestrator",
    # --- infrastructure.foundation.robust_parsing ---
    "parse_float_robust": ".infrastructure.foundation.robust_parsing",
    "parse_bool_robust": ".infrastructure.foundation.robust_parsing",
    "parse_json_robust": ".infrastructure.foundation.robust_parsing",
    "AdaptiveThreshold": ".infrastructure.foundation.robust_parsing",
    "EpsilonGreedy": ".infrastructure.foundation.robust_parsing",
    "safe_hash": ".infrastructure.foundation.robust_parsing",
    "content_similarity": ".infrastructure.foundation.robust_parsing",
    # --- infrastructure.utils.algorithmic_foundations ---
    "ShapleyValueEstimator": ".infrastructure.utils.algorithmic_foundations",
    "DifferenceRewardEstimator": ".infrastructure.utils.algorithmic_foundations",
    "SurpriseEstimator": ".infrastructure.utils.algorithmic_foundations",
    "MutualInformationRetriever": ".infrastructure.utils.algorithmic_foundations",
    "ContentGate": ".infrastructure.utils.algorithmic_foundations",
    # --- infrastructure.utils.algorithmic_foundations (formerly context.global_context_guard) ---
    # GlobalContextGuard and OverflowDetector are stubs in algorithmic_foundations;
    # patch_dspy_with_guard and unpatch_dspy live in infrastructure.context.context_manager.
    "GlobalContextGuard": ".infrastructure.utils.algorithmic_foundations",
    "OverflowDetector": ".infrastructure.utils.algorithmic_foundations",
    "patch_dspy_with_guard": ".infrastructure.utils.algorithmic_foundations",
    "unpatch_dspy": ".infrastructure.utils.algorithmic_foundations",
    # --- intelligence.reasoning.factory.agent_factory ---
    "UniversalRetryHandler": ".intelligence.reasoning.factory.agent_factory",
    "RetryStrategy": ".intelligence.reasoning.factory.agent_factory",
    "AgentResult": ".intelligence.reasoning.factory.agent_factory",
    "PatternDetector": ".intelligence.reasoning.factory.agent_factory",
    "LLMCounterfactualCritic": ".intelligence.reasoning.factory.agent_factory",
    "SelfRAGMemoryRetriever": ".intelligence.reasoning.factory.agent_factory",
    "LLMSurpriseEstimator": ".intelligence.reasoning.factory.agent_factory",
    # --- persistence.persistence ---
    # Vault already imported eagerly via jotty.py
    # --- intelligence.orchestration (PolicyExplorer etc.) ---
    "PolicyExplorer": ".intelligence.orchestration",
    "CreditAssignment": ".intelligence.orchestration",
    "AdaptiveLearning": ".intelligence.orchestration",
}

# --- LOTUS (optional, wrapped in try/except at import time) ---
_LOTUS_NAMES: dict[str, str] = {
    "LotusConfig": ".lotus",
    "CascadeThresholds": ".lotus",
    "CacheConfig": ".lotus",
    "ModelCascade": ".lotus",
    "CascadeResult": ".lotus",
    "ModelTier": ".lotus",
    "SemanticCache": ".lotus",
    "CacheEntry": ".lotus",
    "CacheStats": ".lotus",
    "BatchExecutor": ".lotus",
    "BatchResult": ".lotus",
    "BatchConfig": ".lotus",
    "AdaptiveValidator": ".lotus",
    "ValidationDecision": ".lotus",
    "SemanticOperator": ".lotus",
    "SemFilter": ".lotus",
    "SemMap": ".lotus",
    "SemExtract": ".lotus",
    "SemTopK": ".lotus",
    "SemanticDataFrame": ".lotus",
    "LotusOptimizer": ".lotus",
    "enhance_swarm_manager": ".lotus.integration",
    "create_cascaded_lm": ".lotus.integration",
    "setup_lotus_optimization": ".lotus.integration",
    "LotusEnhancement": ".lotus.integration",
    "LotusSwarmMixin": ".lotus.integration",
}

LOTUS_AVAILABLE = False  # Set to True on first successful lotus import


def __getattr__(name: str) -> Any:
    global LOTUS_AVAILABLE

    # Standard lazy imports
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value

    # LOTUS lazy imports (optional)
    if name in _LOTUS_NAMES:
        try:
            module_path = _LOTUS_NAMES[name]
            module = _importlib.import_module(module_path, __name__)
            value = getattr(module, name)
            globals()[name] = value
            LOTUS_AVAILABLE = True
            return value
        except ImportError:
            raise AttributeError(f"LOTUS not installed: {name!r}")

    if name == "LOTUS_AVAILABLE":
        return LOTUS_AVAILABLE

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

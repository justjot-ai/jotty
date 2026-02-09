"""
JOTTY Core Module - Multi-Agent AI Framework
=============================================

V2 Architecture (SwarmManager-based):
- SwarmManager: Main orchestrator
- JottyCore: Alias for SwarmManager (backward compat)
- Cortex: Hierarchical memory (HierarchicalMemory)
- Axon: Agent communication channel
- Roadmap: Task planning (MarkovianTODO)

Heavy modules are lazily loaded — ``import core`` is lightweight.
"""

# =============================================================================
# PYDANTIC WARNING SUPPRESSION (Must be before any imports)
# =============================================================================
import os
import warnings
import importlib as _importlib

os.environ.setdefault('PYDANTIC_WARNINGS', 'none')

warnings.filterwarnings('ignore', category=UserWarning, module='pydantic.main',
                         message='.*PydanticSerializationUnexpectedValue.*')
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic.*',
                         message='.*serialized value may not be as expected.*')
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic.*',
                         message='.*Expected.*fields but got.*')
warnings.filterwarnings('ignore', category=UserWarning,
                         message='.*Pydantic serializer warnings.*')

# =============================================================================
# EAGER: Only truly lightweight (stdlib/typing) — everything else is LAZY.
# =============================================================================
# Backward compat: these resolve to None until the lazy import fires.
PersistenceManager = None  # Deprecated
create_reval = None  # Deprecated

# =============================================================================
# LAZY: Heavy modules loaded on first attribute access
# =============================================================================

_LAZY_IMPORTS: dict[str, str] = {
    # --- jotty.py facade (was eager, now lazy for 2.9s startup saving) ---
    "SwarmManager": ".jotty",
    "JottyCore": ".jotty",
    "SwarmConfig": ".jotty",
    "JottyConfig": ".jotty",
    "AgentSpec": ".jotty",
    "AgentConfig": ".jotty",
    "InspectorAgent": ".jotty",
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
    "create_conductor": ".jotty",
    "create_cortex": ".jotty",
    "create_axon": ".jotty",
    "create_roadmap": ".jotty",
    # --- jotty.py use cases / api / server ---
    "BaseUseCase": ".jotty",
    "ChatUseCase": ".jotty",
    "WorkflowUseCase": ".jotty",
    "UseCaseResult": ".jotty",
    "UseCaseConfig": ".jotty",
    "ChatContext": ".jotty",
    "ChatMessage": ".jotty",
    "ChatExecutor": ".jotty",
    "ChatOrchestrator": ".jotty",
    "WorkflowContext": ".jotty",
    "WorkflowExecutor": ".jotty",
    "WorkflowOrchestrator": ".jotty",
    "JottyAPI": ".jotty",
    "ChatAPI": ".jotty",
    "WorkflowAPI": ".jotty",
    "OptimizationPipeline": ".jotty",
    "OptimizationConfig": ".jotty",
    "IterationResult": ".jotty",
    "create_optimization_pipeline": ".jotty",
    # --- foundation.data_structures (was eager) ---
    "MemoryEntry": ".foundation.data_structures",
    "MemoryLevel": ".foundation.data_structures",
    "GoalValue": ".foundation.data_structures",
    "ValidationResult": ".foundation.data_structures",
    "EpisodeResult": ".foundation.data_structures",
    "TaggedOutput": ".foundation.data_structures",
    "OutputTag": ".foundation.data_structures",
    "StoredEpisode": ".foundation.data_structures",
    "LearningMetrics": ".foundation.data_structures",
    "AlertType": ".foundation.data_structures",
    "GoalHierarchy": ".foundation.data_structures",
    "GoalNode": ".foundation.data_structures",
    "CausalLink": ".foundation.data_structures",
    "SharedScratchpad": ".foundation.data_structures",
    "AgentMessage": ".foundation.data_structures",
    "CommunicationType": ".foundation.data_structures",
    "AgentContribution": ".foundation.data_structures",
    "ValidationRound": ".foundation.data_structures",
    "ContextType": ".foundation.data_structures",
    "RichObservation": ".foundation.data_structures",
    # --- orchestration.v2.swarm_roadmap ---
    "AgenticState": ".orchestration.v2.swarm_roadmap",
    "TrajectoryStep": ".orchestration.v2.swarm_roadmap",
    "DecomposedQFunction": ".orchestration.v2.swarm_roadmap",
    "MarkovianTODO": ".orchestration.v2.swarm_roadmap",
    "SwarmMarkovianTODO": ".orchestration.v2.swarm_roadmap",
    "SubtaskState": ".orchestration.v2.swarm_roadmap",
    "TaskStatus": ".orchestration.v2.swarm_roadmap",
    "ThoughtLevelCredit": ".orchestration.v2.swarm_roadmap",
    "StateCheckpointer": ".orchestration.v2.swarm_roadmap",
    "TrajectoryPredictor": ".orchestration.v2.swarm_roadmap",
    "TodoItem": ".orchestration.v2.swarm_roadmap",
    # --- integration.framework_decorators ---
    "ContextGuard": ".integration.framework_decorators",
    "JottyDecorator": ".integration.framework_decorators",
    "jotty_wrap": ".integration.framework_decorators",
    "EmptyPromptHandler": ".integration.framework_decorators",
    "JottyEnhanced": ".integration.framework_decorators",
    # --- integration.universal_wrapper ---
    "JottyUniversal": ".integration.universal_wrapper",
    "SmartConfig": ".integration.universal_wrapper",
    "jotty_universal": ".integration.universal_wrapper",
    "detect_actor_type": ".integration.universal_wrapper",
    "ActorType": ".integration.universal_wrapper",
    # --- context.context_manager ---
    "SmartContextManager": ".context.context_manager",
    "ContextChunk": ".context.context_manager",
    "ContextPriority": ".context.context_manager",
    "with_smart_context": ".context.context_manager",
    # --- memory.llm_rag ---
    "LLMRAGRetriever": ".memory.llm_rag",
    "SlidingWindowChunker": ".memory.llm_rag",
    "RecencyValueRanker": ".memory.llm_rag",
    "LLMRelevanceScorer": ".memory.llm_rag",
    "DeduplicationEngine": ".memory.llm_rag",
    "CausalExtractor": ".memory.llm_rag",
    # --- learning.learning ---
    "TDLambdaLearner": ".learning.learning",
    "AdaptiveLearningRate": ".learning.learning",
    "IntermediateRewardCalculator": ".learning.learning",
    "ReasoningCreditAssigner": ".learning.learning",
    "AdaptiveExploration": ".learning.learning",
    "LearningHealthMonitor": ".learning.learning",
    "DynamicBudgetManager": ".learning.learning",
    # --- memory.cortex ---
    "HierarchicalMemory": ".memory.cortex",
    "MemoryCluster": ".memory.cortex",
    "MemoryLevelClassifier": ".memory.cortex",
    # --- agents.inspector ---
    "MultiRoundValidator": ".agents.inspector",
    "InternalReasoningTool": ".agents.inspector",
    "CachingToolWrapper": ".agents.inspector",
    # --- learning.offline_learning ---
    "OfflineLearner": ".learning.offline_learning",
    "PrioritizedEpisodeBuffer": ".learning.offline_learning",
    # --- learning.predictive_cooperation ---
    "CooperationState": ".learning.predictive_cooperation",
    # --- foundation.agent_config ---
    # AgentConfig already imported eagerly
    # --- learning.q_learning ---
    "LLMQPredictor": ".learning.q_learning",
    # --- context.context_guard ---
    "SmartContextGuard": ".context.context_guard",
    # --- learning.predictive_marl ---
    "LLMTrajectoryPredictor": ".learning.predictive_marl",
    "DivergenceMemory": ".learning.predictive_marl",
    "CooperativeCreditAssigner": ".learning.predictive_marl",
    "PredictedTrajectory": ".learning.predictive_marl",
    "ActualTrajectory": ".learning.predictive_marl",
    "Divergence": ".learning.predictive_marl",
    "AgentModel": ".learning.predictive_marl",
    "PredictedAction": ".learning.predictive_marl",
    # --- memory.consolidation_engine ---
    "BrainMode": ".memory.consolidation_engine",
    "BrainModeConfig": ".memory.consolidation_engine",
    "MemoryCandidate": ".memory.consolidation_engine",
    "HippocampalExtractor": ".memory.consolidation_engine",
    "ConsolidationResult": ".memory.consolidation_engine",
    "SharpWaveRippleConsolidator": ".memory.consolidation_engine",
    "BrainStateMachine": ".memory.consolidation_engine",
    "AgentRole": ".memory.consolidation_engine",
    "AgentAbstractor": ".memory.consolidation_engine",
    # --- memory.memory_orchestrator ---
    "SimpleBrain": ".memory.memory_orchestrator",
    "BrainPreset": ".memory.memory_orchestrator",
    "ConsolidationTrigger": ".memory.memory_orchestrator",
    "Experience": ".memory.memory_orchestrator",
    "calculate_chunk_size": ".memory.memory_orchestrator",
    "get_model_context": ".memory.memory_orchestrator",
    "load_brain_config": ".memory.memory_orchestrator",
    # --- foundation.robust_parsing ---
    "parse_float_robust": ".foundation.robust_parsing",
    "parse_bool_robust": ".foundation.robust_parsing",
    "parse_json_robust": ".foundation.robust_parsing",
    "AdaptiveThreshold": ".foundation.robust_parsing",
    "EpsilonGreedy": ".foundation.robust_parsing",
    "safe_hash": ".foundation.robust_parsing",
    "content_similarity": ".foundation.robust_parsing",
    # --- utils.algorithmic_foundations ---
    "ShapleyValueEstimator": ".utils.algorithmic_foundations",
    "DifferenceRewardEstimator": ".utils.algorithmic_foundations",
    "SurpriseEstimator": ".utils.algorithmic_foundations",
    "MutualInformationRetriever": ".utils.algorithmic_foundations",
    "UniversalContextGuard": ".utils.algorithmic_foundations",
    "ContextAwareDocumentProcessor": ".utils.algorithmic_foundations",
    # --- context.global_context_guard ---
    "GlobalContextGuard": ".context.global_context_guard",
    "OverflowDetector": ".context.global_context_guard",
    "patch_dspy_with_guard": ".context.global_context_guard",
    "unpatch_dspy": ".context.global_context_guard",
    # --- integration.integration ---
    "JottyIntegration": ".integration.integration",
    "initialize_jotty": ".integration.integration",
    "get_jotty": ".integration.integration",
    # --- agents.agent_factory ---
    "UniversalRetryHandler": ".agents.agent_factory",
    "RetryStrategy": ".agents.agent_factory",
    "AgentResult": ".agents.agent_factory",
    "PatternDetector": ".agents.agent_factory",
    "LLMCounterfactualCritic": ".agents.agent_factory",
    "SelfRAGMemoryRetriever": ".agents.agent_factory",
    "LLMSurpriseEstimator": ".agents.agent_factory",
    # --- persistence.persistence ---
    # Vault already imported eagerly via jotty.py
    # --- orchestration (PolicyExplorer etc.) ---
    "PolicyExplorer": ".orchestration",
    "CreditAssignment": ".orchestration",
    "AdaptiveLearning": ".orchestration",
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


def __getattr__(name: str):
    global LOTUS_AVAILABLE

    # Standard lazy imports
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        # Handle SwarmMarkovianTODO alias
        attr_name = "MarkovianTODO" if name == "SwarmMarkovianTODO" else name
        value = getattr(module, attr_name)
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

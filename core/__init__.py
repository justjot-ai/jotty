"""
JOTTY v1.0 Core Module - Brain-Inspired Multi-Agent Orchestration
=====================================================================

Evolved from Jotty v6.1 with complete naming overhaul.

JOTTY Terminology:
- Conductor = Main orchestrator (was Conductor)
- JottyCore = Core execution engine 
- Architect = Pre-execution planner 
- Auditor = Post-execution validator 
- Cortex = Hierarchical memory (was HierarchicalMemory)
- Axon = Agent communication channel (was AgentSlack)
- Roadmap = Task planning (was MarkovianTODO)

All old names remain available for backward compatibility.
"""

# =============================================================================
# JOTTY v1.0 EXPORTS (New Names)
# =============================================================================
from .jotty import (
    # Core Orchestration
    Conductor,
    JottyCore,
    SwarmConfig,
    JottyConfig,  # Backward compatibility
    AgentSpec,
    AgentConfig,  # Backward compatibility
    
    # Planning & Validation
    InspectorAgent,
    IterativeAuditor,
    
    # Memory
    Cortex,
    
    # Communication
    Axon,
    
    # Task Management
    Roadmap,
    Checkpoint,
    
    # Learning
    TemporalLearner,
    RewardLearner,
    
    # Credit Assignment
    ContributionEstimator,
    ImpactEstimator,
    
    # Context Management
    ContextSentinel,
    Focus,
    Segmenter,
    Distiller,
    
    # Data Flow
    Datastream,
    Blackboard,
    Catalog,
    
    # Persistence
    Vault,
    Chronicle,
    
    # Cooperation
    CooperationPrinciples,
    NashBargainingSolver,
    CooperationReasoner,
    PredictiveCooperativeAgent,
    
    # Context Gradient
    ContextGradient,
    ContextApplier,
    ContextUpdate,
    
    # Convenience Functions
    create_conductor,
    create_cortex,
    create_axon,
    create_roadmap,
)

# =============================================================================
# BACKWARD COMPATIBILITY: Jotty v6.1 Exports (Old Names Still Work)
# =============================================================================

from .foundation.data_structures import (
    SwarmConfig,
    JottyConfig,  # Backward compatibility
    MemoryEntry,
    MemoryLevel,
    GoalValue,
    ValidationResult,
    EpisodeResult,
    TaggedOutput,
    OutputTag,
    StoredEpisode,
    LearningMetrics,
    AlertType,
    GoalHierarchy,
    GoalNode,
    CausalLink,
    SharedScratchpad,
    AgentMessage,
    CommunicationType,
    AgentContribution,
    ValidationRound,
    # v7.5 A-Team Enhancements
    ContextType,
    RichObservation
)

# v6.1 A-Team Enhancements: Rich State Representation
from .orchestration.roadmap import (
    AgenticState,
    TrajectoryStep,
    DecomposedQFunction,
    MarkovianTODO,
    SubtaskState,
    TaskStatus,
    ThoughtLevelCredit,
    StateCheckpointer,
    TrajectoryPredictor
)

# v6.2 A-Team Critical Fixes (JOTTY v1.0 naming)
from .integration.framework_decorators import (
    ContextGuard,
    JottyDecorator,
    jotty_wrap,
    EmptyPromptHandler,
    JottyEnhanced
)

# v6.3 Universal Wrapper (JOTTY v1.0 naming)
from .integration.universal_wrapper import (
    JottyUniversal,
    SmartConfig,
    jotty_universal,
    detect_actor_type,
    ActorType
)

# v6.4 Smart Context Manager (Task-aware context compression)
from .context.context_manager import (
    SmartContextManager,
    ContextChunk,
    ContextPriority,
    with_smart_context
)

from .memory.llm_rag import (
    LLMRAGRetriever,
    SlidingWindowChunker,
    RecencyValueRanker,
    LLMRelevanceScorer,
    DeduplicationEngine,
    CausalExtractor
)

from .learning.learning import (
    TDLambdaLearner,
    AdaptiveLearningRate,
    IntermediateRewardCalculator,
    ReasoningCreditAssigner,
    AdaptiveExploration,
    LearningHealthMonitor,
    DynamicBudgetManager
)

from .memory.cortex import (
    HierarchicalMemory,
    MemoryCluster,
    # v7.5 A-Team Enhancement
    MemoryLevelClassifier
)

from .agents.inspector import (
    InspectorAgent,
    MultiRoundValidator,
    InternalReasoningTool,
    CachingToolWrapper
)

from .learning.offline_learning import (
    OfflineLearner,
    PrioritizedEpisodeBuffer
)

# 🆕 A-TEAM: Predictive Cooperative Swarm (Game Theory + MARL)
from .learning.predictive_cooperation import (
    CooperationPrinciples,
    NashBargainingSolver,
    CooperationReasoner,
    PredictiveCooperativeAgent,
    CooperationState
)

# 🆕 A-TEAM: Context Gradient (Learning mechanism for LLM-based RL)
from .context.context_gradient import (
    ContextGradient,
    ContextApplier,
    ContextUpdate
)

from .orchestration.jotty_core import (
    JottyCore,
    PersistenceManager,
    create_reval
)

# v9.1 Complete Persistence Manager - NO HARDCODING
from .persistence.persistence import (
    Vault
)

# v7.0 Conductor - Multi-Actor Orchestration (A-Team Enhancement)
from .orchestration.conductor import (
    Conductor,
    create_conductor
)
from .foundation.agent_config import AgentConfig
from .orchestration.roadmap import MarkovianTODO as SwarmMarkovianTODO, TodoItem
from .learning.q_learning import LLMQPredictor
from .context.context_guard import SmartContextGuard
from .orchestration.policy_explorer import PolicyExplorer
# SwarmLearner is now internal to swarm_reval, not exported

# v7.1 Predictive MARL - Multi-Agent Trajectory Prediction (Vision Implementation)
from .learning.predictive_marl import (
    LLMTrajectoryPredictor,
    DivergenceMemory,
    CooperativeCreditAssigner,
    PredictedTrajectory,
    ActualTrajectory,
    Divergence,
    AgentModel,
    PredictedAction
)

# v7.2 Brain-Inspired Modes - Neuroscience-Based Memory and Learning
from .memory.consolidation_engine import (
    BrainMode,
    BrainModeConfig,
    MemoryCandidate,
    HippocampalExtractor,
    ConsolidationResult,
    SharpWaveRippleConsolidator,
    BrainStateMachine,
    AgentRole,
    AgentAbstractor
)

# v7.3 Memory Orchestrator - Unified Memory API
from .memory.memory_orchestrator import (
    SimpleBrain,
    BrainPreset,
    ConsolidationTrigger,
    Experience,
    calculate_chunk_size,
    get_model_context,
    load_brain_config
)

# v7.4 Robust Parsing - A-Team Approved Generic Utilities
from .foundation.robust_parsing import (
    parse_float_robust,
    parse_bool_robust,
    parse_json_robust,
    AdaptiveThreshold,
    EpsilonGreedy,
    safe_hash,
    content_similarity
)

# v8.0 Algorithmic Foundations - NO HARDCODING, Principled Math
from .utils.algorithmic_foundations import (
    # Credit Assignment (Shapley Value, Difference Rewards)
    ShapleyValueEstimator,
    DifferenceRewardEstimator,

    # Information Theory (Self-Information, Mutual Information)
    SurpriseEstimator,
    MutualInformationRetriever,

    # Universal Context Management
    UniversalContextGuard,
    ContextAwareDocumentProcessor
)

# v8.0 Global Context Guard - Protects ALL DSPy calls
from .context.global_context_guard import (
    GlobalContextGuard,
    OverflowDetector,
    patch_dspy_with_guard,
    unpatch_dspy
)

# v8.0 Integration - Ties everything together
from .integration.integration import (
    JottyIntegration,
    initialize_jotty,
    get_jotty
)

# v9.0 Modern Agents - NO HEURISTICS, ONLY AGENTS
from .agents.agent_factory import (
    # Retry handling (replaces heuristic fallbacks)
    UniversalRetryHandler,
    RetryStrategy,
    AgentResult,

    # Pattern detection (replaces ALL keyword matching)
    PatternDetector,

    # Credit assignment (COMA 2018, not Shapley 1953)
    LLMCounterfactualCritic,

    # Memory retrieval (Self-RAG 2023, not embedding models)
    SelfRAGMemoryRetriever,

    # Surprise estimation (LLM-based, not Shannon 1948 formula)
    LLMSurpriseEstimator
)

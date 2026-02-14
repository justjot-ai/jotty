"""
Jotty v6.0 - Advanced Data Structures
=====================================

All enhancements from A-Team review incorporated:
- Dr. Manning: Adaptive learning rates, intermediate rewards, value generalization
- Dr. Chen: Inter-agent communication, multi-round validation, reasoning-based credit
- Dr. Agarwal: LLM-based semantic retrieval, dynamic budget, size-aware storage
- Aristotle: Causal understanding, goal hierarchy, conditional wisdom
- Shannon: Deduplication, compression, mutual information
- Alex: JSON/SQLite persistence, distributed support, caching

NO EMBEDDING MODELS - Uses LLM-based semantic matching with sliding window.

REFACTORING NOTE (Phase 1.1):
This module has been split into organized sub-modules in the types/ package.
All classes are re-exported here for backward compatibility.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
from enum import Enum, auto
from datetime import datetime
from pathlib import Path
import hashlib
import json

# =============================================================================
# RE-EXPORTS FROM TYPES PACKAGE (Backward Compatibility)
# =============================================================================

# Import all types from the new organized structure
from .types import (
    # Enums
    MemoryLevel,
    OutputTag,
    AlertType,
    CommunicationType,
    ValidationRound,
    ContextType,
    TaskStatus,
    ExecutorType,
    # Memory types
    GoalNode,
    GoalHierarchy,
    CausalLink,
    GoalValue,
    MemoryEntry,
    # Learning types
    TaggedOutput,
    EpisodeResult,
    StoredEpisode,
    LearningMetrics,
    # Agent types
    AgentContribution,
    AgentMessage,
    SharedScratchpad,
    # Validation types
    ValidationResult,
    # Workflow types
    RichObservation,
    # SDK types
    ExecutionMode,
    ChannelType,
    SDKEventType,
    ResponseFormat,
    ExecutionContext,
    SDKEvent,
    SDKSession,
    SDKResponse,
    SDKRequest,
)

# Re-export everything for backward compatibility
__all__ = [
    # Enums
    'MemoryLevel',
    'OutputTag',
    'AlertType',
    'CommunicationType',
    'ValidationRound',
    'ContextType',
    'TaskStatus',
    'ExecutorType',
    # Memory types
    'GoalNode',
    'GoalHierarchy',
    'CausalLink',
    'GoalValue',
    'MemoryEntry',
    # Learning types
    'TaggedOutput',
    'EpisodeResult',
    'StoredEpisode',
    'LearningMetrics',
    # Agent types
    'AgentContribution',
    'AgentMessage',
    'SharedScratchpad',
    # Validation types
    'ValidationResult',
    # Workflow types
    'RichObservation',
    # SDK types
    'ExecutionMode',
    'ChannelType',
    'SDKEventType',
    'ResponseFormat',
    'ExecutionContext',
    'SDKEvent',
    'SDKSession',
    'SDKResponse',
    'SDKRequest',
    # Configuration
    'SwarmConfig',
    # Config views
    'PersistenceView',
    'ExecutionView',
    'MemoryView',
    'ContextBudgetView',
    'LearningView',
    'ValidationView',
    'MonitoringView',
    'SwarmIntelligenceView',
    # Parameter aliases
    'DEFAULT_PARAM_ALIASES',
    # Focused configs (Phase 3)
    'PersistenceConfig',
    'ExecutionConfig',
    'MemoryConfig',
    'ContextBudgetConfig',
    'LearningConfig',
    'ValidationConfig',
    'MonitoringConfig',
    'IntelligenceConfig',
]

# Lazy re-export of focused configs for convenient importing
def __getattr__(name: str) -> Any:
    _CONFIG_MAP = {
        'PersistenceConfig': '.configs.persistence',
        'ExecutionConfig': '.configs.execution',
        'MemoryConfig': '.configs.memory',
        'ContextBudgetConfig': '.configs.context_budget',
        'LearningConfig': '.configs.learning',
        'ValidationConfig': '.configs.validation',
        'MonitoringConfig': '.configs.monitoring',
        'IntelligenceConfig': '.configs.intelligence',
    }
    if name in _CONFIG_MAP:
        import importlib
        mod = importlib.import_module(_CONFIG_MAP[name], __package__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# DEFAULT PARAMETER ALIASES (A-Team: Centralized Source of Truth)
# =============================================================================

DEFAULT_PARAM_ALIASES: Dict[str, List[str]] = {
    # Table-related aliases
    'tables': ['relevant_tables', 'selected_tables', 'table_list', 'available_tables', 'get_all_tables'],
    'table_names': ['available_tables', 'all_tables', 'tables', 'relevant_tables', 'selected_tables', 'table_list', 'get_all_tables'],

    # Column-related aliases
    'columns': ['selected_columns', 'column_list', 'relevant_columns', 'available_columns'],
    'columns_metadata': ['column_metadata', 'columns_info'],

    # Business terms aliases
    'resolved_terms': ['terms', 'business_terms', 'term_mapping', 'get_business_terms'],
    'business_terms': ['get_business_terms', 'business_context', 'get_business_context'],

    # Filter/condition aliases
    'filters': ['filter_conditions', 'where_conditions'],

    # Metadata aliases
    'tables_metadata': ['get_all_table_metadata', 'table_metadata', 'schema_info'],

    # Generic content aliases
    'content': ['content', 'text', 'body'],
    'data': ['data', 'output_data', 'results'],
    'file': ['file', 'filepath', 'path', 'file_path'],
    'url': ['url', 'uri', 'link', 'href'],
}


# =============================================================================
# CONFIG VIEW SYSTEM (Phase 1b: Organized Sub-Config Access)
# =============================================================================
# SwarmConfig keeps all flat fields for backward compatibility.
# Views provide organized access: config.execution.max_actor_iters
# Views are read/write proxies — changes propagate to the parent config.

class _ConfigView:
    """Base class for SwarmConfig sub-views. Proxies attribute access to the parent."""
    __slots__ = ('_parent',)
    _FIELDS: frozenset = frozenset()

    def __init__(self, parent: 'SwarmConfig') -> None:
        object.__setattr__(self, '_parent', parent)

    def __getattr__(self, name: str) -> Any:
        if name in type(self)._FIELDS:
            return getattr(object.__getattribute__(self, '_parent'), name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in type(self)._FIELDS:
            setattr(object.__getattribute__(self, '_parent'), name, value)
            return
        object.__setattr__(self, name, value)

    def to_dict(self) -> Dict[str, Any]:
        """Return this view's fields as a flat dictionary."""
        parent = object.__getattribute__(self, '_parent')
        return {f: getattr(parent, f) for f in sorted(type(self)._FIELDS)}

    def __repr__(self) -> str:
        items = ', '.join(f'{k}={v!r}' for k, v in self.to_dict().items())
        return f"{type(self).__name__}({items})"


class PersistenceView(_ConfigView):
    """Storage and persistence settings."""
    _FIELDS = frozenset({
        'output_base_dir', 'create_run_folder',
        'auto_save_interval', 'auto_load_on_start', 'save_interval',
        'persist_memories', 'persist_q_tables', 'persist_brain_state',
        'persist_todos', 'persist_agent_outputs',
        'storage_format', 'compress_large_files',
        'max_runs_to_keep', 'enable_backups', 'backup_interval', 'max_backups',
        'base_path', 'auto_load', 'auto_save',
    })


class ExecutionView(_ConfigView):
    """Runtime execution limits, timeouts, and parallelism."""
    _FIELDS = frozenset({
        'max_actor_iters', 'max_eval_iters', 'max_episode_iterations',
        'async_timeout', 'actor_timeout', 'max_concurrent_agents',
        'max_eval_retries', 'llm_timeout_seconds',
        'parallel_architect', 'parallel_auditor',
        'random_seed', 'numpy_seed', 'torch_seed',
        'python_hash_seed', 'enable_deterministic',
    })


class MemoryView(_ConfigView):
    """Memory capacities and RAG retrieval settings."""
    _FIELDS = frozenset({
        'episodic_capacity', 'semantic_capacity', 'procedural_capacity',
        'meta_capacity', 'causal_capacity', 'max_entry_tokens',
        'enable_llm_rag', 'rag_window_size', 'rag_max_candidates',
        'rag_relevance_threshold', 'rag_use_cot',
        'retrieval_mode', 'synthesis_fetch_size', 'synthesis_max_tokens',
        'chunk_size', 'chunk_overlap',
    })


class ContextBudgetView(_ConfigView):
    """Token budget allocation and context management."""
    _FIELDS = frozenset({
        'max_context_tokens', 'system_prompt_budget', 'current_input_budget',
        'trajectory_budget', 'tool_output_budget',
        'enable_dynamic_budget', 'min_memory_budget', 'max_memory_budget',
        'token_model_name',
    })


class LearningView(_ConfigView):
    """RL, exploration, credit assignment, consolidation, and protection."""
    _FIELDS = frozenset({
        # Q-Learning / TD
        'auto_load_learning', 'per_agent_learning', 'shared_learning',
        'learning_alpha', 'learning_gamma', 'learning_epsilon',
        'max_q_table_size', 'q_prune_percentage', 'enable_domain_transfer',
        'enable_rl', 'rl_verbosity',
        'gamma', 'lambda_trace', 'alpha',
        'enable_adaptive_alpha', 'alpha_min', 'alpha_max', 'alpha_adaptation_rate',
        'q_value_mode',
        # Intermediate rewards
        'enable_intermediate_rewards', 'architect_proceed_reward', 'tool_success_reward',
        # Cooperative rewards
        'base_reward_weight', 'cooperation_bonus', 'predictability_bonus',
        'adaptive_window_size', 'instability_threshold_multiplier',
        'slow_learning_threshold', 'goal_transfer_discount',
        # Exploration
        'epsilon_start', 'epsilon_end', 'epsilon_decay_episodes', 'ucb_coefficient',
        'enable_adaptive_exploration', 'exploration_boost_on_stall',
        'max_exploration_iterations', 'policy_update_threshold',
        # Credit assignment
        'credit_decay', 'min_contribution',
        'enable_reasoning_credit', 'reasoning_weight', 'evidence_weight',
        # Consolidation
        'consolidation_threshold', 'consolidation_interval',
        'min_cluster_size', 'pattern_confidence_threshold',
        # Offline learning
        'episode_buffer_size', 'offline_update_interval',
        'replay_batch_size', 'counterfactual_samples',
        # Adaptive learning
        'enable_adaptive_learning', 'stall_detection_window',
        'stall_threshold', 'learning_boost_factor',
        # Deduplication
        'enable_deduplication', 'similarity_threshold',
        # Components
        'learning_components',
        # Goal hierarchy
        'enable_goal_hierarchy', 'goal_transfer_weight',
        # Causal learning
        'enable_causal_learning', 'causal_confidence_threshold', 'causal_min_support',
        # Protection
        'protected_memory_threshold', 'suspicion_threshold', 'min_rejection_rate',
    })


class ValidationView(_ConfigView):
    """Validation and multi-round settings."""
    _FIELDS = frozenset({
        'max_validation_rounds', 'refinement_timeout',
        'enable_validation', 'validation_mode',
        'require_all_architect', 'require_all_auditor',
        'enable_per_actor_swarm_auditor', 'enable_final_swarm_auditor',
        'swarm_validation_confidence_threshold', 'min_confidence',
        'default_confidence_on_error', 'default_confidence_no_validation',
        'default_confidence_insight_share', 'default_estimated_reward',
        'enable_multi_round', 'refinement_on_low_confidence',
        'refinement_on_disagreement', 'max_refinement_rounds',
    })


class MonitoringView(_ConfigView):
    """Logging, profiling, and budget enforcement."""
    _FIELDS = frozenset({
        'enable_debug_logs', 'log_level',
        'enable_profiling',
        'verbose', 'log_file',
        'enable_debug_logging', 'enable_metrics',
        'enable_monitoring', 'baseline_cost_per_success',
        'max_llm_calls_per_episode', 'max_llm_calls_per_agent',
        'max_total_tokens_per_episode',
        'enable_budget_enforcement', 'budget_warning_threshold',
    })


class SwarmIntelligenceView(_ConfigView):
    """Trust tuning, routing, agent communication, and local mode."""
    _FIELDS = frozenset({
        'trust_decrease_on_struggle', 'trust_increase_on_excel', 'trust_min',
        'adaptation_interval', 'adaptation_struggle_threshold',
        'adaptation_excel_threshold',
        'stigmergy_routing_threshold', 'morph_min_rcs',
        'judge_intervention_confidence',
        'memory_retrieval_budget', 'collective_memory_limit',
        'enable_agent_communication', 'share_tool_results',
        'share_insights', 'max_messages_per_episode',
        'local_mode', 'local_model',
    })


# =============================================================================
# CONFIGURATION (Complete with all enhancements)
# =============================================================================

@dataclass
class SwarmConfig:
    """
    Swarm Orchestration Configuration.

    Schema-versioned for safe JSON persistence roundtrips.
    Pruned to active parameters only — dead params removed 2026-02-10.

    Categories:
    1. PERSISTENCE - State storage
    2. EXECUTION - Runtime limits
    3. MEMORY - Capacity and hierarchy
    4. CONTEXT BUDGET - Token allocation
    5. RL PARAMETERS - TD(λ) learning
    6. EXPLORATION - ε-greedy + UCB
    7. CREDIT ASSIGNMENT - Agent contribution
    8. CONSOLIDATION - Pattern extraction
    9. OFFLINE LEARNING - Batch updates
    10. PROTECTION - Forgetting prevention
    11. VALIDATION - Decision logic
    12. ASYNC - Parallelism
    13. LOGGING - Verbosity
    14. LLM RAG - Semantic retrieval
    15. GOAL HIERARCHY - Knowledge transfer
    16. CAUSAL LEARNING - Why understanding
    17. INTER-AGENT - Communication
    18. MULTI-ROUND - Iterative validation
    19. ADAPTIVE LEARNING - Dynamic parameters
    20. DEDUPLICATION - Redundancy removal
    """

    # Schema version — increment on breaking changes to persisted JSON.
    # Loaded files with a different major version trigger a migration warning.
    schema_version: str = "2.0"

    # =========================================================================
    # 1. PERSISTENCE
    # =========================================================================
    # A-TEAM: Single source of truth for ALL persistence paths
    output_base_dir: str = "./outputs"  # Base directory for all outputs
    create_run_folder: bool = True  # Create timestamped run_YYYYMMDD_HHMMSS/ folders

    # Auto-behavior
    auto_save_interval: int = 3  # Save state every N steps
    auto_load_on_start: bool = True  # Auto-load from outputs/latest/ if exists
    save_interval: int = 1  # Legacy: Episodes between saves (keep for compatibility)

    # What to persist
    persist_memories: bool = True
    persist_q_tables: bool = True
    persist_brain_state: bool = True
    persist_todos: bool = True  # Save session TODOs to markdown
    persist_agent_outputs: bool = True  # Save IOManager outputs

    # Storage format
    storage_format: str = "json"  # "json" or "sqlite" (not pickle!)
    compress_large_files: bool = True  # Gzip files > 1MB

    # Retention & cleanup
    max_runs_to_keep: int = 10  # Auto-cleanup old run folders
    enable_backups: bool = True
    backup_interval: int = 100  # Episodes between backups
    max_backups: int = 10

    # Learning configuration
    auto_load_learning: bool = True  # Automatically load previous learning on startup
    per_agent_learning: bool = True  # Each agent has its own Q-table
    shared_learning: bool = True  # Also maintain shared cross-agent learning
    learning_alpha: float = 0.3  # Q-learning rate
    learning_gamma: float = 0.9  # Discount factor
    learning_epsilon: float = 0.1  # Exploration rate
    max_q_table_size: int = 10000  # Max Q-table entries before pruning
    q_prune_percentage: float = 0.2  # Prune 20% when limit hit
    enable_domain_transfer: bool = True  # Load learning from similar domains

    # Logs
    enable_debug_logs: bool = True  # Keep raw debug logs
    log_level: str = "INFO"  # Logging verbosity

    # Profiling
    enable_profiling: bool = False  # Track execution times for performance analysis

    # Legacy paths (keep for backward compatibility)
    base_path: str = "~/.jotty"  # Old persistence location
    auto_load: bool = True  # Legacy flag
    auto_save: bool = True  # Legacy flag

    # =========================================================================
    # 2. EXECUTION
    # =========================================================================
    max_actor_iters: int = 50 # Configurable! (was hardcoded in agents)
    max_eval_iters: int = 1 # Architect/Auditor ReAct iterations (1=minimal, 2-3=balanced, 5-10=thorough)
    max_episode_iterations: int = 12 # A-TEAM: Max task iterations per episode (used in swarm.run)
    async_timeout: float = 60.0
    actor_timeout: float = 900.0  # Specific timeout for actor execution (15 minutes)
    max_concurrent_agents: int = 10

    # NEW: Agent-specific overrides (can be set in ActorConfig)
    max_eval_retries: int = 3 # Retry attempts for validation (was hardcoded in agent.py)
    llm_timeout_seconds: float = 180.0 # LLM call timeout to prevent API hangs (3 minutes)

    # =========================================================================
    # 2.5 VALIDATION & MULTI-ROUND
    # =========================================================================
    max_validation_rounds: int = 3
    refinement_timeout: float = 30.0

    # NEW: Validation control flags
    enable_validation: bool = True  # Master switch for all validation
    validation_mode: str = 'full'  # 'full' | 'architect_only' | 'auditor_only' | 'none'

    # =========================================================================
    # 3. MEMORY (Hierarchical)
    # =========================================================================
    episodic_capacity: int = 1000
    semantic_capacity: int = 500
    procedural_capacity: int = 200
    meta_capacity: int = 100
    causal_capacity: int = 150  # NEW: Causal knowledge storage

    # NEW: Size limits per entry
    max_entry_tokens: int = 2000  # Prevent oversized entries

    # =========================================================================
    # 4. CONTEXT BUDGET (Shannon)
    # =========================================================================
    max_context_tokens: int = 100000
    system_prompt_budget: int = 5000
    current_input_budget: int = 15000
    trajectory_budget: int = 20000
    tool_output_budget: int = 15000

    # NEW: Dynamic allocation (Dr. Agarwal)
    enable_dynamic_budget: bool = True
    min_memory_budget: int = 10000
    max_memory_budget: int = 60000

    # =========================================================================
    # 4.5 AGENTIC DISCOVERY BUDGET (A-Team: Config-Driven Design)
    # =========================================================================
    # 4.6 TOKEN COUNTING (A-Team: Accurate Token Counting)
    # =========================================================================
    # User requirement: "take token_model_name in config as convention might be different"
    token_model_name: Optional[str] = None  # Override model name for token counting (e.g., 'gpt-4o')
    # If None, uses main model name with automatic mapping

    # =========================================================================
    # 5. RL PARAMETERS (Manning - Corrected TD(λ))
    # =========================================================================
    # When to use RL:
    # - enable_rl=True:  Production systems with repeated, domain-specific tasks
    #                    (e.g., SQL generation, code review, customer support)
    #                    RL learns which agents contribute most to solving that problem class
    # - enable_rl=False: One-off tasks, demos, or completely unrelated tasks
    #                    where past experience doesn't transfer
    enable_rl: bool = True  # Master switch for RL features
    rl_verbosity: str = "quiet"  # "quiet" (minimal), "normal" (info), "verbose" (debug)
    gamma: float = 0.99
    lambda_trace: float = 0.95
    alpha: float = 0.01

    # NEW: Adaptive learning rate (Dr. Manning)
    enable_adaptive_alpha: bool = True
    alpha_min: float = 0.001
    alpha_max: float = 0.1
    alpha_adaptation_rate: float = 0.1

    # 🆕 Q-VALUE CALCULATION MODE (Natural Dependencies vs LLM Prediction)
    # "simple": Average reward per actor (fast, reliable, perfect for natural dependencies)
    # "llm": LLM-based Q-value prediction (USP - semantic generalization across states)
    q_value_mode: str = "simple"  # "simple" or "llm"

    # NEW: Intermediate rewards (Dr. Manning)
    enable_intermediate_rewards: bool = True
    architect_proceed_reward: float = 0.1
    tool_success_reward: float = 0.05

    # 🆕 STANFORD FIX: Cooperative reward weights (multi-agent)
    base_reward_weight: float = 0.3  # Own success contribution
    cooperation_bonus: float = 0.4  # Bonus for helping other agents
    predictability_bonus: float = 0.3  # Bonus for predictable behavior

    # 🆕 STANFORD FIX: Adaptive learning thresholds
    adaptive_window_size: int = 50  # Window for learning rate adaptation
    instability_threshold_multiplier: float = 1.5  # std_dev > mean * this → unstable
    slow_learning_threshold: float = 0.01  # mean_error < this → too slow
    goal_transfer_discount: float = 0.5  # Discount for value transfer to related goals

    # =========================================================================
    # 6. EXPLORATION
    # =========================================================================
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 500
    ucb_coefficient: float = 2.0

    # NEW: Adaptive exploration (Dr. Manning)
    enable_adaptive_exploration: bool = True
    exploration_boost_on_stall: float = 0.1

    # 🆕 STANFORD FIX: Exploration limits
    max_exploration_iterations: int = 10  # Max iterations for policy exploration
    policy_update_threshold: int = 3  # Episodes before updating policy

    # =========================================================================
    # 7. CREDIT ASSIGNMENT
    # =========================================================================
    credit_decay: float = 0.9
    min_contribution: float = 0.1

    # NEW: Reasoning-based credit (Dr. Chen)
    enable_reasoning_credit: bool = True
    reasoning_weight: float = 0.3
    evidence_weight: float = 0.2

    # =========================================================================
    # 8. CONSOLIDATION
    # =========================================================================
    consolidation_threshold: int = 100
    consolidation_interval: int = 3 # A-TEAM: Consolidate every 3 episodes (prevent memory buildup!)
    min_cluster_size: int = 5
    pattern_confidence_threshold: float = 0.7


    # =========================================================================
    # 9. OFFLINE LEARNING
    # =========================================================================
    episode_buffer_size: int = 1000
    offline_update_interval: int = 50
    replay_batch_size: int = 20

    # Counterfactual learning
    counterfactual_samples: int = 5


    # =========================================================================
    # 10. PROTECTION MECHANISMS
    # =========================================================================
    protected_memory_threshold: float = 0.8
    suspicion_threshold: float = 0.95
    min_rejection_rate: float = 0.05

    # =========================================================================
    # 11. VALIDATION
    # =========================================================================
    require_all_architect: bool = True
    require_all_auditor: bool = False

    # A-TEAM FIX: Swarm validation strategy
    enable_per_actor_swarm_auditor: bool = False  # If True, run swarm Auditor after EACH actor (slow!)
    enable_final_swarm_auditor: bool = True       # If True, run swarm Auditor once at END (recommended)
    swarm_validation_confidence_threshold: float = 0.6  # Only retry if confidence below this

    min_confidence: float = 0.5

    # NEW: Default values for validation (was hardcoded 0.3, 0.5, 0.7 in agent.py)
    default_confidence_on_error: float = 0.3  # Confidence when validation errors
    default_confidence_no_validation: float = 0.5  # Confidence when no validation
    default_confidence_insight_share: float = 0.7  # Confidence for shared insights

    # NEW: Reward defaults (was hardcoded in conductor.py)
    default_estimated_reward: float = 0.6  # When no Auditor result yet

    # =========================================================================
    # 12. ASYNC
    # =========================================================================
    parallel_architect: bool = True
    parallel_auditor: bool = True

    # =========================================================================
    # 13. LOGGING
    # =========================================================================
    verbose: int = 1
    log_file: Optional[str] = None

    # 🆕 A-TEAM FIX #2: Debug logging control
    # User reported 144s wasted on debug logs! Default OFF for production.
    enable_debug_logging: bool = False # Set True only for debugging
    enable_metrics: bool = True

    # =========================================================================
    # 🆕 PARAMETER MAPPINGS (A-Team: User-Configurable Genericity!)
    # =========================================================================
    # Allow users to define custom parameter name mappings for their domain
    # Example: {'user_id': ['customer_id', 'uid', 'account_id']}
    custom_param_mappings: Dict[str, List[str]] = field(default_factory=dict)

    # =========================================================================
    # 14. LLM-BASED RAG (NEW - Dr. Agarwal)
    # =========================================================================
    # No embeddings! Uses LLM with sliding window for semantic matching

    enable_llm_rag: bool = True
    rag_window_size: int = 5          # Memories per LLM call for relevance scoring
    rag_max_candidates: int = 50       # Pre-filter before LLM scoring (discrete mode)
    rag_relevance_threshold: float = 0.6  # Minimum relevance score
    rag_use_cot: bool = True          # Chain-of-thought for scoring

    # RETRIEVAL STRATEGY (Brain-Inspired!)
    # synthesize: Fetch broadly + LLM synthesizes wisdom (DEFAULT - neuroscience-aligned!)
    # discrete: Fetch selectively + return discrete memories (legacy, faster but less intelligent)
    retrieval_mode: str = "synthesize"  # "synthesize" or "discrete"
    synthesis_fetch_size: int = 200    # How many memories to fetch for synthesis
    synthesis_max_tokens: int = 800    # Max tokens for synthesized wisdom

    # Sliding window chunking for large content
    chunk_size: int = 500             # Tokens per chunk
    chunk_overlap: int = 50           # Overlap between chunks

    # =========================================================================
    # 15. GOAL HIERARCHY (NEW - Aristotle)
    # =========================================================================
    enable_goal_hierarchy: bool = True
    goal_transfer_weight: float = 0.3  # Weight for transferred knowledge

    # =========================================================================
    # 16. CAUSAL LEARNING (NEW - Aristotle)
    # =========================================================================
    enable_causal_learning: bool = True
    causal_confidence_threshold: float = 0.7
    causal_min_support: int = 3       # Episodes before causal link confirmed

    # =========================================================================
    # 17. INTER-AGENT COMMUNICATION (NEW - Dr. Chen)
    # =========================================================================
    enable_agent_communication: bool = True
    share_tool_results: bool = True    # Cache and share tool results
    share_insights: bool = True        # Share discovered insights
    max_messages_per_episode: int = 20

    # =========================================================================
    # 18. MULTI-ROUND VALIDATION (NEW - Dr. Chen)
    # =========================================================================
    enable_multi_round: bool = True
    refinement_on_low_confidence: float = 0.6  # Trigger refinement below this
    refinement_on_disagreement: bool = True     # Trigger when agents disagree
    max_refinement_rounds: int = 2

    # =========================================================================
    # 19. ADAPTIVE LEARNING (NEW - Dr. Manning)
    # =========================================================================
    enable_adaptive_learning: bool = True
    stall_detection_window: int = 100
    stall_threshold: float = 0.001
    learning_boost_factor: float = 2.0

    # =========================================================================
    # 20. DEDUPLICATION (NEW - Shannon)
    # =========================================================================
    enable_deduplication: bool = True
    similarity_threshold: float = 0.85  # LLM-judged similarity

    # =========================================================================
    # 20.5 LEARNING PIPELINE CONFIGURATION
    # =========================================================================
    # Which components run in post_episode(). None = all (default).
    # Set to a list to enable only specific ones.
    # Valid: 'td_lambda', 'swarm_learner', 'brain_consolidation',
    #   'neurochunk_tiering', 'agent_abstractor', 'transfer_learning',
    #   'swarm_intelligence', 'stigmergy', 'effectiveness', 'mas_learning',
    #   'byzantine', 'credit_assignment', 'auditor_fixes',
    #   'adaptive_learning', 'effectiveness_intervention',
    #   'credit_pruning', 'curriculum'
    learning_components: Optional[List[str]] = None


    # =========================================================================
    # 21.5 LOCAL-FIRST MODE (OpenClaw-Inspired)
    # =========================================================================
    # Enable local-first mode for privacy - no external API calls
    local_mode: bool = False              # Master switch for local-only inference
    local_model: str = "ollama/llama3"    # Local model for agents (Ollama format)

    # =========================================================================
    # 22.5 BUDGET CONTROLS (A-Team Critical Fix)
    # =========================================================================
    # LLM call budget limits
    max_llm_calls_per_episode: int = 100  # Max LLM calls per episode
    max_llm_calls_per_agent: int = 50     # Max LLM calls per agent
    max_total_tokens_per_episode: int = 500000  # Max tokens per episode
    enable_budget_enforcement: bool = True  # Enforce budget limits
    budget_warning_threshold: float = 0.8   # Warn at 80% of budget
    
    # Monitoring framework
    enable_monitoring: bool = False  # Enable comprehensive monitoring (opt-in)
    baseline_cost_per_success: Optional[float] = None  # Baseline for efficiency comparison
    
    # =========================================================================
    # 23. REPRODUCIBILITY & EVALUATION (NEW - OAgents Integration)
    # =========================================================================
    # Reproducibility guarantees
    random_seed: Optional[int] = None  # Fixed random seed for reproducibility
    numpy_seed: Optional[int] = None  # NumPy random seed (if available)
    torch_seed: Optional[int] = None  # PyTorch random seed (if available)
    python_hash_seed: Optional[int] = None  # Python hash randomization seed
    enable_deterministic: bool = True  # Enable deterministic operations
    
    # Context Providers (generic dict for domain-specific context)
    # Examples: {'metadata_manager': obj, 'database': conn, 'api_client': client}
    # JOTTY doesn't know about these - user provides and tools use them
    # NOTE: This is set per-instance, not in config file

    # =========================================================================
    # 22. SWARM INTELLIGENCE TUNING (Previously hardcoded magic numbers)
    # =========================================================================
    # Online adaptation: trust adjustments when agents struggle or excel
    trust_decrease_on_struggle: float = 0.1  # Trust penalty per adaptation window
    trust_increase_on_excel: float = 0.05  # Trust bonus per adaptation window
    trust_min: float = 0.1  # Minimum trust floor
    adaptation_interval: int = 5  # Adapt every N experiences
    adaptation_struggle_threshold: float = 0.3  # Success rate below this = struggling
    adaptation_excel_threshold: float = 0.8  # Success rate above this = excelling

    # Routing thresholds
    stigmergy_routing_threshold: float = 0.5  # Min signal strength for stigmergy routing
    morph_min_rcs: float = 0.3  # Min Role Clarity Score for TRAS routing

    # Validation / judge intervention
    judge_intervention_confidence: float = 0.6  # Auditor confidence below this triggers retry

    # Memory retrieval budgets
    memory_retrieval_budget: int = 3000  # Tokens for memory retrieval in AgentRunner

    # Collective memory
    collective_memory_limit: int = 200  # Max items in swarm collective memory

    # =========================================================================
    # Computed properties
    # =========================================================================
    def __post_init__(self) -> None:
        """Calculate derived config values and validate via focused configs."""
        # Set reproducibility seeds if configured
        if self.random_seed is not None:
            try:
                from ..evaluation.reproducibility import set_reproducible_seeds
                set_reproducible_seeds(
                    random_seed=self.random_seed,
                    numpy_seed=self.numpy_seed or self.random_seed,
                    torch_seed=self.torch_seed or self.random_seed,
                    python_hash_seed=self.python_hash_seed or self.random_seed,
                    enable_deterministic=self.enable_deterministic
                )
            except ImportError:
                pass  # Evaluation module not available

        # Validate via focused configs (catches invalid field combos)
        self._validate_via_focused_configs()

    def _validate_via_focused_configs(self) -> None:
        """Delegate validation to focused configs that have __post_init__ checks."""
        errors = []
        for method_name in [
            'to_memory_config', 'to_learning_config', 'to_context_budget_config',
            'to_execution_config', 'to_persistence_config', 'to_validation_config',
            'to_monitoring_config', 'to_intelligence_config',
        ]:
            try:
                getattr(self, method_name)()
            except (ValueError, TypeError) as e:
                errors.append(f"{method_name}: {e}")
        if errors:
            raise ValueError(
                f"SwarmConfig validation failed:\n" + "\n".join(errors)
            )
        
    @property
    def memory_budget(self) -> int:
        """Compute available tokens for memories."""
        reserved = (
            self.system_prompt_budget +
            self.current_input_budget +
            self.trajectory_budget +
            self.tool_output_budget
        )
        return max(self.min_memory_budget, self.max_context_tokens - reserved)

    @property
    def total_memory_capacity(self) -> int:
        """Total entries across all levels."""
        return (
            self.episodic_capacity +
            self.semantic_capacity +
            self.procedural_capacity +
            self.meta_capacity +
            self.causal_capacity
        )

    # =========================================================================
    # Sub-Config Views (Phase 1b: organized access without breaking flat API)
    # Usage: config.execution.max_actor_iters (read/write proxy to parent)
    # =========================================================================
    @property
    def persistence(self) -> 'PersistenceView':
        """Storage and persistence settings."""
        return PersistenceView(self)

    @property
    def execution(self) -> 'ExecutionView':
        """Runtime execution limits, timeouts, and parallelism."""
        return ExecutionView(self)

    @property
    def memory_settings(self) -> 'MemoryView':
        """Memory capacities and RAG retrieval settings."""
        return MemoryView(self)

    @property
    def context_budget(self) -> 'ContextBudgetView':
        """Token budget allocation."""
        return ContextBudgetView(self)

    @property
    def learning(self) -> 'LearningView':
        """RL, exploration, credit assignment, and consolidation."""
        return LearningView(self)

    @property
    def validation_settings(self) -> 'ValidationView':
        """Validation and multi-round configuration."""
        return ValidationView(self)

    @property
    def monitoring(self) -> 'MonitoringView':
        """Logging, profiling, and budget enforcement."""
        return MonitoringView(self)

    @property
    def intelligence(self) -> 'SwarmIntelligenceView':
        """Trust tuning, routing, and agent communication."""
        return SwarmIntelligenceView(self)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Serialize all config fields to a flat dictionary (replaces dataclasses.asdict)."""
        from dataclasses import fields as dc_fields
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}

    # =========================================================================
    # Focused Config Extraction (Phase 3: subsystems import their own config)
    # Usage: memory_system.init(config.to_memory_config())
    # =========================================================================

    def to_persistence_config(self) -> 'PersistenceConfig':
        """Extract persistence-specific config."""
        from .configs.persistence import PersistenceConfig
        return PersistenceConfig(**{
            f: getattr(self, f) for f in PersistenceView._FIELDS
        })

    def to_execution_config(self) -> 'ExecutionConfig':
        """Extract execution-specific config."""
        from .configs.execution import ExecutionConfig
        return ExecutionConfig(**{
            f: getattr(self, f) for f in ExecutionView._FIELDS
        })

    def to_memory_config(self) -> 'MemoryConfig':
        """Extract memory-specific config."""
        from .configs.memory import MemoryConfig
        return MemoryConfig(**{
            f: getattr(self, f) for f in MemoryView._FIELDS
        })

    def to_context_budget_config(self) -> 'ContextBudgetConfig':
        """Extract context budget config."""
        from .configs.context_budget import ContextBudgetConfig
        return ContextBudgetConfig(**{
            f: getattr(self, f) for f in ContextBudgetView._FIELDS
        })

    def to_learning_config(self) -> 'LearningConfig':
        """Extract learning-specific config."""
        from .configs.learning import LearningConfig
        return LearningConfig(**{
            f: getattr(self, f) for f in LearningView._FIELDS
        })

    def to_validation_config(self) -> 'ValidationConfig':
        """Extract validation-specific config."""
        from .configs.validation import ValidationConfig
        return ValidationConfig(**{
            f: getattr(self, f) for f in ValidationView._FIELDS
        })

    def to_monitoring_config(self) -> 'MonitoringConfig':
        """Extract monitoring-specific config."""
        from .configs.monitoring import MonitoringConfig
        return MonitoringConfig(**{
            f: getattr(self, f) for f in MonitoringView._FIELDS
        })

    def to_intelligence_config(self) -> 'IntelligenceConfig':
        """Extract swarm intelligence config."""
        from .configs.intelligence import IntelligenceConfig
        return IntelligenceConfig(**{
            f: getattr(self, f) for f in SwarmIntelligenceView._FIELDS
        })

    @classmethod
    def from_configs(cls, persistence: 'PersistenceConfig' = None, execution: 'ExecutionConfig' = None, memory: 'MemoryConfig' = None, context_budget: 'ContextBudgetConfig' = None, learning: 'LearningConfig' = None, validation: 'ValidationConfig' = None, monitoring: 'MonitoringConfig' = None, intelligence: 'IntelligenceConfig' = None, **overrides: Any) -> 'SwarmConfig':
        """Build a SwarmConfig by composing focused sub-configs.

        Any fields not covered by sub-configs use defaults.
        Extra keyword overrides are applied last.

        Example:
            config = SwarmConfig.from_configs(
                memory=MemoryConfig(episodic_capacity=2000),
                learning=LearningConfig(gamma=0.95),
            )
        """
        from dataclasses import fields as dc_fields, asdict as dc_asdict
        kwargs: Dict[str, Any] = {}

        for sub_config in [persistence, execution, memory, context_budget,
                           learning, validation, monitoring, intelligence]:
            if sub_config is not None:
                kwargs.update(dc_asdict(sub_config))

        # Overrides win
        kwargs.update(overrides)

        # Filter to valid SwarmConfig fields
        valid_fields = {f.name for f in dc_fields(cls)}
        filtered = {k: v for k, v in kwargs.items() if k in valid_fields}

        return cls(**filtered)

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
    # Configuration
    'SwarmConfig',
    'JottyConfig',  # Backward compatibility alias
    # Parameter aliases
    'DEFAULT_PARAM_ALIASES',
]


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
# CONFIGURATION (Complete with all enhancements)
# =============================================================================

@dataclass
class SwarmConfig:
    """
    Swarm Orchestration Configuration - All A-Team Enhancements

    Categories:
    1. PERSISTENCE - State storage
    2. EXECUTION - Runtime limits
    2.5. TIMEOUT & CIRCUIT BREAKER - Production resilience (NEW)
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
    14. LLM RAG - Semantic retrieval (NEW)
    15. GOAL HIERARCHY - Knowledge transfer (NEW)
    16. CAUSAL LEARNING - Why understanding (NEW)
    17. INTER-AGENT - Communication (NEW)
    18. MULTI-ROUND - Iterative validation (NEW)
    19. ADAPTIVE LEARNING - Dynamic parameters (NEW)
    20. DEDUPLICATION - Redundancy removal (NEW)
    21. DISTRIBUTED - Multi-instance support (NEW)
    """

    # =========================================================================
    # 1. PERSISTENCE (A-Team: Complete Session & State Management)
    # =========================================================================
    # 🔥 A-TEAM: Single source of truth for ALL persistence paths
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
    enable_beautified_logs: bool = True  # Generate human-readable logs
    enable_debug_logs: bool = True  # Keep raw debug logs
    log_level: str = "INFO"  # Logging verbosity

    # Profiling
    enable_profiling: bool = False  # Track execution times for performance analysis
    profiling_verbosity: str = "summary"  # "summary" (end only), "detailed" (per operation)

    # Legacy paths (keep for backward compatibility)
    base_path: str = "~/.jotty"  # Old persistence location
    auto_load: bool = True  # Legacy flag
    auto_save: bool = True  # Legacy flag

    # =========================================================================
    # 2. EXECUTION
    # =========================================================================
    max_actor_iters: int = 50  # ✅ Configurable! (was hardcoded in agents)
    max_eval_iters: int = 1    # ✅ Architect/Auditor ReAct iterations (1=minimal, 2-3=balanced, 5-10=thorough)
    max_episode_iterations: int = 12  # ✅ A-TEAM: Max task iterations per episode (used in swarm.run)
    async_timeout: float = 60.0
    actor_timeout: float = 900.0  # Specific timeout for actor execution (15 minutes)
    max_concurrent_agents: int = 10

    # 🔥 CRITICAL: Natural Dependencies (RL Learning)
    # Allow agents to execute even with missing required parameters
    # Agents can fail naturally when dependencies aren't met (instead of blocking at parameter resolution)
    allow_partial_execution: bool = False  # Default: False (strict parameter checking)
    # Set to True for RL with natural dependencies (agents detect missing data themselves)

    # NEW: Agent-specific overrides (can be set in ActorConfig)
    max_eval_retries: int = 3  # ✅ Retry attempts for validation (was hardcoded in agent.py)
    stream_message_timeout: float = 0.15  # ✅ Streaming timeout (was hardcoded in agents)
    llm_timeout_seconds: float = 180.0  # ⚡ LLM call timeout to prevent API hangs (3 minutes)

    # =========================================================================
    # 2.5 TIMEOUT & CIRCUIT BREAKER (A-Team: Production Resilience)
    # =========================================================================
    # Circuit Breaker Config
    enable_circuit_breakers: bool = True
    llm_circuit_failure_threshold: int = 5  # Failures before opening LLM circuit
    llm_circuit_timeout: float = 60.0  # Seconds before trying half-open
    llm_circuit_success_threshold: int = 2  # Successes to close from half-open
    tool_circuit_failure_threshold: int = 3  # Failures before opening tool circuit
    tool_circuit_timeout: float = 30.0
    tool_circuit_success_threshold: int = 2

    # Adaptive Timeout Config
    enable_adaptive_timeouts: bool = True
    initial_timeout: float = 30.0  # Initial timeout for operations
    timeout_percentile: float = 95.0  # Use 95th percentile of latencies
    min_timeout: float = 5.0  # Minimum adaptive timeout
    max_timeout: float = 300.0  # Maximum adaptive timeout (5 minutes)

    # Dead Letter Queue Config
    enable_dead_letter_queue: bool = True
    dlq_max_size: int = 1000
    dlq_max_retries: int = 3  # Max retry attempts for failed operations

    # NEW: Multi-round limits
    max_validation_rounds: int = 3
    refinement_timeout: float = 30.0

    # ✅ NEW: Validation control flags
    enable_validation: bool = True  # Master switch for all validation
    validation_mode: str = 'full'  # 'full' | 'architect_only' | 'auditor_only' | 'none'
    advisory_confidence_threshold: float = 0.85  # Below this, advisory feedback triggers retry
    max_validation_retries: int = 5  # Increased from 3 for better learning

    # ✅ A-TEAM: Confidence-Based Override Mechanism (Dec 29, 2025)
    enable_confidence_override: bool = True  # Allow confident actors to override uncertain validators
    confidence_override_threshold: float = 0.30  # Min gap (actor - validator) to allow override
    confidence_moving_average_alpha: float = 0.7  # Weight for exponential moving average
    min_confidence_for_override: float = 0.70  # Actor must be at least this confident to override
    max_validator_confidence_to_override: float = 0.95  # Don't override if validator >95% confident

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
    # User requirement: "Have at least 20k tokens" for preview
    # Model has 30k context, should use generously
    preview_token_budget: int = 20000  # For LLM artifact analysis (20k tokens ≈ 80k chars)
    max_description_tokens: int = 5000  # Per artifact description (5k tokens ≈ 20k chars)
    compression_trigger_ratio: float = 0.8  # Only compress when total context > 80% of limit
    chunking_threshold_tokens: int = 15000  # Chunk artifacts larger than 15k tokens

    # Derived values (calculated in __post_init__)
    preview_char_limit: int = None  # preview_token_budget * 4
    max_description_chars: int = None  # max_description_tokens * 4

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
    baseline: float = 0.5
    n_step: int = 3

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
    consolidation_interval: int = 3  # 🔧 A-TEAM: Consolidate every 3 episodes (prevent memory buildup!)
    min_cluster_size: int = 5
    pattern_confidence_threshold: float = 0.7

    # NEW: Causal consolidation (Aristotle)
    enable_causal_extraction: bool = True
    min_causal_evidence: int = 3

    # 🆕 STANFORD FIX: Brain-inspired consolidation (optional features from brain_modes.py)
    brain_reward_salience_weight: float = 0.3  # Weight for reward salience
    brain_novelty_weight: float = 0.4  # Weight for novelty
    brain_goal_relevance_weight: float = 0.3  # Weight for goal relevance
    brain_memory_threshold: float = 0.4  # Threshold for memory retention
    brain_prune_threshold: float = 0.15  # Threshold for memory pruning
    brain_strengthen_threshold: float = 0.85  # Threshold for strengthening
    brain_max_prune_percentage: float = 0.2  # Max % to prune at once
    brain_expected_reward_init: float = 0.5  # Initial expected reward estimate
    brain_alpha: float = 0.1  # Learning rate for brain consolidation

    # =========================================================================
    # 9. OFFLINE LEARNING
    # =========================================================================
    episode_buffer_size: int = 1000
    offline_update_interval: int = 50
    replay_batch_size: int = 20

    # NEW: Counterfactual learning
    enable_counterfactual: bool = True
    counterfactual_samples: int = 5

    # 🆕 STANFORD FIX: Priority replay and credit adjustment
    priority_replay_alpha: float = 0.6  # Alpha for prioritized experience replay
    success_priority: float = 0.5  # Priority for successful episodes
    failure_priority: float = 1.0  # Priority for failed episodes (learn more)
    credit_adjustment_factor: float = 0.2  # Factor for credit adjustments

    # =========================================================================
    # 10. PROTECTION MECHANISMS
    # =========================================================================
    protected_memory_threshold: float = 0.8
    task_memory_ratio: float = 0.3
    suspicion_threshold: float = 0.95
    ood_entropy_threshold: float = 0.8
    min_rejection_rate: float = 0.05
    approval_reward_bonus: float = 0.1
    rejection_penalty: float = 0.05

    # =========================================================================
    # 11. VALIDATION
    # =========================================================================
    require_all_architect: bool = True
    require_all_auditor: bool = False

    # ✅ A-TEAM FIX: Swarm validation strategy
    enable_per_actor_swarm_auditor: bool = False  # If True, run swarm Auditor after EACH actor (slow!)
    enable_final_swarm_auditor: bool = True       # If True, run swarm Auditor once at END (recommended)
    swarm_validation_confidence_threshold: float = 0.6  # Only retry if confidence below this

    # ✅ USER FIX: Task planning strategy (NO HARDCODING!)
    enable_llm_planning: bool = False  # If True, use LLM to create initial TODO (future)
    # Users can provide task_plan in kwargs for full control
    min_confidence: float = 0.5

    # ✅ NEW: Default values for validation (was hardcoded 0.3, 0.5, 0.7 in agent.py)
    default_confidence_on_error: float = 0.3  # Confidence when validation errors
    default_confidence_no_validation: float = 0.5  # Confidence when no validation
    default_confidence_insight_share: float = 0.7  # Confidence for shared insights

    # ✅ NEW: Reward defaults (was hardcoded in conductor.py)
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
    enable_debug_logging: bool = False  # ✅ Set True only for debugging
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

    # 🧠 RETRIEVAL STRATEGY (Brain-Inspired!)
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
    max_transfer_distance: int = 2     # Max hierarchy distance for transfer

    # =========================================================================
    # 16. CAUSAL LEARNING (NEW - Aristotle)
    # =========================================================================
    enable_causal_learning: bool = True
    causal_confidence_threshold: float = 0.7
    causal_min_support: int = 3       # Episodes before causal link confirmed
    causal_transfer_enabled: bool = True  # Apply causal knowledge to new domains

    # =========================================================================
    # 17. INTER-AGENT COMMUNICATION (NEW - Dr. Chen)
    # =========================================================================
    enable_agent_communication: bool = True
    share_tool_results: bool = True    # Cache and share tool results
    share_insights: bool = True        # Share discovered insights
    max_messages_per_episode: int = 20

    # 🆕 STANFORD FIX: Predictive MARL (multi-agent trajectory prediction)
    marl_default_cooperation: float = 0.5  # Default cooperation score
    marl_default_predictability: float = 0.5  # Default predictability score
    marl_action_divergence_weight: float = 0.4  # Weight for action divergence
    marl_state_divergence_weight: float = 0.3  # Weight for state divergence
    marl_reward_divergence_weight: float = 0.3  # Weight for reward divergence

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
    merge_similar_memories: bool = True

    # =========================================================================
    # 21. DISTRIBUTED SUPPORT (NEW - Alex)
    # =========================================================================
    enable_distributed: bool = False
    instance_id: str = "default"
    lock_timeout: float = 5.0

    # Redis config (if distributed)
    redis_host: Optional[str] = None
    redis_port: int = 6379
    redis_db: int = 0

    # =========================================================================
    # 21.5 LOCAL-FIRST MODE (OpenClaw-Inspired)
    # =========================================================================
    # Enable local-first mode for privacy - no external API calls
    local_mode: bool = False              # Master switch for local-only inference
    local_model: str = "ollama/llama3"    # Local model for agents (Ollama format)
    local_voice: bool = True              # Use local voice providers when local_mode=True
    local_embedding: bool = True          # Use local embeddings when local_mode=True (future)

    # =========================================================================
    # 22. DYNAMIC ORCHESTRATION (NEW - A-Team v2.0)
    # =========================================================================
    # Incorporates DeepThink's dynamic planning, state analysis, and recovery
    # All components are domain-agnostic and optional

    # Dynamic Task Planning
    enable_dynamic_planning: bool = False  # LLM-based task decomposition
    planning_complexity_threshold: float = 0.7  # When to plan vs direct execution

    # Agent Registry
    enable_agent_registry: bool = True  # Track actor capabilities and performance
    auto_infer_capabilities: bool = True  # LLM infers if not provided

    # State Analysis
    enable_state_analysis: bool = False  # LLM analyzes execution state
    custom_metrics: Dict[str, Callable] = field(default_factory=dict)  # Pluggable metrics

    # Recovery Management
    enable_recovery_management: bool = False  # Intelligent failure recovery

    # =========================================================================
    # 22. COST TRACKING & MONITORING (NEW - OAgents Integration)
    # =========================================================================
    # Cost tracking and efficiency metrics
    enable_cost_tracking: bool = False  # Track LLM API costs (opt-in)
    cost_budget: Optional[float] = None  # Optional cost limit (in USD)
    cost_tracking_file: Optional[str] = None  # Path to save cost tracking data

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
    monitoring_output_dir: Optional[str] = None  # Directory for monitoring reports
    
    # Efficiency metrics
    enable_efficiency_metrics: bool = False  # Calculate efficiency scores (requires cost tracking)
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
    
    # Evaluation framework
    enable_evaluation: bool = False  # Enable evaluation framework (opt-in)
    evaluation_n_runs: int = 5  # Number of evaluation runs
    evaluation_output_dir: Optional[str] = None  # Directory for evaluation results
    
    # =========================================================================
    # 24. AUDITOR TYPES (NEW - OAgents Integration)
    # =========================================================================
    # Auditor type selection (OAgents verification strategies)
    auditor_type: str = "single"  # "single", "list_wise", "pair_wise", "confidence_based"
    enable_list_wise_verification: bool = False  # Enable list-wise verification (opt-in)
    list_wise_min_results: int = 2  # Minimum results for list-wise verification
    list_wise_max_results: int = 5  # Maximum results for list-wise verification
    list_wise_merge_strategy: str = "best_score"  # "best_score", "consensus", "weighted"
    
    recovery_max_retries: int = 3
    custom_recovery_strategies: Dict[str, Callable] = field(default_factory=dict)  # Pluggable strategies

    # Context Providers (generic dict for domain-specific context)
    # Examples: {'metadata_manager': obj, 'database': conn, 'api_client': client}
    # JOTTY doesn't know about these - user provides and tools use them
    # NOTE: This is set per-instance, not in config file

    # =========================================================================
    # Computed properties
    # =========================================================================
    def __post_init__(self):
        """Calculate derived config values."""
        # A-Team: Calculate char limits from token budgets
        # Rule of thumb: 1 token ≈ 4 characters
        if self.preview_char_limit is None:
            self.preview_char_limit = self.preview_token_budget * 4  # 20k tokens → 80k chars

        if self.max_description_chars is None:
            self.max_description_chars = self.max_description_tokens * 4  # 5k tokens → 20k chars
        
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
        
        # Set auditor type if list-wise verification enabled
        if self.enable_list_wise_verification:
            try:
                from ..orchestration.auditor_types import AuditorType
                self.auditor_type = AuditorType.LIST_WISE.value
            except ImportError:
                pass  # Auditor types not available

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


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Backward compatibility: JottyConfig → SwarmConfig
JottyConfig = SwarmConfig

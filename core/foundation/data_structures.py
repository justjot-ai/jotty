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
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
from enum import Enum, auto
from datetime import datetime
from pathlib import Path
import hashlib
import json


# =============================================================================
# ENUMS
# =============================================================================

class MemoryLevel(Enum):
    """Aristotelian knowledge hierarchy."""
    EPISODIC = "episodic"      # Raw experiences (Techne source)
    SEMANTIC = "semantic"      # Abstracted patterns (Episteme)
    PROCEDURAL = "procedural"  # Action sequences (Techne)
    META = "meta"              # Learning wisdom (Phronesis)
    CAUSAL = "causal"          # NEW: Why things work (Episteme+)


class OutputTag(Enum):
    """Post-validation output classification."""
    ENQUIRY = "enquiry"
    FAIL = "fail"
    USEFUL = "useful"
    PARTIAL = "partial"  # NEW: Partially correct


class AlertType(Enum):
    """Health monitoring alert types."""
    REWARD_HACKING = "reward_hacking"
    DISTRIBUTION_SHIFT = "distribution_shift"
    CONSERVATIVE_COLLAPSE = "conservative_collapse"
    FORGETTING = "forgetting"
    LEARNING_STALL = "learning_stall"  # NEW
    GOAL_DRIFT = "goal_drift"          # NEW


class CommunicationType(Enum):
    """NEW: Inter-agent communication types."""
    TOOL_RESULT = "tool_result"        # Share tool call results
    INSIGHT = "insight"                 # Share discovered insight
    WARNING = "warning"                 # Share concern
    REQUEST = "request"                 # Request info from other agent


class ValidationRound(Enum):
    """NEW: Multi-round validation phases."""
    INITIAL = "initial"
    REFINEMENT = "refinement"
    FINAL = "final"


class ContextType(Enum):
    """A-Team Enhancement: Context types for memory retrieval prioritization."""
    VALIDATION = "validation"     # Prefer PROCEDURAL, META
    DEBUGGING = "debugging"       # Prefer CAUSAL, EPISODIC
    PLANNING = "planning"         # Prefer META, SEMANTIC
    EXPLORATION = "exploration"   # Prefer EPISODIC, CAUSAL
    TRANSFORMATION = "transformation"  # Prefer PROCEDURAL, SEMANTIC
    DEFAULT = "default"           # Equal priority


# =============================================================================
# RICH OBSERVATION (A-Team Enhancement)
# =============================================================================

@dataclass
class RichObservation:
    """
    A-Team Enhancement: Rich observation with linguistic context for LLM understanding.
    
    Instead of a simple string observation, this captures:
    - Natural language summary (for LLM comprehension)
    - State deltas (what changed)
    - Entities affected
    - Confidence signals
    - Anomalies detected
    
    Usage:
        obs = RichObservation(
            raw_result={"success": True, "rows": 100},
            natural_summary="Successfully mapped 100 rows to target schema",
            action_taken="Applied column mapping for 'bank_code'",
            outcome_type="success"
        )
        
        # For LLM context
        context = obs.to_linguistic_string()
    """
    
    # Core
    raw_result: Any = None
    
    # Linguistic (for LLM understanding)
    natural_summary: str = ""     # "The validation agent found 3 issues..."
    action_taken: str = ""        # "Validated column 'bank_code' against schema"
    outcome_type: str = "unknown" # "success", "partial", "failure", "unknown"
    
    # State delta
    state_before: Dict[str, Any] = field(default_factory=dict)
    state_after: Dict[str, Any] = field(default_factory=dict)
    delta_summary: str = ""       # "Added 100 rows, filled 3 columns"
    
    # Entities
    entities_affected: List[str] = field(default_factory=list)
    columns_touched: List[str] = field(default_factory=list)
    agents_involved: List[str] = field(default_factory=list)
    
    # Signals
    confidence_reason: str = ""
    anomalies: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # For learning
    should_remember: bool = True  # Can be set False for routine observations
    memory_level_hint: Optional[str] = None  # Hint for MemoryLevelClassifier
    
    def to_linguistic_string(self) -> str:
        """Convert to rich linguistic representation for LLM context."""
        parts = []
        
        if self.action_taken:
            parts.append(f"ACTION: {self.action_taken}")
        
        if self.outcome_type:
            parts.append(f"OUTCOME: {self.outcome_type.upper()}")
        
        if self.natural_summary:
            parts.append(f"SUMMARY: {self.natural_summary}")
        
        if self.delta_summary:
            parts.append(f"CHANGES: {self.delta_summary}")
        
        if self.entities_affected:
            parts.append(f"ENTITIES: {', '.join(self.entities_affected)}")
        
        if self.columns_touched:
            parts.append(f"COLUMNS: {', '.join(self.columns_touched)}")
        
        if self.anomalies:
            parts.append(f"⚠️ ANOMALIES: {'; '.join(self.anomalies)}")
        
        if self.warnings:
            parts.append(f"⚠️ WARNINGS: {'; '.join(self.warnings)}")
        
        if self.confidence_reason:
            parts.append(f"CONFIDENCE: {self.confidence_reason}")
        
        return "\n".join(parts) if parts else str(self.raw_result)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "natural_summary": self.natural_summary,
            "action_taken": self.action_taken,
            "outcome_type": self.outcome_type,
            "delta_summary": self.delta_summary,
            "entities_affected": self.entities_affected,
            "columns_touched": self.columns_touched,
            "anomalies": self.anomalies,
            "warnings": self.warnings,
            "confidence_reason": self.confidence_reason
        }
    
    @classmethod
    def from_simple(cls, result: Any, action: str, outcome: str) -> 'RichObservation':
        """Create a simple RichObservation from basic info."""
        return cls(
            raw_result=result,
            natural_summary=str(result) if result else "",
            action_taken=action,
            outcome_type=outcome
        )


# =============================================================================
# GOAL HIERARCHY (Aristotle Enhancement)
# =============================================================================

@dataclass
class GoalNode:
    """
    NEW: Hierarchical goal structure.
    
    Enables knowledge transfer between related goals.
    
    Example hierarchy:
        data_analysis
        ├── sql_queries
        │   ├── sales_queries
        │   ├── user_queries
        │   └── order_queries
        └── python_analysis
            ├── pandas_queries
            └── visualization
    """
    goal_id: str
    goal_text: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    # Goal characteristics for matching
    domain: str = "general"           # sql, python, api, etc.
    operation_type: str = "query"     # query, mutation, analysis
    entities: List[str] = field(default_factory=list)  # tables, objects involved
    
    # Learning state
    episode_count: int = 0
    success_rate: float = 0.5
    
    def similarity_score(self, other: 'GoalNode') -> float:
        """Compute structural similarity for knowledge transfer."""
        score = 0.0
        
        # Same domain = high similarity
        if self.domain == other.domain:
            score += 0.4
        
        # Same operation type
        if self.operation_type == other.operation_type:
            score += 0.2
        
        # Entity overlap
        if self.entities and other.entities:
            overlap = len(set(self.entities) & set(other.entities))
            total = len(set(self.entities) | set(other.entities))
            if total > 0:
                score += 0.3 * (overlap / total)
        
        # Hierarchical relationship
        if self.parent_id == other.goal_id or other.parent_id == self.goal_id:
            score += 0.1
        
        return min(score, 1.0)


@dataclass
class GoalHierarchy:
    """
    NEW: Manages the goal hierarchy for knowledge transfer.
    """
    nodes: Dict[str, GoalNode] = field(default_factory=dict)
    root_id: str = "root"
    
    def add_goal(self, goal_text: str, domain: str = "general", 
                 operation_type: str = "query", entities: List[str] = None,
                 parent_id: str = None) -> str:
        """Add a new goal, auto-detecting hierarchy position."""
        goal_id = hashlib.md5(goal_text.encode()).hexdigest()
        
        if goal_id in self.nodes:
            # Update existing
            self.nodes[goal_id].episode_count += 1
            return goal_id
        
        # Find best parent if not specified
        if parent_id is None:
            parent_id = self._find_best_parent(domain, operation_type)
        
        node = GoalNode(
            goal_id=goal_id,
            goal_text=goal_text,
            parent_id=parent_id,
            domain=domain,
            operation_type=operation_type,
            entities=entities or []
        )
        
        self.nodes[goal_id] = node
        
        # Update parent's children
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children_ids.append(goal_id)
        
        return goal_id
    
    def _find_best_parent(self, domain: str, operation_type: str) -> Optional[str]:
        """Find best parent for new goal based on domain."""
        # Look for domain node
        for node_id, node in self.nodes.items():
            if node.domain == domain and node.operation_type == "general":
                return node_id
        return self.root_id if self.root_id in self.nodes else None
    
    def get_related_goals(self, goal_id: str, max_distance: int = 2) -> List[Tuple[str, float]]:
        """Get related goals with similarity scores for knowledge transfer."""
        if goal_id not in self.nodes:
            return []
        
        target = self.nodes[goal_id]
        related = []
        
        for other_id, other in self.nodes.items():
            if other_id == goal_id:
                continue
            
            similarity = target.similarity_score(other)
            if similarity > 0.3:  # Threshold for relevance
                related.append((other_id, similarity))
        
        return sorted(related, key=lambda x: x[1], reverse=True)


# =============================================================================
# CAUSAL KNOWLEDGE (Aristotle Enhancement)
# =============================================================================

@dataclass
class CausalLink:
    """
    NEW: Represents a cause-effect relationship.
    
    Captures WHY something works, not just WHAT works.
    """
    cause: str                    # "Trino parser requires type annotation"
    effect: str                   # "DATE literal needed for date columns"
    confidence: float = 0.5       # Confidence in causal relationship
    
    # Evidence
    supporting_episodes: List[int] = field(default_factory=list)
    contradicting_episodes: List[int] = field(default_factory=list)
    
    # Conditions for applicability
    conditions: List[str] = field(default_factory=list)  # ["database=trino", "column_type=date"]
    exceptions: List[str] = field(default_factory=list)  # ["version<3.0"]
    
    # Generalization info
    domain: str = "general"
    transferable_to: List[str] = field(default_factory=list)  # Other domains where this applies
    
    def applies_in_context(self, context: Dict[str, Any]) -> bool:
        """Check if this causal link applies in given context."""
        # Check conditions
        for condition in self.conditions:
            key, value = condition.split("=", 1) if "=" in condition else (condition, "true")
            if key in context and str(context[key]).lower() != value.lower():
                return False
        
        # Check exceptions
        for exception in self.exceptions:
            key, value = exception.split("=", 1) if "=" in exception else (exception, "true")
            if key in context and str(context[key]).lower() == value.lower():
                return False
        
        return True
    
    def update_confidence(self, supported: bool):
        """Update confidence based on new evidence."""
        if supported:
            self.confidence = self.confidence + 0.1 * (1 - self.confidence)
        else:
            self.confidence = self.confidence - 0.1 * self.confidence
        self.confidence = max(0.1, min(0.99, self.confidence))


# =============================================================================
# MEMORY ENTRIES (Enhanced)
# =============================================================================

@dataclass
class GoalValue:
    """Value estimate for a specific goal."""
    value: float = 0.5
    access_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    eligibility_trace: float = 0.0
    
    # NEW: Uncertainty estimate for UCB
    variance: float = 0.25  # Initial high uncertainty


@dataclass
class MemoryEntry:
    """
    Enhanced memory entry with all A-Team improvements.
    """
    key: str
    content: str
    level: MemoryLevel
    context: Dict[str, Any]
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    
    # Goal-conditioned values (Fix #4)
    goal_values: Dict[str, GoalValue] = field(default_factory=dict)
    default_value: float = 0.5
    
    # Access tracking
    access_count: int = 0
    ucb_visits: int = 0
    
    # NEW: Size tracking for budget
    token_count: int = 0
    
    # NEW: Causal links
    causal_links: List[str] = field(default_factory=list)  # Keys to CausalLink entries
    
    # NEW: Deduplication
    content_hash: str = ""
    similar_entries: List[str] = field(default_factory=list)  # Keys of similar entries
    
    # NEW: Source tracking
    source_episode: int = 0
    source_agent: str = ""
    
    # Protection
    is_protected: bool = False
    protection_reason: str = ""
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.content.encode()).hexdigest()
        if not self.token_count:
            self.token_count = len(self.content) // 4 + 1
    
    def get_value(self, goal: str) -> float:
        """Get value for specific goal (exact match)."""
        if goal in self.goal_values:
            return self.goal_values[goal].value
        return self.default_value
    
    def get_value_with_transfer(self, goal: str, goal_hierarchy: GoalHierarchy, 
                                 transfer_weight: float = 0.3) -> float:
        """
        NEW: Get value with knowledge transfer from related goals.
        """
        # Exact match
        if goal in self.goal_values:
            return self.goal_values[goal].value
        
        # Try to transfer from related goals
        goal_id = hashlib.md5(goal.encode()).hexdigest()
        related = goal_hierarchy.get_related_goals(goal_id)
        
        if not related:
            return self.default_value
        
        # Weighted average of related goal values
        total_weight = 0.0
        weighted_value = 0.0
        
        for related_id, similarity in related:  # ALL related
            related_node = goal_hierarchy.nodes.get(related_id)
            if related_node and related_node.goal_text in self.goal_values:
                weight = similarity * transfer_weight
                weighted_value += self.goal_values[related_node.goal_text].value * weight
                total_weight += weight
        
        if total_weight > 0:
            # Blend transferred value with default
            transferred = weighted_value / total_weight
            return (1 - transfer_weight) * self.default_value + transfer_weight * transferred
        
        return self.default_value
    
    def get_ucb_score(self, goal: str, total_accesses: int, 
                      c: float = 2.0, goal_hierarchy: GoalHierarchy = None) -> float:
        """Get UCB score for exploration-exploitation balance."""
        if goal_hierarchy:
            value = self.get_value_with_transfer(goal, goal_hierarchy)
        else:
            value = self.get_value(goal)
        
        if self.ucb_visits == 0:
            return float('inf')  # Unexplored = highest priority
        
        import math
        exploration_bonus = c * math.sqrt(math.log(total_accesses + 1) / self.ucb_visits)
        return value + exploration_bonus


# =============================================================================
# INTER-AGENT COMMUNICATION (Dr. Chen Enhancement)
# =============================================================================

@dataclass
class AgentMessage:
    """
    NEW: Message for inter-agent communication.
    """
    sender: str
    receiver: str  # "*" for broadcast
    message_type: CommunicationType
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    # For tool result sharing
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    tool_result: Optional[Any] = None
    
    # For insight sharing
    insight: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class SharedScratchpad:
    """
    NEW: Shared memory space for agent communication.
    """
    messages: List[AgentMessage] = field(default_factory=list)
    tool_cache: Dict[str, Any] = field(default_factory=dict)  # Cache tool results
    shared_insights: List[str] = field(default_factory=list)
    
    def add_message(self, message: AgentMessage):
        self.messages.append(message)
        
        # Cache tool results
        if message.message_type == CommunicationType.TOOL_RESULT:
            cache_key = f"{message.tool_name}:{json.dumps(message.tool_args, sort_keys=True)}"
            self.tool_cache[cache_key] = message.tool_result
    
    def get_cached_result(self, tool_name: str, tool_args: Dict) -> Optional[Any]:
        """Check if tool result is already cached."""
        cache_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
        return self.tool_cache.get(cache_key)
    
    def get_messages_for(self, receiver: str) -> List[AgentMessage]:
        """Get all messages for a specific agent."""
        return [m for m in self.messages if m.receiver in (receiver, "*")]
    
    def clear(self):
        self.messages.clear()
        self.tool_cache.clear()
        self.shared_insights.clear()


# =============================================================================
# VALIDATION RESULTS (Enhanced)
# =============================================================================

@dataclass
class ValidationResult:
    """
    Enhanced validation result with reasoning trace.
    """
    agent_name: str
    is_valid: bool
    confidence: float
    reasoning: str
    
    # For Architect
    should_proceed: Optional[bool] = None
    injected_context: Optional[str] = None
    injected_instructions: Optional[str] = None
    
    # For Auditor
    output_tag: Optional[OutputTag] = None
    why_useful: Optional[str] = None
    
    # Execution info
    tool_calls: List[Dict] = field(default_factory=list)
    execution_time: float = 0.0
    
    # NEW: Reasoning quality for credit assignment
    reasoning_steps: List[str] = field(default_factory=list)
    reasoning_quality: float = 0.5  # How well-reasoned was the decision
    
    # NEW: Multi-round info
    validation_round: ValidationRound = ValidationRound.INITIAL
    previous_rounds: List['ValidationResult'] = field(default_factory=list)


# =============================================================================
# AGENT CONTRIBUTION (Enhanced Credit Assignment)
# =============================================================================

@dataclass
class AgentContribution:
    """
    Enhanced contribution tracking with reasoning analysis.
    """
    agent_name: str
    contribution_score: float  # -1 to 1
    
    # Decision analysis
    decision: str              # "approve", "reject", "abstain"
    decision_correct: bool
    counterfactual_impact: float  # Would outcome change without this agent?
    
    # NEW: Reasoning-based credit (Dr. Chen)
    reasoning_quality: float   # How good was the reasoning
    evidence_used: List[str]   # What evidence was cited
    tools_used: List[str]      # What tools were called
    
    # NEW: Temporal credit
    decision_timing: float     # When in episode (0-1)
    temporal_weight: float     # Weight based on timing
    
    def compute_final_contribution(self) -> float:
        """Compute final contribution with all factors."""
        base = self.contribution_score
        
        # Adjust by reasoning quality
        reasoning_factor = 0.5 + 0.5 * self.reasoning_quality
        
        # Adjust by counterfactual impact
        impact_factor = 0.5 + 0.5 * self.counterfactual_impact
        
        # Temporal weighting (early decisions less certain)
        temporal_factor = 0.7 + 0.3 * self.decision_timing
        
        return base * reasoning_factor * impact_factor * temporal_factor


# =============================================================================
# EPISODE STORAGE (Enhanced for Offline Learning)
# =============================================================================

@dataclass
class StoredEpisode:
    """
    Complete episode storage for offline learning.
    """
    episode_id: int
    goal: str
    goal_id: str
    
    # Full trajectory
    trajectory: List[Dict[str, Any]]
    
    # Inputs
    kwargs: Dict[str, Any]
    
    # Results
    success: bool
    final_reward: float
    
    # Agent decisions
    architect_results: List[ValidationResult]
    auditor_results: List[ValidationResult]
    actor_output: Optional[str] = None
    actor_error: Optional[str] = None
    
    # Contributions
    agent_contributions: Dict[str, AgentContribution] = field(default_factory=dict)
    
    # Memory state
    memories_accessed: Dict[str, List[str]] = field(default_factory=dict)  # agent -> [memory_keys]
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time: float = 0.0
    
    # NEW: Causal analysis
    causal_links_used: List[str] = field(default_factory=list)
    causal_links_discovered: List[CausalLink] = field(default_factory=list)
    
    # NEW: For counterfactual learning
    alternative_decisions: Dict[str, str] = field(default_factory=dict)  # agent -> alternative
    estimated_alternative_outcome: Optional[bool] = None


# =============================================================================
# EPISODE RESULT
# =============================================================================

@dataclass
class TaggedOutput:
    """Output tagged by Auditor."""
    name: str
    tag: OutputTag
    why_useful: str
    content: Any = None


@dataclass
class EpisodeResult:
    """
    Complete episode result with all metadata.
    """
    output: Any
    success: bool
    trajectory: List[Dict[str, Any]]
    
    # Tagged outputs
    tagged_outputs: List[TaggedOutput]
    
    # Episode info
    episode: int
    execution_time: float
    
    # Validation results
    architect_results: List[ValidationResult]
    auditor_results: List[ValidationResult]
    
    # Learning info
    agent_contributions: Dict[str, AgentContribution]
    memories_updated: int = 0
    
    # Health
    alerts: List[str] = field(default_factory=list)
    
    # NEW: Causal insights
    causal_insights: List[str] = field(default_factory=list)
    
    # NEW: Multi-round validation summary
    validation_rounds: int = 1
    refinement_improvements: List[str] = field(default_factory=list)
    
    # ✅ A-TEAM: Confidence-based override metadata (Dec 29, 2025)
    override_metadata: Optional[Dict[str, Any]] = None


# =============================================================================
# LEARNING METRICS (Enhanced Health Monitoring)
# =============================================================================

@dataclass
class LearningMetrics:
    """
    Enhanced metrics for learning health monitoring.
    """
    # Basic stats
    episode_count: int = 0
    success_count: int = 0
    block_count: int = 0
    
    # Per-agent stats
    agent_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Rolling windows
    recent_successes: List[bool] = field(default_factory=list)
    recent_rewards: List[float] = field(default_factory=list)
    recent_entropies: List[float] = field(default_factory=list)
    
    # NEW: Learning progress tracking
    value_changes: List[float] = field(default_factory=list)  # Track TD updates
    learning_rate_history: List[float] = field(default_factory=list)
    
    # NEW: Goal diversity
    goals_seen: Set[str] = field(default_factory=set)
    goal_success_rates: Dict[str, float] = field(default_factory=dict)
    
    # NEW: Causal learning progress
    causal_links_discovered: int = 0
    causal_links_validated: int = 0
    
    def get_success_rate(self, window: int = 100) -> float:
        if not self.recent_successes:
            return 0.5
        recent = self.recent_successes[-window:]
        return sum(recent) / len(recent)
    
    def get_learning_velocity(self, window: int = 50) -> float:
        """NEW: Measure how fast values are changing (learning speed)."""
        if len(self.value_changes) < 2:
            return 0.0
        recent = self.value_changes[-window:]
        return sum(abs(v) for v in recent) / len(recent)
    
    def is_learning_stalled(self, threshold: float = 0.001, window: int = 100) -> bool:
        """NEW: Detect if learning has stalled."""
        velocity = self.get_learning_velocity(window)
        return velocity < threshold


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
    max_eval_iters: int = 10   # ✅ Configurable! (was hardcoded=3 in agent.py)
    max_episode_iterations: int = 12  # ✅ A-TEAM: Max task iterations per episode (used in swarm.run)
    async_timeout: float = 60.0
    actor_timeout: float = 900.0  # Specific timeout for actor execution (15 minutes)
    max_concurrent_agents: int = 10
    
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

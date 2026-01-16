"""
Jotty v6.0 - Memory-Related Types
==================================

All memory-related dataclasses including goal-conditioned values,
memory entries, goal hierarchies, and causal knowledge.
Extracted from data_structures.py for better organization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib

from .enums import MemoryLevel


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

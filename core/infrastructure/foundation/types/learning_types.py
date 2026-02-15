"""
Jotty v6.0 - Learning-Related Types
====================================

All learning-related dataclasses including episode results,
metrics, and offline learning storage.
Extracted from data_structures.py for better organization.
"""

from dataclasses import dataclass, field
from datetime import datetime

# ValidationResult and AgentContribution are defined in validation_types.py and agent_types.py
# We use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from .enums import OutputTag
from .memory_types import CausalLink

# =============================================================================
# Forward declarations (to avoid circular imports)
# =============================================================================


if TYPE_CHECKING:
    pass


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
    architect_results: List[Any]  # List['ValidationResult']
    auditor_results: List[Any]  # List['ValidationResult']

    # Learning info
    agent_contributions: Dict[str, Any]  # Dict[str, 'AgentContribution']
    memories_updated: int = 0

    # Health
    alerts: List[str] = field(default_factory=list)

    # NEW: Causal insights
    causal_insights: List[str] = field(default_factory=list)

    # NEW: Multi-round validation summary
    validation_rounds: int = 1
    refinement_improvements: List[str] = field(default_factory=list)

    # A-TEAM: Confidence-based override metadata (Dec 29, 2025)
    override_metadata: Optional[Dict[str, Any]] = None


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
    architect_results: List[Any]  # List['ValidationResult']
    auditor_results: List[Any]  # List['ValidationResult']
    actor_output: Optional[str] = None
    actor_error: Optional[str] = None

    # Contributions
    agent_contributions: Dict[str, Any] = field(
        default_factory=dict
    )  # Dict[str, 'AgentContribution']

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

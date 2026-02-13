"""
Base Classes for Learning Components
=====================================

Provides unified interfaces and base classes for:
1. PatternLearner - Base for pattern-based learners
2. AdaptiveController - Interface for adaptive rate/parameter control
3. ExperienceRecorder - Interface for experience recording

These ensure consistency across learning components.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time


# =============================================================================
# ADAPTIVE CONTROLLER INTERFACE
# =============================================================================

@dataclass
class AdaptiveState:
    """Common state for adaptive controllers."""
    current_rate: float = 1.0
    min_rate: float = 0.01
    max_rate: float = 10.0
    iteration: int = 0
    history: List[float] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)


class AdaptiveController(ABC):
    """
    Interface for adaptive parameter controllers.

    Implementations include:
    - AdaptiveLearningRate (RL learning rate based on TD errors)
    - AdaptiveLearning (orchestration-level based on scores)
    - AdaptiveExploration (epsilon based on success rates)

    All share the pattern of:
    1. Record observations
    2. Compute adjustment
    3. Return updated rate/parameter
    """

    @abstractmethod
    def get_current_rate(self) -> float:
        """Get the current rate/parameter value."""
        pass

    @abstractmethod
    def update(self, observation: float) -> float:
        """
        Update based on new observation.

        Args:
            observation: New data point (TD error, score, success rate, etc.)

        Returns:
            Updated rate/parameter value
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current state for logging/debugging."""
        pass

    def should_adapt(self, min_observations: int = 5) -> bool:
        """Check if enough observations to adapt."""
        state = self.get_state()
        return state.get('iteration', 0) >= min_observations


# =============================================================================
# PATTERN LEARNER BASE CLASS
# =============================================================================

@dataclass
class LearnedPattern:
    """A pattern learned from experience."""
    pattern_id: str
    pattern_type: str  # 'prompt', 'workflow', 'strategy', etc.
    content: Any
    success_count: int = 0
    failure_count: int = 0
    last_used: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    @property
    def confidence(self) -> float:
        """Confidence based on number of uses."""
        total = self.success_count + self.failure_count
        return min(1.0, total / 10)  # Max confidence at 10 uses


class PatternLearner(ABC):
    """
    Base class for pattern-based learners.

    Implementations include:
    - SwarmLearner (learns prompt patterns)
    - SwarmWorkflowLearner (learns workflow patterns)
    - PatternDiscovery (learns episode patterns)

    All share the pattern of:
    1. Extract patterns from experience
    2. Store patterns with success/failure tracking
    3. Retrieve relevant patterns for new situations
    4. Update patterns based on outcomes
    """

    def __init__(self):
        self._patterns: Dict[str, LearnedPattern] = {}

    @abstractmethod
    def extract_pattern(self, experience: Dict[str, Any]) -> Optional[LearnedPattern]:
        """
        Extract a pattern from an experience.

        Args:
            experience: Dict containing state, action, outcome, etc.

        Returns:
            LearnedPattern if pattern found, None otherwise
        """
        pass

    @abstractmethod
    def find_relevant_patterns(self, context: Dict[str, Any], top_k: int = 5) -> List[LearnedPattern]:
        """
        Find patterns relevant to current context.

        Args:
            context: Current situation/state
            top_k: Maximum patterns to return

        Returns:
            List of relevant patterns sorted by relevance
        """
        pass

    def record_outcome(self, pattern_id: str, success: bool):
        """Record outcome for a pattern."""
        if pattern_id in self._patterns:
            pattern = self._patterns[pattern_id]
            if success:
                pattern.success_count += 1
            else:
                pattern.failure_count += 1
            pattern.last_used = time.time()

    def add_pattern(self, pattern: LearnedPattern):
        """Add a new pattern."""
        self._patterns[pattern.pattern_id] = pattern

    def get_pattern(self, pattern_id: str) -> Optional[LearnedPattern]:
        """Get a specific pattern."""
        return self._patterns.get(pattern_id)

    def get_all_patterns(self) -> List[LearnedPattern]:
        """Get all stored patterns."""
        return list(self._patterns.values())

    def get_best_patterns(self, top_k: int = 10) -> List[LearnedPattern]:
        """Get highest success rate patterns."""
        patterns = list(self._patterns.values())
        patterns.sort(key=lambda p: (p.success_rate, p.confidence), reverse=True)
        return patterns[:top_k]

    def prune_weak_patterns(self, min_uses: int = 5, min_success_rate: float = 0.3):
        """Remove patterns with low success rates."""
        to_remove = []
        for pid, pattern in self._patterns.items():
            total = pattern.success_count + pattern.failure_count
            if total >= min_uses and pattern.success_rate < min_success_rate:
                to_remove.append(pid)

        for pid in to_remove:
            del self._patterns[pid]

        return len(to_remove)


# =============================================================================
# EXPERIENCE RECORDER INTERFACE
# =============================================================================

class ExperienceRecorder(ABC):
    """
    Interface for recording learning experiences.

    Implementations include:
    - LearningManager
    - TransferableLearningStore
    - PrioritizedEpisodeBuffer

    All share the pattern of:
    1. Record experiences (state, action, reward, next_state)
    2. Store for later retrieval
    3. Support sampling/querying
    """

    @abstractmethod
    def record(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Optional[Dict[str, Any]] = None,
        done: bool = False,
        **metadata
    ):
        """Record an experience."""
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample experiences for replay."""
        pass

    @abstractmethod
    def get_recent(self, n: int) -> List[Dict[str, Any]]:
        """Get most recent experiences."""
        pass


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'AdaptiveState',
    'AdaptiveController',
    'LearnedPattern',
    'PatternLearner',
    'ExperienceRecorder',
]

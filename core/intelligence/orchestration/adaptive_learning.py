"""
Adaptive Learning Rate and Learning Strategies

Implements:
1. Adaptive learning rate based on improvement velocity
2. Exploration vs exploitation balance
3. Plateau detection
4. Convergence detection
"""

import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class LearningState:
    """State tracking for adaptive learning."""

    # Score history
    score_history: deque = field(default_factory=lambda: deque(maxlen=10))

    # Improvement velocity
    improvement_velocity: float = 0.0  # Rate of score improvement
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=5))

    # Learning rate
    learning_rate: float = 1.0  # Current learning rate
    base_learning_rate: float = 1.0

    # State flags
    is_plateau: bool = False
    is_converging: bool = False
    is_accelerating: bool = False

    # Iteration tracking
    iteration_count: int = 0
    consecutive_no_improvement: int = 0
    consecutive_improvement: int = 0

    # Exploration vs exploitation
    exploration_rate: float = 0.3  # 0.0 = pure exploitation, 1.0 = pure exploration


class AdaptiveLearning:
    """
    Adaptive learning rate controller.

    Adjusts learning strategy based on improvement velocity and convergence.
    """

    def __init__(self, base_learning_rate: float = 1.0) -> None:
        self.state = LearningState(base_learning_rate=base_learning_rate)
        self.state.learning_rate = base_learning_rate

    def update_score(self, score: float) -> Dict[str, Any]:
        """
        Update with new score and adjust learning rate.

        Args:
            score: New evaluation score

        Returns:
            Dictionary with updated learning state and recommendations
        """
        self.state.iteration_count += 1
        self.state.score_history.append(score)

        # Calculate improvement velocity
        if len(self.state.score_history) >= 2:
            recent_scores = list(self.state.score_history)[-5:]  # Last 5 scores
            if len(recent_scores) >= 2:
                # Calculate velocity as slope of recent scores
                velocities = [
                    recent_scores[i] - recent_scores[i - 1] for i in range(1, len(recent_scores))
                ]
                self.state.improvement_velocity = statistics.mean(velocities)
                self.state.velocity_history.append(self.state.improvement_velocity)

        # Detect states
        self._detect_plateau()
        self._detect_convergence()
        self._detect_acceleration()

        # Adjust learning rate
        self._adjust_learning_rate()

        # Adjust exploration rate
        self._adjust_exploration_rate()

        return {
            "learning_rate": self.state.learning_rate,
            "exploration_rate": self.state.exploration_rate,
            "improvement_velocity": self.state.improvement_velocity,
            "is_plateau": self.state.is_plateau,
            "is_converging": self.state.is_converging,
            "is_accelerating": self.state.is_accelerating,
            "recommendation": self._get_recommendation(),
        }

    def _detect_plateau(self) -> None:
        """Detect if learning has plateaued (stuck at low/medium scores)."""
        if len(self.state.score_history) < 5:
            self.state.is_plateau = False
            return

        recent_scores = list(self.state.score_history)[-5:]
        score_variance = statistics.variance(recent_scores) if len(recent_scores) > 1 else 0.0
        score_mean = statistics.mean(recent_scores)

        # Plateau = low variance AND low/medium scores AND no improvement
        # High stable scores (mean >= 0.9) are CONVERGENCE, not plateau
        if score_variance < 0.01 and self.state.improvement_velocity <= 0.01 and score_mean < 0.9:
            self.state.is_plateau = True
            self.state.consecutive_no_improvement += 1
        else:
            self.state.is_plateau = False
            self.state.consecutive_no_improvement = 0

    def _detect_convergence(self) -> None:
        """Detect if learning is converging (high stable scores)."""
        if len(self.state.score_history) < 3:
            self.state.is_converging = False
            return

        recent_scores = list(self.state.score_history)[-3:]

        # Converging if scores are high and stable
        # (velocity > 0 OR already at high stable plateau)
        high_and_stable = (
            recent_scores[-1] >= 0.9 and abs(recent_scores[-1] - recent_scores[-2]) < 0.05
        )
        if high_and_stable:
            self.state.is_converging = True
            self.state.consecutive_improvement += 1
        else:
            self.state.is_converging = False
            if self.state.improvement_velocity > 0.05:
                self.state.consecutive_improvement += 1
            else:
                self.state.consecutive_improvement = 0

    def _detect_acceleration(self) -> None:
        """Detect if learning is accelerating."""
        if len(self.state.velocity_history) < 3:
            self.state.is_accelerating = False
            return

        velocities = list(self.state.velocity_history)

        # Accelerating if velocity is increasing
        if velocities[-1] > velocities[-2] > velocities[-3] and velocities[-1] > 0.05:
            self.state.is_accelerating = True
        else:
            self.state.is_accelerating = False

    def _adjust_learning_rate(self) -> Any:
        """Adjust learning rate based on state."""
        if self.state.is_plateau:
            # Increase learning rate to escape plateau
            self.state.learning_rate = min(2.0, self.state.learning_rate * 1.2)
            logger.info(
                f"Plateau detected, increasing learning rate to {self.state.learning_rate:.2f}"
            )

        elif self.state.is_accelerating:
            # Maintain or slightly increase learning rate
            self.state.learning_rate = min(1.5, self.state.learning_rate * 1.1)
            logger.debug(
                f"Accelerating, maintaining learning rate at {self.state.learning_rate:.2f}"
            )

        elif self.state.is_converging:
            # Decrease learning rate for fine-tuning
            self.state.learning_rate = max(0.5, self.state.learning_rate * 0.9)
            logger.debug(f"Converging, decreasing learning rate to {self.state.learning_rate:.2f}")

        else:
            # Normal operation - slight decay
            self.state.learning_rate = max(0.7, self.state.learning_rate * 0.95)

    def _adjust_exploration_rate(self) -> Any:
        """Adjust exploration vs exploitation balance."""
        if self.state.is_plateau:
            # Increase exploration to find new solutions
            self.state.exploration_rate = min(0.7, self.state.exploration_rate + 0.1)
            logger.info(
                f"Plateau detected, increasing exploration to {self.state.exploration_rate:.2f}"
            )

        elif self.state.is_accelerating:
            # Focus on exploitation (use what works)
            self.state.exploration_rate = max(0.1, self.state.exploration_rate - 0.1)
            logger.debug(
                f"Accelerating, decreasing exploration to {self.state.exploration_rate:.2f}"
            )

        elif self.state.is_converging:
            # Low exploration, high exploitation
            self.state.exploration_rate = max(0.05, self.state.exploration_rate - 0.05)
            logger.debug(f"Converging, decreasing exploration to {self.state.exploration_rate:.2f}")

    def _get_recommendation(self) -> str:
        """Get recommendation based on current state."""
        if self.state.is_plateau:
            return "increase_exploration"
        elif self.state.is_accelerating:
            return "focus_exploitation"
        elif self.state.is_converging:
            return "fine_tune"
        elif self.state.improvement_velocity > 0.1:
            return "maintain_momentum"
        else:
            return "continue_learning"

    def should_stop_early(self, min_iterations: int = 3) -> bool:
        """
        Determine if we should stop early.

        Args:
            min_iterations: Minimum iterations before considering early stop

        Returns:
            True if should stop early
        """
        if self.state.iteration_count < min_iterations:
            return False

        # Stop if converged and high score
        if (
            self.state.is_converging
            and len(self.state.score_history) > 0
            and self.state.score_history[-1] >= 0.95
        ):
            return True

        # Stop if plateau for too long
        if self.state.consecutive_no_improvement >= 5:
            return True

        return False

    def get_state(self) -> LearningState:
        """Get current learning state."""
        return self.state

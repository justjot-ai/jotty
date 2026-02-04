"""
LearningManager - Manages Q-learning, TD(λ), and experience replay.

Extracted from conductor.py to improve maintainability and testability.
All RL-related logic is centralized here.
"""
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LearningUpdate:
    """Result of a learning update."""
    actor: str
    reward: float
    q_value: Optional[float] = None
    td_error: Optional[float] = None


class LearningManager:
    """
    Centralized learning management for Q-learning, TD(λ), and MARL.

    Responsibilities:
    - Q-learner instance management
    - Q-value prediction
    - Experience recording and updates
    - TD(λ) updates
    - Credit assignment
    - Offline learning batches

    This manager owns the single Q-learner instance (no duplicate instances!)
    """

    def __init__(self, config):
        """
        Initialize learning manager.

        Args:
            config: JottyConfig with RL parameters
        """
        self.config = config
        self.q_learner = None
        self.td_lambda_learner = None

        # Initialize Q-learner if available
        try:
            from ...learning.q_learning import LLMQPredictor as NaturalLanguageQTable
            self.q_learner = NaturalLanguageQTable(self.config)
            logger.info("🧠 Q-Learning initialized (LearningManager)")
            logger.info(f"   Mode: {config.q_value_mode} Q-values")
            logger.info("   - Tier 1: Working Memory (always in context)")
            logger.info("   - Tier 2: Semantic Clusters (retrieval-based)")
            logger.info("   - Tier 3: Long-term Archive (causal impact pruning)")
        except ImportError:
            logger.warning("⚠️  Q-Learning not available")

        # Initialize TD(λ) learner if enabled
        if config.enable_rl:
            try:
                from ...learning.learning import TDLambdaLearner
                self.td_lambda_learner = TDLambdaLearner(self.config)
                logger.info("🎯 TD(λ) Learner initialized (LearningManager)")
            except ImportError:
                logger.warning("⚠️  TD(λ) Learning not available")

    def predict_q_value(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        goal: str = ""
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Predict Q-value for a state-action pair.

        Args:
            state: Current state dict
            action: Action dict (contains 'actor' key)
            goal: Goal description

        Returns:
            (q_value, confidence, alternative_suggestion)
        """
        if not self.q_learner:
            return 0.5, 0.1, None

        try:
            return self.q_learner.predict_q_value(state, action, goal)
        except Exception as e:
            logger.warning(f"Q-value prediction failed: {e}")
            return 0.5, 0.1, None

    def record_outcome(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Optional[Dict[str, Any]] = None,
        done: bool = False
    ) -> LearningUpdate:
        """
        Record outcome and update Q-values.

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: State after action
            done: Whether episode is complete

        Returns:
            LearningUpdate with results
        """
        actor = action.get('actor', 'unknown')

        if not self.q_learner:
            return LearningUpdate(actor=actor, reward=reward)

        try:
            # Record in Q-learner (updates experience buffer)
            self.q_learner.record_outcome(state, action, reward, next_state, done)

            # Get updated Q-value
            q_value, _, _ = self.q_learner.predict_q_value(state, action)

            return LearningUpdate(
                actor=actor,
                reward=reward,
                q_value=q_value
            )
        except Exception as e:
            logger.error(f"Learning update failed for {actor}: {e}")
            return LearningUpdate(actor=actor, reward=reward)

    def update_td_lambda(
        self,
        trajectory: list,
        final_reward: float,
        gamma: float = 0.99,
        lambda_trace: float = 0.95
    ) -> None:
        """
        Perform TD(λ) update on trajectory.

        Args:
            trajectory: List of (state, action, reward) tuples
            final_reward: Final episode reward
            gamma: Discount factor
            lambda_trace: Eligibility trace decay
        """
        if not self.td_lambda_learner:
            return

        try:
            self.td_lambda_learner.update(trajectory, final_reward, gamma, lambda_trace)
        except Exception as e:
            logger.error(f"TD(λ) update failed: {e}")

    def get_learned_context(
        self,
        state: Dict[str, Any],
        action: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get learned context to inject into prompts.

        This is how learning manifests in LLM agents!

        Args:
            state: Current state
            action: Optional action being considered

        Returns:
            Natural language lessons learned
        """
        if not self.q_learner:
            return ""

        try:
            return self.q_learner.get_learned_context(state, action)
        except Exception as e:
            logger.warning(f"Failed to get learned context: {e}")
            return ""

    def promote_demote_memories(self, episode_reward: float) -> None:
        """
        Promote/demote memories based on episode performance.

        Args:
            episode_reward: Total reward from episode
        """
        if not self.q_learner:
            return

        try:
            if hasattr(self.q_learner, '_promote_demote_memories'):
                self.q_learner._promote_demote_memories(episode_reward=episode_reward)
        except Exception as e:
            logger.warning(f"Memory promotion/demotion failed: {e}")

    def prune_tier3(self, sample_rate: float = 0.1) -> None:
        """
        Prune Tier 3 memories by causal impact.

        Args:
            sample_rate: Fraction of memories to sample
        """
        if not self.q_learner:
            return

        try:
            if hasattr(self.q_learner, 'prune_tier3_by_causal_impact'):
                self.q_learner.prune_tier3_by_causal_impact(sample_rate=sample_rate)
        except Exception as e:
            logger.warning(f"Tier 3 pruning failed: {e}")

    def get_q_table_summary(self) -> str:
        """
        Get summary of Q-table for logging.

        Returns:
            Human-readable Q-table summary
        """
        if not self.q_learner:
            return "Q-learner not available"

        try:
            if hasattr(self.q_learner, 'get_q_table_summary'):
                return self.q_learner.get_q_table_summary()
            else:
                buffer_size = len(self.q_learner.experience_buffer) if hasattr(self.q_learner, 'experience_buffer') else 0
                return f"Q-learner active: {buffer_size} experiences"
        except Exception as e:
            return f"Q-table summary failed: {e}"

    def add_experience(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Optional[Dict[str, Any]] = None,
        done: bool = False
    ) -> None:
        """
        Add experience to buffer (alias for record_outcome).

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: State after action
            done: Whether episode is complete
        """
        self.record_outcome(state, action, reward, next_state, done)

"""
Jotty v6.0 - Base Learning Manager Interface
=============================================

Abstract base class for all learning managers in Jotty.

Defines a common interface for:
- TD(λ) Learning (TDLambdaLearner)
- Q-Learning (LLMQPredictor)
- Shaped Rewards (ShapedRewardManager)
- Offline Learning (OfflineLearner)
- MARL Learning (LLMTrajectoryPredictor)
- Credit Assignment (AlgorithmicCreditAssigner)

This interface enables:
1. Polymorphism - swap learners without changing orchestrator
2. Standardization - consistent method signatures
3. Testing - mock learners for unit tests
4. Extensions - easy to add new learners

REFACTORING PHASE 5: Learning System Interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from ..foundation.data_structures import SwarmConfig


class BaseLearningManager(ABC):
    """
    Abstract base class for all learning managers.
    
    Defines the contract that all learning systems must implement.
    """
    
    def __init__(self, config: SwarmConfig):
        """
        Initialize learning manager with configuration.
        
        Args:
            config: Jotty configuration with learning hyperparameters
        """
        self.config = config
    
    @abstractmethod
    def reset(self):
        """
        Reset learning state for new episode.
        
        Clears episode-specific state while preserving learned knowledge.
        """
        pass
    
    @abstractmethod
    def start_episode(self, goal: str):
        """
        Initialize new learning episode.
        
        Args:
            goal: Episode goal description
        """
        pass
    
    @abstractmethod
    def record_experience(
        self,
        state: Dict[str, Any],
        action: Any,
        reward: float,
        next_state: Optional[Dict[str, Any]] = None,
        done: bool = False,
        **kwargs
    ):
        """
        Record an experience tuple for learning.
        
        Args:
            state: Current state representation
            action: Action taken
            reward: Immediate reward received
            next_state: Resulting state (None if terminal)
            done: Whether episode ended
            **kwargs: Learner-specific additional data
        """
        pass
    
    @abstractmethod
    def end_episode(
        self,
        final_reward: float,
        success: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        End episode and perform learning updates.
        
        Args:
            final_reward: Final episode reward
            success: Whether episode succeeded
            **kwargs: Learner-specific additional data
        
        Returns:
            Learning statistics (TD errors, Q-values, etc.)
        """
        pass
    
    @abstractmethod
    def get_value(
        self,
        state: Dict[str, Any],
        action: Optional[Any] = None,
        **kwargs
    ) -> float:
        """
        Get estimated value for state(-action pair).
        
        Args:
            state: State to evaluate
            action: Optional action (for Q-learning)
            **kwargs: Learner-specific parameters
        
        Returns:
            Estimated value
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get learning statistics.
        
        Returns:
            Dictionary with learner-specific statistics
        """
        return {
            "learner_type": self.__class__.__name__,
            "config": self.config.__dict__
        }
    
    def save_state(self, path: str):
        """
        Save learned knowledge to file.
        
        Args:
            path: File path to save to
        """
        # Default: no-op (subclasses can override)
        pass
    
    def load_state(self, path: str) -> bool:
        """
        Load learned knowledge from file.
        
        Args:
            path: File path to load from
        
        Returns:
            True if successful, False otherwise
        """
        # Default: no-op (subclasses can override)
        return False


class ValueBasedLearningManager(BaseLearningManager):
    """
    Base for value-based learning (TD(λ), Q-learning).
    
    Adds methods specific to value-based reinforcement learning.
    """
    
    @abstractmethod
    def get_best_action(
        self,
        state: Dict[str, Any],
        available_actions: List[Any]
    ) -> Tuple[Any, float, str]:
        """
        Get best action for state using learned values.
        
        Args:
            state: Current state
            available_actions: List of possible actions
        
        Returns:
            (best_action, estimated_value, reasoning)
        """
        pass


class RewardShapingManager(BaseLearningManager):
    """
    Base for reward shaping systems.
    
    Adds methods for computing intermediate/shaped rewards.
    """
    
    @abstractmethod
    def check_rewards(
        self,
        state: Dict[str, Any],
        action: Any,
        trajectory: Optional[List[Dict]] = None
    ) -> float:
        """
        Compute shaped reward for state/action.
        
        Args:
            state: Current state
            action: Action taken
            trajectory: Episode trajectory so far
        
        Returns:
            Shaped reward value
        """
        pass
    
    @abstractmethod
    def get_total_reward(self) -> float:
        """
        Get cumulative shaped reward for episode.
        
        Returns:
            Total shaped reward
        """
        pass


class MultiAgentLearningManager(BaseLearningManager):
    """
    Base for multi-agent reinforcement learning.
    
    Adds methods for agent coordination and communication.
    """
    
    @abstractmethod
    def predict_agent_action(
        self,
        agent_name: str,
        state: Dict[str, Any]
    ) -> Any:
        """
        Predict what another agent will do.
        
        Args:
            agent_name: Name of agent to predict
            state: Current state
        
        Returns:
            Predicted action
        """
        pass
    
    @abstractmethod
    def update_agent_model(
        self,
        agent_name: str,
        actual_action: Any,
        predicted_action: Any
    ):
        """
        Update model of another agent based on prediction error.
        
        Args:
            agent_name: Agent that acted
            actual_action: What they actually did
            predicted_action: What we predicted
        """
        pass

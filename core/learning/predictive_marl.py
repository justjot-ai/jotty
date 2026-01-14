"""
Predictive Multi-Agent RL for ReVal
====================================

THE VISION:
- Each agent predicts what OTHER agents will do
- Compare predictions with actual outcomes  
- Learn from divergence (like weight updates)
- Emergent cooperation through predictive modeling

THE KEY INSIGHT:
LLM IS the DQN. Prompts ARE weights. Divergence learning updates prompts.

Traditional RL: θ ← θ - α * ∇L(predicted, actual)
Agentic RL:     prompt ← prompt + "When predicted X but got Y, learn..."

This is IN-CONTEXT LEARNING as WEIGHT UPDATE.
"""

import json
import time
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PredictedAction:
    """Predicted action by an agent."""
    agent_name: str
    action_type: str  # What action they'll take
    action_params: Dict[str, Any]
    confidence: float
    reasoning: str


@dataclass
class PredictedTrajectory:
    """Predicted sequence of agent actions."""
    horizon: int
    steps: List[Dict[str, Any]]  # Each step: {agent, action, predicted_state}
    predicted_reward: float
    confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ActualTrajectory:
    """What actually happened."""
    steps: List[Dict[str, Any]]
    actual_reward: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Divergence:
    """Divergence between predicted and actual."""
    predicted: PredictedTrajectory
    actual: ActualTrajectory
    
    # Divergence metrics
    action_divergence: float      # How different were actions?
    state_divergence: float       # How different were states?
    reward_divergence: float      # How different was reward?
    
    # Information content (higher = more surprising = more valuable)
    information_content: float
    
    # Extracted learning
    learning: str
    
    def total_divergence(self) -> float:
        """Weighted total divergence."""
        return (0.4 * self.action_divergence + 
                0.3 * self.state_divergence + 
                0.3 * self.reward_divergence)


@dataclass 
class AgentModel:
    """Model of another agent's behavior (Theory of Mind)."""
    agent_name: str
    
    # Behavioral patterns learned
    action_patterns: List[Dict[str, Any]] = field(default_factory=list)
    
    # When this agent tends to deviate from predictions
    deviation_patterns: List[Dict[str, Any]] = field(default_factory=list)
    
    # Cooperation style
    cooperation_score: float = 0.5  # 0 = competitive, 1 = cooperative
    predictability_score: float = 0.5  # 0 = chaotic, 1 = predictable
    
    # Statistics
    total_predictions: int = 0
    correct_predictions: int = 0
    
    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.5
        return self.correct_predictions / self.total_predictions


# =============================================================================
# LLM TRAJECTORY PREDICTOR
# =============================================================================

class TrajectoryPredictionSignature(dspy.Signature):
    """
    Predict what will happen over the next H steps in a multi-agent system.
    
    You are predicting the FUTURE trajectory of a multi-agent swarm.
    Think about:
    1. What will the current agent do?
    2. How will this affect the state?
    3. What will OTHER agents do in response?
    4. What will be the final outcome?
    """
    
    current_state = dspy.InputField(
        desc="Current state: TODO progress, agent states, recent history"
    )
    acting_agent = dspy.InputField(
        desc="Which agent is about to act"
    )
    proposed_action = dspy.InputField(
        desc="The action the agent is considering"
    )
    other_agents = dspy.InputField(
        desc="List of other agents and their recent behaviors"
    )
    agent_models = dspy.InputField(
        desc="Models of other agents' typical behaviors and cooperation styles"
    )
    horizon = dspy.InputField(
        desc="How many steps ahead to predict"
    )
    goal = dspy.InputField(
        desc="The final goal we're trying to achieve"
    )
    
    step_by_step_prediction = dspy.OutputField(
        desc="Step-by-step prediction: For each step, predict {agent, action, resulting_state}"
    )
    predicted_trajectory = dspy.OutputField(
        desc="JSON list of predicted steps: [{agent, action, state}, ...]"
    )
    predicted_final_reward = dspy.OutputField(
        desc="Predicted reward at the end (0.0 to 1.0)"
    )
    prediction_confidence = dspy.OutputField(
        desc="How confident are you in this prediction? (0.0 to 1.0)"
    )
    cooperation_assessment = dspy.OutputField(
        desc="Will agents cooperate well? What might go wrong?"
    )


class LLMTrajectoryPredictor:
    """
    LLM-based trajectory prediction for multi-agent systems.
    
    This is the DQN equivalent for agentic systems:
    - DQN: Neural network predicts Q(s,a)
    - This: LLM predicts trajectory and reward given (state, action, other_agents)
    
    The key innovation: LLM can REASON about other agents' intentions,
    not just pattern-match like a traditional DQN.
    """
    
    def __init__(self, config, horizon: int = 5):
        self.config = config
        self.horizon = horizon
        self.predictor = dspy.ChainOfThought(TrajectoryPredictionSignature) if DSPY_AVAILABLE else None
        
        # Experience buffer for learning
        self.experience_buffer: List[Tuple[PredictedTrajectory, ActualTrajectory]] = []
        self.max_buffer_size = 100
        
        # Learned patterns (this IS the "weights")
        self.learned_patterns: List[str] = []
        self.agent_models: Dict[str, AgentModel] = {}
        
        logger.info(f"🔮 LLMTrajectoryPredictor initialized (horizon={horizon})")
    
    def predict(
        self,
        current_state: Dict[str, Any],
        acting_agent: str,
        proposed_action: Dict[str, Any],
        other_agents: List[str],
        goal: str
    ) -> PredictedTrajectory:
        """
        Predict trajectory over horizon H.
        
        This is the forward pass of our "DQN".
        """
        if not self.predictor:
            return self._fallback_prediction()
        
        # Get agent models
        models_str = self._format_agent_models(other_agents)
        
        # Add learned patterns to prompt (this is the "weight injection")
        patterns_context = self._format_learned_patterns()
        
        try:
            result = self.predictor(
                current_state=json.dumps(current_state, default=str),
                acting_agent=acting_agent,
                proposed_action=json.dumps(proposed_action, default=str),
                other_agents=json.dumps(other_agents),
                agent_models=models_str,
                horizon=str(self.horizon),
                goal=goal + f"\n\nLEARNED PATTERNS:\n{patterns_context}"
            )
            
            # Parse prediction
            try:
                steps = json.loads(result.predicted_trajectory or "[]")
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.debug(f"Trajectory parsing failed: {e}")
                steps = []
            
            reward = self._parse_float(result.predicted_final_reward, 0.5)
            confidence = self._parse_float(result.prediction_confidence, 0.5)
            
            return PredictedTrajectory(
                horizon=self.horizon,
                steps=steps,
                predicted_reward=reward,
                confidence=confidence
            )
            
        except Exception as e:
            logger.warning(f"Trajectory prediction failed: {e}")
            return self._fallback_prediction()
    
    def compute_divergence(
        self,
        predicted: PredictedTrajectory,
        actual: ActualTrajectory
    ) -> Divergence:
        """
        Compute divergence between predicted and actual trajectories.
        
        This is analogous to computing loss in traditional RL.
        """
        # Action divergence: How different were the actions?
        action_div = self._compute_action_divergence(
            predicted.steps, actual.steps
        )
        
        # State divergence: How different were the states?
        state_div = self._compute_state_divergence(
            predicted.steps, actual.steps
        )
        
        # Reward divergence
        reward_div = abs(predicted.predicted_reward - actual.actual_reward)
        
        # Information content: -log P(actual | predicted)
        # Higher divergence = higher information = more surprising
        total_div = 0.4 * action_div + 0.3 * state_div + 0.3 * reward_div
        info_content = -math.log(max(0.01, 1 - total_div))
        
        # Extract learning from divergence
        learning = self._extract_learning(predicted, actual, total_div)
        
        return Divergence(
            predicted=predicted,
            actual=actual,
            action_divergence=action_div,
            state_divergence=state_div,
            reward_divergence=reward_div,
            information_content=info_content,
            learning=learning
        )
    
    def update_from_divergence(self, divergence: Divergence):
        """
        Update predictor based on divergence.
        
        THIS IS THE KEY INNOVATION:
        Traditional RL: θ ← θ - α * ∇L
        Agentic RL: self.learned_patterns.append(learning)
        
        The "weight update" happens by modifying the context/prompt.
        """
        # Only learn from high-information divergences
        if divergence.information_content < 0.5:
            return
        
        # Add to learned patterns
        self.learned_patterns.append(divergence.learning)
        
        # Keep patterns bounded (forget old ones)
        if len(self.learned_patterns) > 50:
            self.learned_patterns = self.learned_patterns
        
        # Update agent models
        self._update_agent_models(divergence)
        
        logger.info(f"📚 Learned: {divergence.learning}...")
    
    def _compute_action_divergence(
        self, 
        predicted_steps: List[Dict], 
        actual_steps: List[Dict]
    ) -> float:
        """Compute how different the actions were."""
        if not predicted_steps or not actual_steps:
            return 0.5
        
        matches = 0
        total = min(len(predicted_steps), len(actual_steps))
        
        for i in range(total):
            pred_agent = predicted_steps[i].get('agent', '')
            actual_agent = actual_steps[i].get('agent', '')
            
            pred_action = predicted_steps[i].get('action', '')
            actual_action = actual_steps[i].get('action', '')
            
            if pred_agent == actual_agent:
                matches += 0.5
            if str(pred_action) == str(actual_action):
                matches += 0.5
        
        return 1 - (matches / total) if total > 0 else 0.5
    
    def _compute_state_divergence(
        self,
        predicted_steps: List[Dict],
        actual_steps: List[Dict]
    ) -> float:
        """Compute how different the states were."""
        # Simplified: compare final states
        if not predicted_steps or not actual_steps:
            return 0.5
        
        pred_final = predicted_steps[-1].get('state', {})
        actual_final = actual_steps[-1].get('state', {})
        
        # Simple key overlap comparison
        pred_keys = set(str(pred_final).split())
        actual_keys = set(str(actual_final).split())
        
        if not pred_keys or not actual_keys:
            return 0.5
        
        overlap = len(pred_keys & actual_keys)
        total = len(pred_keys | actual_keys)
        
        return 1 - (overlap / total) if total > 0 else 0.5
    
    def _extract_learning(
        self,
        predicted: PredictedTrajectory,
        actual: ActualTrajectory,
        divergence: float
    ) -> str:
        """Extract human-readable learning from divergence."""
        if divergence < 0.2:
            return "Prediction was accurate - patterns confirmed"
        
        # What was different?
        differences = []
        
        for i, (pred, act) in enumerate(zip(predicted.steps, actual.steps)):
            pred_agent = pred.get('agent', '')
            act_agent = act.get('agent', '')
            
            # ✅ FIX: Only log if we have actual agent data
            if pred_agent and act_agent and pred_agent != act_agent:
                differences.append(
                    f"Step {i}: Expected {pred_agent} but {act_agent} acted"
                )
            elif pred_agent and act_agent and pred.get('action') != act.get('action'):
                differences.append(
                    f"Step {i}: {pred_agent} did {act.get('action')} not {pred.get('action')}"
                )
        
        if predicted.predicted_reward > actual.actual_reward + 0.2:
            differences.append(
                f"Outcome worse than expected ({actual.actual_reward:.2f} vs {predicted.predicted_reward:.2f})"
            )
        elif predicted.predicted_reward < actual.actual_reward - 0.2:
            differences.append(
                f"Outcome better than expected ({actual.actual_reward:.2f} vs {predicted.predicted_reward:.2f})"
            )
        
        if differences:
            return "; ".join(differences)
        return "Minor divergence in execution details"
    
    def _update_agent_models(self, divergence: Divergence):
        """Update models of agent behavior."""
        for step_pred, step_actual in zip(
            divergence.predicted.steps,
            divergence.actual.steps
        ):
            agent = step_actual.get('agent', '')
            if not agent:
                continue
            
            if agent not in self.agent_models:
                self.agent_models[agent] = AgentModel(agent_name=agent)
            
            model = self.agent_models[agent]
            model.total_predictions += 1
            
            # Was prediction correct?
            if step_pred.get('action') == step_actual.get('action'):
                model.correct_predictions += 1
            else:
                # Record deviation pattern
                model.deviation_patterns.append({
                    'predicted': step_pred.get('action'),
                    'actual': step_actual.get('action'),
                    'context': step_pred.get('state', {}),
                    'timestamp': time.time()
                })
                # Keep bounded
                if len(model.deviation_patterns) > 20:
                    model.deviation_patterns = model.deviation_patterns
    
    def _format_agent_models(self, agents: List[str]) -> str:
        """Format agent models for prompt injection."""
        models_str = []
        for agent in agents:
            if agent in self.agent_models:
                model = self.agent_models[agent]
                models_str.append(
                    f"- {agent}: accuracy={model.accuracy:.2f}, "
                    f"predictability={model.predictability_score:.2f}"
                )
                if model.deviation_patterns:
                    recent = model.deviation_patterns
                    for dev in recent:
                        models_str.append(
                            f"  └ Recently: predicted {dev['predicted']} but did {dev['actual']}"
                        )
        return "\n".join(models_str) if models_str else "No agent models yet"
    
    def _format_learned_patterns(self) -> str:
        """Format learned patterns for prompt injection."""
        if not self.learned_patterns:
            return "No patterns learned yet"
        
        # Recent patterns
        recent = self.learned_patterns
        return "\n".join(f"- {p}" for p in recent)
    
    def _fallback_prediction(self) -> PredictedTrajectory:
        """Fallback when LLM not available."""
        return PredictedTrajectory(
            horizon=self.horizon,
            steps=[],
            predicted_reward=0.5,
            confidence=0.3
        )
    
    def _parse_float(self, s: str, default: float) -> float:
        """Parse float from string - A-Team approved, no regex."""
        from ..foundation.robust_parsing import parse_float_robust
        result = parse_float_robust(s, default=default)
        if result is not None:
            return max(0.0, min(1.0, result))
        return default


# =============================================================================
# DIVERGENCE MEMORY
# =============================================================================

class DivergenceMemory:
    """
    Memory that stores prediction-vs-reality divergences.
    
    Key insight from Shannon: Store by INFORMATION CONTENT, not recency.
    High-surprise events are more valuable for learning.
    """
    
    def __init__(self, config, max_size: int = 500):
        self.config = config
        self.max_size = max_size
        self.memories: List[Divergence] = []
    
    def store(self, divergence: Divergence):
        """Store divergence, prioritizing high-information ones."""
        self.memories.append(divergence)
        
        # Evict by information content (keep surprising ones)
        if len(self.memories) > self.max_size:
            # Sort by information content (ascending)
            self.memories.sort(key=lambda d: d.information_content)
            # Remove lowest information ones
            self.memories = self.memories[len(self.memories) - self.max_size:]
    
    def get_relevant(self, state: Dict[str, Any], k: int = 5) -> List[Divergence]:
        """Get most relevant divergences for current state."""
        # Simple: return highest information ones
        sorted_by_info = sorted(
            self.memories,
            key=lambda d: d.information_content,
            reverse=True
        )
        return sorted_by_info[:k]
    
    def get_patterns_for_agent(self, agent_name: str) -> List[str]:
        """Get learned patterns related to specific agent."""
        patterns = []
        for div in self.memories:
            if any(s.get('agent') == agent_name for s in div.predicted.steps):
                patterns.append(div.learning)
        return patterns  # Recent ones


# =============================================================================
# COOPERATIVE CREDIT ASSIGNER
# =============================================================================

class CooperativeCreditAssigner:
    """
    Assign credit based on COOPERATION, not just individual success.
    
    Key insight: An agent that helps others succeed deserves credit,
    even if their own task seemed unimportant.
    
    A-Team v8.0: No hardcoded weights! Uses adaptive learned weights.
    """
    
    def __init__(self, config):
        self.config = config
        
        # A-Team v8.0: Adaptive weights instead of hardcoded 0.3, 0.2
        from ..foundation.robust_parsing import AdaptiveThreshold
        self.coop_weight_tracker = AdaptiveThreshold(initial_mean=0.3, initial_std=0.1)
        self.pred_weight_tracker = AdaptiveThreshold(initial_mean=0.2, initial_std=0.1)
    
    def assign_credit(
        self,
        trajectory: ActualTrajectory,
        predictions: Dict[str, PredictedTrajectory],
        final_reward: float
    ) -> Dict[str, float]:
        """
        Assign credit to each agent.
        
        Credit = (task_success) * (cooperation_factor) * (prediction_factor)
        
        A-Team v8.0: Weights are adaptive, not hardcoded.
        """
        credits = {}
        
        # Get current adaptive weights
        coop_weight = self.coop_weight_tracker.mean
        pred_weight = self.pred_weight_tracker.mean
        
        for step in trajectory.steps:
            agent = step.get('agent', '')
            if not agent:
                continue
            
            # Base credit from task success
            task_success = step.get('success', 0.5)
            
            # Cooperation bonus: Did this agent's action help others?
            coop_bonus = self._compute_cooperation_bonus(
                agent, step, trajectory.steps
            )
            
            # Prediction accuracy bonus: Was this agent predictable?
            pred_bonus = self._compute_prediction_bonus(
                agent, step, predictions.get(agent)
            )
            
            # A-Team v8.0: Use adaptive weights
            credits[agent] = task_success * (1 + coop_weight * coop_bonus) * (1 + pred_weight * pred_bonus)
        
        # Normalize
        total = sum(credits.values())
        if total > 0:
            credits = {k: v / total * final_reward for k, v in credits.items()}
        
        # Update adaptive weights based on this episode's outcome
        if final_reward > 0.5:
            # Success - current weights worked, strengthen them
            avg_coop = sum(self._compute_cooperation_bonus(a, s, trajectory.steps) 
                          for a, s in [(step.get('agent', ''), step) for step in trajectory.steps]
                          if a) / max(1, len(trajectory.steps))
            self.coop_weight_tracker.update(avg_coop)
        
        return credits
    
    def _compute_cooperation_bonus(
        self,
        agent: str,
        step: Dict,
        all_steps: List[Dict]
    ) -> float:
        """Bonus for helping other agents succeed."""
        # Find steps after this agent
        agent_idx = next(
            (i for i, s in enumerate(all_steps) if s.get('agent') == agent),
            -1
        )
        if agent_idx < 0:
            return 0.0
        
        # Check if subsequent agents succeeded
        subsequent = all_steps[agent_idx + 1:]
        if not subsequent:
            return 0.0
        
        successes = sum(1 for s in subsequent if s.get('success', False))
        return successes / len(subsequent)
    
    def _compute_prediction_bonus(
        self,
        agent: str,
        actual_step: Dict,
        prediction: Optional[PredictedTrajectory]
    ) -> float:
        """Bonus for being predictable (helps others plan)."""
        if not prediction or not prediction.steps:
            return 0.0
        
        # Find predicted step for this agent
        for pred_step in prediction.steps:
            if pred_step.get('agent') == agent:
                if pred_step.get('action') == actual_step.get('action'):
                    return 1.0
                else:
                    return 0.0
        return 0.0


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'PredictedAction',
    'PredictedTrajectory',
    'ActualTrajectory',
    'Divergence',
    'AgentModel',
    'LLMTrajectoryPredictor',
    'DivergenceMemory',
    'CooperativeCreditAssigner'
]


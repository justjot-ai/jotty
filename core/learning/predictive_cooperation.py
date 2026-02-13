"""
PREDICTIVE COOPERATIVE SWARM
================================

STATUS: UNUSED — No classes in this module are instantiated outside core/learning/.
Candidate for archival. See MODULE_STATUS.md for details.

Breakthrough swarm architecture combining:
- Game Theory (Von Neumann, Nash)
- LLM Reasoning (Anthropic)
- Context Learning (Cursor)
- MARL (A-Team)

Key Innovation: Agents optimize for SYSTEM reward, not individual reward!

See also: predictive_marl.py (the actively used multi-agent RL module).
"""

import dspy
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# COOPERATION PRINCIPLES (Constitutional AI Approach)
# =============================================================================

class CooperationPrinciples:
    """
    Natural language cooperation principles (like Constitutional AI).
    
    Instead of hardcoded rules, agents follow these principles in their prompts.
    """
    
    @staticmethod
    def get_principles() -> str:
        return """
COOPERATION PRINCIPLES:

1. HELPFULNESS: Share information that helps others succeed
   - If you discover data another agent needs, proactively share it
   - Don't withhold information that could unblock others
   
2. HONESTY: Don't mislead or provide incomplete data
   - If unsure, say so explicitly
   - Provide confidence scores with your outputs
   
3. HARMLESSNESS: Don't block others unnecessarily
   - Only reject if there's a real issue
   - Provide constructive feedback, not just "no"
   
4. HUMILITY: Defer to agents with better information
   - If another agent has more relevant data, use theirs
   - Don't duplicate work others already did
   
5. EFFICIENCY: Reuse existing work
   - Check if another agent already solved this
   - Build on others' outputs instead of starting from scratch
   
6. SYSTEM REWARD: Optimize for swarm success, not individual credit
   - Your goal is the swarm succeeds, not that you look good
   - Help others even if you don't get direct credit
"""


# =============================================================================
# NASH BARGAINING FOR REWARD DISTRIBUTION
# =============================================================================

class NashBargainingSignature(dspy.Signature):
    """
    Determine fair reward distribution using Nash Bargaining Solution.
    
    Nash Bargaining maximizes: (R₁ - d₁) × (R₂ - d₂) × ... × (Rₙ - dₙ)
    where Rᵢ = agent i's reward, dᵢ = agent i's fallback (do nothing) reward
    
    This ensures:
    - Pareto efficiency (can't improve one without hurting another)
    - Symmetry (identical agents get identical rewards)
    - Independence of irrelevant alternatives
    - Invariance to affine transformations
    """
    
    agents_and_contributions = dspy.InputField(
        desc="List of agents with their contributions (what they did, how it helped)"
    )
    system_reward = dspy.InputField(
        desc="Total system reward achieved"
    )
    fallback_rewards = dspy.InputField(
        desc="What each agent would get if they did nothing (baseline)"
    )
    cooperation_context = dspy.InputField(
        desc="How agents cooperated (who helped whom, synergies)"
    )
    
    reasoning = dspy.OutputField(
        desc="Step-by-step Nash Bargaining reasoning"
    )
    reward_distribution = dspy.OutputField(
        desc="JSON dict of {agent_name: reward}, must sum to system_reward"
    )
    fairness_score = dspy.OutputField(
        desc="How fair is this distribution (0.0-1.0)"
    )
    cooperation_bonus = dspy.OutputField(
        desc="Extra reward for cooperation (shared among cooperating agents)"
    )


class NashBargainingSolver:
    """
    Solve Nash Bargaining problem using LLM reasoning.
    
    Traditional approach: Solve optimization problem
    Our approach: LLM reasons about fairness using Nash principles
    """
    
    def __init__(self):
        self.bargainer = dspy.ChainOfThought(NashBargainingSignature)
    
    def distribute_rewards(
        self,
        agents: List[str],
        contributions: Dict[str, Dict],
        system_reward: float,
        cooperation_events: List[Dict]
    ) -> Dict[str, float]:
        """
        Distribute system reward fairly using Nash Bargaining.
        
        Args:
            agents: List of agent names
            contributions: {agent_name: {action, impact, confidence}}
            system_reward: Total reward achieved
            cooperation_events: List of cooperation instances
        
        Returns:
            {agent_name: reward}
        """
        logger.info(f" Nash Bargaining: Distributing {system_reward:.2f} among {len(agents)} agents")
        
        # Format inputs
        agents_desc = self._format_contributions(agents, contributions)
        fallbacks = self._estimate_fallbacks(agents, contributions)
        coop_context = self._format_cooperation(cooperation_events)
        
        # LLM reasoning
        result = self.bargainer(
            agents_and_contributions=agents_desc,
            system_reward=str(system_reward),
            fallback_rewards=str(fallbacks),
            cooperation_context=coop_context
        )
        
        # Parse distribution
        try:
            import json
            distribution = json.loads(result.reward_distribution)
            
            # Validate
            total = sum(distribution.values())
            if abs(total - system_reward) > 0.01:
                logger.warning(f" Distribution sum ({total:.2f}) != system_reward ({system_reward:.2f})")
                # Normalize
                factor = system_reward / total if total > 0 else 1.0
                distribution = {k: v * factor for k, v in distribution.items()}
            
            logger.info(f" Nash Bargaining complete:")
            for agent, reward in distribution.items():
                logger.info(f"   {agent}: {reward:.3f}")
            logger.info(f"   Fairness: {result.fairness_score}")
            logger.info(f"   Cooperation Bonus: {result.cooperation_bonus}")
            
            return distribution
        
        except Exception as e:
            logger.error(f" Failed to parse distribution: {e}")
            # Fallback: Equal distribution
            equal_share = system_reward / len(agents)
            return {agent: equal_share for agent in agents}
    
    def _format_contributions(self, agents: List[str], contributions: Dict) -> str:
        """Format contributions for LLM."""
        lines = []
        for agent in agents:
            contrib = contributions.get(agent, {})
            lines.append(f"- {agent}:")
            lines.append(f"  Action: {contrib.get('action', 'N/A')}")
            lines.append(f"  Impact: {contrib.get('impact', 'Unknown')}")
            lines.append(f"  Confidence: {contrib.get('confidence', 0.5)}")
        return "\n".join(lines)
    
    def _estimate_fallbacks(self, agents: List[str], contributions: Dict) -> Dict[str, float]:
        """Estimate fallback rewards (what agents get if they do nothing)."""
        # Simple heuristic: 0.0 (no contribution = no reward)
        return {agent: 0.0 for agent in agents}
    
    def _format_cooperation(self, cooperation_events: List[Dict]) -> str:
        """Format cooperation events for LLM."""
        if not cooperation_events:
            return "No explicit cooperation events recorded."
        
        lines = []
        for event in cooperation_events:
            lines.append(f"- {event.get('from_agent')} → {event.get('to_agent')}")
            lines.append(f"  Type: {event.get('type', 'unknown')}")
            lines.append(f"  Impact: {event.get('impact', 'unknown')}")
        return "\n".join(lines)


# =============================================================================
# COOPERATION REASONER
# =============================================================================

class CooperationReasoningSignature(dspy.Signature):
    """
    Reason about cooperation to maximize system reward.
    
    Given:
    - My current state and available actions
    - Predictions about what other agents will do (from DQN)
    - Cooperation principles
    - Past interactions
    
    Decide:
    - Which action maximizes SYSTEM reward (not individual)
    - What to communicate to other agents
    - How to help unblock others
    """
    
    my_agent_name = dspy.InputField(desc="My agent name")
    my_state = dspy.InputField(desc="My current state and context")
    available_actions = dspy.InputField(desc="Actions I can take")
    
    predicted_other_actions = dspy.InputField(
        desc="DQN predictions: What other agents will likely do"
    )
    cooperation_principles = dspy.InputField(
        desc="Cooperation principles to follow"
    )
    past_interactions = dspy.InputField(
        desc="Memory of past interactions with other agents"
    )
    swarm_goal = dspy.InputField(desc="Overall swarm goal")
    
    reasoning = dspy.OutputField(
        desc="Step-by-step reasoning about cooperation and system reward"
    )
    best_action = dspy.OutputField(
        desc="Action that maximizes SYSTEM reward (not just my reward)"
    )
    communication_plan = dspy.OutputField(
        desc="What to share with whom (agent_name: message)"
    )
    confidence = dspy.OutputField(
        desc="Confidence in this decision (0.0-1.0)"
    )
    expected_system_reward = dspy.OutputField(
        desc="Expected system reward if I take this action"
    )


class CooperationReasoner:
    """
    LLM-based cooperation reasoner.
    
    Key innovation: Agents reason about SYSTEM reward, not individual reward!
    """
    
    def __init__(self):
        self.reasoner = dspy.ChainOfThought(CooperationReasoningSignature)
        self.principles = CooperationPrinciples.get_principles()
    
    def decide_cooperative_action(
        self,
        agent_name: str,
        state: Dict,
        available_actions: List[str],
        dqn_predictions: Dict[str, List[str]],
        memory: List[Dict],
        swarm_goal: str
    ) -> Dict:
        """
        Decide action that maximizes SYSTEM reward.
        
        Args:
            agent_name: My name
            state: My current state
            available_actions: Actions I can take
            dqn_predictions: {other_agent: [predicted_actions]}
            memory: Past interactions
            swarm_goal: Overall goal
        
        Returns:
            {
                'action': chosen_action,
                'reasoning': why_this_action,
                'communication': {agent: message},
                'confidence': float,
                'expected_system_reward': float
            }
        """
        logger.info(f" {agent_name}: Reasoning about cooperation...")
        
        # Format inputs
        state_desc = self._format_state(state)
        actions_desc = self._format_actions(available_actions)
        predictions_desc = self._format_predictions(dqn_predictions)
        memory_desc = self._format_memory(memory)
        
        # LLM reasoning
        result = self.reasoner(
            my_agent_name=agent_name,
            my_state=state_desc,
            available_actions=actions_desc,
            predicted_other_actions=predictions_desc,
            cooperation_principles=self.principles,
            past_interactions=memory_desc,
            swarm_goal=swarm_goal
        )
        
        # Parse communication plan
        try:
            import json
            comm_plan = json.loads(result.communication_plan)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"Communication plan parsing failed: {e}")
            comm_plan = {}
        
        decision = {
            'action': result.best_action,
            'reasoning': result.reasoning,
            'communication': comm_plan,
            'confidence': float(result.confidence) if result.confidence else 0.5,
            'expected_system_reward': float(result.expected_system_reward) if result.expected_system_reward else 0.5
        }
        
        logger.info(f" {agent_name} decided: {decision['action']}")
        logger.info(f"   Expected system reward: {decision['expected_system_reward']:.2f}")
        logger.info(f"   Will communicate with: {list(comm_plan.keys())}")
        
        return decision
    
    def _format_state(self, state: Dict) -> str:
        """Format state for LLM."""
        lines = []
        for key, value in state.items():
            if isinstance(value, (str, int, float, bool)):
                lines.append(f"- {key}: {value}")
            else:
                lines.append(f"- {key}: {type(value).__name__}")
        return "\n".join(lines) if lines else "No state information"
    
    def _format_actions(self, actions: List[str]) -> str:
        """Format actions for LLM."""
        return "\n".join(f"- {action}" for action in actions)
    
    def _format_predictions(self, predictions: Dict[str, List[str]]) -> str:
        """Format DQN predictions for LLM."""
        lines = []
        for agent, actions in predictions.items():
            lines.append(f"- {agent} will likely:")
            for action in actions[:3]:  # Top 3 predictions
                lines.append(f"  → {action}")
        return "\n".join(lines) if lines else "No predictions available"
    
    def _format_memory(self, memory: List[Dict]) -> str:
        """Format memory for LLM."""
        if not memory:
            return "No past interactions"
        
        lines = []
        for event in memory[-5:]:  # Last 5 interactions
            lines.append(f"- {event.get('summary', 'Unknown event')}")
        return "\n".join(lines)


# =============================================================================
# PREDICTIVE COOPERATIVE AGENT (MAIN CLASS)
# =============================================================================

@dataclass
class CooperationState:
    """Track cooperation state for an agent."""
    agent_name: str
    cooperation_events: List[Dict] = field(default_factory=list)
    help_given: Dict[str, int] = field(default_factory=dict)  # {other_agent: count}
    help_received: Dict[str, int] = field(default_factory=dict)
    total_system_reward_contributed: float = 0.0
    fairness_scores: List[float] = field(default_factory=list)


class PredictiveCooperativeAgent:
    """
    Agent that uses predictive cooperation to maximize system reward.
    
    Combines:
    - DQN predictions (what others will do)
    - LLM reasoning (cooperation strategy)
    - Nash Bargaining (fair reward distribution)
    - Shapley Value (credit assignment)
    """
    
    def __init__(self, agent_name: str, dqn_predictor=None, shapley_estimator=None):
        self.agent_name = agent_name
        self.dqn = dqn_predictor  # From predictive_marl.py
        self.shapley = shapley_estimator  # From algorithmic_credit.py
        
        # Cooperation components
        self.cooperation_reasoner = CooperationReasoner()
        self.nash_bargainer = NashBargainingSolver()
        
        # State
        self.cooperation_state = CooperationState(agent_name=agent_name)
        
        logger.info(f" PredictiveCooperativeAgent '{agent_name}' initialized")
    
    def decide_action(
        self,
        state: Dict,
        available_actions: List[str],
        other_agents: List[str],
        swarm_goal: str,
        memory: List[Dict]
    ) -> Dict:
        """
        Decide action using predictive cooperation.
        
        Workflow:
        1. Predict what other agents will do (DQN)
        2. Reason about cooperation (LLM)
        3. Choose action that maximizes SYSTEM reward
        
        Returns:
            {
                'action': chosen_action,
                'reasoning': why,
                'communication': {agent: message},
                'predictions': {agent: predicted_actions}
            }
        """
        logger.info(f" {self.agent_name}: Deciding cooperative action...")
        
        # STEP 1: Predict other agents (DQN)
        predictions = {}
        if self.dqn:
            for other_agent in other_agents:
                if other_agent != self.agent_name:
                    pred = self.dqn.predict_next(
                        agent=other_agent,
                        current_state=state,
                        n=3  # Top 3 predictions
                    )
                    predictions[other_agent] = pred
        
        # STEP 2: Reason about cooperation (LLM)
        decision = self.cooperation_reasoner.decide_cooperative_action(
            agent_name=self.agent_name,
            state=state,
            available_actions=available_actions,
            dqn_predictions=predictions,
            memory=memory,
            swarm_goal=swarm_goal
        )
        
        # Add predictions to decision
        decision['predictions'] = predictions
        
        return decision
    
    def record_cooperation_event(
        self,
        event_type: str,
        other_agent: str,
        description: str,
        impact: float
    ):
        """Record a cooperation event."""
        event = {
            'type': event_type,
            'from_agent': self.agent_name,
            'to_agent': other_agent,
            'description': description,
            'impact': impact,
            'timestamp': __import__('time').time()
        }
        
        self.cooperation_state.cooperation_events.append(event)
        
        if event_type in ['help', 'share', 'unblock']:
            self.cooperation_state.help_given[other_agent] = \
                self.cooperation_state.help_given.get(other_agent, 0) + 1
        
        logger.info(f" {self.agent_name} → {other_agent}: {event_type} ({description})")
    
    def distribute_system_reward(
        self,
        all_agents: List[str],
        contributions: Dict[str, Dict],
        system_reward: float
    ) -> Dict[str, float]:
        """
        Distribute system reward using Nash Bargaining.
        
        Args:
            all_agents: All agent names
            contributions: {agent: {action, impact, confidence}}
            system_reward: Total reward
        
        Returns:
            {agent: reward}
        """
        return self.nash_bargainer.distribute_rewards(
            agents=all_agents,
            contributions=contributions,
            system_reward=system_reward,
            cooperation_events=self.cooperation_state.cooperation_events
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'CooperationPrinciples',
    'NashBargainingSolver',
    'CooperationReasoner',
    'PredictiveCooperativeAgent',
    'CooperationState',
    'NashBargainingSignature',
    'CooperationReasoningSignature'
]


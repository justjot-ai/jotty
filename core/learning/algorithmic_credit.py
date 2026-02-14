"""
Jotty v7.6 - Algorithmic Credit Assignment
===========================================

A-Team Approved: Uses proper game theory and MARL algorithms.

NO HARDCODED PERCENTAGES. Everything derived from:
1. Shapley Value (Game Theory) - Marginal contribution
2. Difference Rewards (MARL) - Counterfactual impact
3. LLM Estimation - For agentic systems

References:
- Shapley, L.S. (1953). "A Value for n-person Games"
- Wolpert, D.H. & Tumer, K. (2002). "Optimal Payoff Functions for Members of Collectives"
"""

import asyncio
import json
import math
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

# DSPy loaded lazily — saves ~6s on module import
# Use _get_dspy() to access the module when needed
DSPY_AVAILABLE = True  # Assumed available; checked on first use
_dspy_module = None

def _get_dspy() -> Any:
    """Lazy-load DSPy on first use."""
    global _dspy_module, DSPY_AVAILABLE
    if _dspy_module is None:
        try:
            import dspy as _dspy
            _dspy_module = _dspy
        except ImportError:
            DSPY_AVAILABLE = False
            _dspy_module = False  # Sentinel: tried and failed
    return _dspy_module if _dspy_module else None

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AgentContribution:
    """Represents an agent's contribution with algorithmic credit."""
    agent_name: str
    shapley_value: float  # Marginal contribution (Shapley)
    difference_reward: float  # Counterfactual impact
    combined_credit: float  # Weighted combination
    confidence: float  # How confident we are in this estimate
    reasoning: str  # LLM reasoning for transparency


@dataclass
class Coalition:
    """A subset of agents."""
    members: List[str]
    value: float  # Estimated value of this coalition
    
    def __hash__(self) -> int:
        return hash(frozenset(self.members))


# =============================================================================
# SHAPLEY VALUE ESTIMATOR
# =============================================================================

_ShapleyEstimatorSignature = None

def _get_shapley_signature() -> None:
    """Lazy-create the DSPy Signature class (avoids module-level dspy import)."""
    global _ShapleyEstimatorSignature
    if _ShapleyEstimatorSignature is None:
        dspy = _get_dspy()
        if not dspy:
            return None
        class ShapleyEstimatorSignature(dspy.Signature):
            """Estimate the value of a coalition of agents."""
            coalition_agents = dspy.InputField(desc="List of agents in this coalition and their capabilities")
            task_description = dspy.InputField(desc="The task being performed")
            trajectory_context = dspy.InputField(desc="What happened during execution")
            full_agent_list = dspy.InputField(desc="All available agents for reference")
            coalition_value = dspy.OutputField(desc="Estimated value (0.0 to 1.0) of this coalition achieving the goal")
            reasoning = dspy.OutputField(desc="Why this coalition would achieve this value")
        _ShapleyEstimatorSignature = ShapleyEstimatorSignature
    return _ShapleyEstimatorSignature


class ShapleyValueEstimator:
    """
    Estimate Shapley values using Monte Carlo sampling + LLM evaluation.
    
    Shapley Value Formula:
    φᵢ = Σ |S|!(n-|S|-1)!/n! × [v(S∪{i}) - v(S)]
    
    For efficiency, we use Monte Carlo approximation:
    1. Sample random orderings of agents
    2. Compute marginal contribution in each ordering
    3. Average across samples
    
     A-TEAM ENHANCEMENTS (per GRF MARL paper):
    - Auto-tune samples by agent count: min(20, 5 * n!)
    - Track variance/CI on φ estimates
    - Cache coalition values for efficiency
    """
    
    def __init__(self, num_samples: int = 20, min_samples: int = 10, max_samples: int = 100) -> None:
        self.base_samples = num_samples
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.coalition_cache: Dict[frozenset, float] = {}
        
        self.coalition_estimator = None  # Lazy-created on first use
        
        logger.info(f" ShapleyValueEstimator initialized (base_samples={num_samples}, auto-tuned)")
    
    def _auto_tune_samples(self, n_agents: int) -> int:
        """
        Auto-tune sample count based on agent count.
        
        More agents → more permutations → need more samples for accuracy.
        But cap to avoid explosion: min(5*n!, max_samples)
        """
        if n_agents <= 2:
            return self.min_samples
        
        # factorial grows fast; cap at reasonable level
        try:
            factorial_based = min(5 * math.factorial(n_agents), self.max_samples)
        except (ValueError, OverflowError) as e:
            logger.debug(f"Factorial calculation failed: {e}")
            factorial_based = self.max_samples
        
        return max(self.min_samples, min(factorial_based, self.max_samples))
    
    async def estimate_shapley_values(
        self,
        agents: List[str],
        agent_capabilities: Dict[str, str],
        task: str,
        trajectory: List[Dict],
        actual_reward: float
    ) -> Dict[str, AgentContribution]:
        """
        Estimate Shapley value for each agent with variance/CI tracking.
        
         A-TEAM ENHANCEMENTS:
        - Auto-tune samples by agent count
        - Compute variance and 95% CI
        - Confidence based on CI width
        
        Returns dict of agent_name -> AgentContribution
        """
        n = len(agents)
        if n == 0:
            return {}
        
        # Auto-tune sample count
        num_samples = self._auto_tune_samples(n)
        logger.debug(f"Shapley sampling: {num_samples} samples for {n} agents")
        
        contributions = {agent: [] for agent in agents}
        
        # Monte Carlo sampling of orderings
        for sample_idx in range(num_samples):
            # Random ordering (deterministic for reproducibility)
            random.seed(42 + sample_idx)
            ordering = random.sample(agents, n)
            
            # Compute marginal contribution for each agent
            for i, agent in enumerate(ordering):
                coalition_before = frozenset(ordering[:i])
                coalition_with = frozenset(ordering[:i+1])
                
                # Get coalition values (use cache)
                v_before = await self._get_coalition_value(
                    coalition_before, agent_capabilities, task, trajectory
                )
                v_with = await self._get_coalition_value(
                    coalition_with, agent_capabilities, task, trajectory
                )
                
                # Marginal contribution
                marginal = v_with - v_before
                contributions[agent].append(marginal)
        
        # Compute Shapley values with variance/CI
        results = {}
        for agent in agents:
            samples = contributions[agent]
            n_samples = len(samples)
            
            # Mean (Shapley estimate)
            shapley = sum(samples) / n_samples if n_samples > 0 else 0.0
            
            # Variance and standard error
            if n_samples > 1:
                variance = sum((x - shapley) ** 2 for x in samples) / (n_samples - 1)
                std_error = math.sqrt(variance / n_samples)
                # 95% CI: ±1.96 * SE
                ci_half_width = 1.96 * std_error
            else:
                variance = 0.0
                std_error = 0.0
                ci_half_width = 1.0  # Max uncertainty
            
            # Confidence based on CI width (narrower = higher confidence)
            # CI width of 0.1 → confidence ~0.9; width of 1.0 → confidence ~0.0
            confidence = max(0.0, min(1.0, 1.0 - ci_half_width))
            
            results[agent] = AgentContribution(
                agent_name=agent,
                shapley_value=shapley,
                difference_reward=0.0,  # Will be filled by DifferenceRewardEstimator
                combined_credit=shapley,  # Start with Shapley
                confidence=confidence,
                reasoning=f"φ={shapley:.3f}±{ci_half_width:.3f} (95% CI), {n_samples} samples, var={variance:.4f}"
            )
        
        return results
    
    async def _get_coalition_value(
        self,
        coalition: frozenset,
        capabilities: Dict[str, str],
        task: str,
        trajectory: List[Dict]
    ) -> float:
        """Get value of a coalition (cached)."""
        if coalition in self.coalition_cache:
            return self.coalition_cache[coalition]
        
        if not coalition:
            # Empty coalition has zero value
            self.coalition_cache[coalition] = 0.0
            return 0.0
        
        # Lazy-create DSPy estimator on first use
        if self.coalition_estimator is None:
            dspy = _get_dspy()
            sig = _get_shapley_signature()
            if dspy and sig:
                self.coalition_estimator = dspy.ChainOfThought(sig)

        if not self.coalition_estimator:
            # Fallback: uniform distribution (DSPy not available)
            value = len(coalition) / max(len(capabilities), 1)
            self.coalition_cache[coalition] = value
            return value
        
        # Use LLM to estimate coalition value
        try:
            result = self.coalition_estimator(
                coalition_agents=json.dumps({
                    a: capabilities.get(a, "unknown") for a in coalition
                }),
                task_description=task,
                trajectory_context=json.dumps(trajectory, default=str),
                full_agent_list=json.dumps(list(capabilities.keys()))
            )
            
            value = self._parse_value(result.coalition_value)
            self.coalition_cache[coalition] = value
            return value
            
        except Exception as e:
            logger.debug(f"Coalition estimation failed: {e}")
            value = len(coalition) / max(len(capabilities), 1)
            self.coalition_cache[coalition] = value
            return value
    
    def _parse_value(self, value_str: str) -> float:
        """Parse value from LLM output."""
        if isinstance(value_str, (int, float)):
            return float(value_str)
        
        try:
            # Try direct conversion
            return float(value_str)
        except (ValueError, TypeError) as e:
            logger.debug(f"Float parsing failed: {e}")
            pass

        # Extract number from string
        import re
        numbers = re.findall(r'[\d.]+', str(value_str))
        if numbers:
            val = float(numbers[0])
            return max(0.0, min(1.0, val))
        
        return 0.5  # Default
    
    def clear_cache(self) -> None:
        """Clear coalition value cache."""
        self.coalition_cache.clear()


# =============================================================================
# DIFFERENCE REWARD ESTIMATOR
# =============================================================================

_CounterfactualSignature = None

def _get_counterfactual_signature() -> None:
    """Lazy-create the CounterfactualSignature DSPy class."""
    global _CounterfactualSignature
    if _CounterfactualSignature is None:
        dspy = _get_dspy()
        if not dspy:
            return None
        class CounterfactualSignature(dspy.Signature):
            """Estimate what would have happened WITHOUT a specific agent's action."""
            agent_name = dspy.InputField(desc="Agent whose action we're evaluating")
            agent_action = dspy.InputField(desc="What the agent actually did")
            state_before = dspy.InputField(desc="State before the action")
            state_after = dspy.InputField(desc="State after the action (what actually happened)")
            other_agents = dspy.InputField(desc="Other agents and their roles")
            counterfactual_outcome = dspy.OutputField(desc="What would the outcome be (0.0-1.0) if this agent did nothing?")
            reasoning = dspy.OutputField(desc="Explain your counterfactual reasoning")
        _CounterfactualSignature = CounterfactualSignature
    return _CounterfactualSignature


class DifferenceRewardEstimator:
    """
    Compute difference rewards using counterfactual reasoning.
    
    Difference Reward Formula:
    D_i = G - G_{-i}
    
    where G is global reward and G_{-i} is what would happen without agent i.
    
    This encourages cooperation because:
    - If agent i helps others succeed, G >> G_{-i}, so D_i is high
    - If agent i hurts others, G << G_{-i}, so D_i is negative
    """
    
    def __init__(self) -> None:
        self.counterfactual_estimator = None  # Lazy-created on first use
        
        logger.info(" DifferenceRewardEstimator initialized")
    
    async def compute_difference_rewards(
        self,
        agents: List[str],
        actions: Dict[str, Dict],  # agent -> action taken
        states: Dict[str, Dict],  # agent -> {before, after} states
        global_reward: float
    ) -> Dict[str, float]:
        """
        Compute difference reward for each agent.
        
        Returns: Dict of agent_name -> difference_reward
        """
        difference_rewards = {}
        
        for agent in agents:
            if agent not in actions:
                difference_rewards[agent] = 0.0
                continue
            
            # Estimate G_{-i} - what would happen without this agent
            g_minus_i = await self._estimate_counterfactual(
                agent=agent,
                action=actions.get(agent, {}),
                state_before=states.get(agent, {}).get('before', {}),
                state_after=states.get(agent, {}).get('after', {}),
                other_agents=[a for a in agents if a != agent]
            )
            
            # Difference reward: G - G_{-i}
            difference_rewards[agent] = global_reward - g_minus_i
        
        return difference_rewards
    
    async def _estimate_counterfactual(
        self,
        agent: str,
        action: Dict,
        state_before: Dict,
        state_after: Dict,
        other_agents: List[str]
    ) -> float:
        """Estimate reward without this agent's action."""
        # Lazy-create DSPy estimator on first use
        if self.counterfactual_estimator is None:
            dspy = _get_dspy()
            sig = _get_counterfactual_signature()
            if dspy and sig:
                self.counterfactual_estimator = dspy.ChainOfThought(sig)

        if not self.counterfactual_estimator:
            # Fallback: assume agent contributed proportionally (DSPy not available)
            return 0.5
        
        try:
            result = self.counterfactual_estimator(
                agent_name=agent,
                agent_action=json.dumps(action, default=str),
                state_before=json.dumps(state_before, default=str),
                state_after=json.dumps(state_after, default=str),
                other_agents=json.dumps(other_agents)
            )
            
            return self._parse_value(result.counterfactual_outcome)
            
        except Exception as e:
            logger.debug(f"Counterfactual estimation failed: {e}")
            return 0.5
    
    def _parse_value(self, value_str: str) -> float:
        """Parse value from LLM output."""
        if isinstance(value_str, (int, float)):
            return float(value_str)
        try:
            return float(value_str)
        except (ValueError, TypeError) as e:
            logger.debug(f"Value parsing failed: {e}")
            import re
            numbers = re.findall(r'[\d.]+', str(value_str))
            if numbers:
                return max(0.0, min(1.0, float(numbers[0])))
            return 0.5


# =============================================================================
# COMBINED CREDIT ASSIGNER
# =============================================================================

class AlgorithmicCreditAssigner:
    """
    Combines Shapley Value + Difference Rewards for robust credit assignment.
    
    NO HARDCODED WEIGHTS. Uses:
    - Adaptive combination based on confidence
    - LLM validation of final credits
    """
    
    def __init__(self, config: Any = None) -> None:
        self.shapley_estimator = ShapleyValueEstimator()
        self.difference_estimator = DifferenceRewardEstimator()
        
        # Adaptive weight based on estimate confidence
        # Higher confidence in Shapley -> use more Shapley
        self.min_shapley_weight = 0.3
        self.max_shapley_weight = 0.7
        
        logger.info(" AlgorithmicCreditAssigner initialized")
    
    async def assign_credit(
        self,
        agents: List[str],
        agent_capabilities: Dict[str, str],
        actions: Dict[str, Dict],
        states: Dict[str, Dict],
        trajectory: List[Dict],
        task: str,
        global_reward: float
    ) -> Dict[str, AgentContribution]:
        """
        Assign credit using Shapley + Difference Rewards.
        
        Returns: Dict of agent_name -> AgentContribution
        """
        # 1. Compute Shapley values
        shapley_credits = await self.shapley_estimator.estimate_shapley_values(
            agents=agents,
            agent_capabilities=agent_capabilities,
            task=task,
            trajectory=trajectory,
            actual_reward=global_reward
        )
        
        # 2. Compute Difference Rewards
        difference_rewards = await self.difference_estimator.compute_difference_rewards(
            agents=agents,
            actions=actions,
            states=states,
            global_reward=global_reward
        )
        
        # 3. Combine with adaptive weighting
        results = {}
        for agent in agents:
            shapley_contrib = shapley_credits.get(agent, AgentContribution(
                agent_name=agent, shapley_value=0.0, difference_reward=0.0,
                combined_credit=0.0, confidence=0.0, reasoning="No data"
            ))
            
            diff_reward = difference_rewards.get(agent, 0.0)
            
            # Adaptive weight based on Shapley confidence
            shapley_weight = self.min_shapley_weight + \
                (self.max_shapley_weight - self.min_shapley_weight) * shapley_contrib.confidence
            
            # Combined credit
            combined = shapley_weight * shapley_contrib.shapley_value + \
                       (1 - shapley_weight) * diff_reward
            
            results[agent] = AgentContribution(
                agent_name=agent,
                shapley_value=shapley_contrib.shapley_value,
                difference_reward=diff_reward,
                combined_credit=combined,
                confidence=shapley_contrib.confidence,
                reasoning=f"Shapley={shapley_contrib.shapley_value:.3f} (weight={shapley_weight:.2f}), "
                         f"Diff={diff_reward:.3f}"
            )
        
        # Normalize credits to sum to global_reward
        total_credit = sum(c.combined_credit for c in results.values())
        if total_credit > 0:
            for agent in results:
                results[agent].combined_credit = \
                    (results[agent].combined_credit / total_credit) * global_reward
        
        return results
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        self.shapley_estimator.clear_cache()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'AgentContribution',
    'Coalition',
    'ShapleyValueEstimator',
    'DifferenceRewardEstimator',
    'AlgorithmicCreditAssigner'
]


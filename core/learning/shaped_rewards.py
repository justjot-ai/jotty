"""
Shaped Rewards - Intermediate Rewards for Long-Horizon Tasks
============================================================

 ADDRESSES: Sparse reward problem (GRF MARL paper, Song et al. 2023)

Long-horizon tasks (like SQL generation) have sparse final rewards.
Shaped rewards provide intermediate signals for partial progress.

 NO HARDCODING - Reward conditions are LLM-evaluated, not rule-based.

Research Foundations:
- Yang & Tang (2020): "Adaptive inner-reward shaping in sparse reward games"
- Wang et al. (2022): "Individual Reward Assisted MARL"
- Song et al. (2023): "GRF Multi-agent Scenarios" - Reward shaping for long games
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
import time

logger = logging.getLogger(__name__)

# DSPy loaded lazily for agentic reward evaluation
_dspy_module = None
DSPY_AVAILABLE = None  # Determined on first access

def _get_dspy():
    global _dspy_module, DSPY_AVAILABLE
    if DSPY_AVAILABLE is None:
        try:
            import dspy
            _dspy_module = dspy
            DSPY_AVAILABLE = True
        except ImportError:
            DSPY_AVAILABLE = False
    return _dspy_module


# =============================================================================
# REWARD CONDITIONS
# =============================================================================

@dataclass
class RewardCondition:
    """
    A condition that triggers an intermediate reward.
    
     NO HARDCODING - Condition is evaluated by LLM, not regex.
    """
    name: str
    description: str
    reward_value: float
    
    # When to check this condition
    check_after: str = "any"  # "any", "actor_complete", "tool_call", "validation"
    
    # One-time or recurring
    one_time: bool = True
    
    # Tracking
    triggered: bool = False
    triggered_at: Optional[float] = None
    trigger_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'reward': self.reward_value,
            'check_after': self.check_after,
            'one_time': self.one_time,
            'triggered': self.triggered,
            'count': self.trigger_count,
        }


# =============================================================================
# AGENTIC REWARD EVALUATOR
# =============================================================================

_RewardConditionSignature = None
def _get_reward_condition_signature():
    global _RewardConditionSignature
    if _RewardConditionSignature is None:
        dspy = _get_dspy()
        if dspy is None:
            return None
        class RewardConditionSignature(dspy.Signature):
            """Evaluate if a reward condition is met based on current state."""
            condition_name = dspy.InputField(desc="Name of the condition to check")
            condition_description = dspy.InputField(desc="Description of what needs to be true")
            current_state = dspy.InputField(desc="Current state including outputs, context, trajectory")
            trajectory_summary = dspy.InputField(desc="What has been done so far")
            is_met = dspy.OutputField(desc="YES or NO - is the condition met?")
            evidence = dspy.OutputField(desc="Evidence supporting the decision")
            confidence = dspy.OutputField(desc="Confidence 0.0-1.0")
        _RewardConditionSignature = RewardConditionSignature
    return _RewardConditionSignature


class AgenticRewardEvaluator:
    """
     NO HARDCODING - LLM evaluates reward conditions.
    
    Instead of regex checking for "table_name in output", the LLM
    reasons about whether progress conditions are met.
    """
    
    def __init__(self):
        sig = _get_reward_condition_signature()
        if sig is not None:
            dspy = _get_dspy()
            self.evaluator = dspy.ChainOfThought(sig)
        else:
            self.evaluator = None
        logger.info(" AgenticRewardEvaluator initialized (pure LLM, no regex)")
    
    def evaluate_condition(
        self,
        condition: RewardCondition,
        state: Dict[str, Any],
        trajectory: List[Dict]
    ) -> tuple[bool, float, str]:
        """
        Evaluate if a condition is met.
        
        Returns: (is_met: bool, confidence: float, evidence: str)
        """
        if not self.evaluator:
            return False, 0.0, "No evaluator available"
        
        try:
            # Format trajectory
            traj_summary = self._format_trajectory(trajectory)
            
            # Format state
            state_summary = self._format_state(state)
            
            # LLM evaluation
            result = self.evaluator(
                condition_name=condition.name,
                condition_description=condition.description,
                current_state=state_summary,
                trajectory_summary=traj_summary,
            )
            
            # Parse result
            is_met = result.is_met.upper().startswith("YES")
            try:
                confidence = float(result.confidence)
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug(f"Confidence parsing failed: {e}")
                confidence = 0.5
            
            return is_met, confidence, result.evidence
            
        except Exception as e:
            logger.warning(f" Condition evaluation failed: {e}")
            return False, 0.0, str(e)
    
    def _format_trajectory(self, trajectory: List[Dict]) -> str:
        """Format trajectory for LLM."""
        if not trajectory:
            return "No actions taken yet."
        
        steps = []
        for i, step in enumerate(trajectory[-10:]):  # Last 10 steps
            actor = step.get('actor', 'unknown')
            action = step.get('action', step.get('type', 'action'))
            result = step.get('result', step.get('output', ''))
            steps.append(f"{i+1}. [{actor}] {action}: {str(result)[:100]}")
        
        return "\n".join(steps)
    
    def _format_state(self, state: Dict[str, Any]) -> str:
        """Format state for LLM."""
        parts = []
        for key, value in list(state.items())[:15]:
            val_str = str(value)[:200]
            parts.append(f"- {key}: {val_str}")
        return "\n".join(parts)


# =============================================================================
# SHAPED REWARD MANAGER
# =============================================================================

class ShapedRewardManager:
    """
    Manages intermediate shaped rewards for long-horizon tasks.
    
     ADDRESSES: Sparse reward problem (GRF MARL paper)
     NO HARDCODING: Conditions evaluated by LLM
    
    Standard Conditions (generic, not domain-specific):
    - input_validated: All required inputs are present and valid
    - dependency_resolved: Dependencies on other agents are resolved
    - tool_call_success: Tool was called and returned valid result
    - partial_output: Agent produced partial output
    - full_output: Agent produced complete output
    - validation_passed: Validation (Auditor) approved output
    - execution_success: External execution succeeded (e.g., SQL ran)
    - goal_achieved: Final goal is achieved
    """
    
    # Standard conditions (generic for any domain)
    STANDARD_CONDITIONS = [
        RewardCondition(
            name="input_validated",
            description="All required inputs for the agent are present and appear valid",
            reward_value=0.05,
            check_after="actor_start",
        ),
        RewardCondition(
            name="dependency_resolved",
            description="Outputs from dependent agents have been received and are usable",
            reward_value=0.05,
            check_after="actor_start",
        ),
        RewardCondition(
            name="tool_call_success",
            description="A tool was called and returned a successful result (not error)",
            reward_value=0.1,
            check_after="tool_call",
            one_time=False,  # Can trigger multiple times
        ),
        RewardCondition(
            name="partial_output",
            description="The agent has produced some output, even if incomplete",
            reward_value=0.1,
            check_after="actor_complete",
        ),
        RewardCondition(
            name="full_output",
            description="The agent has produced a complete output with all required fields",
            reward_value=0.15,
            check_after="actor_complete",
        ),
        RewardCondition(
            name="validation_passed",
            description="The Auditor/validation agent approved the output",
            reward_value=0.2,
            check_after="validation",
        ),
        RewardCondition(
            name="execution_success",
            description="External execution (e.g., query, API call) succeeded with valid results",
            reward_value=0.25,
            check_after="tool_call",
        ),
        RewardCondition(
            name="goal_achieved",
            description="The final goal of the task has been achieved successfully",
            reward_value=0.5,
            check_after="actor_complete",
        ),
    ]
    
    def __init__(self, custom_conditions: Optional[List[RewardCondition]] = None):
        """
        Initialize with standard + optional custom conditions.
        
        Parameters:
        -----------
        custom_conditions : List[RewardCondition], optional
            Additional domain-specific conditions.
        """
        self.conditions = [
            RewardCondition(**c.__dict__) for c in self.STANDARD_CONDITIONS
        ]
        
        if custom_conditions:
            self.conditions.extend(custom_conditions)
        
        self.evaluator = AgenticRewardEvaluator()
        
        # Tracking
        self.total_shaped_reward = 0.0
        self.reward_history: List[Dict] = []
        
        logger.info(f" ShapedRewardManager initialized with {len(self.conditions)} conditions")
    
    def check_rewards(
        self,
        event_type: str,
        state: Dict[str, Any],
        trajectory: List[Dict],
    ) -> float:
        """
        Check all applicable conditions and return total shaped reward.
        
        Parameters:
        -----------
        event_type : str
            What triggered the check: "actor_start", "tool_call", 
            "actor_complete", "validation"
            
        state : Dict
            Current state including outputs, context.
            
        trajectory : List[Dict]
            Execution trajectory.
        
        Returns:
        --------
        float: Total shaped reward from triggered conditions.
        """
        total_reward = 0.0
        
        for condition in self.conditions:
            # Skip if not applicable to this event
            if condition.check_after != "any" and condition.check_after != event_type:
                continue
            
            # Skip if one-time and already triggered
            if condition.one_time and condition.triggered:
                continue
            
            # Evaluate condition
            is_met, confidence, evidence = self.evaluator.evaluate_condition(
                condition=condition,
                state=state,
                trajectory=trajectory,
            )
            
            if is_met and confidence >= 0.7:  # High confidence threshold
                # Trigger reward
                reward = condition.reward_value * confidence
                total_reward += reward
                
                condition.triggered = True
                condition.triggered_at = time.time()
                condition.trigger_count += 1
                
                # Log
                self.reward_history.append({
                    'condition': condition.name,
                    'reward': reward,
                    'confidence': confidence,
                    'evidence': evidence,
                    'timestamp': time.time(),
                })
                
                logger.info(f" Shaped reward: {condition.name} = {reward:.3f} (confidence={confidence:.2f})")
                logger.debug(f"   Evidence: {evidence}")
        
        self.total_shaped_reward += total_reward
        return total_reward
    
    def get_total_reward(self) -> float:
        """Get total accumulated shaped reward."""
        return self.total_shaped_reward
    
    def reset(self) -> None:
        """Reset all conditions for new episode."""
        for condition in self.conditions:
            condition.triggered = False
            condition.triggered_at = None
            condition.trigger_count = 0
        
        self.total_shaped_reward = 0.0
        self.reward_history.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of shaped rewards."""
        return {
            'total_shaped_reward': self.total_shaped_reward,
            'conditions_triggered': sum(1 for c in self.conditions if c.triggered),
            'total_conditions': len(self.conditions),
            'reward_history': self.reward_history[-10:],  # Last 10
            'condition_status': {c.name: c.triggered for c in self.conditions},
        }
    
    def add_condition(self, condition: RewardCondition) -> None:
        """Add a custom condition at runtime."""
        self.conditions.append(condition)
        logger.info(f" Added custom condition: {condition.name}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ShapedRewardManager',
    'RewardCondition',
    'AgenticRewardEvaluator',
]


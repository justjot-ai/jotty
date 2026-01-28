"""
Jotty v6.0 - Enhanced Learning Module
=====================================

All A-Team learning enhancements:
- Dr. Manning: Adaptive α, intermediate rewards, value generalization
- Dr. Chen: Reasoning-based credit, temporal weighting
- Shannon: Information-theoretic value estimation

Correct TD(λ) with all fixes applied.
"""

import math
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

from ..foundation.data_structures import (
    JottyConfig, MemoryEntry, MemoryLevel, GoalValue,
    ValidationResult, AgentContribution, StoredEpisode,
    LearningMetrics, AlertType, GoalHierarchy, CausalLink
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..memory.cortex import HierarchicalMemory


# =============================================================================
# ADAPTIVE LEARNING RATE (Dr. Manning Enhancement)
# =============================================================================

class AdaptiveLearningRate:
    """
    Dynamically adjusts learning rate based on learning progress.
    
    Increases α when learning is slow, decreases when unstable.
    
    Methods:
    - Gradient magnitude tracking
    - Success rate monitoring
    - TD error variance analysis
    """
    
    def __init__(self, config: JottyConfig):
        self.config = config
        self.alpha = config.alpha
        self.min_alpha = config.alpha_min
        self.max_alpha = config.alpha_max
        self.adaptation_rate = config.alpha_adaptation_rate
        
        # Tracking
        self.td_errors: List[float] = []
        self.success_rates: List[float] = []
        self.window_size = config.adaptive_window_size  # 🔧 STANFORD FIX
    
    def record_td_error(self, td_error: float):
        """Record a TD error for variance analysis."""
        self.td_errors.append(abs(td_error))
        if len(self.td_errors) > self.window_size * 2:
            self.td_errors = self.td_errors[-self.window_size:]
    
    def record_success(self, success: bool):
        """Record episode outcome."""
        self.success_rates.append(1.0 if success else 0.0)
        if len(self.success_rates) > self.window_size * 2:
            self.success_rates = self.success_rates[-self.window_size:]
    
    def get_adapted_alpha(self) -> float:
        """
        Get current learning rate with adaptations.
        
        Adaptation logic:
        1. High TD error variance → decrease α (unstable)
        2. Low TD error magnitude → increase α (slow learning)
        3. Success rate improving → maintain α
        4. Success rate declining → increase α (need to learn faster)
        """
        if not self.config.enable_adaptive_alpha:
            return self.config.alpha
        
        if len(self.td_errors) < 10:
            return self.alpha
        
        recent_errors = self.td_errors[-self.window_size:]
        
        # Compute variance
        mean_error = sum(recent_errors) / len(recent_errors)
        variance = sum((e - mean_error) ** 2 for e in recent_errors) / len(recent_errors)
        std_dev = math.sqrt(variance)
        
        # Adaptation decision
        adjustment = 0.0
        
        # High variance → unstable → decrease α
        if std_dev > mean_error * self.config.instability_threshold_multiplier:  # 🔧 STANFORD FIX
            adjustment -= self.adaptation_rate
        
        # Low mean error → slow learning → increase α
        elif mean_error < self.config.slow_learning_threshold:  # 🔧 STANFORD FIX
            adjustment += self.adaptation_rate
        
        # Check success rate trend
        if len(self.success_rates) >= 20:
            recent = sum(self.success_rates) / 10
            older = sum(self.success_rates[-20:-10]) / 10
            
            if recent < older - 0.1:  # Declining
                adjustment += self.adaptation_rate * 0.5
        
        # Apply adjustment with bounds
        self.alpha = self.alpha * (1 + adjustment)
        self.alpha = max(self.min_alpha, min(self.max_alpha, self.alpha))
        
        return self.alpha
    
    def reset(self):
        """Reset tracking."""
        self.alpha = self.config.alpha
        self.td_errors.clear()
        self.success_rates.clear()


# =============================================================================
# INTERMEDIATE REWARDS (Dr. Manning Enhancement)
# =============================================================================

class IntermediateRewardCalculator:
    """
    Calculates intermediate rewards for dense learning signal.
    
    Instead of only terminal reward, provides rewards for:
    - Architect proceed decisions
    - Successful tool calls
    - Partial task completion
    - Good reasoning steps
    """
    
    def __init__(self, config: JottyConfig):
        self.config = config
        self.step_rewards: List[float] = []
    
    def reset(self):
        """Reset for new episode."""
        self.step_rewards.clear()
    
    def reward_architect_proceed(self, confidence: float) -> float:
        """Reward for Architect proceed decision."""
        if not self.config.enable_intermediate_rewards:
            return 0.0
        
        # Higher confidence proceed → higher reward
        reward = self.config.architect_proceed_reward * confidence
        self.step_rewards.append(reward)
        return reward
    
    def reward_tool_success(self, tool_name: str, success: bool) -> float:
        """Reward for successful tool call."""
        if not self.config.enable_intermediate_rewards:
            return 0.0
        
        if success:
            reward = self.config.tool_success_reward
        else:
            reward = -self.config.tool_success_reward * 0.5
        
        self.step_rewards.append(reward)
        return reward
    
    def reward_partial_completion(self, completion_fraction: float) -> float:
        """Reward for partial task completion."""
        if not self.config.enable_intermediate_rewards:
            return 0.0
        
        reward = completion_fraction * 0.3  # Max 0.3 for partial
        self.step_rewards.append(reward)
        return reward
    
    def get_total_intermediate_reward(self) -> float:
        """Get sum of intermediate rewards."""
        return sum(self.step_rewards)
    
    def get_discounted_intermediate_reward(self, gamma: float) -> float:
        """Get discounted sum of intermediate rewards."""
        total = 0.0
        for i, r in enumerate(self.step_rewards):
            total += r * (gamma ** i)
        return total


# =============================================================================
# CORRECTED TD(λ) LEARNER
# =============================================================================

class TDLambdaLearner:
    """
    Correct TD(λ) implementation with all fixes.
    
    TD Error: δ = R + γV(s') - V(s)
    Eligibility Trace: e_t(s) = γλe_{t-1}(s) + 1_{s_t=s}  (accumulating)
    Value Update: V(s) ← V(s) + αδe(s)
    
    Key corrections:
    1. δ uses actual V(s'), not V(s)
    2. Traces are accumulating, not replacing
    3. Goal-conditioned values
    4. Terminal state has V(s') = 0
    """
    
    def __init__(self, config: JottyConfig, adaptive_lr: AdaptiveLearningRate = None):
        self.config = config
        self.gamma = config.gamma
        self.lambda_trace = config.lambda_trace
        self.alpha = config.alpha
        
        self.adaptive_lr = adaptive_lr
        
        # Current episode state
        self.traces: Dict[str, float] = {}  # memory_key -> eligibility trace
        self.values_at_access: Dict[str, float] = {}  # memory_key -> V(s) at access
        self.current_goal: str = ""
        self.access_sequence: List[str] = []  # Order of accesses
        
        # Intermediate rewards
        self.intermediate_calc = IntermediateRewardCalculator(config)
    
    def start_episode(self, goal: str):
        """Initialize for new episode."""
        self.traces.clear()
        self.values_at_access.clear()
        self.access_sequence.clear()
        self.current_goal = goal
        self.intermediate_calc.reset()
    
    def update(self, state: Dict[str, Any], action: Dict[str, Any], reward: float, next_state: Dict[str, Any]):
        """
        Simple update method for compatibility with agent runners.
        
        This is a simplified interface for single-step updates.
        For proper TD(λ) learning, use start_episode/record_access/end_episode.
        
        Args:
            state: Current state dict (e.g., {'goal': '...'})
            action: Action taken (e.g., {'output': '...'})
            reward: Reward received (typically 0.0 or 1.0)
            next_state: Next state dict (e.g., {'goal': '...', 'completed': True})
        
        Note: This method does a simple update without full TD(λ) traces.
        For episode-based learning, use the proper episode methods.
        """
        # Extract goal from state
        goal = state.get('goal', 'general')
        
        # If this is a new episode (goal changed), start it
        if goal != self.current_goal:
            self.start_episode(goal)
        
        # For simple updates, we'll just record that learning happened
        # Full TD(λ) updates should use end_episode() with memories
        # This is a compatibility method - actual learning happens in end_episode()
        
        # Store reward for potential use in end_episode
        if reward > 0:
            self.intermediate_calc.step_rewards.append(reward)
        
        # Log that update was called (for debugging)
        logger.debug(f"TDLambdaLearner.update called: goal={goal}, reward={reward}")
        
        # Note: Actual value updates require memories and should use end_episode()
        # This method exists for compatibility but doesn't perform full TD(λ) updates
    
    def record_access(self, memory: MemoryEntry, step_reward: float = 0.0) -> float:
        """
        Record memory access and update trace.
        
        Parameters:
            memory: The accessed memory
            step_reward: Intermediate reward at this step
        
        Returns:
            Current trace value for this memory
        """
        key = memory.key
        
        # Decay all existing traces (γλ decay)
        # A-TEAM FIX: Prune near-zero traces for computational efficiency
        traces_to_prune = []
        for k in self.traces:
            self.traces[k] *= self.gamma * self.lambda_trace
            # Prune traces below threshold (Shannon: negligible information content)
            if self.traces[k] < 1e-8:
                traces_to_prune.append(k)
        
        # Remove pruned traces
        for k in traces_to_prune:
            del self.traces[k]
        
        # Accumulating trace (add 1, don't replace)
        self.traces[key] = self.traces.get(key, 0.0) + 1.0
        
        # Record value at access time
        self.values_at_access[key] = memory.get_value(self.current_goal)
        
        # Track access order
        if key not in self.access_sequence:
            self.access_sequence.append(key)
        
        # Record intermediate reward
        if step_reward != 0:
            self.intermediate_calc.step_rewards.append(step_reward)
        
        return self.traces[key]
    
    def record_access_from_hierarchical_memory(
        self,
        memory: 'HierarchicalMemory',
        domain: str,
        task_type: str,
        content: str,
        level: MemoryLevel,
        step_reward: float = 0.0
    ) -> float:
        """
        Record memory access from HierarchicalMemory using domain/task_type.
        
        This method integrates RL layer with HierarchicalMemory by:
        1. Generating hierarchical key
        2. Finding or creating memory entry
        3. Recording access for TD(λ) learning
        
        Args:
            memory: HierarchicalMemory instance
            domain: Domain identifier
            task_type: Task type
            content: Memory content
            level: Memory level to check
            step_reward: Intermediate reward
        
        Returns:
            Current trace value
        """
        # Generate hierarchical key
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        key = f"{domain}:{task_type}:{content_hash}"
        
        # Find memory entry across all levels (check specified level first, then others)
        memory_entry = None
        levels_to_check = [level] + [l for l in MemoryLevel if l != level]
        
        for check_level in levels_to_check:
            if check_level in memory.memories:
                if key in memory.memories[check_level]:
                    memory_entry = memory.memories[check_level][key]
                    break
        
        # If not found, create it (store in EPISODIC level)
        if not memory_entry:
            memory_entry = memory.store(
                content=content,
                level=MemoryLevel.EPISODIC,
                context={'domain': domain, 'task_type': task_type},
                goal=self.current_goal or 'general',
                domain=domain,
                task_type=task_type
            )
            key = memory_entry.key  # Use the actual key (may have been migrated)
        
        # Record access using existing method
        return self.record_access(memory_entry, step_reward)
    
    def end_episode(self, 
                    final_reward: float,
                    memories: Dict[str, MemoryEntry],
                    goal_hierarchy: Optional[GoalHierarchy] = None) -> List[Tuple[str, float, float]]:
        """
        Perform TD updates at episode end.
        
        Parameters:
            final_reward: Terminal reward (+1 success, -0.5 failure)
            memories: Dictionary of memory_key -> MemoryEntry
            goal_hierarchy: Optional for value generalization
        
        Returns:
            List of (memory_key, old_value, new_value) updates
        """
        # Get adapted learning rate
        if self.adaptive_lr:
            alpha = self.adaptive_lr.get_adapted_alpha()
        else:
            alpha = self.alpha
        
        # Include intermediate rewards
        total_reward = final_reward + self.intermediate_calc.get_discounted_intermediate_reward(self.gamma)
        
        updates = []
        td_errors = []
        
        # For terminal state, V(s') = 0
        # TD error at terminal: δ = R + γ*0 - V(s)
        
        for key, trace in self.traces.items():
            if key not in memories:
                continue
            
            memory = memories[key]
            
            # Get current value for this goal
            old_value = self.values_at_access.get(key, memory.get_value(self.current_goal))
            
            # TD error (terminal state)
            td_error = total_reward - old_value
            
            # Value update: V(s) ← V(s) + αδe(s)
            delta_v = alpha * td_error * trace
            new_value = old_value + delta_v
            
            # Clip to [0, 1]
            new_value = max(0.0, min(1.0, new_value))
            
            # Update memory's goal-conditioned value
            if self.current_goal not in memory.goal_values:
                memory.goal_values[self.current_goal] = GoalValue()
            
            memory.goal_values[self.current_goal].value = new_value
            memory.goal_values[self.current_goal].last_updated = datetime.now()
            memory.goal_values[self.current_goal].access_count += 1
            
            updates.append((key, old_value, new_value))
            td_errors.append(td_error)
        
        # Record TD errors for adaptive learning rate
        if self.adaptive_lr:
            for td_error in td_errors:
                self.adaptive_lr.record_td_error(td_error)
        
        # Optional: Value transfer to related goals
        if goal_hierarchy and self.config.enable_goal_hierarchy:
            self._transfer_values(memories, goal_hierarchy, updates)
        
        return updates
    
    def end_episode_with_hierarchical_memory(
        self,
        memory: 'HierarchicalMemory',
        domain: str,
        goal: str,
        final_reward: float,
        goal_hierarchy: Optional[GoalHierarchy] = None
    ) -> List[Tuple[str, float, float]]:
        """
        Perform TD updates on HierarchicalMemory at episode end.
        
        This method integrates RL layer with HierarchicalMemory by:
        1. Finding memories accessed during episode (by trace keys)
        2. Updating values directly in HierarchicalMemory
        3. Preserving all existing functionality
        
        Args:
            memory: HierarchicalMemory instance
            domain: Domain identifier (for filtering)
            goal: Goal for value updates
            final_reward: Terminal reward
            goal_hierarchy: Optional for value generalization
        
        Returns:
            List of (memory_key, old_value, new_value) updates
        """
        # Get adapted learning rate
        if self.adaptive_lr:
            alpha = self.adaptive_lr.get_adapted_alpha()
        else:
            alpha = self.alpha
        
        # Include intermediate rewards
        total_reward = final_reward + self.intermediate_calc.get_discounted_intermediate_reward(self.gamma)
        
        updates = []
        td_errors = []
        
        # Find memories across all levels by trace keys
        for key, trace in self.traces.items():
            # Find memory entry across all levels
            memory_entry = None
            for level in MemoryLevel:
                if level in memory.memories:
                    if key in memory.memories[level]:
                        memory_entry = memory.memories[level][key]
                        break
            
            if not memory_entry:
                continue
            
            # Get current value for this goal
            old_value = self.values_at_access.get(key, memory_entry.get_value(goal))
            
            # TD error (terminal state)
            td_error = total_reward - old_value
            
            # Value update: V(s) ← V(s) + αδe(s)
            delta_v = alpha * td_error * trace
            new_value = old_value + delta_v
            
            # Clip to [0, 1]
            new_value = max(0.0, min(1.0, new_value))
            
            # Update memory's goal-conditioned value directly in HierarchicalMemory
            if goal not in memory_entry.goal_values:
                memory_entry.goal_values[goal] = GoalValue()
            
            memory_entry.goal_values[goal].value = new_value
            memory_entry.goal_values[goal].last_updated = datetime.now()
            memory_entry.goal_values[goal].access_count += 1
            
            updates.append((key, old_value, new_value))
            td_errors.append(td_error)
        
        # Record TD errors for adaptive learning rate
        if self.adaptive_lr:
            for td_error in td_errors:
                self.adaptive_lr.record_td_error(td_error)
        
        # Optional: Value transfer to related goals
        if goal_hierarchy and self.config.enable_goal_hierarchy:
            # Convert HierarchicalMemory to dict format for _transfer_values
            memories_dict = {}
            for level in MemoryLevel:
                if level in memory.memories:
                    memories_dict.update(memory.memories[level])
            self._transfer_values(memories_dict, goal_hierarchy, updates)
        
        return updates
    
    def _transfer_values(self, 
                          memories: Dict[str, MemoryEntry],
                          goal_hierarchy: GoalHierarchy,
                          updates: List[Tuple[str, float, float]]):
        """Transfer value updates to related goals with discounting."""
        transfer_weight = self.config.goal_transfer_weight * self.config.goal_transfer_discount  # 🔧 STANFORD FIX
        
        # Find related goals
        goal_id = None
        for gid, node in goal_hierarchy.nodes.items():
            if node.goal_text == self.current_goal:
                goal_id = gid
                break
        
        if not goal_id:
            return
        
        related = goal_hierarchy.get_related_goals(goal_id, max_distance=2)
        
        for mem_key, old_v, new_v in updates:
            if mem_key not in memories:
                continue
            
            memory = memories[mem_key]
            delta = new_v - old_v
            
            for related_id, similarity in related:
                related_node = goal_hierarchy.nodes.get(related_id)
                if not related_node:
                    continue
                
                related_goal = related_node.goal_text
                
                # Partial transfer
                transfer_delta = delta * similarity * transfer_weight
                
                if related_goal not in memory.goal_values:
                    memory.goal_values[related_goal] = GoalValue()
                
                current = memory.goal_values[related_goal].value
                memory.goal_values[related_goal].value = max(0, min(1, current + transfer_delta))


# =============================================================================
# REASONING-BASED CREDIT ASSIGNER (Dr. Chen Enhancement)
# =============================================================================

class ReasoningCreditAssigner:
    """
    Enhanced credit assignment using reasoning quality analysis.
    
    Factors:
    1. Decision correctness (counterfactual)
    2. Reasoning quality (how well-reasoned)
    3. Evidence usage (what data was used)
    4. Temporal position (early vs late decisions)
    """
    
    def __init__(self, config: JottyConfig):
        self.config = config
        self.reasoning_weight = config.reasoning_weight
        self.evidence_weight = config.evidence_weight
    
    def analyze_contributions(self,
                               success: bool,
                               architect_results: List[ValidationResult],
                               auditor_results: List[ValidationResult],
                               actor_succeeded: bool,
                               trajectory: List[Dict]) -> Dict[str, AgentContribution]:
        """
        Analyze each agent's contribution with reasoning quality.
        """
        contributions = {}
        
        total_steps = len(trajectory)
        
        # Analyze Architect agents
        for i, result in enumerate(architect_results):
            contrib = self._analyze_single_agent(
                result=result,
                episode_success=success,
                is_architect=True,
                actor_succeeded=actor_succeeded,
                step_position=i / max(1, total_steps)
            )
            contributions[result.agent_name] = contrib
        
        # Analyze Auditor agents
        for i, result in enumerate(auditor_results):
            contrib = self._analyze_single_agent(
                result=result,
                episode_success=success,
                is_architect=False,
                actor_succeeded=actor_succeeded,
                step_position=(len(architect_results) + i) / max(1, total_steps)
            )
            contributions[result.agent_name] = contrib
        
        return contributions
    
    def _analyze_single_agent(self,
                               result: ValidationResult,
                               episode_success: bool,
                               is_architect: bool,
                               actor_succeeded: bool,
                               step_position: float) -> AgentContribution:
        """Analyze a single agent's contribution."""
        
        # Determine decision
        if is_architect:
            decision = "approve" if result.should_proceed else "reject"
        else:
            decision = "approve" if result.is_valid else "reject"
        
        # Was decision correct?
        if is_architect:
            # Architect approve → Actor runs → Check if actor succeeded
            if decision == "approve":
                decision_correct = actor_succeeded
            else:
                # Architect reject → Can't know if it was right
                # Assume correct if episode would have failed
                decision_correct = not episode_success  # Pessimistic
        else:
            # Auditor: approve should align with success
            decision_correct = (decision == "approve") == episode_success
        
        # Counterfactual impact
        # How much would outcome change without this agent?
        if decision_correct:
            counterfactual = result.confidence  # High confidence → high impact
        else:
            counterfactual = -result.confidence
        
        # Reasoning quality
        reasoning_quality = self._assess_reasoning_quality(result)
        
        # Base contribution score
        if decision_correct:
            base_score = 0.5 + 0.5 * result.confidence
        else:
            base_score = -0.5 * result.confidence
        
        # Temporal weight (later decisions more certain)
        temporal_weight = 0.7 + 0.3 * step_position
        
        return AgentContribution(
            agent_name=result.agent_name,
            contribution_score=base_score,
            decision=decision,
            decision_correct=decision_correct,
            counterfactual_impact=abs(counterfactual),
            reasoning_quality=reasoning_quality,
            evidence_used=self._extract_evidence(result),
            tools_used=[tc.get('tool', '') for tc in result.tool_calls],
            decision_timing=step_position,
            temporal_weight=temporal_weight
        )
    
    def _assess_reasoning_quality(self, result: ValidationResult) -> float:
        """
        Assess quality of reasoning in result.
        
        Heuristics:
        - Length of reasoning (longer often better, up to a point)
        - Use of evidence/data
        - Logical connectors
        - Confidence calibration
        """
        reasoning = result.reasoning or ""
        
        score = 0.5  # Base
        
        # Length factor (50-500 chars is good)
        length = len(reasoning)
        if 50 <= length <= 500:
            score += 0.1
        elif length > 500:
            score += 0.05  # Diminishing returns
        
        # A-Team Fix: Replace keyword patterns with structure-based heuristics
        # Reasoning quality indicators WITHOUT keyword matching:
        
        # 1. Has tool calls (used evidence)
        if result.tool_calls and len(result.tool_calls) > 0:
            score += 0.15  # Actually used tools = grounded reasoning
        
        # 2. Reasoning length indicates depth of analysis
        # (longer reasoning = more thorough, up to a point)
        reasoning_depth = min(len(reasoning) / 800, 0.2)
        score += reasoning_depth
        
        # 3. Has structured output (numbered steps, comparisons)
        # Check for digit presence which indicates structured thinking
        digit_count = sum(c.isdigit() for c in reasoning)
        if digit_count > 2:  # Has numbers = likely quantitative analysis
            score += 0.05
        
        # 4. Confidence calibration (extreme confidence often bad)
        # This is domain-agnostic - overconfidence is always suspicious
        if 0.6 <= result.confidence <= 0.9:
            score += 0.1
        elif result.confidence > 0.95 or result.confidence < 0.3:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _extract_evidence(self, result: ValidationResult) -> List[str]:
        """
        Extract evidence cited in reasoning.
        
        A-Team v8.0: NO REGEX! Uses character-by-character parsing.
        """
        evidence = []
        reasoning = result.reasoning or ""
        
        # Extract quoted content WITHOUT regex
        quotes = self._extract_quoted_strings(reasoning)
        evidence.extend(quotes)  # Max 3
        
        # Look for tool results
        for tc in result.tool_calls:
            if 'result' in tc:
                evidence.append(f"Tool:{tc.get('tool', 'unknown')}")
        
        return evidence
    
    def _extract_quoted_strings(self, text: str) -> List[str]:
        """
        Extract strings between double quotes without regex.
        
        A-Team v8.0: Character-by-character parsing for robustness.
        """
        quotes = []
        in_quote = False
        current = []
        
        for char in text:
            if char == '"':
                if in_quote:
                    # End of quote
                    if current:
                        quotes.append(''.join(current))
                    current = []
                in_quote = not in_quote
            elif in_quote:
                current.append(char)
        
        return quotes


# =============================================================================
# EXPLORATION STRATEGY (Enhanced)
# =============================================================================

class AdaptiveExploration:
    """
    Adaptive exploration with UCB and stall detection.
    
    ε-greedy baseline with:
    - UCB bonus for underexplored memories
    - Exploration boost when learning stalls
    - Goal-specific exploration rates
    """
    
    def __init__(self, config: JottyConfig):
        self.config = config
        self.epsilon = config.epsilon_start
        self.ucb_c = config.ucb_coefficient
        
        # Per-goal exploration
        self.goal_epsilons: Dict[str, float] = {}
        self.goal_visit_counts: Dict[str, int] = defaultdict(int)
        
        # Stall detection
        self.recent_values: List[float] = []
        self.stall_boost_active = False
    
    def get_epsilon(self, goal: str, episode: int) -> float:
        """Get current epsilon with all adaptations."""
        
        # Base decay
        decay_progress = min(1.0, episode / self.config.epsilon_decay_episodes)
        base_epsilon = self.config.epsilon_start - decay_progress * (
            self.config.epsilon_start - self.config.epsilon_end
        )
        
        # Goal-specific adjustment
        goal_visits = self.goal_visit_counts[goal]
        if goal_visits < 5:
            # New goal → more exploration
            base_epsilon = min(0.5, base_epsilon * 1.5)
        
        # Stall boost
        if self.stall_boost_active:
            base_epsilon = min(0.5, base_epsilon + self.config.exploration_boost_on_stall)
        
        self.epsilon = base_epsilon
        return self.epsilon
    
    def record_goal_visit(self, goal: str):
        """Record visit to a goal."""
        self.goal_visit_counts[goal] += 1
    
    def record_value_change(self, delta: float):
        """Record value change for stall detection."""
        self.recent_values.append(abs(delta))
        if len(self.recent_values) > self.config.stall_detection_window:
            self.recent_values = self.recent_values[-self.config.stall_detection_window:]
        
        # Check for stall
        if len(self.recent_values) >= 50:
            avg_change = sum(self.recent_values) / 50
            self.stall_boost_active = avg_change < self.config.stall_threshold
    
    def should_explore(self, goal: str, episode: int) -> bool:
        """Decide whether to explore (True) or exploit (False)."""
        import random
        return random.random() < self.get_epsilon(goal, episode)
    
    def get_ucb_score(self, memory: MemoryEntry, goal: str, 
                      total_accesses: int) -> float:
        """Get UCB score for a memory."""
        value = memory.get_value(goal)
        
        if memory.ucb_visits == 0:
            return float('inf')
        
        exploration_bonus = self.ucb_c * math.sqrt(
            math.log(total_accesses + 1) / memory.ucb_visits
        )
        
        return value + exploration_bonus


# =============================================================================
# LEARNING HEALTH MONITOR (Enhanced)
# =============================================================================

class LearningHealthMonitor:
    """
    Monitors learning health with all pathological behavior detection.
    
    Alerts:
    - Reward hacking
    - Distribution shift
    - Conservative collapse
    - Catastrophic forgetting
    - Learning stall
    - Goal drift
    """
    
    def __init__(self, config: JottyConfig):
        self.config = config
        self.metrics = LearningMetrics()
        
        # Tracking
        self.approval_rates: Dict[str, List[float]] = defaultdict(list)
        self.goal_distributions: List[Dict[str, int]] = []
        self.value_snapshots: List[Dict[str, float]] = []
    
    def record_episode(self,
                        success: bool,
                        goal: str,
                        architect_decisions: List[bool],
                        auditor_decisions: List[bool],
                        value_updates: List[Tuple[str, float, float]]) -> List[str]:
        """
        Record episode and check for issues.
        
        Returns list of alert messages.
        """
        alerts = []
        
        # Update metrics
        self.metrics.episode_count += 1
        self.metrics.success_count += 1 if success else 0
        self.metrics.recent_successes.append(success)
        self.metrics.goals_seen.add(goal)
        
        # Track value changes
        for key, old_v, new_v in value_updates:
            self.metrics.value_changes.append(new_v - old_v)
        
        # Check for reward hacking
        if self._detect_reward_hacking():
            alerts.append(f"ALERT[{AlertType.REWARD_HACKING.value}]: Success rate suspiciously high (>{self.config.suspicion_threshold:.0%})")
        
        # Check for conservative collapse
        approval_rate = sum(architect_decisions) / len(architect_decisions) if architect_decisions else 0.5
        if self._detect_conservative_collapse(approval_rate):
            alerts.append(f"ALERT[{AlertType.CONSERVATIVE_COLLAPSE.value}]: Rejection rate too high, may be over-conservative")
        
        # Check for learning stall
        if self._detect_learning_stall():
            alerts.append(f"ALERT[{AlertType.LEARNING_STALL.value}]: Learning appears stalled, values not changing")
        
        # Check for goal drift
        drift = self._detect_goal_drift(goal)
        if drift:
            alerts.append(f"ALERT[{AlertType.GOAL_DRIFT.value}]: Goal distribution shifting: {drift}")
        
        return alerts
    
    def _detect_reward_hacking(self) -> bool:
        """Detect suspiciously high success rates."""
        if len(self.metrics.recent_successes) < 50:
            return False
        
        recent = self.metrics.recent_successes
        rate = sum(recent) / len(recent)
        
        return rate > self.config.suspicion_threshold
    
    def _detect_conservative_collapse(self, current_approval: float) -> bool:
        """Detect if agents are rejecting too much."""
        # Need history
        if self.metrics.episode_count < 20:
            return False
        
        # Very low approval rate
        return current_approval < self.config.min_rejection_rate
    
    def _detect_learning_stall(self) -> bool:
        """Detect if learning has stalled."""
        if len(self.metrics.value_changes) < 100:
            return False
        
        recent = self.metrics.value_changes
        avg_change = sum(abs(v) for v in recent) / len(recent)
        
        return avg_change < self.config.stall_threshold
    
    def _detect_goal_drift(self, current_goal: str) -> Optional[str]:
        """Detect if goal distribution is shifting unusually."""
        # Track goal frequency
        goal_id = current_goal  # Truncate for tracking
        
        if not hasattr(self, 'goal_counts'):
            self.goal_counts: Dict[str, int] = defaultdict(int)
            self.recent_goals: List[str] = []
        
        self.goal_counts[goal_id] += 1
        self.recent_goals.append(goal_id)
        
        if len(self.recent_goals) > 100:
            self.recent_goals = self.recent_goals
        
        # Check if single goal is dominating
        if len(self.recent_goals) >= 50:
            recent_unique = len(set(self.recent_goals))
            if recent_unique == 1:
                return f"Single goal dominating: {goal_id}..."
        
        return None
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of learning health."""
        return {
            "episode_count": self.metrics.episode_count,
            "success_rate": self.metrics.get_success_rate(),
            "learning_velocity": self.metrics.get_learning_velocity(),
            "is_stalled": self.metrics.is_learning_stalled(),
            "unique_goals": len(self.metrics.goals_seen),
            "causal_links": self.metrics.causal_links_discovered
        }


# =============================================================================
# CONTEXT BUDGET MANAGER (Enhanced)
# =============================================================================

class DynamicBudgetManager:
    """
    Dynamic context budget allocation.
    
    Instead of fixed allocation, adapts based on:
    - Query complexity
    - Trajectory length
    - Tool output sizes
    - Available relevant memories
    """
    
    def __init__(self, config: JottyConfig):
        self.config = config
        self.total_budget = config.max_context_tokens
        
        # Base allocations
        self.base_allocations = {
            'system_prompt': config.system_prompt_budget,
            'current_input': config.current_input_budget,
            'trajectory': config.trajectory_budget,
            'tool_output': config.tool_output_budget,
        }
    
    def compute_allocation(self,
                           system_prompt_tokens: int,
                           input_tokens: int,
                           trajectory_tokens: int,
                           tool_output_tokens: int) -> Dict[str, int]:
        """
        Compute dynamic budget allocation.
        
        Returns dict with actual tokens to allocate for each category.
        """
        if not self.config.enable_dynamic_budget:
            # Static allocation
            return {
                'system_prompt': self.base_allocations['system_prompt'],
                'current_input': self.base_allocations['current_input'],
                'trajectory': self.base_allocations['trajectory'],
                'tool_output': self.base_allocations['tool_output'],
                'memory': self.config.memory_budget
            }
        
        # Use actual sizes
        actual_usage = {
            'system_prompt': system_prompt_tokens,
            'current_input': input_tokens,
            'trajectory': trajectory_tokens,
            'tool_output': tool_output_tokens
        }
        
        # Compute remaining for memory
        used = sum(actual_usage.values())
        memory_budget = self.total_budget - used
        
        # Enforce bounds
        memory_budget = max(
            self.config.min_memory_budget,
            min(self.config.max_memory_budget, memory_budget)
        )
        
        # If memory budget would cause overflow, reduce trajectory
        total_with_memory = used + memory_budget
        if total_with_memory > self.total_budget:
            overage = total_with_memory - self.total_budget
            actual_usage['trajectory'] = max(5000, actual_usage['trajectory'] - overage)
        
        actual_usage['memory'] = memory_budget
        return actual_usage
    
    def select_within_budget(self,
                              items: List[MemoryEntry],
                              budget: int,
                              goal: str,
                              max_items: int = 50) -> List[MemoryEntry]:
        """
        Select items within budget - NO TRUNCATION.
        
        Items are included fully or not at all.
        """
        # Sort by value
        sorted_items = sorted(
            items,
            key=lambda m: m.get_value(goal),
            reverse=True
        )
        
        selected = []
        tokens_used = 0
        
        for item in sorted_items:
            if len(selected) >= max_items:
                break
            
            # Check size limit
            if item.token_count > self.config.max_entry_tokens:
                continue  # Skip oversized
            
            if tokens_used + item.token_count <= budget:
                selected.append(item)
                tokens_used += item.token_count
        
        return selected
    
    def get_learned_context(self, memories: Dict[str, MemoryEntry], goal: str = None) -> str:
        """
        Get learned context to inject into prompts.
        
        THIS IS HOW TD(λ) LEARNING MANIFESTS IN LLM AGENTS!
        
        Returns natural language lessons from value updates.
        """
        if not memories:
            return ""
        
        goal = goal or self.current_goal
        if not goal:
            return ""
        
        # Collect memories with significant learned values
        high_value_memories = []
        low_value_memories = []
        improved_memories = []
        
        for key, memory in memories.items():
            if goal not in memory.goal_values:
                continue
            
            goal_val = memory.goal_values[goal]
            value = goal_val.value
            
            # Check if value was updated significantly
            if key in self.values_at_access:
                old_value = self.values_at_access[key]
                improvement = value - old_value
                
                if abs(improvement) > 0.1:  # Significant update
                    improved_memories.append((memory, value, improvement))
            
            if value > 0.7:
                high_value_memories.append((memory, value))
            elif value < 0.3:
                low_value_memories.append((memory, value))
        
        if not (high_value_memories or low_value_memories or improved_memories):
            return ""
        
        context = "# TD(λ) Learned Values:\n"
        
        # High-value lessons
        if high_value_memories:
            context += "\n## High-Value Patterns (Learned from Success):\n"
            for memory, value in sorted(high_value_memories, key=lambda x: x[1], reverse=True)[:5]:
                context += f"- {memory.content[:150]}... (V={value:.3f})\n"
        
        # Low-value lessons (what to avoid)
        if low_value_memories:
            context += "\n## Low-Value Patterns (Learned from Failure):\n"
            for memory, value in sorted(low_value_memories, key=lambda x: x[1])[:5]:
                context += f"- AVOID: {memory.content[:150]}... (V={value:.3f})\n"
        
        # Recently improved
        if improved_memories:
            context += "\n## Recently Updated Understanding:\n"
            for memory, value, improvement in sorted(improved_memories, key=lambda x: abs(x[2]), reverse=True)[:3]:
                direction = "↑" if improvement > 0 else "↓"
                context += f"- {direction} {memory.content[:150]}... (V={value:.3f}, Δ={improvement:+.3f})\n"
        
        # Add eligibility trace info (which memories are most relevant now)
        if self.traces:
            context += "\n## Currently Relevant (High Eligibility):\n"
            sorted_traces = sorted(self.traces.items(), key=lambda x: x[1], reverse=True)[:3]
            for key, trace in sorted_traces:
                if key in memories and trace > 0.5:
                    memory = memories[key]
                    context += f"- {memory.content[:150]}... (trace={trace:.2f})\n"
        
        return context
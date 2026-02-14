"""
TD(λ) Learner and Grouped Value Baseline.

Core temporal-difference learning:
- TDLambdaLearner: Episode-based TD(λ) with eligibility traces + single-step TD(0)
- GroupedValueBaseline: Task-type grouped baselines for variance reduction
"""

import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from datetime import datetime
logger = logging.getLogger(__name__)

from ..foundation.data_structures import (
    SwarmConfig, MemoryEntry, MemoryLevel, GoalValue,
    ValidationResult, AgentContribution, StoredEpisode,
    LearningMetrics, AlertType, GoalHierarchy, CausalLink
)
from ..foundation.configs.learning import LearningConfig as FocusedLearningConfig

if TYPE_CHECKING:
    from ..memory.cortex import SwarmMemory

from .adaptive_components import AdaptiveLearningRate, IntermediateRewardCalculator


def _ensure_swarm_config(config: Any) -> Any:
    """Accept LearningConfig or SwarmConfig, return SwarmConfig."""
    if isinstance(config, FocusedLearningConfig):
        return SwarmConfig.from_configs(learning=config)
    return config




# =============================================================================
# HRPO-STYLE GROUPED LEARNING (DrZero-Inspired)
# =============================================================================

class GroupedValueBaseline:
    """
    HRPO-inspired grouped learning for more efficient value estimation.

    DrZero insight: Instead of learning from each task independently,
    group similar tasks and compute group-level baselines. This:
    1. REDUCES VARIANCE: Group baseline smooths noisy rewards
    2. IMPROVES EFFICIENCY: Learn patterns across task types
    3. ENABLES TRANSFER: Similar tasks share value estimates

    HRPO (Hop-grouped Relative Policy Optimization):
    - Clusters structurally similar questions/tasks
    - Constructs group-level baselines
    - Reduces computational overhead while maintaining performance

    TD Error with baseline: δ = R - baseline + γV(s') - V(s)
    """

    def __init__(self, config: Any = None, ema_alpha: float = 0.1) -> None:
        """
        Initialize grouped baseline tracker.

        Args:
            config: SwarmConfig (optional)
            ema_alpha: Exponential moving average alpha for baseline updates
        """
        self.config = config
        self.ema_alpha = ema_alpha

        # Group baselines by task_type (regular dict — no phantom entries on read)
        self.group_baselines: Dict[str, float] = {}

        # Group statistics for variance tracking
        self.group_samples: Dict[str, List[float]] = {}
        self.group_counts: Dict[str, int] = {}
        self.max_samples_per_group = 100

        # Domain-level baselines (higher abstraction)
        self.domain_baselines: Dict[str, float] = {}

        # Cross-group transfer weights
        self.transfer_matrix: Dict[str, Dict[str, float]] = {}

        logger.info("GroupedValueBaseline initialized (HRPO-inspired)")

    def get_baseline(self, task_type: str, domain: str = None,
                     action_type: str = None) -> float:
        """
        Get baseline value for a task type.

        Uses hierarchical lookup:
        1. Task-type + action-type composite (most specific, e.g. 'research:api')
        2. Task-type specific baseline
        3. Domain-level baseline (if task_type unseen)
        4. Global default (0.5)

        Args:
            task_type: Type of task (e.g., 'aggregation', 'analysis')
            domain: Optional domain (e.g., 'ml', 'data')
            action_type: Optional execution strategy ('api', 'gui', 'deeplink', 'rpa').
                         When provided, enables the system to learn which strategy
                         works best per task type (MAS-Bench hybrid routing).

        Returns:
            Baseline value for TD error computation
        """
        # Most specific: task_type + action_type composite key
        if action_type:
            composite_key = f"{task_type}:{action_type}"
            if self.group_counts.get(composite_key, 0) >= 3:
                return self.group_baselines.get(composite_key, 0.5)

        # If we have enough samples for this task type, use its baseline
        if self.group_counts.get(task_type, 0) >= 3:
            return self.group_baselines.get(task_type, 0.5)

        # Fall back to domain baseline if available
        if domain and self.group_counts.get(f"domain:{domain}", 0) >= 3:
            return self.domain_baselines.get(domain, 0.5)

        # Check for similar task types via transfer
        similar_baseline = self._get_transferred_baseline(task_type)
        if similar_baseline is not None:
            return similar_baseline

        # Default baseline
        return 0.5

    def update_group(self, task_type: str, reward: float, domain: str = None, action_type: str = None) -> Any:
        """
        Update group baseline from new sample.

        Uses exponential moving average for stability:
        baseline_new = (1 - alpha) * baseline_old + alpha * reward

        Args:
            task_type: Type of task
            reward: Observed reward
            domain: Optional domain for hierarchical update
            action_type: Optional execution strategy ('api', 'gui', 'deeplink', 'rpa').
                         Tracks per-task-type strategy effectiveness for MAS-Bench
                         hybrid routing. Updates both composite and base keys.
        """
        # Update task-type baseline (EMA)
        old_baseline = self.group_baselines.get(task_type, 0.5)
        self.group_baselines[task_type] = (
            (1 - self.ema_alpha) * old_baseline + self.ema_alpha * reward
        )

        # Track samples for variance estimation
        if task_type not in self.group_samples:
            self.group_samples[task_type] = []
        self.group_samples[task_type].append(reward)
        if len(self.group_samples[task_type]) > self.max_samples_per_group:
            self.group_samples[task_type] = self.group_samples[task_type][-self.max_samples_per_group:]

        self.group_counts[task_type] = self.group_counts.get(task_type, 0) + 1

        # Update composite key for action_type dimension (MAS-Bench hybrid routing)
        if action_type:
            composite_key = f"{task_type}:{action_type}"
            old_composite = self.group_baselines.get(composite_key, 0.5)
            self.group_baselines[composite_key] = (
                (1 - self.ema_alpha) * old_composite + self.ema_alpha * reward
            )
            if composite_key not in self.group_samples:
                self.group_samples[composite_key] = []
            self.group_samples[composite_key].append(reward)
            if len(self.group_samples[composite_key]) > self.max_samples_per_group:
                self.group_samples[composite_key] = self.group_samples[composite_key][-self.max_samples_per_group:]
            self.group_counts[composite_key] = self.group_counts.get(composite_key, 0) + 1

        # Update domain baseline if provided
        if domain:
            old_domain = self.domain_baselines.get(domain, 0.5)
            self.domain_baselines[domain] = (
                (1 - self.ema_alpha * 0.5) * old_domain + (self.ema_alpha * 0.5) * reward
            )
            domain_key = f"domain:{domain}"
            self.group_counts[domain_key] = self.group_counts.get(domain_key, 0) + 1

        # Update transfer weights between similar task types
        self._update_transfer_weights(task_type, reward)

        logger.debug(
            f"Group baseline updated: {task_type} {old_baseline:.3f} -> {self.group_baselines[task_type]:.3f}"
            + (f" (action_type={action_type})" if action_type else "")
        )

    def get_group_variance(self, task_type: str) -> float:
        """
        Get variance of rewards for a task type.

        Useful for:
        - Confidence estimation
        - Exploration decisions
        - Adaptive learning rate
        """
        samples = self.group_samples.get(task_type, [])
        if len(samples) < 2:
            return 0.25  # Default high variance

        mean = sum(samples) / len(samples)
        variance = sum((s - mean) ** 2 for s in samples) / len(samples)
        return variance

    def _get_transferred_baseline(self, task_type: str) -> Optional[float]:
        """
        Get baseline transferred from similar task types.

        DrZero insight: Similar tasks should have similar baselines.
        """
        if task_type not in self.transfer_matrix:
            return None

        transfer_weights = self.transfer_matrix.get(task_type)
        if not transfer_weights:
            return None

        # Weighted average of similar task baselines
        total_weight = 0.0
        weighted_sum = 0.0

        for similar_type, weight in transfer_weights.items():
            if self.group_counts.get(similar_type, 0) >= 3 and weight > 0.1:
                weighted_sum += weight * self.group_baselines.get(similar_type, 0.5)
                total_weight += weight

        if total_weight > 0.3:  # Require meaningful transfer
            return weighted_sum / total_weight

        return None

    def _update_transfer_weights(self, task_type: str, reward: float) -> Any:
        """
        Update transfer weights based on reward similarity.

        If two task types consistently have similar rewards,
        increase transfer weight between them.
        """
        for other_type, samples in self.group_samples.items():
            if other_type == task_type or len(samples) < 3:
                continue

            # Compare reward distributions
            other_mean = sum(samples) / len(samples)
            current_mean = self.group_baselines[task_type]

            # Similarity based on baseline proximity
            similarity = 1.0 - min(1.0, abs(other_mean - current_mean))

            # Update transfer weight (EMA)
            if task_type not in self.transfer_matrix:
                self.transfer_matrix[task_type] = {}
            old_weight = self.transfer_matrix[task_type].get(other_type, 0.0)
            self.transfer_matrix[task_type][other_type] = (
                0.9 * old_weight + 0.1 * similarity
            )

    def get_best_action_type(self, task_type: str,
                            candidates: List[str] = None) -> Optional[str]:
        """
        Get the best-performing action_type for a task_type.

        Compares composite baselines (task_type:action_type) and returns
        the action_type with the highest learned baseline value.
        Useful for MAS-Bench hybrid routing decisions.

        Args:
            task_type: Type of task (e.g., 'automation', 'research')
            candidates: Action types to consider (default: ['api', 'gui', 'deeplink', 'rpa'])

        Returns:
            Best action_type, or None if insufficient data.
        """
        if candidates is None:
            candidates = ['api', 'gui', 'deeplink', 'rpa']

        best_type = None
        best_value = -1.0
        min_samples = 3

        for action_type in candidates:
            composite_key = f"{task_type}:{action_type}"
            count = self.group_counts.get(composite_key, 0)
            if count >= min_samples:
                value = self.group_baselines.get(composite_key, 0.5)
                if value > best_value:
                    best_value = value
                    best_type = action_type

        return best_type

    def compute_relative_advantage(self, task_type: str, reward: float) -> float:
        """
        Compute advantage relative to group baseline.

        HRPO insight: Using group-relative advantage reduces variance
        and improves learning stability.

        Returns:
            reward - baseline (positive = better than average)
        """
        baseline = self.get_baseline(task_type)
        return reward - baseline

    def get_statistics(self) -> Dict[str, Any]:
        """Get grouped learning statistics."""
        return {
            'num_groups': len([k for k, v in self.group_counts.items() if v > 0 and not k.startswith('domain:')]),
            'num_domains': len([k for k, v in self.group_counts.items() if k.startswith('domain:')]),
            'group_baselines': dict(self.group_baselines),
            'group_counts': dict(self.group_counts),
            'total_samples': sum(len(s) for s in self.group_samples.values()),
        }

    def to_dict(self) -> Dict:
        """Serialize for persistence."""
        return {
            'group_baselines': dict(self.group_baselines),
            'group_counts': dict(self.group_counts),
            'domain_baselines': dict(self.domain_baselines),
            'ema_alpha': self.ema_alpha,
        }

    @classmethod
    def from_dict(cls, data: Dict, config: Any = None) -> 'GroupedValueBaseline':
        """Deserialize from persistence."""
        instance = cls(config, ema_alpha=data.get('ema_alpha', 0.1))
        instance.group_baselines = dict(data.get('group_baselines', {}))
        instance.group_counts = dict(data.get('group_counts', {}))
        instance.domain_baselines = dict(data.get('domain_baselines', {}))
        return instance


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
    
    def __init__(self, config: Any, adaptive_lr: AdaptiveLearningRate = None) -> None:
        self.config = _ensure_swarm_config(config)
        self.gamma = self.config.gamma
        self.lambda_trace = self.config.lambda_trace
        self.alpha = self.config.alpha

        self.adaptive_lr = adaptive_lr

        # Current episode state
        self.traces: Dict[str, float] = {}  # memory_key -> eligibility trace
        self.values_at_access: Dict[str, float] = {}  # memory_key -> V(s) at access
        self.current_goal: str = ""
        self.current_task_type: str = ""  # DrZero: track task type for grouped learning
        self.current_domain: str = ""  # DrZero: track domain for hierarchical baselines
        self.access_sequence: List[str] = []  # Order of accesses

        # Intermediate rewards
        self.intermediate_calc = IntermediateRewardCalculator(config)

        # DrZero HRPO-style grouped learning
        self.grouped_baseline = GroupedValueBaseline(config)

        # =====================================================================
        # PREDICTIVE MAINTENANCE: Tool Reliability Tracking
        # =====================================================================
        # Track success/failure rates for tools (APIs, skills, external services)
        # Used to predict when a tool is degrading and route around it
        # =====================================================================
        self.tool_reliability: Dict[str, Dict[str, Any]] = {}  # tool_name -> stats
        self.tool_failure_threshold = 0.75  # Alert if success rate < 75%
        self.tool_degraded_threshold = 0.85  # Warning if success rate < 85%
        self.min_tool_samples = 5  # Need 5+ samples before making judgements
    
    def start_episode(self, goal: str, task_type: str = "", domain: str = "") -> None:
        """
        Initialize for new episode.

        Args:
            goal: The goal for this episode
            task_type: Task type for HRPO grouped learning (e.g., 'aggregation', 'analysis')
            domain: Domain for hierarchical baselines (e.g., 'ml', 'data')
        """
        self.traces.clear()
        self.values_at_access.clear()
        self.access_sequence.clear()
        self.current_goal = goal
        self.current_task_type = task_type or self._infer_task_type(goal)
        self.current_domain = domain
        self.intermediate_calc.reset()
    
    def update(self, state: Dict[str, Any], action: Dict[str, Any], reward: float, next_state: Dict[str, Any]) -> None:
        """
        Single-step TD(0) update for compatibility with agent runners.

        Performs a real value update using a state key derived from the goal
        and action. This is simpler than the full episode-based TD(λ) path
        but actually learns — the grouped baseline and state values are updated.

        For multi-step episodes with eligibility traces, use
        start_episode/record_access/end_episode instead.

        Args:
            state: Current state dict (must contain 'goal')
            action: Action taken (e.g., {'output': '...', 'type': '...'})
            reward: Reward received (0.0 = failure, 1.0 = success)
            next_state: Next state dict (e.g., {'completed': True})
        """
        goal = state.get('goal', 'general')

        # Start new episode if goal changed
        if goal != self.current_goal:
            self.start_episode(goal)

        # Derive a state key from goal + action type (KISS: no hashing, just concat)
        action_type = action.get('type', action.get('action', 'default'))
        state_key = f"{self.current_task_type}:{action_type}"

        # Get adapted learning rate
        alpha = self.adaptive_lr.get_adapted_alpha() if self.adaptive_lr else self.alpha

        # Get current value estimate for this state-goal pair (uses .get — no phantom entries)
        old_value = self.grouped_baseline.group_baselines.get(state_key, 0.5)

        # TD(0) update: V(s) ← V(s) + α(R - V(s))
        # At terminal step (next_state.completed), V(s') = 0 so δ = R - V(s)
        is_terminal = next_state.get('completed', False)
        if is_terminal:
            td_error = reward - old_value
        else:
            # Non-terminal: δ = R + γV(s') - V(s), but we don't know V(s')
            # so use the group baseline as a proxy for V(s')
            next_value = self.grouped_baseline.get_baseline(self.current_task_type)
            td_error = reward + self.gamma * next_value - old_value

        new_value = old_value + alpha * td_error
        new_value = max(0.0, min(1.0, new_value))

        # Update the state-level baseline (explicit write — no defaultdict auto-create)
        self.grouped_baseline.group_baselines[state_key] = new_value
        self.grouped_baseline.group_counts[state_key] = (
            self.grouped_baseline.group_counts.get(state_key, 0) + 1
        )

        # Also update task-type group baseline (for transfer to related states)
        self.grouped_baseline.update_group(self.current_task_type, reward, self.current_domain)

        # Record intermediate reward for episode-level aggregation
        if reward != 0:
            self.intermediate_calc.step_rewards.append(reward)

        # Adaptive learning rate feedback
        if self.adaptive_lr:
            self.adaptive_lr.record_td_error(td_error)

        logger.debug(
            f"TD(0) update: state={state_key}, V: {old_value:.3f}→{new_value:.3f}, "
            f"δ={td_error:.3f}, α={alpha:.4f}"
        )
    
    def record_access(self, memory: MemoryEntry, step_reward: float = 0.0) -> float:
        """
        Record memory access and update eligibility trace.

        ELIGIBILITY TRACES EXPLAINED:
        Traces track "credit eligibility" for each memory state. When a reward
        arrives, states with higher traces get more credit. This solves the
        temporal credit assignment problem: "which earlier decisions led to this reward?"

        Math: e_t(s) = γλe_{t-1}(s) + 1_{s_t=s}
        - γ (gamma): Discount factor (how much future matters)
        - λ (lambda): Trace decay (how fast credit fades)
        - Accumulating: Each access ADDS 1.0 to the trace (not replace)

        Example: If we access memory A at t=1, then B at t=2, then get reward at t=3:
        - e(A) at t=3 = (γλ)² (decayed twice)
        - e(B) at t=3 = γλ (decayed once)
        - e(C) at t=3 = 0 (never accessed)
        So B gets more credit than A, and C gets none.

        Parameters:
            memory: The accessed memory
            step_reward: Intermediate reward at this step

        Returns:
            Current trace value for this memory
        """
        key = memory.key

        # STEP 1: Decay all existing traces by γλ
        # This is the "forgetting" factor - recent accesses matter more
        # Why γλ? Gamma is standard RL discount, lambda controls trace speed
        # A-TEAM FIX: Prune near-zero traces for computational efficiency
        traces_to_prune = []
        for k in self.traces:
            self.traces[k] *= self.gamma * self.lambda_trace  # Decay by γλ
            # Prune traces below threshold (Shannon: negligible information content)
            # Below 1e-8, the trace contributes < 0.00000001 to updates
            if self.traces[k] < 1e-8:
                traces_to_prune.append(k)

        # Remove pruned traces (keeps dict size bounded)
        for k in traces_to_prune:
            del self.traces[k]

        # STEP 2: Accumulating trace - ADD 1.0 to current memory's trace
        # Why add instead of replace? Accumulating traces give more credit
        # to states visited multiple times in an episode
        # Example: If we access A twice, e(A) = 1 + γλ*1 = 1 + γλ (not just 1)
        self.traces[key] = self.traces.get(key, 0.0) + 1.0

        # STEP 3: Record current value estimate V(s) at access time
        # We need this later to compute TD error: δ = R + γV(s') - V(s)
        self.values_at_access[key] = memory.get_value(self.current_goal)

        # STEP 4: Track access order for debugging/analysis
        if key not in self.access_sequence:
            self.access_sequence.append(key)

        # STEP 5: Record intermediate reward (partial success during episode)
        if step_reward != 0:
            self.intermediate_calc.step_rewards.append(step_reward)

        return self.traces[key]
    
    def record_access_from_hierarchical_memory(
        self,
        memory: 'SwarmMemory',
        domain: str,
        task_type: str,
        content: str,
        level: MemoryLevel,
        step_reward: float = 0.0
    ) -> float:
        """
        Record memory access from SwarmMemory using domain/task_type.
        
        This method integrates RL layer with SwarmMemory by:
        1. Generating hierarchical key
        2. Finding or creating memory entry
        3. Recording access for TD(λ) learning
        
        Args:
            memory: SwarmMemory instance
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
    
    def _infer_task_type(self, goal: str) -> str:
        """
        Infer task type from goal string for HRPO grouping.

        Simple keyword-based inference (can be replaced with LLM-based).
        """
        goal_lower = goal.lower()

        task_type_keywords = {
            'aggregation': ['count', 'sum', 'average', 'total', 'aggregate'],
            'analysis': ['analyze', 'analysis', 'pattern', 'correlation', 'trend'],
            'transformation': ['transform', 'convert', 'normalize', 'merge', 'join'],
            'validation': ['validate', 'check', 'verify', 'test', 'ensure'],
            'filtering': ['filter', 'select', 'where', 'top', 'remove'],
            'planning': ['plan', 'decompose', 'prioritize', 'schedule', 'organize'],
        }

        for task_type, keywords in task_type_keywords.items():
            if any(kw in goal_lower for kw in keywords):
                return task_type

        return 'general'

    def end_episode(self,
                    final_reward: float,
                    memories: Dict[str, MemoryEntry],
                    goal_hierarchy: Optional[GoalHierarchy] = None) -> List[Tuple[str, float, float]]:
        """
        Perform TD updates at episode end with HRPO group-relative baselines.

        DrZero HRPO Enhancement:
        - Uses group baseline for variance reduction
        - TD error: δ = R - baseline + γV(s') - V(s)
        - Updates group baselines for future episodes

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

        # =====================================================================
        # HRPO GROUP BASELINE COMPUTATION (DrZero-inspired variance reduction)
        # =====================================================================
        # PROBLEM: Raw rewards are noisy. Success on task X doesn't tell us if
        # we did well or just got lucky. We need a baseline: "what's normal for
        # tasks like this?"
        #
        # SOLUTION: Group similar tasks together and compute average reward.
        # Then use advantage (reward - average) instead of raw reward.
        #
        # MATH: δ = (R - baseline) + γV(s') - V(s)
        # At terminal state: δ = (R - baseline) - V(s)  [since V(s') = 0]
        #
        # BENEFIT: If baseline = 0.7 and we get R = 0.8, advantage = +0.1
        # (slight improvement). But if baseline = 0.3 and R = 0.8, advantage = +0.5
        # (huge success!). This gives more meaningful learning signal.
        # =====================================================================
        group_baseline = self.grouped_baseline.get_baseline(
            self.current_task_type,
            self.current_domain
        )

        # Compute group-relative reward (HRPO insight: reduces variance)
        # This is the "advantage" - how much better/worse than typical
        relative_reward = total_reward - group_baseline

        updates = []
        td_errors = []

        # =====================================================================
        # TD(λ) UPDATE LOOP - Apply credit to all accessed memories
        # =====================================================================
        # For each memory we accessed during the episode, update its value
        # based on the final reward and its eligibility trace.
        #
        # TERMINAL STATE NOTE: At episode end, there's no next state, so
        # V(s') = 0. This simplifies TD error to: δ = (R - baseline) - V(s)
        # =====================================================================

        for key, trace in self.traces.items():
            if key not in memories:
                continue

            memory = memories[key]

            # Get current value estimate V(s) for this goal
            # This is what we thought this memory was worth when we accessed it
            old_value = self.values_at_access.get(key, memory.get_value(self.current_goal))

            # =====================================================================
            # TD ERROR COMPUTATION (the core of TD learning)
            # =====================================================================
            # TD error δ measures "surprise" - difference between what we expected
            # and what we got.
            #
            # HRPO version: δ = (R - baseline) - V(s)
            # - If δ > 0: We did better than expected → increase V(s)
            # - If δ < 0: We did worse than expected → decrease V(s)
            #
            # Using group baseline reduces noise: we compare to "typical" not absolute
            # =====================================================================
            td_error = relative_reward - old_value

            # =====================================================================
            # VALUE UPDATE (the learning step)
            # =====================================================================
            # V(s) ← V(s) + α * δ * e(s)
            #
            # Breaking it down:
            # - α (alpha): Learning rate - how much to trust this new evidence
            # - δ (td_error): How wrong we were (positive = underestimated)
            # - e(s) (trace): How "eligible" this state is for credit
            #
            # EXAMPLE: If we accessed state A early (trace = 0.3) and state B
            # recently (trace = 0.9), and got reward +1.0, then:
            # - State B gets most credit: ΔV(B) = α * 1.0 * 0.9 = 0.9α
            # - State A gets less: ΔV(A) = α * 1.0 * 0.3 = 0.3α
            # This makes sense: recent decisions matter more!
            # =====================================================================
            delta_v = alpha * td_error * trace
            new_value = old_value + delta_v

            # Clip to [0, 1] probability range (values are success likelihoods)
            new_value = max(0.0, min(1.0, new_value))

            # Update memory's goal-conditioned value (stored in memory itself)
            # Goal-conditioned: same memory can have different values for different goals
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

        # DrZero HRPO: Update group baseline from this episode
        self.grouped_baseline.update_group(
            self.current_task_type,
            total_reward,
            self.current_domain
        )

        # Optional: Value transfer to related goals
        if goal_hierarchy and self.config.enable_goal_hierarchy:
            self._transfer_values(memories, goal_hierarchy, updates)

        return updates
    
    def end_episode_with_hierarchical_memory(
        self,
        memory: 'SwarmMemory',
        domain: str,
        goal: str,
        final_reward: float,
        goal_hierarchy: Optional[GoalHierarchy] = None,
        task_type: str = None
    ) -> List[Tuple[str, float, float]]:
        """
        Perform TD updates on SwarmMemory at episode end with HRPO baselines.

        This method integrates RL layer with SwarmMemory by:
        1. Finding memories accessed during episode (by trace keys)
        2. Updating values directly in SwarmMemory
        3. Using HRPO group-relative baselines for variance reduction
        4. Preserving all existing functionality

        Args:
            memory: SwarmMemory instance
            domain: Domain identifier (for filtering and HRPO)
            goal: Goal for value updates
            final_reward: Terminal reward
            goal_hierarchy: Optional for value generalization
            task_type: Optional task type for HRPO grouping

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

        # DrZero HRPO: Get group baseline
        effective_task_type = task_type or self.current_task_type or self._infer_task_type(goal)
        group_baseline = self.grouped_baseline.get_baseline(effective_task_type, domain)
        relative_reward = total_reward - group_baseline

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

            # HRPO TD error (group-relative)
            td_error = relative_reward - old_value

            # Value update: V(s) ← V(s) + αδe(s)
            delta_v = alpha * td_error * trace
            new_value = old_value + delta_v

            # Clip to [0, 1]
            new_value = max(0.0, min(1.0, new_value))

            # Update memory's goal-conditioned value directly in SwarmMemory
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

        # DrZero HRPO: Update group baseline
        self.grouped_baseline.update_group(effective_task_type, total_reward, domain)

        # Optional: Value transfer to related goals
        if goal_hierarchy and self.config.enable_goal_hierarchy:
            # Convert SwarmMemory to dict format for _transfer_values
            memories_dict = {}
            for level in MemoryLevel:
                if level in memory.memories:
                    memories_dict.update(memory.memories[level])
            self._transfer_values(memories_dict, goal_hierarchy, updates)

        return updates

    def get_grouped_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about HRPO grouped learning."""
        return self.grouped_baseline.get_statistics()
    
    def _transfer_values(self, memories: Dict[str, MemoryEntry], goal_hierarchy: GoalHierarchy, updates: List[Tuple[str, float, float]]) -> Any:
        """Transfer value updates to related goals with discounting."""
        transfer_weight = self.config.goal_transfer_weight * self.config.goal_transfer_discount # STANFORD FIX
        
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

    # =========================================================================
    # PREDICTIVE MAINTENANCE: Tool Reliability Tracking
    # =========================================================================

    def update_tool_reliability(
        self,
        tool_name: str,
        success: bool,
        latency_ms: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Track tool reliability for predictive maintenance.

        PROBLEM: External tools (APIs, services) can degrade over time
        - Search API starts returning 404s (20% failure rate)
        - Database query times out frequently
        - LLM provider has intermittent outages

        SOLUTION: Track success/failure rates + latency for each tool
        When reliability drops below threshold, alert and suggest alternatives

        WHY THIS MATTERS:
        - Proactive routing: Avoid failing tools before users notice
        - Cost savings: Don't waste API calls on broken tools
        - User experience: Graceful degradation instead of hard failures

        EXAMPLE:
        Tool: "web-search-api"
        - Successes: 85, Failures: 15, Success rate: 85%
        - Status: DEGRADED (below 85% threshold)
        - Action: Route to alternative search tool

        Args:
            tool_name: Name of the tool/API/service
            success: Whether the tool call succeeded
            latency_ms: Optional latency in milliseconds

        Returns:
            Dict with 'status', 'success_rate', 'recommendations'
        """
        # Initialize tool stats if first time seeing this tool
        if tool_name not in self.tool_reliability:
            self.tool_reliability[tool_name] = {
                'successes': 0,
                'failures': 0,
                'total': 0,
                'latencies': [],  # Last 100 latencies
                'first_seen': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }

        stats = self.tool_reliability[tool_name]

        # Update counts
        stats['total'] += 1
        if success:
            stats['successes'] += 1
        else:
            stats['failures'] += 1

        # Track latency (keep last 100 samples for moving average)
        if latency_ms is not None:
            stats['latencies'].append(latency_ms)
            if len(stats['latencies']) > 100:
                stats['latencies'].pop(0)  # Remove oldest

        stats['last_updated'] = datetime.now().isoformat()

        # Compute success rate
        success_rate = stats['successes'] / stats['total'] if stats['total'] > 0 else 1.0

        # Determine tool health status
        status = self._assess_tool_health(tool_name, success_rate, stats)

        return status

    def _assess_tool_health(
        self,
        tool_name: str,
        success_rate: float,
        stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess tool health and generate recommendations.

        HEALTH LEVELS:
        - HEALTHY: Success rate >= 85% (no action needed)
        - DEGRADED: 75% <= Success rate < 85% (warning, monitor closely)
        - FAILING: Success rate < 75% (critical, route around immediately)

        Returns status dict with recommendations.
        """
        total = stats['total']

        # Need minimum samples before making judgements
        if total < self.min_tool_samples:
            return {
                'tool': tool_name,
                'status': 'LEARNING',
                'success_rate': success_rate,
                'total_calls': total,
                'message': f'Collecting data ({total}/{self.min_tool_samples} samples)',
                'recommendations': []
            }

        # Calculate average latency
        avg_latency = None
        if stats['latencies']:
            avg_latency = sum(stats['latencies']) / len(stats['latencies'])

        # Assess health
        if success_rate >= self.tool_degraded_threshold:
            status = 'HEALTHY'
            message = f'Tool operating normally ({success_rate:.1%} success)'
            recommendations = []

        elif success_rate >= self.tool_failure_threshold:
            status = 'DEGRADED'
            message = (
                f'Tool showing signs of degradation ({success_rate:.1%} success, '
                f'below {self.tool_degraded_threshold:.0%} threshold)'
            )
            recommendations = [
                'Monitor closely for further degradation',
                'Consider alternative tools if available',
                'Check tool provider status page'
            ]
            logger.warning(f" {message}")

        else:
            status = 'FAILING'
            message = (
                f'Tool is failing frequently ({success_rate:.1%} success, '
                f'below {self.tool_failure_threshold:.0%} threshold)'
            )
            recommendations = [
                '🚨 CRITICAL: Route to alternative tool immediately',
                'Investigate root cause of failures',
                'Alert on-call engineer if production-critical'
            ]
            logger.error(f"🚨 {message}")

        return {
            'tool': tool_name,
            'status': status,
            'success_rate': success_rate,
            'total_calls': total,
            'successes': stats['successes'],
            'failures': stats['failures'],
            'avg_latency_ms': avg_latency,
            'message': message,
            'recommendations': recommendations
        }

    def get_tool_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive health report for all tools.

        Returns:
            Dict with 'healthy', 'degraded', 'failing' tool lists
        """
        healthy = []
        degraded = []
        failing = []

        for tool_name, stats in self.tool_reliability.items():
            if stats['total'] < self.min_tool_samples:
                continue  # Skip tools still learning

            success_rate = stats['successes'] / stats['total']
            status = self._assess_tool_health(tool_name, success_rate, stats)

            if status['status'] == 'HEALTHY':
                healthy.append(status)
            elif status['status'] == 'DEGRADED':
                degraded.append(status)
            elif status['status'] == 'FAILING':
                failing.append(status)

        return {
            'timestamp': datetime.now().isoformat(),
            'total_tools': len(self.tool_reliability),
            'healthy': healthy,
            'degraded': degraded,
            'failing': failing,
            'summary': {
                'healthy_count': len(healthy),
                'degraded_count': len(degraded),
                'failing_count': len(failing)
            }
        }

    def should_avoid_tool(self, tool_name: str) -> bool:
        """
        Check if a tool should be avoided due to poor reliability.

        Use this in routing logic to skip failing tools.

        Example:
            if not td_learner.should_avoid_tool("web-search-api"):
                result = call_web_search()

        Returns:
            True if tool is FAILING and should be avoided, False otherwise
        """
        if tool_name not in self.tool_reliability:
            return False  # Unknown tool = no data = don't avoid

        stats = self.tool_reliability[tool_name]
        if stats['total'] < self.min_tool_samples:
            return False  # Not enough data yet

        success_rate = stats['successes'] / stats['total']
        return success_rate < self.tool_failure_threshold


# =============================================================================
# Q-LEARNING FOR SKILL SELECTION
# =============================================================================

class SkillQTable:
    """Q-value table for skill selection: Q(task_type, skill) → expected reward.

    Used by planners to prefer skills with higher historical success rates
    for a given task type. Updated after each skill execution.

    Usage:
        q = SkillQTable()
        best = q.select("research", ["web-search", "calculator", "pdf-gen"])
        # ... execute skill ...
        q.update("research", "web-search", reward=0.9)
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.15) -> None:
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate (epsilon-greedy)
        self._q: Dict[str, Dict[str, float]] = {}  # task_type → {skill → Q-value}
        self._counts: Dict[str, Dict[str, int]] = {}  # task_type → {skill → count}

    def get_q(self, task_type: str, skill: str) -> float:
        """Get Q-value for a (task_type, skill) pair. Default 0.5 (optimistic)."""
        return self._q.get(task_type, {}).get(skill, 0.5)

    def update(self, task_type: str, skill: str, reward: float) -> float:
        """Update Q-value after skill execution. Returns TD error."""
        if task_type not in self._q:
            self._q[task_type] = {}
            self._counts[task_type] = {}

        old_q = self._q[task_type].get(skill, 0.5)
        td_error = reward - old_q
        new_q = old_q + self.alpha * td_error
        self._q[task_type][skill] = max(0.0, min(1.0, new_q))
        self._counts[task_type][skill] = self._counts[task_type].get(skill, 0) + 1

        logger.debug(f"Q-update: ({task_type}, {skill}): {old_q:.3f} → {new_q:.3f}")
        return td_error

    def select(self, task_type: str, available_skills: List[str]) -> List[str]:
        """Rank skills by Q-value with epsilon-greedy exploration.

        Returns skills sorted by Q-value (highest first). With probability
        epsilon, shuffles to encourage exploration.
        """
        import random
        if not available_skills:
            return []

        if random.random() < self.epsilon:
            # Explore: random shuffle
            shuffled = list(available_skills)
            random.shuffle(shuffled)
            return shuffled

        # Exploit: sort by Q-value descending
        return sorted(available_skills, key=lambda s: self.get_q(task_type, s), reverse=True)

    def get_top_skills(self, task_type: str, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N skills by Q-value for a task type."""
        skills = self._q.get(task_type, {})
        sorted_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)
        return sorted_skills[:n]

    def to_dict(self) -> Dict:
        return {'q': self._q, 'counts': self._counts,
                'alpha': self.alpha, 'gamma': self.gamma, 'epsilon': self.epsilon}

    @classmethod
    def from_dict(cls, data: Dict) -> 'SkillQTable':
        obj = cls(alpha=data.get('alpha', 0.1), gamma=data.get('gamma', 0.9),
                  epsilon=data.get('epsilon', 0.15))
        obj._q = data.get('q', {})
        obj._counts = data.get('counts', {})
        return obj


# =============================================================================
# COMA COUNTERFACTUAL CREDIT
# =============================================================================

class COMACredit:
    """Counterfactual Multi-Agent credit assignment.

    Answers: "How much did each agent/skill contribute to the outcome?"
    by computing: credit(agent) = team_reward - reward_without_agent

    The counterfactual baseline is approximated by the average team reward
    when that agent is excluded from similar past episodes.

    Usage:
        coma = COMACredit()
        coma.record_episode(
            team_reward=0.85,
            agent_contributions={"researcher": 0.4, "analyzer": 0.35, "writer": 0.1}
        )
        credits = coma.get_credits("researcher")  # ~0.15 (team with - team without)
    """

    def __init__(self) -> None:
        # Per-agent history of (team_reward, agent_contribution)
        self._history: Dict[str, List[Tuple[float, float]]] = {}
        # Team reward history when agent was NOT present
        self._counterfactual: Dict[str, List[float]] = {}

    def record_episode(self, team_reward: float, agent_contributions: Dict[str, float]) -> None:
        """Record an episode's reward and per-agent contributions.

        Args:
            team_reward: Total team reward (0-1)
            agent_contributions: {agent_name: contribution_score}
        """
        all_agents = set(agent_contributions.keys())
        for agent, contrib in agent_contributions.items():
            if agent not in self._history:
                self._history[agent] = []
                self._counterfactual[agent] = []
            self._history[agent].append((team_reward, contrib))
            # Keep bounded
            if len(self._history[agent]) > 200:
                self._history[agent] = self._history[agent][-200:]

        # For each agent, record this as a counterfactual for agents NOT in this episode
        for agent in self._history:
            if agent not in all_agents:
                self._counterfactual[agent].append(team_reward)
                if len(self._counterfactual[agent]) > 200:
                    self._counterfactual[agent] = self._counterfactual[agent][-200:]

    def get_credit(self, agent: str) -> float:
        """Get counterfactual credit for an agent.

        credit = avg(team_reward when present) - avg(team_reward when absent)
        Positive = agent helps the team. Negative = agent hurts.
        """
        history = self._history.get(agent, [])
        counterfactual = self._counterfactual.get(agent, [])

        if not history:
            return 0.0

        avg_with = sum(r for r, _ in history) / len(history)
        avg_without = sum(counterfactual) / len(counterfactual) if counterfactual else 0.5

        return avg_with - avg_without

    def get_all_credits(self) -> Dict[str, float]:
        """Get credits for all known agents."""
        return {agent: self.get_credit(agent) for agent in self._history}

    def to_dict(self) -> Dict:
        return {'history': self._history, 'counterfactual': self._counterfactual}

    @classmethod
    def from_dict(cls, data: Dict) -> 'COMACredit':
        obj = cls()
        obj._history = data.get('history', {})
        obj._counterfactual = data.get('counterfactual', {})
        return obj


# =============================================================================
# LEARNED CONTEXT GENERATOR
# =============================================================================

def get_learned_context(
    td_learner: TDLambdaLearner,
    skill_q: Optional[SkillQTable] = None,
    coma: Optional[COMACredit] = None,
    task_type: str = "",
    domain: str = "",
    max_lines: int = 10,
) -> str:
    """Convert learned values into natural language for LLM prompt injection.

    Produces a concise summary of what the system has learned about
    task types, skill effectiveness, and agent contributions.

    Args:
        td_learner: TDLambdaLearner with learned baselines
        skill_q: Optional SkillQTable with Q-values
        coma: Optional COMACredit with agent credits
        task_type: Current task type to focus on
        domain: Current domain
        max_lines: Maximum output lines

    Returns:
        Natural language string suitable for LLM system prompt injection
    """
    lines: List[str] = []

    # 1. Task type baseline info
    stats = td_learner.get_grouped_learning_stats()
    baselines = stats.get('group_baselines', {})
    if task_type and task_type in baselines:
        baseline = baselines[task_type]
        variance = td_learner.grouped_baseline.get_group_variance(task_type)
        confidence = "high" if variance < 0.05 else ("medium" if variance < 0.15 else "low")
        lines.append(
            f"Historical success for '{task_type}' tasks: {baseline:.0%} "
            f"(confidence: {confidence})"
        )

    # 2. Top skills from Q-table
    if skill_q and task_type:
        top_skills = skill_q.get_top_skills(task_type, n=3)
        if top_skills:
            skill_strs = [f"{name} ({q:.0%})" for name, q in top_skills]
            lines.append(f"Best skills for '{task_type}': {', '.join(skill_strs)}")

    # 3. Agent credits from COMA
    if coma:
        credits = coma.get_all_credits()
        if credits:
            top_agents = sorted(credits.items(), key=lambda x: x[1], reverse=True)[:3]
            if any(c > 0.05 for _, c in top_agents):
                agent_strs = [f"{name} (+{credit:.0%})" for name, credit in top_agents if credit > 0.05]
                if agent_strs:
                    lines.append(f"Most effective agents: {', '.join(agent_strs)}")

    # 4. Cross-task transfer insights
    transfer = td_learner.grouped_baseline.transfer_matrix.get(task_type, {})
    similar = [(t, w) for t, w in transfer.items() if w > 0.5]
    if similar:
        similar_strs = [f"{t} ({w:.0%})" for t, w in sorted(similar, key=lambda x: -x[1])[:2]]
        lines.append(f"Similar task types: {', '.join(similar_strs)}")

    if not lines:
        return ""

    return "LEARNED CONTEXT:\n" + "\n".join(f"- {l}" for l in lines[:max_lines])


# =============================================================================
# REASONING-BASED CREDIT ASSIGNER (Dr. Chen Enhancement)
# =============================================================================

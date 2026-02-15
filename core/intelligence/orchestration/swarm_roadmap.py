"""
Jotty v6.1 - Enhanced State Representation
==========================================

A-Team Consensus Implementation:
- AgenticState: Rich state with trajectory, memories, predictions
- DecomposedQFunction: Multi-objective value estimation
- SwarmTaskBoard: Long-horizon task management
- ThoughtLevelCredit: Reasoning step credit assignment
- TrajectoryPredictor: DQN-style next-state prediction
- StateCheckpointer: Full resume support

Dr. Manning: "State approximation is the key to intelligent agents"
Dr. Chen: "CoT steps are the state trajectory"
Dr. Agarwal: "100+ step tasks need Markovian tracking"
Aristotle: "Understanding WHY enables prediction"
Shannon: "Efficient state compression maximizes information"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum, auto
from datetime import datetime
from pathlib import Path
import json
import hashlib
import logging

logger = logging.getLogger(__name__)
_dspy_module = None
def _get_dspy() -> Any:
    global _dspy_module
    if _dspy_module is None:
        import dspy
        _dspy_module = dspy
    return _dspy_module

# REFACTORING PHASE 1.2: Import TaskStatus from canonical location
from Jotty.core.infrastructure.foundation.types import TaskStatus


# =============================================================================
# TASK ITEM - Shared across modules
# =============================================================================

@dataclass
class TodoItem:
    """A single task item with RL metadata (Roadmap task)."""
    id: str
    description: str
    actor: str  # Which actor handles this
    status: str  # pending, in_progress, completed, failed, blocked
    priority: float  # 0-1, learned priority
    estimated_reward: float  # Q-value estimate
    dependencies: List[str] = field(default_factory=list)
    attempts: int = 0
    max_attempts: int = 5
    failure_reasons: List[str] = field(default_factory=list)
    completion_time: Optional[float] = None


# =============================================================================
# AGENTIC STATE - Rich State Representation
# =============================================================================

@dataclass
class TrajectoryStep:
    """Single step in agent trajectory."""
    step_idx: int
    timestamp: datetime
    
    # Action taken
    action_type: str  # 'thought', 'tool_call', 'decision', 'output'
    action_content: str
    
    # Context at this step
    context_summary: str
    activated_memories: List[str]
    
    # Outcome
    observation: str
    reward: float
    
    # Predictions made at this step
    predicted_outcome: Optional[str] = None
    prediction_confidence: float = 0.0
    actual_divergence: float = 0.0  # How wrong was prediction?


@dataclass
class AgenticState:
    """
    Rich state representation for LLM agents.
    
    A-Team Enhancement: Goes beyond (goal, agent_name) to capture
    the full context needed for intelligent policy learning.
    """
    
    # Identity
    state_id: str = ""
    agent_name: str = ""
    episode_id: str = ""
    
    # Task Context
    task_description: str = ""
    task_decomposition: List[str] = field(default_factory=list)
    current_subtask_idx: int = 0
    subtask_completion: Dict[str, float] = field(default_factory=dict)
    
    # Agent Trajectory - THE KEY ENHANCEMENT
    trajectory: List[TrajectoryStep] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    tool_calls: List[Dict] = field(default_factory=list)
    
    # Memory Context
    activated_memories: List[str] = field(default_factory=list)
    memory_relevance_scores: Dict[str, float] = field(default_factory=dict)
    
    # Predictions (DQN-style)
    predicted_next_action: str = ""
    action_confidence: float = 0.5
    predicted_outcome: str = ""
    predicted_reward: float = 0.0
    uncertainty: float = 0.5
    
    # Causal Understanding
    active_causal_chains: List[Tuple[str, str]] = field(default_factory=list)
    intervention_effects: Dict[str, float] = field(default_factory=dict)
    
    # Meta
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        if not self.state_id:
            self.state_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique state ID from content."""
        content = f"{self.agent_name}:{self.task_description}:{len(self.trajectory)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def add_trajectory_step(self, action_type: str, action_content: str, observation: str, reward: float, context_summary: str = '', activated_memories: List[str] = None) -> Any:
        """Add a step to the trajectory."""
        step = TrajectoryStep(
            step_idx=len(self.trajectory),
            timestamp=datetime.now(),
            action_type=action_type,
            action_content=action_content,
            context_summary=context_summary,
            activated_memories=activated_memories or [],
            observation=observation,
            reward=reward
        )
        self.trajectory.append(step)
        self.last_updated = datetime.now()
    
    def add_reasoning_step(self, thought: str) -> None:
        """Add a CoT reasoning step."""
        self.reasoning_trace.append(thought)
        self.last_updated = datetime.now()
    
    def add_tool_call(self, tool_name: str, args: Dict, result: Any, success: bool) -> None:
        """Record a tool call."""
        self.tool_calls.append({
            'tool': tool_name,
            'args': args,
            'result': str(result), # NO TRUNCATION - FULL content
            'success': success,
            'step_idx': len(self.trajectory)
        })
    
    def to_key(self) -> str:
        """Generate state key for Q-table lookup."""
        # Include trajectory summary for richer state representation
        trajectory_summary = f"steps:{len(self.trajectory)}"
        if self.trajectory:
            last_action = self.trajectory[-1].action_type
            trajectory_summary += f":last:{last_action}"
        
        return f"{self.agent_name}|{self.task_description}|{trajectory_summary}"
    
    def to_llm_summary(self) -> str:
        """Generate LLM-friendly state summary for Q-value estimation."""
        summary_parts = [
            f"Agent: {self.agent_name}",
            f"Task: {self.task_description}",
            f"Progress: {self.current_subtask_idx}/{len(self.task_decomposition)} subtasks",
            f"Trajectory: {len(self.trajectory)} steps taken",
        ]
        
        if self.reasoning_trace:
            recent_thoughts = self.reasoning_trace
            summary_parts.append(f"Recent reasoning: {' -> '.join(recent_thoughts)}")
        
        if self.tool_calls:
            recent_tools = [tc['tool'] for tc in self.tool_calls]
            summary_parts.append(f"Recent tools: {', '.join(recent_tools)}")
        
        if self.predicted_outcome:
            summary_parts.append(f"Prediction: {self.predicted_outcome} (conf: {self.action_confidence:.2f})")
        
        return "\n".join(summary_parts)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'state_id': self.state_id,
            'agent_name': self.agent_name,
            'episode_id': self.episode_id,
            'task_description': self.task_description,
            'task_decomposition': self.task_decomposition,
            'current_subtask_idx': self.current_subtask_idx,
            'subtask_completion': self.subtask_completion,
            'trajectory': [
                {
                    'step_idx': s.step_idx,
                    'timestamp': s.timestamp.isoformat(),
                    'action_type': s.action_type,
                    'action_content': s.action_content,
                    'observation': s.observation,
                    'reward': s.reward
                }
                for s in self.trajectory
            ],
            'reasoning_trace': self.reasoning_trace,  # Keep recent
            'tool_calls': self.tool_calls,
            'activated_memories': self.activated_memories,
            'predictions': {
                'next_action': self.predicted_next_action,
                'confidence': self.action_confidence,
                'outcome': self.predicted_outcome,
                'reward': self.predicted_reward,
                'uncertainty': self.uncertainty
            },
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AgenticState':
        """Deserialize from dictionary."""
        state = cls(
            state_id=data.get('state_id', ''),
            agent_name=data.get('agent_name', ''),
            episode_id=data.get('episode_id', ''),
            task_description=data.get('task_description', ''),
            task_decomposition=data.get('task_decomposition', []),
            current_subtask_idx=data.get('current_subtask_idx', 0),
            subtask_completion=data.get('subtask_completion', {})
        )
        
        # Restore predictions
        preds = data.get('predictions', {})
        state.predicted_next_action = preds.get('next_action', '')
        state.action_confidence = preds.get('confidence', 0.5)
        state.predicted_outcome = preds.get('outcome', '')
        state.predicted_reward = preds.get('reward', 0.0)
        state.uncertainty = preds.get('uncertainty', 0.5)
        
        return state


# =============================================================================
# DECOMPOSED Q-FUNCTION
# =============================================================================

class DecomposedQFunction:
    """
    Multi-objective Q-function for agentic systems.
    
    A-Team Enhancement: Instead of single Q-value, decompose into:
    - Q_task: How good is action for task completion?
    - Q_explore: How informative is action for learning?
    - Q_causal: Does action help understand causality?
    - Q_safety: Does action satisfy constraints?
    
    Dr. Manning: "Different objectives require different value estimates"
    """
    
    def __init__(self, config: Dict = None) -> None:
        self.config = config or {}
        
        # Separate Q-tables for each objective
        self.q_task: Dict[Tuple[str, str], float] = {}
        self.q_explore: Dict[Tuple[str, str], float] = {}
        self.q_causal: Dict[Tuple[str, str], float] = {}
        self.q_safety: Dict[Tuple[str, str], float] = {}
        
        # Adaptive weights (can change based on phase)
        self.weights = {
            'task': self.config.get('task_weight', 0.5),
            'explore': self.config.get('explore_weight', 0.2),
            'causal': self.config.get('causal_weight', 0.15),
            'safety': self.config.get('safety_weight', 0.15)
        }
        
        # Default values
        self.default_value = self.config.get('default_value', 0.5)
        
        # Learning rates per objective
        self.alphas = {
            'task': self.config.get('alpha_task', 0.05),
            'explore': self.config.get('alpha_explore', 0.1),
            'causal': self.config.get('alpha_causal', 0.08),
            'safety': self.config.get('alpha_safety', 0.03)
        }
    
    def get_q_value(self, state: AgenticState, action: str, objective: str = None) -> float:
        """Get Q-value for state-action pair."""
        state_key = state.to_key()
        key = (state_key, action)
        
        if objective:
            q_table = getattr(self, f'q_{objective}', self.q_task)
            return q_table.get(key, self.default_value)
        
        # Combined value
        return self.get_combined_value(state, action)
    
    def get_combined_value(self, state: AgenticState, action: str) -> float:
        """Get weighted combination of all Q-values."""
        state_key = state.to_key()
        key = (state_key, action)
        
        q_t = self.q_task.get(key, self.default_value)
        q_e = self.q_explore.get(key, self.default_value)
        q_c = self.q_causal.get(key, self.default_value)
        q_s = self.q_safety.get(key, self.default_value)
        
        return (self.weights['task'] * q_t +
                self.weights['explore'] * q_e +
                self.weights['causal'] * q_c +
                self.weights['safety'] * q_s)
    
    def update(self, state: AgenticState, action: str, reward_decomposition: Dict[str, float], next_state: AgenticState, gamma: float = 0.95) -> Any:
        """
        Update all Q-functions with decomposed rewards.
        
        reward_decomposition should have keys: 'task', 'explore', 'causal', 'safety'
        """
        state_key = state.to_key()
        next_state_key = next_state.to_key()
        key = (state_key, action)
        
        for objective in ['task', 'explore', 'causal', 'safety']:
            q_table = getattr(self, f'q_{objective}')
            alpha = self.alphas[objective]
            
            # Current Q-value
            current_q = q_table.get(key, self.default_value)
            
            # Reward for this objective
            reward = reward_decomposition.get(objective, 0.0)
            
            # Max Q for next state (greedy)
            next_q_values = [
                q_table.get((next_state_key, a), self.default_value)
                for a in self._get_possible_actions(next_state)
            ]
            max_next_q = max(next_q_values) if next_q_values else self.default_value
            
            # TD update
            td_target = reward + gamma * max_next_q
            td_error = td_target - current_q
            new_q = current_q + alpha * td_error
            
            q_table[key] = new_q
    
    def _get_possible_actions(self, state: AgenticState) -> List[str]:
        """Get possible actions for a state (placeholder)."""
        # In practice, this would query the action space
        return ['proceed', 'retry', 'refine', 'escalate']
    
    def adjust_weights(self, phase: str) -> None:
        """Adjust weights based on learning phase."""
        if phase == 'exploration':
            self.weights = {'task': 0.3, 'explore': 0.4, 'causal': 0.2, 'safety': 0.1}
        elif phase == 'exploitation':
            self.weights = {'task': 0.6, 'explore': 0.1, 'causal': 0.15, 'safety': 0.15}
        elif phase == 'safety_critical':
            self.weights = {'task': 0.3, 'explore': 0.1, 'causal': 0.1, 'safety': 0.5}
    
    def get_action_ranking(self, state: AgenticState, actions: List[str]) -> List[Tuple[str, float]]:
        """Rank actions by combined Q-value."""
        rankings = []
        for action in actions:
            value = self.get_combined_value(state, action)
            rankings.append((action, value))
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'q_task': {f"{k[0]}|{k[1]}": v for k, v in self.q_task.items()},
            'q_explore': {f"{k[0]}|{k[1]}": v for k, v in self.q_explore.items()},
            'q_causal': {f"{k[0]}|{k[1]}": v for k, v in self.q_causal.items()},
            'q_safety': {f"{k[0]}|{k[1]}": v for k, v in self.q_safety.items()},
            'weights': self.weights,
            'alphas': self.alphas
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DecomposedQFunction':
        """Deserialize from dictionary."""
        qfunc = cls()
        
        def parse_key(key_str: Any) -> Any:
            parts = key_str.rsplit('|', 1)
            return (parts[0], parts[1]) if len(parts) == 2 else (key_str, '')
        
        qfunc.q_task = {parse_key(k): v for k, v in data.get('q_task', {}).items()}
        qfunc.q_explore = {parse_key(k): v for k, v in data.get('q_explore', {}).items()}
        qfunc.q_causal = {parse_key(k): v for k, v in data.get('q_causal', {}).items()}
        qfunc.q_safety = {parse_key(k): v for k, v in data.get('q_safety', {}).items()}
        qfunc.weights = data.get('weights', qfunc.weights)
        qfunc.alphas = data.get('alphas', qfunc.alphas)
        
        return qfunc


# =============================================================================
# MARKOVIAN Task List - Long Horizon Task Management
# =============================================================================
# TaskStatus enum now imported from core.foundation.types (see import section above)

@dataclass
class SubtaskState:
    """
    State of an individual subtask.
    
    UNIFIED TASK REPRESENTATION - Merges features from TaskItem and SubtaskState
    to provide complete RL-ready task tracking with NO HARDCODED VALUES.
    """
    # Identity
    task_id: str
    description: str
    actor: str = ""  # Which actor/agent executes this task
    
    # Status
    status: TaskStatus = TaskStatus.PENDING
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    
    # Progress
    attempts: int = 0
    max_attempts: int = 3  # Can be overridden per task
    progress: float = 0.0  # 0.0 to 1.0
    
    # RL Attributes (for Q-learning)
    priority: float = 1.0  # Task priority (0-inf, higher = more important)
    estimated_reward: float = 0.5  # Q-value estimate (0-1)
    confidence: float = 0.5  # Confidence in Q-value (0-1)
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: float = 60.0  # seconds (can be learned)
    
    # Intermediary Values (NO HARDCODING - stores runtime metrics)
    intermediary_values: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"llm_calls": 8, "time_elapsed": 4.2, "protection_blocks": 2}
    
    # Predictions (for predictive MARL)
    predicted_next_task: Optional[str] = None
    predicted_duration: Optional[float] = None
    predicted_reward: Optional[float] = None
    
    # Learning History
    failure_reasons: List[str] = field(default_factory=list)
    
    # Outcome
    result: Optional[Dict] = None
    error: Optional[str] = None
    
    def can_start(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_tasks for dep in self.depends_on)
    
    def start(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()
        self.attempts += 1
        logger.info(f"▶ Task {self.task_id} STARTED (attempt {self.attempts}/{self.max_attempts})")
    
    def complete(self, result: Dict = None) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.progress = 1.0
        self.result = result
    
    def fail(self, error: str) -> None:
        """Mark task as failed."""
        if self.attempts >= self.max_attempts:
            self.status = TaskStatus.FAILED
            logger.error(f"Task {self.task_id} FAILED (attempts={self.attempts}, max={self.max_attempts})")
        else:
            self.status = TaskStatus.PENDING  # Retry
            logger.warning(f"Task {self.task_id} will RETRY (attempts={self.attempts}, max={self.max_attempts})")
        self.error = error


@dataclass
class SwarmTaskBoard:
    """
    Long-horizon task management with Markov state tracking.
    
    A-Team Enhancement: Handles 100+ step tasks with:
    - Task hierarchy decomposition
    - Dependency tracking
    - State transition probabilities
    - Progress estimation
    - Checkpoint/resume support
    
    Dr. Agarwal: "Real tasks need structured tracking"
    """
    
    # Identity
    todo_id: str = ""
    root_task: str = ""
    
    # Task Hierarchy
    subtasks: Dict[str, SubtaskState] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    current_task_id: Optional[str] = None
    
    # State Tracking
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    
    # Transition Probabilities (learned)
    transition_probs: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # Predictions
    estimated_remaining_steps: int = 0
    completion_probability: float = 0.5
    risk_factors: List[str] = field(default_factory=list)
    
    # Checkpointing
    checkpoints: List[Dict] = field(default_factory=list)
    last_checkpoint: Optional[datetime] = None
    
    def __post_init__(self) -> None:
        if not self.todo_id:
            self.todo_id = hashlib.md5(
                f"{self.root_task}:{datetime.now().isoformat()}".encode()
            ).hexdigest()
    
    def add_task(self, task_id: str, description: str, actor: str = '', depends_on: List[str] = None, estimated_duration: float = 60.0, priority: float = 1.0, max_attempts: int = 3) -> Any:
        """
        Add a subtask - NO HARDCODING.

        All parameters are configurable at runtime.

        Args:
            max_attempts: Maximum retry attempts (default 3).
                          Set to 1 for natural dependency learning (no retries).
        """
        self.subtasks[task_id] = SubtaskState(
            task_id=task_id,
            description=description,
            actor=actor,
            depends_on=depends_on or [],
            estimated_duration=estimated_duration,
            priority=priority,
            max_attempts=max_attempts
        )
        if task_id not in self.execution_order:
            self.execution_order.append(task_id)
        self._update_blocks()
        self._estimate_remaining()
    
    def _update_blocks(self) -> Any:
        """Update which tasks block which."""
        for task_id, task in self.subtasks.items():
            for dep_id in task.depends_on:
                if dep_id in self.subtasks:
                    self.subtasks[dep_id].blocks.append(task_id)
    
    def get_next_task(self, q_predictor: Any = None, current_state: Any = None, goal: Any = None, epsilon: Any = 0.1) -> Optional[SubtaskState]:
        """
        Get next task OBJECT that can be started.

        CRITICAL FIX: Returns SubtaskState object (not string ID) to match
        SwarmReVal interface expectations (task.actor, task.description, etc.)

        🆕 RL-AWARE SELECTION: If q_predictor is provided, uses ε-greedy Q-value selection
        instead of fixed execution_order. This allows RL to actually improve agent ordering!

        Args:
            q_predictor: Optional Q-learning predictor for value-based selection
            current_state: Current environment state for Q-value prediction
            goal: Current goal/task description
            epsilon: Exploration rate (default 0.1 = 10% random exploration)

        Returns:
            Next task to execute (Q-value based if RL enabled, else fixed order)
        """
        # Get all pending tasks that can start
        available_tasks = [
            task for task_id, task in self.subtasks.items()
            if task.status == TaskStatus.PENDING and task.can_start(self.completed_tasks)
        ]

        # Debug: Show all task statuses
        logger.info(f" Task statuses: {', '.join([f'{t.task_id}={t.status.name}' for t in self.subtasks.values()])}")

        if not available_tasks:
            logger.info(" No available tasks (all completed/failed/blocked)")
            return None

        # If only one task available, return it
        if len(available_tasks) == 1:
            return available_tasks[0]

        # Log Q-predictor availability and parameters
        logger.info(f" [get_next_task] q_predictor={q_predictor is not None}, state={current_state is not None}, goal={goal is not None}, epsilon={epsilon}")
        logger.info(f" [get_next_task] Available tasks: {[t.actor for t in available_tasks]}")

        # RL-AWARE SELECTION: Use Q-values to choose best agent
        if q_predictor and current_state and goal:
            import random

            logger.info(" [get_next_task] Using Q-value-based selection!")

            # ε-greedy: explore with probability epsilon
            rand_value = random.random()
            if rand_value < epsilon:
                # EXPLORE: Random selection
                selected_task = random.choice(available_tasks)
                logger.debug(f"EXPLORE ({rand_value:.2f} < {epsilon}) -> {selected_task.actor}")
                logger.info(f" [get_next_task] EXPLORE mode (rand={rand_value:.3f} < eps={epsilon:.3f}) → selected {selected_task.actor}")
                return selected_task
            else:
                logger.debug(f"EXPLOIT ({rand_value:.2f} >= {epsilon})")
                logger.info(f" [get_next_task] EXPLOIT mode (rand={rand_value:.3f} >= eps={epsilon:.3f})")

                # EXPLOIT: Select task with highest Q-value
                best_task = None
                best_q_value = float('-inf')
                q_values_debug = []

                for task in available_tasks:
                    action = {'actor': task.actor, 'task': task.description}
                    try:
                        q_value, _, _ = q_predictor.predict_q_value(current_state, action, goal)
                        q_values_debug.append(f"{task.actor}={q_value:.3f}" if q_value is not None else f"{task.actor}=None")
                        if q_value is not None and q_value > best_q_value:
                            best_q_value = q_value
                            best_task = task
                    except Exception as e:
                        # If Q-prediction fails, skip this task
                        logger.warning(f" [get_next_task] Q-prediction failed for {task.actor}: {e}")
                        pass

                logger.debug(f"Q-values: {', '.join(q_values_debug)}")
                logger.debug(f"Best task: {best_task.actor if best_task else 'None'} (Q={best_q_value:.3f})")
                logger.info(f" [get_next_task] Q-values: {', '.join(q_values_debug)}")
                logger.info(f" [get_next_task] Best task: {best_task.actor if best_task else 'None'} (Q={best_q_value:.3f})")

                # If we found a task with valid Q-value, use it
                if best_task:
                    return best_task

        # FALLBACK: Use fixed execution order (original behavior)
        logger.info(" [get_next_task] Falling back to fixed execution order")
        for task_id in self.execution_order:
            task = self.subtasks[task_id]
            if task.status == TaskStatus.PENDING and task.can_start(self.completed_tasks):
                return task  # ← Return OBJECT, not task_id

        return None

    def unblock_ready_tasks(self) -> int:
        """
        Unblock tasks whose dependencies have been satisfied.
        
        This enables true "await dependency" semantics:
        - A task can be BLOCKED while waiting for upstream outputs.
        - Once all deps are completed, it becomes PENDING again.
        
        Returns:
            Number of tasks unblocked.
        """
        unblocked = 0
        for task in self.subtasks.values():
            if task.status == TaskStatus.BLOCKED and task.can_start(self.completed_tasks):
                task.status = TaskStatus.PENDING
                unblocked += 1
        return unblocked
    
    def start_task(self, task_id: str) -> None:
        """Start a task."""
        if task_id in self.subtasks:
            self.subtasks[task_id].start()
            self.current_task_id = task_id
    
    def complete_task(self, task_id: str, result: Dict = None) -> None:
        """Mark task as completed."""
        if task_id in self.subtasks:
            self.subtasks[task_id].complete(result)
            self.completed_tasks.add(task_id)
            if self.current_task_id == task_id:
                self.current_task_id = None
            self._estimate_remaining()
            
            # Update transition probabilities
            if len(self.completed_tasks) > 1:
                prev_task = list(self.completed_tasks)[-2]
                key = (prev_task, task_id)
                self.transition_probs[key] = self.transition_probs.get(key, 0) + 1
    
    def fail_task(self, task_id: str, error: str) -> None:
        """Mark task as failed."""
        if task_id in self.subtasks:
            task = self.subtasks[task_id]
            task.fail(error)
            if task.status == TaskStatus.FAILED:
                self.failed_tasks.add(task_id)
            self._estimate_remaining()
    
    def _estimate_remaining(self) -> Any:
        """Estimate remaining steps and completion probability."""
        remaining = [
            t for t in self.subtasks.values()
            if t.status not in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]
        ]
        self.estimated_remaining_steps = len(remaining)
        
        # Simple completion probability estimate
        if len(self.subtasks) > 0:
            completed_ratio = len(self.completed_tasks) / len(self.subtasks)
            failure_ratio = len(self.failed_tasks) / len(self.subtasks)
            self.completion_probability = completed_ratio * (1 - failure_ratio * 2)
            self.completion_probability = max(0, min(1, self.completion_probability))
    
    def checkpoint(self) -> Dict:
        """Create checkpoint of current state."""
        checkpoint = {
            'todo_id': self.todo_id,
            'timestamp': datetime.now().isoformat(),
            'current_task_id': self.current_task_id,
            'completed_tasks': list(self.completed_tasks),
            'failed_tasks': list(self.failed_tasks),
            'subtask_states': {
                tid: {
                    'status': t.status.value,
                    'progress': t.progress,
                    'attempts': t.attempts
                }
                for tid, t in self.subtasks.items()
            }
        }
        self.checkpoints.append(checkpoint)
        self.last_checkpoint = datetime.now()
        return checkpoint
    
    def restore_from_checkpoint(self, checkpoint: Dict) -> None:
        """Restore state from checkpoint."""
        self.current_task_id = checkpoint.get('current_task_id')
        self.completed_tasks = set(checkpoint.get('completed_tasks', []))
        self.failed_tasks = set(checkpoint.get('failed_tasks', []))
        
        for tid, state in checkpoint.get('subtask_states', {}).items():
            if tid in self.subtasks:
                self.subtasks[tid].status = TaskStatus(state['status'])
                self.subtasks[tid].progress = state['progress']
                self.subtasks[tid].attempts = state['attempts']
    
    def get_progress_summary(self) -> str:
        """Get human-readable progress summary."""
        total = len(self.subtasks)
        completed = len(self.completed_tasks)
        failed = len(self.failed_tasks)
        in_progress = sum(1 for t in self.subtasks.values() if t.status == TaskStatus.IN_PROGRESS)
        
        return (
            f"Task List Progress: {completed}/{total} completed "
            f"({in_progress} in progress, {failed} failed)\n"
            f"Current: {self.current_task_id or 'None'}\n"
            f"Estimated remaining: {self.estimated_remaining_steps} steps\n"
            f"Completion probability: {self.completion_probability:.1%}"
        )
    
    def should_replan(
        self,
        elapsed_time: float,
        global_deadline: float = 300.0,  # 5 min default
        replan_interval: float = 60.0,   # Check every 1 min
        success_threshold: float = 0.7   # Replan if progress < 70%
    ) -> Tuple[bool, str]:
        """
         A-TEAM ENHANCEMENT: Periodic success-based replanning + global deadlines.
        
        Returns:
            (should_replan: bool, reason: str)
        """
        total = len(self.subtasks)
        if total == 0:
            return False, "No tasks to replan"
        
        completed = len(self.completed_tasks)
        failed = len(self.failed_tasks)
        progress = completed / total
        
        # 1. GLOBAL DEADLINE CHECK
        time_remaining = global_deadline - elapsed_time
        if time_remaining <= 0:
            return True, f"DEADLINE_EXCEEDED: elapsed={elapsed_time:.1f}s > deadline={global_deadline:.1f}s"
        
        # 2. PROGRESS VS TIME CHECK
        time_ratio = elapsed_time / global_deadline
        if time_ratio > 0.5 and progress < time_ratio * success_threshold:
            return True, f"BEHIND_SCHEDULE: progress={progress:.1%} < expected={time_ratio * success_threshold:.1%}"
        
        # 3. HIGH FAILURE RATE CHECK
        if total > 0 and failed / total > 0.3:
            return True, f"HIGH_FAILURE_RATE: {failed}/{total} = {failed/total:.1%} failed"
        
        # 4. STUCK CHECK (tasks in_progress too long)
        stuck_tasks = [
            t for t in self.subtasks.values()
            if t.status == TaskStatus.IN_PROGRESS and t.attempts > 3
        ]
        if stuck_tasks:
            return True, f"STUCK_TASKS: {len(stuck_tasks)} tasks with >3 attempts"
        
        return False, "On track"
    
    def replan(self, observation: str = "") -> List[str]:
        """
         A-TEAM ENHANCEMENT: Trigger replanning.
        
        Actions:
        1. Skip blocked tasks with failed dependencies
        2. Reprioritize based on remaining time
        3. Add emergency subtasks if needed
        
        Returns: List of actions taken
        """
        actions = []
        
        # 1. Skip tasks blocked by failures
        for task_id, task in self.subtasks.items():
            if task.status in [TaskStatus.PENDING, TaskStatus.BLOCKED]:
                # Check if any dependency failed
                failed_deps = [dep for dep in task.depends_on if dep in self.failed_tasks]
                if failed_deps:
                    task.status = TaskStatus.SKIPPED
                    actions.append(f"SKIP:{task_id} (deps failed: {failed_deps})")
        
        # 2. Reprioritize remaining tasks
        pending = [
            t for t in self.subtasks.values()
            if t.status in [TaskStatus.PENDING, TaskStatus.BLOCKED]
        ]
        if pending:
            # Sort by: priority * (1 / (1 + len(depends_on)))
            pending.sort(
                key=lambda t: t.priority * (1.0 / (1.0 + len(t.depends_on))),
                reverse=True
            )
            new_order = [t.task_id for t in pending]
            actions.append(f"REPRIORITIZE: {new_order[:3]}...")
        
        # 3. Add observation to risk factors
        if observation:
            self.risk_factors.append(f"[REPLAN] {observation}")
            actions.append(f"RISK_NOTED: {observation[:50]}")
        
        self._estimate_remaining()
        
        return actions
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'todo_id': self.todo_id,
            'root_task': self.root_task,
            'subtasks': {
                tid: {
                    'description': t.description,
                    'status': t.status.value,
                    'depends_on': t.depends_on,
                    'progress': t.progress,
                    'attempts': t.attempts
                }
                for tid, t in self.subtasks.items()
            },
            'execution_order': self.execution_order,
            'current_task_id': self.current_task_id,
            'completed_tasks': list(self.completed_tasks),
            'failed_tasks': list(self.failed_tasks),
            'checkpoints': self.checkpoints  # Keep last 5
        }
    
    # =========================================================================
    # NEW METHODS - Complete SwarmReVal Integration (NO HARDCODING)
    # =========================================================================
    
    def get_state_summary(self) -> str:
        """
        Get comprehensive state summary for context/display.
        
        NO HARDCODING: Dynamically generates summary from current state.
        Used by SwarmReVal for context building and display.
        """
        pending = [t for t in self.subtasks.values() if t.status == TaskStatus.PENDING]
        in_progress = [t for t in self.subtasks.values() if t.status == TaskStatus.IN_PROGRESS]
        
        summary = f"### Task List State\n"
        summary += f"**Root Task:** {self.root_task}\n"
        summary += f"**Progress:** {len(self.completed_tasks)}/{len(self.subtasks)} completed\n\n"
        
        if in_progress:
            summary += "#### In Progress\n"
            for task in in_progress:
                q_val = task.estimated_reward
                priority = task.priority
                summary += f"- {task.description} (Actor: {task.actor}, Q={q_val:.2f}, P={priority:.1f}, Attempt #{task.attempts+1})\n"
            summary += "\n"
        
        if pending:
            summary += "#### ⏳ Next Up\n"
            # Sort by priority * Q-value (NO HARDCODED THRESHOLD)
            sorted_pending = sorted(pending, key=lambda t: t.priority * t.estimated_reward, reverse=True)
            for task in sorted_pending:  # Top 3
                score = task.priority * task.estimated_reward
                summary += f"- {task.description} (Actor: {task.actor}, Score={score:.2f})\n"
            summary += "\n"
        
        if self.failed_tasks:
            summary += "#### Failed (Need Retry/Exploration)\n"
            for task_id in list(self.failed_tasks):  # Show 2
                task = self.subtasks[task_id]
                reason = task.failure_reasons[-1] if task.failure_reasons else "Unknown"
                summary += f"- {task.description} - {reason}\n"
            summary += "\n"
        
        # Add progress statistics (NO HARDCODED THRESHOLDS)
        summary += f"**Completion Probability:** {self.completion_probability:.1%}\n"
        summary += f"**Estimated Remaining:** {self.estimated_remaining_steps} tasks\n"
        
        return summary
    
    def update_q_value(self, task_id: str, q_value: float, confidence: float) -> None:
        """
        Update Q-value for a task.
        
        NO HARDCODING: Values are learned/predicted, not hardcoded.
        """
        if task_id in self.subtasks:
            self.subtasks[task_id].estimated_reward = max(0.0, min(1.0, q_value))
            self.subtasks[task_id].confidence = max(0.0, min(1.0, confidence))
    
    def record_intermediary_values(self, task_id: str, values: Dict[str, Any]) -> None:
        """
        Record intermediary runtime values (LLM calls, time, etc).
        
        NO HARDCODING: Stores any runtime metrics dynamically.
        Perfect for tracking performance without hardcoded fields.
        """
        if task_id in self.subtasks:
            self.subtasks[task_id].intermediary_values.update(values)
    
    def predict_next(self, task_id: str, next_task_id: Optional[str] = None, duration: Optional[float] = None, reward: Optional[float] = None) -> Any:
        """
        Record predictions for trajectory planning.
        
        NO HARDCODING: Predictions come from LLM or learned models.
        """
        if task_id in self.subtasks:
            self.subtasks[task_id].predicted_next_task = next_task_id
            if duration is not None:
                self.subtasks[task_id].predicted_duration = duration
            if reward is not None:
                self.subtasks[task_id].predicted_reward = reward
    
    def get_task_by_id(self, task_id: str) -> Optional[SubtaskState]:
        """Get task object by ID."""
        return self.subtasks.get(task_id)
    
    # =========================================================================
    # Compatibility Properties (for backward compatibility with old code)
    # =========================================================================
    
    @property
    def items(self) -> Dict[str, SubtaskState]:
        """Alias for .subtasks (backward compatibility)."""
        return self.subtasks
    
    @property
    def completed(self) -> Set[str]:
        """Alias for .completed_tasks (backward compatibility)."""
        return self.completed_tasks


# =============================================================================
# THOUGHT-LEVEL CREDIT ASSIGNMENT
# =============================================================================

class ThoughtLevelCredit:
    """
    Assign credit to individual reasoning steps.
    
    A-Team Enhancement: Goes beyond agent-level credit to
    attribute success/failure to specific thoughts and tool calls.
    
    Dr. Chen: "CoT steps should get individual credit"
    """
    
    def __init__(self, config: Dict = None) -> None:
        self.config = config or {}
        self.temporal_weight = self.config.get('temporal_weight', 0.3)
        self.tool_weight = self.config.get('tool_weight', 0.4)
        self.decision_weight = self.config.get('decision_weight', 0.3)
        # STRICT POLICY: no fuzzy/keyword matching. If an LM is provided, use it.
        self.lm = self.config.get('lm')
    
    def assign_credit(self,
                      reasoning_trace: List[str],
                      tool_calls: List[Dict],
                      outcome: float,
                      trajectory: List[TrajectoryStep] = None) -> Dict[int, float]:
        """
        Assign credit to each step in reasoning trace.
        
        Returns: Dict mapping step index to credit value
        """
        if not reasoning_trace:
            return {}
        
        credits = {}
        n_steps = len(reasoning_trace)
        
        # 1. Temporal credit (later steps get more credit)
        for i, thought in enumerate(reasoning_trace):
            temporal_factor = (i + 1) / n_steps
            credits[i] = outcome * temporal_factor * self.temporal_weight
        
        # 2. Tool-linked credit
        tool_step_outcomes = self._compute_tool_credits(reasoning_trace, tool_calls)
        for step_idx, tool_credit in tool_step_outcomes.items():
            credits[step_idx] = credits.get(step_idx, 0) + tool_credit * self.tool_weight
        
        # 3. Decision point credit
        decision_steps = self._identify_decision_steps(reasoning_trace)
        for step_idx in decision_steps:
            credits[step_idx] = credits.get(step_idx, 0) + outcome * self.decision_weight / max(len(decision_steps), 1)
        
        # Normalize to sum to outcome
        total = sum(credits.values())
        if total > 0:
            credits = {k: v / total * abs(outcome) for k, v in credits.items()}
        
        return credits
    
    def _compute_tool_credits(self, 
                              reasoning_trace: List[str],
                              tool_calls: List[Dict]) -> Dict[int, float]:
        """Compute credit for tool-linked reasoning steps."""
        tool_credits = {}
        
        for tool_call in tool_calls:
            tool_name = tool_call.get('tool', '')
            tool_success = 1.0 if tool_call.get('success', False) else -0.5
            
            # Find which reasoning step led to this tool call
            linked_idx = self._find_linked_thought(reasoning_trace, tool_name)
            if linked_idx is not None:
                tool_credits[linked_idx] = tool_credits.get(linked_idx, 0) + tool_success
        
        return tool_credits
    
    def _find_linked_thought(self, reasoning_trace: List[str], tool_name: str) -> Optional[int]:
        """
        Find which reasoning step led to a tool call.
        
        STRICT POLICY: no regex/fuzzy/keyword heuristics. If no LM is available,
        return None (no linkage).
        """
        if not reasoning_trace or not tool_name:
            return None

        if not self.lm:
            return None

        try:
            import dspy

            class ToolLinkSignature(dspy.Signature):
                reasoning_steps = dspy.InputField(desc="JSON list of reasoning steps in order")
                tool_name = dspy.InputField(desc="Exact tool name that was called")
                output = dspy.OutputField(
                    desc=(
                        "JSON object: {\"linked_index\": int | null, \"reason\": str}. "
                        "linked_index must be an integer index into reasoning_steps, or null if not linkable."
                    )
                )

            linker = dspy.ChainOfThought(ToolLinkSignature)
            with dspy.context(lm=self.lm):
                res = linker(
                    reasoning_steps=json.dumps(reasoning_trace),
                    tool_name=tool_name,
                )

            parsed = json.loads(res.output)
            idx = parsed.get("linked_index")
            if isinstance(idx, int) and 0 <= idx < len(reasoning_trace):
                return idx
            return None
        except Exception:
            return None
    
    def _identify_decision_steps(self, reasoning_trace: List[str]) -> List[int]:
        """
        Identify steps that are decision points.
        
        A-Team Fix: Replace keyword matching with structure-based identification:
        - Steps with output references (action was taken)
        - Final steps (typically conclusions)
        - Steps with high confidence assertions
        """
        decision_steps = []
        
        for i, thought in enumerate(reasoning_trace):
            is_decision = False
            
            # Heuristic 1: Step is in final third of reasoning (likely conclusion)
            if len(reasoning_trace) > 2 and i >= len(reasoning_trace) * 0.7:
                is_decision = True
            
            # Heuristic 2: Short, definitive statements (decisions are often concise)
            # Very long statements are usually analysis, not decisions
            if 20 < len(thought) < 150:
                is_decision = True
            
            # Heuristic 3: Step immediately follows a tool result
            # (Tool result → decision about what to do with it)
            if i > 0:
                prev_thought = reasoning_trace[i - 1]
                if len(prev_thought) > len(thought) * 2:
                    # Previous was detailed analysis, this might be the decision
                    is_decision = True
            
            if is_decision:
                decision_steps.append(i)
        
        return decision_steps
    
    def get_step_value_summary(self, credits: Dict[int, float], reasoning_trace: List[str]) -> str:
        """Generate summary of credit assignment."""
        if not credits:
            return "No credits assigned"
        
        lines = ["Step Credit Assignment:"]
        for idx, credit in sorted(credits.items()):
            thought_preview = reasoning_trace[idx] if idx < len(reasoning_trace) else "?"
            lines.append(f"  Step {idx}: {credit:.3f} - '{thought_preview}...'")
        
        return "\n".join(lines)


# =============================================================================
# STATE CHECKPOINTER
# =============================================================================

class StateCheckpointer:
    """
    Full state checkpointing with resume support.
    
    A-Team Enhancement: Save complete state at decision points,
    enable resume from any checkpoint.
    """
    
    def __init__(self, checkpoint_dir: str = 'agent_generated/checkpoints') -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self,
                        state: AgenticState,
                        q_function: DecomposedQFunction,
                        todo: SwarmTaskBoard,
                        episode_id: str) -> str:
        """Save complete checkpoint."""
        checkpoint_id = f"{episode_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        checkpoint = {
            'checkpoint_id': checkpoint_id,
            'episode_id': episode_id,
            'timestamp': datetime.now().isoformat(),
            'state': state.to_dict(),
            'q_function': q_function.to_dict(),
            'todo': todo.to_dict()
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: str) -> Tuple[AgenticState, DecomposedQFunction, SwarmTaskBoard]:
        """Load checkpoint and restore state."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")
        
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        state = AgenticState.from_dict(checkpoint['state'])
        q_function = DecomposedQFunction.from_dict(checkpoint['q_function'])
        
        todo = SwarmTaskBoard(
            root_task=checkpoint['todo'].get('root_task', ''),
            todo_id=checkpoint['todo'].get('todo_id', '')
        )
        # Restore todo state...
        
        return state, q_function, todo
    
    def list_checkpoints(self, episode_id: str = None) -> List[Dict]:
        """List available checkpoints."""
        checkpoints = []
        for cp_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(cp_file, 'r') as f:
                    data = json.load(f)
                if episode_id is None or data.get('episode_id') == episode_id:
                    checkpoints.append({
                        'checkpoint_id': data['checkpoint_id'],
                        'episode_id': data['episode_id'],
                        'timestamp': data['timestamp']
                    })
            except (IOError, OSError, json.JSONDecodeError, KeyError) as e:
                logger.debug(f"Could not load checkpoint from {cp_file}: {e}")
                pass
        
        return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)
    
    def get_latest_checkpoint(self, episode_id: str = None) -> Optional[str]:
        """Get ID of most recent checkpoint."""
        checkpoints = self.list_checkpoints(episode_id)
        return checkpoints[0]['checkpoint_id'] if checkpoints else None


# =============================================================================
# TRAJECTORY PREDICTOR
# =============================================================================

_TrajectoryPredictorSignature = None
def _get_trajectory_predictor_signature() -> Any:
    global _TrajectoryPredictorSignature
    if _TrajectoryPredictorSignature is None:
        dspy = _get_dspy()
        class TrajectoryPredictorSignature(dspy.Signature):
            """Predict next action and outcome given current state."""
            state_summary: str = dspy.InputField(desc="Current state summary")
            trajectory_history: str = dspy.InputField(desc="Recent trajectory steps")
            available_actions: str = dspy.InputField(desc="Possible actions")
            reasoning: str = dspy.OutputField(desc="Analysis of likely next steps")
            predicted_action: str = dspy.OutputField(desc="Most likely next action")
            confidence: float = dspy.OutputField(desc="Confidence 0.0-1.0")
            predicted_outcome: str = dspy.OutputField(desc="Expected outcome")
            uncertainty_factors: str = dspy.OutputField(desc="What could go wrong")
        _TrajectoryPredictorSignature = TrajectoryPredictorSignature
    return _TrajectoryPredictorSignature


class TrajectoryPredictor:
    """
    DQN-style trajectory prediction.
    
    A-Team Enhancement: Predict next states and outcomes
    to enable look-ahead planning.
    """
    
    def __init__(self) -> None:
        self.predictor = _get_dspy().ChainOfThought(_get_trajectory_predictor_signature())
        self.prediction_history: List[Dict] = []
    
    def predict(self, 
                state: AgenticState,
                available_actions: List[str]) -> Dict:
        """Predict next action and outcome."""
        try:
            result = self.predictor(
                state_summary=state.to_llm_summary(),
                trajectory_history=self._format_trajectory(state.trajectory),
                available_actions=", ".join(available_actions)
            )
            
            prediction = {
                'predicted_action': result.predicted_action,
                'confidence': float(result.confidence),
                'predicted_outcome': result.predicted_outcome,
                'uncertainty_factors': result.uncertainty_factors,
                'reasoning': result.reasoning
            }
            
            self.prediction_history.append({
                'timestamp': datetime.now().isoformat(),
                'state_key': state.to_key(),
                'prediction': prediction
            })
            
            return prediction
            
        except Exception as e:
            return {
                'predicted_action': available_actions[0] if available_actions else 'unknown',
                'confidence': 0.3,
                'predicted_outcome': 'uncertain',
                'uncertainty_factors': str(e),
                'reasoning': 'Prediction failed'
            }
    
    def _format_trajectory(self, trajectory: List[TrajectoryStep]) -> str:
        """Format trajectory for LLM."""
        if not trajectory:
            return "No trajectory yet"
        
        lines = []
        for step in trajectory:
            lines.append(f"Step {step.step_idx}: {step.action_type} - {step.action_content}")
            if step.reward != 0:
                lines.append(f"  Reward: {step.reward:.2f}")
        
        return "\n".join(lines)
    
    def evaluate_prediction_accuracy(self, 
                                     prediction: Dict,
                                     actual_action: str,
                                     actual_outcome: str) -> float:
        """Evaluate how accurate a prediction was."""
        action_match = 1.0 if prediction['predicted_action'] == actual_action else 0.0
        
        # Simple outcome similarity
        pred_outcome = prediction['predicted_outcome'].lower()
        actual_lower = actual_outcome.lower()
        outcome_match = 0.5  # Default partial
        if 'success' in pred_outcome and 'success' in actual_lower:
            outcome_match = 1.0
        elif 'fail' in pred_outcome and 'fail' in actual_lower:
            outcome_match = 1.0
        elif 'success' in pred_outcome and 'fail' in actual_lower:
            outcome_match = 0.0
        elif 'fail' in pred_outcome and 'success' in actual_lower:
            outcome_match = 0.0
        
        return (action_match + outcome_match) / 2


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'AgenticState',
    'TrajectoryStep',
    'DecomposedQFunction',
    'SwarmTaskBoard',
    'SubtaskState',
    'TaskStatus',
    'ThoughtLevelCredit',
    'StateCheckpointer',
    'TrajectoryPredictor'
]


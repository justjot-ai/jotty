"""
Curriculum Generator (DrZero + Agent0 Inspired)
=================================================

Self-curriculum generation for agent training:
- SyntheticTask: Self-generated task for agent training
- CurriculumGenerator: DrZero + Agent0 inspired curriculum system

Generates progressively harder tasks to train agents without external data.
Includes tool-awareness (Agent0) and memory integration.

Extracted from swarm_intelligence.py for modularity.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .swarm_data_structures import AgentProfile

logger = logging.getLogger(__name__)


@dataclass
class SyntheticTask:
    """A self-generated task for agent training."""
    task_id: str
    task_type: str
    description: str
    difficulty: float  # 0.0 (easy) to 1.0 (hard)
    target_agent: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class CurriculumGenerator:
    """
    DrZero + Agent0 inspired self-curriculum generator.

    Generates progressively harder tasks to train agents without external data.

    Key concepts from DrZero:
    1. PROPOSER ROLE: Generate tasks slightly harder than current ability
    2. DIFFICULTY SCALING: Tasks adapt to agent performance
    3. WEAKNESS TARGETING: Focus on low-success task types
    4. DIVERSITY: Ensure coverage of all task types

    Agent0 enhancements (arXiv:2511.16043):
    5. TOOL-AWARE TASKS: Tasks designed for tool usage
    6. MEMORY-INFORMED: Query memory for weakness patterns
    7. EXECUTOR FEEDBACK: Closed-loop curriculum adaptation

    This enables AUTONOMOUS SKILL IMPROVEMENT without user intervention.
    """

    def __init__(self, config=None, state_manager=None, memory_system=None):
        self.config = config

        # Agent0: Connect to existing infrastructure (DRY - don't duplicate)
        self._state_manager = state_manager  # SwarmStateManager for tool stats
        self._memory_system = memory_system  # HierarchicalMemory for context

        # Task type templates (domain-agnostic)
        self.task_templates: Dict[str, List[str]] = {
            'aggregation': [
                "Count items matching criteria: {criteria}",
                "Sum values where: {condition}",
                "Calculate average of: {field}",
            ],
            'analysis': [
                "Analyze patterns in: {domain}",
                "Find correlations between: {field_a} and {field_b}",
                "Identify anomalies in: {dataset}",
            ],
            'transformation': [
                "Transform data from {format_a} to {format_b}",
                "Normalize values in: {field}",
                "Merge datasets: {dataset_a} and {dataset_b}",
            ],
            'validation': [
                "Validate data quality for: {field}",
                "Check constraints: {constraints}",
                "Verify consistency between: {source_a} and {source_b}",
            ],
            'filtering': [
                "Filter records where: {condition}",
                "Select top {n} by: {criteria}",
                "Remove duplicates from: {dataset}",
            ],
            'planning': [
                "Plan execution steps for: {goal}",
                "Decompose task: {complex_task}",
                "Prioritize items: {items}",
            ],
        }

        # Difficulty progression tracking
        self.difficulty_by_type: Dict[str, float] = defaultdict(lambda: 0.3)  # Start at 30%

        # Task history for diversity
        self.generated_tasks: List[SyntheticTask] = []
        self.max_history = 100

        # Curriculum statistics
        self.total_generated = 0
        self.tasks_by_difficulty: Dict[str, int] = defaultdict(int)

        # Agent0: Tool-aware task templates (uses existing tools, doesn't duplicate)
        self.tool_task_templates: Dict[str, Dict[str, Any]] = {
            'search_analyze': {
                'description': "Search for {topic} and analyze the results",
                'tools_hint': ['search', 'web_search', 'grep'],
                'complexity': 'chain',
            },
            'read_transform': {
                'description': "Read {source} and transform to {format}",
                'tools_hint': ['read', 'file_read', 'converter'],
                'complexity': 'chain',
            },
            'execute_validate': {
                'description': "Execute {command} and validate output matches {criteria}",
                'tools_hint': ['bash', 'execute', 'validate'],
                'complexity': 'chain',
            },
            'multi_source': {
                'description': "Gather information from multiple sources about {topic}",
                'tools_hint': ['search', 'read', 'fetch'],
                'complexity': 'parallel',
            },
        }

        # Agent0: Track tool success rates from executor feedback
        self._tool_success_rates: Dict[str, Tuple[int, int]] = {}  # tool -> (success, total)
        self._executor_feedback_history: List[Dict] = []
        self._max_feedback_history = 100

        logger.info("CurriculumGenerator initialized (DrZero + Agent0 self-curriculum)")

    def generate_training_task(
        self,
        profiles: Dict[str, 'AgentProfile'],
        target_agent: Optional[str] = None
    ) -> SyntheticTask:
        """
        Generate a training task targeting current agent weaknesses.

        DrZero insight: Tasks should be slightly harder than current ability
        to maximize learning signal (zone of proximal development).

        Args:
            profiles: Current agent performance profiles
            target_agent: Optionally target specific agent

        Returns:
            SyntheticTask for training
        """
        # 1. Identify weakest task type across agents
        task_type, difficulty = self._select_task_type_by_weakness(profiles, target_agent)

        # 2. Generate task description from template
        description = self._generate_description(task_type, difficulty)

        # 3. Create synthetic task
        task = SyntheticTask(
            task_id=f"curriculum_{self.total_generated}_{int(time.time())}",
            task_type=task_type,
            description=description,
            difficulty=difficulty,
            target_agent=target_agent,
            metadata={
                'curriculum_round': self.total_generated,
                'weakness_targeted': task_type,
            }
        )

        # 4. Track for diversity
        self.generated_tasks.append(task)
        if len(self.generated_tasks) > self.max_history:
            self.generated_tasks = self.generated_tasks[-self.max_history:]

        self.total_generated += 1
        self.tasks_by_difficulty[f"{int(difficulty * 10) / 10:.1f}"] += 1

        logger.debug(f"Generated curriculum task: type={task_type}, difficulty={difficulty:.2f}")
        return task

    def _select_task_type_by_weakness(
        self,
        profiles: Dict[str, 'AgentProfile'],
        target_agent: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Select task type based on agent weaknesses.

        DrZero insight: Focus on areas where agents struggle most.
        """
        # Aggregate success rates by task type
        type_success_rates: Dict[str, List[float]] = defaultdict(list)

        for agent_name, profile in profiles.items():
            if target_agent and agent_name != target_agent:
                continue

            for task_type, (success, total) in profile.task_success.items():
                if total > 0:
                    rate = success / total
                    type_success_rates[task_type].append(rate)

        # Find weakest task type (lowest average success rate)
        weakest_type = None
        lowest_rate = 1.0

        for task_type, rates in type_success_rates.items():
            avg_rate = sum(rates) / len(rates) if rates else 0.5
            if avg_rate < lowest_rate:
                lowest_rate = avg_rate
                weakest_type = task_type

        # If no data, pick random type for exploration
        if weakest_type is None:
            import random
            weakest_type = random.choice(list(self.task_templates.keys()))
            lowest_rate = 0.5

        # Ensure diversity: occasionally pick other types
        import random
        if random.random() < 0.2:  # 20% exploration
            weakest_type = random.choice(list(self.task_templates.keys()))

        # Calculate difficulty: slightly above current ability
        # DrZero insight: optimal learning at current_ability + epsilon
        current_ability = lowest_rate
        difficulty = min(1.0, current_ability + 0.15)  # 15% harder than current

        # Update tracked difficulty for progressive curriculum
        self.difficulty_by_type[weakest_type] = difficulty

        return weakest_type, difficulty

    def _generate_description(self, task_type: str, difficulty: float) -> str:
        """
        Generate task description from template with difficulty scaling.
        """
        import random

        templates = self.task_templates.get(task_type, ["Perform {task_type} task"])
        template = random.choice(templates)

        # Generate placeholder values based on difficulty
        placeholders = self._generate_placeholders(difficulty)

        try:
            description = template.format(**placeholders)
        except KeyError:
            description = f"Perform {task_type} task (difficulty: {difficulty:.1%})"

        return description

    def _generate_placeholders(self, difficulty: float) -> Dict[str, str]:
        """
        Generate placeholder values scaled by difficulty.

        Higher difficulty = more complex constraints.
        """
        import random

        # Sample domains/fields
        domains = ['users', 'transactions', 'events', 'logs', 'metrics', 'records']
        fields = ['timestamp', 'value', 'count', 'status', 'category', 'score']
        formats = ['json', 'csv', 'parquet', 'sql', 'xml']

        # Complexity scales with difficulty
        num_conditions = max(1, int(difficulty * 3))

        conditions = []
        for _ in range(num_conditions):
            field_name = random.choice(fields)
            op = random.choice(['>', '<', '=', '!=', 'contains', 'between'])
            conditions.append(f"{field_name} {op} value")

        return {
            'criteria': ' AND '.join(conditions[:2]),
            'condition': conditions[0] if conditions else 'value > 0',
            'field': random.choice(fields),
            'field_a': random.choice(fields),
            'field_b': random.choice(fields),
            'domain': random.choice(domains),
            'dataset': random.choice(domains),
            'dataset_a': random.choice(domains),
            'dataset_b': random.choice(domains),
            'format_a': random.choice(formats),
            'format_b': random.choice(formats),
            'constraints': ', '.join(conditions[:num_conditions]),
            'source_a': random.choice(domains),
            'source_b': random.choice(domains),
            'n': str(random.randint(5, 20) * int(difficulty * 2 + 1)),
            'goal': f"Complete {random.choice(domains)} processing",
            'complex_task': f"Analyze and transform {random.choice(domains)}",
            'items': ', '.join(random.sample(fields, min(3, len(fields)))),
            'task_type': random.choice(list(self.task_templates.keys())),
            'topic': random.choice(domains),
            'source': random.choice(domains),
            'format': random.choice(formats),
            'command': f"process_{random.choice(domains)}",
        }

    def update_from_result(self, task: SyntheticTask, success: bool, execution_time: float):
        """
        Update curriculum based on task result.

        DrZero insight: Adjust difficulty based on success rate.
        - Too easy (always succeeds) -> increase difficulty
        - Too hard (always fails) -> decrease difficulty
        """
        task_type = task.task_type
        current_difficulty = self.difficulty_by_type[task_type]

        if success:
            # Task was achievable, increase difficulty slightly
            self.difficulty_by_type[task_type] = min(1.0, current_difficulty + 0.05)
        else:
            # Task was too hard, decrease difficulty
            self.difficulty_by_type[task_type] = max(0.1, current_difficulty - 0.1)

        logger.debug(
            f"Curriculum updated: {task_type} difficulty "
            f"{current_difficulty:.2f} -> {self.difficulty_by_type[task_type]:.2f}"
        )

    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get statistics about the curriculum."""
        return {
            'total_generated': self.total_generated,
            'difficulty_by_type': dict(self.difficulty_by_type),
            'tasks_by_difficulty': dict(self.tasks_by_difficulty),
            'recent_task_types': [t.task_type for t in self.generated_tasks[-10:]],
            'tool_success_rates': dict(self._tool_success_rates),
            'feedback_count': len(self._executor_feedback_history),
        }

    # =========================================================================
    # Agent0 Enhancements: Tool-awareness & Memory Integration
    # =========================================================================

    def connect_state_manager(self, state_manager):
        """
        Connect to SwarmStateManager for tool success tracking.

        DRY: Uses existing AgentStateTracker.tool_usage instead of duplicating.
        """
        self._state_manager = state_manager
        logger.debug("CurriculumGenerator connected to SwarmStateManager")

    def connect_memory(self, memory_system):
        """
        Connect to memory system for weakness detection.

        DRY: Uses existing HierarchicalMemory/SimpleBrain.
        """
        self._memory_system = memory_system
        logger.debug("CurriculumGenerator connected to memory system")

    def receive_executor_feedback(
        self,
        task_id: str,
        success: bool,
        tools_used: List[str],
        execution_time: float = 0.0,
        error_type: str = None
    ):
        """
        Agent0: Receive feedback from executor to adapt curriculum.

        Closes the loop: Executor performance -> Curriculum adaptation.
        """
        feedback = {
            'task_id': task_id,
            'success': success,
            'tools_used': tools_used,
            'execution_time': execution_time,
            'error_type': error_type,
            'timestamp': time.time(),
        }

        self._executor_feedback_history.append(feedback)
        if len(self._executor_feedback_history) > self._max_feedback_history:
            self._executor_feedback_history = self._executor_feedback_history[-self._max_feedback_history:]

        # Update tool success rates
        for tool in tools_used:
            current = self._tool_success_rates.get(tool, (0, 0))
            if success:
                self._tool_success_rates[tool] = (current[0] + 1, current[1] + 1)
            else:
                self._tool_success_rates[tool] = (current[0], current[1] + 1)

        logger.debug(f"Received executor feedback: success={success}, tools={tools_used}")

    def _sync_tool_stats_from_state_manager(self):
        """
        Sync tool success rates from AgentStateTracker.

        DRY: Pulls from existing infrastructure instead of maintaining duplicate state.
        Uses replace strategy: builds into local dict, then atomic-assigns to
        self._tool_success_rates. This prevents inflation on repeated calls since
        the method pulls ALL current state each time (snapshot, not delta).
        """
        if not self._state_manager:
            return

        try:
            # Get all agent trackers from state manager
            if hasattr(self._state_manager, 'agent_trackers'):
                # Build fresh snapshot into local dict
                synced_rates: Dict[str, Tuple[int, int]] = {}

                for agent_name, tracker in self._state_manager.agent_trackers.items():
                    state = tracker.get_state()
                    tool_usage = state.get('tool_usage', {})

                    for tool, count in tool_usage.get('successful', {}).items():
                        current = synced_rates.get(tool, (0, 0))
                        failed_count = tool_usage.get('failed', {}).get(tool, 0)
                        synced_rates[tool] = (
                            current[0] + count,
                            current[1] + count + failed_count
                        )

                # Atomic replace — no inflation on repeated calls
                self._tool_success_rates = synced_rates
        except Exception as e:
            logger.debug(f"Could not sync tool stats: {e}")

    def _query_memory_for_weaknesses(self, target_agent: str = None) -> List[str]:
        """
        Agent0: Query memory for patterns where agent struggled.

        DRY: Uses existing memory.recall() or memory.query() API.
        """
        if not self._memory_system:
            return []

        weaknesses = []
        try:
            # Query for error patterns
            query = f"errors failures mistakes {target_agent or 'agent'}"

            if hasattr(self._memory_system, 'recall'):
                results = self._memory_system.recall(query, top_k=5)
                if results:
                    weaknesses = [str(r)[:100] for r in results[:3]]

            elif hasattr(self._memory_system, 'query'):
                results = self._memory_system.query(query, limit=5)
                if results:
                    weaknesses = [r.get('content', '')[:100] for r in results[:3]]

        except Exception as e:
            logger.debug(f"Could not query memory for weaknesses: {e}")

        return weaknesses

    def generate_tool_aware_task(
        self,
        profiles: Dict[str, 'AgentProfile'],
        target_agent: Optional[str] = None,
        prefer_weak_tools: bool = True
    ) -> SyntheticTask:
        """
        Agent0: Generate a task designed for tool usage.

        Uses existing SyntheticTask with tool hints in metadata (DRY - no new class).
        """
        import random

        # Sync tool stats from state manager
        self._sync_tool_stats_from_state_manager()

        # Find tools with low success rate (weaknesses)
        weak_tools = []
        if prefer_weak_tools:
            for tool, (success, total) in self._tool_success_rates.items():
                if total > 0 and success / total < 0.6:
                    weak_tools.append(tool)

        # Select template - prefer ones using weak tools
        template_name, template = self._select_tool_template(weak_tools)

        # Query memory for additional context
        memory_hints = self._query_memory_for_weaknesses(target_agent)

        # Calculate difficulty based on tool complexity
        complexity_difficulty = {
            'single': 0.3,
            'chain': 0.5,
            'parallel': 0.7,
            'conditional': 0.8,
        }
        base_difficulty = complexity_difficulty.get(template.get('complexity', 'single'), 0.4)

        # Adjust based on executor feedback
        recent_success_rate = self._get_recent_success_rate()
        difficulty = min(1.0, base_difficulty + (recent_success_rate - 0.5) * 0.2)

        # Generate description
        placeholders = self._generate_placeholders(difficulty)
        try:
            description = template['description'].format(**placeholders)
        except KeyError:
            description = template['description']

        # Create task with tool hints in metadata
        task = SyntheticTask(
            task_id=f"tool_curriculum_{self.total_generated}_{int(time.time())}",
            task_type=template_name,
            description=description,
            difficulty=difficulty,
            target_agent=target_agent,
            metadata={
                'curriculum_round': self.total_generated,
                'tool_aware': True,
                'tools_hint': template.get('tools_hint', []),
                'complexity': template.get('complexity', 'single'),
                'weak_tools_targeted': weak_tools[:3],
                'memory_context': memory_hints[0] if memory_hints else None,
            }
        )

        self.generated_tasks.append(task)
        self.total_generated += 1

        logger.debug(f"Generated tool-aware task: {template_name}, difficulty={difficulty:.2f}")
        return task

    def _select_tool_template(self, weak_tools: List[str]) -> Tuple[str, Dict]:
        """Select template preferring ones that use weak tools."""
        import random

        # Score templates by weak tool overlap
        scored = []
        for name, template in self.tool_task_templates.items():
            tools_hint = template.get('tools_hint', [])
            overlap = len(set(weak_tools) & set(tools_hint))
            scored.append((name, template, overlap))

        # Sort by overlap (descending)
        scored.sort(key=lambda x: x[2], reverse=True)

        # 20% exploration - pick random
        if random.random() < 0.2:
            name, template, _ = random.choice(scored)
        else:
            name, template, _ = scored[0]

        return name, template

    def _get_recent_success_rate(self) -> float:
        """Get success rate from recent executor feedback."""
        recent = self._executor_feedback_history[-20:]
        if not recent:
            return 0.5
        return sum(1 for f in recent if f['success']) / len(recent)

    def to_dict(self) -> Dict:
        """Serialize for persistence - includes full learning history."""
        # Convert tool_success_rates tuples to lists for JSON
        serializable_rates = {}
        for tool, rate in self._tool_success_rates.items():
            if isinstance(rate, tuple):
                serializable_rates[tool] = list(rate)
            else:
                serializable_rates[tool] = rate

        return {
            'difficulty_by_type': dict(self.difficulty_by_type),
            'total_generated': self.total_generated,
            'tasks_by_difficulty': dict(self.tasks_by_difficulty),
            'tool_success_rates': serializable_rates,
            'feedback_history': self._executor_feedback_history[-100:],
        }

    @classmethod
    def from_dict(cls, data: Dict, config=None, state_manager=None, memory_system=None) -> 'CurriculumGenerator':
        """Deserialize from persistence - restores full learning state."""
        instance = cls(config, state_manager=state_manager, memory_system=memory_system)
        instance.difficulty_by_type = defaultdict(lambda: 0.3, data.get('difficulty_by_type', {}))
        instance.total_generated = data.get('total_generated', 0)
        instance.tasks_by_difficulty = defaultdict(int, data.get('tasks_by_difficulty', {}))

        # Restore tool_success_rates (convert lists back to tuples)
        raw_rates = data.get('tool_success_rates', {})
        for tool, rate in raw_rates.items():
            if isinstance(rate, list) and len(rate) == 2:
                instance._tool_success_rates[tool] = (rate[0], rate[1])
            else:
                instance._tool_success_rates[tool] = rate

        # Restore feedback history
        instance._executor_feedback_history = data.get('feedback_history', [])

        return instance


__all__ = [
    'SyntheticTask',
    'CurriculumGenerator',
]

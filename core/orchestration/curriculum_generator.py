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
        self._memory_system = memory_system  # SwarmMemory for context

        # Task type templates — CONCRETE, EXECUTABLE tasks.
        # Every template must be a real instruction an LLM agent can
        # attempt without additional context.  No {placeholder} strings
        # that produce "Analyze patterns in: users" (non-executable).
        self.task_templates: Dict[str, List[str]] = {
            'aggregation': [
                "How many countries in the world have a population over 100 million?",
                "What is the approximate total market cap of the top 5 US tech companies?",
                "List the 3 most common programming languages on GitHub and their approximate usage share.",
            ],
            'analysis': [
                "Compare the pros and cons of Python vs Rust for web backend development.",
                "Analyze why SQLite is preferred for mobile apps over PostgreSQL.",
                "What are the key differences between REST and GraphQL APIs? Give concrete examples.",
            ],
            'transformation': [
                "Convert this CSV header row into a SQL CREATE TABLE statement: name,age,email,created_at",
                "Rewrite this Python 2 code to Python 3: print 'hello'; raw_input('name: ')",
                "Convert this JSON to YAML: {\"name\": \"test\", \"version\": 1, \"tags\": [\"a\", \"b\"]}",
            ],
            'validation': [
                "Is this a valid email address? Explain why or why not: user@.example.com",
                "Check if this SQL query has any syntax errors: SELECT * FROM users WHERE name = 'John AND age > 30",
                "Is this JSON valid? If not, fix it: {name: 'test', items: [1, 2,]}",
            ],
            'coding': [
                "Write a Python function that checks if a string is a palindrome.",
                "Write a bash one-liner that finds all .py files modified in the last 24 hours.",
                "Write a Python function that flattens a nested list of arbitrary depth.",
            ],
            'research': [
                "What are the main differences between Docker and Podman?",
                "Explain how HTTPS/TLS handshake works in simple terms.",
                "What is the CAP theorem and how does it apply to distributed databases?",
            ],
            'planning': [
                "Create a step-by-step plan to migrate a monolith web app to microservices.",
                "Plan the steps needed to set up CI/CD for a Python project using GitHub Actions.",
                "Outline the steps to debug a memory leak in a Node.js application.",
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

        # Agent0: Tool-aware task templates — CONCRETE tasks that exercise
        # specific tool categories. No placeholders.
        self.tool_task_templates: Dict[str, Dict[str, Any]] = {
            'search_analyze': {
                'description': "Search the web for the latest Python 3.13 release notes and summarize the top 3 new features.",
                'tools_hint': ['search', 'web_search', 'grep'],
                'complexity': 'chain',
            },
            'read_transform': {
                'description': "Read the file /etc/hostname and convert its content to uppercase.",
                'tools_hint': ['read', 'file_read', 'converter'],
                'complexity': 'chain',
            },
            'execute_validate': {
                'description': "Run 'python3 --version' and verify the output contains 'Python 3'.",
                'tools_hint': ['bash', 'execute', 'validate'],
                'complexity': 'chain',
            },
            'multi_source': {
                'description': "Find the current weather in Tokyo and the current USD/JPY exchange rate, then present both together.",
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
        Pick a concrete, executable task from the template pool.

        Templates are already complete sentences — no placeholder substitution.
        Difficulty is used only for logging/tracking, not to generate harder
        phrasing (that would require an LLM call).
        """
        import random

        templates = self.task_templates.get(task_type)
        if not templates:
            # If task_type doesn't match any template category, pick from any
            all_templates = [t for ts in self.task_templates.values() for t in ts]
            return random.choice(all_templates)

        return random.choice(templates)

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

        DRY: Uses existing SwarmMemory/SimpleBrain.
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

        # Templates are now concrete — no placeholder substitution needed
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

    # =========================================================================
    # REPLAY BUFFER: Generate tasks from real past executions
    # =========================================================================

    def generate_replay_task(
        self,
        collective_memory,
        profiles: Dict[str, 'AgentProfile'],
        target_agent: Optional[str] = None
    ) -> Optional[SyntheticTask]:
        """
        Generate a training task from REAL past executions (replay buffer).

        Instead of template-based synthetic tasks with placeholder strings like
        "Filter records where: category contains value" (which are non-executable),
        this pulls actual tasks from collective_memory and re-presents them for
        practice, focusing on previously failed tasks.

        DrZero insight: Replaying failed tasks is more valuable than synthetic ones
        because they represent real gaps in agent capability.

        Args:
            collective_memory: SwarmIntelligence.collective_memory (deque of dicts)
            profiles: Current agent profiles
            target_agent: Optionally target specific agent

        Returns:
            SyntheticTask based on a real past task, or None if no suitable replay
        """
        import random

        if not collective_memory:
            return None

        mem_list = list(collective_memory)

        # Prioritize failed tasks (these are where learning value is highest)
        failed_tasks = [
            m for m in mem_list
            if not m.get('success', True)
            and m.get('context', {}).get('query', '')  # Must have a real query
        ]

        # Also include successful tasks for reinforcement (lower priority)
        successful_tasks = [
            m for m in mem_list
            if m.get('success', True)
            and m.get('context', {}).get('query', '')
        ]

        # If targeting a specific agent, filter to their failures
        if target_agent:
            agent_failed = [m for m in failed_tasks if m.get('agent') == target_agent]
            if agent_failed:
                failed_tasks = agent_failed

        # Select replay candidate: 80% failed tasks, 20% successful (reinforcement)
        candidates = []
        if failed_tasks and random.random() < 0.8:
            candidates = failed_tasks
        elif successful_tasks:
            candidates = successful_tasks
        elif failed_tasks:
            candidates = failed_tasks
        else:
            return None  # No suitable replay candidates

        # Pick a random candidate (with recency bias: more recent = more likely)
        weights = [i + 1 for i in range(len(candidates))]  # Linear recency weight
        selected = random.choices(candidates, weights=weights, k=1)[0]

        # Extract real task description
        context = selected.get('context', {})
        task_description = (
            context.get('query', '')
            or context.get('task', '')
            or context.get('goal', '')
            or f"Repeat {selected.get('task_type', 'general')} task"
        )

        task_type = selected.get('task_type', 'general')
        was_success = selected.get('success', False)

        # Calculate difficulty from historical performance
        difficulty = self.difficulty_by_type.get(task_type, 0.3)
        if not was_success:
            difficulty = min(1.0, difficulty + 0.1)  # Failed tasks are harder

        task = SyntheticTask(
            task_id=f"replay_{self.total_generated}_{int(time.time())}",
            task_type=task_type,
            description=task_description,
            difficulty=difficulty,
            target_agent=target_agent or selected.get('agent'),
            metadata={
                'source': 'replay_buffer',
                'original_success': was_success,
                'original_agent': selected.get('agent', 'unknown'),
                'original_time': selected.get('execution_time', 0),
                'curriculum_round': self.total_generated,
            }
        )

        self.generated_tasks.append(task)
        if len(self.generated_tasks) > self.max_history:
            self.generated_tasks = self.generated_tasks[-self.max_history:]

        self.total_generated += 1

        logger.debug(
            f"Replay task generated: type={task_type}, "
            f"original_success={was_success}, difficulty={difficulty:.2f}"
        )
        return task

    def generate_smart_task(
        self,
        profiles: Dict[str, 'AgentProfile'],
        collective_memory=None,
        target_agent: Optional[str] = None,
    ) -> SyntheticTask:
        """
        Smart task generation: prefer replay buffer, fall back to templates.

        This is the recommended entry point. It tries:
        1. Replay buffer (real past tasks) - preferred because executable
        2. Tool-aware templates - if no replay data
        3. Basic templates - last resort

        Args:
            profiles: Current agent profiles
            collective_memory: Optional collective memory for replay
            target_agent: Optionally target specific agent

        Returns:
            SyntheticTask (from replay or template)
        """
        # Try replay buffer first — even 1 real past task is better than
        # a template because it's a task the system actually encountered.
        if collective_memory and len(list(collective_memory)) > 0:
            replay_task = self.generate_replay_task(
                collective_memory=collective_memory,
                profiles=profiles,
                target_agent=target_agent
            )
            if replay_task:
                return replay_task

        # Fall back to tool-aware template
        if self._tool_success_rates:
            return self.generate_tool_aware_task(
                profiles=profiles,
                target_agent=target_agent,
                prefer_weak_tools=True
            )

        # Last resort: basic template
        return self.generate_training_task(
            profiles=profiles,
            target_agent=target_agent
        )

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

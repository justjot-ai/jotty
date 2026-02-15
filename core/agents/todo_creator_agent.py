"""TodoCreatorAgent — validates DAGs, assigns actors, manages todos."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import dspy

from ..orchestration.swarm_roadmap import SwarmTaskBoard
from ..foundation.data_structures import SwarmLearningConfig, MemoryLevel
from ..foundation.exceptions import AgentExecutionError
from .base import AgentResult

logger = logging.getLogger(__name__)

from .dag_types import (
    DAGAgentMixin, SwarmResources, Actor, ExecutableDAG,
    ActorAssignmentSignature,
    DAGValidationSignature, OptimizeDAGSignature,
)

class TodoCreatorAgent(DAGAgentMixin):
    """
    DSPy agent for DAG validation, actor assignment, and optimization.

    Uses SHARED SwarmResources:
    - memory: Shared knowledge across all agents
    - context: Shared taskboard for coordination
    - bus: Inter-agent communication
    - learner: Shared learning from outcomes

    Inherits from:
    - DAGAgentMixin: For BaseAgent infrastructure (metrics, hooks)
    """

    def __init__(self, config: Optional[SwarmConfig] = None, lm: Optional[dspy.LM] = None) -> None:
        self.jotty_config = config or SwarmConfig()
        self.lm = lm or getattr(dspy.settings, 'lm', None)

        # Initialize BaseAgent infrastructure via mixin
        self._init_agent_infrastructure("TodoCreatorAgent")

        if self.lm is None:
            raise AgentExecutionError(
                "No language model configured. Either pass 'lm' parameter or "
                "configure globally via dspy.configure(lm=...)"
            )

        # Get SHARED swarm resources (singleton)
        self.swarm = SwarmResources.get_instance(self.jotty_config)
        self.memory = self.swarm.memory      # SHARED memory
        self.context = self.swarm.context    # SHARED taskboard
        self.bus = self.swarm.bus            # SHARED communication
        self.learner = self.swarm.learner    # SHARED learner

        # DSPy modules
        self.dag_optimizer = dspy.ChainOfThought(OptimizeDAGSignature)
        self.actor_assigner = dspy.ChainOfThought(ActorAssignmentSignature)
        self.dag_validator = dspy.ChainOfThought(DAGValidationSignature)

        # Register with message bus for inter-agent communication
        self.bus.subscribe("TodoCreatorAgent", self._handle_message)

        logger.info(" TodoCreatorAgent initialized with SHARED swarm resources")

    def _handle_message(self, message: Any) -> Any:
        """Handle incoming messages from other agents."""
        logger.info(f"TodoCreatorAgent received from {message.from_agent}: {message.data}")

    async def execute(self, markovian_todo: SwarmTaskBoard, available_actors: List[Dict[str, Any]], **kwargs: Any) -> AgentResult:
        """
        Execute DAG creation with BaseAgent-compatible interface.

        Args:
            markovian_todo: SwarmTaskBoard with tasks
            available_actors: List of actor dicts with 'name' and 'capabilities'

        Returns:
            AgentResult with ExecutableDAG as output
        """
        import time
        start_time = time.time()

        self._run_pre_hooks(
            markovian_todo=markovian_todo,
            available_actors=available_actors,
            **kwargs
        )

        try:
            # Call the synchronous create_executable_dag method
            executable_dag = self.create_executable_dag(markovian_todo, available_actors)

            execution_time = time.time() - start_time
            self._track_execution(success=True, execution_time=execution_time)

            result = AgentResult(
                success=True,
                output=executable_dag,
                agent_name="TodoCreatorAgent",
                execution_time=execution_time,
                metadata={
                    "task_count": executable_dag.total_tasks,
                    "assignments": len(executable_dag.assignments),
                    "validation_passed": executable_dag.validation_passed,
                    "todo_id": executable_dag.markovian_todo.todo_id,
                }
            )

            self._run_post_hooks(
                result,
                markovian_todo=markovian_todo,
                available_actors=available_actors,
                **kwargs
            )
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self._track_execution(success=False, execution_time=execution_time)
            logger.error(f"TodoCreatorAgent execution failed: {e}")

            return AgentResult(
                success=False,
                output=None,
                agent_name="TodoCreatorAgent",
                execution_time=execution_time,
                error=str(e)
            )

    def create_executable_dag(
        self,
        markovian_todo: SwarmTaskBoard,
        available_actors: List[Dict[str, Any]]
    ) -> ExecutableDAG:
        """
        Create an executable DAG with actor assignments and validation.

        Args:
            markovian_todo: SwarmTaskBoard with tasks
            available_actors: List of actor dicts with 'name' and 'capabilities'

        Returns:
            ExecutableDAG with assignments and validation results
        """
        logger.info(f"Creating ExecutableDAG for: {markovian_todo.root_task[:50]}")
        logger.info(f"Tasks: {len(markovian_todo.subtasks)}, Actors: {len(available_actors)}")

        # Convert actor dicts to Actor objects
        actors = [
            Actor(
                name=a["name"],
                capabilities=a["capabilities"],
                description=a.get("description"),
                max_concurrent_tasks=a.get("max_concurrent_tasks", 1)
            )
            for a in available_actors
        ]

        # Retrieve relevant patterns from memory
        relevant_memories = self.memory.retrieve(
            query=f"actor assignment for {markovian_todo.root_task}",
            goal="optimal actor assignment",
            budget_tokens=1000
        )

        # Step 1: Assign actors to tasks
        logger.info("Step 1: Assigning actors to tasks...")
        assignments = self._assign_actors_to_tasks(markovian_todo, actors)
        logger.info(f"Assigned actors to {len(assignments)}/{len(markovian_todo.subtasks)} tasks")

        # Update actor field in SubtaskState
        for task_id, actor in assignments.items():
            if task_id in markovian_todo.subtasks:
                markovian_todo.subtasks[task_id].actor = actor.name

        # Step 2: Collapse consecutive tasks by same actor
        logger.info("Step 2: Collapsing consecutive tasks by same actor...")
        original_count = len(markovian_todo.subtasks)
        markovian_todo, assignments = self._collapse_consecutive_tasks(markovian_todo, assignments)
        collapsed = original_count - len(markovian_todo.subtasks)
        logger.info(f"Collapsed {collapsed} tasks: {original_count} -> {len(markovian_todo.subtasks)}")

        # Step 3: Validate DAG
        logger.info("Step 3: Validating DAG structure...")
        is_valid, issues = self._validate_dag(markovian_todo, assignments, actors)

        # Store pattern in SHARED memory
        self.memory.store(
            content=f"Assignment pattern: {len(actors)} actors, {len(markovian_todo.subtasks)} tasks",
            level=MemoryLevel.PROCEDURAL,
            context={
                "actor_names": [a.name for a in actors],
                "task_count": len(markovian_todo.subtasks),
                "collapsed_count": collapsed,
                "valid": is_valid,
                "agent": "TodoCreatorAgent"
            },
            goal="actor assignment pattern learning"
        )

        if is_valid:
            logger.info(" DAG validation passed")
        else:
            logger.warning(f" DAG validation found {len(issues)} issues")
            for i, issue in enumerate(issues, 1):
                logger.warning(f"   {i}. {issue}")

        executable_dag = ExecutableDAG(
            markovian_todo=markovian_todo,
            assignments=assignments,
            validation_passed=is_valid,
            validation_issues=issues,
            fixes_applied=[]
        )

        # Store to SHARED context (taskboard) for other agents
        self.context.set("executable_dag", executable_dag)
        self.context.set("assignments", {tid: a.name for tid, a in assignments.items()})
        self.context.set("dag_valid", is_valid)
        self.context.set("last_assignment_time", datetime.now().isoformat())

        # Notify other agents via bus
        from .axon import Message
        self.bus.publish(Message(
            from_agent="TodoCreatorAgent",
            to_agent="ExecutionAgent",  # Next agent in pipeline
            data={"dag_id": markovian_todo.todo_id, "valid": is_valid, "task_count": len(markovian_todo.subtasks)},
            format="dict",
            size_bytes=150,
            timestamp=datetime.now().timestamp()
        ))

        return executable_dag

    def _assign_actors_to_tasks(
        self,
        todo: SwarmTaskBoard,
        actors: List[Actor]
    ) -> Dict[str, Actor]:
        """Assign actors to all tasks using LLM-based selection."""
        assignments = {}
        actor_history = {
            actor.name: {"task_count": 0, "recent_tasks": []}
            for actor in actors
        }

        with dspy.context(lm=self.lm):
            for task_id, task in todo.subtasks.items():
                task_type = task.intermediary_values.get("task_type", "implementation")

                result = self.actor_assigner(
                    task_id=task_id,
                    task_name=task.description[:100],
                    task_type=task_type,
                    task_description=task.description,
                    available_actors=str([
                        {"name": a.name, "capabilities": a.capabilities}
                        for a in actors
                    ]),
                    current_assignments=str(actor_history)
                )

                # Find assigned actor
                assigned_actor = next(
                    (a for a in actors if a.name == result.assigned_actor_name),
                    actors[0]  # Fallback to first actor
                )

                assignments[task_id] = assigned_actor
                actor_history[assigned_actor.name]["task_count"] += 1
                actor_history[assigned_actor.name]["recent_tasks"].append({
                    "task_id": task_id,
                    "task_name": task.description[:50]
                })

                logger.debug(f"Assigned {task_id} to {assigned_actor.name}: {result.reasoning}")

        return assignments

    def _has_internal_dependencies(self, todo: SwarmTaskBoard, task_ids: List[str]) -> bool:
        """Check if any task in the group depends on another task in the same group."""
        task_id_set = set(task_ids)
        for task_id in task_ids:
            task = todo.subtasks.get(task_id)
            if task:
                for dep_id in task.depends_on:
                    if dep_id in task_id_set:
                        return True
        return False

    def _collapse_consecutive_tasks(
        self,
        todo: SwarmTaskBoard,
        assignments: Dict[str, Actor]
    ) -> Tuple[SwarmTaskBoard, Dict[str, Actor]]:
        """
        Collapse consecutive tasks assigned to the same actor.

        Same-actor tasks are merged regardless of internal dependencies.
        The actor handles execution order internally.
        """
        # Get topological order
        try:
            task_order = self._get_topological_order(todo)
        except ValueError as e:
            logger.warning(f"Cannot collapse tasks - topological sort failed: {e}")
            return todo, assignments

        # Group consecutive tasks by actor
        groups = []
        current_group = []
        current_actor = None

        for task_id in task_order:
            if task_id not in assignments:
                if current_group:
                    groups.append((current_actor, current_group))
                current_group = []
                current_actor = None
                continue

            task_actor = assignments[task_id]

            if current_actor is None or current_actor.name != task_actor.name:
                if current_group:
                    groups.append((current_actor, current_group))
                current_group = [task_id]
                current_actor = task_actor
            else:
                current_group.append(task_id)

        if current_group:
            groups.append((current_actor, current_group))

        # Merge groups (with limits to prevent mega-tasks)
        MAX_TASKS_PER_MERGE = 5  # Don't merge more than 5 tasks into one
        new_assignments = {}
        tasks_to_remove = set()

        for actor, task_ids in groups:
            if len(task_ids) == 1:
                new_assignments[task_ids[0]] = actor
                continue

            # Split large groups into chunks of MAX_TASKS_PER_MERGE
            chunks = [task_ids[i:i + MAX_TASKS_PER_MERGE] for i in range(0, len(task_ids), MAX_TASKS_PER_MERGE)]

            for chunk in chunks:
                if len(chunk) == 1:
                    new_assignments[chunk[0]] = actor
                    continue

                # Merge this chunk
                has_internal = self._has_internal_dependencies(todo, chunk)
                if has_internal:
                    logger.info(
                        f"Merging {len(chunk)} tasks for {actor.name} "
                        f"(internal deps, same actor handles order): {chunk}"
                    )
                else:
                    logger.info(f"Collapsing {len(chunk)} tasks for {actor.name}: {chunk}")

                self._merge_task_group(todo, chunk, actor, new_assignments, tasks_to_remove)

        # Remove merged tasks
        for task_id in tasks_to_remove:
            if task_id in todo.subtasks:
                del todo.subtasks[task_id]
            if task_id in todo.execution_order:
                todo.execution_order.remove(task_id)

        return todo, new_assignments

    def _merge_task_group(self, todo: SwarmTaskBoard, task_ids: List[str], actor: Actor, new_assignments: Dict[str, Actor], tasks_to_remove: set) -> Any:
        """Merge a group of tasks into a single combined task."""
        combined_id = "_".join(task_ids)
        tasks = [todo.subtasks[tid] for tid in task_ids if tid in todo.subtasks]

        if not tasks:
            return

        # Combine descriptions
        combined_description = "\n\n".join([
            f"**{t.task_id}**: {t.description}"
            for t in tasks
        ])

        # Collect external dependencies
        task_id_set = set(task_ids)
        all_external_deps = set()
        for task in tasks:
            for dep_id in task.depends_on:
                if dep_id not in task_id_set:
                    all_external_deps.add(dep_id)

        # Create combined task
        todo.add_task(
            task_id=combined_id,
            description=combined_description[:500],
            actor=actor.name,
            depends_on=list(all_external_deps),
            priority=max(t.priority for t in tasks),
        )

        # Copy intermediary values
        combined_task = todo.subtasks[combined_id]
        combined_task.intermediary_values["merged_from"] = task_ids
        combined_task.intermediary_values["task_type"] = tasks[0].intermediary_values.get("task_type", "implementation")

        new_assignments[combined_id] = actor

        # Update dependents
        for other_id, other_task in todo.subtasks.items():
            if other_id == combined_id:
                continue
            for old_id in task_ids:
                if old_id in other_task.depends_on:
                    other_task.depends_on.remove(old_id)
                    if combined_id not in other_task.depends_on:
                        other_task.depends_on.append(combined_id)

        tasks_to_remove.update(task_ids)

    def _get_topological_order(self, todo: SwarmTaskBoard) -> List[str]:
        """Get topological order of tasks."""
        in_degree = {tid: 0 for tid in todo.subtasks}

        for task in todo.subtasks.values():
            for dep_id in task.depends_on:
                if dep_id in in_degree:
                    in_degree[task.task_id] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            task_id = queue.pop(0)
            result.append(task_id)

            for other_id, other_task in todo.subtasks.items():
                if task_id in other_task.depends_on:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)

        if len(result) != len(todo.subtasks):
            raise ValueError("Cycle detected in task dependencies")

        return result

    def _validate_dag(
        self,
        todo: SwarmTaskBoard,
        assignments: Dict[str, Actor],
        actors: List[Actor]
    ) -> Tuple[bool, List[str]]:
        """Validate DAG structure and assignments."""
        issues = []

        # Check for cycles
        try:
            self._get_topological_order(todo)
            cycle_info = "No cycle detected"
        except ValueError:
            cycle_info = "Cycle detected in dependencies"
            issues.append(cycle_info)

        # Check missing dependencies
        for task_id, task in todo.subtasks.items():
            for dep_id in task.depends_on:
                if dep_id not in todo.subtasks:
                    issues.append(f"Task {task_id} depends on non-existent task {dep_id}")

        # Check unassigned tasks
        unassigned = [tid for tid in todo.subtasks if tid not in assignments]
        if unassigned:
            issues.append(f"Unassigned tasks: {', '.join(unassigned)}")

        # LLM-based validation
        stats = {
            "total_tasks": len(todo.subtasks),
            "assigned": len(assignments),
            "actors": len(actors)
        }

        with dspy.context(lm=self.lm):
            result = self.dag_validator(
                dag_summary=str(stats),
                tasks_info=str([
                    {
                        "id": t.task_id,
                        "desc": t.description[:50],
                        "depends_on": t.depends_on
                    }
                    for t in todo.subtasks.values()
                ]),
                assignments_info=str({
                    tid: actor.name
                    for tid, actor in assignments.items()
                }),
                cycle_check=cycle_info
            )

        is_valid = result.is_valid.upper() == "YES"

        if result.issues_found.strip().lower() not in ["none", "no issues", ""]:
            llm_issues = [
                issue.strip()
                for issue in result.issues_found.split("\n")
                if issue.strip() and not issue.strip().lower().startswith("none")
            ]
            issues.extend(llm_issues)

        return (len(issues) == 0 and is_valid), issues

    def update_from_execution(self, executable_dag: ExecutableDAG, outcomes: Dict[str, bool]) -> Any:
        """
        Learn from execution outcomes using TD learning.

        Args:
            executable_dag: The executed DAG
            outcomes: Dict of task_id -> success (True/False)
        """
        for task_id, success in outcomes.items():
            if task_id in executable_dag.assignments:
                actor = executable_dag.assignments[task_id]
                task = executable_dag.markovian_todo.subtasks.get(task_id)

                if task:
                    # Record in learner
                    task_type = task.intermediary_values.get('task_type', 'unknown')
                    reward = 1.0 if success else -0.5

                    # Update Q-value for this actor-task_type pair
                    self.learner.update(
                        state={'goal': task_type, 'actor': actor.name},
                        action={'output': 'executed', 'success': success},
                        reward=reward,
                        next_state={'goal': task_type, 'completed': True}
                    )

                    # Store in memory
                    self.memory.store(
                        content=f"Actor {actor.name} {'succeeded' if success else 'failed'} on {task.description[:50]}",
                        level=MemoryLevel.EPISODIC if not success else MemoryLevel.PROCEDURAL,
                        context={
                            "actor": actor.name,
                            "task_type": task.intermediary_values.get("task_type"),
                            "success": success
                        },
                        goal="execution outcome learning"
                    )

    def visualize_assignments(self, executable_dag: ExecutableDAG) -> str:
        """Generate a visualization of task-actor assignments."""
        output = []
        output.append("\n" + "=" * 80)
        output.append("TASK-ACTOR ASSIGNMENTS")
        output.append("=" * 80 + "\n")

        # Group by actor
        actor_tasks: Dict[str, List[SubtaskState]] = {}
        for task_id, actor in executable_dag.assignments.items():
            if actor.name not in actor_tasks:
                actor_tasks[actor.name] = []
            task = executable_dag.markovian_todo.subtasks.get(task_id)
            if task:
                actor_tasks[actor.name].append(task)

        for actor_name, tasks in actor_tasks.items():
            actor = next(
                (a for a in executable_dag.assignments.values() if a.name == actor_name),
                None
            )
            output.append(f"\n {actor_name}")
            if actor:
                output.append(f"   Capabilities: {', '.join(actor.capabilities)}")
            output.append(f"   Assigned Tasks: {len(tasks)}")
            output.append("   " + "-" * 70)

            for task in tasks:
                task_type = task.intermediary_values.get("task_type", "unknown")
                output.append(f"   • {task.task_id}: {task.description[:60]} ({task_type})")
                if task.depends_on:
                    output.append(f"     ↳ Depends on: {', '.join(task.depends_on)}")

        output.append("\n" + "=" * 80 + "\n")
        return "\n".join(output)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


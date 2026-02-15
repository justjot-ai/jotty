"""TaskBreakdownAgent — converts plans into executable DAG workflows."""

import logging
import re
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime

import dspy

from ..orchestration.swarm_roadmap import SwarmTaskBoard
from ..foundation.data_structures import SwarmConfig, SwarmLearningConfig, MemoryLevel
from ..persistence.shared_context import SharedContext
from .base import AgentResult

logger = logging.getLogger(__name__)

from .dag_types import (
    DAGAgentMixin, SwarmResources,
    ExtractTasksSignature, IdentifyDependenciesSignature,
    OptimizeWorkflowSignature,
)

class TaskBreakdownAgent(dspy.Module, DAGAgentMixin):
    """
    DSPy Chain of Thought agent that breaks down implementation plans into DAG workflows.

    Uses SHARED SwarmResources:
    - memory: Shared knowledge across all agents
    - context: Shared taskboard for coordination
    - bus: Inter-agent communication

    Inherits from:
    - dspy.Module: For DSPy integration (forward() method)
    - DAGAgentMixin: For BaseAgent infrastructure (metrics, hooks)
    """

    def __init__(self, config: Optional[SwarmConfig] = None) -> None:
        super().__init__()
        self.jotty_config = config or SwarmConfig()

        # Initialize BaseAgent infrastructure via mixin
        self._init_agent_infrastructure("TaskBreakdownAgent")

        # Get SHARED swarm resources (singleton)
        self.swarm = SwarmResources.get_instance(self.jotty_config)
        self.memory = self.swarm.memory      # SHARED memory
        self.context = self.swarm.context    # SHARED taskboard
        self.bus = self.swarm.bus            # SHARED communication

        # Chain of Thought modules
        self.extract_tasks = dspy.ChainOfThought(ExtractTasksSignature)
        self.identify_dependencies = dspy.ChainOfThought(IdentifyDependenciesSignature)
        self.optimize_workflow = dspy.ChainOfThought(OptimizeWorkflowSignature)

        # Register with message bus for inter-agent communication
        self.bus.subscribe("TaskBreakdownAgent", self._handle_message)

        logger.info(" TaskBreakdownAgent initialized with SHARED swarm resources")

    async def execute(self, implementation_plan: str, **kwargs: Any) -> AgentResult:
        """
        Execute task breakdown with BaseAgent-compatible interface.

        Args:
            implementation_plan: Complete implementation plan as string

        Returns:
            AgentResult with SwarmTaskBoard as output
        """
        import time
        start_time = time.time()

        self._run_pre_hooks(implementation_plan=implementation_plan, **kwargs)

        try:
            # Call the DSPy forward method
            markovian_todo = self.forward(implementation_plan)

            execution_time = time.time() - start_time
            self._track_execution(success=True, execution_time=execution_time)

            result = AgentResult(
                success=True,
                output=markovian_todo,
                agent_name="TaskBreakdownAgent",
                execution_time=execution_time,
                metadata={
                    "task_count": len(markovian_todo.subtasks),
                    "todo_id": markovian_todo.todo_id,
                }
            )

            self._run_post_hooks(result, implementation_plan=implementation_plan, **kwargs)
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self._track_execution(success=False, execution_time=execution_time)
            logger.error(f"TaskBreakdownAgent execution failed: {e}")

            return AgentResult(
                success=False,
                output=None,
                agent_name="TaskBreakdownAgent",
                execution_time=execution_time,
                error=str(e)
            )

    def _handle_message(self, message: Any) -> Any:
        """Handle incoming messages from other agents."""
        logger.debug(f"TaskBreakdownAgent received: {message}")

    def forward(self, implementation_plan: str) -> SwarmTaskBoard:
        """
        Break down implementation plan into executable DAG workflow.

        Args:
            implementation_plan: Complete implementation plan as string

        Returns:
            SwarmTaskBoard: Markovian task manager with all tasks
        """
        logger.info("[TASK BREAKDOWN] Starting plan analysis...")

        # Retrieve relevant patterns from memory
        relevant_patterns = self.memory.retrieve(
            query=f"task breakdown patterns for: {implementation_plan[:200]}",
            goal="effective task decomposition",
            budget_tokens=1000
        )

        # Step 1: Extract tasks
        logger.info("[TASK BREAKDOWN] Step 1: Extracting granular tasks...")
        tasks_response = self.extract_tasks(implementation_plan=implementation_plan)
        tasks_list = tasks_response.tasks_list
        logger.info(f"[TASK BREAKDOWN] Extracted tasks:\n{tasks_list[:500]}...")

        # Step 2: Identify dependencies
        logger.info("[TASK BREAKDOWN] Step 2: Identifying dependencies...")
        deps_response = self.identify_dependencies(tasks_list=tasks_list)
        dependencies_graph = deps_response.dependencies_graph
        logger.info(f"[TASK BREAKDOWN] Dependencies:\n{dependencies_graph[:500]}...")

        # Step 3: Optimize workflow
        logger.info("[TASK BREAKDOWN] Step 3: Optimizing workflow...")
        tasks_with_deps = f"TASKS:\n{tasks_list}\n\nDEPENDENCIES:\n{dependencies_graph}"
        workflow_response = self.optimize_workflow(tasks_with_dependencies=tasks_with_deps)
        optimized_workflow = workflow_response.optimized_workflow
        logger.info(f"[TASK BREAKDOWN] Optimized workflow:\n{optimized_workflow[:500]}...")

        # Step 4: Build SwarmTaskBoard
        logger.info("[TASK BREAKDOWN] Step 4: Building SwarmTaskBoard...")
        markovian_todo = self._build_markovian_todo(
            tasks_list, dependencies_graph, optimized_workflow, implementation_plan
        )

        # Step 5: Validate structure
        if self._detect_cycles(markovian_todo):
            logger.warning("[TASK BREAKDOWN] Cycle detected, fixing...")
            markovian_todo = self._fix_cycles(markovian_todo)

        # Step 6: Post-process to limit task count (aggregate if >10 tasks)
        MAX_TASKS = 10
        if len(markovian_todo.subtasks) > MAX_TASKS:
            logger.info(f"[TASK BREAKDOWN] Aggregating {len(markovian_todo.subtasks)} tasks to max {MAX_TASKS}...")
            markovian_todo = self._aggregate_tasks(markovian_todo, MAX_TASKS)

        # Store pattern in SHARED memory
        self.memory.store(
            content=f"Breakdown pattern: {len(markovian_todo.subtasks)} tasks from plan",
            level=MemoryLevel.PROCEDURAL,
            context={
                "plan_hash": hashlib.md5(implementation_plan.encode()).hexdigest()[:8],
                "task_count": len(markovian_todo.subtasks),
                "stages": len(self._get_execution_stages(markovian_todo)),
                "agent": "TaskBreakdownAgent"
            },
            goal="task breakdown pattern learning"
        )

        # Store to SHARED context (taskboard) for other agents
        self.context.set("current_todo", markovian_todo)
        self.context.set("task_count", len(markovian_todo.subtasks))
        self.context.set("last_breakdown_time", datetime.now().isoformat())

        # Notify other agents via bus
        from .axon import Message
        self.bus.publish(Message(
            from_agent="TaskBreakdownAgent",
            to_agent="TodoCreatorAgent",
            data={"todo_id": markovian_todo.todo_id, "task_count": len(markovian_todo.subtasks)},
            format="dict",
            size_bytes=100,
            timestamp=datetime.now().timestamp()
        ))

        logger.info(f"[TASK BREAKDOWN] Created SwarmTaskBoard with {len(markovian_todo.subtasks)} tasks (shared)")
        return markovian_todo

    def _build_markovian_todo(
        self,
        tasks_list: str,
        dependencies_graph: str,
        optimized_workflow: str,
        original_plan: str
    ) -> SwarmTaskBoard:
        """Build SwarmTaskBoard from parsed task information."""
        todo = SwarmTaskBoard(root_task=original_plan[:100])

        # Parse tasks
        tasks_dict = self._parse_tasks(tasks_list)

        # Parse dependencies
        dependencies_dict = self._parse_dependencies(dependencies_graph)

        # Add tasks to SwarmTaskBoard
        for task_id, task_info in tasks_dict.items():
            todo.add_task(
                task_id=task_id,
                description=task_info.get("description", task_info.get("name", f"Task {task_id}")),
                actor="",  # Will be assigned by TodoCreatorAgent
                depends_on=dependencies_dict.get(task_id, []),
                priority=task_info.get("priority", 1.0),
            )

            # Store task type in intermediary_values
            if task_id in todo.subtasks:
                todo.subtasks[task_id].intermediary_values["task_type"] = task_info.get("type", "implementation")
                todo.subtasks[task_id].intermediary_values["files"] = task_info.get("files_to_create", [])

        return todo

    def _parse_tasks(self, tasks_list: str) -> Dict[str, Dict[str, Any]]:
        """Parse tasks from LLM output (handles JSON, numbered list, and text format)."""
        tasks = {}
        task_counter = 1

        # Try JSON parsing first
        try:
            import json
            # Handle string that looks like JSON array
            if tasks_list.strip().startswith('['):
                parsed = json.loads(tasks_list)
                if isinstance(parsed, list):
                    for item in parsed:
                        task_id = f"task_{task_counter}"
                        tasks[task_id] = {
                            "name": item.get("description", item.get("name", f"Task {task_counter}")),
                            "description": item.get("details", item.get("description", "")),
                            "type": item.get("type", "implementation"),
                            "files_to_create": item.get("files", []),
                        }
                        task_counter += 1
                    if tasks:
                        return tasks
        except (json.JSONDecodeError, TypeError):
            pass

        # Try parsing as Python literal (LLM might return repr of list)
        try:
            import ast
            parsed = ast.literal_eval(tasks_list)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        task_id = f"task_{task_counter}"
                        tasks[task_id] = {
                            "name": item.get("description", item.get("name", f"Task {task_counter}")),
                            "description": item.get("details", item.get("description", "")),
                            "type": item.get("type", "implementation"),
                            "files_to_create": item.get("files", []),
                        }
                        task_counter += 1
                if tasks:
                    return tasks
        except (ValueError, SyntaxError):
            pass

        # Try TASK: format parsing
        task_entries = re.split(r'\n(?=TASK:)', tasks_list)
        for entry in task_entries:
            if not entry.strip() or not entry.startswith("TASK:"):
                continue

            task_id = f"task_{task_counter}"
            task_info = {}

            # Extract name
            name_match = re.search(r'TASK:\s*([^|]+)', entry)
            if name_match:
                task_info["name"] = name_match.group(1).strip()

            # Extract type
            type_match = re.search(r'TYPE:\s*([^|]+)', entry)
            if type_match:
                task_info["type"] = type_match.group(1).strip().lower()

            # Extract description
            desc_match = re.search(r'DESC:\s*([^|]+?)(?:\s*\||$)', entry, re.DOTALL)
            if desc_match:
                task_info["description"] = desc_match.group(1).strip()

            # Extract files
            files_match = re.search(r'FILES:\s*([^|]+)', entry)
            if files_match:
                files_str = files_match.group(1).strip()
                files = [f.strip() for f in files_str.split(',') if f.strip()]
                task_info["files_to_create"] = files

            if task_info:
                tasks[task_id] = task_info
                task_counter += 1

        if tasks:
            return tasks

        # Fallback: Numbered list format (1. task, 2. task, etc.)
        # Matches patterns like "1. Task description" or "1) Task description" or "- Task description"
        lines = tasks_list.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match numbered items: "1. text", "1) text", "- text", "* text"
            match = re.match(r'^(?:\d+[\.\)]\s*|\-\s*|\*\s*)(.+)$', line)
            if match:
                task_text = match.group(1).strip()
                if task_text:
                    task_id = f"task_{task_counter}"
                    # Infer task type from keywords
                    task_type = "implementation"
                    task_lower = task_text.lower()
                    if any(kw in task_lower for kw in ["test", "verify", "check", "assert"]):
                        task_type = "testing"
                    elif any(kw in task_lower for kw in ["valid", "error", "exception", "handle"]):
                        task_type = "validation"
                    elif any(kw in task_lower for kw in ["setup", "create file", "initialize", "configure"]):
                        task_type = "setup"

                    # Extract potential file references
                    files = []
                    file_match = re.findall(r'[\w_]+\.(?:py|js|ts|java|go|rb)', task_text)
                    if file_match:
                        files = file_match

                    tasks[task_id] = {
                        "name": task_text[:80],
                        "description": task_text,
                        "type": task_type,
                        "files_to_create": files,
                    }
                    task_counter += 1

        return tasks

    def _parse_dependencies(self, dependencies_graph: str) -> Dict[str, List[str]]:
        """Parse dependencies from LLM output (handles JSON and text format)."""
        dependencies = {}

        # Try JSON parsing first
        try:
            import json
            if dependencies_graph.strip().startswith('[') or dependencies_graph.strip().startswith('{'):
                parsed = json.loads(dependencies_graph)
                if isinstance(parsed, list):
                    for item in parsed:
                        task_id = f"task_{item.get('task_id', item.get('id', len(dependencies) + 1))}"
                        deps = item.get('depends_on', item.get('dependencies', []))
                        if isinstance(deps, str):
                            deps = [d.strip() for d in deps.split(',') if d.strip() and d.lower() != 'none']
                        dependencies[task_id] = [f"task_{d}" if not str(d).startswith('task_') else str(d) for d in deps]
                    if dependencies:
                        return dependencies
                elif isinstance(parsed, dict):
                    for task_id, deps in parsed.items():
                        if isinstance(deps, str):
                            deps = [d.strip() for d in deps.split(',') if d.strip() and d.lower() != 'none']
                        dependencies[task_id] = deps
                    if dependencies:
                        return dependencies
        except (json.JSONDecodeError, TypeError):
            pass

        # Try Python literal
        try:
            import ast
            parsed = ast.literal_eval(dependencies_graph)
            if isinstance(parsed, (list, dict)):
                # Similar processing as JSON
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            task_id = f"task_{item.get('task_id', item.get('id', len(dependencies) + 1))}"
                            deps = item.get('depends_on', item.get('dependencies', []))
                            dependencies[task_id] = [f"task_{d}" if not str(d).startswith('task_') else str(d) for d in (deps if isinstance(deps, list) else [])]
                    if dependencies:
                        return dependencies
        except (ValueError, SyntaxError):
            pass

        # Fallback: Text format
        dep_entries = re.split(r'\n(?=TASK_ID:)', dependencies_graph)

        for entry in dep_entries:
            if not entry.strip() or not entry.startswith("TASK_ID:"):
                continue

            task_id_match = re.search(r'TASK_ID:\s*([^|]+)', entry)
            if not task_id_match:
                continue

            task_id = task_id_match.group(1).strip()

            deps_match = re.search(r'DEPENDS_ON:\s*([^|]+)', entry)
            if deps_match:
                deps_str = deps_match.group(1).strip()
                if deps_str.lower() in ['none', 'null', '']:
                    dependencies[task_id] = []
                else:
                    deps = [d.strip() for d in deps_str.split(',') if d.strip()]
                    dependencies[task_id] = deps
            else:
                dependencies[task_id] = []

        return dependencies

    def _detect_cycles(self, todo: SwarmTaskBoard) -> bool:
        """Detect cycles in the DAG."""
        visited = set()
        rec_stack = set()

        def dfs(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)

            task = todo.subtasks.get(task_id)
            if task:
                for dep_id in task.depends_on:
                    if dep_id not in visited:
                        if dfs(dep_id):
                            return True
                    elif dep_id in rec_stack:
                        return True

            rec_stack.remove(task_id)
            return False

        for task_id in todo.subtasks:
            if task_id not in visited:
                if dfs(task_id):
                    return True
        return False

    def _fix_cycles(self, todo: SwarmTaskBoard) -> SwarmTaskBoard:
        """Remove dependencies that cause cycles."""
        # Simple fix: remove back-edges
        for task_id, task in todo.subtasks.items():
            task.depends_on = [
                dep for dep in task.depends_on
                if dep in todo.subtasks
            ]
        return todo

    def _aggregate_tasks(self, todo: SwarmTaskBoard, max_tasks: int) -> SwarmTaskBoard:
        """
        Aggregate tasks to reduce count to max_tasks.

        Groups tasks by type and merges them.
        """
        if len(todo.subtasks) <= max_tasks:
            return todo

        # Group tasks by type
        type_groups: Dict[str, List[str]] = {}
        for task_id, task in todo.subtasks.items():
            task_type = task.intermediary_values.get("task_type", "implementation")
            if task_type not in type_groups:
                type_groups[task_type] = []
            type_groups[task_type].append(task_id)

        # Calculate how many tasks to keep per type
        total_tasks = len(todo.subtasks)
        tasks_per_type = max(1, max_tasks // len(type_groups)) if type_groups else max_tasks

        # Create new aggregated todo
        new_todo = SwarmTaskBoard(root_task=todo.root_task)
        merged_count = 0

        for task_type, task_ids in type_groups.items():
            if len(task_ids) <= tasks_per_type:
                # Keep as-is
                for tid in task_ids:
                    task = todo.subtasks[tid]
                    new_todo.add_task(
                        task_id=tid,
                        description=task.description,
                        actor=task.actor,
                        depends_on=task.depends_on,
                        priority=task.priority
                    )
                    new_todo.subtasks[tid].intermediary_values = task.intermediary_values.copy()
            else:
                # Merge into chunks
                chunk_size = max(1, len(task_ids) // tasks_per_type)
                chunks = [task_ids[i:i + chunk_size] for i in range(0, len(task_ids), chunk_size)]

                for i, chunk in enumerate(chunks[:tasks_per_type]):
                    merged_id = f"{task_type}_{i+1}"
                    descriptions = [todo.subtasks[tid].description for tid in chunk]
                    merged_desc = f"[{task_type.upper()}] " + "; ".join(descriptions)[:500]

                    # Collect dependencies (external to this chunk)
                    chunk_set = set(chunk)
                    all_deps = set()
                    for tid in chunk:
                        for dep in todo.subtasks[tid].depends_on:
                            if dep not in chunk_set:
                                all_deps.add(dep)

                    new_todo.add_task(
                        task_id=merged_id,
                        description=merged_desc,
                        actor="",
                        depends_on=list(all_deps),
                        priority=max(todo.subtasks[tid].priority for tid in chunk)
                    )
                    new_todo.subtasks[merged_id].intermediary_values["task_type"] = task_type
                    new_todo.subtasks[merged_id].intermediary_values["merged_from"] = chunk
                    merged_count += len(chunk) - 1

        logger.info(f"[TASK BREAKDOWN] Aggregated: {total_tasks} -> {len(new_todo.subtasks)} tasks")
        return new_todo

    def _get_execution_stages(self, todo: SwarmTaskBoard) -> List[List[str]]:
        """Get tasks grouped by execution stage."""
        stages = []
        completed = set()
        remaining = set(todo.subtasks.keys())

        while remaining:
            stage = [
                task_id for task_id in remaining
                if all(dep in completed for dep in todo.subtasks[task_id].depends_on)
            ]
            if not stage:
                break
            stages.append(stage)
            completed.update(stage)
            remaining -= set(stage)

        return stages

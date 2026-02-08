"""
DAG Agents - Task Breakdown and Actor Assignment for Jotty Swarm

Integrates:
- TaskBreakdownAgent: Converts implementation plans into executable DAG workflows
- TodoCreatorAgent: Validates DAGs, assigns actors, collapses consecutive tasks

Uses Jotty's SHARED infrastructure:
- SharedContext: Common taskboard for all agents
- HierarchicalMemory: SHARED 5-level memory (not per-agent)
- SmartAgentSlack: Inter-agent communication bus
- TDLambdaLearner: Shared learning from execution outcomes

Refactored to use BaseAgent hierarchy (Feb 2026).

Author: A-Team
Date: February 2026
"""

from __future__ import annotations

import logging
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import dspy

# Jotty core imports
from ..orchestration.v2.swarm_roadmap import (
    SubtaskState, MarkovianTODO, TaskStatus, AgenticState, TrajectoryStep
)
from ..foundation.data_structures import JottyConfig, MemoryLevel
from ..memory.cortex import HierarchicalMemory
from ..learning.learning import TDLambdaLearner, AdaptiveLearningRate
from ..persistence.shared_context import SharedContext
from .axon import SmartAgentSlack, MessageBus

# Import base class infrastructure
from .base import BaseAgent, AgentConfig, AgentResult, DomainAgent, DomainAgentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# BASE AGENT MIXIN FOR DAG AGENTS
# =============================================================================

class DAGAgentMixin:
    """
    Mixin providing BaseAgent-compatible infrastructure to DAG agents.

    Delegates to BaseAgent infrastructure when possible:
    - Metrics tracking (mirrors BaseAgent._metrics)
    - Pre/post execution hooks (mirrors BaseAgent hooks)
    - Unified logging
    - Error handling patterns

    Used by TaskBreakdownAgent (dspy.Module + mixin) and TodoCreatorAgent.

    Note: This mixin exists because DSPy agents need to inherit from dspy.Module
    for the forward() method, but we still want BaseAgent's infrastructure.
    """

    def _init_agent_infrastructure(self, name: str):
        """Initialize BaseAgent-compatible infrastructure."""
        # Use BaseAgent's config structure
        self._agent_config = AgentConfig(
            name=name,
            enable_memory=True,
            enable_context=True,
        )
        # Mirror BaseAgent's metrics structure
        self._metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_retries": 0,  # Added for BaseAgent compatibility
            "total_execution_time": 0.0,
        }
        self._pre_hooks = []
        self._post_hooks = []
        self._initialized = False

    def add_pre_hook(self, hook):
        """Add a pre-execution hook (BaseAgent-compatible)."""
        self._pre_hooks.append(hook)

    def add_post_hook(self, hook):
        """Add a post-execution hook (BaseAgent-compatible)."""
        self._post_hooks.append(hook)

    def _run_pre_hooks(self, **kwargs):
        """Run all pre-execution hooks (sync version for DSPy forward())."""
        for hook in self._pre_hooks:
            try:
                hook(self, **kwargs)
            except Exception as e:
                logger.warning(f"Pre-hook failed: {e}")

    def _run_post_hooks(self, result, **kwargs):
        """Run all post-execution hooks (sync version for DSPy forward())."""
        for hook in self._post_hooks:
            try:
                hook(self, result, **kwargs)
            except Exception as e:
                logger.warning(f"Post-hook failed: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent execution metrics (BaseAgent-compatible format)."""
        metrics = self._metrics.copy()
        if metrics["total_executions"] > 0:
            metrics["success_rate"] = (
                metrics["successful_executions"] / metrics["total_executions"]
            )
            metrics["avg_execution_time"] = (
                metrics["total_execution_time"] / metrics["total_executions"]
            )
        else:
            metrics["success_rate"] = 0.0
            metrics["avg_execution_time"] = 0.0
        return metrics

    def reset_metrics(self):
        """Reset all metrics to zero (BaseAgent-compatible)."""
        self._metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_retries": 0,
            "total_execution_time": 0.0,
        }

    def _track_execution(self, success: bool, execution_time: float):
        """Track execution metrics."""
        self._metrics["total_executions"] += 1
        if success:
            self._metrics["successful_executions"] += 1
        else:
            self._metrics["failed_executions"] += 1
        self._metrics["total_execution_time"] += execution_time

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent state (BaseAgent-compatible)."""
        return {
            "name": self._agent_config.name,
            "class": self.__class__.__name__,
            "metrics": self.get_metrics(),
            "initialized": getattr(self, '_initialized', False),
        }


# =============================================================================
# SHARED SWARM RESOURCES (Singleton pattern for true sharing)
# =============================================================================

class SwarmResources:
    """
    Singleton container for shared swarm resources.

    All DAG agents share:
    - memory: Common knowledge base
    - context: Shared taskboard/state
    - bus: Inter-agent communication
    - learner: Shared learning from outcomes
    """
    _instance = None

    def __new__(cls, config: JottyConfig = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: JottyConfig = None):
        if self._initialized:
            return

        self.config = config or JottyConfig()

        # Shared memory - ALL agents use this
        self.memory = HierarchicalMemory(
            config=self.config,
            agent_name="SwarmShared"  # Single shared instance
        )

        # Shared context (taskboard)
        self.context = SharedContext()

        # Shared message bus
        self.bus = MessageBus()

        # Shared learner
        adaptive_lr = AdaptiveLearningRate(self.config)
        self.learner = TDLambdaLearner(
            config=self.config,
            adaptive_lr=adaptive_lr
        )

        self._initialized = True
        logger.info("🔗 SwarmResources initialized (shared memory, context, bus, learner)")

    @classmethod
    def get_instance(cls, config: JottyConfig = None) -> 'SwarmResources':
        """Get or create the singleton instance."""
        return cls(config)

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        cls._instance = None


# =============================================================================
# TASK TYPES (Extended from Jotty)
# =============================================================================

class TaskType(Enum):
    """Task type classification for DAG nodes."""
    SETUP = "setup"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    EXECUTION = "execution"
    DOCUMENTATION = "documentation"
    VALIDATION = "validation"
    RESEARCH = "research"
    ANALYSIS = "analysis"


# =============================================================================
# DSPy SIGNATURES - Task Breakdown
# =============================================================================

class ExtractTasksSignature(dspy.Signature):
    """Extract PRODUCTION-QUALITY tasks from an implementation plan.

    You are a SENIOR ARCHITECT. Design tasks that produce WORLD-CLASS output.

    QUALITY PRINCIPLES:
    1. CONSOLIDATE: One unified app, not separate micro-apps
    2. ARCHITECTURE: Clean separation (models, services, UI, storage)
    3. UI/UX EXCELLENCE: Modern patterns, responsive, accessible, beautiful
    4. BEST PRACTICES: Type hints, error handling, logging, documentation

    TASK STRUCTURE (4-6 tasks max):
    1. Core Domain Models - All entities with validation, relationships, serialization
    2. Business Logic Layer - Services with CRUD, filtering, sorting, business rules
    3. Storage/Persistence - File/DB abstraction, caching, migrations
    4. User Interface - Modern UI with best UX patterns for the platform
    5. Integration Layer - Connect all components, dependency injection
    6. Quality Assurance - Tests, validation, error scenarios

    UI EXCELLENCE REQUIREMENTS:
    - Terminal UI: Use rich library, panels, tables, progress bars, live updates
    - Web UI: Modern CSS, responsive, dark mode, animations, accessibility
    - Desktop UI: Native look, keyboard shortcuts, system tray integration

    CONSOLIDATION RULES:
    - If multiple similar apps mentioned (Jira, Asana, Any.do), create ONE unified app
    - Combine best features from all: boards + projects + daily planning + recurring
    - Single comprehensive solution, not separate implementations

    OUTPUT FORMAT:
    TASK: <name> | TYPE: <type> | DESC: <detailed requirements with quality criteria> | FILES: <files>
    QUALITY: <specific quality requirements for this task>

    Example:
    TASK: Build unified task management UI | TYPE: implementation | DESC: Rich terminal UI with Kanban board view (Jira), project hierarchy (Asana), and daily planner sections (Any.do). Include drag-drop, keyboard navigation, color themes | FILES: ui.py
    QUALITY: Use rich library, support vim keybindings, 60fps animations, <100ms response time
    """
    implementation_plan: str = dspy.InputField(desc="The implementation plan - consolidate similar requirements into ONE unified solution")

    tasks_list: str = dspy.OutputField(
        desc="4-6 PRODUCTION-QUALITY tasks. Consolidate similar apps. Include quality criteria for each."
    )


class IdentifyDependenciesSignature(dspy.Signature):
    """Identify dependencies between tasks.

    You are a DEPENDENCY ANALYZER. Examine the tasks and identify which tasks
    depend on other tasks. A task depends on another if it needs the output
    or side effects of that task.

    RULES:
    1. Setup tasks typically have no dependencies
    2. Implementation tasks may depend on setup or other implementation
    3. Testing tasks depend on the code they test
    4. Validation tasks depend on implementation and testing

    OUTPUT FORMAT:
    TASK_ID: task_1 | DEPENDS_ON: none
    TASK_ID: task_2 | DEPENDS_ON: task_1
    TASK_ID: task_3 | DEPENDS_ON: task_1, task_2
    """
    tasks_list: str = dspy.InputField(desc="List of tasks extracted from plan")

    dependencies_graph: str = dspy.OutputField(
        desc="Dependency graph in format: TASK_ID: id | DEPENDS_ON: dep1, dep2"
    )


class OptimizeWorkflowSignature(dspy.Signature):
    """Optimize the workflow for parallel execution.

    You are a WORKFLOW OPTIMIZER. Analyze the tasks and dependencies to:
    1. Identify tasks that can run in parallel
    2. Suggest optimal execution order
    3. Flag potential bottlenecks

    OUTPUT FORMAT:
    STAGE 1 (parallel): task_1, task_2
    STAGE 2 (sequential): task_3
    STAGE 3 (parallel): task_4, task_5
    BOTTLENECK: task_3 blocks 2 downstream tasks
    """
    tasks_with_dependencies: str = dspy.InputField(
        desc="Tasks with their dependencies"
    )

    optimized_workflow: str = dspy.OutputField(
        desc="Optimized workflow with stages and parallel execution groups"
    )


# =============================================================================
# DSPy SIGNATURES - Actor Assignment
# =============================================================================

class ActorAssignmentSignature(dspy.Signature):
    """Assign the best actor to a task based on capabilities.

    You are an ACTOR ASSIGNER. Given a task and available actors with their
    capabilities, select the most suitable actor.

    Consider:
    1. Actor capabilities vs task requirements
    2. Current actor workload (prefer balanced distribution)
    3. Actor specialization (coding vs research vs testing)

    IMPORTANT: Return the EXACT actor name from available_actors.
    """
    task_id: str = dspy.InputField(desc="Task identifier")
    task_name: str = dspy.InputField(desc="Task name/title")
    task_type: str = dspy.InputField(desc="Task type: setup, implementation, testing, etc.")
    task_description: str = dspy.InputField(desc="Detailed task description")
    available_actors: str = dspy.InputField(desc="JSON list of actors with capabilities")
    current_assignments: str = dspy.InputField(desc="Current actor workload/assignments")

    assigned_actor_name: str = dspy.OutputField(
        desc="Name of the assigned actor (must match exactly from available_actors)"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why this actor was chosen"
    )


class DAGValidationSignature(dspy.Signature):
    """Validate DAG structure and assignments.

    You are a DAG VALIDATOR. Check for:
    1. Circular dependencies (cycles)
    2. Missing dependencies (tasks depending on non-existent tasks)
    3. Unassigned tasks
    4. Feasibility issues (actor can't handle assigned task)

    OUTPUT:
    - is_valid: YES or NO
    - issues_found: List of issues or "none"
    - recommendations: Suggestions for fixing issues
    """
    dag_summary: str = dspy.InputField(desc="DAG statistics and structure summary")
    tasks_info: str = dspy.InputField(desc="JSON of all tasks with dependencies")
    assignments_info: str = dspy.InputField(desc="JSON of task-to-actor assignments")
    cycle_check: str = dspy.InputField(desc="Result of cycle detection")

    is_valid: str = dspy.OutputField(desc="YES or NO")
    issues_found: str = dspy.OutputField(desc="List of issues or 'none'")
    recommendations: str = dspy.OutputField(desc="Suggestions for improvement")


class OptimizeDAGSignature(dspy.Signature):
    """Optimize DAG by removing unnecessary tasks.

    You are a DAG OPTIMIZER. Identify tasks that should be removed:
    1. Tasks that can't be executed by available capabilities
    2. Redundant or duplicate tasks
    3. Tasks outside the core implementation scope

    OUTPUT FORMAT:
    KEEP: task_1 | REASON: Core implementation task
    REMOVE: task_5 | REASON: Docker setup not needed for local dev
    """
    tasks_summary: str = dspy.InputField(desc="Summary of all tasks")
    available_capabilities: str = dspy.InputField(desc="Combined capabilities of all actors")

    optimization_plan: str = dspy.OutputField(
        desc="List of KEEP/REMOVE decisions with reasons"
    )


# =============================================================================
# ACTOR REPRESENTATION
# =============================================================================

@dataclass
class Actor:
    """Represents an actor/agent that can execute tasks."""
    name: str
    capabilities: List[str]  # e.g., ["git", "coding", "web_search", "testing"]
    description: Optional[str] = None
    max_concurrent_tasks: int = 1

    def can_handle(self, task_type: str) -> bool:
        """Check if actor has capability for task type."""
        type_to_capability = {
            "setup": ["setup", "coding", "git"],
            "implementation": ["coding", "implementation"],
            "testing": ["testing", "coding"],
            "execution": ["execution", "coding"],
            "documentation": ["documentation", "writing"],
            "validation": ["validation", "testing", "review"],
            "research": ["research", "web_search"],
            "analysis": ["analysis", "research"],
        }
        required = type_to_capability.get(task_type, [task_type])
        return any(cap in self.capabilities for cap in required)


# =============================================================================
# EXECUTABLE DAG (Enhanced with Jotty integration)
# =============================================================================

@dataclass
class ExecutableDAG:
    """
    DAG with actor assignments and validation results.

    Integrates with Jotty's MarkovianTODO for execution tracking.
    """
    markovian_todo: MarkovianTODO
    assignments: Dict[str, Actor]  # task_id -> Actor
    validation_passed: bool
    validation_issues: List[str] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)

    # Execution tracking (from AgenticState)
    trajectory: List[TrajectoryStep] = field(default_factory=list)

    @property
    def total_tasks(self) -> int:
        return len(self.markovian_todo.subtasks)

    def get_execution_stages(self) -> List[List[str]]:
        """Get tasks grouped by execution stage (parallel groups)."""
        stages = []
        completed = set()
        remaining = set(self.markovian_todo.subtasks.keys())

        while remaining:
            # Find tasks with all dependencies satisfied
            stage = [
                task_id for task_id in remaining
                if all(dep in completed for dep in self.markovian_todo.subtasks[task_id].depends_on)
            ]
            if not stage:
                # Cycle detected or stuck
                break
            stages.append(stage)
            completed.update(stage)
            remaining -= set(stage)

        return stages

    def add_trajectory_step(
        self,
        task_id: str,
        action_type: str,
        action_content: str,
        observation: str,
        reward: float
    ):
        """Record an execution step."""
        step = TrajectoryStep(
            step_idx=len(self.trajectory),
            timestamp=datetime.now(),
            action_type=action_type,
            action_content=action_content,
            context_summary=f"Task: {task_id}",
            activated_memories=[],
            observation=observation,
            reward=reward
        )
        self.trajectory.append(step)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "markovian_todo": {
                "todo_id": self.markovian_todo.todo_id,
                "root_task": self.markovian_todo.root_task,
                "subtasks": {
                    tid: {
                        "task_id": t.task_id,
                        "description": t.description,
                        "actor": t.actor,
                        "status": t.status.value,
                        "depends_on": t.depends_on,
                        "priority": t.priority,
                        "estimated_reward": t.estimated_reward,
                    }
                    for tid, t in self.markovian_todo.subtasks.items()
                },
                "execution_order": self.markovian_todo.execution_order,
            },
            "assignments": {
                tid: {"name": a.name, "capabilities": a.capabilities}
                for tid, a in self.assignments.items()
            },
            "validation_passed": self.validation_passed,
            "validation_issues": self.validation_issues,
            "fixes_applied": self.fixes_applied,
            "trajectory": [
                {
                    "step_idx": s.step_idx,
                    "timestamp": s.timestamp.isoformat(),
                    "action_type": s.action_type,
                    "action_content": s.action_content,
                    "observation": s.observation,
                    "reward": s.reward,
                }
                for s in self.trajectory
            ]
        }


# =============================================================================
# TASK BREAKDOWN AGENT
# =============================================================================

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

    def __init__(self, config: Optional[JottyConfig] = None):
        super().__init__()
        self.jotty_config = config or JottyConfig()

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

        logger.info("✓ TaskBreakdownAgent initialized with SHARED swarm resources")

    async def execute(self, implementation_plan: str, **kwargs) -> AgentResult:
        """
        Execute task breakdown with BaseAgent-compatible interface.

        Args:
            implementation_plan: Complete implementation plan as string

        Returns:
            AgentResult with MarkovianTODO as output
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

    def _handle_message(self, message):
        """Handle incoming messages from other agents."""
        logger.debug(f"TaskBreakdownAgent received: {message}")

    def forward(self, implementation_plan: str) -> MarkovianTODO:
        """
        Break down implementation plan into executable DAG workflow.

        Args:
            implementation_plan: Complete implementation plan as string

        Returns:
            MarkovianTODO: Markovian task manager with all tasks
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

        # Step 4: Build MarkovianTODO
        logger.info("[TASK BREAKDOWN] Step 4: Building MarkovianTODO...")
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

        logger.info(f"[TASK BREAKDOWN] Created MarkovianTODO with {len(markovian_todo.subtasks)} tasks (shared)")
        return markovian_todo

    def _build_markovian_todo(
        self,
        tasks_list: str,
        dependencies_graph: str,
        optimized_workflow: str,
        original_plan: str
    ) -> MarkovianTODO:
        """Build MarkovianTODO from parsed task information."""
        todo = MarkovianTODO(root_task=original_plan[:100])

        # Parse tasks
        tasks_dict = self._parse_tasks(tasks_list)

        # Parse dependencies
        dependencies_dict = self._parse_dependencies(dependencies_graph)

        # Add tasks to MarkovianTODO
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

    def _detect_cycles(self, todo: MarkovianTODO) -> bool:
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

    def _fix_cycles(self, todo: MarkovianTODO) -> MarkovianTODO:
        """Remove dependencies that cause cycles."""
        # Simple fix: remove back-edges
        for task_id, task in todo.subtasks.items():
            task.depends_on = [
                dep for dep in task.depends_on
                if dep in todo.subtasks
            ]
        return todo

    def _aggregate_tasks(self, todo: MarkovianTODO, max_tasks: int) -> MarkovianTODO:
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
        new_todo = MarkovianTODO(root_task=todo.root_task)
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

    def _get_execution_stages(self, todo: MarkovianTODO) -> List[List[str]]:
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


# =============================================================================
# TODO CREATOR AGENT
# =============================================================================

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

    def __init__(self, config: Optional[JottyConfig] = None, lm: Optional[dspy.LM] = None):
        self.jotty_config = config or JottyConfig()
        self.lm = lm or getattr(dspy.settings, 'lm', None)

        # Initialize BaseAgent infrastructure via mixin
        self._init_agent_infrastructure("TodoCreatorAgent")

        if self.lm is None:
            raise RuntimeError(
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

        logger.info("✓ TodoCreatorAgent initialized with SHARED swarm resources")

    def _handle_message(self, message):
        """Handle incoming messages from other agents."""
        logger.info(f"TodoCreatorAgent received from {message.from_agent}: {message.data}")

    async def execute(
        self,
        markovian_todo: MarkovianTODO,
        available_actors: List[Dict[str, Any]],
        **kwargs
    ) -> AgentResult:
        """
        Execute DAG creation with BaseAgent-compatible interface.

        Args:
            markovian_todo: MarkovianTODO with tasks
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
        markovian_todo: MarkovianTODO,
        available_actors: List[Dict[str, Any]]
    ) -> ExecutableDAG:
        """
        Create an executable DAG with actor assignments and validation.

        Args:
            markovian_todo: MarkovianTODO with tasks
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
            logger.info("✓ DAG validation passed")
        else:
            logger.warning(f"⚠ DAG validation found {len(issues)} issues")
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
        todo: MarkovianTODO,
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

    def _has_internal_dependencies(self, todo: MarkovianTODO, task_ids: List[str]) -> bool:
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
        todo: MarkovianTODO,
        assignments: Dict[str, Actor]
    ) -> Tuple[MarkovianTODO, Dict[str, Actor]]:
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

    def _merge_task_group(
        self,
        todo: MarkovianTODO,
        task_ids: List[str],
        actor: Actor,
        new_assignments: Dict[str, Actor],
        tasks_to_remove: set
    ):
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

    def _get_topological_order(self, todo: MarkovianTODO) -> List[str]:
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
        todo: MarkovianTODO,
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

    def update_from_execution(
        self,
        executable_dag: ExecutableDAG,
        outcomes: Dict[str, bool]
    ):
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
            output.append(f"\n👤 {actor_name}")
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

def get_swarm_resources(config: Optional[JottyConfig] = None) -> SwarmResources:
    """Get the shared swarm resources singleton."""
    return SwarmResources.get_instance(config)


def create_task_breakdown_agent(config: Optional[JottyConfig] = None) -> TaskBreakdownAgent:
    """Factory function to create TaskBreakdownAgent with shared resources."""
    return TaskBreakdownAgent(config=config)


def create_todo_creator_agent(
    config: Optional[JottyConfig] = None,
    lm: Optional[dspy.LM] = None
) -> TodoCreatorAgent:
    """Factory function to create TodoCreatorAgent with shared resources."""
    return TodoCreatorAgent(config=config, lm=lm)


def reset_swarm_resources():
    """Reset shared resources (for testing)."""
    SwarmResources.reset()
    logger.info("🔄 SwarmResources reset")

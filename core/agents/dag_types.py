"""DAG Agent types — enums, signatures, data structures."""

from __future__ import annotations


"""DAG Agents — extracted module."""


import logging
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import dspy

from ..orchestration.v2.swarm_roadmap import (
    SubtaskState, MarkovianTODO, TaskStatus, AgenticState, TrajectoryStep
)
from ..foundation.data_structures import JottyConfig, MemoryLevel
from ..memory.cortex import HierarchicalMemory
from ..learning.learning import TDLambdaLearner, AdaptiveLearningRate
from ..persistence.shared_context import SharedContext
from .axon import SmartAgentSlack, MessageBus
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


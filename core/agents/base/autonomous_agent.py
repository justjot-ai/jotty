"""
AutonomousAgent - Open-Ended Problem Solver Base Class

Base class for agents that autonomously solve complex, multi-step tasks:
- Skill discovery via SkillsRegistry
- AgenticPlanner integration for execution planning
- Multi-step execution with dependency handling
- Adaptive replanning on failure

Used by:
- AutoAgent: General-purpose autonomous executor

Author: A-Team
Date: February 2026
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from .base_agent import BaseAgent, AgentConfig, AgentResult
from .skill_plan_executor import SkillPlanExecutor

logger = logging.getLogger(__name__)


# =============================================================================
# AUTONOMOUS AGENT CONFIG
# =============================================================================

@dataclass
class AutonomousAgentConfig(AgentConfig):
    """Configuration specific to AutonomousAgent."""
    max_steps: int = 10
    enable_replanning: bool = True
    max_replans: int = 3
    skill_filter: Optional[str] = None
    default_output_skill: Optional[str] = None
    enable_output: bool = False
    enable_ensemble: bool = False
    ensemble_strategy: str = "multi_perspective"


# =============================================================================
# EXECUTION STEP
# =============================================================================

@dataclass
class ExecutionStep:
    """A step in the execution plan."""
    skill_name: str
    tool_name: str
    params: Dict[str, Any]
    description: str
    depends_on: List[int] = field(default_factory=list)
    output_key: str = ""
    optional: bool = False


# =============================================================================
# AUTONOMOUS AGENT
# =============================================================================

class AutonomousAgent(BaseAgent):
    """
    Base class for autonomous, open-ended problem solving.

    Provides infrastructure for:
    - Skill discovery and selection (via BaseAgent.discover_skills + SkillPlanExecutor)
    - Execution planning via AgenticPlanner
    - Multi-step execution with dependency resolution
    - Adaptive replanning on failures

    Delegates planning/execution mechanics to SkillPlanExecutor while keeping
    the main orchestration loop (with replanning) here since it's specific
    to autonomous mode.

    Subclasses (AutoAgent) can customize:
    - Task type inference
    - Skill discovery strategy
    - Execution behavior
    """

    def __init__(self, config: AutonomousAgentConfig = None):
        """
        Initialize AutonomousAgent.

        Args:
            config: Agent configuration
        """
        config = config or AutonomousAgentConfig(name=self.__class__.__name__)
        # Enable skills for autonomous agents
        config.enable_skills = True
        super().__init__(config)

        # Lazy-loaded components
        self._planner = None
        self._executor = None

        # Execution state
        self._excluded_skills: Set[str] = set()

    def _ensure_initialized(self):
        """Initialize planner, executor, and skills registry."""
        super()._ensure_initialized()

        if self._planner is None:
            try:
                from ..agentic_planner import AgenticPlanner
                self._planner = AgenticPlanner()
                logger.debug("Initialized AgenticPlanner")
            except Exception as e:
                logger.warning(f"Could not initialize AgenticPlanner: {e}")

    @property
    def planner(self):
        """Get the AgenticPlanner instance."""
        self._ensure_initialized()
        return self._planner

    @property
    def executor(self) -> SkillPlanExecutor:
        """Lazy-load SkillPlanExecutor, sharing the agent's planner instance."""
        if self._executor is None:
            config: AutonomousAgentConfig = self.config
            self._executor = SkillPlanExecutor(
                skills_registry=self.skills_registry,
                max_steps=config.max_steps,
                enable_replanning=config.enable_replanning,
                max_replans=config.max_replans,
                planner=self._planner,
            )
        return self._executor

    # =========================================================================
    # SKILL DISCOVERY (autonomous-specific filtering)
    # =========================================================================

    async def _discover_skills(self, task: str) -> List[Dict[str, Any]]:
        """
        Discover relevant skills with autonomous-specific filtering.

        Uses BaseAgent.discover_skills() for keyword matching, then applies
        excluded_skills and category filter on top.

        Args:
            task: Task description

        Returns:
            List of skill dicts with name, description, tools
        """
        config: AutonomousAgentConfig = self.config

        # Use BaseAgent's keyword-based discovery
        all_skills = self.discover_skills(task)

        # Apply autonomous-specific filters
        filtered = []
        for skill in all_skills:
            # Skip excluded skills
            if skill['name'] in self._excluded_skills:
                continue
            # Apply category filter if set
            if config.skill_filter and skill.get('category', 'general') != config.skill_filter:
                continue
            filtered.append(skill)

        return filtered

    async def _select_skills(
        self,
        task: str,
        available_skills: List[Dict[str, Any]],
        max_skills: int = 8,
    ) -> List[Dict[str, Any]]:
        """Select best skills using the executor's planner."""
        return await self.executor.select_skills(task, available_skills, max_skills)

    # =========================================================================
    # EXECUTION PLANNING (delegates to executor)
    # =========================================================================

    async def _create_plan(
        self,
        task: str,
        task_type: str,
        skills: List[Dict[str, Any]],
        previous_outputs: Optional[Dict[str, Any]] = None,
    ) -> list:
        """Create an execution plan via SkillPlanExecutor."""
        return await self.executor.create_plan(task, task_type, skills, previous_outputs)

    # =========================================================================
    # STEP EXECUTION (delegates to executor)
    # =========================================================================

    async def _execute_step(
        self,
        step,
        outputs: Dict[str, Any],
        status_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Execute a single step via SkillPlanExecutor."""
        return await self.executor.execute_step(step, outputs, status_callback)

    # =========================================================================
    # MAIN EXECUTION (orchestration loop with replanning)
    # =========================================================================

    async def _execute_impl(self, task: str = "", **kwargs) -> Dict[str, Any]:
        """
        Execute an autonomous task.

        Orchestrates the full pipeline: infer task type -> discover skills ->
        select skills -> create plan -> execute steps (with replanning).

        This orchestration loop stays in AutonomousAgent because it includes
        autonomous-specific behavior like replanning and excluded_skills
        management. The individual steps delegate to SkillPlanExecutor.

        Args:
            task: Task description
            **kwargs: Additional arguments

        Returns:
            Dict with execution results
        """
        config: AutonomousAgentConfig = self.config
        start_time = time.time()
        status_callback = kwargs.get('status_callback')

        def _status(stage: str, detail: str = ""):
            if status_callback:
                try:
                    status_callback(stage, detail)
                except Exception:
                    pass
            logger.info(f"{stage}: {detail}" if detail else stage)

        # Reset excluded skills for new execution
        self._excluded_skills.clear()
        self.executor.clear_exclusions()

        # Step 1: Infer task type
        _status("Analyzing", "inferring task type")
        task_type = self._infer_task_type(task)
        _status("Task type", task_type)

        # Step 2: Discover skills (with autonomous filtering)
        _status("Discovering", "finding relevant skills")
        all_skills = await self._discover_skills(task)
        _status("Skills found", f"{len(all_skills)} potential")

        # Step 3: Select best skills
        _status("Selecting", "choosing best skills")
        skills = await self._select_skills(task, all_skills)
        _status("Skills selected", ", ".join(s['name'] for s in skills[:5]))

        # Step 4: Create plan
        _status("Planning", "creating execution plan")
        steps = await self._create_plan(task, task_type, skills)
        _status("Plan ready", f"{len(steps)} steps")

        if not steps:
            return {
                "success": False,
                "task": task,
                "error": "No valid execution plan created",
                "skills_used": [],
                "steps_executed": 0,
                "outputs": {},
            }

        # Step 5: Execute steps
        outputs = {}
        skills_used = []
        errors = []
        replan_count = 0

        for i, step in enumerate(steps):
            _status(f"Step {i+1}/{len(steps)}", f"{step.skill_name}: {step.description[:50]}")

            result = await self._execute_step(step, outputs, status_callback)

            if result.get('success'):
                outputs[step.output_key or f'step_{i}'] = result
                skills_used.append(step.skill_name)
                _status(f"Step {i+1}", "completed")
            else:
                error_msg = result.get('error', 'Unknown error')
                errors.append(f"Step {i+1}: {error_msg}")
                _status(f"Step {i+1}", f"failed: {error_msg}")

                # Try replanning if enabled
                if (config.enable_replanning and
                        replan_count < config.max_replans and
                        not step.optional):

                    # Exclude failed skill if domain mismatch
                    if any(kw in error_msg.lower() for kw in
                           ['not found', '404', 'invalid', 'delisted']):
                        self._excluded_skills.add(step.skill_name)
                        self.executor.exclude_skill(step.skill_name)

                    _status("Replanning", "adapting to failure")
                    new_steps = await self._create_plan(
                        task, task_type, skills, outputs
                    )
                    if new_steps:
                        steps = steps[:i+1] + new_steps
                        replan_count += 1
                        _status("Replanned", f"{len(new_steps)} new steps")

        # Determine final output
        final_output = list(outputs.values())[-1] if outputs else None

        return {
            "success": len(outputs) > 0,
            "task": task,
            "task_type": task_type,
            "skills_used": list(set(skills_used)),
            "steps_executed": len(outputs),
            "outputs": outputs,
            "final_output": final_output,
            "errors": errors,
            "execution_time": time.time() - start_time,
        }

    def _infer_task_type(self, task: str) -> str:
        """
        Infer task type from description.

        Delegates to SkillPlanExecutor which uses planner + keyword fallback.
        Can be overridden by subclasses for custom inference.
        """
        return self.executor.infer_task_type(task)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_autonomous_agent(
    max_steps: int = 10,
    enable_replanning: bool = True,
    skill_filter: Optional[str] = None,
    model: str = "sonnet",
) -> AutonomousAgent:
    """
    Factory function to create an AutonomousAgent.

    Args:
        max_steps: Maximum execution steps
        enable_replanning: Enable adaptive replanning
        skill_filter: Optional skill category filter
        model: LLM model to use

    Returns:
        Configured AutonomousAgent
    """
    config = AutonomousAgentConfig(
        name="AutonomousAgent",
        model=model,
        max_steps=max_steps,
        enable_replanning=enable_replanning,
        skill_filter=skill_filter,
    )
    return AutonomousAgent(config)


__all__ = [
    'AutonomousAgent',
    'AutonomousAgentConfig',
    'ExecutionStep',
    'create_autonomous_agent',
]

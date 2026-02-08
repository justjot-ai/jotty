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

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from .base_agent import BaseAgent, AgentConfig, AgentResult

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
    - Skill discovery and selection
    - Execution planning via AgenticPlanner
    - Multi-step execution with dependency resolution
    - Adaptive replanning on failures

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

        # Execution state
        self._excluded_skills: Set[str] = set()

    def _ensure_initialized(self):
        """Initialize planner and skills registry."""
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

    # =========================================================================
    # SKILL DISCOVERY
    # =========================================================================

    async def _discover_skills(self, task: str) -> List[Dict[str, Any]]:
        """
        Discover relevant skills for the task.

        Args:
            task: Task description

        Returns:
            List of skill dicts with name, description, tools
        """
        if self.skills_registry is None:
            logger.warning("Skills registry not available")
            return []

        config: AutonomousAgentConfig = self.config
        task_lower = task.lower()

        # Stop words for better matching
        stop_words = {
            'the', 'and', 'for', 'with', 'how', 'what', 'are', 'is',
            'to', 'of', 'in', 'on', 'a', 'an'
        }
        task_words = [
            w for w in task_lower.split()
            if len(w) > 2 and w not in stop_words
        ]

        skills = []
        skill_names_added = set()

        # Search through registered skills
        for skill_name, skill_def in self.skills_registry.loaded_skills.items():
            # Skip excluded skills
            if skill_name in self._excluded_skills:
                continue

            # Apply category filter if set
            if config.skill_filter:
                category = getattr(skill_def, 'category', 'general')
                if category != config.skill_filter:
                    continue

            skill_name_lower = skill_name.lower()
            desc = getattr(skill_def, 'description', '') or ''
            desc_lower = desc.lower()

            # Calculate relevance score
            score = 0
            for word in task_words:
                if word in skill_name_lower:
                    score += 3  # Strong match
                if word in desc_lower:
                    score += 1  # Weak match

            if score > 0:
                tools = list(skill_def.tools.keys()) if hasattr(skill_def, 'tools') else []
                skills.append({
                    'name': skill_name,
                    'description': desc or skill_name,
                    'category': getattr(skill_def, 'category', 'general'),
                    'tools': tools,
                    'relevance_score': score
                })
                skill_names_added.add(skill_name)

        # Sort by relevance
        skills.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        # If no matches, return first N available skills
        if not skills:
            for skill_name, skill_def in list(self.skills_registry.loaded_skills.items())[:15]:
                if skill_name in self._excluded_skills:
                    continue
                desc = getattr(skill_def, 'description', '') or ''
                tools = list(skill_def.tools.keys()) if hasattr(skill_def, 'tools') else []
                skills.append({
                    'name': skill_name,
                    'description': desc or skill_name,
                    'category': getattr(skill_def, 'category', 'general'),
                    'tools': tools,
                    'relevance_score': 0
                })

        return skills[:15]

    async def _select_skills(
        self,
        task: str,
        available_skills: List[Dict[str, Any]],
        max_skills: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Select best skills using the planner.

        Args:
            task: Task description
            available_skills: List of discovered skills
            max_skills: Maximum skills to select

        Returns:
            List of selected skill dicts
        """
        if self._planner is None:
            return available_skills[:max_skills]

        try:
            selected, reasoning = self._planner.select_skills(
                task=task,
                available_skills=available_skills,
                max_skills=max_skills
            )
            logger.debug(f"Skill selection reasoning: {reasoning}")
            return selected
        except Exception as e:
            logger.warning(f"Skill selection failed: {e}")
            return available_skills[:max_skills]

    # =========================================================================
    # EXECUTION PLANNING
    # =========================================================================

    async def _create_plan(
        self,
        task: str,
        task_type: str,
        skills: List[Dict[str, Any]],
        previous_outputs: Optional[Dict[str, Any]] = None
    ) -> List[ExecutionStep]:
        """
        Create an execution plan using the planner.

        Args:
            task: Task description
            task_type: Inferred task type
            skills: Available skills
            previous_outputs: Outputs from previous steps

        Returns:
            List of ExecutionStep objects
        """
        if self._planner is None:
            logger.warning("No planner available, returning empty plan")
            return []

        config: AutonomousAgentConfig = self.config

        try:
            steps, reasoning = self._planner.plan_execution(
                task=task,
                task_type=task_type,
                skills=skills,
                previous_outputs=previous_outputs,
                max_steps=config.max_steps
            )
            logger.debug(f"Plan reasoning: {reasoning}")
            return steps
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return []

    # =========================================================================
    # STEP EXECUTION
    # =========================================================================

    async def _execute_step(
        self,
        step: ExecutionStep,
        outputs: Dict[str, Any],
        status_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Execute a single step.

        Args:
            step: ExecutionStep to execute
            outputs: Previous step outputs for param resolution
            status_callback: Optional progress callback

        Returns:
            Step result dict
        """
        if self.skills_registry is None:
            return {'success': False, 'error': 'Skills registry not available'}

        # Get skill
        skill = self.skills_registry.get_skill(step.skill_name)
        if not skill:
            return {
                'success': False,
                'error': f'Skill not found: {step.skill_name}'
            }

        # Get tool
        tool = skill.tools.get(step.tool_name) if hasattr(skill, 'tools') else None
        if not tool:
            return {
                'success': False,
                'error': f'Tool not found: {step.tool_name}'
            }

        # Resolve params with template variables
        resolved_params = self._resolve_params(step.params, outputs)

        # Execute tool
        try:
            if inspect.iscoroutinefunction(tool):
                result = await tool(resolved_params)
            else:
                result = tool(resolved_params)

            # Normalize result
            if result is None:
                result = {'success': False, 'error': 'Tool returned None'}
            elif not isinstance(result, dict):
                result = {'success': True, 'output': result}

            return result

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {'success': False, 'error': str(e)}

    def _resolve_params(
        self,
        params: Dict[str, Any],
        outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve template variables in params.

        Supports:
        - ${step_key.field}
        - {step_key.field}
        - {{step_key}}

        Args:
            params: Parameters with potential templates
            outputs: Previous outputs to resolve from

        Returns:
            Resolved parameters
        """
        import re

        resolved = {}

        for key, value in params.items():
            if isinstance(value, str):
                # Handle ${...} format
                pattern = r'\$\{([^}]+)\}'

                def replacer(match):
                    path = match.group(1)
                    return self._resolve_path(path, outputs)

                value = re.sub(pattern, replacer, value)

                # Handle {...} format
                pattern2 = r'\{([a-zA-Z_][a-zA-Z0-9_.\[\]]*)\}'
                value = re.sub(pattern2, replacer, value)

                resolved[key] = value

            elif isinstance(value, dict):
                resolved[key] = self._resolve_params(value, outputs)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_params(item, outputs)
                    if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                resolved[key] = value

        return resolved

    def _resolve_path(self, path: str, outputs: Dict[str, Any]) -> str:
        """Resolve a path like 'step_key.field' from outputs."""
        parts = path.split('.')
        value = outputs

        for part in parts:
            if value is None:
                return path  # Return original if unresolved

            # Handle array indexing
            if '[' in part:
                key = part.split('[')[0]
                try:
                    idx = int(part.split('[')[1].split(']')[0])
                    if isinstance(value, dict):
                        value = value.get(key, [])
                    if isinstance(value, list) and idx < len(value):
                        value = value[idx]
                    else:
                        return path
                except (ValueError, IndexError):
                    return path
            else:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return path

        if value is None:
            return path

        if isinstance(value, (dict, list)):
            import json
            return json.dumps(value, default=str)

        return str(value)

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================

    async def _execute_impl(self, task: str = "", **kwargs) -> Dict[str, Any]:
        """
        Execute an autonomous task.

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

        # Step 1: Infer task type
        _status("Analyzing", "inferring task type")
        task_type = self._infer_task_type(task)
        _status("Task type", task_type)

        # Step 2: Discover skills
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

        Can be overridden by subclasses for custom inference.
        """
        if self._planner is not None:
            try:
                task_type, reasoning, confidence = self._planner.infer_task_type(task)
                return task_type.value
            except Exception as e:
                logger.warning(f"Task type inference failed: {e}")

        # Fallback: keyword-based inference
        task_lower = task.lower()
        if any(kw in task_lower for kw in ['vs', 'compare', 'versus']):
            return 'comparison'
        elif any(kw in task_lower for kw in ['create', 'build', 'write', 'generate']):
            return 'creation'
        elif any(kw in task_lower for kw in ['research', 'find', 'search']):
            return 'research'
        elif any(kw in task_lower for kw in ['analyze', 'calculate', 'evaluate']):
            return 'analysis'
        else:
            return 'unknown'


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

"""
SkillPlanExecutor - Reusable Planning/Execution Service

Extracted from AutonomousAgent to enable any agent to use skill-based
planning and execution. Provides:
- LLM-based skill selection via AgenticPlanner
- Execution plan creation
- Multi-step execution with dependency handling
- Template variable resolution
- Task type inference

Usage:
    executor = SkillPlanExecutor(skills_registry)
    result = await executor.plan_and_execute(task, discovered_skills)

Author: A-Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SkillPlanExecutor:
    """
    Reusable planning and execution service.

    Encapsulates the skill selection, planning, and step execution
    pipeline that was originally inline in AutonomousAgent. Any agent
    can compose with this class to gain skill-based task execution.

    Args:
        skills_registry: Initialized SkillsRegistry instance
        max_steps: Maximum execution steps per plan
        enable_replanning: Whether to replan on step failure
        max_replans: Maximum number of replanning attempts
    """

    def __init__(
        self,
        skills_registry,
        max_steps: int = 10,
        enable_replanning: bool = True,
        max_replans: int = 3,
        planner=None,
    ):
        self._skills_registry = skills_registry
        self._max_steps = max_steps
        self._enable_replanning = enable_replanning
        self._max_replans = max_replans
        self._planner = planner
        self._excluded_skills: set = set()

    # =========================================================================
    # LAZY INITIALIZATION
    # =========================================================================

    @property
    def planner(self):
        """Lazy-load AgenticPlanner."""
        if self._planner is None:
            try:
                from ..agentic_planner import AgenticPlanner
                self._planner = AgenticPlanner()
                logger.debug("SkillPlanExecutor: Initialized AgenticPlanner")
            except Exception as e:
                logger.warning(f"SkillPlanExecutor: Could not initialize AgenticPlanner: {e}")
        return self._planner

    # =========================================================================
    # SKILL SELECTION
    # =========================================================================

    async def select_skills(
        self,
        task: str,
        available_skills: List[Dict[str, Any]],
        max_skills: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Select best skills for a task using the planner.

        Args:
            task: Task description
            available_skills: List of discovered skill dicts
            max_skills: Maximum skills to select

        Returns:
            List of selected skill dicts
        """
        if self.planner is None:
            return available_skills[:max_skills]

        try:
            selected, reasoning = self.planner.select_skills(
                task=task,
                available_skills=available_skills,
                max_skills=max_skills,
            )
            logger.debug(f"Skill selection reasoning: {reasoning}")
            return selected
        except Exception as e:
            logger.warning(f"Skill selection failed: {e}")
            return available_skills[:max_skills]

    # =========================================================================
    # EXECUTION PLANNING
    # =========================================================================

    async def create_plan(
        self,
        task: str,
        task_type: str,
        skills: List[Dict[str, Any]],
        previous_outputs: Optional[Dict[str, Any]] = None,
    ) -> list:
        """
        Create an execution plan using the planner.

        Args:
            task: Task description
            task_type: Inferred task type string
            skills: Available skills
            previous_outputs: Outputs from previous steps (for replanning)

        Returns:
            List of ExecutionStep objects
        """
        if self.planner is None:
            logger.warning("No planner available, returning empty plan")
            return []

        try:
            steps, reasoning = self.planner.plan_execution(
                task=task,
                task_type=task_type,
                skills=skills,
                previous_outputs=previous_outputs,
                max_steps=self._max_steps,
            )
            logger.debug(f"Plan reasoning: {reasoning}")
            return steps
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return []

    # =========================================================================
    # STEP EXECUTION
    # =========================================================================

    async def execute_step(
        self,
        step,
        outputs: Dict[str, Any],
        status_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single ExecutionStep.

        Args:
            step: ExecutionStep to execute
            outputs: Previous step outputs for param resolution
            status_callback: Optional progress callback

        Returns:
            Step result dict with 'success' key
        """
        def _status(stage: str, detail: str = ""):
            if status_callback:
                try:
                    status_callback(stage, detail)
                except Exception:
                    pass

        if self._skills_registry is None:
            return {'success': False, 'error': 'Skills registry not available'}

        _status("Loading", f"skill: {step.skill_name}")
        skill = self._skills_registry.get_skill(step.skill_name)
        if not skill:
            return {
                'success': False,
                'error': f'Skill not found: {step.skill_name}',
            }

        # If skill extends BaseSkill, set the status callback for live updates
        try:
            from ...registry.skills_registry import BaseSkill
            if isinstance(skill, BaseSkill):
                skill.set_status_callback(status_callback)
        except ImportError:
            pass

        tool = skill.tools.get(step.tool_name) if hasattr(skill, 'tools') else None

        # Fallback: if tool not found, try to find a matching tool or use first available
        if not tool and hasattr(skill, 'tools') and skill.tools:
            # Try to find tool with similar name
            tool_name_lower = step.tool_name.lower() if step.tool_name else ''
            for name, func in skill.tools.items():
                if tool_name_lower in name.lower() or name.lower() in tool_name_lower:
                    tool = func
                    logger.info(f"Tool '{step.tool_name}' not found, using similar: '{name}'")
                    break

            # Last resort: use first tool (most skills have one primary tool)
            if not tool:
                first_tool_name = list(skill.tools.keys())[0]
                tool = skill.tools[first_tool_name]
                logger.info(f"Tool '{step.tool_name}' not found, using first tool: '{first_tool_name}'")

        if not tool:
            return {
                'success': False,
                'error': f'Tool not found: {step.tool_name}',
            }

        resolved_params = self.resolve_params(step.params, outputs)

        # Pass status callback to skill via params (skills can optionally use it)
        if status_callback:
            resolved_params['_status_callback'] = status_callback

        # Emit status based on skill type
        skill_status_map = {
            'research': '📚 Researching...',
            'search': '🔍 Searching...',
            'web': '🌐 Fetching web data...',
            'pdf': '📄 Creating PDF...',
            'chart': '📊 Generating charts...',
            'telegram': '📨 Sending message...',
            'slack': '💬 Posting to Slack...',
            'file': '📁 Processing files...',
            'data': '📈 Analyzing data...',
            'stock': '💹 Fetching stock data...',
        }

        for key, msg in skill_status_map.items():
            if key in step.skill_name.lower() or key in step.tool_name.lower():
                _status("Executing", msg)
                break
        else:
            _status("Executing", f"🔧 Running {step.skill_name}...")

        try:
            if inspect.iscoroutinefunction(tool):
                result = await tool(resolved_params)
            else:
                result = tool(resolved_params)

            if result is None:
                result = {'success': False, 'error': 'Tool returned None'}
            elif not isinstance(result, dict):
                result = {'success': True, 'output': result}

            _status("Done", f"✓ {step.skill_name}")
            return result

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            _status("Error", f"✗ {step.skill_name}")
            return {'success': False, 'error': str(e)}

    # =========================================================================
    # PARAMETER RESOLUTION
    # =========================================================================

    def resolve_params(
        self,
        params: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Resolve template variables in parameters.

        Supports ${step_key.field}, {step_key.field}, {{step_key}} formats.

        Args:
            params: Parameters with potential template variables
            outputs: Previous outputs to resolve from

        Returns:
            Resolved parameters
        """
        import re

        resolved = {}

        for key, value in params.items():
            if isinstance(value, str):
                pattern = r'\$\{([^}]+)\}'

                def replacer(match):
                    path = match.group(1)
                    return self.resolve_path(path, outputs)

                value = re.sub(pattern, replacer, value)

                pattern2 = r'\{([a-zA-Z_][a-zA-Z0-9_.\[\]]*)\}'
                value = re.sub(pattern2, replacer, value)

                resolved[key] = value

            elif isinstance(value, dict):
                resolved[key] = self.resolve_params(value, outputs)
            elif isinstance(value, list):
                resolved[key] = [
                    self.resolve_params(item, outputs)
                    if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                resolved[key] = value

        return resolved

    def resolve_path(self, path: str, outputs: Dict[str, Any]) -> str:
        """
        Resolve a dotted path like 'step_key.field' from outputs.

        Args:
            path: Dotted path string (e.g. 'step_0.output.data')
            outputs: Dictionary of previous step outputs

        Returns:
            Resolved string value, or original path if unresolvable
        """
        parts = path.split('.')
        value = outputs

        for part in parts:
            if value is None:
                return path

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
    # TASK TYPE INFERENCE
    # =========================================================================

    def infer_task_type(self, task: str) -> str:
        """
        Infer task type from description using planner or keyword fallback.

        Args:
            task: Task description

        Returns:
            Task type string (e.g. 'research', 'creation', 'comparison')
        """
        if self.planner is not None:
            try:
                task_type, reasoning, confidence = self.planner.infer_task_type(task)
                return task_type.value
            except Exception as e:
                logger.warning(f"Task type inference failed: {e}")

        # Keyword fallback
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

    # =========================================================================
    # EXCLUSION MANAGEMENT
    # =========================================================================

    def exclude_skill(self, skill_name: str):
        """Add a skill to the exclusion set (e.g. after repeated failures)."""
        self._excluded_skills.add(skill_name)

    def clear_exclusions(self):
        """Clear all skill exclusions."""
        self._excluded_skills.clear()

    @property
    def excluded_skills(self) -> set:
        """Get the current set of excluded skill names."""
        return self._excluded_skills

    # =========================================================================
    # FULL PIPELINE
    # =========================================================================

    async def plan_and_execute(
        self,
        task: str,
        discovered_skills: List[Dict[str, Any]],
        status_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Full planning and execution pipeline entry point.

        Runs the complete flow: task type inference -> skill selection ->
        plan creation -> step execution (with optional replanning).

        Args:
            task: Task description
            discovered_skills: Pre-discovered skills (from BaseAgent.discover_skills)
            status_callback: Optional callback(stage, detail) for progress

        Returns:
            Dict with success, task, task_type, skills_used, steps_executed,
            outputs, final_output, errors, execution_time
        """
        start_time = time.time()

        def _status(stage: str, detail: str = ""):
            if status_callback:
                try:
                    status_callback(stage, detail)
                except Exception:
                    pass
            logger.info(f"{stage}: {detail}" if detail else stage)

        self._excluded_skills.clear()

        # Step 1: Infer task type
        _status("Analyzing", "inferring task type")
        task_type = self.infer_task_type(task)
        _status("Task type", task_type)

        # Step 2: Select best skills
        _status("Selecting", "choosing best skills")
        skills = await self.select_skills(task, discovered_skills)
        _status("Skills selected", ", ".join(s['name'] for s in skills[:5]))

        # Step 3: Create plan
        _status("Planning", "creating execution plan")
        steps = await self.create_plan(task, task_type, skills)
        _status("Plan ready", f"{len(steps)} steps")

        if not steps:
            return {
                "success": False,
                "task": task,
                "task_type": task_type,
                "error": "No valid execution plan created",
                "skills_used": [],
                "steps_executed": 0,
                "outputs": {},
                "final_output": None,
                "errors": ["No valid execution plan created"],
                "execution_time": time.time() - start_time,
            }

        # Step 4: Execute steps
        outputs = {}
        skills_used = []
        errors = []
        replan_count = 0

        for i, step in enumerate(steps):
            _status(
                f"Step {i + 1}/{len(steps)}",
                f"{step.skill_name}: {step.description[:50]}",
            )

            result = await self.execute_step(step, outputs, status_callback)

            if result.get('success'):
                outputs[step.output_key or f'step_{i}'] = result
                skills_used.append(step.skill_name)
                _status(f"Step {i + 1}", "completed")
            else:
                error_msg = result.get('error', 'Unknown error')
                errors.append(f"Step {i + 1}: {error_msg}")
                _status(f"Step {i + 1}", f"failed: {error_msg}")

                # Replan on failure with intelligent skill exclusion
                if (self._enable_replanning and
                        replan_count < self._max_replans and
                        not step.optional):

                    # Exclude skill for permanent/structural errors
                    exclusion_keywords = [
                        'not found', '404', 'invalid', 'delisted',
                        'not implemented', 'unsupported', 'deprecated',
                        'permission denied', 'unauthorized', 'forbidden',
                        'module not found', 'import error', 'no module',
                    ]
                    if any(kw in error_msg.lower() for kw in exclusion_keywords):
                        self._excluded_skills.add(step.skill_name)
                        logger.info(f"🚫 Excluded skill '{step.skill_name}' due to: {error_msg[:50]}")

                    # For transient errors, retry once before replanning
                    transient_keywords = ['timeout', 'connection', 'rate limit', 'retry']
                    if any(kw in error_msg.lower() for kw in transient_keywords) and replan_count == 0:
                        _status(f"Step {i + 1}", "retrying after transient error")
                        await asyncio.sleep(2)  # Brief backoff
                        retry_result = await self._execute_step(step, i, outputs)
                        if retry_result.get('success'):
                            outputs[step.output_key or f'step_{i}'] = retry_result
                            skills_used.append(step.skill_name)
                            _status(f"Step {i + 1}", "retry succeeded")
                            continue

                    _status("Replanning", "adapting to failure")
                    new_steps = await self.create_plan(
                        task, task_type, skills, outputs
                    )
                    if new_steps:
                        steps = steps[:i + 1] + new_steps
                        replan_count += 1
                        _status("Replanned", f"{len(new_steps)} new steps")

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


__all__ = [
    'SkillPlanExecutor',
]

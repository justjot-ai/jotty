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

    async def create_reflective_plan(
        self,
        task: str,
        task_type: str,
        skills: List[Dict[str, Any]],
        failed_steps: List[Dict[str, Any]],
        completed_outputs: Optional[Dict[str, Any]] = None,
        max_steps: int = 0,
    ) -> tuple:
        """
        Create a reflective replan after step failure.

        Uses AgenticPlanner.replan_with_reflection() if available,
        otherwise falls back to regular plan_execution().

        Args:
            task: Original task description
            task_type: Task type string
            skills: Available skills
            failed_steps: List of dicts with {skill_name, tool_name, error, params}
            completed_outputs: Outputs from successful steps
            max_steps: Maximum remaining steps (0 = use self._max_steps)

        Returns:
            (steps, reflection, reasoning)
        """
        remaining_steps = max_steps or self._max_steps

        if self.planner is None:
            logger.warning("No planner available for reflective replanning")
            return [], "No planner available", ""

        # Use reflective replanning if planner supports it
        if hasattr(self.planner, 'replan_with_reflection'):
            try:
                excluded = list(self._excluded_skills)
                steps, reflection, reasoning = self.planner.replan_with_reflection(
                    task=task,
                    task_type=task_type,
                    skills=skills,
                    failed_steps=failed_steps,
                    completed_outputs=completed_outputs,
                    excluded_skills=excluded,
                    max_steps=remaining_steps,
                )
                logger.info(f"Reflective replan: {len(steps)} steps, reflection: {reflection[:80]}")
                return steps, reflection, reasoning
            except Exception as e:
                logger.warning(f"Reflective replanning failed: {e}, falling back")

        # Fallback to regular plan_execution
        try:
            steps, reasoning = self.planner.plan_execution(
                task=task,
                task_type=task_type,
                skills=skills,
                previous_outputs=completed_outputs,
                max_steps=remaining_steps,
            )
            return steps, "Regular replanning (no reflection)", reasoning
        except Exception as e:
            logger.error(f"Fallback replanning failed: {e}")
            return [], f"Replanning failed: {e}", ""

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

        # If skill is a BaseSkill instance (class-based skill), set the status callback
        # Note: get_skill() returns SkillDefinition; check the underlying object if present
        try:
            from Jotty.core.registry.skills_registry import BaseSkill
            underlying = getattr(skill, '_skill_instance', None)
            if isinstance(underlying, BaseSkill):
                underlying.set_status_callback(status_callback)
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
            import asyncio as _asyncio

            # Determine per-step timeout (default 120s, web skills 60s)
            step_timeout = 120.0
            if any(kw in step.skill_name.lower() for kw in ['web', 'search', 'http', 'scrape']):
                step_timeout = 60.0

            if inspect.iscoroutinefunction(tool):
                result = await _asyncio.wait_for(tool(resolved_params), timeout=step_timeout)
            else:
                # Run sync tools in executor with timeout
                loop = _asyncio.get_event_loop()
                result = await _asyncio.wait_for(
                    loop.run_in_executor(None, tool, resolved_params),
                    timeout=step_timeout,
                )

            if result is None:
                result = {'success': False, 'error': 'Tool returned None'}
            elif not isinstance(result, dict):
                result = {'success': True, 'output': result}

            _status("Done", f"✓ {step.skill_name}")
            return result

        except _asyncio.TimeoutError:
            logger.error(f"Tool execution timed out after {step_timeout}s: {step.skill_name}")
            _status("Timeout", f"✗ {step.skill_name} ({step_timeout}s)")
            return {'success': False, 'error': f'Tool timed out after {step_timeout}s'}

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
        Also handles aggregation of multiple research outputs.

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
                    resolved = self.resolve_path(path, outputs)
                    # Escape backslashes so re.sub doesn't interpret \u etc.
                    return resolved.replace('\\', '\\\\')

                value = re.sub(pattern, replacer, value)

                pattern2 = r'\{([a-zA-Z_][a-zA-Z0-9_.\[\]]*)\}'
                value = re.sub(pattern2, replacer, value)

                # Post-resolution check: if value still contains unresolved
                # research references, try to aggregate all research outputs
                if '${research_' in value or '{research_' in value:
                    aggregated = self._aggregate_research_outputs(outputs)
                    if aggregated:
                        value = re.sub(
                            r'\$\{research_\d+\.results\}',
                            aggregated,
                            value,
                        )

                # Fallback: detect unresolved placeholder-like strings
                # (e.g. "{CONTENT_FROM_STEP_1}", "{PREVIOUS_OUTPUT}") and
                # replace with the most recent output if available.
                unresolved_pattern = r'\{[A-Z_]+\}'
                if re.search(unresolved_pattern, value) and outputs:
                    last_output = list(outputs.values())[-1]
                    if isinstance(last_output, dict):
                        import json as _json
                        replacement = _json.dumps(last_output, default=str)[:8000]
                    else:
                        replacement = str(last_output)[:8000]
                    # Escape backslashes so re.sub doesn't interpret \u etc.
                    replacement = replacement.replace('\\', '\\\\')
                    value = re.sub(unresolved_pattern, replacement, value)
                    logger.info(f"Resolved unrecognised placeholder in param '{key}' with last step output")

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

    def _aggregate_research_outputs(self, outputs: Dict[str, Any]) -> str:
        """
        Aggregate all research_* outputs into a formatted string for synthesis.
        
        Used when multiple research steps feed into a synthesis/summary step.
        """
        research_parts = []
        for key in sorted(outputs.keys()):
            if key.startswith('research_') and isinstance(outputs[key], dict):
                result = outputs[key]
                query = result.get('query', key)
                results_list = result.get('results', [])
                
                if results_list:
                    part = f"\n## Research: {query}\n"
                    for i, r in enumerate(results_list[:5], 1):
                        title = r.get('title', 'Untitled') if isinstance(r, dict) else str(r)
                        snippet = r.get('snippet', '') if isinstance(r, dict) else ''
                        url = r.get('url', '') if isinstance(r, dict) else ''
                        part += f"\n### {i}. {title}\n{snippet}\n"
                        if url:
                            part += f"Source: {url}\n"
                    research_parts.append(part)
        
        return '\n'.join(research_parts) if research_parts else ''

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
                f"{step.skill_name}: {step.description[:100]}",
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
                        retry_result = await self.execute_step(step, outputs, status_callback)
                        if retry_result.get('success'):
                            outputs[step.output_key or f'step_{i}'] = retry_result
                            skills_used.append(step.skill_name)
                            _status(f"Step {i + 1}", "retry succeeded")
                            continue

                    _status("Replanning", "adapting to failure with reflection")
                    failed_step_info = {
                        'skill_name': step.skill_name,
                        'tool_name': step.tool_name,
                        'error': error_msg,
                        'params': step.params,
                    }
                    new_steps, reflection, _ = await self.create_reflective_plan(
                        task, task_type, skills,
                        [failed_step_info], outputs,
                        max_steps=self._max_steps - (i + 1),
                    )
                    if new_steps:
                        steps = steps[:i + 1] + new_steps
                        replan_count += 1
                        _status("Replanned", f"{len(new_steps)} new steps (reflection: {reflection[:60]})")

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

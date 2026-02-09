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

        # Skill selection cache: task_type → selected skill names
        # Avoids redundant LLM calls when the same type of task is seen again.
        # Cache is keyed by (task_type, frozenset(available_skill_names)).
        self._skill_cache: Dict[tuple, List[Dict[str, Any]]] = {}

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
        task_type: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Select best skills for a task using the planner.

        Uses native async DSPy (.acall) when available, falling back to
        run_in_executor for sync planners. Non-blocking either way.

        Caches results by task_type to avoid redundant LLM calls.
        """
        if self.planner is None:
            return available_skills[:max_skills]

        # Cache lookup: same task_type + same available skills → reuse
        avail_names = frozenset(s.get('name', '') for s in available_skills)
        cache_key = (task_type, avail_names)
        if cache_key in self._skill_cache:
            cached = self._skill_cache[cache_key]
            logger.info(f"Skill selection cache HIT: {task_type} → {[s['name'] for s in cached]}")
            return cached

        SKILL_SELECT_TIMEOUT = 30.0

        try:
            # Prefer native async (no thread pool overhead)
            if hasattr(self.planner, 'aselect_skills'):
                selected, reasoning = await asyncio.wait_for(
                    self.planner.aselect_skills(
                        task=task,
                        available_skills=available_skills,
                        max_skills=max_skills,
                        task_type=task_type,
                    ),
                    timeout=SKILL_SELECT_TIMEOUT,
                )
            else:
                # Fallback: run sync planner in thread pool
                loop = asyncio.get_running_loop()
                selected, reasoning = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.planner.select_skills(
                            task=task,
                            available_skills=available_skills,
                            max_skills=max_skills,
                            task_type=task_type,
                        ),
                    ),
                    timeout=SKILL_SELECT_TIMEOUT,
                )
            logger.debug(f"Skill selection reasoning: {reasoning}")

            # Cache the result (bounded to 50 entries)
            if task_type and len(self._skill_cache) < 50:
                self._skill_cache[cache_key] = selected
                logger.debug(f"Skill selection cached: {task_type}")

            return selected
        except asyncio.TimeoutError:
            logger.warning(f"Skill selection timed out after {SKILL_SELECT_TIMEOUT}s — using top skills")
            return available_skills[:max_skills]
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

        Uses native async DSPy (.acall) when available, falling back to
        run_in_executor for sync planners. Non-blocking either way.
        """
        if self.planner is None:
            logger.warning("No planner available, returning empty plan")
            return []

        PLAN_TIMEOUT = 45.0

        try:
            # Prefer native async (no thread pool overhead)
            if hasattr(self.planner, 'aplan_execution'):
                steps, reasoning = await asyncio.wait_for(
                    self.planner.aplan_execution(
                        task=task,
                        task_type=task_type,
                        skills=skills,
                        previous_outputs=previous_outputs,
                        max_steps=self._max_steps,
                    ),
                    timeout=PLAN_TIMEOUT,
                )
            else:
                # Fallback: run sync planner in thread pool
                loop = asyncio.get_running_loop()
                steps, reasoning = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.planner.plan_execution(
                            task=task,
                            task_type=task_type,
                            skills=skills,
                            previous_outputs=previous_outputs,
                            max_steps=self._max_steps,
                        ),
                    ),
                    timeout=PLAN_TIMEOUT,
                )
            logger.debug(f"Plan reasoning: {reasoning}")
            return steps
        except asyncio.TimeoutError:
            logger.warning(f"Planning timed out after {PLAN_TIMEOUT}s — using direct LLM fallback")
            return []
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

        REPLAN_TIMEOUT = 30.0  # Replanning should be fast; if it takes >30s, skip it

        # Prefer native async reflective replanning
        if hasattr(self.planner, 'areplan_with_reflection'):
            try:
                excluded = list(self._excluded_skills)
                steps, reflection, reasoning = await asyncio.wait_for(
                    self.planner.areplan_with_reflection(
                        task=task,
                        task_type=task_type,
                        skills=skills,
                        failed_steps=failed_steps,
                        completed_outputs=completed_outputs,
                        excluded_skills=excluded,
                        max_steps=remaining_steps,
                    ),
                    timeout=REPLAN_TIMEOUT,
                )
                logger.info(f"Reflective replan: {len(steps)} steps, reflection: {reflection[:80]}")
                return steps, reflection, reasoning
            except asyncio.TimeoutError:
                logger.warning(f"Reflective replanning timed out after {REPLAN_TIMEOUT}s — skipping")
                return [], "Replan timed out", ""
            except Exception as e:
                logger.warning(f"Async reflective replanning failed: {e}, falling back")

        # Fallback: sync reflective replanning via thread pool
        elif hasattr(self.planner, 'replan_with_reflection'):
            try:
                excluded = list(self._excluded_skills)
                loop = asyncio.get_running_loop()
                steps, reflection, reasoning = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.planner.replan_with_reflection(
                            task=task,
                            task_type=task_type,
                            skills=skills,
                            failed_steps=failed_steps,
                            completed_outputs=completed_outputs,
                            excluded_skills=excluded,
                            max_steps=remaining_steps,
                        ),
                    ),
                    timeout=REPLAN_TIMEOUT,
                )
                logger.info(f"Reflective replan: {len(steps)} steps, reflection: {reflection[:80]}")
                return steps, reflection, reasoning
            except asyncio.TimeoutError:
                logger.warning(f"Reflective replanning timed out after {REPLAN_TIMEOUT}s — skipping")
                return [], "Replan timed out", ""
            except Exception as e:
                logger.warning(f"Reflective replanning failed: {e}, falling back")

        # Final fallback: regular plan_execution (prefer async)
        try:
            if hasattr(self.planner, 'aplan_execution'):
                steps, reasoning = await asyncio.wait_for(
                    self.planner.aplan_execution(
                        task=task,
                        task_type=task_type,
                        skills=skills,
                        previous_outputs=completed_outputs,
                        max_steps=remaining_steps,
                    ),
                    timeout=REPLAN_TIMEOUT,
                )
            else:
                loop = asyncio.get_running_loop()
                steps, reasoning = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.planner.plan_execution(
                            task=task,
                            task_type=task_type,
                            skills=skills,
                            previous_outputs=completed_outputs,
                            max_steps=remaining_steps,
                        ),
                    ),
                    timeout=REPLAN_TIMEOUT,
                )
            return steps, "Regular replanning (no reflection)", reasoning
        except asyncio.TimeoutError:
            logger.warning(f"Fallback replanning timed out after {REPLAN_TIMEOUT}s")
            return [], "Replan timed out", ""
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

            # Determine per-step timeout (default 90s, web skills 30s)
            # Reduced from 120s/60s — smart_fetch now has a 20s hard budget per URL,
            # so even search_and_scrape (3 URLs parallel) finishes within ~25s.
            step_timeout = 90.0
            if any(kw in step.skill_name.lower() for kw in ['web', 'search', 'http', 'scrape']):
                step_timeout = 30.0

            if inspect.iscoroutinefunction(tool):
                result = await _asyncio.wait_for(tool(resolved_params), timeout=step_timeout)
            else:
                # Run sync tools in executor with timeout
                loop = _asyncio.get_running_loop()
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

                def replacer(match, _param_name=key):
                    ref_path = match.group(1)
                    raw_resolved = self.resolve_path(ref_path, outputs)

                    # Smart extraction: if the resolved value is a JSON dict
                    # and the target param hints at what field to extract, do so.
                    # e.g. params["path"] = "${step_2}" -> extract step_2["path"]
                    if raw_resolved.startswith('{'):
                        try:
                            import json as _json
                            obj = _json.loads(raw_resolved)
                            if isinstance(obj, dict):
                                # If param name matches a key in the resolved dict, extract it
                                if _param_name in obj:
                                    return str(obj[_param_name]).replace('\\', '\\\\')
                                # Common extractions: path, content, output, stdout
                                for extract_key in ('path', 'output', 'content', 'stdout', 'result'):
                                    if _param_name.lower() in (extract_key, f'file_{extract_key}', f'{extract_key}_path'):
                                        if extract_key in obj:
                                            return str(obj[extract_key]).replace('\\', '\\\\')
                                # If param is 'path' but dict doesn't have it, scan all outputs
                                if _param_name == 'path':
                                    for _k in reversed(list(outputs.keys())):
                                        _sd = outputs[_k]
                                        if isinstance(_sd, dict) and 'path' in _sd:
                                            return str(_sd['path']).replace('\\', '\\\\')
                        except (ValueError, KeyError):
                            pass

                    # Escape backslashes so re.sub doesn't interpret \u etc.
                    return raw_resolved.replace('\\', '\\\\')

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

                # Fallback: if the entire value is a bare output key (no ${} wrapping),
                # resolve it directly. LLMs often write params like {"path": "step_2"}
                # instead of {"path": "${step_2}"}.
                if value in outputs and isinstance(outputs[value], dict):
                    step_data = outputs[value]
                    # Smart field extraction based on param name
                    if key in step_data:
                        value = str(step_data[key])
                        logger.info(f"Resolved bare output key '{key}={params[key]}' → {value[:100]}")
                    elif key == 'path' and 'path' in step_data:
                        value = str(step_data['path'])
                        logger.info(f"Resolved bare output key '{key}={params[key]}' → {value[:100]}")
                    elif key in ('content', 'text') and 'output' in step_data:
                        value = str(step_data['output'])[:10000]
                        logger.info(f"Resolved bare output key '{key}={params[key]}' → content ({len(value)} chars)")
                    elif key == 'path':
                        # The referenced step doesn't have a path — scan ALL outputs
                        # for the most recent step with a path field (common for
                        # verify steps that reference the wrong step index).
                        for _k in reversed(list(outputs.keys())):
                            _sd = outputs[_k]
                            if isinstance(_sd, dict) and 'path' in _sd:
                                value = str(_sd['path'])
                                logger.info(f"Resolved bare output key '{key}={params[key]}' → path from '{_k}': {value[:100]}")
                                break
                    else:
                        # Serialize the whole dict as fallback
                        import json as _json
                        value = _json.dumps(step_data, default=str)[:8000]
                        logger.info(f"Resolved bare output key '{key}={params[key]}' → full dict ({len(value)} chars)")

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

        # Step 2: Select best skills (pre-filtered by task type capabilities)
        _status("Selecting", "choosing best skills")
        skills = await self.select_skills(task, discovered_skills, task_type=task_type)
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

        # Step 4: Execute steps with total execution budget
        # Instead of letting individual steps pile up to a full timeout,
        # enforce a total wall-clock budget. When exhausted, return partial
        # results (what we have so far) instead of a full timeout failure.
        TOTAL_EXECUTION_BUDGET = 240.0  # 4 minutes hard cap for all steps
        outputs = {}
        skills_used = []
        errors = []
        replan_count = 0
        budget_exhausted = False

        for i, step in enumerate(steps):
            # Check total budget before starting each step
            elapsed = time.time() - start_time
            remaining_budget = TOTAL_EXECUTION_BUDGET - elapsed
            if remaining_budget < 5.0:
                _status("Budget", f"execution budget exhausted ({elapsed:.0f}s) — returning partial results")
                errors.append(f"Budget exhausted after step {i}/{len(steps)} ({elapsed:.0f}s)")
                budget_exhausted = True
                break

            _status(
                f"Step {i + 1}/{len(steps)}",
                f"{step.skill_name}: {step.description[:100]} (budget: {remaining_budget:.0f}s left)",
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
                        logger.info(f"Excluded skill '{step.skill_name}' due to: {error_msg[:50]}")

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

                    # Check budget before replanning (replanning = LLM call = expensive)
                    if (time.time() - start_time) > (TOTAL_EXECUTION_BUDGET * 0.75):
                        _status("Skip replan", "budget low — continuing with remaining steps")
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
            "budget_exhausted": budget_exhausted,
            "execution_time": time.time() - start_time,
        }


__all__ = [
    'SkillPlanExecutor',
]

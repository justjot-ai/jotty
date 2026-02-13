"""
SkillPlanExecutor - Reusable Planning/Execution Service

Extracted from AutonomousAgent to enable any agent to use skill-based
planning and execution. Provides:
- LLM-based skill selection via TaskPlanner
- Execution plan creation
- Multi-step execution with dependency handling
- Template variable resolution (via ParameterResolver in step_processors.py)
- Task type inference

Usage:
    executor = SkillPlanExecutor(skills_registry)
    result = await executor.plan_and_execute(task, discovered_skills)

Author: A-Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import time
import threading
from pathlib import Path

from Jotty.core.utils.async_utils import StatusReporter
from typing import Any, Callable, Dict, List, Optional, Tuple

from .step_processors import ParameterResolver, ToolResultProcessor

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL CALL CACHE
# =============================================================================

class ToolCallCache:
    """Thread-safe TTL + LRU cache for tool call results.

    Prevents redundant tool executions when the same skill+tool+params
    combination is called multiple times within a session.

    Usage:
        cache = ToolCallCache(ttl_seconds=300)
        key = cache.make_key("web-search", "search_web_tool", {"query": "AI"})
        cached = cache.get(key)
        if cached is None:
            result = await tool(params)
            cache.set(key, result)
    """

    def __init__(self, ttl_seconds: int = 300, max_size: int = 100):
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def make_key(skill_name: str, tool_name: str, params: Dict[str, Any]) -> str:
        """Create a deterministic cache key from skill, tool, and params."""
        # Sort params for deterministic hashing, skip non-serializable values
        try:
            param_str = json.dumps(params, sort_keys=True, default=str)
        except (TypeError, ValueError):
            param_str = str(sorted(params.items()))
        raw = f"{skill_name}:{tool_name}:{param_str}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get a cached result if it exists and hasn't expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            result, timestamp = entry
            if time.time() - timestamp > self._ttl:
                del self._cache[key]
                return None
            return result

    def set(self, key: str, value: Any) -> None:
        """Cache a tool result."""
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            self._cache[key] = (value, time.time())

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)


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

        # Tool call cache: prevents redundant tool executions
        self._tool_cache = ToolCallCache(ttl_seconds=300)

    # =========================================================================
    # LAZY INITIALIZATION
    # =========================================================================

    @property
    def planner(self):
        """Lazy-load TaskPlanner."""
        if self._planner is None:
            try:
                from ..agentic_planner import TaskPlanner
                self._planner = TaskPlanner()
                logger.debug("SkillPlanExecutor: Initialized TaskPlanner")
            except Exception as e:
                logger.warning(f"SkillPlanExecutor: Could not initialize TaskPlanner: {e}")
        return self._planner

    def _ensure_lm(self) -> None:
        """Ensure DSPy LM is configured before tool execution.

        Tools like claude-cli-llm check dspy.settings.lm and fail if None.
        This is normally set by BaseAgent._init_dspy_lm(), but when running
        the executor standalone (without an agent), the global LM is missing.
        """
        try:
            import dspy
            if hasattr(dspy.settings, 'lm') and dspy.settings.lm is not None:
                return  # Already configured
        except ImportError:
            return

        import os
        # Load API keys from .env.anthropic if not in environment
        from pathlib import Path as _Path
        env_file = _Path(__file__).parents[4] / '.env.anthropic'
        if env_file.exists():
            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if '=' in line:
                            k, v = line.split('=', 1)
                            k, v = k.strip(), v.strip()
                            if v and k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass

        try:
            from Jotty.core.foundation.direct_anthropic_lm import DirectAnthropicLM, is_api_key_available
            if is_api_key_available():
                lm = DirectAnthropicLM()
                dspy.configure(lm=lm)
                logger.info("SkillPlanExecutor: Auto-configured DSPy LM with DirectAnthropicLM")
                return
        except Exception as e:
            logger.debug(f"DirectAnthropicLM not available: {e}")

        try:
            from Jotty.core.foundation.persistent_claude_lm import PersistentClaudeCLI
            lm = PersistentClaudeCLI()
            dspy.configure(lm=lm)
            logger.info("SkillPlanExecutor: Auto-configured DSPy LM with PersistentClaudeCLI")
        except Exception as e:
            logger.warning(f"SkillPlanExecutor: Could not configure DSPy LM: {e}")

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

        Uses TaskPlanner.replan_with_reflection() if available,
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
        _status = StatusReporter(status_callback)

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

        # Strict fallback: case-insensitive exact match, then single-tool-skill fallback
        if not tool and hasattr(skill, 'tools') and skill.tools:
            tool = self._strict_tool_lookup(skill, step.tool_name)

            # Single-tool-skill fallback (unambiguous — only one tool available)
            if not tool and len(skill.tools) == 1:
                first_tool_name = list(skill.tools.keys())[0]
                tool = skill.tools[first_tool_name]
                logger.info(f"Tool '{step.tool_name}' not found, using only tool: '{first_tool_name}'")

        if not tool:
            return {
                'success': False,
                'error': f'Tool not found: {step.tool_name}',
            }

        # Build ToolSchema for type-aware resolution + auto-wiring
        tool_schema = None
        try:
            tool_schema = skill.get_tool_schema(step.tool_name)
        except Exception:
            pass  # Schema is optional — degrade gracefully

        resolved_params = self.resolve_params(step.params, outputs, step=step, tool_schema=tool_schema)

        # Error-feedback-retry: validate after resolution and attempt fixes
        if tool_schema is not None:
            validation = tool_schema.validate(resolved_params, coerce=True)
            resolved_params.update(validation.coerced_params)
            if not validation.valid:
                fixed = self._attempt_param_fix(
                    resolved_params, validation, tool_schema, outputs, step
                )
                if fixed is not None:
                    resolved_params = fixed

        # Guard: detect duplicate file writes — if we're writing to the same
        # path as a previous step, try to infer the correct filename from the
        # step description. LLM planners sometimes reuse the same path for
        # multiple file-operations/write_file_tool steps.
        if (step.skill_name == 'file-operations' and 'path' in resolved_params
                and 'content' in resolved_params):
            _write_path = resolved_params['path']
            # Check if this path was already written in a previous output
            for _prev_key, _prev_out in outputs.items():
                if isinstance(_prev_out, dict) and _prev_out.get('path') == _write_path:
                    # Duplicate! Try to extract correct filename from step description
                    import re as _re
                    _desc = step.description or ''
                    _file_match = _re.search(
                        r'(?:save|write|named?|called?|to)\s+["\']?'
                        r'([a-zA-Z0-9_]+\.\w{1,5})["\']?',
                        _desc, _re.IGNORECASE
                    )
                    if _file_match:
                        _correct_path = _file_match.group(1)
                        if _correct_path != _write_path:
                            logger.info(f" Duplicate write to '{_write_path}' detected — "
                                       f"corrected to '{_correct_path}' from step description")
                            resolved_params['path'] = _correct_path
                    break

        # Log resolved params for debugging (especially file paths)
        if resolved_params and step.skill_name in ('file-operations', 'shell-exec'):
            _param_preview = {k: (str(v)[:80] + '...' if len(str(v)) > 80 else str(v))
                             for k, v in resolved_params.items() if k != 'content'}
            logger.info(f" Resolved params: {_param_preview}")

        # Emit status based on skill type
        skill_status_map = {
            'research': ' Researching...',
            'search': ' Searching...',
            'web': ' Fetching web data...',
            'pdf': ' Creating PDF...',
            'chart': ' Generating charts...',
            'telegram': ' Sending message...',
            'slack': ' Posting to Slack...',
            'file': ' Processing files...',
            'data': ' Analyzing data...',
            'stock': ' Fetching stock data...',
        }

        for key, msg in skill_status_map.items():
            if key in step.skill_name.lower() or key in step.tool_name.lower():
                _status("Executing", msg)
                break
        else:
            _status("Executing", f" Running {step.skill_name}...")

        try:
            import asyncio as _asyncio
            import time as _time

            # Check tool call cache first (skip for side-effect tools)
            _side_effect_skills = {'telegram', 'slack', 'email', 'file-operations', 'shell-exec'}
            _skip_cache = any(s in step.skill_name.lower() for s in _side_effect_skills)
            cache_key = None

            if not _skip_cache:
                cache_key = self._tool_cache.make_key(step.skill_name, step.tool_name, resolved_params)
                cached_result = self._tool_cache.get(cache_key)
                if cached_result is not None:
                    logger.info(f"Tool cache HIT: {step.skill_name}/{step.tool_name}")
                    return cached_result

            # Determine per-step timeout:
            # - LLM generation tools (claude-cli-llm): 180s (complex content takes time)
            # - Web/search tools: 30s (smart_fetch has 20s hard budget per URL)
            # - Default: 90s for everything else
            step_timeout = 90.0
            _skill_lower = step.skill_name.lower()
            if any(kw in _skill_lower for kw in ['claude-cli', 'llm', 'openai', 'groq']):
                step_timeout = 180.0
            elif any(kw in _skill_lower for kw in ['web', 'search', 'http', 'scrape']):
                step_timeout = 30.0

            _step_start = _time.time()

            if inspect.iscoroutinefunction(tool):
                result = await _asyncio.wait_for(tool(resolved_params), timeout=step_timeout)
            else:
                # Run sync tools in executor with timeout
                loop = _asyncio.get_running_loop()
                result = await _asyncio.wait_for(
                    loop.run_in_executor(None, tool, resolved_params),
                    timeout=step_timeout,
                )

            _step_elapsed = _time.time() - _step_start

            if result is None:
                result = {'success': False, 'error': 'Tool returned None'}
            elif not isinstance(result, dict):
                result = {'success': True, 'output': result}

            # Sanitize/truncate tool output (JSON-aware: preserves keys)
            result = ToolResultProcessor().process(result, elapsed=_step_elapsed)

            # Cache successful results for reuse
            if cache_key and result.get('success', True):
                self._tool_cache.set(cache_key, result)

            _status("Done", f" {step.skill_name}")
            return result

        except _asyncio.TimeoutError:
            logger.error(f"Tool execution timed out after {step_timeout}s: {step.skill_name}")
            _status("Timeout", f" {step.skill_name} ({step_timeout}s)")
            return {'success': False, 'error': f'Tool timed out after {step_timeout}s'}

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            _status("Error", f" {step.skill_name}")
            return {'success': False, 'error': str(e)}

    # =========================================================================
    # PLAN-TIME VALIDATION
    # =========================================================================

    def validate_plan(self, steps: list) -> List[Dict]:
        """Validate a plan before execution (warning-only, does not block).

        Checks each step for:
        - Skill existence in the registry
        - Tool existence (strict, case-insensitive)
        - Parameter validation against schema
        - depends_on index validity

        Returns list of ``{step_index, step_description, errors}`` dicts
        for steps with issues. Empty list means the plan looks valid.
        """
        issues: List[Dict] = []
        num_steps = len(steps)

        for i, step in enumerate(steps):
            step_errors: List[str] = []

            # Check skill exists
            skill_name = getattr(step, 'skill_name', '') or ''
            skill = self._skills_registry.get_skill(skill_name) if self._skills_registry else None
            if not skill:
                step_errors.append(f"Skill not found: {skill_name}")
            else:
                # Check tool exists (strict lookup)
                tool_name = getattr(step, 'tool_name', '') or ''
                tool = skill.tools.get(tool_name)
                if not tool:
                    tool = self._strict_tool_lookup(skill, tool_name)
                if not tool and len(skill.tools) != 1:
                    available = list(skill.tools.keys())
                    step_errors.append(
                        f"Tool '{tool_name}' not found in {skill_name}. "
                        f"Available: {available}"
                    )

                # Validate params against schema
                if tool:
                    try:
                        schema = skill.get_tool_schema(tool_name)
                        if schema:
                            params = getattr(step, 'params', {}) or {}
                            result = schema.validate(params)
                            step_errors.extend(result.errors)
                    except Exception:
                        pass  # Schema building may fail — not fatal

            # Check depends_on validity
            depends = getattr(step, 'depends_on', []) or []
            for dep_idx in depends:
                if not isinstance(dep_idx, int) or dep_idx < 0 or dep_idx >= num_steps:
                    step_errors.append(f"Invalid depends_on index: {dep_idx}")
                elif dep_idx >= i:
                    step_errors.append(f"Forward dependency: step {i} depends on step {dep_idx}")

            if step_errors:
                issues.append({
                    'step_index': i,
                    'step_description': getattr(step, 'description', '')[:100],
                    'errors': step_errors,
                })

        return issues

    # =========================================================================
    # STRICT TOOL LOOKUP
    # =========================================================================

    @staticmethod
    def _strict_tool_lookup(skill, tool_name: str) -> Optional[Callable]:
        """Case-insensitive exact match for tool lookup. No substring matching.

        Returns the tool callable if found, None otherwise.
        Avoids dangerous substring matches that could invoke the wrong tool.
        """
        if not tool_name or not hasattr(skill, 'tools'):
            return None
        tool_name_lower = tool_name.lower()
        for name, func in skill.tools.items():
            if name.lower() == tool_name_lower:
                logger.info(f"Strict tool lookup: matched '{tool_name}' → '{name}'")
                return func
        return None

    # =========================================================================
    # ERROR-FEEDBACK-RETRY
    # =========================================================================

    def _attempt_param_fix(
        self,
        params: Dict[str, Any],
        validation,
        tool_schema,
        outputs: Dict[str, Any],
        step,
    ) -> Optional[Dict[str, Any]]:
        """Attempt to fix validation errors before tool execution.

        Strategies:
        1. Missing required params → auto_wire from outputs
        2. Type mismatch → retry coercion with alternate type names
        3. Bad path → search outputs for valid path

        Returns fixed params dict, or None if unfixable.
        """
        from Jotty.core.agents._execution_types import TypeCoercer

        fixed = dict(params)
        any_fixed = False

        for error in validation.errors:
            # Strategy 1: Missing required → try auto_wire
            if error.startswith("Missing required parameter:"):
                param_name = error.split(": ", 1)[1].strip()
                tp = tool_schema.get_param(param_name)
                if tp and outputs:
                    wired = tool_schema.auto_wire({param_name: None}, outputs)
                    if param_name in wired and wired[param_name] is not None:
                        fixed[param_name] = wired[param_name]
                        any_fixed = True
                        logger.info(f"Param fix: auto-wired missing '{param_name}' from outputs")
                        continue

            # Strategy 2: Type error → retry with alternate coercion
            if error.startswith("Type error for '"):
                param_name = error.split("'")[1]
                if param_name in fixed:
                    value = fixed[param_name]
                    tp = tool_schema.get_param(param_name)
                    if tp:
                        # Try string conversion first
                        coerced, err = TypeCoercer.coerce(str(value), tp.type_hint)
                        if not err:
                            fixed[param_name] = coerced
                            any_fixed = True
                            logger.info(f"Param fix: re-coerced '{param_name}' via str() intermediate")
                            continue

            # Strategy 3: Path-like params that failed → find from outputs
            if 'path' in error.lower():
                resolver = ParameterResolver(outputs)
                found = resolver._find_path_from_outputs(step)
                if found:
                    # Find which param is the path
                    for tp in tool_schema.params:
                        if tp.type_hint in ('path', 'file_path') and tp.name in fixed:
                            fixed[tp.name] = found
                            any_fixed = True
                            logger.info(f"Param fix: replaced bad path for '{tp.name}' with '{found}'")
                            break

        if any_fixed:
            # Re-validate to confirm fixes worked
            recheck = tool_schema.validate(fixed, coerce=True)
            fixed.update(recheck.coerced_params)
            if recheck.valid:
                logger.info("Param fix: all errors resolved")
                return fixed
            else:
                logger.warning(f"Param fix: {len(recheck.errors)} error(s) remain after fix attempt")
                return fixed  # Return partially fixed params anyway

        return None

    # =========================================================================
    # PARAMETER RESOLUTION (delegates to ParameterResolver)
    # =========================================================================

    def resolve_params(
        self,
        params: Dict[str, Any],
        outputs: Dict[str, Any],
        step: Any = None,
        tool_schema: Any = None,
    ) -> Dict[str, Any]:
        """Resolve template variables in parameters. Delegates to ParameterResolver.

        When *tool_schema* is provided, enables auto-wiring and validation.
        """
        return ParameterResolver(outputs).resolve(params, step, tool_schema=tool_schema)

    def resolve_path(self, path: str, outputs: Dict[str, Any]) -> str:
        """Resolve a dotted path like 'step_key.field' from outputs."""
        return ParameterResolver(outputs).resolve_path(path)

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
    # ARTIFACT TAGGING & LARGE OUTPUT SPILL
    # =========================================================================

    @staticmethod
    def _infer_artifact_tags(step, result: Dict[str, Any]) -> List[str]:
        """Infer semantic tags for a step result based on skill name and content.

        Used to populate ``SwarmArtifactStore`` entries so downstream
        consumers can query by meaning (e.g. ``store.query_by_tag('analysis')``).
        """
        tags: List[str] = []
        skill = getattr(step, 'skill_name', '') or ''
        skill_lower = skill.lower()

        tag_keywords = {
            'research': ['research', 'search', 'web-search'],
            'analysis': ['analy', 'data', 'chart', 'stock', 'csv'],
            'generation': ['generate', 'create', 'write', 'pdf', 'doc'],
            'communication': ['telegram', 'slack', 'email', 'discord', 'messaging'],
            'code': ['code', 'python', 'shell', 'exec'],
            'file': ['file', 'read', 'write'],
        }
        for tag, keywords in tag_keywords.items():
            if any(kw in skill_lower for kw in keywords):
                tags.append(tag)

        if result.get('path'):
            tags.append('file_output')
        if not tags:
            tags.append('general')
        return tags

    @staticmethod
    def _spill_large_values(
        result: Dict[str, Any],
        threshold: int = 500_000,
        run_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Replace string values exceeding *threshold* bytes with a FileReference.

        Writes the large value to disk under *run_dir* (or ``/tmp``) and
        substitutes a lightweight ``FileReference`` in the result dict.
        """
        from Jotty.core.agents._execution_types import FileReference
        import hashlib as _hl

        spill_dir = run_dir or "/tmp/jotty_spill"
        spilled = dict(result)
        for key, value in result.items():
            if isinstance(value, str) and len(value) > threshold:
                os.makedirs(spill_dir, exist_ok=True)
                digest = _hl.md5(value.encode("utf-8", errors="replace")).hexdigest()[:12]
                fname = f"{key}_{digest}.txt"
                fpath = os.path.join(spill_dir, fname)
                Path(fpath).write_text(value, encoding="utf-8")
                spilled[key] = FileReference(
                    path=fpath,
                    content_type="text/plain",
                    size_bytes=len(value.encode("utf-8", errors="replace")),
                    checksum=digest,
                    step_key=key,
                    description=f"Spilled large value ({len(value)} chars)",
                )
                logger.info(f"Spilled large output '{key}' ({len(value)} chars) → {fpath}")
        return spilled

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
    # DAG PARALLEL EXECUTION
    # =========================================================================

    def _build_dependency_graph(self, steps: list) -> Dict[int, List[int]]:
        """Build a dependency graph from execution steps.

        Returns dict mapping step_index → list of dependency indices.
        Steps with no depends_on field depend on nothing (can run first).
        """
        graph: Dict[int, List[int]] = {}
        for i, step in enumerate(steps):
            deps = getattr(step, 'depends_on', None) or []
            graph[i] = [d for d in deps if isinstance(d, int) and 0 <= d < len(steps)]
        return graph

    def _find_parallel_groups(self, steps: list) -> List[List[int]]:
        """Identify groups of steps that can execute in parallel (DAG layers).

        Uses topological layering: steps with all deps satisfied form one layer.
        Returns list of layers, each layer is a list of step indices.
        """
        graph = self._build_dependency_graph(steps)
        completed: set = set()
        layers: List[List[int]] = []
        remaining = set(range(len(steps)))

        while remaining:
            # Find steps whose dependencies are all completed
            ready = [i for i in remaining if all(d in completed for d in graph[i])]
            if not ready:
                # Circular dependency or bug — break cycle by taking first
                ready = [min(remaining)]
                logger.warning(f"DAG cycle detected, forcing step {ready[0]}")
            layers.append(ready)
            completed.update(ready)
            remaining -= set(ready)

        return layers

    async def _execute_steps_dag(
        self,
        steps: list,
        status_callback: Optional[Callable] = None,
        start_time: float = 0.0,
        total_budget: float = 240.0,
    ) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """Execute steps respecting dependencies, parallelizing independent ones.

        Args:
            steps: List of ExecutionStep objects
            status_callback: Progress callback
            start_time: Execution start timestamp
            total_budget: Total time budget in seconds

        Returns:
            (outputs_dict, skills_used_list, errors_list)
        """
        _status = StatusReporter(status_callback, logger)
        outputs: Dict[str, Any] = {}
        skills_used: List[str] = []
        errors: List[str] = []
        layers = self._find_parallel_groups(steps)

        for layer_idx, layer in enumerate(layers):
            # Check budget
            elapsed = time.time() - start_time
            if elapsed > total_budget - 5.0:
                errors.append(f"Budget exhausted at layer {layer_idx}")
                break

            if len(layer) == 1:
                # Single step — execute normally
                i = layer[0]
                step = steps[i]
                _status(f"Step {i + 1}/{len(steps)}", f"{step.skill_name}: {step.description[:80]}")
                result = await self.execute_step(step, outputs, status_callback)
                if result.get('success'):
                    result = self._spill_large_values(result)
                    result['_tags'] = self._infer_artifact_tags(step, result)
                    outputs[step.output_key or f'step_{i}'] = result
                    skills_used.append(step.skill_name)
                else:
                    errors.append(f"Step {i + 1}: {result.get('error', 'Unknown')}")
            else:
                # Multiple independent steps — run in parallel
                _status(f"Parallel", f"executing {len(layer)} steps concurrently")
                async def _run_step(idx):
                    s = steps[idx]
                    return idx, await self.execute_step(s, outputs, status_callback)

                results = await asyncio.gather(
                    *[_run_step(i) for i in layer],
                    return_exceptions=True,
                )
                for item in results:
                    if isinstance(item, Exception):
                        errors.append(f"Parallel step exception: {item}")
                        continue
                    idx, result = item
                    step = steps[idx]
                    if result.get('success'):
                        result = self._spill_large_values(result)
                        result['_tags'] = self._infer_artifact_tags(step, result)
                        outputs[step.output_key or f'step_{idx}'] = result
                        skills_used.append(step.skill_name)
                    else:
                        errors.append(f"Step {idx + 1}: {result.get('error', 'Unknown')}")

        return outputs, skills_used, errors

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

        _status = StatusReporter(status_callback, logger)

        self._excluded_skills.clear()

        # Ensure DSPy LM is configured (tools depend on it)
        self._ensure_lm()

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

        # Step 3b: Validate plan (warning-only, does not block execution)
        if steps:
            plan_issues = self.validate_plan(steps)
            if plan_issues:
                for issue in plan_issues:
                    _status("Plan warning",
                            f"Step {issue['step_index']}: {'; '.join(issue['errors'][:2])}")
                logger.info(f"Plan validation: {len(plan_issues)} step(s) with issues")

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
                result = self._spill_large_values(result)
                result['_tags'] = self._infer_artifact_tags(step, result)
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
                            retry_result = self._spill_large_values(retry_result)
                            retry_result['_tags'] = self._infer_artifact_tags(step, retry_result)
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

    # =========================================================================
    # Research output aggregation
    # =========================================================================

    def _aggregate_research_outputs(self, outputs: Dict[str, Any]) -> str:
        """Aggregate multiple research step outputs into a single markdown string.

        Args:
            outputs: Dict keyed by step name (e.g. 'research_0') with values
                     containing 'query', 'success', and 'results' list.

        Returns:
            Formatted markdown string, or '' if no outputs.
        """
        if not outputs:
            return ''

        sections = []
        for key in sorted(outputs):
            entry = outputs[key]
            query = entry.get('query', key)
            results = entry.get('results', [])
            lines = [f"## Research: {query}"]
            for r in results:
                title = r.get('title', '')
                snippet = r.get('snippet', '')
                url = r.get('url', '')
                lines.append(f"- **{title}**: {snippet} ({url})")
            sections.append('\n'.join(lines))

        return '\n\n'.join(sections)


__all__ = [
    'ParameterResolver',
    'ToolResultProcessor',
    'ToolCallCache',
    'SkillPlanExecutor',
]

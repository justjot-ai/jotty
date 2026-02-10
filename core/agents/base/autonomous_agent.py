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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base_agent import BaseAgent, AgentConfig, AgentResult
from Jotty.core.utils.async_utils import StatusReporter, AgentEventBroadcaster, AgentEvent
from .skill_plan_executor import SkillPlanExecutor
from .._execution_types import ExecutionStep

logger = logging.getLogger(__name__)


# =============================================================================
# AUTONOMOUS AGENT CONFIG
# =============================================================================

@dataclass
class AutonomousAgentConfig(AgentConfig):
    """Configuration specific to AutonomousAgent."""
    max_steps: int = 10
    enable_replanning: bool = False  # Disabled: stop immediately on failure
    max_replans: int = 3
    skill_filter: Optional[str] = None
    default_output_skill: Optional[str] = None
    enable_output: bool = False
    enable_ensemble: bool = False
    ensemble_strategy: str = "multi_perspective"


# =============================================================================
# EXECUTION CONTEXT MANAGER
# =============================================================================

class ExecutionContextManager:
    """
    Tracks execution history and auto-compresses when approaching limits.

    Uses trajectory-preserving compression: on context overflow, summarise
    the OLDEST 60% of history while keeping the CURRENT trajectory intact.
    This prevents losing progress mid-execution while keeping context small.

    Usage::

        ctx = ExecutionContextManager(max_history_size=100000)
        ctx.add_step({"step": 0, "output": "..."})
        context = ctx.get_context()
    """

    def __init__(self, max_history_size: int = 100_000):
        self._history: List[dict] = []
        self._max_history_size = max_history_size
        self._compression_ratio: float = 0.7
        self._total_chars: int = 0

    def add_step(self, step_info: dict) -> None:
        """Append a step result and trigger compression if needed."""
        import json
        self._total_chars += len(json.dumps(step_info, default=str))
        self._history.append(step_info)
        if self._total_chars > self._max_history_size:
            self._compress()

    def _compress(self) -> None:
        """Summarise oldest 60% of history, keep recent 40% intact."""
        if len(self._history) < 3:
            return

        split_idx = max(1, int(len(self._history) * 0.6))
        old_entries = self._history[:split_idx]
        recent_entries = self._history[split_idx:]

        summary_parts = []
        for i, entry in enumerate(old_entries):
            skill = entry.get('skill_name', entry.get('step', f'step_{i}'))
            success = entry.get('success', '?')
            output = ''
            for key in ('output', 'response', 'text', 'content', 'result'):
                if key in entry and isinstance(entry[key], str):
                    output = entry[key][:150]
                    break
            summary_parts.append(f"  {skill}: success={success}" + (f" | {output}..." if output else ""))

        compressed_entry = {
            '_compressed': True,
            '_original_count': len(old_entries),
            'summary': f"Compressed {len(old_entries)} earlier steps:\n" + "\n".join(summary_parts),
        }

        self._history = [compressed_entry] + recent_entries

        # Recalculate and compound ratio
        import json
        self._total_chars = sum(len(json.dumps(e, default=str)) for e in self._history)
        self._compression_ratio *= 0.7
        logger.info(
            f"ExecutionContextManager: compressed {len(old_entries)} entries -> 1 summary, "
            f"history now {len(self._history)} entries, {self._total_chars} chars"
        )

    def get_context(self) -> List[dict]:
        """Return current history (possibly compressed) for replanning."""
        return list(self._history)

    def get_trajectory(self) -> List[dict]:
        """Return only the current uncompressed steps."""
        return [e for e in self._history if not e.get('_compressed')]


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
        Get all skills with autonomous-specific exclusion filtering.

        Gets full skill catalog from BaseAgent.discover_skills(), then applies
        excluded_skills (from executor, single source of truth) and category
        filter. The LLM in select_skills() handles semantic matching.

        Args:
            task: Task description

        Returns:
            List of skill dicts with name, description, tools
        """
        config: AutonomousAgentConfig = self.config

        # Get full skill catalog
        all_skills = self.discover_skills(task)

        # Get exclusions from executor (single source of truth)
        excluded = self.executor.excluded_skills

        # Apply autonomous-specific filters
        filtered = []
        for skill in all_skills:
            # Skip excluded skills
            if skill['name'] in excluded:
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
        task_type: str = "",
    ) -> List[Dict[str, Any]]:
        """Select best skills using the executor's planner (with caching)."""
        return await self.executor.select_skills(task, available_skills, max_skills, task_type=task_type)

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
    # EVENT EMISSION HELPER
    # =========================================================================

    def _emit(self, event_type: str, **data) -> None:
        """Emit an AgentEvent via the global broadcaster."""
        try:
            broadcaster = AgentEventBroadcaster.get_instance()
            broadcaster.emit(AgentEvent(
                type=event_type,
                data=data,
                agent_id=self.config.name,
            ))
        except Exception:
            pass

    # =========================================================================
    # MAIN EXECUTION (orchestration loop with replanning)
    # =========================================================================

    async def _execute_impl(self, task: str = "", **kwargs) -> Dict[str, Any]:
        """
        Execute an autonomous task.

        Orchestrates the full pipeline: infer task type -> discover skills ->
        select skills -> create plan -> execute steps (with replanning).

        Args:
            task: Task description
            **kwargs: Additional arguments
                direct_llm: If True, bypass skill pipeline and use LLM directly.
        """
        config: AutonomousAgentConfig = self.config
        start_time = time.time()
        status_callback = kwargs.pop('status_callback', None)
        learning_context = kwargs.pop('learning_context', None)
        workspace_dir = kwargs.pop('workspace_dir', None)
        direct_llm = kwargs.pop('direct_llm', False)

        _status = StatusReporter(status_callback, logger)

        self._emit("step_start", phase="execute_impl", task=task[:200])

        # Direct LLM mode (multi-agent sub-agents bypass skill pipeline)
        if direct_llm:
            return await self._handle_direct_llm(task, start_time, _status)

        # Reset excluded skills for new execution
        self.executor.clear_exclusions()

        # Steps 1+2: Infer task type AND discover skills in parallel
        task_type, all_skills = await self._discover_and_classify(task, _status)

        # Step 3: Select and optimize skills
        skills = await self._select_and_optimize_skills(
            task, task_type, all_skills, learning_context, _status
        )

        # Enrich system context with VVP when visual skills are present
        sys_context = self._get_system_context(all_skills)
        if sys_context and self.VVP_PROMPT in sys_context:
            _status("VVP", "visual verification protocol enabled")

        self._emit("status", phase="skills_selected", count=len(skills),
                   names=[s.get('name') for s in skills[:8]])

        # TOO_HARD early-exit
        if not all_skills and not skills:
            self._emit("error", phase="too_hard", reason="no relevant skills")
            _status("TOO_HARD", "no relevant skills found for this sub-task")
            return self._build_result(
                task, task_type, {}, [], ["TOO_HARD: No relevant skills found"],
                [], start_time, stopped=True, too_hard=True,
            )

        # Step 4: Create plan or go direct
        steps, skill_names = await self._create_or_skip_plan(
            task, task_type, skills, learning_context, workspace_dir, _status,
        )

        # No plan → direct LLM response
        if not steps:
            return await self._handle_no_plan_fallback(
                task, task_type, skill_names, start_time, _status,
            )

        # Step 5: Execute plan steps with replanning
        outputs, skills_used, errors, warnings, stopped = await self._run_execution_loop(
            steps, task, task_type, skills, _status, status_callback,
        )

        # Post-execution: verify expected output files exist
        self._verify_output_files(task, outputs)

        is_success = len(outputs) > 0 and not stopped and len(errors) == 0
        self._emit(
            "step_end", phase="execute_impl", success=is_success,
            steps=len(outputs), errors=len(errors),
            elapsed=round(time.time() - start_time, 2),
        )
        return self._build_result(
            task, task_type, outputs, skills_used, errors,
            warnings, start_time, stopped=stopped,
        )

    # =========================================================================
    # EXTRACTED SUB-METHODS (from _execute_impl)
    # =========================================================================

    async def _handle_direct_llm(
        self, task: str, start_time: float, status: StatusReporter,
    ) -> Dict[str, Any]:
        """Handle direct LLM mode — bypass skill pipeline entirely."""
        status("Direct LLM", "bypassing skill pipeline (multi-agent sub-agent)")
        self._emit("tool_start", skill="direct-llm", mode="direct_llm")
        llm_response = await self._fallback_llm_response(task)
        elapsed = time.time() - start_time
        if llm_response:
            self._emit("tool_end", skill="direct-llm", success=True, elapsed=round(elapsed, 2))
            self._emit("step_end", phase="execute_impl", success=True, steps=1, elapsed=round(elapsed, 2))
            return {
                "success": True, "task": task, "task_type": "direct_llm",
                "skills_used": ["direct-llm"], "steps_executed": 1,
                "outputs": {"llm_response": llm_response},
                "final_output": llm_response, "errors": [],
                "execution_time": elapsed,
            }
        self._emit("tool_end", skill="direct-llm", success=False, elapsed=round(elapsed, 2))
        self._emit("step_end", phase="execute_impl", success=False, elapsed=round(elapsed, 2))
        return {
            "success": False, "task": task,
            "error": "Direct LLM response failed",
            "skills_used": [], "steps_executed": 0, "outputs": {},
            "execution_time": elapsed,
        }

    async def _discover_and_classify(
        self, task: str, status: StatusReporter,
    ) -> tuple:
        """Infer task type and discover skills in parallel."""
        import asyncio as _asyncio
        status("Analyzing", "inferring task type + discovering skills (parallel)")

        async def _infer_type_async():
            return self._infer_task_type(task)

        task_type, all_skills = await _asyncio.gather(
            _infer_type_async(),
            self._discover_skills(task),
        )
        status("Task type", task_type)
        status("Skills found", f"{len(all_skills)} potential")

        if self.skills_registry and hasattr(self.skills_registry, 'get_failed_skills'):
            failed = self.skills_registry.get_failed_skills()
            if failed:
                status("Warning", f"{len(failed)} skills failed to load: {', '.join(failed.keys())}")
                logger.warning(f"Failed skills: {failed}")

        return task_type, all_skills

    async def _select_and_optimize_skills(
        self, task: str, task_type: str,
        all_skills: List[Dict], learning_context: Optional[str],
        status: StatusReporter,
    ) -> List[Dict]:
        """Select best skills and strip heavy ones for knowledge tasks."""
        status("Selecting", "choosing best skills")

        # Inject approach guidance from learning context
        selection_task = task
        if learning_context:
            approach_lines = [
                ln for ln in learning_context.split('\n')
                if any(kw in ln.lower() for kw in ['avoid', 'failed approach', 'successful approach', 'use:'])
            ]
            if approach_lines:
                selection_task = f"{task}\n\n[Skill guidance from past experience]:\n" + '\n'.join(approach_lines[:5])

        skills = await self._select_skills(selection_task, all_skills, task_type=task_type)

        # Strip heavy/irrelevant skills for knowledge-only tasks
        skills = self._optimize_for_knowledge_tasks(task, task_type, skills, all_skills, status)

        status("Skills selected", ", ".join(s['name'] for s in skills[:5]))
        return skills

    def _optimize_for_knowledge_tasks(
        self, task: str, task_type: str,
        skills: List[Dict], all_skills: List[Dict],
        status: StatusReporter,
    ) -> List[Dict]:
        """Strip heavy skills for tasks the LLM can answer directly."""
        _KNOWLEDGE_TASK_TYPES = {'analysis', 'explanation', 'design', 'architecture',
                                 'algorithm', 'tutorial', 'comparison', 'review'}
        _KNOWLEDGE_USEFUL = {'claude-cli-llm', 'file-operations', 'calculator',
                             'text-utils', 'shell-exec', 'unit-converter'}

        task_lower = task.lower()
        is_knowledge = (
            task_type in _KNOWLEDGE_TASK_TYPES
            or any(kw in task_lower for kw in [
                'design', 'architect', 'explain', 'compare', 'review',
                'algorithm', 'pattern', 'best practice', 'how to implement',
                'trade-off', 'pros and cons', 'what is', 'describe',
            ])
        )
        wants_heavy = any(kw in task_lower for kw in [
            'search', 'find online', 'latest', 'recent', 'news', 'current',
            'scrape', 'fetch', 'url', 'http', 'website', 'browse',
            'arxiv', 'paper', 'slides', 'presentation', 'mindmap',
            'research', 'benchmark', 'github stars', 'statistics',
            'released in', 'top 5', 'top 10', 'trending',
        ])

        if is_knowledge and not wants_heavy:
            before = len(skills)
            remaining = [s for s in skills if s.get('name') in _KNOWLEDGE_USEFUL]
            if not remaining:
                remaining = [s for s in all_skills if s.get('name') == 'claude-cli-llm'][:1]
            if remaining and len(remaining) < before:
                stripped = before - len(remaining)
                status("Optimized", f"kept {len(remaining)} useful skills for knowledge task (stripped {stripped} heavy)")
                return remaining

        return skills

    async def _create_or_skip_plan(
        self, task: str, task_type: str, skills: List[Dict],
        learning_context: Optional[str], workspace_dir: Optional[str],
        status: StatusReporter,
    ) -> tuple:
        """Create execution plan, or skip planning for simple tasks."""
        skill_names = {s['name'] for s in skills}
        _direct_llm_only = {'claude-cli-llm', 'calculator', 'unit-converter',
                            'text-utils', 'file-operations', 'shell-exec'}
        needs_planning = not skill_names.issubset(_direct_llm_only)

        # Code generation tasks ALWAYS need planning
        if not needs_planning and task_type in ('creation', 'generation', 'automation'):
            needs_planning = True
            logger.info(f"Forcing planning for {task_type} task (code gen needs a multi-step plan)")

        # Tasks requiring actual filesystem/shell interaction need planning
        # even when only "simple" skills are selected
        if not needs_planning:
            _tool_action_kws = [
                'list', 'read', 'write', 'create', 'delete', 'find', 'search',
                'run', 'execute', 'install', 'directory', 'folder', 'file',
                'save', 'count', 'check', 'scan', 'rename', 'move', 'copy',
            ]
            _tool_skills = skill_names & {'file-operations', 'shell-exec'}
            if _tool_skills and any(kw in task.lower() for kw in _tool_action_kws):
                needs_planning = True
                logger.info(f"Forcing planning: task requires actual {_tool_skills} execution")

        steps = []
        if needs_planning:
            status("Planning", "creating execution plan")
            prev_outputs = {}
            if learning_context:
                prev_outputs['_learning_guidance'] = learning_context[:2000]
            if workspace_dir:
                try:
                    from Jotty.core.prompts.rules import load_project_rules
                    rules = load_project_rules(workspace_dir)
                    if rules:
                        prev_outputs['_project_rules'] = rules[:2000]
                except Exception:
                    pass
            steps = await self._create_plan(
                task, task_type, skills,
                previous_outputs=prev_outputs if prev_outputs else None,
            )
            status("Plan ready", f"{len(steps)} steps")
        else:
            status("Direct mode", f"simple task — skipping planner (skills: {', '.join(skill_names)})")

        return steps, skill_names

    async def _handle_no_plan_fallback(
        self, task: str, task_type: str, skill_names: set,
        start_time: float, status: StatusReporter,
    ) -> Dict[str, Any]:
        """Generate direct LLM response when no plan was created."""
        status("Generating", "direct LLM response")
        self._emit("tool_start", skill="claude-cli-llm", mode="no_plan_fallback")
        llm_response = await self._fallback_llm_response(task)

        if llm_response:
            # Auto-save if file-operations was selected and task wants output saved
            if 'file-operations' in skill_names:
                try:
                    _save_kws = ['save', 'write to', 'create a file', 'output to']
                    if any(kw in task.lower() for kw in _save_kws) and self.skills_registry:
                        fo_skill = self.skills_registry.get_skill('file-operations')
                        if fo_skill and fo_skill.tools:
                            write_tool = fo_skill.tools.get('write_file_tool')
                            if write_tool:
                                from Jotty.core.agents._plan_utils_mixin import PlanUtilsMixin
                                fname = PlanUtilsMixin._infer_filename_from_task(None, task) or 'output.md'
                                write_tool({'path': fname, 'content': llm_response})
                                status("Saved", f"output written to {fname}")
                except Exception as e:
                    logger.debug(f"Auto-save failed (non-critical): {e}")

            elapsed = round(time.time() - start_time, 2)
            self._emit("tool_end", skill="claude-cli-llm", success=True, elapsed=elapsed)
            self._emit("step_end", phase="execute_impl", success=True, steps=1, elapsed=elapsed)
            return {
                "success": True, "task": task, "task_type": task_type,
                "skills_used": ["claude-cli-llm"], "steps_executed": 1,
                "outputs": {"llm_response": llm_response},
                "final_output": llm_response, "errors": [],
                "execution_time": time.time() - start_time,
            }
        elapsed = round(time.time() - start_time, 2)
        self._emit("tool_end", skill="claude-cli-llm", success=False, elapsed=elapsed)
        self._emit("step_end", phase="execute_impl", success=False, elapsed=elapsed)
        return {
            "success": False, "task": task,
            "error": "Could not generate response",
            "skills_used": [], "steps_executed": 0, "outputs": {},
        }

    async def _run_execution_loop(
        self, steps: list, task: str, task_type: str,
        skills: List[Dict], status: StatusReporter,
        status_callback: Optional[Callable],
    ) -> tuple:
        """Execute plan steps with replanning on failure."""
        config: AutonomousAgentConfig = self.config
        TOTAL_BUDGET = config.timeout  # wall-clock budget (seconds)
        budget_start = time.time()
        outputs = {}
        skills_used = []
        errors = []
        warnings = []
        replan_count = 0
        execution_stopped = False
        ctx = ExecutionContextManager()
        i = 0

        while i < len(steps):
            # Enforce wall-clock budget
            elapsed = time.time() - budget_start
            remaining = TOTAL_BUDGET - elapsed
            if remaining < 5.0:
                status("Budget", f"execution budget exhausted ({elapsed:.0f}s) — returning partial results")
                errors.append(f"Budget exhausted after step {i}/{len(steps)} ({elapsed:.0f}s)")
                execution_stopped = True
                break

            step = steps[i]
            status(f"Step {i+1}/{len(steps)}", f"{step.skill_name}: {step.description[:100]} (budget: {remaining:.0f}s left)")

            self._emit("step_start", step=i + 1, total=len(steps), skill=step.skill_name)
            result = await self._execute_step(step, outputs, status_callback)

            if result.get('success'):
                self._emit("step_end", step=i + 1, skill=step.skill_name, success=True)
                outputs[step.output_key or f'step_{i}'] = result
                skills_used.append(step.skill_name)
                # Track in context manager for trajectory-preserving compression
                ctx.add_step({
                    'skill_name': step.skill_name,
                    'step': i,
                    'success': True,
                    **{k: v for k, v in result.items() if k != 'success'},
                })
                status(f"Step {i+1}", "completed")
                i += 1
            else:
                error_msg = result.get('error', 'Unknown error')
                self._emit("step_end", step=i + 1, skill=step.skill_name, success=False, error=error_msg[:200])
                errors.append(f"Step {i+1}: {error_msg}")
                status(f"Step {i+1}", f"failed: {error_msg}")

                if step.optional:
                    status("Continuing", "optional step failed, skipping")
                    i += 1
                    continue

                # Try reflective replanning
                if config.enable_replanning and replan_count < config.max_replans:
                    status("Replanning", "adapting to failure with reflection")

                    exclusion_keywords = [
                        'not found', '404', 'invalid', 'delisted',
                        'not implemented', 'unsupported', 'deprecated',
                        'permission denied', 'unauthorized', 'forbidden',
                        'module not found', 'import error', 'no module',
                    ]
                    if any(kw in error_msg.lower() for kw in exclusion_keywords):
                        self.executor.exclude_skill(step.skill_name)
                        logger.info(f"Excluded skill '{step.skill_name}' due to: {error_msg[:50]}")

                    failed_step_info = {
                        'skill_name': step.skill_name,
                        'tool_name': step.tool_name,
                        'error': error_msg,
                        'params': step.params,
                    }
                    new_steps, reflection, _ = await self.executor.create_reflective_plan(
                        task, task_type, skills,
                        [failed_step_info], outputs,
                        max_steps=config.max_steps - (i + 1),
                    )
                    if new_steps:
                        steps = steps[:i + 1] + new_steps
                        replan_count += 1
                        if errors:
                            warnings.append(errors.pop())
                        status("Replanned", f"{len(new_steps)} new steps (reflection: {reflection[:200]})")
                        i += 1
                        continue

                status("Stopped", f"execution halted: {error_msg[:80]}")
                execution_stopped = True
                break

        return outputs, skills_used, errors, warnings, execution_stopped

    def _verify_output_files(self, task: str, outputs: Dict[str, Any]):
        """Auto-save missing files mentioned in the task."""
        try:
            import re as _re
            mentioned_files = _re.findall(
                r'\b([a-zA-Z][a-zA-Z0-9_.-]*\.\w{1,5})\b', task
            )
            seen = set()
            skip_exts = {'com', 'org', 'net', 'io', 'ai', 'dev', 'app', 'co'}

            for fname in mentioned_files:
                ext = fname.rsplit('.', 1)[-1].lower() if '.' in fname else ''
                if (fname in seen or fname.startswith('0.')
                        or '/' in fname or ext in skip_exts or len(fname) >= 100):
                    continue
                seen.add(fname)

                target_path = Path(fname)
                if target_path.exists() or not outputs:
                    continue

                # Find best content to save
                best_content = self._find_best_content_for_file(target_path, outputs)
                if len(best_content) > 100:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    # Atomic write: write to temp file then rename
                    import tempfile
                    fd, tmp_path = tempfile.mkstemp(
                        dir=str(target_path.parent),
                        suffix='.tmp',
                    )
                    try:
                        with open(fd, 'w') as f:
                            f.write(best_content)
                        Path(tmp_path).replace(target_path)
                    except Exception:
                        Path(tmp_path).unlink(missing_ok=True)
                        raise
                    logger.info(f"Auto-saved missing file: {fname} ({len(best_content)} chars)")
                    outputs[f'auto_save_{fname}'] = {
                        'success': True, 'path': str(target_path),
                        'bytes_written': len(best_content),
                    }
        except Exception as e:
            logger.debug(f"Post-exec file check skipped: {e}")

    @staticmethod
    def _find_best_content_for_file(
        target_path: Path, outputs: Dict[str, Any],
    ) -> str:
        """Find the best output content to save for a target file."""
        best = ""
        target_stem = target_path.stem.lower()
        content_keys = ('text', 'content', 'output', 'stdout')

        # First pass: prefer outputs whose key matches the filename
        for key, val in outputs.items():
            if isinstance(val, dict) and target_stem in key.lower():
                for ck in content_keys:
                    txt = str(val.get(ck, ''))
                    if len(txt) > len(best):
                        best = txt

        # Fallback: largest content from any output
        if len(best) < 100:
            for val in outputs.values():
                if isinstance(val, dict):
                    for ck in content_keys:
                        txt = str(val.get(ck, ''))
                        if len(txt) > len(best):
                            best = txt
        return best

    @staticmethod
    def _build_result(
        task: str, task_type: str,
        outputs: Dict, skills_used: List[str],
        errors: List[str], warnings: List[str],
        start_time: float, stopped: bool = False,
        too_hard: bool = False,
    ) -> Dict[str, Any]:
        """Assemble the execution result dict."""
        result = {
            "success": len(outputs) > 0 and not stopped and len(errors) == 0,
            "task": task,
            "task_type": task_type,
            "skills_used": list(set(skills_used)),
            "steps_executed": len(outputs),
            "outputs": outputs,
            "final_output": list(outputs.values())[-1] if outputs else None,
            "errors": errors,
            "warnings": warnings,
            "execution_time": time.time() - start_time,
            "stopped_early": stopped,
        }
        if too_hard:
            result["too_hard"] = True
        return result

    async def _fallback_llm_response(self, task: str) -> Optional[str]:
        """
        Generate direct LLM response when no skills are needed.

        Used for simple conversational queries where no execution plan
        is required - the LLM can respond directly.

        Priority: DSPy API (fast, ~10-30s) > claude-cli-llm skill (slow, shells out to CLI)

        Args:
            task: The user's query/task

        Returns:
            LLM response string or None if generation fails
        """
        # Inject system_prompt if configured (essential for multi-agent roles)
        prompt = task
        if self.config.system_prompt:
            prompt = f"[System: {self.config.system_prompt}]\n\n{task}"

        # 1. Try DSPy API first (fastest — direct HTTP to Anthropic/OpenAI)
        # Run in executor so sync HTTP doesn't block the event loop
        # (critical for multi-agent parallel execution)
        try:
            import dspy
            import asyncio as _aio
            if hasattr(dspy.settings, 'lm') and dspy.settings.lm is not None:
                lm = dspy.settings.lm
                loop = _aio.get_running_loop()
                response = await loop.run_in_executor(None, lambda: lm(prompt=prompt))
                if isinstance(response, list):
                    text = response[0] if response else None
                else:
                    text = str(response) if response else None
                if text and len(text.strip()) > 10:
                    return text.strip()
        except Exception as e:
            logger.debug(f"DSPy LLM call failed: {e}")

        # 2. Fallback to claude-cli-llm skill (slower — shells out to CLI binary)
        try:
            if self.skills_registry:
                skill = self.skills_registry.get_skill('claude-cli-llm')
                if skill and skill.tools:
                    llm_tool = skill.tools.get('claude_cli_llm_tool') or skill.tools.get('generate_text_tool')
                    if llm_tool:
                        result = llm_tool({'prompt': prompt})
                        if isinstance(result, dict):
                            return result.get('response') or result.get('content') or result.get('text')
                        return str(result) if result else None
        except Exception as e:
            logger.warning(f"Fallback LLM response failed: {e}")

        return None

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
    model: str = "",
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
    from Jotty.core.foundation.config_defaults import DEFAULT_MODEL_ALIAS
    model = model or DEFAULT_MODEL_ALIAS
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
    'ExecutionContextManager',
    'ExecutionStep',
    'create_autonomous_agent',
]

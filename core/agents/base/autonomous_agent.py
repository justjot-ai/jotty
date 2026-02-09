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
from typing import Any, Callable, Dict, List, Optional

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
    enable_replanning: bool = False  # Disabled: stop immediately on failure
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
    verification: str = ""
    fallback_skill: str = ""


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
                direct_llm: If True, bypass skill pipeline entirely and use
                    LLM directly with system_prompt. Used by multi-agent mode
                    for specialized sub-agents that do analysis/synthesis.

        Returns:
            Dict with execution results
        """
        config: AutonomousAgentConfig = self.config
        start_time = time.time()
        status_callback = kwargs.pop('status_callback', None)
        # Learning context is kept separate from task to prevent pollution
        # of search queries, entity extraction, and skill params
        learning_context = kwargs.pop('learning_context', None)
        direct_llm = kwargs.pop('direct_llm', False)

        def _status(stage: str, detail: str = ""):
            if status_callback:
                try:
                    status_callback(stage, detail)
                except Exception:
                    pass
            logger.info(f"{stage}: {detail}" if detail else stage)

        # ── DIRECT LLM MODE ────────────────────────────────────────────
        # For multi-agent sub-agents that are specialized via system_prompt:
        # Skip the entire skill pipeline (task inference, skill discovery,
        # skill selection, planning) and call the LLM directly.
        # This turns a ~120s pipeline into a ~15-20s direct LLM call.
        if direct_llm:
            _status("Direct LLM", "bypassing skill pipeline (multi-agent sub-agent)")
            llm_response = await self._fallback_llm_response(task)
            if llm_response:
                return {
                    "success": True,
                    "task": task,
                    "task_type": "direct_llm",
                    "skills_used": ["direct-llm"],
                    "steps_executed": 1,
                    "outputs": {"llm_response": llm_response},
                    "final_output": llm_response,
                    "errors": [],
                    "execution_time": time.time() - start_time,
                }
            return {
                "success": False,
                "task": task,
                "error": "Direct LLM response failed",
                "skills_used": [],
                "steps_executed": 0,
                "outputs": {},
                "execution_time": time.time() - start_time,
            }

        # Reset excluded skills for new execution (single source of truth: executor)
        self.executor.clear_exclusions()

        # Steps 1+2: Infer task type AND discover skills IN PARALLEL
        # These are independent: task_type uses LLM/keywords, discovery queries registry.
        # Running them concurrently saves ~5-10s vs sequential.
        import asyncio as _asyncio

        _status("Analyzing", "inferring task type + discovering skills (parallel)")

        async def _infer_type_async():
            return self._infer_task_type(task)

        task_type, all_skills = await _asyncio.gather(
            _infer_type_async(),
            self._discover_skills(task),
        )
        _status("Task type", task_type)
        _status("Skills found", f"{len(all_skills)} potential")

        # Report any skills that failed to load (helps debugging)
        if self.skills_registry and hasattr(self.skills_registry, 'get_failed_skills'):
            failed = self.skills_registry.get_failed_skills()
            if failed:
                _status("Warning", f"{len(failed)} skills failed to load: {', '.join(failed.keys())}")
                logger.warning(f"Failed skills: {failed}")

        # Step 3: Select best skills (uses CLEAN task, cached by task_type)
        # Inject approach guidance from learning context so skill selector
        # knows which tool combos worked/failed for this task type.
        _status("Selecting", "choosing best skills")
        _selection_task = task
        if learning_context:
            # Extract only the approach-relevant hints (avoid/use lines)
            _approach_lines = [
                ln for ln in learning_context.split('\n')
                if any(kw in ln.lower() for kw in ['avoid', 'failed approach', 'successful approach', 'use:'])
            ]
            if _approach_lines:
                _selection_task = f"{task}\n\n[Skill guidance from past experience]:\n" + '\n'.join(_approach_lines[:5])
        skills = await self._select_skills(_selection_task, all_skills, task_type=task_type)

        # OPTIMIZATION: Strip heavy/irrelevant skills for knowledge-only tasks.
        # Tasks about design, architecture, algorithms, explanations etc. don't
        # need web research, arxiv papers, slideshows, or mindmaps — the LLM
        # already knows the answer. These tools add 60-300s of latency and
        # lead the planner to create unnecessarily complex multi-step plans.
        _KNOWLEDGE_TASK_TYPES = {'analysis', 'explanation', 'design', 'architecture',
                                 'algorithm', 'tutorial', 'comparison', 'review'}
        _HEAVY_SKILLS = {
            'web-search', 'web-scraper',           # web scraping: 30-120s waste
            'arxiv-downloader',                     # paper download: unnecessary for design
            'mindmap-generator', 'pptx-editor',    # presentation tools: not needed
            'image-generator',                      # image gen: not needed
            'screener-financials',                  # finance: irrelevant
        }
        # Skills that ARE useful for knowledge tasks — keep these
        _KNOWLEDGE_USEFUL = {'claude-cli-llm', 'file-operations', 'calculator',
                             'text-utils', 'shell-exec', 'unit-converter'}
        _task_lower = task.lower()
        _is_knowledge_task = (
            task_type in _KNOWLEDGE_TASK_TYPES
            or any(kw in _task_lower for kw in [
                'design', 'architect', 'explain', 'compare', 'review',
                'algorithm', 'pattern', 'best practice', 'how to implement',
                'trade-off', 'pros and cons', 'what is', 'describe',
            ])
        )
        # Only strip if the user didn't explicitly ask for web/papers/slides
        _wants_heavy = any(kw in _task_lower for kw in [
            'search', 'find online', 'latest', 'recent', 'news', 'current',
            'scrape', 'fetch', 'url', 'http', 'website', 'browse',
            'arxiv', 'paper', 'slides', 'presentation', 'mindmap',
        ])
        if _is_knowledge_task and not _wants_heavy:
            _before = len(skills)
            # For pure knowledge/design tasks: keep ONLY useful skills, strip everything else
            # The LLM knows the content; we just need to generate text and optionally save it.
            _remaining = [s for s in skills if s.get('name') in _KNOWLEDGE_USEFUL]
            if not _remaining:
                # Ensure at least claude-cli-llm is available
                _remaining = [s for s in all_skills if s.get('name') == 'claude-cli-llm'][:1]
            if _remaining and len(_remaining) < _before:
                skills = _remaining
                _stripped = _before - len(skills)
                _status("Optimized", f"kept {len(skills)} useful skills for knowledge task (stripped {_stripped} heavy)")

        _status("Skills selected", ", ".join(s['name'] for s in skills[:5]))

        # TOO_HARD early-exit: if no skills found at all, signal difficulty
        # rather than wasting a full execution attempt (MAS-ZERO inspired)
        if not all_skills and not skills:
            _status("TOO_HARD", "no relevant skills found for this sub-task")
            return {
                "success": False,
                "task": task,
                "task_type": task_type,
                "too_hard": True,
                "skills_used": [],
                "steps_executed": 0,
                "outputs": {},
                "final_output": None,
                "errors": ["TOO_HARD: No relevant skills found for this sub-task"],
                "execution_time": time.time() - start_time,
            }

        # Step 4: Create plan — or skip planning for direct-LLM tasks
        #
        # OPTIMIZATION: The planner LLM call returns 0 steps ~80% of the time
        # for simple tasks (analysis, Q&A).  Detect this early and skip the
        # planning LLM call entirely — go straight to direct LLM response.
        # This saves ~15s per simple task.
        #
        # Extended: Also skip planning when only LLM + simple utility skills
        # are selected. The planner adds no value for "think + write to file" tasks.
        skill_names = {s['name'] for s in skills}
        # Skills that don't need a multi-step LLM plan — they're simple enough
        # for direct invocation or heuristic routing.
        _direct_llm_only = {'claude-cli-llm', 'calculator', 'unit-converter',
                            'text-utils', 'file-operations', 'shell-exec'}
        _needs_planning = not skill_names.issubset(_direct_llm_only)

        # OVERRIDE: Code generation/creation tasks ALWAYS need planning,
        # even with simple skills. Without a plan, the agent tries to shell-exec
        # the task description as code instead of generating code first.
        if not _needs_planning and task_type in ('creation', 'generation', 'automation'):
            _needs_planning = True
            logger.info(f"📋 Forcing planning for {task_type} task (code gen needs a multi-step plan)")

        steps = []
        if _needs_planning:
            _status("Planning", "creating execution plan")
            # Pass learning context separately — it gets stripped by _abstract_task_for_planning
            # if embedded in the task, so we pass it as previous_outputs metadata
            prev_outputs = {}
            if learning_context:
                prev_outputs['_learning_guidance'] = learning_context[:2000]
            steps = await self._create_plan(task, task_type, skills, previous_outputs=prev_outputs if prev_outputs else None)
            _status("Plan ready", f"{len(steps)} steps")
        else:
            _status("Direct mode", f"simple task — skipping planner (skills: {', '.join(skill_names)})")

        if not steps:
            # No multi-step plan needed — direct LLM response
            _status("Generating", "direct LLM response")
            llm_response = await self._fallback_llm_response(task)
            if llm_response:
                # If file-operations was selected and task wants output saved,
                # write the LLM response to a file automatically.
                if 'file-operations' in skill_names:
                    try:
                        _save_kws = ['save', 'write to', 'create a file', 'output to']
                        if any(kw in task.lower() for kw in _save_kws) and self.skills_registry:
                            fo_skill = self.skills_registry.get_skill('file-operations')
                            if fo_skill and fo_skill.tools:
                                write_tool = fo_skill.tools.get('write_file_tool')
                                if write_tool:
                                    # Infer filename from planner utility
                                    from Jotty.core.agents._plan_utils_mixin import PlanUtilsMixin
                                    _fname = PlanUtilsMixin._infer_filename_from_task(None, task) or 'output.md'
                                    write_tool({'path': _fname, 'content': llm_response})
                                    _status("Saved", f"output written to {_fname}")
                    except Exception as e:
                        logger.debug(f"Auto-save failed (non-critical): {e}")

                return {
                    "success": True,
                    "task": task,
                    "task_type": task_type,
                    "skills_used": ["claude-cli-llm"],
                    "steps_executed": 1,
                    "outputs": {"llm_response": llm_response},
                    "final_output": llm_response,
                    "errors": [],
                    "execution_time": time.time() - start_time,
                }
            return {
                "success": False,
                "task": task,
                "error": "Could not generate response",
                "skills_used": [],
                "steps_executed": 0,
                "outputs": {},
            }

        # Step 5: Execute steps (using while loop to support dynamic step modification)
        outputs = {}
        skills_used = []
        errors = []          # Unrecovered fatal errors
        warnings = []        # Recovered errors (replanning fixed them)
        replan_count = 0
        execution_stopped = False
        i = 0

        while i < len(steps):
            step = steps[i]
            _status(f"Step {i+1}/{len(steps)}", f"{step.skill_name}: {step.description[:100]}")

            result = await self._execute_step(step, outputs, status_callback)

            if result.get('success'):
                outputs[step.output_key or f'step_{i}'] = result
                skills_used.append(step.skill_name)
                _status(f"Step {i+1}", "completed")
                i += 1
            else:
                error_msg = result.get('error', 'Unknown error')
                errors.append(f"Step {i+1}: {error_msg}")
                _status(f"Step {i+1}", f"failed: {error_msg}")

                # For optional steps, continue without stopping
                if step.optional:
                    _status("Continuing", "optional step failed, skipping")
                    i += 1
                    continue

                # Try reflective replanning before stopping
                if config.enable_replanning and replan_count < config.max_replans:
                    _status("Replanning", "adapting to failure with reflection")

                    # Exclude skill for structural failures
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
                        # Splice new steps into remaining execution
                        steps = steps[:i + 1] + new_steps
                        replan_count += 1
                        # Move error to warnings since replanning is recovering
                        if errors:
                            warnings.append(errors.pop())
                        _status("Replanned", f"{len(new_steps)} new steps (reflection: {reflection[:200]})")
                        i += 1
                        continue

                # No replanning possible or replanning produced no steps - stop
                _status("Stopped", f"execution halted: {error_msg[:80]}")
                execution_stopped = True
                break

        # Determine final output
        final_output = list(outputs.values())[-1] if outputs else None

        # Success if we produced output and didn't stop early from unrecovered errors
        # Recovered errors (replanning fixed them) are warnings, not failures
        is_success = len(outputs) > 0 and not execution_stopped and len(errors) == 0

        return {
            "success": is_success,
            "task": task,
            "task_type": task_type,
            "skills_used": list(set(skills_used)),
            "steps_executed": len(outputs),
            "outputs": outputs,
            "final_output": final_output,
            "errors": errors,
            "warnings": warnings,
            "execution_time": time.time() - start_time,
            "stopped_early": execution_stopped,
        }

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
    'ExecutionStep',
    'create_autonomous_agent',
]

"""Pilot Swarm - Autonomous goal-completion engine.

The universal "do anything" swarm. Decomposes any goal into subtasks
and executes using the right tools: web search, coding, terminal,
skill creation, or delegation to specialized swarms.

Execution loop:
    Phase 1: PLAN — decompose goal into ordered subtasks
    Phase 2: EXECUTE — run each subtask with the right agent
    Phase 3: VALIDATE — check if goal was achieved
    Phase 4: ADAPT — re-plan if validation fails (up to max_retries)

Usage:
    from Jotty.core.swarms.pilot_swarm import pilot
    result = await pilot("Set up a FastAPI project with JWT auth and tests")
"""

import asyncio
import json
import logging
import os
import subprocess
from typing import Dict, Any, Optional, List
from pathlib import Path

import dspy

from ..base_swarm import (
    BaseSwarm, SwarmBaseConfig, SwarmResult, AgentRole,
    register_swarm, ExecutionTrace,
)
from ..base import DomainSwarm, AgentTeam, PhaseExecutor

from .types import (
    SubtaskType, SubtaskStatus, Subtask,
    PilotConfig, PilotResult, AVAILABLE_SWARMS,
)
from .agents import (
    PilotPlannerAgent, PilotSearchAgent, PilotCoderAgent,
    PilotTerminalAgent, PilotSkillWriterAgent, PilotValidatorAgent,
)

logger = logging.getLogger(__name__)


@register_swarm("pilot")
class PilotSwarm(DomainSwarm):
    """
    Autonomous Pilot Swarm — the general-purpose goal-completion engine.

    Takes ANY goal, decomposes it into subtasks, and executes using
    the right combination of tools:

    - Search: web search + LLM synthesis
    - Code: write/edit files with production-quality code
    - Terminal: execute safe shell commands
    - Create Skill: build new reusable Jotty skills
    - Delegate: hand off to specialized swarms (coding, research, testing, etc.)
    - Analyze: LLM-powered reasoning and synthesis

    6 Specialized Agents:
    1. PlannerAgent — decomposes goals into ordered subtasks (Sonnet)
    2. SearchAgent — web search + result synthesis
    3. CoderAgent — code/file generation (Sonnet)
    4. TerminalAgent — safe command execution
    5. SkillWriterAgent — creates new Jotty skills (Sonnet)
    6. ValidatorAgent — validates goal completion
    """

    AGENT_TEAM = AgentTeam.define(
        (PilotPlannerAgent, "Planner", "_planner"),
        (PilotSearchAgent, "Searcher", "_searcher"),
        (PilotCoderAgent, "Coder", "_coder"),
        (PilotTerminalAgent, "Terminal", "_terminal"),
        (PilotSkillWriterAgent, "SkillWriter", "_skill_writer"),
        (PilotValidatorAgent, "Validator", "_validator"),
    )

    def __init__(self, config: PilotConfig = None):
        super().__init__(config or PilotConfig())

    async def _execute_domain(self, topic: str = None, **kwargs) -> PilotResult:
        """Execute goal completion."""
        return await self.run_goal(goal=topic, **kwargs)

    async def run_goal(
        self,
        goal: str,
        context: str = "",
        send_telegram: bool = None,
    ) -> PilotResult:
        """
        Execute an autonomous goal-completion loop.

        Args:
            goal: The goal to accomplish (any natural language description)
            context: Optional context from previous work
            send_telegram: Whether to send results to Telegram

        Returns:
            PilotResult with execution details and artifacts
        """
        self._init_agents()
        config = self.config

        logger.info(f"PilotSwarm starting: {goal[:100]}")

        async def _run_phases(executor: PhaseExecutor) -> PilotResult:
            return await self._execute_phases(executor, goal, context, config, send_telegram)

        return await self._safe_execute_domain(
            task_type='pilot_execution',
            default_tools=['planning', 'search', 'code', 'terminal', 'skill_write', 'validate'],
            result_class=PilotResult,
            execute_fn=_run_phases,
            output_data_fn=lambda r: {
                'subtasks_completed': r.subtasks_completed if hasattr(r, 'subtasks_completed') else 0,
                'artifacts_count': len(r.artifacts) if hasattr(r, 'artifacts') else 0,
            },
            input_data_fn=lambda: {'goal': goal},
        )

    async def _execute_phases(
        self,
        executor: PhaseExecutor,
        goal: str,
        context: str,
        config: PilotConfig,
        send_telegram: bool = None,
    ) -> PilotResult:
        """Execute the Plan → Execute → Validate loop."""

        all_results = {}
        artifacts = []
        skills_created = []
        delegated_to = []

        # ==============================================================
        # PHASE 1: Planning
        # ==============================================================
        plan = await executor.run_phase(
            1, "Goal Decomposition", "Planner", AgentRole.PLANNER,
            self._planner.plan(
                goal=goal,
                available_swarms='\n'.join(AVAILABLE_SWARMS),
                context=context or "",
            ),
            input_data={'goal': goal},
            tools_used=['planning'],
        )

        subtasks_raw = plan.get('subtasks', []) if isinstance(plan, dict) else []
        reasoning = plan.get('reasoning', '') if isinstance(plan, dict) else ''

        # Parse subtasks into Subtask objects
        subtasks = []
        for st in subtasks_raw:
            if isinstance(st, dict):
                try:
                    st_type = SubtaskType(st.get('type', 'analyze'))
                except ValueError:
                    st_type = SubtaskType.ANALYZE
                subtasks.append(Subtask(
                    id=st.get('id', f's{len(subtasks) + 1}'),
                    type=st_type,
                    description=st.get('description', ''),
                    tool_hint=st.get('tool_hint', ''),
                    depends_on=st.get('depends_on', []),
                ))

        # Limit subtasks
        subtasks = subtasks[:config.max_subtasks]

        logger.info(f"Plan: {len(subtasks)} subtasks — {reasoning[:100]}")
        for st in subtasks:
            logger.info(f"  [{st.id}] {st.type.value}: {st.description[:80]}")

        # ==============================================================
        # PHASE 2 + 3 + 4: Execute → Validate → Retry loop
        # ==============================================================
        retry_count = 0
        success = False
        assessment = ''

        while True:
            # ---- Phase 2: Execute subtasks (parallel waves) ----
            completed_ids = {st.id for st in subtasks if st.status == SubtaskStatus.COMPLETED}
            pending_subtasks = [st for st in subtasks if st.status == SubtaskStatus.PENDING]

            waves = self._compute_waves(pending_subtasks)
            semaphore = asyncio.Semaphore(config.max_concurrent)

            for wave in waves:
                async def _run_in_wave(st: Subtask) -> None:
                    # Check dependencies
                    unmet = [d for d in st.depends_on if d not in completed_ids]
                    if unmet:
                        logger.warning(f"  [{st.id}] Skipping — unmet deps: {unmet}")
                        st.status = SubtaskStatus.SKIPPED
                        return

                    st.status = SubtaskStatus.RUNNING
                    logger.info(f"  [{st.id}] Executing: {st.type.value} — {st.description[:60]}")

                    prev_context = self._build_context(all_results)

                    async with semaphore:
                        try:
                            result = await self._execute_subtask_with_retry(st, prev_context, config)
                            all_results[st.id] = result
                            st.result = json.dumps(result, default=str)[:2000]
                            st.status = SubtaskStatus.COMPLETED
                            completed_ids.add(st.id)

                            # Track artifacts and metadata
                            if st.type == SubtaskType.CODE:
                                for op in result.get('file_operations', []):
                                    if isinstance(op, dict) and op.get('file_path'):
                                        artifacts.append(op['file_path'])
                            elif st.type == SubtaskType.CREATE_SKILL:
                                if result.get('skill_name'):
                                    skills_created.append(result['skill_name'])
                                    artifacts.append(f"skills/{result['skill_name']}/")
                            elif st.type == SubtaskType.DELEGATE:
                                if result.get('swarm_used'):
                                    delegated_to.append(result['swarm_used'])

                            logger.info(f"  [{st.id}] Completed")

                        except Exception as e:
                            st.status = SubtaskStatus.FAILED
                            st.error = str(e)
                            logger.error(f"  [{st.id}] Failed: {e}")
                            all_results[st.id] = {'error': str(e)}
                            completed_ids.add(st.id)

                tasks = [_run_in_wave(st) for st in wave]
                await asyncio.gather(*tasks, return_exceptions=True)

            # ---- Phase 3: Validation ----
            results_summary = self._build_results_summary(subtasks, all_results)

            validation = await executor.run_phase(
                3, "Goal Validation", "Validator", AgentRole.REVIEWER,
                self._validator.validate(goal=goal, results_summary=results_summary),
                tools_used=['validation'],
            )

            success = validation.get('success', False) if isinstance(validation, dict) else False
            assessment = validation.get('assessment', '') if isinstance(validation, dict) else ''
            remaining_gaps = validation.get('remaining_gaps', []) if isinstance(validation, dict) else []

            if success or retry_count >= config.max_retries:
                break

            # ---- Phase 4: Re-plan from gaps ----
            logger.info(f"  Retry {retry_count + 1}/{config.max_retries}: re-planning for gaps: {remaining_gaps}")

            replan_context = self._build_replan_context(goal, subtasks, all_results, remaining_gaps)

            replan = await executor.run_phase(
                4, "Re-planning", "Planner", AgentRole.PLANNER,
                self._planner.plan(
                    goal=goal,
                    available_swarms='\n'.join(AVAILABLE_SWARMS),
                    context=replan_context,
                ),
                input_data={'retry': retry_count + 1, 'gaps': remaining_gaps},
                tools_used=['planning'],
            )

            new_subtasks_raw = replan.get('subtasks', []) if isinstance(replan, dict) else []
            for st_raw in new_subtasks_raw:
                if isinstance(st_raw, dict):
                    try:
                        st_type = SubtaskType(st_raw.get('type', 'analyze'))
                    except ValueError:
                        st_type = SubtaskType.ANALYZE
                    new_id = f"r{retry_count + 1}_{st_raw.get('id', f's{len(subtasks) + 1}')}"
                    # Remap depends_on to prefixed IDs
                    raw_deps = st_raw.get('depends_on', [])
                    deps = [f"r{retry_count + 1}_{d}" if d not in completed_ids else d for d in raw_deps]
                    subtasks.append(Subtask(
                        id=new_id,
                        type=st_type,
                        description=st_raw.get('description', ''),
                        tool_hint=st_raw.get('tool_hint', ''),
                        depends_on=deps,
                    ))

            retry_count += 1

        completed_count = sum(1 for st in subtasks if st.status == SubtaskStatus.COMPLETED)

        logger.info(f"PilotSwarm {'SUCCEEDED' if success else 'PARTIAL'}: "
                    f"{completed_count}/{len(subtasks)} subtasks, "
                    f"{len(artifacts)} artifacts")

        # Cost summary
        try:
            from Jotty.core.foundation.direct_anthropic_lm import get_cost_tracker
            tracker = get_cost_tracker()
            metrics = tracker.get_metrics()
            logger.info(f"COST: ${metrics.total_cost:.4f} | {metrics.total_calls} calls")
        except Exception:
            pass

        exec_time = executor.elapsed()

        return PilotResult(
            success=success,
            swarm_name=config.name,
            domain=config.domain,
            output={
                'goal': goal,
                'assessment': assessment,
                'reasoning': reasoning,
                'subtasks_completed': completed_count,
                'subtasks_total': len(subtasks),
                'artifacts': artifacts,
            },
            execution_time=exec_time,
            goal=goal,
            subtasks_completed=completed_count,
            subtasks_total=len(subtasks),
            artifacts=artifacts,
            skills_created=skills_created,
            delegated_to=delegated_to,
            retry_count=retry_count,
        )

    # =========================================================================
    # SUBTASK EXECUTORS
    # =========================================================================

    async def _execute_subtask(
        self,
        subtask: Subtask,
        context: str,
        config: PilotConfig,
    ) -> Dict[str, Any]:
        """Execute a single subtask using the appropriate agent."""

        dispatch = {
            SubtaskType.SEARCH: self._execute_search,
            SubtaskType.CODE: self._execute_code,
            SubtaskType.TERMINAL: self._execute_terminal,
            SubtaskType.CREATE_SKILL: self._execute_create_skill,
            SubtaskType.DELEGATE: self._execute_delegate,
            SubtaskType.ANALYZE: self._execute_analyze,
            SubtaskType.BROWSE: self._execute_browse,
        }

        handler = dispatch.get(subtask.type, self._execute_analyze)
        return await handler(subtask, context, config)

    async def _execute_search(self, subtask: Subtask, context: str, config: PilotConfig) -> Dict[str, Any]:
        """Execute a search subtask."""
        return await self._searcher.search(task=subtask.description, context=context)

    async def _execute_code(self, subtask: Subtask, context: str, config: PilotConfig) -> Dict[str, Any]:
        """Execute a coding subtask — generates code, reads, edits, or writes files."""
        result = await self._coder.code(task=subtask.description, context=context)

        if result.get('file_operations'):
            for op in result['file_operations']:
                if not isinstance(op, dict):
                    continue

                file_path = op.get('file_path', '')
                content = op.get('content', '')
                action = op.get('action', 'create')

                if action == 'read' and file_path:
                    op['read_content'] = self._read_file(file_path)
                    logger.info(f"    Read: {file_path}")
                    continue

                if action == 'edit' and file_path and config.allow_file_write:
                    old_content = op.get('old_content', '')
                    if old_content and content:
                        success = self._edit_file(file_path, old_content, content)
                        op['edit_success'] = success
                    else:
                        op['edit_success'] = False
                        logger.warning(f"    Edit skipped — missing old_content or content for {file_path}")
                    continue

                if file_path and content and config.allow_file_write:
                    try:
                        path = Path(file_path)
                        path.parent.mkdir(parents=True, exist_ok=True)

                        if action == 'append' and path.exists():
                            with open(path, 'a') as f:
                                f.write('\n' + content)
                        else:
                            with open(path, 'w') as f:
                                f.write(content)

                        logger.info(f"    Wrote: {file_path} ({len(content)} chars)")
                    except Exception as e:
                        logger.warning(f"    Failed to write {file_path}: {e}")

        return result

    async def _execute_terminal(self, subtask: Subtask, context: str, config: PilotConfig) -> Dict[str, Any]:
        """Execute a terminal subtask — runs safe shell commands."""
        result = await self._terminal.execute(task=subtask.description, context=context)

        if not config.allow_terminal:
            result['note'] = 'Terminal execution disabled — commands generated but not run'
            return result

        command_outputs = []
        for cmd in result.get('commands', []):
            if not isinstance(cmd, dict):
                continue

            command = cmd.get('command', '')
            is_safe = cmd.get('safe', False)

            if not is_safe:
                logger.warning(f"    Skipping unsafe command: {command}")
                command_outputs.append({
                    'command': command, 'skipped': True,
                    'reason': 'Flagged as unsafe by TerminalAgent',
                })
                continue

            try:
                cwd = config.working_directory or str(Path.cwd())
                proc = subprocess.run(
                    command, shell=True, capture_output=True,
                    text=True, timeout=30, cwd=cwd,
                )
                command_outputs.append({
                    'command': command,
                    'stdout': proc.stdout[:2000] if proc.stdout else '',
                    'stderr': proc.stderr[:500] if proc.stderr else '',
                    'returncode': proc.returncode,
                })
                logger.info(f"    Ran: {command} (rc={proc.returncode})")
            except subprocess.TimeoutExpired:
                command_outputs.append({'command': command, 'error': 'Timed out (30s)'})
            except Exception as e:
                command_outputs.append({'command': command, 'error': str(e)})

        result['command_outputs'] = command_outputs
        return result

    async def _execute_create_skill(self, subtask: Subtask, context: str, config: PilotConfig) -> Dict[str, Any]:
        """Execute a skill creation subtask."""
        # Derive skill name from tool_hint or description
        raw_name = subtask.tool_hint or subtask.description.split()[:3]
        if isinstance(raw_name, list):
            raw_name = '-'.join(raw_name)
        skill_name = ''.join(c for c in raw_name.lower().replace(' ', '-') if c.isalnum() or c == '-')[:30]

        result = await self._skill_writer.write_skill(
            description=subtask.description,
            skill_name=skill_name,
        )

        # Write skill files
        if config.allow_file_write and result.get('skill_yaml') and result.get('tools_py'):
            skills_dir = Path(__file__).parent.parent.parent.parent / "skills" / skill_name
            try:
                skills_dir.mkdir(parents=True, exist_ok=True)
                (skills_dir / "skill.yaml").write_text(result['skill_yaml'])
                (skills_dir / "tools.py").write_text(result['tools_py'])
                result['skill_name'] = skill_name
                result['skill_path'] = str(skills_dir)
                logger.info(f"    Created skill: {skills_dir}")
            except Exception as e:
                logger.warning(f"    Failed to write skill files: {e}")

        return result

    async def _execute_delegate(self, subtask: Subtask, context: str, config: PilotConfig) -> Dict[str, Any]:
        """Delegate to a specialized swarm."""
        if not config.allow_delegation:
            return {'note': 'Delegation disabled', 'skipped': True}

        swarm_name = subtask.tool_hint or 'coding'

        try:
            from ..base_swarm import SwarmRegistry
            swarm_class = SwarmRegistry.get(swarm_name)

            if swarm_class is None:
                return {'error': f'Swarm "{swarm_name}" not found in registry'}

            logger.info(f"    Delegating to {swarm_name} swarm...")
            swarm_instance = swarm_class()
            result = await swarm_instance.execute(subtask.description)

            return {
                'swarm_used': swarm_name,
                'success': getattr(result, 'success', False),
                'output': str(getattr(result, 'output', {}))[:1000],
            }
        except Exception as e:
            logger.error(f"    Delegation to {swarm_name} failed: {e}")
            return {'error': f'Delegation failed: {e}', 'swarm_used': swarm_name}

    async def _execute_analyze(self, subtask: Subtask, context: str, config: PilotConfig) -> Dict[str, Any]:
        """Execute an analysis subtask using LLM reasoning."""
        return await self._searcher.search(task=subtask.description, context=context)

    async def _execute_browse(self, subtask: Subtask, context: str, config: PilotConfig) -> Dict[str, Any]:
        """Execute a browse subtask — smart dispatch based on target type.

        - URLs → web scraper / fetch_webpage to get content
        - File paths → visual-inspector VLM for image/PDF/code analysis
        - General → browser-automation screenshot + extract
        """
        import re

        target = subtask.tool_hint or ''
        if not target:
            # Extract URL or file path from description
            url_match = re.search(r'https?://[^\s\'"<>]+', subtask.description)
            if url_match:
                target = url_match.group(0)
            else:
                path_match = re.search(r'[/~][\w./\-]+\.\w+', subtask.description)
                if path_match:
                    target = path_match.group(0)

        is_url = target.startswith('http://') or target.startswith('https://')

        # URL → fetch webpage content
        if is_url:
            return await self._browse_url(target, subtask.description, context)

        # File path → VLM visual inspection
        if target and not is_url:
            return await self._browse_file(target, subtask.description, context)

        # No clear target → try to extract from description with browser
        return await self._browse_url_fallback(subtask.description, context)

    async def _browse_url(self, url: str, task: str, context: str) -> Dict[str, Any]:
        """Fetch and analyze a URL using web-search/fetch_webpage_tool or web-scraper."""
        skills_root = Path(__file__).parent.parent.parent.parent / "skills"

        # Try web-search fetch_webpage_tool first (lighter weight)
        fetch_tool = self._load_skill_tool(skills_root / "web-search" / "tools.py", "web_search_skill",
                                           "fetch_webpage_tool")
        if fetch_tool:
            try:
                result = fetch_tool({'url': url, 'max_length': 15000})
                if isinstance(result, dict) and result.get('success'):
                    logger.info(f"    Fetched URL: {url} ({result.get('length', 0)} chars)")
                    return {
                        'url': url,
                        'title': result.get('title', ''),
                        'content': result.get('content', '')[:5000],
                        'synthesis': result.get('content', '')[:3000],
                        'fetched_via': result.get('fetched_via', 'fetch_webpage'),
                    }
            except Exception as e:
                logger.debug(f"    fetch_webpage_tool failed: {e}")

        # Try web-scraper
        scrape_tool = self._load_skill_tool(skills_root / "web-scraper" / "tools.py", "web_scraper_skill",
                                            "scrape_website_tool")
        if scrape_tool:
            try:
                result = scrape_tool({'url': url, 'output_format': 'markdown'})
                if isinstance(result, dict) and result.get('success'):
                    logger.info(f"    Scraped URL: {url} ({result.get('content_length', 0)} chars)")
                    return {
                        'url': url,
                        'title': result.get('title', ''),
                        'content': result.get('content', '')[:5000],
                        'synthesis': result.get('content', '')[:3000],
                        'fetched_via': 'web-scraper',
                    }
            except Exception as e:
                logger.debug(f"    scrape_website_tool failed: {e}")

        # Try browser-automation as last resort
        browser_tool = self._load_skill_tool(skills_root / "browser-automation" / "tools.py",
                                             "browser_automation_skill", "browser_navigate_tool")
        if browser_tool:
            try:
                result = browser_tool({'url': url, 'extract_text': True, 'screenshot': False})
                if isinstance(result, dict) and result.get('success'):
                    logger.info(f"    Browsed URL: {url}")
                    return {
                        'url': url,
                        'title': result.get('title', ''),
                        'content': result.get('text', '')[:5000],
                        'synthesis': result.get('text', '')[:3000],
                        'fetched_via': 'browser-automation',
                    }
            except Exception as e:
                logger.debug(f"    browser_navigate_tool failed: {e}")

        logger.warning(f"    No web tool available for {url}, falling back to search")
        return await self._searcher.search(task=f"Fetch content from {url}: {task}", context=context)

    async def _browse_file(self, file_path: str, task: str, context: str) -> Dict[str, Any]:
        """Inspect a local file using VLM visual-inspector."""
        skills_root = Path(__file__).parent.parent.parent.parent / "skills"
        tool = self._load_skill_tool(skills_root / "visual-inspector" / "tools.py",
                                     "visual_inspector_skill",
                                     "inspect_file_visually_tool", "visual_inspect_tool")
        if tool:
            try:
                result = tool({
                    'image_path': file_path,
                    'question': task,
                    'task_context': context,
                })
                logger.info(f"    VLM analyzed: {file_path}")
                return {
                    'visual_analysis': result.get('visual_state', result.get('result', str(result))),
                    'model': result.get('model', 'unknown'),
                    'file_path': file_path,
                    'synthesis': result.get('visual_state', ''),
                }
            except Exception as e:
                logger.error(f"    VLM inspection failed: {e}")

        # Fallback: just read the file as text
        try:
            p = Path(file_path)
            if p.exists() and p.stat().st_size < 50000:
                content = p.read_text(errors='replace')[:5000]
                return {'file_path': file_path, 'content': content, 'synthesis': content[:3000]}
        except Exception:
            pass

        return await self._searcher.search(task=task, context=context)

    async def _browse_url_fallback(self, task: str, context: str) -> Dict[str, Any]:
        """No clear target — fall back to search with browse intent."""
        return await self._searcher.search(task=f"Browse and find: {task}", context=context)

    @staticmethod
    def _load_skill_tool(tools_path: Path, module_name: str, *func_names: str):
        """Load a tool function from a skill's tools.py by absolute path."""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(module_name, str(tools_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for name in func_names:
                tool = getattr(mod, name, None)
                if tool:
                    return tool
        except Exception:
            pass
        return None

    # =========================================================================
    # SUBTASK RETRY WRAPPER
    # =========================================================================

    _TRANSIENT_ERRORS = (TimeoutError, ConnectionError, OSError)
    _TRANSIENT_STRINGS = ("rate limit", "rate_limit", "429", "503", "timeout", "connection reset")

    async def _execute_subtask_with_retry(
        self,
        subtask: Subtask,
        context: str,
        config: PilotConfig,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """Execute a subtask with retry on transient errors.

        Retries on: TimeoutError, ConnectionError, OSError, rate-limit strings.
        Does NOT retry on: ValueError, KeyError, TypeError, or other logic errors.
        Backoff: 1s after first failure, 3s after second.
        """
        backoff = [1, 3]
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                return await self._execute_subtask(subtask, context, config)
            except self._TRANSIENT_ERRORS as e:
                last_exc = e
                if attempt < max_retries:
                    logger.warning(f"  [{subtask.id}] Transient error (attempt {attempt + 1}): {e}, retrying...")
                    await asyncio.sleep(backoff[min(attempt, len(backoff) - 1)])
                    continue
                raise
            except Exception as e:
                err_str = str(e).lower()
                if any(s in err_str for s in self._TRANSIENT_STRINGS) and attempt < max_retries:
                    last_exc = e
                    logger.warning(f"  [{subtask.id}] Transient error (attempt {attempt + 1}): {e}, retrying...")
                    await asyncio.sleep(backoff[min(attempt, len(backoff) - 1)])
                    continue
                raise

        raise last_exc  # pragma: no cover — defensive

    # =========================================================================
    # PARALLEL WAVE COMPUTATION
    # =========================================================================

    @staticmethod
    def _compute_waves(subtasks: List[Subtask]) -> List[List[Subtask]]:
        """Compute execution waves from subtask dependencies.

        Wave 0: subtasks with no depends_on.
        Wave N: subtasks whose deps are all in waves 0..N-1.
        Circular deps: dumped into the final wave as a defensive fallback.
        """
        if not subtasks:
            return []

        # Map id -> subtask for lookup
        by_id = {st.id: st for st in subtasks}
        assigned: Dict[str, int] = {}  # subtask_id -> wave number
        waves: List[List[Subtask]] = []

        remaining = list(subtasks)
        max_iterations = len(subtasks) + 1  # safety cap

        for _ in range(max_iterations):
            if not remaining:
                break

            current_wave = []
            still_remaining = []

            for st in remaining:
                deps = [d for d in st.depends_on if d in by_id]
                if all(d in assigned for d in deps):
                    current_wave.append(st)
                    assigned[st.id] = len(waves)
                else:
                    still_remaining.append(st)

            if current_wave:
                waves.append(current_wave)
                remaining = still_remaining
            else:
                # Circular deps — dump all remaining into final wave
                waves.append(still_remaining)
                break

        return waves

    # =========================================================================
    # FILE READ/EDIT HELPERS
    # =========================================================================

    @staticmethod
    def _read_file(file_path: str, max_chars: int = 5000) -> str:
        """Read a file's content, returning the text or an error message."""
        try:
            p = Path(file_path)
            if not p.exists():
                return f"[ERROR] File not found: {file_path}"
            content = p.read_text(errors='replace')
            if len(content) > max_chars:
                return content[:max_chars] + f"\n... [truncated at {max_chars} chars, total {len(content)}]"
            return content
        except Exception as e:
            return f"[ERROR] Could not read {file_path}: {e}"

    @staticmethod
    def _edit_file(file_path: str, old_content: str, new_content: str) -> bool:
        """Surgically replace old_content with new_content in a file.

        Returns True on success, False if old_content not found or file missing.
        """
        try:
            p = Path(file_path)
            if not p.exists():
                logger.warning(f"    Edit failed — file not found: {file_path}")
                return False
            text = p.read_text(errors='replace')
            if old_content not in text:
                logger.warning(f"    Edit failed — old_content not found in {file_path}")
                return False
            text = text.replace(old_content, new_content, 1)
            p.write_text(text)
            logger.info(f"    Edited: {file_path}")
            return True
        except Exception as e:
            logger.warning(f"    Edit failed for {file_path}: {e}")
            return False

    # =========================================================================
    # REPLAN CONTEXT BUILDER
    # =========================================================================

    @staticmethod
    def _build_replan_context(
        goal: str,
        subtasks: List[Subtask],
        all_results: Dict[str, Any],
        remaining_gaps: List[str],
    ) -> str:
        """Build enriched context for re-planning after validation failure."""
        parts = [f"ORIGINAL GOAL: {goal}", "", "COMPLETED WORK:"]

        for st in subtasks:
            status = st.status.value
            result_preview = ""
            if st.id in all_results:
                r = all_results[st.id]
                if isinstance(r, dict):
                    for key in ['synthesis', 'explanation', 'content', 'assessment']:
                        if r.get(key):
                            result_preview = f" — {str(r[key])[:300]}"
                            break
            parts.append(f"  [{st.id}] {st.type.value} ({status}): {st.description[:80]}{result_preview}")

        parts.append("")
        parts.append("REMAINING GAPS (must be addressed):")
        for gap in remaining_gaps:
            parts.append(f"  - {gap}")

        parts.append("")
        parts.append("Re-plan to address ONLY the remaining gaps. Do NOT repeat completed work.")
        return '\n'.join(parts)

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _build_context(self, all_results: Dict[str, Any]) -> str:
        """Build context string from previous subtask results."""
        if not all_results:
            return "No previous results."

        parts = []
        for sid, result in all_results.items():
            if isinstance(result, dict):
                summary_parts = []

                # Direct keys
                for key in ['synthesis', 'explanation', 'assessment', 'key_findings',
                            'content', 'visual_analysis', 'skill_name', 'output',
                            'title', 'read_content']:
                    if key in result and result[key]:
                        val = result[key]
                        if isinstance(val, list):
                            val = '; '.join(str(v) for v in val[:5])
                        summary_parts.append(f"{key}: {str(val)[:800]}")

                # Extract read_content from file_operations
                for op in result.get('file_operations', []):
                    if isinstance(op, dict) and op.get('read_content'):
                        summary_parts.append(f"read_content ({op.get('file_path', '?')}): "
                                             f"{str(op['read_content'])[:800]}")

                # Terminal command outputs
                for cmd_out in result.get('command_outputs', []):
                    if isinstance(cmd_out, dict):
                        cmd = cmd_out.get('command', '')
                        stdout = cmd_out.get('stdout', '')
                        if stdout:
                            summary_parts.append(f"$ {cmd}\n{stdout[:1000]}")
                        elif cmd_out.get('error'):
                            summary_parts.append(f"$ {cmd} — ERROR: {cmd_out['error'][:200]}")

                if summary_parts:
                    parts.append(f"[{sid}] {'; '.join(summary_parts)}")

        return '\n'.join(parts[-8:]) if parts else "No usable results yet."

    def _build_results_summary(self, subtasks: List[Subtask], all_results: Dict) -> str:
        """Build a summary of all results for validation."""
        parts = []
        for st in subtasks:
            status = st.status.value
            result_preview = ''
            if st.id in all_results:
                r = all_results[st.id]
                if isinstance(r, dict):
                    if r.get('error'):
                        result_preview = f" — ERROR: {r['error'][:300]}"
                    elif r.get('command_outputs'):
                        # Terminal results: show actual command output
                        cmd_summaries = []
                        for co in r['command_outputs']:
                            if isinstance(co, dict) and co.get('stdout'):
                                cmd_summaries.append(f"$ {co.get('command','')}: {co['stdout'][:300]}")
                        if cmd_summaries:
                            result_preview = f" — {'; '.join(cmd_summaries)}"
                    elif r.get('content'):
                        result_preview = f" — {str(r['content'])[:400]}"
                    elif r.get('synthesis'):
                        result_preview = f" — {str(r['synthesis'])[:400]}"
                    elif r.get('visual_analysis'):
                        result_preview = f" — {str(r['visual_analysis'])[:400]}"
                    elif r.get('explanation'):
                        result_preview = f" — {str(r['explanation'])[:400]}"
                    elif r.get('skill_name'):
                        result_preview = f" — Created skill: {r['skill_name']}"
            parts.append(f"[{st.id}] {st.type.value} ({status}): {st.description[:100]}{result_preview}")

        return '\n'.join(parts)

    # =========================================================================
    # GOLD STANDARDS
    # =========================================================================

    def seed_gold_standards(self) -> None:
        """Seed default gold standards for evaluation."""
        self._init_agents()
        self.add_gold_standard(
            task_type='pilot_execution',
            input_data={'goal': 'any'},
            expected_output={
                'subtasks_completed': 1,
                'has_plan': True,
                'has_validation': True,
            },
            evaluation_criteria={
                'subtasks_completed': 0.5,
                'has_plan': 0.25,
                'has_validation': 0.25,
            }
        )
        logger.info("Seeded gold standards for PilotSwarm")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def pilot(
    goal: str,
    context: str = "",
    send_telegram: bool = False,
    allow_terminal: bool = True,
    allow_file_write: bool = True,
) -> PilotResult:
    """
    One-liner autonomous goal completion.

    Usage:
        from Jotty.core.swarms.pilot_swarm import pilot

        # Simple
        result = await pilot("Find the top 5 Python web frameworks and compare them")

        # With file creation
        result = await pilot("Create a FastAPI project with JWT auth and tests")

        # Create a new Jotty skill
        result = await pilot("Create a Jotty skill that converts CSV to JSON")
    """
    config = PilotConfig(
        allow_terminal=allow_terminal,
        allow_file_write=allow_file_write,
    )
    swarm = PilotSwarm(config)
    return await swarm.run_goal(goal=goal, context=context, send_telegram=send_telegram)


def pilot_sync(
    goal: str,
    context: str = "",
    send_telegram: bool = False,
    allow_terminal: bool = True,
    allow_file_write: bool = True,
) -> PilotResult:
    """Synchronous version of pilot."""
    import asyncio
    return asyncio.run(pilot(
        goal=goal, context=context, send_telegram=send_telegram,
        allow_terminal=allow_terminal, allow_file_write=allow_file_write,
    ))


__all__ = [
    'PilotSwarm', 'PilotConfig', 'PilotResult',
    'pilot', 'pilot_sync',
]

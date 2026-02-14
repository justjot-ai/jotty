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
        # PHASE 2: Execute subtasks
        # ==============================================================
        completed_ids = set()

        for st in subtasks:
            # Check dependencies
            unmet = [d for d in st.depends_on if d not in completed_ids]
            if unmet:
                logger.warning(f"  [{st.id}] Skipping — unmet deps: {unmet}")
                st.status = SubtaskStatus.SKIPPED
                continue

            st.status = SubtaskStatus.RUNNING
            logger.info(f"  [{st.id}] Executing: {st.type.value} — {st.description[:60]}")

            # Build context from previous results
            prev_context = self._build_context(all_results)

            try:
                result = await self._execute_subtask(st, prev_context, config)
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

        # ==============================================================
        # PHASE 3: Validation
        # ==============================================================
        results_summary = self._build_results_summary(subtasks, all_results)

        validation = await executor.run_phase(
            3, "Goal Validation", "Validator", AgentRole.REVIEWER,
            self._validator.validate(goal=goal, results_summary=results_summary),
            tools_used=['validation'],
        )

        success = validation.get('success', False) if isinstance(validation, dict) else False
        assessment = validation.get('assessment', '') if isinstance(validation, dict) else ''

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
            SubtaskType.BROWSE: self._execute_search,  # fallback to search
        }

        handler = dispatch.get(subtask.type, self._execute_analyze)
        return await handler(subtask, context, config)

    async def _execute_search(self, subtask: Subtask, context: str, config: PilotConfig) -> Dict[str, Any]:
        """Execute a search subtask."""
        return await self._searcher.search(task=subtask.description, context=context)

    async def _execute_code(self, subtask: Subtask, context: str, config: PilotConfig) -> Dict[str, Any]:
        """Execute a coding subtask — generates code and writes files."""
        result = await self._coder.code(task=subtask.description, context=context)

        if config.allow_file_write and result.get('file_operations'):
            for op in result['file_operations']:
                if isinstance(op, dict):
                    file_path = op.get('file_path', '')
                    content = op.get('content', '')
                    action = op.get('action', 'create')

                    if file_path and content:
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
                for key in ['synthesis', 'explanation', 'assessment', 'key_findings',
                            'stdout', 'skill_name', 'output']:
                    if key in result and result[key]:
                        val = result[key]
                        if isinstance(val, list):
                            val = '; '.join(str(v) for v in val[:3])
                        summary_parts.append(f"{key}: {str(val)[:300]}")
                if summary_parts:
                    parts.append(f"[{sid}] {'; '.join(summary_parts)}")

        return '\n'.join(parts[-5:]) if parts else "No usable results yet."

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
                        result_preview = f" — ERROR: {r['error'][:200]}"
                    elif r.get('synthesis'):
                        result_preview = f" — {str(r['synthesis'])[:200]}"
                    elif r.get('explanation'):
                        result_preview = f" — {str(r['explanation'])[:200]}"
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

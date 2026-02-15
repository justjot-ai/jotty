"""
ParadigmExecutor - Discussion paradigm implementations
======================================================

Extracted from ExecutionEngine for decomposition.
Implements MALLM-inspired paradigms: relay, debate, refinement.
Each paradigm uses _run_agent() for per-agent execution with fast-path.

Usage:
    executor = ParadigmExecutor(orchestrator)
    result = await executor.relay(goal, **kwargs)
"""

import asyncio
import logging
import time
from typing import Any, Dict

from Jotty.core.infrastructure.foundation.data_structures import EpisodeResult
from Jotty.core.infrastructure.foundation.exceptions import AgentExecutionError, LLMError
from Jotty.core.infrastructure.utils.async_utils import safe_status

logger = logging.getLogger(__name__)


def _extract_output_text(output: Any) -> str:
    """Extract clean text content from an agent output, avoiding nested repr() strings."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    # AgenticExecutionResult has .final_output or .outputs
    if hasattr(output, "final_output") and output.final_output:
        return _extract_output_text(output.final_output)
    if hasattr(output, "outputs") and isinstance(output.outputs, dict):
        # Get the last step's output
        for key in reversed(list(output.outputs.keys())):
            val = output.outputs[key]
            if isinstance(val, dict):
                for field in ("content", "response", "text", "output", "article", "code", "result"):
                    if field in val and val[field]:
                        return str(val[field])
            elif isinstance(val, str) and len(val) > 20:
                return val
    # EpisodeResult has .output
    if hasattr(output, "output") and output.output is not None:
        return _extract_output_text(output.output)
    # dict
    if isinstance(output, dict):
        for field in ("content", "response", "text", "output", "article", "code", "result"):
            if field in output and output[field]:
                return str(output[field])
    # Last resort: summary if available, otherwise str()
    if hasattr(output, "summary"):
        return output.summary
    return str(output)


class ParadigmExecutor:
    """
    Discussion paradigm implementations (relay, debate, refinement).

    Composed by ExecutionEngine. Holds a reference to the Orchestrator
    for access to agents, runners, semaphore, and learning hooks.
    """

    def __init__(self, manager: Any) -> None:
        self._manager = manager

    async def run_agent(
        self, runner: Any, sub_goal: str, agent_name: str, **kwargs: Any
    ) -> EpisodeResult:
        """
        Run a single agent within a paradigm, using fast-path when possible.

        LATENCY OPTIMIZATION: For simple sub-goals (no tool use needed),
        bypass the full AutoAgent pipeline and make a direct LLM call.
        Saves 2-4 LLM calls per agent.
        """
        _tool_keywords = [
            "search",
            "fetch",
            "scrape",
            "download",
            "upload",
            "send",
            "email",
            "telegram",
            "slack",
            "create file",
            "save file",
            "write file",
            "read file",
            "execute",
            "run code",
            "compile",
            "deploy",
            "database",
            "sql",
            "api call",
        ]
        _needs_tools = any(kw in sub_goal.lower() for kw in _tool_keywords)

        if not _needs_tools:
            # FAST PATH: Direct LLM call
            import dspy as _dspy

            lm = _dspy.settings.lm
            if lm:
                try:
                    _start = time.time()

                    hook_ctx = runner._run_hooks(
                        "pre_run",
                        goal=sub_goal,
                        agent_name=agent_name,
                        fast_path=True,
                    )
                    sub_goal = hook_ctx.get("goal", sub_goal)

                    _last_err = None
                    response = None
                    for _attempt in range(4):
                        try:
                            response = lm(prompt=sub_goal)
                            break
                        except Exception as _e:
                            _err_s = str(_e)
                            if (
                                "429" in _err_s
                                or "RateLimit" in _err_s
                                or "rate limit" in _err_s.lower()
                            ):
                                _delay = 8.0 * (2**_attempt)
                                logger.info(
                                    f"Paradigm fast-path rate limited, retry in {_delay:.0f}s"
                                )
                                time.sleep(_delay)
                                _last_err = _e
                            else:
                                raise
                    if response is None and _last_err:
                        raise _last_err
                    if isinstance(response, list):
                        response = response[0] if response else ""
                    response = str(response).strip()

                    _elapsed = time.time() - _start
                    logger.info(
                        f" Paradigm fast-path: {agent_name} " f"({_elapsed:.1f}s, 1 LLM call)"
                    )

                    result = EpisodeResult(
                        output=response,
                        success=bool(response),
                        trajectory=[],
                        tagged_outputs={agent_name: response},
                        episode=0,
                        execution_time=_elapsed,
                        architect_results=[],
                        auditor_results=[],
                        agent_contributions={agent_name: response[:200]},
                    )

                    runner._run_hooks(
                        "post_run",
                        goal=sub_goal,
                        agent_name=agent_name,
                        result=result,
                        success=result.success,
                        elapsed=_elapsed,
                    )
                    return result
                except (AgentExecutionError, LLMError) as e:
                    logger.warning(
                        f"Fast-path failed for {agent_name} (recoverable): {e}, falling back to full pipeline"
                    )
                except Exception as e:
                    logger.warning(
                        f"Fast-path failed for {agent_name} (unexpected): {e}, falling back to full pipeline"
                    )

        # FULL PATH: Use the complete AgentRunner pipeline
        return await runner.run(goal=sub_goal, **kwargs)

    async def relay(self, goal: str, **kwargs: Any) -> EpisodeResult:
        """
        Relay paradigm: agents execute sequentially, each building on prior output.

        When agents expose AgentIOSchema, output fields are auto-wired to
        the next agent's input fields by name/type match.
        """
        sm = self._manager
        status_callback = kwargs.pop("status_callback", None)
        kwargs.setdefault("ensemble", False)

        all_results = {}
        enriched_goal = goal
        prev_schema = None
        prev_output = None

        for agent_config in sm.agents:
            runner = sm.runners.get(agent_config.name)
            if not runner:
                continue

            caps = getattr(agent_config, "capabilities", None)
            sub_goal = caps[0] if caps else enriched_goal

            # Schema-aware wiring
            if prev_schema is not None and prev_output is not None:
                try:
                    cur_agent = getattr(runner, "agent", None)
                    if cur_agent and hasattr(cur_agent, "get_io_schema"):
                        cur_schema = cur_agent.get_io_schema()
                        wired_kwargs = prev_schema.map_outputs(prev_output, cur_schema)
                        if wired_kwargs:
                            kwargs.update(wired_kwargs)
                            logger.info(
                                f"Relay schema wiring: {prev_schema.agent_name} -> "
                                f"{cur_schema.agent_name}: {list(wired_kwargs.keys())}"
                            )
                except Exception as e:
                    logger.debug(f"Relay schema wiring skipped: {e}")

            safe_status(status_callback, f"Relay -> {agent_config.name}", sub_goal[:60])

            async with sm.agent_semaphore:
                result = await self.run_agent(runner, sub_goal, agent_config.name, **kwargs)

            all_results[agent_config.name] = result

            if result.success and result.output:
                enriched_goal = (
                    f"{goal}\n\n"
                    f"[Previous agent '{agent_config.name}' output]:\n"
                    f"{_extract_output_text(result.output)[:2000]}"
                )
                try:
                    cur_agent = getattr(runner, "agent", None)
                    if cur_agent and hasattr(cur_agent, "get_io_schema"):
                        prev_schema = cur_agent.get_io_schema()
                        out = result.output
                        prev_output = out if isinstance(out, dict) else {"output": str(out)}
                    else:
                        prev_schema = None
                        prev_output = None
                except Exception:
                    prev_schema = None
                    prev_output = None
            else:
                logger.warning(f"Relay: {agent_config.name} failed, continuing with original goal")
                prev_schema = None
                prev_output = None

        combined = self.aggregate_results(all_results, goal)
        sm._schedule_background_learning(combined, goal)
        return combined

    async def debate(self, goal: str, **kwargs: Any) -> EpisodeResult:
        """
        Debate paradigm: agents produce drafts, then critique each other in rounds.
        """
        sm = self._manager
        status_callback = kwargs.pop("status_callback", None)
        max_debate_rounds = kwargs.pop("debate_rounds", 2)
        kwargs.setdefault("ensemble", False)

        all_results: Dict[str, EpisodeResult] = {}

        # Round 1: All agents produce initial drafts
        safe_status(status_callback, "Debate round 1", "all agents drafting")

        draft_tasks = []
        for agent_config in sm.agents:
            runner = sm.runners.get(agent_config.name)
            if not runner:
                continue
            sub_goal = (
                getattr(agent_config, "capabilities", None) and agent_config.capabilities[0] or goal
            )
            draft_tasks.append((agent_config.name, runner, sub_goal))

        async def _run_draft(name: Any, runner: Any, sub_goal: Any) -> Any:
            async with sm.agent_semaphore:
                return name, await self.run_agent(runner, sub_goal, name, **kwargs)

        draft_results = await asyncio.gather(
            *[_run_draft(n, r, g) for n, r, g in draft_tasks],
            return_exceptions=True,
        )

        drafts = {}
        for item in draft_results:
            if not isinstance(item, tuple):
                logger.warning(f"Debate draft failed: {item}")
                continue
            name, result = item
            all_results[name] = result
            if result.success and result.output:
                drafts[name] = _extract_output_text(result.output)[:1500]

        if len(drafts) < 2:
            combined = self.aggregate_results(all_results, goal)
            sm._schedule_background_learning(combined, goal)
            return combined

        # Rounds 2+: Critique
        for round_num in range(2, max_debate_rounds + 1):
            safe_status(
                status_callback, f"Debate round {round_num}", "agents critiquing & refining"
            )

            critique_tasks = []
            for agent_config in sm.agents:
                runner = sm.runners.get(agent_config.name)
                if not runner or agent_config.name not in drafts:
                    continue

                others = "\n\n".join(
                    f"[{name}'s draft]: {text}"
                    for name, text in drafts.items()
                    if name != agent_config.name
                )
                sub_goal = (
                    getattr(agent_config, "capabilities", None)
                    and agent_config.capabilities[0]
                    or goal
                )
                critique_goal = (
                    f"{sub_goal}\n\n"
                    f"Other agents produced these solutions. "
                    f"Critique them and improve your answer:\n{others}"
                )
                critique_tasks.append((agent_config.name, runner, critique_goal))

            critique_results = await asyncio.gather(
                *[_run_draft(n, r, g) for n, r, g in critique_tasks],
                return_exceptions=True,
            )

            for item in critique_results:
                if not isinstance(item, tuple):
                    logger.warning(f"Debate critique failed: {item}")
                    continue
                name, result = item
                all_results[name] = result
                if result.success and result.output:
                    drafts[name] = _extract_output_text(result.output)[:1500]

        combined = self.aggregate_results(all_results, goal)
        sm._schedule_background_learning(combined, goal)
        return combined

    async def refinement(self, goal: str, **kwargs: Any) -> EpisodeResult:
        """
        Collective refinement paradigm: iterative improvement until quality stabilizes.
        """
        sm = self._manager
        status_callback = kwargs.pop("status_callback", None)
        max_iterations = kwargs.pop("refinement_iterations", 3)
        kwargs.setdefault("ensemble", False)

        first_agent = sm.agents[0]
        runner = sm.runners.get(first_agent.name)
        sub_goal = (
            getattr(first_agent, "capabilities", None) and first_agent.capabilities[0] or goal
        )

        safe_status(status_callback, "Refinement", f"initial draft by {first_agent.name}")

        async with sm.agent_semaphore:
            result = await self.run_agent(runner, sub_goal, first_agent.name, **kwargs)

        current_draft = _extract_output_text(result.output)[:3000]
        all_results = {first_agent.name: result}
        prev_draft = ""

        for iteration in range(1, max_iterations + 1):
            if current_draft and prev_draft and current_draft[:200] == prev_draft[:200]:
                logger.info(f"Refinement: converged at iteration {iteration}")
                break

            try:
                if sm.learning.adaptive_learning.should_stop_early(min_iterations=2):
                    logger.info(
                        f"Refinement: adaptive learning recommends early stop "
                        f"at iteration {iteration}"
                    )
                    break
            except Exception as e:
                logger.debug(f"Adaptive early-stop check failed: {e}")

            prev_draft = current_draft

            for agent_config in sm.agents[1:]:
                refiner = sm.runners.get(agent_config.name)
                if not refiner:
                    continue

                refine_sub = (
                    getattr(agent_config, "capabilities", None)
                    and agent_config.capabilities[0]
                    or goal
                )
                refine_goal = (
                    f"{refine_sub}\n\n" f"Here is the current draft. Improve it:\n{current_draft}"
                )

                safe_status(
                    status_callback,
                    f"Refinement iter {iteration}",
                    f"{agent_config.name} improving",
                )

                async with sm.agent_semaphore:
                    ref_result = await self.run_agent(
                        refiner, refine_goal, agent_config.name, **kwargs
                    )

                all_results[agent_config.name] = ref_result
                if ref_result.success and ref_result.output:
                    current_draft = _extract_output_text(ref_result.output)[:3000]

        combined = self.aggregate_results(all_results, goal)
        sm._schedule_background_learning(combined, goal)
        return combined

    def aggregate_results(self, results: Dict[str, EpisodeResult], goal: str) -> EpisodeResult:
        """
        Combine all agent outputs into a single EpisodeResult.

        Uses CandidateVerifier for intelligent selection when multiple agents produce output.
        """
        sm = self._manager
        if not results:
            return EpisodeResult(
                output=None,
                success=False,
                trajectory=[],
                tagged_outputs=[],
                episode=getattr(sm, "episode_count", 0),
                execution_time=0.0,
                architect_results=[],
                auditor_results=[],
                agent_contributions={},
                alerts=["No tasks executed"],
            )

        if len(results) == 1:
            return list(results.values())[0]

        verified_output = sm._mas_zero_verify(goal, results)

        if verified_output is not None:
            combined_output = verified_output
        else:
            combined_output = {name: r.output for name, r in results.items()}

        all_success = all(r.success for r in results.values())

        merged_trajectory = []
        for name, r in results.items():
            for step in r.trajectory or []:
                step_copy = dict(step)
                step_copy["agent"] = name
                merged_trajectory.append(step_copy)

        merged_contributions = {}
        for r in results.values():
            if hasattr(r, "agent_contributions") and r.agent_contributions:
                merged_contributions.update(r.agent_contributions)

        return EpisodeResult(
            output=combined_output,
            success=all_success,
            trajectory=merged_trajectory,
            tagged_outputs=[],
            episode=getattr(sm, "episode_count", 0),
            execution_time=sum(getattr(r, "execution_time", 0) for r in results.values()),
            architect_results=[],
            auditor_results=[],
            agent_contributions=merged_contributions,
        )

    def assign_cooperative_credit(self, results: Dict[str, EpisodeResult], goal: str) -> None:
        """
        Compute cooperative reward decomposition across agents.

        Uses adaptive learned weights instead of hardcoded values.
        """
        sm = self._manager
        if not results or len(results) < 2:
            return

        episode_success = all(r.success for r in results.values())

        for agent_name, result in results.items():
            base_reward = 1.0 if result.success else 0.0

            other_successes = sum(1 for n, r in results.items() if n != agent_name and r.success)
            total_others = len(results) - 1
            cooperation_bonus = other_successes / total_others if total_others > 0 else 0.0

            predictability_bonus = 0.5

            cooperative_reward = (
                sm.credit_weights.get("base_reward") * base_reward
                + sm.credit_weights.get("cooperation_bonus") * cooperation_bonus
                + sm.credit_weights.get("predictability_bonus") * predictability_bonus
            )

            try:
                state = {"query": goal, "agent": agent_name, "cooperative": True}
                action = {"actor": agent_name, "task": goal[:100]}
                sm.learning_manager.record_outcome(state, action, cooperative_reward, done=True)
            except Exception as e:
                logger.debug(f"Cooperative credit recording skipped for {agent_name}: {e}")

        if episode_success:
            sm.credit_weights.update_from_feedback("cooperation_bonus", 0.1, reward=1.0)
        else:
            sm.credit_weights.update_from_feedback("base_reward", 0.05, reward=0.0)

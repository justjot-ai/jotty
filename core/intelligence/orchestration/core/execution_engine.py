"""
Execution Engine
================

Executes tasks via single/multi-agent paradigms for the Orchestrator.

Responsibilities:
    - run(): Main orchestration method (fast path, model tier routing, etc.)
    - _execute_single_agent(): MAS-ZERO single agent execution
    - _execute_multi_agent(): Multi-agent orchestration with paradigms
    - Paradigm delegation (relay, debate, refinement)
    - autonomous_setup(): Research, install, configure

Extracted from swarm_manager.py for maintainability.
"""

# mypy: disable-error-code="has-type"
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from .swarm_manager import Orchestrator

from Jotty.core.infrastructure.foundation.data_structures import EpisodeResult
from Jotty.core.infrastructure.foundation.exceptions import (  # type: ignore[import-not-found]
    AgentExecutionError,
    ConfigurationError,
    LearningError,
    LLMError,
)
from Jotty.core.infrastructure.utils.async_utils import (  # type: ignore[import-not-found, import]
    StatusReporter,
    safe_status,
)

from ..coordination.paradigm_executor import ParadigmExecutor
from ..execution.agent_runner import AgentRunner, AgentRunnerConfig
from ..routing.model_tier_router import ModelTierRouter

# Optional feedback channel imports
try:
    from Jotty.core.intelligence.reasoning.tools.feedback_channel import (  # type: ignore[import-not-found]
        FeedbackMessage,
        FeedbackType,
    )
except ImportError:
    FeedbackMessage = None  # type: ignore
    FeedbackType = None  # type: ignore

# Optional observability imports
try:
    from Jotty.core.infrastructure.monitoring.observability import (  # type: ignore[import-not-found, import]
        get_metrics,
        get_tracer,
    )
except ImportError:
    get_metrics = None  # type: ignore
    get_tracer = None  # type: ignore

# Optional dspy import
try:
    import dspy
except ImportError:
    dspy = None  # type: ignore

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Executes tasks via single/multi-agent paradigms."""

    def __init__(self, manager: "Orchestrator") -> None:
        self._manager = manager
        self._paradigms = ParadigmExecutor(manager)

    async def run(self, goal: str, **kwargs: Any) -> EpisodeResult:
        """
        Run task execution with full autonomy.

        Supports zero-config: natural language goal -> autonomous execution.
        For simple tool-calling tasks, use ChatExecutor directly instead.

        Args:
            goal: Task goal/description (natural language supported)
            context: Optional ExecutionContext from ModeRouter
            skip_autonomous_setup: If True, skip research/install/configure (fast mode)
            status_callback: Optional callback(stage, detail) for progress updates
            ensemble: Enable prompt ensembling for multi-perspective analysis
            ensemble_strategy: Strategy for ensembling
            **kwargs: Additional arguments

        Returns:
            EpisodeResult with output and metadata
        """
        sm = self._manager
        import time as _time

        run_start_time = _time.time()

        # Observability: Start trace and root span
        _tracer = None
        if get_tracer:
            _tracer = get_tracer()
            _tracer.new_trace(metadata={"goal": goal[:200], "mode": sm.mode})

        # Lazy init: Build runners on first run
        sm._ensure_runners()

        # Wait for background learning init (optional/short for latency-sensitive deployments)
        # learning_wait_timeout_seconds <= 0 skips wait; default 5.0s.
        learning_wait_timeout = getattr(sm.config, "learning_wait_timeout_seconds", 5.0)
        if learning_wait_timeout > 0:
            try:
                await asyncio.wait_for(
                    sm._learning_ready.wait(), timeout=float(learning_wait_timeout)
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Learning init timed out after %.1fs, proceeding without full learning state",
                    learning_wait_timeout,
                )
        else:
            logger.debug("Learning wait skipped (learning_wait_timeout_seconds <= 0)")

        # MAS-ZERO: Reset per-problem experience library
        sm._reset_experience()
        sm._efficiency_stats = {}  # Reset per-run
        sm._execution._efficiency_stats = {}  # Reset orchestrator stats too

        # Extract ExecutionContext if provided by ModeRouter
        exec_context = kwargs.pop("context", None)

        # Extract special kwargs (pop ALL so they don't leak into **kwargs downstream)
        skip_autonomous_setup = kwargs.pop("skip_autonomous_setup", False)
        skip_validation = kwargs.pop("skip_validation", None)  # Explicit override
        status_callback = kwargs.pop("status_callback", None)
        ensemble = kwargs.pop("ensemble", None)  # None = auto-detect, True/False = explicit
        ensemble_strategy = kwargs.pop("ensemble_strategy", "multi_perspective")

        # If ExecutionContext provided, extract callbacks from it
        if exec_context is not None:
            if status_callback is None and hasattr(exec_context, "status_callback"):
                status_callback = exec_context.status_callback
            # Emit start event
            if hasattr(exec_context, "emit_event"):
                from Jotty.core.infrastructure.foundation.types.sdk_types import (
                    SDKEventType,  # type: ignore[import-not-found, import]
                )

                exec_context.emit_event(
                    SDKEventType.PLANNING,
                    {
                        "goal": goal,
                        "mode": sm.mode,
                        "agents": len(sm.agents),
                    },
                )

        # Auto-detect ensemble for certain task types (if not explicitly set)
        # Optima-inspired: adaptive sizing returns (should_ensemble, max_perspectives)
        max_perspectives = 4  # default
        if ensemble is None:
            ensemble, max_perspectives = sm._should_auto_ensemble(goal)
            if ensemble:
                logger.info(
                    f"Auto-enabled ensemble: {max_perspectives} perspectives (use ensemble=False to override)"
                )

        # Ensure DSPy LM is configured (critical for all agent operations)
        if dspy and (not hasattr(dspy.settings, "lm") or dspy.settings.lm is None):
            lm = sm.swarm_provider_gateway.get_lm()
            if lm:
                dspy.configure(lm=lm)
                logger.info(f" DSPy LM configured: {getattr(lm, 'model', 'unknown')}")

        _status = StatusReporter(status_callback, logger, emoji="")

        # ── FAST PATH: Simple tasks bypass the entire agent pipeline ──
        # ValidationGate classifies task complexity using a cheap LLM (or heuristic).
        # For DIRECT tasks (Q&A, lookups, lists): skip zero-config, ensemble, and
        # AutonomousAgent overhead. Call the LLM directly. Target: <10s.
        # OPTIMIZATION: Skip gate for multi-agent mode — fast path only works
        # for single-agent, so running the gate wastes an LLM call.
        from Jotty.core.intelligence.orchestration.execution.validation_gate import (
            ValidationMode,
            get_validation_gate,
        )

        _gate_decision = None
        if sm.mode == "single":
            _fast_gate = get_validation_gate()

            # Determine if user explicitly asked for skip/full validation
            _force_mode = None
            if skip_validation is True:
                _force_mode = ValidationMode.DIRECT
            elif skip_validation is False:
                _force_mode = ValidationMode.FULL

            _gate_decision = await _fast_gate.decide(
                goal=goal,
                agent_name=sm.agents[0].name if sm.agents else "auto",  # type: ignore[union-attr]
                force_mode=_force_mode,
            )

        if _gate_decision and _gate_decision.mode == ValidationMode.DIRECT and sm.mode == "single":
            _status(
                "Fast path", f"DIRECT mode — bypassing agent pipeline ({_gate_decision.reason})"
            )

            # Model tier routing: use cheapest LM for DIRECT tasks
            try:
                if sm._model_tier_router is None:
                    sm._model_tier_router = ModelTierRouter()
                tier_decision = sm._model_tier_router.get_model_for_mode(ValidationMode.DIRECT)
                tier_lm = sm._model_tier_router.get_lm_for_mode(ValidationMode.DIRECT)  # type: ignore[func-returns-value]
                if tier_lm:
                    lm = tier_lm
                    _status(
                        "Model tier",
                        f"{tier_decision.tier.value} ({tier_decision.model}) — cost ratio {tier_decision.estimated_cost_ratio:.1f}x",
                    )
                else:
                    lm = dspy.settings.lm if dspy else None
                    if lm is None:
                        raise AgentExecutionError("No LM configured")

                _fast_start = _time.time()

                # Try multiple calling conventions (DSPy 3.x is finicky)
                # Fast path: NO aggressive retry on rate limits.
                # If rate limited, fall through to full pipeline immediately.
                # Fast path must be FAST — wasting 56s on retries defeats the purpose.
                response = None
                _rate_limited = False
                for call_fn in [
                    lambda: lm(messages=[{"role": "user", "content": goal}]),
                    lambda: lm(prompt=goal),
                    lambda: lm(goal),
                ]:
                    try:
                        response = call_fn()
                        if response:
                            break
                    except LLMError as e:
                        logger.info(f"Fast path LLM error (recoverable): {e}")
                        err_str = str(e)
                        is_rate_limit = (
                            "429" in err_str
                            or "RateLimit" in err_str
                            or "rate limit" in err_str.lower()
                        )
                        if is_rate_limit:
                            logger.info(
                                "Fast path rate limited — falling through to full pipeline (no retry)"
                            )
                            _rate_limited = True
                            break
                        continue
                    except Exception as e:
                        err_str = str(e)
                        is_rate_limit = (
                            "429" in err_str
                            or "RateLimit" in err_str
                            or "rate limit" in err_str.lower()
                        )
                        if is_rate_limit:
                            logger.info(
                                "Fast path rate limited — falling through to full pipeline (no retry)"
                            )
                            _rate_limited = True
                            break
                        continue

                if response is None:
                    raise AgentExecutionError(
                        "All LM calling conventions failed"
                        + (" (rate limited)" if _rate_limited else "")
                    )

                if isinstance(response, list):
                    response = response[0] if response else ""
                elif hasattr(response, "text"):
                    response = response.text
                response = str(response).strip()

                _fast_elapsed = _time.time() - _fast_start
                _status("Fast path complete", f"{_fast_elapsed:.1f}s")

                # Record outcome for gate learning
                _fast_gate.record_outcome(ValidationMode.DIRECT, bool(response))

                # Build minimal EpisodeResult
                fast_result = EpisodeResult(
                    output=response,
                    success=bool(response),
                    trajectory=[{"step": 1, "action": "direct_llm", "output": response[:200]}],
                    tagged_outputs=[],
                    episode=sm.episode_count,
                    execution_time=_fast_elapsed,
                    architect_results=[],
                    auditor_results=[],
                    agent_contributions={},
                )
                sm.episode_count += 1

                # Save learning (lightweight)
                try:
                    sm._save_learnings()
                except LearningError as e:
                    logger.debug(f"Fast-path learning save failed (learning): {e}")
                except Exception as e:
                    logger.debug(f"Fast-path learning save failed (unexpected): {e}", exc_info=True)

                total_elapsed = _time.time() - run_start_time
                _status("Complete", f"fast path success ({total_elapsed:.1f}s)")
                return fast_result

            except (AgentExecutionError, LLMError) as e:
                logger.info(
                    f"Fast path failed (recoverable: {type(e).__name__}): {e}, falling back to full pipeline"
                )
                # Fall through to normal pipeline
            except Exception as e:
                logger.info(
                    f"Fast path failed (unexpected: {type(e).__name__}): {e}, falling back to full pipeline"
                )
                # Fall through to normal pipeline

        # Store gate decision for downstream use — pass to AgentRunner to avoid redundant decide()
        kwargs["gate_decision"] = _gate_decision

        # ── MODEL TIER ROUTING: Select LM quality based on task complexity ──
        # DIRECT → cheap (Haiku), AUDIT_ONLY → balanced (Sonnet), FULL → quality (Opus/Sonnet)
        if _gate_decision and _gate_decision.mode != ValidationMode.DIRECT:
            try:
                if sm._model_tier_router is None:
                    sm._model_tier_router = ModelTierRouter()
                tier_decision = sm._model_tier_router.get_model_for_mode(_gate_decision.mode)
                tier_lm = sm._model_tier_router.get_lm_for_mode(_gate_decision.mode)  # type: ignore[func-returns-value]
                if tier_lm and dspy:
                    dspy.configure(lm=tier_lm)
                    _status("Model tier", f"{tier_decision.tier.value} ({tier_decision.model})")
            except (ConfigurationError, LLMError) as e:
                logger.debug(f"Model tier routing skipped (recoverable): {e}")
            except Exception as e:
                logger.debug(f"Model tier routing skipped (unexpected): {e}", exc_info=True)

        # Zero-config: LLM decides single vs multi-agent at RUN TIME (when goal is available)
        # SKIP when gate already classified as DIRECT — it's a simple task, no need
        # to burn an LLM call deciding single vs multi-agent.
        _gate_is_direct = _gate_decision and _gate_decision.mode == ValidationMode.DIRECT
        if sm.enable_zero_config and sm.mode == "single" and not _gate_is_direct:
            _status("Analyzing task", "deciding single vs multi-agent")
            new_agents = sm._create_zero_config_agents(goal, status_callback)
            if len(new_agents) > 1:
                # LLM detected parallel sub-goals - upgrade to multi-agent
                sm.agents = new_agents
                sm.mode = "multi"
                logger.info(
                    f" Zero-config: Upgraded to {len(sm.agents)} agents for parallel execution"
                )

                # Create runners for new agents
                for agent_config in sm.agents:
                    if agent_config.name not in sm.runners:
                        runner_config = AgentRunnerConfig(
                            architect_prompts=sm.architect_prompts,
                            auditor_prompts=sm.auditor_prompts,
                            config=sm.config,
                            agent_name=agent_config.name,
                            enable_learning=True,
                            enable_memory=True,
                        )
                        runner = AgentRunner(
                            agent=agent_config.agent,
                            config=runner_config,
                            task_planner=sm.swarm_planner,
                            task_board=sm.swarm_task_board,
                            swarm_memory=sm.swarm_memory,
                            swarm_state_manager=sm.swarm_state_manager,
                            learning_manager=sm.learning,
                            transfer_learning=sm.transfer_learning,
                            swarm_terminal=sm.swarm_terminal,  # Shared intelligent terminal
                        )
                        sm.runners[agent_config.name] = runner

        agent_info = (
            f"{len(sm.agents)} AutoAgent(s)" if len(sm.agents) > 1 else "AutoAgent (zero-config)"
        )
        _status("Starting", agent_info)

        # Profile execution if enabled
        if sm.swarm_profiler:
            profile_context = sm.swarm_profiler.profile(
                "Orchestrator.run", metadata={"goal": goal, "mode": sm.mode}
            )
            profile_context.__enter__()
        else:
            profile_context = None

        try:
            # Store goal in shared context for state management
            if sm.shared_context:
                sm.shared_context.set("goal", goal)
                sm.shared_context.set("query", goal)

            # Autonomous planning: Research, install, configure if needed
            # For multi-agent mode with zero-config: skip full setup (agents have specific sub-goals)
            # This reduces latency significantly for parallel agent execution
            if not skip_autonomous_setup and sm.mode == "single":
                _status("Autonomous setup", "analyzing requirements")
                await self.autonomous_setup(goal, status_callback=status_callback)
            elif not skip_autonomous_setup and sm.mode == "multi":
                _status("Fast mode", "multi-agent (agents configured with sub-goals)")
            else:
                _status("Fast mode", "skipping autonomous setup")

            # Set root task in SwarmTaskBoard
            sm.swarm_task_board.root_task = goal

            # Record swarm-level step: goal received
            if sm.swarm_state_manager:
                sm.swarm_state_manager.record_swarm_step(
                    {
                        "step": "goal_received",
                        "goal": goal,
                        "mode": sm.mode,
                        "agent_count": len(sm.agents),
                    }
                )

            # Ensemble mode: Multi-perspective analysis
            # For multi-agent: ensemble happens per-agent (each agent has different sub-goal)
            # For single-agent: ensemble happens at swarm level
            ensemble_result = None
            if ensemble and sm.mode == "single":
                _status(
                    "Ensembling", f"strategy={ensemble_strategy}, perspectives={max_perspectives}"
                )
                _ens_start = _time.time()
                ensemble_result = await sm._execute_ensemble(
                    goal,
                    strategy=ensemble_strategy,
                    status_callback=status_callback,
                    max_perspectives=max_perspectives,
                )
                sm._efficiency_stats["ensemble_time"] = _time.time() - _ens_start
                if ensemble_result.get("success"):
                    # Pass ensemble context via kwargs (not embedded in goal string)
                    # to avoid polluting search queries and downstream skill params.
                    kwargs["ensemble_context"] = ensemble_result

                    # Show quality scores if available
                    quality_scores = ensemble_result.get("quality_scores", {})
                    if quality_scores:
                        avg_quality = sum(quality_scores.values()) / len(quality_scores)
                        scores_str = ", ".join(f"{k}:{v:.0%}" for k, v in quality_scores.items())
                        _status("Ensemble quality", f"avg={avg_quality:.0%} ({scores_str})")
                    else:
                        _status(
                            "Ensemble complete",
                            f"{len(ensemble_result.get('perspectives_used', []))} perspectives",
                        )
            elif ensemble and sm.mode == "multi":
                # Multi-agent mode: SKIP per-agent ensemble (too expensive)
                # With N agents x 4 perspectives = 4N LLM calls - massive overkill
                # Each agent already has a specific sub-goal, no need for multi-perspective
                _status(
                    "Ensemble mode", "DISABLED for multi-agent (agents have specific sub-goals)"
                )
                ensemble = False  # Disable for agents

            # Keep gate_decision in kwargs so AgentRunner can reuse it (no second decide() call)

            # Single-agent mode: Simple execution
            if sm.mode == "single":
                agent_name = sm.agents[0].name if sm.agents else "auto"  # type: ignore[union-attr]
                _status("Executing", f"agent '{agent_name}' with skill orchestration")
                # Architect -> Actor -> Auditor pipeline (now fast with max_eval_iters=2)
                # skip_validation: explicit kwarg wins, else derive from skip_autonomous_setup
                skip_val = skip_validation if skip_validation is not None else skip_autonomous_setup
                # Forward ensemble flag so AutoAgent doesn't auto-detect independently
                if ensemble is not None:
                    kwargs["ensemble"] = ensemble
                # Avoid duplicate kwarg: ensemble_context may already be in kwargs (line 748)
                kwargs.pop("ensemble_context", None)
                result = await self._execute_single_agent(
                    goal,
                    skip_validation=skip_val,
                    status_callback=status_callback,
                    ensemble_context=ensemble_result if ensemble else None,
                    **kwargs,
                )

                # Optima-inspired efficiency summary
                total_elapsed = _time.time() - run_start_time
                ens_t = sm._efficiency_stats.get("ensemble_time", 0)
                exec_t = getattr(result, "execution_time", total_elapsed - ens_t)
                overhead = max(0, total_elapsed - exec_t)
                overhead_pct = (overhead / total_elapsed * 100) if total_elapsed > 0 else 0
                sm._efficiency_stats.update(
                    {"total_time": total_elapsed, "overhead_pct": overhead_pct}
                )
                _status(
                    "Complete",
                    f"{'success' if result.success else 'failed'} ({total_elapsed:.1f}s, overhead={overhead_pct:.0f}%)",
                )

                # Surface user-friendly summary with artifacts
                sm._log_execution_summary(result)

                # Exit profiling context
                if profile_context:
                    profile_context.__exit__(None, None, None)

                # Auto-drain one training task if result succeeded and tasks pending
                sm._maybe_drain_training_task(result)

                return result

            # Multi-agent mode: Use SwarmTaskBoard for coordination
            else:
                # MAS-ZERO: Dynamic reduction — check if multi-agent is overkill
                if sm._mas_zero_should_reduce(goal):
                    _status("MAS-ZERO reduction", "reverting to single agent (simpler is better)")
                    sm.mode = "single"
                    result = await self._execute_single_agent(
                        goal,
                        skip_validation=(
                            skip_validation
                            if skip_validation is not None
                            else skip_autonomous_setup
                        ),
                        status_callback=status_callback,
                        **kwargs,
                    )
                else:
                    paradigm = kwargs.pop("discussion_paradigm", "auto")
                    _status("Executing", f"{len(sm.agents)} agents — paradigm: {paradigm}")
                    _sv = skip_validation if skip_validation is not None else skip_autonomous_setup
                    result = await self._execute_multi_agent(
                        goal,
                        ensemble_context=ensemble_result if ensemble else None,
                        status_callback=status_callback,
                        ensemble=ensemble,
                        ensemble_strategy=ensemble_strategy,
                        discussion_paradigm=paradigm,
                        skip_validation=_sv,
                        **kwargs,
                    )

                # Optima-inspired efficiency summary
                total_elapsed = _time.time() - run_start_time
                ens_t = sm._efficiency_stats.get("ensemble_time", 0)
                exec_t = getattr(result, "execution_time", total_elapsed - ens_t)
                overhead = max(0, total_elapsed - exec_t)
                overhead_pct = (overhead / total_elapsed * 100) if total_elapsed > 0 else 0
                sm._efficiency_stats.update(
                    {"total_time": total_elapsed, "overhead_pct": overhead_pct}
                )
                _status(
                    "Complete",
                    f"{'success' if result.success else 'failed'} ({total_elapsed:.1f}s, overhead={overhead_pct:.0f}%)",
                )

                # Surface user-friendly summary with artifacts
                sm._log_execution_summary(result)

                # Exit profiling context
                if profile_context:
                    profile_context.__exit__(None, None, None)

                return result
        except (AgentExecutionError, LLMError) as e:
            _status("Error", f"{type(e).__name__}: {str(e)[:50]}")
            # Exit profiling context on error
            if profile_context:
                profile_context.__exit__(type(e), e, None)
            raise
        except Exception as e:
            _status("Error", f"unexpected: {str(e)[:50]}")
            logger.error(f"Unexpected error in run(): {e}", exc_info=True)
            # Exit profiling context on error
            if profile_context:
                profile_context.__exit__(type(e), e, None)
            raise

    async def _execute_single_agent(self, goal: str, **kwargs: Any) -> EpisodeResult:
        """
        Execute single-agent mode.

        MAS-ZERO: For comparison/analysis tasks, runs building blocks
        (direct + ensemble) in parallel and verifies the best answer.
        For simple tasks, runs direct execution as before.

        Args:
            goal: Task goal
            **kwargs: Additional arguments

        Returns:
            EpisodeResult
        """
        sm = self._manager
        # Remove ensemble_context from kwargs before passing to runner
        kwargs.pop("ensemble_context", None)
        status_callback = kwargs.pop("status_callback", None)

        _status = StatusReporter(status_callback)

        # ── Pre-execution strategy: leverage learned intelligence ──
        # Use stigmergy + MAS learning to inform agent selection
        try:
            if hasattr(sm, "learning") and sm.learning:
                components = sm.learning.learning_components
                stig = components.get("stigmergy")
                mas = components.get("mas_learning")

                # Stigmergy: recommend approach from historical signals
                if stig:
                    approach = stig.recommend_approach(goal[:100])
                    if approach:
                        kwargs.setdefault("learning_context", "")
                        kwargs[
                            "learning_context"
                        ] += f"\n[Stigmergy] Recommended approach: {approach}"

                # MAS Learning: get execution strategy from history
                if mas and hasattr(mas, "get_execution_strategy"):
                    strategy = mas.get_execution_strategy(
                        goal, [a.name if hasattr(a, "name") else str(a) for a in sm.agents]
                    )
                    if (
                        strategy
                        and hasattr(strategy, "recommended_agents")
                        and strategy.recommended_agents
                    ):
                        logger.info(f"MAS strategy recommends: {strategy.recommended_agents[:3]}")
        except Exception as e:
            logger.debug(f"Pre-execution strategy skipped: {e}")

        # ── Router-guided agent selection (closes RL loop for single-agent) ──
        # When multiple agents are available, ask the router instead of
        # blindly taking agents[0].  Router uses SwarmIntelligence (RL +
        # stigmergy + TRAS) when learning data exists.
        agent_config = sm.agents[0]
        if len(sm.agents) > 1:
            try:
                route = sm._router.select_agent(goal)
                picked = route.get("agent")
                if picked and route.get("method") != "fallback":
                    for ac in sm.agents:
                        if getattr(ac, "name", None) == picked:
                            agent_config = ac
                            logger.info(
                                f"Router selected '{picked}' for single-agent "
                                f"(method={route['method']}, conf={route['confidence']:.2f})"
                            )
                            break
            except (ConfigurationError, LearningError) as e:
                logger.debug(f"Router select_agent skipped (recoverable): {e}")
            except Exception as e:
                logger.debug(f"Router select_agent skipped (unexpected): {e}", exc_info=True)

        runner = sm.runners[agent_config.name]  # type: ignore[union-attr]

        # ── READ LEARNED INTELLIGENCE (closes the learning loop) ──────────
        # Post-episode writes: stigmergy outcomes, byzantine trust, morph scores.
        # Pre-execution reads them back to influence this run.
        # Without this read, the write side is wasted — learning never loops.
        learned_hints = []
        try:
            lp = sm.learning
            task_type = lp.transfer_learning.extractor.extract_task_type(goal)

            # 1. Stigmergy APPROACH guidance (single-agent-aware).
            #    Instead of "use agent X" (useless with one agent),
            #    inject "use these tools/approaches" and "avoid these".
            approach_rec = lp.stigmergy.recommend_approach(task_type, top_k=2)
            for good in approach_rec.get("use", []):
                tools_str = ", ".join(good["tools"][:5]) if good.get("tools") else "N/A"
                learned_hints.append(
                    f"[Learned] Successful approach for '{task_type}': "
                    f"{good['approach'][:120]} (tools: {tools_str}, "
                    f"quality: {good.get('quality', 0):.0%})"
                )
            for bad in approach_rec.get("avoid", []):
                tools_str = ", ".join(bad["tools"][:5]) if bad.get("tools") else "N/A"
                learned_hints.append(
                    f"[Learned] Failed approach for '{task_type}': "
                    f"{bad['approach'][:120]} — avoid this"
                )
            # Fallback: if no approach data yet, use old agent-routing signals
            if not approach_rec.get("use") and not approach_rec.get("avoid"):
                warnings = lp.get_stigmergy_warnings(task_type)
                if warnings:
                    warn_msgs = [
                        (
                            getattr(w, "content", {}).get("goal", "unknown")
                            if isinstance(getattr(w, "content", None), dict)
                            else str(w)
                        )
                        for w in warnings[:3]
                    ]
                    learned_hints.append(
                        f"[Learned] Previous failures on '{task_type}': " f"{'; '.join(warn_msgs)}"
                    )

            # 2. Effectiveness: are we improving or degrading on this type?
            #    Reads from BOTH in-memory EffectivenessTracker (fast, current session)
            #    AND LearningService (SQLite, cross-session history).
            eff_report = lp.effectiveness.improvement_report()
            task_eff = eff_report.get(task_type)
            if task_eff and task_eff.get("trend") is not None:
                trend = task_eff["trend"]
                if trend < -0.1:
                    _recent = task_eff.get("recent_success_rate", 0)
                    _hist = task_eff.get("historical_success_rate", 0)
                    learned_hints.append(
                        f"[Learned] Performance DECLINING on '{task_type}' "
                        f"(recent={_recent:.0%} vs "
                        f"historical={_hist:.0%}). "
                        f"Consider a different approach."
                    )
                elif trend > 0.1:
                    learned_hints.append(
                        f"[Learned] Performance IMPROVING on '{task_type}' — "
                        f"current approach is working."
                    )

            # Cross-session effectiveness from LearningService (SQLite)
            try:
                from Jotty.core.intelligence.learning.learning_service import LearningService

                ls = LearningService.get_instance()
                ls_report = ls.improvement_report(domain=task_type)
                if ls_report and ls_report.get("recent_success_rate") is not None:
                    ls_trend = ls_report.get("trend", 0)
                    if ls_trend < -0.15 and not task_eff:
                        learned_hints.append(
                            f"[Learned/History] Cross-session performance declining "
                            f"on '{task_type}'. Try alternative approaches."
                        )
            except Exception:
                pass

            # 3. Transfer learning: relevant past learnings
            learnings = lp.transfer_learning.get_relevant_learnings(goal, top_k=2)
            for exp in learnings.get("similar_experiences", []):
                query_text = exp.get("query", "") if isinstance(exp, dict) else str(exp)
                success = exp.get("success", True) if isinstance(exp, dict) else True
                if success:
                    learned_hints.append(f"[Learned] Succeeded on similar task: {query_text[:100]}")
                else:
                    err = (exp.get("error", "")[:80]) if isinstance(exp, dict) else ""
                    learned_hints.append(
                        f"[Learned] Failed on similar task: {query_text[:80]}"
                        + (f" (error: {err})" if err else "")
                    )
            if learnings.get("task_pattern"):
                learned_hints.append(f"[Learned] Pattern: {str(learnings['task_pattern'])[:150]}")
            for err_pat in learnings.get("error_patterns", [])[:2]:
                learned_hints.append(f"[Learned] Common error: {str(err_pat)[:120]}")

            # Inject learned context into kwargs for the agent
            if learned_hints:
                existing_ctx = kwargs.get("learning_context", "")
                kwargs["learning_context"] = (
                    existing_ctx + "\n\n" + "\n".join(learned_hints)
                ).strip()
                _status("Intelligence", f"{len(learned_hints)} learned hints applied")
                logger.info(
                    f" Single-agent intelligence: {len(learned_hints)} hints "
                    f"for task_type='{task_type}'"
                )
                for i, hint in enumerate(learned_hints, 1):
                    logger.info(f" Hint {i}: {hint}")

        except LearningError as e:
            logger.warning(f"Pre-execution intelligence read failed (learning): {e}")
        except Exception as e:
            logger.warning(
                f"Pre-execution intelligence read failed (unexpected): {e}", exc_info=True
            )

        # Pass status_callback back for downstream
        if status_callback:
            kwargs["status_callback"] = status_callback

        # Standard single-agent execution
        import time as _t

        _exec_start = _t.time()

        # Observability: trace agent execution
        _tracer = get_tracer() if get_tracer else None
        _metrics = get_metrics() if get_metrics else None

        if _tracer:
            with _tracer.span("agent_execute", agent=agent_config.name, mode="single") as _span:  # type: ignore[union-attr]
                result = await runner.run(goal=goal, **kwargs)
                _exec_elapsed = _t.time() - _exec_start
                _span.set_attribute("success", result.success)
                _span.set_attribute("execution_time_s", round(_exec_elapsed, 2))
                if hasattr(result, "execution_time"):
                    _span.set_attribute("inner_time_s", round(result.execution_time, 2))
        else:
            result = await runner.run(goal=goal, **kwargs)
            _exec_elapsed = _t.time() - _exec_start

        # Observability: record metrics
        if _metrics:
            _metrics.record_execution(
                agent_name=agent_config.name,  # type: ignore[union-attr]
                task_type="single_agent",
                duration_s=_exec_elapsed,
                success=result.success,
            )

        # Learn from execution (DRY: reuse workflow learner)
        if result.success:
            sm._learn_from_result(result, agent_config, goal=goal)

        # Post-episode learning + auto-save (fire-and-forget background)
        sm._schedule_background_learning(result, goal)

        return result

    async def _execute_multi_agent(self, goal: str, **kwargs: Any) -> EpisodeResult:
        """
        Execute multi-agent mode with configurable discussion paradigm.

        MALLM-inspired paradigms (Becker et al., EMNLP 2025):
            fanout      -- All agents run in parallel on decomposed tasks (default)
            relay       -- Sequential chain; each agent builds on previous output
            debate      -- Agents critique each other's outputs in rounds
            refinement  -- Iterative improve loop until quality stabilizes

        DRY: All paradigms reuse the same AgentRunner.run() and semaphore.
        """
        sm = self._manager
        import time as _time
        from dataclasses import dataclass as _dc
        from dataclasses import field as _fl

        @_dc
        class ActualTrajectory:
            steps: list
            actual_reward: float
            timestamp: float = _fl(default_factory=_time.time)

        from Jotty.core.intelligence.orchestration.state.swarm_roadmap import TaskStatus

        # Extract callbacks and ensemble params before passing to runners
        kwargs.pop("ensemble_context", None)
        status_callback = kwargs.pop("status_callback", None)
        ensemble = kwargs.pop("ensemble", False)
        ensemble_strategy = kwargs.pop("ensemble_strategy", "multi_perspective")
        discussion_paradigm = kwargs.pop("discussion_paradigm", "auto")

        # ── Intelligence-guided agent selection (single entry point: router) ──
        # Router delegates to LearningPipeline.order_agents_for_goal (trust + stigmergy + TRAS).
        # This closes the learning loop: post_episode writes -> run() reads.
        _intelligence_applied = False
        try:
            ordered = sm._router.order_agents_for_goal(goal)
            if ordered:
                sm.agents = ordered
                sm._runners_built = False
                sm._ensure_runners()
                _intelligence_applied = bool(getattr(sm, "learning", None))
                if _intelligence_applied and sm.agents:
                    top = sm.agents[0].name if hasattr(sm.agents[0], "name") else "?"
                    logger.info(f" Router: agents ordered for goal (lead={top})")
        except LearningError as e:
            logger.warning(f"Intelligence-guided selection failed (learning): {e}")
        except Exception as e:
            logger.warning(f"Intelligence-guided selection failed (unexpected): {e}", exc_info=True)

        # Track guidance for A/B effectiveness metrics (per task_type)
        sm._last_run_guided = _intelligence_applied
        try:
            _im_task_type = sm.learning.transfer_learning.extractor.extract_task_type(goal)
        except Exception as e:
            logger.debug(f"Task type extraction for A/B metrics failed: {e}")
            _im_task_type = "_global"
        sm._last_task_type = _im_task_type

        for _tt in (_im_task_type, "_global") if _im_task_type != "_global" else ("_global",):
            if _tt not in sm._intelligence_metrics:
                sm._intelligence_metrics[_tt] = {
                    "guided_runs": 0,
                    "guided_successes": 0,
                    "unguided_runs": 0,
                    "unguided_successes": 0,
                }
            if _intelligence_applied:
                sm._intelligence_metrics[_tt]["guided_runs"] += 1
            else:
                sm._intelligence_metrics[_tt]["unguided_runs"] += 1

        # Auto paradigm selection: use learning data to pick the best paradigm
        if discussion_paradigm == "auto":
            try:
                _task_type = sm.learning.transfer_learning.extractor.extract_task_type(goal)
                discussion_paradigm = sm.learning.recommend_paradigm(_task_type)
                logger.info(
                    f" Auto paradigm: selected '{discussion_paradigm}' "
                    f"for task_type='{_task_type}'"
                )
            except LearningError as e:
                logger.debug(f"Auto paradigm selection failed (learning): {e}")
                discussion_paradigm = "fanout"
            except Exception as e:
                logger.debug(f"Auto paradigm selection failed (unexpected): {e}", exc_info=True)
                discussion_paradigm = "fanout"

        # Track which paradigm is used (for _post_episode_learning)
        sm._last_paradigm = discussion_paradigm

        # MALLM-inspired: Dispatch to alternative paradigms before fan-out
        if discussion_paradigm == "relay":
            # Wire coordination: initiate handoff between relay agents
            if sm.swarm_intelligence:
                try:
                    available = [a.name for a in sm.agents]  # type: ignore[union-attr]
                    if len(available) >= 2:
                        sm.swarm_intelligence.initiate_handoff(
                            from_agent=available[0],
                            to_agent=available[1],
                            task=goal,
                            context={"paradigm": "relay", "agents": available},
                        )
                except Exception as e:
                    logger.debug(f"Relay handoff coordination skipped: {e}")
            return await self._paradigm_relay(goal, status_callback=status_callback, **kwargs)
        elif discussion_paradigm == "debate":
            return await self._paradigm_debate(goal, status_callback=status_callback, **kwargs)
        elif discussion_paradigm == "refinement":
            return await self._paradigm_refinement(goal, status_callback=status_callback, **kwargs)
        # else: 'fanout' — fall through to existing parallel execution below

        # Wire coordination: form coalition for fanout tasks (agents collaborate)
        if sm.swarm_intelligence and len(sm.agents) >= 2:
            try:
                available = [a.name for a in sm.agents]  # type: ignore[union-attr]
                _task_type = sm.learning.transfer_learning.extractor.extract_task_type(goal)
                sm.swarm_intelligence.form_coalition(
                    task_type=_task_type,
                    min_agents=min(2, len(available)),
                    available_agents=available,
                )
                logger.info(f" Coalition formed for '{_task_type}' with {len(available)} agents")
            except LearningError as e:
                logger.debug(f"Coalition formation skipped (learning): {e}")
            except Exception as e:
                logger.debug(f"Coalition formation skipped (unexpected): {e}", exc_info=True)

        # Status update at method start
        safe_status(
            status_callback, "Multi-agent exec", f"starting {len(sm.agents)} parallel agents"
        )

        max_attempts = getattr(sm.config, "max_task_attempts", 2)

        # Clear task board for fresh run (avoid stale tasks from previous runs)
        sm.swarm_task_board.subtasks.clear()
        sm.swarm_task_board.completed_tasks.clear()
        sm.swarm_task_board.execution_order.clear()

        # Add tasks to SwarmTaskBoard
        # Zero-config agents from LLM are PARALLEL (independent sub-goals)
        # Only add dependencies if explicitly specified in agent config
        for i, agent_config in enumerate(sm.agents):
            task_id = f"task_{i+1}"
            # Check if agent has explicit dependencies
            deps = getattr(agent_config, "depends_on", []) or []

            # Use agent's sub-goal (from capabilities) as task description
            sub_goal = (
                agent_config.capabilities[0]
                if agent_config.capabilities
                else f"{goal} (agent: {agent_config.name})"
            )

            sm.swarm_task_board.add_task(
                task_id=task_id,
                description=sub_goal,
                actor=agent_config.name,
                depends_on=deps,  # Empty for parallel execution
            )
            logger.info(
                f" Added task {task_id} for {agent_config.name}: {sub_goal[:50]}... (parallel: {len(deps)==0})"
            )

        all_results = {}  # agent_name -> EpisodeResult
        attempt_counts: Dict[str, Any] = {}  # task_id -> attempts
        _max_iters = getattr(sm.config, "max_episode_iterations", 12)
        _iter_count = 0

        while _iter_count < _max_iters:
            # Collect all ready tasks (no unresolved dependencies)
            batch = []

            while True:
                next_task = sm.swarm_task_board.get_next_task(
                    q_predictor=sm.q_predictor,
                    current_state={"goal": goal, "iteration": _iter_count},
                    goal=goal,
                )
                if next_task is None:
                    break
                # Mark as IN_PROGRESS so it's not returned again
                next_task.status = TaskStatus.IN_PROGRESS
                batch.append(next_task)

            if not batch:
                break
            _iter_count += 1

            # Show batch info immediately with agent names for better UX
            if status_callback and len(batch) > 0:
                try:
                    agent_names = [t.actor for t in batch]
                    status_callback(
                        "Running batch", f"{len(batch)} agents: {', '.join(agent_names[:5])}"
                    )
                    # Show each agent's task for clarity
                    for task in batch:
                        agent_cfg = next((a for a in sm.agents if a.name == task.actor), None)  # type: ignore[union-attr]
                        sub_goal = (
                            agent_cfg.capabilities[0]  # type: ignore[union-attr]
                            if agent_cfg and agent_cfg.capabilities  # type: ignore[union-attr]
                            else task.description[:50]
                        )
                        status_callback(f"  {task.actor}", sub_goal[:60])
                except Exception as e:
                    logger.debug(f"Batch status callback failed: {e}")

            # Pre-execution: trajectory prediction (non-blocking, run in background)
            predictions = {}
            # Skip trajectory prediction to reduce latency - agents start immediately
            # Prediction can happen asynchronously after execution starts
            if sm.trajectory_predictor and len(batch) <= 2:  # Only for small batches
                for task in batch:
                    try:
                        prediction = sm.trajectory_predictor.predict(
                            current_state=sm.get_current_state(),
                            acting_agent=task.actor,
                            proposed_action={"task": task.description},
                            other_agents=[a.name for a in sm.agents if a.name != task.actor],  # type: ignore[union-attr]
                            goal=goal,
                        )
                        predictions[task.actor] = prediction
                    except Exception as e:
                        logger.debug(f"Trajectory prediction skipped for {task.actor}: {e}")

            # Execute batch concurrently (status_callback already extracted at method start)
            # AIOS-inspired: Semaphore limits how many agents call LLM simultaneously.
            # Without this, N agents x (architect + agent + auditor) = 3N concurrent API calls.
            async def _run_task(task: Any) -> Any:
                # Check if we'll need to wait for a slot
                if sm.agent_semaphore._value == 0:
                    sm._scheduling_stats["total_waited"] += 1
                    safe_status(status_callback, f"Agent {task.actor}", "waiting for LLM slot...")

                async with sm.agent_semaphore:
                    # Track concurrency stats
                    sm._scheduling_stats["total_scheduled"] += 1
                    sm._scheduling_stats["current_concurrent"] += 1
                    if (
                        sm._scheduling_stats["current_concurrent"]
                        > sm._scheduling_stats["peak_concurrent"]
                    ):
                        sm._scheduling_stats["peak_concurrent"] = sm._scheduling_stats[
                            "current_concurrent"
                        ]

                    try:
                        # Show which agent is executing
                        agent_cfg = next((a for a in sm.agents if a.name == task.actor), None)  # type: ignore[union-attr]
                        sub_goal = (
                            agent_cfg.capabilities[0]  # type: ignore[union-attr]
                            if agent_cfg and agent_cfg.capabilities  # type: ignore[union-attr]
                            else task.description[:60]
                        )

                        safe_status(status_callback, f"Agent {task.actor}", f"starting: {sub_goal}")

                        # Create agent-specific status callback that prefixes with agent name
                        _agent_reporter = StatusReporter(status_callback).with_prefix(
                            f"  [{task.actor}]"
                        )
                        agent_status_callback = _agent_reporter

                        runner = sm.runners[task.actor]
                        # Pass the agent-specific callback and ensemble params
                        task_kwargs = dict(kwargs)
                        task_kwargs["status_callback"] = agent_status_callback
                        # Forward ensemble flag explicitly so AutoAgent doesn't auto-detect
                        task_kwargs["ensemble"] = ensemble
                        if ensemble:
                            task_kwargs["ensemble_strategy"] = ensemble_strategy
                        # Forward the swarm-level gate decision to skip redundant
                        # per-agent architect/auditor if the swarm already decided
                        task_kwargs.pop(
                            "gate_decision", None
                        )  # single-agent only; don't pass to multi-agent tasks

                        # MULTI-AGENT OPTIMIZATION: Sub-agents with system_prompt
                        # are specialized for analysis/synthesis — they don't need
                        # the full skill pipeline (saves ~100s per agent).
                        # Use direct_llm=True to bypass skill discovery/selection/planning.
                        if agent_cfg and hasattr(agent_cfg, "agent") and agent_cfg.agent:
                            _agent_obj = agent_cfg.agent
                            _has_system_prompt = (
                                hasattr(_agent_obj, "config")
                                and hasattr(_agent_obj.config, "system_prompt")
                                and _agent_obj.config.system_prompt
                            )
                            if _has_system_prompt:
                                task_kwargs["direct_llm"] = True

                        return task, await runner.run(goal=task.description, **task_kwargs)
                    finally:
                        sm._scheduling_stats["current_concurrent"] -= 1

            # Per-agent timeout: prevent any single agent from blocking the entire swarm.
            # Reads from SwarmConfig.actor_timeout (default 900s).
            PER_AGENT_TIMEOUT = getattr(sm.config, "actor_timeout", 900.0)

            async def _run_task_with_timeout(task: Any) -> Any:
                try:
                    return await asyncio.wait_for(_run_task(task), timeout=PER_AGENT_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning(f"Agent {task.actor} timed out after {PER_AGENT_TIMEOUT}s")
                    safe_status(
                        status_callback,
                        f"Agent {task.actor}",
                        f"TIMEOUT ({PER_AGENT_TIMEOUT:.0f}s)",
                    )
                    return task, EpisodeResult(
                        output=f"Agent {task.actor} timed out after {PER_AGENT_TIMEOUT:.0f}s",
                        success=False,
                        trajectory=[{"step": 0, "action": "timeout"}],
                        tagged_outputs=[],
                        episode=sm.episode_count,
                        execution_time=PER_AGENT_TIMEOUT,
                        architect_results=[],
                        auditor_results=[],
                        agent_contributions={},
                    )

            coro_results = await asyncio.gather(
                *[_run_task_with_timeout(t) for t in batch], return_exceptions=True
            )

            # Process results
            for coro_result in coro_results:
                if isinstance(coro_result, Exception):
                    logger.error(f"Task execution exception: {coro_result}")
                    safe_status(status_callback, "Agent error", str(coro_result)[:60])
                    continue

                task, result = coro_result  # type: ignore[misc]
                attempt_counts[task.task_id] = attempt_counts.get(task.task_id, 0) + 1

                # Show agent completion status
                status_icon = "" if result.success else ""
                safe_status(
                    status_callback,
                    f"{status_icon} Agent {task.actor}",
                    "completed" if result.success else "failed",
                )
                reward = 1.0 if result.success else -0.5

                # Post-execution: divergence learning
                if sm.trajectory_predictor and task.actor in predictions:
                    try:
                        prediction = predictions[task.actor]
                        actual = ActualTrajectory(
                            steps=result.trajectory or [], actual_reward=reward
                        )
                        divergence = sm.trajectory_predictor.compute_divergence(prediction, actual)
                        sm.divergence_memory.store(divergence)
                        sm.trajectory_predictor.update_from_divergence(divergence)

                        # Use divergence as TD error weight for Q-update
                        divergence_penalty = 1.0 - min(1.0, divergence.total_divergence())
                        adjusted_reward = reward * divergence_penalty
                        state = {"query": goal, "agent": task.actor}
                        action = {"actor": task.actor, "task": task.description[:100]}
                        from Jotty.core.intelligence.learning.learning_service import (
                            LearningService,
                        )

                        svc = LearningService.get_instance()
                        svc.record_outcome(
                            unit_name=task.actor,
                            state=f"divergence:{goal[:80]}",
                            action=f"actor:{task.actor}|task:{task.description[:60]}",
                            reward=adjusted_reward,
                            domain=getattr(sm.config, "domain", "general"),
                        )
                    except LearningError as e:
                        logger.debug(
                            f"Divergence learning skipped for {task.actor} (learning): {e}"
                        )
                    except Exception as e:
                        logger.debug(
                            f"Divergence learning skipped for {task.actor} (unexpected): {e}",
                            exc_info=True,
                        )

                if result.success:
                    sm.swarm_task_board.complete_task(
                        task.task_id, result={"output": result.output}
                    )
                    all_results[task.actor] = result

                    agent_config = next((a for a in sm.agents if a.name == task.actor), None)  # type: ignore[union-attr]
                    if agent_config:
                        sm._learn_from_result(result, agent_config, goal=task.description or goal)
                else:
                    # Retry with enriched context if attempts remain
                    if attempt_counts.get(task.task_id, 1) < max_attempts:
                        error_msg = str(getattr(result, "error", None) or "Execution failed")
                        fb = FeedbackMessage(
                            source_actor="swarm_manager",
                            target_actor=task.actor,
                            feedback_type=FeedbackType.ERROR,
                            content=f"Previous attempt failed: {error_msg}. Please try a different approach.",
                            context={"attempt": attempt_counts[task.task_id], "error": error_msg},
                            requires_response=False,
                            priority=1,
                        )
                        sm.feedback_channel.send(fb)
                        # Reset task to PENDING for retry
                        try:
                            sm.swarm_task_board.add_task(
                                task_id=f"{task.task_id}_retry{attempt_counts[task.task_id]}",
                                description=task.description,
                                actor=task.actor,
                            )
                        except Exception:
                            sm.swarm_task_board.fail_task(task.task_id, error=error_msg)
                    else:
                        sm.swarm_task_board.fail_task(
                            task.task_id,
                            error=str(getattr(result, "error", None) or "Execution failed"),
                        )
                    all_results[task.actor] = result

        # MAS-ZERO: Handle TOO_HARD signals from agents
        # If any agent signaled TOO_HARD, try re-routing its task
        for agent_name, result in list(all_results.items()):
            output = result.output if hasattr(result, "output") else result
            is_too_hard = (isinstance(output, dict) and output.get("too_hard")) or (
                hasattr(result, "output")
                and isinstance(result.output, dict)
                and result.output.get("too_hard")
            )
            if is_too_hard and not result.success:
                logger.info(
                    f"MAS-ZERO: Agent '{agent_name}' signaled TOO_HARD, "
                    f"task may need re-decomposition"
                )

        # MAS-ZERO: Meta-feedback evaluation (solvability + completeness)
        meta_feedback = sm._mas_zero_evaluate(goal, all_results)

        # MAS-ZERO: Iterative refinement if meta-feedback says refine
        if meta_feedback.get("should_refine") and len(all_results) > 0:
            safe_status(status_callback, "MAS-Evolve", "refining based on meta-feedback")
            all_results = await sm._mas_zero_evolve(
                goal,
                all_results,
                max_iterations=2,
                status_callback=status_callback,
                **kwargs,
            )

        # Cooperative credit assignment
        self._assign_cooperative_credit(all_results, goal)

        # Post-episode learning + auto-save (fire-and-forget background)
        combined_result = self._aggregate_results(all_results, goal)
        sm._schedule_background_learning(combined_result, goal)

        # Observability: record per-agent metrics
        if get_metrics:
            try:
                _metrics = get_metrics()
                for agent_name, result in all_results.items():
                    _metrics.record_execution(
                        agent_name=agent_name,
                        task_type="multi_agent",
                        duration_s=getattr(result, "execution_time", 0.0),
                        success=result.success if hasattr(result, "success") else False,
                    )
            except Exception:
                pass

        return combined_result

    # =========================================================================
    # DISCUSSION PARADIGMS -- delegated to ParadigmExecutor
    # =========================================================================

    async def _paradigm_run_agent(
        self, runner: Any, sub_goal: Any, agent_name: Any, **kwargs: Any
    ) -> Any:
        return await self._paradigms.run_agent(runner, sub_goal, agent_name, **kwargs)

    async def _paradigm_relay(self, goal: Any, **kwargs: Any) -> Any:
        return await self._paradigms.relay(goal, **kwargs)

    async def _paradigm_debate(self, goal: Any, **kwargs: Any) -> Any:
        return await self._paradigms.debate(goal, **kwargs)

    async def _paradigm_refinement(self, goal: Any, **kwargs: Any) -> Any:
        return await self._paradigms.refinement(goal, **kwargs)

    def _aggregate_results(self, results: Any, goal: Any) -> Any:
        return self._paradigms.aggregate_results(results, goal)

    def _assign_cooperative_credit(self, results: Any, goal: Any) -> Any:
        return self._paradigms.assign_cooperative_credit(results, goal)

    async def autonomous_setup(self, goal: str, status_callback: Any = None) -> Any:
        """
        Autonomous setup: Research, install, configure.

        Public method for manual autonomous setup.
        DRY: Reuses all autonomous components.

        Args:
            goal: Task goal
            status_callback: Optional callback(stage, detail) for progress

        Example:
            await swarm.autonomous_setup("Set up Reddit scraping")
        """
        sm = self._manager
        _status = StatusReporter(status_callback, logger, emoji="")

        # Cache check: skip if already set up for this goal
        cache_key = hash(goal)
        if cache_key in sm._setup_cache:  # type: ignore[comparison-overlap]
            _status("Setup", "using cached")
            return

        # Parse intent to understand requirements
        _status("Parsing intent", goal)
        task_graph = sm.swarm_intent_parser.parse(goal)

        # Research solutions if needed
        if task_graph.requirements or task_graph.integrations:
            # Filter out stop words and meaningless single-word requirements
            stop_words = {
                "existing",
                "use",
                "check",
                "find",
                "get",
                "the",
                "a",
                "an",
                "and",
                "or",
                "for",
                "with",
            }
            meaningful_requirements = [
                req
                for req in task_graph.requirements
                if req.lower() not in stop_words and len(req.split()) > 1
            ]

            if meaningful_requirements:
                _status("Researching", f"{len(meaningful_requirements)} requirements")

            for i, requirement in enumerate(meaningful_requirements):
                if not requirement.strip():
                    continue
                _status("Research", f"[{i+1}/{len(meaningful_requirements)}] {requirement[:30]}")
                research_result = await sm.swarm_researcher.research(requirement)
                if research_result.tools_found:
                    _status("Found tools", ", ".join(research_result.tools_found[:3]))
                    for tool in research_result.tools_found:
                        _status("Installing", tool)
                        await sm.swarm_installer.install(tool)

        # Configure integrations (handled externally if needed)
        if task_graph.integrations:
            _status("Configuring", f"{len(task_graph.integrations)} integrations noted")
            for integration in task_graph.integrations:
                logger.info(f"Integration required: {integration}")

        # Mark as cached
        sm._setup_cache[cache_key] = True  # type: ignore[index]
        _status("Setup complete", "")


def _build_response_digest(content: str, max_len: int = 1500) -> str:
    """
    Build a structural digest of a response for storage in episode outcomes.
    Shows outline + representative samples instead of an arbitrary character cutoff.
    """
    import re as _re

    lines = content.split("\n")
    words = content.split()

    # Extract headings (skip code blocks)
    headings = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if not in_code and stripped.startswith("#"):
            title = stripped.lstrip("# ").strip()
            if title:
                headings.append(title)

    code_blocks = len(_re.findall(r"```\w*\n.*?```", content, _re.DOTALL))

    parts = [f"[{len(words)} words, {len(headings)} sections, {code_blocks} code blocks]"]

    if headings:
        parts.append("Outline: " + " → ".join(headings[:10]))

    # Opening paragraph
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("```"):
            parts.append("Opens: " + stripped[:200])
            break

    # One sample from the middle
    mid_sections: list = []
    current_heading = ""
    current_text: list = []
    in_code_blk = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_blk = not in_code_blk
        if not in_code_blk and stripped.startswith("#") and stripped.lstrip("# ").strip():
            if current_text and current_heading:
                body = "\n".join(current_text).strip()
                if len(body) > 80:
                    mid_sections.append((current_heading, body))
            current_heading = stripped.lstrip("# ").strip()
            current_text = []
        else:
            current_text.append(line)
    if current_text and current_heading:
        body = "\n".join(current_text).strip()
        if len(body) > 80:
            mid_sections.append((current_heading, body))

    if mid_sections:
        mid = mid_sections[len(mid_sections) // 2]
        paras = [p.strip() for p in mid[1].split("\n\n") if p.strip()]
        sample = paras[0] if paras else mid[1]
        parts.append(f"[{mid[0]}]: {sample[:300]}")

    # First code block sample (if present)
    code_match = _re.search(r"```\w*\n(.*?)```", content, _re.DOTALL)
    if code_match:
        code_lines = code_match.group(1).strip().split("\n")
        if len(code_lines) > 8:
            code_sample = "\n".join(code_lines[:6]) + f"\n... ({len(code_lines)} lines)"
        else:
            code_sample = "\n".join(code_lines)
        parts.append(f"Code: ```\n{code_sample}\n```")

    digest = "\n".join(parts)
    return digest[:max_len]

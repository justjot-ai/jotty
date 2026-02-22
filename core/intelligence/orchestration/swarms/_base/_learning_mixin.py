"""
Learning mixin for SwarmLearning.

Contains the core learning lifecycle methods: pre/post-execution learning hooks,
improvement cycles, and trace recording.

Coordination protocols are in _coordination_mixin.py.
Knowledge retrieval/storage/context building are in _knowledge_mixin.py.

Extracted from base_swarm.py to reduce file size.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from Jotty.core.infrastructure.foundation.exceptions import (
    AgentExecutionError,
    LearningError,
    LLMError,
    MemoryError,
    MemoryStorageError,
)

from ._coordination_mixin import SwarmCoordinationMixin
from ._knowledge_mixin import SwarmKnowledgeMixin
from .swarm_types import (
    AgentRole,
    EvaluationResult,
    ExecutionTrace,
    ImprovementSuggestion,
    ImprovementType,
)

logger = logging.getLogger(__name__)


class SwarmLearningMixin(SwarmCoordinationMixin, SwarmKnowledgeMixin):
    """Mixin providing learning infrastructure for SwarmLearning.

    Inherits from:
    - SwarmCoordinationMixin: circuit breakers, gossip, coalitions, etc.
    - SwarmKnowledgeMixin: expert knowledge, failure analysis, context building

    Methods in this mixin handle the core learning lifecycle:
    - Pre-execution: load relevant context from memory/learner
    - Post-execution: store results, run improvement cycle, update learner
    - Trace recording: log execution traces for debugging

    MAPLE paper integration: Learning is split into hot-path (sync, needed
    for next execution) and cold-path (deferred, background async task).

    Expects to be mixed into SwarmLearning which provides:
    - self._memory, self._context_manager, self._learner
    - self._gold_db, self._improvement_history, self._evaluation_history
    - self.config, self._traces, self._execution_count
    - self._improvement_agents dict
    """

    # Background learning tasks (MAPLE: cold-path deferred learning)
    _bg_learning_tasks: set = set()

    def _classify_error(self, success: bool, result: Any, execution_time: float) -> Optional[str]:
        """Classify execution outcome for learning.

        Returns an actionable error category so the curriculum can adapt
        differently for invalid input vs infrastructure failures vs timeouts.
        """
        if success:
            return None

        error_msg = str(getattr(result, "error", "") or "") if result else ""

        if "No valid stock ticker" in error_msg or "not found" in error_msg.lower():
            return "invalid_input"
        if "timeout" in error_msg.lower() or execution_time > 120:
            return "timeout"
        if "api" in error_msg.lower() or "connection" in error_msg.lower():
            return "infrastructure"
        if "permission" in error_msg.lower() or "auth" in error_msg.lower():
            return "authentication"
        return "execution_failure"

    async def _pre_execute_learning(self) -> Dict[str, Any]:
        """
        Pre-execution learning hook. Called at start of execute().

        Auto-connects SwarmIntelligence, loads saved state, runs warmup
        on first run, computes MorphAgent scores, analyzes tool performance,
        runs coordination protocols (circuit breakers, gossip, auctions,
        coalitions, supervisor tree, backpressure), and stitches all findings
        into a learned context dict.

        Returns:
            Dict with learning context (has_learning, tool_performance,
            agent_scores, weak_tools, recommendations, coordination, etc.)
        """
        learned_context = {
            "has_learning": False,
            "tool_performance": {},
            "agent_scores": {},
            "weak_tools": [],
            "strong_tools": [],
            "recommendations": [],
            "warmup_completed": False,
            "coordination": {},
        }

        try:
            # 1. Auto-connect SwarmIntelligence if not connected
            if not self._swarm_intelligence:  # type: ignore[attr-defined]
                self.connect_swarm_intelligence()  # type: ignore[attr-defined]

            si = self._swarm_intelligence  # type: ignore[attr-defined]
            if not si:
                self._learned_context = learned_context
                return learned_context

            # 2. Auto-warmup if first run (no feedback history yet)
            stats = si.curriculum_generator.get_curriculum_stats()
            save_path = self._get_intelligence_save_path()  # type: ignore[attr-defined]
            if stats["feedback_count"] == 0 and not Path(save_path).exists():
                await self._run_auto_warmup()  # type: ignore[attr-defined]
                learned_context["warmup_completed"] = True
                logger.info("Auto-warmup complete — seeded initial learning data")

            # 3. Compute MorphAgent scores for all registered agents
            if si.agent_profiles:
                morph_scores = si.morph_scorer.compute_all_scores(si.agent_profiles)
                for agent_name, scores in morph_scores.items():
                    profile = si.agent_profiles.get(agent_name)
                    learned_context["agent_scores"][agent_name] = {
                        "rcs": scores.rcs,
                        "rds": scores.rds,
                        "tras": scores.tras,
                        "consistency": scores.rcs_components.get("consistency", 0.5),
                        "focus": scores.rcs_components.get("focus", 0.5),
                        "specialization": scores.rcs_components.get("specialization", 0.5),
                        "total_tasks": profile.total_tasks if profile else 0,
                    }

            # 4. Analyze tool success rates via ToolManager
            tool_analysis = self._manage_tools()  # type: ignore[attr-defined]
            # Cache for reuse in _post_execute_learning (avoids redundant recomputation)
            self._cached_tool_analysis = tool_analysis
            learned_context["tool_performance"] = stats.get("tool_success_rates", {})
            learned_context["weak_tools"] = tool_analysis.get("weak_tools", [])
            learned_context["strong_tools"] = tool_analysis.get("strong_tools", [])

            # 5. STITCH: Combine weak tool + inconsistent agent = PRIORITY
            recommendations = []
            for weak in learned_context["weak_tools"]:  # type: ignore[attr-defined]
                tool_name = weak["tool"]
                rate = weak["success_rate"]
                for agent_name, agent_data in learned_context["agent_scores"].items():  # type: ignore[attr-defined]
                    consistency = agent_data.get("consistency", 0.5)
                    if consistency < 0.5:
                        recommendations.insert(
                            0,
                            {
                                "priority": "HIGH",
                                "type": "tool_and_agent",
                                "tool": tool_name,
                                "tool_rate": rate,
                                "agent": agent_name,
                                "consistency": consistency,
                                "action": f"PRIORITY: Replace {tool_name} ({rate:.0%} success) AND "
                                f"stabilize {agent_name} (consistency={consistency:.2f})",
                            },
                        )
                    else:
                        recommendations.append(
                            {
                                "priority": "MEDIUM",
                                "type": "tool_only",
                                "tool": tool_name,
                                "tool_rate": rate,
                                "action": f"Replace {tool_name} ({rate:.0%} success) — agent {agent_name} is stable",
                            }
                        )

            # Add agent-only warnings (consistent tool but inconsistent agent)
            for agent_name, agent_data in learned_context["agent_scores"].items():  # type: ignore[attr-defined]
                consistency = agent_data.get("consistency", 0.5)
                if consistency < 0.5 and not any(
                    r.get("agent") == agent_name for r in recommendations
                ):
                    recommendations.append(
                        {
                            "priority": "LOW",
                            "type": "agent_only",
                            "agent": agent_name,
                            "consistency": consistency,
                            "action": f"Warn {agent_name}: outputs inconsistent (consistency={consistency:.2f})",
                        }
                    )

            learned_context["recommendations"] = recommendations

            # 6. Retrieve expert domain knowledge from SwarmMemory
            expert_knowledge = self._retrieve_expert_knowledge()
            learned_context["expert_knowledge"] = expert_knowledge

            # 7. Analyze prior failures for recovery
            prior_failures = self._analyze_prior_failures()
            learned_context["prior_failures"] = prior_failures

            # 8. Analyze morph score trends (improving vs declining)
            score_trends = {}
            if si and si.morph_score_history and len(si.morph_score_history) >= 2:
                latest = si.morph_score_history[-1].get("scores", {})
                # Compare with 3 runs ago (or earliest available)
                compare_idx = max(0, len(si.morph_score_history) - 4)
                earlier = si.morph_score_history[compare_idx].get("scores", {})
                for agent_name_key in latest:
                    curr_rcs = latest[agent_name_key].get("rcs", 0)
                    prev_rcs = earlier.get(agent_name_key, {}).get("rcs", 0)
                    if prev_rcs > 0:
                        delta = curr_rcs - prev_rcs
                        if abs(delta) > 0.02:  # Only report meaningful changes
                            score_trends[agent_name_key] = {
                                "current": curr_rcs,
                                "previous": prev_rcs,
                                "delta": delta,
                                "direction": "improving" if delta > 0 else "declining",
                            }
            learned_context["score_trends"] = score_trends

            # 9. Coordination protocols (circuit breakers, gossip, auctions,
            #    coalitions, supervisor tree, backpressure)
            coordination = self._coordinate_pre_execution(si)
            learned_context["coordination"] = coordination

            learned_context["has_learning"] = bool(
                learned_context["tool_performance"]
                or learned_context["agent_scores"]
                or learned_context["warmup_completed"]
                or learned_context["expert_knowledge"]
                or learned_context["prior_failures"]
                or learned_context["score_trends"]
                or learned_context["coordination"]
            )

            self._learned_context = learned_context
            if learned_context["has_learning"]:
                expert_count = len(learned_context.get("expert_knowledge", []))  # type: ignore[arg-type]
                logger.info(
                    f"Pre-execution learning: {len(learned_context['tool_performance'])} tools tracked, "  # type: ignore[arg-type]
                    f"{len(learned_context['agent_scores'])} agents scored, "  # type: ignore[arg-type]
                    f"{len(recommendations)} recommendations, "
                    f"{expert_count} expert patterns loaded"
                )

        except (ImportError, AttributeError) as e:
            # Expected: missing optional dependencies or uninitialized components
            logger.debug(f"Pre-execution learning skipped (optional dep): {e}")
            self._learned_context = learned_context
        except (LearningError, MemoryError) as e:
            # Recoverable: learning or memory subsystem failed but execution can proceed
            logger.warning(f"Pre-execution learning failed (recoverable): {e}")
            self._learned_context = learned_context
        except Exception as e:
            # Unexpected: log at warning so degradation is visible
            logger.warning(
                f"Pre-execution learning failed unexpectedly: {type(e).__name__}: {e}",
                exc_info=True,
            )
            self._learned_context = learned_context

        return learned_context

    async def _post_execute_learning(
        self,
        success: bool,
        execution_time: float,
        tools_used: List[str],
        task_type: str,
        output_data: Dict[str, Any] | None = None,
        input_data: Dict[str, Any] | None = None,
        result: Any = None,
    ) -> Any:
        """
        Post-execution learning hook. Called at end of execute().

        MAPLE paper: Split into hot-path (sync, needed for next execution)
        and cold-path (deferred async, background learning).

        Hot-path: error classification, executor feedback, MorphAgent scores,
        tool re-analysis, coordination protocols.
        Cold-path: evaluation, learning extraction, improvement cycle, state save.
        """
        try:
            # === HOT PATH (sync — needed for next execution) ===

            # 1. Send executor feedback (tools, success, timing, error classification)
            error_type = self._classify_error(success, result, execution_time)
            error_message = str(getattr(result, "error", "") or "")[:500] if not success else None
            input_query = str(getattr(self, "_run_input", {}).get("query", ""))[:200] or None

            self._send_executor_feedback(  # type: ignore[attr-defined]
                task_type=task_type,
                success=success,
                tools_used=tools_used,
                execution_time=execution_time,
                error_type=error_type,
                error_message=error_message,
                input_query=input_query,
            )

            si = self._swarm_intelligence  # type: ignore[attr-defined]
            if not si:
                return

            # 2. Recompute MorphAgent scores with new data
            if si.agent_profiles:
                morph_scores = si.morph_scorer.compute_all_scores(si.agent_profiles)
                si.morph_score_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "scores": {
                            name: {"rcs": s.rcs, "rds": s.rds, "tras": s.tras}
                            for name, s in morph_scores.items()
                        },
                    }
                )

            # 3. Re-analyze tools and update assignments.
            if not getattr(self, "_cached_tool_analysis", None):
                self._manage_tools()  # type: ignore[attr-defined]

            # 3a. Post-execution coordination protocols
            self._coordinate_post_execution(
                si=si,
                success=success,
                execution_time=execution_time,
                tools_used=tools_used,
                task_type=task_type,
            )

            # 3b. Feed tool execution data to learning system
            try:
                from Jotty.core.infrastructure.integration.tool_execution_guard import (
                    get_tool_execution_guard,
                )
                from Jotty.core.intelligence.learning.tool_learning import (
                    get_tool_learning_feedback,
                )

                guard = get_tool_execution_guard()
                feedback = get_tool_learning_feedback()
                for _interceptor in guard.interceptor_registry._interceptors.values():
                    if _interceptor.get_all_calls():
                        feedback.feed_from_interceptor(_interceptor)
                guard.interceptor_registry.clear_all()
            except Exception as e:
                logger.debug(f"Tool learning feedback skipped: {e}")

            # === COLD PATH (deferred — schedule as background task) ===
            self._schedule_deferred_learning(
                si=si,
                success=success,
                execution_time=execution_time,
                tools_used=tools_used,
                task_type=task_type,
                output_data=output_data,
                input_data=input_data,
            )

        except (ImportError, AttributeError) as e:
            logger.debug(f"Post-execution learning skipped (optional dep): {e}")
        except (LearningError, MemoryError) as e:
            logger.warning(
                f"Post-execution learning failed (recoverable): {e}. "
                f"Learning data for this execution may be incomplete."
            )
        except Exception as e:
            logger.warning(
                f"Post-execution learning failed unexpectedly: {type(e).__name__}: {e}. "
                f"Learning data for this execution may be lost.",
                exc_info=True,
            )

    def _schedule_deferred_learning(
        self,
        si: Any,
        success: bool,
        execution_time: float,
        tools_used: List[str],
        task_type: str,
        output_data: Optional[Dict[str, Any]] = None,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Schedule cold-path learning as a background task (MAPLE paper).

        Falls back to synchronous execution when no event loop is available
        (e.g. test environments).
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self._deferred_post_learning(
                    si=si,
                    success=success,
                    execution_time=execution_time,
                    tools_used=tools_used,
                    task_type=task_type,
                    output_data=output_data,
                    input_data=input_data,
                )
            )
            self._bg_learning_tasks.add(task)
            task.add_done_callback(self._bg_learning_tasks.discard)
        except RuntimeError:
            # No event loop — run synchronously (test/CLI environments)
            import asyncio as _aio

            try:
                _aio.get_event_loop().run_until_complete(
                    self._deferred_post_learning(
                        si=si,
                        success=success,
                        execution_time=execution_time,
                        tools_used=tools_used,
                        task_type=task_type,
                        output_data=output_data,
                        input_data=input_data,
                    )
                )
            except Exception as e:
                logger.debug(f"Sync fallback for deferred learning failed: {e}")

    async def _deferred_post_learning(
        self,
        si: Any,
        success: bool,
        execution_time: float,
        tools_used: List[str],
        task_type: str,
        output_data: Optional[Dict[str, Any]] = None,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Cold-path learning: evaluation, extraction, improvement, state save.

        MAPLE paper: These steps are not needed for the next execution and
        can safely run in the background without blocking response return.
        Errors are logged but never propagated.
        """
        try:
            # 4. Evaluate output against gold standard
            evaluation = None
            if success and output_data and self.config.enable_self_improvement:  # type: ignore[attr-defined]
                try:
                    evaluation = await self._evaluate_output(  # type: ignore[attr-defined]
                        output=output_data, task_type=task_type, input_data=input_data or {}
                    )
                    if evaluation:
                        logger.info(
                            f"Evaluation: {evaluation.result.value} "
                            f"(score: {evaluation.overall_score:.2f})"
                        )
                except (LLMError, AgentExecutionError) as eval_err:
                    logger.warning(f"Evaluation failed (LLM/agent): {eval_err}")
                except Exception as eval_err:
                    logger.debug(f"Evaluation skipped (unexpected): {eval_err}")

            # 4a. Audit evaluation quality (non-blocking)
            if evaluation and self._auditor and output_data:  # type: ignore[attr-defined]
                try:
                    audit_result = await self._auditor.audit_evaluation(  # type: ignore[attr-defined]
                        evaluation={
                            "scores": evaluation.scores,
                            "overall_score": evaluation.overall_score,
                            "result": evaluation.result.value,
                            "feedback": evaluation.feedback,
                        },
                        output_data=output_data,
                        context=json.dumps({"task_type": task_type}),
                    )
                    if not audit_result.get("passed", True):
                        logger.warning(
                            f"Audit failed for evaluation: {audit_result.get('reasoning', 'unknown')}"
                        )
                except Exception:
                    pass  # Non-blocking

            # 4b. Record iteration in benchmarks
            if si:
                try:
                    score = evaluation.overall_score if evaluation else (1.0 if success else 0.0)
                    si.benchmarks.record_iteration(
                        iteration_id=f"{task_type}_{int(__import__('time').time())}",
                        task_type=task_type,
                        score=score,
                        execution_time=execution_time,
                        success=success,
                    )
                except LearningError as e:
                    logger.warning(f"Benchmark recording failed (learning): {e}")
                except Exception as e:
                    logger.debug(f"Benchmark recording failed (unexpected): {e}")

            # 4c. Auto-curate gold standard from excellent outputs
            if (
                evaluation
                and evaluation.overall_score >= 0.9
                and evaluation.result in (EvaluationResult.EXCELLENT, EvaluationResult.GOOD)
                and self._gold_db  # type: ignore[attr-defined]
                and output_data
                and input_data
            ):
                try:
                    self._curate_gold_standard(task_type, input_data, output_data, evaluation)  # type: ignore[attr-defined]
                except (LearningError, MemoryStorageError) as e:
                    logger.warning(f"Gold standard curation failed (recoverable): {e}")
                except Exception as e:
                    logger.debug(f"Gold standard curation failed (unexpected): {e}")

            # 4d. Extract learnings from excellent executions
            if (
                evaluation
                and evaluation.overall_score >= 0.9
                and self._learner  # type: ignore[attr-defined]
                and output_data
                and input_data
            ):
                try:
                    learnings = await self._learner.extract_learnings(  # type: ignore[attr-defined]
                        input_data=input_data,
                        output_data=output_data,
                        evaluation={
                            "scores": evaluation.scores,
                            "overall_score": evaluation.overall_score,
                            "feedback": evaluation.feedback,
                        },
                        domain=self.config.domain,  # type: ignore[attr-defined]
                    )
                    if learnings and self._improvement_history:  # type: ignore[attr-defined]
                        now = datetime.now().isoformat()
                        for learning in learnings:
                            suggestion = ImprovementSuggestion(
                                agent_role=AgentRole.ACTOR,
                                improvement_type=ImprovementType.TRAINING_DATA,
                                description=learning,
                                priority=3,
                                expected_impact=0.5,
                                implementation_details={"source": "learner_extraction"},
                                based_on_evaluations=[evaluation.gold_standard_id],
                            )
                            sid = hashlib.md5(
                                f"{suggestion.agent_role.value}:{suggestion.description}:{now}".encode()
                            ).hexdigest()[:12]
                            self._improvement_history.history.append(  # type: ignore[attr-defined]
                                {
                                    "id": sid,
                                    "suggestion": asdict(suggestion),
                                    "status": "completed",
                                    "created_at": now,
                                    "applied_at": now,
                                    "outcome": "success",
                                    "impact_measured": 0.5,
                                    "notes": "Auto-extracted from excellent execution",
                                }
                            )
                        self._improvement_history._save_history()  # type: ignore[attr-defined]
                        logger.info(
                            f"Extracted {len(learnings)} learnings from excellent execution"
                        )
                except (LearningError, LLMError) as e:
                    logger.warning(f"Learning extraction failed (recoverable): {e}")
                except Exception as e:
                    logger.debug(f"Learning extraction failed (unexpected): {e}")

            # 5. Run improvement cycle if evaluation below threshold
            if evaluation and evaluation.overall_score < self.config.improvement_threshold:  # type: ignore[attr-defined]
                try:
                    suggestions = await self._run_improvement_cycle()
                    if suggestions:
                        logger.info(f"Generated {len(suggestions)} improvement suggestions")
                except (LearningError, LLMError) as imp_err:
                    logger.warning(f"Improvement cycle failed (recoverable): {imp_err}")
                except Exception as imp_err:
                    logger.debug(f"Improvement cycle skipped (unexpected): {imp_err}")

            # 6. Save state to disk
            try:
                save_path = self._get_intelligence_save_path()  # type: ignore[attr-defined]
                si.save(save_path)
                logger.debug(f"Post-execution learning state saved to {save_path}")
            except (OSError, IOError) as save_err:
                logger.warning(f"Failed to save post-execution state (I/O): {save_err}")
            except Exception as save_err:
                logger.warning(
                    f"Failed to save post-execution state: {type(save_err).__name__}: {save_err}"
                )

            # 7. Store execution outcome as expert improvement in SwarmMemory
            self._store_execution_as_improvement(
                success=success,
                execution_time=execution_time,
                tools_used=tools_used,
                task_type=task_type,
            )

        except Exception as e:
            # MAPLE: cold-path errors never propagate to caller
            logger.warning(
                f"Deferred learning failed: {type(e).__name__}: {e}",
                exc_info=True,
            )

    async def _run_improvement_cycle(self) -> List[ImprovementSuggestion]:
        """Run the self-improvement cycle."""
        if not self.config.enable_self_improvement or not self._reviewer:  # type: ignore[attr-defined]
            return []

        # Check if improvement is needed (persistent across sessions)
        recent_evals = self._evaluation_history.get_recent(10)  # type: ignore[attr-defined]
        avg_score = self._evaluation_history.get_average_score(10)  # type: ignore[attr-defined]
        if not recent_evals or avg_score >= self.config.improvement_threshold:  # type: ignore[attr-defined]
            logger.info(f"Performance good ({avg_score:.2f}), skipping improvement cycle")
            return []

        # Get suggestions from reviewer
        agent_configs = {
            AgentRole.EXPERT: self._expert.config if self._expert else None,  # type: ignore[attr-defined]
            AgentRole.REVIEWER: self._reviewer.config if self._reviewer else None,  # type: ignore[attr-defined]
            AgentRole.PLANNER: self._planner.config if self._planner else None,  # type: ignore[attr-defined]
            AgentRole.ACTOR: self._actor.config if self._actor else None,  # type: ignore[attr-defined]
            AgentRole.AUDITOR: self._auditor.config if self._auditor else None,  # type: ignore[attr-defined]
            AgentRole.LEARNER: self._learner.config if self._learner else None,  # type: ignore[attr-defined]
        }
        agent_configs = {k: v for k, v in agent_configs.items() if v}

        suggestions = await self._reviewer.analyze_and_suggest(recent_evals, agent_configs)  # type: ignore[attr-defined]

        # Record suggestions
        for suggestion in suggestions:
            self._improvement_history.record_suggestion(suggestion)  # type: ignore[attr-defined]

        return suggestions

    def _record_trace(
        self,
        agent_name: str,
        agent_role: AgentRole,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        execution_time: float,
        success: bool,
        error: Optional[str] = None,
        tools_used: List[str] | None = None,
    ) -> Any:
        """Record execution trace for learning and Agent0 feedback."""
        trace = ExecutionTrace(
            agent_name=agent_name,
            agent_role=agent_role,
            input_data=input_data,
            output_data=output_data,
            execution_time=execution_time,
            success=success,
            error=error,
        )
        self._traces.append(trace)  # type: ignore[attr-defined]

        # Fire TUI trace callback if active
        try:
            from .coding_swarm import utils as _cu

            if _cu._active_trace_callback is not None:
                _cu._active_trace_callback(
                    {
                        "agent": agent_name,
                        "role": agent_role.value if agent_role else "",
                        "time": execution_time,
                        "success": success,
                        "error": error,
                        "output_summary": str(output_data)[:100] if output_data else "",
                    }
                )
        except Exception as e:
            logger.debug(f"Trace recording failed: {e}")

        # Agent0: Per-phase swarm-level feedback removed — swarm-level recording
        # is handled once by _post_execute_learning() at end of execute().
        # Only per-agent recording happens here (below).

        # MorphAgent: Update agent profile for per-agent tracking
        swarm_name = self.config.name or "base_swarm"  # type: ignore[attr-defined]
        si = self._swarm_intelligence  # type: ignore[attr-defined]
        if si and hasattr(si, "agent_profiles"):
            si.register_agent(agent_name)
            # Record task result under individual agent name (not swarm name)
            # so per-agent profiles accumulate real task_success data
            if agent_name != swarm_name:
                task_type_label = agent_role.value if agent_role else "unknown"
                si.record_task_result(
                    agent_name=agent_name,
                    task_type=task_type_label,
                    success=success,
                    execution_time=execution_time,
                )

            # Byzantine verification: verify agent's result against actual outcome.
            # Unlike classical BFT where agents self-report, here we verify the
            # execution result (output_data) against the success flag to catch
            # silent failures (agent returns garbage but claims success).
            try:
                si.byzantine.verify_claim(
                    agent=agent_name,
                    claimed_success=success,
                    actual_result=output_data,
                    task_type=task_type_label,
                )
            except Exception as e:
                logger.debug(f"Byzantine verification failed for {agent_name}: {e}")

            # Circuit breaker: track per-agent failures so repeatedly failing
            # agents get temporarily blocked from receiving new tasks.
            try:
                if success:
                    si.record_circuit_success(agent_name)
                else:
                    si.record_circuit_failure(agent_name)
            except Exception as e:
                logger.debug(f"Circuit breaker update failed for {agent_name}: {e}")

        # Store in memory for learning using outcome-weighted storage
        if self._memory and self.config.enable_learning:  # type: ignore[attr-defined]
            try:
                outcome = "success" if success else "failure"
                task_type_label = agent_role.value if agent_role else "unknown"
                domain = getattr(self.config, "domain", None) or self.config.name or "general"  # type: ignore[attr-defined]

                self._memory.store_with_outcome(  # type: ignore[attr-defined]
                    content=json.dumps(asdict(trace), default=str),
                    outcome=outcome,
                    context={"swarm": self.config.name, "agent": agent_name},  # type: ignore[attr-defined]
                    goal=f"Execution trace: {agent_name}",
                    domain=domain,
                    task_type=task_type_label,
                )
            except (MemoryStorageError, MemoryError) as e:
                logger.warning(f"Failed to store trace in memory (memory): {e}")
            except Exception as e:
                logger.debug(f"Failed to store trace in memory (unexpected): {e}")

        # Auto-persist memory to disk after trace storage
        if hasattr(self, "_memory_persistence") and self._memory_persistence:
            try:
                self._memory_persistence.save()
            except Exception as e:
                logger.debug(f"Memory auto-save failed: {e}")

    def record_improvement_outcome(
        self, suggestion_id: str, success: bool, impact: float, notes: str = ""
    ) -> Any:
        """Record the outcome of an applied improvement."""
        if self._improvement_history:  # type: ignore[attr-defined]
            self._improvement_history.record_outcome(suggestion_id, success, impact, notes)  # type: ignore[attr-defined]


# =============================================================================
# EXPORTS
# =============================================================================

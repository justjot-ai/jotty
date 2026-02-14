"""
Learning mixin for BaseSwarm.

Contains all learning-related methods: pre/post-execution learning hooks,
context building, expert knowledge retrieval, failure analysis, improvement
cycles, and trace recording.

Extracted from base_swarm.py to reduce file size.
"""

import logging
import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from .swarm_types import (
    AgentRole,
    ExecutionTrace,
    ImprovementSuggestion,
    ImprovementType,
)
from ..foundation.exceptions import (
    LearningError,
    RewardCalculationError,
    CreditAssignmentError,
    PolicyUpdateError,
    MemoryError,
    MemoryStorageError,
    MemoryRetrievalError,
    ExecutionError,
    AgentExecutionError,
    LLMError,
    ConfigurationError,
)

logger = logging.getLogger(__name__)


class SwarmLearningMixin:
    """Mixin providing learning infrastructure for BaseSwarm.

    Methods in this mixin handle the learning lifecycle:
    - Pre-execution: load relevant context from memory/learner
    - Post-execution: store results, run improvement cycle, update learner
    - Context building: format learned knowledge for agent prompts
    - Expert knowledge: retrieve and apply prior knowledge
    - Trace recording: log execution traces for debugging

    Expects to be mixed into BaseSwarm which provides:
    - self._memory, self._context_manager, self._learner
    - self._gold_db, self._improvement_history, self._evaluation_history
    - self.config, self._traces, self._execution_count
    - self._improvement_agents dict
    """

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
            'has_learning': False,
            'tool_performance': {},
            'agent_scores': {},
            'weak_tools': [],
            'strong_tools': [],
            'recommendations': [],
            'warmup_completed': False,
            'coordination': {},
        }

        try:
            # 1. Auto-connect SwarmIntelligence if not connected
            if not self._swarm_intelligence:
                self.connect_swarm_intelligence()

            si = self._swarm_intelligence
            if not si:
                self._learned_context = learned_context
                return learned_context

            # 2. Auto-warmup if first run (no feedback history yet)
            stats = si.curriculum_generator.get_curriculum_stats()
            save_path = self._get_intelligence_save_path()
            if stats['feedback_count'] == 0 and not Path(save_path).exists():
                warmup_result = await self._run_auto_warmup()
                learned_context['warmup_completed'] = True
                logger.info("Auto-warmup complete — seeded initial learning data")

            # 3. Compute MorphAgent scores for all registered agents
            if si.agent_profiles:
                morph_scores = si.morph_scorer.compute_all_scores(si.agent_profiles)
                for agent_name, scores in morph_scores.items():
                    profile = si.agent_profiles.get(agent_name)
                    learned_context['agent_scores'][agent_name] = {
                        'rcs': scores.rcs,
                        'rds': scores.rds,
                        'tras': scores.tras,
                        'consistency': scores.rcs_components.get('consistency', 0.5),
                        'focus': scores.rcs_components.get('focus', 0.5),
                        'specialization': scores.rcs_components.get('specialization', 0.5),
                        'total_tasks': profile.total_tasks if profile else 0,
                    }

            # 4. Analyze tool success rates via ToolManager
            tool_analysis = self._manage_tools()
            learned_context['tool_performance'] = stats.get('tool_success_rates', {})
            learned_context['weak_tools'] = tool_analysis.get('weak_tools', [])
            learned_context['strong_tools'] = tool_analysis.get('strong_tools', [])

            # 5. STITCH: Combine weak tool + inconsistent agent = PRIORITY
            recommendations = []
            for weak in learned_context['weak_tools']:
                tool_name = weak['tool']
                rate = weak['success_rate']
                for agent_name, agent_data in learned_context['agent_scores'].items():
                    consistency = agent_data.get('consistency', 0.5)
                    if consistency < 0.5:
                        recommendations.insert(0, {
                            'priority': 'HIGH',
                            'type': 'tool_and_agent',
                            'tool': tool_name,
                            'tool_rate': rate,
                            'agent': agent_name,
                            'consistency': consistency,
                            'action': f"PRIORITY: Replace {tool_name} ({rate:.0%} success) AND "
                                      f"stabilize {agent_name} (consistency={consistency:.2f})"
                        })
                    else:
                        recommendations.append({
                            'priority': 'MEDIUM',
                            'type': 'tool_only',
                            'tool': tool_name,
                            'tool_rate': rate,
                            'action': f"Replace {tool_name} ({rate:.0%} success) — agent {agent_name} is stable"
                        })

            # Add agent-only warnings (consistent tool but inconsistent agent)
            for agent_name, agent_data in learned_context['agent_scores'].items():
                consistency = agent_data.get('consistency', 0.5)
                if consistency < 0.5 and not any(r.get('agent') == agent_name for r in recommendations):
                    recommendations.append({
                        'priority': 'LOW',
                        'type': 'agent_only',
                        'agent': agent_name,
                        'consistency': consistency,
                        'action': f"Warn {agent_name}: outputs inconsistent (consistency={consistency:.2f})"
                    })

            learned_context['recommendations'] = recommendations

            # 6. Retrieve expert domain knowledge from SwarmMemory
            expert_knowledge = self._retrieve_expert_knowledge()
            learned_context['expert_knowledge'] = expert_knowledge

            # 7. Analyze prior failures for recovery
            prior_failures = self._analyze_prior_failures()
            learned_context['prior_failures'] = prior_failures

            # 8. Analyze morph score trends (improving vs declining)
            score_trends = {}
            if si and si.morph_score_history and len(si.morph_score_history) >= 2:
                latest = si.morph_score_history[-1].get('scores', {})
                # Compare with 3 runs ago (or earliest available)
                compare_idx = max(0, len(si.morph_score_history) - 4)
                earlier = si.morph_score_history[compare_idx].get('scores', {})
                for agent_name_key in latest:
                    curr_rcs = latest[agent_name_key].get('rcs', 0)
                    prev_rcs = earlier.get(agent_name_key, {}).get('rcs', 0)
                    if prev_rcs > 0:
                        delta = curr_rcs - prev_rcs
                        if abs(delta) > 0.02:  # Only report meaningful changes
                            score_trends[agent_name_key] = {
                                'current': curr_rcs,
                                'previous': prev_rcs,
                                'delta': delta,
                                'direction': 'improving' if delta > 0 else 'declining'
                            }
            learned_context['score_trends'] = score_trends

            # 9. Coordination protocols (circuit breakers, gossip, auctions,
            #    coalitions, supervisor tree, backpressure)
            coordination = self._coordinate_pre_execution(si)
            learned_context['coordination'] = coordination

            learned_context['has_learning'] = bool(
                learned_context['tool_performance'] or
                learned_context['agent_scores'] or
                learned_context['warmup_completed'] or
                learned_context['expert_knowledge'] or
                learned_context['prior_failures'] or
                learned_context['score_trends'] or
                learned_context['coordination']
            )

            self._learned_context = learned_context
            if learned_context['has_learning']:
                expert_count = len(learned_context.get('expert_knowledge', []))
                logger.info(
                    f"Pre-execution learning: {len(learned_context['tool_performance'])} tools tracked, "
                    f"{len(learned_context['agent_scores'])} agents scored, "
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
            logger.warning(f"Pre-execution learning failed unexpectedly: {type(e).__name__}: {e}", exc_info=True)
            self._learned_context = learned_context

        return learned_context

    # =========================================================================
    # COORDINATION PROTOCOLS (Wire auctions, coalitions, gossip, etc.)
    # =========================================================================

    def _coordinate_pre_execution(self, si: Any) -> Dict[str, Any]:
        """
        Run coordination protocols before task execution.

        Wires the orphaned SwarmIntelligence coordination features into the
        actual execution lifecycle:

        1. Circuit breakers: Filter out agents with open circuits
        2. Gossip: Process pending messages so agents have latest info
        3. Supervisor tree: Build hierarchy on first run for O(log n) routing
        4. Backpressure: Check if swarm is overwhelmed
        5. Load balancing: Rebalance work if agents are overloaded

        Returns:
            Dict with coordination state for downstream use.
        """
        coord = {
            'available_agents': [],
            'circuit_blocked': [],
            'gossip_messages_processed': 0,
            'supervisor_tree_built': False,
            'backpressure': 0.0,
            'should_accept': True,
            'load_balanced': 0,
        }

        if not si:
            return coord

        try:
            # 1. Circuit breakers: get list of available agents (not blocked)
            all_agents = list(si.agent_profiles.keys())
            available = si.get_available_agents(all_agents)
            blocked = [a for a in all_agents if a not in available]
            coord['available_agents'] = available
            coord['circuit_blocked'] = blocked
            if blocked:
                logger.info(
                    f"Circuit breakers blocking {len(blocked)} agents: "
                    f"{', '.join(blocked)}"
                )

            # 2. Gossip: receive and process pending messages for all agents.
            # This ensures agents have the latest coordination info (handoffs,
            # coalition announcements, work-steal notifications, etc.) before
            # any new execution starts.
            total_gossip = 0
            for agent_name in available:
                messages = si.gossip_receive(agent_name)
                total_gossip += len(messages)
            coord['gossip_messages_processed'] = total_gossip
            if total_gossip > 0:
                logger.debug(f"Processed {total_gossip} gossip messages across agents")

            # 3. Supervisor tree: build on first run if we have enough agents.
            # This enables O(log n) hierarchical routing for large swarms.
            if not si._tree_built and len(available) >= 4:
                si.build_supervisor_tree(available)
                coord['supervisor_tree_built'] = True

            # 4. Backpressure: check if swarm is overwhelmed.
            # High backpressure means we should throttle or reject low-priority tasks.
            coord['backpressure'] = si.calculate_backpressure()
            coord['should_accept'] = si.should_accept_task(priority=5)
            if not coord['should_accept']:
                logger.warning(
                    f"Swarm backpressure high ({coord['backpressure']:.2f}), "
                    f"may throttle low-priority tasks"
                )

            # 5. Load balancing: if any agents are idle while others are
            # overloaded, redistribute pending work via work-stealing.
            try:
                actions = si.balance_load()
                coord['load_balanced'] = len(actions)
                if actions:
                    si.metrics.record_coordination('load_balance', success=True)
            except Exception as e:
                logger.debug(f"Load balancing failed: {e}")
                si.metrics.record_error('coordination', 'load_balance', str(e))

            # 6. Stigmergy evaporation: decay stale signals so the pheromone
            # landscape stays fresh. Without this, old signals persist
            # indefinitely since decay only happens lazily on sense().
            try:
                pruned = si.stigmergy.evaporate()
                coord['stigmergy_pruned'] = pruned
                coord['stigmergy_active'] = len(si.stigmergy.signals)
            except Exception as e:
                logger.debug(f"Stigmergy evaporation failed: {e}")
                si.metrics.record_error('coordination', 'stigmergy', str(e))

            # 7. Coalition formation: for swarms with 3+ available agents,
            # form a coalition for the upcoming task. This groups agents
            # into a coordinated team with a leader, enabling the
            # coalition_broadcast/dissolve lifecycle to actually trigger.
            try:
                coord['coalition_formed'] = None
                if len(available) >= 3 and not si.coalitions:
                    swarm_name = getattr(self.config, 'name', None) or 'base_swarm'
                    task_type = getattr(self.config, 'domain', 'general')
                    coalition = si.form_coalition(
                        task_type=task_type,
                        min_agents=min(2, len(available)),
                        max_agents=min(5, len(available))
                    )
                    if coalition:
                        coord['coalition_formed'] = coalition.coalition_id
                        coord['coalition_members'] = coalition.members
                        coord['coalition_leader'] = coalition.leader
                        si.metrics.record_coordination('coalition', success=True)
                        logger.info(
                            f"Coalition formed: {coalition.coalition_id} "
                            f"({len(coalition.members)} agents, leader={coalition.leader})"
                        )
            except Exception as e:
                logger.debug(f"Coalition formation failed: {e}")
                si.metrics.record_error('coordination', 'coalition', str(e))

        except (AttributeError, ImportError) as e:
            logger.debug(f"Pre-execution coordination skipped (optional dep): {e}")
        except (LearningError, ExecutionError) as e:
            logger.warning(f"Pre-execution coordination failed (recoverable): {e}")
        except Exception as e:
            logger.warning(f"Pre-execution coordination failed (unexpected): {type(e).__name__}: {e}", exc_info=True)

        return coord

    def _coordinate_post_execution(self, si: Any, success: bool, execution_time: float, tools_used: List[str], task_type: str) -> Any:
        """
        Run coordination protocols after task execution.

        Wires the orphaned SwarmIntelligence coordination features into the
        actual execution lifecycle:

        1. Byzantine verify: verify swarm-level result consistency
        2. Gossip broadcast: propagate execution outcome to all agents
        3. Coalition cleanup: dissolve any active coalition for this task
        4. Failure recovery: on failure, auto-reassign via auction
        5. Agent retirement: check if any agents should be retired

        Args:
            si: SwarmIntelligence instance
            success: Whether the overall execution succeeded
            execution_time: Total execution time
            tools_used: Tools used during execution
            task_type: Type of task executed
        """
        if not si:
            return

        swarm_name = self.config.name or 'base_swarm'

        try:
            # 1. Byzantine verify: verify the swarm-level execution result.
            # This catches cases where agents claimed intermediate success
            # but the overall result is a failure (or vice versa).
            try:
                si.byzantine.verify_claim(
                    agent=swarm_name,
                    claimed_success=success,
                    actual_result={'success': success, 'time': execution_time},
                    task_type=task_type
                )
                si.metrics.record_coordination('byzantine', agent=swarm_name, success=True)
            except Exception as e:
                logger.debug(f"Byzantine verification failed: {e}")
                si.metrics.record_error('coordination', 'byzantine', str(e))

            # 2. Gossip broadcast: propagate execution outcome so all agents
            # have awareness of what just happened. This feeds into the gossip
            # protocol for decentralized coordination.
            try:
                si.gossip_broadcast(
                    origin_agent=swarm_name,
                    message_type="execution_result",
                    content={
                        "task_type": task_type,
                        "success": success,
                        "execution_time": execution_time,
                        "tools_used": tools_used,
                    },
                    ttl=2  # Propagate through 2 hops
                )
                si.metrics.record_coordination('gossip', agent=swarm_name, success=True)
            except Exception as e:
                logger.debug(f"Gossip broadcast failed: {e}")
                si.metrics.record_error('coordination', 'gossip', str(e))

            # 3. Coalition cleanup: dissolve any active coalitions that were
            # formed for this execution. Agents become available for new coalitions.
            try:
                to_dissolve = []
                for cid, coalition in si.coalitions.items():
                    if coalition.task_type == task_type and coalition.active:
                        to_dissolve.append(cid)
                for cid in to_dissolve:
                    si.dissolve_coalition(cid)
                    si.metrics.record_coordination('coalition_dissolve', success=True)
                    logger.debug(f"Dissolved coalition {cid} after {task_type} execution")
            except Exception as e:
                logger.debug(f"Coalition cleanup failed: {e}")
                si.metrics.record_error('coordination', 'coalition_cleanup', str(e))

            # 4. Failure recovery: if the execution failed, use auction-based
            # reassignment to find an alternative agent for retry.
            if not success:
                try:
                    new_agent = si.record_failure(
                        task_id=f"{task_type}_{int(__import__('time').time())}",
                        agent=swarm_name,
                        task_type=task_type,
                        error_type='execution_failure',
                        context={'tools_used': tools_used, 'execution_time': execution_time}
                    )
                    if new_agent:
                        si.metrics.record_coordination('failure_recovery', agent=new_agent, success=True)
                        logger.info(f"Failure recovery: reassigned {task_type} to {new_agent}")
                except Exception as e:
                    logger.debug(f"Failure recovery failed: {e}")
                    si.metrics.record_error('coordination', 'failure_recovery', str(e))

            # 5. Agent retirement: check if any agents have degraded enough
            # to warrant retirement (low trust, high failure rate, circuit open).
            # Retired agents get their circuit opened permanently so they stop
            # receiving tasks until manually reset or trust recovers.
            try:
                for agent_name in list(si.agent_profiles.keys()):
                    if si.should_retire_agent(agent_name):
                        trust = si.agent_profiles[agent_name].trust_score
                        logger.warning(
                            f"Retiring agent {agent_name} "
                            f"(trust={trust:.2f}) — circuit opened"
                        )
                        # Actually block the agent by opening its circuit
                        si.record_circuit_failure(agent_name, threshold=1)
                        si.metrics.record_coordination(
                            'retirement', agent=agent_name, success=True
                        )
            except Exception as e:
                logger.debug(f"Retirement check failed: {e}")
                si.metrics.record_error('coordination', 'retirement_check', str(e))

            # 6. Stigmergy evaporation: clean up stale signals post-execution.
            try:
                si.stigmergy.evaporate()
            except Exception as e:
                logger.debug(f"Post-execution stigmergy evaporation failed: {e}")
                si.metrics.record_error('coordination', 'stigmergy_evaporate', str(e))

        except (AttributeError, ImportError) as e:
            logger.debug(f"Post-execution coordination skipped (optional dep): {e}")
        except (LearningError, ExecutionError) as e:
            logger.warning(f"Post-execution coordination failed (recoverable): {e}")
            try:
                si.metrics.record_error('coordination', 'post_execution', str(e))
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Post-execution coordination failed (unexpected): {type(e).__name__}: {e}", exc_info=True)
            try:
                si.metrics.record_error('coordination', 'post_execution', str(e))
            except Exception:
                pass  # Last resort: metrics system itself is broken

    def _build_learned_context_string(self, agent_name: str = None) -> str:
        """
        Convert self._learned_context into injectable prompt text.

        Produces a compact string suitable for appending to DSPy agent inputs,
        so agents are aware of prior tool performance, agent consistency, and
        priority actions.

        Args:
            agent_name: Optional specific agent to tailor context for

        Returns:
            String like:
            '## Prior Learning
            Tool Performance: arxiv_fetch 100% RELIABLE, content_generate 45% WEAK
            Agent Notes: ContentPolisher has inconsistent outputs (consistency=0.3)
            Action: Validate content_generate output carefully before using.'
        """
        if not self._learned_context or not self._learned_context.get('has_learning'):
            return ""

        ctx = self._learned_context
        lines = ["## Prior Learning"]

        # Tool performance summary
        tool_parts = []
        for tool_info in ctx.get('strong_tools', []):
            tool_parts.append(f"{tool_info.get('tool', '?')} {tool_info.get('success_rate', 0):.0%} RELIABLE")
        for tool_info in ctx.get('weak_tools', []):
            tool_parts.append(f"{tool_info.get('tool', '?')} {tool_info.get('success_rate', 0):.0%} WEAK")

        if tool_parts:
            lines.append(f"Tool Performance: {', '.join(tool_parts)}")

        # Agent-specific context (both positive reinforcement and warnings)
        agent_notes = []
        scores = ctx.get('agent_scores', {})
        if agent_name and agent_name in scores:
            agent_data = scores[agent_name]
            rcs = agent_data.get('rcs', 0)
            consistency = agent_data.get('consistency', 0.5)
            focus = agent_data.get('focus', 0.5)
            total_tasks = agent_data.get('total_tasks', 0)
            # Competence and focus feedback require enough history to be meaningful
            if total_tasks >= 2:
                # Tiered competence feedback — always push for higher
                if rcs >= 0.85:
                    agent_notes.append(
                        f"Competence {rcs:.2f} — excellent, maintain this standard"
                    )
                elif rcs >= 0.6:
                    agent_notes.append(
                        f"Competence {rcs:.2f} — good but target >0.85, push harder on quality"
                    )
                elif rcs >= 0.4:
                    agent_notes.append(
                        f"Competence {rcs:.2f} — needs improvement, aim for >0.6"
                    )
                elif rcs > 0:
                    agent_notes.append(
                        f"Competence {rcs:.2f} — critical, significant quality issues"
                    )
                # Focus feedback
                if focus >= 0.85:
                    agent_notes.append("Focus is excellent — stay specialized")
                elif focus >= 0.6:
                    agent_notes.append(f"Focus {focus:.2f} — good but tighten specialization")
                elif focus > 0 and focus < 0.4:
                    agent_notes.append(f"Focus {focus:.2f} — too scattered, narrow your scope")
            # Consistency warnings stay unguarded (always useful even for new agents)
            if consistency < 0.5:
                agent_notes.append(
                    f"Consistency {consistency:.2f} — outputs vary too much, "
                    f"be extra careful with accuracy"
                )
        elif scores:
            # Summary for orchestrator or unmatched agents
            high_performers = []
            needs_improvement = []
            low_performers = []
            for name, agent_data in scores.items():
                rcs = agent_data.get('rcs', 0)
                consistency = agent_data.get('consistency', 0.5)
                if rcs >= 0.85:
                    high_performers.append(name)
                elif rcs < 0.5 and rcs > 0:
                    needs_improvement.append(f"{name}({rcs:.2f})")
                if consistency < 0.5:
                    low_performers.append(
                        f"{name} inconsistent ({consistency:.2f})"
                    )
            if high_performers:
                agent_notes.append(f"Strong agents: {', '.join(high_performers)}")
            if needs_improvement:
                agent_notes.append(f"Need improvement: {', '.join(needs_improvement)}")
            if low_performers:
                agent_notes.extend(low_performers)

        # Specialization label from AgentProfile
        si = self._swarm_intelligence
        if si and agent_name and agent_name in getattr(si, 'agent_profiles', {}):
            from ..orchestration.swarm_intelligence import AgentSpecialization
            profile = si.agent_profiles[agent_name]
            spec = profile.specialization
            if spec != AgentSpecialization.GENERALIST:
                agent_notes.append(f"Specialization: {spec.value} — leverage this strength")

            # Per-agent time budget from profile
            if profile.avg_execution_time > 0 and profile.total_tasks >= 2:
                avg_t = profile.avg_execution_time
                agent_notes.append(f"Avg execution: {avg_t:.0f}s over {profile.total_tasks} tasks")

        if agent_notes:
            lines.append(f"Agent Notes: {'; '.join(agent_notes)}")

        # Execution patterns from collective memory (what works, typical timings)
        if si and si.collective_memory:
            recent = list(si.collective_memory)[-20:]
            successes = [m for m in recent if m.get('success')]
            if successes:
                # Build timing expectations per task type
                from collections import defaultdict
                task_times = defaultdict(list)
                for m in successes:
                    tt = m.get('task_type', '')
                    if tt and m.get('execution_time', 0) > 0:
                        task_times[tt].append(m['execution_time'])
                if task_times:
                    timing_parts = []
                    for tt, times in task_times.items():
                        avg = sum(times) / len(times)
                        timing_parts.append(f"{tt}~{avg:.0f}s")
                    if len(timing_parts) <= 6:
                        lines.append(f"Typical timing: {', '.join(timing_parts)}")
                # Success streak info
                total_recent = len(recent)
                success_rate = len(successes) / total_recent if total_recent else 0
                if success_rate >= 0.9 and total_recent >= 5:
                    lines.append(
                        f"Track record: {len(successes)}/{total_recent} recent successes — "
                        f"maintain this standard"
                    )

        # Coordination protocol status (circuit breakers, backpressure, etc.)
        coord = ctx.get('coordination', {})
        if coord:
            coord_parts = []
            blocked = coord.get('circuit_blocked', [])
            if blocked:
                coord_parts.append(f"Circuit-blocked agents: {', '.join(blocked)}")
            backpressure = coord.get('backpressure', 0)
            if backpressure > 0.5:
                coord_parts.append(f"Backpressure: {backpressure:.0%} (high — prioritize efficiency)")
            load_balanced = coord.get('load_balanced', 0)
            if load_balanced > 0:
                coord_parts.append(f"{load_balanced} tasks rebalanced via work-stealing")
            if coord_parts:
                lines.append(f"Swarm Status: {'; '.join(coord_parts)}")

        # Priority recommendations (HIGH first)
        recommendations = ctx.get('recommendations', [])
        high_priority = [r for r in recommendations if r.get('priority') == 'HIGH']
        if high_priority:
            actions = [r['action'] for r in high_priority[:3]]
            lines.append(f"Action: {'; '.join(actions)}")
        elif recommendations:
            lines.append(f"Action: {recommendations[0]['action']}")

        # Evaluation quality bar from persistent history
        if hasattr(self, '_evaluation_history'):
            avg_score = self._evaluation_history.get_average_score(5)
            eval_count = len(self._evaluation_history.evaluations)
            if eval_count >= 2 and avg_score > 0:
                if avg_score >= 0.9:
                    lines.append(
                        f"Quality bar: avg {avg_score:.0%} over {eval_count} evaluations — "
                        f"excellent standard, don't regress"
                    )
                elif avg_score >= 0.7:
                    lines.append(
                        f"Quality bar: avg {avg_score:.0%} over {eval_count} evaluations — "
                        f"good but push for higher"
                    )
                else:
                    lines.append(
                        f"Quality bar: avg {avg_score:.0%} over {eval_count} evaluations — "
                        f"needs significant improvement"
                    )

        # Improvement suggestions from prior cycles (what to improve + what worked)
        if hasattr(self, '_improvement_history') and self._improvement_history:
            pending = self._improvement_history.get_pending_suggestions()
            successful = self._improvement_history.get_successful_improvements()
            if pending or successful:
                imp_lines = []
                # Show successful improvements so agents know what works
                for s in successful[-3:]:
                    suggestion = s.get('suggestion', {})
                    desc = suggestion.get('description', '')
                    if desc:
                        imp_lines.append(f"- Applied successfully: {desc[:120]}")
                # Show pending improvements as directives
                for p in pending[-3:]:
                    suggestion = p.get('suggestion', {})
                    desc = suggestion.get('description', '')
                    priority = suggestion.get('priority', 'MEDIUM')
                    target_agent = suggestion.get('agent_role', '')
                    # Only show agent-specific improvements to that agent
                    if agent_name and target_agent and target_agent != agent_name:
                        continue
                    if desc:
                        imp_lines.append(f"- [{priority}] TASK: {desc[:120]}")
                if imp_lines:
                    lines.append("## Improvement Directives")
                    lines.extend(imp_lines)

        # Expert domain knowledge from SwarmMemory
        expert_knowledge = ctx.get('expert_knowledge', [])
        if expert_knowledge:
            expert_lines = []
            for imp in expert_knowledge[:5]:  # Top 5 patterns
                pattern = imp.get('learned_pattern', '')
                if pattern:
                    # Truncate long patterns for prompt efficiency
                    if len(pattern) > 150:
                        pattern = pattern[:147] + "..."
                    expert_lines.append(f"- {pattern}")

            if expert_lines:
                lines.append("## Expert Knowledge")
                lines.extend(expert_lines)

        # Failure recovery from prior runs
        prior_failures = ctx.get('prior_failures', [])
        if prior_failures:
            failure_lines = ["## Prior Failures (Avoid Repeating)"]
            for f in prior_failures[:3]:
                if f.get('source') == 'evaluation':
                    feedback = f.get('feedback', '')
                    if feedback:
                        failure_lines.append(f"- Previous run scored {f.get('score', 0):.0%}: {feedback[:100]}")
                elif f.get('source') == 'collective_memory':
                    failure_lines.append(
                        f"- Agent {f.get('agent', '?')} failed task {f.get('task_type', '?')}"
                    )
                elif f.get('source') == 'memory':
                    failure_lines.append(f"- {f.get('pattern', 'unknown failure')}")
            if len(failure_lines) > 1:
                lines.extend(failure_lines)

        # Morph score trends — show improvement/decline direction
        score_trends = ctx.get('score_trends', {})
        if score_trends:
            trend_lines = []
            for trend_agent, trend_data in score_trends.items():
                # Show trend for this specific agent or all agents for orchestrator
                if agent_name and trend_agent != agent_name:
                    continue
                delta = trend_data['delta']
                direction = trend_data['direction']
                current = trend_data['current']
                if direction == 'improving':
                    trend_lines.append(
                        f"{trend_agent}: {current:.2f} (+{delta:.2f}) improving — keep pushing"
                    )
                else:
                    trend_lines.append(
                        f"{trend_agent}: {current:.2f} ({delta:.2f}) declining — investigate and fix"
                    )
            if not agent_name:
                # Orchestrator sees all trends
                for trend_agent, trend_data in score_trends.items():
                    delta = trend_data['delta']
                    current = trend_data['current']
                    direction = trend_data['direction']
                    if direction == 'improving':
                        trend_lines.append(
                            f"{trend_agent}: {current:.2f} (+{delta:.2f}) improving"
                        )
                    else:
                        trend_lines.append(
                            f"{trend_agent}: {current:.2f} ({delta:.2f}) DECLINING"
                        )
            if trend_lines:
                lines.append("Score trends: " + "; ".join(trend_lines))

        return "\n".join(lines) if len(lines) > 1 else ""

    @classmethod
    def test_learning_pathways(cls) -> Dict[str, Dict[str, Any]]:
        """Diagnostic: inject synthetic data to verify all 5 learning pathways produce prompt text."""
        import tempfile
        results = {}

        # Create concrete subclass to bypass ABC restriction
        class _TestSwarm(cls):
            async def execute(self, *args: Any, **kwargs: Any) -> Any:
                pass

        # Create minimal swarm instance for testing (no disk I/O beyond tempdir)
        config = SwarmConfig(name='pathway_tester', enable_self_improvement=True)
        instance = _TestSwarm.__new__(_TestSwarm)
        instance.config = config
        instance._swarm_intelligence = None
        instance._memory = None
        instance._learned_context = None
        tmp = tempfile.mkdtemp()
        instance._evaluation_history = EvaluationHistory(path=tmp + '/eval')
        instance._improvement_history = ImprovementHistory(path=tmp + '/imp')

        # === Pathway 1: weak_tools ===
        instance._learned_context = {
            'has_learning': True,
            'tool_performance': {'bad_tool': 0.3},
            'agent_scores': {},
            'weak_tools': [{'tool': 'bad_tool', 'success_rate': 0.3, 'total': 5}],
            'strong_tools': [{'tool': 'good_tool', 'success_rate': 0.95, 'total': 10}],
            'recommendations': [],
            'warmup_completed': True,
            'expert_knowledge': [],
            'prior_failures': [],
            'score_trends': {},
        }
        text = instance._build_learned_context_string()
        results['weak_tools'] = {
            'triggered': 'WEAK' in text,
            'prompt_snippet': text[:200] if text else '(empty)',
        }

        # === Pathway 2: expert_knowledge ===
        instance._learned_context['expert_knowledge'] = [
            {'learned_pattern': 'Always validate API responses before processing'},
            {'learned_pattern': 'Use batch processing for >100 items'},
        ]
        text = instance._build_learned_context_string()
        results['expert_knowledge'] = {
            'triggered': 'Expert Knowledge' in text,
            'prompt_snippet': text[text.find('Expert'):text.find('Expert') + 150] if 'Expert' in text else '(empty)',
        }

        # === Pathway 3: prior_failures ===
        instance._learned_context['prior_failures'] = [
            {'source': 'evaluation', 'score': 0.3, 'feedback': 'Missing key concepts', 'timestamp': datetime.now().isoformat()},
            {'source': 'collective_memory', 'agent': 'ConceptExtractor', 'task_type': 'expert', 'timestamp': datetime.now().isoformat()},
        ]
        text = instance._build_learned_context_string()
        results['prior_failures'] = {
            'triggered': 'Prior Failures' in text,
            'prompt_snippet': text[text.find('Prior Failures'):text.find('Prior Failures') + 200] if 'Prior Failures' in text else '(empty)',
        }

        # === Pathway 4: improvement_directives ===
        # ImprovementHistory uses self.history (list of dicts)
        # get_pending_suggestions() checks status == 'pending'
        # get_successful_improvements() checks outcome == 'success'
        instance._improvement_history.history = [
            {
                'id': 'test_pending_1',
                'suggestion': {'description': 'Improve concept extraction depth', 'priority': 5, 'agent_role': ''},
                'status': 'pending',
                'outcome': None,
            },
            {
                'id': 'test_success_1',
                'suggestion': {'description': 'Use more examples in explanations', 'priority': 3, 'agent_role': ''},
                'status': 'completed',
                'outcome': 'success',
            },
        ]
        text = instance._build_learned_context_string()
        results['improvement_directives'] = {
            'triggered': 'Improvement Directives' in text,
            'prompt_snippet': text[text.find('Improvement'):text.find('Improvement') + 200] if 'Improvement' in text else '(empty)',
        }

        # === Pathway 5: recommendations ===
        instance._learned_context['recommendations'] = [
            {'priority': 'HIGH', 'type': 'tool_and_agent', 'tool': 'bad_tool', 'tool_rate': 0.3,
             'agent': 'SlowAgent', 'consistency': 0.3,
             'action': 'PRIORITY: Replace bad_tool (30% success) AND stabilize SlowAgent (consistency=0.30)'}
        ]
        text = instance._build_learned_context_string()
        results['recommendations'] = {
            'triggered': 'Action:' in text and 'PRIORITY' in text,
            'prompt_snippet': text[text.find('Action:'):text.find('Action:') + 150] if 'Action:' in text else '(empty)',
        }

        # === Pathway 6: new_agent_no_misleading_rcs ===
        instance._learned_context['agent_scores'] = {
            'BrandNewAgent': {
                'rcs': 0.5, 'rds': 0.5, 'tras': 0.5,
                'consistency': 0.5, 'focus': 0.5, 'specialization': 0.5,
                'total_tasks': 0,
            }
        }
        text = instance._build_learned_context_string(agent_name='BrandNewAgent')
        results['new_agent_no_misleading_rcs'] = {
            'triggered': 'needs improvement' not in text,
            'prompt_snippet': text[:200] if text else '(no misleading feedback — correct)',
        }

        # Summary
        all_passed = all(r['triggered'] for r in results.values())
        results['_summary'] = {
            'total': 6,
            'passed': sum(1 for r in results.values() if isinstance(r, dict) and r.get('triggered')),
            'all_passed': all_passed,
        }

        return results

    def _retrieve_expert_knowledge(self) -> List[Dict[str, Any]]:
        """
        Retrieve expert-learned domain patterns from SwarmMemory.

        Queries SwarmMemory for improvements stored by BaseExpert agents
        (via memory_integration.py). Returns patterns relevant to this swarm's
        domain so they can be injected into DSPy agent prompts.

        Returns:
            List of expert improvement dicts with 'learned_pattern', 'domain',
            'source', etc. Empty list if memory unavailable.
        """
        if not self._memory:
            # Try initializing shared resources to get memory
            if not self._initialized:
                self._init_shared_resources()
            if not self._memory:
                return []

        domain = getattr(self.config, 'domain', None) or self.config.name or 'general'
        swarm_name = self.config.name or 'base_swarm'

        try:
            from ..foundation.data_structures import MemoryLevel

            # Primary: domain-scoped retrieval (key-prefix filtering)
            memory_entries = self._memory.retrieve_by_domain(
                domain=domain,
                goal=f"expert_{domain}_improvements",
                budget_tokens=5000,
                levels=[MemoryLevel.PROCEDURAL, MemoryLevel.META, MemoryLevel.SEMANTIC]
            )

            if not memory_entries:
                # Fallback: context-aware retrieval prioritizing wisdom (META > SEMANTIC)
                memory_entries = self._memory.retrieve_for_context(
                    query=f"expert improvements for {swarm_name}",
                    goal=f"expert_{domain}_improvements",
                    context_type="planning",
                    budget_tokens=3000,
                )

            improvements = []
            for entry in memory_entries[:10]:  # Cap at 10 patterns
                try:
                    improvement_data = json.loads(entry.content)
                    if isinstance(improvement_data, dict):
                        improvements.append(improvement_data)
                    elif isinstance(improvement_data, list):
                        improvements.extend(improvement_data[:5])
                except (json.JSONDecodeError, TypeError):
                    # Raw text pattern from consolidation
                    if entry.content and len(entry.content) > 10:
                        improvements.append({
                            'learned_pattern': entry.content,
                            'domain': domain,
                            'source': 'expert_memory',
                            'memory_level': entry.level.value if hasattr(entry, 'level') else 'unknown',
                        })

            if improvements:
                logger.info(f"Retrieved {len(improvements)} expert patterns from memory for domain '{domain}'")

            return improvements

        except (MemoryRetrievalError, MemoryError) as e:
            logger.warning(f"Expert knowledge retrieval failed (memory): {e}")
            return []
        except Exception as e:
            logger.debug(f"Expert knowledge retrieval skipped (unexpected): {e}")
            return []

    def _analyze_prior_failures(self) -> List[Dict[str, Any]]:
        """Analyze prior execution failures from collective_memory and evaluation history.
        Returns list of failure patterns with avoidance suggestions."""
        failures = []

        # Source 1: Evaluation history failures
        if hasattr(self, '_evaluation_history'):
            eval_failures = self._evaluation_history.get_failures(20)
            for f in eval_failures[-5:]:  # Last 5 failures
                failures.append({
                    'source': 'evaluation',
                    'score': f.get('overall_score', 0),
                    'feedback': f.get('feedback', ''),
                    'timestamp': f.get('timestamp', ''),
                })

        # Source 2: Collective memory from SwarmIntelligence
        si = self._swarm_intelligence
        if si and si.collective_memory:
            failed_tasks = [
                m for m in list(si.collective_memory)[-50:]
                if not m.get('success', True)
            ]
            for m in failed_tasks[-5:]:
                failures.append({
                    'source': 'collective_memory',
                    'agent': m.get('agent', 'unknown'),
                    'task_type': m.get('task_type', 'unknown'),
                    'timestamp': m.get('timestamp', ''),
                })

        # Source 3: Execution traces stored in memory
        if self._memory:
            try:
                from ..foundation.data_structures import MemoryLevel
                failure_entries = self._memory.retrieve(
                    query=f"failed execution error {self.config.name or 'swarm'}",
                    goal="failure_analysis",
                    budget_tokens=2000,
                    levels=[MemoryLevel.META]
                )
                for entry in failure_entries[:3]:
                    try:
                        data = json.loads(entry.content)
                        if isinstance(data, dict) and not data.get('success', True):
                            failures.append({
                                'source': 'memory',
                                'pattern': data.get('learned_pattern', entry.content[:100]),
                            })
                    except (json.JSONDecodeError, TypeError):
                        pass
            except (MemoryRetrievalError, MemoryError) as e:
                logger.warning(f"Failure analysis from memory failed (memory): {e}")
            except Exception as e:
                logger.debug(f"Failure analysis from memory failed (unexpected): {e}")

        return failures

    def _store_execution_as_improvement(self, success: bool, execution_time: float, tools_used: List[str], task_type: str) -> Any:
        """
        Store execution outcome as an expert improvement in SwarmMemory.

        This bridges swarm execution results back into the expert memory system,
        so future expert training and swarm executions can learn from outcomes.

        Args:
            success: Whether execution succeeded
            execution_time: Total execution time in seconds
            tools_used: List of tool names used
            task_type: Type of task executed
        """
        if not self._memory:
            return

        domain = getattr(self.config, 'domain', None) or self.config.name or 'general'
        swarm_name = self.config.name or 'base_swarm'

        try:
            from ..foundation.data_structures import MemoryLevel

            # Build improvement from execution outcome
            if success:
                pattern = (
                    f"Successful {task_type} execution by {swarm_name}: "
                    f"tools [{', '.join(tools_used)}] completed in {execution_time:.1f}s"
                )
                level = MemoryLevel.PROCEDURAL
            else:
                pattern = (
                    f"Failed {task_type} execution by {swarm_name}: "
                    f"tools [{', '.join(tools_used)}] failed after {execution_time:.1f}s — "
                    f"consider alternative approach or tool substitution"
                )
                level = MemoryLevel.META  # Failures are learning wisdom

            improvement = {
                'timestamp': datetime.now().isoformat(),
                'task': task_type,
                'learned_pattern': pattern,
                'improvement_type': 'execution_outcome',
                'source': f'swarm_{swarm_name}',
                'success': success,
                'execution_time': execution_time,
                'tools_used': tools_used,
            }

            context = {
                'expert_name': swarm_name,
                'domain': domain,
                'task': task_type,
                'improvement_type': 'execution_outcome',
                'source': 'swarm_lifecycle',
            }

            self._memory.store(
                content=json.dumps(improvement, ensure_ascii=False),
                level=level,
                context=context,
                goal=f"expert_{domain}_improvements",
                initial_value=0.8 if success else 1.0,  # Failures are more valuable for learning
            )

            logger.debug(f"Stored execution outcome to expert memory: {task_type} {'success' if success else 'failure'}")

        except (MemoryStorageError, MemoryError) as e:
            logger.warning(f"Failed to store execution improvement (memory): {e}")
        except Exception as e:
            logger.debug(f"Failed to store execution improvement (unexpected): {e}")

    async def _post_execute_learning(self, success: bool, execution_time: float, tools_used: List[str], task_type: str, output_data: Dict[str, Any] = None, input_data: Dict[str, Any] = None) -> Any:
        """
        Post-execution learning hook. Called at end of execute().

        Sends executor feedback, recomputes MorphAgent scores, re-analyzes
        tools, evaluates output, runs improvement cycle, and saves all state.

        Args:
            success: Whether execution succeeded
            execution_time: Total execution time in seconds
            tools_used: List of tool names used during execution
            task_type: Type of task that was executed
            output_data: Optional dict of output metrics for evaluation
            input_data: Optional dict of input params for evaluation matching
        """
        _learning_errors = []  # Track failures for observability
        try:
            # 1. Send executor feedback (tools, success, timing)
            self._send_executor_feedback(
                task_type=task_type,
                success=success,
                tools_used=tools_used,
                execution_time=execution_time,
                error_type=None if success else 'execution_failure'
            )

            si = self._swarm_intelligence
            if not si:
                return

            # 2. Recompute MorphAgent scores with new data
            if si.agent_profiles:
                morph_scores = si.morph_scorer.compute_all_scores(si.agent_profiles)
                si.morph_score_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'scores': {
                        name: {'rcs': s.rcs, 'rds': s.rds, 'tras': s.tras}
                        for name, s in morph_scores.items()
                    }
                })
                # Note: morph_score_history is a deque(maxlen=100), self-bounding.
                # No manual truncation needed.

            # 3. Re-analyze tools and update assignments
            self._manage_tools()

            # 3a. Post-execution coordination protocols (byzantine verify,
            #     circuit breakers, gossip broadcast, coalition cleanup,
            #     load balancing, failure recovery)
            self._coordinate_post_execution(
                si=si,
                success=success,
                execution_time=execution_time,
                tools_used=tools_used,
                task_type=task_type
            )

            # 4. Evaluate output against gold standard (centralized for all swarms)
            evaluation = None
            if success and output_data and self.config.enable_self_improvement:
                try:
                    evaluation = await self._evaluate_output(
                        output=output_data,
                        task_type=task_type,
                        input_data=input_data or {}
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
            if evaluation and self._auditor and output_data:
                try:
                    audit_result = await self._auditor.audit_evaluation(
                        evaluation={'scores': evaluation.scores,
                                    'overall_score': evaluation.overall_score,
                                    'result': evaluation.result.value,
                                    'feedback': evaluation.feedback},
                        output_data=output_data,
                        context=json.dumps({'task_type': task_type})
                    )
                    if not audit_result.get('passed', True):
                        logger.warning(
                            f"Audit failed for evaluation: {audit_result.get('reasoning', 'unknown')}"
                        )
                except Exception:
                    pass  # Non-blocking

            # 4b. Record iteration in benchmarks (always, not just when evaluation exists)
            if si:
                try:
                    score = evaluation.overall_score if evaluation else (1.0 if success else 0.0)
                    si.benchmarks.record_iteration(
                        iteration_id=f"{task_type}_{int(__import__('time').time())}",
                        task_type=task_type,
                        score=score,
                        execution_time=execution_time,
                        success=success
                    )
                except LearningError as e:
                    logger.warning(f"Benchmark recording failed (learning): {e}")
                except Exception as e:
                    logger.debug(f"Benchmark recording failed (unexpected): {e}")

            # 4c. Auto-curate gold standard from excellent outputs
            if (evaluation and evaluation.overall_score >= 0.9 and
                evaluation.result in (EvaluationResult.EXCELLENT, EvaluationResult.GOOD) and
                self._gold_db and output_data and input_data):
                try:
                    self._curate_gold_standard(task_type, input_data, output_data, evaluation)
                except (LearningError, MemoryStorageError) as e:
                    logger.warning(f"Gold standard curation failed (recoverable): {e}")
                except Exception as e:
                    logger.debug(f"Gold standard curation failed (unexpected): {e}")

            # 4d. Extract learnings from excellent executions
            if (evaluation and evaluation.overall_score >= 0.9 and
                self._learner and output_data and input_data):
                try:
                    learnings = await self._learner.extract_learnings(
                        input_data=input_data,
                        output_data=output_data,
                        evaluation={'scores': evaluation.scores,
                                    'overall_score': evaluation.overall_score,
                                    'feedback': evaluation.feedback},
                        domain=self.config.domain
                    )
                    if learnings and self._improvement_history:
                        now = datetime.now().isoformat()
                        for learning in learnings:
                            suggestion = ImprovementSuggestion(
                                agent_role=AgentRole.ACTOR,
                                improvement_type=ImprovementType.TRAINING_DATA,
                                description=learning,
                                priority=3,
                                expected_impact=0.5,
                                implementation_details={'source': 'learner_extraction'},
                                based_on_evaluations=[evaluation.gold_standard_id]
                            )
                            sid = hashlib.md5(
                                f"{suggestion.agent_role.value}:{suggestion.description}:{now}".encode()
                            ).hexdigest()[:12]
                            self._improvement_history.history.append({
                                'id': sid,
                                'suggestion': asdict(suggestion),
                                'status': 'completed',
                                'created_at': now,
                                'applied_at': now,
                                'outcome': 'success',
                                'impact_measured': 0.5,
                                'notes': 'Auto-extracted from excellent execution',
                            })
                        self._improvement_history._save_history()
                        logger.info(f"Extracted {len(learnings)} learnings from excellent execution")
                except (LearningError, LLMError) as e:
                    logger.warning(f"Learning extraction failed (recoverable): {e}")
                except Exception as e:
                    logger.debug(f"Learning extraction failed (unexpected): {e}")

            # 5. Run improvement cycle if evaluation below threshold
            if evaluation and evaluation.overall_score < self.config.improvement_threshold:
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
                save_path = self._get_intelligence_save_path()
                si.save(save_path)
                logger.debug(f"Post-execution learning state saved to {save_path}")
            except (OSError, IOError) as save_err:
                logger.warning(f"Failed to save post-execution state (I/O): {save_err}")
            except Exception as save_err:
                logger.warning(f"Failed to save post-execution state: {type(save_err).__name__}: {save_err}")

            # 7. Store execution outcome as expert improvement in SwarmMemory
            self._store_execution_as_improvement(
                success=success,
                execution_time=execution_time,
                tools_used=tools_used,
                task_type=task_type
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
                exc_info=True
            )

    async def _run_improvement_cycle(self) -> List[ImprovementSuggestion]:
        """Run the self-improvement cycle."""
        if not self.config.enable_self_improvement or not self._reviewer:
            return []

        # Check if improvement is needed (persistent across sessions)
        recent_evals = self._evaluation_history.get_recent(10)
        avg_score = self._evaluation_history.get_average_score(10)
        if not recent_evals or avg_score >= self.config.improvement_threshold:
            logger.info(f"Performance good ({avg_score:.2f}), skipping improvement cycle")
            return []

        # Get suggestions from reviewer
        agent_configs = {
            AgentRole.EXPERT: self._expert.config if self._expert else None,
            AgentRole.REVIEWER: self._reviewer.config if self._reviewer else None,
            AgentRole.PLANNER: self._planner.config if self._planner else None,
            AgentRole.ACTOR: self._actor.config if self._actor else None,
            AgentRole.AUDITOR: self._auditor.config if self._auditor else None,
            AgentRole.LEARNER: self._learner.config if self._learner else None,
        }
        agent_configs = {k: v for k, v in agent_configs.items() if v}

        suggestions = await self._reviewer.analyze_and_suggest(
            recent_evals,
            agent_configs
        )

        # Record suggestions
        for suggestion in suggestions:
            self._improvement_history.record_suggestion(suggestion)

        return suggestions

    def _record_trace(self, agent_name: str, agent_role: AgentRole, input_data: Dict[str, Any], output_data: Dict[str, Any], execution_time: float, success: bool, error: Optional[str] = None, tools_used: List[str] = None) -> Any:
        """Record execution trace for learning and Agent0 feedback."""
        trace = ExecutionTrace(
            agent_name=agent_name,
            agent_role=agent_role,
            input_data=input_data,
            output_data=output_data,
            execution_time=execution_time,
            success=success,
            error=error
        )
        self._traces.append(trace)

        # Fire TUI trace callback if active
        try:
            from .coding_swarm import utils as _cu
            if _cu._active_trace_callback is not None:
                _cu._active_trace_callback({
                    "agent": agent_name,
                    "role": agent_role.value if agent_role else "",
                    "time": execution_time,
                    "success": success,
                    "error": error,
                    "output_summary": str(output_data)[:100] if output_data else "",
                })
        except Exception as e:
            logger.debug(f"Trace recording failed: {e}")

        # Agent0: Per-phase swarm-level feedback removed — swarm-level recording
        # is handled once by _post_execute_learning() at end of execute().
        # Only per-agent recording happens here (below).

        # MorphAgent: Update agent profile for per-agent tracking
        swarm_name = self.config.name or 'base_swarm'
        si = self._swarm_intelligence
        if si and hasattr(si, 'agent_profiles'):
            si.register_agent(agent_name)
            # Record task result under individual agent name (not swarm name)
            # so per-agent profiles accumulate real task_success data
            if agent_name != swarm_name:
                task_type_label = agent_role.value if agent_role else 'unknown'
                si.record_task_result(
                    agent_name=agent_name,
                    task_type=task_type_label,
                    success=success,
                    execution_time=execution_time
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
                    task_type=task_type_label
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

        # Store in memory for learning
        if self._memory and self.config.enable_learning:
            try:
                from ..foundation.data_structures import MemoryLevel
                self._memory.store(
                    content=json.dumps(asdict(trace), default=str),
                    level=MemoryLevel.EPISODIC,
                    context={'swarm': self.config.name, 'agent': agent_name},
                    goal=f"Execution trace: {agent_name}"
                )
            except (MemoryStorageError, MemoryError) as e:
                logger.warning(f"Failed to store trace in memory (memory): {e}")
            except Exception as e:
                logger.debug(f"Failed to store trace in memory (unexpected): {e}")

    def record_improvement_outcome(self, suggestion_id: str, success: bool, impact: float, notes: str = '') -> Any:
        """Record the outcome of an applied improvement."""
        if self._improvement_history:
            self._improvement_history.record_outcome(suggestion_id, success, impact, notes)



# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'AgentRole',
    'EvaluationResult',
    'ImprovementType',

    # Data classes
    'GoldStandard',
    'Evaluation',
    'ImprovementSuggestion',
    'AgentConfig',
    'ExecutionTrace',
    'SwarmConfig',
    'SwarmResult',

    # DSPy Signatures
    'ExpertEvaluationSignature',
    'ReviewerAnalysisSignature',
    'PlannerOptimizationSignature',
    'ActorExecutionSignature',
    'AuditorVerificationSignature',
    'LearnerExtractionSignature',

    # Core classes
    'GoldStandardDB',
    'ImprovementHistory',
    'ExpertAgent',
    'ReviewerAgent',
    'PlannerAgent',
    'ActorAgent',
    'AuditorAgent',
    'LearnerAgent',
    'BaseSwarm',
    'SwarmRegistry',
    'register_swarm',
]

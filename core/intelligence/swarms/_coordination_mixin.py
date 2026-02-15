"""
Coordination mixin for BaseSwarm.

Contains coordination protocol methods: pre/post-execution wiring of
circuit breakers, gossip, supervisor tree, backpressure, load balancing,
stigmergy, coalitions, byzantine verification, and failure recovery.

Extracted from _learning_mixin.py to separate coordination concerns
from learning and knowledge concerns.
"""

import logging
from typing import Any, Dict, List, Optional

from Jotty.core.infrastructure.foundation.exceptions import ExecutionError, LearningError

logger = logging.getLogger(__name__)


class SwarmCoordinationMixin:
    """Mixin providing coordination protocol infrastructure for BaseSwarm.

    Methods in this mixin handle the coordination lifecycle:
    - Pre-execution: circuit breakers, gossip, supervisor tree, backpressure,
      load balancing, stigmergy, coalition formation
    - Post-execution: byzantine verify, gossip broadcast, coalition cleanup,
      failure recovery, agent retirement, stigmergy evaporation

    Expects to be mixed into BaseSwarm which provides:
    - self.config with .name, .domain attributes
    - self._swarm_intelligence (SwarmIntelligence instance)
    """

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
            "available_agents": [],
            "circuit_blocked": [],
            "gossip_messages_processed": 0,
            "supervisor_tree_built": False,
            "backpressure": 0.0,
            "should_accept": True,
            "load_balanced": 0,
        }

        if not si:
            return coord

        try:
            # 1. Circuit breakers: get list of available agents (not blocked)
            all_agents = list(si.agent_profiles.keys())
            available = si.get_available_agents(all_agents)
            blocked = [a for a in all_agents if a not in available]
            coord["available_agents"] = available
            coord["circuit_blocked"] = blocked
            if blocked:
                logger.info(
                    f"Circuit breakers blocking {len(blocked)} agents: " f"{', '.join(blocked)}"
                )

            # 2. Gossip: receive and process pending messages for all agents.
            # This ensures agents have the latest coordination info (handoffs,
            # coalition announcements, work-steal notifications, etc.) before
            # any new execution starts.
            total_gossip = 0
            for agent_name in available:
                messages = si.gossip_receive(agent_name)
                total_gossip += len(messages)
            coord["gossip_messages_processed"] = total_gossip
            if total_gossip > 0:
                logger.debug(f"Processed {total_gossip} gossip messages across agents")

            # 3. Supervisor tree: build on first run if we have enough agents.
            # This enables O(log n) hierarchical routing for large swarms.
            if not si._tree_built and len(available) >= 4:
                si.build_supervisor_tree(available)
                coord["supervisor_tree_built"] = True

            # 4. Backpressure: check if swarm is overwhelmed.
            # High backpressure means we should throttle or reject low-priority tasks.
            coord["backpressure"] = si.calculate_backpressure()
            coord["should_accept"] = si.should_accept_task(priority=5)
            if not coord["should_accept"]:
                logger.warning(
                    f"Swarm backpressure high ({coord['backpressure']:.2f}), "
                    f"may throttle low-priority tasks"
                )

            # 5. Load balancing: if any agents are idle while others are
            # overloaded, redistribute pending work via work-stealing.
            try:
                actions = si.balance_load()
                coord["load_balanced"] = len(actions)
                if actions:
                    si.metrics.record_coordination("load_balance", success=True)
            except Exception as e:
                logger.debug(f"Load balancing failed: {e}")
                si.metrics.record_error("coordination", "load_balance", str(e))

            # 6. Stigmergy evaporation: decay stale signals so the pheromone
            # landscape stays fresh. Without this, old signals persist
            # indefinitely since decay only happens lazily on sense().
            try:
                pruned = si.stigmergy.evaporate()
                coord["stigmergy_pruned"] = pruned
                coord["stigmergy_active"] = len(si.stigmergy.signals)
            except Exception as e:
                logger.debug(f"Stigmergy evaporation failed: {e}")
                si.metrics.record_error("coordination", "stigmergy", str(e))

            # 7. Coalition formation: for swarms with 3+ available agents,
            # form a coalition for the upcoming task. This groups agents
            # into a coordinated team with a leader, enabling the
            # coalition_broadcast/dissolve lifecycle to actually trigger.
            try:
                coord["coalition_formed"] = None
                if len(available) >= 3 and not si.coalitions:
                    swarm_name = getattr(self.config, "name", None) or "base_swarm"
                    task_type = getattr(self.config, "domain", "general")
                    coalition = si.form_coalition(
                        task_type=task_type,
                        min_agents=min(2, len(available)),
                        max_agents=min(5, len(available)),
                    )
                    if coalition:
                        coord["coalition_formed"] = coalition.coalition_id
                        coord["coalition_members"] = coalition.members
                        coord["coalition_leader"] = coalition.leader
                        si.metrics.record_coordination("coalition", success=True)
                        logger.info(
                            f"Coalition formed: {coalition.coalition_id} "
                            f"({len(coalition.members)} agents, leader={coalition.leader})"
                        )
            except Exception as e:
                logger.debug(f"Coalition formation failed: {e}")
                si.metrics.record_error("coordination", "coalition", str(e))

        except (AttributeError, ImportError) as e:
            logger.debug(f"Pre-execution coordination skipped (optional dep): {e}")
        except (LearningError, ExecutionError) as e:
            logger.warning(f"Pre-execution coordination failed (recoverable): {e}")
        except Exception as e:
            logger.warning(
                f"Pre-execution coordination failed (unexpected): {type(e).__name__}: {e}",
                exc_info=True,
            )

        return coord

    def _coordinate_post_execution(
        self, si: Any, success: bool, execution_time: float, tools_used: List[str], task_type: str
    ) -> Any:
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

        swarm_name = self.config.name or "base_swarm"

        try:
            # 1. Byzantine verify: verify the swarm-level execution result.
            # This catches cases where agents claimed intermediate success
            # but the overall result is a failure (or vice versa).
            try:
                si.byzantine.verify_claim(
                    agent=swarm_name,
                    claimed_success=success,
                    actual_result={"success": success, "time": execution_time},
                    task_type=task_type,
                )
                si.metrics.record_coordination("byzantine", agent=swarm_name, success=True)
            except Exception as e:
                logger.debug(f"Byzantine verification failed: {e}")
                si.metrics.record_error("coordination", "byzantine", str(e))

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
                    ttl=2,  # Propagate through 2 hops
                )
                si.metrics.record_coordination("gossip", agent=swarm_name, success=True)
            except Exception as e:
                logger.debug(f"Gossip broadcast failed: {e}")
                si.metrics.record_error("coordination", "gossip", str(e))

            # 3. Coalition cleanup: dissolve any active coalitions that were
            # formed for this execution. Agents become available for new coalitions.
            try:
                to_dissolve = []
                for cid, coalition in si.coalitions.items():
                    if coalition.task_type == task_type and coalition.active:
                        to_dissolve.append(cid)
                for cid in to_dissolve:
                    si.dissolve_coalition(cid)
                    si.metrics.record_coordination("coalition_dissolve", success=True)
                    logger.debug(f"Dissolved coalition {cid} after {task_type} execution")
            except Exception as e:
                logger.debug(f"Coalition cleanup failed: {e}")
                si.metrics.record_error("coordination", "coalition_cleanup", str(e))

            # 4. Failure recovery: if the execution failed, use auction-based
            # reassignment to find an alternative agent for retry.
            if not success:
                try:
                    new_agent = si.record_failure(
                        task_id=f"{task_type}_{int(__import__('time').time())}",
                        agent=swarm_name,
                        task_type=task_type,
                        error_type="execution_failure",
                        context={"tools_used": tools_used, "execution_time": execution_time},
                    )
                    if new_agent:
                        si.metrics.record_coordination(
                            "failure_recovery", agent=new_agent, success=True
                        )
                        logger.info(f"Failure recovery: reassigned {task_type} to {new_agent}")
                except Exception as e:
                    logger.debug(f"Failure recovery failed: {e}")
                    si.metrics.record_error("coordination", "failure_recovery", str(e))

            # 5. Agent retirement: check if any agents have degraded enough
            # to warrant retirement (low trust, high failure rate, circuit open).
            # Retired agents get their circuit opened permanently so they stop
            # receiving tasks until manually reset or trust recovers.
            try:
                for agent_name in list(si.agent_profiles.keys()):
                    if si.should_retire_agent(agent_name):
                        trust = si.agent_profiles[agent_name].trust_score
                        logger.warning(
                            f"Retiring agent {agent_name} " f"(trust={trust:.2f}) — circuit opened"
                        )
                        # Actually block the agent by opening its circuit
                        si.record_circuit_failure(agent_name, threshold=1)
                        si.metrics.record_coordination("retirement", agent=agent_name, success=True)
            except Exception as e:
                logger.debug(f"Retirement check failed: {e}")
                si.metrics.record_error("coordination", "retirement_check", str(e))

            # 6. Stigmergy evaporation: clean up stale signals post-execution.
            try:
                si.stigmergy.evaporate()
            except Exception as e:
                logger.debug(f"Post-execution stigmergy evaporation failed: {e}")
                si.metrics.record_error("coordination", "stigmergy_evaporate", str(e))

        except (AttributeError, ImportError) as e:
            logger.debug(f"Post-execution coordination skipped (optional dep): {e}")
        except (LearningError, ExecutionError) as e:
            logger.warning(f"Post-execution coordination failed (recoverable): {e}")
            try:
                si.metrics.record_error("coordination", "post_execution", str(e))
            except Exception:
                pass
        except Exception as e:
            logger.warning(
                f"Post-execution coordination failed (unexpected): {type(e).__name__}: {e}",
                exc_info=True,
            )
            try:
                si.metrics.record_error("coordination", "post_execution", str(e))
            except Exception:
                pass  # Last resort: metrics system itself is broken

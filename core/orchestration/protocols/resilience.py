"""
Resilience protocol mixin: circuit breakers, failure tracking, backpressure, adaptive timeouts.

Extracted from SwarmIntelligence for modularity.
These are mixed into SwarmIntelligence at class definition.
"""

import asyncio
import time
import logging
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import defaultdict

from ..swarm_data_structures import (
    AgentSpecialization, AgentProfile, ConsensusVote, SwarmDecision,
    AgentSession, HandoffContext, Coalition, AuctionBid, GossipMessage, SupervisorNode,
)

logger = logging.getLogger(__name__)


class ResilienceMixin:
    """Resilience protocol mixin: circuit breakers, failure tracking, backpressure, adaptive timeouts."""


    # =========================================================================
    # FAILURE RECOVERY (Auto-retry with different agent)
    # =========================================================================

    def record_failure(
        self,
        task_id: str,
        agent: str,
        task_type: str,
        error_type: str = "unknown",
        context: Dict = None
    ) -> Optional[str]:
        """
        Record task failure and auto-reassign to different agent.

        Failure recovery pattern: Automatic retry with alternative.

        Returns:
            New assigned agent or None if no alternatives.
        """
        # Update agent profile (reduce trust)
        if agent in self.agent_profiles:
            profile = self.agent_profiles[agent]
            profile.trust_score = max(0.1, profile.trust_score - 0.1)
            profile.update_task_result(task_type, False, 0.0)

        # Find alternative agent
        failed_agents = [agent]
        if task_id in self.pending_handoffs:
            failed_agents.extend(self.pending_handoffs[task_id].handoff_chain)

        available = [a for a in self.agent_profiles.keys() if a not in failed_agents]

        if not available:
            logger.warning(f"Task {task_id} failed, no alternatives available")
            return None

        # Use auction to find best alternative
        new_agent = self.auto_auction(f"{task_id}_retry", task_type, available)

        if new_agent:
            # Create handoff with failure context
            self.initiate_handoff(
                task_id=f"{task_id}_retry",
                from_agent=agent,
                to_agent=new_agent,
                task_type=task_type,
                context={**(context or {}), "retry_reason": error_type, "failed_agent": agent},
                progress=0.0
            )
            logger.info(f"Task {task_id} reassigned: {agent} (failed) → {new_agent}")

        return new_agent



    def get_failure_rate(self, agent: str, task_type: str = None) -> float:
        """Get failure rate for an agent (optionally for specific task type)."""
        profile = self.agent_profiles.get(agent)
        if not profile:
            return 0.5

        if task_type:
            return 1.0 - profile.get_success_rate(task_type)

        # Overall failure rate
        total_success = sum(s for s, t in profile.task_success.values())
        total_tasks = sum(t for s, t in profile.task_success.values())
        return 1.0 - (total_success / total_tasks) if total_tasks > 0 else 0.5

    # =========================================================================
    # PRIORITY QUEUE (Handle urgent tasks first)
    # =========================================================================



    # =========================================================================
    # CIRCUIT BREAKER (Stop sending to failing agents)
    # =========================================================================

    def __init_circuit_breakers(self) -> Any:
        """Initialize circuit breakers if not exists."""
        if not hasattr(self, 'circuit_breakers'):
            self.circuit_breakers: Dict[str, Dict] = {}  # agent -> {state, failures, last_failure}



    def get_circuit_state(self, agent: str) -> str:
        """Get circuit breaker state: 'closed' (ok), 'open' (blocked), 'half-open' (testing)."""
        self.__init_circuit_breakers()
        cb = self.circuit_breakers.get(agent, {})
        return cb.get('state', 'closed')



    def record_circuit_failure(self, agent: str, threshold: int = 3, cooldown: float = 60.0) -> None:
        """
        Record failure for circuit breaker.

        After `threshold` failures, circuit opens (blocks agent).
        After `cooldown` seconds, circuit becomes half-open (allows one test).
        """
        self.__init_circuit_breakers()

        if agent not in self.circuit_breakers:
            self.circuit_breakers[agent] = {'state': 'closed', 'failures': 0, 'last_failure': 0}

        cb = self.circuit_breakers[agent]
        cb['failures'] += 1
        cb['last_failure'] = time.time()

        if cb['failures'] >= threshold:
            cb['state'] = 'open'
            logger.warning(f"Circuit OPEN for {agent} after {cb['failures']} failures")



    def record_circuit_success(self, agent: str) -> None:
        """Record success - resets circuit breaker."""
        self.__init_circuit_breakers()

        if agent in self.circuit_breakers:
            self.circuit_breakers[agent] = {'state': 'closed', 'failures': 0, 'last_failure': 0}



    def check_circuit(self, agent: str, cooldown: float = 60.0) -> bool:
        """
        Check if agent is available (circuit not open).

        Returns True if agent can receive tasks.
        """
        self.__init_circuit_breakers()

        cb = self.circuit_breakers.get(agent)
        if not cb:
            return True

        if cb['state'] == 'closed':
            return True

        if cb['state'] == 'open':
            # Check if cooldown passed
            if time.time() - cb['last_failure'] > cooldown:
                cb['state'] = 'half-open'
                logger.info(f"Circuit HALF-OPEN for {agent} (testing)")
                return True
            return False

        # half-open - allow one test
        return True



    def get_available_agents(self, agents: List[str] = None) -> List[str]:
        """Get agents with closed or half-open circuits."""
        agents = agents or list(self.agent_profiles.keys())
        return [a for a in agents if self.check_circuit(a)]

    # =========================================================================
    # BACKPRESSURE (Slow down when overwhelmed)
    # =========================================================================



    # =========================================================================
    # BACKPRESSURE (Slow down when overwhelmed)
    # =========================================================================

    def calculate_backpressure(self) -> float:
        """
        Calculate swarm backpressure (0-1).

        High backpressure means swarm is overwhelmed.
        """
        if not self.agent_profiles:
            return 0.0

        # Factors contributing to backpressure
        avg_load = sum(self.get_agent_load(a) for a in self.agent_profiles) / len(self.agent_profiles)
        pending_ratio = min(1.0, len(self.pending_handoffs) / max(1, len(self.agent_profiles) * 3))
        queue_pressure = min(1.0, len(getattr(self, 'priority_queue', [])) / 20)

        backpressure = (avg_load * 0.4 + pending_ratio * 0.4 + queue_pressure * 0.2)
        return min(1.0, backpressure)



    def should_accept_task(self, priority: int = 5) -> bool:
        """
        Check if swarm should accept new task based on backpressure.

        High priority tasks (>=8) always accepted.
        """
        if priority >= 8:
            return True

        backpressure = self.calculate_backpressure()

        # Accept based on priority vs backpressure
        # Priority 5 needs backpressure < 0.7
        # Priority 3 needs backpressure < 0.5
        threshold = 0.5 + (priority - 5) * 0.1
        return backpressure < threshold

    # =========================================================================
    # EMERGENT LEADERSHIP (Dynamic leader election)
    # =========================================================================



    # =========================================================================
    # BYZANTINE CONSENSUS (Fault-tolerant agreement)
    # =========================================================================

    def byzantine_vote(
        self,
        question: str,
        options: List[str],
        voters: List[str] = None,
        threshold: float = 0.67
    ) -> Dict[str, Any]:
        """
        Byzantine fault-tolerant voting.

        Requires 2/3 majority for consensus (can tolerate 1/3 faulty nodes).

        Args:
            question: Question to vote on
            options: Available options
            voters: Participating agents (all if None)
            threshold: Required majority (default 2/3)

        Returns:
            Dict with decision, consensus reached, vote distribution.
        """
        voters = voters or list(self.agent_profiles.keys())

        # Collect votes (weighted by trust)
        votes = defaultdict(float)
        vote_details = []

        for agent in voters:
            profile = self.agent_profiles.get(agent, AgentProfile(agent))

            # Agent votes for option based on specialization match
            # (In real implementation, this would call agent's vote method)
            best_option = options[0]
            best_score = 0

            for opt in options:
                # Score based on past success with similar tasks
                score = profile.get_success_rate(opt) + profile.trust_score * 0.5
                if score > best_score:
                    best_score = score
                    best_option = opt

            # Weight vote by trust
            weight = profile.trust_score
            votes[best_option] += weight
            vote_details.append({
                "agent": agent,
                "vote": best_option,
                "weight": weight
            })

        # Determine winner
        total_weight = sum(votes.values())
        if total_weight == 0:
            return {
                "decision": options[0],
                "consensus": False,
                "reason": "no_votes",
                "votes": vote_details
            }

        winner = max(votes.keys(), key=lambda k: votes[k])
        winner_share = votes[winner] / total_weight

        consensus = winner_share >= threshold

        result = {
            "decision": winner,
            "consensus": consensus,
            "share": winner_share,
            "threshold": threshold,
            "votes": vote_details,
            "distribution": dict(votes)
        }

        if consensus:
            logger.info(f"Byzantine consensus reached: {winner} ({winner_share:.0%})")
        else:
            logger.warning(f"Byzantine consensus FAILED: {winner} only {winner_share:.0%} < {threshold:.0%}")

        return result

    # =========================================================================
    # CIRCUIT BREAKER (Stop sending to failing agents)
    # =========================================================================


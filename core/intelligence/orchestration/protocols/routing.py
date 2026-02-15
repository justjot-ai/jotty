from __future__ import annotations


from pathlib import Path
"""
Routing protocol mixin: smart routing, load balancing, work stealing.

Extracted from SwarmIntelligence for modularity.
These are mixed into SwarmIntelligence at class definition.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable, TYPE_CHECKING, Set

from ..swarm_data_structures import (
    AgentSpecialization, AgentProfile, ConsensusVote, SwarmDecision,
    AgentSession, HandoffContext, Coalition, AuctionBid, GossipMessage, SupervisorNode,
)

logger = logging.getLogger(__name__)


class RoutingMixin:
    """Routing protocol mixin: smart routing, load balancing, work stealing."""


    # =========================================================================
    # INTEGRATED ROUTING (Combines All Patterns)
    # =========================================================================

    def smart_route(
        self,
        task_id: str,
        task_type: str,
        task_description: str = "",
        prefer_coalition: bool = False,
        use_auction: bool = False,
        use_hierarchy: bool = True
    ) -> Dict[str, Any]:
        """

    if TYPE_CHECKING:
        # Comprehensive attribute declarations for type checking
        # These are provided by parent class or other mixins in composition

    # Orchestration attributes
    agents: List[Any]
    agent_profiles: Dict[str, Any]
    agent_name: Optional[str]
    config: Dict[str, Any]
    name: str
    history: List[Any]
    handoff_history: List[Any]
    pending_handoffs: List[Any]
    active_auctions: Dict[str, Any]
    agent_coalitions: Dict[str, Any]
    coalitions: Dict[str, Any]
    circuit_breakers: Dict[str, Any]
    gossip_inbox: List[Any]
    gossip_seen: Set[str]
    consensus_history: List[Any]
    morph_score_history: List[Any]
    morph_scorer: Optional[Any]

    def register_agent(self, agent: Any) -> None: ...
    def find_idle_agents(self) -> List[Any]: ...
    def find_overloaded_agents(self) -> List[Any]: ...
    def get_agent_load(self, agent_id: str) -> float: ...
    def get_available_agents(self) -> List[Any]: ...
    def initiate_handoff(self, from_agent: str, to_agent: str, **kwargs: Any) -> None: ...
    def auto_auction(self, task: Any) -> Any: ...
    def form_coalition(self, agents: List[str]) -> str: ...
    def check_circuit(self, operation: str) -> bool: ...
    def gossip_broadcast(self, message: Any) -> None: ...
    def get_swarm_health(self) -> Dict[str, Any]: ...



    def _rl_informed_select(self, available: List[str], task_type: str) -> Optional[str]:
        """
        Select agent using RL-learned values (closes the RL loop).

        Uses TD-Lambda grouped baseline advantages: agents whose task_type
        baseline is above average get preference. Combined with success rate
        for a blended RL+empirical score.

        Returns:
            Best agent name, or None if no RL data available.
        """
        # Access the grouped baseline from the learning system
        td_learner = getattr(self, '_td_learner', None)
        if td_learner is None:
            return None

        grouped = getattr(td_learner, 'grouped_baseline', None)
        if grouped is None:
            return None

        # Need at least some learning data
        if grouped.group_counts.get(task_type, 0) < 2:
            return None

        baseline = grouped.get_baseline(task_type)
        best_agent = None
        best_score = -1.0

        for agent_name in available:
            profile = self.agent_profiles.get(agent_name)
            if not profile:
                continue

            # Empirical success rate for this task type
            success_rate = profile.get_success_rate(task_type)

            # RL advantage: how much better is this agent's recent performance
            # vs the group baseline for this task type
            advantage = success_rate - baseline

            # Trust-weighted blend of empirical + RL advantage
            trust = profile.trust_score
            score = (
                0.5 * success_rate +
                0.3 * (0.5 + advantage) +  # center advantage around 0.5
                0.2 * trust
            )

            if score > best_score:
                best_score = score
                best_agent = agent_name

        return best_agent

    def _get_rl_advantage(self, agent_name: str, task_type: str) -> float:
        """
        Get the RL advantage for an agent on a task type.

        Returns the difference between agent's success rate and the
        group baseline (positive = agent is above average for this task type).
        """
        td_learner = getattr(self, '_td_learner', None)
        if td_learner is None:
            return 0.0

        grouped = getattr(td_learner, 'grouped_baseline', None)
        if grouped is None:
            return 0.0

        baseline = grouped.get_baseline(task_type)
        profile = self.agent_profiles.get(agent_name)
        if not profile:
            return 0.0

        return profile.get_success_rate(task_type) - baseline

    # =========================================================================
    # WORK-STEALING (Idle agents steal from busy ones)
    # =========================================================================



    # =========================================================================
    # WORK-STEALING (Idle agents steal from busy ones)
    # =========================================================================

    def get_agent_load(self, agent: str) -> float:
        """
        Get current load of an agent (0-1).

        Based on: pending handoffs, coalition membership, recent task count.
        """
        load = 0.0

        # Pending handoffs to this agent
        pending = len([h for h in self.pending_handoffs.values() if h.to_agent == agent])
        load += min(0.4, pending * 0.1)

        # Coalition membership
        if agent in self.agent_coalitions:
            load += 0.2

        # Recent tasks (from collective memory)
        # Use list() for deque compatibility (deque doesn't support slicing in all Python versions)
        mem_list = list(self.collective_memory)
        recent = [m for m in mem_list[-20:]
                  if m.get('agent') == agent and time.time() - m.get('timestamp', 0) < 60]
        load += min(0.4, len(recent) * 0.1)

        return min(1.0, load)



    def find_overloaded_agents(self, threshold: float = 0.7) -> List[str]:
        """Find agents with load above threshold."""
        return [a for a in self.agent_profiles.keys()
                if self.get_agent_load(a) > threshold]



    def find_idle_agents(self, threshold: float = 0.3) -> List[str]:
        """Find agents with load below threshold."""
        return [a for a in self.agent_profiles.keys()
                if self.get_agent_load(a) < threshold]



    def work_steal(self, idle_agent: str) -> Optional[HandoffContext]:
        """
        Idle agent steals work from overloaded agent.

        Work-stealing pattern: Automatic load balancing.

        Returns:
            HandoffContext if work was stolen, None otherwise.
        """
        overloaded = self.find_overloaded_agents()
        if not overloaded:
            return None

        # Find task to steal (pending handoff to overloaded agent)
        for busy_agent in overloaded:
            for task_id, handoff in list(self.pending_handoffs.items()):
                if handoff.to_agent == busy_agent:
                    # Steal this task
                    handoff.to_agent = idle_agent
                    handoff.add_to_chain(busy_agent)  # Record original target

                    logger.info(f"Work stolen: {idle_agent} took {task_id} from {busy_agent}")

                    # Notify via gossip
                    self.gossip_broadcast(
                        origin_agent=idle_agent,
                        message_type="work_steal",
                        content={"task_id": task_id, "from": busy_agent, "to": idle_agent}
                    )

                    return handoff

        return None



    def balance_load(self) -> List[Dict]:
        """
        Rebalance work across the swarm.

        Returns list of rebalancing actions taken.
        """
        actions = []
        idle = self.find_idle_agents()
        overloaded = self.find_overloaded_agents()

        for idle_agent in idle:
            if not overloaded:
                break

            result = self.work_steal(idle_agent)
            if result:
                actions.append({
                    "action": "work_steal",
                    "from": result.handoff_chain[-1] if result.handoff_chain else "unknown",
                    "to": idle_agent,
                    "task_id": result.task_id
                })
                # Recalculate overloaded
                overloaded = self.find_overloaded_agents()

        if actions:
            logger.info(f"Load balanced: {len(actions)} tasks redistributed")

        return actions

    # =========================================================================
    # FAILURE RECOVERY (Auto-retry with different agent)
    # =========================================================================


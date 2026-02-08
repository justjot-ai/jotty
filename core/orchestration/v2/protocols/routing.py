"""
Routing protocol mixin: smart routing, load balancing, work stealing.

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
        Smart routing combining all arXiv swarm patterns.

        Integrates: handoff, hierarchy, auction, coalition, gossip.

        Args:
            task_id: Task identifier
            task_type: Type of task
            task_description: Optional description
            prefer_coalition: Form coalition for complex tasks
            use_auction: Use auction for allocation
            use_hierarchy: Use hierarchical routing

        Returns:
            Dict with routing decision and metadata
        """
        result = {
            "task_id": task_id,
            "assigned_agent": None,
            "coalition": None,
            "method": "direct",
            "confidence": 0.5
        }

        available = list(self.agent_profiles.keys())
        if not available:
            return result

        # Strategy 1: Coalition for complex tasks
        if prefer_coalition:
            coalition = self.form_coalition(task_type, min_agents=2, max_agents=4)
            if coalition:
                result["assigned_agent"] = coalition.leader
                result["coalition"] = coalition.coalition_id
                result["method"] = "coalition"
                result["confidence"] = 0.9
                return result

        # Strategy 2: Auction for competitive allocation
        if use_auction:
            winner = self.auto_auction(task_id, task_type, available)
            if winner:
                result["assigned_agent"] = winner
                result["method"] = "auction"
                result["confidence"] = 0.85
                return result

        # Strategy 3: Hierarchical routing
        if use_hierarchy and self._tree_built:
            agent = self.route_via_hierarchy(task_type)
            if agent:
                result["assigned_agent"] = agent
                result["method"] = "hierarchy"
                result["confidence"] = 0.8
                return result

        # Strategy 4: MorphAgent TRAS scoring
        if task_description:
            profiles = {a: self.agent_profiles[a] for a in available}
            best = self.morph_scorer.get_best_agent_by_tras(
                profiles=profiles,
                task=task_description,
                task_type=task_type
            )
            if best:
                result["assigned_agent"] = best
                result["method"] = "morph_tras"
                result["confidence"] = 0.75
                return result

        # Strategy 5: Fallback to simple routing
        best = self.get_best_agent_for_task(task_type, available, task_description)
        result["assigned_agent"] = best
        result["method"] = "simple"
        result["confidence"] = 0.6
        return result

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
        recent = [m for m in self.collective_memory[-20:]
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


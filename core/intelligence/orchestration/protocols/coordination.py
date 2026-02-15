"""
Coordination protocol mixin: handoff, auction, coalition, gossip, supervisor hierarchy.

Extracted from SwarmIntelligence for modularity.
These are mixed into SwarmIntelligence at class definition.
"""

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

from ..swarm_data_structures import (
    AgentProfile,
    AuctionBid,
    Coalition,
    GossipMessage,
    HandoffContext,
    SupervisorNode,
)

logger = logging.getLogger(__name__)


class CoordinationMixin:
    """Coordination protocol mixin: handoff, auction, coalition, gossip, supervisor hierarchy."""

    # =========================================================================
    # AGENT HANDOFF (SwarmAgentic Pattern)
    # =========================================================================

    def initiate_handoff(
        self,
        task_id: str,
        from_agent: str,
        to_agent: str,
        task_type: str,
        context: Dict = None,
        partial_result: Any = None,
        progress: float = 0.0,
        priority: int = 5,
    ) -> HandoffContext:
        """
        Initiate task handoff between agents with context preservation.

        SwarmAgentic pattern: Seamless task transfer without losing state.

        Args:
            task_id: Unique task identifier
            from_agent: Agent initiating handoff
            to_agent: Agent receiving task
            task_type: Type of task being handed off
            context: Task context to preserve
            partial_result: Any partial work completed
            progress: Completion progress 0-1
            priority: Task priority 1-10

        Returns:
            HandoffContext for tracking
        """
        handoff = HandoffContext(
            task_id=task_id,
            from_agent=from_agent,
            to_agent=to_agent,
            task_type=task_type,
            context=context or {},
            partial_result=partial_result,
            progress=progress,
            priority=priority,
        )
        handoff.add_to_chain(from_agent)

        self.pending_handoffs[task_id] = handoff

        # Notify via gossip
        self.gossip_broadcast(
            origin_agent=from_agent,
            message_type="handoff",
            content={"task_id": task_id, "to": to_agent, "type": task_type},
        )

        logger.info(f"Handoff initiated: {from_agent} → {to_agent} for task {task_id}")
        return handoff

    def accept_handoff(self, task_id: str, agent: str) -> Optional[HandoffContext]:
        """
        Accept a pending handoff.

        Returns the handoff context for the receiving agent to continue work.
        """
        handoff = self.pending_handoffs.pop(task_id, None)
        if handoff and handoff.to_agent == agent:
            handoff.add_to_chain(agent)
            self.handoff_history.append(handoff)
            logger.info(f"Handoff accepted: {agent} received task {task_id}")
            return handoff
        return None

    def reject_handoff(self, task_id: str, agent: str, reason: str = "") -> bool:
        """
        Reject a handoff and find alternative agent.

        Returns True if successfully rerouted, False if no alternative.
        """
        handoff = self.pending_handoffs.get(task_id)
        if not handoff or handoff.to_agent != agent:
            return False

        # Find alternative via auction
        available = [
            a for a in self.agent_profiles.keys() if a != agent and a not in handoff.handoff_chain
        ]

        if not available:
            logger.warning(f"Handoff rejected, no alternatives: {task_id}")
            return False

        # Quick auction for rerouting
        best = self.get_best_agent_for_task(handoff.task_type, available)
        if best:
            handoff.to_agent = best
            logger.info(f"Handoff rerouted: {task_id} → {best} (rejected by {agent}: {reason})")
            return True

        return False

    def get_pending_handoffs(self, agent: str) -> List[HandoffContext]:
        """Get all pending handoffs for an agent."""
        return [h for h in self.pending_handoffs.values() if h.to_agent == agent]

    # =========================================================================
    # HIERARCHICAL SUPERVISOR TREE (SwarmSys O(log n) Pattern)
    # =========================================================================

    # =========================================================================
    # HIERARCHICAL SUPERVISOR TREE (SwarmSys O(log n) Pattern)
    # =========================================================================

    def build_supervisor_tree(self, agents: List[str] = None, branching_factor: int = 3) -> None:
        """
        Build hierarchical supervisor tree for O(log n) coordination.

        SwarmSys pattern: Layered supervisors reduce communication complexity
        from O(n) to O(log n).

        Args:
            agents: List of agents (uses all registered if None)
            branching_factor: Children per supervisor (default 3)
        """
        agents = agents or list(self.agent_profiles.keys())
        if not agents:
            return

        self.supervisor_tree.clear()

        # Level 0: All agents as leaves
        level = 0
        current_level = []
        for i, agent in enumerate(agents):
            node_id = f"L{level}_{i}"
            node = SupervisorNode(
                node_id=node_id, agent_name=agent, level=level, supervised_agents=[agent]
            )
            self.supervisor_tree[node_id] = node
            current_level.append(node_id)

        # Build supervisor levels until we have a single root
        while len(current_level) > 1:
            level += 1
            next_level = []

            for i in range(0, len(current_level), branching_factor):
                children = current_level[i : i + branching_factor]
                if not children:
                    continue

                # Pick best agent as supervisor (highest trust)
                child_agents = [self.supervisor_tree[c].agent_name for c in children]
                supervisor_agent = max(
                    child_agents,
                    key=lambda a: self.agent_profiles.get(a, AgentProfile(a)).trust_score,
                )

                node_id = f"L{level}_{len(next_level)}"
                supervised = []
                for c in children:
                    supervised.extend(self.supervisor_tree[c].supervised_agents)
                    self.supervisor_tree[c].parent = node_id

                node = SupervisorNode(
                    node_id=node_id,
                    agent_name=supervisor_agent,
                    level=level,
                    children=children,
                    supervised_agents=supervised,
                )
                self.supervisor_tree[node_id] = node
                next_level.append(node_id)

            current_level = next_level

        self._tree_built = True
        logger.info(
            f"Supervisor tree built: {len(agents)} agents, {level + 1} levels, O(log {len(agents)}) = O({level + 1})"
        )

    def get_supervisor(self, agent: str) -> Optional[str]:
        """Get the supervisor agent for a given agent."""
        for node in self.supervisor_tree.values():
            if node.agent_name == agent and node.parent:
                parent_node = self.supervisor_tree.get(node.parent)
                if parent_node:
                    return parent_node.agent_name
        return None

    def get_supervised_agents(self, supervisor: str) -> List[str]:
        """Get all agents supervised by a given supervisor."""
        for node in self.supervisor_tree.values():
            if node.agent_name == supervisor:
                return node.supervised_agents
        return []

    def route_via_hierarchy(self, task_type: str, from_agent: str = None) -> Optional[str]:
        """
        Route task through hierarchy for O(log n) routing.

        Instead of checking all agents, routes through supervisor tree.
        """
        if not self._tree_built:
            self.build_supervisor_tree()

        # Find root
        root = None
        for node in self.supervisor_tree.values():
            if node.parent is None and node.level > 0:
                root = node
                break

        if not root:
            # Fallback to flat routing
            return self.get_best_agent_for_task(task_type, list(self.agent_profiles.keys()))

        # Traverse down tree finding best path
        current = root
        while current.children:
            best_child = None
            best_score = -1

            for child_id in current.children:
                child = self.supervisor_tree.get(child_id)
                if not child:
                    continue

                # Score based on task success in subtree
                subtree_score = 0
                for agent in child.supervised_agents:
                    profile = self.agent_profiles.get(agent)
                    if profile:
                        subtree_score += profile.get_success_rate(task_type)

                if subtree_score > best_score:
                    best_score = subtree_score
                    best_child = child

            if best_child:
                current = best_child
            else:
                break

        return current.agent_name

    # =========================================================================
    # GOSSIP PROTOCOL (SwarmSys O(log n) Dissemination)
    # =========================================================================

    # =========================================================================
    # GOSSIP PROTOCOL (SwarmSys O(log n) Dissemination)
    # =========================================================================

    def gossip_broadcast(
        self, origin_agent: str, message_type: str, content: Dict[str, Any], ttl: int = 3
    ) -> str:
        """
        Broadcast message via gossip protocol.

        SwarmSys pattern: O(log n) information spread without central coordinator.

        Args:
            origin_agent: Agent originating the message
            message_type: Type of message (info, warning, route, capability)
            content: Message content
            ttl: Time-to-live in hops

        Returns:
            Message ID
        """
        msg_id = hashlib.md5(f"{origin_agent}:{message_type}:{time.time()}".encode()).hexdigest()[
            :12
        ]

        message = GossipMessage(
            message_id=msg_id,
            content=content,
            origin_agent=origin_agent,
            message_type=message_type,
            ttl=ttl,
            seen_by=[origin_agent],
        )

        # Distribute to random subset of agents (gossip fanout)
        import random

        all_agents = [a for a in self.agent_profiles.keys() if a != origin_agent]
        fanout = min(3, len(all_agents))  # Gossip to 3 random agents
        targets = random.sample(all_agents, fanout) if all_agents else []

        for target in targets:
            if target not in self.gossip_inbox:
                self.gossip_inbox[target] = []
            self.gossip_inbox[target].append(message)

        self.gossip_seen[msg_id] = True
        logger.debug(
            f"Gossip broadcast: {message_type} from {origin_agent} to {len(targets)} agents"
        )
        return msg_id

    def gossip_receive(self, agent: str) -> List[GossipMessage]:
        """
        Receive and process gossip messages for an agent.

        Agent processes messages and propagates if TTL > 0.
        """
        messages = self.gossip_inbox.pop(agent, [])
        to_propagate = []

        for msg in messages:
            if msg.mark_seen(agent):
                to_propagate.append(msg)

        # Propagate with reduced TTL
        for msg in to_propagate:
            import random

            other_agents = [
                a for a in self.agent_profiles.keys() if a != agent and a not in msg.seen_by
            ]
            if other_agents:
                target = random.choice(other_agents)
                if target not in self.gossip_inbox:
                    self.gossip_inbox[target] = []
                self.gossip_inbox[target].append(msg)

        return messages

    def gossip_query(self, query_type: str, agent: str = None) -> List[Dict]:
        """
        Query recent gossip messages by type.

        Args:
            query_type: Message type to filter
            agent: Optional agent to filter by origin
        """
        results = []
        for inbox in self.gossip_inbox.values():
            for msg in inbox:
                if msg.message_type == query_type:
                    if agent is None or msg.origin_agent == agent:
                        results.append(
                            {
                                "id": msg.message_id,
                                "content": msg.content,
                                "from": msg.origin_agent,
                                "age": time.time() - msg.created_at,
                            }
                        )
        return results

    # =========================================================================
    # AUCTION-BASED TASK ALLOCATION (SwarmSys Contract-Net)
    # =========================================================================

    # =========================================================================
    # AUCTION-BASED TASK ALLOCATION (SwarmSys Contract-Net)
    # =========================================================================

    def start_auction(
        self,
        task_id: str,
        task_type: str,
        task_description: str = "",
        deadline_seconds: float = 5.0,
    ) -> str:
        """
        Start auction for task allocation.

        SwarmSys contract-net pattern: Agents bid based on capability.

        Args:
            task_id: Unique task identifier
            task_type: Type of task
            task_description: Optional description
            deadline_seconds: Auction duration

        Returns:
            Auction task_id
        """
        self.active_auctions[task_id] = []

        # Broadcast auction announcement via gossip
        self.gossip_broadcast(
            origin_agent="auctioneer",
            message_type="auction",
            content={
                "task_id": task_id,
                "task_type": task_type,
                "description": task_description,
                "deadline": time.time() + deadline_seconds,
            },
        )

        logger.info(f"Auction started: {task_id} ({task_type})")
        return task_id

    def submit_bid(
        self,
        task_id: str,
        agent_name: str,
        estimated_time: float = 10.0,
        confidence: float = 0.8,
        reasoning: str = "",
    ) -> Optional[AuctionBid]:
        """
        Submit bid for an auction.

        Args:
            task_id: Auction task_id
            agent_name: Bidding agent
            estimated_time: Estimated completion time
            confidence: Confidence level 0-1
            reasoning: Optional explanation

        Returns:
            AuctionBid if accepted
        """
        if task_id not in self.active_auctions:
            return None

        profile = self.agent_profiles.get(agent_name)
        if not profile:
            self.register_agent(agent_name)
            profile = self.agent_profiles[agent_name]

        # Calculate bid value from profile
        bid_value = profile.trust_score

        # Specialization match (check if agent specializes in this task type)
        expected_spec = self._task_type_to_specialization(
            task_id.split("_")[0] if "_" in task_id else "general"
        )
        spec_match = 1.0 if profile.specialization == expected_spec else 0.5

        # Current load (based on pending handoffs)
        current_load = (
            len([h for h in self.pending_handoffs.values() if h.to_agent == agent_name]) / 5.0
        )
        current_load = min(1.0, current_load)

        bid = AuctionBid(
            agent_name=agent_name,
            task_id=task_id,
            bid_value=bid_value,
            estimated_time=estimated_time,
            confidence=confidence,
            specialization_match=spec_match,
            current_load=current_load,
            reasoning=reasoning,
        )

        self.active_auctions[task_id].append(bid)
        logger.debug(f"Bid submitted: {agent_name} for {task_id} (score: {bid.score:.2f})")
        return bid

    def close_auction(self, task_id: str) -> Optional[str]:
        """
        Close auction and determine winner.

        Returns winning agent name or None.
        """
        bids = self.active_auctions.pop(task_id, [])
        if not bids:
            return None

        # Sort by combined score
        bids.sort(key=lambda b: b.score, reverse=True)
        winner = bids[0]

        logger.info(f"Auction closed: {task_id} → {winner.agent_name} (score: {winner.score:.2f})")
        return winner.agent_name

    def auto_auction(
        self, task_id: str, task_type: str, available_agents: List[str] = None
    ) -> Optional[str]:
        """
        Run instant auction (no delay) for immediate task allocation.

        Convenience method combining start, bids, and close.
        """
        agents = available_agents or list(self.agent_profiles.keys())
        if not agents:
            return None

        self.start_auction(task_id, task_type)

        for agent in agents:
            self.submit_bid(task_id, agent)

        return self.close_auction(task_id)

    # =========================================================================
    # COALITION FORMATION (SwarmAgentic Dynamic Teams)
    # =========================================================================

    # =========================================================================
    # COALITION FORMATION (SwarmAgentic Dynamic Teams)
    # =========================================================================

    def form_coalition(
        self,
        task_type: str,
        required_roles: List[str] = None,
        min_agents: int = 2,
        max_agents: int = 5,
    ) -> Optional[Coalition]:
        """
        Form dynamic coalition for complex tasks.

        SwarmAgentic pattern: Assemble optimal team based on capabilities.

        Args:
            task_type: Type of task requiring coalition
            required_roles: Specific roles needed (e.g., ["analyzer", "validator"])
            min_agents: Minimum team size
            max_agents: Maximum team size

        Returns:
            Coalition if successfully formed
        """
        # =====================================================================
        # COALITION FORMATION EXPLAINED
        # =====================================================================
        # PROBLEM: Some tasks are too complex for a single agent.
        #
        # EXAMPLES:
        # - "Research quantum computing AND write a 10-page report"
        #   → Needs: Researcher + Writer
        # - "Find bugs in code AND fix them AND write tests"
        #   → Needs: Analyzer + Coder + Tester
        # - "Analyze financial data AND create visualizations AND present findings"
        #   → Needs: DataAnalyst + Visualizer + Presenter
        #
        # CHALLENGES:
        # 1. Which agents should work together?
        # 2. How many agents are needed?
        # 3. Who should lead the coalition?
        # 4. How to ensure complementary skills?
        # 5. How to avoid assigning same agent to multiple coalitions?
        #
        # SOLUTION: Multi-Criteria Team Selection Algorithm
        #
        # ALGORITHM OVERVIEW:
        # 1. Filter available agents (not already in other coalitions)
        # 2. Score each agent using weighted criteria:
        #    - Success rate on this task type (40% weight)
        #    - Trust score (30% weight)
        #    - Role match (30% weight)
        # 3. Select team:
        #    a. First fill required roles (e.g., MUST have analyzer)
        #    b. Then add highest-scored agents up to max_agents
        # 4. Choose leader (agent with highest trust)
        # 5. Register coalition and broadcast formation
        #
        # SCORING FORMULA:
        # score = success_rate × 0.4 + trust × 0.3 + role_match × 0.3
        #
        # Where:
        # - success_rate: Historical success on this task type (0-1)
        # - trust: Overall trust score from past interactions (0-1)
        # - role_match: 1.0 if has required role, 0.5 otherwise
        #
        # EXAMPLE:
        # Task: "Research and write report"
        # Required roles: ["researcher", "writer"]
        # Available agents:
        #
        # Agent A (Researcher):
        #   - Success rate on research: 0.8
        #   - Trust score: 0.9
        #   - Role: researcher (required) → 1.0
        #   - Score: 0.8 × 0.4 + 0.9 × 0.3 + 1.0 × 0.3 = 0.32 + 0.27 + 0.3 = 0.89
        #
        # Agent B (Writer):
        #   - Success rate on writing: 0.7
        #   - Trust score: 0.8
        #   - Role: writer (required) → 1.0
        #   - Score: 0.7 × 0.4 + 0.8 × 0.3 + 1.0 × 0.3 = 0.28 + 0.24 + 0.3 = 0.82
        #
        # Agent C (Coder):
        #   - Success rate on coding: 0.9
        #   - Trust score: 0.7
        #   - Role: coder (not required) → 0.5
        #   - Score: 0.9 × 0.4 + 0.7 × 0.3 + 0.5 × 0.3 = 0.36 + 0.21 + 0.15 = 0.72
        #
        # Selection process:
        # 1. Fill required roles first: A (researcher), B (writer)
        # 2. Still have space (max_agents=5, current=2)
        # 3. Add highest-scored remaining: C (coder, score=0.72)
        # 4. Final team: [A, B, C]
        # 5. Leader: A (highest trust=0.9)
        #
        # WHY THIS WORKS:
        # 1. Required roles guaranteed (task won't fail due to missing skills)
        # 2. Best performers selected (40% weight on success rate)
        # 3. Trustworthy agents lead (reduces coordination overhead)
        # 4. Flexible team size (adapts to task complexity)
        # 5. No double-booking (agents can only join one coalition at a time)
        # =====================================================================


        # STEP 1: Filter available agents (not in other coalitions)
        available = [a for a in self.agent_profiles.keys() if a not in self.agent_coalitions]

        if len(available) < min_agents:
            logger.warning(f"Not enough agents for coalition: {len(available)} < {min_agents}")
            return None

        # =====================================================================
        # STEP 2: SCORE AGENTS USING MULTI-CRITERIA FORMULA
        # =====================================================================
        # Each agent gets a score based on three weighted factors:
        #
        # Factor 1: Success rate on this task type (40% weight)
        #   - How well has this agent performed on similar tasks?
        #   - Range: 0.0 (never succeeded) to 1.0 (always succeeded)
        #   - Example: Agent with 8/10 success rate → 0.8
        #
        # Factor 2: Trust score (30% weight)
        #   - Overall reliability and cooperation
        #   - Tracks: response time, error rate, handoff quality
        #   - Range: 0.0 (unreliable) to 1.0 (highly reliable)
        #
        # Factor 3: Role match (30% weight)
        #   - Does agent's specialization match required role?
        #   - 1.0 if exact match (required)
        #   - 0.5 if not required (can still contribute)
        #
        # WEIGHTS RATIONALE:
        # - Success rate (40%): Most important - we want agents that succeed
        # - Trust (30%): Important - unreliable agents slow down coalition
        # - Role (30%): Important - ensures skill coverage
        # =====================================================================
        scored = []
        for agent in available:
            profile = self.agent_profiles[agent]
            score = (
                profile.get_success_rate(task_type) * 0.4
                + profile.trust_score * 0.3  # Past performance
                + (  # Reliability
                    1.0 if profile.specialization.value in (required_roles or []) else 0.5
                )
                * 0.3  # Role fit
            )
            scored.append((agent, score, profile.specialization.value))

        # Sort by score (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)

        # =====================================================================
        # STEP 3: SELECT TEAM MEMBERS
        # =====================================================================
        # Two-phase selection:
        #
        # Phase 1: Fill required roles (MUST-HAVE)
        #   - Ensures critical skills are covered
        #   - Example: Can't do "write tests" without a Tester
        #   - Picks first agent matching each required role
        #
        # Phase 2: Add highest-scored agents (NICE-TO-HAVE)
        #   - Fills remaining slots up to max_agents
        #   - Adds complementary skills
        #   - Example: Adding a Reviewer improves output quality
        #
        # WHY TWO PHASES:
        # If we only did Phase 2, we might fill the team with high-scoring
        # generalists but miss critical specialists. Phase 1 guarantees
        # we have essential skills before optimizing for score.
        # =====================================================================
        selected = []
        roles_filled = {}

        # Phase 1: Fill required roles first
        for role in required_roles or []:
            for agent, score, spec in scored:
                if agent not in selected and spec == role:
                    selected.append(agent)
                    roles_filled[agent] = role
                    break

        # Phase 2: Add highest-scored agents until we hit max_agents
        for agent, score, spec in scored:
            if len(selected) >= max_agents:
                break
            if agent not in selected:
                selected.append(agent)
                roles_filled[agent] = spec

        # Verify minimum team size
        if len(selected) < min_agents:
            return None

        # =====================================================================
        # STEP 4: SELECT LEADER
        # =====================================================================
        # Leader is the agent with highest trust score among selected members.
        #
        # WHY TRUST SCORE FOR LEADERSHIP:
        # - High trust = reliable, responsive, cooperative
        # - Leader coordinates coalition (hands off tasks, aggregates results)
        # - Unreliable leader = coalition fails even if members are good
        #
        # EXAMPLE:
        # Team: [AgentA (trust=0.9), AgentB (trust=0.7), AgentC (trust=0.8)]
        # Leader: AgentA (highest trust)
        # =====================================================================
        leader = max(selected, key=lambda a: self.agent_profiles[a].trust_score)

        coalition_id = hashlib.md5(f"coalition:{task_type}:{time.time()}".encode()).hexdigest()[:12]

        coalition = Coalition(
            coalition_id=coalition_id,
            task_type=task_type,
            leader=leader,
            members=selected,
            roles=roles_filled,
        )

        # =====================================================================
        # STEP 5: REGISTER COALITION
        # =====================================================================
        # Register in two places:
        # 1. self.coalitions[id] → Lookup by coalition ID
        # 2. self.agent_coalitions[agent] → Prevent double-booking
        #
        # This ensures no agent can be in two coalitions simultaneously.
        # =====================================================================
        self.coalitions[coalition_id] = coalition
        for agent in selected:
            self.agent_coalitions[agent] = coalition_id

        # =====================================================================
        # STEP 6: BROADCAST FORMATION VIA GOSSIP
        # =====================================================================
        # Announce coalition formation to all agents via gossip protocol.
        # This allows:
        # - Other agents to know who's busy (avoid routing tasks to them)
        # - Monitoring systems to track active coalitions
        # - Learning systems to observe successful team compositions
        # =====================================================================
        self.gossip_broadcast(
            origin_agent=leader,
            message_type="coalition",
            content={
                "coalition_id": coalition_id,
                "task_type": task_type,
                "members": selected,
                "leader": leader,
            },
        )

        logger.info(
            f"Coalition formed: {coalition_id} with {len(selected)} agents, leader: {leader}"
        )
        return coalition

    def dissolve_coalition(self, coalition_id: str) -> None:
        """Dissolve a coalition after task completion."""
        coalition = self.coalitions.pop(coalition_id, None)
        if coalition:
            for agent in coalition.members:
                self.agent_coalitions.pop(agent, None)
            coalition.active = False
            logger.info(f"Coalition dissolved: {coalition_id}")

    def get_coalition(self, agent: str) -> Optional[Coalition]:
        """Get coalition an agent belongs to."""
        coalition_id = self.agent_coalitions.get(agent)
        return self.coalitions.get(coalition_id) if coalition_id else None

    def coalition_broadcast(self, coalition_id: str, message: Dict[str, Any]) -> None:
        """Broadcast message to all coalition members."""
        coalition = self.coalitions.get(coalition_id)
        if not coalition:
            return

        for agent in coalition.members:
            if agent not in self.gossip_inbox:
                self.gossip_inbox[agent] = []
            self.gossip_inbox[agent].append(
                GossipMessage(
                    message_id=f"cb_{coalition_id}_{time.time()}",
                    content=message,
                    origin_agent=coalition.leader,
                    message_type="coalition_msg",
                    ttl=1,  # Direct delivery only
                )
            )

    # =========================================================================
    # INTEGRATED ROUTING (Combines All Patterns)
    # =========================================================================

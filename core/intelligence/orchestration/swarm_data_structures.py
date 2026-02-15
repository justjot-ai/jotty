"""
Swarm Intelligence Data Structures
====================================

Core data classes for swarm intelligence:
- AgentSpecialization: Emergent agent specializations
- AgentProfile: Dynamic performance-tracking profile
- ConsensusVote: Vote in consensus decisions
- SwarmDecision: Result of swarm consensus
- AgentSession: Isolated session (moltbot pattern)

Extracted from swarm_intelligence.py for modularity.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple


class AgentSpecialization(Enum):
    """Emergent agent specializations."""

    GENERALIST = "generalist"
    AGGREGATOR = "aggregator"  # Good at count/sum/avg
    ANALYZER = "analyzer"  # Good at analysis tasks
    TRANSFORMER = "transformer"  # Good at data transformation
    VALIDATOR = "validator"  # Good at validation/checking
    PLANNER = "planner"  # Good at planning/decomposition
    EXECUTOR = "executor"  # Good at execution/action
    # Match real AgentRole task types from _record_trace()
    ACTOR = "actor"  # Action/execution specialist
    EXPERT = "expert"  # Domain knowledge specialist
    REVIEWER = "reviewer"  # Quality/review specialist
    ORCHESTRATOR = "orchestrator"  # Coordination specialist
    RESEARCHER = "researcher"  # Research/learning specialist


@dataclass
class AgentProfile:
    """Dynamic profile that evolves based on performance."""

    agent_name: str
    specialization: AgentSpecialization = AgentSpecialization.GENERALIST

    # Performance tracking by task type
    task_success: Dict[str, Tuple[int, int]] = field(
        default_factory=dict
    )  # task_type -> (success, total)

    # Collaboration stats
    helped_others: int = 0
    received_help: int = 0
    consensus_agreements: int = 0
    consensus_disagreements: int = 0

    # Timing stats
    avg_execution_time: float = 0.0
    total_tasks: int = 0

    # Trust score (how reliable is this agent)
    trust_score: float = 0.5

    def update_task_result(self, task_type: str, success: bool, execution_time: float) -> None:
        """Update profile after task completion."""
        if task_type not in self.task_success:
            self.task_success[task_type] = (0, 0)

        succ, total = self.task_success[task_type]
        self.task_success[task_type] = (succ + (1 if success else 0), total + 1)

        # Update timing
        self.total_tasks += 1
        self.avg_execution_time = (
            self.avg_execution_time * (self.total_tasks - 1) + execution_time
        ) / self.total_tasks

        # Update trust score
        overall_success = sum(s for s, t in self.task_success.values())
        overall_total = sum(t for s, t in self.task_success.values())
        if overall_total > 0:
            self.trust_score = 0.3 + 0.7 * (overall_success / overall_total)

        # Update specialization
        self._update_specialization()

    def _update_specialization(self) -> None:
        """Determine specialization based on performance."""
        if not self.task_success:
            return

        # Single-task-type agents need less data to specialize
        num_task_types = len(self.task_success)
        min_samples = 2 if num_task_types == 1 else 3

        # Find best task type
        best_type = None
        best_rate = 0.0

        for task_type, (succ, total) in self.task_success.items():
            if total >= min_samples:  # Adaptive threshold
                rate = succ / total
                if rate > best_rate:
                    best_rate = rate
                    best_type = task_type

        if best_type and best_rate > 0.7:
            # Map task type to specialization
            specialization_map = {
                "aggregation": AgentSpecialization.AGGREGATOR,
                "analysis": AgentSpecialization.ANALYZER,
                "transformation": AgentSpecialization.TRANSFORMER,
                "validation": AgentSpecialization.VALIDATOR,
                "planning": AgentSpecialization.PLANNER,
                "filtering": AgentSpecialization.EXECUTOR,
                # Real task types from _record_trace() AgentRole values
                "actor": AgentSpecialization.ACTOR,
                "expert": AgentSpecialization.EXPERT,
                "planner": AgentSpecialization.PLANNER,
                "reviewer": AgentSpecialization.REVIEWER,
                "orchestrator": AgentSpecialization.ORCHESTRATOR,
                # Domain-specific task types from swarms
                "paper_learning": AgentSpecialization.RESEARCHER,
                "code_generation": AgentSpecialization.EXECUTOR,
                "test_generation": AgentSpecialization.VALIDATOR,
                "fundamental_analysis": AgentSpecialization.ANALYZER,
                "data_analysis": AgentSpecialization.ANALYZER,
                "devops": AgentSpecialization.EXECUTOR,
                "code_review": AgentSpecialization.REVIEWER,
                "idea_writing": AgentSpecialization.EXPERT,
                "swarm_learning": AgentSpecialization.RESEARCHER,
            }
            self.specialization = specialization_map.get(best_type, AgentSpecialization.GENERALIST)

    def get_success_rate(self, task_type: str) -> float:
        """Get success rate for a specific task type."""
        if task_type not in self.task_success:
            return 0.5  # Unknown
        succ, total = self.task_success[task_type]
        return succ / total if total > 0 else 0.5


@dataclass
class ConsensusVote:
    """A vote in a consensus decision."""

    agent_name: str
    decision: str
    confidence: float
    reasoning: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class SwarmDecision:
    """Result of swarm consensus."""

    question: str
    votes: List[ConsensusVote]
    final_decision: str
    consensus_strength: float  # 0-1, how much agreement
    dissenting_views: List[str]


@dataclass
class AgentSession:
    """Isolated session for an agent (moltbot pattern)."""

    session_id: str
    agent_name: str
    context: str  # "main", "group", "task_{id}"
    messages: List[Dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def add_message(self, from_agent: str, content: str, metadata: Dict = None) -> None:
        """Add message to session."""
        self.messages.append(
            {
                "from": from_agent,
                "content": content,
                "metadata": metadata or {},
                "timestamp": time.time(),
            }
        )
        self.last_active = time.time()

        # Keep bounded
        if len(self.messages) > 100:
            self.messages = self.messages[-100:]


# =============================================================================
# ARXIV SWARM ENHANCEMENTS (SwarmSys, SwarmAgentic patterns)
# =============================================================================


@dataclass
class HandoffContext:
    """
    Context preservation during agent handoff (SwarmAgentic pattern).

    Enables seamless task transfer between agents with full context.
    """

    task_id: str
    from_agent: str
    to_agent: str
    task_type: str
    context: Dict[str, Any] = field(default_factory=dict)
    partial_result: Any = None
    progress: float = 0.0  # 0-1 completion
    priority: int = 5  # 1-10, higher = more urgent
    deadline: float = None  # Unix timestamp
    handoff_chain: List[str] = field(default_factory=list)  # Previous agents
    timestamp: float = field(default_factory=time.time)

    def add_to_chain(self, agent: str) -> None:
        """Track handoff history."""
        if agent not in self.handoff_chain:
            self.handoff_chain.append(agent)


@dataclass
class Coalition:
    """
    Dynamic team of agents (SwarmAgentic coalition formation).

    Groups agents for complex tasks requiring collaboration.
    """

    coalition_id: str
    task_type: str
    leader: str  # Coordinator agent
    members: List[str] = field(default_factory=list)
    roles: Dict[str, str] = field(default_factory=dict)  # agent -> role
    formed_at: float = field(default_factory=time.time)
    active: bool = True
    shared_context: Dict[str, Any] = field(default_factory=dict)

    def add_member(self, agent: str, role: str = "worker") -> None:
        """Add agent to coalition with role."""
        if agent not in self.members:
            self.members.append(agent)
            self.roles[agent] = role

    def remove_member(self, agent: str) -> None:
        """Remove agent from coalition."""
        if agent in self.members:
            self.members.remove(agent)
            self.roles.pop(agent, None)


@dataclass
class AuctionBid:
    """
    Bid in contract-net protocol (SwarmSys auction pattern).

    Agents bid on tasks based on capability and availability.
    """

    agent_name: str
    task_id: str
    bid_value: float  # 0-1, higher = more capable/available
    estimated_time: float  # Seconds
    confidence: float  # 0-1
    specialization_match: float  # 0-1
    current_load: float  # 0-1, 0 = idle
    reasoning: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def score(self) -> float:
        """Combined bid score for ranking."""
        return (
            self.bid_value * 0.3
            + self.confidence * 0.25
            + self.specialization_match * 0.25
            + (1 - self.current_load) * 0.2
        )


@dataclass
class GossipMessage:
    """
    Message for gossip protocol (SwarmSys efficient dissemination).

    Enables O(log n) information spread across swarm.
    """

    message_id: str
    content: Dict[str, Any]
    origin_agent: str
    message_type: str  # "info", "warning", "route", "capability"
    ttl: int = 3  # Hops remaining
    seen_by: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def mark_seen(self, agent: str) -> bool:
        """Mark as seen, return True if should propagate."""
        if agent in self.seen_by:
            return False
        self.seen_by.append(agent)
        self.ttl -= 1
        return self.ttl > 0


@dataclass
class SupervisorNode:
    """
    Node in hierarchical supervisor tree (SwarmSys O(log n) pattern).

    Enables efficient coordination without flat O(n) communication.
    """

    node_id: str
    agent_name: str
    level: int  # 0 = leaf, higher = supervisor
    parent: str = None  # Parent node_id
    children: List[str] = field(default_factory=list)  # Child node_ids
    supervised_agents: List[str] = field(default_factory=list)
    load: float = 0.0  # Current workload 0-1

    def is_leaf(self) -> bool:
        return self.level == 0

    def is_root(self) -> bool:
        return self.parent is None


__all__ = [
    "AgentSpecialization",
    "AgentProfile",
    "ConsensusVote",
    "SwarmDecision",
    "AgentSession",
    # arXiv swarm enhancements
    "HandoffContext",
    "Coalition",
    "AuctionBid",
    "GossipMessage",
    "SupervisorNode",
]

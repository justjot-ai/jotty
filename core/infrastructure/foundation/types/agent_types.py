"""
Jotty v6.0 - Agent-Related Types
=================================

All agent-related dataclasses including agent contributions,
inter-agent communication, and shared scratchpads.
Extracted from data_structures.py for better organization.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .enums import CommunicationType

# =============================================================================
# AGENT CONTRIBUTION (Enhanced Credit Assignment)
# =============================================================================


@dataclass
class AgentContribution:
    """
    Enhanced contribution tracking with reasoning analysis.
    """

    agent_name: str
    contribution_score: float  # -1 to 1

    # Decision analysis
    decision: str  # "approve", "reject", "abstain"
    decision_correct: bool
    counterfactual_impact: float  # Would outcome change without this agent?

    # NEW: Reasoning-based credit (Dr. Chen)
    reasoning_quality: float  # How good was the reasoning
    evidence_used: List[str]  # What evidence was cited
    tools_used: List[str]  # What tools were called

    # NEW: Temporal credit
    decision_timing: float  # When in episode (0-1)
    temporal_weight: float  # Weight based on timing

    def compute_final_contribution(self) -> float:
        """Compute final contribution with all factors."""
        base = self.contribution_score

        # Adjust by reasoning quality
        reasoning_factor = 0.5 + 0.5 * self.reasoning_quality

        # Adjust by counterfactual impact
        impact_factor = 0.5 + 0.5 * self.counterfactual_impact

        # Temporal weighting (early decisions less certain)
        temporal_factor = 0.7 + 0.3 * self.decision_timing

        return base * reasoning_factor * impact_factor * temporal_factor


# =============================================================================
# INTER-AGENT COMMUNICATION (Dr. Chen Enhancement)
# =============================================================================


@dataclass
class AgentMessage:
    """
    NEW: Message for inter-agent communication.
    """

    sender: str
    receiver: str  # "*" for broadcast
    message_type: CommunicationType
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    # For tool result sharing
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    tool_result: Optional[Any] = None

    # For insight sharing
    insight: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class SharedScratchpad:
    """
    NEW: Shared memory space for agent communication.
    """

    messages: List[AgentMessage] = field(default_factory=list)
    tool_cache: Dict[str, Any] = field(default_factory=dict)  # Cache tool results
    shared_insights: List[str] = field(default_factory=list)

    def add_message(self, message: AgentMessage) -> None:
        self.messages.append(message)

        # Cache tool results
        if message.message_type == CommunicationType.TOOL_RESULT:
            cache_key = f"{message.tool_name}:{json.dumps(message.tool_args, sort_keys=True)}"
            self.tool_cache[cache_key] = message.tool_result

    def get_cached_result(self, tool_name: str, tool_args: Dict) -> Optional[Any]:
        """Check if tool result is already cached."""
        cache_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
        return self.tool_cache.get(cache_key)

    def get_messages_for(self, receiver: str) -> List[AgentMessage]:
        """Get all messages for a specific agent."""
        return [m for m in self.messages if m.receiver in (receiver, "*")]

    def clear(self) -> None:
        self.messages.clear()
        self.tool_cache.clear()
        self.shared_insights.clear()

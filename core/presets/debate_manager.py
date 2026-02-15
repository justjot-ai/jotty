"""
Debate Manager - Multi-Round Expert Consensus
==============================================

Manages multi-round expert debates until 100% consensus is achieved.

Based on: /var/www/sites/personal/stock_market/SYNAPSE A-TEAM ROSTER.md

The debate protocol:
1. Each expert reviews the problem
2. Experts raise concerns and suggestions
3. Group discusses solutions
4. Iterate until ALL concerns addressed
5. 100% CONSENSUS REQUIRED

Usage:
    from core.presets.debate_manager import DebateManager, DebateRound

    manager = DebateManager(experts=experts, max_rounds=5)
    result = await manager.run_debate(task_description, goal_context)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import dspy

from .ateam_roster import Expert

logger = logging.getLogger(__name__)


# =============================================================================
# DEBATE STATES
# =============================================================================


class DebateState(Enum):
    """Debate state machine."""

    INITIAL_REVIEW = "initial_review"
    CONCERNS_RAISED = "concerns_raised"
    DISCUSSION = "discussion"
    CONSENSUS_CHECK = "consensus_check"
    CONSENSUS_REACHED = "consensus_reached"
    MAX_ROUNDS_EXCEEDED = "max_rounds_exceeded"


class ExpertVote(Enum):
    """Expert vote on a proposal."""

    APPROVE = "approve"
    CONCERNS = "concerns"
    REJECT = "reject"


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ExpertResponse:
    """Single expert's response in a debate round."""

    expert_name: str
    vote: ExpertVote
    reasoning: str
    concerns: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class DebateRound:
    """A single round of debate."""

    round_number: int
    expert_responses: List[ExpertResponse]
    consensus_reached: bool
    consensus_percentage: float
    summary: str = ""


@dataclass
class ConsensusResult:
    """Final consensus decision."""

    consensus_reached: bool
    total_rounds: int
    final_decision: str
    comprehensive_task: str
    reasoning: str
    expert_votes: Dict[str, str]  # expert_name -> vote
    debate_history: List[DebateRound]


# =============================================================================
# DSPY SIGNATURES FOR EXPERT REVIEWS
# =============================================================================


class ExpertReviewSignature(dspy.Signature):
    """Expert reviews a task proposal."""

    expert_name: str = dspy.InputField(desc="Expert's name")
    expert_role: str = dspy.InputField(desc="Expert's role and expertise")
    expert_questions: str = dspy.InputField(desc="Key questions this expert asks")
    task_description: str = dspy.InputField(desc="Task/problem to review")
    goal_context: str = dspy.InputField(desc="Overall goal and context")
    previous_round_summary: str = dspy.InputField(desc="Summary of previous round (if any)")

    vote: str = dspy.OutputField(desc="Vote: 'approve', 'concerns', or 'reject'")
    reasoning: str = dspy.OutputField(desc="Detailed reasoning for this vote")
    concerns: str = dspy.OutputField(desc="Specific concerns (comma-separated, or 'none')")
    suggestions: str = dspy.OutputField(
        desc="Specific suggestions for improvement (comma-separated, or 'none')"
    )


class ConsensusSynthesisSignature(dspy.Signature):
    """Synthesize expert consensus into final decision."""

    task_description: str = dspy.InputField(desc="Original task/problem")
    goal_context: str = dspy.InputField(desc="Overall goal")
    expert_feedback: str = dspy.InputField(desc="All expert feedback from final round")
    debate_history: str = dspy.InputField(desc="Summary of debate rounds")

    consensus_decision: str = dspy.OutputField(desc="Final consensus decision (what was agreed)")
    comprehensive_task: str = dspy.OutputField(desc="Single comprehensive implementation task")
    reasoning: str = dspy.OutputField(desc="Why this approach was chosen")


# =============================================================================
# DEBATE MANAGER
# =============================================================================


class DebateManager:
    """
    Manages multi-round expert debates.

    Implements the A-TEAM ROSTER debate protocol:
    1. UNDERSTAND CONTEXT
    2. EXPERTISE ROTATION (each expert reviews)
    3. DEBATE PROTOCOL (iterate until consensus)
    4. DOCUMENTATION (track reasoning)
    """

    def __init__(
        self, experts: List[Expert], max_rounds: int = 5, consensus_threshold: float = 1.0
    ) -> None:
        """
        Initialize debate manager.

        Args:
            experts: List of Expert objects to participate
            max_rounds: Maximum debate rounds (default: 5)
            consensus_threshold: Required consensus (default: 1.0 = 100%)
        """
        self.experts = experts
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold

        # DSPy modules
        self.expert_reviewer = dspy.ChainOfThought(ExpertReviewSignature)
        self.consensus_synthesizer = dspy.ChainOfThought(ConsensusSynthesisSignature)

        # State
        self.debate_history: List[DebateRound] = []
        self.current_round = 0

        logger.info(
            f" DebateManager initialized: {len(experts)} experts, "
            f"max_rounds={max_rounds}, threshold={consensus_threshold*100}%"
        )

    async def run_debate(self, task_description: str, goal_context: str) -> ConsensusResult:
        """
        Run multi-round debate until consensus.

        Args:
            task_description: Task/problem to review
            goal_context: Overall goal and context

        Returns:
            ConsensusResult with final decision
        """
        logger.info(f" Starting A-TEAM debate: {len(self.experts)} experts")
        logger.info(f"   Task: {task_description[:100]}...")

        self.current_round = 0
        self.debate_history = []
        previous_summary = "This is the first review round."

        while self.current_round < self.max_rounds:
            self.current_round += 1
            logger.info(f"\n Round {self.current_round}/{self.max_rounds}")

            # Run debate round
            debate_round = await self._run_round(
                task_description=task_description,
                goal_context=goal_context,
                previous_summary=previous_summary,
            )

            self.debate_history.append(debate_round)

            logger.info(f"   Consensus: {debate_round.consensus_percentage*100:.1f}%")

            # Check consensus
            if debate_round.consensus_reached:
                logger.info(f" CONSENSUS REACHED in round {self.current_round}!")
                break

            # Prepare for next round
            previous_summary = self._generate_round_summary(debate_round)

        # Synthesize final decision
        final_result = await self._synthesize_consensus(
            task_description=task_description, goal_context=goal_context
        )

        logger.info(f"\n Debate complete: {final_result.total_rounds} rounds")
        logger.info(f"   Consensus: {final_result.consensus_reached}")

        return final_result

    async def _run_round(
        self, task_description: str, goal_context: str, previous_summary: str
    ) -> DebateRound:
        """Run a single debate round."""
        expert_responses: List[ExpertResponse] = []

        # Each expert reviews
        for expert in self.experts:
            response = await self._get_expert_response(
                expert=expert,
                task_description=task_description,
                goal_context=goal_context,
                previous_summary=previous_summary,
            )
            expert_responses.append(response)

            vote_emoji = {ExpertVote.APPROVE: "", ExpertVote.CONCERNS: "", ExpertVote.REJECT: ""}
            logger.info(f"   {vote_emoji[response.vote]} {expert.name}: {response.vote.value}")

        # Calculate consensus
        approvals = sum(1 for r in expert_responses if r.vote == ExpertVote.APPROVE)
        consensus_percentage = approvals / len(expert_responses)
        consensus_reached = consensus_percentage >= self.consensus_threshold

        debate_round = DebateRound(
            round_number=self.current_round,
            expert_responses=expert_responses,
            consensus_reached=consensus_reached,
            consensus_percentage=consensus_percentage,
        )

        return debate_round

    async def _get_expert_response(
        self, expert: Expert, task_description: str, goal_context: str, previous_summary: str
    ) -> ExpertResponse:
        """Get a single expert's response."""
        # Format expert context
        expert_questions_str = "\n".join([f"- {q}" for q in expert.key_questions])

        # Run expert review
        try:
            result = self.expert_reviewer(
                expert_name=expert.name,
                expert_role=f"{expert.title}: {expert.role}",
                expert_questions=expert_questions_str,
                task_description=task_description,
                goal_context=goal_context,
                previous_round_summary=previous_summary,
            )

            # Parse vote
            vote_str = result.vote.lower().strip()
            if "approve" in vote_str:
                vote = ExpertVote.APPROVE
            elif "reject" in vote_str:
                vote = ExpertVote.REJECT
            else:
                vote = ExpertVote.CONCERNS

            # Parse concerns and suggestions
            concerns = [
                c.strip()
                for c in result.concerns.split(",")
                if c.strip() and c.strip().lower() != "none"
            ]
            suggestions = [
                s.strip()
                for s in result.suggestions.split(",")
                if s.strip() and s.strip().lower() != "none"
            ]

            return ExpertResponse(
                expert_name=expert.name,
                vote=vote,
                reasoning=result.reasoning,
                concerns=concerns,
                suggestions=suggestions,
            )

        except Exception as e:
            logger.error(f"Error getting {expert.name} response: {e}")
            # Default to concerns vote on error
            return ExpertResponse(
                expert_name=expert.name,
                vote=ExpertVote.CONCERNS,
                reasoning=f"Error during review: {str(e)}",
                concerns=["Unable to complete review due to error"],
                suggestions=[],
            )

    def _generate_round_summary(self, debate_round: DebateRound) -> str:
        """Generate summary of a debate round for next iteration."""
        concerns_list = []
        suggestions_list = []

        for response in debate_round.expert_responses:
            if response.vote != ExpertVote.APPROVE:
                concerns_list.extend(response.concerns)
                suggestions_list.extend(response.suggestions)

        summary = f"Round {debate_round.round_number} Summary:\n"
        summary += f"Consensus: {debate_round.consensus_percentage*100:.1f}%\n"

        if concerns_list:
            summary += f"\nKey Concerns Raised:\n"
            for concern in concerns_list[:10]:  # Top 10
                summary += f"- {concern}\n"

        if suggestions_list:
            summary += f"\nSuggestions for Improvement:\n"
            for suggestion in suggestions_list[:10]:  # Top 10
                summary += f"- {suggestion}\n"

        return summary

    async def _synthesize_consensus(
        self, task_description: str, goal_context: str
    ) -> ConsensusResult:
        """Synthesize final consensus from debate history."""
        # Get final round
        final_round = self.debate_history[-1] if self.debate_history else None
        consensus_reached = final_round.consensus_reached if final_round else False

        # Format expert feedback
        expert_feedback_parts = []
        expert_votes = {}

        if final_round:
            for response in final_round.expert_responses:
                expert_votes[response.expert_name] = response.vote.value
                expert_feedback_parts.append(
                    f"**{response.expert_name}** ({response.vote.value}):\n{response.reasoning}"
                )

        expert_feedback = "\n\n".join(expert_feedback_parts)

        # Format debate history
        debate_history_str = "\n".join(
            [
                f"Round {r.round_number}: {r.consensus_percentage*100:.1f}% consensus"
                for r in self.debate_history
            ]
        )

        # Synthesize final decision
        try:
            result = self.consensus_synthesizer(
                task_description=task_description,
                goal_context=goal_context,
                expert_feedback=expert_feedback,
                debate_history=debate_history_str,
            )

            final_decision = result.consensus_decision
            comprehensive_task = result.comprehensive_task
            reasoning = result.reasoning

        except Exception as e:
            logger.error(f"Error synthesizing consensus: {e}")
            final_decision = "Unable to synthesize consensus due to error"
            comprehensive_task = task_description
            reasoning = str(e)

        return ConsensusResult(
            consensus_reached=consensus_reached,
            total_rounds=len(self.debate_history),
            final_decision=final_decision,
            comprehensive_task=comprehensive_task,
            reasoning=reasoning,
            expert_votes=expert_votes,
            debate_history=self.debate_history,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DebateManager",
    "DebateRound",
    "ExpertResponse",
    "ConsensusResult",
    "DebateState",
    "ExpertVote",
]

"""
SwarmIntelligence Consensus Mixin
==================================

Extracted from swarm_intelligence.py — handles consensus gathering
and weighted voting between agents.

MALLM-inspired decision protocols (Becker et al., EMNLP 2025):
    Adds majority, supermajority, unanimity, ranked, and approval voting
    alongside the original weighted-trust protocol.

    decision = await swarm_intelligence.gather_consensus(
        question, options, agents, vote_func,
        protocol='supermajority'
    )
"""

import logging
from typing import Dict, List, Any, Tuple, Callable, Optional
from collections import defaultdict

from .swarm_data_structures import ConsensusVote, SwarmDecision

logger = logging.getLogger(__name__)

# Supported protocols (MALLM-inspired)
DECISION_PROTOCOLS = (
    'weighted',       # Original: confidence × trust (default)
    'majority',       # >50% of votes
    'supermajority',  # ≥2/3 of votes
    'unanimity',      # 100% agreement
    'ranked',         # Ranked-choice (instant runoff)
    'approval',       # Each agent approves multiple options
)


class ConsensusMixin:
    """Mixin for swarm consensus mechanisms."""

    async def gather_consensus(
        self,
        question: str,
        options: List[str],
        agents: List[str],
        vote_func: Callable[[str, str, List[str]], Tuple[str, float, str]],
        protocol: str = 'weighted',
    ) -> SwarmDecision:
        """
        Gather consensus from multiple agents using configurable protocol.

        MALLM insight: Different tasks benefit from different decision
        protocols.  Weighted is best for heterogeneous-trust teams,
        supermajority for high-stakes decisions, unanimity when all
        agents must agree.

        DRY: All protocols share the same vote collection; only the
        tallying logic differs.

        Args:
            question: The question to decide
            options: Available options
            agents: Agents participating in consensus
            vote_func: (agent_name, question, options) -> (decision, confidence, reasoning)
            protocol: Decision protocol — one of DECISION_PROTOCOLS (default 'weighted')

        Returns:
            SwarmDecision with final consensus
        """
        # ---- Collect votes (shared by ALL protocols) ----
        votes = []
        for agent_name in agents:
            try:
                decision, confidence, reasoning = vote_func(agent_name, question, options)
                votes.append(ConsensusVote(
                    agent_name=agent_name,
                    decision=decision,
                    confidence=confidence,
                    reasoning=reasoning
                ))
            except Exception as e:
                logger.warning(f"Agent {agent_name} failed to vote: {e}")

        if not votes:
            return SwarmDecision(
                question=question,
                votes=[],
                final_decision=options[0] if options else "",
                consensus_strength=0.0,
                dissenting_views=[]
            )

        # ---- Tally votes based on protocol ----
        if protocol == 'weighted':
            final_decision, consensus_strength = self._tally_weighted(votes)
        elif protocol == 'majority':
            final_decision, consensus_strength = self._tally_threshold(votes, threshold=0.5)
        elif protocol == 'supermajority':
            final_decision, consensus_strength = self._tally_threshold(votes, threshold=2/3)
        elif protocol == 'unanimity':
            final_decision, consensus_strength = self._tally_threshold(votes, threshold=1.0)
        elif protocol == 'ranked':
            final_decision, consensus_strength = self._tally_ranked(votes, options)
        elif protocol == 'approval':
            final_decision, consensus_strength = self._tally_approval(votes)
        else:
            logger.warning(f"Unknown protocol '{protocol}', falling back to weighted")
            final_decision, consensus_strength = self._tally_weighted(votes)

        # ---- Dissenting views (shared) ----
        dissenting = [
            f"{v.agent_name}: {v.reasoning}"
            for v in votes
            if v.decision != final_decision
        ]

        decision = SwarmDecision(
            question=question,
            votes=votes,
            final_decision=final_decision,
            consensus_strength=consensus_strength,
            dissenting_views=dissenting
        )

        # ---- Update consensus stats (shared) ----
        for vote in votes:
            self.register_agent(vote.agent_name)
            profile = self.agent_profiles[vote.agent_name]
            if vote.decision == final_decision:
                profile.consensus_agreements += 1
            else:
                profile.consensus_disagreements += 1

        self.consensus_history.append(decision)
        return decision

    # =========================================================================
    # TALLYING STRATEGIES (MALLM-inspired)
    # =========================================================================

    def _tally_weighted(self, votes: List[ConsensusVote]) -> Tuple[str, float]:
        """Original: weight = confidence × trust."""
        vote_weights: Dict[str, float] = defaultdict(float)
        for vote in votes:
            self.register_agent(vote.agent_name)
            trust = self.agent_profiles[vote.agent_name].trust_score
            vote_weights[vote.decision] += vote.confidence * trust

        winner = max(vote_weights, key=vote_weights.get)
        total = sum(vote_weights.values())
        strength = vote_weights[winner] / total if total > 0 else 0.0
        return winner, strength

    def _tally_threshold(
        self, votes: List[ConsensusVote], threshold: float
    ) -> Tuple[str, float]:
        """
        Threshold consensus: majority (>50%), supermajority (≥66%), unanimity (100%).

        MALLM uses threshold_percent to parameterize all three.
        """
        counts: Dict[str, int] = defaultdict(int)
        for vote in votes:
            counts[vote.decision] += 1

        total = len(votes)
        winner = max(counts, key=counts.get)
        fraction = counts[winner] / total if total > 0 else 0.0

        # consensus_strength = fraction if threshold met, else scaled down
        if fraction >= threshold:
            return winner, fraction
        else:
            # Below threshold — return best option but low strength
            return winner, fraction * 0.5  # penalize for not meeting threshold

    def _tally_ranked(
        self, votes: List[ConsensusVote], options: List[str]
    ) -> Tuple[str, float]:
        """
        Ranked-choice / instant-runoff voting.

        KISS simplification: Use confidence as rank weight.
        Higher confidence = stronger preference for that option.
        If no majority, eliminate lowest and redistribute.
        """
        if not options:
            return votes[0].decision if votes else "", 0.0

        # Simple: count first-choice votes, eliminate weakest iteratively
        remaining = set(options)
        vote_prefs = [(v.decision, v.confidence) for v in votes]

        for _ in range(len(options) - 1):
            if len(remaining) <= 1:
                break

            counts: Dict[str, float] = defaultdict(float)
            for decision, conf in vote_prefs:
                if decision in remaining:
                    counts[decision] += 1

            if not counts:
                break

            # Check for majority
            total = sum(counts.values())
            leader = max(counts, key=counts.get)
            if counts[leader] / total > 0.5:
                return leader, counts[leader] / total

            # Eliminate weakest
            weakest = min(counts, key=counts.get)
            remaining.discard(weakest)

        winner = remaining.pop() if remaining else votes[0].decision
        fraction = sum(1 for v in votes if v.decision == winner) / len(votes)
        return winner, fraction

    def _tally_approval(self, votes: List[ConsensusVote]) -> Tuple[str, float]:
        """
        Approval voting: count all votes where confidence > 0.5 as "approved".

        MALLM insight: Agents with high confidence approve; low confidence
        is treated as abstention.  Option with most approvals wins.
        """
        approvals: Dict[str, int] = defaultdict(int)
        for vote in votes:
            if vote.confidence >= 0.5:
                approvals[vote.decision] += 1

        if not approvals:
            # No confident votes — fall back to simple count
            counts: Dict[str, int] = defaultdict(int)
            for v in votes:
                counts[v.decision] += 1
            winner = max(counts, key=counts.get)
            return winner, 0.3

        winner = max(approvals, key=approvals.get)
        total_voters = len(votes)
        strength = approvals[winner] / total_voters if total_voters > 0 else 0.0
        return winner, strength

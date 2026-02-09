"""
SwarmIntelligence Consensus Mixin
==================================

Extracted from swarm_intelligence.py — handles consensus gathering
and weighted voting between agents.
"""

import logging
from typing import Dict, List, Any, Tuple, Callable
from collections import defaultdict

from .swarm_data_structures import ConsensusVote, SwarmDecision

logger = logging.getLogger(__name__)


class ConsensusMixin:
    """Mixin for swarm consensus mechanisms."""

    async def gather_consensus(
        self,
        question: str,
        options: List[str],
        agents: List[str],
        vote_func: Callable[[str, str, List[str]], Tuple[str, float, str]]
    ) -> SwarmDecision:
        """
        Gather consensus from multiple agents.

        Args:
            question: The question to decide
            options: Available options
            agents: Agents participating in consensus
            vote_func: Function(agent_name, question, options) -> (decision, confidence, reasoning)

        Returns:
            SwarmDecision with final consensus
        """
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

        # Weighted voting based on confidence and trust
        vote_weights = defaultdict(float)
        for vote in votes:
            self.register_agent(vote.agent_name)
            trust = self.agent_profiles[vote.agent_name].trust_score
            weight = vote.confidence * trust
            vote_weights[vote.decision] += weight

        # Find winner
        final_decision = max(vote_weights.keys(), key=lambda k: vote_weights[k])
        total_weight = sum(vote_weights.values())
        consensus_strength = vote_weights[final_decision] / total_weight if total_weight > 0 else 0.0

        # Find dissenting views
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

        # Update consensus stats
        for vote in votes:
            profile = self.agent_profiles[vote.agent_name]
            if vote.decision == final_decision:
                profile.consensus_agreements += 1
            else:
                profile.consensus_disagreements += 1

        self.consensus_history.append(decision)

        return decision

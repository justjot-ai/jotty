"""
Byzantine Fault Tolerance
==========================

Verify agent claims vs actual results:
- ByzantineVerifier: Byzantine fault tolerance for multi-agent systems

Implements trust-weighted voting and inconsistency detection.

Extracted from swarm_intelligence.py for modularity.
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Any, Tuple, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from .swarm_intelligence import SwarmIntelligence

logger = logging.getLogger(__name__)


class ByzantineVerifier:
    """
    Verify agent claims vs actual results.

    Implements Byzantine fault tolerance for multi-agent systems:
    - Agents can lie about success/failure
    - Verify claims against actual results
    - Apply trust penalties for inconsistent agents
    - Use trust-weighted voting for critical decisions

    This protects the swarm from malicious or malfunctioning agents.
    """

    def __init__(self, swarm_intelligence: 'SwarmIntelligence'):
        self.si = swarm_intelligence

        # Track claim history for verification
        self.claim_history: List[Dict] = []

        # Verification statistics
        self.verified_count = 0
        self.inconsistent_count = 0

    def verify_claim(self, agent: str, claimed_success: bool, actual_result: Any,
                    task_type: str = None) -> bool:
        """
        Verify agent claim and apply trust penalty if inconsistent.

        Args:
            agent: Agent making the claim
            claimed_success: What the agent claimed (success/failure)
            actual_result: The actual result to verify against
            task_type: Type of task for context

        Returns:
            True if claim was consistent, False otherwise
        """
        # Determine actual success from result
        actual_success = self._determine_success(actual_result)

        # Record claim
        claim_record = {
            'agent': agent,
            'claimed': claimed_success,
            'actual': actual_success,
            'task_type': task_type,
            'timestamp': time.time(),
            'consistent': claimed_success == actual_success
        }
        self.claim_history.append(claim_record)

        # Keep bounded
        if len(self.claim_history) > 500:
            self.claim_history = self.claim_history[-500:]

        # Check consistency
        is_consistent = claimed_success == actual_success
        self.verified_count += 1

        if not is_consistent:
            self.inconsistent_count += 1

            # Apply trust penalty
            self.si.register_agent(agent)
            profile = self.si.agent_profiles[agent]
            old_trust = profile.trust_score

            # Penalty depends on severity:
            # - Claiming success when failed: -0.15 (serious)
            # - Claiming failure when succeeded: -0.05 (less serious, might be cautious)
            if claimed_success and not actual_success:
                penalty = 0.15
            else:
                penalty = 0.05

            profile.trust_score = max(0.0, profile.trust_score - penalty)

            logger.warning(
                f"Byzantine: {agent} claim inconsistent "
                f"(claimed={claimed_success}, actual={actual_success}). "
                f"Trust: {old_trust:.2f} -> {profile.trust_score:.2f}"
            )

            # Deposit warning signal
            self.si.deposit_warning_signal(
                agent=agent,
                task_type=task_type or 'unknown',
                warning=f"Inconsistent claim: {claimed_success} vs {actual_success}"
            )

        return is_consistent

    def majority_vote(self, claims: Dict[str, Any]) -> Tuple[Any, float]:
        """
        Weight votes by trust score for critical decisions.

        Args:
            claims: Dict mapping agent_name to their claim/vote

        Returns:
            (winning_claim, confidence) - the trust-weighted winner
        """
        if not claims:
            return None, 0.0

        # Aggregate votes weighted by trust
        vote_weights: Dict[str, float] = defaultdict(float)

        for agent, claim in claims.items():
            self.si.register_agent(agent)
            trust = self.si.agent_profiles[agent].trust_score

            # Convert claim to string key for aggregation
            claim_key = str(claim)
            vote_weights[claim_key] += trust

        if not vote_weights:
            return None, 0.0

        # Find winner
        winner_key = max(vote_weights.keys(), key=lambda k: vote_weights[k])
        total_weight = sum(vote_weights.values())
        confidence = vote_weights[winner_key] / total_weight if total_weight > 0 else 0.0

        # Convert back to original claim value
        for agent, claim in claims.items():
            if str(claim) == winner_key:
                return claim, confidence

        return None, 0.0

    def get_untrusted_agents(self, threshold: float = 0.3) -> List[str]:
        """Get list of agents with trust below threshold."""
        untrusted = []
        for name, profile in self.si.agent_profiles.items():
            if profile.trust_score < threshold:
                untrusted.append(name)
        return untrusted

    def get_agent_consistency_rate(self, agent: str) -> float:
        """Get consistency rate for a specific agent."""
        agent_claims = [c for c in self.claim_history if c['agent'] == agent]
        if not agent_claims:
            return 1.0  # No claims = assume trustworthy

        consistent = sum(1 for c in agent_claims if c['consistent'])
        return consistent / len(agent_claims)

    def _determine_success(self, result: Any) -> bool:
        """Determine success from result object."""
        if result is None:
            return False

        # Handle common result types
        if isinstance(result, bool):
            return result

        if isinstance(result, dict):
            # Check for success field
            if 'success' in result:
                return bool(result['success'])
            if 'error' in result:
                return False
            # Non-empty dict without error = success
            return True

        # Check for success attribute
        if hasattr(result, 'success'):
            return bool(result.success)

        # Default: truthy = success
        return bool(result)

    def format_trust_report(self) -> str:
        """Generate report on agent trustworthiness."""
        lines = [
            "# Byzantine Trust Report",
            "=" * 40,
            f"Total verifications: {self.verified_count}",
            f"Inconsistencies: {self.inconsistent_count}",
            f"Overall rate: {1 - self.inconsistent_count/max(1, self.verified_count):.1%}",
            "",
            "## Agent Trust Scores"
        ]

        # Sort by trust score
        sorted_agents = sorted(
            self.si.agent_profiles.items(),
            key=lambda x: x[1].trust_score,
            reverse=True
        )

        for name, profile in sorted_agents:
            consistency = self.get_agent_consistency_rate(name)
            status = "OK" if profile.trust_score >= 0.5 else "WARNING"
            lines.append(
                f"  {status} {name}: trust={profile.trust_score:.2f}, "
                f"consistency={consistency:.1%}"
            )

        # Untrusted agents warning
        untrusted = self.get_untrusted_agents()
        if untrusted:
            lines.append("")
            lines.append("## Untrusted Agents")
            for agent in untrusted:
                lines.append(f"  - {agent}")

        return "\n".join(lines)


__all__ = [
    'ByzantineVerifier',
]

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

        if is_consistent:
            # Reward honesty: small trust boost for consistent claims
            self.si.register_agent(agent)
            profile = self.si.agent_profiles[agent]
            if profile.trust_score < 1.0:
                boost = 0.05  # Small, asymmetric with penalty (penalize faster than reward)
                profile.trust_score = min(1.0, profile.trust_score + boost)

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

    def to_dict(self) -> Dict:
        """Serialize for persistence."""
        return {
            'claim_history': self.claim_history[-500:],
            'verified_count': self.verified_count,
            'inconsistent_count': self.inconsistent_count,
        }

    def restore_from_dict(self, data: Dict):
        """Restore state from persistence. Requires existing SI reference."""
        self.claim_history = data.get('claim_history', [])
        self.verified_count = data.get('verified_count', 0)
        self.inconsistent_count = data.get('inconsistent_count', 0)


class ConsistencyChecker:
    """
    Multi-agent output consistency checker.

    Repurposed from ByzantineVerifier's trust model. Instead of checking if
    agents "lie" (they can't -- the system observes results directly), this
    checks for **consistency across multiple agents working on the same task**.

    Use cases:
    - Parallel teams: Compare outputs from 2+ agents on the same task
    - Hallucination detection: Flag when agents disagree on factual claims
    - Confidence estimation: High agreement = high confidence in result
    - Quality gating: Reject outputs that deviate too far from consensus

    Architecture:
    - Takes multiple agent outputs for the same task
    - Compares them for agreement/disagreement
    - Uses trust scores to weight contributions
    - Returns consensus result with confidence
    """

    def __init__(self, byzantine_verifier: ByzantineVerifier):
        """
        Args:
            byzantine_verifier: Underlying verifier for trust scores
        """
        self.verifier = byzantine_verifier
        self.consistency_history: List[Dict] = []

    def check_consistency(
        self,
        outputs: Dict[str, Any],
        task_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Check consistency across multiple agent outputs.

        Args:
            outputs: Dict mapping agent_name -> their output for the same task
            task_type: Type of task for context

        Returns:
            Dict with:
            - 'consistent': bool - whether outputs agree
            - 'agreement_rate': float 0-1 - how much agents agree
            - 'consensus_output': Any - the majority/trust-weighted winner
            - 'confidence': float 0-1 - confidence in consensus
            - 'outliers': List[str] - agents whose output disagrees
            - 'details': Dict - per-agent analysis
        """
        if len(outputs) < 2:
            agent = list(outputs.keys())[0] if outputs else "unknown"
            output = list(outputs.values())[0] if outputs else None
            return {
                'consistent': True,
                'agreement_rate': 1.0,
                'consensus_output': output,
                'confidence': 0.5,  # Low confidence with single agent
                'outliers': [],
                'details': {agent: {'agrees': True}} if agent != "unknown" else {},
            }

        # Use ByzantineVerifier's majority_vote for trust-weighted consensus
        consensus_output, confidence = self.verifier.majority_vote(outputs)

        # Calculate agreement rate
        agreements = 0
        total_comparisons = 0
        outliers = []

        for agent, output in outputs.items():
            if str(output) == str(consensus_output):
                agreements += 1
            else:
                outliers.append(agent)
            total_comparisons += 1

        agreement_rate = agreements / total_comparisons if total_comparisons > 0 else 0.0
        consistent = agreement_rate >= 0.5  # Majority agrees

        # Record for trend tracking
        record = {
            'task_type': task_type,
            'agent_count': len(outputs),
            'agreement_rate': agreement_rate,
            'consistent': consistent,
            'outlier_count': len(outliers),
            'timestamp': __import__('time').time(),
        }
        self.consistency_history.append(record)
        if len(self.consistency_history) > 500:
            self.consistency_history = self.consistency_history[-500:]

        # Apply trust penalties to outliers (mild -- they might be right)
        for outlier in outliers:
            if outlier in self.verifier.si.agent_profiles:
                profile = self.verifier.si.agent_profiles[outlier]
                profile.trust_score = max(0.1, profile.trust_score - 0.02)

        # Boost trust for agents in consensus
        for agent in outputs:
            if agent not in outliers and agent in self.verifier.si.agent_profiles:
                profile = self.verifier.si.agent_profiles[agent]
                profile.trust_score = min(1.0, profile.trust_score + 0.01)

        # Build details
        details = {}
        for agent, output in outputs.items():
            details[agent] = {
                'agrees': agent not in outliers,
                'trust_score': self.verifier.si.agent_profiles.get(
                    agent, type('P', (), {'trust_score': 0.5})()
                ).trust_score,
                'output_preview': str(output)[:100],
            }

        result = {
            'consistent': consistent,
            'agreement_rate': agreement_rate,
            'consensus_output': consensus_output,
            'confidence': confidence,
            'outliers': outliers,
            'details': details,
        }

        if not consistent:
            logger.warning(
                f"Inconsistent outputs for {task_type}: "
                f"{len(outliers)}/{len(outputs)} agents disagree "
                f"(agreement={agreement_rate:.0%})"
            )

        return result

    def detect_hallucination(
        self,
        primary_output: Any,
        verification_outputs: Dict[str, Any],
        task_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Detect potential hallucination by comparing primary output against
        verification outputs from other agents.

        Args:
            primary_output: The main output to verify
            verification_outputs: Dict of agent_name -> their verification output
            task_type: Type of task

        Returns:
            Dict with 'likely_hallucination', 'confidence', 'evidence'
        """
        all_outputs = {'primary': primary_output}
        all_outputs.update(verification_outputs)

        result = self.check_consistency(all_outputs, task_type)

        # Primary is hallucinating if it's an outlier with low agreement
        likely_hallucination = (
            'primary' in result['outliers']
            and result['agreement_rate'] > 0.5
            and len(verification_outputs) >= 2
        )

        return {
            'likely_hallucination': likely_hallucination,
            'confidence': result['confidence'],
            'agreement_rate': result['agreement_rate'],
            'evidence': (
                f"Primary output disagrees with {len(result['outliers'])} "
                f"of {len(all_outputs)} agents"
                if likely_hallucination
                else f"Primary output consistent with consensus "
                     f"(agreement={result['agreement_rate']:.0%})"
            ),
        }

    def get_consistency_stats(self) -> Dict[str, Any]:
        """Get overall consistency statistics."""
        if not self.consistency_history:
            return {'total_checks': 0}

        total = len(self.consistency_history)
        consistent_count = sum(1 for r in self.consistency_history if r['consistent'])
        avg_agreement = sum(r['agreement_rate'] for r in self.consistency_history) / total

        # Per task-type breakdown
        by_task = {}
        for record in self.consistency_history:
            tt = record['task_type']
            if tt not in by_task:
                by_task[tt] = {'total': 0, 'consistent': 0}
            by_task[tt]['total'] += 1
            if record['consistent']:
                by_task[tt]['consistent'] += 1

        return {
            'total_checks': total,
            'consistency_rate': consistent_count / total,
            'avg_agreement': avg_agreement,
            'by_task_type': by_task,
        }


__all__ = [
    'ByzantineVerifier',
    'ConsistencyChecker',
]

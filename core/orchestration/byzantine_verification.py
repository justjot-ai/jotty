"""
Byzantine Verification & Consistency Checking
===============================================

Multi-agent output verification:
- ByzantineVerifier: Trust tracking for agent claims vs actual results
- ConsistencyChecker: Semantic comparison of multi-agent outputs

The consistency checker uses key-fact extraction (not string equality)
to compare LLM outputs, since two correct answers are almost never
character-identical.
"""

from __future__ import annotations

import re
import time
import logging
from typing import Dict, List, Any, Tuple, Set, TYPE_CHECKING
from collections import defaultdict, Counter

if TYPE_CHECKING:
    from .swarm_intelligence import SwarmIntelligence

logger = logging.getLogger(__name__)


class ByzantineVerifier:
    """
    Output quality verification and multi-agent trust tracking.

    Two modes of operation:

    1. MULTI-AGENT (original): Compare agent claims vs actual results.
       Trust-weighted voting for critical decisions.

    2. SINGLE-AGENT (added): Heuristic output quality verification.
       Catches cases where agent reports success but output is actually
       empty, an error message, or a restatement of the question.
       This gives byzantine verification real meaning even with one agent.

    Trust score adjustments:
    - Consistent claims / quality output:  +0.05 (slow build)
    - Inconsistent claims / poor output:   -0.15 (fast penalty)
    - Asymmetric by design: trust is hard to earn, easy to lose.
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

    def verify_output_quality(
        self,
        agent: str,
        claimed_success: bool,
        output: Any,
        goal: str = "",
        task_type: str = "general",
        trust_level: str = "safe",
    ) -> Dict[str, Any]:
        """
        Heuristic output quality verification for single-agent mode.

        Instead of comparing agents against each other (meaningless with 1),
        this checks whether the output actually has substance. Catches:
        - Empty or near-empty output
        - Output that's just the error message
        - Output that restates the question without answering
        - Agent claiming success on garbage output

        Trust-level aware (Cline auto-approve pattern):
        - SAFE tools: basic checks only (fast path)
        - SIDE_EFFECT tools: full heuristic suite
        - DESTRUCTIVE tools: stricter thresholds + extra checks

        Returns:
            Dict with 'quality_ok' (bool), 'issues' (list), 'adjusted_success' (bool)
        """
        issues = []
        output_text = ""

        # Extract text from various result types
        if output is None:
            output_text = ""
        elif isinstance(output, str):
            output_text = output
        elif hasattr(output, 'output'):
            output_text = str(getattr(output, 'output', ''))
        elif isinstance(output, dict):
            output_text = str(output.get('output', output.get('result', '')))
        else:
            output_text = str(output)

        output_text = output_text.strip()

        # Trust-level strictness: DESTRUCTIVE tools get tighter thresholds
        _min_length = 20 if trust_level != "destructive" else 50
        _overlap_threshold = 0.8 if trust_level != "destructive" else 0.6

        # SAFE tools: skip deep checks if output has basic substance
        if trust_level == "safe" and claimed_success and len(output_text) >= _min_length:
            # Fast path: minimal verification for read-only tools
            quality_ok = True
            claim_record = {
                'agent': agent, 'claimed': True, 'actual': True,
                'task_type': task_type, 'timestamp': time.time(),
                'consistent': True, 'quality_issues': [],
            }
            self.claim_history.append(claim_record)
            if len(self.claim_history) > 500:
                self.claim_history = self.claim_history[-500:]
            self.verified_count += 1
            if quality_ok:
                self.si.register_agent(agent)
                profile = self.si.agent_profiles[agent]
                profile.trust_score = min(1.0, profile.trust_score + 0.02)
            return {
                'quality_ok': True, 'issues': [],
                'adjusted_success': True, 'trust_level': trust_level,
            }

        # Check 1: Empty or trivially short output
        if len(output_text) < _min_length:
            issues.append("output_too_short")

        # Check 2: Output is mostly an error message
        error_indicators = ['error:', 'exception:', 'traceback', 'failed to', 'could not']
        lower_output = output_text.lower()[:500]
        if any(ind in lower_output for ind in error_indicators) and claimed_success:
            issues.append("error_in_successful_output")

        # Check 3: Output just restates the goal (no actual work done)
        if goal and len(goal) > 10 and len(output_text) > 0:
            goal_words = set(goal.lower().split())
            output_words = set(output_text.lower().split())
            if output_words and goal_words:
                overlap = len(output_words & goal_words) / len(output_words)
                if overlap > _overlap_threshold and len(output_text) < len(goal) * 2:
                    issues.append("output_restates_goal")

        # Check 4: Agent says success but output is actually a refusal
        refusal_phrases = [
            "i cannot", "i can't", "i'm unable", "i am unable",
            "i don't have access", "not possible", "beyond my capabilities",
        ]
        if claimed_success and any(phrase in lower_output for phrase in refusal_phrases):
            issues.append("success_claimed_on_refusal")

        # Check 5 (DESTRUCTIVE only): extra paranoia — check for confirmation keywords
        if trust_level == "destructive" and claimed_success:
            danger_words = ['deleted', 'removed', 'dropped', 'purged', 'destroyed']
            if any(w in lower_output for w in danger_words):
                # Output says something was destroyed — flag for human review
                issues.append("destructive_action_detected")

        quality_ok = len(issues) == 0
        adjusted_success = claimed_success and quality_ok

        # Record as a byzantine claim for trust tracking
        claim_record = {
            'agent': agent,
            'claimed': claimed_success,
            'actual': adjusted_success,
            'task_type': task_type,
            'timestamp': time.time(),
            'consistent': claimed_success == adjusted_success,
            'quality_issues': issues,
        }
        self.claim_history.append(claim_record)
        if len(self.claim_history) > 500:
            self.claim_history = self.claim_history[-500:]

        self.verified_count += 1

        # Adjust trust based on quality
        if not quality_ok and claimed_success:
            # Agent claimed success but output quality is poor
            self.inconsistent_count += 1
            self.si.register_agent(agent)
            profile = self.si.agent_profiles[agent]
            old_trust = profile.trust_score
            profile.trust_score = max(0.0, profile.trust_score - 0.10)
            logger.info(
                f"Byzantine quality check: {agent} claimed success but "
                f"issues={issues}. Trust: {old_trust:.2f} → {profile.trust_score:.2f}"
            )
        elif quality_ok:
            # Good output — small trust boost
            self.si.register_agent(agent)
            profile = self.si.agent_profiles[agent]
            if profile.trust_score < 1.0:
                profile.trust_score = min(1.0, profile.trust_score + 0.03)

        return {
            'quality_ok': quality_ok,
            'issues': issues,
            'adjusted_success': adjusted_success,
        }

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
    Multi-agent output consistency checker using semantic comparison.

    LLM outputs are never character-identical, so string equality is useless.
    Instead, this extracts **key facts** (numbers, proper nouns, conclusions)
    from each output and compares fact overlap using Jaccard similarity.

    Use cases:
    - Parallel teams: Compare outputs from 2+ agents on the same task
    - Hallucination detection: Flag when agents disagree on factual claims
    - Confidence estimation: High fact overlap = high confidence
    - Quality gating: Reject outputs that deviate from consensus

    Comparison approach (no LLM call required):
    1. Extract key facts from each output (numbers, names, conclusions)
    2. Compute pairwise Jaccard similarity on fact sets
    3. Cluster agents by similarity > threshold
    4. Largest cluster = consensus; others = outliers
    """

    # Regex patterns for key fact extraction
    _NUMBER_RE = re.compile(r'\b\d[\d,.]*%?\b')
    _PROPER_NOUN_RE = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
    _CONCLUSION_MARKERS = re.compile(
        r'(?:therefore|thus|in conclusion|the answer is|result is|'
        r'overall|in summary|key takeaway|recommendation)[:\s]*(.*?)(?:\.|$)',
        re.IGNORECASE
    )

    def __init__(self, byzantine_verifier: ByzantineVerifier,
                 similarity_threshold: float = 0.3):
        """
        Args:
            byzantine_verifier: Underlying verifier for trust scores
            similarity_threshold: Jaccard threshold for "agreement" (0-1).
                0.3 is deliberately low because LLM outputs are verbose;
                even agreeing answers share only ~30-50% of extracted facts.
        """
        self.verifier = byzantine_verifier
        self.similarity_threshold = similarity_threshold
        self.consistency_history: List[Dict] = []

    @classmethod
    def _extract_key_facts(cls, text: str) -> Set[str]:
        """
        Extract key facts from text for comparison.

        Returns a set of normalized fact strings (numbers, proper nouns,
        conclusion phrases). Two outputs that agree on substance will
        share most of these even if phrased differently.
        """
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()

        facts: Set[str] = set()

        # Numbers (e.g., "42", "3.14", "85%")
        for m in cls._NUMBER_RE.finditer(text):
            facts.add(f"NUM:{m.group().rstrip('%.,')}")

        # Proper nouns (e.g., "Python", "United States")
        for m in cls._PROPER_NOUN_RE.finditer(text):
            noun = m.group().strip()
            # Skip very short or common words that look like proper nouns
            if len(noun) > 2 and noun not in {'The', 'This', 'That', 'Here', 'There'}:
                facts.add(f"NAME:{noun.lower()}")

        # Conclusion/answer phrases
        for m in cls._CONCLUSION_MARKERS.finditer(text):
            conclusion = m.group(1).strip().lower()[:80]
            if conclusion:
                # Normalize to key words only
                words = set(re.findall(r'[a-z]+', conclusion))
                words -= {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'to', 'of', 'and', 'or', 'in', 'for', 'it'}
                for w in words:
                    if len(w) > 3:
                        facts.add(f"CONC:{w}")

        # If very few facts extracted, fall back to significant words
        if len(facts) < 3:
            words = set(re.findall(r'[a-z]{4,}', text.lower()))
            # Remove stop words
            words -= {'this', 'that', 'with', 'from', 'have', 'been', 'will',
                       'would', 'could', 'should', 'about', 'their', 'there',
                       'which', 'these', 'those', 'then', 'than', 'more', 'also',
                       'very', 'just', 'some', 'other', 'into', 'only', 'over'}
            for w in sorted(words)[:20]:  # Top 20 significant words
                facts.add(f"WORD:{w}")

        return facts

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        """Jaccard similarity between two sets."""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def check_consistency(
        self,
        outputs: Dict[str, Any],
        task_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Check consistency across multiple agent outputs using semantic comparison.

        Args:
            outputs: Dict mapping agent_name -> their output for the same task
            task_type: Type of task for context

        Returns:
            Dict with consistent, agreement_rate, consensus_output,
            confidence, outliers, details
        """
        if len(outputs) < 2:
            agent = list(outputs.keys())[0] if outputs else "unknown"
            output = list(outputs.values())[0] if outputs else None
            return {
                'consistent': True,
                'agreement_rate': 1.0,
                'consensus_output': output,
                'confidence': 0.5,
                'outliers': [],
                'details': {agent: {'agrees': True, 'facts_extracted': 0}} if agent != "unknown" else {},
            }

        # Step 1: Extract facts from each output
        agent_facts: Dict[str, Set[str]] = {}
        for agent, output in outputs.items():
            agent_facts[agent] = self._extract_key_facts(str(output))

        agents = list(outputs.keys())

        # Step 2: Compute pairwise similarity matrix
        similarity: Dict[Tuple[str, str], float] = {}
        for i, a1 in enumerate(agents):
            for a2 in agents[i + 1:]:
                sim = self._jaccard(agent_facts[a1], agent_facts[a2])
                similarity[(a1, a2)] = sim
                similarity[(a2, a1)] = sim

        # Step 3: Find consensus cluster (greedy: start from agent with highest
        # average similarity to others, expand cluster)
        avg_sim = {}
        for a in agents:
            sims = [similarity.get((a, other), 0.0) for other in agents if other != a]
            avg_sim[a] = sum(sims) / len(sims) if sims else 0.0

        # Seed with most central agent
        seed = max(agents, key=lambda a: avg_sim[a])
        cluster = {seed}
        outliers = []

        for a in agents:
            if a == seed:
                continue
            # Agent joins cluster if similar to majority of cluster members
            cluster_sims = [similarity.get((a, c), 0.0) for c in cluster]
            avg_to_cluster = sum(cluster_sims) / len(cluster_sims) if cluster_sims else 0.0
            if avg_to_cluster >= self.similarity_threshold:
                cluster.add(a)
            else:
                outliers.append(a)

        agreement_rate = len(cluster) / len(agents)
        consistent = agreement_rate > 0.5

        # Step 4: Pick consensus output (trust-weighted within cluster)
        cluster_outputs = {a: outputs[a] for a in cluster}
        if cluster_outputs:
            consensus_output, confidence = self.verifier.majority_vote(cluster_outputs)
            # If majority_vote returns None (all different strings), pick highest trust
            if consensus_output is None:
                best_agent = max(cluster, key=lambda a: avg_sim[a])
                consensus_output = outputs[best_agent]
                confidence = agreement_rate
        else:
            consensus_output = list(outputs.values())[0]
            confidence = 0.0

        # Step 5: Trust adjustments (mild)
        for outlier in outliers:
            if outlier in self.verifier.si.agent_profiles:
                profile = self.verifier.si.agent_profiles[outlier]
                profile.trust_score = max(0.1, profile.trust_score - 0.02)

        for agent in cluster:
            if agent in self.verifier.si.agent_profiles:
                profile = self.verifier.si.agent_profiles[agent]
                profile.trust_score = min(1.0, profile.trust_score + 0.01)

        # Record
        record = {
            'task_type': task_type,
            'agent_count': len(outputs),
            'agreement_rate': agreement_rate,
            'consistent': consistent,
            'outlier_count': len(outliers),
            'avg_similarity': sum(avg_sim.values()) / len(avg_sim) if avg_sim else 0.0,
            'timestamp': time.time(),
        }
        self.consistency_history.append(record)
        if len(self.consistency_history) > 500:
            self.consistency_history = self.consistency_history[-500:]

        # Build details
        details = {}
        for agent in agents:
            details[agent] = {
                'agrees': agent in cluster,
                'trust_score': getattr(
                    self.verifier.si.agent_profiles.get(agent), 'trust_score', 0.5
                ),
                'facts_extracted': len(agent_facts[agent]),
                'avg_similarity': avg_sim.get(agent, 0.0),
                'output_preview': str(outputs[agent])[:100],
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
                f"{len(outliers)}/{len(agents)} agents are outliers "
                f"(agreement={agreement_rate:.0%}, avg_sim={record['avg_similarity']:.2f})"
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
        """
        all_outputs = {'primary': primary_output}
        all_outputs.update(verification_outputs)

        result = self.check_consistency(all_outputs, task_type)

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
                f"Primary output disagrees with consensus "
                f"({len(result['outliers'])} outliers of {len(all_outputs)} agents)"
                if likely_hallucination
                else f"Primary consistent with consensus "
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
        avg_similarity = sum(r.get('avg_similarity', 0) for r in self.consistency_history) / total

        by_task = defaultdict(lambda: {'total': 0, 'consistent': 0})
        for record in self.consistency_history:
            tt = record['task_type']
            by_task[tt]['total'] += 1
            if record['consistent']:
                by_task[tt]['consistent'] += 1

        return {
            'total_checks': total,
            'consistency_rate': consistent_count / total,
            'avg_agreement': avg_agreement,
            'avg_fact_similarity': avg_similarity,
            'by_task_type': dict(by_task),
        }


__all__ = [
    'ByzantineVerifier',
    'ConsistencyChecker',
]

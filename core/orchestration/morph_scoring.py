"""
MorphAgent-Inspired Scoring System
====================================

LLM-based scoring system adapted from MorphAgent paper (arXiv:2410.15048):
- MorphScores: Per-agent/swarm score container
- MorphScorer: Computes RCS, RDS, TRAS metrics
- TaskAgentAlignmentSignature: DSPy signature for LLM alignment

Key adaptations:
- NO EMBEDDINGS: Uses LLM-based semantic matching
- Uses task_success distributions instead of profile text embeddings
- Integrates with existing AgentProfile structure

Extracted from swarm_intelligence.py for modularity.
"""

import time
import hashlib
import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from .swarm_data_structures import AgentProfile, AgentSpecialization

logger = logging.getLogger(__name__)


if DSPY_AVAILABLE:
    class TaskAgentAlignmentSignature(dspy.Signature):
        """
        LLM-based task-agent alignment scoring (replaces embedding similarity).
        MorphAgent TRAS semantic component adapted for Jotty.
        """
        task_description: str = dspy.InputField(desc="The task to be executed")
        agent_profile: str = dspy.InputField(desc="Agent's capabilities and specialization")
        agent_history: str = dspy.InputField(desc="Agent's task success history summary")

        reasoning: str = dspy.OutputField(desc="Why this agent is/isn't suited for this task")
        alignment_score: float = dspy.OutputField(desc="Alignment score 0.0-1.0")
        confidence: float = dspy.OutputField(desc="Confidence in assessment 0.0-1.0")


@dataclass
class MorphScores:
    """MorphAgent-inspired scores for an agent or swarm."""
    rcs: float = 0.5  # Role Clarity Score
    rds: float = 0.5  # Role Differentiation Score (swarm-level)
    tras: float = 0.5  # Task-Role Alignment Score

    # Component breakdowns
    rcs_components: Dict[str, float] = field(default_factory=dict)
    tras_components: Dict[str, float] = field(default_factory=dict)

    # Metadata
    computed_at: float = field(default_factory=time.time)
    task_context: str = ""


class MorphScorer:
    """
    MorphAgent-inspired scoring system adapted for Jotty v2.

    Key adaptations from MorphAgent paper (arXiv:2410.15048):
    - NO EMBEDDINGS: Uses LLM-based semantic matching (Jotty philosophy)
    - Uses task_success distributions instead of profile text embeddings
    - Integrates with existing AgentProfile structure

    Three metrics optimized:
    1. RCS (Role Clarity Score): How clear/specific is agent's role
    2. RDS (Role Differentiation Score): How diverse is the swarm
    3. TRAS (Task-Role Alignment): How well does agent match task

    Formula adaptations:
    - RCS: beta1*FOCUS + beta2*CONSISTENCY + beta3*SPECIALIZATION
    - RDS: h(mean pairwise task-distribution dissimilarity)
    - TRAS: alpha*LLM_ALIGNMENT + (1-alpha)*CAPABILITY_MATCH
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config

        # Weights for RCS (Role Clarity Score)
        self.rcs_weights = {
            'focus': 0.4,        # How concentrated on specific task types
            'consistency': 0.3,  # How consistent success rate is
            'specialization': 0.3  # Has clear specialization emerged
        }

        # Weight for TRAS (Task-Role Alignment)
        self.tras_alpha = 0.6  # Weight for LLM alignment vs capability match

        # Cache for expensive computations
        self._rds_cache: Dict[str, Tuple[float, float]] = {}  # hash -> (score, timestamp)
        self._cache_ttl = 300  # 5 minutes

        # LLM scorer (lazy init)
        self._llm_scorer = None

        logger.info("MorphScorer initialized (MorphAgent-inspired, LLM-based)")

    # =========================================================================
    # RCS: ROLE CLARITY SCORE (Per-Agent)
    # =========================================================================

    def compute_rcs(self, profile: 'AgentProfile') -> Tuple[float, Dict[str, float]]:
        """
        Compute Role Clarity Score for an agent.

        MorphAgent RCS measures how clear/specific an agent's role is.
        Adapted formula: RCS = beta1*FOCUS + beta2*CONSISTENCY + beta3*SPECIALIZATION
        """
        components = {}

        # 1. FOCUS: Task concentration (inverse entropy)
        focus = self._compute_focus(profile)
        components['focus'] = focus

        # 2. CONSISTENCY: Success rate stability
        consistency = self._compute_consistency(profile)
        components['consistency'] = consistency

        # 3. SPECIALIZATION: Has clear role emerged
        specialization = self._compute_specialization_clarity(profile)
        components['specialization'] = specialization

        # Weighted combination
        rcs = (
            self.rcs_weights['focus'] * focus +
            self.rcs_weights['consistency'] * consistency +
            self.rcs_weights['specialization'] * specialization
        )

        return rcs, components

    def _compute_focus(self, profile: 'AgentProfile') -> float:
        """
        Compute task focus using inverse normalized entropy.

        Low entropy = agent focuses on few task types = high focus score
        High entropy = agent spreads across many types = low focus score
        """
        if not profile.task_success or profile.total_tasks == 0:
            return 0.5  # Neutral for new agents

        # Get task counts
        task_counts = [total for _, total in profile.task_success.values()]
        total = sum(task_counts)

        if total == 0:
            return 0.5

        # Compute normalized entropy
        probabilities = [count / total for count in task_counts if count > 0]
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

        # Max entropy for uniform distribution
        max_entropy = math.log2(len(probabilities)) if len(probabilities) > 1 else 1.0

        # Normalize and invert (high entropy = low focus)
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
            focus = 1.0 - normalized_entropy
        else:
            focus = 1.0

        return max(0.0, min(1.0, focus))

    def _compute_consistency(self, profile: 'AgentProfile') -> float:
        """
        Compute success rate consistency (low variance = high consistency).
        """
        if not profile.task_success:
            return 0.5

        success_rates = []
        for succ, total in profile.task_success.values():
            if total >= 2:  # Need enough data
                success_rates.append(succ / total)

        if len(success_rates) < 2:
            return 0.5

        # Compute variance
        mean_rate = sum(success_rates) / len(success_rates)
        variance = sum((r - mean_rate) ** 2 for r in success_rates) / len(success_rates)

        # Convert to consistency score (low variance = high consistency)
        # Max variance for [0,1] values is 0.25
        consistency = 1.0 - min(1.0, variance / 0.25)

        return max(0.0, min(1.0, consistency))

    def _compute_specialization_clarity(self, profile: 'AgentProfile') -> float:
        """
        Compute how clearly an agent has specialized.
        """
        if profile.specialization == AgentSpecialization.GENERALIST:
            # Check if close to specializing
            if not profile.task_success:
                return 0.3

            best_rate = 0.0
            for succ, total in profile.task_success.values():
                if total >= 3:
                    rate = succ / total
                    best_rate = max(best_rate, rate)

            # Approaching specialization threshold (0.7)
            if best_rate >= 0.6:
                return 0.6
            elif best_rate >= 0.5:
                return 0.4
            return 0.3

        # Has emerged specialization
        return 0.9

    # =========================================================================
    # RDS: ROLE DIFFERENTIATION SCORE (Swarm-Level)
    # =========================================================================

    def compute_rds(self, profiles: Dict[str, 'AgentProfile']) -> float:
        """
        Compute Role Differentiation Score for the swarm.

        MorphAgent RDS measures how diverse/differentiated agents are.
        Formula: RDS = h3(mean pairwise dissimilarity)
        """
        if len(profiles) < 2:
            return 1.0  # Single agent is maximally differentiated

        # Check cache
        cache_key = self._get_profiles_hash(profiles)
        if cache_key in self._rds_cache:
            score, timestamp = self._rds_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return score

        # Compute pairwise dissimilarities
        profile_list = list(profiles.values())
        n = len(profile_list)
        total_dissimilarity = 0.0
        pair_count = 0

        for i in range(n):
            for j in range(i + 1, n):
                dissim = self._compute_pairwise_dissimilarity(profile_list[i], profile_list[j])
                total_dissimilarity += dissim
                pair_count += 1

        # Average dissimilarity
        avg_dissimilarity = total_dissimilarity / pair_count if pair_count > 0 else 0.5

        # Apply sigmoid normalization (h3 in MorphAgent)
        rds = self._sigmoid_normalize(avg_dissimilarity)

        # Cache result
        self._rds_cache[cache_key] = (rds, time.time())

        return rds

    def _compute_pairwise_dissimilarity(self, p1: 'AgentProfile', p2: 'AgentProfile') -> float:
        """
        Compute dissimilarity between two agent profiles.

        Uses task success distribution comparison (no embeddings).
        d(a_i, a_j) = 1 - cosine_similarity(task_vectors)
        """
        # Build task success vectors
        all_task_types = set(p1.task_success.keys()) | set(p2.task_success.keys())

        if not all_task_types:
            return 0.5  # Unknown

        # Create success rate vectors
        v1 = []
        v2 = []

        for task_type in sorted(all_task_types):
            # Agent 1 success rate
            if task_type in p1.task_success:
                s, t = p1.task_success[task_type]
                v1.append(s / t if t > 0 else 0.5)
            else:
                v1.append(0.0)

            # Agent 2 success rate
            if task_type in p2.task_success:
                s, t = p2.task_success[task_type]
                v2.append(s / t if t > 0 else 0.5)
            else:
                v2.append(0.0)

        # Compute cosine similarity
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a ** 2 for a in v1)) or 1.0
        norm2 = math.sqrt(sum(b ** 2 for b in v2)) or 1.0

        cosine_sim = dot_product / (norm1 * norm2)

        # Dissimilarity = 1 - similarity
        dissimilarity = 1.0 - cosine_sim

        # Also factor in specialization difference
        spec_diff = 0.0 if p1.specialization == p2.specialization else 0.3

        return min(1.0, dissimilarity + spec_diff)

    def _get_profiles_hash(self, profiles: Dict[str, 'AgentProfile']) -> str:
        """Create hash for cache key."""
        key_parts = []
        for name in sorted(profiles.keys()):
            p = profiles[name]
            key_parts.append(f"{name}:{p.total_tasks}:{p.specialization.value}")
        return hashlib.md5('|'.join(key_parts).encode()).hexdigest()

    def _sigmoid_normalize(self, x: float, k: float = 5.0) -> float:
        """Sigmoid normalization h3(x) = 1 / (1 + exp(-k*(x - 0.5)))"""
        return 1.0 / (1.0 + math.exp(-k * (x - 0.5)))

    # =========================================================================
    # TRAS: TASK-ROLE ALIGNMENT SCORE
    # =========================================================================

    def compute_tras(
        self,
        task: str,
        task_type: str,
        profile: 'AgentProfile',
        use_llm: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute Task-Role Alignment Score.

        MorphAgent TRAS measures how well an agent matches a task.
        Formula: TRAS = alpha*S_sim + (1-alpha)*S_cap
        """
        components = {}

        # 1. Capability matching (always computed)
        capability_match = self._compute_capability_match(task_type, profile)
        components['capability_match'] = capability_match

        # 2. Semantic alignment (LLM-based if available)
        if use_llm and DSPY_AVAILABLE:
            try:
                semantic_align = self._compute_llm_alignment(task, profile)
                components['semantic_alignment'] = semantic_align
            except Exception as e:
                logger.debug(f"LLM alignment failed, using fallback: {e}")
                semantic_align = capability_match  # Fallback
                components['semantic_alignment'] = semantic_align
        else:
            # Fallback: use keyword matching
            semantic_align = self._compute_keyword_alignment(task, task_type, profile)
            components['semantic_alignment'] = semantic_align

        # Weighted combination
        tras = (
            self.tras_alpha * semantic_align +
            (1 - self.tras_alpha) * capability_match
        )

        return tras, components

    def _compute_capability_match(self, task_type: str, profile: 'AgentProfile') -> float:
        """
        Compute capability match: does agent's capability match task requirements?
        """
        # Estimate task complexity by type
        task_complexity = {
            'aggregation': 0.3,
            'filtering': 0.3,
            'analysis': 0.6,
            'transformation': 0.5,
            'validation': 0.4,
            'planning': 0.7,
        }.get(task_type, 0.5)

        # Agent capability = success rate for this type (or overall)
        if task_type in profile.task_success:
            s, t = profile.task_success[task_type]
            agent_capability = s / t if t > 0 else 0.5
            has_experience = t >= 3
        else:
            # Use overall success rate
            total_s = sum(s for s, t in profile.task_success.values())
            total_t = sum(t for s, t in profile.task_success.values())
            agent_capability = total_s / total_t if total_t > 0 else 0.5
            has_experience = False

        # Complexity match component
        complexity_match = 1.0 - abs(task_complexity - agent_capability)

        # Combined score
        if has_experience:
            capability_match = 0.7 * agent_capability + 0.3 * complexity_match
        else:
            capability_match = 0.5 * profile.trust_score + 0.5 * complexity_match

        return max(0.0, min(1.0, capability_match))

    def _compute_llm_alignment(self, task: str, profile: 'AgentProfile') -> float:
        """Compute semantic alignment using LLM."""
        if self._llm_scorer is None:
            self._llm_scorer = dspy.ChainOfThought(TaskAgentAlignmentSignature)

        # Format profile for LLM
        profile_desc = self._format_profile_for_llm(profile)
        history_summary = self._format_history_for_llm(profile)

        try:
            result = self._llm_scorer(
                task_description=task[:500],
                agent_profile=profile_desc,
                agent_history=history_summary
            )
            return float(result.alignment_score)
        except Exception:
            return 0.5

    def _compute_keyword_alignment(self, task: str, task_type: str, profile: 'AgentProfile') -> float:
        """Fallback keyword-based alignment when LLM unavailable."""
        if task_type in profile.task_success:
            s, t = profile.task_success[task_type]
            if t >= 3 and s / t >= 0.6:
                return 0.8  # Good match
            elif t >= 3:
                return 0.4  # Poor match
        return 0.5  # Unknown

    def _format_profile_for_llm(self, profile: 'AgentProfile') -> str:
        """Format agent profile for LLM input."""
        parts = [
            f"Agent: {profile.agent_name}",
            f"Specialization: {profile.specialization.value}",
            f"Trust Score: {profile.trust_score:.2f}",
            f"Total Tasks: {profile.total_tasks}",
        ]
        return "; ".join(parts)

    def _format_history_for_llm(self, profile: 'AgentProfile') -> str:
        """Format task history for LLM input."""
        if not profile.task_success:
            return "No task history yet"

        parts = []
        for task_type, (s, t) in profile.task_success.items():
            rate = s / t * 100 if t > 0 else 0
            parts.append(f"{task_type}: {rate:.0f}% ({s}/{t})")
        return "; ".join(parts)

    # =========================================================================
    # COMBINED SCORING
    # =========================================================================

    def compute_all_scores(
        self,
        profiles: Dict[str, 'AgentProfile'],
        task: str = None,
        task_type: str = None
    ) -> Dict[str, MorphScores]:
        """
        Compute all MorphAgent scores for the swarm.
        """
        # Swarm-level RDS (same for all agents)
        rds = self.compute_rds(profiles)

        results = {}
        for name, profile in profiles.items():
            # Per-agent RCS
            rcs, rcs_components = self.compute_rcs(profile)

            # Per-agent TRAS (if task provided)
            if task and task_type:
                tras, tras_components = self.compute_tras(task, task_type, profile)
            else:
                tras = 0.5
                tras_components = {}

            results[name] = MorphScores(
                rcs=rcs,
                rds=rds,
                tras=tras,
                rcs_components=rcs_components,
                tras_components=tras_components,
                task_context=task[:100] if task else ""
            )

        return results

    def get_best_agent_by_tras(
        self,
        profiles: Dict[str, 'AgentProfile'],
        task: str,
        task_type: str,
        min_rcs: float = 0.3
    ) -> Optional[str]:
        """
        Get best agent for task using TRAS scoring.

        Filters by minimum RCS (role clarity) then ranks by TRAS.
        """
        candidates = []

        for name, profile in profiles.items():
            rcs, _ = self.compute_rcs(profile)
            if rcs < min_rcs:
                continue

            tras, _ = self.compute_tras(task, task_type, profile)
            candidates.append((name, tras, rcs))

        if not candidates:
            # Fallback: return agent with highest trust
            return max(profiles.keys(), key=lambda n: profiles[n].trust_score, default=None)

        # Sort by TRAS (primary), RCS (secondary)
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return candidates[0][0]


__all__ = [
    'MorphScores',
    'MorphScorer',
]

# Conditionally export TaskAgentAlignmentSignature
if DSPY_AVAILABLE:
    __all__.append('TaskAgentAlignmentSignature')

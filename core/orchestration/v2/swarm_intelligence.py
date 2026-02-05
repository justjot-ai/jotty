"""
World-Class Swarm Intelligence Module
=====================================

Implements advanced swarm intelligence patterns:

1. EMERGENT SPECIALIZATION: Agents naturally specialize based on performance
2. SWARM CONSENSUS: Agents vote on decisions for better outcomes
3. ONLINE ADAPTATION: Learn during execution, not just after
4. COLLECTIVE MEMORY: Shared experiences across all agents
5. DYNAMIC ROUTING: Route tasks to best-fit agents automatically
6. SESSION ISOLATION: Per-context isolated agent sessions (moltbot pattern)
7. AGENT-TO-AGENT MESSAGING: Direct inter-agent communication tools
8. SELF-CURRICULUM: DrZero-inspired self-generated training tasks
9. MORPHAGENT SCORES: RCS/RDS/TRAS for profile optimization (NEW)

Inspired by: biological swarms, moltbot architecture, multi-agent RL, DrZero, MorphAgent
"""

import asyncio
import time
import logging
import hashlib
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# MORPHAGENT-INSPIRED SCORING (LLM-Based, No Embeddings)
# =============================================================================

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
    - RCS: β₁·FOCUS + β₂·CONSISTENCY + β₃·SPECIALIZATION
    - RDS: h(mean pairwise task-distribution dissimilarity)
    - TRAS: α·LLM_ALIGNMENT + (1-α)·CAPABILITY_MATCH
    """

    def __init__(self, config=None):
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
        Adapted formula: RCS = β₁·FOCUS + β₂·CONSISTENCY + β₃·SPECIALIZATION

        Components:
        - FOCUS: Entropy-based measure of task type concentration (low entropy = focused)
        - CONSISTENCY: Variance of success rates (low variance = consistent)
        - SPECIALIZATION: Whether agent has emerged dominant specialization

        Args:
            profile: AgentProfile to evaluate

        Returns:
            (rcs_score, component_dict)
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
        Formula: RDS = h₃(mean pairwise dissimilarity)

        Higher RDS = agents have complementary, non-overlapping capabilities
        Lower RDS = agents are too similar (redundant)

        Args:
            profiles: Dict of agent_name -> AgentProfile

        Returns:
            rds_score (0.0 to 1.0)
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

        # Apply sigmoid normalization (h₃ in MorphAgent)
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
        """Sigmoid normalization h₃(x) = 1 / (1 + exp(-k*(x - 0.5)))"""
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
        Formula: TRAS = α·S_sim + (1-α)·S_cap

        Components:
        - S_sim: Semantic similarity (LLM-based in Jotty)
        - S_cap: Capability matching (task complexity vs agent capability)

        Args:
            task: Task description
            task_type: Inferred task type
            profile: Agent profile to evaluate
            use_llm: Whether to use LLM for semantic alignment

        Returns:
            (tras_score, component_dict)
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

        Enhanced MorphAgent formula that rewards high capability:
        S_cap = α * success_rate + (1-α) * (1 - |complexity - capability|)

        This ensures:
        - High success rate agents are preferred
        - But also considers complexity matching
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

        # Combined score: weight actual success rate heavily
        # Agents with high success rate should always score higher
        if has_experience:
            # 70% success rate, 30% complexity match
            capability_match = 0.7 * agent_capability + 0.3 * complexity_match
        else:
            # No experience: use trust score as proxy
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
        # Check if agent has success with this task type
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

        Args:
            profiles: All agent profiles
            task: Optional task for TRAS computation
            task_type: Optional task type for TRAS

        Returns:
            Dict of agent_name -> MorphScores
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

        Args:
            profiles: Agent profiles
            task: Task description
            task_type: Task type
            min_rcs: Minimum role clarity threshold

        Returns:
            Best agent name or None
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


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class AgentSpecialization(Enum):
    """Emergent agent specializations."""
    GENERALIST = "generalist"
    AGGREGATOR = "aggregator"  # Good at count/sum/avg
    ANALYZER = "analyzer"      # Good at analysis tasks
    TRANSFORMER = "transformer" # Good at data transformation
    VALIDATOR = "validator"    # Good at validation/checking
    PLANNER = "planner"        # Good at planning/decomposition
    EXECUTOR = "executor"      # Good at execution/action
    # Match real AgentRole task types from _record_trace()
    ACTOR = "actor"            # Action/execution specialist
    EXPERT = "expert"          # Domain knowledge specialist
    REVIEWER = "reviewer"      # Quality/review specialist
    ORCHESTRATOR = "orchestrator"  # Coordination specialist
    RESEARCHER = "researcher"  # Research/learning specialist


@dataclass
class AgentProfile:
    """Dynamic profile that evolves based on performance."""
    agent_name: str
    specialization: AgentSpecialization = AgentSpecialization.GENERALIST

    # Performance tracking by task type
    task_success: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # task_type -> (success, total)

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

    def update_task_result(self, task_type: str, success: bool, execution_time: float):
        """Update profile after task completion."""
        if task_type not in self.task_success:
            self.task_success[task_type] = (0, 0)

        succ, total = self.task_success[task_type]
        self.task_success[task_type] = (succ + (1 if success else 0), total + 1)

        # Update timing
        self.total_tasks += 1
        self.avg_execution_time = (
            (self.avg_execution_time * (self.total_tasks - 1) + execution_time) / self.total_tasks
        )

        # Update trust score
        overall_success = sum(s for s, t in self.task_success.values())
        overall_total = sum(t for s, t in self.task_success.values())
        if overall_total > 0:
            self.trust_score = 0.3 + 0.7 * (overall_success / overall_total)

        # Update specialization
        self._update_specialization()

    def _update_specialization(self):
        """Determine specialization based on performance."""
        if not self.task_success:
            return

        # Find best task type
        best_type = None
        best_rate = 0.0

        for task_type, (succ, total) in self.task_success.items():
            if total >= 3:  # Need enough data
                rate = succ / total
                if rate > best_rate:
                    best_rate = rate
                    best_type = task_type

        if best_type and best_rate > 0.7:
            # Map task type to specialization
            specialization_map = {
                'aggregation': AgentSpecialization.AGGREGATOR,
                'analysis': AgentSpecialization.ANALYZER,
                'transformation': AgentSpecialization.TRANSFORMER,
                'validation': AgentSpecialization.VALIDATOR,
                'planning': AgentSpecialization.PLANNER,
                'filtering': AgentSpecialization.EXECUTOR,
                # Real task types from _record_trace() AgentRole values
                'actor': AgentSpecialization.ACTOR,
                'expert': AgentSpecialization.EXPERT,
                'planner': AgentSpecialization.PLANNER,
                'reviewer': AgentSpecialization.REVIEWER,
                'orchestrator': AgentSpecialization.ORCHESTRATOR,
                # Domain-specific task types from swarms
                'paper_learning': AgentSpecialization.RESEARCHER,
                'code_generation': AgentSpecialization.EXECUTOR,
                'test_generation': AgentSpecialization.VALIDATOR,
                'fundamental_analysis': AgentSpecialization.ANALYZER,
                'data_analysis': AgentSpecialization.ANALYZER,
                'devops': AgentSpecialization.EXECUTOR,
                'code_review': AgentSpecialization.REVIEWER,
                'idea_writing': AgentSpecialization.EXPERT,
                'swarm_learning': AgentSpecialization.RESEARCHER,
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

    def add_message(self, from_agent: str, content: str, metadata: Dict = None):
        """Add message to session."""
        self.messages.append({
            'from': from_agent,
            'content': content,
            'metadata': metadata or {},
            'timestamp': time.time()
        })
        self.last_active = time.time()

        # Keep bounded
        if len(self.messages) > 100:
            self.messages = self.messages[-100:]


# =============================================================================
# STIGMERGY LAYER (Indirect Agent Coordination)
# =============================================================================

@dataclass
class StigmergySignal:
    """A pheromone-like signal in the shared environment."""
    signal_id: str
    signal_type: str  # 'success', 'warning', 'route', 'resource'
    content: Any
    strength: float = 1.0  # Decays over time
    created_at: float = field(default_factory=time.time)
    created_by: str = ""
    metadata: Dict = field(default_factory=dict)

    def decay(self, decay_rate: float = 0.1) -> float:
        """Apply time-based decay to signal strength."""
        age_hours = (time.time() - self.created_at) / 3600.0
        decay_factor = max(0.0, 1.0 - decay_rate * age_hours)
        self.strength *= decay_factor
        return self.strength


class StigmergyLayer:
    """
    Shared artifact store for indirect agent coordination.

    Implements ant-colony-inspired stigmergy:
    - Agents deposit signals ("pheromones") in shared environment
    - Other agents sense and react to signals
    - Signals decay over time (forgotten if not reinforced)
    - Successful paths get reinforced (positive feedback)

    This enables emergent coordination without direct communication.
    """

    def __init__(self, decay_rate: float = 0.1, max_signals: int = 500):
        self.signals: Dict[str, StigmergySignal] = {}
        self.decay_rate = decay_rate
        self.max_signals = max_signals

        # Index by type for efficient querying
        self._type_index: Dict[str, List[str]] = defaultdict(list)

    def deposit(self, signal_type: str, content: Any, agent: str,
                strength: float = 1.0, metadata: Dict = None) -> str:
        """
        Deposit a pheromone signal.

        Args:
            signal_type: Type of signal ('success', 'warning', 'route', etc.)
            content: Signal content (task type, route info, etc.)
            agent: Agent depositing the signal
            strength: Initial signal strength (0-1)
            metadata: Additional context

        Returns:
            Signal ID
        """
        signal_id = hashlib.md5(
            f"{signal_type}:{content}:{agent}:{time.time()}".encode()
        ).hexdigest()[:12]

        signal = StigmergySignal(
            signal_id=signal_id,
            signal_type=signal_type,
            content=content,
            strength=min(1.0, max(0.0, strength)),
            created_by=agent,
            metadata=metadata or {}
        )

        self.signals[signal_id] = signal
        self._type_index[signal_type].append(signal_id)

        # Prune old signals if over limit
        self._prune_weak_signals()

        logger.debug(f"Stigmergy: {agent} deposited {signal_type} signal: {content}")
        return signal_id

    def sense(self, signal_type: str = None, min_strength: float = 0.1) -> List[StigmergySignal]:
        """
        Sense signals with decay applied.

        Args:
            signal_type: Filter by type (None = all types)
            min_strength: Minimum strength to return

        Returns:
            List of signals above threshold
        """
        # Apply decay to all signals
        self._apply_decay()

        # Filter and return
        results = []
        for signal in self.signals.values():
            if signal.strength < min_strength:
                continue
            if signal_type and signal.signal_type != signal_type:
                continue
            results.append(signal)

        # Sort by strength (strongest first)
        results.sort(key=lambda s: s.strength, reverse=True)
        return results

    def reinforce(self, signal_id: str, amount: float = 0.2) -> bool:
        """
        Reinforce existing signal (like ants reinforcing trails).

        Args:
            signal_id: ID of signal to reinforce
            amount: Amount to add to strength (clamped to 1.0)

        Returns:
            True if signal found and reinforced
        """
        if signal_id not in self.signals:
            return False

        signal = self.signals[signal_id]
        signal.strength = min(1.0, signal.strength + amount)
        signal.created_at = time.time()  # Reset decay timer
        return True

    def get_route_signals(self, task_type: str) -> Dict[str, float]:
        """
        Get routing recommendations from environment.

        Returns agent->strength mapping for task routing decisions.
        """
        route_signals = self.sense(signal_type='route', min_strength=0.1)

        recommendations = defaultdict(float)
        for signal in route_signals:
            content = signal.content
            if isinstance(content, dict):
                if content.get('task_type') == task_type:
                    agent = content.get('agent', '')
                    if agent:
                        recommendations[agent] += signal.strength

        return dict(recommendations)

    def _apply_decay(self):
        """Apply time-based decay to all signals."""
        for signal in self.signals.values():
            signal.decay(self.decay_rate)

    def _prune_weak_signals(self):
        """Remove weak signals and keep under limit."""
        # Remove signals below threshold
        to_remove = [sid for sid, s in self.signals.items() if s.strength < 0.01]
        for sid in to_remove:
            del self.signals[sid]

        # If still over limit, remove weakest
        if len(self.signals) > self.max_signals:
            sorted_signals = sorted(
                self.signals.items(),
                key=lambda x: x[1].strength
            )
            excess = len(self.signals) - self.max_signals
            for sid, _ in sorted_signals[:excess]:
                del self.signals[sid]

        # Rebuild type index
        self._type_index = defaultdict(list)
        for sid, signal in self.signals.items():
            self._type_index[signal.signal_type].append(sid)

    def to_dict(self) -> Dict:
        """Serialize for persistence."""
        return {
            'signals': {
                sid: {
                    'signal_id': s.signal_id,
                    'signal_type': s.signal_type,
                    'content': s.content,
                    'strength': s.strength,
                    'created_at': s.created_at,
                    'created_by': s.created_by,
                    'metadata': s.metadata
                }
                for sid, s in self.signals.items()
            },
            'decay_rate': self.decay_rate
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'StigmergyLayer':
        """Deserialize from persistence."""
        instance = cls(decay_rate=data.get('decay_rate', 0.1))

        for sid, s_data in data.get('signals', {}).items():
            signal = StigmergySignal(
                signal_id=s_data['signal_id'],
                signal_type=s_data['signal_type'],
                content=s_data['content'],
                strength=s_data['strength'],
                created_at=s_data['created_at'],
                created_by=s_data['created_by'],
                metadata=s_data.get('metadata', {})
            )
            instance.signals[sid] = signal
            instance._type_index[signal.signal_type].append(sid)

        return instance


# =============================================================================
# SWARM BENCHMARKS
# =============================================================================

@dataclass
class SwarmMetrics:
    """Metrics for evaluating swarm performance."""
    communication_overhead: float = 0.0   # Time spent in inter-agent communication
    specialization_diversity: float = 0.0  # How diverse are agent specializations
    single_vs_multi_ratio: float = 1.0     # Speedup from multi-agent vs single
    cooperation_index: float = 0.0         # How well agents cooperate
    task_distribution_entropy: float = 0.0 # How evenly work is distributed


class SwarmBenchmarks:
    """
    Benchmark suite for comparing swarm performance.

    Tracks:
    - Single-agent vs multi-agent speedup
    - Communication overhead
    - Specialization emergence
    - Cooperation effectiveness
    """

    def __init__(self):
        # Single-agent baselines: task_type -> [(time, success)]
        self.single_agent_runs: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)

        # Multi-agent runs: task_type -> [(time, agents_count, success)]
        self.multi_agent_runs: Dict[str, List[Tuple[float, int, bool]]] = defaultdict(list)

        # Communication overhead tracking
        self.communication_events: List[Dict] = []

        # Cooperation events
        self.cooperation_events: List[Dict] = []

    def record_single_agent_run(self, task_type: str, execution_time: float, success: bool = True):
        """Record single-agent baseline run."""
        self.single_agent_runs[task_type].append((execution_time, success))

        # Keep bounded
        if len(self.single_agent_runs[task_type]) > 100:
            self.single_agent_runs[task_type] = self.single_agent_runs[task_type][-100:]

    def record_multi_agent_run(self, task_type: str, execution_time: float,
                               agents_count: int, success: bool = True):
        """Record multi-agent run."""
        self.multi_agent_runs[task_type].append((execution_time, agents_count, success))

        # Keep bounded
        if len(self.multi_agent_runs[task_type]) > 100:
            self.multi_agent_runs[task_type] = self.multi_agent_runs[task_type][-100:]

    def record_communication(self, from_agent: str, to_agent: str, message_size: int = 0):
        """Record inter-agent communication event."""
        self.communication_events.append({
            'from': from_agent,
            'to': to_agent,
            'size': message_size,
            'timestamp': time.time()
        })

        # Keep bounded
        if len(self.communication_events) > 1000:
            self.communication_events = self.communication_events[-1000:]

    def record_cooperation(self, helper: str, helped: str, task_type: str, success: bool):
        """Record cooperation event between agents."""
        self.cooperation_events.append({
            'helper': helper,
            'helped': helped,
            'task_type': task_type,
            'success': success,
            'timestamp': time.time()
        })

        # Keep bounded
        if len(self.cooperation_events) > 500:
            self.cooperation_events = self.cooperation_events[-500:]

    def compute_metrics(self, agent_profiles: Dict[str, 'AgentProfile'] = None) -> SwarmMetrics:
        """Compute current swarm metrics."""
        metrics = SwarmMetrics()

        # 1. Single vs Multi speedup ratio
        speedups = []
        for task_type in self.multi_agent_runs:
            single_times = [t for t, s in self.single_agent_runs.get(task_type, []) if s]
            multi_times = [t for t, n, s in self.multi_agent_runs.get(task_type, []) if s]

            if single_times and multi_times:
                avg_single = sum(single_times) / len(single_times)
                avg_multi = sum(multi_times) / len(multi_times)
                if avg_multi > 0:
                    speedups.append(avg_single / avg_multi)

        if speedups:
            metrics.single_vs_multi_ratio = sum(speedups) / len(speedups)

        # 2. Communication overhead (messages per hour)
        recent_comms = [c for c in self.communication_events
                       if time.time() - c['timestamp'] < 3600]
        metrics.communication_overhead = len(recent_comms)

        # 3. Specialization diversity (entropy of specializations)
        if agent_profiles:
            spec_counts = defaultdict(int)
            for profile in agent_profiles.values():
                spec_counts[profile.specialization.value] += 1

            total = sum(spec_counts.values())
            if total > 0:
                import math
                entropy = 0.0
                for count in spec_counts.values():
                    p = count / total
                    if p > 0:
                        entropy -= p * math.log2(p)
                # Normalize by max entropy (log2(num_specializations))
                max_entropy = math.log2(len(AgentSpecialization))
                metrics.specialization_diversity = entropy / max_entropy if max_entropy > 0 else 0

        # 4. Cooperation index
        recent_coop = [c for c in self.cooperation_events
                      if time.time() - c['timestamp'] < 3600]
        if recent_coop:
            successful = sum(1 for c in recent_coop if c['success'])
            metrics.cooperation_index = successful / len(recent_coop)

        return metrics

    def format_benchmark_report(self, agent_profiles: Dict[str, 'AgentProfile'] = None) -> str:
        """Generate human-readable benchmark report."""
        metrics = self.compute_metrics(agent_profiles)

        lines = [
            "# Swarm Benchmark Report",
            "=" * 40,
            "",
            f"## Performance Metrics",
            f"  - Multi-agent speedup ratio: {metrics.single_vs_multi_ratio:.2f}x",
            f"  - Communication overhead: {metrics.communication_overhead:.0f} msgs/hour",
            f"  - Cooperation index: {metrics.cooperation_index:.2%}",
            f"  - Specialization diversity: {metrics.specialization_diversity:.2%}",
            "",
        ]

        # Task-specific breakdown
        if self.multi_agent_runs:
            lines.append("## Task Performance")
            for task_type in sorted(self.multi_agent_runs.keys()):
                multi = self.multi_agent_runs[task_type]
                single = self.single_agent_runs.get(task_type, [])

                multi_success = sum(1 for _, _, s in multi if s) / len(multi) if multi else 0
                single_success = sum(1 for _, s in single if s) / len(single) if single else 0

                lines.append(f"  {task_type}:")
                lines.append(f"    - Multi-agent success: {multi_success:.1%} ({len(multi)} runs)")
                lines.append(f"    - Single-agent success: {single_success:.1%} ({len(single)} runs)")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialize for persistence."""
        return {
            'single_agent_runs': dict(self.single_agent_runs),
            'multi_agent_runs': dict(self.multi_agent_runs),
            'communication_events': self.communication_events[-200:],
            'cooperation_events': self.cooperation_events[-100:]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SwarmBenchmarks':
        """Deserialize from persistence."""
        instance = cls()
        instance.single_agent_runs = defaultdict(list, data.get('single_agent_runs', {}))
        instance.multi_agent_runs = defaultdict(list, data.get('multi_agent_runs', {}))
        instance.communication_events = data.get('communication_events', [])
        instance.cooperation_events = data.get('cooperation_events', [])
        return instance


# =============================================================================
# BYZANTINE FAULT TOLERANCE
# =============================================================================

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
                f"Trust: {old_trust:.2f} → {profile.trust_score:.2f}"
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
            status = "✓" if profile.trust_score >= 0.5 else "⚠️"
            lines.append(
                f"  {status} {name}: trust={profile.trust_score:.2f}, "
                f"consistency={consistency:.1%}"
            )

        # Untrusted agents warning
        untrusted = self.get_untrusted_agents()
        if untrusted:
            lines.append("")
            lines.append("## ⚠️ Untrusted Agents")
            for agent in untrusted:
                lines.append(f"  - {agent}")

        return "\n".join(lines)


# =============================================================================
# CURRICULUM GENERATOR (DrZero-Inspired)
# =============================================================================

@dataclass
class SyntheticTask:
    """A self-generated task for agent training."""
    task_id: str
    task_type: str
    description: str
    difficulty: float  # 0.0 (easy) to 1.0 (hard)
    target_agent: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class CurriculumGenerator:
    """
    DrZero + Agent0 inspired self-curriculum generator.

    Generates progressively harder tasks to train agents without external data.

    Key concepts from DrZero:
    1. PROPOSER ROLE: Generate tasks slightly harder than current ability
    2. DIFFICULTY SCALING: Tasks adapt to agent performance
    3. WEAKNESS TARGETING: Focus on low-success task types
    4. DIVERSITY: Ensure coverage of all task types

    Agent0 enhancements (arXiv:2511.16043):
    5. TOOL-AWARE TASKS: Tasks designed for tool usage
    6. MEMORY-INFORMED: Query memory for weakness patterns
    7. EXECUTOR FEEDBACK: Closed-loop curriculum adaptation

    This enables AUTONOMOUS SKILL IMPROVEMENT without user intervention.
    """

    def __init__(self, config=None, state_manager=None, memory_system=None):
        self.config = config

        # Agent0: Connect to existing infrastructure (DRY - don't duplicate)
        self._state_manager = state_manager  # SwarmStateManager for tool stats
        self._memory_system = memory_system  # HierarchicalMemory for context

        # Task type templates (domain-agnostic)
        self.task_templates: Dict[str, List[str]] = {
            'aggregation': [
                "Count items matching criteria: {criteria}",
                "Sum values where: {condition}",
                "Calculate average of: {field}",
            ],
            'analysis': [
                "Analyze patterns in: {domain}",
                "Find correlations between: {field_a} and {field_b}",
                "Identify anomalies in: {dataset}",
            ],
            'transformation': [
                "Transform data from {format_a} to {format_b}",
                "Normalize values in: {field}",
                "Merge datasets: {dataset_a} and {dataset_b}",
            ],
            'validation': [
                "Validate data quality for: {field}",
                "Check constraints: {constraints}",
                "Verify consistency between: {source_a} and {source_b}",
            ],
            'filtering': [
                "Filter records where: {condition}",
                "Select top {n} by: {criteria}",
                "Remove duplicates from: {dataset}",
            ],
            'planning': [
                "Plan execution steps for: {goal}",
                "Decompose task: {complex_task}",
                "Prioritize items: {items}",
            ],
        }

        # Difficulty progression tracking
        self.difficulty_by_type: Dict[str, float] = defaultdict(lambda: 0.3)  # Start at 30%

        # Task history for diversity
        self.generated_tasks: List[SyntheticTask] = []
        self.max_history = 100

        # Curriculum statistics
        self.total_generated = 0
        self.tasks_by_difficulty: Dict[str, int] = defaultdict(int)

        # Agent0: Tool-aware task templates (uses existing tools, doesn't duplicate)
        self.tool_task_templates: Dict[str, Dict[str, Any]] = {
            'search_analyze': {
                'description': "Search for {topic} and analyze the results",
                'tools_hint': ['search', 'web_search', 'grep'],
                'complexity': 'chain',
            },
            'read_transform': {
                'description': "Read {source} and transform to {format}",
                'tools_hint': ['read', 'file_read', 'converter'],
                'complexity': 'chain',
            },
            'execute_validate': {
                'description': "Execute {command} and validate output matches {criteria}",
                'tools_hint': ['bash', 'execute', 'validate'],
                'complexity': 'chain',
            },
            'multi_source': {
                'description': "Gather information from multiple sources about {topic}",
                'tools_hint': ['search', 'read', 'fetch'],
                'complexity': 'parallel',
            },
        }

        # Agent0: Track tool success rates from executor feedback
        self._tool_success_rates: Dict[str, Tuple[int, int]] = {}  # tool -> (success, total)
        self._executor_feedback_history: List[Dict] = []
        self._max_feedback_history = 100

        logger.info("CurriculumGenerator initialized (DrZero + Agent0 self-curriculum)")

    def generate_training_task(
        self,
        profiles: Dict[str, 'AgentProfile'],
        target_agent: Optional[str] = None
    ) -> SyntheticTask:
        """
        Generate a training task targeting current agent weaknesses.

        DrZero insight: Tasks should be slightly harder than current ability
        to maximize learning signal (zone of proximal development).

        Args:
            profiles: Current agent performance profiles
            target_agent: Optionally target specific agent

        Returns:
            SyntheticTask for training
        """
        # 1. Identify weakest task type across agents
        task_type, difficulty = self._select_task_type_by_weakness(profiles, target_agent)

        # 2. Generate task description from template
        description = self._generate_description(task_type, difficulty)

        # 3. Create synthetic task
        task = SyntheticTask(
            task_id=f"curriculum_{self.total_generated}_{int(time.time())}",
            task_type=task_type,
            description=description,
            difficulty=difficulty,
            target_agent=target_agent,
            metadata={
                'curriculum_round': self.total_generated,
                'weakness_targeted': task_type,
            }
        )

        # 4. Track for diversity
        self.generated_tasks.append(task)
        if len(self.generated_tasks) > self.max_history:
            self.generated_tasks = self.generated_tasks[-self.max_history:]

        self.total_generated += 1
        self.tasks_by_difficulty[f"{int(difficulty * 10) / 10:.1f}"] += 1

        logger.debug(f"Generated curriculum task: type={task_type}, difficulty={difficulty:.2f}")
        return task

    def _select_task_type_by_weakness(
        self,
        profiles: Dict[str, 'AgentProfile'],
        target_agent: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Select task type based on agent weaknesses.

        DrZero insight: Focus on areas where agents struggle most.
        """
        # Aggregate success rates by task type
        type_success_rates: Dict[str, List[float]] = defaultdict(list)

        for agent_name, profile in profiles.items():
            if target_agent and agent_name != target_agent:
                continue

            for task_type, (success, total) in profile.task_success.items():
                if total > 0:
                    rate = success / total
                    type_success_rates[task_type].append(rate)

        # Find weakest task type (lowest average success rate)
        weakest_type = None
        lowest_rate = 1.0

        for task_type, rates in type_success_rates.items():
            avg_rate = sum(rates) / len(rates) if rates else 0.5
            if avg_rate < lowest_rate:
                lowest_rate = avg_rate
                weakest_type = task_type

        # If no data, pick random type for exploration
        if weakest_type is None:
            import random
            weakest_type = random.choice(list(self.task_templates.keys()))
            lowest_rate = 0.5

        # Ensure diversity: occasionally pick other types
        import random
        if random.random() < 0.2:  # 20% exploration
            weakest_type = random.choice(list(self.task_templates.keys()))

        # Calculate difficulty: slightly above current ability
        # DrZero insight: optimal learning at current_ability + epsilon
        current_ability = lowest_rate
        difficulty = min(1.0, current_ability + 0.15)  # 15% harder than current

        # Update tracked difficulty for progressive curriculum
        self.difficulty_by_type[weakest_type] = difficulty

        return weakest_type, difficulty

    def _generate_description(self, task_type: str, difficulty: float) -> str:
        """
        Generate task description from template with difficulty scaling.
        """
        import random

        templates = self.task_templates.get(task_type, ["Perform {task_type} task"])
        template = random.choice(templates)

        # Generate placeholder values based on difficulty
        placeholders = self._generate_placeholders(difficulty)

        try:
            description = template.format(**placeholders)
        except KeyError:
            description = f"Perform {task_type} task (difficulty: {difficulty:.1%})"

        return description

    def _generate_placeholders(self, difficulty: float) -> Dict[str, str]:
        """
        Generate placeholder values scaled by difficulty.

        Higher difficulty = more complex constraints.
        """
        import random

        # Sample domains/fields
        domains = ['users', 'transactions', 'events', 'logs', 'metrics', 'records']
        fields = ['timestamp', 'value', 'count', 'status', 'category', 'score']
        formats = ['json', 'csv', 'parquet', 'sql', 'xml']

        # Complexity scales with difficulty
        num_conditions = max(1, int(difficulty * 3))

        conditions = []
        for _ in range(num_conditions):
            field = random.choice(fields)
            op = random.choice(['>', '<', '=', '!=', 'contains', 'between'])
            conditions.append(f"{field} {op} value")

        return {
            'criteria': ' AND '.join(conditions[:2]),
            'condition': conditions[0] if conditions else 'value > 0',
            'field': random.choice(fields),
            'field_a': random.choice(fields),
            'field_b': random.choice(fields),
            'domain': random.choice(domains),
            'dataset': random.choice(domains),
            'dataset_a': random.choice(domains),
            'dataset_b': random.choice(domains),
            'format_a': random.choice(formats),
            'format_b': random.choice(formats),
            'constraints': ', '.join(conditions[:num_conditions]),
            'source_a': random.choice(domains),
            'source_b': random.choice(domains),
            'n': str(random.randint(5, 20) * int(difficulty * 2 + 1)),
            'goal': f"Complete {random.choice(domains)} processing",
            'complex_task': f"Analyze and transform {random.choice(domains)}",
            'items': ', '.join(random.sample(fields, min(3, len(fields)))),
            'task_type': random.choice(list(self.task_templates.keys())),
        }

    def update_from_result(self, task: SyntheticTask, success: bool, execution_time: float):
        """
        Update curriculum based on task result.

        DrZero insight: Adjust difficulty based on success rate.
        - Too easy (always succeeds) → increase difficulty
        - Too hard (always fails) → decrease difficulty
        """
        task_type = task.task_type
        current_difficulty = self.difficulty_by_type[task_type]

        if success:
            # Task was achievable, increase difficulty slightly
            self.difficulty_by_type[task_type] = min(1.0, current_difficulty + 0.05)
        else:
            # Task was too hard, decrease difficulty
            self.difficulty_by_type[task_type] = max(0.1, current_difficulty - 0.1)

        logger.debug(
            f"Curriculum updated: {task_type} difficulty "
            f"{current_difficulty:.2f} → {self.difficulty_by_type[task_type]:.2f}"
        )

    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get statistics about the curriculum."""
        return {
            'total_generated': self.total_generated,
            'difficulty_by_type': dict(self.difficulty_by_type),
            'tasks_by_difficulty': dict(self.tasks_by_difficulty),
            'recent_task_types': [t.task_type for t in self.generated_tasks[-10:]],
            'tool_success_rates': dict(self._tool_success_rates),
            'feedback_count': len(self._executor_feedback_history),
        }

    # =========================================================================
    # Agent0 Enhancements: Tool-awareness & Memory Integration
    # =========================================================================

    def connect_state_manager(self, state_manager):
        """
        Connect to SwarmStateManager for tool success tracking.

        DRY: Uses existing AgentStateTracker.tool_usage instead of duplicating.
        """
        self._state_manager = state_manager
        logger.debug("CurriculumGenerator connected to SwarmStateManager")

    def connect_memory(self, memory_system):
        """
        Connect to memory system for weakness detection.

        DRY: Uses existing HierarchicalMemory/SimpleBrain.
        """
        self._memory_system = memory_system
        logger.debug("CurriculumGenerator connected to memory system")

    def receive_executor_feedback(
        self,
        task_id: str,
        success: bool,
        tools_used: List[str],
        execution_time: float = 0.0,
        error_type: str = None
    ):
        """
        Agent0: Receive feedback from executor to adapt curriculum.

        Closes the loop: Executor performance → Curriculum adaptation.
        """
        feedback = {
            'task_id': task_id,
            'success': success,
            'tools_used': tools_used,
            'execution_time': execution_time,
            'error_type': error_type,
            'timestamp': time.time(),
        }

        self._executor_feedback_history.append(feedback)
        if len(self._executor_feedback_history) > self._max_feedback_history:
            self._executor_feedback_history = self._executor_feedback_history[-self._max_feedback_history:]

        # Update tool success rates
        for tool in tools_used:
            current = self._tool_success_rates.get(tool, (0, 0))
            if success:
                self._tool_success_rates[tool] = (current[0] + 1, current[1] + 1)
            else:
                self._tool_success_rates[tool] = (current[0], current[1] + 1)

        logger.debug(f"Received executor feedback: success={success}, tools={tools_used}")

    def _sync_tool_stats_from_state_manager(self):
        """
        Sync tool success rates from AgentStateTracker.

        DRY: Pulls from existing infrastructure instead of maintaining duplicate state.
        """
        if not self._state_manager:
            return

        try:
            # Get all agent trackers from state manager
            if hasattr(self._state_manager, 'agent_trackers'):
                for agent_name, tracker in self._state_manager.agent_trackers.items():
                    state = tracker.get_state()
                    tool_usage = state.get('tool_usage', {})

                    for tool, count in tool_usage.get('successful', {}).items():
                        current = self._tool_success_rates.get(tool, (0, 0))
                        # Merge counts (avoid duplicates by using max)
                        self._tool_success_rates[tool] = (
                            max(current[0], count),
                            max(current[1], count + tool_usage.get('failed', {}).get(tool, 0))
                        )
        except Exception as e:
            logger.debug(f"Could not sync tool stats: {e}")

    def _query_memory_for_weaknesses(self, target_agent: str = None) -> List[str]:
        """
        Agent0: Query memory for patterns where agent struggled.

        DRY: Uses existing memory.recall() or memory.query() API.
        """
        if not self._memory_system:
            return []

        weaknesses = []
        try:
            # Query for error patterns
            query = f"errors failures mistakes {target_agent or 'agent'}"

            if hasattr(self._memory_system, 'recall'):
                results = self._memory_system.recall(query, top_k=5)
                if results:
                    weaknesses = [str(r)[:100] for r in results[:3]]

            elif hasattr(self._memory_system, 'query'):
                results = self._memory_system.query(query, limit=5)
                if results:
                    weaknesses = [r.get('content', '')[:100] for r in results[:3]]

        except Exception as e:
            logger.debug(f"Could not query memory for weaknesses: {e}")

        return weaknesses

    def generate_tool_aware_task(
        self,
        profiles: Dict[str, 'AgentProfile'],
        target_agent: Optional[str] = None,
        prefer_weak_tools: bool = True
    ) -> SyntheticTask:
        """
        Agent0: Generate a task designed for tool usage.

        Uses existing SyntheticTask with tool hints in metadata (DRY - no new class).
        """
        import random

        # Sync tool stats from state manager
        self._sync_tool_stats_from_state_manager()

        # Find tools with low success rate (weaknesses)
        weak_tools = []
        if prefer_weak_tools:
            for tool, (success, total) in self._tool_success_rates.items():
                if total > 0 and success / total < 0.6:
                    weak_tools.append(tool)

        # Select template - prefer ones using weak tools
        template_name, template = self._select_tool_template(weak_tools)

        # Query memory for additional context
        memory_hints = self._query_memory_for_weaknesses(target_agent)

        # Calculate difficulty based on tool complexity
        complexity_difficulty = {
            'single': 0.3,
            'chain': 0.5,
            'parallel': 0.7,
            'conditional': 0.8,
        }
        base_difficulty = complexity_difficulty.get(template.get('complexity', 'single'), 0.4)

        # Adjust based on executor feedback
        recent_success_rate = self._get_recent_success_rate()
        difficulty = min(1.0, base_difficulty + (recent_success_rate - 0.5) * 0.2)

        # Generate description
        placeholders = self._generate_placeholders(difficulty)
        try:
            description = template['description'].format(**placeholders)
        except KeyError:
            description = template['description']

        # Create task with tool hints in metadata
        task = SyntheticTask(
            task_id=f"tool_curriculum_{self.total_generated}_{int(time.time())}",
            task_type=template_name,
            description=description,
            difficulty=difficulty,
            target_agent=target_agent,
            metadata={
                'curriculum_round': self.total_generated,
                'tool_aware': True,
                'tools_hint': template.get('tools_hint', []),
                'complexity': template.get('complexity', 'single'),
                'weak_tools_targeted': weak_tools[:3],
                'memory_context': memory_hints[0] if memory_hints else None,
            }
        )

        self.generated_tasks.append(task)
        self.total_generated += 1

        logger.debug(f"Generated tool-aware task: {template_name}, difficulty={difficulty:.2f}")
        return task

    def _select_tool_template(self, weak_tools: List[str]) -> Tuple[str, Dict]:
        """Select template preferring ones that use weak tools."""
        import random

        # Score templates by weak tool overlap
        scored = []
        for name, template in self.tool_task_templates.items():
            tools_hint = template.get('tools_hint', [])
            overlap = len(set(weak_tools) & set(tools_hint))
            scored.append((name, template, overlap))

        # Sort by overlap (descending)
        scored.sort(key=lambda x: x[2], reverse=True)

        # 20% exploration - pick random
        if random.random() < 0.2:
            name, template, _ = random.choice(scored)
        else:
            name, template, _ = scored[0]

        return name, template

    def _get_recent_success_rate(self) -> float:
        """Get success rate from recent executor feedback."""
        recent = self._executor_feedback_history[-20:]
        if not recent:
            return 0.5
        return sum(1 for f in recent if f['success']) / len(recent)

    def to_dict(self) -> Dict:
        """Serialize for persistence - includes full learning history."""
        # Convert tool_success_rates tuples to lists for JSON
        serializable_rates = {}
        for tool, rate in self._tool_success_rates.items():
            if isinstance(rate, tuple):
                serializable_rates[tool] = list(rate)
            else:
                serializable_rates[tool] = rate

        return {
            'difficulty_by_type': dict(self.difficulty_by_type),
            'total_generated': self.total_generated,
            'tasks_by_difficulty': dict(self.tasks_by_difficulty),
            'tool_success_rates': serializable_rates,
            'feedback_history': self._executor_feedback_history[-100:],
        }

    @classmethod
    def from_dict(cls, data: Dict, config=None, state_manager=None, memory_system=None) -> 'CurriculumGenerator':
        """Deserialize from persistence - restores full learning state."""
        instance = cls(config, state_manager=state_manager, memory_system=memory_system)
        instance.difficulty_by_type = defaultdict(lambda: 0.3, data.get('difficulty_by_type', {}))
        instance.total_generated = data.get('total_generated', 0)
        instance.tasks_by_difficulty = defaultdict(int, data.get('tasks_by_difficulty', {}))

        # Restore tool_success_rates (convert lists back to tuples)
        raw_rates = data.get('tool_success_rates', {})
        for tool, rate in raw_rates.items():
            if isinstance(rate, list) and len(rate) == 2:
                instance._tool_success_rates[tool] = (rate[0], rate[1])
            else:
                instance._tool_success_rates[tool] = rate

        # Restore feedback history
        instance._executor_feedback_history = data.get('feedback_history', [])

        return instance


# =============================================================================
# TOOL MANAGER (Agent0 Dynamic Tool Management)
# =============================================================================

class ToolManager:
    """
    Dynamic tool management based on Agent0 learned performance.

    Bridges CurriculumGenerator._tool_success_rates with runtime tool selection.
    Tracks per-swarm tool additions/removals and suggests replacements for
    consistently failing tools.
    """

    FAILURE_THRESHOLD = 0.6   # Below 60% success = failing tool
    MIN_SAMPLES = 3           # Need 3+ uses before judging

    def __init__(self):
        self._tool_assignments: Dict[str, List[str]] = {}    # swarm_name → [added tools]
        self._deactivated_tools: Dict[str, List[str]] = {}   # swarm_name → [removed tools]

    def analyze_tools(self, tool_success_rates: Dict[str, Tuple[int, int]], swarm_name: str) -> Dict[str, Any]:
        """
        Classify tools as weak/strong based on Agent0 success rates.

        Args:
            tool_success_rates: Dict of tool_name → (successes, total_uses)
            swarm_name: Name of the swarm being analyzed

        Returns:
            Dict with weak_tools, strong_tools, suggested_removals, replacements
        """
        weak_tools = []
        strong_tools = []
        suggested_removals = []
        replacements = {}

        for tool, rate_data in tool_success_rates.items():
            if isinstance(rate_data, (list, tuple)) and len(rate_data) == 2:
                successes, total = rate_data
            else:
                continue

            if total < self.MIN_SAMPLES:
                continue

            success_rate = successes / total if total > 0 else 0.0

            if success_rate < self.FAILURE_THRESHOLD:
                weak_tools.append({
                    'tool': tool,
                    'success_rate': success_rate,
                    'successes': successes,
                    'total': total
                })
                suggested_removals.append(tool)
                tool_replacements = self.find_replacements(tool)
                if tool_replacements:
                    replacements[tool] = tool_replacements
            else:
                strong_tools.append({
                    'tool': tool,
                    'success_rate': success_rate,
                    'successes': successes,
                    'total': total
                })

        return {
            'weak_tools': weak_tools,
            'strong_tools': strong_tools,
            'suggested_removals': suggested_removals,
            'replacements': replacements,
            'swarm_name': swarm_name
        }

    def find_replacements(self, failing_tool: str) -> List[Dict[str, str]]:
        """
        Search for replacement tools by keyword matching on tool name.

        Args:
            failing_tool: Name of the tool that is performing poorly

        Returns:
            List of dicts with name, description, reason for each candidate
        """
        replacements = []
        keywords = failing_tool.lower().replace('_', ' ').split()

        # Built-in tool categories that could serve as replacements
        tool_alternatives = {
            'fetch': ['web_search', 'api_query', 'scrape'],
            'search': ['grep', 'web_search', 'semantic_search'],
            'generate': ['create', 'synthesize', 'compose'],
            'analyze': ['evaluate', 'inspect', 'assess'],
            'extract': ['parse', 'mine', 'collect'],
            'validate': ['verify', 'check', 'confirm'],
            'transform': ['convert', 'normalize', 'reshape'],
        }

        for keyword in keywords:
            if keyword in tool_alternatives:
                for alt in tool_alternatives[keyword]:
                    replacements.append({
                        'name': alt,
                        'description': f"Alternative for {failing_tool} (keyword: {keyword})",
                        'reason': f"{failing_tool} below {self.FAILURE_THRESHOLD*100:.0f}% success"
                    })

        return replacements

    def get_active_tools(self, swarm_name: str, defaults: List[str] = None) -> List[str]:
        """
        Return merged tool list: defaults + dynamic additions - deactivated.

        Args:
            swarm_name: Name of the swarm
            defaults: Default tool list for this swarm

        Returns:
            Active tool list after applying additions and removals
        """
        active = set(defaults or [])
        active.update(self._tool_assignments.get(swarm_name, []))
        active -= set(self._deactivated_tools.get(swarm_name, []))
        return list(active)

    def update_assignments(self, swarm_name: str, add: List[str] = None, remove: List[str] = None):
        """
        Track tool additions/removals per swarm.

        Args:
            swarm_name: Name of the swarm
            add: Tools to add to this swarm's active set
            remove: Tools to deactivate for this swarm
        """
        if add:
            if swarm_name not in self._tool_assignments:
                self._tool_assignments[swarm_name] = []
            for tool in add:
                if tool not in self._tool_assignments[swarm_name]:
                    self._tool_assignments[swarm_name].append(tool)

        if remove:
            if swarm_name not in self._deactivated_tools:
                self._deactivated_tools[swarm_name] = []
            for tool in remove:
                if tool not in self._deactivated_tools[swarm_name]:
                    self._deactivated_tools[swarm_name].append(tool)

    def to_dict(self) -> Dict:
        """Serialize tool assignments and deactivations for persistence."""
        return {
            'tool_assignments': dict(self._tool_assignments),
            'deactivated_tools': dict(self._deactivated_tools)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ToolManager':
        """Restore ToolManager from persisted data."""
        instance = cls()
        instance._tool_assignments = data.get('tool_assignments', {})
        instance._deactivated_tools = data.get('deactivated_tools', {})
        return instance


# =============================================================================
# SWARM INTELLIGENCE ENGINE
# =============================================================================

class SwarmIntelligence:
    """
    World-class swarm intelligence coordinator.

    Features:
    - Emergent specialization
    - Swarm consensus
    - Online adaptation
    - Dynamic task routing
    - Session isolation
    - Agent-to-agent messaging
    """

    def __init__(self, config=None):
        self.config = config

        # Agent profiles (emergent specialization)
        self.agent_profiles: Dict[str, AgentProfile] = {}

        # Session management (moltbot pattern)
        self.sessions: Dict[str, AgentSession] = {}

        # Collective memory (shared across swarm)
        self.collective_memory: List[Dict] = []
        self.memory_embeddings: Dict[str, Any] = {}

        # Online adaptation buffer
        self.adaptation_buffer: List[Dict] = []
        self.adaptation_interval = 5  # Adapt every N experiences

        # Consensus history
        self.consensus_history: List[SwarmDecision] = []

        # Task routing stats
        self.routing_success: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Stigmergy layer (indirect coordination via shared artifacts)
        self.stigmergy = StigmergyLayer()

        # Swarm benchmarks (performance tracking)
        self.benchmarks = SwarmBenchmarks()

        # Byzantine fault tolerance (verify agent claims)
        self.byzantine = ByzantineVerifier(self)

        # DrZero-inspired curriculum generator (self-generated training tasks)
        self.curriculum_generator = CurriculumGenerator(config)

        # MorphAgent-inspired scorer (RCS/RDS/TRAS)
        self.morph_scorer = MorphScorer(config)

        # Agent0: Dynamic tool management
        self.tool_manager = ToolManager()

        # Track swarm-level MorphAgent scores over time
        self.morph_score_history: List[Dict[str, Any]] = []

        # Training mode configuration (Agent0 inspired)
        self._training_mode = False
        self._memory_system = None

        logger.info("SwarmIntelligence initialized (DrZero curriculum + MorphAgent scoring)")

    def enable_training_mode(self, enabled: bool = True, memory_system=None):
        """
        Enable/disable curriculum-based training mode.

        Agent0 insight: Training mode generates tasks that target agent weaknesses.

        Args:
            enabled: Whether training mode is active
            memory_system: Optional HierarchicalMemory for context-aware tasks
        """
        self._training_mode = enabled

        if memory_system:
            self._memory_system = memory_system
            self.curriculum_generator.connect_memory(memory_system)

        logger.info(f"Training mode {'enabled' if enabled else 'disabled'}")

    def get_training_task(self, target_agent: str = None, tool_aware: bool = True) -> Optional[SyntheticTask]:
        """
        Get a curriculum-generated training task.

        Agent0: Uses tool-aware generation when tool_aware=True.

        Args:
            target_agent: Optionally target specific agent's weaknesses
            tool_aware: Use tool-aware task generation (Agent0 style)

        Returns:
            SyntheticTask or None if training mode disabled
        """
        if not self._training_mode:
            return None

        if tool_aware:
            return self.curriculum_generator.generate_tool_aware_task(
                profiles=self.agent_profiles,
                target_agent=target_agent,
                prefer_weak_tools=True
            )
        else:
            return self.curriculum_generator.generate_training_task(
                profiles=self.agent_profiles,
                target_agent=target_agent
            )

    def receive_executor_feedback(
        self,
        task_id: str,
        success: bool,
        tools_used: List[str],
        execution_time: float = 0.0,
        error_type: str = None,
        task_type: str = None
    ):
        """
        Receive feedback from executor after task completion.

        Agent0 closed-loop: Executor feedback → Curriculum adaptation.

        Args:
            task_id: Task identifier
            success: Whether task succeeded
            tools_used: List of tools used during execution
            execution_time: Time taken to execute
            error_type: Type of error if failed
            task_type: Type of task (for curriculum update)
        """
        # Forward to curriculum generator
        self.curriculum_generator.receive_executor_feedback(
            task_id=task_id,
            success=success,
            tools_used=tools_used,
            execution_time=execution_time,
            error_type=error_type
        )

        # Update curriculum difficulty if this was a synthetic task
        if task_type:
            task = SyntheticTask(
                task_id=task_id,
                task_type=task_type,
                description="",
                difficulty=0.5,
                target_agent=None
            )
            self.curriculum_generator.update_from_result(task, success, execution_time)

    # =========================================================================
    # EMERGENT SPECIALIZATION
    # =========================================================================

    def register_agent(self, agent_name: str):
        """Register an agent for tracking."""
        if agent_name not in self.agent_profiles:
            self.agent_profiles[agent_name] = AgentProfile(agent_name=agent_name)

    def record_task_result(
        self,
        agent_name: str,
        task_type: str,
        success: bool,
        execution_time: float,
        context: Dict = None,
        is_multi_agent: bool = False,
        agents_count: int = 1
    ):
        """Record task result for specialization learning."""
        self.register_agent(agent_name)
        self.agent_profiles[agent_name].update_task_result(task_type, success, execution_time)

        # Add to collective memory
        self.collective_memory.append({
            'agent': agent_name,
            'task_type': task_type,
            'success': success,
            'execution_time': execution_time,
            'context': context or {},
            'timestamp': time.time()
        })

        # Bound collective memory
        if len(self.collective_memory) > 1000:
            self.collective_memory = self.collective_memory[-1000:]

        # Online adaptation
        self.adaptation_buffer.append({
            'agent': agent_name,
            'task_type': task_type,
            'success': success
        })
        if len(self.adaptation_buffer) >= self.adaptation_interval:
            self._perform_online_adaptation()

        # Stigmergy: deposit success/warning signals
        if success:
            self.deposit_success_signal(agent_name, task_type, execution_time)
        else:
            self.deposit_warning_signal(agent_name, task_type, "Task failed")

        # Benchmarks: record run
        if is_multi_agent:
            self.benchmarks.record_multi_agent_run(task_type, execution_time, agents_count, success)
        else:
            self.benchmarks.record_single_agent_run(task_type, execution_time, success)

    def get_agent_specialization(self, agent_name: str) -> AgentSpecialization:
        """Get current specialization of an agent."""
        if agent_name in self.agent_profiles:
            return self.agent_profiles[agent_name].specialization
        return AgentSpecialization.GENERALIST

    def get_specialization_summary(self) -> Dict[str, str]:
        """Get summary of all agent specializations."""
        return {
            name: profile.specialization.value
            for name, profile in self.agent_profiles.items()
        }

    # =========================================================================
    # DYNAMIC TASK ROUTING
    # =========================================================================

    def get_best_agent_for_task(
        self,
        task_type: str,
        available_agents: List[str],
        task_description: str = None,
        use_morph_scoring: bool = True
    ) -> Optional[str]:
        """
        Route task to best-fit agent based on learned performance.

        Enhanced with MorphAgent TRAS scoring for better task-agent alignment.

        Uses:
        - MorphAgent TRAS (Task-Role Alignment Score) - NEW
        - MorphAgent RCS (Role Clarity Score) as filter - NEW
        - Historical success rate
        - Specialization match
        - Trust score
        - Stigmergy routing signals
        """
        if not available_agents:
            return None

        # Ensure all agents are registered
        for agent_name in available_agents:
            self.register_agent(agent_name)

        # Build profile dict for available agents
        profiles = {name: self.agent_profiles[name] for name in available_agents}

        # Strategy 1: Use MorphAgent TRAS scoring if enabled and task description available
        if use_morph_scoring and task_description and self.morph_scorer:
            best = self.morph_scorer.get_best_agent_by_tras(
                profiles=profiles,
                task=task_description,
                task_type=task_type,
                min_rcs=0.3  # Require minimum role clarity
            )
            if best:
                logger.debug(f"MorphAgent TRAS routing: {task_type} -> {best}")
                return best

        # Strategy 2: Check stigmergy routing signals
        route_signals = self.stigmergy.get_route_signals(task_type)
        if route_signals:
            # Filter to available agents
            available_signals = {a: s for a, s in route_signals.items() if a in available_agents}
            if available_signals:
                best_from_stigmergy = max(available_signals.keys(), key=lambda a: available_signals[a])
                if available_signals[best_from_stigmergy] > 0.5:  # Strong signal
                    logger.debug(f"Stigmergy routing: {task_type} -> {best_from_stigmergy}")
                    return best_from_stigmergy

        # Strategy 3: Fallback to traditional scoring
        best_agent = None
        best_score = -1.0

        for agent_name in available_agents:
            profile = self.agent_profiles[agent_name]

            # Base: success rate for this task type
            success_rate = profile.get_success_rate(task_type)

            # Bonus for specialization match
            spec_bonus = 0.0
            expected_spec = self._task_type_to_specialization(task_type)
            if profile.specialization == expected_spec:
                spec_bonus = 0.2

            # Trust score weight
            trust_weight = profile.trust_score

            # MorphAgent RCS bonus (clear roles get preference)
            rcs_bonus = 0.0
            if self.morph_scorer:
                rcs, _ = self.morph_scorer.compute_rcs(profile)
                rcs_bonus = rcs * 0.1  # Up to 0.1 bonus for clear roles

            # Combined score
            score = (
                success_rate * 0.4 +
                trust_weight * 0.25 +
                spec_bonus * 0.15 +
                rcs_bonus * 0.2
            )

            if score > best_score:
                best_score = score
                best_agent = agent_name

        return best_agent

    def _task_type_to_specialization(self, task_type: str) -> AgentSpecialization:
        """Map task type to expected specialization."""
        mapping = {
            'aggregation': AgentSpecialization.AGGREGATOR,
            'analysis': AgentSpecialization.ANALYZER,
            'transformation': AgentSpecialization.TRANSFORMER,
            'validation': AgentSpecialization.VALIDATOR,
            'planning': AgentSpecialization.PLANNER,
            'filtering': AgentSpecialization.EXECUTOR,
            'generation': AgentSpecialization.EXECUTOR,
        }
        return mapping.get(task_type, AgentSpecialization.GENERALIST)

    # =========================================================================
    # SWARM CONSENSUS
    # =========================================================================

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

        # Gather votes (can be parallelized)
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

    # =========================================================================
    # ONLINE ADAPTATION
    # =========================================================================

    def _perform_online_adaptation(self):
        """
        Adapt routing and specialization based on recent performance.

        Called periodically during execution, not just at end.
        """
        if not self.adaptation_buffer:
            return

        # Analyze recent performance
        recent_by_agent = defaultdict(list)
        for item in self.adaptation_buffer:
            recent_by_agent[item['agent']].append(item['success'])

        # Check for struggling agents
        for agent_name, results in recent_by_agent.items():
            recent_rate = sum(results) / len(results)
            profile = self.agent_profiles.get(agent_name)

            if profile and recent_rate < 0.3 and len(results) >= 3:
                # Agent is struggling - trigger adaptation
                logger.info(f"Online adaptation: {agent_name} struggling ({recent_rate:.0%}), may need different task types")
                profile.trust_score = max(0.1, profile.trust_score - 0.1)
            elif profile and recent_rate > 0.8 and len(results) >= 3:
                # Agent is excelling - boost trust
                profile.trust_score = min(1.0, profile.trust_score + 0.05)

        # Clear buffer
        self.adaptation_buffer = []

    # =========================================================================
    # SESSION MANAGEMENT (moltbot pattern)
    # =========================================================================

    def create_session(self, agent_name: str, context: str = "main") -> str:
        """Create isolated session for an agent."""
        session_id = hashlib.md5(f"{agent_name}:{context}:{time.time()}".encode()).hexdigest()[:12]

        self.sessions[session_id] = AgentSession(
            session_id=session_id,
            agent_name=agent_name,
            context=context
        )

        return session_id

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def session_send(self, session_id: str, from_agent: str, content: str, metadata: Dict = None):
        """Send message to a session (moltbot sessions_send pattern)."""
        session = self.sessions.get(session_id)
        if session:
            session.add_message(from_agent, content, metadata)
            return True
        return False

    def session_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Get session history (moltbot sessions_history pattern)."""
        session = self.sessions.get(session_id)
        if session:
            return session.messages[-limit:]
        return []

    def sessions_list(self, agent_name: str = None) -> List[Dict]:
        """List sessions (moltbot sessions_list pattern)."""
        sessions = []
        for sid, session in self.sessions.items():
            if agent_name is None or session.agent_name == agent_name:
                sessions.append({
                    'session_id': sid,
                    'agent': session.agent_name,
                    'context': session.context,
                    'message_count': len(session.messages),
                    'last_active': session.last_active
                })
        return sessions

    # =========================================================================
    # STIGMERGY INTEGRATION
    # =========================================================================

    def deposit_success_signal(self, agent: str, task_type: str, execution_time: float = 0.0):
        """
        Deposit success signal so other agents can learn from this success.

        Creates two signals:
        1. A 'success' signal for general awareness
        2. A 'route' signal for task routing recommendations
        """
        # Success signal
        self.stigmergy.deposit(
            signal_type='success',
            content={'agent': agent, 'task_type': task_type},
            agent=agent,
            strength=1.0,
            metadata={'execution_time': execution_time}
        )

        # Route signal (for task routing)
        self.stigmergy.deposit(
            signal_type='route',
            content={'agent': agent, 'task_type': task_type},
            agent=agent,
            strength=1.0
        )

        logger.debug(f"Stigmergy: Deposited success signal for {agent} on {task_type}")

    def deposit_warning_signal(self, agent: str, task_type: str, warning: str):
        """Deposit warning signal so other agents can avoid mistakes."""
        self.stigmergy.deposit(
            signal_type='warning',
            content={'agent': agent, 'task_type': task_type, 'warning': warning},
            agent=agent,
            strength=0.8
        )

    def get_stigmergy_recommendation(self, task_type: str) -> Optional[str]:
        """
        Get agent recommendation from pheromone signals.

        Returns the agent with the strongest route signal for this task type.
        """
        route_signals = self.stigmergy.get_route_signals(task_type)

        if not route_signals:
            return None

        # Return agent with highest accumulated strength
        best_agent = max(route_signals.keys(), key=lambda a: route_signals[a])
        return best_agent

    def get_warnings_for_task(self, task_type: str) -> List[str]:
        """Get warnings from stigmergy for a task type."""
        warnings = []
        for signal in self.stigmergy.sense(signal_type='warning', min_strength=0.3):
            content = signal.content
            if isinstance(content, dict) and content.get('task_type') == task_type:
                warnings.append(content.get('warning', ''))
        return [w for w in warnings if w]

    # =========================================================================
    # COLLECTIVE INTELLIGENCE
    # =========================================================================

    def get_swarm_wisdom(self, query: str, task_type: str = None) -> Dict[str, Any]:
        """
        Get collective wisdom from the swarm for a task.

        Returns:
        - Best agent recommendation
        - Similar past experiences
        - Success patterns
        - Warnings from failures
        """
        wisdom = {
            'recommended_agent': None,
            'similar_experiences': [],
            'success_patterns': [],
            'warnings': [],
            'confidence': 0.0
        }

        # Get best agent
        available = list(self.agent_profiles.keys())
        if task_type and available:
            wisdom['recommended_agent'] = self.get_best_agent_for_task(task_type, available)

        # Find similar past experiences
        if self.collective_memory:
            for mem in self.collective_memory[-50:]:  # Recent memories
                if task_type and mem.get('task_type') == task_type:
                    wisdom['similar_experiences'].append({
                        'agent': mem['agent'],
                        'success': mem['success'],
                        'execution_time': mem['execution_time']
                    })

        # Extract patterns
        successes = [m for m in wisdom['similar_experiences'] if m['success']]
        failures = [m for m in wisdom['similar_experiences'] if not m['success']]

        if successes:
            wisdom['success_patterns'].append(
                f"{len(successes)} successful executions for {task_type} tasks"
            )

        if failures:
            wisdom['warnings'].append(
                f"{len(failures)} failures recorded - consider validation"
            )

        # Confidence based on data
        total = len(wisdom['similar_experiences'])
        if total > 0:
            wisdom['confidence'] = min(1.0, total / 10)  # Max confidence at 10+ examples

        return wisdom

    def format_swarm_context(self, query: str, task_type: str = None) -> str:
        """Format swarm wisdom as context for agents."""
        wisdom = self.get_swarm_wisdom(query, task_type)

        lines = ["# Swarm Intelligence Context:\n"]

        if wisdom['recommended_agent']:
            lines.append(f"## Recommended Agent: {wisdom['recommended_agent']}")

        if wisdom['success_patterns']:
            lines.append("\n## Success Patterns:")
            for pattern in wisdom['success_patterns']:
                lines.append(f"  - {pattern}")

        if wisdom['warnings']:
            lines.append("\n## Warnings:")
            for warning in wisdom['warnings']:
                lines.append(f"  - ⚠️ {warning}")

        # Add specialization info
        specs = self.get_specialization_summary()
        if specs:
            lines.append("\n## Agent Specializations:")
            for agent, spec in specs.items():
                lines.append(f"  - {agent}: {spec}")

        return "\n".join(lines)

    # =========================================================================
    # MORPHAGENT SCORING INTEGRATION
    # =========================================================================

    def compute_morph_scores(self, task: str = None, task_type: str = None) -> Dict[str, MorphScores]:
        """
        Compute MorphAgent scores (RCS/RDS/TRAS) for all agents.

        Args:
            task: Optional task description for TRAS computation
            task_type: Optional task type for TRAS computation

        Returns:
            Dict of agent_name -> MorphScores
        """
        if not self.morph_scorer:
            return {}

        scores = self.morph_scorer.compute_all_scores(
            profiles=self.agent_profiles,
            task=task,
            task_type=task_type
        )

        # Record in history
        self.morph_score_history.append({
            'timestamp': time.time(),
            'scores': {name: {'rcs': s.rcs, 'rds': s.rds, 'tras': s.tras} for name, s in scores.items()},
            'task_context': task[:50] if task else ''
        })

        # Keep bounded
        if len(self.morph_score_history) > 100:
            self.morph_score_history = self.morph_score_history[-100:]

        return scores

    def get_swarm_health(self) -> Dict[str, Any]:
        """
        Get overall swarm health using MorphAgent metrics.

        Returns comprehensive health assessment:
        - avg_rcs: Average Role Clarity (are roles well-defined?)
        - rds: Role Differentiation (is swarm diverse?)
        - avg_trust: Average trust score
        - specialization_coverage: How many specializations are covered
        - recommendations: Improvement suggestions
        """
        health = {
            'avg_rcs': 0.5,
            'rds': 0.5,
            'avg_trust': 0.5,
            'specialization_coverage': 0.0,
            'agent_count': len(self.agent_profiles),
            'total_tasks': sum(p.total_tasks for p in self.agent_profiles.values()),
            'recommendations': []
        }

        if not self.agent_profiles:
            health['recommendations'].append("No agents registered - add agents to swarm")
            return health

        # Compute MorphAgent scores
        if self.morph_scorer:
            # RCS for each agent
            rcs_scores = []
            for profile in self.agent_profiles.values():
                rcs, _ = self.morph_scorer.compute_rcs(profile)
                rcs_scores.append(rcs)
            health['avg_rcs'] = sum(rcs_scores) / len(rcs_scores) if rcs_scores else 0.5

            # RDS (swarm-level)
            health['rds'] = self.morph_scorer.compute_rds(self.agent_profiles)

        # Average trust
        trust_scores = [p.trust_score for p in self.agent_profiles.values()]
        health['avg_trust'] = sum(trust_scores) / len(trust_scores) if trust_scores else 0.5

        # Specialization coverage
        unique_specs = set(p.specialization for p in self.agent_profiles.values())
        health['specialization_coverage'] = len(unique_specs) / len(AgentSpecialization)

        # Generate recommendations
        if health['avg_rcs'] < 0.4:
            health['recommendations'].append(
                "Low role clarity - consider warmup training to specialize agents"
            )

        if health['rds'] < 0.4:
            health['recommendations'].append(
                "Low role differentiation - agents are too similar, consider diversifying"
            )

        if health['avg_trust'] < 0.5:
            health['recommendations'].append(
                "Low average trust - some agents have inconsistent performance"
            )

        if health['total_tasks'] < 10:
            health['recommendations'].append(
                "Limited task history - consider warmup() for self-training"
            )

        if not health['recommendations']:
            health['recommendations'].append("Swarm health is good - no issues detected")

        return health

    def optimize_profiles_morph(self, num_iterations: int = 5, threshold: float = 0.1) -> Dict[str, Any]:
        """
        MorphAgent-inspired profile optimization.

        Iteratively improves agent profiles by:
        1. Computing RCS/RDS scores
        2. Identifying low-scoring agents
        3. Generating curriculum tasks targeting weaknesses
        4. Simulating improvement through task type rebalancing

        This is used during warmup phase to optimize agent differentiation.

        Args:
            num_iterations: Max optimization iterations
            threshold: Convergence threshold for score improvement

        Returns:
            Optimization results with before/after scores
        """
        if not self.agent_profiles:
            return {'success': False, 'reason': 'No agents to optimize'}

        results = {
            'iterations': 0,
            'initial_rds': 0.0,
            'final_rds': 0.0,
            'initial_avg_rcs': 0.0,
            'final_avg_rcs': 0.0,
            'improvements': []
        }

        # Initial scores
        if self.morph_scorer:
            results['initial_rds'] = self.morph_scorer.compute_rds(self.agent_profiles)
            rcs_scores = [self.morph_scorer.compute_rcs(p)[0] for p in self.agent_profiles.values()]
            results['initial_avg_rcs'] = sum(rcs_scores) / len(rcs_scores) if rcs_scores else 0.5

        prev_score = results['initial_rds'] + results['initial_avg_rcs']

        for iteration in range(num_iterations):
            results['iterations'] = iteration + 1

            # Find agents with low RCS (unclear roles)
            low_rcs_agents = []
            for name, profile in self.agent_profiles.items():
                if self.morph_scorer:
                    rcs, components = self.morph_scorer.compute_rcs(profile)
                    if rcs < 0.5:
                        low_rcs_agents.append((name, rcs, components))

            # Generate curriculum tasks targeting low-RCS agents
            for agent_name, rcs, components in low_rcs_agents[:3]:  # Top 3 worst
                # Identify which component is lowest
                if components.get('focus', 1.0) < 0.5:
                    # Agent needs to focus - generate tasks in their best type
                    profile = self.agent_profiles[agent_name]
                    if profile.task_success:
                        best_type = max(
                            profile.task_success.keys(),
                            key=lambda t: profile.task_success[t][0] / max(1, profile.task_success[t][1])
                        )
                        results['improvements'].append(
                            f"{agent_name}: Focus training on {best_type} (RCS: {rcs:.2f})"
                        )

            # Compute new scores
            if self.morph_scorer:
                new_rds = self.morph_scorer.compute_rds(self.agent_profiles)
                new_rcs_scores = [self.morph_scorer.compute_rcs(p)[0] for p in self.agent_profiles.values()]
                new_avg_rcs = sum(new_rcs_scores) / len(new_rcs_scores) if new_rcs_scores else 0.5

                new_score = new_rds + new_avg_rcs

                # Check convergence
                if abs(new_score - prev_score) < threshold:
                    break

                prev_score = new_score
                results['final_rds'] = new_rds
                results['final_avg_rcs'] = new_avg_rcs

        return results

    def format_morph_report(self) -> str:
        """Generate human-readable MorphAgent scores report."""
        lines = [
            "# MorphAgent Scores Report",
            "=" * 50,
            ""
        ]

        if not self.agent_profiles:
            lines.append("No agents registered.")
            return "\n".join(lines)

        # Swarm-level RDS
        if self.morph_scorer:
            rds = self.morph_scorer.compute_rds(self.agent_profiles)
            lines.append(f"## Swarm Role Differentiation (RDS): {rds:.2f}")
            lines.append(f"   {'✓ Good diversity' if rds >= 0.5 else '⚠️ Agents too similar'}")
            lines.append("")

        # Per-agent RCS
        lines.append("## Per-Agent Role Clarity (RCS)")
        lines.append("-" * 40)

        for name, profile in self.agent_profiles.items():
            if self.morph_scorer:
                rcs, components = self.morph_scorer.compute_rcs(profile)
                status = "✓" if rcs >= 0.5 else "⚠️"
                lines.append(f"  {status} {name}: RCS={rcs:.2f}")
                lines.append(f"      Focus: {components.get('focus', 0):.2f}, "
                           f"Consistency: {components.get('consistency', 0):.2f}, "
                           f"Specialization: {components.get('specialization', 0):.2f}")

        # Health summary
        lines.append("")
        health = self.get_swarm_health()
        lines.append("## Health Summary")
        lines.append(f"  - Average RCS: {health['avg_rcs']:.2f}")
        lines.append(f"  - RDS: {health['rds']:.2f}")
        lines.append(f"  - Average Trust: {health['avg_trust']:.2f}")
        lines.append(f"  - Specialization Coverage: {health['specialization_coverage']:.1%}")

        if health['recommendations']:
            lines.append("")
            lines.append("## Recommendations")
            for rec in health['recommendations']:
                lines.append(f"  - {rec}")

        return "\n".join(lines)

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def save(self, path: str):
        """Save swarm intelligence state."""
        import json

        data = {
            'agent_profiles': {
                name: {
                    'agent_name': p.agent_name,
                    'specialization': p.specialization.value,
                    'task_success': p.task_success,
                    'helped_others': p.helped_others,
                    'received_help': p.received_help,
                    'consensus_agreements': p.consensus_agreements,
                    'consensus_disagreements': p.consensus_disagreements,
                    'avg_execution_time': p.avg_execution_time,
                    'total_tasks': p.total_tasks,
                    'trust_score': p.trust_score,
                }
                for name, p in self.agent_profiles.items()
            },
            'collective_memory': self.collective_memory[-200:],  # Keep recent
            'routing_success': dict(self.routing_success),
            'stigmergy': self.stigmergy.to_dict(),  # Persist stigmergy state
            'benchmarks': self.benchmarks.to_dict(),  # Persist benchmark data
            'curriculum': self.curriculum_generator.to_dict(),  # DrZero curriculum state
            'morph_score_history': self.morph_score_history[-50:],  # MorphAgent score history
            'tool_manager': self.tool_manager.to_dict(),  # Agent0 tool management state
        }

        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved swarm intelligence: {len(self.agent_profiles)} profiles, {len(self.stigmergy.signals)} stigmergy signals, curriculum tasks={self.curriculum_generator.total_generated}")

    def load(self, path: str) -> bool:
        """Load swarm intelligence state."""
        import json
        from pathlib import Path

        if not Path(path).exists():
            return False

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            # Restore profiles
            for name, p_data in data.get('agent_profiles', {}).items():
                profile = AgentProfile(
                    agent_name=p_data['agent_name'],
                    specialization=AgentSpecialization(p_data['specialization']),
                    task_success=p_data['task_success'],
                    helped_others=p_data['helped_others'],
                    received_help=p_data['received_help'],
                    consensus_agreements=p_data['consensus_agreements'],
                    consensus_disagreements=p_data['consensus_disagreements'],
                    avg_execution_time=p_data['avg_execution_time'],
                    total_tasks=p_data['total_tasks'],
                    trust_score=p_data['trust_score'],
                )
                self.agent_profiles[name] = profile

            self.collective_memory = data.get('collective_memory', [])

            # Load stigmergy state
            if 'stigmergy' in data:
                self.stigmergy = StigmergyLayer.from_dict(data['stigmergy'])

            # Load benchmarks
            if 'benchmarks' in data:
                self.benchmarks = SwarmBenchmarks.from_dict(data['benchmarks'])

            # Load DrZero curriculum state
            if 'curriculum' in data:
                self.curriculum_generator = CurriculumGenerator.from_dict(data['curriculum'], self.config)

            # Load MorphAgent score history
            if 'morph_score_history' in data:
                self.morph_score_history = data['morph_score_history']

            # Load Agent0 tool manager state
            if 'tool_manager' in data:
                self.tool_manager = ToolManager.from_dict(data['tool_manager'])

            logger.info(f"Loaded swarm intelligence: {len(self.agent_profiles)} profiles, {len(self.stigmergy.signals)} stigmergy signals, curriculum tasks={self.curriculum_generator.total_generated}")
            return True

        except Exception as e:
            logger.warning(f"Could not load swarm intelligence: {e}")
            return False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SwarmIntelligence',
    'AgentProfile',
    'AgentSpecialization',
    'ConsensusVote',
    'SwarmDecision',
    'AgentSession',
    'StigmergySignal',
    'StigmergyLayer',
    'SwarmMetrics',
    'SwarmBenchmarks',
    'ByzantineVerifier',
    # DrZero-inspired curriculum
    'CurriculumGenerator',
    'SyntheticTask',
    # MorphAgent-inspired scoring
    'MorphScorer',
    'MorphScores',
    # Agent0 tool management
    'ToolManager',
]

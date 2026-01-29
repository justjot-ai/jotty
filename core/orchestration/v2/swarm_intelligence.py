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

Inspired by: biological swarms, moltbot architecture, multi-agent RL research
"""

import asyncio
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


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

        logger.info("SwarmIntelligence initialized")

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

    def get_best_agent_for_task(self, task_type: str, available_agents: List[str]) -> Optional[str]:
        """
        Route task to best-fit agent based on learned performance.

        Uses:
        - Historical success rate
        - Specialization match
        - Trust score
        - Current load (if available)
        """
        if not available_agents:
            return None

        best_agent = None
        best_score = -1.0

        for agent_name in available_agents:
            self.register_agent(agent_name)
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

            # Combined score
            score = (success_rate * 0.5 + trust_weight * 0.3 + spec_bonus * 0.2)

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
        }

        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved swarm intelligence: {len(self.agent_profiles)} profiles, {len(self.stigmergy.signals)} stigmergy signals")

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

            logger.info(f"Loaded swarm intelligence: {len(self.agent_profiles)} profiles, {len(self.stigmergy.signals)} stigmergy signals")
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
]

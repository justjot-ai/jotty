"""
Stigmergy Layer (Indirect Agent Coordination)
===============================================

Shared signal store for agent coordination without direct communication.

Key mechanism: after each task execution, record_outcome() deposits a
signal. Over time, successful agent-task pairings accumulate strong
signals while failed ones decay. recommend_agent() reads these signals
to route new tasks to the best-performing agent.

This is the actual value of stigmergy: emergent routing from accumulated
experience, not a fancy name for a TTL cache.
"""

import time
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


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

    def record_outcome(self, agent: str, task_type: str, success: bool,
                       quality: float = 0.0, metadata: Dict = None) -> str:
        """
        Record a task outcome as a signal. Call this after every execution.

        Successful outcomes deposit a strong 'route' signal that reinforces
        the agent-task pairing. Failed outcomes deposit a weak 'warning'
        signal. Over many executions, the signal landscape converges to
        reflect which agents are best for which task types.

        Args:
            agent: Agent that executed the task
            task_type: Type of task (e.g., 'analysis', 'coding', 'research')
            success: Whether the task succeeded
            quality: Output quality 0-1 (optional, refines signal strength)
            metadata: Additional context

        Returns:
            Signal ID
        """
        if success:
            strength = 0.5 + 0.5 * max(0.0, min(1.0, quality))  # 0.5-1.0
            signal_type = 'route'
            content = {'task_type': task_type, 'agent': agent, 'success': True}
        else:
            strength = 0.3
            signal_type = 'warning'
            content = {'task_type': task_type, 'agent': agent, 'success': False}

        # Check if there's an existing route signal for this agent+task_type
        # If so, reinforce it instead of creating a new one (DRY: single signal per pairing)
        for sid, signal in self.signals.items():
            if (signal.signal_type == 'route'
                    and isinstance(signal.content, dict)
                    and signal.content.get('agent') == agent
                    and signal.content.get('task_type') == task_type):
                if success:
                    self.reinforce(sid, amount=0.15 + 0.1 * quality)
                else:
                    # Weaken on failure
                    signal.strength = max(0.01, signal.strength - 0.1)
                return sid

        return self.deposit(
            signal_type=signal_type,
            content=content,
            agent=agent,
            strength=strength,
            metadata=metadata or {},
        )

    def recommend_agent(self, task_type: str, candidates: List[str] = None) -> List[Tuple[str, float]]:
        """
        Recommend agents for a task type based on accumulated signals.

        Returns a ranked list of (agent, score) pairs. Agents with more
        successful executions of this task type score higher.

        Args:
            task_type: Type of task to route
            candidates: Optional whitelist of agent names to consider

        Returns:
            List of (agent_name, score) sorted by score descending.
            Empty list if no signals exist for this task type.
        """
        self._apply_decay()

        scores: Dict[str, float] = defaultdict(float)
        penalties: Dict[str, float] = defaultdict(float)

        # Accumulate positive signals from route signals
        for signal in self.signals.values():
            if not isinstance(signal.content, dict):
                continue
            if signal.content.get('task_type') != task_type:
                continue

            agent = signal.content.get('agent', '')
            if not agent:
                continue
            if candidates and agent not in candidates:
                continue

            if signal.signal_type == 'route' and signal.content.get('success'):
                scores[agent] += signal.strength
            elif signal.signal_type == 'warning':
                penalties[agent] += signal.strength * 0.5

        # Net score = positive - penalties
        result = []
        all_agents = set(scores.keys()) | set(penalties.keys())
        for agent in all_agents:
            net = scores.get(agent, 0.0) - penalties.get(agent, 0.0)
            if net > 0:
                result.append((agent, net))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def get_route_signals(self, task_type: str) -> Dict[str, float]:
        """
        Get routing recommendations from environment.

        Returns agent->strength mapping for task routing decisions.
        Prefer recommend_agent() for ranked results.
        """
        return dict(self.recommend_agent(task_type))

    # =====================================================================
    # APPROACH TRACKING (useful in single-agent mode)
    # =====================================================================
    #
    # In single-agent mode, agent routing is meaningless — there's one agent.
    # What IS useful: tracking which *approach* worked for which task type.
    # An "approach" is a summary of what the agent actually did:
    #   - which tools/skills it called
    #   - what strategy it used (e.g., "used web-search then summarized")
    #   - what failed (e.g., "tried code execution but hit timeout")
    #
    # These signals are deposited via record_approach_outcome() and read
    # via recommend_approach() to give the single agent actionable guidance
    # for the *next* execution of a similar task type.

    def record_approach_outcome(
        self,
        task_type: str,
        approach_summary: str,
        tools_used: List[str],
        success: bool,
        quality: float = 0.5,
        agent: str = "default",
    ) -> str:
        """
        Record what approach/tools worked (or failed) for a task type.

        Unlike record_outcome which just says "agent X on task Y",
        this records HOW the task was done, making the signal useful
        even when there's only one agent.

        Args:
            task_type: Type of task (e.g., 'analysis', 'coding')
            approach_summary: Short description of the approach taken
            tools_used: List of tool/skill names used during execution
            success: Whether the approach succeeded
            quality: Output quality 0-1
            agent: Agent name (for tracking, even if there's only one)

        Returns:
            Signal ID
        """
        content = {
            'task_type': task_type,
            'approach': approach_summary[:200],
            'tools': tools_used[:10],
            'success': success,
            'quality': quality,
        }

        if success:
            strength = 0.5 + 0.5 * max(0.0, min(1.0, quality))
            signal_type = 'approach_success'
        else:
            strength = 0.3
            signal_type = 'approach_warning'

        # Check for existing approach signal for this task_type + approach combo
        approach_key = f"{task_type}:{approach_summary[:50]}"
        for sid, signal in self.signals.items():
            if (signal.signal_type in ('approach_success', 'approach_warning')
                    and isinstance(signal.content, dict)
                    and signal.content.get('task_type') == task_type
                    and signal.content.get('approach', '')[:50] == approach_summary[:50]):
                if success:
                    self.reinforce(sid, amount=0.15 + 0.1 * quality)
                else:
                    signal.strength = max(0.01, signal.strength - 0.1)
                return sid

        return self.deposit(
            signal_type=signal_type,
            content=content,
            agent=agent,
            strength=strength,
        )

    def recommend_approach(self, task_type: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Get approach recommendations for a task type.

        Returns actionable guidance: what tools to use, what approaches
        worked, and what to avoid. This is the single-agent analog of
        recommend_agent().

        Args:
            task_type: Type of task
            top_k: Number of top approaches to return

        Returns:
            Dict with 'use' (successful approaches) and 'avoid' (failed ones).
            Each entry has approach summary, tools used, and signal strength.
        """
        self._apply_decay()

        successes = []
        failures = []

        for signal in self.signals.values():
            if not isinstance(signal.content, dict):
                continue
            if signal.content.get('task_type') != task_type:
                continue
            if signal.strength < 0.05:
                continue

            entry = {
                'approach': signal.content.get('approach', ''),
                'tools': signal.content.get('tools', []),
                'quality': signal.content.get('quality', 0.5),
                'strength': signal.strength,
            }

            if signal.signal_type == 'approach_success':
                successes.append(entry)
            elif signal.signal_type == 'approach_warning':
                failures.append(entry)

        successes.sort(key=lambda x: x['strength'], reverse=True)
        failures.sort(key=lambda x: x['strength'], reverse=True)

        return {
            'use': successes[:top_k],
            'avoid': failures[:top_k],
        }

    def evaporate(self, decay_rate: float = None) -> int:
        """
        Public evaporation method — decay all signals and prune dead ones.

        Call this periodically (e.g., pre/post execution) to clean up stale
        signals that no one is actively sensing. Without periodic evaporation,
        signals only decay lazily on sense() calls, meaning old signals persist
        indefinitely if nobody queries their type.

        Args:
            decay_rate: Override decay rate (uses instance default if None)

        Returns:
            Number of signals pruned
        """
        rate = decay_rate if decay_rate is not None else self.decay_rate
        before = len(self.signals)

        # Apply decay to every signal
        for signal in self.signals.values():
            signal.decay(rate)

        # Remove dead signals
        self._prune_weak_signals()

        pruned = before - len(self.signals)
        if pruned > 0:
            logger.debug(f"Stigmergy evaporation: pruned {pruned} signals ({len(self.signals)} remaining)")
        return pruned

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
# CROSS-SWARM STIGMERGY (Shared coordination across swarm instances)
# =============================================================================

class CrossSwarmStigmergy:
    """
    Cross-swarm coordination via shared stigmergy signals.

    Unlike the per-swarm StigmergyLayer (which duplicates AgentProfile.task_success),
    this provides something AgentProfile can't: **inter-swarm communication**.

    Use cases:
    - CodingSwarm deposits "code_quality:high" -> TestingSwarm picks it up
    - DataAnalysisSwarm warns about bad data -> any swarm avoids that source
    - ResearchSwarm shares discovered patterns -> all swarms benefit
    - Any swarm can broadcast capability announcements for task delegation

    Signal types unique to cross-swarm:
    - 'capability': Announce what a swarm can do well
    - 'data_quality': Share data quality observations
    - 'delegation': Request another swarm to handle a sub-task
    - 'insight': Share discovered patterns across swarms
    """

    # Singleton shared layer for all swarms in this process
    _shared_instance: 'CrossSwarmStigmergy' = None
    _lock = None

    def __init__(self, decay_rate: float = 0.05, max_signals: int = 1000):
        self._layer = StigmergyLayer(decay_rate=decay_rate, max_signals=max_signals)
        self._swarm_registry: Dict[str, Dict[str, Any]] = {}  # swarm_name -> info

    @classmethod
    def get_shared(cls) -> 'CrossSwarmStigmergy':
        """Get the process-wide shared stigmergy layer."""
        if cls._shared_instance is None:
            cls._shared_instance = cls()
        return cls._shared_instance

    def register_swarm(self, swarm_name: str, domain: str = "general",
                       capabilities: List[str] = None):
        """Register a swarm for cross-swarm coordination."""
        self._swarm_registry[swarm_name] = {
            'domain': domain,
            'capabilities': capabilities or [],
            'registered_at': time.time()
        }
        # Announce capabilities
        if capabilities:
            self._layer.deposit(
                signal_type='capability',
                content={
                    'swarm': swarm_name,
                    'domain': domain,
                    'capabilities': capabilities
                },
                agent=swarm_name,
                strength=1.0
            )

    def share_insight(self, from_swarm: str, insight_type: str,
                      content: Any, strength: float = 0.8) -> str:
        """
        Share an insight from one swarm to all others.

        Args:
            from_swarm: Originating swarm name
            insight_type: Type of insight ('data_quality', 'pattern', 'warning')
            content: Insight content
            strength: Signal strength

        Returns:
            Signal ID
        """
        return self._layer.deposit(
            signal_type='insight',
            content={
                'from_swarm': from_swarm,
                'insight_type': insight_type,
                'data': content
            },
            agent=from_swarm,
            strength=strength,
            metadata={'insight_type': insight_type}
        )

    def request_delegation(self, from_swarm: str, task_type: str,
                           task_description: str, urgency: float = 0.5) -> str:
        """
        Request another swarm to handle a sub-task.

        Args:
            from_swarm: Requesting swarm
            task_type: Type of task to delegate
            task_description: What needs to be done
            urgency: How urgent (0-1)

        Returns:
            Signal ID for tracking
        """
        return self._layer.deposit(
            signal_type='delegation',
            content={
                'from_swarm': from_swarm,
                'task_type': task_type,
                'description': task_description,
                'urgency': urgency
            },
            agent=from_swarm,
            strength=urgency
        )

    def check_delegations(self, for_swarm: str) -> List[Dict]:
        """
        Check for pending delegation requests relevant to this swarm.

        Returns delegation requests where the swarm has matching capabilities.
        """
        delegations = self._layer.sense(signal_type='delegation', min_strength=0.1)
        swarm_info = self._swarm_registry.get(for_swarm, {})
        capabilities = swarm_info.get('capabilities', [])

        relevant = []
        for signal in delegations:
            content = signal.content
            if isinstance(content, dict):
                # Don't return own delegations
                if content.get('from_swarm') == for_swarm:
                    continue
                # Check capability match
                task_type = content.get('task_type', '')
                if any(cap in task_type for cap in capabilities) or not capabilities:
                    relevant.append({
                        'signal_id': signal.signal_id,
                        'from_swarm': content.get('from_swarm'),
                        'task_type': task_type,
                        'description': content.get('description', ''),
                        'urgency': content.get('urgency', 0.5),
                        'strength': signal.strength
                    })

        return sorted(relevant, key=lambda x: x['urgency'], reverse=True)

    def get_insights_for(self, swarm_name: str, insight_type: str = None,
                         min_strength: float = 0.2) -> List[Dict]:
        """
        Get cross-swarm insights relevant to a swarm.

        Args:
            swarm_name: Receiving swarm
            insight_type: Filter by type (None = all)
            min_strength: Minimum signal strength

        Returns:
            List of insight dicts
        """
        signals = self._layer.sense(signal_type='insight', min_strength=min_strength)
        insights = []
        for signal in signals:
            content = signal.content
            if isinstance(content, dict):
                if content.get('from_swarm') == swarm_name:
                    continue  # Skip own insights
                if insight_type and content.get('insight_type') != insight_type:
                    continue
                insights.append({
                    'from_swarm': content.get('from_swarm'),
                    'insight_type': content.get('insight_type'),
                    'data': content.get('data'),
                    'strength': signal.strength
                })

        return insights

    def find_capable_swarm(self, task_type: str) -> Optional[str]:
        """
        Find a swarm capable of handling a task type.

        Uses capability announcements to match task to swarm.
        """
        capability_signals = self._layer.sense(
            signal_type='capability', min_strength=0.3
        )

        best_swarm = None
        best_strength = 0.0

        for signal in capability_signals:
            content = signal.content
            if isinstance(content, dict):
                capabilities = content.get('capabilities', [])
                if any(cap in task_type or task_type in cap for cap in capabilities):
                    if signal.strength > best_strength:
                        best_strength = signal.strength
                        best_swarm = content.get('swarm')

        return best_swarm

    def evaporate(self) -> None:
        """Evaporate stale cross-swarm signals."""
        self._layer.evaporate()

    @property
    def registered_swarms(self) -> List[str]:
        """Get list of registered swarm names."""
        return list(self._swarm_registry.keys())


__all__ = [
    'StigmergySignal',
    'StigmergyLayer',
    'CrossSwarmStigmergy',
]

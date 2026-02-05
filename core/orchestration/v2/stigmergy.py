"""
Stigmergy Layer (Indirect Agent Coordination)
===============================================

Implements ant-colony-inspired stigmergy:
- StigmergySignal: Pheromone-like signal in shared environment
- StigmergyLayer: Shared artifact store for indirect coordination

Agents deposit signals, other agents sense and react.
Signals decay over time; successful paths get reinforced.

Extracted from swarm_intelligence.py for modularity.
"""

import time
import hashlib
import logging
from typing import Dict, List, Any
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


__all__ = [
    'StigmergySignal',
    'StigmergyLayer',
]

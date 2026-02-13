"""
Swarm Benchmarks
=================

Performance benchmarking for swarm intelligence:
- SwarmMetrics: Metrics container for swarm evaluation
- SwarmBenchmarks: Benchmark suite for comparing performance

Tracks single-agent vs multi-agent speedup, communication overhead,
specialization emergence, and cooperation effectiveness.

Extracted from swarm_intelligence.py for modularity.
"""

import time
import math
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .swarm_data_structures import AgentProfile, AgentSpecialization

logger = logging.getLogger(__name__)


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

        # Iteration history for self-improvement tracking
        self.iteration_history: List[Dict] = []

    def record_iteration(self, iteration_id: str, task_type: str,
                         score: float, execution_time: float, success: bool):
        """Record a self-improvement iteration."""
        self.iteration_history.append({
            'iteration_id': iteration_id, 'task_type': task_type,
            'score': score, 'execution_time': execution_time,
            'success': success, 'timestamp': time.time()
        })
        if len(self.iteration_history) > 200:
            self.iteration_history = self.iteration_history[-200:]

    def get_improvement_trend(self, task_type: str = None, window: int = 10) -> Dict[str, Any]:
        """Split-half comparison with stddev-based noise-aware threshold."""
        history = getattr(self, 'iteration_history', [])
        if task_type:
            history = [h for h in history if h['task_type'] == task_type]
        if len(history) < 2:
            return {'trend': 'insufficient_data', 'delta': 0.0, 'iterations': len(history)}
        recent = history[-window:]
        mid = len(recent) // 2
        avg_first = sum(h['score'] for h in recent[:mid]) / mid
        avg_second = sum(h['score'] for h in recent[mid:]) / len(recent[mid:])
        delta = avg_second - avg_first

        # Compute stddev of the full recent window for noise-aware threshold
        scores = [h['score'] for h in recent]
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        stddev = math.sqrt(variance)
        threshold = max(0.5 * stddev, 0.01)

        trend = 'improving' if delta > threshold else ('declining' if delta < -threshold else 'stable')
        return {'trend': trend, 'delta': delta, 'avg_recent': avg_second,
                'avg_earlier': avg_first, 'iterations': len(recent),
                'stddev': stddev, 'threshold': threshold}

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
            'cooperation_events': self.cooperation_events[-100:],
            'iteration_history': getattr(self, 'iteration_history', [])[-200:],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SwarmBenchmarks':
        """Deserialize from persistence."""
        instance = cls()
        instance.single_agent_runs = defaultdict(list, data.get('single_agent_runs', {}))
        instance.multi_agent_runs = defaultdict(list, data.get('multi_agent_runs', {}))
        instance.communication_events = data.get('communication_events', [])
        instance.cooperation_events = data.get('cooperation_events', [])
        instance.iteration_history = data.get('iteration_history', [])
        return instance


__all__ = [
    'SwarmMetrics',
    'SwarmBenchmarks',
]

"""
MetricsCollector: Observability for the MAS/Swarm System
=========================================================

Answers the question: "Is the system getting better over time?"

Tracks:
- Success rate by swarm/agent/task_type over time windows
- Average execution time trends
- Memory value distribution shifts
- Coordination protocol hit rates
- RL learning progress (TD-Lambda convergence)
- Error rates by category

No external dependencies -- uses simple in-memory time-series with
configurable retention. Can be serialized to disk alongside
SwarmIntelligence state.
"""

import time
import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class TimeWindow:
    """
    Fixed-size time window for metric aggregation.

    Stores recent data points and provides efficient aggregation
    (mean, sum, count, percentiles) over configurable time windows.
    """

    def __init__(self, max_age_seconds: float = 3600, max_points: int = 10000):
        self.max_age = max_age_seconds
        self.max_points = max_points
        self._points: deque = deque(maxlen=max_points)

    def add(self, value: float, labels: Dict[str, str] = None) -> None:
        """Add a data point."""
        self._points.append(MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {}
        ))

    def _prune(self):
        """Remove expired points."""
        cutoff = time.time() - self.max_age
        while self._points and self._points[0].timestamp < cutoff:
            self._points.popleft()

    def count(self) -> int:
        self._prune()
        return len(self._points)

    def mean(self) -> float:
        self._prune()
        if not self._points:
            return 0.0
        return sum(p.value for p in self._points) / len(self._points)

    def sum(self) -> float:
        self._prune()
        return sum(p.value for p in self._points)

    def rate(self) -> float:
        """Rate: fraction of values that are > 0 (for success tracking)."""
        self._prune()
        if not self._points:
            return 0.0
        return sum(1 for p in self._points if p.value > 0) / len(self._points)

    def percentile(self, p: float) -> float:
        """Get p-th percentile (0-100)."""
        self._prune()
        if not self._points:
            return 0.0
        values = sorted(pt.value for pt in self._points)
        idx = int(len(values) * p / 100)
        idx = min(idx, len(values) - 1)
        return values[idx]

    def trend(self, window_split: int = 2) -> float:
        """
        Compute trend: positive = improving, negative = declining.

        Splits the window into halves and compares means.
        Returns the difference (second_half_mean - first_half_mean).
        """
        self._prune()
        if len(self._points) < 4:
            return 0.0

        points_list = list(self._points)
        mid = len(points_list) // 2
        first_half = [p.value for p in points_list[:mid]]
        second_half = [p.value for p in points_list[mid:]]

        first_mean = sum(first_half) / len(first_half)
        second_mean = sum(second_half) / len(second_half)

        return second_mean - first_mean

    def by_label(self, label_key: str) -> Dict[str, 'TimeWindow']:
        """Split points by a label value into separate windows."""
        self._prune()
        groups: Dict[str, TimeWindow] = {}
        for point in self._points:
            label_val = point.labels.get(label_key, 'unknown')
            if label_val not in groups:
                groups[label_val] = TimeWindow(self.max_age, self.max_points)
            groups[label_val]._points.append(point)
        return groups

    def to_list(self, limit: int = 100) -> List[Dict]:
        """Serialize recent points."""
        self._prune()
        return [
            {'t': p.timestamp, 'v': p.value, 'l': p.labels}
            for p in list(self._points)[-limit:]
        ]


class MetricsCollector:
    """
    Central metrics collector for the MAS/Swarm system.

    Usage:
        metrics = MetricsCollector()

        # Record events
        metrics.record_task('CodingSwarm', 'Developer', 'coding', success=True, duration=2.5)
        metrics.record_task('CodingSwarm', 'Reviewer', 'review', success=False, duration=1.0)
        metrics.record_coordination('circuit_breaker', agent='Developer')
        metrics.record_learning(td_error=0.15, task_type='coding')

        # Query metrics
        report = metrics.get_report()
        print(report['success_rate'])  # Overall success rate
        print(report['trends'])        # Is system improving?
    """

    # Singleton for process-wide metrics
    _instance: 'MetricsCollector' = None

    def __init__(self, retention_seconds: float = 7200):
        """
        Args:
            retention_seconds: How long to keep data points (default: 2 hours)
        """
        self.retention = retention_seconds

        # Task metrics
        self.task_success = TimeWindow(retention_seconds)
        self.task_duration = TimeWindow(retention_seconds)
        self.task_count = TimeWindow(retention_seconds)

        # Coordination protocol metrics
        self.coordination_events = TimeWindow(retention_seconds)

        # Learning metrics
        self.td_errors = TimeWindow(retention_seconds)
        self.rl_values = TimeWindow(retention_seconds)
        self.learning_updates = TimeWindow(retention_seconds)

        # Error metrics
        self.errors = TimeWindow(retention_seconds)

        # Memory metrics
        self.memory_retrievals = TimeWindow(retention_seconds)
        self.memory_stores = TimeWindow(retention_seconds)

        logger.debug("MetricsCollector initialized")

    @classmethod
    def get_global(cls) -> 'MetricsCollector':
        """Get process-wide metrics collector singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # =========================================================================
    # RECORDING METHODS
    # =========================================================================

    def record_task(
        self,
        swarm: str,
        agent: str,
        task_type: str,
        success: bool,
        duration: float = 0.0,
    ):
        """Record a task execution."""
        labels = {'swarm': swarm, 'agent': agent, 'task_type': task_type}
        self.task_success.add(1.0 if success else 0.0, labels)
        self.task_duration.add(duration, labels)
        self.task_count.add(1.0, labels)

    def record_coordination(
        self,
        protocol: str,
        agent: str = "",
        success: bool = True,
    ):
        """
        Record a coordination protocol event.

        Args:
            protocol: Protocol name ('circuit_breaker', 'gossip', 'coalition',
                      'auction', 'byzantine', 'load_balance', 'work_steal')
            agent: Agent involved
            success: Whether the protocol action succeeded
        """
        self.coordination_events.add(
            1.0 if success else 0.0,
            {'protocol': protocol, 'agent': agent}
        )

    def record_learning(
        self,
        td_error: float = 0.0,
        value_update: float = 0.0,
        task_type: str = "",
    ):
        """Record a learning update from TD-Lambda."""
        labels = {'task_type': task_type}
        if td_error != 0.0:
            self.td_errors.add(abs(td_error), labels)
        if value_update != 0.0:
            self.learning_updates.add(value_update, labels)

    def record_error(
        self,
        category: str,
        component: str = "",
        message: str = "",
    ):
        """
        Record an error event.

        Args:
            category: Error category ('coordination', 'learning', 'execution', etc.)
            component: Component that produced the error
            message: Error message (truncated)
        """
        self.errors.add(1.0, {
            'category': category,
            'component': component,
            'message': message[:100],
        })

    def record_memory(self, operation: str, count: int = 1) -> None:
        """Record memory operation (retrieve, store, consolidate)."""
        if operation == 'retrieve':
            self.memory_retrievals.add(float(count))
        elif operation == 'store':
            self.memory_stores.add(float(count))

    # =========================================================================
    # QUERYING METHODS
    # =========================================================================

    def get_report(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics report.

        Returns a dict suitable for logging, dashboards, or API responses.
        """
        report = {}

        # Overall success rate
        report['success_rate'] = self.task_success.rate()
        report['total_tasks'] = self.task_count.count()
        report['avg_duration'] = self.task_duration.mean()
        report['p95_duration'] = self.task_duration.percentile(95)

        # Trends
        report['trends'] = {
            'success_rate': self.task_success.trend(),
            'duration': self.task_duration.trend(),
            'td_error': self.td_errors.trend(),
            'error_rate': self.errors.trend(),
        }

        # Interpret trends
        trend_summary = []
        sr_trend = report['trends']['success_rate']
        if sr_trend > 0.05:
            trend_summary.append("Success rate IMPROVING")
        elif sr_trend < -0.05:
            trend_summary.append("Success rate DECLINING")

        td_trend = report['trends']['td_error']
        if td_trend < -0.01:
            trend_summary.append("TD errors CONVERGING (learning working)")
        elif td_trend > 0.05:
            trend_summary.append("TD errors DIVERGING (learning unstable)")

        report['trend_summary'] = trend_summary or ["Stable"]

        # By swarm
        report['by_swarm'] = {}
        swarm_windows = self.task_success.by_label('swarm')
        for swarm_name, window in swarm_windows.items():
            report['by_swarm'][swarm_name] = {
                'success_rate': window.rate(),
                'count': window.count(),
                'trend': window.trend(),
            }

        # By agent
        report['by_agent'] = {}
        agent_windows = self.task_success.by_label('agent')
        for agent_name, window in agent_windows.items():
            report['by_agent'][agent_name] = {
                'success_rate': window.rate(),
                'count': window.count(),
                'trend': window.trend(),
            }

        # By task type
        report['by_task_type'] = {}
        type_windows = self.task_success.by_label('task_type')
        for task_type, window in type_windows.items():
            report['by_task_type'][task_type] = {
                'success_rate': window.rate(),
                'count': window.count(),
            }

        # Coordination protocol usage
        report['coordination'] = {}
        protocol_windows = self.coordination_events.by_label('protocol')
        for protocol, window in protocol_windows.items():
            report['coordination'][protocol] = {
                'count': window.count(),
                'success_rate': window.rate(),
            }

        # Error summary
        report['errors'] = {
            'total': self.errors.count(),
            'rate_per_minute': self.errors.count() / max(1, self.retention / 60),
        }
        error_categories = self.errors.by_label('category')
        report['errors']['by_category'] = {
            cat: window.count() for cat, window in error_categories.items()
        }

        # Learning progress
        report['learning'] = {
            'avg_td_error': self.td_errors.mean(),
            'td_error_trend': self.td_errors.trend(),
            'total_updates': self.learning_updates.count(),
            'converging': self.td_errors.trend() < 0,
        }

        return report

    def format_report(self) -> str:
        """Format metrics report as human-readable text."""
        report = self.get_report()
        lines = [
            "# MAS Metrics Report",
            "=" * 50,
            "",
            f"## Overview",
            f"  Tasks: {report['total_tasks']}",
            f"  Success Rate: {report['success_rate']:.1%}",
            f"  Avg Duration: {report['avg_duration']:.2f}s",
            f"  P95 Duration: {report['p95_duration']:.2f}s",
            "",
            f"## Trends: {', '.join(report['trend_summary'])}",
        ]

        if report['by_swarm']:
            lines.append("\n## By Swarm")
            for name, data in report['by_swarm'].items():
                trend_arrow = "^" if data['trend'] > 0 else "v" if data['trend'] < 0 else "="
                lines.append(
                    f"  {name}: {data['success_rate']:.0%} "
                    f"({data['count']} tasks) {trend_arrow}"
                )

        if report['coordination']:
            lines.append("\n## Coordination Protocols")
            for protocol, data in report['coordination'].items():
                lines.append(
                    f"  {protocol}: {data['count']} events "
                    f"({data['success_rate']:.0%} success)"
                )

        if report['errors']['total'] > 0:
            lines.append(f"\n## Errors: {report['errors']['total']} total")
            for cat, count in report['errors']['by_category'].items():
                lines.append(f"  {cat}: {count}")

        lines.append(f"\n## Learning")
        lines.append(f"  Avg TD Error: {report['learning']['avg_td_error']:.4f}")
        lines.append(f"  Converging: {'Yes' if report['learning']['converging'] else 'No'}")
        lines.append(f"  Total Updates: {report['learning']['total_updates']}")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialize for persistence (recent data only)."""
        return {
            'task_success': self.task_success.to_list(200),
            'task_duration': self.task_duration.to_list(200),
            'coordination': self.coordination_events.to_list(200),
            'td_errors': self.td_errors.to_list(200),
            'errors': self.errors.to_list(200),
        }


__all__ = [
    'MetricsCollector',
    'TimeWindow',
    'MetricPoint',
]

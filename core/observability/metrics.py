"""
Metrics Collector - Per-Agent Execution Metrics
================================================

Tracks execution metrics across agents with statistical aggregation.
Provides p50/p95/p99 latencies, success rates, token usage, and cost breakdowns.

Usage:
    metrics = get_metrics()

    # Record executions
    metrics.record_execution("auto", "research", 12.5, success=True, tokens=1500)
    metrics.record_execution("auto", "research", 8.3, success=True, tokens=1200)
    metrics.record_execution("analyst", "analysis", 5.1, success=False, tokens=800)

    # Get summary
    summary = metrics.get_summary()
    agent_stats = metrics.get_agent_metrics("auto")
"""

import time
import threading
import math
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRecord:
    """Record of a single execution."""
    agent_name: str
    task_type: str
    duration_s: float
    success: bool
    timestamp: float = field(default_factory=time.time)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMetrics:
    """Aggregated metrics for a single agent."""
    agent_name: str
    total_executions: int = 0
    successful: int = 0
    failed: int = 0
    total_duration_s: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_llm_calls: int = 0
    durations: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.successful / self.total_executions

    @property
    def avg_duration_s(self) -> float:
        if not self.durations:
            return 0.0
        return sum(self.durations) / len(self.durations)

    @property
    def avg_tokens_per_execution(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.total_tokens / self.total_executions

    @property
    def avg_cost_per_execution(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.total_cost_usd / self.total_executions

    def _percentile(self, values: List[float], pct: float) -> float:
        """Calculate percentile from sorted values."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = (pct / 100) * (len(sorted_vals) - 1)
        lower = int(math.floor(idx))
        upper = int(math.ceil(idx))
        if lower == upper:
            return sorted_vals[lower]
        frac = idx - lower
        return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac

    @property
    def p50_duration_s(self) -> float:
        return self._percentile(self.durations, 50)

    @property
    def p95_duration_s(self) -> float:
        return self._percentile(self.durations, 95)

    @property
    def p99_duration_s(self) -> float:
        return self._percentile(self.durations, 99)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_name': self.agent_name,
            'total_executions': self.total_executions,
            'successful': self.successful,
            'failed': self.failed,
            'success_rate': round(self.success_rate, 4),
            'latency': {
                'avg_s': round(self.avg_duration_s, 3),
                'p50_s': round(self.p50_duration_s, 3),
                'p95_s': round(self.p95_duration_s, 3),
                'p99_s': round(self.p99_duration_s, 3),
            },
            'tokens': {
                'total': self.total_tokens,
                'avg_per_execution': round(self.avg_tokens_per_execution, 0),
            },
            'cost': {
                'total_usd': round(self.total_cost_usd, 6),
                'avg_per_execution_usd': round(self.avg_cost_per_execution, 6),
            },
            'llm_calls': self.total_llm_calls,
        }


class MetricsCollector:
    """
    Collects execution metrics across agents.

    Thread-safe. Maintains a bounded history of execution records
    and provides aggregated metrics per agent and globally.
    """

    def __init__(self, max_history: int = 10000):
        self._records: List[ExecutionRecord] = []
        self._agent_metrics: Dict[str, AgentMetrics] = {}
        self._task_type_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._max_history = max_history

        # Session-level counters
        self._session_start = time.time()
        self._total_cost_usd = 0.0
        self._total_tokens = 0
        self._total_llm_calls = 0

    def record_execution(
        self,
        agent_name: str,
        task_type: str,
        duration_s: float,
        success: bool,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        llm_calls: int = 0,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record an agent execution.

        Args:
            agent_name: Name of the agent
            task_type: Type of task (research, analysis, creation, etc.)
            duration_s: Duration in seconds
            success: Whether execution succeeded
            input_tokens: Input tokens consumed
            output_tokens: Output tokens generated
            cost_usd: Cost in USD
            llm_calls: Number of LLM API calls
            error: Error message if failed
            metadata: Additional metadata
        """
        record = ExecutionRecord(
            agent_name=agent_name,
            task_type=task_type,
            duration_s=duration_s,
            success=success,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            llm_calls=llm_calls,
            error=error,
            metadata=metadata or {},
        )

        with self._lock:
            # Store record
            self._records.append(record)
            if len(self._records) > self._max_history:
                self._records = self._records[-self._max_history:]

            # Update agent metrics
            if agent_name not in self._agent_metrics:
                self._agent_metrics[agent_name] = AgentMetrics(agent_name=agent_name)

            am = self._agent_metrics[agent_name]
            am.total_executions += 1
            if success:
                am.successful += 1
            else:
                am.failed += 1
            am.total_duration_s += duration_s
            am.durations.append(duration_s)
            am.total_tokens += input_tokens + output_tokens
            am.total_cost_usd += cost_usd
            am.total_llm_calls += llm_calls

            # Keep durations bounded
            if len(am.durations) > 1000:
                am.durations = am.durations[-1000:]

            # Global counters
            self._task_type_counts[task_type] += 1
            self._total_cost_usd += cost_usd
            self._total_tokens += input_tokens + output_tokens
            self._total_llm_calls += llm_calls

    def get_agent_metrics(self, agent_name: str) -> Optional[AgentMetrics]:
        """Get metrics for a specific agent."""
        with self._lock:
            return self._agent_metrics.get(agent_name)

    def get_all_agent_metrics(self) -> Dict[str, AgentMetrics]:
        """Get metrics for all agents."""
        with self._lock:
            return dict(self._agent_metrics)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive metrics summary.

        Returns:
            Dict with global stats, per-agent stats, and task type breakdown.
        """
        with self._lock:
            total_records = len(self._records)
            total_success = sum(1 for r in self._records if r.success)
            session_duration = time.time() - self._session_start

            # Global latency percentiles
            all_durations = [r.duration_s for r in self._records]
            all_durations_sorted = sorted(all_durations)

            def _pct(vals, p):
                if not vals:
                    return 0.0
                idx = (p / 100) * (len(vals) - 1)
                lo, hi = int(math.floor(idx)), int(math.ceil(idx))
                if lo == hi:
                    return vals[lo]
                frac = idx - lo
                return vals[lo] * (1 - frac) + vals[hi] * frac

            return {
                'session': {
                    'duration_s': round(session_duration, 1),
                    'start_time': self._session_start,
                },
                'global': {
                    'total_executions': total_records,
                    'successful': total_success,
                    'failed': total_records - total_success,
                    'success_rate': round(total_success / total_records, 4) if total_records else 0,
                    'total_tokens': self._total_tokens,
                    'total_cost_usd': round(self._total_cost_usd, 6),
                    'total_llm_calls': self._total_llm_calls,
                    'latency': {
                        'p50_s': round(_pct(all_durations_sorted, 50), 3),
                        'p95_s': round(_pct(all_durations_sorted, 95), 3),
                        'p99_s': round(_pct(all_durations_sorted, 99), 3),
                    },
                },
                'per_agent': {
                    name: am.to_dict()
                    for name, am in self._agent_metrics.items()
                },
                'task_types': dict(self._task_type_counts),
            }

    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get cost breakdown by agent and model."""
        with self._lock:
            by_agent = {}
            by_model = defaultdict(float)

            for record in self._records:
                if record.agent_name not in by_agent:
                    by_agent[record.agent_name] = {
                        'cost_usd': 0.0, 'tokens': 0, 'calls': 0
                    }
                by_agent[record.agent_name]['cost_usd'] += record.cost_usd
                by_agent[record.agent_name]['tokens'] += record.input_tokens + record.output_tokens
                by_agent[record.agent_name]['calls'] += record.llm_calls

                model = record.metadata.get('model', 'unknown')
                by_model[model] += record.cost_usd

            return {
                'total_cost_usd': round(self._total_cost_usd, 6),
                'by_agent': {
                    k: {kk: round(vv, 6) if isinstance(vv, float) else vv
                         for kk, vv in v.items()}
                    for k, v in by_agent.items()
                },
                'by_model': {k: round(v, 6) for k, v in by_model.items()},
            }

    def recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent error records."""
        with self._lock:
            errors = [r for r in reversed(self._records) if not r.success and r.error]
            return [
                {
                    'agent': r.agent_name,
                    'task_type': r.task_type,
                    'error': r.error,
                    'timestamp': r.timestamp,
                    'duration_s': r.duration_s,
                }
                for r in errors[:limit]
            ]

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._records.clear()
            self._agent_metrics.clear()
            self._task_type_counts.clear()
            self._session_start = time.time()
            self._total_cost_usd = 0.0
            self._total_tokens = 0
            self._total_llm_calls = 0


# =========================================================================
# SINGLETON
# =========================================================================

_global_metrics: Optional[MetricsCollector] = None
_metrics_lock = threading.Lock()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector singleton."""
    global _global_metrics
    if _global_metrics is None:
        with _metrics_lock:
            if _global_metrics is None:
                _global_metrics = MetricsCollector()
    return _global_metrics


def reset_metrics() -> None:
    """Reset the global metrics collector (for testing)."""
    global _global_metrics
    with _metrics_lock:
        if _global_metrics:
            _global_metrics.reset()
        _global_metrics = None

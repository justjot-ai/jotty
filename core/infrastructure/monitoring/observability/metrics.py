"""
Prometheus Metrics for Jotty

Tracks:
- Skill execution counts and latencies
- Agent performance metrics
- LLM token usage and costs
- Memory operation stats
- Error rates
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import Prometheus client, fall back to no-op
try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info("Prometheus client not installed. Install with: pip install prometheus-client")


class NoOpMetric:
    """No-op metric when Prometheus not available."""

    def inc(self, *args: Any, **kwargs: Any) -> None:
        pass

    def dec(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set(self, *args: Any, **kwargs: Any) -> None:
        pass

    def observe(self, *args: Any, **kwargs: Any) -> None:
        pass

    def labels(self, *args: Any, **kwargs: Any) -> Any:
        return self


# =============================================================================
# ExecutionRecord
# =============================================================================


@dataclass
class ExecutionRecord:
    """A single recorded execution event."""

    agent_name: str
    task_type: str
    duration_s: float
    success: bool
    timestamp: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# =============================================================================
# AgentMetrics
# =============================================================================


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
        """Compute the *pct*-th percentile of *values* using linear interpolation."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n == 1:
            return sorted_vals[0]
        idx = (pct / 100.0) * (n - 1)
        lower = int(idx)
        upper = lower + 1
        if upper >= n:
            return sorted_vals[-1]
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
        """Serialize to a dictionary."""
        return {
            "agent_name": self.agent_name,
            "total_executions": self.total_executions,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": round(self.success_rate, 4),
            "latency": {
                "avg_s": round(self.avg_duration_s, 3),
                "p50_s": round(self.p50_duration_s, 3),
                "p95_s": round(self.p95_duration_s, 3),
                "p99_s": round(self.p99_duration_s, 3),
            },
            "tokens": {
                "total": self.total_tokens,
                "avg_per_execution": self.avg_tokens_per_execution,
            },
            "cost": {
                "total_usd": round(self.total_cost_usd, 6),
                "avg_per_execution_usd": round(self.avg_cost_per_execution, 6),
            },
            "llm_calls": self.total_llm_calls,
        }


# =============================================================================
# MetricsCollector
# =============================================================================


class MetricsCollector:
    """
    Jotty metrics collector with optional Prometheus export.

    Provides observability into:
    - Skill executions (count, duration, errors)
    - Agent performance
    - LLM usage (tokens, cost)
    - Memory operations

    Internal tracking (records, agent metrics, errors) is always active
    regardless of whether Prometheus is available.
    """

    def __init__(
        self,
        enabled: bool = True,
        registry: Optional["CollectorRegistry"] = None,
        max_history: int = 10000,
    ) -> None:
        """
        Initialize metrics collector.

        Args:
            enabled: Enable Prometheus metrics export
            registry: Prometheus registry (creates new if None)
            max_history: Maximum number of ExecutionRecords to retain
        """
        self._prometheus_enabled = enabled and PROMETHEUS_AVAILABLE

        # --- Internal tracking (always active) ---
        self._lock = threading.Lock()
        self._max_history: int = max_history
        self._records: List[ExecutionRecord] = []
        self._agent_metrics: Dict[str, AgentMetrics] = {}
        self._session_start: float = time.time()
        self._total_cost_usd: float = 0.0
        self._total_tokens: int = 0
        self._total_llm_calls: int = 0
        self._task_types: Dict[str, int] = {}

        # --- Prometheus metrics (optional) ---
        if self._prometheus_enabled:
            self.registry = registry or CollectorRegistry()

            self.skill_executions = Counter(
                "jotty_skill_executions_total",
                "Total skill executions",
                ["skill_name", "status"],
                registry=self.registry,
            )
            self.skill_duration = Histogram(
                "jotty_skill_duration_seconds",
                "Skill execution duration",
                ["skill_name"],
                registry=self.registry,
            )
            self.agent_executions = Counter(
                "jotty_agent_executions_total",
                "Total agent executions",
                ["agent_name", "status"],
                registry=self.registry,
            )
            self.llm_tokens = Counter(
                "jotty_llm_tokens_total",
                "Total LLM tokens used",
                ["model", "type"],
                registry=self.registry,
            )
            self.llm_cost = Counter(
                "jotty_llm_cost_dollars",
                "Total LLM cost in dollars",
                ["model"],
                registry=self.registry,
            )
            self.llm_calls = Counter(
                "jotty_llm_calls_total",
                "Total LLM API calls",
                ["model", "status"],
                registry=self.registry,
            )
            self.memory_operations = Counter(
                "jotty_memory_operations_total",
                "Memory operations",
                ["operation", "level"],
                registry=self.registry,
            )
            self.memory_size = Gauge(
                "jotty_memory_size_bytes",
                "Memory storage size",
                ["level"],
                registry=self.registry,
            )
            self.errors = Counter(
                "jotty_errors_total",
                "Total errors",
                ["component", "error_type"],
                registry=self.registry,
            )
            logger.info("Prometheus metrics enabled")
        else:
            self.skill_executions = NoOpMetric()
            self.skill_duration = NoOpMetric()
            self.agent_executions = NoOpMetric()
            self.llm_tokens = NoOpMetric()
            self.llm_cost = NoOpMetric()
            self.llm_calls = NoOpMetric()
            self.memory_operations = NoOpMetric()
            self.memory_size = NoOpMetric()
            self.errors = NoOpMetric()

            if not PROMETHEUS_AVAILABLE:
                logger.info("Prometheus not available - using no-op metrics")

    # --- legacy Prometheus-only helpers ----------------------------------------

    def record_skill_execution(self, skill_name: str, duration: float, success: bool) -> Any:
        """Record skill execution metrics (Prometheus only)."""
        status = "success" if success else "error"
        self.skill_executions.labels(skill_name=skill_name, status=status).inc()
        self.skill_duration.labels(skill_name=skill_name).observe(duration)

    def record_llm_call(
        self, model: str, input_tokens: int, output_tokens: int, cost: float, success: bool
    ) -> Any:
        """Record LLM API call metrics (Prometheus only)."""
        status = "success" if success else "error"
        self.llm_calls.labels(model=model, status=status).inc()
        self.llm_tokens.labels(model=model, type="input").inc(input_tokens)
        self.llm_tokens.labels(model=model, type="output").inc(output_tokens)
        self.llm_cost.labels(model=model).inc(cost)

    def record_error(self, component: str, error_type: str) -> Any:
        """Record error occurrence (Prometheus only)."""
        self.errors.labels(component=component, error_type=error_type).inc()

    # --- main recording method ------------------------------------------------

    def record_execution(
        self,
        agent_name: str,
        task_type: str,
        duration_s: float,
        success: bool,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        cost_usd: float = 0.0,
        llm_calls: int = 0,
        error: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Record execution metrics for an agent/tier.

        Args:
            agent_name: Name of the agent/tier
            task_type: Type of task
            duration_s: Duration in seconds
            success: Whether execution succeeded
            input_tokens: Input tokens used
            output_tokens: Output tokens generated
            cost: Cost in dollars (legacy)
            cost_usd: Cost in USD
            llm_calls: Number of LLM calls
            error: Error message if failed
            metadata: Additional metadata
        """
        actual_cost = cost_usd if cost_usd else cost
        if metadata is None:
            metadata = {}

        record = ExecutionRecord(
            agent_name=agent_name,
            task_type=task_type,
            duration_s=duration_s,
            success=success,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=actual_cost,
            llm_calls=llm_calls,
            error=error,
            metadata=metadata,
        )

        with self._lock:
            # Append record and trim to max_history
            self._records.append(record)
            if len(self._records) > self._max_history:
                self._records = self._records[-self._max_history :]

            # Update global counters
            total_tokens = input_tokens + output_tokens
            self._total_tokens += total_tokens
            self._total_cost_usd += actual_cost
            self._total_llm_calls += llm_calls

            # Update task type counts
            self._task_types[task_type] = self._task_types.get(task_type, 0) + 1

            # Update per-agent aggregated metrics
            if agent_name not in self._agent_metrics:
                self._agent_metrics[agent_name] = AgentMetrics(agent_name=agent_name)
            am = self._agent_metrics[agent_name]
            am.total_executions += 1
            if success:
                am.successful += 1
            else:
                am.failed += 1
            am.total_duration_s += duration_s
            am.total_tokens += total_tokens
            am.total_cost_usd += actual_cost
            am.total_llm_calls += llm_calls
            am.durations.append(duration_s)
            # Trim durations to 1000
            if len(am.durations) > 1000:
                am.durations = am.durations[-1000:]

        # Prometheus (best-effort, never blocks internal tracking)
        if self._prometheus_enabled:
            status = "success" if success else "error"
            self.agent_executions.labels(agent_name=agent_name, status=status).inc()
            if input_tokens or output_tokens:
                model = "claude"
                self.llm_tokens.labels(model=model, type="input").inc(input_tokens)
                self.llm_tokens.labels(model=model, type="output").inc(output_tokens)
            if actual_cost:
                self.llm_cost.labels(model="claude").inc(actual_cost)

    # --- query methods --------------------------------------------------------

    def get_agent_metrics(self, agent_name: str) -> Optional[AgentMetrics]:
        """Return aggregated metrics for *agent_name*, or ``None`` if unknown."""
        with self._lock:
            return self._agent_metrics.get(agent_name)

    def get_all_agent_metrics(self) -> Dict[str, AgentMetrics]:
        """Return a *copy* of the per-agent metrics dict."""
        with self._lock:
            return dict(self._agent_metrics)

    def get_summary(self) -> Dict[str, Any]:
        """Return a full summary of all collected metrics."""
        with self._lock:
            total_execs = sum(am.total_executions for am in self._agent_metrics.values())
            total_successful = sum(am.successful for am in self._agent_metrics.values())
            total_failed = sum(am.failed for am in self._agent_metrics.values())
            all_durations = [r.duration_s for r in self._records]

            # Build a temporary AgentMetrics just for percentile computation
            _helper = AgentMetrics(agent_name="_global", durations=all_durations)

            return {
                "session": {
                    "start_time": self._session_start,
                    "duration_s": round(time.time() - self._session_start, 3),
                },
                "global": {
                    "total_executions": total_execs,
                    "successful": total_successful,
                    "failed": total_failed,
                    "success_rate": (total_successful / total_execs) if total_execs else 0,
                    "total_tokens": self._total_tokens,
                    "total_cost_usd": round(self._total_cost_usd, 6),
                    "total_llm_calls": self._total_llm_calls,
                    "latency": {
                        "p50_s": round(_helper.p50_duration_s, 3),
                        "p95_s": round(_helper.p95_duration_s, 3),
                        "p99_s": round(_helper.p99_duration_s, 3),
                    },
                },
                "per_agent": {name: am.to_dict() for name, am in self._agent_metrics.items()},
                "task_types": dict(self._task_types),
            }

    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Return cost breakdown by agent and by model."""
        with self._lock:
            by_agent: Dict[str, Dict[str, Any]] = {}
            by_model: Dict[str, float] = {}
            total_cost = 0.0

            for record in self._records:
                total_cost += record.cost_usd

                # By agent
                if record.agent_name not in by_agent:
                    by_agent[record.agent_name] = {"cost_usd": 0.0, "tokens": 0, "calls": 0}
                entry = by_agent[record.agent_name]
                entry["cost_usd"] += record.cost_usd
                entry["tokens"] += record.input_tokens + record.output_tokens
                entry["calls"] += record.llm_calls

                # By model
                model = record.metadata.get("model", "unknown")
                by_model[model] = by_model.get(model, 0.0) + record.cost_usd

            return {
                "total_cost_usd": total_cost,
                "by_agent": by_agent,
                "by_model": by_model,
            }

    def recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent error records (most recent first)."""
        with self._lock:
            errors: List[Dict[str, Any]] = []
            for record in reversed(self._records):
                if not record.success and record.error is not None:
                    errors.append(
                        {
                            "agent": record.agent_name,
                            "task_type": record.task_type,
                            "error": record.error,
                            "timestamp": record.timestamp,
                            "duration_s": record.duration_s,
                        }
                    )
                    if len(errors) >= limit:
                        break
            return errors

    # --- lifecycle ------------------------------------------------------------

    def reset(self) -> None:
        """Reset all internal tracking state."""
        with self._lock:
            self._records.clear()
            self._agent_metrics.clear()
            self._task_types.clear()
            self._total_cost_usd = 0.0
            self._total_tokens = 0
            self._total_llm_calls = 0
            self._session_start = time.time()

    def export_metrics(self) -> bytes:
        """
        Export metrics in Prometheus format.

        Returns:
            Metrics as bytes (Prometheus exposition format)
        """
        if self._prometheus_enabled:
            return generate_latest(self.registry)
        return b""


# =============================================================================
# Singleton
# =============================================================================

_metrics: Optional[MetricsCollector] = None


def get_metrics(enabled: bool = True) -> MetricsCollector:
    """
    Get singleton metrics collector.

    Args:
        enabled: Enable metrics

    Returns:
        MetricsCollector instance
    """
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector(enabled=enabled)
    return _metrics


def reset_metrics() -> None:
    """Reset the singleton metrics collector (used in tests)."""
    global _metrics
    _metrics = None

"""
Observability Module for Jotty

Provides:
- Distributed tracing (OpenTelemetry)
- Metrics export (Prometheus)
- Performance monitoring
- Health checks

Usage:
    from Jotty.core.infrastructure.monitoring.observability import tracer, metrics

    with tracer.start_span("skill_execution"):
        result = tool(params)

    metrics.skill_execution_count.inc()
"""

from .tracing import (
    get_tracer, reset_tracer, trace_skill, trace_agent, trace_swarm,
    Span, Trace, TracingContext, SpanStatus,
)
from .metrics import (
    get_metrics, reset_metrics, MetricsCollector,
    ExecutionRecord, AgentMetrics,
)
from .health import HealthCheck

__all__ = [
    'DistributedTracer',
    'get_distributed_tracer',
    'get_tracer',
    'reset_tracer',
    'trace_skill',
    'trace_agent',
    'trace_swarm',
    'Span',
    'Trace',
    'TracingContext',
    'SpanStatus',
    'get_metrics',
    'reset_metrics',
    'MetricsCollector',
    'ExecutionRecord',
    'AgentMetrics',
    'HealthCheck',
]

from .distributed_tracing import (
    DistributedTracer,
    get_distributed_tracer,
)

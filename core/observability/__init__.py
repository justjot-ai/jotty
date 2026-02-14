"""
Observability Module for Jotty

Provides:
- Distributed tracing (OpenTelemetry)
- Metrics export (Prometheus)
- Performance monitoring
- Health checks

Usage:
    from Jotty.core.observability import tracer, metrics

    with tracer.start_span("skill_execution"):
        result = tool(params)

    metrics.skill_execution_count.inc()
"""

from .tracing import get_tracer, trace_skill, trace_agent, trace_swarm
from .metrics import get_metrics, MetricsCollector
from .health import HealthCheck

__all__ = [
    'get_tracer',
    'trace_skill',
    'trace_agent',
    'trace_swarm',
    'get_metrics',
    'MetricsCollector',
    'HealthCheck',
]

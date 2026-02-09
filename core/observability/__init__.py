"""
Observability Module - Tracing, Metrics, and Cost Tracking
==========================================================

Provides production-grade observability for the Jotty agent framework:

1. TRACING: OpenTelemetry-style span trees for debugging multi-agent runs
2. METRICS: Per-agent execution metrics with p50/p95/p99 latencies
3. COST TRACKING: Token/cost tracking per LLM call, per agent, per run

Usage:
    from Jotty.core.observability import get_tracer, get_metrics

    # Tracing
    tracer = get_tracer()
    with tracer.span("swarm_run", goal="Research AI") as span:
        with tracer.span("agent_execute", agent="auto") as child:
            result = await agent.run(goal)
            child.set_attribute("success", result.success)
    trace = tracer.export()  # Full trace tree

    # Metrics
    metrics = get_metrics()
    metrics.record_execution("auto", "research", 12.5, success=True, tokens=1500)
    summary = metrics.get_summary()
"""

from .tracing import (
    Span,
    SpanStatus,
    Trace,
    TracingContext,
    get_tracer,
    reset_tracer,
)
from .metrics import (
    ExecutionRecord,
    AgentMetrics,
    MetricsCollector,
    get_metrics,
    reset_metrics,
)

__all__ = [
    # Tracing
    'Span',
    'SpanStatus',
    'Trace',
    'TracingContext',
    'get_tracer',
    'reset_tracer',
    # Metrics
    'ExecutionRecord',
    'AgentMetrics',
    'MetricsCollector',
    'get_metrics',
    'reset_metrics',
]

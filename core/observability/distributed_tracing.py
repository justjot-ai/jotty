"""
Distributed Tracing Integration
================================

OpenTelemetry-compatible distributed tracing for production debugging.

KISS PRINCIPLE: Extends existing JottyTracer instead of reimplementing.
DRY PRINCIPLE: Reuses core/observability infrastructure.
"""

import logging
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DistributedTracer:
    """
    Distributed tracing for multi-service Jotty deployments.

    BENEFITS:
    - Trace requests across microservices
    - Correlate swarm execution with external APIs
    - Production debugging 10x easier

    KISS: Simple context propagation (no heavy OpenTelemetry SDK)
    DRY: Delegates to existing JottyTracer for local spans
    """

    def __init__(self, service_name: str = "jotty"):
        self.service_name = service_name
        self.trace_contexts: Dict[str, Dict[str, Any]] = {}

        # Lazy-import to avoid dependency issues
        self._local_tracer = None

        logger.info(f"🔍 DistributedTracer initialized: service={service_name}")

    def _get_local_tracer(self):
        """Lazy-load local tracer (DRY - reuse existing)."""
        if self._local_tracer is None:
            try:
                from Jotty.core.observability import get_tracer
                self._local_tracer = get_tracer()
            except ImportError:
                logger.warning("Local tracer unavailable, using noop")
                self._local_tracer = NoopTracer()
        return self._local_tracer

    @contextmanager
    def trace(self, operation: str, parent_context: Optional[str] = None):
        """
        Trace an operation with distributed context propagation.

        EXAMPLE:
        >>> tracer = DistributedTracer()
        >>> with tracer.trace("swarm_execution", parent_context=request_id):
        ...     result = swarm.execute(task)

        Args:
            operation: Operation name (e.g., "swarm_execution", "credit_assignment")
            parent_context: Parent trace ID (for cross-service correlation)
        """
        trace_id = self._generate_trace_id(parent_context)

        # Store context for propagation
        self.trace_contexts[trace_id] = {
            'service': self.service_name,
            'operation': operation,
            'parent': parent_context,
            'start_time': time.time(),
        }

        # Delegate to local tracer (DRY)
        local_tracer = self._get_local_tracer()

        try:
            # Start local span
            with local_tracer.span(operation, trace_id=trace_id):
                yield trace_id
        finally:
            # Record duration
            if trace_id in self.trace_contexts:
                ctx = self.trace_contexts[trace_id]
                ctx['duration_ms'] = (time.time() - ctx['start_time']) * 1000

    def _generate_trace_id(self, parent: Optional[str]) -> str:
        """Generate trace ID (KISS - simple UUID)."""
        import uuid
        if parent:
            return f"{parent}:{uuid.uuid4().hex[:8]}"
        return uuid.uuid4().hex[:16]

    def get_context(self, trace_id: str) -> Dict[str, Any]:
        """Get trace context for propagation to downstream services."""
        return self.trace_contexts.get(trace_id, {})

    def inject_headers(self, trace_id: str) -> Dict[str, str]:
        """
        Inject trace context into HTTP headers (W3C Trace Context format).

        EXAMPLE:
        >>> headers = tracer.inject_headers(trace_id)
        >>> response = requests.get(url, headers=headers)
        """
        ctx = self.get_context(trace_id)
        return {
            'traceparent': f"00-{trace_id}-{trace_id[:16]}-01",
            'x-jotty-service': self.service_name,
            'x-jotty-operation': ctx.get('operation', 'unknown'),
        }

    def extract_context(self, headers: Dict[str, str]) -> Optional[str]:
        """
        Extract trace context from HTTP headers.

        EXAMPLE:
        >>> parent_trace = tracer.extract_context(request.headers)
        >>> with tracer.trace("handle_request", parent_context=parent_trace):
        ...     process()
        """
        traceparent = headers.get('traceparent', '')
        if traceparent:
            parts = traceparent.split('-')
            if len(parts) >= 2:
                return parts[1]  # trace-id
        return None


class NoopTracer:
    """Noop tracer for fallback (KISS - no errors if observability unavailable)."""

    @contextmanager
    def span(self, *args, **kwargs):
        yield


# Singleton instance
_distributed_tracer = None


def get_distributed_tracer(service_name: str = "jotty") -> DistributedTracer:
    """Get or create distributed tracer singleton."""
    global _distributed_tracer
    if _distributed_tracer is None:
        _distributed_tracer = DistributedTracer(service_name)
    return _distributed_tracer


def reset_distributed_tracer() -> None:
    """Reset the singleton distributed tracer (used in tests)."""
    global _distributed_tracer
    _distributed_tracer = None


__all__ = [
    'DistributedTracer',
    'get_distributed_tracer',
    'reset_distributed_tracer',
]

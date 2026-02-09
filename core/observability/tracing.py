"""
Tracing Module - OpenTelemetry-style Span Trees
================================================

Lightweight tracing for multi-agent execution debugging.
Captures hierarchical span trees with timing, attributes, and status.

No external dependencies. Compatible with OpenTelemetry export if needed later.

Usage:
    tracer = get_tracer()

    with tracer.span("swarm_run", goal="Research AI") as root:
        with tracer.span("planning") as plan_span:
            plan_span.set_attribute("task_type", "research")
            ...
        with tracer.span("agent_execute", agent="auto") as agent_span:
            result = await agent.run(goal)
            agent_span.set_attribute("success", result.success)
            agent_span.set_attribute("tokens", 1500)

    # Export full trace
    trace = tracer.get_current_trace()
    print(trace.to_dict())    # JSON-serializable
    print(trace.summary())    # Human-readable summary
"""

import time
import uuid
import threading
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SpanStatus(Enum):
    """Status of a span."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class Span:
    """
    A single unit of work in a trace.

    Spans form a tree: each span has an optional parent and zero or more children.
    """
    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: Optional[str] = None
    trace_id: Optional[str] = None

    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    # Metadata
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: SpanStatus = SpanStatus.UNSET
    status_message: str = ""

    # Children (populated by TracingContext)
    children: List['Span'] = field(default_factory=list)

    # Cost tracking (embedded for convenience)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return self.duration_ms / 1000

    def set_attribute(self, key: str, value: Any):
        """Set a span attribute."""
        self.attributes[key] = value

    def set_status(self, status: SpanStatus, message: str = ""):
        """Set span status."""
        self.status = status
        self.status_message = message

    def add_cost(self, input_tokens: int = 0, output_tokens: int = 0,
                 cost_usd: float = 0.0):
        """Add LLM cost to this span."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += cost_usd
        self.llm_calls += 1

    def end(self):
        """End this span."""
        if self.end_time is None:
            self.end_time = time.time()
        if self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK

    @property
    def total_tokens(self) -> int:
        """Total tokens including children."""
        total = self.input_tokens + self.output_tokens
        for child in self.children:
            total += child.total_tokens
        return total

    @property
    def total_cost(self) -> float:
        """Total cost including children."""
        total = self.cost_usd
        for child in self.children:
            total += child.total_cost
        return total

    @property
    def total_llm_calls(self) -> int:
        """Total LLM calls including children."""
        total = self.llm_calls
        for child in self.children:
            total += child.total_llm_calls
        return total

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            'name': self.name,
            'span_id': self.span_id,
            'parent_id': self.parent_id,
            'trace_id': self.trace_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ms': round(self.duration_ms, 2),
            'status': self.status.value,
            'attributes': self.attributes,
        }
        if self.input_tokens or self.output_tokens or self.cost_usd:
            result['cost'] = {
                'input_tokens': self.input_tokens,
                'output_tokens': self.output_tokens,
                'total_tokens': self.total_tokens,
                'cost_usd': round(self.total_cost, 6),
                'llm_calls': self.total_llm_calls,
            }
        if self.status_message:
            result['status_message'] = self.status_message
        if self.children:
            result['children'] = [c.to_dict() for c in self.children]
        return result


@dataclass
class Trace:
    """
    A complete execution trace — a tree of spans.
    """
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:32])
    root_spans: List[Span] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000

    def _collect_all_spans(self) -> List[Span]:
        """Collect all spans in the trace (DFS)."""
        all_spans = []
        stack = list(self.root_spans)
        while stack:
            span = stack.pop()
            all_spans.append(span)
            stack.extend(span.children)
        return all_spans

    @property
    def total_cost(self) -> float:
        return sum(s.total_cost for s in self.root_spans)

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.root_spans)

    @property
    def total_llm_calls(self) -> int:
        return sum(s.total_llm_calls for s in self.root_spans)

    @property
    def span_count(self) -> int:
        return len(self._collect_all_spans())

    def summary(self) -> str:
        """Human-readable trace summary."""
        lines = [
            f"Trace {self.trace_id[:8]}",
            f"  Duration: {self.duration_ms:.0f}ms",
            f"  Spans: {self.span_count}",
            f"  LLM Calls: {self.total_llm_calls}",
            f"  Tokens: {self.total_tokens:,}",
            f"  Cost: ${self.total_cost:.4f}",
        ]

        def _format_span(span: Span, indent: int = 1):
            prefix = "  " * indent
            status_icon = {
                SpanStatus.OK: "+",
                SpanStatus.ERROR: "!",
                SpanStatus.UNSET: "?",
            }[span.status]
            line = f"{prefix}[{status_icon}] {span.name} ({span.duration_ms:.0f}ms)"
            if span.llm_calls:
                line += f" [{span.llm_calls} LLM, {span.input_tokens + span.output_tokens} tok, ${span.cost_usd:.4f}]"
            attrs = {k: v for k, v in span.attributes.items()
                     if k not in ('goal',)}  # Skip verbose attrs
            if attrs:
                line += f" {attrs}"
            lines.append(line)
            for child in span.children:
                _format_span(child, indent + 1)

        for root in self.root_spans:
            _format_span(root)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trace_id': self.trace_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ms': round(self.duration_ms, 2),
            'metadata': self.metadata,
            'summary': {
                'span_count': self.span_count,
                'total_llm_calls': self.total_llm_calls,
                'total_tokens': self.total_tokens,
                'total_cost_usd': round(self.total_cost, 6),
            },
            'spans': [s.to_dict() for s in self.root_spans],
        }


class TracingContext:
    """
    Manages a trace tree with span creation and nesting.

    Thread-safe. Each thread maintains its own span stack.

    Usage:
        tracer = TracingContext()
        with tracer.span("operation", key="value") as span:
            # Nested spans automatically become children
            with tracer.span("sub_operation") as child:
                child.add_cost(input_tokens=100, output_tokens=50, cost_usd=0.001)

        trace = tracer.get_current_trace()
    """

    def __init__(self):
        self._trace: Optional[Trace] = None
        self._span_stack: threading.local = threading.local()
        self._lock = threading.Lock()
        self._all_traces: List[Trace] = []

    def new_trace(self, metadata: Optional[Dict[str, Any]] = None) -> Trace:
        """Start a new trace. Ends the previous one if active."""
        with self._lock:
            if self._trace and self._trace.end_time is None:
                self._trace.end_time = time.time()
                self._all_traces.append(self._trace)
            self._trace = Trace(metadata=metadata or {})
            return self._trace

    def get_current_trace(self) -> Optional[Trace]:
        """Get the current active trace."""
        return self._trace

    def _get_stack(self) -> List[Span]:
        if not hasattr(self._span_stack, 'stack'):
            self._span_stack.stack = []
        return self._span_stack.stack

    @contextmanager
    def span(self, name: str, **attributes):
        """
        Create a span as a context manager.

        Automatically nests under the current active span.
        Automatically ends the span on exit.

        Args:
            name: Span name (e.g., "agent_execute", "llm_call")
            **attributes: Key-value attributes to attach

        Yields:
            Span instance (can add attributes, cost, etc.)
        """
        # Ensure we have an active trace
        if self._trace is None:
            self.new_trace()

        stack = self._get_stack()
        parent = stack[-1] if stack else None

        new_span = Span(
            name=name,
            parent_id=parent.span_id if parent else None,
            trace_id=self._trace.trace_id,
            attributes=dict(attributes),
        )

        # Add to parent or trace root
        if parent:
            parent.children.append(new_span)
        else:
            self._trace.root_spans.append(new_span)

        stack.append(new_span)

        try:
            yield new_span
        except Exception as e:
            new_span.set_status(SpanStatus.ERROR, str(e)[:200])
            raise
        finally:
            new_span.end()
            stack.pop()

    def get_active_span(self) -> Optional[Span]:
        """Get the currently active span (innermost)."""
        stack = self._get_stack()
        return stack[-1] if stack else None

    def add_cost_to_current(self, input_tokens: int = 0, output_tokens: int = 0,
                            cost_usd: float = 0.0):
        """Add LLM cost to the currently active span."""
        span = self.get_active_span()
        if span:
            span.add_cost(input_tokens, output_tokens, cost_usd)

    def get_trace_history(self) -> List[Trace]:
        """Get all completed traces."""
        return list(self._all_traces)

    def end_trace(self):
        """End the current trace."""
        if self._trace:
            self._trace.end_time = time.time()
            self._all_traces.append(self._trace)
            self._trace = None

    def reset(self):
        """Reset all traces and state."""
        with self._lock:
            self._trace = None
            self._all_traces.clear()
            self._span_stack = threading.local()


# =========================================================================
# SINGLETON
# =========================================================================

_global_tracer: Optional[TracingContext] = None
_tracer_lock = threading.Lock()


def get_tracer() -> TracingContext:
    """Get the global tracer singleton."""
    global _global_tracer
    if _global_tracer is None:
        with _tracer_lock:
            if _global_tracer is None:
                _global_tracer = TracingContext()
    return _global_tracer


def reset_tracer():
    """Reset the global tracer (for testing)."""
    global _global_tracer
    with _tracer_lock:
        if _global_tracer:
            _global_tracer.reset()
        _global_tracer = None

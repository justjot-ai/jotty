"""
Distributed Tracing for Jotty

Provides observability for:
- Skill executions
- Agent operations
- Swarm workflows
- LLM calls
- Memory operations

Includes a lightweight Span/Trace/TracingContext layer that works without
OpenTelemetry, plus optional OpenTelemetry export via JottyTracer.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Optional, Dict, List, Any, Callable, Generator
import threading
import time
import uuid
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# SpanStatus enum
# =============================================================================

class SpanStatus(Enum):
    """Status of a tracing span."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


# =============================================================================
# Span dataclass
# =============================================================================

def _new_span_id() -> str:
    return uuid.uuid4().hex[:16]


def _new_trace_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Span:
    """A single span in a trace tree."""

    name: str
    span_id: str = field(default_factory=_new_span_id)
    trace_id: str = ""
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    status_message: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    children: List[Span] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    # --- computed properties --------------------------------------------------

    @property
    def duration_ms(self) -> float:
        end = self.end_time if self.end_time is not None else time.time()
        return (end - self.start_time) * 1000

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0

    @property
    def total_tokens(self) -> int:
        own = self.input_tokens + self.output_tokens
        return own + sum(c.total_tokens for c in self.children)

    @property
    def total_cost(self) -> float:
        return self.cost_usd + sum(c.total_cost for c in self.children)

    @property
    def total_llm_calls(self) -> int:
        return self.llm_calls + sum(c.total_llm_calls for c in self.children)

    # --- mutators -------------------------------------------------------------

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, status: SpanStatus, message: str = "") -> None:
        self.status = status
        self.status_message = message

    def add_cost(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += cost_usd
        self.llm_calls += 1

    def end(self) -> None:
        """End the span (idempotent)."""
        if self.end_time is not None:
            return
        self.end_time = time.time()
        if self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK

    # --- serialization --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "status": self.status.value,
            "start_time": self.start_time,
            "duration_ms": round(self.duration_ms, 3),
        }
        if self.end_time is not None:
            d["end_time"] = self.end_time
        if self.parent_id:
            d["parent_id"] = self.parent_id
        if self.status_message:
            d["status_message"] = self.status_message
        if self.attributes:
            d["attributes"] = dict(self.attributes)
        if self.input_tokens or self.output_tokens or self.cost_usd:
            d["cost"] = {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost_usd": self.cost_usd,
                "llm_calls": self.llm_calls,
            }
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


# =============================================================================
# Trace dataclass
# =============================================================================

@dataclass
class Trace:
    """A collection of root spans forming a logical trace."""

    trace_id: str = field(default_factory=_new_trace_id)
    root_spans: List[Span] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        end = self.end_time if self.end_time is not None else time.time()
        return (end - self.start_time) * 1000

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
        def _count(span: Span) -> int:
            return 1 + sum(_count(c) for c in span.children)
        return sum(_count(s) for s in self.root_spans)

    def summary(self) -> str:
        """Human-readable summary of the trace."""
        lines: List[str] = []

        def _tree(span: Span, indent: int = 0) -> None:
            prefix = "  " * indent
            icon = "[!] " if span.status == SpanStatus.ERROR else ""
            lines.append(f"{prefix}{icon}{span.name} ({round(span.duration_ms, 1)}ms)")
            for c in span.children:
                _tree(c, indent + 1)

        for s in self.root_spans:
            _tree(s)

        header = (
            f"Trace {self.trace_id[:8]}  |  "
            f"Spans: {self.span_count}  |  "
            f"LLM Calls: {self.total_llm_calls}  |  "
            f"Cost: ${self.total_cost:.4f}"
        )
        return header + "\n" + "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 3),
            "metadata": dict(self.metadata),
            "summary": {
                "span_count": self.span_count,
                "total_tokens": self.total_tokens,
                "total_cost": self.total_cost,
                "total_llm_calls": self.total_llm_calls,
            },
            "spans": [s.to_dict() for s in self.root_spans],
        }


# =============================================================================
# TracingContext
# =============================================================================

class TracingContext:
    """
    Lightweight tracing context manager.

    Manages the current trace, span stack, and trace history.
    Works without any external dependency (OpenTelemetry is optional).
    """

    def __init__(self) -> None:
        self._current_trace: Optional[Trace] = None
        self._span_stack: List[Span] = []
        self._trace_history: List[Trace] = []
        self._lock = threading.Lock()

    # --- trace lifecycle ------------------------------------------------------

    def new_trace(self, metadata: Optional[Dict[str, Any]] = None) -> Trace:
        """Start a new trace. Ends any current trace first."""
        if self._current_trace is not None:
            self.end_trace()
        trace = Trace(metadata=metadata or {})
        self._current_trace = trace
        self._span_stack = []
        return trace

    def get_current_trace(self) -> Optional[Trace]:
        return self._current_trace

    def end_trace(self) -> None:
        """End the current trace and add it to history."""
        if self._current_trace is not None:
            self._current_trace.end_time = time.time()
            self._trace_history.append(self._current_trace)
            self._current_trace = None
            self._span_stack = []

    def get_trace_history(self) -> List[Trace]:
        return list(self._trace_history)

    def reset(self) -> None:
        """Clear all state."""
        self._current_trace = None
        self._span_stack = []
        self._trace_history = []

    # --- span management ------------------------------------------------------

    def get_active_span(self) -> Optional[Span]:
        if self._span_stack:
            return self._span_stack[-1]
        return None

    def add_cost_to_current(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Add cost info to the currently active span (no-op if none active)."""
        active = self.get_active_span()
        if active is not None:
            active.add_cost(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Generator[Span, None, None]:
        """Context manager that creates, tracks, and auto-ends a Span."""
        # Auto-create a trace if one does not exist
        if self._current_trace is None:
            self.new_trace()
        assert self._current_trace is not None  # for type checker

        parent = self.get_active_span()
        s = Span(
            name=name,
            trace_id=self._current_trace.trace_id,
            parent_id=parent.span_id if parent else None,
        )
        for k, v in attributes.items():
            s.set_attribute(k, v)

        # Push onto stack
        self._span_stack.append(s)

        # Attach to parent or to trace root_spans
        if parent is not None:
            parent.children.append(s)
        else:
            self._current_trace.root_spans.append(s)

        try:
            yield s
        except Exception as exc:
            s.set_status(SpanStatus.ERROR, str(exc))
            raise
        finally:
            s.end()
            # Pop from stack
            if self._span_stack and self._span_stack[-1] is s:
                self._span_stack.pop()


# =============================================================================
# Legacy OpenTelemetry integration (kept for backward compatibility)
# =============================================================================

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


class SpanWrapper:
    """Wrapper for OpenTelemetry spans to add custom methods."""
    def __init__(self, span: Any) -> None:
        self._span = span

    def __enter__(self) -> SpanWrapper:
        self._span.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._span.__exit__(*args)

    def set_attribute(self, key: str, value: Any) -> None:
        if hasattr(self._span, 'set_attribute'):
            self._span.set_attribute(key, value)

    def set_status(self, status: str, description: str = "") -> None:
        if hasattr(self._span, 'set_status'):
            self._span.set_status(status)

    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None:
        if hasattr(self._span, 'add_event'):
            self._span.add_event(name, attributes=attributes)

    def add_cost(self, input_tokens: int, output_tokens: int, cost: float) -> None:
        """Add cost tracking as span attributes."""
        self.set_attribute("llm.input_tokens", input_tokens)
        self.set_attribute("llm.output_tokens", output_tokens)
        self.set_attribute("llm.cost_usd", cost)


class NoOpSpan:
    """No-op span when OpenTelemetry not available."""
    def __enter__(self) -> Any: return self
    def __exit__(self, *args: Any) -> None: pass
    def set_attribute(self, key: str, value: Any) -> None: pass
    def set_status(self, status: str, description: str = "") -> None: pass
    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None: pass
    def add_cost(self, input_tokens: int, output_tokens: int, cost: float) -> None: pass


class NoOpTracer:
    """No-op tracer when OpenTelemetry not available."""
    def start_span(self, name: str, **kwargs: Any) -> NoOpSpan: return NoOpSpan()
    def start_as_current_span(self, name: str, **kwargs: Any) -> NoOpSpan: return NoOpSpan()


class JottyTracer:
    """Jotty distributed tracing wrapper (OpenTelemetry backend)."""

    def __init__(self, enabled: bool = True, console_export: bool = False) -> None:
        self.enabled = enabled and OTEL_AVAILABLE

        if self.enabled:
            resource = Resource.create({"service.name": "jotty", "service.version": "1.0.0"})
            provider = TracerProvider(resource=resource)

            if console_export:
                console_exporter = ConsoleSpanExporter()
                provider.add_span_processor(BatchSpanProcessor(console_exporter))

            otel_trace.set_tracer_provider(provider)
            self.tracer = otel_trace.get_tracer("jotty")
            logger.info("OpenTelemetry tracing enabled")
        else:
            self.tracer = NoOpTracer()

    def trace(self, span_name: Optional[str] = None, **attributes: Any) -> Any:
        """Decorator for tracing functions."""
        def decorator(func: Callable) -> Callable:
            name = span_name or f"{func.__module__}.{func.__name__}"

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.tracer.start_as_current_span(name) as span:
                    for key, value in attributes.items():
                        if hasattr(span, 'set_attribute'):
                            span.set_attribute(key, value)
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        if hasattr(span, 'set_attribute'):
                            span.set_attribute("success", True)
                        return result
                    except Exception as e:
                        if hasattr(span, 'set_attribute'):
                            span.set_attribute("success", False)
                            span.set_attribute("error.type", type(e).__name__)
                            span.set_attribute("error.message", str(e))
                        raise
                    finally:
                        duration = time.time() - start_time
                        if hasattr(span, 'set_attribute'):
                            span.set_attribute("duration_ms", duration * 1000)

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.tracer.start_as_current_span(name) as span:
                    for key, value in attributes.items():
                        if hasattr(span, 'set_attribute'):
                            span.set_attribute(key, value)
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        if hasattr(span, 'set_attribute'):
                            span.set_attribute("success", True)
                        return result
                    except Exception as e:
                        if hasattr(span, 'set_attribute'):
                            span.set_attribute("success", False)
                            span.set_attribute("error.type", type(e).__name__)
                        raise
                    finally:
                        duration = time.time() - start_time
                        if hasattr(span, 'set_attribute'):
                            span.set_attribute("duration_ms", duration * 1000)

            import asyncio
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def span(self, name: str, **attributes: Any) -> Any:
        """Create a span context manager (OpenTelemetry backend)."""
        raw_span = self.tracer.start_as_current_span(name)
        return SpanWrapper(raw_span)


# =============================================================================
# Singleton (returns TracingContext)
# =============================================================================

_tracer: Optional[TracingContext] = None
_tracer_lock = threading.Lock()


def get_tracer(**kwargs: Any) -> TracingContext:
    """Get singleton TracingContext instance."""
    global _tracer
    if _tracer is None:
        with _tracer_lock:
            if _tracer is None:
                _tracer = TracingContext()
    return _tracer


def reset_tracer() -> None:
    """Reset the singleton tracer instance (used in tests)."""
    global _tracer
    with _tracer_lock:
        _tracer = None


# =============================================================================
# Convenience decorators (use JottyTracer internally)
# =============================================================================

_jotty_tracer: Optional[JottyTracer] = None


def _get_jotty_tracer() -> JottyTracer:
    global _jotty_tracer
    if _jotty_tracer is None:
        _jotty_tracer = JottyTracer()
    return _jotty_tracer


def trace_skill(skill_name: str) -> Any:
    """Decorator for tracing skill execution."""
    return _get_jotty_tracer().trace(f"skill.{skill_name}", skill_name=skill_name, component="skill")


def trace_agent(agent_name: str) -> Any:
    """Decorator for tracing agent execution."""
    return _get_jotty_tracer().trace(f"agent.{agent_name}", agent_name=agent_name, component="agent")


def trace_swarm(swarm_name: str) -> Any:
    """Decorator for tracing swarm execution."""
    return _get_jotty_tracer().trace(f"swarm.{swarm_name}", swarm_name=swarm_name, component="swarm")

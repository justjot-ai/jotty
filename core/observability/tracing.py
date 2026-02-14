"""
Distributed Tracing with OpenTelemetry

Provides observability for:
- Skill executions
- Agent operations  
- Swarm workflows
- LLM calls
- Memory operations
"""
from typing import Optional, Dict, Any, Callable
from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry, fall back to no-op if not installed
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.info("OpenTelemetry not installed. Install with: pip install opentelemetry-api opentelemetry-sdk")


class NoOpSpan:
    """No-op span when OpenTelemetry not available."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def set_attribute(self, key: str, value: Any): pass
    def add_event(self, name: str, attributes: Optional[Dict] = None): pass


class NoOpTracer:
    """No-op tracer when OpenTelemetry not available."""
    def start_span(self, name: str, **kwargs): return NoOpSpan()
    def start_as_current_span(self, name: str, **kwargs): return NoOpSpan()


class JottyTracer:
    """Jotty distributed tracing wrapper."""

    def __init__(self, enabled: bool = True, console_export: bool = False):
        self.enabled = enabled and OTEL_AVAILABLE

        if self.enabled:
            resource = Resource.create({"service.name": "jotty", "service.version": "1.0.0"})
            provider = TracerProvider(resource=resource)

            if console_export:
                console_exporter = ConsoleSpanExporter()
                provider.add_span_processor(BatchSpanProcessor(console_exporter))

            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer("jotty")
            logger.info("✅ OpenTelemetry tracing enabled")
        else:
            self.tracer = NoOpTracer()

    def trace(self, span_name: Optional[str] = None, **attributes):
        """Decorator for tracing functions."""
        def decorator(func: Callable) -> Callable:
            name = span_name or f"{func.__module__}.{func.__name__}"

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
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
            async def async_wrapper(*args, **kwargs):
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


_tracer: Optional[JottyTracer] = None


def get_tracer(enabled: bool = True, console_export: bool = False) -> JottyTracer:
    """Get singleton tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = JottyTracer(enabled=enabled, console_export=console_export)
    return _tracer


def trace_skill(skill_name: str):
    """Decorator for tracing skill execution."""
    return get_tracer().trace(f"skill.{skill_name}", skill_name=skill_name, component="skill")


def trace_agent(agent_name: str):
    """Decorator for tracing agent execution."""
    return get_tracer().trace(f"agent.{agent_name}", agent_name=agent_name, component="agent")


def trace_swarm(swarm_name: str):
    """Decorator for tracing swarm execution."""
    return get_tracer().trace(f"swarm.{swarm_name}", swarm_name=swarm_name, component="swarm")

"""
Tracing Coverage Tests
======================

Tests for core/observability/tracing.py:
- Span, Trace, TracingContext, SpanStatus
- get_tracer / reset_tracer singletons
- Nested span trees, cost aggregation, serialization
"""

import threading
import time

import pytest

from Jotty.core.infrastructure.monitoring.observability.tracing import (
    Span,
    SpanStatus,
    Trace,
    TracingContext,
    get_tracer,
    reset_tracer,
)

# =============================================================================
# Span Tests
# =============================================================================


@pytest.mark.unit
class TestSpan:
    """Tests for Span dataclass."""

    def test_span_defaults(self):
        span = Span(name="test")
        assert span.name == "test"
        assert span.status == SpanStatus.UNSET
        assert span.end_time is None
        assert span.parent_id is None
        assert span.input_tokens == 0
        assert span.output_tokens == 0
        assert span.cost_usd == 0.0
        assert span.llm_calls == 0
        assert span.children == []
        assert span.attributes == {}

    def test_span_ids_unique(self):
        s1 = Span(name="a")
        s2 = Span(name="b")
        assert s1.span_id != s2.span_id

    def test_span_duration_while_open(self):
        span = Span(name="test", start_time=time.time() - 1.0)
        assert span.duration_ms >= 900  # At least ~900ms
        assert span.duration_s >= 0.9

    def test_span_duration_after_end(self):
        start = time.time()
        span = Span(name="test", start_time=start, end_time=start + 0.5)
        assert abs(span.duration_ms - 500) < 1
        assert abs(span.duration_s - 0.5) < 0.001

    def test_set_attribute(self):
        span = Span(name="test")
        span.set_attribute("key", "value")
        span.set_attribute("count", 42)
        assert span.attributes == {"key": "value", "count": 42}

    def test_set_status(self):
        span = Span(name="test")
        span.set_status(SpanStatus.OK, "all good")
        assert span.status == SpanStatus.OK
        assert span.status_message == "all good"

    def test_set_status_error(self):
        span = Span(name="test")
        span.set_status(SpanStatus.ERROR, "something broke")
        assert span.status == SpanStatus.ERROR
        assert span.status_message == "something broke"

    def test_add_cost(self):
        span = Span(name="test")
        span.add_cost(input_tokens=100, output_tokens=50, cost_usd=0.001)
        assert span.input_tokens == 100
        assert span.output_tokens == 50
        assert span.cost_usd == 0.001
        assert span.llm_calls == 1

    def test_add_cost_accumulates(self):
        span = Span(name="test")
        span.add_cost(input_tokens=100, output_tokens=50, cost_usd=0.001)
        span.add_cost(input_tokens=200, output_tokens=100, cost_usd=0.002)
        assert span.input_tokens == 300
        assert span.output_tokens == 150
        assert span.cost_usd == pytest.approx(0.003)
        assert span.llm_calls == 2

    def test_end_span(self):
        span = Span(name="test")
        assert span.end_time is None
        span.end()
        assert span.end_time is not None
        assert span.status == SpanStatus.OK  # Auto-set on end

    def test_end_span_preserves_error_status(self):
        span = Span(name="test")
        span.set_status(SpanStatus.ERROR, "fail")
        span.end()
        assert span.status == SpanStatus.ERROR  # Not overwritten

    def test_end_idempotent(self):
        span = Span(name="test")
        span.end()
        end1 = span.end_time
        span.end()
        assert span.end_time == end1  # Not re-stamped

    def test_total_tokens_with_children(self):
        parent = Span(name="parent")
        parent.add_cost(input_tokens=10, output_tokens=5)
        child = Span(name="child")
        child.add_cost(input_tokens=20, output_tokens=10)
        parent.children.append(child)
        assert parent.total_tokens == 45  # 15 + 30

    def test_total_cost_with_children(self):
        parent = Span(name="parent")
        parent.add_cost(cost_usd=0.01)
        child = Span(name="child")
        child.add_cost(cost_usd=0.02)
        parent.children.append(child)
        assert parent.total_cost == pytest.approx(0.03)

    def test_total_llm_calls_with_children(self):
        parent = Span(name="parent")
        parent.add_cost(input_tokens=1)
        child = Span(name="child")
        child.add_cost(input_tokens=1)
        child.add_cost(input_tokens=1)
        parent.children.append(child)
        assert parent.total_llm_calls == 3

    def test_to_dict_basic(self):
        span = Span(name="test", span_id="abc123", trace_id="trace1")
        span.end()
        d = span.to_dict()
        assert d["name"] == "test"
        assert d["span_id"] == "abc123"
        assert d["trace_id"] == "trace1"
        assert d["status"] == "ok"
        assert "duration_ms" in d
        assert "children" not in d  # No children

    def test_to_dict_with_cost(self):
        span = Span(name="test")
        span.add_cost(input_tokens=100, output_tokens=50, cost_usd=0.001)
        span.end()
        d = span.to_dict()
        assert "cost" in d
        assert d["cost"]["input_tokens"] == 100
        assert d["cost"]["output_tokens"] == 50

    def test_to_dict_with_children(self):
        parent = Span(name="parent")
        child = Span(name="child")
        child.end()
        parent.children.append(child)
        parent.end()
        d = parent.to_dict()
        assert "children" in d
        assert len(d["children"]) == 1
        assert d["children"][0]["name"] == "child"

    def test_to_dict_with_status_message(self):
        span = Span(name="test")
        span.set_status(SpanStatus.ERROR, "timeout")
        span.end()
        d = span.to_dict()
        assert d["status_message"] == "timeout"


# =============================================================================
# SpanStatus Tests
# =============================================================================


@pytest.mark.unit
class TestSpanStatus:
    """Tests for SpanStatus enum."""

    def test_values(self):
        assert SpanStatus.UNSET.value == "unset"
        assert SpanStatus.OK.value == "ok"
        assert SpanStatus.ERROR.value == "error"

    def test_members(self):
        assert len(SpanStatus) == 3


# =============================================================================
# Trace Tests
# =============================================================================


@pytest.mark.unit
class TestTrace:
    """Tests for Trace dataclass."""

    def test_trace_defaults(self):
        trace = Trace()
        assert trace.trace_id is not None
        assert len(trace.trace_id) == 32
        assert trace.root_spans == []
        assert trace.end_time is None
        assert trace.metadata == {}

    def test_trace_ids_unique(self):
        t1 = Trace()
        t2 = Trace()
        assert t1.trace_id != t2.trace_id

    def test_duration_while_open(self):
        trace = Trace(start_time=time.time() - 1.0)
        assert trace.duration_ms >= 900

    def test_duration_after_end(self):
        start = time.time()
        trace = Trace(start_time=start, end_time=start + 2.0)
        assert abs(trace.duration_ms - 2000) < 1

    def test_total_cost(self):
        trace = Trace()
        s1 = Span(name="a")
        s1.add_cost(cost_usd=0.01)
        s2 = Span(name="b")
        s2.add_cost(cost_usd=0.02)
        trace.root_spans = [s1, s2]
        assert trace.total_cost == pytest.approx(0.03)

    def test_total_tokens(self):
        trace = Trace()
        s1 = Span(name="a")
        s1.add_cost(input_tokens=100, output_tokens=50)
        trace.root_spans = [s1]
        assert trace.total_tokens == 150

    def test_total_llm_calls(self):
        trace = Trace()
        s1 = Span(name="a")
        s1.add_cost(input_tokens=1)
        s1.add_cost(input_tokens=1)
        trace.root_spans = [s1]
        assert trace.total_llm_calls == 2

    def test_span_count(self):
        trace = Trace()
        parent = Span(name="parent")
        child = Span(name="child")
        grandchild = Span(name="grandchild")
        child.children.append(grandchild)
        parent.children.append(child)
        trace.root_spans = [parent]
        assert trace.span_count == 3

    def test_span_count_empty(self):
        trace = Trace()
        assert trace.span_count == 0

    def test_summary(self):
        trace = Trace()
        s = Span(name="test_op")
        s.add_cost(input_tokens=100, output_tokens=50, cost_usd=0.001)
        s.end()
        trace.root_spans = [s]
        summary = trace.summary()
        assert "test_op" in summary
        assert "Spans: 1" in summary
        assert "LLM Calls: 1" in summary

    def test_summary_with_error_span(self):
        trace = Trace()
        s = Span(name="failed_op")
        s.set_status(SpanStatus.ERROR, "boom")
        s.end()
        trace.root_spans = [s]
        summary = trace.summary()
        assert "[!]" in summary  # Error icon

    def test_to_dict(self):
        trace = Trace(metadata={"goal": "test"})
        s = Span(name="op")
        s.end()
        trace.root_spans = [s]
        trace.end_time = time.time()
        d = trace.to_dict()
        assert d["trace_id"] == trace.trace_id
        assert d["metadata"] == {"goal": "test"}
        assert "summary" in d
        assert d["summary"]["span_count"] == 1
        assert len(d["spans"]) == 1


# =============================================================================
# TracingContext Tests
# =============================================================================


@pytest.mark.unit
class TestTracingContext:
    """Tests for TracingContext manager."""

    def test_new_trace(self):
        ctx = TracingContext()
        trace = ctx.new_trace(metadata={"goal": "test"})
        assert trace is not None
        assert trace.metadata == {"goal": "test"}

    def test_get_current_trace(self):
        ctx = TracingContext()
        assert ctx.get_current_trace() is None
        trace = ctx.new_trace()
        assert ctx.get_current_trace() is trace

    def test_span_creates_trace_if_none(self):
        ctx = TracingContext()
        assert ctx.get_current_trace() is None
        with ctx.span("auto") as s:
            assert ctx.get_current_trace() is not None

    def test_span_basic(self):
        ctx = TracingContext()
        ctx.new_trace()
        with ctx.span("my_op", key="val") as s:
            assert s.name == "my_op"
            assert s.attributes == {"key": "val"}
        assert s.end_time is not None
        assert s.status == SpanStatus.OK

    def test_nested_spans(self):
        ctx = TracingContext()
        ctx.new_trace()
        with ctx.span("parent") as parent:
            with ctx.span("child") as child:
                assert child.parent_id == parent.span_id
            assert len(parent.children) == 1
            assert parent.children[0] is child

    def test_deeply_nested_spans(self):
        ctx = TracingContext()
        ctx.new_trace()
        with ctx.span("L1") as l1:
            with ctx.span("L2") as l2:
                with ctx.span("L3") as l3:
                    assert l3.parent_id == l2.span_id
                assert l2.parent_id == l1.span_id
        assert len(l1.children) == 1
        assert len(l1.children[0].children) == 1

    def test_span_error_handling(self):
        ctx = TracingContext()
        ctx.new_trace()
        with pytest.raises(ValueError, match="test error"):
            with ctx.span("failing") as s:
                raise ValueError("test error")
        assert s.status == SpanStatus.ERROR
        assert "test error" in s.status_message
        assert s.end_time is not None  # Still ended

    def test_get_active_span(self):
        ctx = TracingContext()
        ctx.new_trace()
        assert ctx.get_active_span() is None
        with ctx.span("outer") as outer:
            assert ctx.get_active_span() is outer
            with ctx.span("inner") as inner:
                assert ctx.get_active_span() is inner
            assert ctx.get_active_span() is outer
        assert ctx.get_active_span() is None

    def test_add_cost_to_current(self):
        ctx = TracingContext()
        ctx.new_trace()
        with ctx.span("op") as s:
            ctx.add_cost_to_current(input_tokens=100, output_tokens=50, cost_usd=0.001)
        assert s.input_tokens == 100
        assert s.output_tokens == 50
        assert s.cost_usd == 0.001

    def test_add_cost_no_active_span(self):
        ctx = TracingContext()
        ctx.new_trace()
        # Should not raise when no active span
        ctx.add_cost_to_current(input_tokens=100)

    def test_multiple_root_spans(self):
        ctx = TracingContext()
        ctx.new_trace()
        with ctx.span("first"):
            pass
        with ctx.span("second"):
            pass
        trace = ctx.get_current_trace()
        assert len(trace.root_spans) == 2

    def test_end_trace(self):
        ctx = TracingContext()
        ctx.new_trace()
        with ctx.span("op"):
            pass
        ctx.end_trace()
        assert ctx.get_current_trace() is None

    def test_end_trace_adds_to_history(self):
        ctx = TracingContext()
        ctx.new_trace()
        ctx.end_trace()
        history = ctx.get_trace_history()
        assert len(history) == 1

    def test_new_trace_ends_previous(self):
        ctx = TracingContext()
        t1 = ctx.new_trace()
        t2 = ctx.new_trace()
        assert t1.end_time is not None
        assert ctx.get_current_trace() is t2
        assert len(ctx.get_trace_history()) == 1

    def test_reset(self):
        ctx = TracingContext()
        ctx.new_trace()
        with ctx.span("op"):
            pass
        ctx.end_trace()
        ctx.reset()
        assert ctx.get_current_trace() is None
        assert len(ctx.get_trace_history()) == 0

    def test_trace_id_propagated_to_spans(self):
        ctx = TracingContext()
        trace = ctx.new_trace()
        with ctx.span("op") as s:
            assert s.trace_id == trace.trace_id


# =============================================================================
# Singleton Tests
# =============================================================================


@pytest.mark.unit
class TestTracingSingleton:
    """Tests for get_tracer / reset_tracer."""

    def setup_method(self):
        reset_tracer()

    def teardown_method(self):
        reset_tracer()

    def test_get_tracer_returns_tracing_context(self):
        tracer = get_tracer()
        assert isinstance(tracer, TracingContext)

    def test_get_tracer_singleton(self):
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2

    def test_reset_tracer(self):
        t1 = get_tracer()
        reset_tracer()
        t2 = get_tracer()
        assert t1 is not t2

    def test_concurrent_get_tracer(self):
        results = []
        errors = []

        def _get():
            try:
                results.append(get_tracer())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_get) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(results) == 8
        assert all(r is results[0] for r in results)

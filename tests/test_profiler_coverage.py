"""
Profiler Coverage Tests
=======================

Tests for core/monitoring/profiler.py:
- PerformanceProfiler, ProfileSegment, ProfileReport
- profile() context manager, profile_function() decorator
"""

import time
import pytest

from Jotty.core.infrastructure.monitoring.monitoring.profiler import (
    PerformanceProfiler, ProfileSegment, ProfileReport, profile_function,
)


# =============================================================================
# ProfileSegment Tests
# =============================================================================

@pytest.mark.unit
class TestProfileSegment:
    """Tests for ProfileSegment dataclass."""

    def test_basic_creation(self):
        seg = ProfileSegment(
            name="test", duration=0.5,
            start_time=1000.0, end_time=1000.5,
        )
        assert seg.name == "test"
        assert seg.duration == 0.5
        assert seg.metadata == {}
        assert seg.children == []

    def test_with_metadata(self):
        seg = ProfileSegment(
            name="llm_call", duration=1.0,
            start_time=0.0, end_time=1.0,
            metadata={"model": "claude"},
        )
        assert seg.metadata == {"model": "claude"}

    def test_to_dict(self):
        seg = ProfileSegment(
            name="test", duration=0.5,
            start_time=1000.0, end_time=1000.5,
            metadata={"key": "val"},
        )
        d = seg.to_dict()
        assert d['name'] == "test"
        assert d['duration'] == 0.5
        assert d['metadata'] == {"key": "val"}
        assert d['children'] == []

    def test_to_dict_with_children(self):
        child = ProfileSegment(
            name="child", duration=0.1,
            start_time=0.0, end_time=0.1,
        )
        parent = ProfileSegment(
            name="parent", duration=0.5,
            start_time=0.0, end_time=0.5,
            children=[child],
        )
        d = parent.to_dict()
        assert len(d['children']) == 1
        assert d['children'][0]['name'] == "child"


# =============================================================================
# ProfileReport Tests
# =============================================================================

@pytest.mark.unit
class TestProfileReport:
    """Tests for ProfileReport dataclass."""

    def test_basic_creation(self):
        seg = ProfileSegment(name="test", duration=1.0, start_time=0, end_time=1)
        report = ProfileReport(
            total_duration=2.0,
            segments=[seg],
            slowest_segments=[seg],
        )
        assert report.total_duration == 2.0
        assert len(report.segments) == 1
        assert len(report.slowest_segments) == 1
        assert report.call_counts == {}
        assert report.memory_usage is None

    def test_to_dict(self):
        seg = ProfileSegment(name="test", duration=1.0, start_time=0, end_time=1)
        report = ProfileReport(
            total_duration=2.0,
            segments=[seg],
            slowest_segments=[seg],
            call_counts={"total_calls": 50},
            memory_usage=128.5,
        )
        d = report.to_dict()
        assert d['total_duration'] == 2.0
        assert len(d['segments']) == 1
        assert len(d['slowest_segments']) == 1
        assert d['call_counts'] == {"total_calls": 50}
        assert d['memory_usage'] == 128.5


# =============================================================================
# PerformanceProfiler Tests
# =============================================================================

@pytest.mark.unit
class TestPerformanceProfiler:
    """Tests for PerformanceProfiler."""

    def test_init_defaults(self):
        p = PerformanceProfiler()
        assert p.segments == []
        assert p.current_segment is None
        assert p.segment_stack == []
        assert p.enable_cprofile is False
        assert p.profiler is None

    def test_init_with_cprofile(self):
        p = PerformanceProfiler(enable_cprofile=True)
        assert p.enable_cprofile is True
        assert p.profiler is not None

    def test_profile_basic(self):
        p = PerformanceProfiler()
        with p.profile("sleep_op"):
            time.sleep(0.05)
        assert len(p.segments) == 1
        seg = p.segments[0]
        assert seg.name == "sleep_op"
        assert seg.duration >= 0.04
        assert seg.end_time > seg.start_time

    def test_profile_with_metadata(self):
        p = PerformanceProfiler()
        with p.profile("op", metadata={"model": "claude"}) as seg:
            pass
        assert seg.metadata == {"model": "claude"}

    def test_profile_nested(self):
        p = PerformanceProfiler()
        with p.profile("parent"):
            with p.profile("child"):
                time.sleep(0.01)
        assert len(p.segments) == 1  # Only root
        parent = p.segments[0]
        assert parent.name == "parent"
        assert len(parent.children) == 1
        assert parent.children[0].name == "child"

    def test_profile_deeply_nested(self):
        p = PerformanceProfiler()
        with p.profile("L1"):
            with p.profile("L2"):
                with p.profile("L3"):
                    pass
        assert len(p.segments) == 1
        assert len(p.segments[0].children) == 1
        assert len(p.segments[0].children[0].children) == 1
        assert p.segments[0].children[0].children[0].name == "L3"

    def test_profile_sequential(self):
        p = PerformanceProfiler()
        with p.profile("first"):
            pass
        with p.profile("second"):
            pass
        assert len(p.segments) == 2
        assert p.segments[0].name == "first"
        assert p.segments[1].name == "second"

    def test_profile_current_segment_restored(self):
        p = PerformanceProfiler()
        assert p.current_segment is None
        with p.profile("outer"):
            assert p.current_segment is not None
            with p.profile("inner"):
                pass
            assert p.current_segment.name == "outer"
        assert p.current_segment is None

    def test_profile_function_decorator(self):
        p = PerformanceProfiler()

        @p.profile_function("my_func")
        def my_func(x):
            return x * 2

        result = my_func(21)
        assert result == 42
        assert len(p.segments) == 1
        assert p.segments[0].name == "my_func"
        assert p.segments[0].metadata == {"function": "my_func"}

    def test_profile_function_auto_name(self):
        p = PerformanceProfiler()

        @p.profile_function()
        def another_func():
            return "hello"

        result = another_func()
        assert result == "hello"
        assert p.segments[0].name == "another_func"

    def test_get_report(self):
        p = PerformanceProfiler()
        with p.profile("slow"):
            time.sleep(0.05)
        with p.profile("fast"):
            pass
        report = p.get_report(top_n=5)
        assert isinstance(report, ProfileReport)
        assert report.total_duration > 0
        assert len(report.segments) == 2
        # Slowest should be the sleep
        assert report.slowest_segments[0].name == "slow"

    def test_get_report_includes_nested(self):
        p = PerformanceProfiler()
        with p.profile("parent"):
            with p.profile("slow_child"):
                time.sleep(0.05)
        report = p.get_report(top_n=10)
        # Flattened should include both parent and child
        names = [s.name for s in report.slowest_segments]
        assert "parent" in names
        assert "slow_child" in names

    def test_get_report_top_n_limit(self):
        p = PerformanceProfiler()
        for i in range(20):
            with p.profile(f"op_{i}"):
                pass
        report = p.get_report(top_n=5)
        assert len(report.slowest_segments) == 5

    def test_get_report_with_cprofile(self):
        p = PerformanceProfiler(enable_cprofile=True)
        with p.profile("op"):
            sum(range(100))
        report = p.get_report()
        assert "total_calls" in report.call_counts

    def test_print_report_no_error(self):
        """print_report should not raise."""
        p = PerformanceProfiler()
        with p.profile("op"):
            pass
        p.print_report()  # Should not raise

    def test_reset(self):
        p = PerformanceProfiler()
        with p.profile("op"):
            pass
        assert len(p.segments) == 1
        p.reset()
        assert len(p.segments) == 0
        assert p.current_segment is None
        assert p.segment_stack == []

    def test_reset_with_cprofile(self):
        p = PerformanceProfiler(enable_cprofile=True)
        old_profiler = p.profiler
        p.reset()
        assert p.profiler is not old_profiler  # New profiler instance

    def test_exception_in_profile(self):
        """Profile segment still records duration on exception."""
        p = PerformanceProfiler()
        with pytest.raises(RuntimeError):
            with p.profile("failing"):
                raise RuntimeError("boom")
        assert len(p.segments) == 1
        assert p.segments[0].duration > 0
        assert p.current_segment is None  # Stack restored


# =============================================================================
# Standalone profile_function Tests
# =============================================================================

@pytest.mark.unit
class TestStandaloneProfileFunction:
    """Tests for the standalone profile_function decorator."""

    def test_standalone_decorator(self):
        @profile_function("standalone")
        def my_func():
            return 42

        result = my_func()
        assert result == 42

    def test_standalone_decorator_no_name(self):
        @profile_function()
        def named_func():
            return "ok"

        result = named_func()
        assert result == "ok"

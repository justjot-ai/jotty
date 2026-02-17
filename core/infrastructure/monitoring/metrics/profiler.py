"""
Performance Profiler
====================

Profiles execution to identify bottlenecks and performance issues.
Includes both cProfile-based PerformanceProfiler and lightweight ExecutionTimer
(merged from utils/profiler.py).
"""

import asyncio
import cProfile
import functools
import io
import logging
import pstats
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProfileSegment:
    """A profiled segment of code."""

    name: str
    duration: float
    start_time: float
    end_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["ProfileSegment"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "duration": self.duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class ProfileReport:
    """Complete profiling report."""

    total_duration: float
    segments: List[ProfileSegment]
    slowest_segments: List[ProfileSegment]
    call_counts: Dict[str, int] = field(default_factory=dict)
    memory_usage: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_duration": self.total_duration,
            "segments": [s.to_dict() for s in self.segments],
            "slowest_segments": [s.to_dict() for s in self.slowest_segments],
            "call_counts": self.call_counts,
            "memory_usage": self.memory_usage,
        }


class PerformanceProfiler:
    """
    Performance profiler for identifying bottlenecks.

    Usage:
        profiler = PerformanceProfiler()

        with profiler.profile("my_function"):
            my_function()

        report = profiler.get_report()
        print(f"Slowest: {report.slowest_segments[0].name}")
    """

    def __init__(self, enable_cprofile: bool = False) -> None:
        """
        Initialize profiler.

        Args:
            enable_cprofile: Enable cProfile for detailed call analysis
        """
        self.segments: List[ProfileSegment] = []
        self.current_segment: Optional[ProfileSegment] = None
        self.segment_stack: List[ProfileSegment] = []
        self.enable_cprofile = enable_cprofile
        self.profiler = cProfile.Profile() if enable_cprofile else None
        self.start_time = time.time()

    @contextmanager  # type: ignore[arg-type]
    def profile(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:  # type: ignore[misc]
        """
        Profile a code segment.

        Args:
            name: Segment name
            metadata: Optional metadata

        Example:
            with profiler.profile("llm_call", {"model": "claude"}):
                result = llm.generate(prompt)
        """
        segment = ProfileSegment(
            name=name, duration=0.0, start_time=time.time(), end_time=0.0, metadata=metadata or {}
        )

        # Push to stack
        if self.current_segment:
            self.current_segment.children.append(segment)
            self.segment_stack.append(self.current_segment)
        else:
            self.segments.append(segment)

        self.current_segment = segment

        # Start profiling if enabled
        if self.profiler:
            self.profiler.enable()

        try:
            yield segment
        finally:
            # End profiling
            if self.profiler:
                self.profiler.disable()

            # Calculate duration
            segment.end_time = time.time()
            segment.duration = segment.end_time - segment.start_time

            # Pop from stack
            if self.segment_stack:
                self.current_segment = self.segment_stack.pop()
            else:
                self.current_segment = None

    def profile_function(self, name: Optional[str] = None) -> Any:
        """
        Decorator to profile a function.

        Example:
            @profiler.profile_function("my_function")
            def my_function():
                ...
        """

        def decorator(func: Callable) -> Any:
            func_name = name or func.__name__

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.profile(func_name, {"function": func.__name__}):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def get_report(self, top_n: int = 10) -> ProfileReport:
        """
        Get profiling report.

        Args:
            top_n: Number of slowest segments to include

        Returns:
            ProfileReport
        """
        total_duration = time.time() - self.start_time

        # Flatten all segments (including nested)
        all_segments = []

        def collect_segments(segments: List[ProfileSegment]) -> None:
            for segment in segments:
                all_segments.append(segment)
                if segment.children:
                    collect_segments(segment.children)

        collect_segments(self.segments)

        # Sort by duration
        slowest = sorted(all_segments, key=lambda s: s.duration, reverse=True)[:top_n]

        # Get call counts if cProfile enabled
        call_counts = {}
        if self.profiler:
            s = io.StringIO()
            stats = pstats.Stats(self.profiler, stream=s)
            stats.sort_stats("cumulative")
            # Extract call counts (simplified)
            call_counts = {"total_calls": stats.total_calls}  # type: ignore[attr-defined]

        return ProfileReport(
            total_duration=total_duration,
            segments=self.segments,
            slowest_segments=slowest,
            call_counts=call_counts,
        )

    def print_report(self, top_n: int = 10) -> None:
        """Print profiling report."""
        report = self.get_report(top_n)

        logger.info("=" * 60)
        logger.info("Performance Profiling Report")
        logger.info("=" * 60)
        logger.info("Total Duration: %.2fs", report.total_duration)
        logger.info("Top %d Slowest Segments:", top_n)

        for i, segment in enumerate(report.slowest_segments, 1):
            percentage = (
                (segment.duration / report.total_duration * 100) if report.total_duration > 0 else 0
            )
            logger.info("%d. %s", i, segment.name)
            logger.info("   Duration: %.2fs (%.1f%%)", segment.duration, percentage)
            if segment.metadata:
                logger.info("   Metadata: %s", segment.metadata)

        if self.profiler:
            logger.info("=" * 60)
            logger.info("cProfile Details")
            logger.info("=" * 60)
            s = io.StringIO()
            stats = pstats.Stats(self.profiler, stream=s)
            stats.sort_stats("cumulative")
            stats.print_stats(20)  # Top 20
            logger.info(s.getvalue())

    def reset(self) -> None:
        """Reset profiler."""
        self.segments = []
        self.current_segment = None
        self.segment_stack = []
        self.start_time = time.time()
        if self.profiler:
            self.profiler = cProfile.Profile()


def profile_function(name: Optional[str] = None) -> Any:
    """
    Standalone decorator for profiling functions.

    Usage:
        @profile_function("my_function")
        def my_function():
            ...
    """
    profiler = PerformanceProfiler()
    return profiler.profile_function(name)


# =============================================================================
# LIGHTWEIGHT TIMING (merged from utils/profiler.py)
# =============================================================================

# Import ProfilingReport for detailed profiling
try:
    from Jotty.core.infrastructure.utils.profiling_report import ProfilingReport

    PROFILING_REPORT_AVAILABLE = True
except ImportError:
    PROFILING_REPORT_AVAILABLE = False
    ProfilingReport = None  # type: ignore[assignment, misc]


class ExecutionTimer:
    """Track execution times for different operations."""

    def __init__(self) -> None:
        self.timings: Dict[str, List[float]] = {}
        self.enabled = True
        self.profiling_report: Optional[Any] = None

    def record(self, operation: str, duration: float) -> None:
        """Record a timing for an operation."""
        if not self.enabled:
            return
        if operation not in self.timings:
            self.timings[operation] = []
        self.timings[operation].append(duration)

    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self.timings or not self.timings[operation]:
            return {}
        times = self.timings[operation]
        return {
            "count": len(times),
            "total": sum(times),
            "avg": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {op: self.get_stats(op) for op in self.timings.keys()}

    def print_summary(self) -> None:
        """Print a formatted summary of all timings."""
        if not self.timings:
            return
        sorted_ops = sorted(self.timings.items(), key=lambda x: sum(x[1]), reverse=True)
        for operation, times in sorted_ops:
            stats = self.get_stats(operation)
            logger.info(
                f" {operation}: count={stats['count']}, "
                f"total={stats['total']:.2f}s, avg={stats['avg']:.2f}s"
            )

    def reset(self) -> None:
        """Clear all timing data."""
        self.timings.clear()
        if self.profiling_report and hasattr(self.profiling_report, "entries"):
            self.profiling_report.entries.clear()

    def set_profiling_report(self, report: Any) -> None:
        """Set the profiling report for detailed tracking."""
        self.profiling_report = report


# Global timer instance
_global_timer = ExecutionTimer()


def get_timer() -> ExecutionTimer:
    """Get the global execution timer."""
    return _global_timer


def set_output_dir(output_dir: str) -> None:
    """Initialize ProfilingReport with output directory."""
    if PROFILING_REPORT_AVAILABLE and ProfilingReport is not None:
        _global_timer.profiling_report = ProfilingReport(output_dir)


def set_overall_timing(start_time: float, end_time: float) -> None:
    """Set overall execution timing for the profiling report."""
    if _global_timer.profiling_report:
        _global_timer.profiling_report.set_overall_timing(start_time, end_time)


@contextmanager  # type: ignore[arg-type]
def timed_block(  # type: ignore[misc]
    operation: str, component: str = "Other", enabled: bool = True, **metadata: Any
) -> None:
    """Context manager for timing a block of code."""
    if not enabled:
        yield
        return
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        duration = end - start
        _global_timer.record(operation, duration)
        if _global_timer.profiling_report:
            _global_timer.profiling_report.record_timing(
                operation=operation, component=component, start_time=start, end_time=end, **metadata
            )


def timed(operation: Optional[str] = None, enabled: bool = True) -> Any:
    """Decorator for timing function execution."""

    def decorator(func: Any) -> Any:
        op_name = operation or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not enabled:
                return func(*args, **kwargs)
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                _global_timer.record(op_name, time.time() - start)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not enabled:
                return await func(*args, **kwargs)
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                _global_timer.record(op_name, time.time() - start)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def enable_profiling() -> None:
    """Enable global profiling."""
    _global_timer.enabled = True


def disable_profiling() -> None:
    """Disable global profiling."""
    _global_timer.enabled = False


def print_profile_summary() -> None:
    """Print the global profiling summary."""
    _global_timer.print_summary()


def save_profiling_reports() -> Any:
    """Save detailed profiling reports to files."""
    if _global_timer.profiling_report:
        return _global_timer.profiling_report.save_reports()
    return None


def reset_profiling() -> None:
    """Reset all profiling data."""
    _global_timer.reset()

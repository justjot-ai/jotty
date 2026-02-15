"""
Performance Profiler

Profiles execution to identify bottlenecks and performance issues.
Based on OAgents profiling approach.
"""

import cProfile
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

    @contextmanager
    def profile(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
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
            call_counts = {"total_calls": stats.total_calls}

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

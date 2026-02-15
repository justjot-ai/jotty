"""
Simple Profiling Utilities for Jotty
=====================================

Lightweight timing decorators and context managers for performance analysis.
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import ProfilingReport for detailed profiling
try:
    from .profiling_report import ProfilingReport

    PROFILING_REPORT_AVAILABLE = True
except ImportError:
    PROFILING_REPORT_AVAILABLE = False
    ProfilingReport = None


class ExecutionTimer:
    """Track execution times for different operations."""

    def __init__(self) -> None:
        self.timings: Dict[str, List[float]] = {}
        self.enabled = True
        self.profiling_report: Optional[ProfilingReport] = None

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
            logger.info("⏱ No timing data collected")
            return

        logger.info("=" * 70)
        logger.info("⏱ PERFORMANCE SUMMARY")
        logger.info("=" * 70)

        # Sort by total time descending
        sorted_ops = sorted(self.timings.items(), key=lambda x: sum(x[1]), reverse=True)

        for operation, times in sorted_ops:
            stats = self.get_stats(operation)
            logger.info(f"\n {operation}")
            logger.info(f"   Count: {stats['count']}")
            logger.info(f"   Total: {stats['total']:.2f}s")
            logger.info(f"   Avg:   {stats['avg']:.2f}s")
            logger.info(f"   Range: {stats['min']:.2f}s - {stats['max']:.2f}s")

        logger.info("=" * 70)

    def reset(self) -> None:
        """Clear all timing data."""
        self.timings.clear()
        if self.profiling_report:
            self.profiling_report.entries.clear()

    def set_profiling_report(self, report: "ProfilingReport") -> None:
        """Set the profiling report for detailed tracking."""
        self.profiling_report = report


# Global timer instance
_global_timer = ExecutionTimer()


def get_timer() -> ExecutionTimer:
    """Get the global execution timer."""
    return _global_timer


def set_output_dir(output_dir: str) -> None:
    """Initialize ProfilingReport with output directory."""
    if PROFILING_REPORT_AVAILABLE:
        _global_timer.profiling_report = ProfilingReport(output_dir)
        logger.debug(f"⏱ Profiling report initialized: {output_dir}")


def set_overall_timing(start_time: float, end_time: float) -> None:
    """Set overall execution timing for the profiling report."""
    if _global_timer.profiling_report:
        _global_timer.profiling_report.set_overall_timing(start_time, end_time)


@contextmanager
def timed_block(
    operation: str, component: str = "Other", enabled: bool = True, **metadata: Any
) -> None:
    """
    Context manager for timing a block of code.

    Args:
        operation: Name of the operation (e.g., "IssueDetector")
        component: Component category (e.g., "Agent", "ParameterResolution", "LLMCall")
        enabled: Whether timing is enabled
        **metadata: Additional metadata to record

    Usage:
        with timed_block("IssueDetector", component="Agent", name="Issue Detection"):
            result = agent.execute(...)
    """
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

        # Record in detailed profiling report if available
        if _global_timer.profiling_report:
            _global_timer.profiling_report.record_timing(
                operation=operation, component=component, start_time=start, end_time=end, **metadata
            )


def timed(operation: Optional[str] = None, enabled: bool = True) -> Any:
    """
    Decorator for timing function execution.

    Usage:
        @timed("process_data")
        def process(data):
            ...

        @timed()  # Uses function name
        def fetch_data():
            ...
    """

    def decorator(func: Any) -> Any:
        op_name = operation or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not enabled:
                return func(*args, **kwargs)

            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                _global_timer.record(op_name, duration)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not enabled:
                return await func(*args, **kwargs)

            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                _global_timer.record(op_name, duration)

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def enable_profiling() -> None:
    """Enable global profiling."""
    _global_timer.enabled = True
    logger.info("⏱ Profiling enabled")


def disable_profiling() -> None:
    """Disable global profiling."""
    _global_timer.enabled = False
    logger.info("⏱ Profiling disabled")


def print_profile_summary() -> None:
    """Print the global profiling summary."""
    _global_timer.print_summary()


def save_profiling_reports() -> None:
    """Save detailed profiling reports to files."""
    if _global_timer.profiling_report:
        files = _global_timer.profiling_report.save_reports()
        logger.info("⏱ Profiling reports saved:")
        for report_type, path in files.items():
            logger.info(f" {report_type}: {path}")
        return files
    else:
        logger.warning("⏱ No profiling report available to save")
        return None


def reset_profiling() -> None:
    """Reset all profiling data."""
    _global_timer.reset()

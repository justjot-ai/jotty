"""
Profiler — Re-export Shim
===========================

All profiling functionality now lives in ``monitoring.metrics.profiler``.
This module re-exports for backward compatibility.
"""

from Jotty.core.infrastructure.monitoring.metrics.profiler import (
    ExecutionTimer,
    _global_timer,
    disable_profiling,
    enable_profiling,
    get_timer,
    print_profile_summary,
    reset_profiling,
    save_profiling_reports,
    set_output_dir,
    set_overall_timing,
    timed,
    timed_block,
)

__all__ = [
    "ExecutionTimer",
    "_global_timer",
    "get_timer",
    "timed",
    "timed_block",
    "enable_profiling",
    "disable_profiling",
    "reset_profiling",
    "print_profile_summary",
    "save_profiling_reports",
    "set_output_dir",
    "set_overall_timing",
]

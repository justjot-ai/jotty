"""
Backward-compatibility shim — delegates to ``core.intelligence.swarms``.

All swarm implementations live in ``core.intelligence.swarms``.
This module re-exports them so that existing ``core.execution.swarms``
imports continue to work.
"""

from typing import Any

from Jotty.core.execution.base import PhaseExecutor
from Jotty.core.intelligence import swarms as _canonical
from Jotty.core.intelligence.swarms import (
    AgentSpec,
    CoordinationPattern,
    MergeStrategy,
    SwarmTemplate,
    TeamCoordinator,
    TeamResult,
)
from Jotty.core.intelligence.swarms import __all__ as _intel_all


def __getattr__(name: str) -> Any:
    """Delegate any lazy attribute to the canonical intelligence.swarms module."""
    return getattr(_canonical, name)


__all__ = [*_intel_all, "PhaseExecutor"]

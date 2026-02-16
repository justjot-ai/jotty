"""
Backward-compatibility shim.

Canonical location: ``core.execution.swarms``

All attributes are forwarded transparently so existing
``from Jotty.core.intelligence.swarms import X`` still works.
"""

import Jotty.core.execution.swarms as _canonical
from Jotty.core.execution.swarms import (
    AgentSpec,
    CoordinationPattern,
    MergeStrategy,
    SwarmTemplate,
    TeamCoordinator,
    TeamResult,
)

__all__ = _canonical.__all__


def __getattr__(name: str):
    return getattr(_canonical, name)

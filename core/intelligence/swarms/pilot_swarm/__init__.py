"""Pilot Swarm — Autonomous goal-completion engine.

The universal "do anything" swarm. Decomposes any goal into subtasks
and executes using the right combination of tools: web search, coding,
terminal, skill creation, or delegation to specialized swarms.

Quick start:
    from Jotty.core.intelligence.swarms.pilot_swarm import pilot
    result = await pilot("Create a FastAPI project with JWT auth")

    # Synchronous
    from Jotty.core.intelligence.swarms.pilot_swarm import pilot_sync
    result = pilot_sync("Find and compare the top 5 Python web frameworks")
"""

from .swarm import PilotSwarm, pilot, pilot_sync
from .types import AVAILABLE_SWARMS, PilotConfig, PilotResult, Subtask, SubtaskStatus, SubtaskType

__all__ = [
    # Swarm
    "PilotSwarm",
    "pilot",
    "pilot_sync",
    # Types
    "SubtaskType",
    "SubtaskStatus",
    "Subtask",
    "PilotConfig",
    "PilotResult",
    "AVAILABLE_SWARMS",
]

"""
DAG Agents - Task Breakdown and Actor Assignment for Jotty Swarm

This module provides a minimal SwarmResources stub since swarms work fine without it.
"""

from typing import Optional

from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

from .swarm_resources_stub import SwarmResources


def get_swarm_resources(config: Optional[SwarmConfig] = None) -> SwarmResources:
    """Get or create shared SwarmResources stub."""
    return SwarmResources(config or SwarmConfig())


def reset_swarm_resources() -> None:
    """Reset shared resources (for testing)."""
    SwarmResources._instance = None

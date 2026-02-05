"""
Swarm Registry
===============

Registry for discovering and creating swarm instances:
- SwarmRegistry: Central registry for all available swarms
- register_swarm: Decorator for registering swarm classes

Extracted from base_swarm.py for modularity.
"""

from __future__ import annotations

from typing import Dict, Optional, List, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base_swarm import BaseSwarm

from .swarm_types import SwarmConfig


class SwarmRegistry:
    """Registry for all available swarms."""

    _swarms: Dict[str, Type['BaseSwarm']] = {}

    @classmethod
    def register(cls, name: str, swarm_class: Type['BaseSwarm']):
        """Register a swarm class."""
        cls._swarms[name] = swarm_class

    @classmethod
    def get(cls, name: str) -> Optional[Type['BaseSwarm']]:
        """Get a swarm class by name."""
        return cls._swarms.get(name)

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered swarms."""
        return list(cls._swarms.keys())

    @classmethod
    def create(cls, name: str, config: SwarmConfig = None) -> Optional['BaseSwarm']:
        """Create a swarm instance by name."""
        swarm_class = cls.get(name)
        if not swarm_class:
            return None

        if config is None:
            config = SwarmConfig(name=name, domain=name)

        return swarm_class(config)


def register_swarm(name: str):
    """Decorator to register a swarm class."""
    def decorator(cls):
        SwarmRegistry.register(name, cls)
        return cls
    return decorator


__all__ = [
    'SwarmRegistry',
    'register_swarm',
]

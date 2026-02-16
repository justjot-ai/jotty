"""
Minimal SwarmResources Stub
============================

Provides a minimal SwarmResources implementation that doesn't require
all the complex dependencies, since swarms work fine without it anyway.
"""

from typing import Any, Optional

from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig


class SwarmResources:
    """
    Minimal stub for SwarmResources.

    Swarms check if these resources are available but work fine without them.
    This stub allows imports to succeed without requiring all dependencies.
    """

    _instance = None

    def __new__(cls, config: Optional[SwarmConfig] = None) -> Any:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False  # type: ignore[has-type]
        return cls._instance

    def __init__(self, config: Optional[SwarmConfig] = None) -> None:
        if self._initialized:  # type: ignore[has-type]
            return

        self.config = config or SwarmConfig()
        self.memory = None  # Swarms check if this exists before using
        self.context = None  # Swarms check if this exists before using
        self.bus = None  # Swarms check if this exists before using
        self.learner = None  # Swarms check if this exists before using
        self._initialized = True

    @classmethod
    def get_instance(cls, config: Optional[SwarmConfig] = None) -> "SwarmResources":
        """Get or create the singleton instance."""
        return cls(config)

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

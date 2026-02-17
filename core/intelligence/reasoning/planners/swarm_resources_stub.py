"""
SwarmResources
==============

Provides shared resources for all swarms: persistent SwarmMemory,
context, message bus, and TD-Lambda learner.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

logger = logging.getLogger(__name__)


class SwarmResources:
    """
    Shared resources for all swarms.

    Provides a real SwarmMemory with disk persistence so memories
    survive process restarts. Other resources (context, bus, learner)
    remain None until explicitly wired.
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

        # Create real SwarmMemory with persistence
        self.memory = None
        self._memory_persistence = None
        try:
            from Jotty.core.intelligence.memory.cortex import SwarmMemory
            from Jotty.core.intelligence.memory.memory_persistence import (
                enable_memory_persistence,
            )

            self.memory = SwarmMemory(config=self.config, agent_name="swarm_shared")
            persistence_dir = Path.home() / "jotty" / "memory" / "swarm_shared"
            self._memory_persistence = enable_memory_persistence(
                memory=self.memory,
                persistence_dir=persistence_dir,
            )
            logger.debug("SwarmResources: SwarmMemory with persistence initialized")
        except Exception as e:
            logger.warning(f"SwarmResources: Could not initialize memory: {e}")

        self.context = None
        self.bus = None
        self.learner = None
        self._initialized = True

    def save_memory(self) -> bool:
        """Persist memory to disk. Returns True if successful."""
        if self._memory_persistence:
            return self._memory_persistence.save()
        return False

    @classmethod
    def get_instance(cls, config: Optional[SwarmConfig] = None) -> "SwarmResources":
        """Get or create the singleton instance."""
        return cls(config)

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

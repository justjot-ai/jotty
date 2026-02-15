"""
Jotty Focused Configuration Dataclasses
========================================

Each config covers one subsystem's parameters. SwarmConfig composes all of them.

Usage:
    # Import a focused config for your subsystem
    from Jotty.core.infrastructure.foundation.configs import MemoryConfig, LearningConfig

    # Or import all
    from Jotty.core.infrastructure.foundation.configs import (
        PersistenceConfig, ExecutionConfig, MemoryConfig,
        ContextBudgetConfig, LearningConfig, ValidationConfig,
        MonitoringConfig, IntelligenceConfig,
    )

    # Subsystems accept their own config type
    def init_memory(config: MemoryConfig): ...
"""

from .persistence import PersistenceConfig
from .execution import ExecutionConfig
from .memory import MemoryConfig
from .context_budget import ContextBudgetConfig
from .learning import LearningConfig
from .validation import ValidationConfig
from .monitoring import MonitoringConfig
from .intelligence import IntelligenceConfig

__all__ = [
    'PersistenceConfig',
    'ExecutionConfig',
    'MemoryConfig',
    'ContextBudgetConfig',
    'LearningConfig',
    'ValidationConfig',
    'MonitoringConfig',
    'IntelligenceConfig',
]

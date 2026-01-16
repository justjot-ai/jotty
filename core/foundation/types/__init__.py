"""
Jotty v6.0 - Foundation Types Package
======================================

All core data structures organized by category.
This package provides a clean import interface while maintaining
backward compatibility with the original data_structures.py module.

Usage:
    # New organized imports
    from Jotty.core.foundation.types import MemoryLevel, OutputTag
    from Jotty.core.foundation.types import MemoryEntry, GoalValue
    from Jotty.core.foundation.types import ValidationResult
    from Jotty.core.foundation.types import EpisodeResult, LearningMetrics
    from Jotty.core.foundation.types import AgentContribution, AgentMessage

    # Or import all at once
    from Jotty.core.foundation.types import *
"""

# Re-export all types for convenient importing
from .enums import (
    MemoryLevel,
    OutputTag,
    AlertType,
    CommunicationType,
    ValidationRound,
    ContextType,
)

from .memory_types import (
    GoalNode,
    GoalHierarchy,
    CausalLink,
    GoalValue,
    MemoryEntry,
)

from .learning_types import (
    TaggedOutput,
    EpisodeResult,
    StoredEpisode,
    LearningMetrics,
)

from .agent_types import (
    AgentContribution,
    AgentMessage,
    SharedScratchpad,
)

from .validation_types import (
    ValidationResult,
)

from .workflow_types import (
    RichObservation,
)

__all__ = [
    # Enums
    'MemoryLevel',
    'OutputTag',
    'AlertType',
    'CommunicationType',
    'ValidationRound',
    'ContextType',
    # Memory types
    'GoalNode',
    'GoalHierarchy',
    'CausalLink',
    'GoalValue',
    'MemoryEntry',
    # Learning types
    'TaggedOutput',
    'EpisodeResult',
    'StoredEpisode',
    'LearningMetrics',
    # Agent types
    'AgentContribution',
    'AgentMessage',
    'SharedScratchpad',
    # Validation types
    'ValidationResult',
    # Workflow types
    'RichObservation',
]

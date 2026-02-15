"""
Jotty v6.0 - Foundation Types Package
======================================

All core data structures organized by category.
This package provides a clean import interface while maintaining
backward compatibility with the original data_structures.py module.

Usage:
    # New organized imports
    from Jotty.core.infrastructure.foundation.types import MemoryLevel, OutputTag
    from Jotty.core.infrastructure.foundation.types import MemoryEntry, GoalValue
    from Jotty.core.infrastructure.foundation.types import ValidationResult
    from Jotty.core.infrastructure.foundation.types import EpisodeResult, LearningMetrics
    from Jotty.core.infrastructure.foundation.types import AgentContribution, AgentMessage

    # Or import all at once
    from Jotty.core.infrastructure.foundation.types import *
"""

from .agent_types import AgentContribution, AgentMessage, SharedScratchpad

# Re-export all types for convenient importing
from .enums import (
    AlertType,
    CommunicationType,
    ContextType,
    ExecutorType,
    MemoryLevel,
    OutputTag,
    TaskStatus,
    ValidationRound,
)
from .execution_types import CoordinationPattern, MergeStrategy
from .learning_types import EpisodeResult, LearningMetrics, StoredEpisode, TaggedOutput
from .memory_types import CausalLink, GoalHierarchy, GoalNode, GoalValue, MemoryEntry
from .sdk_types import (
    ChannelType,
    ExecutionContext,
    ExecutionMode,
    ResponseFormat,
    SDKEvent,
    SDKEventType,
    SDKRequest,
    SDKResponse,
    SDKSession,
)
from .validation_types import ValidationResult
from .workflow_types import RichObservation

__all__ = [
    # Enums
    "MemoryLevel",
    "OutputTag",
    "AlertType",
    "CommunicationType",
    "ValidationRound",
    "ContextType",
    "TaskStatus",
    "ExecutorType",
    # Memory types
    "GoalNode",
    "GoalHierarchy",
    "CausalLink",
    "GoalValue",
    "MemoryEntry",
    # Learning types
    "TaggedOutput",
    "EpisodeResult",
    "StoredEpisode",
    "LearningMetrics",
    # Agent types
    "AgentContribution",
    "AgentMessage",
    "SharedScratchpad",
    # Validation types
    "ValidationResult",
    # Workflow types
    "RichObservation",
    # SDK types
    "ExecutionMode",
    "ChannelType",
    "SDKEventType",
    "ResponseFormat",
    "ExecutionContext",
    "SDKEvent",
    "SDKSession",
    "SDKResponse",
    "SDKRequest",
    # Execution types (shared between agents/ and swarms/)
    "CoordinationPattern",
    "MergeStrategy",
]

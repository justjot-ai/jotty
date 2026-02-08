"""
FOUNDATION Layer - Cross-cutting (Exceptions, Config, Types)
"""
# Exceptions (30+ types)
from Jotty.core.foundation.exceptions import (
    JottyError,
    ConfigurationError, InvalidConfigError, MissingConfigError,
    ExecutionError, AgentExecutionError, ToolExecutionError, TimeoutError, CircuitBreakerError,
    ContextError, ContextOverflowError, CompressionError, ChunkingError,
    MemoryError, MemoryRetrievalError, MemoryStorageError, ConsolidationError,
    LearningError, RewardCalculationError, CreditAssignmentError, PolicyUpdateError,
    CommunicationError, MessageDeliveryError, FeedbackRoutingError,
    ValidationError, InputValidationError, OutputValidationError,
    PersistenceError, StorageError, RetrievalError,
    IntegrationError, LLMError, DSPyError, ExternalToolError,
    wrap_exception,
)

# Config
from Jotty.core.foundation.data_structures import JottyConfig

try:
    from Jotty.core.foundation.agent_config import AgentConfig
except ImportError:
    AgentConfig = None

try:
    from Jotty.cli.config.schema import CLIConfig
except ImportError:
    CLIConfig = None

# Types
try:
    from Jotty.core.foundation.types import (
        MemoryLevel, OutputTag, MemoryEntry, GoalValue,
        ValidationResult, EpisodeResult, LearningMetrics,
        AgentContribution, AgentMessage,
    )
except ImportError:
    pass

__all__ = [
    # Exceptions
    "JottyError",
    "ConfigurationError", "InvalidConfigError", "MissingConfigError",
    "ExecutionError", "AgentExecutionError", "ToolExecutionError", "TimeoutError", "CircuitBreakerError",
    "ContextError", "ContextOverflowError", "CompressionError", "ChunkingError",
    "MemoryError", "MemoryRetrievalError", "MemoryStorageError", "ConsolidationError",
    "LearningError", "RewardCalculationError", "CreditAssignmentError", "PolicyUpdateError",
    "CommunicationError", "MessageDeliveryError", "FeedbackRoutingError",
    "ValidationError", "InputValidationError", "OutputValidationError",
    "PersistenceError", "StorageError", "RetrievalError",
    "IntegrationError", "LLMError", "DSPyError", "ExternalToolError",
    "wrap_exception",
    # Config
    "JottyConfig", "AgentConfig", "CLIConfig",
    # Types
    "MemoryLevel", "OutputTag", "MemoryEntry", "GoalValue",
    "ValidationResult", "EpisodeResult", "LearningMetrics", "AgentContribution", "AgentMessage",
]

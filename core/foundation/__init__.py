"""
Foundation Layer - Core Types and Protocols
============================================

This layer contains the fundamental building blocks that all other
components depend on. These files have minimal or no dependencies.

Modules:
--------
- data_structures: Core data types (SwarmConfig, EpisodeResult, etc.)
- protocols: Core protocols (MetadataProvider, etc.)
- agent_config: Agent configuration
- robust_parsing: Robust parsing utilities
- token_counter: Token counting utilities
- exceptions: Exception hierarchy for structured error handling
"""

from .data_structures import (
    SwarmConfig,
    JottyConfig,  # Backward compatibility
    MemoryLevel,
    ValidationResult,
    MemoryEntry,
    GoalValue,
    EpisodeResult,
    TaggedOutput,
    OutputTag,
    StoredEpisode,
    LearningMetrics,
    GoalHierarchy,
    GoalNode,
    CausalLink,
    AlertType,
    CommunicationType,
    ValidationRound,
    ContextType,
    RichObservation,
    AgentMessage,
    SharedScratchpad,
    AgentContribution,
)
from .protocols import (
    MetadataProvider,
    DataProvider,
    ContextExtractor,
)
from .agent_config import AgentConfig
from .robust_parsing import (
    parse_float_robust,
    parse_bool_robust,
    parse_json_robust,
    AdaptiveThreshold,
    safe_hash
)
from .token_counter import (
    count_tokens_accurate,
    estimate_tokens,
    get_tokenizer_info,
)
from .config_defaults import JottyDefaults, DEFAULTS
from .exceptions import (
    JottyError,
    ConfigurationError,
    InvalidConfigError,
    MissingConfigError,
    ExecutionError,
    AgentExecutionError,
    ToolExecutionError,
    TimeoutError,
    CircuitBreakerError,
    ContextError,
    ContextOverflowError,
    CompressionError,
    ChunkingError,
    MemoryError,
    MemoryRetrievalError,
    MemoryStorageError,
    ConsolidationError,
    LearningError,
    RewardCalculationError,
    CreditAssignmentError,
    PolicyUpdateError,
    CommunicationError,
    MessageDeliveryError,
    FeedbackRoutingError,
    ValidationError,
    InputValidationError,
    OutputValidationError,
    PersistenceError,
    StorageError,
    RetrievalError,
    IntegrationError,
    LLMError,
    DSPyError,
    ExternalToolError,
    wrap_exception,
)

__all__ = [
    # From data_structures
    'SwarmConfig',
    'JottyConfig',  # Backward compatibility
    'MemoryLevel',
    'ValidationResult',
    'MemoryEntry',
    'GoalValue',
    'EpisodeResult',
    'TaggedOutput',
    'OutputTag',
    'StoredEpisode',
    'LearningMetrics',
    'GoalHierarchy',
    'GoalNode',
    'CausalLink',
    'AlertType',
    'CommunicationType',
    'ValidationRound',
    'ContextType',
    'RichObservation',
    'AgentMessage',
    'SharedScratchpad',
    'AgentContribution',

    # From protocols
    'MetadataProvider',
    'DataProvider',
    'ContextExtractor',

    # From agent_config
    'AgentConfig',

    # From robust_parsing
    'parse_float_robust',
    'parse_bool_robust',
    'parse_json_robust',
    'AdaptiveThreshold',
    'safe_hash',

    # From token_counter
    'count_tokens_accurate',
    'estimate_tokens',
    'get_tokenizer_info',

    # From config_defaults
    'JottyDefaults',
    'DEFAULTS',

    # From exceptions
    'JottyError',
    'ConfigurationError',
    'InvalidConfigError',
    'MissingConfigError',
    'ExecutionError',
    'AgentExecutionError',
    'ToolExecutionError',
    'TimeoutError',
    'CircuitBreakerError',
    'ContextError',
    'ContextOverflowError',
    'CompressionError',
    'ChunkingError',
    'MemoryError',
    'MemoryRetrievalError',
    'MemoryStorageError',
    'ConsolidationError',
    'LearningError',
    'RewardCalculationError',
    'CreditAssignmentError',
    'PolicyUpdateError',
    'CommunicationError',
    'MessageDeliveryError',
    'FeedbackRoutingError',
    'ValidationError',
    'InputValidationError',
    'OutputValidationError',
    'PersistenceError',
    'StorageError',
    'RetrievalError',
    'IntegrationError',
    'LLMError',
    'DSPyError',
    'ExternalToolError',
    'wrap_exception',
]

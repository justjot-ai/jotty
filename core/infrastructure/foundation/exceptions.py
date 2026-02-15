from typing import Any

"""
Jotty Exception Hierarchy
==========================

A-Team Approved: Structured exception handling for all Jotty components.

This module defines the complete exception hierarchy for Jotty, replacing
bare `except:` and broad `except Exception:` patterns with specific,
meaningful exceptions.

Exception Hierarchy:
-------------------
JottyError (base)
├── ConfigurationError
│   ├── InvalidConfigError
│   └── MissingConfigError
├── ExecutionError
│   ├── AgentExecutionError
│   ├── ToolExecutionError
│   └── TimeoutError
├── ContextError
│   ├── ContextOverflowError
│   ├── CompressionError
│   └── ChunkingError
├── MemoryError
│   ├── MemoryRetrievalError
│   ├── MemoryStorageError
│   └── ConsolidationError
├── LearningError
│   ├── RewardCalculationError
│   ├── CreditAssignmentError
│   └── PolicyUpdateError
├── CommunicationError
│   ├── MessageDeliveryError
│   └── FeedbackRoutingError
├── ValidationError
│   ├── InputValidationError
│   └── OutputValidationError
├── PersistenceError
│   ├── StorageError
│   └── RetrievalError
└── IntegrationError
    ├── LLMError
    ├── DSPyError
    └── ExternalToolError

Usage:
------
    from core.foundation.exceptions import AgentExecutionError, ContextOverflowError

    try:
        result = agent.execute(task)
    except AgentExecutionError as e:
        logger.error(f"Agent execution failed: {e}")
        # Handle gracefully
    except ContextOverflowError as e:
        logger.warning(f"Context overflow: {e}")
        # Compress and retry
"""

from typing import Any, Dict, Optional

# =============================================================================
# BASE EXCEPTION
# =============================================================================


class JottyError(Exception):
    """
    Base exception for all Jotty framework errors.

    All custom exceptions in Jotty should inherit from this class.
    This allows catching all Jotty-specific errors with a single handler.

    Attributes:
        message: Human-readable error message
        context: Additional context about the error (dict)
        original_error: Original exception if this wraps another error
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        self.message = message
        self.context = context or {}
        self.original_error = original_error

        # Build full message
        full_message = message
        if context:
            full_message += f" | Context: {context}"
        if original_error:
            full_message += f" | Caused by: {type(original_error).__name__}: {original_error}"

        super().__init__(full_message)


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================


class ConfigurationError(JottyError):
    """Base class for configuration-related errors."""

    pass


class InvalidConfigError(ConfigurationError):
    """Raised when configuration values are invalid."""

    pass


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""

    pass


# =============================================================================
# EXECUTION ERRORS
# =============================================================================


class ExecutionError(JottyError):
    """Base class for execution-related errors."""

    pass


class AgentExecutionError(ExecutionError):
    """
    Raised when agent execution fails.

    This includes DSPy module failures, prompt errors, and other
    agent-specific execution problems.
    """

    pass


class ToolExecutionError(ExecutionError):
    """Raised when a tool/function call fails."""

    pass


class TimeoutError(ExecutionError):
    """Raised when an operation exceeds its timeout."""

    pass


class CircuitBreakerError(ExecutionError):
    """Raised when circuit breaker is open and blocks execution."""

    pass


# =============================================================================
# CONTEXT ERRORS
# =============================================================================


class ContextError(JottyError):
    """Base class for context management errors."""

    pass


class ContextOverflowError(ContextError):
    """
    Raised when context exceeds token limit.

    This is a recoverable error - the system should attempt compression
    and retry before failing.
    """

    def __init__(
        self,
        message: str,
        detected_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.detected_tokens = detected_tokens
        self.max_tokens = max_tokens

        context = kwargs.get("context", {})
        if detected_tokens:
            context["detected_tokens"] = detected_tokens
        if max_tokens:
            context["max_tokens"] = max_tokens
        kwargs["context"] = context

        super().__init__(message, **kwargs)


class CompressionError(ContextError):
    """Raised when content compression fails."""

    pass


class ChunkingError(ContextError):
    """Raised when content chunking fails."""

    pass


# =============================================================================
# MEMORY ERRORS
# =============================================================================


class MemoryError(JottyError):
    """Base class for memory system errors."""

    pass


class MemoryRetrievalError(MemoryError):
    """Raised when memory retrieval fails."""

    pass


class MemoryStorageError(MemoryError):
    """Raised when memory storage fails."""

    pass


class ConsolidationError(MemoryError):
    """Raised when memory consolidation fails."""

    pass


# =============================================================================
# LEARNING ERRORS
# =============================================================================


class LearningError(JottyError):
    """Base class for learning/RL errors."""

    pass


class RewardCalculationError(LearningError):
    """Raised when reward calculation fails."""

    pass


class CreditAssignmentError(LearningError):
    """Raised when credit assignment fails."""

    pass


class PolicyUpdateError(LearningError):
    """Raised when policy update fails."""

    pass


# =============================================================================
# COMMUNICATION ERRORS
# =============================================================================


class CommunicationError(JottyError):
    """Base class for agent communication errors."""

    pass


class MessageDeliveryError(CommunicationError):
    """Raised when message delivery fails."""

    pass


class FeedbackRoutingError(CommunicationError):
    """Raised when feedback routing fails."""

    pass


# =============================================================================
# VALIDATION ERRORS
# =============================================================================


class ValidationError(JottyError):
    """Base class for validation errors.

    Supports param/value tracking for tool parameter validation
    and structured error responses via to_dict().
    """

    def __init__(
        self,
        message: str,
        param: str = None,
        value: Any = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, context=context, original_error=original_error)
        self.param = param
        self.value = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for tool responses."""
        return {"success": False, "error": self.message, "param": self.param}


class InputValidationError(ValidationError):
    """Raised when input validation fails."""

    pass


class OutputValidationError(ValidationError):
    """Raised when output validation fails."""

    pass


# =============================================================================
# PERSISTENCE ERRORS
# =============================================================================


class PersistenceError(JottyError):
    """Base class for persistence/storage errors."""

    pass


class StorageError(PersistenceError):
    """Raised when storage operation fails."""

    pass


class RetrievalError(PersistenceError):
    """Raised when retrieval operation fails."""

    pass


# =============================================================================
# INTEGRATION ERRORS
# =============================================================================


class IntegrationError(JottyError):
    """Base class for external integration errors."""

    pass


class LLMError(IntegrationError):
    """
    Raised when LLM API call fails.

    This includes OpenAI, Anthropic, and other LLM provider errors.
    """

    pass


class DSPyError(IntegrationError):
    """Raised when DSPy operation fails."""

    pass


class ExternalToolError(IntegrationError):
    """Raised when external tool integration fails."""

    pass


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def wrap_exception(
    original: Exception, jotty_exception_class: type[JottyError], message: str, **kwargs: Any
) -> JottyError:
    """
    Wrap a generic exception in a Jotty-specific exception.

    Args:
        original: The original exception to wrap
        jotty_exception_class: The Jotty exception class to use
        message: Human-readable message
        **kwargs: Additional context

    Returns:
        JottyError instance wrapping the original exception

    Example:
        try:
            result = risky_operation()
        except ValueError as e:
            raise wrap_exception(
                e,
                AgentExecutionError,
                "Agent failed to process input",
                agent_name="MyAgent"
            )
    """
    return jotty_exception_class(message=message, context=kwargs, original_error=original)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base
    "JottyError",
    # Configuration
    "ConfigurationError",
    "InvalidConfigError",
    "MissingConfigError",
    # Execution
    "ExecutionError",
    "AgentExecutionError",
    "ToolExecutionError",
    "TimeoutError",
    "CircuitBreakerError",
    # Context
    "ContextError",
    "ContextOverflowError",
    "CompressionError",
    "ChunkingError",
    # Memory
    "MemoryError",
    "MemoryRetrievalError",
    "MemoryStorageError",
    "ConsolidationError",
    # Learning
    "LearningError",
    "RewardCalculationError",
    "CreditAssignmentError",
    "PolicyUpdateError",
    # Communication
    "CommunicationError",
    "MessageDeliveryError",
    "FeedbackRoutingError",
    # Validation
    "ValidationError",
    "InputValidationError",
    "OutputValidationError",
    # Persistence
    "PersistenceError",
    "StorageError",
    "RetrievalError",
    # Integration
    "IntegrationError",
    "LLMError",
    "DSPyError",
    "ExternalToolError",
    # Utilities
    "wrap_exception",
]

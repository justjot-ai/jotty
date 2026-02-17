"""
Error Handling for Context Management
======================================

Provides error classification, retry strategies, and compression results
for context-aware error recovery.

Migrated from utils/context_utils.py to consolidate all context management
into the canonical context subsystem.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Categorized error types for different handling strategies."""

    CONTEXT_LENGTH = "context_length"
    TIMEOUT = "timeout"
    PARSE_ERROR = "parse_error"
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    TOOL_ERROR = "tool_error"
    UNKNOWN = "unknown"


@dataclass
class CompressionResult:
    """Result of context compression."""

    original_length: int
    compressed_length: int
    compression_ratio: float
    content: str
    preserved_trajectory: str = ""


class ErrorDetector:
    """
    Detect and categorize errors for appropriate handling.

    Different error types require different strategies:
    - Context length: Compress and retry
    - Timeout: Exponential backoff retry
    - Parse error: Simplify prompt and retry
    - Rate limit: Wait and retry
    """

    # Error patterns by category
    CONTEXT_LENGTH_PATTERNS = [
        "input is too long",
        "context length exceeded",
        "maximum context length",
        "token limit exceeded",
        "too many tokens",
        "context window exceeded",
        "prompt is too long",
        "request too large",
    ]

    TIMEOUT_PATTERNS = [
        "timeout",
        "timed out",
        "request timeout",
        "connection timeout",
        "read timeout",
    ]

    PARSE_PATTERNS = [
        "failed to parse",
        "json decode error",
        "invalid json",
        "parse error",
        "adapter parse error",
        "validation error",
    ]

    RATE_LIMIT_PATTERNS = [
        "rate limit",
        "too many requests",
        "quota exceeded",
        "throttled",
        "429",
    ]

    NETWORK_PATTERNS = [
        "network error",
        "connection error",
        "connection refused",
        "dns",
        "socket",
        "ssl",
        "certificate",
    ]

    @classmethod
    def detect(cls, error: Exception) -> ErrorType:
        """
        Detect the type of error.

        Args:
            error: The exception to categorize

        Returns:
            ErrorType enum value
        """
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()

        # Check each category
        if cls._matches_patterns(error_str, cls.CONTEXT_LENGTH_PATTERNS):
            return ErrorType.CONTEXT_LENGTH

        if cls._matches_patterns(error_str, cls.TIMEOUT_PATTERNS) or "timeout" in error_type_name:
            return ErrorType.TIMEOUT

        if cls._matches_patterns(error_str, cls.PARSE_PATTERNS) or "parse" in error_type_name:
            return ErrorType.PARSE_ERROR

        if cls._matches_patterns(error_str, cls.RATE_LIMIT_PATTERNS):
            return ErrorType.RATE_LIMIT

        if cls._matches_patterns(error_str, cls.NETWORK_PATTERNS):
            return ErrorType.NETWORK

        return ErrorType.UNKNOWN

    @classmethod
    def _matches_patterns(cls, text: str, patterns: List[str]) -> bool:
        """Check if text matches any pattern."""
        return any(pattern in text for pattern in patterns)

    @classmethod
    def get_retry_strategy(cls, error_type: ErrorType) -> Dict[str, Any]:
        """
        Get recommended retry strategy for error type.

        Returns:
            Dict with retry configuration
        """
        strategies = {
            ErrorType.CONTEXT_LENGTH: {
                "should_retry": True,
                "action": "compress",
                "max_retries": 3,
                "delay_seconds": 0,
            },
            ErrorType.TIMEOUT: {
                "should_retry": True,
                "action": "backoff",
                "max_retries": 3,
                "delay_seconds": 2,
            },
            ErrorType.PARSE_ERROR: {
                "should_retry": True,
                "action": "simplify",
                "max_retries": 2,
                "delay_seconds": 0,
            },
            ErrorType.RATE_LIMIT: {
                "should_retry": True,
                "action": "wait",
                "max_retries": 3,
                "delay_seconds": 30,
            },
            ErrorType.NETWORK: {
                "should_retry": True,
                "action": "backoff",
                "max_retries": 3,
                "delay_seconds": 1,
            },
            ErrorType.UNKNOWN: {
                "should_retry": False,
                "action": "fail",
                "max_retries": 0,
                "delay_seconds": 0,
            },
        }
        return strategies.get(error_type, strategies[ErrorType.UNKNOWN])


def detect_error_type(error: Exception) -> Tuple[ErrorType, Dict[str, Any]]:
    """
    Convenience function to detect error type and get retry strategy.

    Returns:
        Tuple of (ErrorType, retry_strategy_dict)
    """
    error_type = ErrorDetector.detect(error)
    strategy = ErrorDetector.get_retry_strategy(error_type)
    return error_type, strategy


__all__ = [
    "ErrorType",
    "CompressionResult",
    "ErrorDetector",
    "detect_error_type",
]

"""
Jotty Configuration Defaults
============================

A-Team Approved: Centralized, documented defaults for all Jotty components.

This module consolidates all hardcoded configuration values from across the
codebase into a single, well-documented location. This makes it easy to:
- Tune parameters for different models
- Adjust budgets for different use cases
- Understand what each value controls

Usage:
------
    from core.foundation.config_defaults import JottyDefaults

    config = JottyConfig(
        max_context_tokens=JottyDefaults.MAX_CONTEXT_TOKENS,
        episodic_capacity=JottyDefaults.EPISODIC_CAPACITY
    )
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class JottyDefaults:
    """
    Centralized default values for all Jotty configuration parameters.

    These values are tuned for Claude Sonnet/GPT-4 class models with
    ~100K context windows. Adjust for smaller/larger models as needed.

    Categories:
    -----------
    1. Token Budgets - How much context to allocate to different components
    2. Memory Capacity - How many items to store in different memory levels
    3. Learning Parameters - RL and credit assignment settings
    4. Chunking & Compression - Content processing thresholds
    5. Timeouts & Limits - Execution constraints
    """

    # =========================================================================
    # TOKEN BUDGETS
    # =========================================================================

    # Total context available to the model
    MAX_CONTEXT_TOKENS: int = 100_000
    """Total tokens available in model context window."""

    # Safety margin to avoid hitting exact limit
    SAFETY_MARGIN: int = 2_000
    """Tokens to reserve as safety buffer."""

    # Component-specific budgets
    SYSTEM_PROMPT_BUDGET: int = 5_000
    """Max tokens for system prompts and instructions."""

    CURRENT_INPUT_BUDGET: int = 15_000
    """Max tokens for current user input."""

    TRAJECTORY_BUDGET: int = 20_000
    """Max tokens for execution trajectory/history."""

    TOOL_OUTPUT_BUDGET: int = 15_000
    """Max tokens for tool call outputs."""

    MIN_MEMORY_BUDGET: int = 10_000
    """Minimum tokens to reserve for memories."""

    MAX_MEMORY_BUDGET: int = 60_000
    """Maximum tokens for all memories combined."""

    PREVIEW_TOKEN_BUDGET: int = 20_000
    """Max tokens for data preview/exploration."""

    MAX_DESCRIPTION_TOKENS: int = 5_000
    """Max tokens for descriptions and metadata."""

    # Entry size limits
    MAX_ENTRY_TOKENS: int = 2_000
    """Maximum tokens per single memory entry."""

    # Chunking threshold
    CHUNKING_THRESHOLD_TOKENS: int = 15_000
    """When to activate auto-chunking (60% of usable context)."""

    # =========================================================================
    # MEMORY CAPACITY
    # =========================================================================

    EPISODIC_CAPACITY: int = 1_000
    """Max entries in episodic memory (recent experiences)."""

    SEMANTIC_CAPACITY: int = 500
    """Max entries in semantic memory (facts and concepts)."""

    PROCEDURAL_CAPACITY: int = 100
    """Max entries in procedural memory (how-to knowledge)."""

    META_CAPACITY: int = 50
    """Max entries in meta memory (learning about learning)."""

    CAUSAL_CAPACITY: int = 200
    """Max entries in causal memory (cause-effect relationships)."""

    EPISODE_BUFFER_SIZE: int = 1_000
    """Max episodes to keep in RL buffer."""

    DLQ_MAX_SIZE: int = 1_000
    """Max items in dead letter queue."""

    # =========================================================================
    # LEARNING PARAMETERS
    # =========================================================================

    # TD(λ) Learning
    LEARNING_RATE: float = 0.1
    """Base learning rate for TD updates."""

    DISCOUNT_FACTOR: float = 0.99
    """Gamma - how much to value future rewards."""

    LAMBDA_VALUE: float = 0.9
    """Lambda - trace decay parameter for TD(λ)."""

    # Credit Assignment
    MIN_CREDIT: float = 0.01
    """Minimum credit to assign to any agent."""

    MAX_CREDIT: float = 1.0
    """Maximum credit any single agent can receive."""

    # Exploration
    EPSILON_START: float = 0.3
    """Starting epsilon for epsilon-greedy exploration."""

    EPSILON_MIN: float = 0.05
    """Minimum epsilon (always explore a bit)."""

    EPSILON_DECAY: float = 0.995
    """Epsilon decay rate per episode."""

    # =========================================================================
    # CHUNKING & COMPRESSION
    # =========================================================================

    # Chunk sizes
    DEFAULT_CHUNK_SIZE: int = 1_000
    """Default chunk size in tokens."""

    CHUNK_OVERLAP: int = 200
    """Overlap between consecutive chunks."""

    MIN_CHUNK_SIZE: int = 100
    """Minimum viable chunk size."""

    MAX_CHUNKS: int = 100
    """Maximum chunks to process."""

    # Compression targets
    COMPRESSION_RATIO: float = 0.7
    """Target: reduce to 70% of original size."""

    MIN_COMPRESSION_SIZE: int = 1_000
    """Don't compress if smaller than this."""

    QUALITY_THRESHOLD: float = 0.6
    """Minimum quality score for compressed content."""

    # =========================================================================
    # TIMEOUTS & LIMITS
    # =========================================================================

    # Execution timeouts
    DEFAULT_TIMEOUT_SECONDS: int = 300
    """Default timeout for agent execution (5 minutes)."""

    TOOL_TIMEOUT_SECONDS: int = 60
    """Timeout for individual tool calls (1 minute)."""

    LLM_TIMEOUT_SECONDS: int = 120
    """Timeout for LLM API calls (2 minutes)."""

    # Circuit breaker
    CIRCUIT_BREAKER_THRESHOLD: int = 5
    """Failures before circuit opens."""

    CIRCUIT_BREAKER_TIMEOUT: int = 60
    """Seconds to wait before retrying after circuit opens."""

    # Retry limits
    MAX_RETRIES: int = 3
    """Maximum retry attempts for failed operations."""

    RETRY_BACKOFF_SECONDS: float = 1.0
    """Base backoff time between retries."""

    # =========================================================================
    # VALIDATION & QUALITY
    # =========================================================================

    MIN_CONFIDENCE: float = 0.5
    """Minimum confidence for parameter matching."""

    MIN_RELEVANCE_SCORE: float = 0.6
    """Minimum relevance for memory retrieval."""

    MIN_SURPRISE_LEVEL: int = 1
    """Minimum surprise level (1-10 scale)."""

    MAX_SURPRISE_LEVEL: int = 10
    """Maximum surprise level (1-10 scale)."""

    # Validation rounds
    MAX_VALIDATION_ROUNDS: int = 3
    """Maximum rounds of iterative validation."""

    # =========================================================================
    # LOGGING & MONITORING
    # =========================================================================

    LOG_PREVIEW_LENGTH: int = 200
    """Characters to show in log previews."""

    MAX_LOG_ENTRIES: int = 10_000
    """Maximum log entries to keep in memory."""

    HEALTH_CHECK_INTERVAL: int = 300
    """Seconds between health checks."""

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @classmethod
    def for_model(cls, model_name: str) -> 'JottyDefaults':
        """
        Get recommended defaults for a specific model.

        Args:
            model_name: Model identifier (e.g., "gpt-4", "claude-3-opus")

        Returns:
            JottyDefaults instance tuned for that model

        Example:
            defaults = JottyDefaults.for_model("gpt-4-32k")
        """
        # This is a placeholder for model-specific tuning
        # In the future, we can return different configurations
        # based on known model characteristics
        return cls()

    @classmethod
    def conservative(cls) -> 'JottyDefaults':
        """
        Get conservative defaults for safety.

        Returns:
            JottyDefaults with reduced limits and higher safety margins
        """
        # Return a modified instance with conservative settings
        # This is a placeholder for future implementation
        return cls()

    @classmethod
    def aggressive(cls) -> 'JottyDefaults':
        """
        Get aggressive defaults for maximum performance.

        Returns:
            JottyDefaults with higher limits and tighter margins
        """
        # Return a modified instance with aggressive settings
        # This is a placeholder for future implementation
        return cls()


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

# Make defaults available directly
DEFAULTS = JottyDefaults()

# Common aliases for backward compatibility
MAX_TOKENS = DEFAULTS.MAX_CONTEXT_TOKENS
SAFETY_MARGIN = DEFAULTS.SAFETY_MARGIN
EPISODIC_CAPACITY = DEFAULTS.EPISODIC_CAPACITY
MAX_ENTRY_TOKENS = DEFAULTS.MAX_ENTRY_TOKENS

__all__ = [
    'JottyDefaults',
    'DEFAULTS',
    'MAX_TOKENS',
    'SAFETY_MARGIN',
    'EPISODIC_CAPACITY',
    'MAX_ENTRY_TOKENS',
]

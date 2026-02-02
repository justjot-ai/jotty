"""
Budget Tracker
==============

Track and enforce LLM call budgets to prevent runaway costs.

A-Team Critical Fix: Add cost/budget controls to prevent unbounded LLM usage.

Features:
- Track calls per episode/agent
- Track token usage
- Raise BudgetExceededError when limit hit
- Emit warnings at threshold
- Integration with circuit breakers
"""

import time
import logging
import threading
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class BudgetScope(Enum):
    """Scope levels for budget tracking."""
    GLOBAL = "global"           # Total across everything
    EPISODE = "episode"         # Per episode/session
    AGENT = "agent"             # Per individual agent
    OPERATION = "operation"     # Per operation type


class BudgetExceededError(Exception):
    """Raised when a budget limit is exceeded."""

    def __init__(
        self,
        message: str,
        scope: BudgetScope,
        current: int,
        limit: int,
        resource: str = "calls"
    ):
        super().__init__(message)
        self.scope = scope
        self.current = current
        self.limit = limit
        self.resource = resource


@dataclass
class BudgetUsage:
    """Tracks usage for a single budget category."""
    calls: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    estimated_cost_usd: float = 0.0
    last_call_time: float = 0.0
    warnings_emitted: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.tokens_input + self.tokens_output

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'calls': self.calls,
            'tokens_input': self.tokens_input,
            'tokens_output': self.tokens_output,
            'total_tokens': self.total_tokens,
            'estimated_cost_usd': self.estimated_cost_usd,
            'last_call_time': self.last_call_time,
            'warnings_emitted': self.warnings_emitted,
        }


@dataclass
class BudgetConfig:
    """Configuration for budget limits."""
    # Call limits
    max_llm_calls_per_episode: int = 100
    max_llm_calls_per_agent: int = 50
    max_llm_calls_global: int = 10000

    # Token limits
    max_total_tokens_per_episode: int = 500000
    max_tokens_per_agent: int = 100000
    max_tokens_per_call: int = 50000

    # Cost limits (USD)
    max_cost_per_episode: Optional[float] = None
    max_cost_global: Optional[float] = None

    # Warning thresholds (fraction of limit)
    warning_threshold: float = 0.8  # Warn at 80%

    # Enforcement
    enable_enforcement: bool = True
    soft_limit_mode: bool = False  # If True, warn but don't block

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'max_llm_calls_per_episode': self.max_llm_calls_per_episode,
            'max_llm_calls_per_agent': self.max_llm_calls_per_agent,
            'max_llm_calls_global': self.max_llm_calls_global,
            'max_total_tokens_per_episode': self.max_total_tokens_per_episode,
            'max_tokens_per_agent': self.max_tokens_per_agent,
            'max_tokens_per_call': self.max_tokens_per_call,
            'max_cost_per_episode': self.max_cost_per_episode,
            'max_cost_global': self.max_cost_global,
            'warning_threshold': self.warning_threshold,
            'enable_enforcement': self.enable_enforcement,
            'soft_limit_mode': self.soft_limit_mode,
        }


# Default cost per 1000 tokens for estimation (can be updated per model)
DEFAULT_COST_PER_1K_INPUT = 0.01  # $0.01 per 1K input tokens
DEFAULT_COST_PER_1K_OUTPUT = 0.03  # $0.03 per 1K output tokens


class BudgetTracker:
    """
    Track and enforce LLM call budgets.

    Features:
    - Multi-scope tracking (global, episode, agent)
    - Call and token counting
    - Cost estimation
    - Configurable limits and warnings
    - Thread-safe operations

    Usage:
        tracker = BudgetTracker(config=BudgetConfig())

        # Start new episode
        tracker.start_episode("episode_001")

        # Track a call
        tracker.record_call(
            agent_name="SQLGenerator",
            tokens_input=500,
            tokens_output=200
        )

        # Check if limit reached
        if tracker.can_make_call("SQLGenerator"):
            result = make_llm_call()
            tracker.record_call("SQLGenerator", ...)

        # End episode
        tracker.end_episode()
    """

    _instances: Dict[str, 'BudgetTracker'] = {}

    def __init__(
        self,
        config: Optional[BudgetConfig] = None,
        cost_per_1k_input: float = DEFAULT_COST_PER_1K_INPUT,
        cost_per_1k_output: float = DEFAULT_COST_PER_1K_OUTPUT
    ):
        """
        Initialize budget tracker.

        Args:
            config: Budget configuration
            cost_per_1k_input: Cost per 1000 input tokens (USD)
            cost_per_1k_output: Cost per 1000 output tokens (USD)
        """
        self.config = config or BudgetConfig()
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output

        # Usage tracking
        self._global_usage = BudgetUsage()
        self._episode_usage = BudgetUsage()
        self._agent_usage: Dict[str, BudgetUsage] = defaultdict(BudgetUsage)

        # Current episode
        self._current_episode: Optional[str] = None

        # Thread safety
        self._lock = threading.RLock()

        # Warning callbacks
        self._warning_callbacks: list = []

        logger.info(
            f"BudgetTracker initialized: "
            f"max_calls_per_episode={self.config.max_llm_calls_per_episode}, "
            f"max_tokens_per_episode={self.config.max_total_tokens_per_episode}"
        )

    @classmethod
    def get_instance(cls, name: str = "default", **kwargs) -> 'BudgetTracker':
        """Get a singleton instance by name."""
        if name not in cls._instances:
            cls._instances[name] = cls(**kwargs)
        return cls._instances[name]

    def start_episode(self, episode_id: str) -> None:
        """
        Start a new episode, resetting episode-level tracking.

        Args:
            episode_id: Identifier for this episode
        """
        with self._lock:
            self._current_episode = episode_id
            self._episode_usage = BudgetUsage()
            self._agent_usage.clear()

            logger.info(f"Budget episode started: {episode_id}")

    def end_episode(self) -> Dict[str, Any]:
        """
        End current episode and return usage summary.

        Returns:
            Dictionary with episode usage statistics
        """
        with self._lock:
            summary = {
                'episode_id': self._current_episode,
                'episode_usage': self._episode_usage.to_dict(),
                'agent_usage': {
                    agent: usage.to_dict()
                    for agent, usage in self._agent_usage.items()
                },
                'global_usage': self._global_usage.to_dict(),
            }

            logger.info(
                f"Budget episode ended: {self._current_episode} - "
                f"{self._episode_usage.calls} calls, "
                f"{self._episode_usage.total_tokens} tokens, "
                f"${self._episode_usage.estimated_cost_usd:.4f}"
            )

            self._current_episode = None
            return summary

    def can_make_call(
        self,
        agent_name: str,
        estimated_tokens: int = 0
    ) -> bool:
        """
        Check if an LLM call can be made within budget.

        Args:
            agent_name: Name of the agent making the call
            estimated_tokens: Estimated tokens for this call

        Returns:
            True if call is within budget
        """
        if not self.config.enable_enforcement:
            return True

        with self._lock:
            # Check episode call limit
            if self._episode_usage.calls >= self.config.max_llm_calls_per_episode:
                if not self.config.soft_limit_mode:
                    return False

            # Check agent call limit
            agent_usage = self._agent_usage[agent_name]
            if agent_usage.calls >= self.config.max_llm_calls_per_agent:
                if not self.config.soft_limit_mode:
                    return False

            # Check token limit
            projected_tokens = self._episode_usage.total_tokens + estimated_tokens
            if projected_tokens > self.config.max_total_tokens_per_episode:
                if not self.config.soft_limit_mode:
                    return False

            # Check single call token limit
            if estimated_tokens > self.config.max_tokens_per_call:
                if not self.config.soft_limit_mode:
                    return False

            return True

    def record_call(
        self,
        agent_name: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        model: str = "",
        cost_override: Optional[float] = None
    ) -> None:
        """
        Record an LLM call and update usage.

        Args:
            agent_name: Name of the agent making the call
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens
            model: Model name (for future model-specific pricing)
            cost_override: Override automatic cost calculation

        Raises:
            BudgetExceededError: If budget exceeded and enforcement enabled
        """
        with self._lock:
            # Calculate cost
            if cost_override is not None:
                cost = cost_override
            else:
                cost = self._estimate_cost(tokens_input, tokens_output)

            current_time = time.time()

            # Update global usage
            self._global_usage.calls += 1
            self._global_usage.tokens_input += tokens_input
            self._global_usage.tokens_output += tokens_output
            self._global_usage.estimated_cost_usd += cost
            self._global_usage.last_call_time = current_time

            # Update episode usage
            self._episode_usage.calls += 1
            self._episode_usage.tokens_input += tokens_input
            self._episode_usage.tokens_output += tokens_output
            self._episode_usage.estimated_cost_usd += cost
            self._episode_usage.last_call_time = current_time

            # Update agent usage
            agent_usage = self._agent_usage[agent_name]
            agent_usage.calls += 1
            agent_usage.tokens_input += tokens_input
            agent_usage.tokens_output += tokens_output
            agent_usage.estimated_cost_usd += cost
            agent_usage.last_call_time = current_time

            # Check limits and emit warnings/errors
            self._check_limits(agent_name)

            logger.debug(
                f"Budget recorded: {agent_name} - "
                f"tokens={tokens_input+tokens_output}, cost=${cost:.4f}"
            )

    def _estimate_cost(self, tokens_input: int, tokens_output: int) -> float:
        """Estimate cost from token counts."""
        input_cost = (tokens_input / 1000) * self.cost_per_1k_input
        output_cost = (tokens_output / 1000) * self.cost_per_1k_output
        return input_cost + output_cost

    def _check_limits(self, agent_name: str) -> None:
        """Check limits and emit warnings/errors."""
        # Check episode call limit
        episode_call_ratio = (
            self._episode_usage.calls / self.config.max_llm_calls_per_episode
        )
        if episode_call_ratio >= 1.0:
            self._handle_limit_exceeded(
                BudgetScope.EPISODE,
                self._episode_usage.calls,
                self.config.max_llm_calls_per_episode,
                "calls"
            )
        elif episode_call_ratio >= self.config.warning_threshold:
            self._emit_warning(
                BudgetScope.EPISODE,
                "calls",
                self._episode_usage.calls,
                self.config.max_llm_calls_per_episode
            )

        # Check agent call limit
        agent_usage = self._agent_usage[agent_name]
        agent_call_ratio = (
            agent_usage.calls / self.config.max_llm_calls_per_agent
        )
        if agent_call_ratio >= 1.0:
            self._handle_limit_exceeded(
                BudgetScope.AGENT,
                agent_usage.calls,
                self.config.max_llm_calls_per_agent,
                "calls"
            )
        elif agent_call_ratio >= self.config.warning_threshold:
            self._emit_warning(
                BudgetScope.AGENT,
                "calls",
                agent_usage.calls,
                self.config.max_llm_calls_per_agent,
                agent_name
            )

        # Check token limit
        token_ratio = (
            self._episode_usage.total_tokens /
            self.config.max_total_tokens_per_episode
        )
        if token_ratio >= 1.0:
            self._handle_limit_exceeded(
                BudgetScope.EPISODE,
                self._episode_usage.total_tokens,
                self.config.max_total_tokens_per_episode,
                "tokens"
            )
        elif token_ratio >= self.config.warning_threshold:
            self._emit_warning(
                BudgetScope.EPISODE,
                "tokens",
                self._episode_usage.total_tokens,
                self.config.max_total_tokens_per_episode
            )

        # Check cost limits
        if self.config.max_cost_per_episode:
            cost_ratio = (
                self._episode_usage.estimated_cost_usd /
                self.config.max_cost_per_episode
            )
            if cost_ratio >= 1.0:
                self._handle_limit_exceeded(
                    BudgetScope.EPISODE,
                    int(self._episode_usage.estimated_cost_usd * 100),
                    int(self.config.max_cost_per_episode * 100),
                    "cost_cents"
                )
            elif cost_ratio >= self.config.warning_threshold:
                self._emit_warning(
                    BudgetScope.EPISODE,
                    "cost",
                    self._episode_usage.estimated_cost_usd,
                    self.config.max_cost_per_episode
                )

    def _handle_limit_exceeded(
        self,
        scope: BudgetScope,
        current: int,
        limit: int,
        resource: str
    ) -> None:
        """Handle a limit being exceeded."""
        message = (
            f"Budget limit exceeded: {scope.value} {resource} "
            f"({current}/{limit})"
        )

        if self.config.enable_enforcement and not self.config.soft_limit_mode:
            logger.error(f"❌ {message}")
            raise BudgetExceededError(
                message=message,
                scope=scope,
                current=current,
                limit=limit,
                resource=resource
            )
        else:
            logger.warning(f"⚠️ {message} (soft limit mode)")

    def _emit_warning(
        self,
        scope: BudgetScope,
        resource: str,
        current: Any,
        limit: Any,
        context: str = ""
    ) -> None:
        """Emit a warning about approaching limit."""
        # Track warnings to avoid spamming
        usage = (
            self._episode_usage if scope == BudgetScope.EPISODE
            else self._agent_usage.get(context, self._global_usage)
        )

        # Only warn once per threshold crossing
        ratio = current / limit if limit > 0 else 0
        warning_level = int(ratio * 10)  # 0-10

        if warning_level > usage.warnings_emitted:
            usage.warnings_emitted = warning_level

            message = (
                f"Budget warning: {scope.value} {resource} at "
                f"{ratio:.0%} ({current}/{limit})"
            )
            if context:
                message += f" for {context}"

            logger.warning(f"⚠️ {message}")

            # Call registered callbacks
            for callback in self._warning_callbacks:
                try:
                    callback(scope, resource, current, limit, context)
                except Exception as e:
                    logger.error(f"Warning callback error: {e}")

    def add_warning_callback(self, callback: Callable) -> None:
        """Add a callback to be called on warnings."""
        self._warning_callbacks.append(callback)

    def get_usage(self, scope: BudgetScope = BudgetScope.EPISODE) -> Dict[str, Any]:
        """Get current usage for a scope."""
        with self._lock:
            if scope == BudgetScope.GLOBAL:
                return self._global_usage.to_dict()
            elif scope == BudgetScope.EPISODE:
                return self._episode_usage.to_dict()
            else:
                return {
                    agent: usage.to_dict()
                    for agent, usage in self._agent_usage.items()
                }

    def get_remaining(self, scope: BudgetScope = BudgetScope.EPISODE) -> Dict[str, int]:
        """Get remaining budget for a scope."""
        with self._lock:
            if scope == BudgetScope.EPISODE:
                return {
                    'calls': max(0, self.config.max_llm_calls_per_episode - self._episode_usage.calls),
                    'tokens': max(0, self.config.max_total_tokens_per_episode - self._episode_usage.total_tokens),
                }
            elif scope == BudgetScope.GLOBAL:
                return {
                    'calls': max(0, self.config.max_llm_calls_global - self._global_usage.calls),
                }
            return {}

    def reset_global(self) -> None:
        """Reset global usage counters."""
        with self._lock:
            self._global_usage = BudgetUsage()
            logger.info("Global budget counters reset")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_budget_tracker(name: str = "default", **kwargs) -> BudgetTracker:
    """Get a named budget tracker instance."""
    return BudgetTracker.get_instance(name, **kwargs)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BudgetTracker',
    'BudgetConfig',
    'BudgetUsage',
    'BudgetScope',
    'BudgetExceededError',
    'get_budget_tracker',
]

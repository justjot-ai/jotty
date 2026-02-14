"""
Utilities Subsystem Facade
============================

Clean, discoverable API for cross-cutting utility components.
No new business logic — just imports + convenience accessors.

Usage:
    from Jotty.core.utils.facade import get_budget_tracker, list_components

    tracker = get_budget_tracker()
    tracker.check_budget("my-scope")
"""

import threading
from typing import Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from Jotty.core.utils.budget_tracker import BudgetTracker
    from Jotty.core.utils.timeouts import CircuitBreaker
    from Jotty.core.utils.llm_cache import LLMCallCache
    from Jotty.core.utils.tokenizer import SmartTokenizer

_lock = threading.Lock()
_singletons: Dict[str, object] = {}


def get_budget_tracker(name: str = "default") -> 'BudgetTracker':
    """
    Return a BudgetTracker instance for LLM spending control.

    Args:
        name: Budget tracker name/scope.

    Returns:
        BudgetTracker instance.
    """
    from Jotty.core.utils.budget_tracker import get_budget_tracker as _get
    return _get(name)


def get_circuit_breaker(name: str = "default") -> 'CircuitBreaker':
    """
    Return a CircuitBreaker instance for fail-fast patterns.

    Args:
        name: Circuit breaker name.

    Returns:
        CircuitBreaker instance.
    """
    from Jotty.core.utils.timeouts import CircuitBreaker, CircuitBreakerConfig
    return CircuitBreaker(config=CircuitBreakerConfig(name=name))


def get_llm_cache() -> 'LLMCallCache':
    """
    Return an LLMCallCache instance for semantic caching.

    Returns:
        LLMCallCache instance.
    """
    from Jotty.core.utils.llm_cache import get_cache
    return get_cache()


def get_tokenizer(encoding_name: Optional[str] = None) -> 'SmartTokenizer':
    """
    Return a SmartTokenizer singleton for model-aware token counting.

    Thread-safe with double-checked locking.

    Args:
        encoding_name: Optional encoding name for token counting.

    Returns:
        SmartTokenizer instance.
    """
    key = f'tokenizer_{encoding_name}'
    if key not in _singletons:
        with _lock:
            if key not in _singletons:
                from Jotty.core.utils.tokenizer import SmartTokenizer
                _singletons[key] = SmartTokenizer(encoding_name=encoding_name)
    return _singletons[key]


def list_components() -> Dict[str, str]:
    """
    List all utility components with descriptions.

    Returns:
        Dict mapping component name to description.
    """
    return {
        "BudgetTracker": "Track and limit LLM spending per scope with alerts",
        "CircuitBreaker": "Fail-fast pattern: open/closed/half-open state machine",
        "LLMCallCache": "Semantic caching for LLM calls with TTL and stats",
        "SmartTokenizer": "Model-aware token counting (tiktoken/fallback)",
        "EnhancedLogger": "Structured logging with context requirements",
        "TrajectoryParser": "Parse agent trajectories from raw output",
        "DeadLetterQueue": "Queue for failed operations (retry later)",
        "AdaptiveTimeout": "Timeouts that adapt based on historical latency",
        "StatusReporter": "Safe callback wrapper for status updates",
    }

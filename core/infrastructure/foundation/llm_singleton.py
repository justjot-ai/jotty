"""
LLM Provider Singleton
======================

Single global LLM instance shared across entire application.
Prevents duplicate API calls and rate limit issues.

All components (ValidationGate, DirectChatExecutor, etc.) should use
get_global_lm() instead of creating their own LLM instances.
"""

import logging
import threading
from typing import Optional

from dspy.clients.base_lm import BaseLM

logger = logging.getLogger(__name__)

_global_lm: Optional[BaseLM] = None
_lm_lock = threading.Lock()


def get_global_lm(provider: Optional[str] = None, model: Optional[str] = None, **kwargs) -> BaseLM:
    """
    Get or create global LLM instance.

    All components should use this instead of creating their own.
    First call configures the LLM, subsequent calls return the same instance.

    Args:
        provider: Provider name (anthropic, openai, etc.) - only used on first call
        model: Model name - only used on first call
        **kwargs: Additional config - only used on first call

    Returns:
        Shared LLM instance

    Example:
        # First call (configuration)
        lm = get_global_lm(provider="anthropic", model="claude-haiku-3-5-20241022")

        # Subsequent calls (reuse)
        lm = get_global_lm()  # Returns same instance
    """
    global _global_lm

    with _lm_lock:
        if _global_lm is None:
            from .unified_lm_provider import configure_dspy_lm

            _global_lm = configure_dspy_lm(provider=provider, model=model, **kwargs)
            logger.info(
                f"Global LLM initialized: "
                f"provider={getattr(_global_lm, 'provider', 'unknown')}, "
                f"model={getattr(_global_lm, 'model', 'unknown')}"
            )
        return _global_lm


def reset_global_lm():
    """
    Reset global LM (for testing/reconfiguration).

    Rarely needed - only use if you need to switch providers at runtime.
    """
    global _global_lm
    with _lm_lock:
        _global_lm = None
        logger.info("Global LLM reset")

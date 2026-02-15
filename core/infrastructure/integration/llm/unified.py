"""
Unified LLM Interface

Single interface for all LLM providers with automatic fallback support.
Includes cost tracking integration.
"""

import logging
import os
import time
from typing import Any, List, Optional

from Jotty.core.infrastructure.foundation.config_defaults import (
    DEFAULT_MODEL_ALIAS,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_TIMEOUT_SECONDS,
)

from .providers import (
    PROVIDERS,
    LLMResponse,
    get_provider,
)

logger = logging.getLogger(__name__)

# Optional cost tracking import (if available)
try:
    from ..monitoring.cost_tracker import CostTracker

    COST_TRACKING_AVAILABLE = True
except ImportError:
    COST_TRACKING_AVAILABLE = False
    CostTracker = None


# Default provider order for fallback
DEFAULT_FALLBACK_ORDER = ["claude-cli", "anthropic", "gemini", "openai"]


class UnifiedLLM:
    """
    Unified LLM interface with provider routing and fallback.

    Usage:
        from core.llm import UnifiedLLM

        llm = UnifiedLLM()
        response = llm.generate("What is Python?")

        # Or with specific provider
        response = llm.generate("What is Python?", provider="anthropic")

        # Or with fallback
        response = llm.generate("What is Python?", fallback=True)
    """

    def __init__(
        self,
        default_provider: str = "claude-cli",
        default_model: str = DEFAULT_MODEL_ALIAS,
        fallback_order: Optional[List[str]] = None,
        timeout: int = LLM_TIMEOUT_SECONDS,
        cost_tracker: Optional["CostTracker"] = None,
    ) -> None:
        """
        Initialize UnifiedLLM.

        Args:
            default_provider: Default provider to use
            default_model: Default model name
            fallback_order: Order of providers to try on failure
            timeout: Default timeout in seconds
            cost_tracker: Optional CostTracker instance for cost tracking
        """
        self.default_provider = default_provider
        self.default_model = default_model
        self.fallback_order = fallback_order or DEFAULT_FALLBACK_ORDER
        self.timeout = timeout
        self.cost_tracker = cost_tracker

    def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_tokens: int = LLM_MAX_OUTPUT_TOKENS,
        fallback: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate text using LLM.

        Args:
            prompt: The prompt text
            provider: Provider to use (default: self.default_provider)
            model: Model name (default: self.default_model)
            timeout: Timeout in seconds
            max_tokens: Maximum tokens for API providers
            fallback: If True, try other providers on failure
            **kwargs: Additional provider-specific arguments

        Returns:
            LLMResponse with result or error
        """
        provider = provider or self.default_provider
        model = model or self.default_model
        timeout = timeout or self.timeout

        # Track start time for duration calculation
        start_time = time.time()

        # Try primary provider
        response = self._call_provider(
            provider=provider,
            prompt=prompt,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Track cost if cost tracker is available
        duration = time.time() - start_time
        self._track_cost(response, provider, model, duration)

        # If successful or fallback disabled, return
        if response.success or not fallback:
            return response

        # Try fallback providers
        logger.info(f"Primary provider {provider} failed, trying fallbacks...")

        for fallback_provider in self.fallback_order:
            if fallback_provider == provider:
                continue  # Skip the one that already failed

            logger.info(f"Trying fallback provider: {fallback_provider}")

            fallback_start_time = time.time()
            response = self._call_provider(
                provider=fallback_provider,
                prompt=prompt,
                model=model,
                timeout=timeout,
                max_tokens=max_tokens,
                **kwargs,
            )

            # Track cost for fallback call
            fallback_duration = time.time() - fallback_start_time
            self._track_cost(response, fallback_provider, model, fallback_duration)

            if response.success:
                logger.info(f"Fallback provider {fallback_provider} succeeded")
                return response

        # All providers failed
        return LLMResponse(
            success=False,
            error=f"All providers failed. Last error: {response.error}",
            provider=provider,
            model=model,
        )

    def _track_cost(self, response: LLMResponse, provider: str, model: str, duration: float) -> Any:
        """
        Track cost for an LLM call.

        Args:
            response: LLMResponse from provider
            provider: Provider name
            model: Model name
            duration: Call duration in seconds
        """
        if not self.cost_tracker or not COST_TRACKING_AVAILABLE:
            return

        # Extract token counts from response
        input_tokens = 0
        output_tokens = 0

        if response.usage:
            input_tokens = response.usage.get("input_tokens", 0)
            output_tokens = response.usage.get("output_tokens", 0)

        # Record the call
        self.cost_tracker.record_llm_call(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=response.success,
            error=response.error if not response.success else None,
            duration=duration,
        )

    def _call_provider(
        self, provider: str, prompt: str, model: str, timeout: int, max_tokens: int, **kwargs: Any
    ) -> LLMResponse:
        """Call a specific provider."""
        provider_class = get_provider(provider)

        if not provider_class:
            return LLMResponse(
                success=False,
                error=f"Unknown provider: {provider}. Available: {list(PROVIDERS.keys())}",
                provider=provider,
                model=model,
            )

        return provider_class.generate(
            prompt=prompt, model=model, timeout=timeout, max_tokens=max_tokens, **kwargs
        )

    def is_available(self, provider: str) -> bool:
        """Check if a provider is available (has required credentials)."""
        if provider == "claude-cli":
            # Check if claude CLI exists
            import shutil

            return shutil.which("claude") is not None

        elif provider == "anthropic":
            return bool(os.environ.get("ANTHROPIC_API_KEY"))

        elif provider == "gemini":
            return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))

        elif provider == "openai":
            return bool(os.environ.get("OPENAI_API_KEY"))

        return False

    def list_available_providers(self) -> List[str]:
        """List providers that are currently available."""
        return [p for p in PROVIDERS.keys() if self.is_available(p)]


# Singleton instance for convenience
_default_llm: Optional[UnifiedLLM] = None


def get_llm() -> UnifiedLLM:
    """Get the default UnifiedLLM instance."""
    global _default_llm
    if _default_llm is None:
        _default_llm = UnifiedLLM()
    return _default_llm


def generate(
    prompt: str,
    provider: str = "claude-cli",
    model: str = DEFAULT_MODEL_ALIAS,
    timeout: int = LLM_TIMEOUT_SECONDS,
    fallback: bool = False,
    **kwargs: Any,
) -> LLMResponse:
    """
    Convenience function for quick LLM calls.

    Usage:
        from core.llm import generate

        response = generate("What is Python?")
        if response.success:
            print(response.text)
    """
    return get_llm().generate(
        prompt=prompt, provider=provider, model=model, timeout=timeout, fallback=fallback, **kwargs
    )


def generate_text(
    prompt: str,
    provider: str = "claude-cli",
    model: str = DEFAULT_MODEL_ALIAS,
    timeout: int = LLM_TIMEOUT_SECONDS,
    fallback: bool = False,
    **kwargs: Any,
) -> str:
    """
    Convenience function that returns just the text or raises an error.

    Usage:
        from core.llm import generate_text

        text = generate_text("What is Python?")
    """
    response = generate(
        prompt=prompt, provider=provider, model=model, timeout=timeout, fallback=fallback, **kwargs
    )

    if not response.success:
        raise RuntimeError(f"LLM generation failed: {response.error}")

    return response.text

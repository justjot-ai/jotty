"""
Provider Health Manager
========================

Central per-provider circuit breakers with exponential backoff deferral.

Reuses existing CircuitBreaker from timeouts.py but adds:
- Per-provider health tracking
- Exponential backoff on repeated failures (30s → 60s → 120s → ... → 600s max)
- Automatic recovery: backoff resets on successful half-open test

Design: Deferral, NOT blacklist. Providers always get another chance after timeout.

Usage:
    from Jotty.core.infrastructure.utils.provider_health import get_provider_health

    health = get_provider_health()
    if health.is_healthy("serper"):
        try:
            result = call_serper(query)
            health.record_success("serper")
        except Exception as e:
            health.record_failure("serper", e)
    else:
        # Skip serper, use fallback
        ...
"""

import logging
import threading
import time
from typing import Any, Dict, Optional

from Jotty.core.infrastructure.utils.timeouts import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)

logger = logging.getLogger(__name__)


class ProviderHealthManager:
    """
    Central health tracker for external providers (serper, searxng, openai, etc.).

    Each provider gets its own CircuitBreaker. On circuit open, timeout starts
    at base_timeout and doubles on each re-open (exponential backoff) up to
    max_timeout. On successful recovery from HALF_OPEN, backoff resets.

    Thread-safe singleton via get_provider_health().
    """

    def __init__(
        self,
        base_timeout: float = 30.0,
        max_timeout: float = 600.0,
        failure_threshold: int = 2,
    ) -> None:
        self._base_timeout = base_timeout
        self._max_timeout = max_timeout
        self._failure_threshold = failure_threshold
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._backoff_multipliers: Dict[str, int] = {}
        self._lock = threading.Lock()

    def _get_breaker(self, provider: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a provider. Must be called under lock."""
        if provider not in self._breakers:
            timeout = self._compute_timeout(provider)
            self._breakers[provider] = CircuitBreaker(
                config=CircuitBreakerConfig(
                    name=f"provider:{provider}",
                    failure_threshold=self._failure_threshold,
                    success_threshold=1,
                    timeout=timeout,
                )
            )
            self._backoff_multipliers.setdefault(provider, 0)
        return self._breakers[provider]

    def _compute_timeout(self, provider: str) -> float:
        """Compute current timeout for provider based on backoff multiplier."""
        multiplier = self._backoff_multipliers.get(provider, 0)
        timeout = self._base_timeout * (2**multiplier)
        return min(timeout, self._max_timeout)

    def is_healthy(self, provider: str) -> bool:
        """
        Check if a provider is currently healthy (circuit closed or half-open).

        This is a query-only method — does NOT increment call counts.

        Args:
            provider: Provider name (e.g. "serper", "openai")

        Returns:
            True if provider can be tried, False if deferred
        """
        with self._lock:
            breaker = self._get_breaker(provider)
            if breaker.state == CircuitState.CLOSED:
                return True
            if breaker.state == CircuitState.OPEN:
                # Check if timeout expired → transition to half-open
                if breaker.last_failure_time:
                    elapsed = time.time() - breaker.last_failure_time
                    if elapsed >= breaker.config.timeout:
                        breaker.state = CircuitState.HALF_OPEN
                        breaker.success_count = 0
                        logger.info(
                            f"Provider '{provider}' entering HALF_OPEN "
                            f"(testing recovery after {breaker.config.timeout:.0f}s)"
                        )
                        return True
                return False
            # HALF_OPEN — allow test request
            return True

    def record_success(self, provider: str) -> None:
        """
        Record a successful call to a provider.

        If recovering from HALF_OPEN, resets backoff multiplier to 0.

        Args:
            provider: Provider name
        """
        with self._lock:
            breaker = self._get_breaker(provider)
            was_half_open = breaker.state == CircuitState.HALF_OPEN
            breaker.record_success()
            if was_half_open and breaker.state == CircuitState.CLOSED:
                # Recovery successful — reset exponential backoff
                self._backoff_multipliers[provider] = 0
                breaker.config.timeout = self._base_timeout
                logger.info(
                    f"Provider '{provider}' recovered — backoff reset to {self._base_timeout:.0f}s"
                )

    def record_failure(self, provider: str, error: Optional[Exception] = None) -> None:
        """
        Record a failed call to a provider.

        If circuit opens (or re-opens from half-open), applies exponential backoff.

        Args:
            provider: Provider name
            error: Optional exception that caused the failure
        """
        with self._lock:
            breaker = self._get_breaker(provider)
            was_open = breaker.state == CircuitState.OPEN
            was_half_open = breaker.state == CircuitState.HALF_OPEN
            breaker.record_failure(error or Exception("unknown"))

            if breaker.state == CircuitState.OPEN and (not was_open or was_half_open):
                # Circuit just opened (or re-opened from half-open test failure)
                self._backoff_multipliers[provider] = self._backoff_multipliers.get(provider, 0) + 1
                new_timeout = self._compute_timeout(provider)
                breaker.config.timeout = new_timeout
                # Reset last_failure_time so the new timeout applies from now
                breaker.last_failure_time = time.time()
                logger.warning(
                    f"Provider '{provider}' circuit OPEN — "
                    f"deferred for {new_timeout:.0f}s "
                    f"(backoff level {self._backoff_multipliers[provider]})"
                )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get health stats for all tracked providers.

        Returns:
            Dict mapping provider name to its circuit breaker stats + backoff info
        """
        with self._lock:
            stats: Dict[str, Any] = {}
            for provider, breaker in self._breakers.items():
                breaker_stats = breaker.get_stats()
                breaker_stats["backoff_multiplier"] = self._backoff_multipliers.get(provider, 0)
                breaker_stats["current_timeout"] = breaker.config.timeout
                stats[provider] = breaker_stats
            return stats


# =============================================================================
# Singleton
# =============================================================================

_lock = threading.Lock()
_instance: Optional[ProviderHealthManager] = None


def get_provider_health() -> ProviderHealthManager:
    """Get singleton ProviderHealthManager instance."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ProviderHealthManager()
    return _instance

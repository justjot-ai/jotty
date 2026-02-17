"""
Unit tests for ProviderHealthManager.

Tests exponential backoff deferral, recovery, max timeout cap, and thread safety.
"""

import threading
import time
from unittest.mock import patch

import pytest

from Jotty.core.infrastructure.utils.provider_health import ProviderHealthManager
from Jotty.core.infrastructure.utils.timeouts import CircuitState


@pytest.mark.unit
class TestProviderHealthManager:
    """Tests for ProviderHealthManager."""

    def setup_method(self):
        """Fresh manager for each test."""
        self.manager = ProviderHealthManager(
            base_timeout=30.0, max_timeout=600.0, failure_threshold=2
        )

    def test_healthy_by_default(self):
        """New provider should be healthy."""
        assert self.manager.is_healthy("serper") is True
        assert self.manager.is_healthy("openai") is True
        assert self.manager.is_healthy("unknown_provider") is True

    def test_opens_after_threshold(self):
        """Circuit should open after failure_threshold consecutive failures."""
        self.manager.record_failure("serper", Exception("400 Bad Request"))
        assert self.manager.is_healthy("serper") is True  # 1 failure, threshold is 2

        self.manager.record_failure("serper", Exception("400 Bad Request"))
        assert self.manager.is_healthy("serper") is False  # 2 failures → open

    def test_exponential_backoff(self):
        """Timeout should double on each re-open."""
        # First open: 2 failures → timeout = 30 * 2^1 = 60s
        self.manager.record_failure("serper", Exception("err"))
        self.manager.record_failure("serper", Exception("err"))
        breaker = self.manager._breakers["serper"]
        assert breaker.state == CircuitState.OPEN
        assert breaker.config.timeout == 60.0  # base * 2^1

        # Simulate timeout expiry → half-open
        breaker.last_failure_time = time.time() - 61
        assert self.manager.is_healthy("serper") is True  # transitions to half-open

        # Fail again in half-open → re-opens with timeout = 30 * 2^2 = 120s
        self.manager.record_failure("serper", Exception("still broken"))
        assert breaker.state == CircuitState.OPEN
        assert breaker.config.timeout == 120.0  # base * 2^2

    def test_recovery_resets_backoff(self):
        """Successful recovery from half-open should reset backoff to 0."""
        # Open circuit
        self.manager.record_failure("serper", Exception("err"))
        self.manager.record_failure("serper", Exception("err"))
        breaker = self.manager._breakers["serper"]
        assert breaker.state == CircuitState.OPEN

        # Simulate timeout expiry → half-open
        breaker.last_failure_time = time.time() - 61
        assert self.manager.is_healthy("serper") is True

        # Succeed in half-open → should close circuit and reset backoff
        self.manager.record_success("serper")
        assert breaker.state == CircuitState.CLOSED
        assert self.manager._backoff_multipliers["serper"] == 0
        assert breaker.config.timeout == 30.0  # back to base

    def test_max_timeout_cap(self):
        """Timeout should never exceed max_timeout (600s)."""
        manager = ProviderHealthManager(base_timeout=30.0, max_timeout=600.0, failure_threshold=2)

        # Simulate many open/half-open/fail cycles to drive backoff high
        for cycle in range(10):
            manager.record_failure("test", Exception("err"))
            manager.record_failure("test", Exception("err"))
            breaker = manager._breakers["test"]
            # Force half-open for next cycle
            breaker.last_failure_time = time.time() - 1000
            manager.is_healthy("test")  # transitions to half-open

        # Verify timeout is capped
        breaker = manager._breakers["test"]
        assert breaker.config.timeout <= 600.0

    def test_thread_safety(self):
        """Concurrent access from multiple threads should not corrupt state."""
        errors = []

        def stress_test(provider_id: int):
            try:
                name = f"provider_{provider_id % 3}"
                for _ in range(50):
                    self.manager.is_healthy(name)
                    self.manager.record_failure(name, Exception("test"))
                    self.manager.record_success(name)
                    self.manager.get_stats()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stress_test, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_get_stats(self):
        """get_stats should return info for all tracked providers."""
        self.manager.is_healthy("serper")
        self.manager.record_failure("openai", Exception("rate limit"))

        stats = self.manager.get_stats()
        assert "serper" in stats
        assert "openai" in stats
        assert stats["openai"]["backoff_multiplier"] >= 0
        assert "current_timeout" in stats["openai"]

    def test_independent_providers(self):
        """Failures on one provider should not affect another."""
        self.manager.record_failure("serper", Exception("err"))
        self.manager.record_failure("serper", Exception("err"))

        assert self.manager.is_healthy("serper") is False
        assert self.manager.is_healthy("openai") is True

    def test_success_while_closed(self):
        """Recording success while closed should reduce failure count."""
        self.manager.record_failure("serper", Exception("err"))
        self.manager.record_success("serper")
        # Still need 2 consecutive failures to open
        self.manager.record_failure("serper", Exception("err"))
        assert self.manager.is_healthy("serper") is True  # Should still be healthy

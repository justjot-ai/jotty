"""
Integration tests for provider deferral.

Tests the end-to-end flow: search_web_tool → ProviderHealthManager → fallback.
All mocked, no real API calls.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from Jotty.core.infrastructure.utils.provider_health import ProviderHealthManager


@pytest.mark.integration
class TestProviderDeferral:
    """Integration tests for search provider deferral."""

    def setup_method(self):
        """Fresh health manager for each test."""
        self.health = ProviderHealthManager(
            base_timeout=30.0, max_timeout=600.0, failure_threshold=2
        )

    def test_search_skips_deferred_serper(self):
        """When Serper circuit is open, search should skip to next provider."""
        # Open Serper circuit
        self.health.record_failure("serper", Exception("400 Bad Request"))
        self.health.record_failure("serper", Exception("400 Bad Request"))
        assert self.health.is_healthy("serper") is False

        # Verify the health manager state is correct
        stats = self.health.get_stats()
        assert stats["serper"]["state"] == "open"
        assert stats["serper"]["backoff_multiplier"] >= 1

        # Verify is_healthy returns False — search_web_tool checks this
        # before attempting Serper, so it will skip to DDG
        assert self.health.is_healthy("serper") is False
        assert self.health.is_healthy("duckduckgo") is True  # Fallback healthy

    def test_search_retries_after_backoff(self):
        """After backoff timeout expires, provider should be re-tested."""
        # Open circuit
        self.health.record_failure("serper", Exception("400"))
        self.health.record_failure("serper", Exception("400"))
        assert self.health.is_healthy("serper") is False

        # Simulate time passing beyond the backoff timeout
        breaker = self.health._breakers["serper"]
        breaker.last_failure_time = time.time() - (breaker.config.timeout + 1)

        # Should now be half-open (allowing a test request)
        assert self.health.is_healthy("serper") is True

        # Successful test request → circuit closes, backoff resets
        self.health.record_success("serper")
        assert self.health.is_healthy("serper") is True
        assert self.health._backoff_multipliers["serper"] == 0

    def test_full_deferral_cycle(self):
        """Full cycle: healthy → fail → defer → timeout → half-open → recover."""
        provider = "test_api"

        # 1. Start healthy
        assert self.health.is_healthy(provider) is True

        # 2. Fail twice → circuit opens
        self.health.record_failure(provider, Exception("500"))
        self.health.record_failure(provider, Exception("500"))
        assert self.health.is_healthy(provider) is False

        # 3. Check backoff is applied
        breaker = self.health._breakers[provider]
        assert breaker.config.timeout == 60.0  # 30 * 2^1

        # 4. Simulate timeout expiry
        breaker.last_failure_time = time.time() - 61
        assert self.health.is_healthy(provider) is True  # half-open

        # 5. Fail again in half-open → re-opens with higher backoff
        self.health.record_failure(provider, Exception("still broken"))
        assert self.health.is_healthy(provider) is False
        assert breaker.config.timeout == 120.0  # 30 * 2^2

        # 6. Simulate another timeout expiry
        breaker.last_failure_time = time.time() - 121
        assert self.health.is_healthy(provider) is True  # half-open again

        # 7. Succeed → circuit closes, backoff resets
        self.health.record_success(provider)
        assert self.health.is_healthy(provider) is True
        assert self.health._backoff_multipliers[provider] == 0
        assert breaker.config.timeout == 30.0  # back to base

    def test_guard_integration_with_health(self):
        """ToolExecutionGuard should record provider health from tool results."""
        from Jotty.core.infrastructure.integration.tool_execution_guard import ToolExecutionGuard

        guard = ToolExecutionGuard()

        def search_tool(query: str) -> dict:
            return {"success": True, "results": ["r1"], "provider": "serper"}

        wrapped = guard.wrap_skill_tools("web-search", {"search": search_tool})
        wrapped["search"](query="test")

        # Verify the health manager recorded the success
        stats = guard.health_manager.get_stats()
        assert "serper" in stats

    def test_guard_records_failure_opens_circuit(self):
        """ToolExecutionGuard failures should open provider circuit."""
        from Jotty.core.infrastructure.integration.tool_execution_guard import ToolExecutionGuard

        guard = ToolExecutionGuard()
        # Override health manager with low threshold for testing
        guard._health = ProviderHealthManager(
            base_timeout=30.0, max_timeout=600.0, failure_threshold=2
        )

        def failing_search(query: str) -> dict:
            raise ConnectionError("Serper 400")

        wrapped = guard.wrap_skill_tools("serper-search", {"search": failing_search})

        # Fail twice to open circuit
        for _ in range(2):
            try:
                wrapped["search"](query="test")
            except ConnectionError:
                pass

        assert guard.health_manager.is_healthy("serper") is False

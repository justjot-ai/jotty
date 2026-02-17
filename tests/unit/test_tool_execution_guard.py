"""
Unit tests for ToolExecutionGuard.

Tests wrapping behavior, interceptor recording, provider inference, and async support.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from Jotty.core.infrastructure.integration.tool_execution_guard import (
    ToolExecutionGuard,
    _infer_provider,
)


@pytest.mark.unit
class TestToolExecutionGuard:
    """Tests for ToolExecutionGuard."""

    def setup_method(self):
        """Fresh guard for each test."""
        self.guard = ToolExecutionGuard()

    def test_wrap_preserves_behavior(self):
        """Wrapped tool should return the same result as the original."""

        def my_tool(query: str) -> dict:
            return {"success": True, "results": [query], "provider": "test"}

        wrapped = self.guard.wrap_skill_tools("test-skill", {"my_tool": my_tool})
        result = wrapped["my_tool"](query="hello")

        assert result["success"] is True
        assert result["results"] == ["hello"]

    def test_records_success_in_interceptor(self):
        """Successful tool call should be tracked in the interceptor registry."""

        def my_tool(x: int) -> dict:
            return {"success": True, "value": x * 2}

        wrapped = self.guard.wrap_skill_tools("math-skill", {"my_tool": my_tool})
        wrapped["my_tool"](x=5)

        all_calls = self.guard.interceptor_registry.get_all_calls()
        assert len(all_calls) == 1
        assert all_calls[0].tool_name == "my_tool"
        assert all_calls[0].success is True
        assert all_calls[0].metadata["skill"] == "math-skill"

    def test_records_failure_in_interceptor(self):
        """Failed tool call should be tracked with error info."""

        def failing_tool() -> dict:
            raise ValueError("something broke")

        wrapped = self.guard.wrap_skill_tools("bad-skill", {"failing_tool": failing_tool})

        with pytest.raises(ValueError, match="something broke"):
            wrapped["failing_tool"]()

        all_calls = self.guard.interceptor_registry.get_all_calls()
        assert len(all_calls) == 1
        assert all_calls[0].success is False
        assert "something broke" in all_calls[0].error

    def test_infers_provider_from_result(self):
        """Provider should be inferred from result dict's 'provider' key."""

        def search_tool(query: str) -> dict:
            return {"success": True, "results": [], "provider": "serper"}

        wrapped = self.guard.wrap_skill_tools("web-search", {"search": search_tool})

        with patch.object(self.guard.health_manager, "record_success") as mock_success:
            wrapped["search"](query="test")
            mock_success.assert_called_once_with("serper")

    def test_records_provider_health_failure(self):
        """Provider failure should be forwarded to ProviderHealthManager."""

        def openai_tool() -> dict:
            raise ConnectionError("rate limited")

        wrapped = self.guard.wrap_skill_tools("openai-chat", {"generate": openai_tool})

        with patch.object(self.guard.health_manager, "record_failure") as mock_fail:
            with pytest.raises(ConnectionError):
                wrapped["generate"]()
            mock_fail.assert_called_once()
            assert mock_fail.call_args[0][0] == "openai"

    @pytest.mark.asyncio
    async def test_wraps_async_tools(self):
        """Async tool wrapping should work correctly."""

        async def async_tool(query: str) -> dict:
            return {"success": True, "data": query, "provider": "test"}

        wrapped = self.guard.wrap_skill_tools("async-skill", {"async_tool": async_tool})
        result = await wrapped["async_tool"](query="hello")

        assert result["success"] is True
        assert result["data"] == "hello"

        all_calls = self.guard.interceptor_registry.get_all_calls()
        assert len(all_calls) == 1
        assert all_calls[0].success is True

    def test_result_based_failure_detection(self):
        """Tool returning success=False in dict should be recorded as failure."""

        def soft_fail_tool() -> dict:
            return {"success": False, "error": "not found"}

        wrapped = self.guard.wrap_skill_tools("test-skill", {"tool": soft_fail_tool})
        result = wrapped["tool"]()

        assert result["success"] is False
        all_calls = self.guard.interceptor_registry.get_all_calls()
        assert len(all_calls) == 1
        assert all_calls[0].success is False

    def test_multiple_tools_wrapped(self):
        """Multiple tools in a skill should all be wrapped."""

        def tool_a() -> dict:
            return {"success": True}

        def tool_b() -> dict:
            return {"success": True}

        wrapped = self.guard.wrap_skill_tools("multi-skill", {"tool_a": tool_a, "tool_b": tool_b})

        wrapped["tool_a"]()
        wrapped["tool_b"]()

        all_calls = self.guard.interceptor_registry.get_all_calls()
        assert len(all_calls) == 2


@pytest.mark.unit
class TestInferProvider:
    """Tests for _infer_provider helper."""

    def test_from_result_dict(self):
        assert _infer_provider("web-search", {"provider": "serper"}) == "serper"
        assert _infer_provider("web-search", {"provider": "duckduckgo"}) == "duckduckgo"

    def test_from_skill_name(self):
        assert _infer_provider("openai-chat", {}) == "openai"
        assert _infer_provider("claude-api-llm", {}) == "anthropic"
        assert _infer_provider("gemini", {}) == "google"
        assert _infer_provider("groq-fast", {}) == "groq"

    def test_no_match(self):
        assert _infer_provider("file-operations", {}) is None
        assert _infer_provider("calculator", {"result": 42}) is None

    def test_result_takes_priority(self):
        """Result dict provider should override skill name inference."""
        assert _infer_provider("openai-chat", {"provider": "azure"}) == "azure"

"""Tests for BaseAgent sentinel pattern — injected resources are respected."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from Jotty.core.intelligence.reasoning.base.base_agent import (
    AgentRuntimeConfig,
    BaseAgent,
)


class ConcreteAgent(BaseAgent):
    """Minimal concrete agent for testing."""

    async def _execute_impl(self, **kwargs):
        return {"ok": True}


class TestSentinelPattern:
    """Verify that injected resources are not overwritten by lazy loading."""

    @pytest.mark.unit
    def test_memory_sentinel_default_is_unset(self) -> None:
        """New agents start with _UNSET sentinel, not None."""
        agent = ConcreteAgent(AgentRuntimeConfig(name="test", enable_memory=False))
        # The sentinel is in place before any property access
        assert agent._memory is BaseAgent._UNSET

    @pytest.mark.unit
    def test_injected_memory_is_respected(self) -> None:
        """When a swarm injects memory, the lazy loader does NOT overwrite it."""
        agent = ConcreteAgent(AgentRuntimeConfig(name="test", enable_memory=True))
        shared_memory = MagicMock(name="SharedMemory")
        agent.memory = shared_memory  # Swarm injects

        # Property access should return the injected instance
        assert agent.memory is shared_memory
        # Internal field should hold the injected value
        assert agent._memory is shared_memory

    @pytest.mark.unit
    def test_injected_none_memory_is_respected(self) -> None:
        """Explicit None injection (from swarm with no memory) is respected."""
        agent = ConcreteAgent(AgentRuntimeConfig(name="test", enable_memory=True))
        agent.memory = None  # Swarm explicitly sets None

        # Should return None without trying to lazy-create
        assert agent.memory is None

    @pytest.mark.unit
    def test_injected_context_is_respected(self) -> None:
        """Injected context manager is not overwritten."""
        agent = ConcreteAgent(AgentRuntimeConfig(name="test", enable_context=True))
        shared_ctx = MagicMock(name="SharedContext")
        agent.context = shared_ctx

        assert agent.context is shared_ctx

    @pytest.mark.unit
    def test_injected_skills_registry_is_respected(self) -> None:
        """Injected skills registry is not overwritten."""
        agent = ConcreteAgent(AgentRuntimeConfig(name="test", enable_skills=True))
        shared_reg = MagicMock(name="SharedRegistry")
        agent.skills_registry = shared_reg

        assert agent.skills_registry is shared_reg

    @pytest.mark.unit
    def test_disabled_memory_returns_none(self) -> None:
        """When enable_memory=False, memory property returns None."""
        agent = ConcreteAgent(AgentRuntimeConfig(name="test", enable_memory=False))
        assert agent.memory is None

    @pytest.mark.unit
    def test_disabled_context_returns_none(self) -> None:
        """When enable_context=False, context property returns None."""
        agent = ConcreteAgent(AgentRuntimeConfig(name="test", enable_context=False))
        assert agent.context is None

    @pytest.mark.unit
    def test_disabled_skills_returns_none(self) -> None:
        """When enable_skills=False, skills_registry property returns None."""
        agent = ConcreteAgent(AgentRuntimeConfig(name="test", enable_skills=False))
        assert agent.skills_registry is None

    @pytest.mark.unit
    def test_multiple_property_accesses_return_same_instance(self) -> None:
        """Injected resource is returned consistently across accesses."""
        agent = ConcreteAgent(AgentRuntimeConfig(name="test", enable_memory=True))
        shared_memory = MagicMock(name="SharedMemory")
        agent.memory = shared_memory

        # Access multiple times
        assert agent.memory is shared_memory
        assert agent.memory is shared_memory
        assert agent.memory is shared_memory

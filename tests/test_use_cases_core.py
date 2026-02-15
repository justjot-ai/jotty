"""
Tests for core/use_cases/base.py
=================================
Covers: UseCaseType, UseCaseConfig, UseCaseResult, BaseUseCase.
"""

import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from Jotty.core.interface.use_cases.base import (
    BaseUseCase,
    UseCaseConfig,
    UseCaseResult,
    UseCaseType,
)

# ===========================================================================
# UseCaseType Enum
# ===========================================================================


@pytest.mark.unit
class TestUseCaseType:
    """Tests for UseCaseType enum."""

    def test_chat_value(self):
        assert UseCaseType.CHAT.value == "chat"

    def test_workflow_value(self):
        assert UseCaseType.WORKFLOW.value == "workflow"

    def test_lookup_by_value(self):
        assert UseCaseType("chat") is UseCaseType.CHAT
        assert UseCaseType("workflow") is UseCaseType.WORKFLOW

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            UseCaseType("invalid")


# ===========================================================================
# UseCaseConfig Dataclass
# ===========================================================================


@pytest.mark.unit
class TestUseCaseConfig:
    """Tests for UseCaseConfig dataclass."""

    def test_defaults(self):
        """UseCaseConfig has sensible defaults."""
        config = UseCaseConfig(use_case_type=UseCaseType.CHAT)
        assert config.use_case_type == UseCaseType.CHAT
        assert config.max_iterations == 100
        assert config.enable_learning is True
        assert config.enable_memory is True
        assert config.enable_streaming is True
        assert config.timeout is None
        assert config.metadata is None

    def test_custom_values(self):
        """UseCaseConfig stores custom values."""
        config = UseCaseConfig(
            use_case_type=UseCaseType.WORKFLOW,
            max_iterations=50,
            enable_learning=False,
            enable_memory=False,
            enable_streaming=False,
            timeout=30.0,
            metadata={"key": "val"},
        )
        assert config.use_case_type == UseCaseType.WORKFLOW
        assert config.max_iterations == 50
        assert config.enable_learning is False
        assert config.timeout == 30.0
        assert config.metadata == {"key": "val"}


# ===========================================================================
# UseCaseResult Dataclass
# ===========================================================================


@pytest.mark.unit
class TestUseCaseResult:
    """Tests for UseCaseResult dataclass."""

    def test_success_result(self):
        """UseCaseResult stores success result."""
        result = UseCaseResult(
            success=True,
            output="Hello",
            metadata={"tokens": 10},
            execution_time=1.5,
            use_case_type=UseCaseType.CHAT,
        )
        assert result.success is True
        assert result.output == "Hello"
        assert result.execution_time == 1.5

    def test_to_dict_string_output(self):
        """to_dict converts non-dict output to string."""
        result = UseCaseResult(
            success=True,
            output=42,
            metadata={},
            execution_time=0.5,
            use_case_type=UseCaseType.CHAT,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["output"] == "42"
        assert d["use_case_type"] == "chat"

    def test_to_dict_dict_output(self):
        """to_dict preserves dict output."""
        result = UseCaseResult(
            success=True,
            output={"key": "val"},
            metadata={"m": 1},
            execution_time=1.0,
            use_case_type=UseCaseType.WORKFLOW,
        )
        d = result.to_dict()
        assert d["output"] == {"key": "val"}
        assert d["metadata"] == {"m": 1}
        assert d["use_case_type"] == "workflow"

    def test_to_dict_list_output(self):
        """to_dict preserves list output."""
        result = UseCaseResult(
            success=True,
            output=[1, 2, 3],
            metadata={},
            execution_time=0.1,
            use_case_type=UseCaseType.CHAT,
        )
        d = result.to_dict()
        assert d["output"] == [1, 2, 3]

    def test_failure_result(self):
        """UseCaseResult stores failure result."""
        result = UseCaseResult(
            success=False,
            output=None,
            metadata={"error": "timeout"},
            execution_time=30.0,
            use_case_type=UseCaseType.WORKFLOW,
        )
        assert result.success is False
        d = result.to_dict()
        assert d["success"] is False
        assert d["output"] == "None"


# ===========================================================================
# BaseUseCase
# ===========================================================================


class _ConcreteChat(BaseUseCase):
    """Concrete implementation for testing."""

    def _get_use_case_type(self) -> UseCaseType:
        return UseCaseType.CHAT

    async def execute(self, goal, context=None, **kwargs):
        return self._create_result(True, f"Executed: {goal}")

    async def stream(self, goal, context=None, **kwargs):
        yield {"type": "token", "data": goal}


class _ConcreteWorkflow(BaseUseCase):
    """Concrete workflow implementation for testing."""

    def _get_use_case_type(self) -> UseCaseType:
        return UseCaseType.WORKFLOW

    async def execute(self, goal, context=None, **kwargs):
        start = time.time()
        return self._create_result(True, "done", execution_time=self._get_execution_time(start))

    async def stream(self, goal, context=None, **kwargs):
        yield {"type": "done"}


@pytest.mark.unit
class TestBaseUseCaseInit:
    """Tests for BaseUseCase initialization."""

    def test_init_with_default_config(self):
        """BaseUseCase creates default config from use case type."""
        conductor = MagicMock()
        uc = _ConcreteChat(conductor)
        assert uc.conductor is conductor
        assert uc.config.use_case_type == UseCaseType.CHAT

    def test_init_with_custom_config(self):
        """BaseUseCase accepts custom config."""
        conductor = MagicMock()
        config = UseCaseConfig(use_case_type=UseCaseType.CHAT, max_iterations=10)
        uc = _ConcreteChat(conductor, config)
        assert uc.config.max_iterations == 10

    def test_init_mismatched_config_raises(self):
        """BaseUseCase raises for mismatched config type."""
        conductor = MagicMock()
        config = UseCaseConfig(use_case_type=UseCaseType.WORKFLOW)
        with pytest.raises(ValueError, match="doesn't match"):
            _ConcreteChat(conductor, config)


@pytest.mark.unit
class TestBaseUseCaseExecute:
    """Tests for BaseUseCase execute and stream."""

    @pytest.mark.asyncio
    async def test_execute(self):
        """execute() returns UseCaseResult."""
        conductor = MagicMock()
        uc = _ConcreteChat(conductor)
        result = await uc.execute("test goal")
        assert isinstance(result, UseCaseResult)
        assert result.success is True
        assert "test goal" in result.output

    @pytest.mark.asyncio
    async def test_stream(self):
        """stream() yields event dicts."""
        conductor = MagicMock()
        uc = _ConcreteChat(conductor)
        events = []
        async for event in uc.stream("hello"):
            events.append(event)
        assert len(events) == 1
        assert events[0]["data"] == "hello"


@pytest.mark.unit
class TestBaseUseCaseEnqueue:
    """Tests for BaseUseCase.enqueue() default."""

    @pytest.mark.asyncio
    async def test_enqueue_raises_not_implemented(self):
        """Default enqueue raises NotImplementedError."""
        conductor = MagicMock()
        uc = _ConcreteChat(conductor)
        with pytest.raises(NotImplementedError, match="does not support"):
            await uc.enqueue("some goal")


@pytest.mark.unit
class TestBaseUseCaseHelpers:
    """Tests for _create_result and _get_execution_time."""

    def test_create_result_defaults(self):
        """_create_result sets type and defaults."""
        conductor = MagicMock()
        uc = _ConcreteChat(conductor)
        result = uc._create_result(True, "output")
        assert result.success is True
        assert result.output == "output"
        assert result.metadata == {}
        assert result.execution_time == 0.0
        assert result.use_case_type == UseCaseType.CHAT

    def test_create_result_with_metadata(self):
        """_create_result accepts metadata and execution_time."""
        conductor = MagicMock()
        uc = _ConcreteChat(conductor)
        result = uc._create_result(False, None, metadata={"err": "x"}, execution_time=5.0)
        assert result.metadata == {"err": "x"}
        assert result.execution_time == 5.0

    def test_get_execution_time(self):
        """_get_execution_time computes time delta."""
        conductor = MagicMock()
        uc = _ConcreteChat(conductor)
        start = time.time() - 1.0
        elapsed = uc._get_execution_time(start)
        assert elapsed >= 1.0
        assert elapsed < 2.0

    @pytest.mark.asyncio
    async def test_workflow_execution_time(self):
        """Workflow execute() records execution time."""
        conductor = MagicMock()
        uc = _ConcreteWorkflow(conductor)
        result = await uc.execute("task")
        assert result.execution_time >= 0
        assert result.use_case_type == UseCaseType.WORKFLOW

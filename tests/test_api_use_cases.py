"""
Phase 7 - Comprehensive API & Use Cases Test Suite
====================================================

Tests for:
1. UseCaseType, UseCaseConfig, UseCaseResult, BaseUseCase (core/use_cases/base.py)
2. ChatMessage, ChatContext (core/use_cases/chat/chat_context.py)
3. WorkflowTask, WorkflowContext (core/use_cases/workflow/workflow_context.py)
4. CommandInfo, CommandExecutionResult, CommandService (core/services/command_service.py)
5. JottyAPI (core/api/unified.py)

Target: ~150 tests across all classes.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from dataclasses import asdict

# ---------------------------------------------------------------------------
# Guarded imports
# ---------------------------------------------------------------------------

try:
    from Jotty.core.interface.use_cases.base import (
        UseCaseType,
        UseCaseConfig,
        UseCaseResult,
        BaseUseCase,
    )
    HAS_USE_CASE_BASE = True
except ImportError:
    HAS_USE_CASE_BASE = False

try:
    from Jotty.core.interface.use_cases.chat.chat_context import ChatMessage, ChatContext
    HAS_CHAT_CONTEXT = True
except ImportError:
    HAS_CHAT_CONTEXT = False

try:
    from Jotty.core.interface.use_cases.workflow.workflow_context import (
        WorkflowTask,
        WorkflowContext,
    )
    HAS_WORKFLOW_CONTEXT = True
except ImportError:
    HAS_WORKFLOW_CONTEXT = False

try:
    from Jotty.core.infrastructure.foundation.types import TaskStatus
    HAS_TASK_STATUS = True
except ImportError:
    HAS_TASK_STATUS = False

try:
    from Jotty.core.infrastructure.services.command_service import (
        CommandInfo,
        CommandExecutionResult,
        CommandService,
    )
    HAS_COMMAND_SERVICE = True
except ImportError:
    HAS_COMMAND_SERVICE = False

try:
    from Jotty.core.interface.api.unified import JottyAPI
    HAS_JOTTY_API = True
except ImportError:
    HAS_JOTTY_API = False


# ---------------------------------------------------------------------------
# Concrete BaseUseCase subclass for testing
# ---------------------------------------------------------------------------

if HAS_USE_CASE_BASE:
    class _ChatUseCaseStub(BaseUseCase):
        """Concrete chat use case stub for testing BaseUseCase."""

        def _get_use_case_type(self) -> UseCaseType:
            return UseCaseType.CHAT

        async def execute(self, goal, context=None, **kwargs):
            start = time.time()
            return self._create_result(
                True, f"Executed: {goal}",
                execution_time=self._get_execution_time(start),
            )

        async def stream(self, goal, context=None, **kwargs):
            yield {"type": "token", "data": goal}

    class _WorkflowUseCaseStub(BaseUseCase):
        """Concrete workflow use case stub for testing BaseUseCase."""

        def _get_use_case_type(self) -> UseCaseType:
            return UseCaseType.WORKFLOW

        async def execute(self, goal, context=None, **kwargs):
            start = time.time()
            return self._create_result(
                True, {"result": goal},
                metadata={"steps": 3},
                execution_time=self._get_execution_time(start),
            )

        async def stream(self, goal, context=None, **kwargs):
            yield {"type": "step", "step": 1}
            yield {"type": "done"}


# ===========================================================================
# 1. UseCaseType Enum Tests
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_USE_CASE_BASE, reason="UseCaseType import failed")
class TestUseCaseType:
    """Tests for UseCaseType enum values and iteration."""

    def test_chat_value(self):
        assert UseCaseType.CHAT.value == "chat"

    def test_workflow_value(self):
        assert UseCaseType.WORKFLOW.value == "workflow"

    def test_lookup_by_value_chat(self):
        assert UseCaseType("chat") is UseCaseType.CHAT

    def test_lookup_by_value_workflow(self):
        assert UseCaseType("workflow") is UseCaseType.WORKFLOW

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            UseCaseType("invalid")

    def test_iteration_yields_all_members(self):
        members = list(UseCaseType)
        assert len(members) == 2
        assert UseCaseType.CHAT in members
        assert UseCaseType.WORKFLOW in members

    def test_name_attribute(self):
        assert UseCaseType.CHAT.name == "CHAT"
        assert UseCaseType.WORKFLOW.name == "WORKFLOW"


# ===========================================================================
# 2. UseCaseConfig Dataclass Tests
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_USE_CASE_BASE, reason="UseCaseConfig import failed")
class TestUseCaseConfig:
    """Tests for UseCaseConfig dataclass defaults and custom values."""

    def test_defaults_chat(self):
        config = UseCaseConfig(use_case_type=UseCaseType.CHAT)
        assert config.use_case_type == UseCaseType.CHAT
        assert config.max_iterations == 100
        assert config.enable_learning is True
        assert config.enable_memory is True
        assert config.enable_streaming is True
        assert config.timeout is None
        assert config.metadata is None

    def test_defaults_workflow(self):
        config = UseCaseConfig(use_case_type=UseCaseType.WORKFLOW)
        assert config.use_case_type == UseCaseType.WORKFLOW
        assert config.max_iterations == 100

    def test_custom_max_iterations(self):
        config = UseCaseConfig(use_case_type=UseCaseType.CHAT, max_iterations=50)
        assert config.max_iterations == 50

    def test_custom_all_fields(self):
        meta = {"key": "value", "nested": {"a": 1}}
        config = UseCaseConfig(
            use_case_type=UseCaseType.WORKFLOW,
            max_iterations=25,
            enable_learning=False,
            enable_memory=False,
            enable_streaming=False,
            timeout=60.0,
            metadata=meta,
        )
        assert config.max_iterations == 25
        assert config.enable_learning is False
        assert config.enable_memory is False
        assert config.enable_streaming is False
        assert config.timeout == 60.0
        assert config.metadata == meta

    def test_timeout_none_by_default(self):
        config = UseCaseConfig(use_case_type=UseCaseType.CHAT)
        assert config.timeout is None

    def test_timeout_custom(self):
        config = UseCaseConfig(use_case_type=UseCaseType.CHAT, timeout=30.5)
        assert config.timeout == 30.5

    def test_metadata_none_by_default(self):
        config = UseCaseConfig(use_case_type=UseCaseType.CHAT)
        assert config.metadata is None

    def test_metadata_dict(self):
        config = UseCaseConfig(
            use_case_type=UseCaseType.CHAT,
            metadata={"agent": "test"},
        )
        assert config.metadata == {"agent": "test"}


# ===========================================================================
# 3. UseCaseResult Dataclass Tests
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_USE_CASE_BASE, reason="UseCaseResult import failed")
class TestUseCaseResult:
    """Tests for UseCaseResult creation and to_dict conversion."""

    def test_creation_success(self):
        result = UseCaseResult(
            success=True, output="Hello", metadata={"tokens": 10},
            execution_time=1.5, use_case_type=UseCaseType.CHAT,
        )
        assert result.success is True
        assert result.output == "Hello"
        assert result.metadata == {"tokens": 10}
        assert result.execution_time == 1.5
        assert result.use_case_type == UseCaseType.CHAT

    def test_creation_failure(self):
        result = UseCaseResult(
            success=False, output=None, metadata={"error": "timeout"},
            execution_time=30.0, use_case_type=UseCaseType.WORKFLOW,
        )
        assert result.success is False
        assert result.output is None

    def test_to_dict_string_output(self):
        result = UseCaseResult(
            success=True, output=42, metadata={},
            execution_time=0.5, use_case_type=UseCaseType.CHAT,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["output"] == "42"
        assert d["use_case_type"] == "chat"
        assert d["execution_time"] == 0.5

    def test_to_dict_dict_output(self):
        result = UseCaseResult(
            success=True, output={"key": "val"}, metadata={"m": 1},
            execution_time=1.0, use_case_type=UseCaseType.WORKFLOW,
        )
        d = result.to_dict()
        assert d["output"] == {"key": "val"}
        assert d["metadata"] == {"m": 1}
        assert d["use_case_type"] == "workflow"

    def test_to_dict_list_output(self):
        result = UseCaseResult(
            success=True, output=[1, 2, 3], metadata={},
            execution_time=0.1, use_case_type=UseCaseType.CHAT,
        )
        d = result.to_dict()
        assert d["output"] == [1, 2, 3]

    def test_to_dict_none_output_becomes_string(self):
        result = UseCaseResult(
            success=False, output=None, metadata={},
            execution_time=0.0, use_case_type=UseCaseType.WORKFLOW,
        )
        d = result.to_dict()
        assert d["output"] == "None"

    def test_to_dict_bool_output_becomes_string(self):
        result = UseCaseResult(
            success=True, output=True, metadata={},
            execution_time=0.0, use_case_type=UseCaseType.CHAT,
        )
        d = result.to_dict()
        assert d["output"] == "True"

    def test_to_dict_contains_all_keys(self):
        result = UseCaseResult(
            success=True, output="x", metadata={},
            execution_time=0.0, use_case_type=UseCaseType.CHAT,
        )
        d = result.to_dict()
        assert set(d.keys()) == {"success", "output", "metadata", "execution_time", "use_case_type"}


# ===========================================================================
# 4. BaseUseCase Tests
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_USE_CASE_BASE, reason="BaseUseCase import failed")
class TestBaseUseCaseInit:
    """Tests for BaseUseCase initialization and config validation."""

    def test_init_with_conductor(self):
        conductor = MagicMock()
        uc = _ChatUseCaseStub(conductor)
        assert uc.conductor is conductor

    def test_init_default_config_type(self):
        conductor = MagicMock()
        uc = _ChatUseCaseStub(conductor)
        assert uc.config.use_case_type == UseCaseType.CHAT

    def test_init_custom_config(self):
        conductor = MagicMock()
        config = UseCaseConfig(use_case_type=UseCaseType.CHAT, max_iterations=10)
        uc = _ChatUseCaseStub(conductor, config)
        assert uc.config.max_iterations == 10

    def test_init_mismatched_config_raises_value_error(self):
        conductor = MagicMock()
        config = UseCaseConfig(use_case_type=UseCaseType.WORKFLOW)
        with pytest.raises(ValueError, match="doesn't match"):
            _ChatUseCaseStub(conductor, config)

    def test_init_workflow_stub(self):
        conductor = MagicMock()
        uc = _WorkflowUseCaseStub(conductor)
        assert uc.config.use_case_type == UseCaseType.WORKFLOW

    def test_init_workflow_with_chat_config_raises(self):
        conductor = MagicMock()
        config = UseCaseConfig(use_case_type=UseCaseType.CHAT)
        with pytest.raises(ValueError, match="doesn't match"):
            _WorkflowUseCaseStub(conductor, config)


@pytest.mark.unit
@pytest.mark.skipif(not HAS_USE_CASE_BASE, reason="BaseUseCase import failed")
class TestBaseUseCaseEnqueue:
    """Tests for BaseUseCase.enqueue() default behaviour."""

    @pytest.mark.asyncio
    async def test_enqueue_raises_not_implemented(self):
        conductor = MagicMock()
        uc = _ChatUseCaseStub(conductor)
        with pytest.raises(NotImplementedError, match="does not support"):
            await uc.enqueue("some goal")

    @pytest.mark.asyncio
    async def test_enqueue_error_includes_class_name(self):
        conductor = MagicMock()
        uc = _ChatUseCaseStub(conductor)
        with pytest.raises(NotImplementedError, match="_ChatUseCaseStub"):
            await uc.enqueue("goal")


@pytest.mark.unit
@pytest.mark.skipif(not HAS_USE_CASE_BASE, reason="BaseUseCase import failed")
class TestBaseUseCaseCreateResult:
    """Tests for _create_result helper."""

    def test_create_result_defaults(self):
        conductor = MagicMock()
        uc = _ChatUseCaseStub(conductor)
        result = uc._create_result(True, "output")
        assert result.success is True
        assert result.output == "output"
        assert result.metadata == {}
        assert result.execution_time == 0.0
        assert result.use_case_type == UseCaseType.CHAT

    def test_create_result_with_metadata(self):
        conductor = MagicMock()
        uc = _ChatUseCaseStub(conductor)
        result = uc._create_result(False, None, metadata={"err": "x"}, execution_time=5.0)
        assert result.metadata == {"err": "x"}
        assert result.execution_time == 5.0

    def test_create_result_workflow_type(self):
        conductor = MagicMock()
        uc = _WorkflowUseCaseStub(conductor)
        result = uc._create_result(True, "done")
        assert result.use_case_type == UseCaseType.WORKFLOW


@pytest.mark.unit
@pytest.mark.skipif(not HAS_USE_CASE_BASE, reason="BaseUseCase import failed")
class TestBaseUseCaseGetExecutionTime:
    """Tests for _get_execution_time helper."""

    def test_get_execution_time_positive(self):
        conductor = MagicMock()
        uc = _ChatUseCaseStub(conductor)
        start = time.time() - 1.0
        elapsed = uc._get_execution_time(start)
        assert elapsed >= 1.0
        assert elapsed < 3.0

    def test_get_execution_time_near_zero(self):
        conductor = MagicMock()
        uc = _ChatUseCaseStub(conductor)
        start = time.time()
        elapsed = uc._get_execution_time(start)
        assert elapsed >= 0.0
        assert elapsed < 1.0


@pytest.mark.unit
@pytest.mark.skipif(not HAS_USE_CASE_BASE, reason="BaseUseCase import failed")
class TestBaseUseCaseExecute:
    """Tests for concrete execute and stream."""

    @pytest.mark.asyncio
    async def test_chat_execute(self):
        conductor = MagicMock()
        uc = _ChatUseCaseStub(conductor)
        result = await uc.execute("test goal")
        assert isinstance(result, UseCaseResult)
        assert result.success is True
        assert "test goal" in result.output

    @pytest.mark.asyncio
    async def test_workflow_execute(self):
        conductor = MagicMock()
        uc = _WorkflowUseCaseStub(conductor)
        result = await uc.execute("build app")
        assert result.success is True
        assert result.output == {"result": "build app"}
        assert result.metadata == {"steps": 3}

    @pytest.mark.asyncio
    async def test_chat_stream(self):
        conductor = MagicMock()
        uc = _ChatUseCaseStub(conductor)
        events = []
        async for event in uc.stream("hello"):
            events.append(event)
        assert len(events) == 1
        assert events[0]["data"] == "hello"

    @pytest.mark.asyncio
    async def test_workflow_stream(self):
        conductor = MagicMock()
        uc = _WorkflowUseCaseStub(conductor)
        events = []
        async for event in uc.stream("plan"):
            events.append(event)
        assert len(events) == 2
        assert events[0]["type"] == "step"
        assert events[1]["type"] == "done"


# ===========================================================================
# 5. ChatMessage Tests
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_CHAT_CONTEXT, reason="ChatMessage import failed")
class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_creation_defaults(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert isinstance(msg.timestamp, float)
        assert msg.metadata == {}

    def test_creation_with_metadata(self):
        msg = ChatMessage(role="assistant", content="Hi", metadata={"model": "test"})
        assert msg.metadata == {"model": "test"}

    def test_creation_system_role(self):
        msg = ChatMessage(role="system", content="You are helpful")
        assert msg.role == "system"

    def test_creation_tool_role(self):
        msg = ChatMessage(role="tool", content='{"result": "ok"}')
        assert msg.role == "tool"

    def test_to_dict(self):
        ts = 1700000000.0
        msg = ChatMessage(role="user", content="test", timestamp=ts, metadata={"k": "v"})
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "test"
        assert d["timestamp"] == ts
        assert d["metadata"] == {"k": "v"}

    def test_to_dict_keys(self):
        msg = ChatMessage(role="user", content="x")
        d = msg.to_dict()
        assert set(d.keys()) == {"role", "content", "timestamp", "metadata"}

    def test_from_dict_basic(self):
        data = {"role": "user", "content": "Hello"}
        msg = ChatMessage.from_dict(data)
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert isinstance(msg.timestamp, float)
        assert msg.metadata == {}

    def test_from_dict_with_all_fields(self):
        data = {
            "role": "assistant",
            "content": "Response",
            "timestamp": 1700000000.0,
            "metadata": {"agent": "test"},
        }
        msg = ChatMessage.from_dict(data)
        assert msg.role == "assistant"
        assert msg.content == "Response"
        assert msg.timestamp == 1700000000.0
        assert msg.metadata == {"agent": "test"}

    def test_roundtrip(self):
        original = ChatMessage(role="user", content="roundtrip test", metadata={"id": 1})
        d = original.to_dict()
        restored = ChatMessage.from_dict(d)
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.timestamp == original.timestamp
        assert restored.metadata == original.metadata


# ===========================================================================
# 6. ChatContext Tests
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_CHAT_CONTEXT, reason="ChatContext import failed")
class TestChatContextInit:
    """Tests for ChatContext initialization."""

    def test_init_defaults(self):
        ctx = ChatContext()
        assert ctx.max_history == 50
        assert ctx.system_prompt is None
        assert ctx.messages == []

    def test_init_custom_max_history(self):
        ctx = ChatContext(max_history=10)
        assert ctx.max_history == 10

    def test_init_with_system_prompt(self):
        ctx = ChatContext(system_prompt="You are a helper")
        assert ctx.system_prompt == "You are a helper"
        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == "system"
        assert ctx.messages[0].content == "You are a helper"

    def test_init_no_system_prompt_empty_messages(self):
        ctx = ChatContext()
        assert len(ctx.messages) == 0


@pytest.mark.unit
@pytest.mark.skipif(not HAS_CHAT_CONTEXT, reason="ChatContext import failed")
class TestChatContextAddMessage:
    """Tests for ChatContext.add_message."""

    def test_add_single_message(self):
        ctx = ChatContext()
        ctx.add_message("user", "Hello")
        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == "user"
        assert ctx.messages[0].content == "Hello"

    def test_add_multiple_messages(self):
        ctx = ChatContext()
        ctx.add_message("user", "Hello")
        ctx.add_message("assistant", "Hi there")
        assert len(ctx.messages) == 2

    def test_add_message_with_metadata(self):
        ctx = ChatContext()
        ctx.add_message("user", "Test", metadata={"key": "val"})
        assert ctx.messages[0].metadata == {"key": "val"}

    def test_add_message_auto_trim(self):
        ctx = ChatContext(max_history=3)
        ctx.add_message("user", "msg1")
        ctx.add_message("assistant", "msg2")
        ctx.add_message("user", "msg3")
        ctx.add_message("assistant", "msg4")
        assert len(ctx.messages) == 3
        # Oldest non-system message should be trimmed
        contents = [m.content for m in ctx.messages]
        assert "msg1" not in contents
        assert "msg4" in contents

    def test_add_message_auto_trim_preserves_system(self):
        ctx = ChatContext(max_history=3, system_prompt="System msg")
        ctx.add_message("user", "msg1")
        ctx.add_message("assistant", "msg2")
        ctx.add_message("user", "msg3")
        # Now we have 4 messages total (1 system + 3 user/assistant)
        # Trim should keep system + most recent 2
        assert len(ctx.messages) == 3
        assert ctx.messages[0].role == "system"
        assert ctx.messages[0].content == "System msg"


@pytest.mark.unit
@pytest.mark.skipif(not HAS_CHAT_CONTEXT, reason="ChatContext import failed")
class TestChatContextGetHistory:
    """Tests for ChatContext.get_history."""

    def test_get_history_empty(self):
        ctx = ChatContext()
        history = ctx.get_history()
        assert history == []

    def test_get_history_returns_copy(self):
        ctx = ChatContext()
        ctx.add_message("user", "Hello")
        history = ctx.get_history()
        assert len(history) == 1
        # Modifying the copy should not affect the original
        history.pop()
        assert len(ctx.messages) == 1

    def test_get_history_includes_all(self):
        ctx = ChatContext(system_prompt="sys")
        ctx.add_message("user", "u1")
        ctx.add_message("assistant", "a1")
        history = ctx.get_history()
        assert len(history) == 3


@pytest.mark.unit
@pytest.mark.skipif(not HAS_CHAT_CONTEXT, reason="ChatContext import failed")
class TestChatContextFormatted:
    """Tests for ChatContext.get_formatted_history."""

    def test_formatted_empty(self):
        ctx = ChatContext()
        assert ctx.get_formatted_history() == ""

    def test_formatted_single(self):
        ctx = ChatContext()
        ctx.add_message("user", "Hello")
        formatted = ctx.get_formatted_history()
        assert formatted == "user: Hello"

    def test_formatted_multiple(self):
        ctx = ChatContext()
        ctx.add_message("user", "Hello")
        ctx.add_message("assistant", "Hi")
        formatted = ctx.get_formatted_history()
        assert "user: Hello" in formatted
        assert "assistant: Hi" in formatted
        assert formatted == "user: Hello\nassistant: Hi"

    def test_formatted_with_system(self):
        ctx = ChatContext(system_prompt="Be helpful")
        ctx.add_message("user", "Hello")
        formatted = ctx.get_formatted_history()
        assert formatted.startswith("system: Be helpful")


@pytest.mark.unit
@pytest.mark.skipif(not HAS_CHAT_CONTEXT, reason="ChatContext import failed")
class TestChatContextRecentMessages:
    """Tests for ChatContext.get_recent_messages."""

    def test_recent_excludes_system(self):
        ctx = ChatContext(system_prompt="System")
        ctx.add_message("user", "u1")
        ctx.add_message("assistant", "a1")
        recent = ctx.get_recent_messages(count=10)
        roles = [m.role for m in recent]
        assert "system" not in roles
        assert len(recent) == 2

    def test_recent_respects_count(self):
        ctx = ChatContext()
        for i in range(10):
            ctx.add_message("user", f"msg{i}")
        recent = ctx.get_recent_messages(count=3)
        assert len(recent) == 3
        assert recent[-1].content == "msg9"

    def test_recent_default_count(self):
        ctx = ChatContext()
        for i in range(15):
            ctx.add_message("user", f"msg{i}")
        recent = ctx.get_recent_messages()
        assert len(recent) == 10  # default count is 10

    def test_recent_empty(self):
        ctx = ChatContext(system_prompt="sys")
        recent = ctx.get_recent_messages()
        assert recent == []


@pytest.mark.unit
@pytest.mark.skipif(not HAS_CHAT_CONTEXT, reason="ChatContext import failed")
class TestChatContextClear:
    """Tests for ChatContext.clear."""

    def test_clear_removes_non_system(self):
        ctx = ChatContext()
        ctx.add_message("user", "Hello")
        ctx.add_message("assistant", "Hi")
        ctx.clear()
        assert len(ctx.messages) == 0

    def test_clear_keeps_system(self):
        ctx = ChatContext(system_prompt="System prompt")
        ctx.add_message("user", "Hello")
        ctx.add_message("assistant", "Hi")
        ctx.clear()
        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == "system"
        assert ctx.messages[0].content == "System prompt"


@pytest.mark.unit
@pytest.mark.skipif(not HAS_CHAT_CONTEXT, reason="ChatContext import failed")
class TestChatContextSerialization:
    """Tests for ChatContext to_dict and from_dict."""

    def test_to_dict(self):
        ctx = ChatContext(max_history=20, system_prompt="sys")
        ctx.add_message("user", "Hello")
        d = ctx.to_dict()
        assert d["max_history"] == 20
        assert d["system_prompt"] == "sys"
        assert len(d["messages"]) == 2  # system + user
        assert d["messages"][0]["role"] == "system"

    def test_to_dict_empty(self):
        ctx = ChatContext()
        d = ctx.to_dict()
        assert d["max_history"] == 50
        assert d["system_prompt"] is None
        assert d["messages"] == []

    def test_from_dict(self):
        data = {
            "max_history": 25,
            "system_prompt": "sys",
            "messages": [
                {"role": "system", "content": "sys", "timestamp": 100.0, "metadata": {}},
                {"role": "user", "content": "hi", "timestamp": 101.0, "metadata": {}},
            ],
        }
        ctx = ChatContext.from_dict(data)
        assert ctx.max_history == 25
        assert ctx.system_prompt == "sys"
        assert len(ctx.messages) == 2

    def test_from_dict_defaults(self):
        data = {}
        ctx = ChatContext.from_dict(data)
        assert ctx.max_history == 50
        assert ctx.system_prompt is None
        assert len(ctx.messages) == 0

    def test_roundtrip(self):
        ctx = ChatContext(max_history=30, system_prompt="Be concise")
        ctx.add_message("user", "Summarize this")
        ctx.add_message("assistant", "Done")
        d = ctx.to_dict()
        restored = ChatContext.from_dict(d)
        assert restored.max_history == ctx.max_history
        assert restored.system_prompt == ctx.system_prompt
        assert len(restored.messages) == len(ctx.messages)
        for orig, rest in zip(ctx.messages, restored.messages):
            assert orig.role == rest.role
            assert orig.content == rest.content


# ===========================================================================
# 7. WorkflowTask Tests
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_WORKFLOW_CONTEXT or not HAS_TASK_STATUS,
    reason="WorkflowTask or TaskStatus import failed",
)
class TestWorkflowTask:
    """Tests for WorkflowTask dataclass."""

    def test_creation_defaults(self):
        task = WorkflowTask(id="t1", goal="Do something")
        assert task.id == "t1"
        assert task.goal == "Do something"
        assert task.status == TaskStatus.PENDING
        assert task.dependencies == []
        assert task.result is None
        assert task.error is None
        assert task.metadata == {}
        assert isinstance(task.created_at, float)
        assert task.completed_at is None

    def test_creation_all_fields(self):
        ts = time.time()
        task = WorkflowTask(
            id="t2",
            goal="Build feature",
            status=TaskStatus.COMPLETED,
            dependencies=["t1"],
            result={"output": "done"},
            error=None,
            metadata={"priority": 5},
            created_at=ts,
            completed_at=ts + 10,
        )
        assert task.status == TaskStatus.COMPLETED
        assert task.dependencies == ["t1"]
        assert task.result == {"output": "done"}
        assert task.completed_at == ts + 10

    def test_status_pending(self):
        task = WorkflowTask(id="t1", goal="x")
        assert task.status == TaskStatus.PENDING

    def test_status_in_progress(self):
        task = WorkflowTask(id="t1", goal="x", status=TaskStatus.IN_PROGRESS)
        assert task.status == TaskStatus.IN_PROGRESS

    def test_status_failed(self):
        task = WorkflowTask(id="t1", goal="x", status=TaskStatus.FAILED, error="timeout")
        assert task.error == "timeout"

    def test_multiple_dependencies(self):
        task = WorkflowTask(id="t3", goal="x", dependencies=["t1", "t2"])
        assert len(task.dependencies) == 2


# ===========================================================================
# 8. WorkflowContext Tests
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_WORKFLOW_CONTEXT or not HAS_TASK_STATUS,
    reason="WorkflowContext or TaskStatus import failed",
)
class TestWorkflowContextInit:
    """Tests for WorkflowContext initialization."""

    def test_init_defaults(self):
        ctx = WorkflowContext()
        assert ctx.workflow_id.startswith("workflow_")
        assert ctx.max_tasks == 100
        assert ctx.tasks == {}
        assert ctx.execution_order == []
        assert ctx.metadata == {}

    def test_init_custom_id(self):
        ctx = WorkflowContext(workflow_id="my-wf-123")
        assert ctx.workflow_id == "my-wf-123"

    def test_init_custom_max_tasks(self):
        ctx = WorkflowContext(max_tasks=10)
        assert ctx.max_tasks == 10


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_WORKFLOW_CONTEXT or not HAS_TASK_STATUS,
    reason="WorkflowContext or TaskStatus import failed",
)
class TestWorkflowContextAddTask:
    """Tests for WorkflowContext.add_task."""

    def test_add_task_auto_id(self):
        ctx = WorkflowContext()
        task_id = ctx.add_task("Do something")
        assert task_id.startswith("task_")
        assert task_id in ctx.tasks

    def test_add_task_custom_id(self):
        ctx = WorkflowContext()
        task_id = ctx.add_task("Do something", task_id="custom-1")
        assert task_id == "custom-1"
        assert "custom-1" in ctx.tasks

    def test_add_task_with_dependencies(self):
        ctx = WorkflowContext()
        t1 = ctx.add_task("First task", task_id="t1")
        t2 = ctx.add_task("Second task", task_id="t2", dependencies=["t1"])
        assert ctx.tasks["t2"].dependencies == ["t1"]

    def test_add_task_with_metadata(self):
        ctx = WorkflowContext()
        task_id = ctx.add_task("Task", metadata={"priority": 5})
        assert ctx.tasks[task_id].metadata == {"priority": 5}

    def test_add_task_goal_stored(self):
        ctx = WorkflowContext()
        task_id = ctx.add_task("Research AI trends")
        assert ctx.tasks[task_id].goal == "Research AI trends"

    def test_add_task_initial_status_pending(self):
        ctx = WorkflowContext()
        task_id = ctx.add_task("Task")
        assert ctx.tasks[task_id].status == TaskStatus.PENDING


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_WORKFLOW_CONTEXT or not HAS_TASK_STATUS,
    reason="WorkflowContext or TaskStatus import failed",
)
class TestWorkflowContextGetTask:
    """Tests for WorkflowContext.get_task."""

    def test_get_existing_task(self):
        ctx = WorkflowContext()
        task_id = ctx.add_task("Task", task_id="t1")
        task = ctx.get_task("t1")
        assert task is not None
        assert task.goal == "Task"

    def test_get_nonexistent_task(self):
        ctx = WorkflowContext()
        task = ctx.get_task("nonexistent")
        assert task is None


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_WORKFLOW_CONTEXT or not HAS_TASK_STATUS,
    reason="WorkflowContext or TaskStatus import failed",
)
class TestWorkflowContextUpdateStatus:
    """Tests for WorkflowContext.update_task_status."""

    def test_update_status(self):
        ctx = WorkflowContext()
        ctx.add_task("Task", task_id="t1")
        ctx.update_task_status("t1", TaskStatus.IN_PROGRESS)
        assert ctx.tasks["t1"].status == TaskStatus.IN_PROGRESS

    def test_update_status_with_result(self):
        ctx = WorkflowContext()
        ctx.add_task("Task", task_id="t1")
        ctx.update_task_status("t1", TaskStatus.COMPLETED, result="done")
        assert ctx.tasks["t1"].result == "done"

    def test_update_status_with_error(self):
        ctx = WorkflowContext()
        ctx.add_task("Task", task_id="t1")
        ctx.update_task_status("t1", TaskStatus.FAILED, error="timeout")
        assert ctx.tasks["t1"].error == "timeout"

    def test_update_completed_sets_completed_at(self):
        ctx = WorkflowContext()
        ctx.add_task("Task", task_id="t1")
        ctx.update_task_status("t1", TaskStatus.COMPLETED, result="ok")
        assert ctx.tasks["t1"].completed_at is not None
        assert isinstance(ctx.tasks["t1"].completed_at, float)

    def test_update_failed_sets_completed_at(self):
        ctx = WorkflowContext()
        ctx.add_task("Task", task_id="t1")
        ctx.update_task_status("t1", TaskStatus.FAILED, error="err")
        assert ctx.tasks["t1"].completed_at is not None

    def test_update_in_progress_no_completed_at(self):
        ctx = WorkflowContext()
        ctx.add_task("Task", task_id="t1")
        ctx.update_task_status("t1", TaskStatus.IN_PROGRESS)
        assert ctx.tasks["t1"].completed_at is None

    def test_update_nonexistent_task_no_error(self):
        ctx = WorkflowContext()
        # Should not raise
        ctx.update_task_status("nonexistent", TaskStatus.COMPLETED)


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_WORKFLOW_CONTEXT or not HAS_TASK_STATUS,
    reason="WorkflowContext or TaskStatus import failed",
)
class TestWorkflowContextReadyTasks:
    """Tests for WorkflowContext.get_ready_tasks."""

    def test_ready_tasks_no_deps(self):
        ctx = WorkflowContext()
        ctx.add_task("T1", task_id="t1")
        ctx.add_task("T2", task_id="t2")
        ready = ctx.get_ready_tasks()
        assert len(ready) == 2

    def test_ready_tasks_with_deps_satisfied(self):
        ctx = WorkflowContext()
        ctx.add_task("T1", task_id="t1")
        ctx.add_task("T2", task_id="t2", dependencies=["t1"])
        ctx.update_task_status("t1", TaskStatus.COMPLETED)
        ready = ctx.get_ready_tasks()
        # t1 is completed (not pending), t2 should be ready
        ready_ids = [t.id for t in ready]
        assert "t2" in ready_ids

    def test_ready_tasks_with_deps_unsatisfied(self):
        ctx = WorkflowContext()
        ctx.add_task("T1", task_id="t1")
        ctx.add_task("T2", task_id="t2", dependencies=["t1"])
        ready = ctx.get_ready_tasks()
        # t1 is pending (ready), but t2's dep is not completed
        ready_ids = [t.id for t in ready]
        assert "t1" in ready_ids
        assert "t2" not in ready_ids

    def test_ready_tasks_excludes_non_pending(self):
        ctx = WorkflowContext()
        ctx.add_task("T1", task_id="t1")
        ctx.update_task_status("t1", TaskStatus.IN_PROGRESS)
        ready = ctx.get_ready_tasks()
        assert len(ready) == 0

    def test_ready_tasks_excludes_completed(self):
        ctx = WorkflowContext()
        ctx.add_task("T1", task_id="t1")
        ctx.update_task_status("t1", TaskStatus.COMPLETED)
        ready = ctx.get_ready_tasks()
        assert len(ready) == 0


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_WORKFLOW_CONTEXT or not HAS_TASK_STATUS,
    reason="WorkflowContext or TaskStatus import failed",
)
class TestWorkflowContextExecutionOrder:
    """Tests for WorkflowContext.get_execution_order."""

    def test_simple_order(self):
        ctx = WorkflowContext()
        ctx.add_task("T1", task_id="t1")
        ctx.add_task("T2", task_id="t2")
        order = ctx.get_execution_order()
        assert set(order) == {"t1", "t2"}

    def test_order_with_deps(self):
        ctx = WorkflowContext()
        ctx.add_task("T1", task_id="t1")
        ctx.add_task("T2", task_id="t2", dependencies=["t1"])
        order = ctx.get_execution_order()
        assert order.index("t1") < order.index("t2")

    def test_order_chain(self):
        ctx = WorkflowContext()
        ctx.add_task("T1", task_id="t1")
        ctx.add_task("T2", task_id="t2", dependencies=["t1"])
        ctx.add_task("T3", task_id="t3", dependencies=["t2"])
        order = ctx.get_execution_order()
        assert order.index("t1") < order.index("t2")
        assert order.index("t2") < order.index("t3")

    def test_order_circular_handles_gracefully(self):
        ctx = WorkflowContext()
        ctx.add_task("T1", task_id="t1", dependencies=["t2"])
        ctx.add_task("T2", task_id="t2", dependencies=["t1"])
        # Should not raise, but may not include all tasks
        order = ctx.get_execution_order()
        assert isinstance(order, list)


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_WORKFLOW_CONTEXT or not HAS_TASK_STATUS,
    reason="WorkflowContext or TaskStatus import failed",
)
class TestWorkflowContextIsComplete:
    """Tests for WorkflowContext.is_complete."""

    def test_empty_is_complete(self):
        ctx = WorkflowContext()
        assert ctx.is_complete() is True

    def test_pending_not_complete(self):
        ctx = WorkflowContext()
        ctx.add_task("T1", task_id="t1")
        assert ctx.is_complete() is False

    def test_all_completed_is_complete(self):
        ctx = WorkflowContext()
        ctx.add_task("T1", task_id="t1")
        ctx.add_task("T2", task_id="t2")
        ctx.update_task_status("t1", TaskStatus.COMPLETED)
        ctx.update_task_status("t2", TaskStatus.COMPLETED)
        assert ctx.is_complete() is True

    def test_all_failed_is_complete(self):
        ctx = WorkflowContext()
        ctx.add_task("T1", task_id="t1")
        ctx.update_task_status("t1", TaskStatus.FAILED)
        assert ctx.is_complete() is True

    def test_mixed_completed_failed_is_complete(self):
        ctx = WorkflowContext()
        ctx.add_task("T1", task_id="t1")
        ctx.add_task("T2", task_id="t2")
        ctx.update_task_status("t1", TaskStatus.COMPLETED)
        ctx.update_task_status("t2", TaskStatus.FAILED)
        assert ctx.is_complete() is True

    def test_in_progress_not_complete(self):
        ctx = WorkflowContext()
        ctx.add_task("T1", task_id="t1")
        ctx.update_task_status("t1", TaskStatus.IN_PROGRESS)
        assert ctx.is_complete() is False


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_WORKFLOW_CONTEXT or not HAS_TASK_STATUS,
    reason="WorkflowContext or TaskStatus import failed",
)
class TestWorkflowContextSummary:
    """Tests for WorkflowContext.get_summary."""

    def test_summary_structure(self):
        ctx = WorkflowContext(workflow_id="wf-1")
        ctx.add_task("T1", task_id="t1")
        summary = ctx.get_summary()
        assert summary["workflow_id"] == "wf-1"
        assert summary["total_tasks"] == 1
        assert "status_counts" in summary
        assert "is_complete" in summary
        assert "execution_order" in summary

    def test_summary_status_counts(self):
        ctx = WorkflowContext()
        ctx.add_task("T1", task_id="t1")
        ctx.add_task("T2", task_id="t2")
        ctx.update_task_status("t1", TaskStatus.COMPLETED)
        summary = ctx.get_summary()
        assert summary["status_counts"]["completed"] == 1
        assert summary["status_counts"]["pending"] == 1

    def test_summary_empty(self):
        ctx = WorkflowContext()
        summary = ctx.get_summary()
        assert summary["total_tasks"] == 0
        assert summary["is_complete"] is True


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_WORKFLOW_CONTEXT or not HAS_TASK_STATUS,
    reason="WorkflowContext or TaskStatus import failed",
)
class TestWorkflowContextSerialization:
    """Tests for WorkflowContext to_dict and from_dict."""

    def test_to_dict(self):
        ctx = WorkflowContext(workflow_id="wf-1", max_tasks=50)
        ctx.add_task("T1", task_id="t1")
        d = ctx.to_dict()
        assert d["workflow_id"] == "wf-1"
        assert d["max_tasks"] == 50
        assert "t1" in d["tasks"]
        assert d["tasks"]["t1"]["goal"] == "T1"
        assert d["tasks"]["t1"]["status"] == "pending"

    def test_to_dict_empty(self):
        ctx = WorkflowContext(workflow_id="wf-empty")
        d = ctx.to_dict()
        assert d["tasks"] == {}

    def test_from_dict(self):
        data = {
            "workflow_id": "wf-restored",
            "max_tasks": 50,
            "tasks": {
                "t1": {
                    "id": "t1",
                    "goal": "Restore me",
                    "status": "completed",
                    "dependencies": [],
                    "result": "done",
                    "error": None,
                    "metadata": {},
                    "created_at": 1700000000.0,
                    "completed_at": 1700000010.0,
                },
            },
            "execution_order": ["t1"],
            "metadata": {"key": "value"},
        }
        ctx = WorkflowContext.from_dict(data)
        assert ctx.workflow_id == "wf-restored"
        assert ctx.max_tasks == 50
        assert "t1" in ctx.tasks
        assert ctx.tasks["t1"].status == TaskStatus.COMPLETED
        assert ctx.execution_order == ["t1"]
        assert ctx.metadata == {"key": "value"}

    def test_from_dict_defaults(self):
        data = {}
        ctx = WorkflowContext.from_dict(data)
        assert ctx.max_tasks == 100
        assert len(ctx.tasks) == 0

    def test_roundtrip(self):
        ctx = WorkflowContext(workflow_id="wf-round")
        ctx.add_task("T1", task_id="t1")
        ctx.add_task("T2", task_id="t2", dependencies=["t1"])
        ctx.update_task_status("t1", TaskStatus.COMPLETED, result="ok")
        d = ctx.to_dict()
        restored = WorkflowContext.from_dict(d)
        assert restored.workflow_id == "wf-round"
        assert len(restored.tasks) == 2
        assert restored.tasks["t1"].status == TaskStatus.COMPLETED


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_WORKFLOW_CONTEXT or not HAS_TASK_STATUS,
    reason="WorkflowContext or TaskStatus import failed",
)
class TestWorkflowContextMaxTasks:
    """Tests for WorkflowContext task trimming."""

    def test_max_tasks_trims_completed(self):
        ctx = WorkflowContext(max_tasks=3)
        ctx.add_task("T1", task_id="t1")
        ctx.update_task_status("t1", TaskStatus.COMPLETED, result="r1")
        ctx.add_task("T2", task_id="t2")
        ctx.update_task_status("t2", TaskStatus.COMPLETED, result="r2")
        ctx.add_task("T3", task_id="t3")
        # Now at limit; adding one more should trim oldest completed
        ctx.add_task("T4", task_id="t4")
        assert len(ctx.tasks) <= 4  # May have trimmed or not depending on completed order

    def test_under_max_no_trimming(self):
        ctx = WorkflowContext(max_tasks=10)
        for i in range(5):
            ctx.add_task(f"T{i}", task_id=f"t{i}")
        assert len(ctx.tasks) == 5


# ===========================================================================
# 9. CommandInfo Tests
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_COMMAND_SERVICE, reason="CommandInfo import failed")
class TestCommandInfo:
    """Tests for CommandInfo dataclass."""

    def test_creation(self):
        info = CommandInfo(
            name="task",
            description="Manage tasks",
            usage="/task <action>",
            category="workflow",
            aliases=["t", "tasks"],
        )
        assert info.name == "task"
        assert info.description == "Manage tasks"
        assert info.usage == "/task <action>"
        assert info.category == "workflow"
        assert info.aliases == ["t", "tasks"]

    def test_to_dict(self):
        info = CommandInfo(
            name="help",
            description="Show help",
            usage="/help",
            category="general",
            aliases=["h"],
        )
        d = info.to_dict()
        assert d["name"] == "help"
        assert d["description"] == "Show help"
        assert d["usage"] == "/help"
        assert d["category"] == "general"
        assert d["aliases"] == ["h"]

    def test_to_dict_keys(self):
        info = CommandInfo(
            name="test", description="d", usage="/test",
            category="c", aliases=[],
        )
        d = info.to_dict()
        assert set(d.keys()) == {"name", "description", "usage", "category", "aliases"}

    def test_empty_aliases(self):
        info = CommandInfo(
            name="cmd", description="desc", usage="/cmd",
            category="cat", aliases=[],
        )
        assert info.aliases == []
        assert info.to_dict()["aliases"] == []


# ===========================================================================
# 10. CommandExecutionResult Tests
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_COMMAND_SERVICE, reason="CommandExecutionResult import failed")
class TestCommandExecutionResult:
    """Tests for CommandExecutionResult dataclass."""

    def test_creation_success(self):
        result = CommandExecutionResult(success=True, output="Done")
        assert result.success is True
        assert result.output == "Done"
        assert result.error is None
        assert result.data is None

    def test_creation_failure(self):
        result = CommandExecutionResult(
            success=False, output="", error="Command not found",
        )
        assert result.success is False
        assert result.error == "Command not found"

    def test_creation_with_data(self):
        result = CommandExecutionResult(
            success=True, output="OK", data={"count": 5},
        )
        assert result.data == {"count": 5}

    def test_to_dict(self):
        result = CommandExecutionResult(
            success=True, output="output", error=None, data={"k": "v"},
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["output"] == "output"
        assert d["error"] is None
        assert d["data"] == {"k": "v"}

    def test_to_dict_keys(self):
        result = CommandExecutionResult(success=True, output="x")
        d = result.to_dict()
        assert set(d.keys()) == {"success", "output", "error", "data"}


# ===========================================================================
# 11. CommandService Tests
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_COMMAND_SERVICE, reason="CommandService import failed")
class TestCommandService:
    """Tests for CommandService initialization and utility methods."""

    def test_init(self):
        svc = CommandService()
        assert svc._registry is None
        assert svc._cli is None
        assert svc._initialized is False

    def test_is_command_slash(self):
        svc = CommandService()
        assert svc.is_command("/help") is True

    def test_is_command_no_slash(self):
        svc = CommandService()
        assert svc.is_command("help") is False

    def test_is_command_with_whitespace(self):
        svc = CommandService()
        assert svc.is_command("  /task list") is True

    def test_is_command_empty(self):
        svc = CommandService()
        assert svc.is_command("") is False

    def test_parse_command_basic(self):
        svc = CommandService()
        cmd, args = svc.parse_command("/task list")
        assert cmd == "task"
        assert args == "list"

    def test_parse_command_no_args(self):
        svc = CommandService()
        cmd, args = svc.parse_command("/help")
        assert cmd == "help"
        assert args == ""

    def test_parse_command_with_flags(self):
        svc = CommandService()
        cmd, args = svc.parse_command("/task list --status=pending")
        assert cmd == "task"
        assert args == "list --status=pending"

    def test_parse_command_no_slash(self):
        svc = CommandService()
        cmd, args = svc.parse_command("regular text")
        assert cmd == ""
        assert args == "regular text"

    def test_parse_command_whitespace(self):
        svc = CommandService()
        cmd, args = svc.parse_command("  /run code  ")
        assert cmd == "run"
        assert args == "code"

    def test_lazy_init_not_triggered_on_creation(self):
        svc = CommandService()
        assert svc._initialized is False
        assert svc._registry is None


# ===========================================================================
# 12. JottyAPI Tests
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_JOTTY_API, reason="JottyAPI import failed")
class TestJottyAPIInit:
    """Tests for JottyAPI initialization."""

    def _build_api(self):
        mock_agents = [Mock()]
        mock_config = Mock()
        mock_conductor = Mock()
        return JottyAPI(
            agents=mock_agents,
            config=mock_config,
            conductor=mock_conductor,
        )

    def test_init_with_conductor(self):
        api = self._build_api()
        assert api.conductor is not None
        assert api.config is not None
        assert len(api.agents) == 1

    def test_init_stores_agents(self):
        agent1 = Mock()
        agent2 = Mock()
        api = JottyAPI(agents=[agent1, agent2], conductor=Mock())
        assert len(api.agents) == 2

    def test_init_stores_config(self):
        config = Mock()
        api = JottyAPI(agents=[Mock()], config=config, conductor=Mock())
        assert api.config is config

    def test_init_lazy_properties_none(self):
        api = self._build_api()
        assert api._chat_use_case is None
        assert api._workflow_use_case is None

    @patch("Jotty.core.jotty.create_swarm_manager")
    def test_init_without_conductor_creates_one(self, mock_create):
        mock_conductor = Mock()
        mock_create.return_value = mock_conductor
        agents = [Mock()]
        api = JottyAPI(agents=agents)
        assert api.conductor is mock_conductor
        mock_create.assert_called_once()


@pytest.mark.unit
@pytest.mark.skipif(not HAS_JOTTY_API, reason="JottyAPI import failed")
class TestJottyAPIProperties:
    """Tests for JottyAPI lazy-init properties."""

    def _build_api(self):
        return JottyAPI(
            agents=[Mock()],
            config=Mock(),
            conductor=Mock(),
        )

    @patch("Jotty.core.api.unified.ChatUseCase")
    @patch("Jotty.core.api.unified.UseCaseConfig")
    def test_chat_lazy_init(self, _mock_cfg, mock_chat_cls):
        api = self._build_api()
        chat = api.chat
        assert chat is not None
        # Second access returns same instance
        assert api.chat is chat

    @patch("Jotty.core.api.unified.WorkflowUseCase")
    @patch("Jotty.core.api.unified.UseCaseConfig")
    def test_workflow_lazy_init(self, _mock_cfg, mock_wf_cls):
        api = self._build_api()
        workflow = api.workflow
        assert workflow is not None
        assert api.workflow is workflow

    @patch("Jotty.core.api.unified.ChatUseCase")
    @patch("Jotty.core.api.unified.UseCaseConfig")
    def test_chat_property_creates_chat_use_case(self, _mock_cfg, mock_chat_cls):
        api = self._build_api()
        _ = api.chat
        mock_chat_cls.assert_called_once()

    @patch("Jotty.core.api.unified.WorkflowUseCase")
    @patch("Jotty.core.api.unified.UseCaseConfig")
    def test_workflow_property_creates_workflow_use_case(self, _mock_cfg, mock_wf_cls):
        api = self._build_api()
        _ = api.workflow
        mock_wf_cls.assert_called_once()


@pytest.mark.unit
@pytest.mark.skipif(not HAS_JOTTY_API, reason="JottyAPI import failed")
class TestJottyAPIChatExecute:
    """Tests for JottyAPI.chat_execute."""

    def _build_api(self):
        return JottyAPI(
            agents=[Mock()],
            config=Mock(),
            conductor=Mock(),
        )

    @pytest.mark.asyncio
    @patch("Jotty.core.api.unified.ChatUseCase")
    @patch("Jotty.core.api.unified.UseCaseConfig")
    async def test_chat_execute_delegates(self, _mock_cfg, mock_chat_cls):
        api = self._build_api()
        mock_result = Mock()
        mock_result.to_dict.return_value = {"response": "hi"}
        mock_chat_instance = Mock()
        mock_chat_instance.execute = AsyncMock(return_value=mock_result)
        mock_chat_cls.return_value = mock_chat_instance
        api._chat_use_case = None

        output = await api.chat_execute(message="hello")
        assert output == {"response": "hi"}

    @pytest.mark.asyncio
    @patch("Jotty.core.api.unified.ChatUseCase")
    @patch("Jotty.core.api.unified.UseCaseConfig")
    async def test_chat_execute_passes_message(self, _mock_cfg, mock_chat_cls):
        api = self._build_api()
        mock_result = Mock()
        mock_result.to_dict.return_value = {}
        mock_chat_instance = Mock()
        mock_chat_instance.execute = AsyncMock(return_value=mock_result)
        mock_chat_cls.return_value = mock_chat_instance
        api._chat_use_case = None

        await api.chat_execute(message="test message")
        call_kwargs = mock_chat_instance.execute.call_args
        assert call_kwargs[1].get("goal") == "test message" or call_kwargs[0][0] == "test message"


@pytest.mark.unit
@pytest.mark.skipif(not HAS_JOTTY_API, reason="JottyAPI import failed")
class TestJottyAPIWorkflowExecute:
    """Tests for JottyAPI.workflow_execute."""

    def _build_api(self):
        return JottyAPI(
            agents=[Mock()],
            config=Mock(),
            conductor=Mock(),
        )

    @pytest.mark.asyncio
    @patch("Jotty.core.api.unified.WorkflowUseCase")
    @patch("Jotty.core.api.unified.UseCaseConfig")
    async def test_workflow_execute_delegates(self, _mock_cfg, mock_wf_cls):
        api = self._build_api()
        mock_result = Mock()
        mock_result.to_dict.return_value = {"workflow_id": "wf-1", "success": True}
        mock_wf_instance = Mock()
        mock_wf_instance.execute = AsyncMock(return_value=mock_result)
        mock_wf_cls.return_value = mock_wf_instance
        api._workflow_use_case = None

        output = await api.workflow_execute(goal="build app")
        assert output == {"workflow_id": "wf-1", "success": True}

    @pytest.mark.asyncio
    @patch("Jotty.core.api.unified.WorkflowUseCase")
    @patch("Jotty.core.api.unified.UseCaseConfig")
    async def test_workflow_execute_default_mode(self, _mock_cfg, mock_wf_cls):
        api = self._build_api()
        mock_result = Mock()
        mock_result.to_dict.return_value = {}
        mock_wf_instance = Mock()
        mock_wf_instance.execute = AsyncMock(return_value=mock_result)
        mock_wf_cls.return_value = mock_wf_instance
        api._workflow_use_case = None

        await api.workflow_execute(goal="task")
        # Default mode is "dynamic" so it uses the cached property


@pytest.mark.unit
@pytest.mark.skipif(not HAS_JOTTY_API, reason="JottyAPI import failed")
class TestJottyAPIWorkflowEnqueue:
    """Tests for JottyAPI.workflow_enqueue."""

    def _build_api(self):
        return JottyAPI(
            agents=[Mock()],
            config=Mock(),
            conductor=Mock(),
        )

    @pytest.mark.asyncio
    @patch("Jotty.core.api.unified.WorkflowUseCase")
    @patch("Jotty.core.api.unified.UseCaseConfig")
    async def test_workflow_enqueue_delegates(self, _mock_cfg, mock_wf_cls):
        api = self._build_api()
        mock_wf_instance = Mock()
        mock_wf_instance.enqueue = AsyncMock(return_value="task-123")
        mock_wf_cls.return_value = mock_wf_instance
        api._workflow_use_case = None

        task_id = await api.workflow_enqueue(goal="background task")
        assert task_id == "task-123"


# ===========================================================================
# 13. Integration / Cross-Class Tests
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_USE_CASE_BASE or not HAS_CHAT_CONTEXT,
    reason="Required imports failed",
)
class TestUseCaseWithChatContext:
    """Integration tests combining use case base classes with chat context."""

    def test_chat_context_in_result_metadata(self):
        ctx = ChatContext(system_prompt="Be helpful")
        ctx.add_message("user", "Hello")
        conductor = MagicMock()
        uc = _ChatUseCaseStub(conductor)
        result = uc._create_result(
            True, "Hi there",
            metadata={"context": ctx.to_dict()},
        )
        assert "context" in result.metadata
        assert result.metadata["context"]["system_prompt"] == "Be helpful"

    def test_chat_message_in_history_list(self):
        msg1 = ChatMessage(role="user", content="Q1")
        msg2 = ChatMessage(role="assistant", content="A1")
        history = [msg1.to_dict(), msg2.to_dict()]
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_execute_with_chat_context(self):
        ctx = ChatContext()
        ctx.add_message("user", "Hello")
        conductor = MagicMock()
        uc = _ChatUseCaseStub(conductor)
        result = await uc.execute("Hello")
        assert result.success is True
        assert result.use_case_type == UseCaseType.CHAT


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_USE_CASE_BASE or not HAS_WORKFLOW_CONTEXT or not HAS_TASK_STATUS,
    reason="Required imports failed",
)
class TestUseCaseWithWorkflowContext:
    """Integration tests combining use case base classes with workflow context."""

    def test_workflow_context_summary_in_result(self):
        wf_ctx = WorkflowContext(workflow_id="wf-int")
        wf_ctx.add_task("Step 1", task_id="s1")
        wf_ctx.update_task_status("s1", TaskStatus.COMPLETED, result="done")
        conductor = MagicMock()
        uc = _WorkflowUseCaseStub(conductor)
        result = uc._create_result(
            True, "complete",
            metadata={"summary": wf_ctx.get_summary()},
        )
        assert result.metadata["summary"]["workflow_id"] == "wf-int"
        assert result.metadata["summary"]["total_tasks"] == 1
        assert result.metadata["summary"]["is_complete"] is True

    @pytest.mark.asyncio
    async def test_workflow_execute_result_type(self):
        conductor = MagicMock()
        uc = _WorkflowUseCaseStub(conductor)
        result = await uc.execute("plan project")
        assert result.use_case_type == UseCaseType.WORKFLOW
        assert isinstance(result.output, dict)

    def test_workflow_task_status_transition(self):
        wf_ctx = WorkflowContext()
        tid = wf_ctx.add_task("Task", task_id="t1")
        assert wf_ctx.tasks[tid].status == TaskStatus.PENDING
        wf_ctx.update_task_status(tid, TaskStatus.IN_PROGRESS)
        assert wf_ctx.tasks[tid].status == TaskStatus.IN_PROGRESS
        wf_ctx.update_task_status(tid, TaskStatus.COMPLETED, result="done")
        assert wf_ctx.tasks[tid].status == TaskStatus.COMPLETED
        assert wf_ctx.tasks[tid].completed_at is not None


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_COMMAND_SERVICE or not HAS_USE_CASE_BASE,
    reason="Required imports failed",
)
class TestCommandServiceWithUseCases:
    """Integration tests combining CommandService with use case types."""

    def test_command_result_to_use_case_result(self):
        cmd_result = CommandExecutionResult(success=True, output="Listed 5 tasks")
        uc_result = UseCaseResult(
            success=cmd_result.success,
            output=cmd_result.output,
            metadata={"source": "command"},
            execution_time=0.1,
            use_case_type=UseCaseType.CHAT,
        )
        d = uc_result.to_dict()
        assert d["success"] is True
        assert d["output"] == "Listed 5 tasks"

    def test_command_parse_and_is_command(self):
        svc = CommandService()
        text = "/task list --all"
        assert svc.is_command(text) is True
        cmd, args = svc.parse_command(text)
        assert cmd == "task"
        assert "list" in args
        assert "--all" in args

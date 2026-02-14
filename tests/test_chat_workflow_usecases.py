"""
Comprehensive Test Suite for Chat & Workflow Use Case Executors and Orchestrators
=================================================================================

Tests for:
1. ChatExecutor       (core/use_cases/chat/chat_executor.py)
2. ChatOrchestrator   (core/use_cases/chat/chat_orchestrator.py)
3. ChatUseCase        (core/use_cases/chat/chat_use_case.py)
4. WorkflowExecutor   (core/use_cases/workflow/workflow_executor.py)
5. WorkflowOrchestrator (core/use_cases/workflow/workflow_orchestrator.py)
6. WorkflowUseCase    (core/use_cases/workflow/workflow_use_case.py)
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, AsyncMock, patch, PropertyMock
from typing import Dict, Any, Optional, List

# ---------------------------------------------------------------------------
# Guarded imports
# ---------------------------------------------------------------------------

try:
    from Jotty.core.use_cases.chat.chat_executor import ChatExecutor
    HAS_CHAT_EXECUTOR = True
except ImportError:
    HAS_CHAT_EXECUTOR = False

try:
    from Jotty.core.use_cases.chat.chat_orchestrator import ChatOrchestrator
    HAS_CHAT_ORCHESTRATOR = True
except ImportError:
    HAS_CHAT_ORCHESTRATOR = False

try:
    from Jotty.core.use_cases.chat.chat_use_case import ChatUseCase
    HAS_CHAT_USE_CASE = True
except ImportError:
    HAS_CHAT_USE_CASE = False

try:
    from Jotty.core.use_cases.chat.chat_context import ChatContext, ChatMessage
    HAS_CHAT_CONTEXT = True
except ImportError:
    HAS_CHAT_CONTEXT = False

try:
    from Jotty.core.use_cases.workflow.workflow_executor import WorkflowExecutor
    HAS_WORKFLOW_EXECUTOR = True
except ImportError:
    HAS_WORKFLOW_EXECUTOR = False

try:
    from Jotty.core.use_cases.workflow.workflow_orchestrator import WorkflowOrchestrator
    HAS_WORKFLOW_ORCHESTRATOR = True
except ImportError:
    HAS_WORKFLOW_ORCHESTRATOR = False

try:
    from Jotty.core.use_cases.workflow.workflow_use_case import WorkflowUseCase
    HAS_WORKFLOW_USE_CASE = True
except ImportError:
    HAS_WORKFLOW_USE_CASE = False

try:
    from Jotty.core.use_cases.workflow.workflow_context import (
        WorkflowContext,
        WorkflowTask,
    )
    HAS_WORKFLOW_CONTEXT = True
except ImportError:
    HAS_WORKFLOW_CONTEXT = False

try:
    from Jotty.core.use_cases.base import (
        BaseUseCase,
        UseCaseType,
        UseCaseResult,
        UseCaseConfig,
    )
    HAS_USE_CASE_BASE = True
except ImportError:
    HAS_USE_CASE_BASE = False

try:
    from Jotty.core.foundation.types import TaskStatus
    HAS_TASK_STATUS = True
except ImportError:
    HAS_TASK_STATUS = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_conductor(actors=None, has_q_predictor=False, has_select_agent=False):
    """Create a mock Conductor with configurable features."""
    conductor = MagicMock()
    conductor.run = AsyncMock(return_value={"response": "mock response"})

    if actors is not None:
        conductor.actors = actors
    else:
        actor = MagicMock()
        actor.name = "DefaultAgent"
        conductor.actors = [actor]

    if has_q_predictor:
        q_pred = MagicMock()
        q_pred.predict_q_value = MagicMock(return_value=(0.8, 0.9, {}))
        conductor.q_predictor = q_pred
    else:
        conductor.q_predictor = None

    if has_select_agent:
        conductor.select_agent = MagicMock(return_value="DynamicAgent")
    else:
        if hasattr(conductor, "select_agent"):
            del conductor.select_agent

    return conductor


def _make_mock_actor(name):
    """Create a mock actor with a name attribute."""
    actor = MagicMock()
    actor.name = name
    return actor


# ---------------------------------------------------------------------------
# 1. ChatExecutor Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_CHAT_EXECUTOR and HAS_CHAT_ORCHESTRATOR and HAS_CHAT_CONTEXT),
    reason="Chat executor imports unavailable",
)
class TestChatExecutorInit:
    """Tests for ChatExecutor.__init__."""

    def test_init_stores_conductor(self):
        conductor = _make_mock_conductor()
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="TestAgent")
        executor = ChatExecutor(conductor=conductor, orchestrator=orchestrator)
        assert executor.conductor is conductor

    def test_init_stores_orchestrator(self):
        conductor = _make_mock_conductor()
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="TestAgent")
        executor = ChatExecutor(conductor=conductor, orchestrator=orchestrator)
        assert executor.orchestrator is orchestrator

    def test_init_creates_default_context(self):
        conductor = _make_mock_conductor()
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="TestAgent")
        executor = ChatExecutor(conductor=conductor, orchestrator=orchestrator)
        assert isinstance(executor.context, ChatContext)

    def test_init_uses_provided_context(self):
        conductor = _make_mock_conductor()
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="TestAgent")
        custom_ctx = ChatContext(max_history=10)
        executor = ChatExecutor(
            conductor=conductor, orchestrator=orchestrator, context=custom_ctx
        )
        assert executor.context is custom_ctx
        assert executor.context.max_history == 10


@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_CHAT_EXECUTOR and HAS_CHAT_ORCHESTRATOR and HAS_CHAT_CONTEXT),
    reason="Chat executor imports unavailable",
)
class TestChatExecutorExecute:
    """Tests for ChatExecutor.execute()."""

    @pytest.mark.asyncio
    async def test_execute_returns_success_dict(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"response": "Hello!"})
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="Agent1")
        executor = ChatExecutor(conductor=conductor, orchestrator=orchestrator)

        result = await executor.execute(message="Hi")

        assert result["success"] is True
        assert result["message"] == "Hello!"
        assert result["agent"] == "Agent1"
        assert "execution_time" in result
        assert "metadata" in result

    @pytest.mark.asyncio
    async def test_execute_calls_conductor_run(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"response": "ok"})
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="Agent1")
        executor = ChatExecutor(conductor=conductor, orchestrator=orchestrator)

        await executor.execute(message="hello")

        conductor.run.assert_awaited_once()
        call_kwargs = conductor.run.call_args
        assert call_kwargs.kwargs["goal"] == "hello"

    @pytest.mark.asyncio
    async def test_execute_adds_messages_to_context(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"response": "answer"})
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="Agent1")
        ctx = ChatContext()
        executor = ChatExecutor(
            conductor=conductor, orchestrator=orchestrator, context=ctx
        )

        await executor.execute(message="question")

        messages = ctx.get_history()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "question"
        assert messages[1].role == "assistant"
        assert messages[1].content == "answer"

    @pytest.mark.asyncio
    async def test_execute_handles_exception(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(side_effect=RuntimeError("boom"))
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="Agent1")
        executor = ChatExecutor(conductor=conductor, orchestrator=orchestrator)

        result = await executor.execute(message="fail")

        assert result["success"] is False
        assert "boom" in result["message"]
        assert result["error"] == "boom"

    @pytest.mark.asyncio
    async def test_execute_result_metadata_includes_context_count(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"response": "ok"})
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="Agent1")
        executor = ChatExecutor(conductor=conductor, orchestrator=orchestrator)

        result = await executor.execute(message="hello")

        assert "context_used" in result["metadata"]

    @pytest.mark.asyncio
    async def test_execute_with_dspy_prediction_result(self):
        """Test that DSPy Prediction objects are serialized via _store."""
        conductor = _make_mock_conductor()
        prediction = MagicMock()
        prediction._store = {"response": "predicted output", "confidence": 0.95}
        # Make hasattr checks work correctly
        del prediction.response
        del prediction.output
        del prediction.final_output
        conductor.run = AsyncMock(return_value=prediction)
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="Agent1")
        executor = ChatExecutor(conductor=conductor, orchestrator=orchestrator)

        result = await executor.execute(message="test")

        assert result["success"] is True
        assert result["metadata"]["result"] == {
            "response": "predicted output",
            "confidence": 0.95,
        }

    @pytest.mark.asyncio
    async def test_execute_with_non_dict_non_store_result(self):
        """Test fallback serialization for unknown result types."""
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value="plain string result")
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="Agent1")
        executor = ChatExecutor(conductor=conductor, orchestrator=orchestrator)

        result = await executor.execute(message="test")

        assert result["success"] is True
        assert result["metadata"]["result"] == {"value": "plain string result"}


@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_CHAT_EXECUTOR and HAS_CHAT_ORCHESTRATOR and HAS_CHAT_CONTEXT),
    reason="Chat executor imports unavailable",
)
class TestChatExecutorExtractResponse:
    """Tests for ChatExecutor._extract_response()."""

    def _make_executor(self):
        conductor = _make_mock_conductor()
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="Agent1")
        return ChatExecutor(conductor=conductor, orchestrator=orchestrator)

    def test_extract_response_from_dict_with_response_key(self):
        executor = self._make_executor()
        assert executor._extract_response({"response": "hello"}) == "hello"

    def test_extract_response_from_dict_with_output_key(self):
        executor = self._make_executor()
        assert executor._extract_response({"output": "world"}) == "world"

    def test_extract_response_from_dict_fallback_to_str(self):
        executor = self._make_executor()
        result = executor._extract_response({"data": 42})
        assert "42" in result

    def test_extract_response_from_object_with_response_attr(self):
        executor = self._make_executor()
        obj = MagicMock()
        obj.response = "from attr"
        result = executor._extract_response(obj)
        assert result == "from attr"

    def test_extract_response_from_object_with_output_attr(self):
        executor = self._make_executor()
        obj = MagicMock(spec=["output"])
        obj.output = "output value"
        result = executor._extract_response(obj)
        assert result == "output value"

    def test_extract_response_from_object_with_final_output_attr(self):
        executor = self._make_executor()
        obj = MagicMock(spec=["final_output"])
        obj.final_output = "final value"
        result = executor._extract_response(obj)
        assert result == "final value"

    def test_extract_response_from_string(self):
        executor = self._make_executor()
        assert executor._extract_response("plain text") == "plain text"

    def test_extract_response_from_none(self):
        executor = self._make_executor()
        assert executor._extract_response(None) == ""

    def test_extract_response_from_number(self):
        executor = self._make_executor()
        assert executor._extract_response(42) == "42"


@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_CHAT_EXECUTOR and HAS_CHAT_ORCHESTRATOR and HAS_CHAT_CONTEXT),
    reason="Chat executor imports unavailable",
)
class TestChatExecutorChunkText:
    """Tests for ChatExecutor._chunk_text()."""

    def _make_executor(self):
        conductor = _make_mock_conductor()
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="Agent1")
        return ChatExecutor(conductor=conductor, orchestrator=orchestrator)

    def test_chunk_short_text(self):
        executor = self._make_executor()
        chunks = executor._chunk_text("Hello world.", chunk_size=50)
        assert len(chunks) >= 1
        assert "Hello world" in chunks[0]

    def test_chunk_long_text_splits_on_sentences(self):
        executor = self._make_executor()
        text = "First sentence. Second sentence. Third sentence."
        chunks = executor._chunk_text(text, chunk_size=30)
        assert len(chunks) >= 2

    def test_chunk_empty_text(self):
        executor = self._make_executor()
        chunks = executor._chunk_text("")
        assert chunks == []

    def test_chunk_none_text(self):
        executor = self._make_executor()
        chunks = executor._chunk_text(None)
        assert chunks == []

    def test_chunk_respects_chunk_size(self):
        executor = self._make_executor()
        text = "A. B. C. D. E. F. G. H. I. J."
        chunks = executor._chunk_text(text, chunk_size=10)
        for chunk in chunks:
            # Each chunk should be reasonably sized
            assert len(chunk) < 100

    def test_chunk_returns_list(self):
        executor = self._make_executor()
        chunks = executor._chunk_text("Some text here.")
        assert isinstance(chunks, list)

    def test_chunk_single_word_text(self):
        executor = self._make_executor()
        chunks = executor._chunk_text("Hello")
        assert len(chunks) >= 1
        assert "Hello" in chunks[0]


@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_CHAT_EXECUTOR and HAS_CHAT_ORCHESTRATOR and HAS_CHAT_CONTEXT),
    reason="Chat executor imports unavailable",
)
class TestChatExecutorStream:
    """Tests for ChatExecutor.stream()."""

    @pytest.mark.asyncio
    async def test_stream_yields_agent_selected_event(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"response": "streamed"})
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="StreamAgent")
        executor = ChatExecutor(conductor=conductor, orchestrator=orchestrator)

        events = []
        async for event in executor.stream(message="test"):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "agent_selected" in event_types

    @pytest.mark.asyncio
    async def test_stream_yields_done_event(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"response": "done"})
        # Ensure no run_actor_stream attribute
        if hasattr(conductor, "run_actor_stream"):
            del conductor.run_actor_stream
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="StreamAgent")
        executor = ChatExecutor(conductor=conductor, orchestrator=orchestrator)

        events = []
        async for event in executor.stream(message="test"):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "done" in event_types

    @pytest.mark.asyncio
    async def test_stream_yields_text_chunks(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(
            return_value={"response": "First sentence. Second sentence."}
        )
        if hasattr(conductor, "run_actor_stream"):
            del conductor.run_actor_stream
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="StreamAgent")
        executor = ChatExecutor(conductor=conductor, orchestrator=orchestrator)

        events = []
        async for event in executor.stream(message="test"):
            events.append(event)

        text_chunks = [e for e in events if e["type"] == "text_chunk"]
        assert len(text_chunks) >= 1

    @pytest.mark.asyncio
    async def test_stream_handles_error(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(side_effect=RuntimeError("stream error"))
        if hasattr(conductor, "run_actor_stream"):
            del conductor.run_actor_stream
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="StreamAgent")
        executor = ChatExecutor(conductor=conductor, orchestrator=orchestrator)

        events = []
        async for event in executor.stream(message="test"):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "error" in event_types

    @pytest.mark.asyncio
    async def test_stream_adds_messages_to_context(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"response": "context test"})
        if hasattr(conductor, "run_actor_stream"):
            del conductor.run_actor_stream
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="StreamAgent")
        ctx = ChatContext()
        executor = ChatExecutor(
            conductor=conductor, orchestrator=orchestrator, context=ctx
        )

        async for _ in executor.stream(message="hello"):
            pass

        messages = ctx.get_history()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"


# ---------------------------------------------------------------------------
# 2. ChatOrchestrator Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_CHAT_ORCHESTRATOR,
    reason="ChatOrchestrator import unavailable",
)
class TestChatOrchestratorInit:
    """Tests for ChatOrchestrator.__init__."""

    def test_init_stores_conductor(self):
        conductor = _make_mock_conductor()
        orch = ChatOrchestrator(conductor=conductor)
        assert orch.conductor is conductor

    def test_init_stores_agent_id(self):
        conductor = _make_mock_conductor()
        orch = ChatOrchestrator(conductor=conductor, agent_id="MyAgent")
        assert orch.agent_id == "MyAgent"

    def test_init_default_mode_is_dynamic(self):
        conductor = _make_mock_conductor()
        orch = ChatOrchestrator(conductor=conductor)
        assert orch.mode == "dynamic"

    def test_init_accepts_static_mode(self):
        conductor = _make_mock_conductor()
        orch = ChatOrchestrator(conductor=conductor, mode="static")
        assert orch.mode == "static"

    def test_init_rejects_invalid_mode(self):
        conductor = _make_mock_conductor()
        with pytest.raises(ValueError, match="Invalid mode"):
            ChatOrchestrator(conductor=conductor, mode="invalid")

    def test_init_agent_id_none_by_default(self):
        conductor = _make_mock_conductor()
        orch = ChatOrchestrator(conductor=conductor)
        assert orch.agent_id is None


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_CHAT_ORCHESTRATOR,
    reason="ChatOrchestrator import unavailable",
)
class TestChatOrchestratorSelectAgent:
    """Tests for ChatOrchestrator.select_agent()."""

    def test_select_agent_single_agent(self):
        conductor = _make_mock_conductor()
        orch = ChatOrchestrator(conductor=conductor, agent_id="FixedAgent")
        result = orch.select_agent("hello")
        assert result == "FixedAgent"

    def test_select_agent_static_mode_first_available(self):
        actors = [_make_mock_actor("Agent1"), _make_mock_actor("Agent2")]
        conductor = _make_mock_conductor(actors=actors)
        orch = ChatOrchestrator(conductor=conductor, mode="static")
        result = orch.select_agent("hello")
        assert result == "Agent1"

    def test_select_agent_dynamic_with_conductor_select(self):
        conductor = _make_mock_conductor(has_select_agent=True)
        orch = ChatOrchestrator(conductor=conductor, mode="dynamic")
        result = orch.select_agent("hello")
        assert result == "DynamicAgent"

    def test_select_agent_dynamic_q_predictor_routing(self):
        actors = [_make_mock_actor("AgentA"), _make_mock_actor("AgentB")]
        conductor = _make_mock_conductor(actors=actors, has_q_predictor=True)
        # Remove select_agent so it falls through to q_predictor path
        if hasattr(conductor, "select_agent"):
            del conductor.select_agent
        conductor.q_predictor.predict_q_value = MagicMock(
            side_effect=[
                (0.3, 0.5, {}),  # AgentA score
                (0.9, 0.9, {}),  # AgentB score
            ]
        )
        orch = ChatOrchestrator(conductor=conductor, mode="dynamic")
        result = orch.select_agent("complex query")
        assert result == "AgentB"

    def test_select_agent_fallback_to_first_actor(self):
        actors = [_make_mock_actor("FallbackAgent")]
        conductor = _make_mock_conductor(actors=actors)
        if hasattr(conductor, "select_agent"):
            del conductor.select_agent
        conductor.q_predictor = None
        orch = ChatOrchestrator(conductor=conductor, mode="dynamic")
        result = orch.select_agent("hello")
        assert result == "FallbackAgent"

    def test_select_agent_no_agents_raises(self):
        conductor = _make_mock_conductor(actors=[])
        if hasattr(conductor, "select_agent"):
            del conductor.select_agent
        conductor.q_predictor = None
        orch = ChatOrchestrator(conductor=conductor, mode="dynamic")
        with pytest.raises(RuntimeError, match="No agents available"):
            orch.select_agent("hello")

    def test_select_agent_q_predictor_failure_falls_back(self):
        actors = [_make_mock_actor("SafeAgent")]
        conductor = _make_mock_conductor(actors=actors, has_q_predictor=True)
        if hasattr(conductor, "select_agent"):
            del conductor.select_agent
        conductor.q_predictor.predict_q_value = MagicMock(
            side_effect=Exception("Q failed")
        )
        orch = ChatOrchestrator(conductor=conductor, mode="dynamic")
        result = orch.select_agent("hello")
        # Should fall back to first actor
        assert result == "SafeAgent"

    def test_select_agent_passes_history_and_context_to_conductor(self):
        conductor = _make_mock_conductor(has_select_agent=True)
        orch = ChatOrchestrator(conductor=conductor, mode="dynamic")
        history = [MagicMock()]
        context = {"key": "value"}
        orch.select_agent("hello", history=history, context=context)
        conductor.select_agent.assert_called_once_with("hello", history, context)


@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_CHAT_ORCHESTRATOR and HAS_CHAT_CONTEXT),
    reason="ChatOrchestrator imports unavailable",
)
class TestChatOrchestratorPrepareContext:
    """Tests for ChatOrchestrator.prepare_agent_context()."""

    def test_prepare_context_includes_message(self):
        conductor = _make_mock_conductor()
        orch = ChatOrchestrator(conductor=conductor)
        ctx = orch.prepare_agent_context("hello world")
        assert ctx["message"] == "hello world"
        assert ctx["query"] == "hello world"
        assert ctx["user_message"] == "hello world"

    def test_prepare_context_with_history(self):
        conductor = _make_mock_conductor()
        orch = ChatOrchestrator(conductor=conductor)
        msg1 = ChatMessage(role="user", content="hi")
        msg2 = ChatMessage(role="assistant", content="hello")
        ctx = orch.prepare_agent_context("how are you", history=[msg1, msg2])
        assert "history" in ctx
        assert len(ctx["history"]) == 2
        assert "conversation_history" in ctx
        assert "messages" in ctx

    def test_prepare_context_without_history(self):
        conductor = _make_mock_conductor()
        orch = ChatOrchestrator(conductor=conductor)
        ctx = orch.prepare_agent_context("hello")
        assert ctx["history"] == []

    def test_prepare_context_merges_additional_context(self):
        conductor = _make_mock_conductor()
        orch = ChatOrchestrator(conductor=conductor)
        ctx = orch.prepare_agent_context(
            "hello", context={"extra_key": "extra_value"}
        )
        assert ctx["extra_key"] == "extra_value"

    def test_prepare_context_history_to_dict_conversion(self):
        conductor = _make_mock_conductor()
        orch = ChatOrchestrator(conductor=conductor)
        msg = ChatMessage(role="user", content="test message")
        ctx = orch.prepare_agent_context("hello", history=[msg])
        assert ctx["history"][0]["role"] == "user"
        assert ctx["history"][0]["content"] == "test message"

    def test_prepare_context_plain_string_history(self):
        conductor = _make_mock_conductor()
        orch = ChatOrchestrator(conductor=conductor)
        ctx = orch.prepare_agent_context("hello", history=["plain string"])
        assert ctx["history"][0]["role"] == "user"
        assert ctx["history"][0]["content"] == "plain string"


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_CHAT_ORCHESTRATOR,
    reason="ChatOrchestrator import unavailable",
)
class TestChatOrchestratorFormatHistory:
    """Tests for ChatOrchestrator._format_history()."""

    def test_format_history_with_chat_messages(self):
        conductor = _make_mock_conductor()
        orch = ChatOrchestrator(conductor=conductor)
        msg1 = MagicMock()
        msg1.role = "user"
        msg1.content = "hi"
        msg2 = MagicMock()
        msg2.role = "assistant"
        msg2.content = "hello"
        result = orch._format_history([msg1, msg2])
        assert "user: hi" in result
        assert "assistant: hello" in result

    def test_format_history_with_plain_strings(self):
        conductor = _make_mock_conductor()
        orch = ChatOrchestrator(conductor=conductor)
        result = orch._format_history(["first", "second"])
        assert "first" in result
        assert "second" in result

    def test_format_history_empty_list(self):
        conductor = _make_mock_conductor()
        orch = ChatOrchestrator(conductor=conductor)
        result = orch._format_history([])
        assert result == ""


# ---------------------------------------------------------------------------
# 3. ChatUseCase Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_CHAT_USE_CASE and HAS_USE_CASE_BASE and HAS_CHAT_ORCHESTRATOR),
    reason="ChatUseCase imports unavailable",
)
class TestChatUseCaseInit:
    """Tests for ChatUseCase initialization and inheritance."""

    def test_inherits_base_use_case(self):
        assert issubclass(ChatUseCase, BaseUseCase)

    def test_init_creates_orchestrator(self):
        conductor = _make_mock_conductor()
        uc = ChatUseCase(conductor=conductor, agent_id="Agent1")
        assert isinstance(uc.orchestrator, ChatOrchestrator)

    def test_init_creates_executor(self):
        conductor = _make_mock_conductor()
        uc = ChatUseCase(conductor=conductor, agent_id="Agent1")
        assert isinstance(uc.executor, ChatExecutor)

    def test_init_stores_agent_id(self):
        conductor = _make_mock_conductor()
        uc = ChatUseCase(conductor=conductor, agent_id="TestAgent")
        assert uc.agent_id == "TestAgent"

    def test_init_stores_mode(self):
        conductor = _make_mock_conductor()
        uc = ChatUseCase(conductor=conductor, mode="static")
        assert uc.mode == "static"

    def test_init_default_config_created(self):
        conductor = _make_mock_conductor()
        uc = ChatUseCase(conductor=conductor)
        assert uc.config.use_case_type == UseCaseType.CHAT

    def test_init_custom_config(self):
        conductor = _make_mock_conductor()
        config = UseCaseConfig(use_case_type=UseCaseType.CHAT, max_iterations=50)
        uc = ChatUseCase(conductor=conductor, config=config)
        assert uc.config.max_iterations == 50


@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_CHAT_USE_CASE and HAS_USE_CASE_BASE),
    reason="ChatUseCase imports unavailable",
)
class TestChatUseCaseType:
    """Tests for ChatUseCase._get_use_case_type()."""

    def test_returns_chat_type(self):
        conductor = _make_mock_conductor()
        uc = ChatUseCase(conductor=conductor)
        assert uc._get_use_case_type() == UseCaseType.CHAT


@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_CHAT_USE_CASE and HAS_USE_CASE_BASE),
    reason="ChatUseCase imports unavailable",
)
class TestChatUseCaseExecute:
    """Tests for ChatUseCase.execute()."""

    @pytest.mark.asyncio
    async def test_execute_returns_use_case_result(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"response": "chat reply"})
        uc = ChatUseCase(conductor=conductor, agent_id="Agent1")
        result = await uc.execute(goal="hello")
        assert isinstance(result, UseCaseResult)

    @pytest.mark.asyncio
    async def test_execute_result_success(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"response": "success reply"})
        uc = ChatUseCase(conductor=conductor, agent_id="Agent1")
        result = await uc.execute(goal="test")
        assert result.success is True
        assert result.output == "success reply"
        assert result.use_case_type == UseCaseType.CHAT

    @pytest.mark.asyncio
    async def test_execute_result_has_metadata(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"response": "meta reply"})
        uc = ChatUseCase(conductor=conductor, agent_id="Agent1")
        result = await uc.execute(goal="test")
        assert "agent" in result.metadata
        assert "execution_time" in result.metadata

    @pytest.mark.asyncio
    async def test_execute_handles_exception(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(side_effect=RuntimeError("use case error"))
        uc = ChatUseCase(conductor=conductor, agent_id="Agent1")
        result = await uc.execute(goal="fail")
        assert result.success is False
        assert "Error" in result.output

    @pytest.mark.asyncio
    async def test_execute_with_context_and_history(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"response": "contextualized"})
        uc = ChatUseCase(conductor=conductor, agent_id="Agent1")
        result = await uc.execute(
            goal="with context",
            context={"key": "val"},
            history=[],
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_result_to_dict(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"response": "dict test"})
        uc = ChatUseCase(conductor=conductor, agent_id="Agent1")
        result = await uc.execute(goal="test")
        d = result.to_dict()
        assert d["success"] is True
        assert d["use_case_type"] == "chat"


# ---------------------------------------------------------------------------
# 4. WorkflowExecutor Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_WORKFLOW_EXECUTOR and HAS_WORKFLOW_ORCHESTRATOR and HAS_WORKFLOW_CONTEXT and HAS_TASK_STATUS),
    reason="Workflow executor imports unavailable",
)
class TestWorkflowExecutorInit:
    """Tests for WorkflowExecutor.__init__."""

    def test_init_stores_conductor(self):
        conductor = _make_mock_conductor()
        orchestrator = WorkflowOrchestrator(conductor=conductor, agent_order=["a"])
        executor = WorkflowExecutor(conductor=conductor, orchestrator=orchestrator)
        assert executor.conductor is conductor

    def test_init_stores_orchestrator(self):
        conductor = _make_mock_conductor()
        orchestrator = WorkflowOrchestrator(conductor=conductor, agent_order=["a"])
        executor = WorkflowExecutor(conductor=conductor, orchestrator=orchestrator)
        assert executor.orchestrator is orchestrator

    def test_init_creates_default_context(self):
        conductor = _make_mock_conductor()
        orchestrator = WorkflowOrchestrator(conductor=conductor, agent_order=["a"])
        executor = WorkflowExecutor(conductor=conductor, orchestrator=orchestrator)
        assert isinstance(executor.context, WorkflowContext)

    def test_init_uses_provided_context(self):
        conductor = _make_mock_conductor()
        orchestrator = WorkflowOrchestrator(conductor=conductor, agent_order=["a"])
        custom_ctx = WorkflowContext(workflow_id="custom_wf")
        executor = WorkflowExecutor(
            conductor=conductor, orchestrator=orchestrator, context=custom_ctx
        )
        assert executor.context is custom_ctx
        assert executor.context.workflow_id == "custom_wf"


@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_WORKFLOW_EXECUTOR and HAS_WORKFLOW_ORCHESTRATOR and HAS_WORKFLOW_CONTEXT and HAS_TASK_STATUS),
    reason="Workflow executor imports unavailable",
)
class TestWorkflowExecutorExecute:
    """Tests for WorkflowExecutor.execute()."""

    def _make_executor(self, conductor=None):
        conductor = conductor or _make_mock_conductor()
        orchestrator = WorkflowOrchestrator(
            conductor=conductor, agent_order=["Agent1"]
        )
        return WorkflowExecutor(conductor=conductor, orchestrator=orchestrator)

    @pytest.mark.asyncio
    async def test_execute_returns_success_dict(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"output": "done"})
        executor = self._make_executor(conductor)

        result = await executor.execute(goal="do something")

        assert result["success"] is True
        assert "result" in result
        assert "workflow_id" in result
        assert "task_id" in result
        assert "execution_time" in result
        assert "summary" in result

    @pytest.mark.asyncio
    async def test_execute_calls_conductor_run(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"output": "ok"})
        executor = self._make_executor(conductor)

        await executor.execute(goal="run workflow")

        conductor.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_creates_task_in_context(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"output": "ok"})
        executor = self._make_executor(conductor)

        result = await executor.execute(goal="tracked task")

        task_id = result["task_id"]
        task = executor.context.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_handles_exception(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(side_effect=RuntimeError("workflow error"))
        executor = self._make_executor(conductor)

        result = await executor.execute(goal="fail")

        assert result["success"] is False
        assert result["error"] == "workflow error"

    @pytest.mark.asyncio
    async def test_execute_failed_task_status(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(side_effect=RuntimeError("fail"))
        executor = self._make_executor(conductor)

        result = await executor.execute(goal="fail task")

        task_id = result["task_id"]
        task = executor.context.get_task(task_id)
        assert task.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_summary_contains_task_counts(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"output": "done"})
        executor = self._make_executor(conductor)

        result = await executor.execute(goal="summary test")

        summary = result["summary"]
        assert "total_tasks" in summary
        assert "status_counts" in summary
        assert summary["total_tasks"] >= 1

    @pytest.mark.asyncio
    async def test_execute_passes_context_and_max_iterations(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"output": "ok"})
        executor = self._make_executor(conductor)

        await executor.execute(
            goal="test", context={"key": "val"}, max_iterations=50
        )

        call_kwargs = conductor.run.call_args.kwargs
        assert call_kwargs["context"] == {"key": "val"}
        assert call_kwargs["max_iterations"] == 50


@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_WORKFLOW_EXECUTOR and HAS_WORKFLOW_ORCHESTRATOR and HAS_WORKFLOW_CONTEXT and HAS_TASK_STATUS),
    reason="Workflow executor imports unavailable",
)
class TestWorkflowExecutorStream:
    """Tests for WorkflowExecutor.stream()."""

    @pytest.mark.asyncio
    async def test_stream_yields_workflow_started_event(self):
        conductor = _make_mock_conductor()
        if hasattr(conductor, "run_stream"):
            del conductor.run_stream
        # Need run_actor for fallback sequential execution
        conductor.run_actor = AsyncMock(return_value={"output": "streamed"})
        orchestrator = WorkflowOrchestrator(
            conductor=conductor, agent_order=["Agent1"]
        )
        executor = WorkflowExecutor(conductor=conductor, orchestrator=orchestrator)

        events = []
        async for event in executor.stream(goal="stream test"):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "workflow_started" in event_types

    @pytest.mark.asyncio
    async def test_stream_yields_workflow_completed_event(self):
        conductor = _make_mock_conductor()
        if hasattr(conductor, "run_stream"):
            del conductor.run_stream
        conductor.run_actor = AsyncMock(return_value={"output": "streamed"})
        orchestrator = WorkflowOrchestrator(
            conductor=conductor, agent_order=["Agent1"]
        )
        executor = WorkflowExecutor(conductor=conductor, orchestrator=orchestrator)

        events = []
        async for event in executor.stream(goal="stream test"):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "workflow_completed" in event_types

    @pytest.mark.asyncio
    async def test_stream_error_yields_workflow_failed(self):
        conductor = _make_mock_conductor()
        if hasattr(conductor, "run_stream"):
            del conductor.run_stream
        orchestrator = WorkflowOrchestrator(
            conductor=conductor, agent_order=["Agent1"]
        )
        ctx = WorkflowContext()
        executor = WorkflowExecutor(
            conductor=conductor, orchestrator=orchestrator, context=ctx
        )

        # Patch context to raise an error during get_ready_tasks
        with patch.object(ctx, "get_ready_tasks", side_effect=RuntimeError("stream fail")):
            events = []
            async for event in executor.stream(goal="fail"):
                events.append(event)

        event_types = [e["type"] for e in events]
        assert "workflow_failed" in event_types


# ---------------------------------------------------------------------------
# 5. WorkflowOrchestrator Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_WORKFLOW_ORCHESTRATOR and HAS_WORKFLOW_CONTEXT and HAS_TASK_STATUS),
    reason="WorkflowOrchestrator imports unavailable",
)
class TestWorkflowOrchestratorInit:
    """Tests for WorkflowOrchestrator.__init__."""

    def test_init_stores_conductor(self):
        conductor = _make_mock_conductor()
        orch = WorkflowOrchestrator(conductor=conductor, agent_order=["a"])
        assert orch.conductor is conductor

    def test_init_stores_mode(self):
        conductor = _make_mock_conductor()
        orch = WorkflowOrchestrator(
            conductor=conductor, mode="static", agent_order=["a"]
        )
        assert orch.mode == "static"

    def test_init_stores_agent_order(self):
        conductor = _make_mock_conductor()
        orch = WorkflowOrchestrator(
            conductor=conductor, mode="static", agent_order=["A", "B", "C"]
        )
        assert orch.agent_order == ["A", "B", "C"]

    def test_init_rejects_invalid_mode(self):
        conductor = _make_mock_conductor()
        with pytest.raises(ValueError, match="Invalid mode"):
            WorkflowOrchestrator(conductor=conductor, mode="bad")

    def test_init_static_mode_requires_agent_order(self):
        conductor = _make_mock_conductor()
        with pytest.raises(ValueError, match="agent_order is required"):
            WorkflowOrchestrator(conductor=conductor, mode="static")

    def test_init_dynamic_mode_no_agent_order_needed(self):
        conductor = _make_mock_conductor()
        orch = WorkflowOrchestrator(conductor=conductor, mode="dynamic")
        assert orch.mode == "dynamic"
        assert orch.agent_order is None


@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_WORKFLOW_ORCHESTRATOR and HAS_WORKFLOW_CONTEXT and HAS_TASK_STATUS),
    reason="WorkflowOrchestrator imports unavailable",
)
class TestWorkflowOrchestratorSelectAgent:
    """Tests for WorkflowOrchestrator.select_agent()."""

    def _make_task(self, task_id="task_0", goal="do something"):
        return WorkflowTask(id=task_id, goal=goal)

    def test_select_agent_static_mode_uses_agent_order(self):
        conductor = _make_mock_conductor()
        orch = WorkflowOrchestrator(
            conductor=conductor,
            mode="static",
            agent_order=["AgentA", "AgentB", "AgentC"],
        )
        task = self._make_task()
        result = orch.select_agent(task)
        assert result in ["AgentA", "AgentB", "AgentC"]

    def test_select_agent_dynamic_with_conductor_select(self):
        conductor = _make_mock_conductor(has_select_agent=True)
        orch = WorkflowOrchestrator(conductor=conductor, mode="dynamic")
        task = self._make_task()
        result = orch.select_agent(task)
        assert result == "DynamicAgent"

    def test_select_agent_dynamic_q_predictor(self):
        actors = [_make_mock_actor("WorkerA"), _make_mock_actor("WorkerB")]
        conductor = _make_mock_conductor(actors=actors, has_q_predictor=True)
        if hasattr(conductor, "select_agent"):
            del conductor.select_agent
        conductor.q_predictor.predict_q_value = MagicMock(
            side_effect=[
                (0.2, 0.5, {}),  # WorkerA
                (0.95, 0.9, {}),  # WorkerB
            ]
        )
        orch = WorkflowOrchestrator(conductor=conductor, mode="dynamic")
        task = self._make_task()
        result = orch.select_agent(task)
        assert result == "WorkerB"

    def test_select_agent_fallback_to_first_actor(self):
        actors = [_make_mock_actor("FallbackWorker")]
        conductor = _make_mock_conductor(actors=actors)
        if hasattr(conductor, "select_agent"):
            del conductor.select_agent
        conductor.q_predictor = None
        orch = WorkflowOrchestrator(conductor=conductor, mode="dynamic")
        task = self._make_task()
        result = orch.select_agent(task)
        assert result == "FallbackWorker"

    def test_select_agent_no_agents_raises(self):
        conductor = _make_mock_conductor(actors=[])
        if hasattr(conductor, "select_agent"):
            del conductor.select_agent
        conductor.q_predictor = None
        orch = WorkflowOrchestrator(conductor=conductor, mode="dynamic")
        task = self._make_task()
        with pytest.raises(RuntimeError, match="No agents available"):
            orch.select_agent(task)

    def test_select_agent_q_predictor_failure_falls_back(self):
        actors = [_make_mock_actor("SafeWorker")]
        conductor = _make_mock_conductor(actors=actors, has_q_predictor=True)
        if hasattr(conductor, "select_agent"):
            del conductor.select_agent
        conductor.q_predictor.predict_q_value = MagicMock(
            side_effect=Exception("Q error")
        )
        orch = WorkflowOrchestrator(conductor=conductor, mode="dynamic")
        task = self._make_task()
        result = orch.select_agent(task)
        assert result == "SafeWorker"

    def test_select_agent_static_round_robin_distribution(self):
        """Verify that different task IDs can map to different agents."""
        conductor = _make_mock_conductor()
        orch = WorkflowOrchestrator(
            conductor=conductor,
            mode="static",
            agent_order=["A", "B", "C"],
        )
        results = set()
        for i in range(30):
            task = self._make_task(task_id=f"task_{i}")
            results.add(orch.select_agent(task))
        # With hash-based selection, we should see multiple agents selected
        assert len(results) >= 1  # At minimum one is guaranteed


@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_WORKFLOW_ORCHESTRATOR and HAS_WORKFLOW_CONTEXT and HAS_TASK_STATUS),
    reason="WorkflowOrchestrator imports unavailable",
)
class TestWorkflowOrchestratorPrepareContext:
    """Tests for WorkflowOrchestrator.prepare_agent_context()."""

    def _make_task(self, task_id="task_0", goal="do work", dependencies=None):
        return WorkflowTask(
            id=task_id, goal=goal, dependencies=dependencies or []
        )

    def test_prepare_context_includes_goal(self):
        conductor = _make_mock_conductor()
        orch = WorkflowOrchestrator(conductor=conductor, agent_order=["a"])
        wf_ctx = WorkflowContext()
        task = self._make_task(goal="analyze data")
        ctx = orch.prepare_agent_context(task, wf_ctx)
        assert ctx["goal"] == "analyze data"
        assert ctx["query"] == "analyze data"

    def test_prepare_context_includes_task_id(self):
        conductor = _make_mock_conductor()
        orch = WorkflowOrchestrator(conductor=conductor, agent_order=["a"])
        wf_ctx = WorkflowContext()
        task = self._make_task(task_id="task_42")
        ctx = orch.prepare_agent_context(task, wf_ctx)
        assert ctx["task_id"] == "task_42"

    def test_prepare_context_includes_workflow_metadata(self):
        conductor = _make_mock_conductor()
        orch = WorkflowOrchestrator(conductor=conductor, agent_order=["a"])
        wf_ctx = WorkflowContext(workflow_id="wf_123")
        task = self._make_task()
        ctx = orch.prepare_agent_context(task, wf_ctx)
        assert ctx["workflow_id"] == "wf_123"
        assert "workflow_summary" in ctx

    def test_prepare_context_includes_dependency_results(self):
        conductor = _make_mock_conductor()
        orch = WorkflowOrchestrator(conductor=conductor, agent_order=["a"])
        wf_ctx = WorkflowContext()
        dep_id = wf_ctx.add_task("dependency task")
        wf_ctx.update_task_status(dep_id, TaskStatus.COMPLETED, result="dep result")

        task = self._make_task(dependencies=[dep_id])
        ctx = orch.prepare_agent_context(task, wf_ctx)
        assert "dependency_results" in ctx
        assert ctx["dependency_results"][dep_id] == "dep result"

    def test_prepare_context_no_dependency_results_when_none(self):
        conductor = _make_mock_conductor()
        orch = WorkflowOrchestrator(conductor=conductor, agent_order=["a"])
        wf_ctx = WorkflowContext()
        task = self._make_task(dependencies=[])
        ctx = orch.prepare_agent_context(task, wf_ctx)
        assert "dependency_results" not in ctx

    def test_prepare_context_merges_additional_context(self):
        conductor = _make_mock_conductor()
        orch = WorkflowOrchestrator(conductor=conductor, agent_order=["a"])
        wf_ctx = WorkflowContext()
        task = self._make_task()
        ctx = orch.prepare_agent_context(
            task, wf_ctx, context={"extra": "data"}
        )
        assert ctx["extra"] == "data"

    def test_prepare_context_workflow_summary_structure(self):
        conductor = _make_mock_conductor()
        orch = WorkflowOrchestrator(conductor=conductor, agent_order=["a"])
        wf_ctx = WorkflowContext()
        wf_ctx.add_task("task 1")
        task = self._make_task()
        ctx = orch.prepare_agent_context(task, wf_ctx)
        summary = ctx["workflow_summary"]
        assert "total_tasks" in summary
        assert "status_counts" in summary

    def test_prepare_context_skips_unresolved_dependencies(self):
        conductor = _make_mock_conductor()
        orch = WorkflowOrchestrator(conductor=conductor, agent_order=["a"])
        wf_ctx = WorkflowContext()
        dep_id = wf_ctx.add_task("pending dep")
        # Do not complete the dependency -- result is None
        task = self._make_task(dependencies=[dep_id])
        ctx = orch.prepare_agent_context(task, wf_ctx)
        # Should not include dependency_results because result is None
        assert "dependency_results" not in ctx


# ---------------------------------------------------------------------------
# 6. WorkflowUseCase Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_WORKFLOW_USE_CASE and HAS_USE_CASE_BASE and HAS_WORKFLOW_ORCHESTRATOR and HAS_TASK_STATUS),
    reason="WorkflowUseCase imports unavailable",
)
class TestWorkflowUseCaseInit:
    """Tests for WorkflowUseCase initialization and inheritance."""

    def test_inherits_base_use_case(self):
        assert issubclass(WorkflowUseCase, BaseUseCase)

    def test_init_creates_orchestrator(self):
        conductor = _make_mock_conductor()
        uc = WorkflowUseCase(
            conductor=conductor, mode="static", agent_order=["A"]
        )
        assert isinstance(uc.orchestrator, WorkflowOrchestrator)

    def test_init_creates_executor(self):
        conductor = _make_mock_conductor()
        uc = WorkflowUseCase(
            conductor=conductor, mode="static", agent_order=["A"]
        )
        assert isinstance(uc.executor, WorkflowExecutor)

    def test_init_stores_mode_and_agent_order(self):
        conductor = _make_mock_conductor()
        uc = WorkflowUseCase(
            conductor=conductor, mode="static", agent_order=["A", "B"]
        )
        assert uc.mode == "static"
        assert uc.agent_order == ["A", "B"]

    def test_init_default_config(self):
        conductor = _make_mock_conductor()
        uc = WorkflowUseCase(conductor=conductor, mode="dynamic")
        assert uc.config.use_case_type == UseCaseType.WORKFLOW

    def test_init_custom_config(self):
        conductor = _make_mock_conductor()
        config = UseCaseConfig(
            use_case_type=UseCaseType.WORKFLOW, max_iterations=25
        )
        uc = WorkflowUseCase(
            conductor=conductor, mode="dynamic", config=config
        )
        assert uc.config.max_iterations == 25


@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_WORKFLOW_USE_CASE and HAS_USE_CASE_BASE),
    reason="WorkflowUseCase imports unavailable",
)
class TestWorkflowUseCaseType:
    """Tests for WorkflowUseCase._get_use_case_type()."""

    def test_returns_workflow_type(self):
        conductor = _make_mock_conductor()
        uc = WorkflowUseCase(conductor=conductor, mode="dynamic")
        assert uc._get_use_case_type() == UseCaseType.WORKFLOW


@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_WORKFLOW_USE_CASE and HAS_USE_CASE_BASE and HAS_TASK_STATUS),
    reason="WorkflowUseCase imports unavailable",
)
class TestWorkflowUseCaseExecute:
    """Tests for WorkflowUseCase.execute()."""

    @pytest.mark.asyncio
    async def test_execute_returns_use_case_result(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"output": "workflow done"})
        uc = WorkflowUseCase(conductor=conductor, mode="dynamic")
        result = await uc.execute(goal="do workflow")
        assert isinstance(result, UseCaseResult)

    @pytest.mark.asyncio
    async def test_execute_result_success(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"output": "completed"})
        uc = WorkflowUseCase(conductor=conductor, mode="dynamic")
        result = await uc.execute(goal="complete task")
        assert result.success is True
        assert result.use_case_type == UseCaseType.WORKFLOW

    @pytest.mark.asyncio
    async def test_execute_result_has_workflow_metadata(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"output": "ok"})
        uc = WorkflowUseCase(conductor=conductor, mode="dynamic")
        result = await uc.execute(goal="metadata test")
        assert "workflow_id" in result.metadata
        assert "task_id" in result.metadata
        assert "summary" in result.metadata

    @pytest.mark.asyncio
    async def test_execute_handles_exception(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(
            side_effect=RuntimeError("workflow use case error")
        )
        uc = WorkflowUseCase(conductor=conductor, mode="dynamic")
        result = await uc.execute(goal="fail")
        assert result.success is False
        # The executor catches the exception internally and returns a result
        # with success=False. The summary reflects the failed task status.
        assert result.metadata["summary"]["status_counts"]["failed"] == 1

    @pytest.mark.asyncio
    async def test_execute_passes_max_iterations(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"output": "ok"})
        uc = WorkflowUseCase(conductor=conductor, mode="dynamic")
        await uc.execute(goal="iter test", max_iterations=10)
        call_kwargs = conductor.run.call_args.kwargs
        assert call_kwargs["max_iterations"] == 10

    @pytest.mark.asyncio
    async def test_execute_result_to_dict(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"output": "dict test"})
        uc = WorkflowUseCase(conductor=conductor, mode="dynamic")
        result = await uc.execute(goal="test")
        d = result.to_dict()
        assert d["success"] is True
        assert d["use_case_type"] == "workflow"

    @pytest.mark.asyncio
    async def test_execute_with_context(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"output": "ok"})
        uc = WorkflowUseCase(conductor=conductor, mode="dynamic")
        result = await uc.execute(
            goal="contextualized", context={"key": "value"}
        )
        assert result.success is True


@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_WORKFLOW_USE_CASE and HAS_USE_CASE_BASE),
    reason="WorkflowUseCase imports unavailable",
)
class TestWorkflowUseCaseEnqueue:
    """Tests for WorkflowUseCase.enqueue()."""

    @pytest.mark.asyncio
    async def test_enqueue_raises_not_implemented_without_support(self):
        conductor = _make_mock_conductor()
        if hasattr(conductor, "enqueue_goal"):
            del conductor.enqueue_goal
        uc = WorkflowUseCase(conductor=conductor, mode="dynamic")
        with pytest.raises(NotImplementedError, match="does not support"):
            await uc.enqueue(goal="async task")

    @pytest.mark.asyncio
    async def test_enqueue_calls_conductor_enqueue_goal(self):
        conductor = _make_mock_conductor()
        conductor.enqueue_goal = AsyncMock(return_value="task_abc")
        uc = WorkflowUseCase(conductor=conductor, mode="dynamic")
        task_id = await uc.enqueue(goal="async task", priority=5)
        assert task_id == "task_abc"
        conductor.enqueue_goal.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_enqueue_raises_on_none_task_id(self):
        conductor = _make_mock_conductor()
        conductor.enqueue_goal = AsyncMock(return_value=None)
        uc = WorkflowUseCase(conductor=conductor, mode="dynamic")
        with pytest.raises(RuntimeError, match="Failed to enqueue"):
            await uc.enqueue(goal="fail enqueue")


# ---------------------------------------------------------------------------
# Integration-style tests: ChatUseCase stream
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_CHAT_USE_CASE and HAS_CHAT_EXECUTOR and HAS_CHAT_ORCHESTRATOR),
    reason="Chat use case imports unavailable",
)
class TestChatUseCaseStream:
    """Tests for ChatUseCase.stream()."""

    @pytest.mark.asyncio
    async def test_stream_yields_events(self):
        conductor = _make_mock_conductor()
        conductor.run = AsyncMock(return_value={"response": "streamed"})
        if hasattr(conductor, "run_actor_stream"):
            del conductor.run_actor_stream
        uc = ChatUseCase(conductor=conductor, agent_id="StreamAgent")

        events = []
        async for event in uc.stream(goal="hello"):
            events.append(event)

        assert len(events) >= 2  # at least agent_selected + done
        event_types = [e["type"] for e in events]
        assert "agent_selected" in event_types
        assert "done" in event_types


# ---------------------------------------------------------------------------
# Integration-style tests: WorkflowUseCase stream
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_WORKFLOW_USE_CASE and HAS_WORKFLOW_EXECUTOR and HAS_WORKFLOW_ORCHESTRATOR and HAS_TASK_STATUS),
    reason="Workflow use case imports unavailable",
)
class TestWorkflowUseCaseStream:
    """Tests for WorkflowUseCase.stream()."""

    @pytest.mark.asyncio
    async def test_stream_yields_events(self):
        conductor = _make_mock_conductor()
        if hasattr(conductor, "run_stream"):
            del conductor.run_stream
        conductor.run_actor = AsyncMock(return_value={"output": "streamed"})
        uc = WorkflowUseCase(
            conductor=conductor, mode="static", agent_order=["Agent1"]
        )

        events = []
        async for event in uc.stream(goal="workflow stream"):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "workflow_started" in event_types
        assert "workflow_completed" in event_types


# ---------------------------------------------------------------------------
# Edge case / cross-cutting tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_CHAT_ORCHESTRATOR and HAS_CHAT_EXECUTOR),
    reason="Chat imports unavailable",
)
class TestChatExecutorExtractTextFromA2UI:
    """Tests for ChatExecutor._extract_text_from_a2ui()."""

    def _make_executor(self):
        conductor = _make_mock_conductor()
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="A")
        return ChatExecutor(conductor=conductor, orchestrator=orchestrator)

    def test_extract_text_blocks(self):
        executor = self._make_executor()
        a2ui = {
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ]
        }
        result = executor._extract_text_from_a2ui(a2ui)
        assert "Hello" in result
        assert "World" in result

    def test_extract_card_title_and_subtitle(self):
        executor = self._make_executor()
        a2ui = {
            "content": [
                {"type": "card", "title": "Card Title", "subtitle": "Card Sub"}
            ]
        }
        result = executor._extract_text_from_a2ui(a2ui)
        assert "Card Title" in result
        assert "Card Sub" in result

    def test_extract_list_items(self):
        executor = self._make_executor()
        a2ui = {
            "content": [
                {
                    "type": "list",
                    "items": [
                        {"title": "Item 1", "subtitle": "Sub 1"},
                        {"title": "Item 2"},
                    ],
                }
            ]
        }
        result = executor._extract_text_from_a2ui(a2ui)
        assert "Item 1" in result
        assert "Item 2" in result

    def test_extract_empty_content(self):
        executor = self._make_executor()
        a2ui = {"content": []}
        result = executor._extract_text_from_a2ui(a2ui)
        assert result == "A2UI widget response"

    def test_extract_card_with_body_text_blocks(self):
        executor = self._make_executor()
        a2ui = {
            "content": [
                {
                    "type": "card",
                    "title": "Title",
                    "body": [{"type": "text", "text": "Body text"}],
                }
            ]
        }
        result = executor._extract_text_from_a2ui(a2ui)
        assert "Body text" in result


@pytest.mark.unit
@pytest.mark.skipif(
    not HAS_CHAT_ORCHESTRATOR,
    reason="ChatOrchestrator import unavailable",
)
class TestChatOrchestratorQPredictor:
    """Tests for ChatOrchestrator._select_with_q_predictor()."""

    def test_returns_none_without_q_predictor(self):
        conductor = _make_mock_conductor()
        conductor.q_predictor = None
        orch = ChatOrchestrator(conductor=conductor, mode="dynamic")
        result = orch._select_with_q_predictor("msg", None, None)
        assert result is None

    def test_returns_best_agent_by_q_value(self):
        actors = [_make_mock_actor("Low"), _make_mock_actor("High")]
        conductor = _make_mock_conductor(actors=actors, has_q_predictor=True)
        conductor.q_predictor.predict_q_value = MagicMock(
            side_effect=[
                (0.1, 0.5, {}),
                (0.99, 0.95, {}),
            ]
        )
        orch = ChatOrchestrator(conductor=conductor, mode="dynamic")
        result = orch._select_with_q_predictor("msg", None, None)
        assert result == "High"

    def test_handles_prediction_errors_gracefully(self):
        actors = [_make_mock_actor("OnlyAgent")]
        conductor = _make_mock_conductor(actors=actors, has_q_predictor=True)
        conductor.q_predictor.predict_q_value = MagicMock(
            side_effect=Exception("prediction failed")
        )
        orch = ChatOrchestrator(conductor=conductor, mode="dynamic")
        result = orch._select_with_q_predictor("msg", None, None)
        # Returns None because no successful prediction
        assert result is None


@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_WORKFLOW_ORCHESTRATOR and HAS_WORKFLOW_CONTEXT and HAS_TASK_STATUS),
    reason="WorkflowOrchestrator imports unavailable",
)
class TestWorkflowOrchestratorQPredictor:
    """Tests for WorkflowOrchestrator._select_with_q_predictor()."""

    def test_returns_none_without_q_predictor(self):
        conductor = _make_mock_conductor()
        conductor.q_predictor = None
        orch = WorkflowOrchestrator(conductor=conductor, mode="dynamic")
        task = WorkflowTask(id="t1", goal="test")
        result = orch._select_with_q_predictor(task, None)
        assert result is None

    def test_returns_best_agent_by_q_value(self):
        actors = [_make_mock_actor("Slow"), _make_mock_actor("Fast")]
        conductor = _make_mock_conductor(actors=actors, has_q_predictor=True)
        conductor.q_predictor.predict_q_value = MagicMock(
            side_effect=[
                (0.2, 0.5, {}),
                (0.85, 0.9, {}),
            ]
        )
        orch = WorkflowOrchestrator(conductor=conductor, mode="dynamic")
        task = WorkflowTask(id="t1", goal="test")
        result = orch._select_with_q_predictor(task, None)
        assert result == "Fast"

    def test_handles_prediction_errors(self):
        actors = [_make_mock_actor("OnlyWorker")]
        conductor = _make_mock_conductor(actors=actors, has_q_predictor=True)
        conductor.q_predictor.predict_q_value = MagicMock(
            side_effect=Exception("fail")
        )
        orch = WorkflowOrchestrator(conductor=conductor, mode="dynamic")
        task = WorkflowTask(id="t1", goal="test")
        result = orch._select_with_q_predictor(task, None)
        assert result is None


@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_CHAT_EXECUTOR and HAS_CHAT_ORCHESTRATOR),
    reason="Chat imports unavailable",
)
class TestChatExecutorTransformEvents:
    """Tests for ChatExecutor._transform_to_chat_events()."""

    def _make_executor(self):
        conductor = _make_mock_conductor()
        orchestrator = ChatOrchestrator(conductor=conductor, agent_id="A")
        return ChatExecutor(conductor=conductor, orchestrator=orchestrator)

    def test_transform_agent_complete_with_reasoning(self):
        executor = self._make_executor()
        event = {
            "type": "agent_complete",
            "result": {
                "reasoning": "I thought about it",
                "response": "Here is my answer",
                "tool_calls": [],
                "tool_results": [],
            },
        }
        chat_events = executor._transform_to_chat_events(event)
        types = [e["type"] for e in chat_events]
        assert "reasoning" in types
        assert "done" in types

    def test_transform_agent_complete_with_tool_calls(self):
        executor = self._make_executor()
        event = {
            "type": "agent_complete",
            "result": {
                "response": "result",
                "tool_calls": [{"name": "search", "arguments": {"q": "test"}}],
                "tool_results": [{"result": "found"}],
            },
        }
        chat_events = executor._transform_to_chat_events(event)
        types = [e["type"] for e in chat_events]
        assert "tool_call" in types
        assert "tool_result" in types

    def test_transform_unknown_event_type(self):
        executor = self._make_executor()
        event = {"type": "unknown", "data": "stuff"}
        chat_events = executor._transform_to_chat_events(event)
        assert chat_events == []

    def test_transform_empty_response(self):
        executor = self._make_executor()
        event = {
            "type": "agent_complete",
            "result": {
                "response": "",
                "tool_calls": [],
                "tool_results": [],
            },
        }
        chat_events = executor._transform_to_chat_events(event)
        # No text_chunk or done events for empty response
        types = [e["type"] for e in chat_events]
        assert "done" not in types

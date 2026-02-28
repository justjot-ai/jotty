"""
Tests for Jotty Core API layer.

Covers:
- RouteResult dataclass (mode_router.py)
- ModeRouter class (mode_router.py)
- JottyAPI unified API (unified.py)
- ChatAPI (chat_api.py)
- WorkflowAPI (workflow_api.py)
"""

from enum import Enum
from unittest.mock import AsyncMock, Mock, patch

import pytest

# --- Guarded imports ---

try:
    from Jotty.core.interface.api.mode_router import RouteResult

    HAS_MODE_ROUTER = True
except ImportError:
    HAS_MODE_ROUTER = False

try:
    from Jotty.core.interface.api.mode_router import ModeRouter

    HAS_MODE_ROUTER_CLASS = True
except ImportError:
    HAS_MODE_ROUTER_CLASS = False

try:
    from Jotty.core.interface.api.mode_router import ExecutionMode

    HAS_EXECUTION_MODE = True
except ImportError:
    HAS_EXECUTION_MODE = False

try:
    from Jotty.core.interface.api.unified import JottyAPI

    HAS_JOTTY_API = True
except ImportError:
    HAS_JOTTY_API = False

try:
    from Jotty.core.interface.api.chat_api import ChatAPI

    HAS_CHAT_API = True
except ImportError:
    HAS_CHAT_API = False

try:
    from Jotty.core.interface.api.workflow_api import WorkflowAPI

    HAS_WORKFLOW_API = True
except ImportError:
    HAS_WORKFLOW_API = False


def _make_execution_mode():
    """Return the real ExecutionMode enum or a mock fallback."""
    if HAS_EXECUTION_MODE:
        return ExecutionMode
    # Provide a lightweight stand-in so tests can still run
    return Enum("ExecutionMode", ["CHAT", "WORKFLOW", "SKILL", "AGENT"])


# ---------------------------------------------------------------------------
# RouteResult tests (1-5)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(not HAS_MODE_ROUTER, reason="RouteResult import failed")
class TestRouteResultDefaults:
    """Verify RouteResult dataclass defaults and __post_init__ behaviour."""

    def _mode(self):
        Mode = _make_execution_mode()
        return Mode.CHAT

    def test_route_result_required_fields_stored(self):
        """RouteResult stores success, content, and mode correctly."""
        mode = self._mode()
        result = RouteResult(success=True, content="hello", mode=mode)
        assert result.success is True
        assert result.content == "hello"
        assert result.mode == mode

    def test_route_result_execution_time_defaults_zero(self):
        """execution_time defaults to 0.0 when not provided."""
        result = RouteResult(success=False, content=None, mode=self._mode())
        assert result.execution_time == 0.0

    def test_route_result_skills_used_defaults_to_empty_list(self):
        """skills_used is initialised to [] by __post_init__ when None."""
        result = RouteResult(success=True, content="x", mode=self._mode())
        assert result.skills_used == []
        assert isinstance(result.skills_used, list)

    def test_route_result_errors_defaults_to_empty_list(self):
        """errors is initialised to [] by __post_init__ when None."""
        result = RouteResult(success=True, content="x", mode=self._mode())
        assert result.errors == []
        assert isinstance(result.errors, list)

    def test_route_result_agents_used_and_metadata_defaults(self):
        """agents_used defaults to [] and metadata defaults to {}."""
        result = RouteResult(success=True, content="x", mode=self._mode())
        assert result.agents_used == []
        assert result.metadata == {}
        assert result.steps_executed == 0
        assert result.error is None
        assert result.stopped_early is False


# ---------------------------------------------------------------------------
# ModeRouter tests (6-8)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(not HAS_MODE_ROUTER_CLASS, reason="ModeRouter import failed")
class TestModeRouter:
    """Verify ModeRouter instantiation and method presence."""

    def test_mode_router_instantiation(self):
        """ModeRouter can be instantiated without arguments."""
        router = ModeRouter()
        assert router is not None
        assert router._initialized is False

    def test_mode_router_has_route_method(self):
        """ModeRouter exposes an async route() method."""
        router = ModeRouter()
        assert hasattr(router, "route")
        assert callable(router.route)

    def test_mode_router_has_chat_method(self):
        """ModeRouter exposes an async chat() convenience method."""
        router = ModeRouter()
        assert hasattr(router, "chat")
        assert callable(router.chat)


# ---------------------------------------------------------------------------
# JottyAPI tests (9-12)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(not HAS_JOTTY_API, reason="JottyAPI import failed")
class TestJottyAPI:
    """Verify JottyAPI instantiation and property exposure."""

    def _build_api(self):
        """Create a JottyAPI with fully mocked dependencies."""
        mock_conductor = Mock()
        api = JottyAPI(
            conductor=mock_conductor,
        )
        return api

    def test_jotty_api_instantiation(self):
        """JottyAPI can be created with a mock conductor."""
        api = self._build_api()
        assert api is not None
        assert api.conductor is not None
        assert api.config is not None

    def test_jotty_api_has_chat_method(self):
        """JottyAPI exposes an async chat() method."""
        api = self._build_api()
        assert hasattr(api, "chat")
        assert callable(api.chat)

    def test_jotty_api_has_run_method(self):
        """JottyAPI exposes an async run() method."""
        api = self._build_api()
        assert hasattr(api, "run")
        assert callable(api.run)

    @pytest.mark.asyncio
    async def test_jotty_api_chat_execute_delegates(self):
        """chat_execute() delegates to the orchestrator.chat method."""
        mock_conductor = Mock()
        mock_result = Mock()
        mock_result.to_dict.return_value = {"response": "hi"}
        mock_conductor.chat = AsyncMock(return_value=mock_result)

        api = JottyAPI(conductor=mock_conductor)
        output = await api.chat_execute(message="hello")
        assert output == {"response": "hi"}
        mock_conductor.chat.assert_called_once()


# ---------------------------------------------------------------------------
# ChatAPI tests (13-14)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(not HAS_CHAT_API, reason="ChatAPI import failed")
class TestChatAPI:
    """Verify ChatAPI instantiation and defaults."""

    def test_chat_api_instantiation(self):
        """ChatAPI can be created with a mock conductor."""
        mock_conductor = Mock()
        chat = ChatAPI(
            conductor=mock_conductor,
            auto_register_chat_assistant=False,
        )
        assert chat is not None
        assert chat.conductor is mock_conductor

    def test_chat_api_has_send_method(self):
        """ChatAPI exposes an async send() method."""
        mock_conductor = Mock()
        chat = ChatAPI(
            conductor=mock_conductor,
            auto_register_chat_assistant=False,
        )
        assert hasattr(chat, "send")
        assert callable(chat.send)

    def test_chat_api_has_stream_method(self):
        """ChatAPI exposes an async stream() method."""
        mock_conductor = Mock()
        chat = ChatAPI(
            conductor=mock_conductor,
            auto_register_chat_assistant=False,
        )
        assert hasattr(chat, "stream")
        assert callable(chat.stream)


# ---------------------------------------------------------------------------
# WorkflowAPI tests (15-17)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(not HAS_WORKFLOW_API, reason="WorkflowAPI import failed")
class TestWorkflowAPI:
    """Verify WorkflowAPI instantiation and method exposure."""

    def test_workflow_api_instantiation(self):
        """WorkflowAPI can be created with a mock conductor."""
        mock_conductor = Mock()
        wf = WorkflowAPI(conductor=mock_conductor)
        assert wf is not None
        assert wf.conductor is mock_conductor

    def test_workflow_api_has_execute_method(self):
        """WorkflowAPI exposes an async execute() method."""
        mock_conductor = Mock()
        wf = WorkflowAPI(conductor=mock_conductor)
        assert hasattr(wf, "execute")
        assert callable(wf.execute)

    def test_workflow_api_has_enqueue_method(self):
        """WorkflowAPI exposes an async enqueue() method."""
        mock_conductor = Mock()
        wf = WorkflowAPI(conductor=mock_conductor)
        assert hasattr(wf, "enqueue")
        assert callable(wf.enqueue)

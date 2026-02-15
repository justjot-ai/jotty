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
from unittest.mock import AsyncMock, MagicMock, Mock, patch

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
        mock_agents = [Mock()]
        mock_config = Mock()
        mock_conductor = Mock()
        api = JottyAPI(
            agents=mock_agents,
            config=mock_config,
            conductor=mock_conductor,
        )
        return api

    def test_jotty_api_instantiation(self):
        """JottyAPI can be created with mock agents, config, and conductor."""
        api = self._build_api()
        assert api is not None
        assert api.conductor is not None
        assert api.config is not None
        assert len(api.agents) == 1

    @patch("Jotty.core.api.unified.ChatUseCase")
    @patch("Jotty.core.api.unified.UseCaseConfig")
    def test_jotty_api_has_chat_property(self, _mock_cfg, _mock_chat_cls):
        """JottyAPI.chat returns a ChatUseCase (lazy-initialised)."""
        api = self._build_api()
        # Accessing the property should trigger lazy init
        chat = api.chat
        assert chat is not None

    @patch("Jotty.core.api.unified.WorkflowUseCase")
    @patch("Jotty.core.api.unified.UseCaseConfig")
    def test_jotty_api_has_workflow_property(self, _mock_cfg, _mock_wf_cls):
        """JottyAPI.workflow returns a WorkflowUseCase (lazy-initialised)."""
        api = self._build_api()
        workflow = api.workflow
        assert workflow is not None

    @pytest.mark.asyncio
    @patch("Jotty.core.api.unified.ChatUseCase")
    @patch("Jotty.core.api.unified.UseCaseConfig")
    async def test_jotty_api_chat_execute_delegates(self, _mock_cfg, mock_chat_cls):
        """chat_execute() delegates to the ChatUseCase.execute method."""
        api = self._build_api()

        # Configure the mock ChatUseCase instance returned by the property
        mock_result = Mock()
        mock_result.to_dict.return_value = {"response": "hi"}
        mock_chat_instance = Mock()
        mock_chat_instance.execute = AsyncMock(return_value=mock_result)
        mock_chat_cls.return_value = mock_chat_instance

        # Force re-init so the patched class is used
        api._chat_use_case = None

        output = await api.chat_execute(message="hello")
        assert output == {"response": "hi"}


# ---------------------------------------------------------------------------
# ChatAPI tests (13-14)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(not HAS_CHAT_API, reason="ChatAPI import failed")
class TestChatAPI:
    """Verify ChatAPI instantiation and defaults."""

    @patch("Jotty.core.api.chat_api.ChatUseCase")
    def test_chat_api_instantiation(self, mock_chat_use_case_cls):
        """ChatAPI can be created with a mock conductor."""
        mock_conductor = Mock()
        chat = ChatAPI(
            conductor=mock_conductor,
            auto_register_chat_assistant=False,
        )
        assert chat is not None
        assert chat.conductor is mock_conductor

    @patch("Jotty.core.api.chat_api.ChatUseCase")
    def test_chat_api_mode_defaults_dynamic(self, mock_chat_use_case_cls):
        """ChatAPI passes mode='dynamic' to the underlying ChatUseCase."""
        mock_conductor = Mock()
        ChatAPI(
            conductor=mock_conductor,
            auto_register_chat_assistant=False,
        )
        # Verify that ChatUseCase was constructed with mode="dynamic"
        call_kwargs = mock_chat_use_case_cls.call_args
        assert call_kwargs is not None
        # Could be positional or keyword; check keyword 'mode'
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("mode") == "dynamic"
        else:
            # mode is the third positional arg (conductor, agent_id, mode)
            assert "dynamic" in call_kwargs.args


# ---------------------------------------------------------------------------
# WorkflowAPI tests (15-17)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.skipif(not HAS_WORKFLOW_API, reason="WorkflowAPI import failed")
class TestWorkflowAPI:
    """Verify WorkflowAPI instantiation and method exposure."""

    @patch("Jotty.core.api.workflow_api.WorkflowUseCase")
    def test_workflow_api_instantiation(self, mock_wf_cls):
        """WorkflowAPI can be created with a mock conductor."""
        mock_conductor = Mock()
        wf = WorkflowAPI(conductor=mock_conductor)
        assert wf is not None
        assert wf.conductor is mock_conductor

    @patch("Jotty.core.api.workflow_api.WorkflowUseCase")
    def test_workflow_api_has_execute_method(self, mock_wf_cls):
        """WorkflowAPI exposes an async execute() method."""
        mock_conductor = Mock()
        wf = WorkflowAPI(conductor=mock_conductor)
        assert hasattr(wf, "execute")
        assert callable(wf.execute)

    @patch("Jotty.core.api.workflow_api.WorkflowUseCase")
    def test_workflow_api_mode_defaults_dynamic(self, mock_wf_cls):
        """WorkflowAPI passes mode='dynamic' to the underlying WorkflowUseCase."""
        mock_conductor = Mock()
        WorkflowAPI(conductor=mock_conductor)
        call_kwargs = mock_wf_cls.call_args
        assert call_kwargs is not None
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("mode") == "dynamic"
        else:
            assert "dynamic" in call_kwargs.args

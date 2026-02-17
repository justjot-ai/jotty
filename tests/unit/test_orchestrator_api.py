"""
Tests for the unified Orchestrator API: run() and chat().

Validates:
- run() auto-routing (default swarm-based)
- run(agent=...) single-agent mode
- run(swarm=...) explicit swarm mode
- run(stages=[...]) pipeline mode
- chat() conversational mode
- chat(stream=True) streaming mode
- Backward-compat: run_pipeline(), JottyAPI, ChatAPI
- LearningService integration in both run() and chat()
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

logger = logging.getLogger(__name__)


# ============================================================================
# Fixtures
# ============================================================================


@dataclass
class MockResult:
    """Minimal result object for testing."""

    success: bool = True
    output: str = "test output"
    content: str = "test content"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self):
        return {
            "success": self.success,
            "output": self.output,
            "metadata": self.metadata,
        }


@pytest.fixture
def mock_learning_service():
    """Mock LearningService singleton."""
    with patch("Jotty.core.intelligence.learning.learning_service.LearningService") as MockLS:
        instance = MagicMock()
        instance.start_episode.return_value = "test-episode-123"
        instance.end_episode.return_value = None
        instance.record.return_value = None
        instance.query.return_value = {"recommendations": []}
        MockLS.get_instance.return_value = instance
        yield instance


@pytest.fixture
def orchestrator():
    """Create an Orchestrator without triggering real components."""
    from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

    orch = object.__new__(Orchestrator)
    # Minimal init to avoid full Orchestrator.__init__
    orch.config = MagicMock()
    orch.config.domain = "general"
    orch.agents = []
    orch.mode = "single"
    orch.runners = {}
    orch._runners_built = False
    orch._efficiency_stats = {}
    orch._intelligence_metrics = {}
    orch._engine = MagicMock()
    orch._engine.run = AsyncMock(return_value=MockResult())
    orch._learning_ready = asyncio.Event()
    orch._learning_ready.set()
    return orch


# ============================================================================
# Test: run() auto-routing
# ============================================================================


class TestRunAutoRouting:
    """Test run() auto-routes to ExecutionEngine when no explicit mode given."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_auto_detect(self, orchestrator, mock_learning_service):
        """run(goal) should delegate to ExecutionEngine.run()."""
        result = await orchestrator.run("What is GDP?")

        assert result is not None
        assert result.success is True
        orchestrator._engine.run.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_records_to_learning(self, orchestrator, mock_learning_service):
        """run() should record outcome to LearningService."""
        await orchestrator.run("Test task")

        mock_learning_service.record.assert_called_once()
        call_kwargs = mock_learning_service.record.call_args[1]
        assert call_kwargs["unit_type"] == "orchestrator"
        assert call_kwargs["task_type"] == "run"
        assert call_kwargs["action"]["mode"] == "auto"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_passes_status_callback(self, orchestrator, mock_learning_service):
        """run() should pass status_callback to engine."""
        callback = MagicMock()
        await orchestrator.run("Test", status_callback=callback)

        call_kwargs = orchestrator._engine.run.call_args[1]
        assert call_kwargs.get("status_callback") == callback


# ============================================================================
# Test: run(agent=...) single agent mode
# ============================================================================


class TestRunAgent:
    """Test run(agent=...) routes to single agent execution."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_with_agent(self, orchestrator, mock_learning_service):
        """run(agent=...) should call agent.execute()."""
        mock_agent = MagicMock()
        mock_agent.execute = AsyncMock(return_value=MockResult())
        mock_agent.__class__.__name__ = "TestAgent"

        result = await orchestrator.run("Review this code", agent=mock_agent)

        assert result.success is True
        mock_agent.execute.assert_awaited_once()
        call_kwargs = mock_agent.execute.call_args
        assert (
            call_kwargs[1].get("task") == "Review this code"
            or call_kwargs[0][0] == "Review this code"
        )

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_agent_records_mode(self, orchestrator, mock_learning_service):
        """run(agent=...) should record mode='agent' to LearningService."""
        mock_agent = MagicMock()
        mock_agent.execute = AsyncMock(return_value=MockResult())

        await orchestrator.run("Test", agent=mock_agent)

        call_kwargs = mock_learning_service.record.call_args[1]
        assert call_kwargs["action"]["mode"] == "agent"


# ============================================================================
# Test: run(swarm=...) swarm mode
# ============================================================================


class TestRunSwarm:
    """Test run(swarm=...) routes to swarm execution."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_with_swarm_instance(self, orchestrator, mock_learning_service):
        """run(swarm=instance) should call swarm.execute()."""
        mock_swarm = MagicMock()
        mock_swarm.execute = AsyncMock(return_value=MockResult())
        mock_swarm.__class__.__name__ = "CodingSwarm"

        result = await orchestrator.run("Build API", swarm=mock_swarm)

        assert result.success is True
        mock_swarm.execute.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_swarm_records_mode(self, orchestrator, mock_learning_service):
        """run(swarm=...) should record mode='swarm' to LearningService."""
        mock_swarm = MagicMock()
        mock_swarm.execute = AsyncMock(return_value=MockResult())

        await orchestrator.run("Test", swarm=mock_swarm)

        call_kwargs = mock_learning_service.record.call_args[1]
        assert call_kwargs["action"]["mode"] == "swarm"


# ============================================================================
# Test: run(stages=[...]) pipeline mode
# ============================================================================


class TestRunPipeline:
    """Test run(stages=[...]) routes to pipeline execution."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_with_stages(self, orchestrator, mock_learning_service):
        """run(stages=[...]) should execute pipeline."""
        mock_swarm = MagicMock()
        mock_swarm.execute = AsyncMock(return_value=MockResult(output="designed"))
        mock_swarm.__class__.__name__ = "CodingSwarm"

        stages = [
            {"name": "design", "swarm": mock_swarm, "task": "Design API"},
        ]

        result = await orchestrator.run("Build API", stages=stages)

        assert result is not None
        mock_swarm.execute.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_pipeline_records_mode(self, orchestrator, mock_learning_service):
        """run(stages=...) should record mode='pipeline' to LearningService."""
        stages = [
            {
                "name": "step1",
                "callable": AsyncMock(return_value="done"),
            },
        ]

        await orchestrator.run("Test pipeline", stages=stages)

        call_kwargs = mock_learning_service.record.call_args[1]
        assert call_kwargs["action"]["mode"] == "pipeline"


# ============================================================================
# Test: run_pipeline() backward compat
# ============================================================================


class TestRunPipelineBackwardCompat:
    """Test run_pipeline() still works as backward-compat alias."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_pipeline_alias(self, orchestrator, mock_learning_service):
        """run_pipeline() should delegate to run(stages=...)."""
        stages = [
            {
                "name": "step1",
                "callable": AsyncMock(return_value="done"),
            },
        ]

        result = await orchestrator.run_pipeline("Test", stages=stages)
        assert result is not None


# ============================================================================
# Test: chat() conversational mode
# ============================================================================


class TestChat:
    """Test chat() conversational mode."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_chat_basic(self, orchestrator, mock_learning_service):
        """chat() should execute via ChatExecutor."""
        mock_executor_result = MockResult(content="Hello! How can I help?")

        with patch(
            "Jotty.core.intelligence.orchestration.execution.unified_executor.ChatExecutor"
        ) as MockCE:
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(return_value=mock_executor_result)
            MockCE.return_value = mock_executor

            result = await orchestrator.chat("Hello!")

            assert result.content == "Hello! How can I help?"
            mock_executor.execute.assert_awaited_once_with("Hello!", history=None)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_chat_with_history(self, orchestrator, mock_learning_service):
        """chat() should pass history to ChatExecutor."""
        history = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is..."},
        ]

        with patch(
            "Jotty.core.intelligence.orchestration.execution.unified_executor.ChatExecutor"
        ) as MockCE:
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(return_value=MockResult())
            MockCE.return_value = mock_executor

            await orchestrator.chat("Follow up", history=history)

            mock_executor.execute.assert_awaited_once_with("Follow up", history=history)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_chat_starts_learning_episode(self, orchestrator, mock_learning_service):
        """chat() should start and end a LearningService episode."""
        with patch(
            "Jotty.core.intelligence.orchestration.execution.unified_executor.ChatExecutor"
        ) as MockCE:
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(return_value=MockResult())
            MockCE.return_value = mock_executor

            await orchestrator.chat("Test")

            mock_learning_service.start_episode.assert_called_once()
            ep_kwargs = mock_learning_service.start_episode.call_args[1]
            assert ep_kwargs["unit_type"] == "chat"
            assert ep_kwargs["task_type"] == "chat"

            mock_learning_service.end_episode.assert_called_once()
            end_kwargs = mock_learning_service.end_episode.call_args[1]
            assert end_kwargs["episode_id"] == "test-episode-123"


# ============================================================================
# Test: JottyAPI facade
# ============================================================================


class TestJottyAPIFacade:
    """Test JottyAPI delegates to Orchestrator."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_jotty_api_run(self, orchestrator, mock_learning_service):
        """JottyAPI.run() delegates to Orchestrator.run()."""
        from Jotty.core.interface.api.unified import JottyAPI

        api = JottyAPI(conductor=orchestrator)
        result = await api.run("Test goal")

        assert result is not None
        orchestrator._engine.run.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_jotty_api_chat(self, orchestrator, mock_learning_service):
        """JottyAPI.chat() delegates to Orchestrator.chat()."""
        from Jotty.core.interface.api.unified import JottyAPI

        with patch(
            "Jotty.core.intelligence.orchestration.execution.unified_executor.ChatExecutor"
        ) as MockCE:
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(return_value=MockResult())
            MockCE.return_value = mock_executor

            api = JottyAPI(conductor=orchestrator)
            result = await api.chat("Hello!")

            assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_jotty_api_backward_compat(self, orchestrator, mock_learning_service):
        """JottyAPI.chat_execute() backward compat still works."""
        from Jotty.core.interface.api.unified import JottyAPI

        with patch(
            "Jotty.core.intelligence.orchestration.execution.unified_executor.ChatExecutor"
        ) as MockCE:
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(return_value=MockResult())
            MockCE.return_value = mock_executor

            api = JottyAPI(conductor=orchestrator)
            result = await api.chat_execute("Hello!")

            assert isinstance(result, dict)
            assert result["success"] is True


# ============================================================================
# Test: ChatAPI facade
# ============================================================================


class TestChatAPIFacade:
    """Test ChatAPI delegates to Orchestrator.chat()."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_chat_api_send(self, orchestrator, mock_learning_service):
        """ChatAPI.send() delegates to Orchestrator.chat()."""
        from Jotty.core.interface.api.chat_api import ChatAPI

        with patch(
            "Jotty.core.intelligence.orchestration.execution.unified_executor.ChatExecutor"
        ) as MockCE:
            mock_executor = MagicMock()
            mock_result = MockResult(content="Hi there!")
            mock_executor.execute = AsyncMock(return_value=mock_result)
            MockCE.return_value = mock_executor

            chat = ChatAPI(conductor=orchestrator)
            result = await chat.send("Hello!")

            assert isinstance(result, dict)
            assert result["success"] is True


# ============================================================================
# Test: WorkflowAPI facade
# ============================================================================


class TestWorkflowAPIFacade:
    """Test WorkflowAPI delegates to Orchestrator.run()."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_workflow_api_execute(self, orchestrator, mock_learning_service):
        """WorkflowAPI.execute() delegates to Orchestrator.run()."""
        from Jotty.core.interface.api.workflow_api import WorkflowAPI

        api = WorkflowAPI(conductor=orchestrator)
        result = await api.execute("Research AI")

        assert isinstance(result, dict)
        assert result["success"] is True


# ============================================================================
# Test: Architecture integrity
# ============================================================================


class TestArchitectureIntegrity:
    """Verify the Orchestrator has the expected public API."""

    @pytest.mark.unit
    def test_orchestrator_has_run(self):
        """Orchestrator must have run() method."""
        from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

        assert hasattr(Orchestrator, "run")
        assert asyncio.iscoroutinefunction(Orchestrator.run) or callable(Orchestrator.run)

    @pytest.mark.unit
    def test_orchestrator_has_chat(self):
        """Orchestrator must have chat() method."""
        from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

        assert hasattr(Orchestrator, "chat")
        assert asyncio.iscoroutinefunction(Orchestrator.chat)

    @pytest.mark.unit
    def test_orchestrator_has_run_pipeline_compat(self):
        """Orchestrator must have run_pipeline() for backward compat."""
        from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

        assert hasattr(Orchestrator, "run_pipeline")

    @pytest.mark.unit
    def test_run_signature(self):
        """run() should accept stream, stages, swarm, agent kwargs."""
        import inspect
        from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

        sig = inspect.signature(Orchestrator.run)
        params = set(sig.parameters.keys())
        assert "goal" in params
        assert "stream" in params
        assert "stages" in params
        assert "swarm" in params
        assert "agent" in params

    @pytest.mark.unit
    def test_chat_signature(self):
        """chat() should accept message, history, stream kwargs."""
        import inspect
        from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

        sig = inspect.signature(Orchestrator.chat)
        params = set(sig.parameters.keys())
        assert "message" in params
        assert "history" in params
        assert "stream" in params

    @pytest.mark.unit
    def test_no_use_cases_in_api(self):
        """JottyAPI should not import from use_cases."""
        import inspect
        from Jotty.core.interface.api import unified

        source = inspect.getsource(unified)
        assert "use_cases" not in source

    @pytest.mark.unit
    def test_no_use_cases_in_chat_api(self):
        """ChatAPI should not import from use_cases."""
        import inspect
        from Jotty.core.interface.api import chat_api

        source = inspect.getsource(chat_api)
        assert "use_cases" not in source

    @pytest.mark.unit
    def test_no_use_cases_in_workflow_api(self):
        """WorkflowAPI should not import from use_cases."""
        import inspect
        from Jotty.core.interface.api import workflow_api

        source = inspect.getsource(workflow_api)
        assert "use_cases" not in source

    @pytest.mark.unit
    def test_run_signature_has_learn(self):
        """run() should accept learn kwarg."""
        import inspect
        from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

        sig = inspect.signature(Orchestrator.run)
        params = set(sig.parameters.keys())
        assert "learn" in params

    @pytest.mark.unit
    def test_chat_signature_has_learn(self):
        """chat() should accept learn kwarg."""
        import inspect
        from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

        sig = inspect.signature(Orchestrator.chat)
        params = set(sig.parameters.keys())
        assert "learn" in params


# ============================================================================
# Test: learn=True/False flag
# ============================================================================


class TestLearnFlag:
    """Test learn=True (default) and learn=False behavior."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_learn_true_records(self, orchestrator, mock_learning_service):
        """run(learn=True) should record to LearningService."""
        mock_learning_service.build_context_string.return_value = ""
        await orchestrator.run("Test task", learn=True)

        mock_learning_service.record.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_learn_false_skips_recording(self, orchestrator, mock_learning_service):
        """run(learn=False) should NOT record to LearningService."""
        await orchestrator.run("Test task", learn=False)

        mock_learning_service.record.assert_not_called()
        mock_learning_service.build_context_string.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_learn_default_is_true(self, orchestrator, mock_learning_service):
        """run() with no learn flag should default to learn=True."""
        mock_learning_service.build_context_string.return_value = ""
        await orchestrator.run("Default task")

        mock_learning_service.record.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_chat_learn_true_starts_episode(self, orchestrator, mock_learning_service):
        """chat(learn=True) should start and end episode."""
        with patch(
            "Jotty.core.intelligence.orchestration.execution.unified_executor.ChatExecutor"
        ) as MockCE:
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(return_value=MockResult())
            MockCE.return_value = mock_executor

            await orchestrator.chat("Hello!", learn=True)

            mock_learning_service.start_episode.assert_called_once()
            mock_learning_service.end_episode.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_chat_learn_false_skips_episode(self, orchestrator, mock_learning_service):
        """chat(learn=False) should NOT start or end episode."""
        with patch(
            "Jotty.core.intelligence.orchestration.execution.unified_executor.ChatExecutor"
        ) as MockCE:
            mock_executor = MagicMock()
            mock_executor.execute = AsyncMock(return_value=MockResult())
            MockCE.return_value = mock_executor

            await orchestrator.chat("Quick test", learn=False)

            mock_learning_service.start_episode.assert_not_called()
            mock_learning_service.end_episode.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_learn_injects_guidance(self, orchestrator, mock_learning_service):
        """run(learn=True) should inject LearningService guidance into context."""
        mock_learning_service.build_context_string.return_value = "[Guidance] Use tool X"
        await orchestrator.run("Test task")

        mock_learning_service.build_context_string.assert_called_once()
        call_kwargs = orchestrator._engine.run.call_args[1]
        assert "[Guidance] Use tool X" in call_kwargs.get("learning_context", "")


# ============================================================================
# Test: SwarmLearningPipeline records to LearningService
# ============================================================================


class TestLearningPipelineIntegration:
    """Test that SwarmLearningPipeline.post_episode() records to LearningService."""

    @pytest.mark.unit
    def test_post_episode_records_to_learning_service(self):
        """post_episode() should call LearningService.record()."""
        from unittest.mock import patch as _patch

        from Jotty.core.infrastructure.foundation.data_structures import (
            EpisodeResult,
            SwarmConfig,
        )
        from Jotty.core.intelligence.orchestration.learning.swarm_learning_pipeline import (
            SwarmLearningPipeline,
        )

        config = MagicMock(spec=SwarmConfig)
        config.base_path = None
        config.domain = "test"
        config.learning_components = None

        with _patch(
            "Jotty.core.intelligence.orchestration.learning."
            "swarm_learning_pipeline.SwarmLearningPipeline.__init__",
            return_value=None,
        ):
            pipeline = SwarmLearningPipeline.__new__(SwarmLearningPipeline)
            pipeline.config = config
            pipeline.episode_count = 0
            pipeline.effectiveness = MagicMock()
            pipeline.effectiveness.improvement_report.return_value = {}
            pipeline.transfer_learning = MagicMock()
            pipeline.transfer_learning.extractor.extract_task_type.return_value = "test_task"

            result = MagicMock(spec=EpisodeResult)
            result.success = True
            result.output = "test output"
            result.trajectory = []
            result.agent_name = "test_agent"

            mock_ls = MagicMock()
            with _patch(
                "Jotty.core.intelligence.learning.learning_service." "LearningService.get_instance",
                return_value=mock_ls,
            ):
                pipeline._record_to_learning_service(
                    {
                        "result": result,
                        "goal": "test goal",
                        "agents": [],
                        "agent_name": "test_agent",
                        "task_type": "test_task",
                        "episode_reward": 0.8,
                    },
                    execution_time=1.5,
                )

                mock_ls.record.assert_called_once()
                call_kwargs = mock_ls.record.call_args[1]
                assert call_kwargs["unit_name"] == "test_agent"
                assert call_kwargs["unit_type"] == "swarm_pipeline"
                assert call_kwargs["success"] is True
                assert call_kwargs["quality"] == 0.8

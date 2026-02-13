"""
Tests for Orchestration Core Modules
=====================================
Tests for ExecutionContext, AgentRunnerConfig (agent_runner.py),
ProviderManager (provider_manager.py), and EnsembleManager (ensemble_manager.py).
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import fields

from Jotty.core.orchestration.agent_runner import ExecutionContext, AgentRunnerConfig
from Jotty.core.foundation.data_structures import SwarmConfig

# Conditional imports for modules with complex dependency chains
try:
    from Jotty.core.orchestration.provider_manager import ProviderManager
    HAS_PROVIDER_MANAGER = True
except ImportError:
    HAS_PROVIDER_MANAGER = False

try:
    from Jotty.core.orchestration.ensemble_manager import EnsembleManager
    HAS_ENSEMBLE_MANAGER = True
except ImportError:
    HAS_ENSEMBLE_MANAGER = False


# =============================================================================
# ExecutionContext Tests (1-6)
# =============================================================================

@pytest.mark.unit
class TestExecutionContext:
    """Tests for the ExecutionContext dataclass used throughout the agent pipeline."""

    def test_defaults_populated(self):
        """ExecutionContext initializes all default fields correctly."""
        ctx = ExecutionContext(goal="test goal", kwargs={"key": "value"})
        assert ctx.start_time == 0.0
        assert ctx.status_callback is None
        assert ctx.gate_decision is None
        assert ctx.skip_architect is False
        assert ctx.skip_auditor is False
        assert ctx.enriched_goal == ""
        assert ctx.proceed is True
        assert ctx.architect_shaped_reward == 0.0
        assert ctx.agent_output is None
        assert ctx.inner_success is False
        assert ctx.success is False
        assert ctx.auditor_reasoning == ""
        assert ctx.auditor_confidence == 0.0
        assert ctx.duration == 0.0
        assert ctx.task_progress is None
        assert ctx.ws_checkpoint_id is None

    def test_goal_stored(self):
        """ExecutionContext stores the goal string."""
        ctx = ExecutionContext(goal="Research quantum computing", kwargs={})
        assert ctx.goal == "Research quantum computing"

    def test_kwargs_stored(self):
        """ExecutionContext stores the kwargs dictionary."""
        kwargs = {"workspace_dir": "/tmp", "status_callback": None}
        ctx = ExecutionContext(goal="test", kwargs=kwargs)
        assert ctx.kwargs == kwargs
        assert ctx.kwargs["workspace_dir"] == "/tmp"

    def test_proceed_defaults_true(self):
        """ExecutionContext.proceed defaults to True so agents run by default."""
        ctx = ExecutionContext(goal="any goal", kwargs={})
        assert ctx.proceed is True

    def test_success_defaults_false(self):
        """ExecutionContext.success defaults to False until auditor validates."""
        ctx = ExecutionContext(goal="any goal", kwargs={})
        assert ctx.success is False

    def test_learning_context_parts_defaults_empty(self):
        """ExecutionContext.learning_context_parts defaults to an empty list."""
        ctx = ExecutionContext(goal="any goal", kwargs={})
        assert ctx.learning_context_parts == []
        assert isinstance(ctx.learning_context_parts, list)

    def test_mutable_default_fields_are_independent(self):
        """Each ExecutionContext instance gets its own mutable default lists/dicts."""
        ctx_a = ExecutionContext(goal="a", kwargs={})
        ctx_b = ExecutionContext(goal="b", kwargs={})
        ctx_a.learning_context_parts.append("item")
        ctx_a.trajectory.append({"step": 1})
        ctx_a.learning_data["key"] = "val"
        # ctx_b should not be affected
        assert ctx_b.learning_context_parts == []
        assert ctx_b.trajectory == []
        assert ctx_b.learning_data == {}


# =============================================================================
# AgentRunnerConfig Tests (7-10)
# =============================================================================

@pytest.mark.unit
class TestAgentRunnerConfig:
    """Tests for the AgentRunnerConfig dataclass."""

    def test_requires_architect_auditor_config(self):
        """AgentRunnerConfig requires architect_prompts, auditor_prompts, and config."""
        config = SwarmConfig()
        runner_config = AgentRunnerConfig(
            architect_prompts=["prompt_a.md"],
            auditor_prompts=["prompt_b.md"],
            config=config,
        )
        assert runner_config.architect_prompts == ["prompt_a.md"]
        assert runner_config.auditor_prompts == ["prompt_b.md"]
        assert runner_config.config is config

    def test_missing_required_fields_raises(self):
        """AgentRunnerConfig raises TypeError when required fields are missing."""
        with pytest.raises(TypeError):
            AgentRunnerConfig()  # missing all required fields
        with pytest.raises(TypeError):
            AgentRunnerConfig(architect_prompts=["a"])  # missing auditor_prompts and config

    def test_agent_name_defaults_agent(self):
        """AgentRunnerConfig.agent_name defaults to 'agent'."""
        config = SwarmConfig()
        runner_config = AgentRunnerConfig(
            architect_prompts=[],
            auditor_prompts=[],
            config=config,
        )
        assert runner_config.agent_name == "agent"

    def test_enable_learning_defaults_true(self):
        """AgentRunnerConfig.enable_learning defaults to True."""
        config = SwarmConfig()
        runner_config = AgentRunnerConfig(
            architect_prompts=[],
            auditor_prompts=[],
            config=config,
        )
        assert runner_config.enable_learning is True

    def test_enable_terminal_defaults_true(self):
        """AgentRunnerConfig.enable_terminal defaults to True."""
        config = SwarmConfig()
        runner_config = AgentRunnerConfig(
            architect_prompts=[],
            auditor_prompts=[],
            config=config,
        )
        assert runner_config.enable_terminal is True

    def test_enable_memory_defaults_true(self):
        """AgentRunnerConfig.enable_memory defaults to True."""
        config = SwarmConfig()
        runner_config = AgentRunnerConfig(
            architect_prompts=[],
            auditor_prompts=[],
            config=config,
        )
        assert runner_config.enable_memory is True


# =============================================================================
# ProviderManager Tests (11-14)
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_PROVIDER_MANAGER, reason="ProviderManager not importable")
class TestProviderManager:
    """Tests for the ProviderManager composed class."""

    def test_init_with_config_and_callable(self):
        """ProviderManager stores config and swarm_intelligence getter."""
        config = SwarmConfig()
        getter = Mock(return_value="si_instance")
        pm = ProviderManager(config=config, get_swarm_intelligence=getter)
        assert pm.config is config
        assert pm._get_si is getter

    def test_swarm_intelligence_property_returns_callable_result(self):
        """ProviderManager.swarm_intelligence property delegates to the getter callable."""
        mock_si = Mock(name="SwarmIntelligence")
        getter = Mock(return_value=mock_si)
        config = SwarmConfig()
        pm = ProviderManager(config=config, get_swarm_intelligence=getter)
        result = pm.swarm_intelligence
        getter.assert_called_once()
        assert result is mock_si

    def test_init_provider_registry_creates_registry(self):
        """ProviderManager.init_provider_registry creates a registry when providers load."""
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = False

        config = SwarmConfig()
        mock_si = Mock(name="SwarmIntelligence")
        pm = ProviderManager(config=config, get_swarm_intelligence=lambda: mock_si)

        # Mock the _load_providers and _provider_cache to simulate provider loading
        mock_registry_instance = Mock()
        mock_registry_instance.get_all_providers.return_value = {"test_provider": Mock()}

        mock_cache = {
            "ProviderRegistry": Mock(return_value=mock_registry_instance),
            "BrowserUseProvider": Mock(return_value=Mock(name="browser-use")),
            "OpenHandsProvider": Mock(return_value=Mock(name="openhands")),
            "AgentSProvider": Mock(return_value=Mock(name="agent-s")),
            "OpenInterpreterProvider": Mock(return_value=Mock(name="open-interpreter")),
            "ResearchAndAnalyzeProvider": Mock(return_value=Mock(name="research")),
            "AutomateWorkflowProvider": Mock(return_value=Mock(name="workflow")),
            "FullStackAgentProvider": Mock(return_value=Mock(name="fullstack")),
        }

        with patch(
            "Jotty.core.orchestration.swarm_manager._load_providers",
            return_value=True,
        ), patch(
            "Jotty.core.orchestration.swarm_manager._provider_cache",
            mock_cache,
        ), patch.object(
            pm, "_get_provider_registry_path", return_value=mock_path_obj,
        ):
            pm.init_provider_registry()
            assert pm.provider_registry is not None

    def test_provider_registry_starts_none(self):
        """ProviderManager.provider_registry is None before init_provider_registry."""
        config = SwarmConfig()
        pm = ProviderManager(config=config, get_swarm_intelligence=lambda: None)
        assert pm.provider_registry is None


# =============================================================================
# EnsembleManager Tests (15-18)
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_ENSEMBLE_MANAGER, reason="EnsembleManager not importable")
class TestEnsembleManager:
    """Tests for the EnsembleManager composed class."""

    def test_instantiation(self):
        """EnsembleManager can be instantiated without arguments."""
        em = EnsembleManager()
        assert em is not None
        assert hasattr(em, "execute_ensemble")
        assert hasattr(em, "should_auto_ensemble")

    @pytest.mark.asyncio
    async def test_execute_ensemble_returns_dict(self):
        """EnsembleManager.execute_ensemble returns a dict with success key."""
        em = EnsembleManager()

        # Mock the skill-based ensemble path via the source module where the
        # local import resolves: Jotty.core.registry.skills_registry
        mock_tool = Mock(return_value={
            "success": True,
            "response": "synthesized answer",
            "quality_scores": {"analytical": 0.9},
        })
        mock_skill = Mock()
        mock_skill.tools = {"ensemble_prompt_tool": mock_tool}
        mock_registry = Mock()
        mock_registry.get_skill.return_value = mock_skill

        with patch(
            "Jotty.core.registry.skills_registry.get_skills_registry",
            return_value=mock_registry,
        ):
            result = await em.execute_ensemble("What is AI?")
            assert isinstance(result, dict)
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_ensemble_strategy_parameter(self):
        """EnsembleManager.execute_ensemble accepts strategy parameter."""
        em = EnsembleManager()

        mock_tool = Mock(return_value={
            "success": True,
            "response": "debate result",
            "quality_scores": {},
        })
        mock_skill = Mock()
        mock_skill.tools = {"ensemble_prompt_tool": mock_tool}
        mock_registry = Mock()
        mock_registry.get_skill.return_value = mock_skill

        with patch(
            "Jotty.core.registry.skills_registry.get_skills_registry",
            return_value=mock_registry,
        ):
            result = await em.execute_ensemble(
                "Analyze climate change",
                strategy="debate",
                max_perspectives=3,
            )
            assert isinstance(result, dict)
            # Verify the tool was called with the strategy parameter
            call_args = mock_tool.call_args[0][0]
            assert call_args["strategy"] == "debate"
            assert call_args["max_perspectives"] == 3

    @pytest.mark.asyncio
    async def test_execute_ensemble_skill_fallback(self):
        """EnsembleManager falls back to DSPy when skill registry unavailable."""
        em = EnsembleManager()

        # Simulate ImportError for skills registry so it falls to DSPy path,
        # then mock DSPy to return a result.
        with patch(
            "Jotty.core.registry.skills_registry.get_skills_registry",
            side_effect=ImportError("no skills"),
        ):
            # The DSPy fallback path will try 'import dspy'. Mock it so
            # dspy.settings.lm exists and produces responses.
            mock_lm = Mock(side_effect=[
                ["analytical perspective"],   # perspective 1
                ["creative perspective"],      # perspective 2
                ["critical perspective"],      # perspective 3
                ["practical perspective"],     # perspective 4
                ["final synthesis"],           # synthesis call
            ])
            mock_dspy = MagicMock()
            mock_dspy.settings.lm = mock_lm

            with patch.dict("sys.modules", {"dspy": mock_dspy}):
                result = await em.execute_ensemble("What is consciousness?")
                assert isinstance(result, dict)
                assert result["success"] is True
                assert "perspectives_used" in result
                assert len(result["perspectives_used"]) > 0

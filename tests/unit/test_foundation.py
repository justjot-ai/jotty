"""
Tests for Foundation Module
============================
Tests for SwarmConfig, AgentConfig (foundation), Exception hierarchy,
model_limits_catalog, robust_parsing functions, and token_counter.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# SwarmConfig Tests
# =============================================================================


class TestSwarmConfig:
    """Tests for the SwarmConfig dataclass."""

    @pytest.mark.unit
    def test_default_creation(self):
        """SwarmConfig creates with sensible defaults."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig()
        assert config.schema_version == "2.0"
        assert config.max_actor_iters > 0
        assert config.episodic_capacity > 0
        assert config.gamma > 0
        assert config.enable_validation is True

    @pytest.mark.unit
    def test_memory_budget_property(self):
        """memory_budget = max_context - reserved, with floor."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig(
            max_context_tokens=10000,
            system_prompt_budget=2000,
            current_input_budget=2000,
            trajectory_budget=2000,
            tool_output_budget=2000,
            min_memory_budget=500,
        )
        # reserved = 2000+2000+2000+2000 = 8000, budget = 10000-8000 = 2000
        assert config.memory_budget == 2000

    @pytest.mark.unit
    def test_memory_budget_floor(self):
        """memory_budget respects min_memory_budget floor."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig(
            max_context_tokens=5000,
            system_prompt_budget=2000,
            current_input_budget=2000,
            trajectory_budget=2000,
            tool_output_budget=2000,
            min_memory_budget=500,
        )
        # reserved = 8000 > max 5000, so budget should be clamped to floor
        assert config.memory_budget == 500

    @pytest.mark.unit
    def test_total_memory_capacity(self):
        """total_memory_capacity sums all 5 levels."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig(
            episodic_capacity=100,
            semantic_capacity=50,
            procedural_capacity=30,
            meta_capacity=20,
            causal_capacity=10,
        )
        assert config.total_memory_capacity == 210

    @pytest.mark.unit
    def test_total_memory_capacity_with_zeros(self):
        """total_memory_capacity handles zero capacities."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig(
            episodic_capacity=0,
            semantic_capacity=0,
            procedural_capacity=0,
            meta_capacity=0,
            causal_capacity=0,
        )
        assert config.total_memory_capacity == 0

    @pytest.mark.unit
    def test_custom_field_values_preserved(self):
        """Explicit field values are not overridden."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig(
            gamma=0.5,
            lambda_trace=0.8,
            alpha=0.01,
            enable_rl=False,
            enable_validation=False,
        )
        assert config.gamma == 0.5
        assert config.lambda_trace == 0.8
        assert config.alpha == 0.01
        assert config.enable_rl is False
        assert config.enable_validation is False

    @pytest.mark.unit
    def test_validation_mode_values(self):
        """SwarmConfig accepts different validation modes."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        for mode in ["quick", "standard", "thorough"]:
            config = SwarmConfig(validation_mode=mode)
            assert config.validation_mode == mode

    @pytest.mark.unit
    def test_budget_controls(self):
        """Budget control fields have sensible defaults."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig()
        assert config.max_llm_calls_per_episode >= 0
        assert config.max_total_tokens_per_episode >= 0

    @pytest.mark.unit
    def test_to_flat_dict(self):
        """to_flat_dict() returns all fields as a flat dictionary."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig(max_actor_iters=42, enable_rl=False)
        d = config.to_flat_dict()
        assert isinstance(d, dict)
        assert d["max_actor_iters"] == 42
        assert d["enable_rl"] is False
        assert "schema_version" in d

    @pytest.mark.unit
    def test_to_flat_dict_roundtrip(self):
        """to_flat_dict() output can recreate SwarmConfig."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        original = SwarmConfig(gamma=0.5, episodic_capacity=42)
        d = original.to_flat_dict()
        restored = SwarmConfig(**d)
        assert restored.gamma == 0.5
        assert restored.episodic_capacity == 42


class TestConfigViews:
    """Tests for the SwarmConfig sub-config view system."""

    @pytest.mark.unit
    def test_execution_view_read(self):
        """ExecutionView reads parent config fields."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig(max_actor_iters=99, actor_timeout=300.0)
        assert config.execution.max_actor_iters == 99
        assert config.execution.actor_timeout == 300.0

    @pytest.mark.unit
    def test_execution_view_write_through(self):
        """ExecutionView writes propagate to parent."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig(max_actor_iters=10)
        config.execution.max_actor_iters = 50
        assert config.max_actor_iters == 50

    @pytest.mark.unit
    def test_memory_view_fields(self):
        """MemoryView exposes memory-related fields."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig(episodic_capacity=200, enable_llm_rag=False)
        assert config.memory_settings.episodic_capacity == 200
        assert config.memory_settings.enable_llm_rag is False

    @pytest.mark.unit
    def test_learning_view_fields(self):
        """LearningView exposes RL and exploration fields."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig(gamma=0.8, epsilon_start=0.5, enable_rl=False)
        assert config.learning.gamma == 0.8
        assert config.learning.epsilon_start == 0.5
        assert config.learning.enable_rl is False

    @pytest.mark.unit
    def test_validation_view_fields(self):
        """ValidationView exposes validation settings."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig(enable_validation=False, max_validation_rounds=5)
        assert config.validation_settings.enable_validation is False
        assert config.validation_settings.max_validation_rounds == 5

    @pytest.mark.unit
    def test_monitoring_view_fields(self):
        """MonitoringView exposes logging and budget fields."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig(log_level="DEBUG", enable_metrics=False)
        assert config.monitoring.log_level == "DEBUG"
        assert config.monitoring.enable_metrics is False

    @pytest.mark.unit
    def test_intelligence_view_fields(self):
        """SwarmIntelligenceView exposes trust and agent comm fields."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig(trust_min=0.2, enable_agent_communication=False)
        assert config.intelligence.trust_min == 0.2
        assert config.intelligence.enable_agent_communication is False

    @pytest.mark.unit
    def test_persistence_view_fields(self):
        """PersistenceView exposes storage fields."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig(output_base_dir="/tmp/test", storage_format="sqlite")
        assert config.persistence.output_base_dir == "/tmp/test"
        assert config.persistence.storage_format == "sqlite"

    @pytest.mark.unit
    def test_context_budget_view_fields(self):
        """ContextBudgetView exposes token budget fields."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig(max_context_tokens=50000, min_memory_budget=2000)
        assert config.context_budget.max_context_tokens == 50000
        assert config.context_budget.min_memory_budget == 2000

    @pytest.mark.unit
    def test_view_to_dict(self):
        """View.to_dict() returns only that view's fields."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig(max_actor_iters=42, gamma=0.5)
        exec_dict = config.execution.to_dict()
        assert "max_actor_iters" in exec_dict
        assert exec_dict["max_actor_iters"] == 42
        # gamma is a LearningView field, not ExecutionView
        assert "gamma" not in exec_dict

    @pytest.mark.unit
    def test_view_attribute_error(self):
        """View raises AttributeError for unknown fields."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig()
        with pytest.raises(AttributeError):
            _ = config.execution.nonexistent_field

    @pytest.mark.unit
    def test_view_repr(self):
        """View __repr__ includes field values."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig(max_actor_iters=42)
        r = repr(config.execution)
        assert "ExecutionView" in r
        assert "max_actor_iters=42" in r


# =============================================================================
# Foundation AgentConfig Tests
# =============================================================================


class TestFoundationAgentConfig:
    """Tests for the foundation AgentConfig (orchestration-level spec)."""

    @pytest.mark.unit
    def test_creation_with_required_fields(self):
        """AgentConfig requires name and agent."""
        from Jotty.core.infrastructure.foundation.agent_config import AgentConfig

        mock_agent = MagicMock()
        config = AgentConfig(name="TestAgent", agent=mock_agent)
        assert config.name == "TestAgent"
        assert config.agent is mock_agent

    @pytest.mark.unit
    def test_prompt_list_initialization(self):
        """None prompts initialized to empty lists."""
        from Jotty.core.infrastructure.foundation.agent_config import AgentConfig

        config = AgentConfig(name="Test", agent=MagicMock())
        assert config.architect_prompts == [] or config.architect_prompts is not None
        assert config.auditor_prompts == [] or config.auditor_prompts is not None

    @pytest.mark.unit
    def test_max_retries_sentinel_resolution(self):
        """max_retries=0 resolves from config_defaults."""
        from Jotty.core.infrastructure.foundation.agent_config import AgentConfig

        config = AgentConfig(name="Test", agent=MagicMock(), max_retries=0)
        # Should be resolved to a positive value from DEFAULTS
        assert config.max_retries > 0

    @pytest.mark.unit
    def test_max_retries_explicit_preserved(self):
        """Explicit max_retries value is preserved."""
        from Jotty.core.infrastructure.foundation.agent_config import AgentConfig

        config = AgentConfig(name="Test", agent=MagicMock(), max_retries=5)
        assert config.max_retries == 5

    @pytest.mark.unit
    def test_default_flags(self):
        """Default boolean flags are correct."""
        from Jotty.core.infrastructure.foundation.agent_config import AgentConfig

        config = AgentConfig(name="Test", agent=MagicMock())
        assert config.enable_architect is True
        assert config.enable_auditor is True
        assert config.enabled is True
        assert config.is_critical is False
        assert config.is_executor is False

    @pytest.mark.unit
    def test_validation_mode_default(self):
        """Default validation mode is 'standard'."""
        from Jotty.core.infrastructure.foundation.agent_config import AgentConfig

        config = AgentConfig(name="Test", agent=MagicMock())
        assert config.validation_mode == "standard"

    @pytest.mark.unit
    def test_dependencies_and_capabilities(self):
        """Dependencies and capabilities stored correctly."""
        from Jotty.core.infrastructure.foundation.agent_config import AgentConfig

        config = AgentConfig(
            name="Test",
            agent=MagicMock(),
            dependencies=["AgentA", "AgentB"],
            capabilities=["search", "analyze"],
        )
        assert config.dependencies == ["AgentA", "AgentB"]
        assert config.capabilities == ["search", "analyze"]

    @pytest.mark.unit
    def test_parameter_mappings(self):
        """Parameter mappings stored as dict."""
        from Jotty.core.infrastructure.foundation.agent_config import AgentConfig

        config = AgentConfig(
            name="Test",
            agent=MagicMock(),
            parameter_mappings={"input_data": "query", "context": "background"},
        )
        assert config.parameter_mappings == {"input_data": "query", "context": "background"}


# =============================================================================
# BaseAgent AgentRuntimeConfig Tests
# =============================================================================


class TestBaseAgentRuntimeConfig:
    """Tests for the BaseAgent-level AgentRuntimeConfig (runtime config)."""

    @pytest.mark.unit
    def test_default_sentinel_resolution(self):
        """Zero sentinels resolve from DEFAULTS."""
        from Jotty.core.modes.agent.base.base_agent import AgentRuntimeConfig

        config = AgentRuntimeConfig()
        assert config.model != ""
        assert config.temperature > 0
        assert config.max_tokens > 0
        assert config.max_retries > 0

    @pytest.mark.unit
    def test_explicit_values_preserved(self):
        """Explicit non-zero values are preserved."""
        from Jotty.core.modes.agent.base.base_agent import AgentRuntimeConfig

        config = AgentRuntimeConfig(
            name="MyAgent",
            model="claude-3",
            temperature=0.7,
            max_tokens=1000,
            max_retries=3,
        )
        assert config.name == "MyAgent"
        assert config.model == "claude-3"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.max_retries == 3

    @pytest.mark.unit
    def test_enable_flags_defaults(self):
        """Enable flags default to True."""
        from Jotty.core.modes.agent.base.base_agent import AgentRuntimeConfig

        config = AgentRuntimeConfig()
        assert config.enable_memory is True
        assert config.enable_context is True
        assert config.enable_monitoring is True
        assert config.enable_skills is True

    @pytest.mark.unit
    def test_enable_flags_override(self):
        """Enable flags can be disabled."""
        from Jotty.core.modes.agent.base.base_agent import AgentRuntimeConfig

        config = AgentRuntimeConfig(
            enable_memory=False,
            enable_context=False,
            enable_monitoring=False,
            enable_skills=False,
        )
        assert config.enable_memory is False
        assert config.enable_context is False
        assert config.enable_monitoring is False
        assert config.enable_skills is False


# =============================================================================
# Exception Hierarchy Tests
# =============================================================================


class TestExceptionHierarchy:
    """Tests for the Jotty exception hierarchy."""

    @pytest.mark.unit
    def test_jotty_error_message(self):
        """JottyError stores message."""
        from Jotty.core.infrastructure.foundation.exceptions import JottyError

        err = JottyError("Something failed")
        assert err.message == "Something failed"
        assert "Something failed" in str(err)

    @pytest.mark.unit
    def test_jotty_error_with_context(self):
        """JottyError stores context dict."""
        from Jotty.core.infrastructure.foundation.exceptions import JottyError

        err = JottyError("Failed", context={"agent": "TestAgent", "step": 3})
        assert err.context == {"agent": "TestAgent", "step": 3}

    @pytest.mark.unit
    def test_jotty_error_with_original(self):
        """JottyError chains original exception."""
        from Jotty.core.infrastructure.foundation.exceptions import JottyError

        original = ValueError("bad value")
        err = JottyError("Wrapped error", original_error=original)
        assert err.original_error is original

    @pytest.mark.unit
    def test_exception_inheritance_chain(self):
        """All exceptions inherit from JottyError."""
        from Jotty.core.infrastructure.foundation.exceptions import (
            AgentExecutionError,
            ConfigurationError,
            ExecutionError,
            JottyError,
            LearningError,
            MemoryRetrievalError,
            TimeoutError,
            ValidationError,
        )

        assert issubclass(ConfigurationError, JottyError)
        assert issubclass(ExecutionError, JottyError)
        assert issubclass(AgentExecutionError, ExecutionError)
        assert issubclass(TimeoutError, ExecutionError)
        assert issubclass(ValidationError, JottyError)
        assert issubclass(MemoryRetrievalError, JottyError)
        assert issubclass(LearningError, JottyError)

    @pytest.mark.unit
    def test_catch_by_parent_class(self):
        """Can catch specific exceptions by parent class."""
        from Jotty.core.infrastructure.foundation.exceptions import (
            AgentExecutionError,
            JottyError,
            TimeoutError,
        )

        with pytest.raises(JottyError):
            raise AgentExecutionError("agent failed")

        with pytest.raises(JottyError):
            raise TimeoutError("timed out")

    @pytest.mark.unit
    def test_validation_error_to_dict(self):
        """ValidationError.to_dict() returns structured response."""
        from Jotty.core.infrastructure.foundation.exceptions import ValidationError

        err = ValidationError("Invalid param", param="temperature", value=2.5)
        result = err.to_dict()
        assert result == {
            "success": False,
            "error": "Invalid param",
            "param": "temperature",
        }

    @pytest.mark.unit
    def test_validation_error_param_and_value(self):
        """ValidationError stores param and value attributes."""
        from Jotty.core.infrastructure.foundation.exceptions import ValidationError

        err = ValidationError("bad", param="x", value=42)
        assert err.param == "x"
        assert err.value == 42

    @pytest.mark.unit
    def test_validation_error_defaults(self):
        """ValidationError defaults param/value to None."""
        from Jotty.core.infrastructure.foundation.exceptions import ValidationError

        err = ValidationError("bad")
        assert err.param is None
        assert err.value is None
        result = err.to_dict()
        assert result["param"] is None

    @pytest.mark.unit
    def test_context_overflow_error(self):
        """ContextOverflowError stores token info."""
        from Jotty.core.infrastructure.foundation.exceptions import ContextOverflowError

        err = ContextOverflowError(
            "Too many tokens",
            detected_tokens=50000,
            max_tokens=32000,
        )
        assert err.detected_tokens == 50000
        assert err.max_tokens == 32000

    @pytest.mark.unit
    def test_wrap_exception(self):
        """wrap_exception() converts generic to Jotty exception."""
        from Jotty.core.infrastructure.foundation.exceptions import (
            AgentExecutionError,
            wrap_exception,
        )

        original = RuntimeError("something broke")
        wrapped = wrap_exception(
            original,
            AgentExecutionError,
            "Agent execution failed",
            agent="TestAgent",
        )
        assert isinstance(wrapped, AgentExecutionError)
        assert wrapped.original_error is original
        assert "Agent execution failed" in wrapped.message

    @pytest.mark.unit
    def test_validation_error_from_core_validation(self):
        """core.validation.ValidationError is the same as foundation one."""
        from Jotty.core.infrastructure.foundation.exceptions import ValidationError as V1
        from Jotty.core.validation import ValidationError as V2

        assert V1 is V2


# =============================================================================
# Singleton Reset Tests
# =============================================================================


class TestSingletonResets:
    """Tests that singleton reset functions work correctly."""

    @pytest.mark.unit
    def test_cost_tracker_reset(self):
        """reset_cost_tracker() clears the singleton."""
        from Jotty.core.infrastructure.foundation.direct_anthropic_lm import (
            get_cost_tracker,
            reset_cost_tracker,
        )

        tracker = get_cost_tracker()
        assert tracker is not None
        reset_cost_tracker()
        # After reset, get_cost_tracker returns a NEW instance
        tracker2 = get_cost_tracker()
        assert tracker2 is not tracker

    @pytest.mark.unit
    def test_jotty_integration_reset(self):
        """JottyIntegration.reset_instance() clears the singleton."""
        from Jotty.core.infrastructure.integration.integration import JottyIntegration

        inst = JottyIntegration.get_instance()
        assert inst is not None
        JottyIntegration.reset_instance()
        assert JottyIntegration._instance is None

    @pytest.mark.unit
    def test_event_broadcaster_reset(self):
        """AgentEventBroadcaster.reset_instance() clears the singleton."""
        from Jotty.core.infrastructure.utils.async_utils import AgentEventBroadcaster

        broadcaster = AgentEventBroadcaster.get_instance()
        assert broadcaster is not None
        AgentEventBroadcaster.reset_instance()
        assert AgentEventBroadcaster._instance is None

    @pytest.mark.unit
    def test_budget_tracker_reset(self):
        """BudgetTracker.reset_instances() clears all named instances."""
        from Jotty.core.infrastructure.utils.budget_tracker import BudgetTracker

        BudgetTracker.get_instance("test_a")
        BudgetTracker.get_instance("test_b")
        assert len(BudgetTracker._instances) >= 2
        BudgetTracker.reset_instances()
        assert len(BudgetTracker._instances) == 0

    @pytest.mark.unit
    def test_llm_cache_reset(self):
        """LLMCallCache.reset_instances() clears all named instances."""
        from Jotty.core.infrastructure.utils.llm_cache import LLMCallCache

        LLMCallCache.get_instance("test_cache")
        assert "test_cache" in LLMCallCache._instances
        LLMCallCache.reset_instances()
        assert len(LLMCallCache._instances) == 0

    @pytest.mark.unit
    def test_prompt_selector_reset(self):
        """reset_prompt_selector() clears the singleton."""
        from Jotty.core.infrastructure.utils.prompt_selector import (
            get_prompt_selector,
            reset_prompt_selector,
        )

        sel = get_prompt_selector()
        assert sel is not None
        reset_prompt_selector()
        sel2 = get_prompt_selector()
        assert sel2 is not sel

    @pytest.mark.unit
    def test_tokenizer_reset(self):
        """SmartTokenizer.reset_instances() clears all cached encodings."""
        from Jotty.core.infrastructure.utils.tokenizer import SmartTokenizer

        SmartTokenizer.get_instance()
        assert len(SmartTokenizer._instances) >= 1
        SmartTokenizer.reset_instances()
        assert len(SmartTokenizer._instances) == 0


# =============================================================================
# ConfigView Tests
# =============================================================================


class TestConfigViews:
    """Tests for _ConfigView proxy system."""

    @pytest.mark.unit
    def test_config_view_getattr_reads_parent(self):
        """ConfigView proxies attribute reads to parent SwarmConfig."""
        from Jotty.core.infrastructure.foundation.data_structures import ExecutionView, SwarmConfig

        config = SwarmConfig(max_actor_iters=42)
        view = ExecutionView(config)
        assert view.max_actor_iters == 42

    @pytest.mark.unit
    def test_config_view_setattr_writes_parent(self):
        """ConfigView proxies attribute writes to parent SwarmConfig."""
        from Jotty.core.infrastructure.foundation.data_structures import ExecutionView, SwarmConfig

        config = SwarmConfig(max_actor_iters=10)
        view = ExecutionView(config)
        view.max_actor_iters = 99
        assert config.max_actor_iters == 99

    @pytest.mark.unit
    def test_config_view_getattr_invalid_raises(self):
        """ConfigView raises AttributeError for fields not in its _FIELDS."""
        from Jotty.core.infrastructure.foundation.data_structures import ExecutionView, SwarmConfig

        config = SwarmConfig()
        view = ExecutionView(config)
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = view.nonexistent_field

    @pytest.mark.unit
    def test_config_view_to_dict(self):
        """ConfigView.to_dict() returns only its fields."""
        from Jotty.core.infrastructure.foundation.data_structures import (
            PersistenceView,
            SwarmConfig,
        )

        config = SwarmConfig()
        view = PersistenceView(config)
        d = view.to_dict()
        assert isinstance(d, dict)
        assert "persist_memories" in d
        # Should NOT have execution fields
        assert "max_actor_iters" not in d

    @pytest.mark.unit
    def test_config_view_repr(self):
        """ConfigView.__repr__() contains class name and field values."""
        from Jotty.core.infrastructure.foundation.data_structures import MonitoringView, SwarmConfig

        config = SwarmConfig()
        view = MonitoringView(config)
        rep = repr(view)
        assert "MonitoringView(" in rep

    @pytest.mark.unit
    def test_persistence_view_fields(self):
        """PersistenceView has expected fields."""
        from Jotty.core.infrastructure.foundation.data_structures import (
            PersistenceView,
            SwarmConfig,
        )

        view = PersistenceView(SwarmConfig())
        d = view.to_dict()
        assert "auto_save_interval" in d
        assert "persist_memories" in d

    @pytest.mark.unit
    def test_memory_view_fields(self):
        """MemoryView has expected fields."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryView, SwarmConfig

        view = MemoryView(SwarmConfig())
        d = view.to_dict()
        assert "episodic_capacity" in d
        assert "enable_llm_rag" in d

    @pytest.mark.unit
    def test_learning_view_fields(self):
        """LearningView has expected fields."""
        from Jotty.core.infrastructure.foundation.data_structures import LearningView, SwarmConfig

        view = LearningView(SwarmConfig())
        d = view.to_dict()
        assert "gamma" in d
        assert "alpha" in d

    @pytest.mark.unit
    def test_swarm_config_view_property(self):
        """SwarmConfig exposes views via properties."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig

        config = SwarmConfig()
        # Check that view properties exist and return the right type
        assert hasattr(config, "execution")
        assert hasattr(config, "persistence")
        assert hasattr(config, "memory_settings")
        assert hasattr(config, "learning")


# =============================================================================
# SharedScratchpad Tests
# =============================================================================


class TestSharedScratchpad:
    """Tests for SharedScratchpad inter-agent communication."""

    @pytest.mark.unit
    def test_add_message(self):
        """add_message appends to messages list."""
        from Jotty.core.infrastructure.foundation.data_structures import (
            AgentMessage,
            CommunicationType,
            SharedScratchpad,
        )

        pad = SharedScratchpad()
        msg = AgentMessage(
            sender="agent1",
            receiver="agent2",
            message_type=CommunicationType.INSIGHT,
            content={"data": "test"},
        )
        pad.add_message(msg)
        assert len(pad.messages) == 1
        assert pad.messages[0].sender == "agent1"

    @pytest.mark.unit
    def test_add_message_caches_tool_result(self):
        """Tool result messages are cached for reuse."""
        from Jotty.core.infrastructure.foundation.data_structures import (
            AgentMessage,
            CommunicationType,
            SharedScratchpad,
        )

        pad = SharedScratchpad()
        msg = AgentMessage(
            sender="agent1",
            receiver="*",
            message_type=CommunicationType.TOOL_RESULT,
            content={},
            tool_name="web_search",
            tool_args={"query": "test"},
            tool_result={"results": ["r1", "r2"]},
        )
        pad.add_message(msg)
        cached = pad.get_cached_result("web_search", {"query": "test"})
        assert cached == {"results": ["r1", "r2"]}

    @pytest.mark.unit
    def test_get_cached_result_miss(self):
        """get_cached_result returns None on cache miss."""
        from Jotty.core.infrastructure.foundation.data_structures import SharedScratchpad

        pad = SharedScratchpad()
        assert pad.get_cached_result("nonexistent", {}) is None

    @pytest.mark.unit
    def test_get_messages_for_specific_receiver(self):
        """get_messages_for returns messages for specific receiver."""
        from Jotty.core.infrastructure.foundation.data_structures import (
            AgentMessage,
            CommunicationType,
            SharedScratchpad,
        )

        pad = SharedScratchpad()
        pad.add_message(
            AgentMessage(
                sender="a", receiver="b", message_type=CommunicationType.INSIGHT, content={}
            )
        )
        pad.add_message(
            AgentMessage(
                sender="a", receiver="c", message_type=CommunicationType.INSIGHT, content={}
            )
        )
        msgs_b = pad.get_messages_for("b")
        assert len(msgs_b) == 1

    @pytest.mark.unit
    def test_get_messages_for_broadcast(self):
        """get_messages_for includes broadcast messages."""
        from Jotty.core.infrastructure.foundation.data_structures import (
            AgentMessage,
            CommunicationType,
            SharedScratchpad,
        )

        pad = SharedScratchpad()
        pad.add_message(
            AgentMessage(
                sender="a", receiver="*", message_type=CommunicationType.INSIGHT, content={}
            )
        )
        pad.add_message(
            AgentMessage(
                sender="a", receiver="b", message_type=CommunicationType.INSIGHT, content={}
            )
        )
        # "c" should get broadcast only
        msgs_c = pad.get_messages_for("c")
        assert len(msgs_c) == 1
        # "b" gets both broadcast and direct
        msgs_b = pad.get_messages_for("b")
        assert len(msgs_b) == 2

    @pytest.mark.unit
    def test_clear(self):
        """clear() empties all data structures."""
        from Jotty.core.infrastructure.foundation.data_structures import (
            AgentMessage,
            CommunicationType,
            SharedScratchpad,
        )

        pad = SharedScratchpad()
        pad.add_message(
            AgentMessage(
                sender="a",
                receiver="*",
                message_type=CommunicationType.TOOL_RESULT,
                content={},
                tool_name="t",
                tool_args={},
                tool_result="r",
            )
        )
        pad.shared_insights.append("insight1")
        pad.clear()
        assert len(pad.messages) == 0
        assert len(pad.tool_cache) == 0
        assert len(pad.shared_insights) == 0


# =============================================================================
# AgentContribution Tests
# =============================================================================


class TestAgentContribution:
    """Tests for AgentContribution credit assignment."""

    @pytest.mark.unit
    def test_compute_final_contribution_all_max(self):
        """Maximum values produce contribution > base score."""
        from Jotty.core.infrastructure.foundation.data_structures import AgentContribution

        contrib = AgentContribution(
            agent_name="planner",
            contribution_score=1.0,
            decision="approve",
            decision_correct=True,
            counterfactual_impact=1.0,
            reasoning_quality=1.0,
            evidence_used=["e1"],
            tools_used=["t1"],
            decision_timing=1.0,
            temporal_weight=1.0,
        )
        final = contrib.compute_final_contribution()
        assert final == 1.0  # 1.0 * 1.0 * 1.0 * 1.0

    @pytest.mark.unit
    def test_compute_final_contribution_all_min(self):
        """Zero values produce reduced contribution."""
        from Jotty.core.infrastructure.foundation.data_structures import AgentContribution

        contrib = AgentContribution(
            agent_name="planner",
            contribution_score=1.0,
            decision="abstain",
            decision_correct=False,
            counterfactual_impact=0.0,
            reasoning_quality=0.0,
            evidence_used=[],
            tools_used=[],
            decision_timing=0.0,
            temporal_weight=0.0,
        )
        final = contrib.compute_final_contribution()
        # 1.0 * 0.5 * 0.5 * 0.7 = 0.175
        assert abs(final - 0.175) < 0.01

    @pytest.mark.unit
    def test_compute_final_contribution_mid_values(self):
        """Mid-range values produce expected contribution."""
        from Jotty.core.infrastructure.foundation.data_structures import AgentContribution

        contrib = AgentContribution(
            agent_name="auditor",
            contribution_score=0.8,
            decision="approve",
            decision_correct=True,
            counterfactual_impact=0.5,
            reasoning_quality=0.6,
            evidence_used=["memory"],
            tools_used=["reason_about"],
            decision_timing=0.5,
            temporal_weight=0.5,
        )
        final = contrib.compute_final_contribution()
        # 0.8 * (0.5 + 0.3) * (0.5 + 0.25) * (0.7 + 0.15) = 0.8 * 0.8 * 0.75 * 0.85
        expected = 0.8 * 0.8 * 0.75 * 0.85
        assert abs(final - expected) < 0.01

    @pytest.mark.unit
    def test_negative_contribution_score(self):
        """Negative base score produces negative contribution."""
        from Jotty.core.infrastructure.foundation.data_structures import AgentContribution

        contrib = AgentContribution(
            agent_name="bad_agent",
            contribution_score=-0.5,
            decision="reject",
            decision_correct=False,
            counterfactual_impact=0.8,
            reasoning_quality=0.3,
            evidence_used=[],
            tools_used=[],
            decision_timing=0.5,
            temporal_weight=0.5,
        )
        final = contrib.compute_final_contribution()
        assert final < 0


# =============================================================================
# LearningMetrics Tests
# =============================================================================


class TestLearningMetrics:
    """Tests for LearningMetrics health monitoring."""

    @pytest.mark.unit
    def test_get_success_rate_empty(self):
        """Success rate defaults to 0.5 when no data."""
        from Jotty.core.infrastructure.foundation.data_structures import LearningMetrics

        metrics = LearningMetrics()
        assert metrics.get_success_rate() == 0.5

    @pytest.mark.unit
    def test_get_success_rate_all_success(self):
        """Success rate is 1.0 when all recent are successful."""
        from Jotty.core.infrastructure.foundation.data_structures import LearningMetrics

        metrics = LearningMetrics()
        metrics.recent_successes = [True] * 10
        assert metrics.get_success_rate() == 1.0

    @pytest.mark.unit
    def test_get_success_rate_mixed(self):
        """Success rate reflects actual mix."""
        from Jotty.core.infrastructure.foundation.data_structures import LearningMetrics

        metrics = LearningMetrics()
        metrics.recent_successes = [True, False, True, True, False]
        assert metrics.get_success_rate() == 0.6

    @pytest.mark.unit
    def test_get_success_rate_windowed(self):
        """Success rate respects window parameter."""
        from Jotty.core.infrastructure.foundation.data_structures import LearningMetrics

        metrics = LearningMetrics()
        metrics.recent_successes = [False] * 100 + [True] * 10
        rate = metrics.get_success_rate(window=10)
        assert rate == 1.0

    @pytest.mark.unit
    def test_get_learning_velocity_empty(self):
        """Learning velocity is 0 with insufficient data."""
        from Jotty.core.infrastructure.foundation.data_structures import LearningMetrics

        metrics = LearningMetrics()
        assert metrics.get_learning_velocity() == 0.0

    @pytest.mark.unit
    def test_get_learning_velocity_with_changes(self):
        """Learning velocity reflects magnitude of value changes."""
        from Jotty.core.infrastructure.foundation.data_structures import LearningMetrics

        metrics = LearningMetrics()
        metrics.value_changes = [0.1, -0.2, 0.15, -0.05, 0.3]
        velocity = metrics.get_learning_velocity()
        expected = sum(abs(v) for v in metrics.value_changes) / len(metrics.value_changes)
        assert abs(velocity - expected) < 0.001

    @pytest.mark.unit
    def test_is_learning_stalled_no_data(self):
        """Stalled check with no data returns True (velocity = 0)."""
        from Jotty.core.infrastructure.foundation.data_structures import LearningMetrics

        metrics = LearningMetrics()
        assert metrics.is_learning_stalled() is True

    @pytest.mark.unit
    def test_is_learning_stalled_active(self):
        """Active learning is not stalled."""
        from Jotty.core.infrastructure.foundation.data_structures import LearningMetrics

        metrics = LearningMetrics()
        metrics.value_changes = [0.1, 0.2, 0.15, 0.1, 0.2]
        assert metrics.is_learning_stalled() is False

    @pytest.mark.unit
    def test_is_learning_stalled_flat(self):
        """Flat learning (tiny changes) is stalled."""
        from Jotty.core.infrastructure.foundation.data_structures import LearningMetrics

        metrics = LearningMetrics()
        metrics.value_changes = [0.0001, 0.0002, 0.00005, 0.0001]
        assert metrics.is_learning_stalled(threshold=0.001) is True


# =============================================================================
# AgentResult Tests
# =============================================================================


class TestAgentResult:
    """Tests for AgentResult serialization."""

    @pytest.mark.unit
    def test_to_dict(self):
        """AgentResult.to_dict() includes all fields."""
        from Jotty.core.modes.agent.base.base_agent import AgentResult

        result = AgentResult(
            success=True,
            output="Hello",
            agent_name="test_agent",
            execution_time=1.5,
            retries=2,
            error=None,
            metadata={"key": "val"},
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["output"] == "Hello"
        assert d["agent_name"] == "test_agent"
        assert d["execution_time"] == 1.5
        assert d["retries"] == 2
        assert d["error"] is None
        assert d["metadata"] == {"key": "val"}
        assert "timestamp" in d

    @pytest.mark.unit
    def test_to_dict_with_error(self):
        """AgentResult.to_dict() serializes errors correctly."""
        from Jotty.core.modes.agent.base.base_agent import AgentResult

        result = AgentResult(
            success=False,
            output=None,
            error="Something went wrong",
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["error"] == "Something went wrong"


# =============================================================================
# BaseAgent Memory and Context Tests
# =============================================================================


class TestBaseAgentMemoryContext:
    """Tests for BaseAgent store_memory, retrieve_memory, context methods."""

    def _make_agent(self):
        from Jotty.core.modes.agent.base.base_agent import AgentRuntimeConfig, BaseAgent

        class _Dummy(BaseAgent):
            async def _execute_impl(self, **kw):
                return {}

        a = _Dummy(AgentRuntimeConfig(name="test"))
        a._initialized = True
        return a

    @pytest.mark.unit
    def test_store_memory_no_memory(self):
        """store_memory is no-op when memory is None."""
        agent = self._make_agent()
        agent._memory = None
        agent.store_memory("test content")  # Should not raise

    @pytest.mark.unit
    def test_store_memory_calls_memory_store(self):
        """store_memory delegates to self.memory.store()."""
        agent = self._make_agent()
        mock_memory = MagicMock()
        agent._memory = mock_memory
        agent.store_memory("test content", level="semantic", goal="test goal")
        mock_memory.store.assert_called_once()
        call_kwargs = mock_memory.store.call_args[1]
        assert call_kwargs["content"] == "test content"
        assert call_kwargs["goal"] == "test goal"

    @pytest.mark.unit
    def test_store_memory_exception_suppressed(self):
        """store_memory suppresses exceptions."""
        agent = self._make_agent()
        mock_memory = MagicMock()
        mock_memory.store.side_effect = RuntimeError("store failed")
        agent._memory = mock_memory
        agent.store_memory("test")  # Should not raise

    @pytest.mark.unit
    def test_retrieve_memory_no_memory(self):
        """retrieve_memory returns empty list when memory is None."""
        agent = self._make_agent()
        agent._memory = None
        result = agent.retrieve_memory("test query")
        assert result == []

    @pytest.mark.unit
    def test_retrieve_memory_calls_memory_retrieve(self):
        """retrieve_memory delegates to self.memory.retrieve()."""
        agent = self._make_agent()
        mock_memory = MagicMock()
        mock_memory.retrieve.return_value = ["mem1", "mem2"]
        agent._memory = mock_memory
        result = agent.retrieve_memory("test query", goal="goal", budget_tokens=500)
        assert result == ["mem1", "mem2"]
        mock_memory.retrieve.assert_called_once_with(
            query="test query", goal="goal", budget_tokens=500
        )

    @pytest.mark.unit
    def test_retrieve_memory_exception_returns_empty(self):
        """retrieve_memory returns [] on exception."""
        agent = self._make_agent()
        mock_memory = MagicMock()
        mock_memory.retrieve.side_effect = RuntimeError("retrieve failed")
        agent._memory = mock_memory
        result = agent.retrieve_memory("test")
        assert result == []

    @pytest.mark.unit
    def test_register_context(self):
        """register_context delegates to context.set()."""
        agent = self._make_agent()
        mock_context = MagicMock()
        agent._context_manager = mock_context
        agent.register_context("key1", "value1")
        mock_context.set.assert_called_once_with("key1", "value1")

    @pytest.mark.unit
    def test_register_context_no_context(self):
        """register_context is no-op when context is None."""
        agent = self._make_agent()
        agent._context_manager = None
        agent.config.enable_context = False
        agent.register_context("key", "val")  # Should not raise

    @pytest.mark.unit
    def test_get_context(self):
        """get_context delegates to context.get()."""
        agent = self._make_agent()
        mock_context = MagicMock()
        mock_context.get.return_value = "found_value"
        agent._context_manager = mock_context
        result = agent.get_context("key1")
        assert result == "found_value"

    @pytest.mark.unit
    def test_get_context_no_context_returns_default(self):
        """get_context returns default when context is None."""
        agent = self._make_agent()
        agent._context_manager = None
        agent.config.enable_context = False
        result = agent.get_context("key1", default="fallback")
        assert result == "fallback"

    @pytest.mark.unit
    def test_get_compressed_context_no_context(self):
        """get_compressed_context returns '' when context is None."""
        agent = self._make_agent()
        agent._context_manager = None
        agent.config.enable_context = False
        assert agent.get_compressed_context() == ""

    @pytest.mark.unit
    def test_get_compressed_context_with_data(self):
        """get_compressed_context returns JSON string of known keys."""
        agent = self._make_agent()
        mock_context = MagicMock()
        mock_context.get.side_effect = lambda key, *a: {
            "current_task": "do X",
            "current_goal": "achieve Y",
        }.get(key)
        agent._context_manager = mock_context
        result = agent.get_compressed_context()
        assert "do X" in result
        assert "achieve Y" in result

    @pytest.mark.unit
    def test_get_compressed_context_truncation(self):
        """get_compressed_context truncates when exceeding max_tokens."""
        agent = self._make_agent()
        mock_context = MagicMock()
        long_value = "x" * 50000
        mock_context.get.side_effect = lambda key, *a: {
            "current_task": long_value,
        }.get(key)
        agent._context_manager = mock_context
        result = agent.get_compressed_context(max_tokens=100)
        assert len(result) <= 100 * 4 + 10  # 4 chars/token + margin for "..."
        assert result.endswith("...")


# =============================================================================
# GoalNode Tests
# =============================================================================


class TestGoalNode:
    """Tests for GoalNode similarity scoring."""

    @pytest.mark.unit
    def test_similarity_same_domain(self):
        """Same domain adds 0.4 to similarity."""
        from Jotty.core.infrastructure.foundation.types.memory_types import GoalNode

        a = GoalNode(goal_id="a", goal_text="query sales", domain="sql", operation_type="query")
        b = GoalNode(goal_id="b", goal_text="query users", domain="sql", operation_type="analysis")
        score = a.similarity_score(b)
        assert score >= 0.4  # At least domain match

    @pytest.mark.unit
    def test_similarity_same_operation(self):
        """Same operation type adds 0.2 to similarity."""
        from Jotty.core.infrastructure.foundation.types.memory_types import GoalNode

        a = GoalNode(goal_id="a", goal_text="query X", domain="sql", operation_type="query")
        b = GoalNode(goal_id="b", goal_text="query Y", domain="python", operation_type="query")
        score = a.similarity_score(b)
        assert score >= 0.2  # At least operation match

    @pytest.mark.unit
    def test_similarity_entity_overlap(self):
        """Entity overlap contributes to similarity."""
        from Jotty.core.infrastructure.foundation.types.memory_types import GoalNode

        a = GoalNode(goal_id="a", goal_text="q1", domain="sql", entities=["users", "orders"])
        b = GoalNode(goal_id="b", goal_text="q2", domain="sql", entities=["users", "products"])
        score = a.similarity_score(b)
        # domain=0.4, entities overlap 1/3 * 0.3 = 0.1
        assert score > 0.4

    @pytest.mark.unit
    def test_similarity_hierarchical_parent(self):
        """Parent-child relationship adds 0.1 to similarity."""
        from Jotty.core.infrastructure.foundation.types.memory_types import GoalNode

        parent = GoalNode(goal_id="parent", goal_text="data analysis", domain="general")
        child = GoalNode(goal_id="child", goal_text="sql queries", domain="sql", parent_id="parent")
        score = child.similarity_score(parent)
        assert score >= 0.1  # At least hierarchical bonus

    @pytest.mark.unit
    def test_similarity_completely_different(self):
        """Completely different nodes have low similarity."""
        from Jotty.core.infrastructure.foundation.types.memory_types import GoalNode

        a = GoalNode(goal_id="a", goal_text="sql query", domain="sql", operation_type="query")
        b = GoalNode(goal_id="b", goal_text="draw chart", domain="viz", operation_type="render")
        score = a.similarity_score(b)
        assert score == 0.0

    @pytest.mark.unit
    def test_similarity_capped_at_one(self):
        """Similarity score cannot exceed 1.0."""
        from Jotty.core.infrastructure.foundation.types.memory_types import GoalNode

        a = GoalNode(
            goal_id="a",
            goal_text="q",
            domain="sql",
            operation_type="query",
            entities=["users"],
            parent_id="b",
        )
        b = GoalNode(
            goal_id="b", goal_text="q", domain="sql", operation_type="query", entities=["users"]
        )
        score = a.similarity_score(b)
        assert score <= 1.0

    @pytest.mark.unit
    def test_similarity_no_entities(self):
        """Missing entities don't add entity overlap score."""
        from Jotty.core.infrastructure.foundation.types.memory_types import GoalNode

        a = GoalNode(goal_id="a", goal_text="q", domain="sql", operation_type="query", entities=[])
        b = GoalNode(
            goal_id="b", goal_text="q", domain="sql", operation_type="query", entities=["users"]
        )
        score = a.similarity_score(b)
        # domain=0.4 + operation=0.2, no entity overlap since a has no entities
        assert abs(score - 0.6) < 0.01


# =============================================================================
# GoalHierarchy Tests
# =============================================================================


class TestGoalHierarchy:
    """Tests for GoalHierarchy goal management and knowledge transfer."""

    @pytest.mark.unit
    def test_add_goal(self):
        """add_goal creates a new node and returns its ID."""
        from Jotty.core.infrastructure.foundation.types.memory_types import GoalHierarchy

        h = GoalHierarchy()
        goal_id = h.add_goal("analyze sales data", domain="sql")
        assert goal_id in h.nodes
        assert h.nodes[goal_id].goal_text == "analyze sales data"
        assert h.nodes[goal_id].domain == "sql"

    @pytest.mark.unit
    def test_add_duplicate_increments_count(self):
        """Adding the same goal text increments episode_count."""
        from Jotty.core.infrastructure.foundation.types.memory_types import GoalHierarchy

        h = GoalHierarchy()
        goal_id = h.add_goal("analyze sales data")
        assert h.nodes[goal_id].episode_count == 0
        goal_id2 = h.add_goal("analyze sales data")
        assert goal_id == goal_id2
        assert h.nodes[goal_id].episode_count == 1

    @pytest.mark.unit
    def test_add_goal_with_parent(self):
        """add_goal with explicit parent_id links to parent."""
        from Jotty.core.infrastructure.foundation.types.memory_types import GoalHierarchy, GoalNode

        h = GoalHierarchy()
        # Create parent first
        parent_id = h.add_goal("data analysis", domain="general", operation_type="general")
        child_id = h.add_goal("sql queries", domain="sql", parent_id=parent_id)
        assert h.nodes[child_id].parent_id == parent_id
        assert child_id in h.nodes[parent_id].children_ids

    @pytest.mark.unit
    def test_find_best_parent_by_domain(self):
        """_find_best_parent returns domain node with operation_type='general'."""
        from Jotty.core.infrastructure.foundation.types.memory_types import GoalHierarchy

        h = GoalHierarchy()
        h.add_goal("sql general", domain="sql", operation_type="general")
        parent = h._find_best_parent("sql", "query")
        assert parent is not None
        assert h.nodes[parent].domain == "sql"

    @pytest.mark.unit
    def test_find_best_parent_no_match(self):
        """_find_best_parent returns None when no matching domain exists and no root."""
        from Jotty.core.infrastructure.foundation.types.memory_types import GoalHierarchy

        h = GoalHierarchy()
        parent = h._find_best_parent("unknown_domain", "query")
        assert parent is None

    @pytest.mark.unit
    def test_get_related_goals_empty(self):
        """get_related_goals returns empty for unknown goal_id."""
        from Jotty.core.infrastructure.foundation.types.memory_types import GoalHierarchy

        h = GoalHierarchy()
        assert h.get_related_goals("nonexistent") == []

    @pytest.mark.unit
    def test_get_related_goals_finds_similar(self):
        """get_related_goals returns goals with similarity > 0.3."""
        from Jotty.core.infrastructure.foundation.types.memory_types import GoalHierarchy

        h = GoalHierarchy()
        id1 = h.add_goal("sales queries", domain="sql", operation_type="query", entities=["sales"])
        id2 = h.add_goal("user queries", domain="sql", operation_type="query", entities=["users"])
        id3 = h.add_goal("draw chart", domain="viz", operation_type="render")
        related = h.get_related_goals(id1)
        related_ids = [r[0] for r in related]
        # id2 has same domain+operation → similarity >= 0.6, should be included
        assert id2 in related_ids
        # id3 has nothing in common → similarity 0.0, should not be included
        assert id3 not in related_ids

    @pytest.mark.unit
    def test_get_related_goals_sorted_by_similarity(self):
        """get_related_goals returns results sorted by similarity descending."""
        from Jotty.core.infrastructure.foundation.types.memory_types import GoalHierarchy

        h = GoalHierarchy()
        id1 = h.add_goal("sales query", domain="sql", operation_type="query", entities=["sales"])
        id2 = h.add_goal("user query", domain="sql", operation_type="query", entities=["users"])
        id3 = h.add_goal(
            "sales analysis", domain="sql", operation_type="analysis", entities=["sales"]
        )
        related = h.get_related_goals(id1)
        if len(related) >= 2:
            assert related[0][1] >= related[1][1]  # First has higher or equal similarity


# =============================================================================
# CausalLink Tests
# =============================================================================


class TestCausalLink:
    """Tests for CausalLink context checking and confidence updates."""

    @pytest.mark.unit
    def test_applies_in_context_no_conditions(self):
        """CausalLink with no conditions applies in any context."""
        from Jotty.core.infrastructure.foundation.types.memory_types import CausalLink

        link = CausalLink(cause="type annotation", effect="correct parsing")
        assert link.applies_in_context({"database": "trino"}) is True

    @pytest.mark.unit
    def test_applies_in_context_matching_condition(self):
        """CausalLink applies when conditions match."""
        from Jotty.core.infrastructure.foundation.types.memory_types import CausalLink

        link = CausalLink(
            cause="type annotation", effect="correct parsing", conditions=["database=trino"]
        )
        assert link.applies_in_context({"database": "trino"}) is True
        assert link.applies_in_context({"database": "postgres"}) is False

    @pytest.mark.unit
    def test_applies_in_context_exception_blocks(self):
        """CausalLink does not apply when exception matches."""
        from Jotty.core.infrastructure.foundation.types.memory_types import CausalLink

        link = CausalLink(
            cause="type annotation",
            effect="correct parsing",
            conditions=["database=trino"],
            exceptions=["version=2.0"],
        )
        assert link.applies_in_context({"database": "trino"}) is True
        assert link.applies_in_context({"database": "trino", "version": "2.0"}) is False

    @pytest.mark.unit
    def test_applies_in_context_missing_condition_key(self):
        """Missing context key passes condition check (not contradicted)."""
        from Jotty.core.infrastructure.foundation.types.memory_types import CausalLink

        link = CausalLink(cause="X", effect="Y", conditions=["database=trino"])
        # "database" not in context → condition not contradicted → applies
        assert link.applies_in_context({"other_key": "val"}) is True

    @pytest.mark.unit
    def test_update_confidence_supported(self):
        """Supported evidence increases confidence."""
        from Jotty.core.infrastructure.foundation.types.memory_types import CausalLink

        link = CausalLink(cause="X", effect="Y", confidence=0.5)
        link.update_confidence(supported=True)
        assert link.confidence > 0.5
        assert link.confidence <= 0.99

    @pytest.mark.unit
    def test_update_confidence_contradicted(self):
        """Contradicting evidence decreases confidence."""
        from Jotty.core.infrastructure.foundation.types.memory_types import CausalLink

        link = CausalLink(cause="X", effect="Y", confidence=0.5)
        link.update_confidence(supported=False)
        assert link.confidence < 0.5
        assert link.confidence >= 0.1

    @pytest.mark.unit
    def test_update_confidence_clamped(self):
        """Confidence stays within [0.1, 0.99] bounds."""
        from Jotty.core.infrastructure.foundation.types.memory_types import CausalLink

        link = CausalLink(cause="X", effect="Y", confidence=0.99)
        for _ in range(20):
            link.update_confidence(supported=True)
        assert link.confidence <= 0.99

        link2 = CausalLink(cause="X", effect="Y", confidence=0.1)
        for _ in range(20):
            link2.update_confidence(supported=False)
        assert link2.confidence >= 0.1


# =============================================================================
# MemoryEntry Tests
# =============================================================================


class TestMemoryEntry:
    """Tests for MemoryEntry value retrieval and UCB scoring."""

    @pytest.mark.unit
    def test_get_value_exact_match(self):
        """get_value returns exact goal value when present."""
        from Jotty.core.infrastructure.foundation.types.memory_types import (
            GoalValue,
            MemoryEntry,
            MemoryLevel,
        )

        entry = MemoryEntry(
            key="k1",
            content="test content",
            level=MemoryLevel.EPISODIC,
            context={},
            goal_values={"goal_a": GoalValue(value=0.8)},
        )
        assert entry.get_value("goal_a") == 0.8

    @pytest.mark.unit
    def test_get_value_default(self):
        """get_value returns default_value when goal not found."""
        from Jotty.core.infrastructure.foundation.types.memory_types import MemoryEntry, MemoryLevel

        entry = MemoryEntry(
            key="k1", content="test", level=MemoryLevel.EPISODIC, context={}, default_value=0.3
        )
        assert entry.get_value("unknown_goal") == 0.3

    @pytest.mark.unit
    def test_get_ucb_score_unexplored(self):
        """Unexplored entry (ucb_visits=0) returns infinity."""
        from Jotty.core.infrastructure.foundation.types.memory_types import MemoryEntry, MemoryLevel

        entry = MemoryEntry(
            key="k1", content="test", level=MemoryLevel.EPISODIC, context={}, ucb_visits=0
        )
        score = entry.get_ucb_score("goal", total_accesses=10)
        assert score == float("inf")

    @pytest.mark.unit
    def test_get_ucb_score_explored(self):
        """Explored entry returns value + exploration bonus."""
        import math

        from Jotty.core.infrastructure.foundation.types.memory_types import (
            GoalValue,
            MemoryEntry,
            MemoryLevel,
        )

        entry = MemoryEntry(
            key="k1",
            content="test",
            level=MemoryLevel.EPISODIC,
            context={},
            ucb_visits=5,
            goal_values={"goal": GoalValue(value=0.7)},
        )
        score = entry.get_ucb_score("goal", total_accesses=100, c=2.0)
        expected_bonus = 2.0 * math.sqrt(math.log(101) / 5)
        expected = 0.7 + expected_bonus
        assert abs(score - expected) < 0.001

    @pytest.mark.unit
    def test_post_init_content_hash(self):
        """__post_init__ computes content_hash from content."""
        import hashlib

        from Jotty.core.infrastructure.foundation.types.memory_types import MemoryEntry, MemoryLevel

        entry = MemoryEntry(key="k", content="hello world", level=MemoryLevel.SEMANTIC, context={})
        expected_hash = hashlib.md5("hello world".encode()).hexdigest()
        assert entry.content_hash == expected_hash

    @pytest.mark.unit
    def test_post_init_token_count(self):
        """__post_init__ estimates token_count from content length."""
        from Jotty.core.infrastructure.foundation.types.memory_types import MemoryEntry, MemoryLevel

        content = "a" * 100
        entry = MemoryEntry(key="k", content=content, level=MemoryLevel.SEMANTIC, context={})
        assert entry.token_count == 100 // 4 + 1


# =============================================================================
# AdaptiveThreshold Tests
# =============================================================================


class TestAdaptiveThreshold:
    """Tests for AdaptiveThreshold Welford's running statistics."""

    @pytest.mark.unit
    def test_initial_state(self):
        """AdaptiveThreshold starts with initial mean and std."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveThreshold

        t = AdaptiveThreshold(initial_mean=0.5, initial_std=0.2)
        assert t.mean == 0.5
        assert t.std == 0.2
        assert t.count == 0

    @pytest.mark.unit
    def test_update_mean(self):
        """update() adjusts mean with Welford's algorithm."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveThreshold

        t = AdaptiveThreshold(initial_mean=0.0, initial_std=0.0)
        t.update(1.0)
        assert t.mean == 1.0
        t.update(3.0)
        assert t.mean == 2.0  # (1+3)/2

    @pytest.mark.unit
    def test_update_std(self):
        """update() computes running standard deviation."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveThreshold

        t = AdaptiveThreshold(initial_mean=0.0, initial_std=0.0)
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
            t.update(v)
        # Expected std for this dataset ≈ 2.0
        assert abs(t.mean - 5.0) < 0.01
        assert t.std > 0

    @pytest.mark.unit
    def test_is_high(self):
        """is_high returns True for values above mean + sigma * std."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveThreshold

        t = AdaptiveThreshold(initial_mean=0.5, initial_std=0.1)
        # High threshold = 0.5 + 1.5 * max(0.1, 0.1) = 0.65
        assert t.is_high(0.8) is True
        assert t.is_high(0.5) is False

    @pytest.mark.unit
    def test_is_low(self):
        """is_low returns True for values below mean - sigma * std."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveThreshold

        t = AdaptiveThreshold(initial_mean=0.5, initial_std=0.1)
        # Low threshold = 0.5 - 1.5 * 0.1 = 0.35
        assert t.is_low(0.2) is True
        assert t.is_low(0.5) is False

    @pytest.mark.unit
    def test_is_extreme(self):
        """is_extreme returns True for values far from mean in either direction."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveThreshold

        t = AdaptiveThreshold(initial_mean=0.5, initial_std=0.1)
        # Extreme threshold = 2.0 * max(0.1, 0.1) = 0.2
        assert t.is_extreme(0.8) is True  # |0.8 - 0.5| = 0.3 > 0.2
        assert t.is_extreme(0.2) is True  # |0.2 - 0.5| = 0.3 > 0.2
        assert t.is_extreme(0.5) is False

    @pytest.mark.unit
    def test_std_floor(self):
        """Std uses floor of 0.1 for is_high/is_low/is_extreme."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveThreshold

        t = AdaptiveThreshold(initial_mean=0.5, initial_std=0.01)
        # std=0.01 but floor is 0.1, so threshold = 0.5 + 1.5 * 0.1 = 0.65
        assert t.is_high(0.7) is True
        assert t.is_high(0.6) is False


# =============================================================================
# EpsilonGreedy Tests
# =============================================================================


class TestEpsilonGreedy:
    """Tests for EpsilonGreedy deterministic exploration."""

    @pytest.mark.unit
    def test_epsilon_decays(self):
        """Epsilon decays after each should_explore call."""
        from Jotty.core.infrastructure.foundation.robust_parsing import EpsilonGreedy

        eg = EpsilonGreedy(initial_epsilon=0.3, decay=0.9, min_epsilon=0.05)
        initial = eg.epsilon
        eg.should_explore()
        assert eg.epsilon < initial
        assert eg.epsilon == max(0.05, initial * 0.9)

    @pytest.mark.unit
    def test_epsilon_min_floor(self):
        """Epsilon never goes below min_epsilon."""
        from Jotty.core.infrastructure.foundation.robust_parsing import EpsilonGreedy

        eg = EpsilonGreedy(initial_epsilon=0.1, decay=0.5, min_epsilon=0.05)
        for _ in range(100):
            eg.should_explore()
        assert eg.epsilon >= 0.05

    @pytest.mark.unit
    def test_decide_exploit(self):
        """decide() returns exploit_decision when not exploring."""
        from Jotty.core.infrastructure.foundation.robust_parsing import EpsilonGreedy

        eg = EpsilonGreedy(initial_epsilon=0.0, min_epsilon=0.0)
        # With epsilon=0, should never explore
        result = eg.decide(exploit_decision=False)
        assert result is False

    @pytest.mark.unit
    def test_decision_count_increments(self):
        """Decision count increments on each should_explore call."""
        from Jotty.core.infrastructure.foundation.robust_parsing import EpsilonGreedy

        eg = EpsilonGreedy()
        assert eg.decision_count == 0
        eg.should_explore()
        assert eg.decision_count == 1
        eg.should_explore()
        assert eg.decision_count == 2


# =============================================================================
# AdaptiveWeight Tests
# =============================================================================


class TestAdaptiveWeight:
    """Tests for AdaptiveWeight momentum-based gradient descent."""

    @pytest.mark.unit
    def test_initial_values(self):
        """AdaptiveWeight initializes with given values."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveWeight

        w = AdaptiveWeight(name="test", value=0.5, learning_rate=0.02)
        assert w.name == "test"
        assert w.value == 0.5
        assert w.learning_rate == 0.02
        assert w.updates == 0

    @pytest.mark.unit
    def test_update_increases_with_positive_gradient(self):
        """Positive gradient increases weight value."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveWeight

        w = AdaptiveWeight(name="test", value=0.5)
        w.update(gradient=1.0, reward=1.0)
        assert w.value > 0.5
        assert w.updates == 1

    @pytest.mark.unit
    def test_update_decreases_with_negative_gradient(self):
        """Negative gradient decreases weight value."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveWeight

        w = AdaptiveWeight(name="test", value=0.5)
        w.update(gradient=-1.0, reward=1.0)
        assert w.value < 0.5

    @pytest.mark.unit
    def test_update_clamps_value(self):
        """Weight value stays within [0.05, 0.95]."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveWeight

        w = AdaptiveWeight(name="test", value=0.94, learning_rate=0.5)
        w.update(gradient=1.0, reward=1.0)
        assert w.value <= 0.95

        w2 = AdaptiveWeight(name="test", value=0.06, learning_rate=0.5)
        w2.update(gradient=-1.0, reward=1.0)
        assert w2.value >= 0.05

    @pytest.mark.unit
    def test_to_dict_roundtrip(self):
        """to_dict/from_dict preserves all fields."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveWeight

        w = AdaptiveWeight(name="test", value=0.6, momentum=0.1, learning_rate=0.02, updates=5)
        d = w.to_dict()
        w2 = AdaptiveWeight.from_dict(d)
        assert w2.name == w.name
        assert w2.value == w.value
        assert w2.momentum == w.momentum
        assert w2.updates == w.updates


# =============================================================================
# AdaptiveWeightGroup Tests
# =============================================================================


class TestAdaptiveWeightGroup:
    """Tests for AdaptiveWeightGroup normalized weight management."""

    @pytest.mark.unit
    def test_initialization_normalizes(self):
        """Weights are normalized to sum to 1.0 on init."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveWeightGroup

        group = AdaptiveWeightGroup({"a": 1.0, "b": 1.0, "c": 1.0})
        all_w = group.get_all()
        assert abs(sum(all_w.values()) - 1.0) < 0.001
        assert abs(all_w["a"] - 1 / 3) < 0.001

    @pytest.mark.unit
    def test_get_returns_weight(self):
        """get() returns current weight value."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveWeightGroup

        group = AdaptiveWeightGroup({"a": 0.5, "b": 0.5})
        assert group.get("a") == 0.5  # Already normalized

    @pytest.mark.unit
    def test_get_unknown_returns_zero(self):
        """get() returns 0.0 for unknown weight name."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveWeightGroup

        group = AdaptiveWeightGroup({"a": 1.0})
        assert group.get("nonexistent") == 0.0

    @pytest.mark.unit
    def test_update_from_feedback_renormalizes(self):
        """update_from_feedback maintains sum=1.0 after update."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveWeightGroup

        group = AdaptiveWeightGroup({"a": 0.5, "b": 0.3, "c": 0.2})
        group.update_from_feedback("a", gradient=0.5, reward=0.8)
        all_w = group.get_all()
        assert abs(sum(all_w.values()) - 1.0) < 0.001

    @pytest.mark.unit
    def test_update_from_feedback_unknown_noop(self):
        """update_from_feedback does nothing for unknown weight."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveWeightGroup

        group = AdaptiveWeightGroup({"a": 0.5, "b": 0.5})
        before = group.get_all()
        group.update_from_feedback("nonexistent", gradient=1.0)
        after = group.get_all()
        assert before == after

    @pytest.mark.unit
    def test_to_dict_roundtrip(self):
        """to_dict/from_dict preserves group state."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveWeightGroup

        group = AdaptiveWeightGroup({"a": 0.6, "b": 0.3, "c": 0.1})
        group.update_from_feedback("a", gradient=0.3, reward=0.5)
        d = group.to_dict()
        group2 = AdaptiveWeightGroup.from_dict(d)
        for name in ["a", "b", "c"]:
            assert abs(group.get(name) - group2.get(name)) < 0.001

    @pytest.mark.unit
    def test_repr(self):
        """__repr__ shows weight names and values."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveWeightGroup

        group = AdaptiveWeightGroup({"alpha": 0.5, "beta": 0.5})
        rep = repr(group)
        assert "AdaptiveWeightGroup" in rep
        assert "alpha=" in rep
        assert "beta=" in rep

    @pytest.mark.unit
    def test_zero_weights_handled(self):
        """All-zero weights don't cause division by zero."""
        from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveWeightGroup

        group = AdaptiveWeightGroup({"a": 0.0, "b": 0.0})
        all_w = group.get_all()
        assert abs(sum(all_w.values()) - 1.0) < 0.001


# ===========================================================================
# Model Limits Catalog Tests
# ===========================================================================


@pytest.mark.unit
class TestModelLimitsCatalog:
    """Tests for core/foundation/model_limits_catalog.py."""

    def test_get_model_limits_exact_match(self):
        from Jotty.core.infrastructure.foundation.model_limits_catalog import get_model_limits

        limits = get_model_limits("gpt-4o")
        assert limits["max_prompt"] == 128000
        assert limits["max_output"] == 16384

    def test_get_model_limits_anthropic(self):
        from Jotty.core.infrastructure.foundation.model_limits_catalog import get_model_limits

        limits = get_model_limits("claude-3-opus-20240229")
        assert limits["max_prompt"] == 200000
        assert limits["max_output"] == 4096

    def test_get_model_limits_case_insensitive(self):
        from Jotty.core.infrastructure.foundation.model_limits_catalog import get_model_limits

        limits = get_model_limits("GPT-4O")
        assert limits["max_prompt"] == 128000

    def test_get_model_limits_partial_match(self):
        from Jotty.core.infrastructure.foundation.model_limits_catalog import get_model_limits

        limits = get_model_limits("gpt-4o-something")
        assert "max_prompt" in limits
        assert "max_output" in limits

    def test_get_model_limits_path_extraction(self):
        """Model path like 'provider/model' extracts base model via partial or path match."""
        from Jotty.core.infrastructure.foundation.model_limits_catalog import get_model_limits

        limits = get_model_limits("openai/gpt-4o")
        # Will match via partial match (gpt-4 is a substring), so gets a result
        assert "max_prompt" in limits
        assert "max_output" in limits

    def test_get_model_limits_unknown_fallback(self):
        from Jotty.core.infrastructure.foundation.model_limits_catalog import get_model_limits

        limits = get_model_limits("totally-unknown-model-xyz")
        assert limits["max_prompt"] == 30000
        assert limits["max_output"] == 4096

    def test_get_model_limits_conservative_mode(self):
        from Jotty.core.infrastructure.foundation.model_limits_catalog import get_model_limits

        limits = get_model_limits("gpt-4o", conservative=True)
        assert limits["max_prompt"] == 30000
        assert limits["max_output"] == 16384

    def test_get_model_limits_conservative_small_model(self):
        """Conservative mode doesn't affect models under 30k."""
        from Jotty.core.infrastructure.foundation.model_limits_catalog import get_model_limits

        limits = get_model_limits("gpt-4", conservative=True)
        assert limits["max_prompt"] == 8192

    def test_list_supported_models(self):
        from Jotty.core.infrastructure.foundation.model_limits_catalog import list_supported_models

        models = list_supported_models()
        assert isinstance(models, dict)
        assert len(models) > 50
        assert "gpt-4o" in models

    def test_list_supported_models_returns_copy(self):
        from Jotty.core.infrastructure.foundation.model_limits_catalog import list_supported_models

        m1 = list_supported_models()
        m2 = list_supported_models()
        assert m1 is not m2

    def test_get_models_by_provider_openai(self):
        from Jotty.core.infrastructure.foundation.model_limits_catalog import get_models_by_provider

        openai = get_models_by_provider("openai")
        assert len(openai) > 10
        assert all(
            "gpt" in k or "o1" in k or "text-embedding" in k or "chatgpt" in k or "ft:" in k
            for k in openai.keys()
        )

    def test_get_models_by_provider_anthropic(self):
        from Jotty.core.infrastructure.foundation.model_limits_catalog import get_models_by_provider

        anthropic = get_models_by_provider("anthropic")
        assert len(anthropic) > 5
        assert all("claude" in k for k in anthropic.keys())

    def test_get_models_by_provider_unknown(self):
        from Jotty.core.infrastructure.foundation.model_limits_catalog import get_models_by_provider

        result = get_models_by_provider("nonexistent")
        assert result == {}

    def test_get_max_context_models(self):
        from Jotty.core.infrastructure.foundation.model_limits_catalog import get_max_context_models

        big = get_max_context_models(min_tokens=200000)
        assert len(big) > 0
        # Should be sorted descending by max_prompt
        values = [v["max_prompt"] for v in big.values()]
        assert values == sorted(values, reverse=True)

    def test_get_max_context_models_all_large(self):
        from Jotty.core.infrastructure.foundation.model_limits_catalog import get_max_context_models

        result = get_max_context_models(min_tokens=1000000)
        assert all(v["max_prompt"] >= 1000000 for v in result.values())

    def test_get_model_info_alias(self):
        from Jotty.core.infrastructure.foundation.model_limits_catalog import (
            get_model_info,
            get_model_limits,
        )

        assert get_model_info("gpt-4o") == get_model_limits("gpt-4o")

    def test_gemini_large_context(self):
        from Jotty.core.infrastructure.foundation.model_limits_catalog import get_model_limits

        limits = get_model_limits("gemini-1.5-pro")
        assert limits["max_prompt"] == 2000000

    def test_meta_llama_models(self):
        from Jotty.core.infrastructure.foundation.model_limits_catalog import get_models_by_provider

        meta = get_models_by_provider("meta")
        assert len(meta) > 5


# ===========================================================================
# Robust Parsing Functions Tests
# ===========================================================================


@pytest.mark.unit
class TestParseFloatRobust:
    """Tests for parse_float_robust."""

    def test_none_returns_default(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_float_robust

        assert parse_float_robust(None) is None
        assert parse_float_robust(None, 0.5) == 0.5

    def test_int_input(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_float_robust

        assert parse_float_robust(7) == 7.0

    def test_float_input(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_float_robust

        assert parse_float_robust(0.7) == 0.7

    def test_string_number(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_float_robust

        assert parse_float_robust("0.7") == 0.7

    def test_string_percentage(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_float_robust

        assert abs(parse_float_robust("70%") - 0.7) < 0.001

    def test_string_percentage_word(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_float_robust

        result = parse_float_robust("70 percent")
        assert result == 0.7

    def test_string_approximate(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_float_robust

        result = parse_float_robust("approximately 0.7")
        assert result == 0.7

    def test_empty_string_returns_default(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_float_robust

        assert parse_float_robust("") is None
        assert parse_float_robust("  ") is None

    def test_dict_with_value_key(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_float_robust

        assert parse_float_robust({"value": 0.8}) == 0.8

    def test_dict_with_score_key(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_float_robust

        assert parse_float_robust({"score": 0.9}) == 0.9

    def test_dict_with_confidence_key(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_float_robust

        assert parse_float_robust({"confidence": 0.6}) == 0.6

    def test_dict_without_known_key(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_float_robust

        assert parse_float_robust({"unknown": 1.0}) is None

    def test_json_string(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_float_robust

        assert parse_float_robust('{"value": 0.5}') == 0.5

    def test_string_with_embedded_number(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_float_robust

        result = parse_float_robust("score is 0.85 out of 1.0")
        assert result == 0.85


@pytest.mark.unit
class TestParseBoolRobust:
    """Tests for parse_bool_robust."""

    def test_none_returns_default(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_bool_robust

        assert parse_bool_robust(None) is False
        assert parse_bool_robust(None, True) is True

    def test_bool_passthrough(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_bool_robust

        assert parse_bool_robust(True) is True
        assert parse_bool_robust(False) is False

    def test_int_input(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_bool_robust

        assert parse_bool_robust(1) is True
        assert parse_bool_robust(0) is False
        assert parse_bool_robust(-1) is False

    def test_float_input(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_bool_robust

        assert parse_bool_robust(0.5) is True
        assert parse_bool_robust(0.0) is False

    def test_positive_strings(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_bool_robust

        for s in ["true", "yes", "1", "proceed", "valid", "accept", "approved", "pass"]:
            assert parse_bool_robust(s) is True, f"Failed for '{s}'"

    def test_negative_strings(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_bool_robust

        for s in ["false", "no", "0", "block", "invalid", "reject", "denied", "fail"]:
            assert parse_bool_robust(s) is False, f"Failed for '{s}'"

    def test_case_insensitive(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_bool_robust

        assert parse_bool_robust("TRUE") is True
        assert parse_bool_robust("FALSE") is False

    def test_unknown_string_returns_default(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_bool_robust

        assert parse_bool_robust("maybe") is False
        assert parse_bool_robust("maybe", True) is True


@pytest.mark.unit
class TestParseJsonRobust:
    """Tests for parse_json_robust."""

    def test_none_returns_none(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_json_robust

        assert parse_json_robust(None) is None

    def test_dict_passthrough(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_json_robust

        d = {"key": "val"}
        assert parse_json_robust(d) is d

    def test_json_string(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_json_robust

        result = parse_json_robust('{"name": "test"}')
        assert result == {"name": "test"}

    def test_markdown_code_block(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_json_robust

        text = 'Here is the result:\n```json\n{"status": "ok"}\n```'
        result = parse_json_robust(text)
        assert result == {"status": "ok"}

    def test_markdown_code_block_no_lang(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_json_robust

        text = 'Result:\n```\n{"status": "ok"}\n```'
        result = parse_json_robust(text)
        assert result == {"status": "ok"}

    def test_embedded_json_in_text(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_json_robust

        text = 'The response is {"key": "value"} and some trailing text.'
        result = parse_json_robust(text)
        assert result == {"key": "value"}

    def test_invalid_json_returns_none(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_json_robust

        assert parse_json_robust("not json at all") is None

    def test_nested_braces(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_json_robust

        text = 'result: {"a": {"b": 1}}'
        result = parse_json_robust(text)
        assert result == {"a": {"b": 1}}

    def test_non_dict_json(self):
        """Non-dict JSON like list returns the parsed value."""
        from Jotty.core.infrastructure.foundation.robust_parsing import parse_json_robust

        result = parse_json_robust("[1, 2, 3]")
        assert result == [1, 2, 3]


@pytest.mark.unit
class TestSafeHash:
    """Tests for safe_hash."""

    def test_none_returns_zero(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import safe_hash

        assert safe_hash(None) == 0

    def test_string_input(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import safe_hash

        h = safe_hash("hello")
        assert isinstance(h, int)
        assert h != 0

    def test_int_input(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import safe_hash

        h = safe_hash(42)
        assert isinstance(h, int)

    def test_max_length_truncation(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import safe_hash

        long_text = "a" * 10000
        h1 = safe_hash(long_text)
        h2 = safe_hash(long_text, max_length=10)
        assert h1 != h2

    def test_deterministic(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import safe_hash

        assert safe_hash("test") == safe_hash("test")


@pytest.mark.unit
class TestContentSimilarity:
    """Tests for content_similarity."""

    def test_identical_content(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import content_similarity

        assert content_similarity("hello world", "hello world") is True

    def test_similar_content(self):
        """Similar content passes with appropriate threshold."""
        from Jotty.core.infrastructure.foundation.robust_parsing import content_similarity

        # Jaccard of 3/5 = 0.6, so use threshold=0.5
        assert (
            content_similarity(
                "the quick brown fox",
                "the quick brown dog",
                threshold=0.5,
            )
            is True
        )

    def test_different_content(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import content_similarity

        assert content_similarity("hello world", "goodbye universe") is False

    def test_none_inputs(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import content_similarity

        assert content_similarity(None, None) is True
        assert content_similarity(None, "test") is False
        assert content_similarity("test", None) is False

    def test_empty_strings(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import content_similarity

        assert content_similarity("", "") is True

    def test_case_insensitive(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import content_similarity

        assert content_similarity("Hello World", "hello world") is True

    def test_custom_threshold(self):
        from Jotty.core.infrastructure.foundation.robust_parsing import content_similarity

        # Low threshold = more similar
        assert content_similarity("a b c", "a b d", threshold=0.3) is True
        # High threshold = stricter
        assert content_similarity("a b c", "a d e", threshold=0.9) is False


# ===========================================================================
# Token Counter Tests
# ===========================================================================


@pytest.mark.unit
class TestTokenCounter:
    """Tests for core/foundation/token_counter.py."""

    def test_init_default_model(self):
        from unittest.mock import patch

        from Jotty.core.infrastructure.foundation.token_counter import TokenCounter

        # Ensure dspy.settings.lm is None so TokenCounter uses its default
        with patch("dspy.settings") as mock_settings:
            mock_settings.lm = None
            counter = TokenCounter()
        assert counter.model == "gpt-4.1"

    def test_init_custom_model(self):
        from Jotty.core.infrastructure.foundation.token_counter import TokenCounter

        counter = TokenCounter(model="claude-3-opus")
        assert counter.model == "claude-3-opus"

    def test_map_model_name_direct(self):
        from Jotty.core.infrastructure.foundation.token_counter import TokenCounter

        counter = TokenCounter(model="gpt-4o")
        assert counter.tokencost_model == "gpt-4o"

    def test_map_model_name_claude(self):
        from Jotty.core.infrastructure.foundation.token_counter import TokenCounter

        counter = TokenCounter(model="claude-3-opus")
        assert counter.tokencost_model == "claude-3-opus-20240229"

    def test_map_model_name_unknown(self):
        from Jotty.core.infrastructure.foundation.token_counter import TokenCounter

        counter = TokenCounter(model="totally-unknown")
        assert counter.tokencost_model == "totally-unknown"

    def test_count_tokens_empty(self):
        from Jotty.core.infrastructure.foundation.token_counter import TokenCounter

        counter = TokenCounter(model="gpt-4o")
        assert counter.count_tokens("") == 0

    def test_count_tokens_nonempty(self):
        from Jotty.core.infrastructure.foundation.token_counter import TokenCounter

        counter = TokenCounter(model="gpt-4o")
        count = counter.count_tokens("Hello, world!")
        assert count > 0

    def test_count_messages_empty(self):
        from Jotty.core.infrastructure.foundation.token_counter import TokenCounter

        counter = TokenCounter(model="gpt-4o")
        assert counter.count_messages([]) == 0

    def test_count_messages_nonempty(self):
        from Jotty.core.infrastructure.foundation.token_counter import TokenCounter

        counter = TokenCounter(model="gpt-4o")
        msgs = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        count = counter.count_messages(msgs)
        assert count > 0

    def test_get_model_limits(self):
        from Jotty.core.infrastructure.foundation.token_counter import TokenCounter

        counter = TokenCounter(model="gpt-4o")
        limits = counter.get_model_limits()
        assert "max_prompt" in limits
        assert "max_output" in limits
        assert limits["max_prompt"] == 128000

    def test_will_overflow_false(self):
        from Jotty.core.infrastructure.foundation.token_counter import TokenCounter

        counter = TokenCounter(model="gpt-4o")
        assert counter.will_overflow(1000, 1000) is False

    def test_will_overflow_true(self):
        from Jotty.core.infrastructure.foundation.token_counter import TokenCounter

        counter = TokenCounter(model="gpt-4o")
        # gpt-4o has 128000 max_prompt, 90% = 115200
        assert counter.will_overflow(115000, 1000) is True

    def test_will_overflow_custom_margin(self):
        from Jotty.core.infrastructure.foundation.token_counter import TokenCounter

        counter = TokenCounter(model="gpt-4o")
        # At 50% margin, 128000*0.5=64000
        assert counter.will_overflow(60000, 5000) is False
        assert counter.will_overflow(60000, 5000, safety_margin=0.5) is True

    def test_get_remaining_tokens(self):
        from Jotty.core.infrastructure.foundation.token_counter import TokenCounter

        counter = TokenCounter(model="gpt-4o")
        remaining = counter.get_remaining_tokens(1000)
        # 128000 * 0.9 - 1000 = 114200
        assert remaining == 114200

    def test_get_remaining_tokens_over_limit(self):
        from Jotty.core.infrastructure.foundation.token_counter import TokenCounter

        counter = TokenCounter(model="gpt-4o")
        remaining = counter.get_remaining_tokens(200000)
        assert remaining == 0


@pytest.mark.unit
class TestTokenCounterConvenience:
    """Tests for token_counter module-level convenience functions."""

    def setup_method(self):
        """Reset global counter between tests."""
        import Jotty.core.infrastructure.foundation.token_counter as tc

        tc._default_counter = None

    def test_get_token_counter_creates_instance(self):
        from Jotty.core.infrastructure.foundation.token_counter import get_token_counter

        counter = get_token_counter()
        assert counter is not None

    def test_get_token_counter_caches(self):
        from Jotty.core.infrastructure.foundation.token_counter import get_token_counter

        c1 = get_token_counter("gpt-4o")
        c2 = get_token_counter()
        assert c1 is c2

    def test_get_token_counter_new_model(self):
        from Jotty.core.infrastructure.foundation.token_counter import get_token_counter

        c1 = get_token_counter("gpt-4o")
        c2 = get_token_counter("claude-3-opus")
        assert c1 is not c2
        assert c2.model == "claude-3-opus"

    def test_count_tokens_function(self):
        from Jotty.core.infrastructure.foundation.token_counter import count_tokens

        count = count_tokens("hello world")
        assert count > 0

    def test_count_tokens_accurate_function(self):
        from Jotty.core.infrastructure.foundation.token_counter import count_tokens_accurate

        count = count_tokens_accurate("hello world")
        assert count > 0

    def test_count_tokens_accurate_empty(self):
        from Jotty.core.infrastructure.foundation.token_counter import count_tokens_accurate

        assert count_tokens_accurate("") == 0

    def test_estimate_tokens(self):
        from Jotty.core.infrastructure.foundation.token_counter import estimate_tokens

        text = "a" * 100
        assert estimate_tokens(text) == 26  # 100 // 4 + 1

    def test_estimate_tokens_empty(self):
        from Jotty.core.infrastructure.foundation.token_counter import estimate_tokens

        assert estimate_tokens("") == 1  # 0 // 4 + 1

    def test_get_model_limits_function(self):
        from Jotty.core.infrastructure.foundation.token_counter import get_model_limits

        limits = get_model_limits("gpt-4o")
        assert limits["max_prompt"] == 128000

    def test_will_overflow_function(self):
        from Jotty.core.infrastructure.foundation.token_counter import will_overflow

        assert will_overflow(1000, 1000, "gpt-4o") is False

    def test_count_message_tokens_safe(self):
        from Jotty.core.infrastructure.foundation.token_counter import count_message_tokens_safe

        msgs = [{"role": "user", "content": "hello"}]
        count = count_message_tokens_safe(msgs, "gpt-4o")
        assert count > 0

    def test_get_tokenizer_info(self):
        from Jotty.core.infrastructure.foundation.token_counter import get_tokenizer_info

        info = get_tokenizer_info("gpt-4o")
        assert "available" in info
        assert "model" in info
        assert info["model"] == "gpt-4o"

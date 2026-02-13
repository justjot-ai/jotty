"""
Tests for Foundation Module
============================
Tests for SwarmConfig, AgentConfig (foundation), and Exception hierarchy.
"""
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any


# =============================================================================
# SwarmConfig Tests
# =============================================================================

class TestSwarmConfig:
    """Tests for the SwarmConfig dataclass."""

    @pytest.mark.unit
    def test_default_creation(self):
        """SwarmConfig creates with sensible defaults."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig()
        assert config.schema_version == "2.0"
        assert config.max_actor_iters > 0
        assert config.episodic_capacity > 0
        assert config.gamma > 0
        assert config.enable_validation is True

    @pytest.mark.unit
    def test_memory_budget_property(self):
        """memory_budget = max_context - reserved, with floor."""
        from Jotty.core.foundation.data_structures import SwarmConfig
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
        from Jotty.core.foundation.data_structures import SwarmConfig
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
        from Jotty.core.foundation.data_structures import SwarmConfig
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
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(
            episodic_capacity=0,
            semantic_capacity=0,
            procedural_capacity=0,
            meta_capacity=0,
            causal_capacity=0,
        )
        assert config.total_memory_capacity == 0

    @pytest.mark.unit
    def test_post_init_char_limits(self):
        """__post_init__ calculates char limits from token budgets."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(
            preview_token_budget=1000,
            max_description_tokens=500,
        )
        assert config.preview_char_limit == 4000  # 1000 * 4
        assert config.max_description_chars == 2000  # 500 * 4

    @pytest.mark.unit
    def test_custom_field_values_preserved(self):
        """Explicit field values are not overridden."""
        from Jotty.core.foundation.data_structures import SwarmConfig
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
        from Jotty.core.foundation.data_structures import SwarmConfig
        for mode in ['quick', 'standard', 'thorough']:
            config = SwarmConfig(validation_mode=mode)
            assert config.validation_mode == mode

    @pytest.mark.unit
    def test_budget_controls(self):
        """Budget control fields have sensible defaults."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig()
        assert config.max_llm_calls_per_episode >= 0
        assert config.max_total_tokens_per_episode >= 0

    @pytest.mark.unit
    def test_to_flat_dict(self):
        """to_flat_dict() returns all fields as a flat dictionary."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(max_actor_iters=42, enable_rl=False)
        d = config.to_flat_dict()
        assert isinstance(d, dict)
        assert d['max_actor_iters'] == 42
        assert d['enable_rl'] is False
        assert 'schema_version' in d

    @pytest.mark.unit
    def test_to_flat_dict_roundtrip(self):
        """to_flat_dict() output can recreate SwarmConfig."""
        from Jotty.core.foundation.data_structures import SwarmConfig
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
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(max_actor_iters=99, actor_timeout=300.0)
        assert config.execution.max_actor_iters == 99
        assert config.execution.actor_timeout == 300.0

    @pytest.mark.unit
    def test_execution_view_write_through(self):
        """ExecutionView writes propagate to parent."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(max_actor_iters=10)
        config.execution.max_actor_iters = 50
        assert config.max_actor_iters == 50

    @pytest.mark.unit
    def test_memory_view_fields(self):
        """MemoryView exposes memory-related fields."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(episodic_capacity=200, enable_llm_rag=False)
        assert config.memory_settings.episodic_capacity == 200
        assert config.memory_settings.enable_llm_rag is False

    @pytest.mark.unit
    def test_learning_view_fields(self):
        """LearningView exposes RL and exploration fields."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(gamma=0.8, epsilon_start=0.5, enable_rl=False)
        assert config.learning.gamma == 0.8
        assert config.learning.epsilon_start == 0.5
        assert config.learning.enable_rl is False

    @pytest.mark.unit
    def test_validation_view_fields(self):
        """ValidationView exposes validation settings."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(enable_validation=False, max_validation_rounds=5)
        assert config.validation_settings.enable_validation is False
        assert config.validation_settings.max_validation_rounds == 5

    @pytest.mark.unit
    def test_monitoring_view_fields(self):
        """MonitoringView exposes logging and budget fields."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(log_level="DEBUG", enable_metrics=False)
        assert config.monitoring.log_level == "DEBUG"
        assert config.monitoring.enable_metrics is False

    @pytest.mark.unit
    def test_intelligence_view_fields(self):
        """SwarmIntelligenceView exposes trust and agent comm fields."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(trust_min=0.2, enable_agent_communication=False)
        assert config.intelligence.trust_min == 0.2
        assert config.intelligence.enable_agent_communication is False

    @pytest.mark.unit
    def test_persistence_view_fields(self):
        """PersistenceView exposes storage fields."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(output_base_dir="/tmp/test", storage_format="sqlite")
        assert config.persistence.output_base_dir == "/tmp/test"
        assert config.persistence.storage_format == "sqlite"

    @pytest.mark.unit
    def test_context_budget_view_fields(self):
        """ContextBudgetView exposes token budget fields."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(max_context_tokens=50000, min_memory_budget=2000)
        assert config.context_budget.max_context_tokens == 50000
        assert config.context_budget.min_memory_budget == 2000

    @pytest.mark.unit
    def test_view_to_dict(self):
        """View.to_dict() returns only that view's fields."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(max_actor_iters=42, gamma=0.5)
        exec_dict = config.execution.to_dict()
        assert 'max_actor_iters' in exec_dict
        assert exec_dict['max_actor_iters'] == 42
        # gamma is a LearningView field, not ExecutionView
        assert 'gamma' not in exec_dict

    @pytest.mark.unit
    def test_view_attribute_error(self):
        """View raises AttributeError for unknown fields."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig()
        with pytest.raises(AttributeError):
            _ = config.execution.nonexistent_field

    @pytest.mark.unit
    def test_view_repr(self):
        """View __repr__ includes field values."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig(max_actor_iters=42)
        r = repr(config.execution)
        assert 'ExecutionView' in r
        assert 'max_actor_iters=42' in r


# =============================================================================
# Foundation AgentConfig Tests
# =============================================================================

class TestFoundationAgentConfig:
    """Tests for the foundation AgentConfig (orchestration-level spec)."""

    @pytest.mark.unit
    def test_creation_with_required_fields(self):
        """AgentConfig requires name and agent."""
        from Jotty.core.foundation.agent_config import AgentConfig
        mock_agent = MagicMock()
        config = AgentConfig(name="TestAgent", agent=mock_agent)
        assert config.name == "TestAgent"
        assert config.agent is mock_agent

    @pytest.mark.unit
    def test_prompt_list_initialization(self):
        """None prompts initialized to empty lists."""
        from Jotty.core.foundation.agent_config import AgentConfig
        config = AgentConfig(name="Test", agent=MagicMock())
        assert config.architect_prompts == [] or config.architect_prompts is not None
        assert config.auditor_prompts == [] or config.auditor_prompts is not None

    @pytest.mark.unit
    def test_max_retries_sentinel_resolution(self):
        """max_retries=0 resolves from config_defaults."""
        from Jotty.core.foundation.agent_config import AgentConfig
        config = AgentConfig(name="Test", agent=MagicMock(), max_retries=0)
        # Should be resolved to a positive value from DEFAULTS
        assert config.max_retries > 0

    @pytest.mark.unit
    def test_max_retries_explicit_preserved(self):
        """Explicit max_retries value is preserved."""
        from Jotty.core.foundation.agent_config import AgentConfig
        config = AgentConfig(name="Test", agent=MagicMock(), max_retries=5)
        assert config.max_retries == 5

    @pytest.mark.unit
    def test_default_flags(self):
        """Default boolean flags are correct."""
        from Jotty.core.foundation.agent_config import AgentConfig
        config = AgentConfig(name="Test", agent=MagicMock())
        assert config.enable_architect is True
        assert config.enable_auditor is True
        assert config.enabled is True
        assert config.is_critical is False
        assert config.is_executor is False

    @pytest.mark.unit
    def test_validation_mode_default(self):
        """Default validation mode is 'standard'."""
        from Jotty.core.foundation.agent_config import AgentConfig
        config = AgentConfig(name="Test", agent=MagicMock())
        assert config.validation_mode == "standard"

    @pytest.mark.unit
    def test_dependencies_and_capabilities(self):
        """Dependencies and capabilities stored correctly."""
        from Jotty.core.foundation.agent_config import AgentConfig
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
        from Jotty.core.foundation.agent_config import AgentConfig
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
        from Jotty.core.agents.base.base_agent import AgentRuntimeConfig
        config = AgentRuntimeConfig()
        assert config.model != ""
        assert config.temperature > 0
        assert config.max_tokens > 0
        assert config.max_retries > 0

    @pytest.mark.unit
    def test_explicit_values_preserved(self):
        """Explicit non-zero values are preserved."""
        from Jotty.core.agents.base.base_agent import AgentRuntimeConfig
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
        from Jotty.core.agents.base.base_agent import AgentRuntimeConfig
        config = AgentRuntimeConfig()
        assert config.enable_memory is True
        assert config.enable_context is True
        assert config.enable_monitoring is True
        assert config.enable_skills is True

    @pytest.mark.unit
    def test_enable_flags_override(self):
        """Enable flags can be disabled."""
        from Jotty.core.agents.base.base_agent import AgentRuntimeConfig
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
        from Jotty.core.foundation.exceptions import JottyError
        err = JottyError("Something failed")
        assert err.message == "Something failed"
        assert "Something failed" in str(err)

    @pytest.mark.unit
    def test_jotty_error_with_context(self):
        """JottyError stores context dict."""
        from Jotty.core.foundation.exceptions import JottyError
        err = JottyError("Failed", context={"agent": "TestAgent", "step": 3})
        assert err.context == {"agent": "TestAgent", "step": 3}

    @pytest.mark.unit
    def test_jotty_error_with_original(self):
        """JottyError chains original exception."""
        from Jotty.core.foundation.exceptions import JottyError
        original = ValueError("bad value")
        err = JottyError("Wrapped error", original_error=original)
        assert err.original_error is original

    @pytest.mark.unit
    def test_exception_inheritance_chain(self):
        """All exceptions inherit from JottyError."""
        from Jotty.core.foundation.exceptions import (
            JottyError, ConfigurationError, ExecutionError,
            AgentExecutionError, TimeoutError, ValidationError,
            MemoryRetrievalError, LearningError,
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
        from Jotty.core.foundation.exceptions import (
            JottyError, AgentExecutionError, TimeoutError,
        )
        with pytest.raises(JottyError):
            raise AgentExecutionError("agent failed")

        with pytest.raises(JottyError):
            raise TimeoutError("timed out")

    @pytest.mark.unit
    def test_validation_error_to_dict(self):
        """ValidationError.to_dict() returns structured response."""
        from Jotty.core.foundation.exceptions import ValidationError
        err = ValidationError("Invalid param", param="temperature", value=2.5)
        result = err.to_dict()
        assert result == {
            'success': False,
            'error': 'Invalid param',
            'param': 'temperature',
        }

    @pytest.mark.unit
    def test_validation_error_param_and_value(self):
        """ValidationError stores param and value attributes."""
        from Jotty.core.foundation.exceptions import ValidationError
        err = ValidationError("bad", param="x", value=42)
        assert err.param == "x"
        assert err.value == 42

    @pytest.mark.unit
    def test_validation_error_defaults(self):
        """ValidationError defaults param/value to None."""
        from Jotty.core.foundation.exceptions import ValidationError
        err = ValidationError("bad")
        assert err.param is None
        assert err.value is None
        result = err.to_dict()
        assert result['param'] is None

    @pytest.mark.unit
    def test_context_overflow_error(self):
        """ContextOverflowError stores token info."""
        from Jotty.core.foundation.exceptions import ContextOverflowError
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
        from Jotty.core.foundation.exceptions import (
            wrap_exception, AgentExecutionError,
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
        from Jotty.core.foundation.exceptions import ValidationError as V1
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
        from Jotty.core.foundation.direct_anthropic_lm import get_cost_tracker, reset_cost_tracker
        tracker = get_cost_tracker()
        assert tracker is not None
        reset_cost_tracker()
        # After reset, get_cost_tracker returns a NEW instance
        tracker2 = get_cost_tracker()
        assert tracker2 is not tracker

    @pytest.mark.unit
    def test_jotty_integration_reset(self):
        """JottyIntegration.reset_instance() clears the singleton."""
        from Jotty.core.integration.integration import JottyIntegration
        inst = JottyIntegration.get_instance()
        assert inst is not None
        JottyIntegration.reset_instance()
        assert JottyIntegration._instance is None

    @pytest.mark.unit
    def test_event_broadcaster_reset(self):
        """AgentEventBroadcaster.reset_instance() clears the singleton."""
        from Jotty.core.utils.async_utils import AgentEventBroadcaster
        broadcaster = AgentEventBroadcaster.get_instance()
        assert broadcaster is not None
        AgentEventBroadcaster.reset_instance()
        assert AgentEventBroadcaster._instance is None

    @pytest.mark.unit
    def test_budget_tracker_reset(self):
        """BudgetTracker.reset_instances() clears all named instances."""
        from Jotty.core.utils.budget_tracker import BudgetTracker
        BudgetTracker.get_instance("test_a")
        BudgetTracker.get_instance("test_b")
        assert len(BudgetTracker._instances) >= 2
        BudgetTracker.reset_instances()
        assert len(BudgetTracker._instances) == 0

    @pytest.mark.unit
    def test_llm_cache_reset(self):
        """LLMCallCache.reset_instances() clears all named instances."""
        from Jotty.core.utils.llm_cache import LLMCallCache
        LLMCallCache.get_instance("test_cache")
        assert "test_cache" in LLMCallCache._instances
        LLMCallCache.reset_instances()
        assert len(LLMCallCache._instances) == 0

    @pytest.mark.unit
    def test_prompt_selector_reset(self):
        """reset_prompt_selector() clears the singleton."""
        from Jotty.core.utils.prompt_selector import get_prompt_selector, reset_prompt_selector
        sel = get_prompt_selector()
        assert sel is not None
        reset_prompt_selector()
        sel2 = get_prompt_selector()
        assert sel2 is not sel

    @pytest.mark.unit
    def test_tokenizer_reset(self):
        """SmartTokenizer.reset_instances() clears all cached encodings."""
        from Jotty.core.utils.tokenizer import SmartTokenizer
        SmartTokenizer.get_instance()
        assert len(SmartTokenizer._instances) >= 1
        SmartTokenizer.reset_instances()
        assert len(SmartTokenizer._instances) == 0

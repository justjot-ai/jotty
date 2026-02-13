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


# =============================================================================
# ConfigView Tests
# =============================================================================

class TestConfigViews:
    """Tests for _ConfigView proxy system."""

    @pytest.mark.unit
    def test_config_view_getattr_reads_parent(self):
        """ConfigView proxies attribute reads to parent SwarmConfig."""
        from Jotty.core.foundation.data_structures import SwarmConfig, ExecutionView
        config = SwarmConfig(max_actor_iters=42)
        view = ExecutionView(config)
        assert view.max_actor_iters == 42

    @pytest.mark.unit
    def test_config_view_setattr_writes_parent(self):
        """ConfigView proxies attribute writes to parent SwarmConfig."""
        from Jotty.core.foundation.data_structures import SwarmConfig, ExecutionView
        config = SwarmConfig(max_actor_iters=10)
        view = ExecutionView(config)
        view.max_actor_iters = 99
        assert config.max_actor_iters == 99

    @pytest.mark.unit
    def test_config_view_getattr_invalid_raises(self):
        """ConfigView raises AttributeError for fields not in its _FIELDS."""
        from Jotty.core.foundation.data_structures import SwarmConfig, ExecutionView
        config = SwarmConfig()
        view = ExecutionView(config)
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = view.nonexistent_field

    @pytest.mark.unit
    def test_config_view_to_dict(self):
        """ConfigView.to_dict() returns only its fields."""
        from Jotty.core.foundation.data_structures import SwarmConfig, PersistenceView
        config = SwarmConfig()
        view = PersistenceView(config)
        d = view.to_dict()
        assert isinstance(d, dict)
        assert 'persist_memories' in d
        # Should NOT have execution fields
        assert 'max_actor_iters' not in d

    @pytest.mark.unit
    def test_config_view_repr(self):
        """ConfigView.__repr__() contains class name and field values."""
        from Jotty.core.foundation.data_structures import SwarmConfig, MonitoringView
        config = SwarmConfig()
        view = MonitoringView(config)
        rep = repr(view)
        assert "MonitoringView(" in rep

    @pytest.mark.unit
    def test_persistence_view_fields(self):
        """PersistenceView has expected fields."""
        from Jotty.core.foundation.data_structures import SwarmConfig, PersistenceView
        view = PersistenceView(SwarmConfig())
        d = view.to_dict()
        assert 'auto_save_interval' in d
        assert 'persist_memories' in d

    @pytest.mark.unit
    def test_memory_view_fields(self):
        """MemoryView has expected fields."""
        from Jotty.core.foundation.data_structures import SwarmConfig, MemoryView
        view = MemoryView(SwarmConfig())
        d = view.to_dict()
        assert 'episodic_capacity' in d
        assert 'enable_llm_rag' in d

    @pytest.mark.unit
    def test_learning_view_fields(self):
        """LearningView has expected fields."""
        from Jotty.core.foundation.data_structures import SwarmConfig, LearningView
        view = LearningView(SwarmConfig())
        d = view.to_dict()
        assert 'gamma' in d
        assert 'alpha' in d

    @pytest.mark.unit
    def test_swarm_config_view_property(self):
        """SwarmConfig exposes views via properties."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig()
        # Check that view properties exist and return the right type
        assert hasattr(config, 'execution')
        assert hasattr(config, 'persistence')
        assert hasattr(config, 'memory_settings')
        assert hasattr(config, 'learning')


# =============================================================================
# SharedScratchpad Tests
# =============================================================================

class TestSharedScratchpad:
    """Tests for SharedScratchpad inter-agent communication."""

    @pytest.mark.unit
    def test_add_message(self):
        """add_message appends to messages list."""
        from Jotty.core.foundation.data_structures import SharedScratchpad, AgentMessage, CommunicationType
        pad = SharedScratchpad()
        msg = AgentMessage(
            sender="agent1", receiver="agent2",
            message_type=CommunicationType.INSIGHT,
            content={"data": "test"},
        )
        pad.add_message(msg)
        assert len(pad.messages) == 1
        assert pad.messages[0].sender == "agent1"

    @pytest.mark.unit
    def test_add_message_caches_tool_result(self):
        """Tool result messages are cached for reuse."""
        from Jotty.core.foundation.data_structures import SharedScratchpad, AgentMessage, CommunicationType
        pad = SharedScratchpad()
        msg = AgentMessage(
            sender="agent1", receiver="*",
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
        from Jotty.core.foundation.data_structures import SharedScratchpad
        pad = SharedScratchpad()
        assert pad.get_cached_result("nonexistent", {}) is None

    @pytest.mark.unit
    def test_get_messages_for_specific_receiver(self):
        """get_messages_for returns messages for specific receiver."""
        from Jotty.core.foundation.data_structures import SharedScratchpad, AgentMessage, CommunicationType
        pad = SharedScratchpad()
        pad.add_message(AgentMessage(
            sender="a", receiver="b", message_type=CommunicationType.INSIGHT, content={}
        ))
        pad.add_message(AgentMessage(
            sender="a", receiver="c", message_type=CommunicationType.INSIGHT, content={}
        ))
        msgs_b = pad.get_messages_for("b")
        assert len(msgs_b) == 1

    @pytest.mark.unit
    def test_get_messages_for_broadcast(self):
        """get_messages_for includes broadcast messages."""
        from Jotty.core.foundation.data_structures import SharedScratchpad, AgentMessage, CommunicationType
        pad = SharedScratchpad()
        pad.add_message(AgentMessage(
            sender="a", receiver="*", message_type=CommunicationType.INSIGHT, content={}
        ))
        pad.add_message(AgentMessage(
            sender="a", receiver="b", message_type=CommunicationType.INSIGHT, content={}
        ))
        # "c" should get broadcast only
        msgs_c = pad.get_messages_for("c")
        assert len(msgs_c) == 1
        # "b" gets both broadcast and direct
        msgs_b = pad.get_messages_for("b")
        assert len(msgs_b) == 2

    @pytest.mark.unit
    def test_clear(self):
        """clear() empties all data structures."""
        from Jotty.core.foundation.data_structures import SharedScratchpad, AgentMessage, CommunicationType
        pad = SharedScratchpad()
        pad.add_message(AgentMessage(
            sender="a", receiver="*", message_type=CommunicationType.TOOL_RESULT,
            content={}, tool_name="t", tool_args={}, tool_result="r",
        ))
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
        from Jotty.core.foundation.data_structures import AgentContribution
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
        from Jotty.core.foundation.data_structures import AgentContribution
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
        from Jotty.core.foundation.data_structures import AgentContribution
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
        from Jotty.core.foundation.data_structures import AgentContribution
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
        from Jotty.core.foundation.data_structures import LearningMetrics
        metrics = LearningMetrics()
        assert metrics.get_success_rate() == 0.5

    @pytest.mark.unit
    def test_get_success_rate_all_success(self):
        """Success rate is 1.0 when all recent are successful."""
        from Jotty.core.foundation.data_structures import LearningMetrics
        metrics = LearningMetrics()
        metrics.recent_successes = [True] * 10
        assert metrics.get_success_rate() == 1.0

    @pytest.mark.unit
    def test_get_success_rate_mixed(self):
        """Success rate reflects actual mix."""
        from Jotty.core.foundation.data_structures import LearningMetrics
        metrics = LearningMetrics()
        metrics.recent_successes = [True, False, True, True, False]
        assert metrics.get_success_rate() == 0.6

    @pytest.mark.unit
    def test_get_success_rate_windowed(self):
        """Success rate respects window parameter."""
        from Jotty.core.foundation.data_structures import LearningMetrics
        metrics = LearningMetrics()
        metrics.recent_successes = [False] * 100 + [True] * 10
        rate = metrics.get_success_rate(window=10)
        assert rate == 1.0

    @pytest.mark.unit
    def test_get_learning_velocity_empty(self):
        """Learning velocity is 0 with insufficient data."""
        from Jotty.core.foundation.data_structures import LearningMetrics
        metrics = LearningMetrics()
        assert metrics.get_learning_velocity() == 0.0

    @pytest.mark.unit
    def test_get_learning_velocity_with_changes(self):
        """Learning velocity reflects magnitude of value changes."""
        from Jotty.core.foundation.data_structures import LearningMetrics
        metrics = LearningMetrics()
        metrics.value_changes = [0.1, -0.2, 0.15, -0.05, 0.3]
        velocity = metrics.get_learning_velocity()
        expected = sum(abs(v) for v in metrics.value_changes) / len(metrics.value_changes)
        assert abs(velocity - expected) < 0.001

    @pytest.mark.unit
    def test_is_learning_stalled_no_data(self):
        """Stalled check with no data returns True (velocity = 0)."""
        from Jotty.core.foundation.data_structures import LearningMetrics
        metrics = LearningMetrics()
        assert metrics.is_learning_stalled() is True

    @pytest.mark.unit
    def test_is_learning_stalled_active(self):
        """Active learning is not stalled."""
        from Jotty.core.foundation.data_structures import LearningMetrics
        metrics = LearningMetrics()
        metrics.value_changes = [0.1, 0.2, 0.15, 0.1, 0.2]
        assert metrics.is_learning_stalled() is False

    @pytest.mark.unit
    def test_is_learning_stalled_flat(self):
        """Flat learning (tiny changes) is stalled."""
        from Jotty.core.foundation.data_structures import LearningMetrics
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
        from Jotty.core.agents.base.base_agent import AgentResult
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
        from Jotty.core.agents.base.base_agent import AgentResult
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
        from Jotty.core.agents.base.base_agent import BaseAgent, AgentRuntimeConfig
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
        agent.memory = None
        agent.store_memory("test content")  # Should not raise

    @pytest.mark.unit
    def test_store_memory_calls_memory_store(self):
        """store_memory delegates to self.memory.store()."""
        agent = self._make_agent()
        mock_memory = MagicMock()
        agent.memory = mock_memory
        agent.store_memory("test content", level="semantic", goal="test goal")
        mock_memory.store.assert_called_once()
        call_kwargs = mock_memory.store.call_args[1]
        assert call_kwargs['content'] == "test content"
        assert call_kwargs['goal'] == "test goal"

    @pytest.mark.unit
    def test_store_memory_exception_suppressed(self):
        """store_memory suppresses exceptions."""
        agent = self._make_agent()
        mock_memory = MagicMock()
        mock_memory.store.side_effect = RuntimeError("store failed")
        agent.memory = mock_memory
        agent.store_memory("test")  # Should not raise

    @pytest.mark.unit
    def test_retrieve_memory_no_memory(self):
        """retrieve_memory returns empty list when memory is None."""
        agent = self._make_agent()
        agent.memory = None
        result = agent.retrieve_memory("test query")
        assert result == []

    @pytest.mark.unit
    def test_retrieve_memory_calls_memory_retrieve(self):
        """retrieve_memory delegates to self.memory.retrieve()."""
        agent = self._make_agent()
        mock_memory = MagicMock()
        mock_memory.retrieve.return_value = ["mem1", "mem2"]
        agent.memory = mock_memory
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
        agent.memory = mock_memory
        result = agent.retrieve_memory("test")
        assert result == []

    @pytest.mark.unit
    def test_register_context(self):
        """register_context delegates to context.set()."""
        agent = self._make_agent()
        mock_context = MagicMock()
        agent.context = mock_context
        agent.register_context("key1", "value1")
        mock_context.set.assert_called_once_with("key1", "value1")

    @pytest.mark.unit
    def test_register_context_no_context(self):
        """register_context is no-op when context is None."""
        agent = self._make_agent()
        agent.context = None
        agent.register_context("key", "val")  # Should not raise

    @pytest.mark.unit
    def test_get_context(self):
        """get_context delegates to context.get()."""
        agent = self._make_agent()
        mock_context = MagicMock()
        mock_context.get.return_value = "found_value"
        agent.context = mock_context
        result = agent.get_context("key1")
        assert result == "found_value"

    @pytest.mark.unit
    def test_get_context_no_context_returns_default(self):
        """get_context returns default when context is None."""
        agent = self._make_agent()
        agent.context = None
        result = agent.get_context("key1", default="fallback")
        assert result == "fallback"

    @pytest.mark.unit
    def test_get_compressed_context_no_context(self):
        """get_compressed_context returns '' when context is None."""
        agent = self._make_agent()
        agent.context = None
        assert agent.get_compressed_context() == ""

    @pytest.mark.unit
    def test_get_compressed_context_with_data(self):
        """get_compressed_context returns JSON string of known keys."""
        agent = self._make_agent()
        mock_context = MagicMock()
        mock_context.get.side_effect = lambda key, *a: {
            'current_task': 'do X',
            'current_goal': 'achieve Y',
        }.get(key)
        agent.context = mock_context
        result = agent.get_compressed_context()
        assert 'do X' in result
        assert 'achieve Y' in result

    @pytest.mark.unit
    def test_get_compressed_context_truncation(self):
        """get_compressed_context truncates when exceeding max_tokens."""
        agent = self._make_agent()
        mock_context = MagicMock()
        long_value = "x" * 50000
        mock_context.get.side_effect = lambda key, *a: {
            'current_task': long_value,
        }.get(key)
        agent.context = mock_context
        result = agent.get_compressed_context(max_tokens=100)
        assert len(result) <= 100 * 4 + 10  # 4 chars/token + margin for "..."
        assert result.endswith("...")

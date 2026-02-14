"""
Tests for Budget Tracker
========================
Comprehensive tests for BudgetScope, BudgetExceededError, BudgetUsage,
BudgetConfig, BudgetTracker, and the get_budget_tracker convenience function.

Covers: episode lifecycle, budget limits, soft limit mode, cost estimation,
warning callbacks, multi-agent tracking, singleton pattern, thread safety,
edge cases, serialization, and reset behaviour.
"""

import time
import threading
import pytest
from unittest.mock import Mock, patch, MagicMock

from Jotty.core.utils.budget_tracker import (
    BudgetScope,
    BudgetExceededError,
    BudgetUsage,
    BudgetConfig,
    BudgetTracker,
    get_budget_tracker,
    DEFAULT_COST_PER_1K_INPUT,
    DEFAULT_COST_PER_1K_OUTPUT,
)


# =============================================================================
# BudgetScope Enum Tests
# =============================================================================

@pytest.mark.unit
class TestBudgetScope:
    """Tests for the BudgetScope enum."""

    def test_global_value(self):
        """GLOBAL scope has value 'global'."""
        assert BudgetScope.GLOBAL.value == "global"

    def test_episode_value(self):
        """EPISODE scope has value 'episode'."""
        assert BudgetScope.EPISODE.value == "episode"

    def test_agent_value(self):
        """AGENT scope has value 'agent'."""
        assert BudgetScope.AGENT.value == "agent"

    def test_operation_value(self):
        """OPERATION scope has value 'operation'."""
        assert BudgetScope.OPERATION.value == "operation"

    def test_all_members(self):
        """BudgetScope has exactly four members."""
        members = list(BudgetScope)
        assert len(members) == 4

    def test_enum_identity(self):
        """Enum members are singletons."""
        assert BudgetScope.GLOBAL is BudgetScope.GLOBAL
        assert BudgetScope.EPISODE is not BudgetScope.AGENT


# =============================================================================
# BudgetExceededError Tests
# =============================================================================

@pytest.mark.unit
class TestBudgetExceededError:
    """Tests for the BudgetExceededError exception."""

    def test_basic_construction(self):
        """Error stores message, scope, current, limit, resource."""
        err = BudgetExceededError(
            message="over budget",
            scope=BudgetScope.EPISODE,
            current=101,
            limit=100,
            resource="calls",
        )
        assert str(err) == "over budget"
        assert err.scope == BudgetScope.EPISODE
        assert err.current == 101
        assert err.limit == 100
        assert err.resource == "calls"

    def test_default_resource(self):
        """Default resource is 'calls'."""
        err = BudgetExceededError(
            message="exceeded",
            scope=BudgetScope.GLOBAL,
            current=10001,
            limit=10000,
        )
        assert err.resource == "calls"

    def test_custom_resource(self):
        """Custom resource value is preserved."""
        err = BudgetExceededError(
            message="exceeded",
            scope=BudgetScope.AGENT,
            current=60000,
            limit=50000,
            resource="tokens",
        )
        assert err.resource == "tokens"

    def test_is_exception(self):
        """BudgetExceededError is a subclass of Exception."""
        assert issubclass(BudgetExceededError, Exception)

    def test_can_be_raised_and_caught(self):
        """Error can be raised and caught with attributes intact."""
        with pytest.raises(BudgetExceededError) as exc_info:
            raise BudgetExceededError(
                message="too many calls",
                scope=BudgetScope.EPISODE,
                current=150,
                limit=100,
                resource="calls",
            )
        assert exc_info.value.current == 150
        assert exc_info.value.limit == 100


# =============================================================================
# BudgetUsage Tests
# =============================================================================

@pytest.mark.unit
class TestBudgetUsage:
    """Tests for the BudgetUsage dataclass."""

    def test_default_values(self):
        """BudgetUsage defaults are all zero."""
        usage = BudgetUsage()
        assert usage.calls == 0
        assert usage.tokens_input == 0
        assert usage.tokens_output == 0
        assert usage.estimated_cost_usd == 0.0
        assert usage.last_call_time == 0.0
        assert usage.warnings_emitted == 0

    def test_total_tokens_property(self):
        """total_tokens returns sum of input and output tokens."""
        usage = BudgetUsage(tokens_input=300, tokens_output=200)
        assert usage.total_tokens == 500

    def test_total_tokens_with_zero(self):
        """total_tokens returns 0 when no tokens recorded."""
        usage = BudgetUsage()
        assert usage.total_tokens == 0

    def test_to_dict_contains_all_fields(self):
        """to_dict includes all fields plus total_tokens."""
        usage = BudgetUsage(
            calls=5,
            tokens_input=1000,
            tokens_output=500,
            estimated_cost_usd=0.025,
            last_call_time=1700000000.0,
            warnings_emitted=2,
        )
        d = usage.to_dict()
        assert d['calls'] == 5
        assert d['tokens_input'] == 1000
        assert d['tokens_output'] == 500
        assert d['total_tokens'] == 1500
        assert d['estimated_cost_usd'] == 0.025
        assert d['last_call_time'] == 1700000000.0
        assert d['warnings_emitted'] == 2

    def test_to_dict_keys(self):
        """to_dict has exactly the expected keys."""
        expected_keys = {
            'calls', 'tokens_input', 'tokens_output', 'total_tokens',
            'estimated_cost_usd', 'last_call_time', 'warnings_emitted',
        }
        assert set(BudgetUsage().to_dict().keys()) == expected_keys

    def test_mutation(self):
        """BudgetUsage fields can be mutated after creation."""
        usage = BudgetUsage()
        usage.calls += 1
        usage.tokens_input += 100
        usage.tokens_output += 50
        assert usage.calls == 1
        assert usage.total_tokens == 150


# =============================================================================
# BudgetConfig Tests
# =============================================================================

@pytest.mark.unit
class TestBudgetConfig:
    """Tests for the BudgetConfig dataclass."""

    def test_default_call_limits(self):
        """Default call limits are set correctly."""
        config = BudgetConfig()
        assert config.max_llm_calls_per_episode == 100
        assert config.max_llm_calls_per_agent == 50
        assert config.max_llm_calls_global == 10000

    def test_default_token_limits(self):
        """Default token limits are set correctly."""
        config = BudgetConfig()
        assert config.max_total_tokens_per_episode == 500000
        assert config.max_tokens_per_agent == 100000
        assert config.max_tokens_per_call == 50000

    def test_default_cost_limits(self):
        """Cost limits default to None."""
        config = BudgetConfig()
        assert config.max_cost_per_episode is None
        assert config.max_cost_global is None

    def test_default_warning_threshold(self):
        """Default warning threshold is 0.8 (80%)."""
        config = BudgetConfig()
        assert config.warning_threshold == 0.8

    def test_default_enforcement(self):
        """Enforcement is enabled by default, soft_limit_mode is off."""
        config = BudgetConfig()
        assert config.enable_enforcement is True
        assert config.soft_limit_mode is False

    def test_custom_config(self):
        """Custom values override defaults."""
        config = BudgetConfig(
            max_llm_calls_per_episode=10,
            max_llm_calls_per_agent=5,
            max_cost_per_episode=1.0,
            warning_threshold=0.9,
            soft_limit_mode=True,
        )
        assert config.max_llm_calls_per_episode == 10
        assert config.max_llm_calls_per_agent == 5
        assert config.max_cost_per_episode == 1.0
        assert config.warning_threshold == 0.9
        assert config.soft_limit_mode is True

    def test_to_dict_contains_all_fields(self):
        """to_dict includes all configuration fields."""
        config = BudgetConfig()
        d = config.to_dict()
        expected_keys = {
            'max_llm_calls_per_episode', 'max_llm_calls_per_agent',
            'max_llm_calls_global', 'max_total_tokens_per_episode',
            'max_tokens_per_agent', 'max_tokens_per_call',
            'max_cost_per_episode', 'max_cost_global',
            'warning_threshold', 'enable_enforcement', 'soft_limit_mode',
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match(self):
        """to_dict returns values matching the config."""
        config = BudgetConfig(max_llm_calls_per_episode=42)
        d = config.to_dict()
        assert d['max_llm_calls_per_episode'] == 42


# =============================================================================
# Constants Tests
# =============================================================================

@pytest.mark.unit
class TestConstants:
    """Tests for module-level constants."""

    def test_default_cost_per_1k_input(self):
        """DEFAULT_COST_PER_1K_INPUT is 0.01."""
        assert DEFAULT_COST_PER_1K_INPUT == 0.01

    def test_default_cost_per_1k_output(self):
        """DEFAULT_COST_PER_1K_OUTPUT is 0.03."""
        assert DEFAULT_COST_PER_1K_OUTPUT == 0.03


# =============================================================================
# BudgetTracker Initialization Tests
# =============================================================================

@pytest.mark.unit
class TestBudgetTrackerInit:
    """Tests for BudgetTracker initialization."""

    def setup_method(self):
        """Reset singleton instances before each test."""
        BudgetTracker.reset_instances()

    def test_default_init(self):
        """BudgetTracker initializes with default config and cost rates."""
        tracker = BudgetTracker()
        assert isinstance(tracker.config, BudgetConfig)
        assert tracker.cost_per_1k_input == DEFAULT_COST_PER_1K_INPUT
        assert tracker.cost_per_1k_output == DEFAULT_COST_PER_1K_OUTPUT

    def test_custom_config(self):
        """BudgetTracker accepts a custom BudgetConfig."""
        config = BudgetConfig(max_llm_calls_per_episode=20)
        tracker = BudgetTracker(config=config)
        assert tracker.config.max_llm_calls_per_episode == 20

    def test_custom_cost_rates(self):
        """Custom cost rates are stored."""
        tracker = BudgetTracker(cost_per_1k_input=0.05, cost_per_1k_output=0.10)
        assert tracker.cost_per_1k_input == 0.05
        assert tracker.cost_per_1k_output == 0.10

    def test_initial_state(self):
        """Tracker starts with zero usage and no episode."""
        tracker = BudgetTracker()
        assert tracker._current_episode is None
        assert tracker._global_usage.calls == 0
        assert tracker._episode_usage.calls == 0


# =============================================================================
# Singleton Pattern Tests
# =============================================================================

@pytest.mark.unit
class TestSingletonPattern:
    """Tests for the BudgetTracker singleton/get_instance pattern."""

    def setup_method(self):
        """Reset singleton instances before each test."""
        BudgetTracker.reset_instances()

    def test_get_instance_returns_same_object(self):
        """get_instance returns the same tracker for the same name."""
        t1 = BudgetTracker.get_instance("test")
        t2 = BudgetTracker.get_instance("test")
        assert t1 is t2

    def test_get_instance_different_names(self):
        """get_instance returns different trackers for different names."""
        t1 = BudgetTracker.get_instance("alpha")
        t2 = BudgetTracker.get_instance("beta")
        assert t1 is not t2

    def test_get_instance_default_name(self):
        """get_instance with no name uses 'default'."""
        t1 = BudgetTracker.get_instance()
        t2 = BudgetTracker.get_instance("default")
        assert t1 is t2

    def test_reset_instances_clears_all(self):
        """reset_instances removes all cached instances."""
        BudgetTracker.get_instance("a")
        BudgetTracker.get_instance("b")
        BudgetTracker.reset_instances()
        # After reset, new instances are created
        t_new = BudgetTracker.get_instance("a")
        assert t_new._global_usage.calls == 0

    def test_get_budget_tracker_convenience(self):
        """get_budget_tracker returns a singleton via get_instance."""
        t1 = get_budget_tracker("mytracker")
        t2 = get_budget_tracker("mytracker")
        assert t1 is t2

    def test_get_budget_tracker_default(self):
        """get_budget_tracker with no args uses 'default'."""
        t1 = get_budget_tracker()
        t2 = BudgetTracker.get_instance("default")
        assert t1 is t2

    def test_get_instance_passes_kwargs(self):
        """get_instance forwards kwargs to constructor on first call."""
        config = BudgetConfig(max_llm_calls_per_episode=7)
        tracker = BudgetTracker.get_instance("custom", config=config)
        assert tracker.config.max_llm_calls_per_episode == 7

    def test_get_instance_ignores_kwargs_on_subsequent_call(self):
        """Subsequent get_instance calls ignore kwargs (returns cached)."""
        config1 = BudgetConfig(max_llm_calls_per_episode=7)
        config2 = BudgetConfig(max_llm_calls_per_episode=99)
        t1 = BudgetTracker.get_instance("dup", config=config1)
        t2 = BudgetTracker.get_instance("dup", config=config2)
        assert t2.config.max_llm_calls_per_episode == 7


# =============================================================================
# Episode Lifecycle Tests
# =============================================================================

@pytest.mark.unit
class TestEpisodeLifecycle:
    """Tests for start_episode and end_episode."""

    def setup_method(self):
        BudgetTracker.reset_instances()
        self.tracker = BudgetTracker()

    def test_start_episode_sets_id(self):
        """start_episode sets the current episode ID."""
        self.tracker.start_episode("ep_001")
        assert self.tracker._current_episode == "ep_001"

    def test_start_episode_resets_episode_usage(self):
        """start_episode resets episode-level usage."""
        self.tracker.start_episode("ep_001")
        self.tracker.record_call("agent1", tokens_input=100, tokens_output=50)
        assert self.tracker._episode_usage.calls == 1

        self.tracker.start_episode("ep_002")
        assert self.tracker._episode_usage.calls == 0
        assert self.tracker._episode_usage.total_tokens == 0

    def test_start_episode_clears_agent_usage(self):
        """start_episode clears per-agent usage tracking."""
        self.tracker.start_episode("ep_001")
        self.tracker.record_call("agent1", tokens_input=100, tokens_output=50)
        assert "agent1" in self.tracker._agent_usage

        self.tracker.start_episode("ep_002")
        assert len(self.tracker._agent_usage) == 0

    def test_start_episode_preserves_global_usage(self):
        """start_episode does NOT reset global usage."""
        self.tracker.start_episode("ep_001")
        self.tracker.record_call("agent1", tokens_input=100, tokens_output=50)
        global_calls = self.tracker._global_usage.calls

        self.tracker.start_episode("ep_002")
        assert self.tracker._global_usage.calls == global_calls

    def test_end_episode_returns_summary(self):
        """end_episode returns a usage summary dict."""
        self.tracker.start_episode("ep_001")
        self.tracker.record_call("agent1", tokens_input=100, tokens_output=50)
        summary = self.tracker.end_episode()

        assert summary['episode_id'] == "ep_001"
        assert 'episode_usage' in summary
        assert 'agent_usage' in summary
        assert 'global_usage' in summary
        assert summary['episode_usage']['calls'] == 1

    def test_end_episode_clears_episode_id(self):
        """end_episode sets current episode to None."""
        self.tracker.start_episode("ep_001")
        self.tracker.end_episode()
        assert self.tracker._current_episode is None

    def test_end_episode_includes_agent_breakdown(self):
        """end_episode summary includes per-agent usage."""
        self.tracker.start_episode("ep_001")
        self.tracker.record_call("agentA", tokens_input=100, tokens_output=50)
        self.tracker.record_call("agentB", tokens_input=200, tokens_output=100)
        summary = self.tracker.end_episode()

        assert "agentA" in summary['agent_usage']
        assert "agentB" in summary['agent_usage']
        assert summary['agent_usage']['agentA']['calls'] == 1
        assert summary['agent_usage']['agentB']['calls'] == 1

    def test_multiple_episodes(self):
        """Multiple episodes can run sequentially with independent usage."""
        self.tracker.start_episode("ep_001")
        self.tracker.record_call("a", tokens_input=100, tokens_output=50)
        summary1 = self.tracker.end_episode()

        self.tracker.start_episode("ep_002")
        self.tracker.record_call("a", tokens_input=200, tokens_output=100)
        self.tracker.record_call("a", tokens_input=200, tokens_output=100)
        summary2 = self.tracker.end_episode()

        assert summary1['episode_usage']['calls'] == 1
        assert summary2['episode_usage']['calls'] == 2
        # Global accumulates across episodes
        assert summary2['global_usage']['calls'] == 3


# =============================================================================
# Record Call Tests
# =============================================================================

@pytest.mark.unit
class TestRecordCall:
    """Tests for recording LLM calls."""

    def setup_method(self):
        BudgetTracker.reset_instances()
        self.tracker = BudgetTracker()
        self.tracker.start_episode("test_ep")

    def test_record_call_increments_counts(self):
        """record_call increments call counts at all scopes."""
        self.tracker.record_call("agent1", tokens_input=100, tokens_output=50)
        assert self.tracker._global_usage.calls == 1
        assert self.tracker._episode_usage.calls == 1
        assert self.tracker._agent_usage["agent1"].calls == 1

    def test_record_call_tracks_tokens(self):
        """record_call accumulates token counts."""
        self.tracker.record_call("agent1", tokens_input=100, tokens_output=50)
        self.tracker.record_call("agent1", tokens_input=200, tokens_output=100)

        assert self.tracker._global_usage.tokens_input == 300
        assert self.tracker._global_usage.tokens_output == 150
        assert self.tracker._episode_usage.tokens_input == 300
        assert self.tracker._agent_usage["agent1"].tokens_input == 300

    def test_record_call_sets_last_call_time(self):
        """record_call updates last_call_time."""
        before = time.time()
        self.tracker.record_call("agent1", tokens_input=100, tokens_output=50)
        after = time.time()

        assert before <= self.tracker._global_usage.last_call_time <= after
        assert before <= self.tracker._episode_usage.last_call_time <= after
        assert before <= self.tracker._agent_usage["agent1"].last_call_time <= after

    def test_record_call_default_tokens(self):
        """record_call works with default zero tokens."""
        self.tracker.record_call("agent1")
        assert self.tracker._episode_usage.calls == 1
        assert self.tracker._episode_usage.total_tokens == 0

    def test_record_call_cost_override(self):
        """cost_override bypasses automatic cost calculation."""
        self.tracker.record_call(
            "agent1", tokens_input=1000, tokens_output=1000,
            cost_override=0.50,
        )
        assert self.tracker._global_usage.estimated_cost_usd == 0.50

    def test_record_call_multiple_agents(self):
        """Calls from different agents are tracked separately."""
        self.tracker.record_call("agent1", tokens_input=100, tokens_output=50)
        self.tracker.record_call("agent2", tokens_input=200, tokens_output=100)

        assert self.tracker._agent_usage["agent1"].calls == 1
        assert self.tracker._agent_usage["agent2"].calls == 1
        assert self.tracker._agent_usage["agent1"].tokens_input == 100
        assert self.tracker._agent_usage["agent2"].tokens_input == 200
        # Episode totals both
        assert self.tracker._episode_usage.calls == 2
        assert self.tracker._episode_usage.tokens_input == 300


# =============================================================================
# Cost Estimation Tests
# =============================================================================

@pytest.mark.unit
class TestCostEstimation:
    """Tests for automatic cost calculation."""

    def setup_method(self):
        BudgetTracker.reset_instances()
        self.tracker = BudgetTracker()
        self.tracker.start_episode("cost_ep")

    def test_default_cost_calculation(self):
        """Cost is calculated from default per-1K rates."""
        # 1000 input tokens at $0.01/1K = $0.01
        # 1000 output tokens at $0.03/1K = $0.03
        self.tracker.record_call("agent1", tokens_input=1000, tokens_output=1000)
        expected = 0.01 + 0.03
        assert abs(self.tracker._global_usage.estimated_cost_usd - expected) < 1e-10

    def test_custom_cost_rates(self):
        """Custom cost rates are used in calculation."""
        tracker = BudgetTracker(cost_per_1k_input=0.05, cost_per_1k_output=0.15)
        tracker.start_episode("ep")
        tracker.record_call("agent1", tokens_input=2000, tokens_output=1000)
        # 2000 input at $0.05/1K = $0.10
        # 1000 output at $0.15/1K = $0.15
        expected = 0.10 + 0.15
        assert abs(tracker._global_usage.estimated_cost_usd - expected) < 1e-10

    def test_zero_tokens_zero_cost(self):
        """Zero tokens results in zero cost."""
        self.tracker.record_call("agent1", tokens_input=0, tokens_output=0)
        assert self.tracker._global_usage.estimated_cost_usd == 0.0

    def test_cost_accumulates(self):
        """Cost accumulates across multiple calls."""
        self.tracker.record_call("agent1", tokens_input=1000, tokens_output=0)
        self.tracker.record_call("agent1", tokens_input=1000, tokens_output=0)
        # Two calls of 1000 input each at $0.01/1K = $0.02 total
        assert abs(self.tracker._global_usage.estimated_cost_usd - 0.02) < 1e-10

    def test_cost_override_ignores_tokens(self):
        """cost_override is used instead of token-based calculation."""
        self.tracker.record_call(
            "agent1", tokens_input=1000, tokens_output=1000,
            cost_override=0.001,
        )
        assert self.tracker._global_usage.estimated_cost_usd == 0.001

    def test_estimate_cost_internal(self):
        """_estimate_cost returns correct value directly."""
        cost = self.tracker._estimate_cost(tokens_input=5000, tokens_output=2000)
        # 5000/1000 * 0.01 = 0.05
        # 2000/1000 * 0.03 = 0.06
        expected = 0.05 + 0.06
        assert abs(cost - expected) < 1e-10

    def test_fractional_tokens(self):
        """Cost calculation works with non-round token counts."""
        cost = self.tracker._estimate_cost(tokens_input=500, tokens_output=250)
        expected = (500 / 1000) * 0.01 + (250 / 1000) * 0.03
        assert abs(cost - expected) < 1e-10


# =============================================================================
# Budget Limits / Enforcement Tests
# =============================================================================

@pytest.mark.unit
class TestBudgetLimits:
    """Tests for budget limit enforcement."""

    def setup_method(self):
        BudgetTracker.reset_instances()

    def test_episode_call_limit_raises(self):
        """Exceeding episode call limit raises BudgetExceededError."""
        config = BudgetConfig(max_llm_calls_per_episode=3, max_llm_calls_per_agent=100)
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        tracker.record_call("agent1", tokens_input=10, tokens_output=5)
        tracker.record_call("agent1", tokens_input=10, tokens_output=5)
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.record_call("agent1", tokens_input=10, tokens_output=5)

        assert exc_info.value.scope == BudgetScope.EPISODE
        assert exc_info.value.resource == "calls"

    def test_agent_call_limit_raises(self):
        """Exceeding agent call limit raises BudgetExceededError."""
        config = BudgetConfig(
            max_llm_calls_per_episode=100,
            max_llm_calls_per_agent=2,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        tracker.record_call("agent1", tokens_input=10, tokens_output=5)
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.record_call("agent1", tokens_input=10, tokens_output=5)

        assert exc_info.value.scope == BudgetScope.AGENT

    def test_token_limit_raises(self):
        """Exceeding episode token limit raises BudgetExceededError."""
        config = BudgetConfig(
            max_llm_calls_per_episode=1000,
            max_llm_calls_per_agent=1000,
            max_total_tokens_per_episode=100,
            max_tokens_per_agent=100000,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.record_call("agent1", tokens_input=80, tokens_output=30)

        assert exc_info.value.resource == "tokens"

    def test_cost_limit_raises(self):
        """Exceeding episode cost limit raises BudgetExceededError."""
        config = BudgetConfig(
            max_llm_calls_per_episode=1000,
            max_llm_calls_per_agent=1000,
            max_total_tokens_per_episode=10000000,
            max_cost_per_episode=0.01,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        # 1000 input at $0.01/1K = $0.01 exactly => hits 100%
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.record_call("agent1", tokens_input=1000, tokens_output=0)

        assert exc_info.value.resource == "cost_cents"

    def test_enforcement_disabled(self):
        """No error when enforcement is disabled."""
        config = BudgetConfig(
            max_llm_calls_per_episode=1,
            enable_enforcement=False,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        # Would exceed limit, but enforcement is off
        tracker.record_call("agent1", tokens_input=10, tokens_output=5)
        tracker.record_call("agent1", tokens_input=10, tokens_output=5)
        # No error raised
        assert tracker._episode_usage.calls == 2


# =============================================================================
# can_make_call Tests
# =============================================================================

@pytest.mark.unit
class TestCanMakeCall:
    """Tests for the can_make_call pre-check method."""

    def setup_method(self):
        BudgetTracker.reset_instances()

    def test_within_budget(self):
        """can_make_call returns True when within all limits."""
        tracker = BudgetTracker()
        tracker.start_episode("ep")
        assert tracker.can_make_call("agent1") is True

    def test_episode_call_limit_exceeded(self):
        """can_make_call returns False when episode calls exhausted."""
        config = BudgetConfig(max_llm_calls_per_episode=2, max_llm_calls_per_agent=100)
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")
        tracker._episode_usage.calls = 2
        assert tracker.can_make_call("agent1") is False

    def test_agent_call_limit_exceeded(self):
        """can_make_call returns False when agent calls exhausted."""
        config = BudgetConfig(max_llm_calls_per_agent=5)
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")
        tracker._agent_usage["agent1"].calls = 5
        assert tracker.can_make_call("agent1") is False

    def test_token_limit_exceeded(self):
        """can_make_call returns False when projected tokens exceed limit."""
        config = BudgetConfig(max_total_tokens_per_episode=1000)
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")
        tracker._episode_usage.tokens_input = 800
        tracker._episode_usage.tokens_output = 100
        # Already 900 tokens; adding 200 projected = 1100 > 1000
        assert tracker.can_make_call("agent1", estimated_tokens=200) is False

    def test_per_call_token_limit_exceeded(self):
        """can_make_call returns False when estimated tokens exceed per-call limit."""
        config = BudgetConfig(max_tokens_per_call=500)
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")
        assert tracker.can_make_call("agent1", estimated_tokens=600) is False

    def test_enforcement_disabled_always_true(self):
        """can_make_call always returns True when enforcement is disabled."""
        config = BudgetConfig(
            max_llm_calls_per_episode=0,
            enable_enforcement=False,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")
        assert tracker.can_make_call("agent1") is True

    def test_soft_limit_mode_returns_true(self):
        """can_make_call returns True in soft_limit_mode even when over budget."""
        config = BudgetConfig(
            max_llm_calls_per_episode=1,
            max_llm_calls_per_agent=1,
            max_total_tokens_per_episode=10,
            max_tokens_per_call=5,
            soft_limit_mode=True,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")
        tracker._episode_usage.calls = 100
        tracker._agent_usage["agent1"].calls = 100
        # Even though all limits are exceeded, soft mode returns True
        assert tracker.can_make_call("agent1", estimated_tokens=1000) is True


# =============================================================================
# Soft Limit Mode Tests
# =============================================================================

@pytest.mark.unit
class TestSoftLimitMode:
    """Tests for soft limit (warn but don't block) mode."""

    def setup_method(self):
        BudgetTracker.reset_instances()

    def test_soft_limit_no_exception_on_episode_calls(self):
        """Soft limit mode logs warning instead of raising on episode call limit."""
        config = BudgetConfig(
            max_llm_calls_per_episode=2,
            max_llm_calls_per_agent=100,
            soft_limit_mode=True,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        # Should not raise even when exceeding limit
        tracker.record_call("agent1")
        tracker.record_call("agent1")
        tracker.record_call("agent1")
        assert tracker._episode_usage.calls == 3

    def test_soft_limit_no_exception_on_agent_calls(self):
        """Soft limit mode does not raise on agent call limit."""
        config = BudgetConfig(
            max_llm_calls_per_episode=1000,
            max_llm_calls_per_agent=1,
            soft_limit_mode=True,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        tracker.record_call("agent1")
        tracker.record_call("agent1")
        assert tracker._agent_usage["agent1"].calls == 2

    def test_soft_limit_no_exception_on_token_limit(self):
        """Soft limit mode does not raise on token limit."""
        config = BudgetConfig(
            max_total_tokens_per_episode=10,
            max_tokens_per_agent=100000,
            max_llm_calls_per_episode=1000,
            max_llm_calls_per_agent=1000,
            soft_limit_mode=True,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        tracker.record_call("agent1", tokens_input=100, tokens_output=50)
        assert tracker._episode_usage.total_tokens == 150

    def test_soft_limit_no_exception_on_cost_limit(self):
        """Soft limit mode does not raise on cost limit."""
        config = BudgetConfig(
            max_cost_per_episode=0.001,
            max_llm_calls_per_episode=1000,
            max_llm_calls_per_agent=1000,
            max_total_tokens_per_episode=10000000,
            soft_limit_mode=True,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        tracker.record_call("agent1", tokens_input=10000, tokens_output=10000)
        # No exception raised
        assert tracker._episode_usage.calls == 1


# =============================================================================
# Warning Callback Tests
# =============================================================================

@pytest.mark.unit
class TestWarningCallbacks:
    """Tests for warning threshold and callback mechanism."""

    def setup_method(self):
        BudgetTracker.reset_instances()

    def test_warning_callback_invoked_at_threshold(self):
        """Callback is invoked when usage crosses warning threshold."""
        config = BudgetConfig(
            max_llm_calls_per_episode=10,
            max_llm_calls_per_agent=100,
            warning_threshold=0.8,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        callback = Mock()
        tracker.add_warning_callback(callback)

        # Make 8 calls to reach 80% of 10 limit
        for i in range(8):
            tracker.record_call("agent1")

        assert callback.called

    def test_warning_callback_receives_args(self):
        """Callback receives scope, resource, current, limit, context."""
        config = BudgetConfig(
            max_llm_calls_per_episode=10,
            max_llm_calls_per_agent=100,
            warning_threshold=0.5,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        callback = Mock()
        tracker.add_warning_callback(callback)

        # Make 5 calls to reach 50%
        for _ in range(5):
            tracker.record_call("agent1")

        # Should have been called with episode scope warning
        assert callback.called
        args = callback.call_args[0]
        assert args[0] == BudgetScope.EPISODE  # scope
        assert args[1] == "calls"               # resource

    def test_multiple_callbacks(self):
        """Multiple registered callbacks are all invoked."""
        config = BudgetConfig(
            max_llm_calls_per_episode=10,
            max_llm_calls_per_agent=100,
            warning_threshold=0.5,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        cb1 = Mock()
        cb2 = Mock()
        tracker.add_warning_callback(cb1)
        tracker.add_warning_callback(cb2)

        for _ in range(5):
            tracker.record_call("agent1")

        assert cb1.called
        assert cb2.called

    def test_callback_exception_does_not_propagate(self):
        """Exception in callback is caught and does not crash the tracker."""
        config = BudgetConfig(
            max_llm_calls_per_episode=10,
            max_llm_calls_per_agent=100,
            warning_threshold=0.5,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        def bad_callback(*args):
            raise ValueError("callback error")

        tracker.add_warning_callback(bad_callback)

        # Should not raise
        for _ in range(6):
            tracker.record_call("agent1")

        assert tracker._episode_usage.calls == 6

    def test_warning_not_emitted_below_threshold(self):
        """No warning emitted when usage is below threshold."""
        config = BudgetConfig(
            max_llm_calls_per_episode=100,
            max_llm_calls_per_agent=100,
            warning_threshold=0.8,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        callback = Mock()
        tracker.add_warning_callback(callback)

        # Make 1 call (1% of 100)
        tracker.record_call("agent1")

        assert not callback.called

    def test_warning_emitted_only_once_per_level(self):
        """Warnings are deduplicated per warning_level integer threshold."""
        config = BudgetConfig(
            max_llm_calls_per_episode=10,
            max_llm_calls_per_agent=100,
            warning_threshold=0.5,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        callback = Mock()
        tracker.add_warning_callback(callback)

        # Calls 5 and 6 are both at warning_level=5 (50%-60%)
        for _ in range(5):
            tracker.record_call("agent1")

        first_call_count = callback.call_count

        # Call 6 is 60%, warning_level=6 => new warning
        tracker.record_call("agent1")
        assert callback.call_count > first_call_count


# =============================================================================
# Multi-Agent Tracking Tests
# =============================================================================

@pytest.mark.unit
class TestMultiAgentTracking:
    """Tests for tracking multiple agents independently."""

    def setup_method(self):
        BudgetTracker.reset_instances()
        self.tracker = BudgetTracker()
        self.tracker.start_episode("multi_ep")

    def test_independent_agent_counts(self):
        """Each agent has independent call counts."""
        self.tracker.record_call("agent_a")
        self.tracker.record_call("agent_a")
        self.tracker.record_call("agent_b")

        assert self.tracker._agent_usage["agent_a"].calls == 2
        assert self.tracker._agent_usage["agent_b"].calls == 1

    def test_independent_agent_tokens(self):
        """Each agent has independent token counts."""
        self.tracker.record_call("agent_a", tokens_input=100, tokens_output=50)
        self.tracker.record_call("agent_b", tokens_input=200, tokens_output=100)

        assert self.tracker._agent_usage["agent_a"].total_tokens == 150
        assert self.tracker._agent_usage["agent_b"].total_tokens == 300

    def test_independent_agent_cost(self):
        """Each agent has independent cost tracking."""
        self.tracker.record_call("agent_a", tokens_input=1000, tokens_output=0)
        self.tracker.record_call("agent_b", tokens_input=0, tokens_output=1000)

        assert abs(self.tracker._agent_usage["agent_a"].estimated_cost_usd - 0.01) < 1e-10
        assert abs(self.tracker._agent_usage["agent_b"].estimated_cost_usd - 0.03) < 1e-10

    def test_agent_limit_independent(self):
        """Hitting agent limit for one agent does not block another."""
        config = BudgetConfig(
            max_llm_calls_per_episode=1000,
            max_llm_calls_per_agent=2,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        tracker.record_call("agent_a")
        with pytest.raises(BudgetExceededError):
            tracker.record_call("agent_a")

        # agent_b should still be able to make calls
        assert tracker.can_make_call("agent_b")
        tracker.record_call("agent_b")
        assert tracker._agent_usage["agent_b"].calls == 1

    def test_global_totals_aggregate_all_agents(self):
        """Global usage sums across all agents."""
        self.tracker.record_call("agent_a", tokens_input=100, tokens_output=50)
        self.tracker.record_call("agent_b", tokens_input=200, tokens_output=100)
        self.tracker.record_call("agent_c", tokens_input=300, tokens_output=150)

        assert self.tracker._global_usage.calls == 3
        assert self.tracker._global_usage.tokens_input == 600
        assert self.tracker._global_usage.tokens_output == 300


# =============================================================================
# get_usage Tests
# =============================================================================

@pytest.mark.unit
class TestGetUsage:
    """Tests for the get_usage method."""

    def setup_method(self):
        BudgetTracker.reset_instances()
        self.tracker = BudgetTracker()
        self.tracker.start_episode("usage_ep")
        self.tracker.record_call("agent1", tokens_input=100, tokens_output=50)

    def test_get_usage_global(self):
        """get_usage for GLOBAL scope returns global stats."""
        usage = self.tracker.get_usage(BudgetScope.GLOBAL)
        assert usage['calls'] == 1
        assert usage['tokens_input'] == 100
        assert usage['tokens_output'] == 50

    def test_get_usage_episode(self):
        """get_usage for EPISODE scope returns episode stats."""
        usage = self.tracker.get_usage(BudgetScope.EPISODE)
        assert usage['calls'] == 1
        assert usage['total_tokens'] == 150

    def test_get_usage_agent(self):
        """get_usage for AGENT scope returns per-agent breakdown."""
        usage = self.tracker.get_usage(BudgetScope.AGENT)
        assert "agent1" in usage
        assert usage["agent1"]['calls'] == 1

    def test_get_usage_default_is_episode(self):
        """get_usage defaults to EPISODE scope."""
        usage = self.tracker.get_usage()
        assert usage['calls'] == 1

    def test_get_usage_operation_returns_empty(self):
        """get_usage for OPERATION scope returns agent dict (fallthrough)."""
        # OPERATION falls into the else branch, same as AGENT
        usage = self.tracker.get_usage(BudgetScope.OPERATION)
        assert isinstance(usage, dict)


# =============================================================================
# get_remaining Tests
# =============================================================================

@pytest.mark.unit
class TestGetRemaining:
    """Tests for the get_remaining method."""

    def setup_method(self):
        BudgetTracker.reset_instances()
        self.tracker = BudgetTracker()
        self.tracker.start_episode("remaining_ep")

    def test_remaining_episode_calls(self):
        """Remaining episode calls decreases with usage."""
        remaining_before = self.tracker.get_remaining(BudgetScope.EPISODE)['calls']
        self.tracker.record_call("agent1")
        remaining_after = self.tracker.get_remaining(BudgetScope.EPISODE)['calls']
        assert remaining_after == remaining_before - 1

    def test_remaining_episode_tokens(self):
        """Remaining episode tokens decreases with usage."""
        remaining_before = self.tracker.get_remaining(BudgetScope.EPISODE)['tokens']
        self.tracker.record_call("agent1", tokens_input=100, tokens_output=50)
        remaining_after = self.tracker.get_remaining(BudgetScope.EPISODE)['tokens']
        assert remaining_after == remaining_before - 150

    def test_remaining_global_calls(self):
        """Remaining global calls decreases with usage."""
        remaining_before = self.tracker.get_remaining(BudgetScope.GLOBAL)['calls']
        self.tracker.record_call("agent1")
        remaining_after = self.tracker.get_remaining(BudgetScope.GLOBAL)['calls']
        assert remaining_after == remaining_before - 1

    def test_remaining_never_negative(self):
        """Remaining clamps to zero, never goes negative."""
        config = BudgetConfig(
            max_llm_calls_per_episode=1,
            max_llm_calls_per_agent=100,
            enable_enforcement=False,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")
        tracker.record_call("agent1")
        tracker.record_call("agent1")

        remaining = tracker.get_remaining(BudgetScope.EPISODE)
        assert remaining['calls'] == 0  # clamped at 0, not -1

    def test_remaining_default_is_episode(self):
        """get_remaining defaults to EPISODE scope."""
        remaining = self.tracker.get_remaining()
        assert 'calls' in remaining
        assert 'tokens' in remaining

    def test_remaining_agent_returns_empty(self):
        """get_remaining for AGENT scope returns empty dict."""
        remaining = self.tracker.get_remaining(BudgetScope.AGENT)
        assert remaining == {}

    def test_remaining_full_budget_initially(self):
        """Before any calls, remaining equals the configured limit."""
        config = BudgetConfig(max_llm_calls_per_episode=100)
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")
        remaining = tracker.get_remaining(BudgetScope.EPISODE)
        assert remaining['calls'] == 100


# =============================================================================
# reset_global Tests
# =============================================================================

@pytest.mark.unit
class TestResetGlobal:
    """Tests for the reset_global method."""

    def setup_method(self):
        BudgetTracker.reset_instances()

    def test_reset_global_clears_global_usage(self):
        """reset_global zeroes out global usage counters."""
        tracker = BudgetTracker()
        tracker.start_episode("ep")
        tracker.record_call("agent1", tokens_input=100, tokens_output=50)
        assert tracker._global_usage.calls == 1

        tracker.reset_global()
        assert tracker._global_usage.calls == 0
        assert tracker._global_usage.tokens_input == 0
        assert tracker._global_usage.tokens_output == 0
        assert tracker._global_usage.estimated_cost_usd == 0.0

    def test_reset_global_preserves_episode_usage(self):
        """reset_global does NOT reset episode-level usage."""
        tracker = BudgetTracker()
        tracker.start_episode("ep")
        tracker.record_call("agent1", tokens_input=100, tokens_output=50)

        tracker.reset_global()
        assert tracker._episode_usage.calls == 1

    def test_reset_global_preserves_agent_usage(self):
        """reset_global does NOT reset agent-level usage."""
        tracker = BudgetTracker()
        tracker.start_episode("ep")
        tracker.record_call("agent1", tokens_input=100, tokens_output=50)

        tracker.reset_global()
        assert tracker._agent_usage["agent1"].calls == 1


# =============================================================================
# Thread Safety Tests
# =============================================================================

@pytest.mark.unit
class TestThreadSafety:
    """Tests for thread-safe operations."""

    def setup_method(self):
        BudgetTracker.reset_instances()

    def test_concurrent_record_calls(self):
        """Concurrent record_call from multiple threads maintains consistency."""
        config = BudgetConfig(
            max_llm_calls_per_episode=10000,
            max_llm_calls_per_agent=10000,
            max_total_tokens_per_episode=100000000,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("thread_ep")

        num_threads = 10
        calls_per_thread = 100
        errors = []

        def worker(agent_name):
            try:
                for _ in range(calls_per_thread):
                    tracker.record_call(agent_name, tokens_input=1, tokens_output=1)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(f"agent_{i}",))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        expected_total = num_threads * calls_per_thread
        assert tracker._global_usage.calls == expected_total
        assert tracker._episode_usage.calls == expected_total

    def test_concurrent_can_make_call(self):
        """can_make_call is safe to call from multiple threads."""
        tracker = BudgetTracker()
        tracker.start_episode("ep")
        results = []

        def worker():
            result = tracker.can_make_call("agent1", estimated_tokens=100)
            results.append(result)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r is True for r in results)


# =============================================================================
# Edge Cases and Miscellaneous Tests
# =============================================================================

@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def setup_method(self):
        BudgetTracker.reset_instances()

    def test_record_call_without_start_episode(self):
        """record_call works even without starting an episode."""
        tracker = BudgetTracker()
        tracker.record_call("agent1", tokens_input=100, tokens_output=50)
        assert tracker._global_usage.calls == 1
        assert tracker._episode_usage.calls == 1

    def test_end_episode_without_start(self):
        """end_episode returns summary even when no episode was started."""
        tracker = BudgetTracker()
        summary = tracker.end_episode()
        assert summary['episode_id'] is None
        assert summary['episode_usage']['calls'] == 0

    def test_exactly_at_limit_triggers_exceeded(self):
        """Reaching exactly the limit triggers BudgetExceededError."""
        config = BudgetConfig(
            max_llm_calls_per_episode=1,
            max_llm_calls_per_agent=100,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        # First call brings count to 1, which equals limit => error
        with pytest.raises(BudgetExceededError):
            tracker.record_call("agent1")

    def test_model_param_accepted(self):
        """record_call accepts model parameter without error."""
        tracker = BudgetTracker()
        tracker.start_episode("ep")
        tracker.record_call(
            "agent1", tokens_input=100, tokens_output=50,
            model="claude-3-sonnet",
        )
        assert tracker._episode_usage.calls == 1

    def test_empty_agent_name(self):
        """Empty string as agent name is accepted."""
        tracker = BudgetTracker()
        tracker.start_episode("ep")
        tracker.record_call("", tokens_input=10, tokens_output=5)
        assert tracker._agent_usage[""].calls == 1

    def test_large_token_values(self):
        """Large token values are handled correctly."""
        tracker = BudgetTracker(
            config=BudgetConfig(
                max_total_tokens_per_episode=10**9,
                max_tokens_per_agent=10**9,
                max_llm_calls_per_episode=10000,
                max_llm_calls_per_agent=10000,
            )
        )
        tracker.start_episode("ep")
        tracker.record_call("agent1", tokens_input=10**6, tokens_output=10**6)
        assert tracker._episode_usage.total_tokens == 2 * 10**6

    def test_cost_override_zero(self):
        """cost_override=0.0 is respected (not treated as falsy None)."""
        tracker = BudgetTracker()
        tracker.start_episode("ep")
        tracker.record_call("agent1", tokens_input=1000, tokens_output=1000, cost_override=0.0)
        assert tracker._global_usage.estimated_cost_usd == 0.0

    def test_can_make_call_creates_agent_entry(self):
        """can_make_call creates a defaultdict entry for the agent."""
        tracker = BudgetTracker()
        tracker.start_episode("ep")
        tracker.can_make_call("new_agent")
        assert "new_agent" in tracker._agent_usage

    def test_warning_threshold_at_zero(self):
        """warning_threshold of 0 means warnings fire immediately."""
        config = BudgetConfig(
            max_llm_calls_per_episode=10,
            max_llm_calls_per_agent=100,
            warning_threshold=0.0,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        callback = Mock()
        tracker.add_warning_callback(callback)
        tracker.record_call("agent1")
        # Warning level 1 (10%) > 0 warnings_emitted => callback fires
        assert callback.called

    def test_warning_threshold_at_one(self):
        """warning_threshold of 1.0 means no early warnings, only limit exceeded."""
        config = BudgetConfig(
            max_llm_calls_per_episode=100,
            max_llm_calls_per_agent=100,
            warning_threshold=1.0,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        callback = Mock()
        tracker.add_warning_callback(callback)
        for _ in range(99):
            tracker.record_call("agent1")
        # At 99/100 (99%), < 1.0 threshold => no warning
        assert not callback.called


# =============================================================================
# _check_limits Internal Logic Tests
# =============================================================================

@pytest.mark.unit
class TestCheckLimitsInternal:
    """Tests for the internal _check_limits logic paths."""

    def setup_method(self):
        BudgetTracker.reset_instances()

    def test_agent_warning_emitted(self):
        """Agent-level warning is emitted when agent usage crosses threshold."""
        config = BudgetConfig(
            max_llm_calls_per_episode=1000,
            max_llm_calls_per_agent=10,
            warning_threshold=0.8,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        callback = Mock()
        tracker.add_warning_callback(callback)

        # Make 8 calls to hit 80% of agent limit (10)
        for _ in range(8):
            tracker.record_call("agent1")

        # Check that callback was invoked with AGENT scope
        agent_scope_calls = [
            c for c in callback.call_args_list
            if c[0][0] == BudgetScope.AGENT
        ]
        assert len(agent_scope_calls) > 0

    def test_token_warning_emitted(self):
        """Episode-level token warning is emitted at threshold."""
        config = BudgetConfig(
            max_llm_calls_per_episode=1000,
            max_llm_calls_per_agent=1000,
            max_total_tokens_per_episode=1000,
            warning_threshold=0.8,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        callback = Mock()
        tracker.add_warning_callback(callback)

        # 800 tokens = 80% of 1000
        tracker.record_call("agent1", tokens_input=800, tokens_output=0)

        token_calls = [
            c for c in callback.call_args_list
            if c[0][1] == "tokens"
        ]
        assert len(token_calls) > 0

    def test_cost_warning_emitted(self):
        """Cost warning is emitted when cost crosses threshold."""
        config = BudgetConfig(
            max_llm_calls_per_episode=1000,
            max_llm_calls_per_agent=1000,
            max_total_tokens_per_episode=10000000,
            max_cost_per_episode=1.0,
            warning_threshold=0.8,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        callback = Mock()
        tracker.add_warning_callback(callback)

        # Need to spend ~$0.80 to hit 80% of $1.00
        # $0.01 per 1K input => need 80K input tokens
        tracker.record_call("agent1", tokens_input=80000, tokens_output=0)

        cost_calls = [
            c for c in callback.call_args_list
            if c[0][1] == "cost"
        ]
        assert len(cost_calls) > 0

    def test_no_cost_check_when_limit_is_none(self):
        """No cost check occurs when max_cost_per_episode is None."""
        config = BudgetConfig(
            max_llm_calls_per_episode=1000,
            max_llm_calls_per_agent=1000,
            max_total_tokens_per_episode=10000000,
            max_cost_per_episode=None,
        )
        tracker = BudgetTracker(config=config)
        tracker.start_episode("ep")

        callback = Mock()
        tracker.add_warning_callback(callback)

        tracker.record_call("agent1", tokens_input=100000, tokens_output=100000)

        cost_calls = [
            c for c in callback.call_args_list
            if len(c[0]) > 1 and c[0][1] == "cost"
        ]
        assert len(cost_calls) == 0


# =============================================================================
# Serialization / to_dict Round-trip Tests
# =============================================================================

@pytest.mark.unit
class TestSerialization:
    """Tests for to_dict serialization consistency."""

    def setup_method(self):
        BudgetTracker.reset_instances()

    def test_end_episode_summary_structure(self):
        """end_episode returns a properly structured summary dict."""
        tracker = BudgetTracker()
        tracker.start_episode("ep_ser")
        tracker.record_call("a1", tokens_input=100, tokens_output=50)
        tracker.record_call("a2", tokens_input=200, tokens_output=100)
        summary = tracker.end_episode()

        # Top-level keys
        assert set(summary.keys()) == {'episode_id', 'episode_usage', 'agent_usage', 'global_usage'}

        # Episode usage has all expected fields
        ep = summary['episode_usage']
        assert all(k in ep for k in ['calls', 'tokens_input', 'tokens_output', 'total_tokens', 'estimated_cost_usd'])

        # Agent usage has entries for both agents
        assert set(summary['agent_usage'].keys()) == {'a1', 'a2'}

    def test_budget_config_roundtrip(self):
        """BudgetConfig to_dict values match field values."""
        config = BudgetConfig(
            max_llm_calls_per_episode=42,
            soft_limit_mode=True,
            warning_threshold=0.95,
        )
        d = config.to_dict()
        assert d['max_llm_calls_per_episode'] == 42
        assert d['soft_limit_mode'] is True
        assert d['warning_threshold'] == 0.95

    def test_budget_usage_roundtrip(self):
        """BudgetUsage to_dict values match field values after mutation."""
        usage = BudgetUsage()
        usage.calls = 10
        usage.tokens_input = 5000
        usage.tokens_output = 2000
        usage.estimated_cost_usd = 0.11
        d = usage.to_dict()
        assert d['calls'] == 10
        assert d['total_tokens'] == 7000
        assert d['estimated_cost_usd'] == 0.11

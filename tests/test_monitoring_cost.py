"""
Comprehensive tests for Jotty monitoring and cost tracking modules.

Tests cover:
- PRICING_TABLE and DEFAULT_PRICING constants
- LLMCallRecord dataclass
- CostMetrics dataclass
- CostTracker class (full lifecycle)
- MonitoringFramework class
- EfficiencyMetrics class
"""

import json
import time
from typing import Any, Dict

import pytest

# --- cost_tracker imports ---
try:
    from Jotty.core.infrastructure.monitoring.monitoring.cost_tracker import (
        DEFAULT_PRICING,
        PRICING_TABLE,
        CostMetrics,
        CostTracker,
        LLMCallRecord,
    )

    _HAS_COST_TRACKER = True
except ImportError:
    _HAS_COST_TRACKER = False

# --- monitoring_framework imports ---
try:
    from Jotty.core.infrastructure.monitoring.monitoring.monitoring_framework import (
        ExecutionMetrics,
        ExecutionStatus,
        MonitoringFramework,
        PerformanceMetrics,
    )

    _HAS_MONITORING_FRAMEWORK = True
except ImportError:
    _HAS_MONITORING_FRAMEWORK = False

# --- efficiency_metrics imports ---
try:
    from Jotty.core.infrastructure.monitoring.monitoring.efficiency_metrics import (
        EfficiencyMetrics,
        EfficiencyReport,
    )

    _HAS_EFFICIENCY_METRICS = True
except ImportError:
    _HAS_EFFICIENCY_METRICS = False


# ============================================================================
# 1. PRICING_TABLE and DEFAULT_PRICING
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_COST_TRACKER, reason="cost_tracker not importable")
class TestPricingTable:
    """Tests for PRICING_TABLE and DEFAULT_PRICING constants."""

    def test_pricing_table_has_claude_models(self):
        """PRICING_TABLE contains Anthropic Claude models."""
        claude_models = [k for k in PRICING_TABLE if "claude" in k.lower()]
        assert (
            len(claude_models) >= 3
        ), f"Expected at least 3 Claude models, found {len(claude_models)}"

    def test_pricing_table_has_openai_models(self):
        """PRICING_TABLE contains OpenAI models."""
        openai_models = [k for k in PRICING_TABLE if k.startswith(("gpt-", "o1", "o3"))]
        assert (
            len(openai_models) >= 2
        ), f"Expected at least 2 OpenAI models, found {len(openai_models)}"

    def test_pricing_table_has_gemini_models(self):
        """PRICING_TABLE contains Gemini models."""
        gemini_models = [k for k in PRICING_TABLE if "gemini" in k.lower()]
        assert (
            len(gemini_models) >= 1
        ), f"Expected at least 1 Gemini model, found {len(gemini_models)}"

    def test_pricing_entries_have_input_output_floats(self):
        """Every entry in PRICING_TABLE has 'input' and 'output' float keys."""
        for model, pricing in PRICING_TABLE.items():
            assert "input" in pricing, f"Model '{model}' missing 'input' key"
            assert "output" in pricing, f"Model '{model}' missing 'output' key"
            assert isinstance(
                pricing["input"], (int, float)
            ), f"Model '{model}' input is not numeric: {type(pricing['input'])}"
            assert isinstance(
                pricing["output"], (int, float)
            ), f"Model '{model}' output is not numeric: {type(pricing['output'])}"

    def test_pricing_values_are_positive(self):
        """All pricing values are positive (non-negative)."""
        for model, pricing in PRICING_TABLE.items():
            assert pricing["input"] >= 0, f"Model '{model}' has negative input price"
            assert pricing["output"] >= 0, f"Model '{model}' has negative output price"

    def test_default_pricing_has_input_output(self):
        """DEFAULT_PRICING has 'input' and 'output' keys."""
        assert "input" in DEFAULT_PRICING
        assert "output" in DEFAULT_PRICING
        assert isinstance(DEFAULT_PRICING["input"], (int, float))
        assert isinstance(DEFAULT_PRICING["output"], (int, float))

    def test_default_pricing_values(self):
        """DEFAULT_PRICING has expected default values."""
        assert DEFAULT_PRICING["input"] == 3.0
        assert DEFAULT_PRICING["output"] == 15.0


# ============================================================================
# 2. LLMCallRecord dataclass
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_COST_TRACKER, reason="cost_tracker not importable")
class TestLLMCallRecord:
    """Tests for LLMCallRecord dataclass."""

    def _make_record(self, **overrides):
        """Factory helper for creating LLMCallRecord instances."""
        defaults = dict(
            timestamp=1000.0,
            provider="anthropic",
            model="claude-sonnet-4",
            input_tokens=500,
            output_tokens=200,
            cost=0.0045,
            success=True,
            error=None,
            duration=1.5,
        )
        defaults.update(overrides)
        return LLMCallRecord(**defaults)

    def test_all_fields_assigned(self):
        """All fields are correctly assigned on construction."""
        record = self._make_record()
        assert record.timestamp == 1000.0
        assert record.provider == "anthropic"
        assert record.model == "claude-sonnet-4"
        assert record.input_tokens == 500
        assert record.output_tokens == 200
        assert record.cost == 0.0045
        assert record.success is True
        assert record.error is None
        assert record.duration == 1.5

    def test_optional_fields_default_none(self):
        """Optional fields (error, duration) default to None."""
        record = LLMCallRecord(
            timestamp=1.0,
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost=0.001,
            success=True,
        )
        assert record.error is None
        assert record.duration is None

    def test_error_field(self):
        """Error field stores the error message."""
        record = self._make_record(success=False, error="API timeout")
        assert record.success is False
        assert record.error == "API timeout"

    def test_to_dict_returns_all_keys(self):
        """to_dict() returns a dict with all expected keys."""
        record = self._make_record()
        d = record.to_dict()
        expected_keys = {
            "timestamp",
            "provider",
            "model",
            "input_tokens",
            "output_tokens",
            "cost",
            "success",
            "error",
            "duration",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match_fields(self):
        """to_dict() values match the dataclass field values."""
        record = self._make_record(
            timestamp=999.0,
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=300,
            output_tokens=100,
            cost=0.00012,
            success=False,
            error="rate_limit",
            duration=2.5,
        )
        d = record.to_dict()
        assert d["timestamp"] == 999.0
        assert d["provider"] == "openai"
        assert d["model"] == "gpt-4o-mini"
        assert d["input_tokens"] == 300
        assert d["output_tokens"] == 100
        assert d["cost"] == 0.00012
        assert d["success"] is False
        assert d["error"] == "rate_limit"
        assert d["duration"] == 2.5


# ============================================================================
# 3. CostMetrics dataclass
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_COST_TRACKER, reason="cost_tracker not importable")
class TestCostMetrics:
    """Tests for CostMetrics dataclass."""

    def _make_metrics(self, **overrides):
        """Factory helper for creating CostMetrics instances."""
        defaults = dict(
            total_cost=0.05,
            total_input_tokens=5000,
            total_output_tokens=2000,
            total_tokens=7000,
            total_calls=10,
            successful_calls=8,
            failed_calls=2,
            avg_cost_per_call=0.005,
            avg_tokens_per_call=700.0,
            cost_per_1k_tokens=0.00714,
            cost_by_provider={"anthropic": 0.03, "openai": 0.02},
            cost_by_model={"claude-sonnet-4": 0.03, "gpt-4o": 0.02},
            calls_by_provider={"anthropic": 6, "openai": 4},
            calls_by_model={"claude-sonnet-4": 6, "gpt-4o": 4},
        )
        defaults.update(overrides)
        return CostMetrics(**defaults)

    def test_all_fields_assigned(self):
        """All fields are correctly assigned on construction."""
        m = self._make_metrics()
        assert m.total_cost == 0.05
        assert m.total_input_tokens == 5000
        assert m.total_output_tokens == 2000
        assert m.total_tokens == 7000
        assert m.total_calls == 10
        assert m.successful_calls == 8
        assert m.failed_calls == 2
        assert m.avg_cost_per_call == 0.005
        assert m.avg_tokens_per_call == 700.0
        assert m.cost_by_provider == {"anthropic": 0.03, "openai": 0.02}
        assert m.cost_by_model == {"claude-sonnet-4": 0.03, "gpt-4o": 0.02}

    def test_to_dict_returns_all_keys(self):
        """to_dict() returns a dict with all expected keys."""
        m = self._make_metrics()
        d = m.to_dict()
        expected_keys = {
            "total_cost",
            "total_input_tokens",
            "total_output_tokens",
            "total_tokens",
            "total_calls",
            "successful_calls",
            "failed_calls",
            "avg_cost_per_call",
            "avg_tokens_per_call",
            "cost_per_1k_tokens",
            "cost_by_provider",
            "cost_by_model",
            "calls_by_provider",
            "calls_by_model",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match(self):
        """to_dict() values match the dataclass fields."""
        m = self._make_metrics()
        d = m.to_dict()
        assert d["total_cost"] == m.total_cost
        assert d["total_calls"] == m.total_calls
        assert d["cost_by_provider"] == m.cost_by_provider
        assert d["cost_by_model"] == m.cost_by_model


# ============================================================================
# 4. CostTracker class
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_COST_TRACKER, reason="cost_tracker not importable")
class TestCostTracker:
    """Tests for CostTracker class."""

    # ---------- __init__ ----------

    def test_init_default_tracking_enabled(self):
        """Default constructor enables tracking with empty calls list."""
        tracker = CostTracker()
        assert tracker.enable_tracking is True
        assert tracker.calls == []
        assert isinstance(tracker.pricing_table, dict)
        assert len(tracker.pricing_table) > 0

    def test_init_tracking_disabled(self):
        """Constructor with enable_tracking=False."""
        tracker = CostTracker(enable_tracking=False)
        assert tracker.enable_tracking is False
        assert tracker.calls == []

    def test_init_pricing_table_is_copy(self):
        """pricing_table is a copy of PRICING_TABLE (not the same object)."""
        tracker = CostTracker()
        assert tracker.pricing_table is not PRICING_TABLE
        assert tracker.pricing_table == PRICING_TABLE

    # ---------- update_pricing ----------

    def test_update_pricing_adds_new_model(self):
        """update_pricing adds a new model to the pricing table."""
        tracker = CostTracker()
        tracker.update_pricing("my-custom-model", 5.0, 25.0)
        assert "my-custom-model" in tracker.pricing_table
        assert tracker.pricing_table["my-custom-model"] == {"input": 5.0, "output": 25.0}

    def test_update_pricing_overrides_existing(self):
        """update_pricing overrides an existing model's pricing."""
        tracker = CostTracker()
        tracker.update_pricing("claude-sonnet-4", 99.0, 199.0)
        assert tracker.pricing_table["claude-sonnet-4"] == {"input": 99.0, "output": 199.0}

    # ---------- _calculate_cost ----------

    def test_calculate_cost_known_model(self):
        """_calculate_cost uses PRICING_TABLE rates for known models."""
        tracker = CostTracker()
        # claude-sonnet-4: input=3.0/1M, output=15.0/1M
        cost = tracker._calculate_cost("claude-sonnet-4", 1_000_000, 1_000_000)
        expected = 3.0 + 15.0  # $3 input + $15 output
        assert cost == pytest.approx(expected)

    def test_calculate_cost_known_model_small_tokens(self):
        """_calculate_cost with small token counts for known model."""
        tracker = CostTracker()
        # gpt-4o: input=2.5/1M, output=10.0/1M
        cost = tracker._calculate_cost("gpt-4o", 1000, 500)
        expected_input = (1000 / 1_000_000) * 2.5  # 0.0025
        expected_output = (500 / 1_000_000) * 10.0  # 0.005
        expected = expected_input + expected_output  # 0.0075
        assert cost == pytest.approx(expected)

    def test_calculate_cost_unknown_model_uses_default(self):
        """_calculate_cost falls back to DEFAULT_PRICING for unknown models."""
        tracker = CostTracker()
        cost = tracker._calculate_cost("unknown-model-xyz", 1_000_000, 1_000_000)
        expected = DEFAULT_PRICING["input"] + DEFAULT_PRICING["output"]  # 3.0 + 15.0
        assert cost == pytest.approx(expected)

    def test_calculate_cost_zero_tokens(self):
        """_calculate_cost returns 0.0 when both token counts are zero."""
        tracker = CostTracker()
        cost = tracker._calculate_cost("claude-sonnet-4", 0, 0)
        assert cost == 0.0

    def test_calculate_cost_large_tokens(self):
        """_calculate_cost handles large token counts correctly."""
        tracker = CostTracker()
        # claude-opus-4: input=15.0/1M, output=75.0/1M
        cost = tracker._calculate_cost("claude-opus-4", 10_000_000, 5_000_000)
        expected_input = 10.0 * 15.0  # 150.0
        expected_output = 5.0 * 75.0  # 375.0
        expected = expected_input + expected_output  # 525.0
        assert cost == pytest.approx(expected)

    # ---------- record_llm_call ----------

    def test_record_llm_call_enabled_creates_record(self):
        """record_llm_call with tracking enabled creates and stores a record."""
        tracker = CostTracker(enable_tracking=True)
        record = tracker.record_llm_call(
            provider="anthropic",
            model="claude-sonnet-4",
            input_tokens=1000,
            output_tokens=500,
            success=True,
        )
        assert isinstance(record, LLMCallRecord)
        assert record.provider == "anthropic"
        assert record.model == "claude-sonnet-4"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.success is True
        assert record.cost > 0.0
        assert len(tracker.calls) == 1
        assert tracker.calls[0] is record

    def test_record_llm_call_disabled_returns_zero_cost(self):
        """record_llm_call with tracking disabled returns zero cost and does not store."""
        tracker = CostTracker(enable_tracking=False)
        record = tracker.record_llm_call(
            provider="anthropic",
            model="claude-sonnet-4",
            input_tokens=1000,
            output_tokens=500,
            success=True,
        )
        assert record.cost == 0.0
        assert len(tracker.calls) == 0

    def test_record_llm_call_custom_cost_override(self):
        """record_llm_call with custom_cost uses the override value."""
        tracker = CostTracker(enable_tracking=True)
        record = tracker.record_llm_call(
            provider="anthropic",
            model="claude-sonnet-4",
            input_tokens=1000,
            output_tokens=500,
            success=True,
            custom_cost=0.42,
        )
        assert record.cost == 0.42

    def test_record_llm_call_error_and_duration_fields(self):
        """record_llm_call correctly stores error and duration."""
        tracker = CostTracker(enable_tracking=True)
        record = tracker.record_llm_call(
            provider="openai",
            model="gpt-4o",
            input_tokens=200,
            output_tokens=100,
            success=False,
            error="Connection timeout",
            duration=5.5,
        )
        assert record.success is False
        assert record.error == "Connection timeout"
        assert record.duration == 5.5

    def test_record_llm_call_calculates_correct_cost(self):
        """record_llm_call calculates cost matching _calculate_cost."""
        tracker = CostTracker(enable_tracking=True)
        model = "gpt-4o-mini"
        input_tokens = 2000
        output_tokens = 800
        expected_cost = tracker._calculate_cost(model, input_tokens, output_tokens)
        record = tracker.record_llm_call(
            provider="openai",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=True,
        )
        assert record.cost == pytest.approx(expected_cost)

    # ---------- get_metrics ----------

    def test_get_metrics_empty_all_zeros(self):
        """get_metrics on empty tracker returns all zeros."""
        tracker = CostTracker()
        m = tracker.get_metrics()
        assert m.total_cost == 0.0
        assert m.total_input_tokens == 0
        assert m.total_output_tokens == 0
        assert m.total_tokens == 0
        assert m.total_calls == 0
        assert m.successful_calls == 0
        assert m.failed_calls == 0
        assert m.avg_cost_per_call == 0.0
        assert m.avg_tokens_per_call == 0.0
        assert m.cost_per_1k_tokens == 0.0
        assert m.cost_by_provider == {}
        assert m.cost_by_model == {}
        assert m.calls_by_provider == {}
        assert m.calls_by_model == {}

    def test_get_metrics_after_calls_correct_aggregation(self):
        """get_metrics after multiple calls returns correct aggregation."""
        tracker = CostTracker(enable_tracking=True)

        # Call 1: anthropic / claude-sonnet-4 (input=3.0/1M, output=15.0/1M)
        tracker.record_llm_call(
            provider="anthropic",
            model="claude-sonnet-4",
            input_tokens=1000,
            output_tokens=500,
            success=True,
        )
        # Call 2: openai / gpt-4o (input=2.5/1M, output=10.0/1M)
        tracker.record_llm_call(
            provider="openai",
            model="gpt-4o",
            input_tokens=2000,
            output_tokens=1000,
            success=True,
        )
        # Call 3: anthropic / claude-sonnet-4, failed
        tracker.record_llm_call(
            provider="anthropic",
            model="claude-sonnet-4",
            input_tokens=500,
            output_tokens=0,
            success=False,
            error="API error",
        )

        m = tracker.get_metrics()

        assert m.total_calls == 3
        assert m.successful_calls == 2
        assert m.failed_calls == 1

        assert m.total_input_tokens == 1000 + 2000 + 500
        assert m.total_output_tokens == 500 + 1000 + 0
        assert m.total_tokens == m.total_input_tokens + m.total_output_tokens

        # Compute exact expected costs from PRICING_TABLE
        cost_1 = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        cost_2 = (2000 / 1_000_000) * 2.5 + (1000 / 1_000_000) * 10.0
        cost_3 = (500 / 1_000_000) * 3.0 + (0 / 1_000_000) * 15.0
        expected_total = cost_1 + cost_2 + cost_3

        assert m.total_cost == pytest.approx(expected_total)
        assert m.avg_cost_per_call == pytest.approx(expected_total / 3)
        assert m.avg_tokens_per_call == pytest.approx(m.total_tokens / 3)
        assert m.cost_per_1k_tokens == pytest.approx(expected_total / m.total_tokens * 1000)

        # by_provider
        assert m.cost_by_provider["anthropic"] == pytest.approx(cost_1 + cost_3)
        assert m.cost_by_provider["openai"] == pytest.approx(cost_2)
        assert m.calls_by_provider["anthropic"] == 2
        assert m.calls_by_provider["openai"] == 1

        # by_model
        assert m.cost_by_model["claude-sonnet-4"] == pytest.approx(cost_1 + cost_3)
        assert m.cost_by_model["gpt-4o"] == pytest.approx(cost_2)
        assert m.calls_by_model["claude-sonnet-4"] == 2
        assert m.calls_by_model["gpt-4o"] == 1

    # ---------- get_efficiency_metrics ----------

    def test_get_efficiency_metrics_basic(self):
        """get_efficiency_metrics returns correct cost_per_success and success_rate."""
        tracker = CostTracker(enable_tracking=True)
        tracker.record_llm_call(
            provider="anthropic",
            model="claude-sonnet-4",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            success=True,
        )
        # total_cost = 3.0 + 15.0 = 18.0

        eff = tracker.get_efficiency_metrics(success_count=10)
        assert eff["cost_per_success"] == pytest.approx(18.0 / 10)
        assert eff["success_count"] == 10
        assert eff["total_cost"] == pytest.approx(18.0)
        assert eff["success_rate"] == pytest.approx(10.0 / 1)  # 10 successes / 1 call

    def test_get_efficiency_metrics_efficiency_score(self):
        """get_efficiency_metrics returns positive efficiency_score."""
        tracker = CostTracker(enable_tracking=True)
        tracker.record_llm_call(
            provider="anthropic",
            model="claude-sonnet-4",
            input_tokens=1000,
            output_tokens=500,
            success=True,
        )
        eff = tracker.get_efficiency_metrics(success_count=5)
        assert eff["efficiency_score"] > 0

    def test_get_efficiency_metrics_zero_success(self):
        """get_efficiency_metrics with zero successes does not raise."""
        tracker = CostTracker(enable_tracking=True)
        tracker.record_llm_call(
            provider="anthropic",
            model="claude-sonnet-4",
            input_tokens=1000,
            output_tokens=500,
            success=False,
        )
        eff = tracker.get_efficiency_metrics(success_count=0)
        # cost_per_success uses max(success_count, 1)
        assert eff["cost_per_success"] == eff["total_cost"]
        assert eff["success_rate"] == 0.0

    # ---------- reset ----------

    def test_reset_clears_all_data(self):
        """reset clears calls and resets start_time."""
        tracker = CostTracker(enable_tracking=True)
        tracker.record_llm_call(
            provider="anthropic",
            model="claude-sonnet-4",
            input_tokens=1000,
            output_tokens=500,
            success=True,
        )
        assert len(tracker.calls) == 1

        old_start = tracker.start_time
        time.sleep(0.01)
        tracker.reset()

        assert len(tracker.calls) == 0
        assert tracker.start_time >= old_start

    def test_reset_metrics_are_zero_after(self):
        """After reset, get_metrics returns all zeros."""
        tracker = CostTracker(enable_tracking=True)
        tracker.record_llm_call(
            provider="anthropic",
            model="claude-sonnet-4",
            input_tokens=1000,
            output_tokens=500,
            success=True,
        )
        tracker.reset()
        m = tracker.get_metrics()
        assert m.total_cost == 0.0
        assert m.total_calls == 0

    # ---------- save_to_file / load_from_file ----------

    def test_save_and_load_roundtrip(self, tmp_path):
        """save_to_file and load_from_file roundtrip preserves data."""
        filepath = str(tmp_path / "cost_data.json")

        tracker = CostTracker(enable_tracking=True)
        tracker.record_llm_call(
            provider="anthropic",
            model="claude-sonnet-4",
            input_tokens=1000,
            output_tokens=500,
            success=True,
        )
        tracker.record_llm_call(
            provider="openai",
            model="gpt-4o",
            input_tokens=2000,
            output_tokens=800,
            success=False,
            error="timeout",
            duration=3.2,
        )
        original_metrics = tracker.get_metrics()

        tracker.save_to_file(filepath)

        # Load into a fresh tracker
        tracker2 = CostTracker(enable_tracking=True)
        tracker2.load_from_file(filepath)

        assert len(tracker2.calls) == 2
        loaded_metrics = tracker2.get_metrics()
        assert loaded_metrics.total_cost == pytest.approx(original_metrics.total_cost)
        assert loaded_metrics.total_calls == original_metrics.total_calls
        assert loaded_metrics.total_input_tokens == original_metrics.total_input_tokens
        assert loaded_metrics.total_output_tokens == original_metrics.total_output_tokens

    def test_save_creates_parent_directories(self, tmp_path):
        """save_to_file creates parent directories if they do not exist."""
        filepath = str(tmp_path / "nested" / "dir" / "cost.json")
        tracker = CostTracker(enable_tracking=True)
        tracker.record_llm_call(
            provider="anthropic",
            model="claude-sonnet-4",
            input_tokens=100,
            output_tokens=50,
            success=True,
        )
        tracker.save_to_file(filepath)
        assert (tmp_path / "nested" / "dir" / "cost.json").exists()

    def test_save_file_is_valid_json(self, tmp_path):
        """save_to_file writes valid JSON with expected keys."""
        filepath = str(tmp_path / "cost.json")
        tracker = CostTracker(enable_tracking=True)
        tracker.record_llm_call(
            provider="anthropic",
            model="claude-sonnet-4",
            input_tokens=100,
            output_tokens=50,
            success=True,
        )
        tracker.save_to_file(filepath)

        with open(filepath, "r") as f:
            data = json.load(f)
        assert "start_time" in data
        assert "calls" in data
        assert "metrics" in data
        assert len(data["calls"]) == 1

    def test_load_from_nonexistent_file(self, tmp_path):
        """load_from_file with nonexistent path does not raise."""
        tracker = CostTracker(enable_tracking=True)
        tracker.load_from_file(str(tmp_path / "nonexistent.json"))
        assert len(tracker.calls) == 0

    def test_load_preserves_call_fields(self, tmp_path):
        """load_from_file preserves all fields of each LLMCallRecord."""
        filepath = str(tmp_path / "cost.json")
        tracker = CostTracker(enable_tracking=True)
        tracker.record_llm_call(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=300,
            output_tokens=150,
            success=False,
            error="rate_limit",
            duration=1.2,
        )
        tracker.save_to_file(filepath)

        tracker2 = CostTracker(enable_tracking=True)
        tracker2.load_from_file(filepath)

        loaded = tracker2.calls[0]
        assert loaded.provider == "openai"
        assert loaded.model == "gpt-4o-mini"
        assert loaded.input_tokens == 300
        assert loaded.output_tokens == 150
        assert loaded.success is False
        assert loaded.error == "rate_limit"
        assert loaded.duration == 1.2


# ============================================================================
# 5. MonitoringFramework
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_MONITORING_FRAMEWORK, reason="monitoring_framework not importable")
class TestMonitoringFramework:
    """Tests for MonitoringFramework and ExecutionStatus."""

    def test_execution_status_enum_values(self):
        """ExecutionStatus enum has expected members."""
        assert ExecutionStatus.SUCCESS.value == "success"
        assert ExecutionStatus.FAILURE.value == "failure"
        assert ExecutionStatus.TIMEOUT.value == "timeout"
        assert ExecutionStatus.ERROR.value == "error"
        assert ExecutionStatus.CANCELLED.value == "cancelled"

    def test_init_defaults(self):
        """MonitoringFramework initializes with monitoring enabled and empty executions."""
        monitor = MonitoringFramework()
        assert monitor.enable_monitoring is True
        assert monitor.executions == []

    def test_start_execution_returns_metrics(self):
        """start_execution returns an ExecutionMetrics instance and stores it."""
        monitor = MonitoringFramework()
        em = monitor.start_execution("test-agent", task_id="task-1")
        assert isinstance(em, ExecutionMetrics)
        assert em.agent_name == "test-agent"
        assert em.task_id == "task-1"
        assert len(monitor.executions) == 1

    def test_start_execution_with_metadata(self):
        """start_execution stores metadata dict."""
        monitor = MonitoringFramework()
        em = monitor.start_execution("agent-a", metadata={"key": "val"})
        assert em.metadata == {"key": "val"}

    def test_finish_execution_sets_status_and_duration(self):
        """finish_execution sets status, end_time, and duration."""
        monitor = MonitoringFramework()
        em = monitor.start_execution("agent-a")
        time.sleep(0.01)
        monitor.finish_execution(em, status=ExecutionStatus.SUCCESS)

        assert em.status == ExecutionStatus.SUCCESS
        assert em.end_time is not None
        assert em.duration > 0

    def test_finish_execution_with_error(self):
        """finish_execution stores error message and failure status."""
        monitor = MonitoringFramework()
        em = monitor.start_execution("agent-a")
        monitor.finish_execution(
            em,
            status=ExecutionStatus.FAILURE,
            error="Something went wrong",
            input_tokens=100,
            output_tokens=50,
            cost=0.01,
        )
        assert em.status == ExecutionStatus.FAILURE
        assert em.error == "Something went wrong"
        assert em.input_tokens == 100
        assert em.output_tokens == 50
        assert em.cost == 0.01

    def test_finish_execution_disabled_monitoring(self):
        """finish_execution is a no-op when monitoring is disabled."""
        monitor = MonitoringFramework(enable_monitoring=False)
        em = monitor.start_execution("agent-a")
        monitor.finish_execution(em, status=ExecutionStatus.SUCCESS)
        # When disabled, finish_execution returns early; end_time stays None
        assert em.end_time is None

    def test_get_performance_metrics_empty(self):
        """get_performance_metrics on empty framework returns zeros."""
        monitor = MonitoringFramework()
        pm = monitor.get_performance_metrics()
        assert pm.total_executions == 0
        assert pm.successful_executions == 0
        assert pm.success_rate == 0.0
        assert pm.executions_by_agent == {}

    def test_get_performance_metrics_after_executions(self):
        """get_performance_metrics returns correct aggregation after executions."""
        monitor = MonitoringFramework()

        em1 = monitor.start_execution("agent-a", task_id="t1")
        monitor.finish_execution(em1, status=ExecutionStatus.SUCCESS, cost=0.01)

        em2 = monitor.start_execution("agent-b", task_id="t2")
        monitor.finish_execution(
            em2,
            status=ExecutionStatus.FAILURE,
            error="Err:detail",
            cost=0.02,
        )

        em3 = monitor.start_execution("agent-a", task_id="t3")
        monitor.finish_execution(em3, status=ExecutionStatus.SUCCESS, cost=0.03)

        pm = monitor.get_performance_metrics()
        assert pm.total_executions == 3
        assert pm.successful_executions == 2
        assert pm.failed_executions == 1
        assert pm.total_cost == pytest.approx(0.06)
        assert pm.success_rate == pytest.approx(2.0 / 3)
        assert pm.error_rate == pytest.approx(1.0 / 3)
        assert pm.executions_by_agent["agent-a"] == 2
        assert pm.executions_by_agent["agent-b"] == 1

    def test_get_performance_metrics_unfinished_ignored(self):
        """get_performance_metrics ignores unfinished executions for aggregation."""
        monitor = MonitoringFramework()
        em1 = monitor.start_execution("agent-a")
        monitor.finish_execution(em1, status=ExecutionStatus.SUCCESS)

        # Start but do NOT finish
        monitor.start_execution("agent-b")

        pm = monitor.get_performance_metrics()
        # Only the finished one counts in aggregation
        assert pm.total_executions == 1
        assert pm.successful_executions == 1


# ============================================================================
# 6. EfficiencyMetrics
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_EFFICIENCY_METRICS, reason="efficiency_metrics not importable")
class TestEfficiencyMetrics:
    """Tests for EfficiencyMetrics static methods."""

    def _make_cost_metrics(self, total_cost=0.10, total_calls=10, **kwargs):
        """Create a CostMetrics instance for testing."""
        defaults = dict(
            total_cost=total_cost,
            total_input_tokens=5000,
            total_output_tokens=2000,
            total_tokens=7000,
            total_calls=total_calls,
            successful_calls=8,
            failed_calls=2,
            avg_cost_per_call=total_cost / max(total_calls, 1),
            avg_tokens_per_call=700.0,
            cost_per_1k_tokens=0.01,
            cost_by_provider={},
            cost_by_model={},
            calls_by_provider={},
            calls_by_model={},
        )
        defaults.update(kwargs)
        return CostMetrics(**defaults)

    def test_calculate_efficiency_basic(self):
        """calculate_efficiency returns an EfficiencyReport with correct fields."""
        cm = self._make_cost_metrics(total_cost=0.10, total_calls=10)
        report = EfficiencyMetrics.calculate_efficiency(cm, success_count=8)

        assert isinstance(report, EfficiencyReport)
        assert report.cost_per_success == pytest.approx(0.10 / 8)
        assert report.success_rate == pytest.approx(8 / 10)
        assert report.total_cost == pytest.approx(0.10)
        assert report.success_count == 8
        assert report.total_attempts == 10

    def test_calculate_efficiency_with_baseline(self):
        """calculate_efficiency with baseline_cost_per_success includes cost_reduction_potential."""
        cm = self._make_cost_metrics(total_cost=0.05, total_calls=10)
        report = EfficiencyMetrics.calculate_efficiency(
            cm, success_count=10, baseline_cost_per_success=0.01
        )
        # cost_per_success = 0.05 / 10 = 0.005
        # cost_reduction = ((0.01 - 0.005) / 0.01) * 100 = 50.0
        assert report.cost_per_success == pytest.approx(0.005)
        assert report.cost_reduction_potential == pytest.approx(50.0)

    def test_calculate_efficiency_no_baseline_no_reduction(self):
        """calculate_efficiency without baseline has cost_reduction_potential=None."""
        cm = self._make_cost_metrics(total_cost=0.10, total_calls=10)
        report = EfficiencyMetrics.calculate_efficiency(cm, success_count=5)
        assert report.cost_reduction_potential is None

    def test_calculate_efficiency_zero_success(self):
        """calculate_efficiency with zero successes does not raise."""
        cm = self._make_cost_metrics(total_cost=0.10, total_calls=5)
        report = EfficiencyMetrics.calculate_efficiency(cm, success_count=0)
        # max(success_count, 1) => 1
        assert report.cost_per_success == pytest.approx(0.10)
        assert report.success_rate == 0.0

    def test_calculate_efficiency_custom_total_attempts(self):
        """calculate_efficiency respects explicit total_attempts override."""
        cm = self._make_cost_metrics(total_cost=0.10, total_calls=10)
        report = EfficiencyMetrics.calculate_efficiency(
            cm,
            success_count=5,
            total_attempts=20,
        )
        assert report.total_attempts == 20
        assert report.success_rate == pytest.approx(5 / 20)

    def test_calculate_efficiency_efficiency_score_bounded(self):
        """efficiency_score is bounded to [0, 1] range."""
        cm = self._make_cost_metrics(total_cost=0.001, total_calls=1)
        report = EfficiencyMetrics.calculate_efficiency(cm, success_count=1)
        # cost_per_success = 0.001, baseline=1.0
        # efficiency_score = min(1.0, 1.0 / 0.001) => 1.0
        assert report.efficiency_score <= 1.0

    def test_calculate_efficiency_to_dict(self):
        """EfficiencyReport.to_dict returns all expected keys."""
        cm = self._make_cost_metrics(total_cost=0.10, total_calls=10)
        report = EfficiencyMetrics.calculate_efficiency(cm, success_count=5)
        d = report.to_dict()
        expected_keys = {
            "cost_per_success",
            "efficiency_score",
            "success_rate",
            "total_cost",
            "success_count",
            "total_attempts",
            "cost_reduction_potential",
            "performance_retention",
        }
        assert set(d.keys()) == expected_keys

    # ---------- compare_efficiency ----------

    def test_compare_efficiency_cost_reduction(self):
        """compare_efficiency calculates correct cost_reduction_percent."""
        result = EfficiencyMetrics.compare_efficiency(
            current_cost_per_success=0.005,
            baseline_cost_per_success=0.010,
            current_performance=0.85,
            baseline_performance=0.90,
        )
        # cost_reduction = ((0.01 - 0.005) / 0.01) * 100 = 50.0
        assert result["cost_reduction_percent"] == pytest.approx(50.0)

    def test_compare_efficiency_performance_retention(self):
        """compare_efficiency calculates correct performance_retention_percent."""
        result = EfficiencyMetrics.compare_efficiency(
            current_cost_per_success=0.005,
            baseline_cost_per_success=0.010,
            current_performance=0.85,
            baseline_performance=0.90,
        )
        # performance_retention = (0.85 / 0.90) * 100
        assert result["performance_retention_percent"] == pytest.approx((0.85 / 0.90) * 100)

    def test_compare_efficiency_improvement(self):
        """compare_efficiency calculates efficiency_improvement_percent."""
        result = EfficiencyMetrics.compare_efficiency(
            current_cost_per_success=0.005,
            baseline_cost_per_success=0.010,
            current_performance=0.90,
            baseline_performance=0.90,
        )
        # current_efficiency = 0.90 / 0.005 = 180
        # baseline_efficiency = 0.90 / 0.010 = 90
        # improvement = ((180 - 90) / 90) * 100 = 100.0
        assert result["efficiency_improvement_percent"] == pytest.approx(100.0)

    def test_compare_efficiency_negative_cost_reduction(self):
        """compare_efficiency can return negative cost_reduction (cost increased)."""
        result = EfficiencyMetrics.compare_efficiency(
            current_cost_per_success=0.020,
            baseline_cost_per_success=0.010,
            current_performance=0.90,
            baseline_performance=0.90,
        )
        # cost_reduction = ((0.01 - 0.02) / 0.01) * 100 = -100.0
        assert result["cost_reduction_percent"] == pytest.approx(-100.0)

    def test_compare_efficiency_zero_baseline_performance(self):
        """compare_efficiency raises ZeroDivisionError when baseline_performance is zero."""
        with pytest.raises(ZeroDivisionError):
            EfficiencyMetrics.compare_efficiency(
                current_cost_per_success=0.005,
                baseline_cost_per_success=0.010,
                current_performance=0.85,
                baseline_performance=0.0,
            )

    def test_compare_efficiency_returns_all_keys(self):
        """compare_efficiency returns dict with all expected keys."""
        result = EfficiencyMetrics.compare_efficiency(
            current_cost_per_success=0.005,
            baseline_cost_per_success=0.010,
            current_performance=0.85,
            baseline_performance=0.90,
        )
        expected_keys = {
            "cost_reduction_percent",
            "performance_retention_percent",
            "efficiency_improvement_percent",
            "cost_per_success_improvement",
        }
        assert set(result.keys()) == expected_keys

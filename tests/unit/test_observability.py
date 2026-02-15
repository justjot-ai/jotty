"""
Tests for Observability Metrics Module
=======================================
Tests for ExecutionRecord, AgentMetrics, MetricsCollector, and singleton functions.
All tests are mocked and offline — no LLM calls or API keys required.
"""

import math
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from Jotty.core.infrastructure.monitoring.observability.metrics import (
    AgentMetrics,
    ExecutionRecord,
    MetricsCollector,
    get_metrics,
    reset_metrics,
)

# =============================================================================
# ExecutionRecord Tests
# =============================================================================


@pytest.mark.unit
class TestExecutionRecord:
    """Tests for the ExecutionRecord dataclass."""

    def test_creation_with_required_fields(self):
        """ExecutionRecord can be created with only required fields."""
        record = ExecutionRecord(
            agent_name="auto",
            task_type="research",
            duration_s=5.0,
            success=True,
        )
        assert record.agent_name == "auto"
        assert record.task_type == "research"
        assert record.duration_s == 5.0
        assert record.success is True

    def test_default_timestamp_is_set(self):
        """ExecutionRecord timestamp defaults to current time."""
        before = time.time()
        record = ExecutionRecord(
            agent_name="auto",
            task_type="research",
            duration_s=1.0,
            success=True,
        )
        after = time.time()
        assert before <= record.timestamp <= after

    def test_default_token_counts_are_zero(self):
        """input_tokens and output_tokens default to zero."""
        record = ExecutionRecord(
            agent_name="auto",
            task_type="analysis",
            duration_s=2.0,
            success=True,
        )
        assert record.input_tokens == 0
        assert record.output_tokens == 0

    def test_default_cost_is_zero(self):
        """cost_usd defaults to 0.0."""
        record = ExecutionRecord(
            agent_name="auto",
            task_type="analysis",
            duration_s=2.0,
            success=True,
        )
        assert record.cost_usd == 0.0

    def test_default_llm_calls_is_zero(self):
        """llm_calls defaults to 0."""
        record = ExecutionRecord(
            agent_name="auto",
            task_type="analysis",
            duration_s=2.0,
            success=True,
        )
        assert record.llm_calls == 0

    def test_default_error_is_none(self):
        """error defaults to None."""
        record = ExecutionRecord(
            agent_name="auto",
            task_type="analysis",
            duration_s=2.0,
            success=False,
        )
        assert record.error is None

    def test_default_metadata_is_empty_dict(self):
        """metadata defaults to an empty dict."""
        record = ExecutionRecord(
            agent_name="auto",
            task_type="analysis",
            duration_s=2.0,
            success=True,
        )
        assert record.metadata == {}
        assert isinstance(record.metadata, dict)

    def test_creation_with_all_fields(self):
        """ExecutionRecord can be created with all fields specified."""
        record = ExecutionRecord(
            agent_name="analyst",
            task_type="creation",
            duration_s=12.5,
            success=False,
            timestamp=1000.0,
            input_tokens=500,
            output_tokens=1000,
            cost_usd=0.05,
            llm_calls=3,
            error="timeout",
            metadata={"model": "claude-opus-4-20250514"},
        )
        assert record.agent_name == "analyst"
        assert record.task_type == "creation"
        assert record.duration_s == 12.5
        assert record.success is False
        assert record.timestamp == 1000.0
        assert record.input_tokens == 500
        assert record.output_tokens == 1000
        assert record.cost_usd == 0.05
        assert record.llm_calls == 3
        assert record.error == "timeout"
        assert record.metadata == {"model": "claude-opus-4-20250514"}

    def test_metadata_instances_are_independent(self):
        """Each ExecutionRecord gets its own metadata dict instance."""
        r1 = ExecutionRecord(agent_name="a", task_type="t", duration_s=1.0, success=True)
        r2 = ExecutionRecord(agent_name="b", task_type="t", duration_s=1.0, success=True)
        r1.metadata["key"] = "value"
        assert "key" not in r2.metadata


# =============================================================================
# AgentMetrics Tests
# =============================================================================


@pytest.mark.unit
class TestAgentMetrics:
    """Tests for the AgentMetrics dataclass and its computed properties."""

    def _make_metrics(self, **kwargs):
        """Helper to create an AgentMetrics with overrides."""
        defaults = {
            "agent_name": "test_agent",
            "total_executions": 0,
            "successful": 0,
            "failed": 0,
            "total_duration_s": 0.0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "total_llm_calls": 0,
            "durations": [],
        }
        defaults.update(kwargs)
        return AgentMetrics(**defaults)

    def test_success_rate_zero_executions(self):
        """success_rate returns 0.0 when no executions recorded."""
        am = self._make_metrics(total_executions=0, successful=0)
        assert am.success_rate == 0.0

    def test_success_rate_all_successful(self):
        """success_rate returns 1.0 when all executions succeed."""
        am = self._make_metrics(total_executions=10, successful=10)
        assert am.success_rate == 1.0

    def test_success_rate_partial(self):
        """success_rate correctly computes a fractional rate."""
        am = self._make_metrics(total_executions=4, successful=3)
        assert am.success_rate == 0.75

    def test_success_rate_all_failed(self):
        """success_rate returns 0.0 when all executions fail."""
        am = self._make_metrics(total_executions=5, successful=0, failed=5)
        assert am.success_rate == 0.0

    def test_avg_duration_empty(self):
        """avg_duration_s returns 0.0 when no durations recorded."""
        am = self._make_metrics(durations=[])
        assert am.avg_duration_s == 0.0

    def test_avg_duration_single_value(self):
        """avg_duration_s with a single duration returns that value."""
        am = self._make_metrics(durations=[5.0])
        assert am.avg_duration_s == 5.0

    def test_avg_duration_multiple_values(self):
        """avg_duration_s correctly averages multiple durations."""
        am = self._make_metrics(durations=[2.0, 4.0, 6.0])
        assert am.avg_duration_s == 4.0

    def test_avg_tokens_per_execution_zero_executions(self):
        """avg_tokens_per_execution returns 0.0 when no executions."""
        am = self._make_metrics(total_executions=0, total_tokens=0)
        assert am.avg_tokens_per_execution == 0.0

    def test_avg_tokens_per_execution_normal(self):
        """avg_tokens_per_execution divides total tokens by total executions."""
        am = self._make_metrics(total_executions=5, total_tokens=2500)
        assert am.avg_tokens_per_execution == 500.0

    def test_avg_cost_per_execution_zero_executions(self):
        """avg_cost_per_execution returns 0.0 when no executions."""
        am = self._make_metrics(total_executions=0, total_cost_usd=0.0)
        assert am.avg_cost_per_execution == 0.0

    def test_avg_cost_per_execution_normal(self):
        """avg_cost_per_execution divides total cost by total executions."""
        am = self._make_metrics(total_executions=4, total_cost_usd=0.20)
        assert am.avg_cost_per_execution == pytest.approx(0.05)

    def test_percentile_empty_list(self):
        """_percentile returns 0.0 for an empty list."""
        am = self._make_metrics()
        assert am._percentile([], 50) == 0.0

    def test_percentile_single_value(self):
        """_percentile returns the single value for any percentile."""
        am = self._make_metrics()
        assert am._percentile([7.0], 50) == 7.0
        assert am._percentile([7.0], 95) == 7.0
        assert am._percentile([7.0], 99) == 7.0

    def test_percentile_median_odd_count(self):
        """_percentile correctly computes the median of an odd-length list."""
        am = self._make_metrics()
        result = am._percentile([1.0, 3.0, 5.0], 50)
        assert result == 3.0

    def test_percentile_median_even_count(self):
        """_percentile interpolates the median for an even-length list."""
        am = self._make_metrics()
        result = am._percentile([1.0, 2.0, 3.0, 4.0], 50)
        assert result == pytest.approx(2.5)

    def test_percentile_unsorted_input(self):
        """_percentile sorts values internally, so unsorted input works."""
        am = self._make_metrics()
        result = am._percentile([5.0, 1.0, 3.0], 50)
        assert result == 3.0

    def test_p50_duration(self):
        """p50_duration_s returns the median of durations."""
        am = self._make_metrics(durations=[1.0, 2.0, 3.0, 4.0, 5.0])
        assert am.p50_duration_s == 3.0

    def test_p95_duration(self):
        """p95_duration_s returns the 95th percentile of durations."""
        durations = list(range(1, 101))  # 1 to 100
        am = self._make_metrics(durations=[float(d) for d in durations])
        # For 100 values, p95 index = 0.95 * 99 = 94.05
        # Interpolation between sorted[94]=95 and sorted[95]=96
        expected = 95.0 * (1 - 0.05) + 96.0 * 0.05
        assert am.p95_duration_s == pytest.approx(expected, rel=1e-3)

    def test_p99_duration(self):
        """p99_duration_s returns the 99th percentile of durations."""
        durations = list(range(1, 101))
        am = self._make_metrics(durations=[float(d) for d in durations])
        # p99 index = 0.99 * 99 = 98.01
        expected = 99.0 * (1 - 0.01) + 100.0 * 0.01
        assert am.p99_duration_s == pytest.approx(expected, rel=1e-3)

    def test_percentile_properties_empty_durations(self):
        """Percentile properties return 0.0 for empty durations."""
        am = self._make_metrics(durations=[])
        assert am.p50_duration_s == 0.0
        assert am.p95_duration_s == 0.0
        assert am.p99_duration_s == 0.0

    def test_to_dict_structure(self):
        """to_dict returns a dict with the expected top-level keys and nested structure."""
        am = self._make_metrics(
            agent_name="analyst",
            total_executions=10,
            successful=8,
            failed=2,
            total_tokens=5000,
            total_cost_usd=0.50,
            total_llm_calls=20,
            durations=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        d = am.to_dict()
        assert d["agent_name"] == "analyst"
        assert d["total_executions"] == 10
        assert d["successful"] == 8
        assert d["failed"] == 2
        assert d["success_rate"] == round(0.8, 4)
        assert "latency" in d
        assert "avg_s" in d["latency"]
        assert "p50_s" in d["latency"]
        assert "p95_s" in d["latency"]
        assert "p99_s" in d["latency"]
        assert "tokens" in d
        assert d["tokens"]["total"] == 5000
        assert d["tokens"]["avg_per_execution"] == 500.0
        assert "cost" in d
        assert d["cost"]["total_usd"] == round(0.50, 6)
        assert d["cost"]["avg_per_execution_usd"] == round(0.05, 6)
        assert d["llm_calls"] == 20

    def test_to_dict_latency_values(self):
        """to_dict latency values match the computed percentiles (rounded)."""
        am = self._make_metrics(durations=[1.0, 2.0, 3.0, 4.0, 5.0])
        d = am.to_dict()
        assert d["latency"]["avg_s"] == round(3.0, 3)
        assert d["latency"]["p50_s"] == round(am.p50_duration_s, 3)
        assert d["latency"]["p95_s"] == round(am.p95_duration_s, 3)
        assert d["latency"]["p99_s"] == round(am.p99_duration_s, 3)

    def test_to_dict_zero_executions(self):
        """to_dict handles zero executions gracefully."""
        am = self._make_metrics()
        d = am.to_dict()
        assert d["total_executions"] == 0
        assert d["success_rate"] == 0.0
        assert d["latency"]["avg_s"] == 0.0
        assert d["tokens"]["avg_per_execution"] == 0.0
        assert d["cost"]["avg_per_execution_usd"] == 0.0


# =============================================================================
# MetricsCollector Tests
# =============================================================================


@pytest.mark.unit
class TestMetricsCollectorRecording:
    """Tests for MetricsCollector.record_execution and basic retrieval."""

    def test_record_single_execution(self):
        """record_execution stores a record and updates agent metrics."""
        mc = MetricsCollector()
        mc.record_execution("auto", "research", 5.0, True, input_tokens=100, output_tokens=200)
        am = mc.get_agent_metrics("auto")
        assert am is not None
        assert am.total_executions == 1
        assert am.successful == 1
        assert am.failed == 0
        assert am.total_tokens == 300

    def test_record_failed_execution(self):
        """record_execution correctly tracks a failed execution."""
        mc = MetricsCollector()
        mc.record_execution("auto", "research", 3.0, False, error="timeout")
        am = mc.get_agent_metrics("auto")
        assert am.total_executions == 1
        assert am.successful == 0
        assert am.failed == 1

    def test_record_multiple_executions_same_agent(self):
        """Multiple recordings for the same agent accumulate correctly."""
        mc = MetricsCollector()
        mc.record_execution(
            "auto",
            "research",
            5.0,
            True,
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.01,
            llm_calls=2,
        )
        mc.record_execution(
            "auto",
            "analysis",
            3.0,
            True,
            input_tokens=50,
            output_tokens=100,
            cost_usd=0.005,
            llm_calls=1,
        )
        mc.record_execution(
            "auto",
            "research",
            7.0,
            False,
            input_tokens=80,
            output_tokens=0,
            cost_usd=0.008,
            llm_calls=1,
        )

        am = mc.get_agent_metrics("auto")
        assert am.total_executions == 3
        assert am.successful == 2
        assert am.failed == 1
        assert am.total_tokens == 530
        assert am.total_cost_usd == pytest.approx(0.023)
        assert am.total_llm_calls == 4
        assert len(am.durations) == 3

    def test_record_multiple_agents(self):
        """Records for different agents are tracked independently."""
        mc = MetricsCollector()
        mc.record_execution("auto", "research", 5.0, True)
        mc.record_execution("analyst", "analysis", 3.0, True)

        auto_m = mc.get_agent_metrics("auto")
        analyst_m = mc.get_agent_metrics("analyst")
        assert auto_m.total_executions == 1
        assert analyst_m.total_executions == 1
        assert auto_m.agent_name == "auto"
        assert analyst_m.agent_name == "analyst"

    def test_get_agent_metrics_unknown_agent(self):
        """get_agent_metrics returns None for an unknown agent."""
        mc = MetricsCollector()
        assert mc.get_agent_metrics("nonexistent") is None

    def test_record_execution_with_metadata(self):
        """record_execution stores metadata correctly."""
        mc = MetricsCollector()
        mc.record_execution(
            "auto", "research", 5.0, True, metadata={"model": "claude-opus-4-20250514"}
        )
        # Verify by checking internal records directly
        assert len(mc._records) == 1
        assert mc._records[0].metadata == {"model": "claude-opus-4-20250514"}

    def test_record_execution_none_metadata_becomes_empty_dict(self):
        """When metadata is None, it is stored as an empty dict."""
        mc = MetricsCollector()
        mc.record_execution("auto", "research", 5.0, True, metadata=None)
        assert mc._records[0].metadata == {}

    def test_duration_tracked_in_agent_metrics(self):
        """Durations are appended to agent metrics correctly."""
        mc = MetricsCollector()
        mc.record_execution("auto", "t1", 2.5, True)
        mc.record_execution("auto", "t2", 4.0, True)
        am = mc.get_agent_metrics("auto")
        assert am.durations == [2.5, 4.0]
        assert am.total_duration_s == pytest.approx(6.5)


@pytest.mark.unit
class TestMetricsCollectorGetAllAgentMetrics:
    """Tests for MetricsCollector.get_all_agent_metrics."""

    def test_empty_collector(self):
        """get_all_agent_metrics returns an empty dict when nothing recorded."""
        mc = MetricsCollector()
        assert mc.get_all_agent_metrics() == {}

    def test_returns_all_agents(self):
        """get_all_agent_metrics returns metrics for every recorded agent."""
        mc = MetricsCollector()
        mc.record_execution("auto", "research", 1.0, True)
        mc.record_execution("analyst", "analysis", 2.0, True)
        mc.record_execution("coder", "coding", 3.0, False)

        all_metrics = mc.get_all_agent_metrics()
        assert set(all_metrics.keys()) == {"auto", "analyst", "coder"}
        for name, am in all_metrics.items():
            assert isinstance(am, AgentMetrics)
            assert am.agent_name == name

    def test_returns_copy(self):
        """get_all_agent_metrics returns a copy of the internal dict."""
        mc = MetricsCollector()
        mc.record_execution("auto", "research", 1.0, True)
        result = mc.get_all_agent_metrics()
        result["injected"] = AgentMetrics(agent_name="injected")
        # Internal state should not be affected
        assert "injected" not in mc.get_all_agent_metrics()


@pytest.mark.unit
class TestMetricsCollectorSummary:
    """Tests for MetricsCollector.get_summary."""

    def test_summary_structure_empty(self):
        """get_summary returns the expected structure even when empty."""
        mc = MetricsCollector()
        summary = mc.get_summary()
        assert "session" in summary
        assert "global" in summary
        assert "per_agent" in summary
        assert "task_types" in summary

    def test_summary_global_counts(self):
        """get_summary global section has correct counts."""
        mc = MetricsCollector()
        mc.record_execution(
            "auto",
            "research",
            5.0,
            True,
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.01,
            llm_calls=2,
        )
        mc.record_execution("auto", "research", 3.0, False, error="fail")

        summary = mc.get_summary()
        g = summary["global"]
        assert g["total_executions"] == 2
        assert g["successful"] == 1
        assert g["failed"] == 1
        assert g["success_rate"] == 0.5
        assert g["total_tokens"] == 300
        assert g["total_cost_usd"] == round(0.01, 6)
        assert g["total_llm_calls"] == 2

    def test_summary_global_latency_percentiles(self):
        """get_summary global latency percentiles are computed correctly."""
        mc = MetricsCollector()
        for d in [1.0, 2.0, 3.0, 4.0, 5.0]:
            mc.record_execution("auto", "research", d, True)

        summary = mc.get_summary()
        latency = summary["global"]["latency"]
        assert "p50_s" in latency
        assert "p95_s" in latency
        assert "p99_s" in latency
        assert latency["p50_s"] == round(3.0, 3)

    def test_summary_per_agent(self):
        """get_summary per_agent section contains all agents with to_dict format."""
        mc = MetricsCollector()
        mc.record_execution("auto", "research", 5.0, True)
        mc.record_execution("analyst", "analysis", 3.0, True)

        summary = mc.get_summary()
        assert "auto" in summary["per_agent"]
        assert "analyst" in summary["per_agent"]
        assert summary["per_agent"]["auto"]["agent_name"] == "auto"

    def test_summary_task_types(self):
        """get_summary task_types section tracks task type counts."""
        mc = MetricsCollector()
        mc.record_execution("auto", "research", 1.0, True)
        mc.record_execution("auto", "research", 2.0, True)
        mc.record_execution("auto", "analysis", 3.0, True)

        summary = mc.get_summary()
        assert summary["task_types"]["research"] == 2
        assert summary["task_types"]["analysis"] == 1

    def test_summary_session_has_duration_and_start(self):
        """get_summary session section includes duration_s and start_time."""
        mc = MetricsCollector()
        summary = mc.get_summary()
        assert "duration_s" in summary["session"]
        assert "start_time" in summary["session"]
        assert summary["session"]["duration_s"] >= 0

    def test_summary_success_rate_zero_when_empty(self):
        """get_summary success_rate is 0 when there are no executions."""
        mc = MetricsCollector()
        summary = mc.get_summary()
        assert summary["global"]["success_rate"] == 0


@pytest.mark.unit
class TestMetricsCollectorCostBreakdown:
    """Tests for MetricsCollector.get_cost_breakdown."""

    def test_cost_breakdown_empty(self):
        """get_cost_breakdown returns structure with zero cost when empty."""
        mc = MetricsCollector()
        cb = mc.get_cost_breakdown()
        assert cb["total_cost_usd"] == 0.0
        assert cb["by_agent"] == {}
        assert cb["by_model"] == {}

    def test_cost_breakdown_by_agent(self):
        """get_cost_breakdown correctly groups costs by agent."""
        mc = MetricsCollector()
        mc.record_execution(
            "auto",
            "research",
            1.0,
            True,
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.01,
            llm_calls=1,
        )
        mc.record_execution(
            "analyst",
            "analysis",
            2.0,
            True,
            input_tokens=50,
            output_tokens=50,
            cost_usd=0.005,
            llm_calls=1,
        )
        mc.record_execution(
            "auto",
            "creation",
            3.0,
            True,
            input_tokens=200,
            output_tokens=300,
            cost_usd=0.02,
            llm_calls=2,
        )

        cb = mc.get_cost_breakdown()
        assert cb["total_cost_usd"] == pytest.approx(0.035, abs=1e-6)
        assert cb["by_agent"]["auto"]["cost_usd"] == pytest.approx(0.03, abs=1e-6)
        assert cb["by_agent"]["auto"]["tokens"] == 800  # 100+200+200+300
        assert cb["by_agent"]["auto"]["calls"] == 3
        assert cb["by_agent"]["analyst"]["cost_usd"] == pytest.approx(0.005, abs=1e-6)
        assert cb["by_agent"]["analyst"]["tokens"] == 100
        assert cb["by_agent"]["analyst"]["calls"] == 1

    def test_cost_breakdown_by_model(self):
        """get_cost_breakdown groups costs by model from metadata."""
        mc = MetricsCollector()
        mc.record_execution(
            "auto", "research", 1.0, True, cost_usd=0.01, metadata={"model": "claude"}
        )
        mc.record_execution(
            "auto", "research", 1.0, True, cost_usd=0.02, metadata={"model": "gpt-4"}
        )
        mc.record_execution(
            "auto", "research", 1.0, True, cost_usd=0.005, metadata={"model": "claude"}
        )

        cb = mc.get_cost_breakdown()
        assert cb["by_model"]["claude"] == pytest.approx(0.015, abs=1e-6)
        assert cb["by_model"]["gpt-4"] == pytest.approx(0.02, abs=1e-6)

    def test_cost_breakdown_unknown_model(self):
        """get_cost_breakdown uses 'unknown' when model is not in metadata."""
        mc = MetricsCollector()
        mc.record_execution("auto", "research", 1.0, True, cost_usd=0.01)

        cb = mc.get_cost_breakdown()
        assert "unknown" in cb["by_model"]
        assert cb["by_model"]["unknown"] == pytest.approx(0.01, abs=1e-6)


@pytest.mark.unit
class TestMetricsCollectorRecentErrors:
    """Tests for MetricsCollector.recent_errors."""

    def test_recent_errors_empty(self):
        """recent_errors returns empty list when no errors exist."""
        mc = MetricsCollector()
        assert mc.recent_errors() == []

    def test_recent_errors_no_failed_records(self):
        """recent_errors returns empty list when all executions succeeded."""
        mc = MetricsCollector()
        mc.record_execution("auto", "research", 1.0, True)
        mc.record_execution("auto", "analysis", 2.0, True)
        assert mc.recent_errors() == []

    def test_recent_errors_returns_failures(self):
        """recent_errors returns records that failed with an error message."""
        mc = MetricsCollector()
        mc.record_execution("auto", "research", 1.0, False, error="timeout")
        mc.record_execution("auto", "research", 2.0, True)
        mc.record_execution("analyst", "analysis", 3.0, False, error="rate_limit")

        errors = mc.recent_errors()
        assert len(errors) == 2
        # Most recent first (reversed order)
        assert errors[0]["error"] == "rate_limit"
        assert errors[1]["error"] == "timeout"

    def test_recent_errors_limit(self):
        """recent_errors respects the limit parameter."""
        mc = MetricsCollector()
        for i in range(20):
            mc.record_execution("auto", "research", 1.0, False, error=f"error_{i}")

        errors = mc.recent_errors(limit=5)
        assert len(errors) == 5

    def test_recent_errors_default_limit(self):
        """recent_errors defaults to limit=10."""
        mc = MetricsCollector()
        for i in range(15):
            mc.record_execution("auto", "research", 1.0, False, error=f"error_{i}")

        errors = mc.recent_errors()
        assert len(errors) == 10

    def test_recent_errors_structure(self):
        """recent_errors returns dicts with the expected keys."""
        mc = MetricsCollector()
        mc.record_execution("auto", "research", 5.5, False, error="something broke")

        errors = mc.recent_errors()
        assert len(errors) == 1
        err = errors[0]
        assert err["agent"] == "auto"
        assert err["task_type"] == "research"
        assert err["error"] == "something broke"
        assert "timestamp" in err
        assert err["duration_s"] == 5.5

    def test_recent_errors_excludes_failed_without_error_message(self):
        """recent_errors only includes records where success=False AND error is not None."""
        mc = MetricsCollector()
        mc.record_execution("auto", "research", 1.0, False)  # failed, no error string
        mc.record_execution("auto", "research", 2.0, False, error="has error")

        errors = mc.recent_errors()
        assert len(errors) == 1
        assert errors[0]["error"] == "has error"


@pytest.mark.unit
class TestMetricsCollectorReset:
    """Tests for MetricsCollector.reset."""

    def test_reset_clears_records(self):
        """reset clears all stored records."""
        mc = MetricsCollector()
        mc.record_execution(
            "auto",
            "research",
            1.0,
            True,
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.01,
            llm_calls=1,
        )
        mc.record_execution("analyst", "analysis", 2.0, False, error="fail")

        mc.reset()

        assert mc.get_agent_metrics("auto") is None
        assert mc.get_agent_metrics("analyst") is None
        assert mc.get_all_agent_metrics() == {}
        assert mc.recent_errors() == []

    def test_reset_clears_global_counters(self):
        """reset zeroes out global cost, tokens, and LLM call counters."""
        mc = MetricsCollector()
        mc.record_execution(
            "auto",
            "research",
            1.0,
            True,
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.01,
            llm_calls=2,
        )
        mc.reset()

        assert mc._total_cost_usd == 0.0
        assert mc._total_tokens == 0
        assert mc._total_llm_calls == 0

    def test_reset_clears_task_types(self):
        """reset clears task type counts."""
        mc = MetricsCollector()
        mc.record_execution("auto", "research", 1.0, True)
        mc.reset()

        summary = mc.get_summary()
        assert summary["task_types"] == {}

    def test_reset_resets_session_start(self):
        """reset updates the session start time."""
        mc = MetricsCollector()
        old_start = mc._session_start
        time.sleep(0.01)
        mc.reset()
        assert mc._session_start >= old_start

    def test_can_record_after_reset(self):
        """Collector works normally after reset."""
        mc = MetricsCollector()
        mc.record_execution("auto", "research", 1.0, True)
        mc.reset()
        mc.record_execution("analyst", "analysis", 2.0, True)

        assert mc.get_agent_metrics("auto") is None
        am = mc.get_agent_metrics("analyst")
        assert am is not None
        assert am.total_executions == 1


@pytest.mark.unit
class TestMetricsCollectorMaxHistory:
    """Tests for MetricsCollector max_history bounding."""

    def test_records_bounded_to_max_history(self):
        """Internal records list is trimmed when exceeding max_history."""
        mc = MetricsCollector(max_history=5)
        for i in range(10):
            mc.record_execution("auto", "research", float(i), True)

        assert len(mc._records) == 5
        # Should keep the most recent 5 (durations 5.0 through 9.0)
        assert mc._records[0].duration_s == 5.0
        assert mc._records[-1].duration_s == 9.0

    def test_agent_metrics_not_reset_by_bounding(self):
        """Agent-level aggregated metrics keep accumulating even when records are trimmed."""
        mc = MetricsCollector(max_history=3)
        for i in range(6):
            mc.record_execution("auto", "research", 1.0, True, input_tokens=10, output_tokens=10)

        am = mc.get_agent_metrics("auto")
        # All 6 executions are tracked in the agent metrics
        assert am.total_executions == 6
        assert am.total_tokens == 120  # 6 * 20

    def test_durations_bounded_to_1000(self):
        """Agent durations list is trimmed to 1000 entries."""
        mc = MetricsCollector(max_history=20000)
        for i in range(1050):
            mc.record_execution("auto", "research", float(i), True)

        am = mc.get_agent_metrics("auto")
        assert len(am.durations) == 1000
        # Should keep the most recent 1000
        assert am.durations[0] == 50.0
        assert am.durations[-1] == 1049.0

    def test_max_history_default(self):
        """Default max_history is 10000."""
        mc = MetricsCollector()
        assert mc._max_history == 10000

    def test_max_history_custom(self):
        """Custom max_history is respected."""
        mc = MetricsCollector(max_history=42)
        assert mc._max_history == 42


@pytest.mark.unit
class TestMetricsCollectorThreadSafety:
    """Tests that MetricsCollector uses a threading lock."""

    def test_has_lock(self):
        """MetricsCollector has a threading lock."""
        mc = MetricsCollector()
        assert hasattr(mc, "_lock")
        assert isinstance(mc._lock, type(MagicMock())) or hasattr(mc._lock, "acquire")

    def test_concurrent_recording(self):
        """Multiple threads can record without data corruption."""
        import threading

        mc = MetricsCollector()
        num_threads = 10
        records_per_thread = 100

        def record_work(agent_name):
            for _ in range(records_per_thread):
                mc.record_execution(agent_name, "task", 1.0, True, input_tokens=1, output_tokens=1)

        threads = [
            threading.Thread(target=record_work, args=(f"agent_{i}",)) for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        all_metrics = mc.get_all_agent_metrics()
        assert len(all_metrics) == num_threads
        total_execs = sum(am.total_executions for am in all_metrics.values())
        assert total_execs == num_threads * records_per_thread


# =============================================================================
# Singleton Tests
# =============================================================================


@pytest.mark.unit
class TestSingleton:
    """Tests for get_metrics() and reset_metrics() singleton functions."""

    def setup_method(self):
        """Reset the singleton before each test."""
        reset_metrics()

    def teardown_method(self):
        """Clean up the singleton after each test."""
        reset_metrics()

    def test_get_metrics_returns_collector(self):
        """get_metrics returns a MetricsCollector instance."""
        m = get_metrics()
        assert isinstance(m, MetricsCollector)

    def test_get_metrics_returns_same_instance(self):
        """get_metrics returns the same instance on repeated calls."""
        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2

    def test_reset_metrics_creates_new_instance(self):
        """After reset_metrics, get_metrics returns a new instance."""
        m1 = get_metrics()
        m1.record_execution("auto", "research", 1.0, True)
        reset_metrics()
        m2 = get_metrics()
        assert m2 is not m1
        assert m2.get_agent_metrics("auto") is None

    def test_reset_metrics_when_no_instance(self):
        """reset_metrics does not raise when no instance exists."""
        reset_metrics()
        reset_metrics()  # Should not raise

    def test_singleton_data_persists_across_calls(self):
        """Data recorded via the singleton persists across get_metrics() calls."""
        m = get_metrics()
        m.record_execution("auto", "research", 5.0, True)

        m2 = get_metrics()
        am = m2.get_agent_metrics("auto")
        assert am is not None
        assert am.total_executions == 1


# =============================================================================
# Edge Case Tests
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Edge case tests for metrics module."""

    def test_zero_duration_execution(self):
        """An execution with zero duration is recorded correctly."""
        mc = MetricsCollector()
        mc.record_execution("auto", "fast", 0.0, True)
        am = mc.get_agent_metrics("auto")
        assert am.avg_duration_s == 0.0
        assert am.p50_duration_s == 0.0

    def test_very_large_duration(self):
        """An execution with a very large duration is recorded correctly."""
        mc = MetricsCollector()
        mc.record_execution("auto", "slow", 99999.99, True)
        am = mc.get_agent_metrics("auto")
        assert am.avg_duration_s == 99999.99

    def test_all_failures_summary(self):
        """Summary correctly reflects 100% failure rate."""
        mc = MetricsCollector()
        for _ in range(5):
            mc.record_execution("auto", "failing_task", 1.0, False, error="always fails")

        summary = mc.get_summary()
        assert summary["global"]["successful"] == 0
        assert summary["global"]["failed"] == 5
        assert summary["global"]["success_rate"] == 0.0

    def test_cost_breakdown_with_zero_cost(self):
        """Cost breakdown handles zero-cost executions."""
        mc = MetricsCollector()
        mc.record_execution("auto", "free_task", 1.0, True, cost_usd=0.0)
        cb = mc.get_cost_breakdown()
        assert cb["total_cost_usd"] == 0.0
        assert cb["by_agent"]["auto"]["cost_usd"] == 0.0

    def test_summary_single_execution(self):
        """Summary works correctly for exactly one execution."""
        mc = MetricsCollector()
        mc.record_execution(
            "auto",
            "research",
            3.0,
            True,
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.01,
            llm_calls=1,
        )

        summary = mc.get_summary()
        g = summary["global"]
        assert g["total_executions"] == 1
        assert g["successful"] == 1
        assert g["failed"] == 0
        assert g["success_rate"] == 1.0
        assert g["latency"]["p50_s"] == 3.0

    def test_many_task_types_tracked(self):
        """Multiple task types are all tracked in the summary."""
        mc = MetricsCollector()
        task_types = ["research", "analysis", "creation", "coding", "review"]
        for tt in task_types:
            mc.record_execution("auto", tt, 1.0, True)

        summary = mc.get_summary()
        assert set(summary["task_types"].keys()) == set(task_types)
        for tt in task_types:
            assert summary["task_types"][tt] == 1

    def test_global_counters_accumulate_across_agents(self):
        """Global session counters (cost, tokens, llm_calls) accumulate across all agents."""
        mc = MetricsCollector()
        mc.record_execution(
            "auto", "r", 1.0, True, input_tokens=100, output_tokens=100, cost_usd=0.01, llm_calls=2
        )
        mc.record_execution(
            "analyst",
            "a",
            1.0,
            True,
            input_tokens=50,
            output_tokens=50,
            cost_usd=0.005,
            llm_calls=1,
        )

        assert mc._total_tokens == 300
        assert mc._total_cost_usd == pytest.approx(0.015)
        assert mc._total_llm_calls == 3

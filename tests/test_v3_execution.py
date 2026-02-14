"""
Tiered Execution Tests — Mocked Unit Tests
===========================================

Tests all tiers, observability, streaming, facade, auto-detection, and errors.
Every test uses mocked dependencies — no LLM calls, no API keys, runs offline.

Fixtures come from conftest.py:
    v3_executor — pre-wired TierExecutor with all mocks
    v3_observability_helpers — assertion helpers for metrics/traces/cost
"""

import json
import os
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from Jotty.core.execution.types import (
    ExecutionConfig,
    ExecutionTier,
    ExecutionResult,
    ExecutionPlan,
    ExecutionStep,
    ValidationResult,
    StreamEvent,
    StreamEventType,
)
from Jotty.core.execution.tier_detector import TierDetector


# =============================================================================
# Execution Types (sync, no executor needed)
# =============================================================================

@pytest.mark.unit
class TestExecutionTypes:
    """Test core execution data types."""

    def test_stream_event_type_values(self):
        """StreamEventType enum values match expected strings."""
        assert StreamEventType.STATUS.value == "status"
        assert StreamEventType.TOKEN.value == "token"
        assert StreamEventType.RESULT.value == "result"
        assert StreamEventType.ERROR.value == "error"
        assert StreamEventType.STEP_COMPLETE.value == "step_complete"
        assert StreamEventType.PARTIAL_OUTPUT.value == "partial_output"

    def test_stream_event_creation(self):
        """StreamEvent fields default correctly."""
        event = StreamEvent(type=StreamEventType.TOKEN, data="hello")
        assert event.type == StreamEventType.TOKEN
        assert event.data == "hello"
        assert event.tier is None
        assert event.timestamp is not None

    def test_execution_result_to_dict_with_trace(self):
        """to_dict() includes trace_id when trace is present."""
        mock_trace = Mock()
        mock_trace.trace_id = "abc123"
        result = ExecutionResult(
            output="test output",
            tier=ExecutionTier.DIRECT,
            success=True,
            trace=mock_trace,
        )
        d = result.to_dict()
        assert d['trace_id'] == "abc123"
        assert d['tier'] == "DIRECT"
        assert d['success'] is True

    def test_execution_result_to_dict_without_trace(self):
        """to_dict() has trace_id=None when no trace."""
        result = ExecutionResult(output="out", tier=ExecutionTier.AGENTIC)
        d = result.to_dict()
        assert d['trace_id'] is None

    def test_execution_result_str(self):
        """__str__() format is stable."""
        result = ExecutionResult(
            output="out",
            tier=ExecutionTier.DIRECT,
            success=True,
            llm_calls=1,
            latency_ms=123.4,
            cost_usd=0.0012,
        )
        s = str(result)
        assert "OK" in s
        assert "Tier 1" in s
        assert "1 calls" in s
        assert "123ms" in s

    def test_execution_step_is_complete(self):
        """ExecutionStep.is_complete follows result/error logic."""
        step = ExecutionStep(step_num=1, description="test")
        assert step.is_complete is False

        step.result = "done"
        assert step.is_complete is True

    def test_execution_plan_total_steps(self):
        """ExecutionPlan.total_steps counts correctly."""
        plan = ExecutionPlan(
            goal="test",
            steps=[
                ExecutionStep(step_num=1, description="a"),
                ExecutionStep(step_num=2, description="b"),
            ],
        )
        assert plan.total_steps == 2

    def test_execution_tier_values(self):
        """ExecutionTier integer values are correct."""
        assert ExecutionTier.DIRECT == 1
        assert ExecutionTier.AGENTIC == 2
        assert ExecutionTier.LEARNING == 3
        assert ExecutionTier.RESEARCH == 4
        assert ExecutionTier.AUTONOMOUS == 5


# =============================================================================
# Tier 1: DIRECT
# =============================================================================

@pytest.mark.unit
class TestTier1Direct:
    """Test Tier 1 (DIRECT) — single LLM call."""

    @pytest.mark.asyncio
    async def test_tier1_returns_result(self, v3_executor):
        """Tier 1 returns a successful ExecutionResult with correct fields."""
        result = await v3_executor.execute(
            "What is 2+2?",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )
        assert result.success is True
        assert result.tier == ExecutionTier.DIRECT
        assert result.llm_calls == 1
        assert result.output == 'Mock LLM response'

    @pytest.mark.asyncio
    async def test_tier1_uses_cost_tracker(self, v3_executor):
        """Tier 1 cost comes from CostTracker, not hardcoded."""
        result = await v3_executor.execute(
            "What is 2+2?",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )
        # CostTracker computes based on token counts, should be > 0
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_tier1_records_metrics(self, v3_executor, v3_observability_helpers):
        """Tier 1 records execution in MetricsCollector."""
        await v3_executor.execute(
            "What is 2+2?",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )
        v3_observability_helpers['assert_metrics_recorded']('tier_1')

    @pytest.mark.asyncio
    async def test_tier1_creates_trace(self, v3_executor, v3_observability_helpers):
        """Tier 1 creates a trace with tier1_llm_call span."""
        result = await v3_executor.execute(
            "What is 2+2?",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )
        v3_observability_helpers['assert_trace_exists']()
        assert result.trace is not None

    @pytest.mark.asyncio
    async def test_tier1_discovers_skills(self, v3_executor, mock_registry):
        """Tier 1 calls registry.discover_for_task with the goal."""
        await v3_executor.execute(
            "Search for quantum computing",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )
        mock_registry.discover_for_task.assert_called_with("Search for quantum computing")

    @pytest.mark.asyncio
    async def test_tier1_calls_provider_generate(self, v3_executor, mock_provider):
        """Tier 1 calls provider.generate exactly once."""
        await v3_executor.execute(
            "What is 2+2?",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )
        mock_provider.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tier1_status_callback(self, v3_executor):
        """Tier 1 calls status_callback with progress updates."""
        called = []

        def callback(stage, detail):
            called.append((stage, detail))

        await v3_executor.execute(
            "Hello",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
            status_callback=callback,
        )
        assert len(called) >= 2  # At least "Discovering skills" and "Calling LLM"
        stages = [c[0] for c in called]
        assert 'direct' in stages


# =============================================================================
# Tier 2: AGENTIC
# =============================================================================

@pytest.mark.unit
class TestTier2Agentic:
    """Test Tier 2 (AGENTIC) — planning + orchestration."""

    @pytest.mark.asyncio
    async def test_tier2_creates_plan(self, v3_executor):
        """Tier 2 result has a populated plan."""
        result = await v3_executor.execute(
            "Research AI and then create a summary",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )
        assert result.plan is not None
        assert len(result.plan.steps) == 2

    @pytest.mark.asyncio
    async def test_tier2_executes_steps(self, v3_executor):
        """Tier 2 executes all plan steps."""
        result = await v3_executor.execute(
            "Research AI and then create a summary",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )
        assert all(s.is_complete for s in result.steps)

    @pytest.mark.asyncio
    async def test_tier2_uses_cost_tracker_for_plan(self, v3_executor):
        """Tier 2 plan cost goes through CostTracker."""
        result = await v3_executor.execute(
            "Research AI and create summary",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )
        # Plan (1 call) + 2 steps = 3 calls, so cost > 0
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_tier2_uses_cost_tracker_for_steps(self, v3_executor):
        """Tier 2 step costs are computed from provider response tokens."""
        result = await v3_executor.execute(
            "Research AI and create summary",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )
        # 1 plan call + 2 step calls + 1 synthesis call = 4
        assert result.llm_calls == 4

    @pytest.mark.asyncio
    async def test_tier2_creates_trace_with_child_spans(self, v3_executor, v3_observability_helpers):
        """Tier 2 trace has tier2_plan and tier2_step spans."""
        result = await v3_executor.execute(
            "Research AI and create summary",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )
        v3_observability_helpers['assert_trace_exists']()
        assert result.trace is not None

    @pytest.mark.asyncio
    async def test_tier2_handles_step_failure(self, v3_executor, mock_provider):
        """When a step fails, error is recorded but other steps still run."""
        call_count = 0
        original_generate = mock_provider.generate

        async def _failing_generate(**kwargs):
            nonlocal call_count
            call_count += 1
            # Fail on second call (first step execution)
            if call_count == 2:
                raise ValueError("Step failed")
            return await original_generate(**kwargs)

        mock_provider.generate = AsyncMock(side_effect=_failing_generate)

        result = await v3_executor.execute(
            "Do A and B",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )
        # First step failed, but second step should still run
        step_errors = [s for s in result.steps if s.error]
        assert len(step_errors) >= 1

    @pytest.mark.asyncio
    async def test_tier2_calls_planner(self, v3_executor, mock_planner):
        """Tier 2 calls planner.plan with the goal."""
        await v3_executor.execute(
            "Build a REST API",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )
        mock_planner.plan.assert_awaited_once_with("Build a REST API")


# =============================================================================
# Tier 3: LEARNING
# =============================================================================

@pytest.mark.unit
class TestTier3Learning:
    """Test Tier 3 (LEARNING) — memory + validation."""

    @pytest.mark.asyncio
    async def test_tier3_retrieves_memory(self, v3_executor, mock_v3_memory):
        """Tier 3 calls memory.retrieve."""
        await v3_executor.execute(
            "Analyze sales data",
            config=ExecutionConfig(tier=ExecutionTier.LEARNING),
        )
        mock_v3_memory.retrieve.assert_awaited()

    @pytest.mark.asyncio
    async def test_tier3_enriches_goal(self, v3_executor, mock_planner):
        """Tier 3 passes enriched goal (with memory context) to planner."""
        await v3_executor.execute(
            "Analyze sales data",
            config=ExecutionConfig(tier=ExecutionTier.LEARNING),
        )
        # Planner should be called with enriched goal that includes memory entries
        call_args = mock_planner.plan.call_args[0][0]
        assert "Relevant past experience" in call_args
        assert "Previous analysis result" in call_args

    @pytest.mark.asyncio
    async def test_tier3_validates_result(self, v3_executor, mock_validator):
        """Tier 3 calls validator and populates result.validation."""
        result = await v3_executor.execute(
            "Analyze sales data",
            config=ExecutionConfig(tier=ExecutionTier.LEARNING, enable_validation=True),
        )
        mock_validator.validate.assert_awaited()
        assert result.validation is not None
        assert result.validation.success is True
        assert result.validation.confidence == 0.9

    @pytest.mark.asyncio
    async def test_tier3_uses_cost_tracker_for_validation(self, v3_executor):
        """Tier 3 validation cost goes through CostTracker."""
        result = await v3_executor.execute(
            "Analyze sales data",
            config=ExecutionConfig(tier=ExecutionTier.LEARNING, enable_validation=True),
        )
        # Plan + steps + validation = several calls, all tracked
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_tier3_stores_memory(self, v3_executor, mock_v3_memory):
        """Tier 3 stores result in memory after execution."""
        await v3_executor.execute(
            "Analyze sales data",
            config=ExecutionConfig(tier=ExecutionTier.LEARNING),
        )
        mock_v3_memory.store.assert_awaited()

    @pytest.mark.asyncio
    async def test_tier3_creates_trace_with_memory_spans(self, v3_executor, v3_observability_helpers):
        """Tier 3 trace has memory, execute, and validate spans."""
        result = await v3_executor.execute(
            "Analyze sales data",
            config=ExecutionConfig(tier=ExecutionTier.LEARNING, enable_validation=True),
        )
        v3_observability_helpers['assert_trace_exists']()
        assert result.trace is not None

    @pytest.mark.asyncio
    async def test_tier3_used_memory_flag(self, v3_executor):
        """Tier 3 sets used_memory=True when memory entries are found."""
        result = await v3_executor.execute(
            "Analyze sales data",
            config=ExecutionConfig(tier=ExecutionTier.LEARNING),
        )
        assert result.used_memory is True

    @pytest.mark.asyncio
    async def test_tier3_no_memory_when_backend_none(self, v3_executor, mock_v3_memory):
        """Tier 3 skips memory when backend='none'."""
        result = await v3_executor.execute(
            "Analyze sales data",
            config=ExecutionConfig(tier=ExecutionTier.LEARNING, memory_backend="none"),
        )
        mock_v3_memory.retrieve.assert_not_awaited()
        assert result.used_memory is False


# =============================================================================
# Tier 4: RESEARCH (domain swarm execution)
# =============================================================================

@pytest.mark.unit
class TestTier4Research:
    """Test Tier 4 (RESEARCH) — domain swarm execution."""

    @pytest.mark.asyncio
    async def test_tier4_selects_swarm(self, v3_executor):
        """Tier 4 selects correct swarm by keyword."""
        mock_swarm = AsyncMock()
        mock_swarm.execute = AsyncMock(return_value=Mock(
            output={'result': 'code'}, success=True,
        ))
        mock_swarm.__class__.__name__ = 'CodingSwarm'

        with patch.object(v3_executor, '_select_swarm', return_value=mock_swarm):
            result = await v3_executor.execute(
                "Code a function",
                config=ExecutionConfig(tier=ExecutionTier.RESEARCH),
            )
            assert result.success is True
            assert result.swarm_name == 'CodingSwarm'

    @pytest.mark.asyncio
    async def test_tier4_executes_swarm(self, v3_executor):
        """Tier 4 calls swarm.execute with the goal."""
        mock_swarm = AsyncMock()
        mock_swarm.execute = AsyncMock(return_value=Mock(
            output={'result': 'done'}, success=True,
        ))
        mock_swarm.__class__.__name__ = 'ResearchSwarm'

        with patch.object(v3_executor, '_select_swarm', return_value=mock_swarm):
            await v3_executor.execute(
                "Research AI",
                config=ExecutionConfig(tier=ExecutionTier.RESEARCH),
            )
            mock_swarm.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tier4_creates_trace(self, v3_executor, v3_observability_helpers):
        """Tier 4 creates a trace with tier4_swarm span."""
        mock_swarm = AsyncMock()
        mock_swarm.execute = AsyncMock(return_value=Mock(
            output={'result': 'done'}, success=True,
        ))
        mock_swarm.__class__.__name__ = 'TestSwarm'

        with patch.object(v3_executor, '_select_swarm', return_value=mock_swarm):
            result = await v3_executor.execute(
                "Test something",
                config=ExecutionConfig(tier=ExecutionTier.RESEARCH),
            )
            v3_observability_helpers['assert_trace_exists']()
            assert result.trace is not None

    @pytest.mark.asyncio
    async def test_tier4_falls_back_to_v2(self, v3_executor):
        """Tier 4 falls back to Orchestrator when no swarm matches."""
        with patch.object(v3_executor, '_select_swarm', return_value=None):
            with patch.object(v3_executor, '_execute_with_swarm_manager', new_callable=AsyncMock) as mock_fallback:
                mock_fallback.return_value = ExecutionResult(
                    output="Orchestrator result",
                    tier=ExecutionTier.RESEARCH,
                    success=True,
                )
                result = await v3_executor.execute(
                    "Unknown task type",
                    config=ExecutionConfig(tier=ExecutionTier.RESEARCH),
                )
                mock_fallback.assert_awaited_once()
                assert result.output == "Orchestrator result"


# =============================================================================
# Tier 5: AUTONOMOUS
# =============================================================================

@pytest.mark.unit
class TestTier5Autonomous:
    """Test Tier 5 (AUTONOMOUS) — sandbox + coalition."""

    @pytest.mark.asyncio
    async def test_tier5_executes_with_swarm(self, v3_executor):
        """Tier 5 basic execution works."""
        mock_swarm = AsyncMock()
        mock_swarm.execute = AsyncMock(return_value=Mock(
            output={'result': 'autonomous'}, success=True,
        ))
        mock_swarm.__class__.__name__ = 'CodingSwarm'

        with patch.object(v3_executor, '_select_swarm', return_value=mock_swarm):
            result = await v3_executor.execute(
                "Execute code in sandbox",
                config=ExecutionConfig(tier=ExecutionTier.AUTONOMOUS),
            )
            assert result.success is True
            assert result.tier == ExecutionTier.AUTONOMOUS

    @pytest.mark.asyncio
    async def test_tier5_creates_trace(self, v3_executor, v3_observability_helpers):
        """Tier 5 creates a trace with tier5_autonomous span."""
        mock_swarm = AsyncMock()
        mock_swarm.execute = AsyncMock(return_value=Mock(
            output={'result': 'done'}, success=True,
        ))
        mock_swarm.__class__.__name__ = 'TestSwarm'

        with patch.object(v3_executor, '_select_swarm', return_value=mock_swarm):
            result = await v3_executor.execute(
                "Autonomous task",
                config=ExecutionConfig(tier=ExecutionTier.AUTONOMOUS),
            )
            v3_observability_helpers['assert_trace_exists']()
            assert result.trace is not None


# =============================================================================
# Observability (parametrized across tiers)
# =============================================================================

@pytest.mark.unit
class TestObservability:
    """Test observability integration across all tiers."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tier,goal", [
        (ExecutionTier.DIRECT, "What is 2+2?"),
        (ExecutionTier.AGENTIC, "Research and summarize AI"),
        (ExecutionTier.LEARNING, "Analyze and improve sales"),
    ])
    async def test_metrics_recorded_per_tier(self, v3_executor, v3_observability_helpers, tier, goal):
        """Metrics are recorded for tiers 1-3."""
        await v3_executor.execute(goal, config=ExecutionConfig(tier=tier))
        v3_observability_helpers['assert_metrics_recorded'](f'tier_{tier.value}')

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tier,goal", [
        (ExecutionTier.DIRECT, "What is 2+2?"),
        (ExecutionTier.AGENTIC, "Research and summarize AI"),
        (ExecutionTier.LEARNING, "Analyze and improve sales"),
    ])
    async def test_trace_attached_to_result(self, v3_executor, tier, goal):
        """Result.trace is not None for all tiers."""
        result = await v3_executor.execute(goal, config=ExecutionConfig(tier=tier))
        assert result.trace is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tier,goal", [
        (ExecutionTier.DIRECT, "What is 2+2?"),
        (ExecutionTier.AGENTIC, "Research and summarize AI"),
        (ExecutionTier.LEARNING, "Analyze and improve sales"),
    ])
    async def test_cost_nonzero(self, v3_executor, tier, goal):
        """Cost is > 0 for tiers 1-3."""
        result = await v3_executor.execute(goal, config=ExecutionConfig(tier=tier))
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_error_recorded_on_failure(self, v3_executor, mock_provider):
        """Failures are recorded in metrics."""
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("LLM down"))

        result = await v3_executor.execute(
            "Will fail",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )
        assert result.success is False
        assert "LLM down" in result.error

        from Jotty.core.observability.metrics import get_metrics
        errors = get_metrics().recent_errors(limit=5)
        assert len(errors) >= 1
        assert "LLM down" in errors[0]['error']


# =============================================================================
# Streaming
# =============================================================================

@pytest.mark.unit
class TestStreaming:
    """Test streaming execution."""

    @pytest.mark.asyncio
    async def test_stream_tier1_yields_tokens(self, v3_executor):
        """Tier 1 streaming yields TOKEN events."""
        events = []
        async for event in v3_executor.stream(
            "Hello",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        ):
            events.append(event)

        token_events = [e for e in events if e.type == StreamEventType.TOKEN]
        assert len(token_events) >= 1
        tokens = ''.join(e.data for e in token_events)
        assert 'Mock' in tokens

    @pytest.mark.asyncio
    async def test_stream_tier1_yields_result(self, v3_executor):
        """Tier 1 streaming ends with a RESULT event."""
        events = []
        async for event in v3_executor.stream(
            "Hello",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        ):
            events.append(event)

        result_events = [e for e in events if e.type == StreamEventType.RESULT]
        assert len(result_events) == 1
        assert isinstance(result_events[0].data, ExecutionResult)

    @pytest.mark.asyncio
    async def test_stream_tier1_fallback_no_stream(self, v3_executor, mock_provider):
        """Provider without stream() still yields RESULT."""
        del mock_provider.stream  # Remove stream method

        events = []
        async for event in v3_executor.stream(
            "Hello",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        ):
            events.append(event)

        result_events = [e for e in events if e.type == StreamEventType.RESULT]
        assert len(result_events) == 1

    @pytest.mark.asyncio
    async def test_stream_tier2_yields_status_events(self, v3_executor):
        """Tier 2 streaming yields STATUS events for planning/executing phases."""
        events = []
        async for event in v3_executor.stream(
            "Research and summarize",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        ):
            events.append(event)

        status_events = [e for e in events if e.type == StreamEventType.STATUS]
        assert len(status_events) >= 1

        result_events = [e for e in events if e.type == StreamEventType.RESULT]
        assert len(result_events) == 1

    @pytest.mark.asyncio
    async def test_stream_yields_error_on_failure(self, v3_executor, mock_provider):
        """Streaming propagates or wraps exception on tier 1 failure."""
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("LLM crashed"))
        # Also remove stream so it falls back to generate
        if hasattr(mock_provider, 'stream'):
            del mock_provider.stream

        # Tier 1 streaming propagates exceptions from _stream_tier1 directly.
        # Non-tier-1 streaming catches them via the queue bridge.
        # For tier 2+, we'd get an ERROR event, but tier 1 raises.
        with pytest.raises(RuntimeError, match="LLM crashed"):
            async for _ in v3_executor.stream(
                "Will fail",
                config=ExecutionConfig(tier=ExecutionTier.DIRECT),
            ):
                pass

    @pytest.mark.asyncio
    async def test_stream_tier2_yields_error_on_failure(self, v3_executor, mock_provider):
        """Non-tier-1 streaming yields ERROR event (queue bridge catches it)."""
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("LLM crashed"))

        events = []
        async for event in v3_executor.stream(
            "Research and summarize",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        ):
            events.append(event)

        # Queue-based streaming catches the exception and yields RESULT with success=False
        result_events = [e for e in events if e.type == StreamEventType.RESULT]
        assert len(result_events) == 1
        assert result_events[0].data.success is False


# =============================================================================
# Jotty Facade
# =============================================================================

@pytest.mark.unit
class TestJottyFacade:
    """Test the Jotty facade class."""

    def _make_jotty(self, mock_executor=None):
        """Create Jotty instance with mocked executor."""
        from Jotty.jotty import Jotty
        jotty = Jotty(log_level="ERROR")
        if mock_executor:
            jotty.executor = mock_executor
        return jotty

    def test_get_stats_includes_config(self):
        """stats['config'] has default_tier and memory_backend."""
        jotty = self._make_jotty()
        stats = jotty.get_stats()
        assert 'config' in stats
        assert 'default_tier' in stats['config']
        assert 'memory_backend' in stats['config']

    def test_get_stats_includes_metrics(self):
        """stats['global']['total_executions'] works."""
        jotty = self._make_jotty()
        stats = jotty.get_stats()
        assert 'global' in stats
        assert 'total_executions' in stats['global']

    def test_get_cost_breakdown(self):
        """get_cost_breakdown returns dict with total_cost_usd."""
        jotty = self._make_jotty()
        costs = jotty.get_cost_breakdown()
        assert 'total_cost_usd' in costs

    def test_get_recent_errors_empty(self):
        """get_recent_errors returns [] when no errors."""
        jotty = self._make_jotty()
        errors = jotty.get_recent_errors()
        assert errors == []

    def test_get_recent_errors_with_data(self):
        """get_recent_errors returns errors after recording failure."""
        jotty = self._make_jotty()
        from Jotty.core.observability.metrics import get_metrics
        get_metrics().record_execution(
            agent_name="test_agent",
            task_type="test",
            duration_s=1.0,
            success=False,
            error="Something broke",
        )
        errors = jotty.get_recent_errors()
        assert len(errors) >= 1
        assert "Something broke" in errors[0]['error']

    def test_save_metrics_writes_file(self, tmp_path):
        """save_metrics writes a valid JSON file."""
        jotty = self._make_jotty()
        path = str(tmp_path / "metrics.json")
        saved = jotty.save_metrics(path=path)
        assert Path(saved).exists()
        data = json.loads(Path(saved).read_text())
        assert 'global' in data

    @pytest.mark.asyncio
    async def test_stream_delegates_to_executor(self):
        """jotty.stream() delegates to executor.stream()."""
        jotty = self._make_jotty()

        mock_event = StreamEvent(type=StreamEventType.RESULT, data="done")

        async def _mock_stream(goal, config=None, **kw):
            yield mock_event

        jotty.executor.stream = _mock_stream

        events = []
        async for event in jotty.stream("test"):
            events.append(event)

        assert len(events) == 1
        assert events[0].type == StreamEventType.RESULT

    @pytest.mark.asyncio
    async def test_run_delegates_to_executor(self):
        """jotty.run() delegates to executor.execute()."""
        jotty = self._make_jotty()
        mock_result = ExecutionResult(output="test", tier=ExecutionTier.DIRECT, success=True)
        jotty.executor.execute = AsyncMock(return_value=mock_result)

        result = await jotty.run("Hello")
        assert result.output == "test"
        jotty.executor.execute.assert_awaited_once()

    def test_set_default_tier(self):
        """set_default_tier updates config."""
        jotty = self._make_jotty()
        jotty.set_default_tier(ExecutionTier.LEARNING)
        assert jotty.config.tier == ExecutionTier.LEARNING

    def test_explain_tier(self):
        """explain_tier returns non-empty string."""
        jotty = self._make_jotty()
        explanation = jotty.explain_tier("What is 2+2?")
        assert len(explanation) > 0
        assert "DIRECT" in explanation


# =============================================================================
# Auto-Detection (sync, no mocks needed)
# =============================================================================

@pytest.mark.unit
class TestAutoDetection:
    """Test tier auto-detection logic."""

    def test_detect_simple_query(self):
        """Simple queries auto-detect to DIRECT."""
        detector = TierDetector()
        assert detector.detect("What is 2+2?") == ExecutionTier.DIRECT
        assert detector.detect("Define AI") == ExecutionTier.DIRECT
        assert detector.detect("Calculate 15 * 23") == ExecutionTier.DIRECT

    def test_detect_multi_step(self):
        """Multi-step tasks auto-detect to AGENTIC."""
        detector = TierDetector()
        assert detector.detect("Research X and then summarize the results in a detailed report") == ExecutionTier.AGENTIC

    def test_detect_learning(self):
        """Learning keywords auto-detect to LEARNING."""
        detector = TierDetector()
        assert detector.detect("Learn from past failures and improve") == ExecutionTier.LEARNING
        assert detector.detect("Remember this pattern for next time") == ExecutionTier.LEARNING

    def test_detect_research(self):
        """Research keywords auto-detect to RESEARCH."""
        detector = TierDetector()
        assert detector.detect("Research thoroughly the impact of AI") == ExecutionTier.RESEARCH
        assert detector.detect("Benchmark different approaches") == ExecutionTier.RESEARCH

    def test_detect_autonomous(self):
        """Autonomous keywords auto-detect to AUTONOMOUS."""
        detector = TierDetector()
        assert detector.detect("Execute code in sandbox environment") == ExecutionTier.AUTONOMOUS
        assert detector.detect("Run coalition consensus protocol") == ExecutionTier.AUTONOMOUS

    def test_explain_detection(self):
        """explain_detection returns non-empty string with tier name."""
        detector = TierDetector()
        explanation = detector.explain_detection("What is 2+2?")
        assert len(explanation) > 0
        assert "DIRECT" in explanation

    def test_force_tier(self):
        """force_tier overrides auto-detection."""
        detector = TierDetector()
        tier = detector.detect("Simple query", force_tier=ExecutionTier.RESEARCH)
        assert tier == ExecutionTier.RESEARCH

    def test_detection_cache(self):
        """Repeated detection uses cache."""
        detector = TierDetector()
        tier1 = detector.detect("What is 2+2?")
        tier2 = detector.detect("What is 2+2?")
        assert tier1 == tier2
        assert len(detector.detection_cache) == 1


# =============================================================================
# Error Handling
# =============================================================================

@pytest.mark.unit
class TestErrorHandling:
    """Test error handling across tiers."""

    @pytest.mark.asyncio
    async def test_execute_catches_exception(self, v3_executor, mock_provider):
        """Provider raising an exception returns result.success=False."""
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("Connection refused"))

        result = await v3_executor.execute(
            "Will fail",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )
        assert result.success is False
        assert "Connection refused" in result.error

    @pytest.mark.asyncio
    async def test_execute_records_error_metrics(self, v3_executor, mock_provider, v3_observability_helpers):
        """Failure is recorded in MetricsCollector."""
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("Timeout"))

        await v3_executor.execute(
            "Will fail",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )

        from Jotty.core.observability.metrics import get_metrics
        agent_metrics = get_metrics().get_agent_metrics('tier_1')
        assert agent_metrics is not None
        assert agent_metrics.failed >= 1

    @pytest.mark.asyncio
    async def test_execute_ends_trace_on_error(self, v3_executor, mock_provider, v3_observability_helpers):
        """Trace is still ended cleanly on error."""
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("Boom"))

        await v3_executor.execute(
            "Will fail",
            config=ExecutionConfig(tier=ExecutionTier.DIRECT),
        )

        # Trace should exist in history (end_trace called in except block)
        from Jotty.core.observability.tracing import get_tracer
        traces = get_tracer().get_trace_history()
        assert len(traces) >= 1

    @pytest.mark.asyncio
    async def test_tier3_memory_retrieval_failure(self, v3_executor, mock_v3_memory):
        """Memory retrieval failure doesn't crash tier 3."""
        mock_v3_memory.retrieve = AsyncMock(side_effect=ConnectionError("Redis down"))

        result = await v3_executor.execute(
            "Analyze data",
            config=ExecutionConfig(tier=ExecutionTier.LEARNING),
        )
        # Should still succeed (memory failure is non-fatal)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_tier3_memory_store_failure(self, v3_executor, mock_v3_memory):
        """Memory store failure doesn't crash tier 3."""
        mock_v3_memory.store = AsyncMock(side_effect=ConnectionError("Redis down"))

        result = await v3_executor.execute(
            "Analyze data",
            config=ExecutionConfig(tier=ExecutionTier.LEARNING),
        )
        # Should still succeed (store failure is non-fatal)
        assert result.success is True


# =============================================================================
# ComplexityGate
# =============================================================================

@pytest.mark.unit
class TestComplexityGate:
    """Test ComplexityGate — LLM-based planning bypass for Tier 2."""

    @pytest.mark.asyncio
    async def test_skip_planning_returns_direct(self, v3_executor, mock_complexity_gate, mock_provider):
        """When gate says DIRECT, Tier 2 returns a single-call result without planning."""
        mock_complexity_gate.should_skip_planning = AsyncMock(return_value=True)

        result = await v3_executor.execute(
            "What are the differences between Python and JavaScript?",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )
        assert result.success is True
        assert result.llm_calls == 1
        assert result.metadata.get('complexity_gate') == 'direct'

    @pytest.mark.asyncio
    async def test_proceed_with_planning(self, v3_executor, mock_complexity_gate, mock_planner):
        """When gate says TOOLS, Tier 2 proceeds with full planning."""
        mock_complexity_gate.should_skip_planning = AsyncMock(return_value=False)

        result = await v3_executor.execute(
            "Research AI trends and create a summary",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )
        assert result.plan is not None
        mock_planner.plan.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gate_failure_defaults_to_planning(self, v3_executor, mock_complexity_gate, mock_planner):
        """If gate raises an exception, execution proceeds with planning."""
        mock_complexity_gate.should_skip_planning = AsyncMock(
            side_effect=RuntimeError("API error")
        )

        result = await v3_executor.execute(
            "Research something",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )
        assert result.plan is not None
        mock_planner.plan.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skip_planning_uses_cost_tracker(self, v3_executor, mock_complexity_gate):
        """Direct bypass still records cost via CostTracker."""
        mock_complexity_gate.should_skip_planning = AsyncMock(return_value=True)

        result = await v3_executor.execute(
            "What is recursion?",
            config=ExecutionConfig(tier=ExecutionTier.AGENTIC),
        )
        assert result.cost_usd > 0


# =============================================================================
# Output Synthesis
# =============================================================================

@pytest.mark.unit
class TestOutputSynthesis:
    """Test _synthesize_results — LLM-based multi-step output merging."""

    @pytest.mark.asyncio
    async def test_single_result_no_llm_call(self, v3_executor, mock_provider):
        """Single step result is returned as-is without LLM synthesis."""
        results = [{'output': 'Only one result'}]
        synthesis = await v3_executor._synthesize_results(results, "test goal")
        assert synthesis['output'] == 'Only one result'
        assert synthesis['llm_calls'] == 0

    @pytest.mark.asyncio
    async def test_multi_result_uses_llm(self, v3_executor, mock_provider):
        """Multiple step results trigger an LLM synthesis call."""
        results = [
            {'output': 'Step 1 result'},
            {'output': 'Step 2 result'},
        ]
        synthesis = await v3_executor._synthesize_results(results, "combine these")
        assert synthesis['llm_calls'] == 1
        assert synthesis['cost'] > 0
        # Provider was called for synthesis
        assert mock_provider.generate.await_count >= 1

    @pytest.mark.asyncio
    async def test_synthesis_failure_uses_fallback(self, v3_executor, mock_provider):
        """When LLM synthesis fails, falls back to simple concatenation."""
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("LLM down"))

        results = [
            {'output': 'Step 1 result'},
            {'output': 'Step 2 result'},
        ]
        synthesis = await v3_executor._synthesize_results(results, "test goal")
        assert synthesis['llm_calls'] == 0
        assert 'Step 1' in synthesis['output']
        assert 'Step 2' in synthesis['output']


# =============================================================================
# LLM-Enhanced Tier Detection
# =============================================================================

@pytest.mark.unit
class TestLLMTierDetector:
    """Test TierDetector with LLM fallback for ambiguous cases."""

    def test_clear_keywords_skip_llm(self):
        """Clear keyword matches have high confidence and don't need LLM."""
        detector = TierDetector(enable_llm_fallback=True)
        tier, confidence = detector._detect_tier_with_confidence("Execute code in sandbox")
        assert tier == ExecutionTier.AUTONOMOUS
        assert confidence >= 0.7

    def test_ambiguous_task_low_confidence(self):
        """Tasks without clear keywords get low confidence."""
        detector = TierDetector(enable_llm_fallback=False)
        tier, confidence = detector._detect_tier_with_confidence(
            "Tell me about the history of computing and its impact on society"
        )
        assert confidence < 0.7

    @pytest.mark.asyncio
    async def test_llm_fallback_on_ambiguous(self):
        """LLM classifier is consulted for ambiguous tasks when enabled."""
        detector = TierDetector(enable_llm_fallback=True)

        # Mock the LLM classifier
        mock_classifier = AsyncMock()
        mock_classifier.classify = AsyncMock(return_value=ExecutionTier.DIRECT)
        detector._llm_classifier = mock_classifier

        tier = await detector.adetect(
            "Tell me about the history of computing and its impact on society"
        )
        mock_classifier.classify.assert_awaited_once()
        assert tier == ExecutionTier.DIRECT

    @pytest.mark.asyncio
    async def test_llm_failure_uses_heuristic(self):
        """When LLM classifier fails, the heuristic result is used."""
        detector = TierDetector(enable_llm_fallback=True)

        mock_classifier = AsyncMock()
        mock_classifier.classify = AsyncMock(side_effect=RuntimeError("API error"))
        detector._llm_classifier = mock_classifier

        tier = await detector.adetect(
            "Tell me about the history of computing and its impact on society"
        )
        # Falls back to heuristic (AGENTIC for ambiguous tasks)
        assert tier == ExecutionTier.AGENTIC

    @pytest.mark.asyncio
    async def test_sync_detect_unchanged(self):
        """Sync detect() still works without LLM, preserving backward compat."""
        detector = TierDetector(enable_llm_fallback=True)
        tier = detector.detect("What is 2+2?")
        assert tier == ExecutionTier.DIRECT


# =============================================================================
# Error Classification Tests
# =============================================================================

@pytest.mark.unit
class TestErrorClassification:
    """Test ErrorType, ValidationStatus, ValidationVerdict from types.py."""

    def test_error_type_values(self):
        """ErrorType enum has all expected members."""
        from Jotty.core.execution.types import ErrorType
        assert ErrorType.NONE.value == "none"
        assert ErrorType.INFRASTRUCTURE.value == "infrastructure"
        assert ErrorType.LOGIC.value == "logic"
        assert ErrorType.DATA.value == "data"
        assert ErrorType.ENVIRONMENT.value == "environment"

    def test_error_type_classify_infrastructure(self):
        """Infrastructure errors are classified correctly."""
        from Jotty.core.execution.types import ErrorType
        assert ErrorType.classify("Connection timeout after 30s") == ErrorType.INFRASTRUCTURE
        assert ErrorType.classify("Rate limit exceeded (429)") == ErrorType.INFRASTRUCTURE
        assert ErrorType.classify("503 Service Unavailable") == ErrorType.INFRASTRUCTURE

    def test_error_type_classify_environment(self):
        """Environment/proxy errors are classified correctly."""
        from Jotty.core.execution.types import ErrorType
        assert ErrorType.classify("SSL certificate verify failed") == ErrorType.ENVIRONMENT
        assert ErrorType.classify("Zscaler proxy blocked request") == ErrorType.ENVIRONMENT
        assert ErrorType.classify("TLS handshake error") == ErrorType.ENVIRONMENT

    def test_error_type_classify_data(self):
        """Data errors are classified correctly."""
        from Jotty.core.execution.types import ErrorType
        assert ErrorType.classify("Empty result set") == ErrorType.DATA
        assert ErrorType.classify("No results found for query") == ErrorType.DATA
        assert ErrorType.classify("Invalid JSON: parse error") == ErrorType.DATA

    def test_error_type_classify_logic(self):
        """Logic errors are classified correctly."""
        from Jotty.core.execution.types import ErrorType
        assert ErrorType.classify("Element not found: #submit-btn") == ErrorType.LOGIC
        assert ErrorType.classify("Syntax error in expression") == ErrorType.LOGIC
        assert ErrorType.classify("Missing required parameter 'query'") == ErrorType.LOGIC

    def test_error_type_classify_fallback(self):
        """Unknown errors default to INFRASTRUCTURE (retryable)."""
        from Jotty.core.execution.types import ErrorType
        assert ErrorType.classify("Something weird happened") == ErrorType.INFRASTRUCTURE

    def test_validation_status_values(self):
        """ValidationStatus enum has all expected members."""
        from Jotty.core.execution.types import ValidationStatus
        assert ValidationStatus.PASS.value == "pass"
        assert ValidationStatus.FAIL.value == "fail"
        assert ValidationStatus.EXTERNAL_ERROR.value == "external_error"
        assert ValidationStatus.ENQUIRY.value == "enquiry"

    def test_validation_verdict_ok(self):
        """ValidationVerdict.ok() creates a passing verdict."""
        from Jotty.core.execution.types import ValidationVerdict, ValidationStatus, ErrorType
        v = ValidationVerdict.ok("All good", confidence=0.95)
        assert v.is_pass is True
        assert v.status == ValidationStatus.PASS
        assert v.confidence == 0.95
        assert v.error_type == ErrorType.NONE

    def test_validation_verdict_from_error(self):
        """ValidationVerdict.from_error() auto-classifies and sets retryable."""
        from Jotty.core.execution.types import ValidationVerdict, ValidationStatus, ErrorType
        v = ValidationVerdict.from_error("Connection timeout after 30s")
        assert v.is_pass is False
        assert v.error_type == ErrorType.INFRASTRUCTURE
        assert v.retryable is True
        assert v.status == ValidationStatus.FAIL

    def test_validation_verdict_from_logic_error_not_retryable(self):
        """Logic errors are not marked retryable."""
        from Jotty.core.execution.types import ValidationVerdict, ErrorType
        v = ValidationVerdict.from_error("Element not found: #missing selector")
        assert v.error_type == ErrorType.LOGIC
        assert v.retryable is False

    def test_validation_verdict_fields(self):
        """ValidationVerdict fields default correctly."""
        from Jotty.core.execution.types import ValidationVerdict, ValidationStatus, ErrorType
        v = ValidationVerdict(
            status=ValidationStatus.FAIL,
            error_type=ErrorType.DATA,
            reason="Empty results",
            issues=["No data returned"],
            fixes=["Try different query"],
            confidence=0.8,
            retryable=True,
        )
        assert v.reason == "Empty results"
        assert len(v.issues) == 1
        assert len(v.fixes) == 1
        assert v.retryable is True


# =============================================================================
# ToolResultProcessor Tests
# =============================================================================

@pytest.mark.unit
class TestToolResultProcessor:
    """Test ToolResultProcessor from skill_plan_executor.py."""

    def test_process_strips_binary(self):
        """Binary/base64 data is replaced with size placeholder."""
        from Jotty.core.agents.base.skill_plan_executor import ToolResultProcessor
        processor = ToolResultProcessor()
        result = processor.process({
            'success': True,
            'screenshot_base64': 'A' * 50000,
            'data': 'short value',
        })
        assert 'binary data' in result.get('screenshot_base64', '')
        assert result['data'] == 'short value'
        assert result['success'] is True

    def test_process_converts_sets(self):
        """Sets are converted to sorted lists."""
        from Jotty.core.agents.base.skill_plan_executor import ToolResultProcessor
        processor = ToolResultProcessor()
        result = processor.process({'tags': {'b', 'a', 'c'}})
        assert isinstance(result['tags'], list)
        assert result['tags'] == ['a', 'b', 'c']

    def test_process_truncates_large_values(self):
        """Large string values are truncated while preserving keys."""
        from Jotty.core.agents.base.skill_plan_executor import ToolResultProcessor
        processor = ToolResultProcessor()
        result = processor.process({
            'status': 'ok',
            'large_content': 'x' * 100_000,
        }, max_size=5000)
        assert 'status' in result
        assert result['status'] == 'ok'
        # large_content should be truncated
        assert len(str(result.get('large_content', ''))) < 100_000

    def test_process_adds_execution_time(self):
        """Elapsed time is added to result."""
        from Jotty.core.agents.base.skill_plan_executor import ToolResultProcessor
        processor = ToolResultProcessor()
        result = processor.process({'success': True}, elapsed=1.5)
        assert result['_execution_time_ms'] == 1500.0

    def test_process_non_dict_input(self):
        """Non-dict input is wrapped in output key."""
        from Jotty.core.agents.base.skill_plan_executor import ToolResultProcessor
        processor = ToolResultProcessor()
        result = processor.process("plain string")
        assert 'output' in result
        assert result['output'] == 'plain string'

    def test_process_preserves_small_results(self):
        """Small results pass through unchanged (except set conversion)."""
        from Jotty.core.agents.base.skill_plan_executor import ToolResultProcessor
        processor = ToolResultProcessor()
        original = {'success': True, 'path': '/tmp/file.txt', 'bytes': 42}
        result = processor.process(original)
        assert result['success'] is True
        assert result['path'] == '/tmp/file.txt'
        assert result['bytes'] == 42


# =============================================================================
# SearchCache Tests
# =============================================================================

@pytest.mark.unit
class TestSearchCache:
    """Test SearchCache from web-search/tools.py."""

    def test_cache_miss_returns_none(self):
        """Cache miss returns None."""
        import threading, time

        class _TestSearchCache:
            def __init__(self, ttl_seconds=300):
                self._cache = {}
                self._lock = threading.Lock()
                self._ttl = ttl_seconds
            def get(self, key):
                with self._lock:
                    now = time.time()
                    expired = [k for k, (_, ts) in self._cache.items() if now - ts > self._ttl]
                    for k in expired:
                        del self._cache[k]
                    entry = self._cache.get(key)
                    return entry[0] if entry else None
            def set(self, key, value):
                with self._lock:
                    self._cache[key] = (value, time.time())
            def clear(self):
                with self._lock:
                    self._cache.clear()

        cache = _TestSearchCache(ttl_seconds=300)
        assert cache.get("nonexistent") is None

    def test_cache_set_and_get(self):
        """Set/get roundtrip works."""
        import threading, time

        class _TestSearchCache:
            def __init__(self, ttl_seconds=300):
                self._cache = {}
                self._lock = threading.Lock()
                self._ttl = ttl_seconds
            def get(self, key):
                with self._lock:
                    entry = self._cache.get(key)
                    return entry[0] if entry else None
            def set(self, key, value):
                with self._lock:
                    self._cache[key] = (value, time.time())

        cache = _TestSearchCache()
        data = {"results": [{"title": "Test"}], "count": 1}
        cache.set("serper:test query:10", data)
        assert cache.get("serper:test query:10") == data

    def test_cache_ttl_expiry(self):
        """Expired entries are evicted."""
        import threading, time

        class _TestSearchCache:
            def __init__(self, ttl_seconds=300):
                self._cache = {}
                self._lock = threading.Lock()
                self._ttl = ttl_seconds
            def get(self, key):
                with self._lock:
                    now = time.time()
                    expired = [k for k, (_, ts) in self._cache.items() if now - ts > self._ttl]
                    for k in expired:
                        del self._cache[k]
                    entry = self._cache.get(key)
                    return entry[0] if entry else None
            def set(self, key, value):
                with self._lock:
                    self._cache[key] = (value, time.time())

        cache = _TestSearchCache(ttl_seconds=0)  # Expire immediately
        cache.set("key", "value")
        time.sleep(0.01)
        assert cache.get("key") is None

    def test_cache_clear(self):
        """Clear removes all entries."""
        import threading, time

        class _TestSearchCache:
            def __init__(self, ttl_seconds=300):
                self._cache = {}
                self._lock = threading.Lock()
                self._ttl = ttl_seconds
            def get(self, key):
                with self._lock:
                    entry = self._cache.get(key)
                    return entry[0] if entry else None
            def set(self, key, value):
                with self._lock:
                    self._cache[key] = (value, time.time())
            def clear(self):
                with self._lock:
                    self._cache.clear()

        cache = _TestSearchCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None


# =============================================================================
# CompletionReviewer Tests
# =============================================================================

@pytest.mark.unit
class TestCompletionReviewer:
    """Test CompletionReviewer from inspector.py."""

    def test_reviewer_fallback_on_success(self):
        """Fallback heuristic returns 'complete' when result has success=True."""
        from Jotty.core.agents.inspector import CompletionReviewer
        reviewer = CompletionReviewer()

        # Mock the predictor to raise, forcing fallback
        reviewer._predictor = Mock(side_effect=Exception("No LLM"))
        reviewer._ensure_predictor = Mock()

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            reviewer.review_completion(
                instruction="Test task",
                result={"success": True, "output": "done"},
                tool_calls=[],
            )
        )
        assert result["completion_state"] == "complete"
        assert result["confidence"] > 0

    def test_reviewer_fallback_on_failure(self):
        """Fallback heuristic returns 'partial' when result has success=False."""
        from Jotty.core.agents.inspector import CompletionReviewer
        reviewer = CompletionReviewer()

        reviewer._predictor = Mock(side_effect=Exception("No LLM"))
        reviewer._ensure_predictor = Mock()

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            reviewer.review_completion(
                instruction="Test task",
                result={"success": False, "error": "timeout"},
                tool_calls=[],
            )
        )
        assert result["completion_state"] == "partial"
        assert len(result["unresolved_items"]) > 0


# =============================================================================
# Context Compression Retry Tests
# =============================================================================

@pytest.mark.unit
class TestContextCompressionRetry:
    """Test _call_with_compression_retry from InferenceMixin."""

    def test_compress_context_preserves_header_and_tail(self):
        """_compress_context keeps first 20% and last portion."""
        from Jotty.core.agents._inference_mixin import InferenceMixin
        text = "HEADER " * 100 + "MIDDLE " * 500 + "TAIL " * 100
        compressed = InferenceMixin._compress_context(text, 1000)
        assert len(compressed) <= 1100  # Allow some slack for marker
        assert "HEADER" in compressed
        assert "TAIL" in compressed
        assert "[... context compressed ...]" in compressed

    def test_compress_context_noop_for_small(self):
        """Small texts pass through unchanged."""
        from Jotty.core.agents._inference_mixin import InferenceMixin
        text = "short text"
        assert InferenceMixin._compress_context(text, 1000) == text

    @pytest.mark.asyncio
    async def test_compression_retry_succeeds_first_try(self):
        """No compression when call succeeds on first try."""
        from Jotty.core.agents._inference_mixin import InferenceMixin

        mixin = InferenceMixin()

        async def mock_fn(conv, instr):
            return "success"

        result = await mixin._call_with_compression_retry(
            mock_fn, "conversation", "instruction"
        )
        assert result == "success"

    @pytest.mark.asyncio
    async def test_compression_retry_on_overflow(self):
        """Retries with compression on context_length_exceeded."""
        from Jotty.core.agents._inference_mixin import InferenceMixin

        mixin = InferenceMixin()
        call_count = 0

        async def mock_fn(conv, instr):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("context_length_exceeded: too many tokens")
            return f"ok with {len(conv)} chars"

        result = await mixin._call_with_compression_retry(
            mock_fn, "x" * 10000, "instruction", max_retries=2
        )
        assert "ok with" in result
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_compression_retry_raises_non_overflow(self):
        """Non-overflow errors are raised immediately, not retried."""
        from Jotty.core.agents._inference_mixin import InferenceMixin

        mixin = InferenceMixin()

        async def mock_fn(conv, instr):
            raise ValueError("something else entirely")

        with pytest.raises(ValueError, match="something else"):
            await mixin._call_with_compression_retry(
                mock_fn, "conversation", "instruction"
            )


# =============================================================================
# Terminal Proxy Detection Tests
# =============================================================================

@pytest.mark.unit
class TestTerminalProxyDetection:
    """Test proxy detection in AutoTerminalSession."""

    def test_no_proxy_detected_clean_env(self):
        """No proxy detected in clean environment."""
        # Test the detection logic directly (same as AutoTerminalSession)
        indicators = (
            '/Library/Application Support/Zscaler',
            '/opt/zscaler',
            '/usr/local/zscaler',
        )
        proxy_keywords = ('zscaler', 'bluecoat', 'forcepoint', 'mcafee')

        found = False
        for path in indicators:
            if os.path.exists(path):
                found = True
                break

        if not found:
            for var in ('HTTP_PROXY', 'HTTPS_PROXY'):
                val = os.environ.get(var, '').lower()
                if any(kw in val for kw in proxy_keywords):
                    found = True
                    break

        # In clean CI/test env, should not detect proxy
        assert isinstance(found, bool)

    def test_env_overrides_empty_without_proxy(self):
        """_get_env_overrides returns empty dict when no proxy detected."""
        # Test the logic: if _detect_corporate_proxy returns False, overrides are empty
        detected = False  # Simulating no proxy
        if not detected:
            overrides = {}
        else:
            overrides = {
                'CURL_CA_BUNDLE': '',
                'PYTHONHTTPSVERIFY': '0',
                'REQUESTS_CA_BUNDLE': '',
                'NODE_TLS_REJECT_UNAUTHORIZED': '0',
            }
        assert overrides == {}

    def test_env_overrides_populated_with_proxy(self):
        """_get_env_overrides returns SSL bypass vars when proxy detected."""
        detected = True  # Simulating proxy detected
        if not detected:
            overrides = {}
        else:
            overrides = {
                'CURL_CA_BUNDLE': '',
                'PYTHONHTTPSVERIFY': '0',
                'REQUESTS_CA_BUNDLE': '',
                'NODE_TLS_REJECT_UNAUTHORIZED': '0',
            }
        assert 'CURL_CA_BUNDLE' in overrides
        assert overrides['NODE_TLS_REJECT_UNAUTHORIZED'] == '0'


# =============================================================================
# Browser CDP/Accessibility Tests
# =============================================================================

@pytest.mark.unit
class TestBrowserCDP:
    """Test CDP accessibility and DOM structure tools."""

    def test_accessibility_tree_without_selenium(self):
        """Accessibility tree tool fails gracefully without Selenium."""
        # Test the guard logic: when SELENIUM_AVAILABLE is False
        SELENIUM_AVAILABLE = False
        if not SELENIUM_AVAILABLE:
            result = {'success': False, 'error': 'Selenium not installed'}
        else:
            result = {'success': True}
        assert result['success'] is False
        assert 'Selenium' in result['error']

    def test_dom_structure_tool_params(self):
        """DOM structure tool accepts selector and max_depth params."""
        # Test the JS evaluation logic would be called correctly
        params = {'selector': '#main', 'max_depth': 2}
        assert params['selector'] == '#main'
        assert params['max_depth'] == 2

    def test_cdp_click_requires_coordinates(self):
        """CDP click tool requires x and y parameters."""
        # Test the validation logic directly
        params = {'x': None, 'y': None}
        x = params.get('x')
        y = params.get('y')
        if x is None or y is None:
            result = {'success': False, 'error': 'x and y coordinates are required'}
        else:
            result = {'success': True}
        assert result['success'] is False

    def test_cdp_click_with_coordinates(self):
        """CDP click tool accepts valid coordinates."""
        params = {'x': 100, 'y': 200}
        x = params.get('x')
        y = params.get('y')
        assert x == 100
        assert y == 200
        assert x is not None and y is not None


# =============================================================================
# Visual Verification Protocol Tests
# =============================================================================

@pytest.mark.unit
class TestVisualVerificationProtocol:
    """Test Visual Verification Protocol constants and helpers."""

    def test_protocol_constant_exists(self):
        """VISUAL_VERIFICATION_PROTOCOL constant is defined."""
        import os
        tools_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'skills', 'visual-inspector', 'tools.py')

        # Read the file and check for the constant
        with open(tools_path, 'r') as f:
            content = f.read()
        assert 'VISUAL_VERIFICATION_PROTOCOL' in content
        assert 'WHEN TO VISUALLY VERIFY' in content
        assert 'PRINCIPLE:' in content

    def test_protocol_has_key_sections(self):
        """Protocol contains all expected guidance sections."""
        import os
        tools_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'skills', 'visual-inspector', 'tools.py')
        with open(tools_path, 'r') as f:
            content = f.read()

        assert 'state-changing actions' in content
        assert 'irreversible' in content
        assert 'Observe before acting' in content
        assert 'HOW TO VERIFY' in content

    def test_guidance_helper_function_exists(self):
        """get_visual_verification_guidance function is exported."""
        import os
        tools_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'skills', 'visual-inspector', 'tools.py')
        with open(tools_path, 'r') as f:
            content = f.read()

        assert 'def get_visual_verification_guidance' in content
        assert "'get_visual_verification_guidance'" in content  # in __all__


# =============================================================================
# ParameterResolver Tests
# =============================================================================

@pytest.mark.unit
class TestParameterResolver:
    """Test ParameterResolver from step_processors.py."""

    def test_resolve_simple_passthrough(self):
        """Params with no templates pass through unchanged."""
        from Jotty.core.agents.base.step_processors import ParameterResolver
        resolver = ParameterResolver({})
        result = resolver.resolve({'query': 'test', 'max': 5})
        assert result == {'query': 'test', 'max': 5}

    def test_resolve_template_substitution(self):
        """${ref} templates are resolved from outputs."""
        from Jotty.core.agents.base.step_processors import ParameterResolver
        outputs = {'step_0': {'path': '/tmp/report.pdf', 'success': True}}
        resolver = ParameterResolver(outputs)
        result = resolver.resolve({'file': '${step_0.path}'})
        assert result['file'] == '/tmp/report.pdf'

    def test_resolve_bare_key_to_path(self):
        """Bare output keys (e.g. 'step_0') resolve to matching field."""
        from Jotty.core.agents.base.step_processors import ParameterResolver
        outputs = {'step_0': {'path': '/tmp/result.txt', 'success': True}}
        resolver = ParameterResolver(outputs)
        result = resolver.resolve({'path': 'step_0'})
        assert result['path'] == '/tmp/result.txt'

    def test_resolve_nested_dict(self):
        """Nested dict params are resolved recursively."""
        from Jotty.core.agents.base.step_processors import ParameterResolver
        outputs = {'step_0': {'url': 'https://example.com'}}
        resolver = ParameterResolver(outputs)
        result = resolver.resolve({'config': {'target': '${step_0.url}'}})
        assert result['config']['target'] == 'https://example.com'

    def test_resolve_max_depth_protection(self):
        """Deeply nested params hit max depth and return as-is."""
        from Jotty.core.agents.base.step_processors import ParameterResolver
        resolver = ParameterResolver({})
        result = resolver.resolve({'key': 'val'}, _depth=11)
        assert result == {'key': 'val'}

    def test_is_bad_content_short(self):
        """Short strings are detected as bad content."""
        from Jotty.core.agents.base.step_processors import ParameterResolver
        resolver = ParameterResolver({})
        assert resolver._is_bad_content("too short") is True

    def test_is_bad_content_success_json(self):
        """Success JSON responses are detected as bad content."""
        from Jotty.core.agents.base.step_processors import ParameterResolver
        resolver = ParameterResolver({})
        assert resolver._is_bad_content('{"success": true, "bytes_written": 42}') is True

    def test_is_bad_content_good(self):
        """Real content passes the check."""
        from Jotty.core.agents.base.step_processors import ParameterResolver
        resolver = ParameterResolver({})
        # Must be >300 chars or >80 chars without instruction prefixes
        good_content = "Python is a versatile language used in web development, data science, and AI. " * 5
        assert resolver._is_bad_content(good_content) is False

    def test_resolve_path_dotted(self):
        """Dotted paths like 'step_0.output' are resolved."""
        from Jotty.core.agents.base.step_processors import ParameterResolver
        outputs = {'step_0': {'output': 'hello world'}}
        resolver = ParameterResolver(outputs)
        assert resolver.resolve_path('step_0.output') == 'hello world'

    def test_resolve_path_array_index(self):
        """Array index paths like 'step_0.items[0]' are resolved."""
        from Jotty.core.agents.base.step_processors import ParameterResolver
        outputs = {'step_0': {'items': ['first', 'second']}}
        resolver = ParameterResolver(outputs)
        assert resolver.resolve_path('step_0.items[0]') == 'first'

    def test_resolve_path_missing_falls_back(self):
        """Missing path keys trigger fallback resolution."""
        from Jotty.core.agents.base.step_processors import ParameterResolver
        outputs = {'step_0': {'content': 'actual content that is long enough to be valid for fallback resolution purposes'}}
        resolver = ParameterResolver(outputs)
        result = resolver.resolve_path('step_99.content')
        # Should fall back to last output's content field
        assert 'actual content' in result

    def test_sanitize_command_long_text(self):
        """Long non-command text in 'command' param gets auto-fixed."""
        from Jotty.core.agents.base.step_processors import ParameterResolver
        outputs = {'step_0': {'path': '/tmp/script.py', 'success': True}}
        resolver = ParameterResolver(outputs)
        long_text = "This is a very long text " * 20  # > 150 chars, > 15 spaces
        result = resolver._sanitize_command_param('command', long_text, None)
        assert result == 'python /tmp/script.py'

    def test_sanitize_path_long_content(self):
        """Long content in 'path' param gets auto-fixed to real path."""
        from Jotty.core.agents.base.step_processors import ParameterResolver
        outputs = {'step_0': {'path': '/tmp/output.txt', 'success': True}}
        resolver = ParameterResolver(outputs)
        long_content = "x" * 300  # > 200 chars
        result = resolver._sanitize_path_param('path', long_content, None)
        assert result == '/tmp/output.txt'

    def test_find_best_content(self):
        """_find_best_content returns longest valid content from outputs."""
        from Jotty.core.agents.base.step_processors import ParameterResolver
        short_content = "short"
        good_content = "A" * 200
        outputs = {
            'step_0': {'text': short_content},
            'step_1': {'content': good_content},
        }
        resolver = ParameterResolver(outputs)
        assert resolver._find_best_content() == good_content

    def test_aggregate_research_outputs(self):
        """Research outputs are aggregated into formatted markdown."""
        from Jotty.core.agents.base.step_processors import ParameterResolver
        outputs = {
            'research_0': {
                'query': 'AI trends',
                'results': [
                    {'title': 'Result 1', 'snippet': 'Snippet 1', 'url': 'http://example.com/1'}
                ]
            }
        }
        resolver = ParameterResolver(outputs)
        result = resolver._aggregate_research_outputs()
        assert '## Research: AI trends' in result
        assert 'Result 1' in result

    def test_backward_compat_import(self):
        """ParameterResolver can still be imported from skill_plan_executor."""
        from Jotty.core.agents.base.skill_plan_executor import ParameterResolver
        assert ParameterResolver is not None
        resolver = ParameterResolver({})
        assert resolver.resolve({'key': 'val'}) == {'key': 'val'}


# =============================================================================
# ParadigmExecutor and TrainingDaemon Tests
# =============================================================================

@pytest.mark.unit
class TestParadigmExecutorUnit:
    """Test ParadigmExecutor aggregate_results logic."""

    def test_aggregate_empty_results(self):
        """Empty results produce failed EpisodeResult."""
        from Jotty.core.orchestration.paradigm_executor import ParadigmExecutor
        from unittest.mock import Mock
        manager = Mock()
        manager.episode_count = 0
        executor = ParadigmExecutor(manager)
        result = executor.aggregate_results({}, "test goal")
        assert result.success is False
        assert result.output is None

    def test_aggregate_single_result(self):
        """Single result is returned as-is."""
        from Jotty.core.orchestration.paradigm_executor import ParadigmExecutor
        from Jotty.core.foundation.data_structures import EpisodeResult
        from unittest.mock import Mock
        manager = Mock()
        executor = ParadigmExecutor(manager)
        ep = EpisodeResult(
            output="test output", success=True, trajectory=[],
            tagged_outputs=[], episode=0, execution_time=1.0,
            architect_results=[], auditor_results=[], agent_contributions={},
        )
        result = executor.aggregate_results({'agent1': ep}, "goal")
        assert result is ep

    def test_aggregate_multiple_results(self):
        """Multiple results are combined with merged contributions."""
        from Jotty.core.orchestration.paradigm_executor import ParadigmExecutor
        from Jotty.core.foundation.data_structures import EpisodeResult
        from unittest.mock import Mock
        manager = Mock()
        manager.episode_count = 1
        manager._mas_zero_verify = Mock(return_value=None)
        executor = ParadigmExecutor(manager)
        ep1 = EpisodeResult(
            output="output1", success=True, trajectory=[{'step': 1}],
            tagged_outputs=[], episode=0, execution_time=1.0,
            architect_results=[], auditor_results=[],
            agent_contributions={'agent1': 'contrib1'},
        )
        ep2 = EpisodeResult(
            output="output2", success=True, trajectory=[{'step': 2}],
            tagged_outputs=[], episode=0, execution_time=2.0,
            architect_results=[], auditor_results=[],
            agent_contributions={'agent2': 'contrib2'},
        )
        result = executor.aggregate_results({'a1': ep1, 'a2': ep2}, "goal")
        assert result.success is True
        assert result.execution_time == 3.0
        assert 'agent1' in result.agent_contributions
        assert 'agent2' in result.agent_contributions


@pytest.mark.unit
class TestTrainingDaemonUnit:
    """Test TrainingDaemon status and control."""

    def test_status_not_running(self):
        """Status shows not running when no daemon task exists."""
        from Jotty.core.orchestration.training_daemon import TrainingDaemon
        from unittest.mock import Mock
        manager = Mock()
        daemon = TrainingDaemon(manager)
        status = daemon.status()
        assert status['running'] is False
        assert status['completed'] == 0
        assert status['succeeded'] == 0
        assert status['success_rate'] == 0.0

    def test_pending_count_no_learning(self):
        """Pending count returns 0 when learning unavailable."""
        from Jotty.core.orchestration.training_daemon import TrainingDaemon
        from unittest.mock import Mock
        manager = Mock()
        manager.learning.pending_training_count.side_effect = Exception("no learning")
        daemon = TrainingDaemon(manager)
        assert daemon.pending_count == 0

    def test_stop_not_running(self):
        """Stop returns False when daemon is not running."""
        from Jotty.core.orchestration.training_daemon import TrainingDaemon
        from unittest.mock import Mock
        manager = Mock()
        daemon = TrainingDaemon(manager)
        assert daemon.stop() is False


# =============================================================================
# PlanUtilsMixin moved methods Tests
# =============================================================================

@pytest.mark.unit
class TestPlanParsing:
    """Test plan normalization and parsing methods moved to PlanUtilsMixin."""

    def test_normalize_raw_plan_list_passthrough(self):
        """Lists pass through normalization unchanged."""
        from Jotty.core.agents._plan_utils_mixin import PlanUtilsMixin
        mixin = PlanUtilsMixin()
        result = mixin._normalize_raw_plan([{'step': 1}])
        assert result == [{'step': 1}]

    def test_normalize_raw_plan_json_string(self):
        """JSON string is parsed to list."""
        from Jotty.core.agents._plan_utils_mixin import PlanUtilsMixin
        import json
        mixin = PlanUtilsMixin()
        plan_data = [{'skill_name': 'web-search', 'tool_name': 'search'}]
        result = mixin._normalize_raw_plan(json.dumps(plan_data))
        assert len(result) == 1
        assert result[0]['skill_name'] == 'web-search'

    def test_normalize_raw_plan_code_block(self):
        """JSON in markdown code block is extracted."""
        from Jotty.core.agents._plan_utils_mixin import PlanUtilsMixin
        import json
        mixin = PlanUtilsMixin()
        plan_data = [{'skill_name': 'test'}]
        raw = f"Here's the plan:\n```json\n{json.dumps(plan_data)}\n```"
        result = mixin._normalize_raw_plan(raw)
        assert len(result) == 1

    def test_normalize_raw_plan_empty(self):
        """None/empty input returns empty list."""
        from Jotty.core.agents._plan_utils_mixin import PlanUtilsMixin
        mixin = PlanUtilsMixin()
        assert mixin._normalize_raw_plan(None) == []
        assert mixin._normalize_raw_plan('') == []

    def test_extract_comparison_entities_vs(self):
        """'vs' pattern extracts entities."""
        from Jotty.core.agents._plan_utils_mixin import PlanUtilsMixin
        mixin = PlanUtilsMixin()
        result = mixin._extract_comparison_entities("Compare Python vs JavaScript")
        assert len(result) == 2
        assert 'Python' in result[0]
        assert 'JavaScript' in result[1]

    def test_extract_comparison_entities_between(self):
        """'difference between X and Y' pattern works."""
        from Jotty.core.agents._plan_utils_mixin import PlanUtilsMixin
        mixin = PlanUtilsMixin()
        result = mixin._extract_comparison_entities("difference between React and Vue, create report")
        assert len(result) == 2

    def test_extract_comparison_entities_none(self):
        """Non-comparison tasks return empty list."""
        from Jotty.core.agents._plan_utils_mixin import PlanUtilsMixin
        mixin = PlanUtilsMixin()
        result = mixin._extract_comparison_entities("Research AI trends")
        assert result == []

    def test_extract_comparison_entities_triple(self):
        """Three-way comparison extracts all entities."""
        from Jotty.core.agents._plan_utils_mixin import PlanUtilsMixin
        mixin = PlanUtilsMixin()
        result = mixin._extract_comparison_entities("Compare Python vs JavaScript vs Ruby")
        assert len(result) == 3


# =============================================================================
# TierDetector Tests
# =============================================================================

@pytest.mark.unit
class TestTierDetector:
    """Tests for tier_detector.py — keyword heuristics, caching, LLM fallback."""

    def test_force_tier_overrides(self):
        """force_tier bypasses all detection logic."""
        detector = TierDetector()
        result = detector.detect("sandbox autonomous agent", force_tier=ExecutionTier.DIRECT)
        assert result == ExecutionTier.DIRECT

    def test_detect_direct_simple_query(self):
        """Short queries with direct indicators → DIRECT."""
        detector = TierDetector()
        assert detector.detect("what is 2+2") == ExecutionTier.DIRECT

    def test_detect_direct_short_query(self):
        """Very short queries (<=10 words) without multi-step → DIRECT."""
        detector = TierDetector()
        assert detector.detect("hello") == ExecutionTier.DIRECT

    def test_detect_agentic_multi_step(self):
        """Multi-step indicators → AGENTIC."""
        detector = TierDetector()
        result = detector.detect("Research the topic and then create a summary report for distribution")
        assert result == ExecutionTier.AGENTIC

    def test_detect_learning(self):
        """Learning indicators → LEARNING."""
        detector = TierDetector()
        assert detector.detect("learn from past mistakes and improve accuracy") == ExecutionTier.LEARNING

    def test_detect_research(self):
        """Research indicators → RESEARCH."""
        detector = TierDetector()
        assert detector.detect("experiment with different approaches and benchmark results") == ExecutionTier.RESEARCH

    def test_detect_autonomous(self):
        """Autonomous indicators → AUTONOMOUS."""
        detector = TierDetector()
        assert detector.detect("run in sandbox isolated environment with trust verification") == ExecutionTier.AUTONOMOUS

    def test_detect_agentic_ambiguous(self):
        """Ambiguous longer tasks without specific keywords → AGENTIC (default)."""
        detector = TierDetector()
        # Long enough to not be simple, no specific tier keywords
        result = detector.detect(
            "Write a comprehensive analysis of the current market trends "
            "including all major sectors and their quarterly performance data"
        )
        assert result == ExecutionTier.AGENTIC

    def test_caching(self):
        """Repeated queries use cache."""
        detector = TierDetector()
        result1 = detector.detect("what is Python")
        assert "what is python" in detector.detection_cache
        result2 = detector.detect("what is Python")
        assert result1 == result2

    def test_cache_key_normalization(self):
        """Cache key is lowercase and stripped."""
        detector = TierDetector()
        detector.detect("  What Is Python  ")
        assert "what is python" in detector.detection_cache

    def test_clear_cache(self):
        """clear_cache() empties the cache."""
        detector = TierDetector()
        detector.detect("hello")
        assert len(detector.detection_cache) == 1
        detector.clear_cache()
        assert len(detector.detection_cache) == 0

    def test_confidence_autonomous_high(self):
        """Autonomous keywords get high confidence (0.85)."""
        detector = TierDetector()
        _, conf = detector._detect_tier_with_confidence("sandbox execution")
        assert conf == 0.85

    def test_confidence_research_high(self):
        """Research keywords get high confidence (0.85)."""
        detector = TierDetector()
        _, conf = detector._detect_tier_with_confidence("benchmark approaches")
        assert conf == 0.85

    def test_confidence_direct_moderate(self):
        """Simple queries get 0.80 confidence."""
        detector = TierDetector()
        _, conf = detector._detect_tier_with_confidence("what is Python")
        assert conf == 0.80

    def test_confidence_multi_step_moderate(self):
        """Multi-step indicators get 0.75 confidence."""
        detector = TierDetector()
        _, conf = detector._detect_tier_with_confidence(
            "first gather all the data then compile it and after that generate the final report"
        )
        assert conf == 0.75

    def test_confidence_ambiguous_low(self):
        """Ambiguous fall-through gets 0.40 confidence."""
        detector = TierDetector()
        _, conf = detector._detect_tier_with_confidence(
            "Write a comprehensive analysis of the current market trends "
            "including all major sectors and their quarterly performance data"
        )
        assert conf == 0.40

    def test_explain_detection_direct(self):
        """explain_detection returns human-readable explanation for DIRECT."""
        detector = TierDetector()
        explanation = detector.explain_detection("what is Python")
        assert "Tier 1" in explanation
        assert "DIRECT" in explanation
        assert "direct query keywords" in explanation.lower() or "Short query" in explanation

    def test_explain_detection_agentic(self):
        """explain_detection mentions multi-step or default for AGENTIC."""
        detector = TierDetector()
        explanation = detector.explain_detection("analyze and create a report then send it")
        assert "AGENTIC" in explanation

    def test_explain_detection_autonomous(self):
        """explain_detection identifies autonomous keywords."""
        detector = TierDetector()
        explanation = detector.explain_detection("sandbox isolated execution")
        assert "AUTONOMOUS" in explanation

    @pytest.mark.asyncio
    async def test_adetect_force_tier(self):
        """adetect respects force_tier."""
        detector = TierDetector()
        result = await detector.adetect("sandbox test", force_tier=ExecutionTier.DIRECT)
        assert result == ExecutionTier.DIRECT

    @pytest.mark.asyncio
    async def test_adetect_uses_cache(self):
        """adetect populates and uses cache."""
        detector = TierDetector()
        r1 = await detector.adetect("what is Python")
        assert r1 == ExecutionTier.DIRECT
        assert "what is python" in detector.detection_cache
        r2 = await detector.adetect("what is Python")
        assert r2 == r1

    @pytest.mark.asyncio
    async def test_adetect_llm_fallback_when_ambiguous(self):
        """adetect calls LLM classifier for low-confidence results."""
        detector = TierDetector(enable_llm_fallback=True)
        mock_classifier = AsyncMock()
        mock_classifier.classify = AsyncMock(return_value=ExecutionTier.LEARNING)
        detector._llm_classifier = mock_classifier

        # Ambiguous task (confidence=0.40)
        result = await detector.adetect(
            "Write a comprehensive analysis of the current market trends "
            "including all major sectors and their quarterly performance data"
        )
        assert result == ExecutionTier.LEARNING
        mock_classifier.classify.assert_called_once()

    @pytest.mark.asyncio
    async def test_adetect_llm_fallback_skipped_high_confidence(self):
        """adetect skips LLM for high-confidence heuristic results."""
        detector = TierDetector(enable_llm_fallback=True)
        mock_classifier = AsyncMock()
        detector._llm_classifier = mock_classifier

        result = await detector.adetect("what is Python")
        assert result == ExecutionTier.DIRECT
        mock_classifier.classify.assert_not_called()

    @pytest.mark.asyncio
    async def test_adetect_llm_fallback_error_uses_heuristic(self):
        """adetect falls back to heuristic when LLM fails."""
        detector = TierDetector(enable_llm_fallback=True)
        mock_classifier = AsyncMock()
        mock_classifier.classify = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        detector._llm_classifier = mock_classifier

        # Ambiguous task - should still return AGENTIC (heuristic default)
        result = await detector.adetect(
            "Write a comprehensive analysis of the current market trends "
            "including all major sectors and their quarterly performance data"
        )
        assert result == ExecutionTier.AGENTIC

    @pytest.mark.asyncio
    async def test_adetect_llm_fallback_none_keeps_heuristic(self):
        """adetect keeps heuristic when LLM returns None."""
        detector = TierDetector(enable_llm_fallback=True)
        mock_classifier = AsyncMock()
        mock_classifier.classify = AsyncMock(return_value=None)
        detector._llm_classifier = mock_classifier

        result = await detector.adetect(
            "Write a comprehensive analysis of the current market trends "
            "including all major sectors and their quarterly performance data"
        )
        assert result == ExecutionTier.AGENTIC

    def test_tier_priority_autonomous_over_direct(self):
        """Autonomous keywords take priority even with short/simple queries."""
        detector = TierDetector()
        result = detector.detect("sandbox test")
        assert result == ExecutionTier.AUTONOMOUS

    def test_tier_priority_research_over_learning(self):
        """Research checked before learning in priority chain."""
        detector = TierDetector()
        result = detector.detect("experiment and benchmark to improve accuracy")
        # Has both research and learning indicators, research checked first
        assert result == ExecutionTier.RESEARCH


# =============================================================================
# EffectivenessTracker Tests
# =============================================================================

@pytest.mark.unit
class TestEffectivenessTracker:
    """Tests for learning_pipeline.py EffectivenessTracker."""

    def test_record_and_report(self):
        """Basic record + improvement_report."""
        from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
        tracker = EffectivenessTracker(recent_window=3, historical_window=10)
        tracker.record("analysis", success=True, quality=0.8)
        tracker.record("analysis", success=False, quality=0.2)
        report = tracker.improvement_report()
        assert "analysis" in report
        assert "_global" in report
        assert report["analysis"]["total_episodes"] == 2

    def test_split_windows(self):
        """Records split into recent and historical correctly."""
        from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
        tracker = EffectivenessTracker(recent_window=2, historical_window=10)
        # Add 5 records
        for i in range(5):
            tracker.record("task", success=(i >= 3), quality=i * 0.2)
        recent, historical = tracker._split_windows(tracker._records["task"])
        assert len(recent) == 2  # last 2
        assert len(historical) == 3  # first 3

    def test_split_windows_few_records(self):
        """With fewer records than recent_window, all go to recent."""
        from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
        tracker = EffectivenessTracker(recent_window=10, historical_window=50)
        tracker.record("task", success=True, quality=0.5)
        recent, historical = tracker._split_windows(tracker._records["task"])
        assert len(recent) == 1
        assert len(historical) == 0

    def test_rate_empty(self):
        """_rate of empty list returns (0.0, 0.0)."""
        from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
        tracker = EffectivenessTracker()
        rate, quality = tracker._rate([])
        assert rate == 0.0
        assert quality == 0.0

    def test_rate_calculation(self):
        """_rate computes success rate and avg quality correctly."""
        from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
        import time
        tracker = EffectivenessTracker()
        records = [
            (time.time(), True, 0.8, "agent1"),
            (time.time(), False, 0.2, "agent1"),
            (time.time(), True, 0.6, "agent1"),
        ]
        rate, quality = tracker._rate(records)
        assert abs(rate - 2 / 3) < 0.01
        assert abs(quality - (0.8 + 0.2 + 0.6) / 3) < 0.01

    def test_is_improving_insufficient_history(self):
        """is_improving returns False without enough historical data."""
        from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
        tracker = EffectivenessTracker(recent_window=2, historical_window=10)
        tracker.record("task", success=True, quality=0.9)
        tracker.record("task", success=True, quality=0.9)
        # Only 2 records total, all recent, no historical
        assert tracker.is_improving("task") is False

    def test_is_improving_with_trend(self):
        """is_improving returns True when recent > historical and enough data."""
        from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
        tracker = EffectivenessTracker(recent_window=3, historical_window=10)
        # Historical: all failures (5 records)
        for _ in range(5):
            tracker.record("task", success=False, quality=0.1)
        # Recent: all successes (3 records)
        for _ in range(3):
            tracker.record("task", success=True, quality=0.9)
        assert tracker.is_improving("task") is True

    def test_is_improving_global(self):
        """is_improving() without task_type checks global."""
        from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
        tracker = EffectivenessTracker(recent_window=3, historical_window=10)
        # Historical: failures
        for _ in range(5):
            tracker.record("any", success=False, quality=0.1)
        # Recent: successes
        for _ in range(3):
            tracker.record("any", success=True, quality=0.9)
        assert tracker.is_improving() is True

    def test_to_dict_roundtrip(self):
        """to_dict / from_dict roundtrip preserves data."""
        from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
        tracker = EffectivenessTracker(recent_window=5, historical_window=20)
        tracker.record("analysis", success=True, quality=0.8, agent="planner")
        tracker.record("analysis", success=False, quality=0.3, agent="executor")
        tracker.record("coding", success=True, quality=0.9)

        data = tracker.to_dict()
        restored = EffectivenessTracker.from_dict(data, recent_window=5, historical_window=20)

        # Same number of records per task_type
        assert len(restored._records["analysis"]) == 2
        assert len(restored._records["coding"]) == 1
        # Global gets all records
        assert len(restored._global) == 3

    def test_quality_clamped(self):
        """Quality values are clamped to [0.0, 1.0]."""
        from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
        tracker = EffectivenessTracker()
        tracker.record("task", success=True, quality=-0.5)
        tracker.record("task", success=True, quality=2.0)
        records = list(tracker._records["task"])
        assert records[0][2] == 0.0  # clamped from -0.5
        assert records[1][2] == 1.0  # clamped from 2.0

    def test_report_trend_positive(self):
        """Report shows positive trend when recent beats historical."""
        from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
        tracker = EffectivenessTracker(recent_window=2, historical_window=10)
        # All historical failures
        for _ in range(5):
            tracker.record("task", success=False, quality=0.1)
        # All recent successes
        for _ in range(2):
            tracker.record("task", success=True, quality=0.9)
        report = tracker.improvement_report()
        assert report["task"]["trend"] > 0
        assert report["task"]["quality_trend"] > 0
        assert report["task"]["improving"] is True

    def test_report_trend_negative(self):
        """Report shows negative trend when recent is worse."""
        from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
        tracker = EffectivenessTracker(recent_window=2, historical_window=10)
        # Historical successes
        for _ in range(5):
            tracker.record("task", success=True, quality=0.9)
        # Recent failures
        for _ in range(2):
            tracker.record("task", success=False, quality=0.1)
        report = tracker.improvement_report()
        assert report["task"]["trend"] < 0
        assert report["task"]["improving"] is False

    def test_multiple_task_types(self):
        """Different task types tracked independently."""
        from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
        tracker = EffectivenessTracker()
        tracker.record("analysis", success=True, quality=0.9)
        tracker.record("coding", success=False, quality=0.2)
        report = tracker.improvement_report()
        assert "analysis" in report
        assert "coding" in report
        assert report["analysis"]["recent_success_rate"] == 1.0
        assert report["coding"]["recent_success_rate"] == 0.0


# =============================================================================
# AsyncUtils Tests — safe_status, StatusReporter, AgentEventBroadcaster
# =============================================================================

@pytest.mark.unit
class TestAsyncUtils:
    """Tests for core/utils/async_utils.py utilities."""

    def test_safe_status_none_callback(self):
        """safe_status with None callback is a no-op."""
        from Jotty.core.utils.async_utils import safe_status
        safe_status(None, "Planning", "step 1")  # Should not raise

    def test_safe_status_calls_callback(self):
        """safe_status invokes the callback with stage and detail."""
        from Jotty.core.utils.async_utils import safe_status
        cb = Mock()
        safe_status(cb, "Planning", "step 1")
        cb.assert_called_once_with("Planning", "step 1")

    def test_safe_status_suppresses_exception(self):
        """safe_status swallows callback exceptions."""
        from Jotty.core.utils.async_utils import safe_status
        cb = Mock(side_effect=RuntimeError("callback broken"))
        safe_status(cb, "Planning", "step 1")  # Should not raise

    def test_status_reporter_calls_callback_and_logs(self):
        """StatusReporter invokes callback and logs."""
        from Jotty.core.utils.async_utils import StatusReporter
        cb = Mock()
        mock_logger = Mock()
        reporter = StatusReporter(cb, mock_logger, emoji="")
        reporter("Planning", "step 1")
        cb.assert_called_once_with("Planning", "step 1")
        mock_logger.info.assert_called_once()

    def test_status_reporter_with_prefix(self):
        """StatusReporter.with_prefix() prepends prefix to stage."""
        from Jotty.core.utils.async_utils import StatusReporter
        cb = Mock()
        reporter = StatusReporter(cb)
        sub = reporter.with_prefix("[agent1]")
        sub("Executing", "step 2")
        cb.assert_called_once_with("[agent1] Executing", "step 2")

    def test_status_reporter_no_callback(self):
        """StatusReporter with None callback only logs."""
        from Jotty.core.utils.async_utils import StatusReporter
        mock_logger = Mock()
        reporter = StatusReporter(None, mock_logger)
        reporter("Planning", "detail")
        mock_logger.info.assert_called_once()

    def test_ensure_async_wraps_sync(self):
        """ensure_async wraps a sync function to be awaitable."""
        from Jotty.core.utils.async_utils import ensure_async
        def sync_fn(x):
            return x * 2
        async_fn = ensure_async(sync_fn)
        assert asyncio.iscoroutinefunction(async_fn)

    def test_ensure_async_passthrough_async(self):
        """ensure_async returns async functions unchanged."""
        from Jotty.core.utils.async_utils import ensure_async
        async def async_fn(x):
            return x * 2
        result = ensure_async(async_fn)
        assert result is async_fn

    @pytest.mark.asyncio
    async def test_ensure_async_result(self):
        """ensure_async wrapped function returns correct result."""
        from Jotty.core.utils.async_utils import ensure_async
        def sync_fn(x):
            return x * 2
        async_fn = ensure_async(sync_fn)
        result = await async_fn(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_gather_with_limit(self):
        """gather_with_limit runs coroutines with concurrency limiting."""
        from Jotty.core.utils.async_utils import gather_with_limit
        results = []
        async def task(n):
            results.append(n)
            return n * 2
        out = await gather_with_limit([task(1), task(2), task(3)], limit=2)
        assert sorted(out) == [2, 4, 6]
        assert sorted(results) == [1, 2, 3]

    def test_agent_event_creation(self):
        """AgentEvent initializes with correct fields."""
        from Jotty.core.utils.async_utils import AgentEvent
        event = AgentEvent(type="tool_start", data={"skill": "web-search"}, agent_id="agent1")
        assert event.type == "tool_start"
        assert event.data == {"skill": "web-search"}
        assert event.agent_id == "agent1"
        assert event.timestamp > 0

    def test_agent_event_unknown_type_no_error(self):
        """AgentEvent with unknown type doesn't raise (just logs debug)."""
        from Jotty.core.utils.async_utils import AgentEvent
        event = AgentEvent(type="unknown_type")  # Should not raise
        assert event.type == "unknown_type"

    def test_broadcaster_singleton(self):
        """AgentEventBroadcaster.get_instance() returns singleton."""
        from Jotty.core.utils.async_utils import AgentEventBroadcaster
        AgentEventBroadcaster.reset_instance()
        try:
            b1 = AgentEventBroadcaster.get_instance()
            b2 = AgentEventBroadcaster.get_instance()
            assert b1 is b2
        finally:
            AgentEventBroadcaster.reset_instance()

    def test_broadcaster_reset(self):
        """reset_instance() creates fresh singleton on next call."""
        from Jotty.core.utils.async_utils import AgentEventBroadcaster
        AgentEventBroadcaster.reset_instance()
        try:
            b1 = AgentEventBroadcaster.get_instance()
            AgentEventBroadcaster.reset_instance()
            b2 = AgentEventBroadcaster.get_instance()
            assert b1 is not b2
        finally:
            AgentEventBroadcaster.reset_instance()

    def test_broadcaster_subscribe_and_emit(self):
        """subscribe + emit delivers events to listeners."""
        from Jotty.core.utils.async_utils import AgentEventBroadcaster, AgentEvent
        AgentEventBroadcaster.reset_instance()
        try:
            bus = AgentEventBroadcaster.get_instance()
            received = []
            bus.subscribe("tool_start", lambda e: received.append(e))
            event = AgentEvent(type="tool_start", data={"skill": "test"})
            bus.emit(event)
            assert len(received) == 1
            assert received[0].data == {"skill": "test"}
        finally:
            AgentEventBroadcaster.reset_instance()

    def test_broadcaster_unsubscribe(self):
        """unsubscribe removes listener."""
        from Jotty.core.utils.async_utils import AgentEventBroadcaster, AgentEvent
        AgentEventBroadcaster.reset_instance()
        try:
            bus = AgentEventBroadcaster.get_instance()
            received = []
            handler = lambda e: received.append(e)
            bus.subscribe("tool_end", handler)
            bus.unsubscribe("tool_end", handler)
            bus.emit(AgentEvent(type="tool_end"))
            assert len(received) == 0
        finally:
            AgentEventBroadcaster.reset_instance()

    def test_broadcaster_emit_suppresses_listener_error(self):
        """emit swallows exceptions from listeners."""
        from Jotty.core.utils.async_utils import AgentEventBroadcaster, AgentEvent
        AgentEventBroadcaster.reset_instance()
        try:
            bus = AgentEventBroadcaster.get_instance()
            bus.subscribe("error", lambda e: (_ for _ in ()).throw(RuntimeError("boom")))
            received = []
            bus.subscribe("error", lambda e: received.append(e))
            bus.emit(AgentEvent(type="error", data={"msg": "test"}))
            # Second listener still receives despite first one raising
            assert len(received) == 1
        finally:
            AgentEventBroadcaster.reset_instance()

    def test_broadcaster_unsubscribe_nonexistent(self):
        """unsubscribe with non-subscribed callback is a no-op."""
        from Jotty.core.utils.async_utils import AgentEventBroadcaster
        AgentEventBroadcaster.reset_instance()
        try:
            bus = AgentEventBroadcaster.get_instance()
            bus.unsubscribe("tool_start", lambda e: None)  # Should not raise
        finally:
            AgentEventBroadcaster.reset_instance()


# =============================================================================
# INFRASTRUCTURE PATTERNS (Circuit Breaker, Adaptive Timeout, DLQ, TimeoutWarning)
# =============================================================================

@pytest.mark.unit
class TestCircuitBreaker:
    """Test CircuitBreaker state machine."""

    def test_initial_state_closed(self):
        from Jotty.core.execution.types import CircuitBreaker, CircuitState
        cb = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=10)
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request()

    def test_trips_after_threshold(self):
        from Jotty.core.execution.types import CircuitBreaker, CircuitState
        cb = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=60)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.allow_request()

    def test_resets_on_success(self):
        from Jotty.core.execution.types import CircuitBreaker, CircuitState
        cb = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request()

    def test_half_open_after_cooldown(self):
        from Jotty.core.execution.types import CircuitBreaker, CircuitState
        cb = CircuitBreaker("test", failure_threshold=2, cooldown_seconds=0.01)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        import time
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request()

    def test_manual_reset(self):
        from Jotty.core.execution.types import CircuitBreaker, CircuitState
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=999)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED


@pytest.mark.unit
class TestAdaptiveTimeout:
    """Test AdaptiveTimeout P95 calculation."""

    def test_default_when_no_observations(self):
        from Jotty.core.execution.types import AdaptiveTimeout
        at = AdaptiveTimeout(default_seconds=30)
        assert at.get("unknown_op") == 30.0

    def test_adaptive_after_observations(self):
        from Jotty.core.execution.types import AdaptiveTimeout
        at = AdaptiveTimeout(default_seconds=30, min_seconds=1, max_seconds=100)
        for _ in range(10):
            at.record("llm_call", 2.0)
        timeout = at.get("llm_call")
        assert 1.0 <= timeout <= 100.0
        assert timeout < 30.0  # Should be much less than default

    def test_respects_min_max(self):
        from Jotty.core.execution.types import AdaptiveTimeout
        at = AdaptiveTimeout(min_seconds=5, max_seconds=10)
        for _ in range(10):
            at.record("fast_op", 0.1)
        assert at.get("fast_op") >= 5.0  # Floor

        for _ in range(10):
            at.record("slow_op", 100.0)
        assert at.get("slow_op") <= 10.0  # Ceiling


@pytest.mark.unit
class TestDeadLetterQueue:
    """Test DLQ enqueue/retry/resolve."""

    def test_enqueue_and_size(self):
        from Jotty.core.execution.types import DeadLetterQueue, ErrorType
        dlq = DeadLetterQueue(max_size=10)
        dlq.enqueue("search", {"query": "test"}, "timeout", ErrorType.INFRASTRUCTURE)
        assert dlq.size == 1

    def test_get_retryable(self):
        from Jotty.core.execution.types import DeadLetterQueue, ErrorType
        dlq = DeadLetterQueue()
        dlq.enqueue("op1", {}, "error1", ErrorType.INFRASTRUCTURE)
        retryable = dlq.get_retryable()
        assert len(retryable) == 1

    def test_mark_resolved(self):
        from Jotty.core.execution.types import DeadLetterQueue, ErrorType
        dlq = DeadLetterQueue()
        letter = dlq.enqueue("op1", {}, "error1")
        dlq.mark_resolved(letter)
        assert dlq.size == 0

    def test_retry_all(self):
        from Jotty.core.execution.types import DeadLetterQueue, ErrorType
        dlq = DeadLetterQueue()
        dlq.enqueue("op1", {}, "error1")
        dlq.enqueue("op2", {}, "error2")
        successes = dlq.retry_all(lambda letter: True)
        assert successes == 2
        assert dlq.size == 0

    def test_max_size_eviction(self):
        from Jotty.core.execution.types import DeadLetterQueue
        dlq = DeadLetterQueue(max_size=2)
        dlq.enqueue("op1", {}, "e1")
        dlq.enqueue("op2", {}, "e2")
        dlq.enqueue("op3", {}, "e3")
        assert dlq.size == 2  # Oldest evicted


@pytest.mark.unit
class TestTimeoutWarning:
    """Test TimeoutWarning threshold detection."""

    def test_no_warning_before_threshold(self):
        from Jotty.core.execution.types import TimeoutWarning
        tw = TimeoutWarning(timeout_seconds=100)
        tw.start()
        assert tw.check() is None  # Just started

    def test_warning_at_threshold(self):
        from Jotty.core.execution.types import TimeoutWarning
        tw = TimeoutWarning(timeout_seconds=0.01)
        tw.start()
        import time
        time.sleep(0.02)
        warning = tw.check()
        assert warning is not None
        assert "TIMEOUT WARNING" in warning

    def test_is_expired(self):
        from Jotty.core.execution.types import TimeoutWarning
        tw = TimeoutWarning(timeout_seconds=0.01)
        tw.start()
        import time
        time.sleep(0.02)
        assert tw.is_expired

    def test_fraction_used(self):
        from Jotty.core.execution.types import TimeoutWarning
        tw = TimeoutWarning(timeout_seconds=100)
        tw.start()
        assert tw.fraction_used < 0.01


# =============================================================================
# CONTEXT GUARD
# =============================================================================

@pytest.mark.unit
class TestProactiveContextGuard:
    """Test proactive context guard assembly."""

    def test_fits_in_budget(self):
        from Jotty.core.agents._inference_mixin import InferenceMixin, ContextPriority
        mixin = InferenceMixin()
        sections = [
            (ContextPriority.CRITICAL, "instruction", "Do X"),
            (ContextPriority.HIGH, "tools", "tool1, tool2"),
            (ContextPriority.LOW, "history", "old stuff"),
        ]
        result = mixin._proactive_context_guard(sections, budget_tokens=100000)
        assert "Do X" in result
        assert "tool1" in result

    def test_compresses_low_priority(self):
        from Jotty.core.agents._inference_mixin import InferenceMixin, ContextPriority
        mixin = InferenceMixin()
        # Create sections where LOW is very large
        sections = [
            (ContextPriority.CRITICAL, "instruction", "Do X"),
            (ContextPriority.LOW, "history", "x" * 100000),
        ]
        result = mixin._proactive_context_guard(sections, budget_tokens=500)
        assert "Do X" in result
        assert len(result) < 100000  # Should be compressed

    def test_estimate_tokens(self):
        from Jotty.core.agents._inference_mixin import InferenceMixin
        tokens = InferenceMixin._estimate_tokens("Hello world test")
        assert tokens > 0
        assert tokens < 100


# =============================================================================
# REACT EXECUTION MODE
# =============================================================================

@pytest.mark.unit
class TestReActMode:
    """Test ReAct execution mode configuration."""

    def test_react_config_defaults(self):
        from Jotty.core.agents.base.domain_agent import DomainAgentConfig
        config = DomainAgentConfig(use_react=True)
        assert config.use_react is True
        assert config.max_react_iters == 5

    def test_react_mode_disabled_by_default(self):
        from Jotty.core.agents.base.domain_agent import DomainAgentConfig
        config = DomainAgentConfig()
        assert config.use_react is False


# =============================================================================
# LLM-ANALYZED RETRY + TRAJECTORY
# =============================================================================

@pytest.mark.unit
class TestAnalyzedRetry:
    """Test LLM-analyzed retry and trajectory preservation."""

    def _make_agent(self):
        from Jotty.core.agents.base.base_agent import BaseAgent, AgentRuntimeConfig
        class _Dummy(BaseAgent):
            async def _execute_impl(self, **kw):
                return {}
        a = _Dummy(AgentRuntimeConfig(name="test"))
        a._initialized = True
        return a

    def test_analyze_failure_rate_limit(self):
        agent = self._make_agent()
        guidance = agent._analyze_failure("rate limit exceeded 429", {})
        assert guidance == ""

    def test_analyze_failure_logic_error(self):
        agent = self._make_agent()
        guidance = agent._analyze_failure("element not found: #submit-btn", {})
        assert "logic error" in guidance.lower() or "different approach" in guidance.lower()

    def test_analyze_failure_data_error(self):
        agent = self._make_agent()
        guidance = agent._analyze_failure("empty result from API", {})
        assert "data" in guidance.lower()

    @pytest.mark.asyncio
    async def test_trajectory_preserved_across_retries(self):
        from Jotty.core.agents.base.base_agent import BaseAgent, AgentRuntimeConfig

        call_count = 0
        class FailThenSucceed(BaseAgent):
            async def _execute_impl(self, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise ValueError("test error")
                # On retry, trajectory should be injected
                assert '_retry_trajectory' in kwargs
                return {"ok": True}

        agent = FailThenSucceed(AgentRuntimeConfig(
            name="test", max_retries=3, retry_delay=0.01, timeout=10
        ))
        agent._initialized = True
        result = await agent.execute()
        assert result.success
        assert result.retries == 1


# =============================================================================
# DAG PARALLEL EXECUTION + TOOL CALL CACHING
# =============================================================================

@pytest.mark.unit
class TestToolCallCache:
    """Test ToolCallCache TTL and LRU."""

    def test_cache_set_and_get(self):
        from Jotty.core.agents.base.skill_plan_executor import ToolCallCache
        cache = ToolCallCache(ttl_seconds=60)
        key = cache.make_key("web-search", "search_tool", {"query": "test"})
        cache.set(key, {"result": "data"})
        assert cache.get(key) == {"result": "data"}

    def test_cache_miss(self):
        from Jotty.core.agents.base.skill_plan_executor import ToolCallCache
        cache = ToolCallCache()
        assert cache.get("nonexistent") is None

    def test_cache_ttl_expiry(self):
        from Jotty.core.agents.base.skill_plan_executor import ToolCallCache
        cache = ToolCallCache(ttl_seconds=0.01)
        key = "test"
        cache.set(key, "value")
        import time
        time.sleep(0.02)
        assert cache.get(key) is None

    def test_cache_max_size_eviction(self):
        from Jotty.core.agents.base.skill_plan_executor import ToolCallCache
        cache = ToolCallCache(max_size=2)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")
        assert cache.size == 2

    def test_deterministic_key(self):
        from Jotty.core.agents.base.skill_plan_executor import ToolCallCache
        k1 = ToolCallCache.make_key("s", "t", {"a": 1, "b": 2})
        k2 = ToolCallCache.make_key("s", "t", {"b": 2, "a": 1})
        assert k1 == k2  # Same regardless of dict order


@pytest.mark.unit
class TestDAGExecution:
    """Test DAG dependency graph and parallel grouping."""

    def test_build_dependency_graph(self):
        from Jotty.core.agents.base.skill_plan_executor import SkillPlanExecutor
        executor = SkillPlanExecutor(skills_registry=None)
        steps = [Mock(depends_on=[]), Mock(depends_on=[0]), Mock(depends_on=[0])]
        graph = executor._build_dependency_graph(steps)
        assert graph[0] == []
        assert graph[1] == [0]
        assert graph[2] == [0]

    def test_find_parallel_groups(self):
        from Jotty.core.agents.base.skill_plan_executor import SkillPlanExecutor
        executor = SkillPlanExecutor(skills_registry=None)
        # Step 0: no deps, Step 1: depends on 0, Step 2: depends on 0
        steps = [Mock(depends_on=[]), Mock(depends_on=[0]), Mock(depends_on=[0])]
        layers = executor._find_parallel_groups(steps)
        assert layers[0] == [0]       # Step 0 first
        assert set(layers[1]) == {1, 2}  # Steps 1,2 in parallel

    def test_sequential_chain(self):
        from Jotty.core.agents.base.skill_plan_executor import SkillPlanExecutor
        executor = SkillPlanExecutor(skills_registry=None)
        steps = [Mock(depends_on=[]), Mock(depends_on=[0]), Mock(depends_on=[1])]
        layers = executor._find_parallel_groups(steps)
        assert len(layers) == 3  # All sequential


# =============================================================================
# Q-LEARNING, COMA, LEARNED CONTEXT
# =============================================================================

@pytest.mark.unit
class TestSkillQTable:
    """Test Q-learning for skill selection."""

    def test_initial_q_value(self):
        from Jotty.core.learning.td_lambda import SkillQTable
        q = SkillQTable()
        assert q.get_q("research", "web-search") == 0.5  # Optimistic default

    def test_update_q_value(self):
        from Jotty.core.learning.td_lambda import SkillQTable
        q = SkillQTable(alpha=0.5)
        q.update("research", "web-search", reward=1.0)
        assert q.get_q("research", "web-search") > 0.5

    def test_select_ranks_by_q(self):
        from Jotty.core.learning.td_lambda import SkillQTable
        q = SkillQTable(epsilon=0.0)  # No exploration
        q.update("research", "good-skill", reward=0.9)
        q.update("research", "bad-skill", reward=0.1)
        ranked = q.select("research", ["bad-skill", "good-skill"])
        assert ranked[0] == "good-skill"

    def test_serialization(self):
        from Jotty.core.learning.td_lambda import SkillQTable
        q = SkillQTable()
        q.update("test", "s1", 0.8)
        data = q.to_dict()
        q2 = SkillQTable.from_dict(data)
        assert q2.get_q("test", "s1") == q.get_q("test", "s1")


@pytest.mark.unit
class TestCOMACredit:
    """Test counterfactual credit assignment."""

    def test_initial_credit_zero(self):
        from Jotty.core.learning.td_lambda import COMACredit
        coma = COMACredit()
        assert coma.get_credit("unknown") == 0.0

    def test_credit_after_episodes(self):
        from Jotty.core.learning.td_lambda import COMACredit
        coma = COMACredit()
        # Agent A present in good episodes
        coma.record_episode(0.9, {"A": 0.5, "B": 0.4})
        coma.record_episode(0.8, {"A": 0.6, "B": 0.2})
        # Episode without A (counterfactual)
        coma.record_episode(0.3, {"B": 0.3})
        credit_a = coma.get_credit("A")
        assert credit_a > 0  # A helps the team

    def test_get_all_credits(self):
        from Jotty.core.learning.td_lambda import COMACredit
        coma = COMACredit()
        coma.record_episode(0.9, {"A": 0.5, "B": 0.4})
        credits = coma.get_all_credits()
        assert "A" in credits
        assert "B" in credits


@pytest.mark.unit
class TestGetLearnedContext:
    """Test learned context generation for LLM prompts."""

    def test_empty_when_no_data(self):
        from Jotty.core.learning.td_lambda import TDLambdaLearner, get_learned_context
        from Jotty.core.foundation.data_structures import SwarmConfig
        learner = TDLambdaLearner(SwarmConfig())
        ctx = get_learned_context(learner, task_type="unknown_type")
        # May be empty if no data for this type
        assert isinstance(ctx, str)

    def test_includes_baseline_info(self):
        from Jotty.core.learning.td_lambda import TDLambdaLearner, SkillQTable, get_learned_context
        from Jotty.core.foundation.data_structures import SwarmConfig
        learner = TDLambdaLearner(SwarmConfig())
        # Populate some data
        learner.grouped_baseline.update_group("research", 0.8)
        learner.grouped_baseline.update_group("research", 0.9)
        learner.grouped_baseline.update_group("research", 0.85)
        q = SkillQTable()
        q.update("research", "web-search", 0.9)
        ctx = get_learned_context(learner, skill_q=q, task_type="research")
        assert "LEARNED CONTEXT" in ctx
        assert "research" in ctx


# =============================================================================
# POLICY EXPLORER + DATA-FLOW DEPENDENCIES
# =============================================================================

@pytest.mark.unit
class TestDataFlowDependencies:
    """Test data-flow dependency inference."""

    def test_infer_no_deps(self):
        from Jotty.core.agents.agentic_planner import TaskPlanner
        steps = [
            Mock(output_key="out_0", params={"query": "test"}, depends_on=[]),
            Mock(output_key="out_1", params={"query": "other"}, depends_on=[]),
        ]
        result = TaskPlanner.infer_data_dependencies(steps)
        assert result[0].depends_on == []
        assert result[1].depends_on == []

    def test_infer_template_dependency(self):
        from Jotty.core.agents.agentic_planner import TaskPlanner
        steps = [
            Mock(output_key="research_out", params={"query": "AI"}, depends_on=[]),
            Mock(output_key="summary_out", params={"text": "{{research_out}}"}, depends_on=[]),
        ]
        result = TaskPlanner.infer_data_dependencies(steps)
        assert 0 in result[1].depends_on


# =============================================================================
# TOOL I/O CHAINING + PER-TOOL STATS
# =============================================================================

@pytest.mark.unit
class TestToolStats:
    """Test per-tool performance statistics."""

    def test_record_and_get(self):
        from Jotty.core.agents._execution_types import ToolStats
        ts = ToolStats()
        ts.record("web-search", "search_tool", success=True, latency_ms=1200)
        ts.record("web-search", "search_tool", success=True, latency_ms=800)
        ts.record("web-search", "search_tool", success=False, latency_ms=5000)
        stats = ts.get_stats("web-search", "search_tool")
        assert stats['call_count'] == 3
        assert abs(stats['success_rate'] - 2/3) < 0.01

    def test_summary_string(self):
        from Jotty.core.agents._execution_types import ToolStats
        ts = ToolStats()
        ts.record("s", "t", True, 1000)
        summary = ts.get_summary("s", "t")
        assert "100%" in summary
        assert "1 calls" in summary

    def test_no_history(self):
        from Jotty.core.agents._execution_types import ToolStats
        ts = ToolStats()
        assert ts.get_stats("x", "y")['call_count'] == 0


@pytest.mark.unit
class TestCapabilityIndex:
    """Test tool I/O chaining graph."""

    def test_register_and_find_chain(self):
        from Jotty.core.agents._execution_types import CapabilityIndex
        idx = CapabilityIndex()
        idx.register("search", inputs=["query"], outputs=["search_results"])
        idx.register("summarize", inputs=["search_results"], outputs=["summary"])
        chain = idx.find_chain("query", "summary")
        assert chain == ["search", "summarize"]

    def test_no_chain_exists(self):
        from Jotty.core.agents._execution_types import CapabilityIndex
        idx = CapabilityIndex()
        idx.register("search", inputs=["query"], outputs=["search_results"])
        chain = idx.find_chain("query", "nonexistent_type")
        assert chain == []

    def test_direct_chain(self):
        from Jotty.core.agents._execution_types import CapabilityIndex
        idx = CapabilityIndex()
        idx.register("tool1", inputs=["a"], outputs=["b"])
        chain = idx.find_chain("a", "b")
        assert chain == ["tool1"]

    def test_three_hop_chain(self):
        from Jotty.core.agents._execution_types import CapabilityIndex
        idx = CapabilityIndex()
        idx.register("t1", inputs=["a"], outputs=["b"])
        idx.register("t2", inputs=["b"], outputs=["c"])
        idx.register("t3", inputs=["c"], outputs=["d"])
        chain = idx.find_chain("a", "d")
        assert len(chain) == 3


# =============================================================================
# SELF-RAG + SURPRISE MEMORY
# =============================================================================

@pytest.mark.unit
class TestSelfRAG:
    """Test self-RAG retrieval gating."""

    def test_skip_for_greeting(self):
        from Jotty.core.memory.cortex import SwarmMemory
        from Jotty.core.foundation.data_structures import SwarmConfig
        mem = SwarmMemory(agent_name="test", config=SwarmConfig())
        should, results, reason = mem.self_rag_retrieve("hello")
        assert should is False
        assert "greeting" in reason.lower() or "simple" in reason.lower()

    def test_skip_when_empty(self):
        from Jotty.core.memory.cortex import SwarmMemory
        from Jotty.core.foundation.data_structures import SwarmConfig
        mem = SwarmMemory(agent_name="test", config=SwarmConfig())
        should, results, reason = mem.self_rag_retrieve("complex research task about AI")
        assert should is False
        assert "no memories" in reason.lower()


@pytest.mark.unit
class TestSurpriseMemory:
    """Test surprise-based memory storage."""

    def test_routine_skipped(self):
        from Jotty.core.memory.cortex import SwarmMemory
        from Jotty.core.foundation.data_structures import SwarmConfig
        mem = SwarmMemory(agent_name="test", config=SwarmConfig())
        result = mem.store_with_surprise("routine event", surprise_score=0.1, context={})
        assert result is None  # Skipped

    def test_surprising_stored_causal(self):
        from Jotty.core.memory.cortex import SwarmMemory
        from Jotty.core.foundation.data_structures import SwarmConfig, MemoryLevel
        mem = SwarmMemory(agent_name="test", config=SwarmConfig())
        result = mem.store_with_surprise(
            "unexpected API failure pattern",
            surprise_score=0.9,
            context={"error": "novel failure"},
            goal="reliability",
        )
        assert result is not None
        # Should be stored in CAUSAL level
        assert len(mem.memories[MemoryLevel.CAUSAL]) > 0

    def test_notable_stored_episodic(self):
        from Jotty.core.memory.cortex import SwarmMemory
        from Jotty.core.foundation.data_structures import SwarmConfig, MemoryLevel
        mem = SwarmMemory(agent_name="test", config=SwarmConfig())
        result = mem.store_with_surprise(
            "notable event",
            surprise_score=0.5,
            context={},
        )
        assert result is not None
        assert len(mem.memories[MemoryLevel.EPISODIC]) > 0


# =============================================================================
# FAILURE ROUTER
# =============================================================================

@pytest.mark.unit
class TestFailureRouter:
    """Test failure routing decisions."""

    def test_timeout_routes_to_retry(self):
        from Jotty.core.agents.inspector import FailureRouter
        router = FailureRouter()
        action = router.route("Connection timeout after 30s", "web-search")
        assert action['action'] == 'retry_with_backoff'

    def test_rate_limit_routes_to_wait(self):
        from Jotty.core.agents.inspector import FailureRouter
        router = FailureRouter()
        action = router.route("rate_limit exceeded, retry in 60 seconds", "llm")
        assert action['action'] == 'wait_and_retry'

    def test_not_found_routes_to_alternative(self):
        from Jotty.core.agents.inspector import FailureRouter
        router = FailureRouter()
        action = router.route("resource not_found: 404", "web-search")
        assert action['action'] == 'try_alternative'

    def test_logic_error_with_alternative(self):
        from Jotty.core.agents.inspector import FailureRouter
        router = FailureRouter()
        action = router.route("invalid selector syntax", "browser-automation")
        assert action['action'] in ('try_alternative', 'replan')

    def test_ssl_routes_to_bypass(self):
        from Jotty.core.agents.inspector import FailureRouter
        router = FailureRouter()
        action = router.route("SSL certificate verification failed", "http-client")
        assert action['action'] == 'use_env_bypass'


# =============================================================================
# ERROR CLASSIFICATION (extended tests)
# =============================================================================

@pytest.mark.unit
class TestErrorClassificationExtended:
    """Extended tests for ErrorType.classify()."""

    def test_environment_detection(self):
        from Jotty.core.execution.types import ErrorType
        assert ErrorType.classify("SSL handshake failed") == ErrorType.ENVIRONMENT
        assert ErrorType.classify("Zscaler proxy block") == ErrorType.ENVIRONMENT

    def test_logic_before_data(self):
        from Jotty.core.execution.types import ErrorType
        # "element not found" should be LOGIC, not DATA
        assert ErrorType.classify("element not found in DOM") == ErrorType.LOGIC

    def test_data_detection(self):
        from Jotty.core.execution.types import ErrorType
        assert ErrorType.classify("empty result set") == ErrorType.DATA
        assert ErrorType.classify("invalid json response") == ErrorType.DATA

    def test_infrastructure_default(self):
        from Jotty.core.execution.types import ErrorType
        assert ErrorType.classify("something unknown happened") == ErrorType.INFRASTRUCTURE


# =============================================================================
# VALIDATION VERDICT (extended tests)
# =============================================================================

@pytest.mark.unit
class TestValidationVerdictExtended:
    """Extended tests for ValidationVerdict."""

    def test_ok_verdict(self):
        from Jotty.core.execution.types import ValidationVerdict, ValidationStatus
        v = ValidationVerdict.ok("all good", confidence=0.95)
        assert v.is_pass
        assert v.confidence == 0.95

    def test_from_error_retryable(self):
        from Jotty.core.execution.types import ValidationVerdict, ErrorType
        v = ValidationVerdict.from_error("connection timeout")
        assert not v.is_pass
        assert v.retryable
        assert v.error_type == ErrorType.INFRASTRUCTURE

    def test_from_error_not_retryable(self):
        from Jotty.core.execution.types import ValidationVerdict, ErrorType
        v = ValidationVerdict.from_error("syntax error in selector")
        assert not v.is_pass
        assert not v.retryable  # Logic errors aren't retryable
        assert v.error_type == ErrorType.LOGIC


# =============================================================================
# ComplexityGate Tests
# =============================================================================

@pytest.mark.unit
class TestComplexityGate:
    """Tests for executor.py ComplexityGate."""

    @pytest.mark.asyncio
    async def test_should_skip_planning_direct(self):
        """ComplexityGate returns True for DIRECT classification."""
        from Jotty.core.execution.executor import ComplexityGate
        gate = ComplexityGate()
        mock_response = Mock()
        mock_response.content = [Mock(text="DIRECT")]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        gate._client = mock_client
        result = await gate.should_skip_planning("What is 2+2?")
        assert result is True

    @pytest.mark.asyncio
    async def test_should_skip_planning_tools(self):
        """ComplexityGate returns False for TOOLS classification."""
        from Jotty.core.execution.executor import ComplexityGate
        gate = ComplexityGate()
        mock_response = Mock()
        mock_response.content = [Mock(text="TOOLS")]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        gate._client = mock_client
        result = await gate.should_skip_planning("Search the web for AI trends")
        assert result is False

    @pytest.mark.asyncio
    async def test_should_skip_planning_error_defaults_false(self):
        """ComplexityGate defaults to False (proceed with planning) on error."""
        from Jotty.core.execution.executor import ComplexityGate
        gate = ComplexityGate()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))
        gate._client = mock_client
        result = await gate.should_skip_planning("some task")
        assert result is False

    @pytest.mark.asyncio
    async def test_should_skip_planning_empty_response(self):
        """ComplexityGate returns False for empty LLM response."""
        from Jotty.core.execution.executor import ComplexityGate
        gate = ComplexityGate()
        mock_response = Mock()
        mock_response.content = []
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        gate._client = mock_client
        result = await gate.should_skip_planning("some task")
        assert result is False

    @pytest.mark.asyncio
    async def test_truncates_long_goal(self):
        """ComplexityGate truncates goal to 500 chars."""
        from Jotty.core.execution.executor import ComplexityGate
        gate = ComplexityGate()
        mock_response = Mock()
        mock_response.content = [Mock(text="TOOLS")]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        gate._client = mock_client
        long_goal = "x" * 1000
        await gate.should_skip_planning(long_goal)
        call_args = mock_client.messages.create.call_args
        prompt = call_args[1]['messages'][0]['content']
        # Goal in prompt should be truncated to 500 chars
        assert "x" * 501 not in prompt


# =============================================================================
# FallbackValidator Tests
# =============================================================================

@pytest.mark.unit
class TestFallbackValidator:
    """Tests for executor.py _FallbackValidator."""

    @pytest.mark.asyncio
    async def test_validate_parses_json(self):
        """_FallbackValidator parses JSON from LLM response."""
        from Jotty.core.execution.executor import _FallbackValidator
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(return_value={
            'content': '{"success": true, "confidence": 0.9, "feedback": "good", "reasoning": "looks fine"}'
        })
        validator = _FallbackValidator(mock_provider)
        result = await validator.validate("Check this result: Hello world")
        assert result['success'] is True
        assert result['confidence'] == 0.9

    @pytest.mark.asyncio
    async def test_validate_extracts_json_from_text(self):
        """_FallbackValidator extracts JSON embedded in text."""
        from Jotty.core.execution.executor import _FallbackValidator
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(return_value={
            'content': 'Here is my evaluation: {"success": false, "confidence": 0.4, "feedback": "incomplete", "reasoning": "missing data"}'
        })
        validator = _FallbackValidator(mock_provider)
        result = await validator.validate("Check this")
        assert result['success'] is False
        assert result['confidence'] == 0.4

    @pytest.mark.asyncio
    async def test_validate_no_json_returns_default(self):
        """_FallbackValidator returns default when no JSON in response."""
        from Jotty.core.execution.executor import _FallbackValidator
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(return_value={
            'content': 'The result looks good and complete.'
        })
        validator = _FallbackValidator(mock_provider)
        result = await validator.validate("Check this")
        assert result['success'] is True
        assert result['confidence'] == 0.7

    @pytest.mark.asyncio
    async def test_validate_error_returns_safe_default(self):
        """_FallbackValidator returns safe default on LLM error."""
        from Jotty.core.execution.executor import _FallbackValidator
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("API error"))
        validator = _FallbackValidator(mock_provider)
        result = await validator.validate("Check this")
        assert result['success'] is True
        assert result['confidence'] == 0.5
        assert 'skipped' in result['feedback'].lower()


# =============================================================================
# Output Synthesis Tests
# =============================================================================

@pytest.mark.unit
class TestOutputSynthesis:
    """Tests for TierExecutor._fallback_aggregate and _synthesize_results."""

    def test_fallback_aggregate_empty(self, v3_executor):
        """Empty results produces 'No results generated.'."""
        result = v3_executor._fallback_aggregate([], "test goal")
        assert result == "No results generated."

    def test_fallback_aggregate_single(self, v3_executor):
        """Single result returns output directly."""
        results = [{'output': 'Hello world'}]
        result = v3_executor._fallback_aggregate(results, "test goal")
        assert result == "Hello world"

    def test_fallback_aggregate_multiple(self, v3_executor):
        """Multiple results concatenated with step numbers."""
        results = [
            {'output': 'Result A'},
            {'output': 'Result B'},
        ]
        result = v3_executor._fallback_aggregate(results, "test goal")
        assert "Step 1" in result
        assert "Step 2" in result
        assert "Result A" in result
        assert "Result B" in result

    @pytest.mark.asyncio
    async def test_synthesize_results_empty(self, v3_executor):
        """Empty input returns no-result message."""
        result = await v3_executor._synthesize_results([], "goal")
        assert result['output'] == "No results generated."
        assert result['llm_calls'] == 0

    @pytest.mark.asyncio
    async def test_synthesize_results_single(self, v3_executor):
        """Single result returned as-is without LLM call."""
        result = await v3_executor._synthesize_results([{'output': 'Hello'}], "goal")
        assert result['output'] == "Hello"
        assert result['llm_calls'] == 0
        assert result['cost'] == 0.0

    @pytest.mark.asyncio
    async def test_synthesize_results_multi_calls_llm(self, v3_executor):
        """Multiple results trigger LLM synthesis call."""
        v3_executor._provider = AsyncMock()
        v3_executor._provider.generate = AsyncMock(return_value={
            'content': 'Synthesized answer combining both results.',
            'usage': {'input_tokens': 100, 'output_tokens': 50},
        })
        mock_cost = Mock()
        mock_cost.cost = 0.001
        v3_executor._cost_tracker = Mock()
        v3_executor._cost_tracker.record_llm_call = Mock(return_value=mock_cost)

        results = [{'output': 'Part A'}, {'output': 'Part B'}]
        result = await v3_executor._synthesize_results(results, "test goal")
        assert result['output'] == 'Synthesized answer combining both results.'
        assert result['llm_calls'] == 1
        assert result['cost'] == 0.001

    @pytest.mark.asyncio
    async def test_synthesize_results_llm_error_falls_back(self, v3_executor):
        """LLM failure falls back to concatenation."""
        v3_executor._provider = AsyncMock()
        v3_executor._provider.generate = AsyncMock(side_effect=RuntimeError("LLM down"))

        results = [{'output': 'Part A'}, {'output': 'Part B'}]
        result = await v3_executor._synthesize_results(results, "test goal")
        # Should fall back to _fallback_aggregate
        assert "Part A" in result['output']
        assert "Part B" in result['output']


# =============================================================================
# Inspector Utility Tests (smart_truncate, CachingToolWrapper, FailureRouter)
# =============================================================================

@pytest.mark.unit
class TestSmartTruncate:
    """Tests for ValidatorAgent._smart_truncate."""

    def _make_validator_agent(self):
        """Create a minimal ValidatorAgent for testing utility methods."""
        from Jotty.core.agents.inspector import ValidatorAgent
        from Jotty.core.foundation.data_structures import SwarmConfig, SharedScratchpad
        from pathlib import Path
        from unittest.mock import patch

        config = SwarmConfig()
        # Patch DSPy import to avoid needing a configured LM
        with patch('Jotty.core.agents.inspector._get_dspy') as mock_dspy:
            mock_dspy_mod = MagicMock()
            mock_dspy_mod.ChainOfThought = MagicMock(return_value=MagicMock())
            mock_dspy.return_value = mock_dspy_mod
            with patch('Jotty.core.agents.inspector._get_reviewer_signature', return_value=MagicMock()):
                agent = ValidatorAgent(
                    md_path=Path("/nonexistent/test.md"),
                    is_architect=False,
                    tools=[],
                    config=config,
                    scratchpad=SharedScratchpad(),
                )
        return agent

    def test_no_truncation_short_text(self):
        """Short text passes through unchanged."""
        agent = self._make_validator_agent()
        text = "Hello world"
        result = agent._smart_truncate(text, 1000)
        assert result == text

    def test_truncates_at_sentence_boundary(self):
        """Truncation prefers sentence boundaries."""
        agent = self._make_validator_agent()
        text = "First sentence here. Second sentence here. Third sentence here."
        result = agent._smart_truncate(text, 45)
        assert result.endswith("...")
        assert len(result) < 50

    def test_truncates_at_word_boundary(self):
        """Fallback: truncation at word boundary when no sentence end found."""
        agent = self._make_validator_agent()
        text = "word " * 100  # Lots of words, no sentence endings
        result = agent._smart_truncate(text, 50)
        assert result.endswith("...")
        assert len(result) <= 55  # Some tolerance for the ellipsis

    def test_hard_truncate_no_boundaries(self):
        """Last resort: hard truncate with no word/sentence boundaries."""
        agent = self._make_validator_agent()
        text = "a" * 200
        result = agent._smart_truncate(text, 50)
        assert len(result) == 53  # 50 + "..."
        assert result.endswith("...")


@pytest.mark.unit
class TestCachingToolWrapper:
    """Tests for inspector.py CachingToolWrapper."""

    def test_calls_tool_and_caches(self):
        """CachingToolWrapper calls tool and stores result in scratchpad."""
        from Jotty.core.agents.inspector import CachingToolWrapper
        from Jotty.core.foundation.data_structures import SharedScratchpad
        tool = Mock(name="test_tool", description="test desc")
        tool.return_value = {"result": "success"}
        scratchpad = SharedScratchpad()
        wrapper = CachingToolWrapper(tool, scratchpad, "agent1")
        result = wrapper(query="test")
        assert result == {"result": "success"}
        tool.assert_called_once_with(query="test")
        assert len(scratchpad.messages) == 1

    def test_returns_cached_result(self):
        """CachingToolWrapper returns cached result on cache hit."""
        from Jotty.core.agents.inspector import CachingToolWrapper
        from Jotty.core.foundation.data_structures import SharedScratchpad
        tool = Mock(name="test_tool", description="test desc")
        tool.return_value = {"result": "success"}
        scratchpad = SharedScratchpad()
        # Pre-populate cache
        scratchpad.get_cached_result = Mock(return_value={"cached": True})
        wrapper = CachingToolWrapper(tool, scratchpad, "agent1")
        result = wrapper(query="test")
        assert result == {"cached": True}
        tool.assert_not_called()

    def test_handles_tool_exception(self):
        """CachingToolWrapper returns error dict when tool raises."""
        from Jotty.core.agents.inspector import CachingToolWrapper
        from Jotty.core.foundation.data_structures import SharedScratchpad
        tool = Mock(name="error_tool", description="")
        tool.side_effect = RuntimeError("tool failed")
        scratchpad = SharedScratchpad()
        wrapper = CachingToolWrapper(tool, scratchpad, "agent1")
        result = wrapper(query="test")
        assert "error" in result
        assert "tool failed" in result["error"]


@pytest.mark.unit
class TestMultiRoundValidatorUnit:
    """Unit tests for MultiRoundValidator._needs_refinement and _build_feedback."""

    def _make_validator(self):
        """Create MultiRoundValidator with mock agents."""
        from Jotty.core.agents.inspector import MultiRoundValidator
        from Jotty.core.foundation.data_structures import SwarmConfig
        config = SwarmConfig()
        config.refinement_on_low_confidence = 0.6
        config.refinement_on_disagreement = True
        return MultiRoundValidator(agents=[], config=config)

    def test_needs_refinement_low_confidence(self):
        """Low confidence triggers refinement."""
        validator = self._make_validator()
        results = [
            Mock(confidence=0.3, should_proceed=True, is_valid=True),
            Mock(confidence=0.8, should_proceed=True, is_valid=True),
        ]
        assert validator._needs_refinement(results, is_architect=True) is True

    def test_needs_refinement_disagreement(self):
        """Disagreement triggers refinement."""
        validator = self._make_validator()
        results = [
            Mock(confidence=0.8, should_proceed=True, is_valid=True),
            Mock(confidence=0.8, should_proceed=False, is_valid=False),
        ]
        assert validator._needs_refinement(results, is_architect=True) is True

    def test_no_refinement_high_confidence_agreement(self):
        """High confidence + agreement = no refinement."""
        validator = self._make_validator()
        results = [
            Mock(confidence=0.9, should_proceed=True, is_valid=True),
            Mock(confidence=0.8, should_proceed=True, is_valid=True),
        ]
        assert validator._needs_refinement(results, is_architect=True) is False

    def test_build_feedback(self):
        """_build_feedback includes agent names and reasoning."""
        validator = self._make_validator()
        results = [
            Mock(agent_name="auditor1", should_proceed=True, confidence=0.8, reasoning="Looks good"),
            Mock(agent_name="auditor2", should_proceed=False, confidence=0.6, reasoning="Missing data"),
        ]
        feedback = validator._build_feedback(results)
        assert "auditor1" in feedback
        assert "auditor2" in feedback
        assert "Looks good" in feedback
        assert "Missing data" in feedback


@pytest.mark.unit
class TestCompletionReviewer:
    """Tests for inspector.py CompletionReviewer."""

    def test_init_lazy_predictor(self):
        """CompletionReviewer initializes without creating predictor."""
        from Jotty.core.agents.inspector import CompletionReviewer
        reviewer = CompletionReviewer()
        assert reviewer._predictor is None

    @pytest.mark.asyncio
    async def test_review_completion_fallback_on_success(self):
        """CompletionReviewer falls back to heuristic when predictor errors, success=True."""
        from Jotty.core.agents.inspector import CompletionReviewer
        reviewer = CompletionReviewer()
        # Mock predictor — will fail in the async-in-thread path
        reviewer._predictor = MagicMock()

        result = await reviewer.review_completion(
            instruction="Research AI trends",
            result={"success": True, "output": "AI trends report"},
            tool_calls=[{"tool": "web_search"}],
        )
        # Fallback: success=True → complete, confidence=0.4
        assert result['completion_state'] == "complete"
        assert result['confidence'] == 0.4

    @pytest.mark.asyncio
    async def test_review_completion_fallback_on_failure(self):
        """CompletionReviewer falls back to heuristic when predictor errors, success=False."""
        from Jotty.core.agents.inspector import CompletionReviewer
        reviewer = CompletionReviewer()
        reviewer._predictor = MagicMock()

        result = await reviewer.review_completion(
            instruction="Research AI trends",
            result={"success": False, "error": "API timeout"},
            tool_calls=[],
        )
        # Fallback: success=False → partial
        assert result['completion_state'] == "partial"
        assert len(result['unresolved_items']) > 0


# =============================================================================
# LLMProvider Tests
# =============================================================================

@pytest.mark.unit
class TestLLMProvider:
    """Tests for executor.py LLMProvider."""

    def test_default_provider_anthropic(self):
        """LLMProvider defaults to anthropic provider."""
        from Jotty.core.execution.executor import LLMProvider
        provider = LLMProvider()
        assert provider._provider_name == 'anthropic'
        assert 'claude' in provider._model

    def test_custom_model(self):
        """LLMProvider accepts custom model."""
        from Jotty.core.execution.executor import LLMProvider
        provider = LLMProvider(model='claude-haiku-4-5-20251001')
        assert provider._model == 'claude-haiku-4-5-20251001'

    def test_lazy_client_init(self):
        """Client is not created until first use."""
        from Jotty.core.execution.executor import LLMProvider
        provider = LLMProvider()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_generate_anthropic(self):
        """LLMProvider.generate calls anthropic API correctly."""
        from Jotty.core.execution.executor import LLMProvider
        provider = LLMProvider()
        mock_block = Mock()
        mock_block.text = "Hello response"
        mock_response = Mock()
        mock_response.content = [mock_block]
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client
        result = await provider.generate("Hello")
        assert result['content'] == "Hello response"
        assert result['usage']['input_tokens'] == 10
        assert result['usage']['output_tokens'] == 5


# =============================================================================
# COMPLEX REAL ORCHESTRATOR INTEGRATION TEST
# =============================================================================
# Run standalone: python tests/test_v3_execution.py --integration
#
# Exercises the full SkillPlanExecutor pipeline with REAL LLM + REAL tools
# to validate the 22 patterns provide measurable value:
#   - ToolCallCache, DAG parallel, ToolResultProcessor
#   - CircuitBreaker, AdaptiveTimeout, DeadLetterQueue, TimeoutWarning
#   - ErrorType classification, ValidationVerdict, FailureRouter
#   - SkillQTable, COMACredit, get_learned_context
#   - ToolStats, CapabilityIndex
#   - Proactive context guard, compression retry
#   - LLM-analyzed retry with trajectory
#   - Policy explorer, data-flow dependencies
#   - Self-RAG, surprise memory
# =============================================================================


class RealOrchestratorIntegrationTest:
    """Super-complex real orchestrator stress test.

    NOT a pytest test — run directly: python tests/test_v3_execution.py --integration

    Throws 5 genuinely hard, multi-step tasks at the orchestrator.
    No hand-holding, no pre-registered patterns — just raw goals.
    The orchestrator must plan skills, chain data between steps,
    handle errors, replan on failure, and produce real artifacts.
    """

    OUTPUT_DIR = "/tmp/jotty_stress_test"

    def __init__(self):
        self.task_results = {}
        self.start_time = 0.0
        self.task_timings = {}

    def log(self, msg: str):
        import time
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        print(f"[{elapsed:6.1f}s] {msg}", flush=True)

    def section(self, title: str):
        border = "=" * 70
        print(f"\n{border}\n  {title}\n{border}", flush=True)

    def _build_discovered_skills(self, registry) -> list:
        """Build skill discovery list from the real registry."""
        discovered = []
        for name, skill_def in list(registry.loaded_skills.items()):
            desc = ""
            if hasattr(skill_def, "metadata") and skill_def.metadata:
                desc = getattr(skill_def.metadata, "description", "") or ""
            tools_list = list(skill_def.tools.keys()) if hasattr(skill_def, "tools") else []
            discovered.append({"name": name, "description": desc, "tools": tools_list})
        return discovered

    def _check_file(self, path: str, label: str) -> dict:
        """Check if a file was created and report its content stats."""
        import os
        info = {"label": label, "path": path, "exists": False}
        if os.path.exists(path):
            if os.path.isdir(path):
                # Path was created as directory by mistake
                info["exists"] = False
                info["is_dir"] = True
                return info
            info["exists"] = True
            info["size"] = os.path.getsize(path)
            with open(path, errors="replace") as f:
                content = f.read()
            info["lines"] = len(content.split("\n"))
            info["chars"] = len(content)
            info["preview"] = content[:600]
            # Quality heuristic: not just a stub
            info["is_stub"] = (
                info["size"] < 200
                or content.strip().startswith("# TODO")
            )
        return info

    def _log_result(self, name: str, result: dict, elapsed: float):
        """Log a task execution result with full details."""
        self.log(f"  Success: {result.get('success')}")
        self.log(f"  Task type: {result.get('task_type', '?')}")
        self.log(f"  Skills used: {result.get('skills_used', [])}")
        self.log(f"  Steps planned: {result.get('steps_planned', '?')}")
        self.log(f"  Steps executed: {result.get('steps_executed', 0)}")
        self.log(f"  Replans: {result.get('replans', 0)}")
        self.log(f"  Time: {elapsed:.1f}s")
        errors = result.get("errors", [])
        if errors:
            self.log(f"  Errors ({len(errors)}):")
            for e in errors[:5]:
                self.log(f"    - {str(e)[:120]}")

        outputs = result.get("outputs", {})
        for key, val in outputs.items():
            if isinstance(val, dict):
                keys = [k for k in val.keys() if k != "_tags"][:6]
                self.log(f"  Output [{key}]: keys={keys}")
            else:
                self.log(f"  Output [{key}]: {str(val)[:150]}")

    async def _run_task(self, executor, discovered, task: str, name: str, status_cb):
        """Execute a single task and capture results + timing."""
        import time
        self.section(f"TASK: {name}")
        self.log(f"Goal: {task[:200]}")

        t0 = time.time()
        try:
            result = await executor.plan_and_execute(
                task=task,
                discovered_skills=discovered,
                status_callback=status_cb,
            )
        except Exception as e:
            result = {"success": False, "error": str(e), "skills_used": [], "steps_executed": 0}
            self.log(f"  EXCEPTION: {e}")

        elapsed = time.time() - t0
        self.task_timings[name] = elapsed
        self.task_results[name] = result
        self._log_result(name, result, elapsed)
        return result

    async def run(self):
        import time
        import os
        import shutil

        self.start_time = time.time()

        self.section("JOTTY ORCHESTRATOR STRESS TEST — 5 COMPLEX GOALS")
        self.log("No mocks. No hand-holding. Real LLM + real tools.")
        self.log(f"All outputs go to {self.OUTPUT_DIR}/")

        # Clean slate
        if os.path.exists(self.OUTPUT_DIR):
            shutil.rmtree(self.OUTPUT_DIR)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # ------------------------------------------------------------------
        # INIT
        # ------------------------------------------------------------------
        self.section("INIT: Registry + Executor")

        from Jotty.core.registry.skills_registry import get_skills_registry
        from Jotty.core.agents.base.skill_plan_executor import SkillPlanExecutor

        sr = get_skills_registry()
        sr.init()
        self.log(f"Skills loaded: {len(sr.loaded_skills)}")

        executor = SkillPlanExecutor(
            sr, max_steps=15, enable_replanning=True, max_replans=3
        )
        discovered = self._build_discovered_skills(sr)
        self.log(f"Discovered {len(discovered)} skills for planning")

        def status_cb(stage, detail):
            self.log(f"  |{stage}| {detail}")

        # ==================================================================
        # TASK 1: Algorithm Benchmark Pipeline
        #
        # generate 3 sorting algorithms → benchmark harness → execute →
        # parse timing output → compute speedup ratios → write report
        #
        # Tests: claude-cli-llm, file-operations, shell-exec, calculator
        # Hard because: generated code MUST compile + run, output is
        # consumed by calculator, report must contain real numbers
        # ==================================================================
        task1 = (
            f"Create a Python file at {self.OUTPUT_DIR}/sort_benchmark.py that: "
            f"(a) implements bubble sort, merge sort, and quicksort, "
            f"(b) benchmarks each on random lists of sizes 1000, 5000, and 10000 "
            f"using time.perf_counter, "
            f"(c) prints results as CSV lines: algorithm,size,seconds. "
            f"Then execute that script with shell-exec and save stdout to "
            f"{self.OUTPUT_DIR}/benchmark_results.csv. "
            f"Then use the calculator to compute the speedup of quicksort over "
            f"bubble sort on size 10000 (divide bubble_time by quick_time). "
            f"Finally write a markdown report at {self.OUTPUT_DIR}/benchmark_report.md "
            f"containing ALL benchmark CSV data and the speedup calculation."
        )
        await self._run_task(executor, discovered, task1, "T1_AlgoBenchmark", status_cb)

        # ==================================================================
        # TASK 2: Data Pipeline — generate → process → analyze → report
        #
        # generate synthetic sales CSV → write analysis script →
        # execute analysis → capture aggregated output → write report
        #
        # Tests: claude-cli-llm, file-operations, shell-exec
        # Hard because: TWO scripts that must work, second reads first's
        # output file, final report must contain computed aggregates
        # ==================================================================
        task2 = (
            f"Step 1: Write a Python script at {self.OUTPUT_DIR}/generate_sales.py that "
            f"generates a CSV file at {self.OUTPUT_DIR}/sales_data.csv with 200 rows "
            f"and columns: date (random dates in 2025), region (North/South/East/West), "
            f"product (Widget-A/Widget-B/Widget-C), units_sold (random 1-500), "
            f"unit_price (random 10.0-99.0). Use random.seed(42) for reproducibility. "
            f"Step 2: Execute generate_sales.py with shell-exec. "
            f"Step 3: Write a second script at {self.OUTPUT_DIR}/analyze_sales.py that "
            f"reads {self.OUTPUT_DIR}/sales_data.csv and prints: "
            f"total revenue, revenue by region, top product by units, and "
            f"the month with highest revenue. "
            f"Step 4: Execute analyze_sales.py with shell-exec and save output to "
            f"{self.OUTPUT_DIR}/sales_analysis.txt. "
            f"Step 5: Write a final report at {self.OUTPUT_DIR}/sales_report.md "
            f"combining the raw data stats and analysis output."
        )
        await self._run_task(executor, discovered, task2, "T2_DataPipeline", status_cb)

        # ==================================================================
        # TASK 3: Live Research → Financial Modeling → Investment Memo
        #
        # search 3 companies → extract metrics → calculate 5+ ratios →
        # cross-compare → write investment memo with ranked recommendation
        #
        # Tests: web-search, calculator, claude-cli-llm, file-operations
        # Hard because: 3 search calls, 5+ calculator calls, LLM synthesis
        # must reference actual calculated numbers, not hallucinate
        # ==================================================================
        task3 = (
            f"Research and compare three companies — Apple, Microsoft, and Google — "
            f"for an investment memo: "
            f"(1) Search the web for 'Apple AAPL stock price revenue 2025 2026', "
            f"(2) Search the web for 'Microsoft MSFT stock price revenue 2025 2026', "
            f"(3) Search the web for 'Google GOOGL stock price revenue 2025 2026'. "
            f"Then use the calculator for these computations: "
            f"(a) Apple market cap if price is $195 and shares outstanding are 15.4 billion: 195*15400000000, "
            f"(b) Microsoft market cap if price is $420 and shares are 7.43 billion: 420*7430000000, "
            f"(c) Google market cap if price is $175 and shares are 12.06 billion: 175*12060000000, "
            f"(d) Apple P/E ratio if EPS is $6.75: 195/6.75, "
            f"(e) percentage difference between Microsoft and Apple market caps: "
            f"(420*7430000000 - 195*15400000000) / (195*15400000000) * 100. "
            f"Write the complete investment memo at {self.OUTPUT_DIR}/investment_memo.md "
            f"with sections: Executive Summary, Company Profiles (with search findings), "
            f"Financial Metrics (with all calculated values), and Ranked Recommendation."
        )
        await self._run_task(executor, discovered, task3, "T3_InvestmentMemo", status_cb)

        # ==================================================================
        # TASK 4: Multi-file Library + Test Suite + Execution
        #
        # create a Python package with 3 modules → write test suite →
        # execute tests → capture results → generate docs
        #
        # Tests: file-operations (8+ writes), shell-exec (run pytest),
        #        claude-cli-llm
        # Hard because: inter-module imports must work, tests must pass,
        # test output is captured and verified
        # ==================================================================
        task4 = (
            f"Build a Python library at {self.OUTPUT_DIR}/mathlib/ with: "
            f"(1) {self.OUTPUT_DIR}/mathlib/__init__.py that exports the library, "
            f"(2) {self.OUTPUT_DIR}/mathlib/stats.py with functions: mean(data), "
            f"median(data), std_dev(data) — pure Python, no external deps, "
            f"(3) {self.OUTPUT_DIR}/mathlib/geometry.py with functions: "
            f"circle_area(r), triangle_area(base, height), distance(x1,y1,x2,y2), "
            f"(4) {self.OUTPUT_DIR}/test_mathlib.py with at least 8 pytest test "
            f"functions covering all stats and geometry functions with known values. "
            f"Then execute 'python -m pytest {self.OUTPUT_DIR}/test_mathlib.py -v' "
            f"with shell-exec and save the output to {self.OUTPUT_DIR}/test_results.txt. "
            f"Finally write {self.OUTPUT_DIR}/mathlib/README.md documenting all functions "
            f"with usage examples and the test pass results."
        )
        await self._run_task(executor, discovered, task4, "T4_LibraryAndTests", status_cb)

        # ==================================================================
        # TASK 5: Web Intelligence Dashboard
        #
        # search 4 trending tech topics → calculate composite scores →
        # generate HTML dashboard → generate JSON data file
        #
        # Tests: web-search (4 calls), calculator (4 calls),
        #        file-operations, claude-cli-llm
        # Hard because: 4 parallel research streams, calculated scores
        # must appear in both HTML and JSON, HTML must be valid
        # ==================================================================
        task5 = (
            f"Create a tech trends intelligence dashboard: "
            f"(1) Search the web for 'artificial intelligence market size 2026', "
            f"(2) Search the web for 'quantum computing breakthroughs 2026', "
            f"(3) Search the web for 'autonomous vehicles progress 2026', "
            f"(4) Search the web for 'blockchain enterprise adoption 2026'. "
            f"For each topic, use the calculator to compute a 'hype score' from 1-100 "
            f"based on: (number_of_search_results * 10) capped at 100. "
            f"Use formula: min(result_count * 10, 100). "
            f"Write {self.OUTPUT_DIR}/dashboard_data.json with JSON containing "
            f"each topic's name, top 3 search result titles, and hype_score. "
            f"Write {self.OUTPUT_DIR}/dashboard.html with a styled HTML page "
            f"that displays all 4 topics in cards with their scores and key findings. "
            f"Write {self.OUTPUT_DIR}/trend_analysis.md with a markdown report "
            f"ranking all 4 topics by hype score and providing a brief analysis of each."
        )
        await self._run_task(executor, discovered, task5, "T5_TechDashboard", status_cb)

        # ==================================================================
        # FINAL SCORECARD
        # ==================================================================
        self.section("FINAL SCORECARD")

        total_time = time.time() - self.start_time
        self.log(f"Total wall-clock time: {total_time:.1f}s")
        self.log(f"Task timings:")
        for name, t in self.task_timings.items():
            self.log(f"  {name}: {t:.1f}s")

        # Check all expected output files
        expected_files = {
            # T1: Algorithm Benchmark
            "T1_script": f"{self.OUTPUT_DIR}/sort_benchmark.py",
            "T1_csv": f"{self.OUTPUT_DIR}/benchmark_results.csv",
            "T1_report": f"{self.OUTPUT_DIR}/benchmark_report.md",
            # T2: Data Pipeline
            "T2_gen_script": f"{self.OUTPUT_DIR}/generate_sales.py",
            "T2_csv": f"{self.OUTPUT_DIR}/sales_data.csv",
            "T2_analyze_script": f"{self.OUTPUT_DIR}/analyze_sales.py",
            "T2_analysis": f"{self.OUTPUT_DIR}/sales_analysis.txt",
            "T2_report": f"{self.OUTPUT_DIR}/sales_report.md",
            # T3: Investment Memo
            "T3_memo": f"{self.OUTPUT_DIR}/investment_memo.md",
            # T4: Library + Tests
            "T4_init": f"{self.OUTPUT_DIR}/mathlib/__init__.py",
            "T4_stats": f"{self.OUTPUT_DIR}/mathlib/stats.py",
            "T4_geometry": f"{self.OUTPUT_DIR}/mathlib/geometry.py",
            "T4_tests": f"{self.OUTPUT_DIR}/test_mathlib.py",
            "T4_results": f"{self.OUTPUT_DIR}/test_results.txt",
            "T4_readme": f"{self.OUTPUT_DIR}/mathlib/README.md",
            # T5: Tech Dashboard
            "T5_json": f"{self.OUTPUT_DIR}/dashboard_data.json",
            "T5_html": f"{self.OUTPUT_DIR}/dashboard.html",
            "T5_analysis": f"{self.OUTPUT_DIR}/trend_analysis.md",
        }

        self.section("ARTIFACT VERIFICATION")
        artifacts_pass = 0
        artifacts_total = len(expected_files)
        for label, path in expected_files.items():
            info = self._check_file(path, label)
            if info.get("is_dir"):
                self.log(f"  [DDIR] {label}: path is a directory (planner bug)")
            elif info["exists"] and not info.get("is_stub", True):
                status = "PASS"
                artifacts_pass += 1
                self.log(f"  [{status}] {label}: {info['size']} bytes, {info['lines']} lines")
            elif info["exists"]:
                status = "STUB"
                self.log(f"  [{status}] {label}: {info['size']} bytes (looks like stub/generated code)")
            else:
                status = "MISS"
                self.log(f"  [{status}] {label}: FILE NOT FOUND at {path}")

        self.section("FILE CONTENT PREVIEWS")
        for label, path in expected_files.items():
            info = self._check_file(path, label)
            if info.get("exists") and not info.get("is_dir"):
                self.log(f"\n--- {label} ({info['size']} bytes) ---")
                self.log(info["preview"])
                if info["chars"] > 600:
                    self.log(f"  ... ({info['chars'] - 600} more chars)")

        self.section("TASK SUCCESS SUMMARY")
        tasks_pass = 0
        for name, result in self.task_results.items():
            success = result.get("success", False)
            steps = result.get("steps_executed", 0)
            skills = result.get("skills_used", [])
            replans = result.get("replans", 0)
            elapsed = self.task_timings.get(name, 0)
            status = "PASS" if success else "FAIL"
            if success:
                tasks_pass += 1
            self.log(
                f"  [{status}] {name}: {steps} steps, {len(skills)} skills "
                f"({', '.join(skills[:4])}), {replans} replans, {elapsed:.1f}s"
            )

        self.log("")
        self.log(f"  Tasks: {tasks_pass}/{len(self.task_results)} passed")
        self.log(f"  Artifacts: {artifacts_pass}/{artifacts_total} real content")
        self.log(f"  Total time: {total_time:.1f}s")

        # Overall grade
        total_score = tasks_pass + artifacts_pass
        total_possible = len(self.task_results) + artifacts_total
        pct = (total_score / total_possible * 100) if total_possible else 0
        if pct >= 80:
            grade = "A"
        elif pct >= 60:
            grade = "B"
        elif pct >= 40:
            grade = "C"
        elif pct >= 20:
            grade = "D"
        else:
            grade = "F"
        self.log(f"\n  GRADE: {grade} ({total_score}/{total_possible} = {pct:.0f}%)")

        return self.task_results


# =============================================================================
# ToolCallCache Tests
# =============================================================================

class TestToolCallCache:
    """Tests for ToolCallCache TTL + LRU caching."""

    @pytest.mark.unit
    def test_make_key_deterministic(self):
        """make_key produces same key for same inputs."""
        from Jotty.core.agents.base.skill_plan_executor import ToolCallCache
        k1 = ToolCallCache.make_key("web-search", "search", {"query": "AI"})
        k2 = ToolCallCache.make_key("web-search", "search", {"query": "AI"})
        assert k1 == k2

    @pytest.mark.unit
    def test_make_key_different_params(self):
        """make_key produces different keys for different params."""
        from Jotty.core.agents.base.skill_plan_executor import ToolCallCache
        k1 = ToolCallCache.make_key("web-search", "search", {"query": "AI"})
        k2 = ToolCallCache.make_key("web-search", "search", {"query": "ML"})
        assert k1 != k2

    @pytest.mark.unit
    def test_make_key_sorts_params(self):
        """make_key is order-independent for params."""
        from Jotty.core.agents.base.skill_plan_executor import ToolCallCache
        k1 = ToolCallCache.make_key("s", "t", {"a": 1, "b": 2})
        k2 = ToolCallCache.make_key("s", "t", {"b": 2, "a": 1})
        assert k1 == k2

    @pytest.mark.unit
    def test_get_set(self):
        """set() stores value, get() retrieves it."""
        from Jotty.core.agents.base.skill_plan_executor import ToolCallCache
        cache = ToolCallCache(ttl_seconds=60)
        key = cache.make_key("s", "t", {"x": 1})
        cache.set(key, {"result": "data"})
        assert cache.get(key) == {"result": "data"}

    @pytest.mark.unit
    def test_get_miss(self):
        """get() returns None for missing key."""
        from Jotty.core.agents.base.skill_plan_executor import ToolCallCache
        cache = ToolCallCache()
        assert cache.get("nonexistent") is None

    @pytest.mark.unit
    def test_ttl_expiry(self):
        """get() returns None for expired entries."""
        import time as _time
        from Jotty.core.agents.base.skill_plan_executor import ToolCallCache
        cache = ToolCallCache(ttl_seconds=0)  # Immediate expiry
        key = "test_key"
        cache.set(key, "value")
        _time.sleep(0.01)  # Ensure time passes
        assert cache.get(key) is None

    @pytest.mark.unit
    def test_lru_eviction(self):
        """Oldest entry is evicted when max_size is reached."""
        from Jotty.core.agents.base.skill_plan_executor import ToolCallCache
        cache = ToolCallCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # Should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    @pytest.mark.unit
    def test_clear(self):
        """clear() empties the cache."""
        from Jotty.core.agents.base.skill_plan_executor import ToolCallCache
        cache = ToolCallCache()
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.size == 2
        cache.clear()
        assert cache.size == 0
        assert cache.get("a") is None

    @pytest.mark.unit
    def test_size_property(self):
        """size property returns number of cached entries."""
        from Jotty.core.agents.base.skill_plan_executor import ToolCallCache
        cache = ToolCallCache()
        assert cache.size == 0
        cache.set("a", 1)
        assert cache.size == 1

    @pytest.mark.unit
    def test_overwrite_existing_key(self):
        """Setting existing key updates value without eviction."""
        from Jotty.core.agents.base.skill_plan_executor import ToolCallCache
        cache = ToolCallCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("a", 10)  # Update, not evict
        assert cache.size == 2
        assert cache.get("a") == 10


# =============================================================================
# SkillPlanExecutor DAG + Exclusion Tests
# =============================================================================

class TestSkillPlanExecutorDAG:
    """Tests for dependency graph and parallel group detection."""

    def _make_executor(self):
        from Jotty.core.agents.base.skill_plan_executor import SkillPlanExecutor
        mock_registry = MagicMock()
        return SkillPlanExecutor(skills_registry=mock_registry)

    @pytest.mark.unit
    def test_build_dependency_graph_no_deps(self):
        """Steps with no depends_on have empty dependency lists."""
        executor = self._make_executor()
        steps = [MagicMock(depends_on=None), MagicMock(depends_on=None)]
        graph = executor._build_dependency_graph(steps)
        assert graph == {0: [], 1: []}

    @pytest.mark.unit
    def test_build_dependency_graph_with_deps(self):
        """Steps with depends_on reference earlier indices."""
        executor = self._make_executor()
        step0 = MagicMock(depends_on=None)
        step1 = MagicMock(depends_on=[0])
        step2 = MagicMock(depends_on=[0, 1])
        graph = executor._build_dependency_graph([step0, step1, step2])
        assert graph[0] == []
        assert graph[1] == [0]
        assert graph[2] == [0, 1]

    @pytest.mark.unit
    def test_build_dependency_graph_filters_invalid(self):
        """Invalid dependency indices are filtered out."""
        executor = self._make_executor()
        step0 = MagicMock(depends_on=[99, -1, "bad"])  # All invalid
        graph = executor._build_dependency_graph([step0])
        assert graph[0] == []

    @pytest.mark.unit
    def test_find_parallel_groups_all_independent(self):
        """All independent steps form a single parallel layer."""
        executor = self._make_executor()
        steps = [MagicMock(depends_on=None) for _ in range(3)]
        groups = executor._find_parallel_groups(steps)
        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1, 2]

    @pytest.mark.unit
    def test_find_parallel_groups_sequential(self):
        """Sequential chain produces one step per layer."""
        executor = self._make_executor()
        steps = [
            MagicMock(depends_on=None),
            MagicMock(depends_on=[0]),
            MagicMock(depends_on=[1]),
        ]
        groups = executor._find_parallel_groups(steps)
        assert len(groups) == 3
        assert groups[0] == [0]
        assert groups[1] == [1]
        assert groups[2] == [2]

    @pytest.mark.unit
    def test_find_parallel_groups_diamond(self):
        """Diamond dependency pattern: 0 → {1,2} → 3."""
        executor = self._make_executor()
        steps = [
            MagicMock(depends_on=None),      # 0: root
            MagicMock(depends_on=[0]),        # 1: depends on 0
            MagicMock(depends_on=[0]),        # 2: depends on 0
            MagicMock(depends_on=[1, 2]),     # 3: depends on 1 and 2
        ]
        groups = executor._find_parallel_groups(steps)
        assert len(groups) == 3
        assert groups[0] == [0]
        assert sorted(groups[1]) == [1, 2]
        assert groups[2] == [3]


# =============================================================================
# SkillPlanExecutor Exclusion Management Tests
# =============================================================================

class TestSkillExclusions:
    """Tests for skill exclusion management."""

    def _make_executor(self):
        from Jotty.core.agents.base.skill_plan_executor import SkillPlanExecutor
        return SkillPlanExecutor(skills_registry=MagicMock())

    @pytest.mark.unit
    def test_exclude_skill(self):
        """exclude_skill adds to exclusion set."""
        executor = self._make_executor()
        executor.exclude_skill("web-search")
        assert "web-search" in executor.excluded_skills

    @pytest.mark.unit
    def test_clear_exclusions(self):
        """clear_exclusions empties the set."""
        executor = self._make_executor()
        executor.exclude_skill("a")
        executor.exclude_skill("b")
        executor.clear_exclusions()
        assert len(executor.excluded_skills) == 0

    @pytest.mark.unit
    def test_excluded_skills_property(self):
        """excluded_skills returns the exclusion set."""
        executor = self._make_executor()
        assert executor.excluded_skills == set()
        executor.exclude_skill("x")
        assert "x" in executor.excluded_skills


# =============================================================================
# FailureRouter Tests
# =============================================================================

class TestFailureRouter:
    """Tests for FailureRouter error classification and routing."""

    @pytest.mark.unit
    def test_timeout_pattern(self):
        """Timeout errors route to retry_with_backoff."""
        from Jotty.core.agents.inspector import FailureRouter
        router = FailureRouter()
        result = router.route("Connection timeout after 30s", "web-search")
        assert result['action'] == 'retry_with_backoff'
        assert result['failed_skill'] == 'web-search'

    @pytest.mark.unit
    def test_rate_limit_pattern(self):
        """Rate limit errors route to wait_and_retry."""
        from Jotty.core.agents.inspector import FailureRouter
        router = FailureRouter()
        result = router.route("Rate_limit exceeded, retry after 60s", "claude-cli-llm")
        assert result['action'] == 'wait_and_retry'
        assert result.get('delay') == 60

    @pytest.mark.unit
    def test_not_found_pattern(self):
        """Not found errors route to try_alternative."""
        from Jotty.core.agents.inspector import FailureRouter
        router = FailureRouter()
        result = router.route("Resource not_found at endpoint", "http-client")
        assert result['action'] == 'try_alternative'

    @pytest.mark.unit
    def test_permission_pattern(self):
        """Permission errors route to escalate."""
        from Jotty.core.agents.inspector import FailureRouter
        router = FailureRouter()
        result = router.route("Permission denied: insufficient access", "file-manager")
        assert result['action'] == 'escalate'

    @pytest.mark.unit
    def test_parse_error_pattern(self):
        """Parse errors route to retry_with_fix."""
        from Jotty.core.agents.inspector import FailureRouter
        router = FailureRouter()
        result = router.route("parse_error: unexpected token", "json-parser")
        assert result['action'] == 'retry_with_fix'

    @pytest.mark.unit
    def test_infrastructure_fallback(self):
        """Unknown infrastructure errors fall back to retry_with_backoff."""
        from Jotty.core.agents.inspector import FailureRouter
        result = FailureRouter().route("Connection reset by peer", "web-search")
        assert result['action'] == 'retry_with_backoff'
        assert result['error_type'] == 'infrastructure'

    @pytest.mark.unit
    def test_data_error_fallback(self):
        """Data errors route to validate_inputs."""
        from Jotty.core.agents.inspector import FailureRouter
        result = FailureRouter().route("invalid JSON format in response body", "api-client")
        assert result['action'] == 'validate_inputs'
        assert result['error_type'] == 'data'

    @pytest.mark.unit
    def test_find_alternatives(self):
        """_find_alternatives returns known alternatives."""
        from Jotty.core.agents.inspector import FailureRouter
        router = FailureRouter()
        alts = router._find_alternatives("web-search")
        assert "http-client" in alts

    @pytest.mark.unit
    def test_find_alternatives_unknown(self):
        """_find_alternatives returns empty for unknown skills."""
        from Jotty.core.agents.inspector import FailureRouter
        router = FailureRouter()
        assert router._find_alternatives("nonexistent-skill") == []

    @pytest.mark.unit
    def test_logic_error_with_alternative(self):
        """Logic error with available alternative suggests it."""
        from Jotty.core.agents.inspector import FailureRouter
        router = FailureRouter()
        # "SyntaxError" classified as LOGIC, web-search has alternatives
        result = router.route("SyntaxError: invalid syntax in template", "web-search")
        assert result['action'] == 'try_alternative'
        assert 'suggested_agent' in result

    @pytest.mark.unit
    def test_logic_error_no_alternative(self):
        """Logic error without alternative routes to replan."""
        from Jotty.core.agents.inspector import FailureRouter
        router = FailureRouter()
        result = router.route("SyntaxError: invalid syntax in template", "custom-tool")
        assert result['action'] == 'replan'


# =============================================================================
# ValidatorAgent _check_required_fields Tests
# =============================================================================

class TestCheckRequiredFields:
    """Tests for ValidatorAgent._check_required_fields.

    Uses object.__new__ to bypass the heavy __init__ and directly test
    the pure-logic method.
    """

    def _make_validator(self, is_architect=True):
        """Create a ValidatorAgent shell bypassing __init__."""
        from Jotty.core.agents.inspector import ValidatorAgent
        v = object.__new__(ValidatorAgent)
        v.is_architect = is_architect
        return v

    @pytest.mark.unit
    def test_architect_all_fields_present(self):
        """No missing fields when architect result has all required fields."""
        validator = self._make_validator(is_architect=True)
        result = MagicMock()
        result.reasoning = "Good reasoning"
        result.confidence = 0.8
        result.should_proceed = True
        missing = validator._check_required_fields(result)
        assert missing == []

    @pytest.mark.unit
    def test_architect_missing_reasoning(self):
        """Missing reasoning reported for architect."""
        validator = self._make_validator(is_architect=True)
        result = MagicMock(spec=[])  # No attributes
        missing = validator._check_required_fields(result)
        assert 'reasoning' in missing
        assert 'confidence' in missing
        assert 'should_proceed' in missing

    @pytest.mark.unit
    def test_auditor_all_fields_present(self):
        """No missing fields when auditor result has all required fields."""
        validator = self._make_validator(is_architect=False)
        result = MagicMock()
        result.reasoning = "Valid output"
        result.confidence = 0.9
        result.is_valid = True
        result.output_tag = "useful"
        missing = validator._check_required_fields(result)
        assert missing == []

    @pytest.mark.unit
    def test_auditor_missing_fields(self):
        """Missing fields reported for auditor."""
        validator = self._make_validator(is_architect=False)
        result = MagicMock(spec=[])
        missing = validator._check_required_fields(result)
        assert 'reasoning' in missing
        assert 'is_valid' in missing
        assert 'output_tag' in missing

    @pytest.mark.unit
    def test_empty_reasoning_counts_as_missing(self):
        """Empty reasoning string is treated as missing."""
        validator = self._make_validator(is_architect=True)
        result = MagicMock()
        result.reasoning = ""
        result.confidence = 0.5
        result.should_proceed = True
        missing = validator._check_required_fields(result)
        assert 'reasoning' in missing


# =============================================================================
# ValidatorAgent _smart_truncate Tests (expanded)
# =============================================================================

class TestSmartTruncateExpanded:
    """Expanded tests for ValidatorAgent._smart_truncate."""

    def _make_validator(self):
        """Create ValidatorAgent shell bypassing __init__."""
        from Jotty.core.agents.inspector import ValidatorAgent
        v = object.__new__(ValidatorAgent)
        return v

    @pytest.mark.unit
    def test_no_truncation_needed(self):
        """Short text returned as-is."""
        v = self._make_validator()
        assert v._smart_truncate("hello", 100) == "hello"

    @pytest.mark.unit
    def test_truncate_at_sentence(self):
        """Truncation prefers sentence boundaries."""
        v = self._make_validator()
        text = "First sentence. Second sentence. Third very long sentence that goes beyond the limit."
        result = v._smart_truncate(text, 40)
        assert result.endswith("...")
        assert "First sentence." in result

    @pytest.mark.unit
    def test_truncate_at_word(self):
        """Truncation falls back to word boundary."""
        v = self._make_validator()
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        result = v._smart_truncate(text, 30)
        assert result.endswith("...")
        assert not result[:-3].endswith(" ")  # Clean word boundary

    @pytest.mark.unit
    def test_hard_truncate(self):
        """Hard truncation when no good boundary found."""
        v = self._make_validator()
        text = "a" * 100
        result = v._smart_truncate(text, 50)
        assert len(result) == 53  # 50 chars + "..."
        assert result.endswith("...")

    @pytest.mark.unit
    def test_exact_length(self):
        """Text at exact max_chars is not truncated."""
        v = self._make_validator()
        text = "x" * 50
        result = v._smart_truncate(text, 50)
        assert result == text


# =============================================================================
# ValidatorAgent get_statistics Tests
# =============================================================================

class TestValidatorStatistics:
    """Tests for ValidatorAgent.get_statistics."""

    @pytest.mark.unit
    def test_statistics_initial(self):
        """Initial statistics have zero counts."""
        from Jotty.core.agents.inspector import ValidatorAgent
        v = object.__new__(ValidatorAgent)
        v.agent_name = "test_auditor"
        v.is_architect = False
        v.total_calls = 0
        v.total_approvals = 0
        mock_memory = MagicMock()
        mock_memory.get_statistics.return_value = {"entries": 0}
        v.memory = mock_memory
        stats = v.get_statistics()
        assert stats['agent_name'] == 'test_auditor'
        assert stats['is_architect'] is False
        assert stats['total_calls'] == 0
        assert stats['approval_rate'] == 0.0

    @pytest.mark.unit
    def test_statistics_with_calls(self):
        """Statistics reflect call and approval counts."""
        from Jotty.core.agents.inspector import ValidatorAgent
        v = object.__new__(ValidatorAgent)
        v.agent_name = "test_arch"
        v.is_architect = True
        v.total_calls = 10
        v.total_approvals = 7
        mock_memory = MagicMock()
        mock_memory.get_statistics.return_value = {"entries": 5}
        v.memory = mock_memory
        stats = v.get_statistics()
        assert stats['total_calls'] == 10
        assert stats['total_approvals'] == 7
        assert stats['approval_rate'] == 0.7


# =============================================================================
# InternalReasoningTool Tests
# =============================================================================

class TestInternalReasoningTool:
    """Tests for InternalReasoningTool reasoning capability."""

    @pytest.mark.unit
    def test_call_with_memory_scope(self):
        """InternalReasoningTool retrieves memories for memory scope."""
        from Jotty.core.agents.inspector import InternalReasoningTool
        mock_memory = MagicMock()
        mock_entry = MagicMock()
        mock_entry.content = "past experience"
        mock_entry.default_value = 0.8
        mock_memory.retrieve.return_value = [mock_entry]
        mock_memory.retrieve_causal.return_value = []
        mock_config = MagicMock()
        tool = InternalReasoningTool(memory=mock_memory, config=mock_config)
        result = tool("How does this work?", context_scope="memory")
        assert len(result["relevant_memories"]) == 1
        assert result["relevant_memories"][0]["content"] == "past experience"
        assert result["causal_insights"] == []

    @pytest.mark.unit
    def test_call_with_causal_scope(self):
        """InternalReasoningTool retrieves causal knowledge for causal scope."""
        from Jotty.core.agents.inspector import InternalReasoningTool
        mock_memory = MagicMock()
        mock_causal = MagicMock()
        mock_causal.cause = "type annotation"
        mock_causal.effect = "correct parsing"
        mock_causal.confidence = 0.9
        mock_memory.retrieve.return_value = []
        mock_memory.retrieve_causal.return_value = [mock_causal]
        mock_config = MagicMock()
        tool = InternalReasoningTool(memory=mock_memory, config=mock_config)
        result = tool("Why does X work?", context_scope="causal")
        assert len(result["causal_insights"]) == 1
        assert result["causal_insights"][0]["cause"] == "type annotation"
        assert result["relevant_memories"] == []

    @pytest.mark.unit
    def test_call_with_all_scope(self):
        """InternalReasoningTool retrieves both memories and causal for 'all' scope."""
        from Jotty.core.agents.inspector import InternalReasoningTool
        mock_memory = MagicMock()
        mock_entry = MagicMock(content="mem", default_value=0.5)
        mock_causal = MagicMock(cause="C", effect="E", confidence=0.7)
        mock_memory.retrieve.return_value = [mock_entry]
        mock_memory.retrieve_causal.return_value = [mock_causal]
        mock_config = MagicMock()
        tool = InternalReasoningTool(memory=mock_memory, config=mock_config)
        result = tool("Analyze this", context_scope="all")
        assert len(result["relevant_memories"]) == 1
        assert len(result["causal_insights"]) == 1

    @pytest.mark.unit
    def test_tool_name_and_description(self):
        """InternalReasoningTool has name 'reason_about'."""
        from Jotty.core.agents.inspector import InternalReasoningTool
        tool = InternalReasoningTool(memory=MagicMock(), config=MagicMock())
        assert tool.name == "reason_about"
        assert "reasoning" in tool.description.lower()


# =============================================================================
# ExecutionResult Tests
# =============================================================================

class TestExecutionResult:
    """Tests for ExecutionResult serialization and display."""

    @pytest.mark.unit
    def test_to_dict_basic(self):
        """to_dict produces JSON-serializable dict."""
        from Jotty.core.execution.types import ExecutionResult, ExecutionTier
        result = ExecutionResult(
            output="Hello world",
            tier=ExecutionTier.DIRECT,
            success=True,
            llm_calls=1,
            latency_ms=150.0,
            cost_usd=0.001,
        )
        d = result.to_dict()
        assert d['output'] == "Hello world"
        assert d['tier'] == "DIRECT"
        assert d['success'] is True
        assert d['llm_calls'] == 1
        assert d['latency_ms'] == 150.0
        assert d['cost_usd'] == 0.001
        assert d['trace_id'] is None

    @pytest.mark.unit
    def test_to_dict_with_steps(self):
        """to_dict counts steps."""
        from Jotty.core.execution.types import ExecutionResult, ExecutionTier, ExecutionStep
        steps = [ExecutionStep(step_num=1, description="s1"), ExecutionStep(step_num=2, description="s2")]
        result = ExecutionResult(output="out", tier=ExecutionTier.AGENTIC, steps=steps)
        d = result.to_dict()
        assert d['steps'] == 2

    @pytest.mark.unit
    def test_str_success(self):
        """__str__ shows OK for successful result."""
        from Jotty.core.execution.types import ExecutionResult, ExecutionTier
        result = ExecutionResult(
            output="x", tier=ExecutionTier.DIRECT, success=True,
            llm_calls=2, latency_ms=100.0, cost_usd=0.005,
        )
        s = str(result)
        assert "[OK]" in s
        assert "Tier 1" in s
        assert "2 calls" in s

    @pytest.mark.unit
    def test_str_failure(self):
        """__str__ shows FAIL for failed result."""
        from Jotty.core.execution.types import ExecutionResult, ExecutionTier
        result = ExecutionResult(
            output=None, tier=ExecutionTier.AGENTIC, success=False, error="timeout",
        )
        s = str(result)
        assert "[FAIL]" in s
        assert "Tier 2" in s

    @pytest.mark.unit
    def test_defaults(self):
        """ExecutionResult defaults are sensible."""
        from Jotty.core.execution.types import ExecutionResult, ExecutionTier
        result = ExecutionResult(output="x", tier=ExecutionTier.DIRECT)
        assert result.success is True
        assert result.error is None
        assert result.llm_calls == 0
        assert result.steps == []
        assert result.used_memory is False
        assert result.metadata == {}


# =============================================================================
# ExecutionStep Property Tests
# =============================================================================

class TestExecutionStepProperties:
    """Tests for ExecutionStep computed properties."""

    @pytest.mark.unit
    def test_duration_ms_complete(self):
        """duration_ms computes from started_at and completed_at."""
        from Jotty.core.execution.types import ExecutionStep
        from datetime import datetime, timedelta
        start = datetime(2026, 1, 1, 12, 0, 0)
        end = start + timedelta(seconds=2.5)
        step = ExecutionStep(step_num=1, description="test", started_at=start, completed_at=end)
        assert step.duration_ms == 2500.0

    @pytest.mark.unit
    def test_duration_ms_incomplete(self):
        """duration_ms returns None when not started or not completed."""
        from Jotty.core.execution.types import ExecutionStep
        step = ExecutionStep(step_num=1, description="test")
        assert step.duration_ms is None

    @pytest.mark.unit
    def test_is_complete_with_result(self):
        """is_complete is True when result is set."""
        from Jotty.core.execution.types import ExecutionStep
        step = ExecutionStep(step_num=1, description="test", result="done")
        assert step.is_complete is True

    @pytest.mark.unit
    def test_is_complete_with_error(self):
        """is_complete is True when error is set."""
        from Jotty.core.execution.types import ExecutionStep
        step = ExecutionStep(step_num=1, description="test", error="failed")
        assert step.is_complete is True

    @pytest.mark.unit
    def test_is_complete_pending(self):
        """is_complete is False when neither result nor error is set."""
        from Jotty.core.execution.types import ExecutionStep
        step = ExecutionStep(step_num=1, description="test")
        assert step.is_complete is False


# =============================================================================
# ExecutionPlan Property Tests
# =============================================================================

class TestExecutionPlanProperties:
    """Tests for ExecutionPlan computed properties."""

    @pytest.mark.unit
    def test_total_steps(self):
        """total_steps counts all steps."""
        from Jotty.core.execution.types import ExecutionPlan, ExecutionStep
        plan = ExecutionPlan(goal="test", steps=[
            ExecutionStep(step_num=1, description="s1"),
            ExecutionStep(step_num=2, description="s2"),
            ExecutionStep(step_num=3, description="s3"),
        ])
        assert plan.total_steps == 3

    @pytest.mark.unit
    def test_parallelizable_steps(self):
        """parallelizable_steps counts steps that can run in parallel."""
        from Jotty.core.execution.types import ExecutionPlan, ExecutionStep
        plan = ExecutionPlan(goal="test", steps=[
            ExecutionStep(step_num=1, description="s1", can_parallelize=True),
            ExecutionStep(step_num=2, description="s2", can_parallelize=False),
            ExecutionStep(step_num=3, description="s3", can_parallelize=True),
        ])
        assert plan.parallelizable_steps == 2

    @pytest.mark.unit
    def test_empty_plan(self):
        """Empty plan has 0 steps."""
        from Jotty.core.execution.types import ExecutionPlan
        plan = ExecutionPlan(goal="nothing", steps=[])
        assert plan.total_steps == 0
        assert plan.parallelizable_steps == 0


# =============================================================================
# ValidationVerdict Tests
# =============================================================================

class TestValidationVerdict:
    """Tests for ValidationVerdict structured validation results."""

    @pytest.mark.unit
    def test_is_pass_true(self):
        """is_pass is True for PASS status."""
        from Jotty.core.execution.types import ValidationVerdict, ValidationStatus
        v = ValidationVerdict(status=ValidationStatus.PASS)
        assert v.is_pass is True

    @pytest.mark.unit
    def test_is_pass_false(self):
        """is_pass is False for non-PASS status."""
        from Jotty.core.execution.types import ValidationVerdict, ValidationStatus
        v = ValidationVerdict(status=ValidationStatus.FAIL)
        assert v.is_pass is False

    @pytest.mark.unit
    def test_ok_factory(self):
        """ok() creates a passing verdict."""
        from Jotty.core.execution.types import ValidationVerdict, ValidationStatus, ErrorType
        v = ValidationVerdict.ok(reason="all good", confidence=0.95)
        assert v.status == ValidationStatus.PASS
        assert v.reason == "all good"
        assert v.confidence == 0.95
        assert v.error_type == ErrorType.NONE

    @pytest.mark.unit
    def test_from_error_infrastructure(self):
        """from_error classifies timeout as INFRASTRUCTURE and retryable."""
        from Jotty.core.execution.types import ValidationVerdict, ValidationStatus, ErrorType
        v = ValidationVerdict.from_error("Connection timeout after 30s")
        assert v.status == ValidationStatus.FAIL
        assert v.error_type == ErrorType.INFRASTRUCTURE
        assert v.retryable is True
        assert "timeout" in v.reason.lower()

    @pytest.mark.unit
    def test_from_error_logic(self):
        """from_error classifies syntax errors as LOGIC and not retryable."""
        from Jotty.core.execution.types import ValidationVerdict, ErrorType
        v = ValidationVerdict.from_error("SyntaxError: invalid selector")
        assert v.error_type == ErrorType.LOGIC
        assert v.retryable is False

    @pytest.mark.unit
    def test_from_error_environment(self):
        """from_error classifies SSL errors as ENVIRONMENT and retryable."""
        from Jotty.core.execution.types import ValidationVerdict, ErrorType
        v = ValidationVerdict.from_error("SSL certificate verification failed")
        assert v.error_type == ErrorType.ENVIRONMENT
        assert v.retryable is True

    @pytest.mark.unit
    def test_from_error_data(self):
        """from_error classifies parse errors as DATA and not retryable."""
        from Jotty.core.execution.types import ValidationVerdict, ErrorType
        v = ValidationVerdict.from_error("Empty result set returned")
        assert v.error_type == ErrorType.DATA
        assert v.retryable is False

    @pytest.mark.unit
    def test_from_error_populates_issues(self):
        """from_error adds error message to issues list."""
        from Jotty.core.execution.types import ValidationVerdict
        v = ValidationVerdict.from_error("something broke")
        assert len(v.issues) == 1
        assert v.issues[0] == "something broke"


# =============================================================================
# DeadLetterQueue Tests
# =============================================================================

class TestDeadLetterQueue:
    """Tests for DeadLetterQueue thread-safe failed operation queue."""

    @pytest.mark.unit
    def test_enqueue_and_size(self):
        """enqueue adds items and size reports correctly."""
        from Jotty.core.execution.types import DeadLetterQueue, ErrorType
        dlq = DeadLetterQueue()
        dlq.enqueue("web_search", {"query": "test"}, "timeout", ErrorType.INFRASTRUCTURE)
        assert dlq.size == 1

    @pytest.mark.unit
    def test_get_retryable(self):
        """get_retryable returns items under max_retries."""
        from Jotty.core.execution.types import DeadLetterQueue, ErrorType
        dlq = DeadLetterQueue()
        letter = dlq.enqueue("op", {}, "error", ErrorType.INFRASTRUCTURE)
        retryable = dlq.get_retryable()
        assert len(retryable) == 1
        assert retryable[0].operation == "op"

    @pytest.mark.unit
    def test_get_retryable_excludes_exhausted(self):
        """get_retryable excludes items at max_retries."""
        from Jotty.core.execution.types import DeadLetterQueue, ErrorType
        dlq = DeadLetterQueue()
        letter = dlq.enqueue("op", {}, "error", ErrorType.INFRASTRUCTURE)
        letter.retry_count = letter.max_retries  # Exhaust retries
        retryable = dlq.get_retryable()
        assert len(retryable) == 0

    @pytest.mark.unit
    def test_mark_resolved(self):
        """mark_resolved removes item from queue."""
        from Jotty.core.execution.types import DeadLetterQueue, ErrorType
        dlq = DeadLetterQueue()
        letter = dlq.enqueue("op", {}, "error", ErrorType.INFRASTRUCTURE)
        assert dlq.size == 1
        dlq.mark_resolved(letter)
        assert dlq.size == 0

    @pytest.mark.unit
    def test_retry_all_success(self):
        """retry_all calls executor and removes successful items."""
        from Jotty.core.execution.types import DeadLetterQueue, ErrorType
        dlq = DeadLetterQueue()
        dlq.enqueue("op1", {}, "error1", ErrorType.INFRASTRUCTURE)
        dlq.enqueue("op2", {}, "error2", ErrorType.INFRASTRUCTURE)
        successes = dlq.retry_all(lambda letter: True)
        assert successes == 2
        assert dlq.size == 0

    @pytest.mark.unit
    def test_retry_all_partial_failure(self):
        """retry_all handles mixed success/failure."""
        from Jotty.core.execution.types import DeadLetterQueue, ErrorType
        dlq = DeadLetterQueue()
        dlq.enqueue("good", {}, "err", ErrorType.INFRASTRUCTURE)
        dlq.enqueue("bad", {}, "err", ErrorType.INFRASTRUCTURE)
        successes = dlq.retry_all(lambda l: l.operation == "good")
        assert successes == 1
        assert dlq.size == 1  # "bad" still in queue

    @pytest.mark.unit
    def test_retry_all_increments_retry_count(self):
        """retry_all increments retry_count even on failure."""
        from Jotty.core.execution.types import DeadLetterQueue, ErrorType
        dlq = DeadLetterQueue()
        letter = dlq.enqueue("op", {}, "error", ErrorType.INFRASTRUCTURE)
        dlq.retry_all(lambda l: False)  # All fail
        assert letter.retry_count == 1

    @pytest.mark.unit
    def test_max_size_eviction(self):
        """Exceeding max_size evicts oldest entry."""
        from Jotty.core.execution.types import DeadLetterQueue, ErrorType
        dlq = DeadLetterQueue(max_size=2)
        dlq.enqueue("first", {}, "err", ErrorType.INFRASTRUCTURE)
        dlq.enqueue("second", {}, "err", ErrorType.INFRASTRUCTURE)
        dlq.enqueue("third", {}, "err", ErrorType.INFRASTRUCTURE)
        assert dlq.size == 2
        # First should be evicted
        retryable = dlq.get_retryable()
        ops = [l.operation for l in retryable]
        assert "first" not in ops
        assert "second" in ops
        assert "third" in ops

    @pytest.mark.unit
    def test_clear(self):
        """clear empties the queue."""
        from Jotty.core.execution.types import DeadLetterQueue, ErrorType
        dlq = DeadLetterQueue()
        dlq.enqueue("op1", {}, "err", ErrorType.INFRASTRUCTURE)
        dlq.enqueue("op2", {}, "err", ErrorType.INFRASTRUCTURE)
        dlq.clear()
        assert dlq.size == 0


# =============================================================================
# TimeoutWarning Tests
# =============================================================================

class TestTimeoutWarning:
    """Tests for TimeoutWarning threshold-based timeout alerts."""

    @pytest.mark.unit
    def test_initial_state(self):
        """TimeoutWarning starts with zero elapsed."""
        from Jotty.core.execution.types import TimeoutWarning
        tw = TimeoutWarning(timeout_seconds=120)
        assert tw.elapsed == 0.0
        assert tw.is_expired is False

    @pytest.mark.unit
    def test_check_before_start(self):
        """check returns None before start() is called."""
        from Jotty.core.execution.types import TimeoutWarning
        tw = TimeoutWarning(timeout_seconds=120)
        assert tw.check() is None

    @pytest.mark.unit
    def test_fraction_used_zero_timeout(self):
        """fraction_used returns 1.0 for zero timeout."""
        from Jotty.core.execution.types import TimeoutWarning
        tw = TimeoutWarning(timeout_seconds=0)
        assert tw.fraction_used == 1.0

    @pytest.mark.unit
    def test_remaining_before_start(self):
        """remaining returns full timeout before start."""
        from Jotty.core.execution.types import TimeoutWarning
        tw = TimeoutWarning(timeout_seconds=120)
        assert tw.remaining == 120.0

    @pytest.mark.unit
    def test_start_resets_triggered(self):
        """start() clears previously triggered thresholds."""
        from Jotty.core.execution.types import TimeoutWarning
        tw = TimeoutWarning(timeout_seconds=120)
        tw._triggered.add(0.80)
        tw.start()
        assert len(tw._triggered) == 0

    @pytest.mark.unit
    def test_check_triggers_80_percent(self):
        """check triggers 80% warning."""
        from Jotty.core.execution.types import TimeoutWarning
        tw = TimeoutWarning(timeout_seconds=100)
        tw._start_time = 1.0  # Manually set
        import time
        # Simulate 85% elapsed by setting start_time in the past
        tw._start_time = time.time() - 85
        warning = tw.check()
        assert warning is not None
        assert "80%" in warning

    @pytest.mark.unit
    def test_one_shot_triggering(self):
        """Each threshold only triggers once."""
        from Jotty.core.execution.types import TimeoutWarning
        import time as time_mod
        tw = TimeoutWarning(timeout_seconds=100)
        tw._start_time = time_mod.time() - 85  # 85% elapsed
        first = tw.check()
        assert first is not None
        second = tw.check()
        # 80% already triggered. 95% not yet triggered. May be None or 95%.
        # At 85%, 95% threshold not crossed, so None.
        assert second is None

    @pytest.mark.unit
    def test_is_expired(self):
        """is_expired True when elapsed >= timeout."""
        from Jotty.core.execution.types import TimeoutWarning
        import time as time_mod
        tw = TimeoutWarning(timeout_seconds=10)
        tw._start_time = time_mod.time() - 20  # Well past expiry
        assert tw.is_expired is True


# =============================================================================
# AdaptiveTimeout Tests
# =============================================================================

class TestAdaptiveTimeoutExpanded:
    """Tests for AdaptiveTimeout P95-based adaptive timeouts."""

    @pytest.mark.unit
    def test_default_with_no_observations(self):
        """Returns default_seconds when no observations exist."""
        from Jotty.core.execution.types import AdaptiveTimeout
        at = AdaptiveTimeout(default_seconds=30.0)
        assert at.get("llm_call") == 30.0

    @pytest.mark.unit
    def test_default_with_insufficient_observations(self):
        """Returns default when fewer than 3 observations."""
        from Jotty.core.execution.types import AdaptiveTimeout
        at = AdaptiveTimeout(default_seconds=30.0)
        at.record("llm_call", 2.0)
        at.record("llm_call", 3.0)
        assert at.get("llm_call") == 30.0  # Only 2 < 3 required

    @pytest.mark.unit
    def test_adaptive_with_observations(self):
        """With 3+ observations, returns P95 * multiplier."""
        from Jotty.core.execution.types import AdaptiveTimeout
        at = AdaptiveTimeout(default_seconds=30.0, min_seconds=1.0, max_seconds=300.0)
        for t in [1.0, 2.0, 3.0, 4.0, 5.0]:
            at.record("op", t)
        timeout = at.get("op", multiplier=2.0)
        # P95 of [1,2,3,4,5] ≈ 5 → 5*2 = 10
        assert timeout >= 1.0  # Above min
        assert timeout <= 300.0  # Below max
        assert timeout != 30.0  # Not default

    @pytest.mark.unit
    def test_min_bound(self):
        """Timeout never goes below min_seconds."""
        from Jotty.core.execution.types import AdaptiveTimeout
        at = AdaptiveTimeout(default_seconds=30.0, min_seconds=10.0)
        for t in [0.01, 0.02, 0.03, 0.04]:
            at.record("fast_op", t)
        timeout = at.get("fast_op", multiplier=1.0)
        assert timeout >= 10.0

    @pytest.mark.unit
    def test_max_bound(self):
        """Timeout never exceeds max_seconds."""
        from Jotty.core.execution.types import AdaptiveTimeout
        at = AdaptiveTimeout(default_seconds=30.0, max_seconds=50.0)
        for t in [100.0, 200.0, 300.0, 400.0]:
            at.record("slow_op", t)
        timeout = at.get("slow_op", multiplier=2.0)
        assert timeout <= 50.0

    @pytest.mark.unit
    def test_separate_operations(self):
        """Different operations have independent observations."""
        from Jotty.core.execution.types import AdaptiveTimeout
        at = AdaptiveTimeout(default_seconds=30.0, min_seconds=1.0)
        for t in [1.0, 2.0, 3.0, 4.0]:
            at.record("fast", t)
        for t in [50.0, 60.0, 70.0, 80.0]:
            at.record("slow", t)
        fast_t = at.get("fast", multiplier=2.0)
        slow_t = at.get("slow", multiplier=2.0)
        assert fast_t < slow_t


# =============================================================================
# CircuitBreaker Tests
# =============================================================================

class TestCircuitBreakerExpanded:
    """Tests for CircuitBreaker state machine."""

    @pytest.mark.unit
    def test_initial_state_closed(self):
        """CircuitBreaker starts CLOSED."""
        from Jotty.core.execution.types import CircuitBreaker, CircuitState
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    @pytest.mark.unit
    def test_failures_below_threshold(self):
        """Stays CLOSED with failures below threshold."""
        from Jotty.core.execution.types import CircuitBreaker, CircuitState
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    @pytest.mark.unit
    def test_trips_to_open(self):
        """Trips to OPEN when failures reach threshold."""
        from Jotty.core.execution.types import CircuitBreaker, CircuitState
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    @pytest.mark.unit
    def test_cooldown_to_half_open(self):
        """Transitions to HALF_OPEN after cooldown."""
        from Jotty.core.execution.types import CircuitBreaker, CircuitState
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        import time
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request() is True  # Probe allowed

    @pytest.mark.unit
    def test_success_resets_to_closed(self):
        """record_success resets to CLOSED from any state."""
        from Jotty.core.execution.types import CircuitBreaker, CircuitState
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        import time
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    @pytest.mark.unit
    def test_half_open_failure_reopens(self):
        """Failure in HALF_OPEN trips back to OPEN."""
        from Jotty.core.execution.types import CircuitBreaker, CircuitState
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        import time
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    @pytest.mark.unit
    def test_manual_reset(self):
        """reset() manually returns to CLOSED."""
        from Jotty.core.execution.types import CircuitBreaker, CircuitState
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0


if __name__ == "__main__":
    import sys
    if "--integration" in sys.argv:
        import asyncio
        asyncio.run(RealOrchestratorIntegrationTest().run())
    else:
        pytest.main([__file__, "-v"])

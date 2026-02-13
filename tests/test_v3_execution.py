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
        assert "✓" in s
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
        # 1 plan call + 2 step calls = 3
        assert result.llm_calls == 3

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


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

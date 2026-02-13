"""
Swarm Tests — Mocked Unit Tests
=================================

Tests for the swarm infrastructure:
- SwarmTypes: SwarmConfig, SwarmResult, ExecutionTrace, enums
- SwarmRegistry: register, get, list, create
- AgentTeam: define, coordination patterns, merge strategies
- PhaseExecutor: run_phase, run_parallel, build_error_result
- DomainSwarm: execute lifecycle, team coordination, to_composite, validate_output

All tests use mocks — no LLM calls, no API keys.

Fixtures from conftest.py:
    make_concrete_agent — factory for BaseAgent subclasses
    make_swarm_result — factory for SwarmResult
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from Jotty.core.swarms.swarm_types import (
    AgentRole,
    EvaluationResult,
    ImprovementType,
    SwarmConfig,
    SwarmResult,
    ExecutionTrace,
    _split_field,
    _safe_join,
    _safe_num,
)
from Jotty.core.swarms.registry import SwarmRegistry, register_swarm
from Jotty.core.swarms.base.agent_team import (
    AgentTeam,
    AgentSpec,
    TeamResult,
    CoordinationPattern,
    MergeStrategy,
)
from Jotty.core.swarms.base.domain_swarm import DomainSwarm, PhaseExecutor


# =============================================================================
# Swarm Types & Enums
# =============================================================================

@pytest.mark.unit
class TestSwarmTypes:
    """Test swarm data structures and enums."""

    def test_agent_role_values(self):
        """AgentRole enum has expected members."""
        assert AgentRole.ACTOR.value == "actor"
        assert AgentRole.EXPERT.value == "expert"
        assert AgentRole.AUDITOR.value == "auditor"
        assert AgentRole.LEARNER.value == "learner"

    def test_evaluation_result_values(self):
        """EvaluationResult enum has expected members."""
        assert EvaluationResult.EXCELLENT.value == "excellent"
        assert EvaluationResult.FAILED.value == "failed"

    def test_improvement_type_values(self):
        """ImprovementType enum has expected members."""
        assert ImprovementType.PROMPT_REFINEMENT.value == "prompt_refinement"
        assert ImprovementType.WORKFLOW_CHANGE.value == "workflow_change"

    def test_swarm_config_defaults(self):
        """SwarmConfig has sensible defaults."""
        config = SwarmConfig(name="Test", domain="test")
        assert config.name == "Test"
        assert config.domain == "test"
        assert config.enable_learning is True
        assert config.max_retries == 3
        assert config.timeout_seconds == 300

    def test_swarm_result_creation(self, make_swarm_result):
        """SwarmResult factory creates valid instances."""
        result = make_swarm_result(
            success=True,
            output={"analysis": "done"},
            name="AnalysisSwarm",
            domain="analysis",
        )
        assert result.success is True
        assert result.swarm_name == "AnalysisSwarm"
        assert result.output == {"analysis": "done"}
        assert result.execution_time == 1.0
        assert result.error is None
        assert result.agent_traces == []

    def test_swarm_result_with_error(self):
        """SwarmResult stores error information."""
        result = SwarmResult(
            success=False,
            swarm_name="FailSwarm",
            domain="test",
            output={},
            execution_time=0.5,
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_execution_trace_creation(self):
        """ExecutionTrace stores agent execution data."""
        trace = ExecutionTrace(
            agent_name="Architect",
            agent_role=AgentRole.ACTOR,
            input_data={"task": "design"},
            output_data={"design": "done"},
            execution_time=2.5,
            success=True,
        )
        assert trace.agent_name == "Architect"
        assert trace.success is True
        assert trace.execution_time == 2.5
        assert isinstance(trace.timestamp, datetime)


# =============================================================================
# Defensive Utilities
# =============================================================================

@pytest.mark.unit
class TestDefensiveUtilities:
    """Test _split_field, _safe_join, _safe_num utilities."""

    def test_split_field_string(self):
        """_split_field splits pipe-delimited strings."""
        assert _split_field("a|b|c") == ["a", "b", "c"]

    def test_split_field_with_spaces(self):
        """_split_field strips whitespace."""
        assert _split_field("a | b | c") == ["a", "b", "c"]

    def test_split_field_list(self):
        """_split_field coerces list items to strings."""
        assert _split_field(["a", "b"]) == ["a", "b"]

    def test_split_field_dict(self):
        """_split_field flattens dict to key-value strings."""
        result = _split_field({"key": "val"})
        assert result == ["key: val"]

    def test_split_field_none(self):
        """_split_field returns empty list for None."""
        assert _split_field(None) == []

    def test_safe_join_list(self):
        """_safe_join joins list items."""
        assert _safe_join(["a", "b", "c"]) == "a, b, c"

    def test_safe_join_string(self):
        """_safe_join returns string as-is."""
        assert _safe_join("already a string") == "already a string"

    def test_safe_join_empty(self):
        """_safe_join returns empty string for empty input."""
        assert _safe_join([]) == ""
        assert _safe_join(None) == ""

    def test_safe_num_int(self):
        """_safe_num returns int directly."""
        assert _safe_num(42) == 42

    def test_safe_num_float(self):
        """_safe_num returns float directly."""
        assert _safe_num(3.14) == 3.14

    def test_safe_num_string(self):
        """_safe_num parses numeric strings."""
        assert _safe_num("42") == 42.0

    def test_safe_num_invalid(self):
        """_safe_num returns default for non-numeric."""
        assert _safe_num("not a number") == 0
        assert _safe_num(None) == 0
        assert _safe_num(None, default=5) == 5


# =============================================================================
# SwarmRegistry
# =============================================================================

@pytest.mark.unit
class TestSwarmRegistry:
    """Test SwarmRegistry CRUD operations."""

    def setup_method(self):
        """Save and clear registry state before each test."""
        self._saved = dict(SwarmRegistry._swarms)
        SwarmRegistry._swarms.clear()

    def teardown_method(self):
        """Restore registry state after each test."""
        SwarmRegistry._swarms.clear()
        SwarmRegistry._swarms.update(self._saved)

    def test_register_and_get(self):
        """Register and retrieve a swarm class."""
        mock_class = Mock()
        SwarmRegistry.register("test-swarm", mock_class)
        assert SwarmRegistry.get("test-swarm") is mock_class

    def test_get_nonexistent(self):
        """get() returns None for unknown swarm."""
        assert SwarmRegistry.get("nonexistent") is None

    def test_list_all(self):
        """list_all() returns registered swarm names."""
        SwarmRegistry.register("alpha", Mock())
        SwarmRegistry.register("beta", Mock())
        names = SwarmRegistry.list_all()
        assert "alpha" in names
        assert "beta" in names

    def test_create_with_config(self):
        """create() instantiates swarm with config."""
        mock_class = Mock()
        SwarmRegistry.register("creator", mock_class)

        config = SwarmConfig(name="creator", domain="test")
        SwarmRegistry.create("creator", config)

        mock_class.assert_called_once_with(config)

    def test_create_without_config(self):
        """create() generates default config when none provided."""
        mock_class = Mock()
        SwarmRegistry.register("auto", mock_class)

        SwarmRegistry.create("auto")
        call_args = mock_class.call_args
        config = call_args[0][0]
        assert config.name == "auto"
        assert config.domain == "auto"

    def test_create_nonexistent(self):
        """create() returns None for unregistered swarm."""
        assert SwarmRegistry.create("ghost") is None

    def test_register_swarm_decorator(self):
        """@register_swarm decorator registers a class."""
        @register_swarm("decorated")
        class MySwarm:
            pass

        assert SwarmRegistry.get("decorated") is MySwarm


# =============================================================================
# AgentSpec
# =============================================================================

@pytest.mark.unit
class TestAgentSpec:
    """Test AgentSpec auto-naming and fields."""

    def test_auto_attr_name_simple(self):
        """AgentSpec generates _lowercase attr from display name."""
        spec = AgentSpec(agent_class=Mock, display_name="Architect")
        assert spec.attr_name == "_architect"

    def test_auto_attr_name_camel_case(self):
        """AgentSpec converts CamelCase to snake_case."""
        spec = AgentSpec(agent_class=Mock, display_name="TestWriter")
        assert spec.attr_name == "_test_writer"

    def test_custom_attr_name(self):
        """AgentSpec uses custom attr_name when provided."""
        spec = AgentSpec(agent_class=Mock, display_name="X", attr_name="_custom")
        assert spec.attr_name == "_custom"

    def test_spec_fields(self):
        """AgentSpec stores all fields correctly."""
        spec = AgentSpec(
            agent_class=Mock,
            display_name="Worker",
            role="worker",
            priority=3,
        )
        assert spec.display_name == "Worker"
        assert spec.role == "worker"
        assert spec.priority == 3


# =============================================================================
# AgentTeam
# =============================================================================

@pytest.mark.unit
class TestAgentTeam:
    """Test AgentTeam definition and coordination execution."""

    def test_define_simple(self):
        """AgentTeam.define() creates team from tuples."""
        team = AgentTeam.define(
            (Mock, "Agent1"),
            (Mock, "Agent2"),
        )
        assert len(team) == 2
        assert team.pattern == CoordinationPattern.NONE
        assert team.get_agent_names() == ["Agent1", "Agent2"]

    def test_define_with_pattern(self):
        """AgentTeam.define() accepts coordination pattern."""
        team = AgentTeam.define(
            (Mock, "A"),
            (Mock, "B"),
            pattern=CoordinationPattern.PIPELINE,
            merge_strategy=MergeStrategy.FIRST,
        )
        assert team.pattern == CoordinationPattern.PIPELINE
        assert team.merge_strategy == MergeStrategy.FIRST

    def test_add_agent(self):
        """add() fluently adds agents to team."""
        team = AgentTeam()
        team.add(Mock, "First").add(Mock, "Second")
        assert len(team) == 2

    def test_get_agents_by_role(self):
        """get_agents_by_role() filters by role."""
        team = AgentTeam.define(
            (Mock, "Manager", None, "manager"),
            (Mock, "Worker1", None, "worker"),
            (Mock, "Worker2", None, "worker"),
        )
        workers = team.get_agents_by_role("worker")
        assert len(workers) == 2

    def test_get_ordered_agents(self):
        """get_ordered_agents() sorts by priority descending."""
        team = AgentTeam.define(
            (Mock, "Low", None, None, 1),
            (Mock, "High", None, None, 10),
            (Mock, "Mid", None, None, 5),
        )
        ordered = team.get_ordered_agents()
        names = [spec.display_name for _, spec in ordered]
        assert names == ["High", "Mid", "Low"]

    def test_iter_team(self):
        """AgentTeam is iterable over (attr_name, spec) pairs."""
        team = AgentTeam.define((Mock, "A"), (Mock, "B"))
        items = list(team)
        assert len(items) == 2
        assert all(isinstance(name, str) for name, _ in items)

    @pytest.mark.asyncio
    async def test_execute_none_pattern(self):
        """NONE pattern returns empty outputs for swarm to handle."""
        team = AgentTeam.define((Mock, "A"))
        team.set_instances({"_a": Mock()})

        result = await team.execute("task")

        assert result.success is True
        assert result.outputs == {}
        assert result.pattern == CoordinationPattern.NONE

    @pytest.mark.asyncio
    async def test_execute_without_instances_raises(self):
        """execute() raises when instances not set."""
        team = AgentTeam.define((Mock, "A"))

        with pytest.raises(RuntimeError, match="instances not set"):
            await team.execute("task")

    @pytest.mark.asyncio
    async def test_execute_pipeline(self, make_concrete_agent):
        """Pipeline executes agents sequentially."""
        a1 = make_concrete_agent(name="A1", output={"step": "1"})
        a2 = make_concrete_agent(name="A2", output={"step": "2"})

        team = AgentTeam.define(
            (Mock, "A1", "_a1", None, 2),
            (Mock, "A2", "_a2", None, 1),
            pattern=CoordinationPattern.PIPELINE,
        )
        team.set_instances({"_a1": a1, "_a2": a2})

        result = await team.execute("start")

        assert result.success is True
        assert result.pattern == CoordinationPattern.PIPELINE
        assert "A1" in result.execution_order
        assert "A2" in result.execution_order

    @pytest.mark.asyncio
    async def test_execute_pipeline_error_stops(self):
        """Pipeline stops when agent raises an uncaught exception."""
        # Use a mock agent that raises directly (bypasses BaseAgent wrapping)
        mock_agent = AsyncMock()
        mock_agent.execute = AsyncMock(side_effect=RuntimeError("fail"))

        a2 = AsyncMock()
        a2.execute = AsyncMock(return_value=Mock(output="never"))

        team = AgentTeam.define(
            (Mock, "A1", "_a1", None, 2),
            (Mock, "A2", "_a2", None, 1),
            pattern=CoordinationPattern.PIPELINE,
        )
        team.set_instances({"_a1": mock_agent, "_a2": a2})

        result = await team.execute("start")

        assert result.success is False
        assert "A1" in result.errors
        a2.execute.assert_not_awaited()  # Pipeline stopped before A2

    @pytest.mark.asyncio
    async def test_execute_parallel(self, make_concrete_agent):
        """Parallel executes all agents concurrently."""
        a1 = make_concrete_agent(name="P1", output={"p1": "done"})
        a2 = make_concrete_agent(name="P2", output={"p2": "done"})

        team = AgentTeam.define(
            (Mock, "P1", "_p1"),
            (Mock, "P2", "_p2"),
            pattern=CoordinationPattern.PARALLEL,
            merge_strategy=MergeStrategy.COMBINE,
        )
        team.set_instances({"_p1": a1, "_p2": a2})

        result = await team.execute("task")

        assert result.success is True
        assert result.pattern == CoordinationPattern.PARALLEL
        assert "P1" in result.outputs
        assert "P2" in result.outputs

    @pytest.mark.asyncio
    async def test_execute_consensus(self, make_concrete_agent):
        """Consensus votes on parallel results."""
        a1 = make_concrete_agent(name="V1", output="yes")
        a2 = make_concrete_agent(name="V2", output="yes")
        a3 = make_concrete_agent(name="V3", output="no")

        team = AgentTeam.define(
            (Mock, "V1", "_v1"),
            (Mock, "V2", "_v2"),
            (Mock, "V3", "_v3"),
            pattern=CoordinationPattern.CONSENSUS,
        )
        team.set_instances({"_v1": a1, "_v2": a2, "_v3": a3})

        result = await team.execute("vote")

        assert result.success is True
        assert result.merged_output == "yes"
        assert "votes" in result.metadata

    @pytest.mark.asyncio
    async def test_execute_round_robin(self, make_concrete_agent):
        """Round robin assigns tasks in rotation."""
        a1 = make_concrete_agent(name="RR1", output="r1")
        a2 = make_concrete_agent(name="RR2", output="r2")

        team = AgentTeam.define(
            (Mock, "RR1", "_rr1"),
            (Mock, "RR2", "_rr2"),
            pattern=CoordinationPattern.ROUND_ROBIN,
        )
        team.set_instances({"_rr1": a1, "_rr2": a2})

        result = await team.execute(["t1", "t2", "t3"])

        assert result.success is True
        assert result.pattern == CoordinationPattern.ROUND_ROBIN
        assert len(result.outputs) == 3

    def test_merge_combine(self):
        """COMBINE returns dict as-is."""
        team = AgentTeam(merge_strategy=MergeStrategy.COMBINE)
        assert team._merge_outputs({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_merge_first(self):
        """FIRST returns first value."""
        team = AgentTeam(merge_strategy=MergeStrategy.FIRST)
        result = team._merge_outputs({"a": "first", "b": "second"})
        assert result == "first"

    def test_merge_concat(self):
        """CONCAT joins string representations."""
        team = AgentTeam(merge_strategy=MergeStrategy.CONCAT)
        result = team._merge_outputs({"a": "hello", "b": "world"})
        assert "hello" in result
        assert "world" in result

    def test_merge_vote(self):
        """VOTE picks most common output."""
        team = AgentTeam(merge_strategy=MergeStrategy.VOTE)
        result = team._merge_outputs({"a": "yes", "b": "yes", "c": "no"})
        assert result == "yes"

    def test_merge_empty(self):
        """Empty outputs returns None."""
        for strategy in MergeStrategy:
            team = AgentTeam(merge_strategy=strategy)
            assert team._merge_outputs({}) is None


# =============================================================================
# PhaseExecutor
# =============================================================================

@pytest.mark.unit
class TestPhaseExecutor:
    """Test PhaseExecutor phase management."""

    def _make_domain_swarm(self):
        """Create a minimal concrete DomainSwarm for testing."""

        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="Test",
                    domain="test",
                    output={"done": True},
                    execution_time=1.0,
                )

        config = SwarmConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)
        swarm._initialized = True
        swarm._traces = []
        return swarm

    @pytest.mark.asyncio
    async def test_run_phase_success(self):
        """run_phase executes coroutine and traces result."""
        swarm = self._make_domain_swarm()
        swarm._trace_phase = Mock()
        executor = PhaseExecutor(swarm)

        async def coro():
            return {"result": "done"}

        result = await executor.run_phase(
            1, "Analysis", "Analyzer", AgentRole.ACTOR, coro(),
        )

        assert result == {"result": "done"}
        swarm._trace_phase.assert_called_once()
        call_kwargs = swarm._trace_phase.call_args
        assert call_kwargs[0][0] == "Analyzer"
        assert call_kwargs[1]["success"] is True

    @pytest.mark.asyncio
    async def test_run_phase_detects_error_dict(self):
        """run_phase marks error dicts as failures."""
        swarm = self._make_domain_swarm()
        swarm._trace_phase = Mock()
        executor = PhaseExecutor(swarm)

        async def coro():
            return {"error": "something broke"}

        result = await executor.run_phase(
            1, "Phase", "Agent", AgentRole.ACTOR, coro(),
        )

        assert result == {"error": "something broke"}
        call_kwargs = swarm._trace_phase.call_args
        assert call_kwargs[1]["success"] is False

    @pytest.mark.asyncio
    async def test_run_parallel_success(self):
        """run_parallel executes tasks concurrently."""
        swarm = self._make_domain_swarm()
        swarm._trace_phase = Mock()
        executor = PhaseExecutor(swarm)

        async def task1():
            return {"t1": "done"}

        async def task2():
            return {"t2": "done"}

        tasks = [
            ("Agent1", AgentRole.ACTOR, task1(), ["tool1"]),
            ("Agent2", AgentRole.ACTOR, task2(), ["tool2"]),
        ]

        results = await executor.run_parallel(1, "Parallel", tasks)

        assert len(results) == 2
        assert results[0] == {"t1": "done"}
        assert results[1] == {"t2": "done"}
        assert swarm._trace_phase.call_count == 2

    @pytest.mark.asyncio
    async def test_run_parallel_handles_exceptions(self):
        """run_parallel converts exceptions to error dicts."""
        swarm = self._make_domain_swarm()
        swarm._trace_phase = Mock()
        executor = PhaseExecutor(swarm)

        async def failing():
            raise ValueError("boom")

        async def succeeding():
            return {"ok": True}

        tasks = [
            ("Fail", AgentRole.ACTOR, failing(), []),
            ("Pass", AgentRole.ACTOR, succeeding(), []),
        ]

        results = await executor.run_parallel(1, "Mixed", tasks)

        assert results[0] == {"error": "boom"}
        assert results[1] == {"ok": True}

    def test_build_error_result(self):
        """build_error_result creates error SwarmResult."""
        swarm = self._make_domain_swarm()
        executor = PhaseExecutor(swarm)

        result = executor.build_error_result(
            SwarmResult,
            ValueError("test error"),
            "TestSwarm",
            "test",
        )

        assert result.success is False
        assert result.error == "test error"
        assert result.swarm_name == "TestSwarm"
        assert result.execution_time >= 0

    def test_elapsed(self):
        """elapsed() returns non-negative seconds."""
        swarm = self._make_domain_swarm()
        executor = PhaseExecutor(swarm)
        assert executor.elapsed() >= 0


# =============================================================================
# DomainSwarm
# =============================================================================

@pytest.mark.unit
class TestDomainSwarm:
    """Test DomainSwarm lifecycle and features."""

    def _make_test_swarm(self, output=None, raises=None):
        """Create a concrete DomainSwarm for testing."""

        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                if raises:
                    raise raises
                return output or SwarmResult(
                    success=True,
                    swarm_name="TestSwarm",
                    domain="test",
                    output={"done": True},
                    execution_time=1.0,
                )

        config = SwarmConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)
        swarm._initialized = True
        swarm._agents_initialized = True
        return swarm

    @pytest.mark.asyncio
    async def test_execute_calls_domain_logic(self):
        """execute() calls _execute_domain and returns result."""
        swarm = self._make_test_swarm()
        # Stub out learning hooks
        swarm._pre_execute_learning = AsyncMock()
        swarm._post_execute_learning = AsyncMock()

        result = await swarm.execute()

        assert result.success is True
        assert result.output == {"done": True}

    @pytest.mark.asyncio
    async def test_execute_calls_pre_learning(self):
        """execute() calls _pre_execute_learning before domain logic."""
        swarm = self._make_test_swarm()
        swarm._pre_execute_learning = AsyncMock()
        swarm._post_execute_learning = AsyncMock()

        await swarm.execute()

        swarm._pre_execute_learning.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_calls_post_learning(self):
        """execute() calls _post_execute_learning after domain logic."""
        swarm = self._make_test_swarm()
        swarm._pre_execute_learning = AsyncMock()
        swarm._post_execute_learning = AsyncMock()

        await swarm.execute()

        swarm._post_execute_learning.assert_awaited()

    @pytest.mark.asyncio
    async def test_execute_pre_learning_failure_non_fatal(self):
        """Pre-learning failure doesn't stop execution."""
        swarm = self._make_test_swarm()
        swarm._pre_execute_learning = AsyncMock(side_effect=RuntimeError("learning broke"))
        swarm._post_execute_learning = AsyncMock()

        result = await swarm.execute()

        assert result.success is True  # Execution still succeeds

    @pytest.mark.asyncio
    async def test_safe_execute_domain_success(self):
        """_safe_execute_domain wraps execute_fn with learning."""
        swarm = self._make_test_swarm()
        swarm._post_execute_learning = AsyncMock()
        swarm._get_active_tools = Mock(return_value=["tool1"])

        expected_result = SwarmResult(
            success=True,
            swarm_name="TestSwarm",
            domain="test",
            output={"result": "ok"},
            execution_time=0.5,
        )

        async def execute_fn(executor):
            return expected_result

        result = await swarm._safe_execute_domain(
            task_type="test_task",
            default_tools=["tool1"],
            result_class=SwarmResult,
            execute_fn=execute_fn,
        )

        assert result.success is True
        assert swarm._learning_recorded is True
        swarm._post_execute_learning.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_safe_execute_domain_error(self):
        """_safe_execute_domain catches exceptions and returns error result."""
        swarm = self._make_test_swarm()
        swarm._post_execute_learning = AsyncMock()
        swarm._get_active_tools = Mock(return_value=[])

        async def execute_fn(executor):
            raise ValueError("domain error")

        result = await swarm._safe_execute_domain(
            task_type="test_task",
            default_tools=[],
            result_class=SwarmResult,
            execute_fn=execute_fn,
        )

        assert result.success is False
        assert "domain error" in result.error
        assert swarm._learning_recorded is True

    def test_phase_executor_creation(self):
        """_phase_executor() creates PhaseExecutor bound to swarm."""
        swarm = self._make_test_swarm()
        executor = swarm._phase_executor()
        assert isinstance(executor, PhaseExecutor)
        assert executor.swarm is swarm

    def test_get_agents_empty(self):
        """get_agents() returns empty dict when no AGENT_TEAM."""
        swarm = self._make_test_swarm()
        assert swarm.get_agents() == {}

    def test_has_team_coordination_false(self):
        """has_team_coordination() returns False with no team."""
        swarm = self._make_test_swarm()
        assert swarm.has_team_coordination() is False

    @pytest.mark.asyncio
    async def test_execute_team_without_team_raises(self):
        """execute_team() raises when no AGENT_TEAM."""
        swarm = self._make_test_swarm()
        with pytest.raises(RuntimeError, match="No AGENT_TEAM"):
            await swarm.execute_team("task")

    def test_to_composite(self):
        """to_composite() creates CompositeAgent wrapping the swarm."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent
        swarm = self._make_test_swarm()
        composite = swarm.to_composite()
        assert isinstance(composite, CompositeAgent)

    def test_repr(self):
        """__repr__ shows agents count and pattern."""
        swarm = self._make_test_swarm()
        r = repr(swarm)
        assert "TestSwarm" in r
        assert "agents=0" in r

    @pytest.mark.asyncio
    async def test_validate_output_fields_auto_populates(self):
        """_validate_output_fields auto-populates missing fields."""
        swarm = self._make_test_swarm()

        # Mock a schema with expected outputs
        mock_schema = Mock()
        mock_param = Mock()
        mock_param.name = "analysis"
        mock_param.type_hint = "str"
        mock_schema.outputs = [mock_param]
        swarm._io_schema = mock_schema

        result = SwarmResult(
            success=True,
            swarm_name="Test",
            domain="test",
            output={},  # Missing 'analysis' field
            execution_time=1.0,
        )

        swarm._validate_output_fields(result)

        assert "analysis" in result.output
        assert result.output["analysis"] == ""

    def test_get_io_schema_none(self):
        """get_io_schema() returns None when no SWARM_SIGNATURE."""
        swarm = self._make_test_swarm()
        assert swarm.get_io_schema() is None

    @pytest.mark.asyncio
    async def test_execute_attaches_traces(self):
        """execute() copies collected traces to result."""
        swarm = self._make_test_swarm()
        swarm._pre_execute_learning = AsyncMock()
        swarm._post_execute_learning = AsyncMock()
        swarm._traces = [
            ExecutionTrace(
                agent_name="Agent1",
                agent_role=AgentRole.ACTOR,
                input_data={},
                output_data={},
                execution_time=1.0,
                success=True,
            )
        ]

        result = await swarm.execute()

        assert len(result.agent_traces) == 1
        assert result.agent_traces[0].agent_name == "Agent1"


# =============================================================================
# TeamResult
# =============================================================================

@pytest.mark.unit
class TestTeamResult:
    """Test TeamResult data class."""

    def test_team_result_creation(self):
        """TeamResult stores execution data."""
        result = TeamResult(
            success=True,
            outputs={"A": "output_a", "B": "output_b"},
            merged_output="combined",
            pattern=CoordinationPattern.PARALLEL,
            execution_order=["A", "B"],
        )
        assert result.success is True
        assert len(result.outputs) == 2
        assert result.merged_output == "combined"
        assert result.errors == {}

    def test_team_result_with_errors(self):
        """TeamResult stores per-agent errors."""
        result = TeamResult(
            success=False,
            outputs={"A": "ok"},
            errors={"B": "failed", "C": "timeout"},
        )
        assert result.success is False
        assert len(result.errors) == 2

    def test_team_result_metadata(self):
        """TeamResult stores metadata."""
        result = TeamResult(
            success=True,
            outputs={},
            metadata={"rounds": 3, "votes": {"yes": 2}},
        )
        assert result.metadata["rounds"] == 3


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

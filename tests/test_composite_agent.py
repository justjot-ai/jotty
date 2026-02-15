"""
Tests for CompositeAgent, CompositeAgentConfig, and UnifiedResult.

Covers the bridge pattern that unifies Agent and Swarm hierarchies,
including configuration defaults, result conversion, sub-agent management,
and factory methods (from_swarm, compose).
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from Jotty.core.modes.agent.base.composite_agent import (
    CompositeAgent,
    CompositeAgentConfig,
    UnifiedResult,
)
from Jotty.core.intelligence.swarms.swarm_types import SwarmResult, ExecutionTrace, AgentRole
from Jotty.core.intelligence.swarms.base.agent_team import CoordinationPattern, MergeStrategy


def _make_mock_base_agent(name="MockAgent", timeout=120.0):
    """Create a mock BaseAgent with the minimum required config surface."""
    agent = MagicMock()
    agent.config = MagicMock()
    agent.config.name = name
    agent.config.timeout = timeout
    agent.execute = AsyncMock()
    agent.get_io_schema = MagicMock(return_value=None)
    agent.to_dict = MagicMock(return_value={"name": name})
    return agent


def _make_mock_domain_swarm(name="TestSwarm", timeout_seconds=300):
    """Create a mock DomainSwarm with config.timeout_seconds."""
    swarm = MagicMock()
    swarm.__class__.__name__ = name
    swarm.config = MagicMock()
    swarm.config.name = name
    swarm.config.timeout_seconds = timeout_seconds
    swarm.execute = AsyncMock()
    return swarm


# =============================================================================
# CompositeAgentConfig
# =============================================================================

@pytest.mark.unit
class TestCompositeAgentConfig:
    """Tests for CompositeAgentConfig defaults and custom values."""

    def test_defaults(self):
        """CompositeAgentConfig defaults to PIPELINE coordination and COMBINE merge."""
        config = CompositeAgentConfig(name="TestComposite")
        assert config.coordination_pattern == CoordinationPattern.PIPELINE
        assert config.merge_strategy == MergeStrategy.COMBINE
        assert config.name == "TestComposite"

    def test_custom_values(self):
        """CompositeAgentConfig accepts custom coordination pattern and merge strategy."""
        config = CompositeAgentConfig(
            name="ParallelComposite",
            coordination_pattern=CoordinationPattern.PARALLEL,
            merge_strategy=MergeStrategy.BEST,
        )
        assert config.coordination_pattern == CoordinationPattern.PARALLEL
        assert config.merge_strategy == MergeStrategy.BEST
        assert config.name == "ParallelComposite"


# =============================================================================
# UnifiedResult
# =============================================================================

@pytest.mark.unit
class TestUnifiedResult:
    """Tests for UnifiedResult creation and bidirectional conversions."""

    def test_creation_with_all_fields(self):
        """UnifiedResult stores all provided fields correctly."""
        traces = [{"agent": "a1", "time": 1.0}]
        evaluation = {"score": 0.95}
        improvements = ["use caching"]
        metadata = {"domain": "coding", "version": "1.0"}

        result = UnifiedResult(
            success=True,
            output={"code": "print('hello')"},
            name="TestAgent",
            execution_time=2.5,
            error=None,
            metadata=metadata,
            agent_traces=traces,
            evaluation=evaluation,
            improvements=improvements,
        )

        assert result.success is True
        assert result.output == {"code": "print('hello')"}
        assert result.name == "TestAgent"
        assert result.execution_time == 2.5
        assert result.error is None
        assert result.metadata == metadata
        assert result.agent_traces == traces
        assert result.evaluation == evaluation
        assert result.improvements == improvements

    def test_to_swarm_result_conversion(self):
        """to_swarm_result produces a SwarmResult with mapped fields."""
        result = UnifiedResult(
            success=True,
            output={"analysis": "done"},
            name="AnalysisComposite",
            execution_time=3.0,
            error=None,
            metadata={"domain": "research"},
            agent_traces=[],
            evaluation=None,
            improvements=[],
        )

        swarm_result = result.to_swarm_result()

        assert isinstance(swarm_result, SwarmResult)
        assert swarm_result.success is True
        assert swarm_result.swarm_name == "AnalysisComposite"
        assert swarm_result.domain == "research"
        assert swarm_result.output == {"analysis": "done"}
        assert swarm_result.execution_time == 3.0
        assert swarm_result.error is None

    def test_to_swarm_result_wraps_non_dict_output(self):
        """to_swarm_result wraps non-dict output in {'result': output}."""
        result = UnifiedResult(
            success=True,
            output="plain string output",
            name="StringAgent",
            execution_time=1.0,
        )

        swarm_result = result.to_swarm_result()
        assert swarm_result.output == {"result": "plain string output"}

    def test_from_swarm_result_classmethod(self):
        """from_swarm_result creates a UnifiedResult from SwarmResult."""
        trace = ExecutionTrace(
            agent_name="worker",
            agent_role=AgentRole.ACTOR,
            input_data={"task": "build"},
            output_data={"code": "done"},
            execution_time=1.2,
            success=True,
        )
        swarm_result = SwarmResult(
            success=True,
            swarm_name="CodingSwarm",
            domain="coding",
            output={"code": "print(1)"},
            execution_time=5.0,
            agent_traces=[trace],
            evaluation=None,
            improvements=[],
            error=None,
            metadata={"tier": "research"},
        )

        unified = UnifiedResult.from_swarm_result(swarm_result)

        assert unified.success is True
        assert unified.output == {"code": "print(1)"}
        assert unified.name == "CodingSwarm"
        assert unified.execution_time == 5.0
        assert unified.error is None
        assert unified.metadata == {"tier": "research"}
        assert len(unified.agent_traces) == 1
        assert unified.agent_traces[0] is trace


# =============================================================================
# CompositeAgent
# =============================================================================

@pytest.mark.unit
class TestCompositeAgent:
    """Tests for CompositeAgent init, sub-agent management, and factory methods."""

    def test_init_with_default_config(self):
        """CompositeAgent() uses default CompositeAgentConfig with name 'CompositeAgent'."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            agent = CompositeAgent()
            assert agent.config.name == "CompositeAgent"
            assert agent.config.coordination_pattern == CoordinationPattern.PIPELINE
            assert agent.config.merge_strategy == MergeStrategy.COMBINE
            assert agent.signature is None
            assert agent._sub_agents == {}
            assert agent._wrapped_swarm is None

    def test_add_agent_and_get_agent(self):
        """add_agent stores a sub-agent retrievable via get_agent."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            mock_agent = _make_mock_base_agent("Coder")

            composite.add_agent("coder", mock_agent)

            retrieved = composite.get_agent("coder")
            assert retrieved is mock_agent
            assert composite.get_agent("nonexistent") is None

    def test_add_agent_is_chainable(self):
        """add_agent returns self, allowing method chaining."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            agent_a = _make_mock_base_agent("A")
            agent_b = _make_mock_base_agent("B")

            result = composite.add_agent("a", agent_a).add_agent("b", agent_b)

            assert result is composite
            assert composite.get_agent("a") is agent_a
            assert composite.get_agent("b") is agent_b

    def test_remove_agent(self):
        """remove_agent removes an existing sub-agent by name."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            mock_agent = _make_mock_base_agent("Temp")

            composite.add_agent("temp", mock_agent)
            assert composite.get_agent("temp") is mock_agent

            returned = composite.remove_agent("temp")
            assert returned is composite
            assert composite.get_agent("temp") is None

    def test_remove_agent_missing_key_is_safe(self):
        """remove_agent on a nonexistent name does not raise."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            composite.remove_agent("nonexistent")  # should not raise

    def test_sub_agents_property_returns_copy(self):
        """sub_agents property returns a shallow copy, not the internal dict."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            mock_agent = _make_mock_base_agent("Worker")
            composite.add_agent("worker", mock_agent)

            snapshot = composite.sub_agents
            assert snapshot == {"worker": mock_agent}
            assert snapshot is not composite._sub_agents

            # Mutating the snapshot must not affect the composite
            snapshot["intruder"] = _make_mock_base_agent("Intruder")
            assert "intruder" not in composite._sub_agents

    def test_from_swarm_wraps_mock_swarm(self):
        """from_swarm creates a CompositeAgent that delegates to the swarm."""
        mock_swarm = _make_mock_domain_swarm("CodingSwarm", timeout_seconds=300)

        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent.from_swarm(mock_swarm)

            assert composite.config.name == "CodingSwarm"
            assert composite.config.timeout == 300.0
            assert composite._wrapped_swarm is mock_swarm
            assert composite.signature is None

    def test_compose_creates_from_named_agents(self):
        """compose builds a CompositeAgent with named sub-agents and correct config."""
        agent_a = _make_mock_base_agent("Architect", timeout=60.0)
        agent_b = _make_mock_base_agent("Developer", timeout=90.0)

        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent.compose(
                "DevPipeline",
                coordination=CoordinationPattern.PIPELINE,
                merge_strategy=MergeStrategy.CONCAT,
                architect=agent_a,
                developer=agent_b,
            )

            assert composite.config.name == "DevPipeline"
            assert composite.config.coordination_pattern == CoordinationPattern.PIPELINE
            assert composite.config.merge_strategy == MergeStrategy.CONCAT
            # Pipeline timeout = sum of sub-agent timeouts
            assert composite.config.timeout == 150.0
            assert composite.config.max_retries == 1
            assert composite.get_agent("architect") is agent_a
            assert composite.get_agent("developer") is agent_b
            assert len(composite.sub_agents) == 2


# Import _COORDINATION_DISPATCH for dispatch table tests
from Jotty.core.modes.agent.base.composite_agent import _COORDINATION_DISPATCH
from Jotty.core.modes.agent.base.base_agent import AgentResult


def _make_successful_agent_result(output, agent_name="TestAgent"):
    """Create a successful AgentResult for mock returns."""
    return AgentResult(
        success=True,
        output=output,
        agent_name=agent_name,
        execution_time=0.5,
        metadata={},
    )


def _make_failed_agent_result(error_msg, agent_name="TestAgent"):
    """Create a failed AgentResult for mock returns."""
    return AgentResult(
        success=False,
        output=None,
        agent_name=agent_name,
        execution_time=0.1,
        error=error_msg,
        metadata={},
    )


# =============================================================================
# TestCompositeAgentExecution
# =============================================================================

@pytest.mark.unit
class TestCompositeAgentExecution:
    """Tests for execution coordination patterns: pipeline, parallel, consensus."""

    @pytest.mark.asyncio
    async def test_pipeline_sequential_execution(self):
        """Pipeline executes sub-agents in order, chaining outputs."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            composite.config.coordination_pattern = CoordinationPattern.PIPELINE

            agent_a = _make_mock_base_agent("AgentA")
            agent_a.execute.return_value = _make_successful_agent_result(
                {"step": "a_done"}, "AgentA"
            )
            agent_b = _make_mock_base_agent("AgentB")
            agent_b.execute.return_value = _make_successful_agent_result(
                {"step": "b_done"}, "AgentB"
            )

            composite.add_agent("a", agent_a).add_agent("b", agent_b)

            result = await composite._execute_pipeline(task="start")

            assert result.success is True
            assert agent_a.execute.call_count == 1
            assert agent_b.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_pipeline_dict_output_merging(self):
        """Pipeline merges dict outputs as kwargs for the next agent."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()

            agent_a = _make_mock_base_agent("AgentA")
            agent_a.execute.return_value = _make_successful_agent_result(
                {"analysis": "complete"}, "AgentA"
            )
            agent_b = _make_mock_base_agent("AgentB")
            agent_b.execute.return_value = _make_successful_agent_result(
                {"final": "output"}, "AgentB"
            )

            composite.add_agent("a", agent_a).add_agent("b", agent_b)

            result = await composite._execute_pipeline(task="go")

            # agent_b should have been called with analysis="complete" merged in
            call_kwargs = agent_b.execute.call_args[1]
            assert call_kwargs.get("analysis") == "complete"

    @pytest.mark.asyncio
    async def test_pipeline_stops_on_first_failure(self):
        """Pipeline stops at first failed agent and returns failure."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()

            agent_a = _make_mock_base_agent("AgentA")
            agent_a.execute.return_value = _make_failed_agent_result("boom", "AgentA")
            agent_b = _make_mock_base_agent("AgentB")

            composite.add_agent("a", agent_a).add_agent("b", agent_b)

            result = await composite._execute_pipeline(task="go")

            assert result.success is False
            assert "Pipeline failed at 'a'" in result.error
            agent_b.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_pipeline_non_dict_output_sets_task_key(self):
        """Pipeline sets 'task' and 'previous_output' keys for non-dict outputs."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()

            agent_a = _make_mock_base_agent("AgentA")
            agent_a.execute.return_value = _make_successful_agent_result(
                "plain string output", "AgentA"
            )
            agent_b = _make_mock_base_agent("AgentB")
            agent_b.execute.return_value = _make_successful_agent_result(
                "final", "AgentB"
            )

            composite.add_agent("a", agent_a).add_agent("b", agent_b)

            result = await composite._execute_pipeline(task="go")

            call_kwargs = agent_b.execute.call_args[1]
            assert call_kwargs.get("task") == "plain string output"
            assert call_kwargs.get("previous_output") == "plain string output"

    @pytest.mark.asyncio
    async def test_pipeline_metadata_contains_stages(self):
        """Pipeline result metadata contains pipeline_stages list."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()

            agent_a = _make_mock_base_agent("A")
            agent_a.execute.return_value = _make_successful_agent_result("ok", "A")
            agent_b = _make_mock_base_agent("B")
            agent_b.execute.return_value = _make_successful_agent_result("ok", "B")

            composite.add_agent("first", agent_a).add_agent("second", agent_b)

            result = await composite._execute_pipeline()

            assert "pipeline_stages" in result.metadata
            assert result.metadata["pipeline_stages"] == ["first", "second"]

    @pytest.mark.asyncio
    async def test_parallel_concurrent_execution(self):
        """Parallel runs all agents and merges successful outputs."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            composite.config.merge_strategy = MergeStrategy.COMBINE

            agent_a = _make_mock_base_agent("AgentA")
            agent_a.execute.return_value = _make_successful_agent_result(
                {"from": "a"}, "AgentA"
            )
            agent_b = _make_mock_base_agent("AgentB")
            agent_b.execute.return_value = _make_successful_agent_result(
                {"from": "b"}, "AgentB"
            )

            composite.add_agent("a", agent_a).add_agent("b", agent_b)

            result = await composite._execute_parallel(task="go")

            assert result.success is True
            assert agent_a.execute.call_count == 1
            assert agent_b.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_parallel_continues_despite_failures(self):
        """Parallel continues even when some agents fail, reports errors."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            composite.config.merge_strategy = MergeStrategy.COMBINE

            agent_a = _make_mock_base_agent("AgentA")
            agent_a.execute.return_value = _make_successful_agent_result(
                {"from": "a"}, "AgentA"
            )
            agent_b = _make_mock_base_agent("AgentB")
            agent_b.execute.return_value = _make_failed_agent_result("fail!", "AgentB")

            composite.add_agent("a", agent_a).add_agent("b", agent_b)

            result = await composite._execute_parallel(task="go")

            # Still succeeds because at least one agent succeeded
            assert result.success is True
            assert result.error is not None
            assert "b" in result.error

    @pytest.mark.asyncio
    async def test_parallel_all_fail_returns_empty_output(self):
        """Parallel with all failures returns success=False (no outputs)."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            composite.config.merge_strategy = MergeStrategy.COMBINE

            agent_a = _make_mock_base_agent("AgentA")
            agent_a.execute.return_value = _make_failed_agent_result("fail a", "AgentA")
            agent_b = _make_mock_base_agent("AgentB")
            agent_b.execute.return_value = _make_failed_agent_result("fail b", "AgentB")

            composite.add_agent("a", agent_a).add_agent("b", agent_b)

            result = await composite._execute_parallel(task="go")

            # success=bool(outputs) where outputs is empty
            assert result.success is False

    @pytest.mark.asyncio
    async def test_parallel_exception_recorded_as_error(self):
        """Parallel records raised exceptions in errors dict."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            composite.config.merge_strategy = MergeStrategy.COMBINE

            agent_a = _make_mock_base_agent("AgentA")
            agent_a.execute.return_value = _make_successful_agent_result("ok", "AgentA")
            agent_b = _make_mock_base_agent("AgentB")
            agent_b.execute.side_effect = RuntimeError("unexpected crash")

            composite.add_agent("a", agent_a).add_agent("b", agent_b)

            result = await composite._execute_parallel(task="go")

            assert result.success is True
            assert "b" in result.metadata.get("errors", {})

    @pytest.mark.asyncio
    async def test_consensus_majority_success(self):
        """Consensus passes when >50% of agents succeed."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()

            agent_a = _make_mock_base_agent("AgentA")
            agent_a.execute.return_value = _make_successful_agent_result("ok", "AgentA")
            agent_b = _make_mock_base_agent("AgentB")
            agent_b.execute.return_value = _make_successful_agent_result("ok", "AgentB")
            agent_c = _make_mock_base_agent("AgentC")
            agent_c.execute.return_value = _make_failed_agent_result("fail", "AgentC")

            composite.add_agent("a", agent_a).add_agent("b", agent_b).add_agent("c", agent_c)

            result = await composite._execute_consensus(task="vote")

            # 2 out of 3 succeed → majority
            assert result.success is True
            assert result.metadata["votes_success"] == 2
            assert result.metadata["votes_total"] == 3

    @pytest.mark.asyncio
    async def test_consensus_majority_failure(self):
        """Consensus fails when <=50% of agents succeed."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()

            agent_a = _make_mock_base_agent("AgentA")
            agent_a.execute.return_value = _make_successful_agent_result("ok", "AgentA")
            agent_b = _make_mock_base_agent("AgentB")
            agent_b.execute.return_value = _make_failed_agent_result("fail", "AgentB")
            agent_c = _make_mock_base_agent("AgentC")
            agent_c.execute.return_value = _make_failed_agent_result("fail", "AgentC")

            composite.add_agent("a", agent_a).add_agent("b", agent_b).add_agent("c", agent_c)

            result = await composite._execute_consensus(task="vote")

            # 1 out of 3 succeed → no majority
            assert result.success is False
            assert result.metadata["votes_success"] == 1
            assert result.metadata["votes_total"] == 3

    @pytest.mark.asyncio
    async def test_consensus_returns_first_successful_output(self):
        """Consensus uses the first successful output as the consensus output."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()

            agent_a = _make_mock_base_agent("AgentA")
            agent_a.execute.return_value = _make_successful_agent_result(
                {"answer": "42"}, "AgentA"
            )
            agent_b = _make_mock_base_agent("AgentB")
            agent_b.execute.return_value = _make_successful_agent_result(
                {"answer": "43"}, "AgentB"
            )

            composite.add_agent("a", agent_a).add_agent("b", agent_b)

            result = await composite._execute_consensus(task="vote")

            assert result.output is not None
            assert result.success is True

    @pytest.mark.asyncio
    async def test_orchestrate_no_sub_agents_returns_failure(self):
        """_orchestrate with no sub-agents returns failure UnifiedResult."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()

            result = await composite._orchestrate(task="nothing")

            assert result.success is False
            assert "No sub-agents configured" in result.error

    def test_coordination_dispatch_map_entries(self):
        """_COORDINATION_DISPATCH maps PIPELINE, PARALLEL, CONSENSUS to method names."""
        assert CoordinationPattern.PIPELINE in _COORDINATION_DISPATCH
        assert CoordinationPattern.PARALLEL in _COORDINATION_DISPATCH
        assert CoordinationPattern.CONSENSUS in _COORDINATION_DISPATCH
        assert _COORDINATION_DISPATCH[CoordinationPattern.PIPELINE] == '_execute_pipeline'
        assert _COORDINATION_DISPATCH[CoordinationPattern.PARALLEL] == '_execute_parallel'
        assert _COORDINATION_DISPATCH[CoordinationPattern.CONSENSUS] == '_execute_consensus'

    @pytest.mark.asyncio
    async def test_orchestrate_routes_to_pipeline(self):
        """_orchestrate routes PIPELINE pattern to _execute_pipeline."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            composite.config.coordination_pattern = CoordinationPattern.PIPELINE

            agent_a = _make_mock_base_agent("A")
            agent_a.execute.return_value = _make_successful_agent_result("ok", "A")
            composite.add_agent("a", agent_a)

            with patch.object(composite, '_execute_pipeline', new_callable=AsyncMock) as mock_pipe:
                mock_pipe.return_value = UnifiedResult(
                    success=True, output="done", name="Test", execution_time=1.0
                )
                result = await composite._orchestrate(task="go")
                mock_pipe.assert_called_once()


# =============================================================================
# TestMergeStrategies
# =============================================================================

@pytest.mark.unit
class TestMergeStrategies:
    """Tests for _merge_outputs with different MergeStrategy values."""

    def _make_composite_with_strategy(self, strategy):
        """Helper to create a CompositeAgent with a given merge strategy."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            composite.config.merge_strategy = strategy
            return composite

    def test_combine_returns_dict_of_outputs(self):
        """COMBINE strategy returns the outputs dict as-is."""
        composite = self._make_composite_with_strategy(MergeStrategy.COMBINE)
        outputs = {"a": {"data": 1}, "b": {"data": 2}}
        result = composite._merge_outputs(outputs)
        assert result == outputs

    def test_first_returns_first_value(self):
        """FIRST strategy returns the first value from outputs dict."""
        composite = self._make_composite_with_strategy(MergeStrategy.FIRST)
        outputs = {"a": "first_value", "b": "second_value"}
        result = composite._merge_outputs(outputs)
        assert result == "first_value"

    def test_concat_joins_strings_with_newlines(self):
        """CONCAT strategy joins string values with newline separators."""
        composite = self._make_composite_with_strategy(MergeStrategy.CONCAT)
        outputs = {"a": "line1", "b": "line2", "c": "line3"}
        result = composite._merge_outputs(outputs)
        assert result == "line1\nline2\nline3"

    def test_concat_converts_non_strings(self):
        """CONCAT strategy converts non-string values via str()."""
        composite = self._make_composite_with_strategy(MergeStrategy.CONCAT)
        outputs = {"a": 42, "b": {"key": "val"}}
        result = composite._merge_outputs(outputs)
        assert "42" in result
        assert "key" in result

    def test_best_returns_largest_value(self):
        """BEST strategy returns the value with the largest string length."""
        composite = self._make_composite_with_strategy(MergeStrategy.BEST)
        outputs = {"a": "short", "b": "this is a longer string", "c": "medium text"}
        result = composite._merge_outputs(outputs)
        assert result == "this is a longer string"

    def test_best_with_dict_values(self):
        """BEST strategy compares dict values by their str() length."""
        composite = self._make_composite_with_strategy(MergeStrategy.BEST)
        outputs = {"a": {"x": 1}, "b": {"x": 1, "y": 2, "z": 3}}
        result = composite._merge_outputs(outputs)
        # {"x": 1, "y": 2, "z": 3} has longer str repr
        assert result == {"x": 1, "y": 2, "z": 3}

    def test_empty_outputs_returns_none(self):
        """Any strategy with empty outputs returns None."""
        composite = self._make_composite_with_strategy(MergeStrategy.COMBINE)
        assert composite._merge_outputs({}) is None

    def test_unknown_strategy_falls_back_to_dict(self):
        """Unrecognized strategy falls through and returns the outputs dict."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            # Set to a mock strategy value that is not in the switch
            composite.config.merge_strategy = MagicMock()
            outputs = {"a": "val"}
            result = composite._merge_outputs(outputs)
            assert result == outputs


# =============================================================================
# TestCompositeAgentFactoryMethods
# =============================================================================

@pytest.mark.unit
class TestCompositeAgentFactoryMethods:
    """Extended tests for from_swarm and compose factory methods."""

    def test_from_swarm_uses_swarm_timeout(self):
        """from_swarm uses the swarm's timeout_seconds (300s default)."""
        mock_swarm = _make_mock_domain_swarm("MySwarm", timeout_seconds=300)

        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent.from_swarm(mock_swarm)

            assert composite.config.timeout == 300.0

    def test_from_swarm_custom_timeout(self):
        """from_swarm uses a custom swarm timeout when set."""
        mock_swarm = _make_mock_domain_swarm("MySwarm", timeout_seconds=600)

        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent.from_swarm(mock_swarm)

            assert composite.config.timeout == 600.0

    def test_from_swarm_preserves_wrapped_swarm_reference(self):
        """from_swarm stores the original swarm in _wrapped_swarm."""
        mock_swarm = _make_mock_domain_swarm("TestSwarm")

        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent.from_swarm(mock_swarm)

            assert composite._wrapped_swarm is mock_swarm

    def test_from_swarm_with_signature(self):
        """from_swarm accepts an optional signature parameter."""
        mock_swarm = _make_mock_domain_swarm("TestSwarm")
        mock_signature = MagicMock()

        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent.from_swarm(mock_swarm, signature=mock_signature)

            assert composite.signature is mock_signature

    def test_from_swarm_fallback_timeout_when_no_attr(self):
        """from_swarm falls back to 300 if swarm.config lacks timeout_seconds."""
        mock_swarm = MagicMock()
        mock_swarm.__class__.__name__ = "NoTimeoutSwarm"
        mock_swarm.config = MagicMock(spec=[])  # empty spec = no attributes
        mock_swarm.config.name = "NoTimeoutSwarm"

        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent.from_swarm(mock_swarm)

            # getattr(swarm.config, 'timeout_seconds', 300) → 300
            assert composite.config.timeout == 300.0

    def test_compose_parallel_timeout_uses_max(self):
        """compose with PARALLEL uses max of sub-agent timeouts."""
        agent_a = _make_mock_base_agent("A", timeout=60.0)
        agent_b = _make_mock_base_agent("B", timeout=120.0)

        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent.compose(
                "ParGroup",
                coordination=CoordinationPattern.PARALLEL,
                a=agent_a,
                b=agent_b,
            )

            assert composite.config.timeout == 120.0

    def test_compose_consensus_timeout_uses_max(self):
        """compose with CONSENSUS uses max of sub-agent timeouts."""
        agent_a = _make_mock_base_agent("A", timeout=50.0)
        agent_b = _make_mock_base_agent("B", timeout=200.0)
        agent_c = _make_mock_base_agent("C", timeout=100.0)

        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent.compose(
                "ConsGroup",
                coordination=CoordinationPattern.CONSENSUS,
                a=agent_a,
                b=agent_b,
                c=agent_c,
            )

            assert composite.config.timeout == 200.0

    def test_compose_forces_max_retries_to_one(self):
        """compose always sets max_retries=1."""
        agent_a = _make_mock_base_agent("A", timeout=30.0)

        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent.compose(
                "SingleAgent",
                coordination=CoordinationPattern.PIPELINE,
                a=agent_a,
            )

            assert composite.config.max_retries == 1

    def test_compose_no_agents_timeout_not_set(self):
        """compose with no agents does not crash on empty timeout list."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent.compose("Empty")
            # No agents means agent_timeouts is empty, so timeout is not overridden
            assert len(composite.sub_agents) == 0

    def test_compose_with_signature(self):
        """compose passes through the signature parameter."""
        agent_a = _make_mock_base_agent("A", timeout=30.0)
        mock_sig = MagicMock()

        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent.compose(
                "WithSig",
                signature=mock_sig,
                a=agent_a,
            )

            assert composite.signature is mock_sig


# =============================================================================
# TestCompositeAgentSubAgentManagement
# =============================================================================

@pytest.mark.unit
class TestCompositeAgentSubAgentManagement:
    """Extended tests for add_agent, remove_agent, get_agent, and sub_agents property."""

    def test_add_agent_returns_self_for_chaining(self):
        """add_agent returns self to allow method chaining."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            agent = _make_mock_base_agent("X")
            result = composite.add_agent("x", agent)
            assert result is composite

    def test_remove_agent_returns_self_for_chaining(self):
        """remove_agent returns self to allow method chaining."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            agent = _make_mock_base_agent("X")
            composite.add_agent("x", agent)
            result = composite.remove_agent("x")
            assert result is composite

    def test_get_agent_returns_none_for_missing(self):
        """get_agent returns None when name is not found."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            assert composite.get_agent("nonexistent") is None

    def test_get_agent_returns_correct_agent(self):
        """get_agent returns the exact agent instance that was added."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            agent_x = _make_mock_base_agent("X")
            agent_y = _make_mock_base_agent("Y")
            composite.add_agent("x", agent_x).add_agent("y", agent_y)

            assert composite.get_agent("x") is agent_x
            assert composite.get_agent("y") is agent_y

    def test_sub_agents_returns_read_only_snapshot(self):
        """sub_agents property returns a snapshot that does not affect internals."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            agent = _make_mock_base_agent("W")
            composite.add_agent("w", agent)

            snapshot = composite.sub_agents
            snapshot["hacker"] = _make_mock_base_agent("Hacker")

            assert "hacker" not in composite._sub_agents
            assert len(composite._sub_agents) == 1

    def test_add_agent_overwrites_existing(self):
        """add_agent overwrites an agent if the same name is used."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            agent_v1 = _make_mock_base_agent("V1")
            agent_v2 = _make_mock_base_agent("V2")

            composite.add_agent("slot", agent_v1)
            assert composite.get_agent("slot") is agent_v1

            composite.add_agent("slot", agent_v2)
            assert composite.get_agent("slot") is agent_v2

    def test_remove_then_readd_agent(self):
        """An agent can be removed and re-added under the same name."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            agent = _make_mock_base_agent("Recyclable")

            composite.add_agent("r", agent)
            composite.remove_agent("r")
            assert composite.get_agent("r") is None

            composite.add_agent("r", agent)
            assert composite.get_agent("r") is agent


# =============================================================================
# TestCompositeAgentUtility
# =============================================================================

@pytest.mark.unit
class TestCompositeAgentUtility:
    """Tests for __repr__, to_dict, get_io_schema, and extract_output."""

    def test_repr_with_wrapped_swarm(self):
        """__repr__ mentions the wrapped swarm class name."""
        mock_swarm = _make_mock_domain_swarm("CodingSwarm")

        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent.from_swarm(mock_swarm)
            r = repr(composite)
            assert "wraps=" in r
            assert "CodingSwarm" in r

    def test_repr_without_wrapped_swarm(self):
        """__repr__ shows agents list and coordination pattern."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            composite.config.coordination_pattern = CoordinationPattern.PARALLEL
            agent = _make_mock_base_agent("Worker")
            composite.add_agent("worker", agent)
            r = repr(composite)
            assert "worker" in r
            assert "parallel" in r

    def test_to_dict_includes_composite_fields(self):
        """to_dict includes type, coordination, merge_strategy, has_signature."""
        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent()
            composite.config.coordination_pattern = CoordinationPattern.PIPELINE
            composite.config.merge_strategy = MergeStrategy.COMBINE

            d = composite.to_dict()
            assert d["type"] == "composite"
            assert d["coordination"] == "pipeline"
            assert d["merge_strategy"] == "combine"
            assert d["has_signature"] is False
            assert d["wrapped_swarm"] is None

    def test_to_dict_with_wrapped_swarm(self):
        """to_dict includes wrapped swarm class name."""
        mock_swarm = _make_mock_domain_swarm("ResearchSwarm")

        with patch.object(CompositeAgentConfig, '__post_init__', lambda self: None):
            composite = CompositeAgent.from_swarm(mock_swarm)
            d = composite.to_dict()
            assert d["wrapped_swarm"] is not None

    def test_extract_output_unwraps_unified_result(self):
        """_extract_output unwraps nested UnifiedResult from AgentResult."""
        inner = UnifiedResult(
            success=True,
            output={"deep": "value"},
            name="Inner",
            execution_time=1.0,
            metadata={"inner_key": "inner_val"},
        )
        agent_result = AgentResult(
            success=True,
            output=inner,
            agent_name="Outer",
            execution_time=2.0,
        )

        output, metadata = CompositeAgent._extract_output(agent_result)
        assert output == {"deep": "value"}
        assert metadata == {"inner_key": "inner_val"}

    def test_extract_output_plain_agent_result(self):
        """_extract_output returns output and metadata directly for plain AgentResult."""
        agent_result = AgentResult(
            success=True,
            output="simple",
            agent_name="Simple",
            execution_time=1.0,
            metadata={"key": "val"},
        )

        output, metadata = CompositeAgent._extract_output(agent_result)
        assert output == "simple"
        assert metadata == {"key": "val"}

    def test_unified_result_to_agent_result_conversion(self):
        """UnifiedResult.to_agent_result produces correct AgentResult."""
        unified = UnifiedResult(
            success=True,
            output={"data": "here"},
            name="TestAgent",
            execution_time=2.0,
            error=None,
            metadata={"domain": "test"},
        )

        agent_result = unified.to_agent_result()

        assert isinstance(agent_result, AgentResult)
        assert agent_result.success is True
        assert agent_result.output == {"data": "here"}
        assert agent_result.agent_name == "TestAgent"
        assert agent_result.execution_time == 2.0

    def test_unified_result_from_agent_result_conversion(self):
        """UnifiedResult.from_agent_result creates correct UnifiedResult."""
        agent_result = AgentResult(
            success=False,
            output=None,
            agent_name="FailAgent",
            execution_time=0.5,
            error="something went wrong",
            metadata={"retry": 2},
        )

        unified = UnifiedResult.from_agent_result(agent_result)

        assert unified.success is False
        assert unified.output is None
        assert unified.name == "FailAgent"
        assert unified.execution_time == 0.5
        assert unified.error == "something went wrong"
        assert unified.metadata == {"retry": 2}

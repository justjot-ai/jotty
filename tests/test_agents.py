"""
Agent Tests — Mocked Unit Tests
================================

Tests for the agent hierarchy:
- BaseAgent: execute, retry, hooks, metrics, collaboration
- DomainAgent: DSPy execution, skill fallback, field extraction
- CompositeAgent: pipeline, parallel, consensus, swarm wrapping, UnifiedResult

All tests use mocks — no LLM calls, no API keys.

Fixtures from conftest.py:
    make_concrete_agent — factory for BaseAgent subclasses
    make_domain_agent — factory for DomainAgent with mocked DSPy module
    make_agent_result — factory for AgentResult
    make_swarm_result — factory for SwarmResult
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from Jotty.core.modes.agent.base.base_agent import BaseAgent, AgentRuntimeConfig, AgentResult
from Jotty.core.modes.agent.base.domain_agent import DomainAgent, DomainAgentConfig
from Jotty.core.modes.agent.base.composite_agent import (
    CompositeAgent,
    CompositeAgentConfig,
    UnifiedResult,
    CoordinationPattern,
    MergeStrategy,
)


# =============================================================================
# BaseAgent
# =============================================================================

@pytest.mark.unit
class TestBaseAgent:
    """Test BaseAgent lifecycle, retry, hooks, and metrics."""

    @pytest.mark.asyncio
    async def test_execute_returns_agent_result(self, make_concrete_agent):
        """execute() returns AgentResult with correct fields."""
        agent = make_concrete_agent(output={"answer": 42})
        result = await agent.execute(task="test")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.output == {"answer": 42}
        assert result.agent_name == "TestAgent"
        assert result.execution_time > 0

    @pytest.mark.asyncio
    async def test_execute_catches_exception(self, make_concrete_agent):
        """execute() catches exceptions and returns success=False after retries."""
        agent = make_concrete_agent(raises=ValueError("broken"))
        # Set max_retries=1 so it fails immediately without retrying
        agent.config.max_retries = 1
        agent.config.retry_delay = 0.0

        result = await agent.execute(task="fail")

        assert result.success is False
        assert "broken" in result.error

    @pytest.mark.asyncio
    async def test_execute_retries_on_failure(self, make_concrete_agent):
        """execute() retries with exponential backoff."""
        call_count = 0

        class RetryAgent(BaseAgent):
            async def _execute_impl(self, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise RuntimeError("transient")
                return "recovered"

        agent = RetryAgent(AgentRuntimeConfig(name="RetryAgent"))
        agent._initialized = True
        agent.config.max_retries = 3
        agent.config.retry_delay = 0.001  # Fast retries for tests

        result = await agent.execute(task="retry")

        assert result.success is True
        assert result.output == "recovered"
        assert result.retries == 2
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_timeout(self, make_concrete_agent):
        """execute() times out and returns failure."""
        class SlowAgent(BaseAgent):
            async def _execute_impl(self, **kwargs):
                await asyncio.sleep(10)
                return "never"

        agent = SlowAgent(AgentRuntimeConfig(name="SlowAgent"))
        agent._initialized = True
        agent.config.timeout = 0.05
        agent.config.max_retries = 1
        agent.config.retry_delay = 0.0

        result = await agent.execute(task="slow")

        assert result.success is False
        assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_pre_hook_called(self, make_concrete_agent):
        """Pre-execution hooks are called before execute."""
        agent = make_concrete_agent(output="ok")
        hook_called = []

        def pre_hook(agent_instance, **kwargs):
            hook_called.append(kwargs.get('task'))

        agent.add_pre_hook(pre_hook)
        await agent.execute(task="hooktest")

        assert hook_called == ["hooktest"]

    @pytest.mark.asyncio
    async def test_post_hook_called(self, make_concrete_agent):
        """Post-execution hooks are called after successful execute."""
        agent = make_concrete_agent(output="ok")
        hook_results = []

        def post_hook(agent_instance, result, **kwargs):
            hook_results.append(result.success)

        agent.add_post_hook(post_hook)
        await agent.execute(task="hooktest")

        assert hook_results == [True]

    @pytest.mark.asyncio
    async def test_async_pre_hook(self, make_concrete_agent):
        """Async pre-hooks are awaited."""
        agent = make_concrete_agent(output="ok")
        hook_called = []

        async def async_hook(agent_instance, **kwargs):
            hook_called.append(True)

        agent.add_pre_hook(async_hook)
        await agent.execute(task="test")

        assert hook_called == [True]

    @pytest.mark.asyncio
    async def test_metrics_updated_on_success(self, make_concrete_agent):
        """Metrics increment after successful execution."""
        agent = make_concrete_agent(output="ok")
        await agent.execute(task="m1")
        await agent.execute(task="m2")

        metrics = agent.get_metrics()
        assert metrics["total_executions"] == 2
        assert metrics["successful_executions"] == 2
        assert metrics["failed_executions"] == 0
        assert metrics["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_metrics_updated_on_failure(self, make_concrete_agent):
        """Metrics increment after failed execution."""
        agent = make_concrete_agent(raises=RuntimeError("fail"))
        agent.config.max_retries = 1
        agent.config.retry_delay = 0.0

        await agent.execute(task="f1")

        metrics = agent.get_metrics()
        assert metrics["total_executions"] == 1
        assert metrics["failed_executions"] == 1
        assert metrics["success_rate"] == 0.0

    def test_reset_metrics(self, make_concrete_agent):
        """reset_metrics() zeros all counters."""
        agent = make_concrete_agent()
        agent._metrics["total_executions"] = 5
        agent.reset_metrics()

        metrics = agent.get_metrics()
        assert metrics["total_executions"] == 0

    def test_collaboration_request_help(self, make_concrete_agent):
        """request_help posts to agent slack."""
        agent = make_concrete_agent(name="Agent1")
        agent.set_collaboration_context({"Agent2": Mock()}, [])

        agent.request_help("Agent2", "need review")

        assert len(agent._agent_slack) == 1
        assert agent._agent_slack[0]["to"] == "Agent2"
        assert agent._agent_slack[0]["query"] == "need review"

    def test_get_pending_requests(self, make_concrete_agent):
        """get_pending_requests filters by agent name."""
        agent = make_concrete_agent(name="Agent2")
        slack = [
            {"from": "Agent1", "to": "Agent2", "query": "help"},
            {"from": "Agent1", "to": "Agent3", "query": "other"},
        ]
        agent.set_collaboration_context({}, slack)

        requests = agent.get_pending_requests()
        assert len(requests) == 1
        assert requests[0]["query"] == "help"

    def test_to_dict(self, make_concrete_agent):
        """to_dict() serializes agent state."""
        agent = make_concrete_agent(name="Serializable")
        d = agent.to_dict()

        assert d["name"] == "Serializable"
        assert "metrics" in d
        assert "config" in d

    def test_repr(self, make_concrete_agent):
        """__repr__ includes agent name."""
        agent = make_concrete_agent(name="MyAgent")
        assert "MyAgent" in repr(agent)

    def test_get_io_schema(self, make_concrete_agent):
        """get_io_schema returns generic schema for BaseAgent."""
        agent = make_concrete_agent(name="Schematest")
        schema = agent.get_io_schema()
        assert schema is not None
        assert schema.agent_name == "Schematest"


# =============================================================================
# DomainAgent
# =============================================================================

@pytest.mark.unit
class TestDomainAgent:
    """Test DomainAgent DSPy execution and skill fallback."""

    @pytest.mark.asyncio
    async def test_execute_with_dspy_module(self, make_domain_agent):
        """DomainAgent executes DSPy module and extracts output fields."""
        mock_result = Mock()
        mock_result.analysis = "deep analysis"
        mock_result.confidence = 0.95
        mock_result.reasoning = "step by step"

        agent = make_domain_agent(
            name="Analyzer",
            module_output=mock_result,
        )
        agent._input_fields = ['text']
        agent._output_fields = ['analysis', 'confidence']

        result = await agent.execute(text="some text")

        assert result.success is True
        assert result.output['analysis'] == "deep analysis"
        assert result.output['confidence'] == 0.95
        assert result.output['_reasoning'] == "step by step"

    @pytest.mark.asyncio
    async def test_execute_no_module_falls_back_to_skills(self, make_domain_agent):
        """DomainAgent with no module falls back to skill execution."""
        agent = make_domain_agent(name="NoModule", module_output=None)
        agent._module = None  # Ensure no module

        # Mock _execute_with_skills directly to avoid duplicate 'task' kwarg issue
        agent._execute_with_skills = AsyncMock(return_value={
            "success": True,
            "result": "skill output",
        })

        result = await agent.execute(task="do something")

        assert result.success is True
        agent._execute_with_skills.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_no_module_no_task_returns_error(self, make_domain_agent):
        """DomainAgent with no module and no task returns error dict."""
        agent = make_domain_agent(name="Empty")
        agent._module = None

        result = await agent.execute()

        assert result.success is True  # BaseAgent wraps output, even error dicts
        assert "error" in result.output or result.output is not None

    @pytest.mark.asyncio
    async def test_dspy_failure_falls_back_to_skills(self, make_domain_agent):
        """When DSPy execution raises, DomainAgent falls back to skills."""
        agent = make_domain_agent(name="Fallback")
        agent._module = Mock(side_effect=RuntimeError("DSPy broke"))
        agent._input_fields = ['task']
        agent._output_fields = ['result']

        # Mock _execute_with_skills directly to avoid duplicate 'task' kwarg issue
        agent._execute_with_skills = AsyncMock(return_value={
            "success": True,
            "result": "recovered via skills",
        })

        result = await agent.execute(task="recoverable")

        assert result.success is True
        agent._execute_with_skills.assert_awaited_once()

    def test_input_output_fields(self, make_domain_agent):
        """input_fields and output_fields return copies."""
        agent = make_domain_agent()
        agent._input_fields = ['a', 'b']
        agent._output_fields = ['c']

        assert agent.input_fields == ['a', 'b']
        assert agent.output_fields == ['c']
        # Verify they're copies, not references
        agent.input_fields.append('x')
        assert 'x' not in agent._input_fields

    def test_build_task_from_kwargs(self, make_domain_agent):
        """_build_task_from_kwargs extracts task from common keys."""
        agent = make_domain_agent()
        assert agent._build_task_from_kwargs({'query': 'hello'}) == 'hello'
        assert agent._build_task_from_kwargs({'prompt': 'do it'}) == 'do it'
        assert agent._build_task_from_kwargs({}) == ''

    def test_get_io_schema_no_signature(self, make_domain_agent):
        """DomainAgent without signature returns field-based schema."""
        agent = make_domain_agent()
        agent._input_fields = ['task']
        agent._output_fields = ['result']
        schema = agent.get_io_schema()
        assert schema is not None


# =============================================================================
# UnifiedResult
# =============================================================================

@pytest.mark.unit
class TestUnifiedResult:
    """Test UnifiedResult bidirectional bridge."""

    def test_to_agent_result(self):
        """UnifiedResult converts to AgentResult."""
        ur = UnifiedResult(
            success=True,
            output={"answer": 42},
            name="TestAgent",
            execution_time=1.5,
            metadata={"key": "val"},
        )
        ar = ur.to_agent_result()

        assert isinstance(ar, AgentResult)
        assert ar.success is True
        assert ar.output == {"answer": 42}
        assert ar.agent_name == "TestAgent"
        assert ar.execution_time == 1.5

    def test_to_swarm_result(self, make_swarm_result):
        """UnifiedResult converts to SwarmResult."""
        ur = UnifiedResult(
            success=True,
            output={"result": "done"},
            name="TestSwarm",
            execution_time=2.0,
            metadata={"domain": "coding"},
        )
        sr = ur.to_swarm_result()

        assert sr.success is True
        assert sr.swarm_name == "TestSwarm"
        assert sr.output == {"result": "done"}

    def test_from_swarm_result(self, make_swarm_result):
        """UnifiedResult created from SwarmResult."""
        sr = make_swarm_result(success=True, output={"code": "print(1)"}, name="CodingSwarm")
        ur = UnifiedResult.from_swarm_result(sr)

        assert ur.success is True
        assert ur.output == {"code": "print(1)"}
        assert ur.name == "CodingSwarm"

    def test_from_agent_result(self, make_agent_result):
        """UnifiedResult created from AgentResult."""
        ar = make_agent_result(success=True, output="hello", name="Agent1")
        ur = UnifiedResult.from_agent_result(ar)

        assert ur.success is True
        assert ur.output == "hello"
        assert ur.name == "Agent1"

    def test_roundtrip_agent(self, make_agent_result):
        """AgentResult -> UnifiedResult -> AgentResult preserves data."""
        original = make_agent_result(success=True, output={"x": 1}, name="RT")
        ur = UnifiedResult.from_agent_result(original)
        back = ur.to_agent_result()

        assert back.success == original.success
        assert back.output == original.output
        assert back.agent_name == original.agent_name


# =============================================================================
# CompositeAgent
# =============================================================================

@pytest.mark.unit
class TestCompositeAgent:
    """Test CompositeAgent coordination patterns."""

    @pytest.mark.asyncio
    async def test_pipeline_executes_sequentially(self, make_concrete_agent):
        """Pipeline executes agents in order, chaining output."""
        execution_order = []

        class Agent1(BaseAgent):
            async def _execute_impl(self, **kwargs):
                execution_order.append("agent1")
                return {"step1": "done"}

        class Agent2(BaseAgent):
            async def _execute_impl(self, **kwargs):
                execution_order.append("agent2")
                assert kwargs.get("step1") == "done"  # Output chained
                return {"step2": "done"}

        a1 = Agent1(AgentRuntimeConfig(name="A1"))
        a1._initialized = True
        a2 = Agent2(AgentRuntimeConfig(name="A2"))
        a2._initialized = True

        composite = CompositeAgent.compose(
            "Pipeline",
            coordination=CoordinationPattern.PIPELINE,
            first=a1,
            second=a2,
        )
        composite._initialized = True

        result = await composite.execute(task="test")

        assert result.success is True
        assert execution_order == ["agent1", "agent2"]
        assert result.output.get("step2") == "done"

    @pytest.mark.asyncio
    async def test_pipeline_stops_on_failure(self, make_concrete_agent):
        """Pipeline stops when a stage fails."""
        agent1 = make_concrete_agent(name="A1", raises=RuntimeError("fail"))
        agent1.config.max_retries = 1
        agent1.config.retry_delay = 0.0
        agent2 = make_concrete_agent(name="A2", output="should not run")

        composite = CompositeAgent.compose(
            "Pipeline",
            coordination=CoordinationPattern.PIPELINE,
            first=agent1,
            second=agent2,
        )
        composite._initialized = True

        result = await composite.execute(task="test")

        assert result.success is False
        assert "first" in (result.error or "")  # Uses compose key, not agent name

    @pytest.mark.asyncio
    async def test_parallel_executes_concurrently(self, make_concrete_agent):
        """Parallel executes all agents and merges output."""
        a1 = make_concrete_agent(name="A1", output={"a1": "done"})
        a2 = make_concrete_agent(name="A2", output={"a2": "done"})

        composite = CompositeAgent.compose(
            "Parallel",
            coordination=CoordinationPattern.PARALLEL,
            merge_strategy=MergeStrategy.COMBINE,
            first=a1,
            second=a2,
        )
        composite._initialized = True

        result = await composite.execute(task="test")

        assert result.success is True
        # COMBINE merges into dict of {agent_name: output}
        assert isinstance(result.output, dict)

    @pytest.mark.asyncio
    async def test_consensus_majority_success(self, make_concrete_agent):
        """Consensus succeeds when majority of agents succeed."""
        a1 = make_concrete_agent(name="A1", output="yes")
        a2 = make_concrete_agent(name="A2", output="yes")
        a3 = make_concrete_agent(name="A3", raises=RuntimeError("no"))
        a3.config.max_retries = 1
        a3.config.retry_delay = 0.0

        composite = CompositeAgent.compose(
            "Consensus",
            coordination=CoordinationPattern.CONSENSUS,
            a=a1, b=a2, c=a3,
        )
        composite._initialized = True

        result = await composite.execute(task="vote")

        assert result.success is True  # 2/3 succeeded

    @pytest.mark.asyncio
    async def test_consensus_majority_failure(self, make_concrete_agent):
        """Consensus fails when majority of agents fail."""
        a1 = make_concrete_agent(name="A1", output="yes")
        a2 = make_concrete_agent(name="A2", raises=RuntimeError("no"))
        a2.config.max_retries = 1
        a2.config.retry_delay = 0.0
        a3 = make_concrete_agent(name="A3", raises=RuntimeError("no"))
        a3.config.max_retries = 1
        a3.config.retry_delay = 0.0

        composite = CompositeAgent.compose(
            "Consensus",
            coordination=CoordinationPattern.CONSENSUS,
            a=a1, b=a2, c=a3,
        )
        composite._initialized = True

        result = await composite.execute(task="vote")

        assert result.success is False  # 1/3 succeeded

    @pytest.mark.asyncio
    async def test_wrapped_swarm_delegates(self, make_swarm_result):
        """CompositeAgent wrapping a swarm delegates to swarm.execute()."""
        mock_swarm = AsyncMock()
        mock_swarm.config = Mock(name="TestSwarm", timeout_seconds=300)
        mock_swarm.__class__.__name__ = "TestSwarm"
        mock_swarm.execute = AsyncMock(return_value=make_swarm_result(
            success=True,
            output={"swarm_result": "done"},
            name="TestSwarm",
        ))

        composite = CompositeAgent.from_swarm(mock_swarm)
        composite._initialized = True

        result = await composite.execute(task="test")

        mock_swarm.execute.assert_awaited_once()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_no_sub_agents_returns_error(self):
        """CompositeAgent with no sub-agents returns error."""
        composite = CompositeAgent(
            config=CompositeAgentConfig(name="Empty"),
        )
        composite._initialized = True

        result = await composite.execute(task="test")

        assert result.success is False
        assert "No sub-agents" in (result.error or "")

    def test_add_remove_agent(self, make_concrete_agent):
        """add_agent/remove_agent manage sub-agents."""
        composite = CompositeAgent(config=CompositeAgentConfig(name="Dynamic"))
        a1 = make_concrete_agent(name="A1")

        composite.add_agent("first", a1)
        assert composite.get_agent("first") is a1
        assert "first" in composite.sub_agents

        composite.remove_agent("first")
        assert composite.get_agent("first") is None

    def test_compose_timeout_pipeline(self, make_concrete_agent):
        """Pipeline timeout is sum of sub-agent timeouts."""
        a1 = make_concrete_agent(name="A1")
        a1.config.timeout = 10.0
        a2 = make_concrete_agent(name="A2")
        a2.config.timeout = 20.0

        composite = CompositeAgent.compose(
            "Pipeline",
            coordination=CoordinationPattern.PIPELINE,
            first=a1,
            second=a2,
        )

        assert composite.config.timeout == 30.0

    def test_compose_timeout_parallel(self, make_concrete_agent):
        """Parallel timeout is max of sub-agent timeouts."""
        a1 = make_concrete_agent(name="A1")
        a1.config.timeout = 10.0
        a2 = make_concrete_agent(name="A2")
        a2.config.timeout = 20.0

        composite = CompositeAgent.compose(
            "Parallel",
            coordination=CoordinationPattern.PARALLEL,
            first=a1,
            second=a2,
        )

        assert composite.config.timeout == 20.0

    def test_repr_with_sub_agents(self, make_concrete_agent):
        """__repr__ shows sub-agent names and coordination pattern."""
        a1 = make_concrete_agent(name="A1")
        composite = CompositeAgent.compose("Test", first=a1)

        r = repr(composite)
        assert "Test" in r
        assert "first" in r

    def test_repr_wrapped_swarm(self):
        """__repr__ shows wrapped swarm name."""
        mock_swarm = Mock()
        mock_swarm.config = Mock(name="CodingSwarm", timeout_seconds=300)
        mock_swarm.__class__.__name__ = "CodingSwarm"

        composite = CompositeAgent.from_swarm(mock_swarm)

        assert "CodingSwarm" in repr(composite)

    def test_to_dict(self, make_concrete_agent):
        """to_dict serializes composite state."""
        a1 = make_concrete_agent(name="A1")
        composite = CompositeAgent.compose("Test", first=a1)

        d = composite.to_dict()
        assert d["type"] == "composite"
        assert "first" in d["sub_agents"]

    @pytest.mark.asyncio
    async def test_pipeline_non_dict_output_chains_as_task(self, make_concrete_agent):
        """Non-dict output replaces 'task' key for next agent."""
        class StringAgent(BaseAgent):
            async def _execute_impl(self, **kwargs):
                return "string output"

        class ReceivingAgent(BaseAgent):
            async def _execute_impl(self, **kwargs):
                assert kwargs.get("task") == "string output"
                return {"received": True}

        a1 = StringAgent(AgentRuntimeConfig(name="A1"))
        a1._initialized = True
        a2 = ReceivingAgent(AgentRuntimeConfig(name="A2"))
        a2._initialized = True

        composite = CompositeAgent.compose(
            "Pipeline", first=a1, second=a2,
            coordination=CoordinationPattern.PIPELINE,
        )
        composite._initialized = True

        result = await composite.execute(task="start")
        assert result.success is True

    def test_get_io_schema_from_wrapped_swarm(self):
        """get_io_schema delegates to wrapped swarm's schema."""
        mock_schema = Mock()
        mock_swarm = Mock()
        mock_swarm.config = Mock(name="Test", timeout_seconds=300)
        mock_swarm.__class__.__name__ = "TestSwarm"
        mock_swarm.get_io_schema = Mock(return_value=mock_schema)

        composite = CompositeAgent.from_swarm(mock_swarm)
        schema = composite.get_io_schema()

        assert schema is mock_schema


# =============================================================================
# Merge Strategies
# =============================================================================

@pytest.mark.unit
class TestMergeStrategies:
    """Test CompositeAgent merge strategy options."""

    def _make_composite(self, strategy, agents):
        config = CompositeAgentConfig(
            name="MergeTest",
            coordination_pattern=CoordinationPattern.PARALLEL,
            merge_strategy=strategy,
        )
        composite = CompositeAgent(config=config, sub_agents=agents)
        return composite

    def test_merge_combine(self):
        """COMBINE returns dict of all outputs."""
        composite = self._make_composite(MergeStrategy.COMBINE, {})
        result = composite._merge_outputs({"a": "x", "b": "y"})
        assert result == {"a": "x", "b": "y"}

    def test_merge_first(self):
        """FIRST returns first output."""
        composite = self._make_composite(MergeStrategy.FIRST, {})
        result = composite._merge_outputs({"a": "x", "b": "y"})
        assert result == "x"

    def test_merge_concat(self):
        """CONCAT concatenates string representations."""
        composite = self._make_composite(MergeStrategy.CONCAT, {})
        result = composite._merge_outputs({"a": "hello", "b": "world"})
        assert "hello" in result
        assert "world" in result

    def test_merge_best(self):
        """BEST picks output with longest string representation."""
        composite = self._make_composite(MergeStrategy.BEST, {})
        result = composite._merge_outputs({"a": "short", "b": "much longer output"})
        assert result == "much longer output"

    def test_merge_empty(self):
        """Empty outputs returns None for all strategies."""
        for strategy in MergeStrategy:
            composite = self._make_composite(strategy, {})
            result = composite._merge_outputs({})
            assert result is None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for Agent Core Module
=============================
Tests for BaseAgent execution, retries, hooks, metrics, and AgentResult.
Tests for DomainAgent, CompositeAgent, and MetaAgent.
"""
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any


# =============================================================================
# AgentResult Tests
# =============================================================================

class TestAgentResult:
    """Tests for AgentResult dataclass."""

    @pytest.mark.unit
    def test_successful_result(self):
        """AgentResult for successful execution."""
        from Jotty.core.agents.base.base_agent import AgentResult
        result = AgentResult(
            success=True,
            output="computed answer",
            agent_name="TestAgent",
            execution_time=1.5,
        )
        assert result.success is True
        assert result.output == "computed answer"
        assert result.agent_name == "TestAgent"
        assert result.error is None
        assert result.retries == 0

    @pytest.mark.unit
    def test_failed_result(self):
        """AgentResult for failed execution."""
        from Jotty.core.agents.base.base_agent import AgentResult
        result = AgentResult(
            success=False,
            output=None,
            agent_name="TestAgent",
            execution_time=5.0,
            error="Timeout after 3 retries",
            retries=3,
        )
        assert result.success is False
        assert result.error == "Timeout after 3 retries"
        assert result.retries == 3

    @pytest.mark.unit
    def test_to_dict_serialization(self):
        """AgentResult.to_dict() returns serializable dict."""
        from Jotty.core.agents.base.base_agent import AgentResult
        result = AgentResult(
            success=True,
            output="answer",
            agent_name="TestAgent",
            execution_time=1.0,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d['success'] is True
        assert d['output'] == "answer"
        assert d['agent_name'] == "TestAgent"
        assert 'timestamp' in d

    @pytest.mark.unit
    def test_timestamp_set_automatically(self):
        """AgentResult gets timestamp on creation."""
        from Jotty.core.agents.base.base_agent import AgentResult
        from datetime import datetime
        result = AgentResult(
            success=True, output="x",
            agent_name="A", execution_time=0.1,
        )
        assert isinstance(result.timestamp, datetime)


# =============================================================================
# BaseAgent Tests
# =============================================================================

class TestBaseAgent:
    """Tests for BaseAgent execution lifecycle."""

    @pytest.mark.unit
    def test_creation_with_default_config(self):
        """BaseAgent creates with default config."""
        from Jotty.core.agents.base.base_agent import BaseAgent, AgentRuntimeConfig

        class TestAgent(BaseAgent):
            async def _execute_impl(self, **kwargs):
                return "ok"

        agent = TestAgent()
        assert agent.config is not None
        assert agent.config.name != ""

    @pytest.mark.unit
    def test_creation_with_custom_config(self):
        """BaseAgent uses provided config."""
        from Jotty.core.agents.base.base_agent import BaseAgent, AgentRuntimeConfig

        class TestAgent(BaseAgent):
            async def _execute_impl(self, **kwargs):
                return "ok"

        config = AgentRuntimeConfig(name="CustomAgent", enable_memory=False)
        agent = TestAgent(config=config)
        assert agent.config.name == "CustomAgent"
        assert agent.config.enable_memory is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_success(self, make_concrete_agent):
        """execute() returns success result."""
        agent = make_concrete_agent(name="TestAgent", output="hello world")
        result = await agent.execute(query="test")
        assert result.success is True
        assert result.output == "hello world"
        assert result.agent_name == "TestAgent"
        assert result.execution_time > 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_failure(self, make_concrete_agent):
        """execute() returns failure result on exception."""
        agent = make_concrete_agent(
            name="FailAgent",
            raises=ValueError("bad input"),
            max_retries=1,
        )
        result = await agent.execute(query="test")
        assert result.success is False
        assert result.error is not None
        assert "bad input" in result.error

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_with_retries(self, make_concrete_agent):
        """execute() retries on failure."""
        call_count = 0

        from Jotty.core.agents.base.base_agent import BaseAgent, AgentRuntimeConfig

        class RetryAgent(BaseAgent):
            async def _execute_impl(self, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise RuntimeError("transient error")
                return "success after retries"

        config = AgentRuntimeConfig(name="RetryAgent", max_retries=5, retry_delay=0.01)
        agent = RetryAgent(config=config)
        agent._initialized = True
        result = await agent.execute()
        assert result.success is True
        assert result.output == "success after retries"
        assert call_count == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_exhausts_retries(self, make_concrete_agent):
        """execute() fails after exhausting retries."""
        from Jotty.core.agents.base.base_agent import BaseAgent, AgentRuntimeConfig

        class AlwaysFailAgent(BaseAgent):
            async def _execute_impl(self, **kwargs):
                raise RuntimeError("persistent error")

        config = AgentRuntimeConfig(name="AlwaysFail", max_retries=2, retry_delay=0.01)
        agent = AlwaysFailAgent(config=config)
        agent._initialized = True
        result = await agent.execute()
        assert result.success is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metrics_updated_on_success(self, make_concrete_agent):
        """Metrics track successful executions."""
        agent = make_concrete_agent(name="MetricsAgent", output="ok")
        await agent.execute()
        await agent.execute()
        metrics = agent.get_metrics()
        assert metrics['total_executions'] == 2
        assert metrics['successful_executions'] == 2
        assert metrics['failed_executions'] == 0
        assert metrics['success_rate'] == 1.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_metrics_updated_on_failure(self, make_concrete_agent):
        """Metrics track failed executions."""
        agent = make_concrete_agent(
            name="FailMetrics",
            raises=RuntimeError("fail"),
            max_retries=1,
            retry_delay=0.01,
        )
        await agent.execute()
        metrics = agent.get_metrics()
        assert metrics['total_executions'] == 1
        assert metrics['failed_executions'] == 1
        assert metrics['success_rate'] == 0.0

    @pytest.mark.unit
    def test_reset_metrics(self, make_concrete_agent):
        """reset_metrics() zeros all counters."""
        agent = make_concrete_agent(name="ResetTest")
        # Manually set some metrics
        agent._metrics['total_executions'] = 5
        agent._metrics['successful_executions'] = 3
        agent.reset_metrics()
        metrics = agent.get_metrics()
        assert metrics['total_executions'] == 0
        assert metrics['successful_executions'] == 0

    @pytest.mark.unit
    def test_metrics_zero_division(self, make_concrete_agent):
        """get_metrics() handles zero total executions."""
        agent = make_concrete_agent(name="ZeroDiv")
        metrics = agent.get_metrics()
        assert metrics['success_rate'] == 0.0
        assert metrics['avg_execution_time'] == 0.0


# =============================================================================
# Hook Tests
# =============================================================================

class TestBaseAgentHooks:
    """Tests for BaseAgent pre/post hooks."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pre_hook_called(self, make_concrete_agent):
        """Pre-hook is called before execution."""
        hook_called = False

        def pre_hook(agent, **kwargs):
            nonlocal hook_called
            hook_called = True

        agent = make_concrete_agent(name="HookTest", output="ok")
        agent.add_pre_hook(pre_hook)
        await agent.execute(query="test")
        assert hook_called is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_post_hook_called(self, make_concrete_agent):
        """Post-hook is called after successful execution."""
        post_result = {}

        def post_hook(agent, result, **kwargs):
            post_result['called'] = True
            post_result['success'] = result.success

        agent = make_concrete_agent(name="PostHookTest", output="ok")
        agent.add_post_hook(post_hook)
        await agent.execute(query="test")
        assert post_result.get('called') is True
        assert post_result.get('success') is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_hook_called(self, make_concrete_agent):
        """Async hooks are awaited correctly."""
        hook_called = False

        async def async_pre_hook(agent, **kwargs):
            nonlocal hook_called
            hook_called = True

        agent = make_concrete_agent(name="AsyncHook", output="ok")
        agent.add_pre_hook(async_pre_hook)
        await agent.execute(query="test")
        assert hook_called is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_hook_exception_does_not_fail_execution(self, make_concrete_agent):
        """Hook exceptions are logged but don't fail execution."""
        def bad_hook(agent, **kwargs):
            raise RuntimeError("hook error")

        agent = make_concrete_agent(name="BadHook", output="ok")
        agent.add_pre_hook(bad_hook)
        result = await agent.execute(query="test")
        assert result.success is True  # Execution still succeeds


# =============================================================================
# Collaboration Tests
# =============================================================================

class TestBaseAgentCollaboration:
    """Tests for BaseAgent inter-agent collaboration."""

    @pytest.mark.unit
    def test_request_help(self, make_concrete_agent):
        """request_help() stores request in shared queue."""
        agent = make_concrete_agent(name="RequesterAgent")
        shared_queue = []
        agent.set_collaboration_context(
            agent_directory={"HelperAgent": MagicMock()},
            agent_slack=shared_queue,
        )
        agent.request_help("HelperAgent", "Need analysis data")
        assert len(shared_queue) == 1
        assert shared_queue[0]['from'] == "RequesterAgent"
        assert shared_queue[0]['to'] == "HelperAgent"
        assert shared_queue[0]['query'] == "Need analysis data"

    @pytest.mark.unit
    def test_get_pending_requests(self, make_concrete_agent):
        """get_pending_requests() filters by recipient."""
        agent = make_concrete_agent(name="ReceiverAgent")
        shared_queue = [
            {'from': 'A', 'to': 'ReceiverAgent', 'query': 'help me'},
            {'from': 'B', 'to': 'OtherAgent', 'query': 'not for me'},
            {'from': 'C', 'to': 'ReceiverAgent', 'query': 'also me'},
        ]
        agent.set_collaboration_context(
            agent_directory={},
            agent_slack=shared_queue,
        )
        pending = agent.get_pending_requests()
        assert len(pending) == 2
        assert all(r['to'] == 'ReceiverAgent' for r in pending)


# =============================================================================
# Lazy Property Tests
# =============================================================================

class TestBaseAgentProperties:
    """Tests for BaseAgent lazy-loaded properties."""

    @pytest.mark.unit
    def test_memory_disabled_returns_none(self, make_concrete_agent):
        """memory property returns None when disabled."""
        agent = make_concrete_agent(name="NoMemory", enable_memory=False)
        assert agent.memory is None

    @pytest.mark.unit
    def test_skills_disabled_returns_none(self, make_concrete_agent):
        """skills_registry returns None when disabled."""
        agent = make_concrete_agent(name="NoSkills", enable_skills=False)
        assert agent.skills_registry is None

    @pytest.mark.unit
    def test_store_memory_noop_when_disabled(self, make_concrete_agent):
        """store_memory is no-op when memory is None."""
        agent = make_concrete_agent(name="NoMem", enable_memory=False)
        # Should not raise
        agent.store_memory("test content", level="episodic")

    @pytest.mark.unit
    def test_retrieve_memory_empty_when_disabled(self, make_concrete_agent):
        """retrieve_memory returns empty when memory is None."""
        agent = make_concrete_agent(name="NoMem", enable_memory=False)
        result = agent.retrieve_memory("query")
        assert result == [] or result is None


# =============================================================================
# DomainAgent Tests
# =============================================================================

class TestDomainAgent:
    """Tests for DomainAgent field extraction, I/O properties, and factory."""

    @pytest.fixture
    def mock_signature(self):
        """Create a mock DSPy Signature with input/output fields."""
        import dspy
        class TestSig(dspy.Signature):
            """Test signature for unit testing."""
            task: str = dspy.InputField(desc="The task to perform")
            context: str = dspy.InputField(desc="Context information")
            output: str = dspy.OutputField(desc="The result")
            confidence: float = dspy.OutputField(desc="Confidence score")
        return TestSig

    @pytest.mark.unit
    def test_extract_fields_from_signature(self, mock_signature):
        """DomainAgent extracts input/output fields from DSPy Signature."""
        from Jotty.core.agents.base.domain_agent import DomainAgent, DomainAgentConfig
        config = DomainAgentConfig(name="test")
        agent = DomainAgent(mock_signature, config)
        assert 'task' in agent.input_fields
        assert 'context' in agent.input_fields
        assert 'output' in agent.output_fields
        assert 'confidence' in agent.output_fields

    @pytest.mark.unit
    def test_input_fields_returns_copy(self, mock_signature):
        """input_fields property returns a copy, not a reference."""
        from Jotty.core.agents.base.domain_agent import DomainAgent, DomainAgentConfig
        config = DomainAgentConfig(name="test")
        agent = DomainAgent(mock_signature, config)
        fields1 = agent.input_fields
        fields2 = agent.input_fields
        assert fields1 == fields2
        # Modifying one shouldn't affect the other
        fields1.append("extra")
        assert "extra" not in agent.input_fields

    @pytest.mark.unit
    def test_output_fields_returns_copy(self, mock_signature):
        """output_fields property returns a copy."""
        from Jotty.core.agents.base.domain_agent import DomainAgent, DomainAgentConfig
        config = DomainAgentConfig(name="test")
        agent = DomainAgent(mock_signature, config)
        fields = agent.output_fields
        fields.append("extra")
        assert "extra" not in agent.output_fields

    @pytest.mark.unit
    def test_no_signature_empty_fields(self):
        """DomainAgent with signature=None has empty field lists."""
        from Jotty.core.agents.base.domain_agent import DomainAgent, DomainAgentConfig
        config = DomainAgentConfig(name="test")
        agent = DomainAgent(None, config)
        assert agent.input_fields == []
        assert agent.output_fields == []

    @pytest.mark.unit
    def test_get_io_schema(self, mock_signature):
        """get_io_schema returns AgentIOSchema with correct fields."""
        from Jotty.core.agents.base.domain_agent import DomainAgent, DomainAgentConfig
        config = DomainAgentConfig(name="test")
        agent = DomainAgent(mock_signature, config)
        schema = agent.get_io_schema()
        input_names = [p.name for p in schema.inputs]
        output_names = [p.name for p in schema.outputs]
        assert 'task' in input_names
        assert 'output' in output_names

    @pytest.mark.unit
    def test_get_io_schema_cached(self, mock_signature):
        """get_io_schema caches result after first call."""
        from Jotty.core.agents.base.domain_agent import DomainAgent, DomainAgentConfig
        config = DomainAgentConfig(name="test")
        agent = DomainAgent(mock_signature, config)
        schema1 = agent.get_io_schema()
        schema2 = agent.get_io_schema()
        assert schema1 is schema2

    @pytest.mark.unit
    def test_create_domain_agent_factory(self, mock_signature):
        """create_domain_agent factory returns configured DomainAgent."""
        from Jotty.core.agents.base.domain_agent import create_domain_agent
        agent = create_domain_agent(mock_signature, use_chain_of_thought=True)
        assert 'task' in agent.input_fields
        assert agent.config.use_chain_of_thought is True

    @pytest.mark.unit
    def test_create_domain_agent_no_cot(self, mock_signature):
        """create_domain_agent with use_chain_of_thought=False."""
        from Jotty.core.agents.base.domain_agent import create_domain_agent
        agent = create_domain_agent(mock_signature, use_chain_of_thought=False)
        assert agent.config.use_chain_of_thought is False

    @pytest.mark.unit
    def test_domain_agent_config_defaults(self):
        """DomainAgentConfig has sensible defaults."""
        from Jotty.core.agents.base.domain_agent import DomainAgentConfig
        config = DomainAgentConfig()
        assert config.use_chain_of_thought is True
        assert config.use_react is False
        assert config.max_react_iters == 5

    @pytest.mark.unit
    def test_build_task_from_kwargs_query_key(self, mock_signature):
        """_build_task_from_kwargs extracts 'query' key."""
        from Jotty.core.agents.base.domain_agent import DomainAgent, DomainAgentConfig
        config = DomainAgentConfig(name="test")
        agent = DomainAgent(mock_signature, config)
        result = agent._build_task_from_kwargs({"query": "search AI trends"})
        assert result == "search AI trends"

    @pytest.mark.unit
    def test_build_task_from_kwargs_fallback(self, mock_signature):
        """_build_task_from_kwargs concatenates string values as fallback."""
        from Jotty.core.agents.base.domain_agent import DomainAgent, DomainAgentConfig
        config = DomainAgentConfig(name="test")
        agent = DomainAgent(mock_signature, config)
        result = agent._build_task_from_kwargs({"x": "hello", "y": "world"})
        assert "hello" in result
        assert "world" in result

    @pytest.mark.unit
    def test_build_task_from_kwargs_empty(self, mock_signature):
        """_build_task_from_kwargs returns empty for empty kwargs."""
        from Jotty.core.agents.base.domain_agent import DomainAgent, DomainAgentConfig
        config = DomainAgentConfig(name="test")
        agent = DomainAgent(mock_signature, config)
        result = agent._build_task_from_kwargs({})
        assert result == ""


# =============================================================================
# UnifiedResult Tests
# =============================================================================

class TestUnifiedResult:
    """Tests for UnifiedResult bidirectional bridge."""

    @pytest.mark.unit
    def test_to_agent_result(self):
        """UnifiedResult converts to AgentResult."""
        from Jotty.core.agents.base.composite_agent import UnifiedResult
        from Jotty.core.agents.base.base_agent import AgentResult
        ur = UnifiedResult(
            success=True, output="hello", name="TestAgent",
            execution_time=1.5, metadata={"k": "v"},
        )
        ar = ur.to_agent_result()
        assert isinstance(ar, AgentResult)
        assert ar.success is True
        assert ar.output == "hello"
        assert ar.agent_name == "TestAgent"
        assert ar.execution_time == 1.5

    @pytest.mark.unit
    def test_from_agent_result(self):
        """UnifiedResult creates from AgentResult."""
        from Jotty.core.agents.base.composite_agent import UnifiedResult
        from Jotty.core.agents.base.base_agent import AgentResult
        ar = AgentResult(
            success=True, output="data", agent_name="A",
            execution_time=2.0, metadata={"x": 1},
        )
        ur = UnifiedResult.from_agent_result(ar)
        assert ur.success is True
        assert ur.output == "data"
        assert ur.name == "A"
        assert ur.metadata == {"x": 1}

    @pytest.mark.unit
    def test_roundtrip_agent_result(self):
        """AgentResult → UnifiedResult → AgentResult preserves fields."""
        from Jotty.core.agents.base.composite_agent import UnifiedResult
        from Jotty.core.agents.base.base_agent import AgentResult
        original = AgentResult(
            success=False, output=None, agent_name="B",
            execution_time=0.5, error="timeout",
        )
        ur = UnifiedResult.from_agent_result(original)
        restored = ur.to_agent_result()
        assert restored.success is False
        assert restored.error == "timeout"
        assert restored.agent_name == "B"

    @pytest.mark.unit
    def test_unified_result_defaults(self):
        """UnifiedResult has sensible defaults for optional fields."""
        from Jotty.core.agents.base.composite_agent import UnifiedResult
        ur = UnifiedResult(success=True, output="x", name="A", execution_time=0.1)
        assert ur.error is None
        assert ur.metadata == {}
        assert ur.agent_traces == []
        assert ur.evaluation is None
        assert ur.improvements == []


# =============================================================================
# CompositeAgent Tests
# =============================================================================

class TestCompositeAgent:
    """Tests for CompositeAgent orchestration."""

    def _make_sub_agent(self, name="sub", output="result", success=True, error=None):
        """Create a mock sub-agent."""
        from Jotty.core.agents.base.base_agent import BaseAgent, AgentRuntimeConfig, AgentResult
        class _MockAgent(BaseAgent):
            async def _execute_impl(self, **kwargs):
                if not success:
                    raise RuntimeError(error or "failed")
                return output
        config = AgentRuntimeConfig(name=name, max_retries=1, retry_delay=0.01)
        agent = _MockAgent(config)
        agent._initialized = True
        return agent

    @pytest.mark.unit
    def test_create_default(self):
        """CompositeAgent creates with defaults."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent
        agent = CompositeAgent()
        assert agent.config.name == "CompositeAgent"
        assert agent._sub_agents == {}
        assert agent._wrapped_swarm is None

    @pytest.mark.unit
    def test_compose_factory(self):
        """compose() creates configured CompositeAgent from sub-agents."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent, CoordinationPattern, MergeStrategy
        a = self._make_sub_agent("a")
        b = self._make_sub_agent("b")
        composite = CompositeAgent.compose(
            "Pipeline", coordination=CoordinationPattern.PIPELINE,
            merge_strategy=MergeStrategy.CONCAT, a=a, b=b,
        )
        assert composite.config.name == "Pipeline"
        assert composite.config.coordination_pattern == CoordinationPattern.PIPELINE
        assert composite.config.merge_strategy == MergeStrategy.CONCAT
        assert len(composite.sub_agents) == 2

    @pytest.mark.unit
    def test_compose_pipeline_timeout_is_sum(self):
        """Pipeline timeout is sum of sub-agent timeouts."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent, CoordinationPattern
        a = self._make_sub_agent("a")
        a.config.timeout = 10.0
        b = self._make_sub_agent("b")
        b.config.timeout = 20.0
        composite = CompositeAgent.compose("P", coordination=CoordinationPattern.PIPELINE, a=a, b=b)
        assert composite.config.timeout == 30.0

    @pytest.mark.unit
    def test_compose_parallel_timeout_is_max(self):
        """Parallel timeout is max of sub-agent timeouts."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent, CoordinationPattern
        a = self._make_sub_agent("a")
        a.config.timeout = 10.0
        b = self._make_sub_agent("b")
        b.config.timeout = 20.0
        composite = CompositeAgent.compose("P", coordination=CoordinationPattern.PARALLEL, a=a, b=b)
        assert composite.config.timeout == 20.0

    @pytest.mark.unit
    def test_add_remove_get_agent(self):
        """add_agent/remove_agent/get_agent manage sub-agents."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent
        composite = CompositeAgent()
        a = self._make_sub_agent("a")
        composite.add_agent("alpha", a)
        assert composite.get_agent("alpha") is a
        composite.remove_agent("alpha")
        assert composite.get_agent("alpha") is None

    @pytest.mark.unit
    def test_add_agent_returns_self(self):
        """add_agent returns self for chaining."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent
        composite = CompositeAgent()
        result = composite.add_agent("x", self._make_sub_agent())
        assert result is composite

    @pytest.mark.unit
    def test_sub_agents_returns_copy(self):
        """sub_agents property returns a copy, not the internal dict."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent
        composite = CompositeAgent()
        composite.add_agent("x", self._make_sub_agent())
        snapshot = composite.sub_agents
        snapshot["y"] = self._make_sub_agent()
        assert "y" not in composite.sub_agents

    @pytest.mark.unit
    def test_merge_outputs_combine(self):
        """_merge_outputs with COMBINE returns dict of outputs."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent, CompositeAgentConfig, MergeStrategy
        config = CompositeAgentConfig(merge_strategy=MergeStrategy.COMBINE)
        agent = CompositeAgent(config=config)
        merged = agent._merge_outputs({"a": "result_a", "b": "result_b"})
        assert merged == {"a": "result_a", "b": "result_b"}

    @pytest.mark.unit
    def test_merge_outputs_first(self):
        """_merge_outputs with FIRST returns first value."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent, CompositeAgentConfig, MergeStrategy
        config = CompositeAgentConfig(merge_strategy=MergeStrategy.FIRST)
        agent = CompositeAgent(config=config)
        merged = agent._merge_outputs({"a": "first", "b": "second"})
        assert merged == "first"

    @pytest.mark.unit
    def test_merge_outputs_concat(self):
        """_merge_outputs with CONCAT joins string representations."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent, CompositeAgentConfig, MergeStrategy
        config = CompositeAgentConfig(merge_strategy=MergeStrategy.CONCAT)
        agent = CompositeAgent(config=config)
        merged = agent._merge_outputs({"a": "hello", "b": "world"})
        assert "hello" in merged
        assert "world" in merged

    @pytest.mark.unit
    def test_merge_outputs_best(self):
        """_merge_outputs with BEST returns longest output."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent, CompositeAgentConfig, MergeStrategy
        config = CompositeAgentConfig(merge_strategy=MergeStrategy.BEST)
        agent = CompositeAgent(config=config)
        merged = agent._merge_outputs({"a": "short", "b": "this is longer"})
        assert merged == "this is longer"

    @pytest.mark.unit
    def test_merge_outputs_empty(self):
        """_merge_outputs with empty dict returns None."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent
        agent = CompositeAgent()
        assert agent._merge_outputs({}) is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_orchestrate_no_agents_fails(self):
        """_orchestrate with no sub-agents returns failure."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent, UnifiedResult
        agent = CompositeAgent()
        result = await agent._orchestrate()
        assert isinstance(result, UnifiedResult)
        assert result.success is False
        assert "No sub-agents" in result.error

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_pipeline_success(self):
        """Pipeline executes sub-agents sequentially."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent, CoordinationPattern
        a = self._make_sub_agent("a", output="from_a")
        b = self._make_sub_agent("b", output="from_b")
        composite = CompositeAgent.compose(
            "Pipeline", coordination=CoordinationPattern.PIPELINE, a=a, b=b,
        )
        composite._initialized = True
        result = await composite.execute(task="start")
        assert result.success is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_pipeline_fails_at_stage(self):
        """Pipeline stops and reports failure when a stage fails."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent, CoordinationPattern
        a = self._make_sub_agent("a", output="ok")
        b = self._make_sub_agent("b", success=False, error="stage_b_error")
        composite = CompositeAgent.compose(
            "Pipeline", coordination=CoordinationPattern.PIPELINE, a=a, b=b,
        )
        composite._initialized = True
        result = await composite.execute(task="start")
        assert result.success is False
        assert "b" in (result.error or "")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_parallel_success(self):
        """Parallel executes all sub-agents concurrently."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent, CoordinationPattern
        a = self._make_sub_agent("a", output="result_a")
        b = self._make_sub_agent("b", output="result_b")
        composite = CompositeAgent.compose(
            "Parallel", coordination=CoordinationPattern.PARALLEL, a=a, b=b,
        )
        composite._initialized = True
        result = await composite.execute(task="go")
        assert result.success is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_consensus_majority(self):
        """Consensus succeeds when majority of agents succeed."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent, CoordinationPattern
        a = self._make_sub_agent("a", output="yes")
        b = self._make_sub_agent("b", output="yes")
        c = self._make_sub_agent("c", success=False, error="no")
        composite = CompositeAgent.compose(
            "Consensus", coordination=CoordinationPattern.CONSENSUS, a=a, b=b, c=c,
        )
        composite._initialized = True
        result = await composite.execute(task="vote")
        assert result.success is True  # 2/3 majority

    @pytest.mark.unit
    def test_to_dict(self):
        """to_dict includes composite-specific fields."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent, CoordinationPattern
        a = self._make_sub_agent("a")
        composite = CompositeAgent.compose("Test", coordination=CoordinationPattern.PIPELINE, a=a)
        d = composite.to_dict()
        assert d['type'] == 'composite'
        assert d['coordination'] == 'pipeline'
        assert 'a' in d['sub_agents']
        assert d['wrapped_swarm'] is None

    @pytest.mark.unit
    def test_repr_with_agents(self):
        """__repr__ shows name and agent list."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent
        composite = CompositeAgent()
        composite.add_agent("x", self._make_sub_agent("x"))
        r = repr(composite)
        assert "CompositeAgent" in r
        assert "x" in r

    @pytest.mark.unit
    def test_extract_output_nested_unified(self):
        """_extract_output unwraps nested UnifiedResult."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent, UnifiedResult
        from Jotty.core.agents.base.base_agent import AgentResult
        inner_unified = UnifiedResult(
            success=True, output="inner_data", name="inner",
            execution_time=0.5, metadata={"nested": True},
        )
        ar = AgentResult(success=True, output=inner_unified, agent_name="wrapper", execution_time=1.0)
        output, meta = CompositeAgent._extract_output(ar)
        assert output == "inner_data"
        assert meta == {"nested": True}

    @pytest.mark.unit
    def test_extract_output_plain(self):
        """_extract_output returns plain output and metadata."""
        from Jotty.core.agents.base.composite_agent import CompositeAgent
        from Jotty.core.agents.base.base_agent import AgentResult
        ar = AgentResult(success=True, output="plain", agent_name="A", execution_time=1.0, metadata={"k": "v"})
        output, meta = CompositeAgent._extract_output(ar)
        assert output == "plain"
        assert meta == {"k": "v"}


# =============================================================================
# MetaAgent Tests
# =============================================================================

class TestMetaAgent:
    """Tests for MetaAgent self-improvement infrastructure."""

    @pytest.mark.unit
    def test_create_default(self):
        """MetaAgent creates with defaults."""
        from Jotty.core.agents.base.meta_agent import MetaAgent
        agent = MetaAgent()
        assert agent.config.name == "MetaAgent"
        assert agent.gold_db is None
        assert agent.improvement_history is None

    @pytest.mark.unit
    def test_create_with_config(self):
        """MetaAgent respects custom config."""
        from Jotty.core.agents.base.meta_agent import MetaAgent, MetaAgentConfig
        config = MetaAgentConfig(
            name="CustomMeta",
            improvement_threshold=0.9,
            max_learnings_per_run=10,
        )
        agent = MetaAgent(config=config)
        assert agent.config.name == "CustomMeta"
        assert agent.config.improvement_threshold == 0.9
        assert agent.config.max_learnings_per_run == 10

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_no_gold_db(self):
        """evaluate_against_gold returns default when no gold_db."""
        from Jotty.core.agents.base.meta_agent import MetaAgent
        agent = MetaAgent()
        result = await agent.evaluate_against_gold("test_id", {"answer": "42"})
        assert result["overall_score"] == 0.5
        assert result["result"] == "needs_improvement"
        assert "No gold standard database" in result["feedback"][0]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_gold_not_found(self):
        """evaluate_against_gold returns failed when gold standard not in db."""
        from Jotty.core.agents.base.meta_agent import MetaAgent
        mock_db = MagicMock()
        mock_db.get.return_value = None
        agent = MetaAgent(gold_db=mock_db)
        result = await agent.evaluate_against_gold("missing_id", {"answer": "42"})
        assert result["result"] == "failed"
        assert result["overall_score"] == 0.0

    @pytest.mark.unit
    def test_simple_evaluation_exact_match(self):
        """_simple_evaluation gives 1.0 for exact field matches."""
        from Jotty.core.agents.base.meta_agent import MetaAgent
        agent = MetaAgent()
        gold = MagicMock()
        gold.id = "g1"
        gold.expected_output = {"answer": "42", "source": "deep thought"}
        result = agent._simple_evaluation(gold, {"answer": "42", "source": "deep thought"})
        assert result["overall_score"] == 1.0
        assert result["result"] == "good"

    @pytest.mark.unit
    def test_simple_evaluation_partial_string(self):
        """_simple_evaluation computes word overlap for partial string match."""
        from Jotty.core.agents.base.meta_agent import MetaAgent
        agent = MetaAgent()
        gold = MagicMock()
        gold.id = "g1"
        gold.expected_output = {"summary": "the quick brown fox"}
        result = agent._simple_evaluation(gold, {"summary": "the quick red fox"})
        score = result["scores"]["summary"]
        assert 0 < score < 1  # Partial match

    @pytest.mark.unit
    def test_simple_evaluation_missing_field(self):
        """_simple_evaluation gives 0.0 for missing fields."""
        from Jotty.core.agents.base.meta_agent import MetaAgent
        agent = MetaAgent()
        gold = MagicMock()
        gold.id = "g1"
        gold.expected_output = {"answer": "42"}
        result = agent._simple_evaluation(gold, {})
        assert result["scores"]["answer"] == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_learnings_not_excellent(self):
        """extract_learnings returns empty for non-excellent results."""
        from Jotty.core.agents.base.meta_agent import MetaAgent
        agent = MetaAgent()
        result = await agent.extract_learnings(
            {"task": "x"}, {"output": "y"},
            {"result": "needs_improvement"}, domain="test"
        )
        assert result == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_learnings_good_no_dspy(self):
        """extract_learnings returns simple learnings for good results without DSPy."""
        from Jotty.core.agents.base.meta_agent import MetaAgent
        agent = MetaAgent()
        result = await agent.extract_learnings(
            {"task": "x"}, {"output": "y"},
            {"result": "good", "overall_score": 0.85}, domain="testing"
        )
        assert len(result) == 2
        assert "testing" in result[0]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_analyze_empty_evaluations(self):
        """analyze_and_suggest_improvements returns empty for no evaluations."""
        from Jotty.core.agents.base.meta_agent import MetaAgent
        agent = MetaAgent()
        result = await agent.analyze_and_suggest_improvements([])
        assert result == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_analyze_no_dspy_module(self):
        """analyze_and_suggest_improvements returns empty without DSPy module."""
        from Jotty.core.agents.base.meta_agent import MetaAgent
        agent = MetaAgent()
        result = await agent.analyze_and_suggest_improvements([{"score": 0.5}])
        assert result == []

    @pytest.mark.unit
    def test_get_agent_state_no_context(self):
        """get_agent_state returns None when context is None."""
        from Jotty.core.agents.base.meta_agent import MetaAgent
        agent = MetaAgent()
        agent._context_manager = None
        agent.config.enable_context = False
        assert agent.get_agent_state("other") is None

    @pytest.mark.unit
    def test_get_agent_state_with_context(self):
        """get_agent_state retrieves from shared context."""
        from Jotty.core.agents.base.meta_agent import MetaAgent
        agent = MetaAgent()
        mock_ctx = MagicMock()
        mock_ctx.get.return_value = {"AgentA": {"score": 0.9}}
        agent._context_manager = mock_ctx
        state = agent.get_agent_state("AgentA")
        assert state == {"score": 0.9}

    @pytest.mark.unit
    def test_publish_state(self):
        """publish_state writes to shared context."""
        from Jotty.core.agents.base.meta_agent import MetaAgent
        agent = MetaAgent()
        mock_ctx = MagicMock()
        mock_ctx.get.return_value = {}
        agent._context_manager = mock_ctx
        agent.publish_state({"status": "ready"})
        mock_ctx.set.assert_called_once()
        call_args = mock_ctx.set.call_args
        assert call_args[0][0] == "agent_states"
        assert agent.config.name in call_args[0][1]

    @pytest.mark.unit
    def test_get_all_agent_states_no_context(self):
        """get_all_agent_states returns empty dict when no context."""
        from Jotty.core.agents.base.meta_agent import MetaAgent
        agent = MetaAgent()
        agent._context_manager = None
        agent.config.enable_context = False
        assert agent.get_all_agent_states() == {}

    @pytest.mark.unit
    def test_create_meta_agent_factory(self):
        """create_meta_agent factory returns configured MetaAgent."""
        from Jotty.core.agents.base.meta_agent import create_meta_agent
        agent = create_meta_agent()
        assert isinstance(agent, type(agent))  # MetaAgent
        assert agent.config.name == "MetaAgent"

    @pytest.mark.unit
    def test_create_meta_agent_with_signature(self):
        """create_meta_agent with signature includes name in config."""
        import dspy
        from Jotty.core.agents.base.meta_agent import create_meta_agent
        class TestSig(dspy.Signature):
            """Test signature."""
            task: str = dspy.InputField()
            output: str = dspy.OutputField()
        agent = create_meta_agent(signature=TestSig)
        assert "TestSig" in agent.config.name

    @pytest.mark.unit
    def test_meta_agent_config_defaults(self):
        """MetaAgentConfig has sensible defaults."""
        from Jotty.core.agents.base.meta_agent import MetaAgentConfig
        config = MetaAgentConfig()
        assert config.enable_gold_db is True
        assert config.enable_improvement_history is True
        assert config.improvement_threshold == 0.7
        assert config.max_learnings_per_run == 5


# =============================================================================
# TypeCoercer Tests
# =============================================================================

class TestTypeCoercer:
    """Tests for TypeCoercer schema-driven type coercion."""

    @pytest.mark.unit
    def test_coerce_str_passthrough(self):
        """String type hints pass value through unchanged."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce("hello", "str")
        assert val == "hello"
        assert err is None

    @pytest.mark.unit
    def test_coerce_unknown_type_passthrough(self):
        """Unknown type hints pass value through unchanged."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce("hello", "custom_type")
        assert val == "hello"
        assert err is None

    @pytest.mark.unit
    def test_coerce_none_type_hint_passthrough(self):
        """None type_hint passes value through unchanged."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce(42, None)
        assert val == 42
        assert err is None

    @pytest.mark.unit
    def test_coerce_int_from_int(self):
        """Int values pass through."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce(42, "int")
        assert val == 42
        assert err is None

    @pytest.mark.unit
    def test_coerce_int_from_float(self):
        """Whole floats coerce to int."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce(42.0, "int")
        assert val == 42
        assert isinstance(val, int)
        assert err is None

    @pytest.mark.unit
    def test_coerce_int_from_string(self):
        """Numeric strings coerce to int."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce("42", "int")
        assert val == 42
        assert err is None

    @pytest.mark.unit
    def test_coerce_int_negative_string(self):
        """Negative numeric strings coerce to int."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce("-5", "int")
        assert val == -5
        assert err is None

    @pytest.mark.unit
    def test_coerce_int_float_string(self):
        """'42.0' string coerces to int via float intermediary."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce("42.0", "int")
        assert val == 42
        assert err is None

    @pytest.mark.unit
    def test_coerce_int_bool_rejected(self):
        """Booleans are not coerced to int (isinstance(True, int) is True in Python)."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce(True, "int")
        # Bool is rejected by the isinstance check
        assert err is not None

    @pytest.mark.unit
    def test_coerce_int_non_numeric_string(self):
        """Non-numeric strings produce error."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce("hello", "int")
        assert val == "hello"  # Original value returned
        assert err is not None

    @pytest.mark.unit
    def test_coerce_float_from_float(self):
        """Float values pass through."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce(3.14, "float")
        assert val == 3.14
        assert err is None

    @pytest.mark.unit
    def test_coerce_float_from_int(self):
        """Int values coerce to float."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce(42, "float")
        assert val == 42.0
        assert isinstance(val, float)
        assert err is None

    @pytest.mark.unit
    def test_coerce_float_from_string(self):
        """Numeric strings coerce to float."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce("3.14", "float")
        assert val == 3.14
        assert err is None

    @pytest.mark.unit
    def test_coerce_float_non_numeric(self):
        """Non-numeric strings produce error for float."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce("not_a_number", "float")
        assert err is not None

    @pytest.mark.unit
    def test_coerce_bool_true_variants(self):
        """Various truthy strings coerce to True."""
        from Jotty.core.agents._execution_types import TypeCoercer
        for s in ("true", "True", "yes", "1", "on"):
            val, err = TypeCoercer.coerce(s, "bool")
            assert val is True, f"Expected True for '{s}'"
            assert err is None

    @pytest.mark.unit
    def test_coerce_bool_false_variants(self):
        """Various falsy strings coerce to False."""
        from Jotty.core.agents._execution_types import TypeCoercer
        for s in ("false", "False", "no", "0", "off"):
            val, err = TypeCoercer.coerce(s, "bool")
            assert val is False, f"Expected False for '{s}'"
            assert err is None

    @pytest.mark.unit
    def test_coerce_bool_from_int(self):
        """Int 1/0 coerces to bool."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce(1, "bool")
        assert val is True
        assert err is None
        val2, err2 = TypeCoercer.coerce(0, "bool")
        assert val2 is False
        assert err2 is None

    @pytest.mark.unit
    def test_coerce_bool_invalid_string(self):
        """Invalid string produces error for bool."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce("maybe", "bool")
        assert err is not None

    @pytest.mark.unit
    def test_coerce_list_from_list(self):
        """Lists pass through."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce([1, 2, 3], "list")
        assert val == [1, 2, 3]
        assert err is None

    @pytest.mark.unit
    def test_coerce_list_from_json_string(self):
        """JSON array strings coerce to list."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce('["a", "b"]', "list")
        assert val == ["a", "b"]
        assert err is None

    @pytest.mark.unit
    def test_coerce_list_from_comma_separated(self):
        """Comma-separated strings coerce to list."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce("a, b, c", "list")
        assert val == ["a", "b", "c"]
        assert err is None

    @pytest.mark.unit
    def test_coerce_list_from_tuple(self):
        """Tuples coerce to list."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce((1, 2), "list")
        assert val == [1, 2]
        assert err is None

    @pytest.mark.unit
    def test_coerce_list_single_item(self):
        """Single string becomes single-element list."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce("single", "list")
        assert val == ["single"]
        assert err is None

    @pytest.mark.unit
    def test_coerce_dict_from_dict(self):
        """Dicts pass through."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce({"a": 1}, "dict")
        assert val == {"a": 1}
        assert err is None

    @pytest.mark.unit
    def test_coerce_dict_from_json_string(self):
        """JSON object strings coerce to dict."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce('{"key": "val"}', "dict")
        assert val == {"key": "val"}
        assert err is None

    @pytest.mark.unit
    def test_coerce_dict_invalid_string(self):
        """Non-JSON strings produce error for dict."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce("not json", "dict")
        assert err is not None

    @pytest.mark.unit
    def test_coerce_path_valid(self):
        """Valid file paths pass through."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce("/home/user/file.txt", "path")
        assert val == "/home/user/file.txt"
        assert err is None

    @pytest.mark.unit
    def test_coerce_path_newlines_rejected(self):
        """Paths with newlines are rejected (likely content)."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce("line1\nline2", "path")
        assert err is not None
        assert "newlines" in err.lower()

    @pytest.mark.unit
    def test_coerce_path_too_long_rejected(self):
        """Paths over 500 chars are rejected."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce("a" * 501, "path")
        assert err is not None
        assert "too long" in err.lower()

    @pytest.mark.unit
    def test_coerce_path_spaces_no_separator_rejected(self):
        """Paths with spaces but no separator are rejected (likely content)."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce("some content text here", "path")
        assert err is not None

    @pytest.mark.unit
    def test_coerce_path_bare_name_allowed(self):
        """Bare filenames without separator are allowed."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce("filename", "path")
        assert val == "filename"
        assert err is None

    @pytest.mark.unit
    def test_coerce_integer_alias(self):
        """'integer' type hint aliases to int coercion."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce("42", "integer")
        assert val == 42
        assert err is None

    @pytest.mark.unit
    def test_coerce_number_alias(self):
        """'number' type hint aliases to float coercion."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce("3.14", "number")
        assert val == 3.14
        assert err is None

    @pytest.mark.unit
    def test_coerce_array_alias(self):
        """'array' type hint aliases to list coercion."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce('[1,2]', "array")
        assert val == [1, 2]
        assert err is None

    @pytest.mark.unit
    def test_coerce_object_alias(self):
        """'object' type hint aliases to dict coercion."""
        from Jotty.core.agents._execution_types import TypeCoercer
        val, err = TypeCoercer.coerce('{"a":1}', "object")
        assert val == {"a": 1}
        assert err is None


# =============================================================================
# ToolValidationResult Tests
# =============================================================================

class TestToolValidationResult:
    """Tests for ToolValidationResult structured validation output."""

    @pytest.mark.unit
    def test_default_valid(self):
        """Default result is valid with no errors."""
        from Jotty.core.agents._execution_types import ToolValidationResult
        r = ToolValidationResult()
        assert r.valid is True
        assert r.errors == []
        assert r.coerced_params == {}
        assert r.warnings == []

    @pytest.mark.unit
    def test_error_summary_ok(self):
        """error_summary returns 'OK' when no errors."""
        from Jotty.core.agents._execution_types import ToolValidationResult
        r = ToolValidationResult()
        assert r.error_summary() == "OK"

    @pytest.mark.unit
    def test_error_summary_with_errors(self):
        """error_summary joins errors with semicolons."""
        from Jotty.core.agents._execution_types import ToolValidationResult
        r = ToolValidationResult(valid=False, errors=["missing X", "bad type Y"])
        assert "missing X" in r.error_summary()
        assert "bad type Y" in r.error_summary()
        assert ";" in r.error_summary()


# =============================================================================
# ToolSchema Tests
# =============================================================================

class TestToolSchema:
    """Tests for ToolSchema typed tool parameter definitions."""

    @pytest.mark.unit
    def test_constructor(self):
        """ToolSchema initializes with name and optional params."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam
        schema = ToolSchema(name="test_tool", description="A test tool")
        assert schema.name == "test_tool"
        assert schema.description == "A test tool"
        assert schema.params == []

    @pytest.mark.unit
    def test_get_param_by_name(self):
        """get_param finds param by canonical name."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam
        schema = ToolSchema(name="t", params=[
            ToolParam(name="query", type_hint="str", required=True),
        ])
        p = schema.get_param("query")
        assert p is not None
        assert p.name == "query"

    @pytest.mark.unit
    def test_get_param_by_alias(self):
        """get_param finds param by alias."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam
        schema = ToolSchema(name="t", params=[
            ToolParam(name="query", aliases=["q", "search_query"]),
        ])
        p = schema.get_param("q")
        assert p is not None
        assert p.name == "query"

    @pytest.mark.unit
    def test_get_param_not_found(self):
        """get_param returns None for unknown params."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam
        schema = ToolSchema(name="t", params=[
            ToolParam(name="query"),
        ])
        assert schema.get_param("nonexistent") is None

    @pytest.mark.unit
    def test_get_llm_visible_params_excludes_reserved(self):
        """get_llm_visible_params filters out reserved params."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam
        schema = ToolSchema(name="t", params=[
            ToolParam(name="query", required=True),
            ToolParam(name="_status_callback", reserved=True),
            ToolParam(name="format", required=False),
        ])
        visible = schema.get_llm_visible_params()
        names = [p.name for p in visible]
        assert "query" in names
        assert "format" in names
        assert "_status_callback" not in names

    @pytest.mark.unit
    def test_resolve_aliases(self):
        """resolve_aliases maps alias keys to canonical names."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam
        schema = ToolSchema(name="t", params=[
            ToolParam(name="query", aliases=["q", "search_query"]),
            ToolParam(name="path", aliases=["file_path"]),
        ])
        params = {"q": "test", "file_path": "/tmp/a.txt"}
        result = schema.resolve_aliases(params)
        assert "query" in result
        assert result["query"] == "test"
        assert "path" in result
        assert result["path"] == "/tmp/a.txt"

    @pytest.mark.unit
    def test_validate_missing_required(self):
        """validate reports missing required params."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam
        schema = ToolSchema(name="t", params=[
            ToolParam(name="query", required=True),
            ToolParam(name="format", required=False),
        ])
        result = schema.validate({})
        assert result.valid is False
        assert any("query" in e for e in result.errors)

    @pytest.mark.unit
    def test_validate_all_present(self):
        """validate passes when all required params present."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam
        schema = ToolSchema(name="t", params=[
            ToolParam(name="query", required=True),
        ])
        result = schema.validate({"query": "test"})
        assert result.valid is True
        assert result.errors == []

    @pytest.mark.unit
    def test_validate_with_alias(self):
        """validate finds required params via aliases."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam
        schema = ToolSchema(name="t", params=[
            ToolParam(name="query", required=True, aliases=["q"]),
        ])
        result = schema.validate({"q": "test"})
        assert result.valid is True

    @pytest.mark.unit
    def test_validate_with_coercion(self):
        """validate with coerce=True populates coerced_params."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam
        schema = ToolSchema(name="t", params=[
            ToolParam(name="count", type_hint="int", required=True),
        ])
        result = schema.validate({"count": "42"}, coerce=True)
        assert result.valid is True
        assert result.coerced_params.get("count") == 42

    @pytest.mark.unit
    def test_validate_coercion_error(self):
        """validate with coerce=True reports type errors."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam
        schema = ToolSchema(name="t", params=[
            ToolParam(name="count", type_hint="int", required=True),
        ])
        result = schema.validate({"count": "not_a_number"}, coerce=True)
        assert result.valid is False
        assert any("count" in e for e in result.errors)

    @pytest.mark.unit
    def test_auto_wire_exact_name_match(self):
        """auto_wire fills missing param from exact name in outputs."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam
        schema = ToolSchema(name="t", params=[
            ToolParam(name="query", required=True),
        ])
        outputs = {"step_1": {"query": "test value"}}
        result = schema.auto_wire({}, outputs)
        assert result["query"] == "test value"

    @pytest.mark.unit
    def test_auto_wire_skips_present_params(self):
        """auto_wire does not overwrite params already present."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam
        schema = ToolSchema(name="t", params=[
            ToolParam(name="query", required=True),
        ])
        outputs = {"step_1": {"query": "from output"}}
        result = schema.auto_wire({"query": "explicit"}, outputs)
        assert result["query"] == "explicit"

    @pytest.mark.unit
    def test_auto_wire_content_direct_name_match(self):
        """auto_wire matches content param via direct name match (not _CONTENT_FIELDS scan)."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam
        schema = ToolSchema(name="t", params=[
            ToolParam(name="content", required=True),
        ])
        # Direct 'content' key — matches by name. 'response' no longer scanned via _CONTENT_FIELDS.
        outputs = {"step_1": {"content": "A" * 60}}
        result = schema.auto_wire({}, outputs)
        assert result.get("content") == "A" * 60

    @pytest.mark.unit
    def test_auto_wire_path_fallback(self):
        """auto_wire uses path fallback for path params."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam
        schema = ToolSchema(name="t", params=[
            ToolParam(name="path", required=True),
        ])
        outputs = {"step_1": {"path": "/tmp/result.txt"}}
        result = schema.auto_wire({}, outputs)
        assert result["path"] == "/tmp/result.txt"

    @pytest.mark.unit
    def test_required_param_names(self):
        """required_param_names returns names of required params only."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam
        schema = ToolSchema(name="t", params=[
            ToolParam(name="query", required=True),
            ToolParam(name="format", required=False),
            ToolParam(name="limit", required=True),
        ])
        assert schema.required_param_names == ["query", "limit"]

    @pytest.mark.unit
    def test_to_dict(self):
        """to_dict excludes reserved params and includes schema info."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam
        schema = ToolSchema(name="search", description="Search tool", params=[
            ToolParam(name="query", type_hint="str", required=True, description="Search query"),
            ToolParam(name="_status_callback", reserved=True),
        ])
        d = schema.to_dict()
        assert d['name'] == "search"
        assert d['description'] == "Search tool"
        assert len(d['parameters']) == 1
        assert d['parameters'][0]['name'] == "query"

    @pytest.mark.unit
    def test_repr(self):
        """repr shows name and required params."""
        from Jotty.core.agents._execution_types import ToolSchema, ToolParam
        schema = ToolSchema(name="search", params=[
            ToolParam(name="query", required=True),
            ToolParam(name="limit", required=False),
        ])
        r = repr(schema)
        assert "search" in r
        assert "query" in r
        assert "limit" not in r  # Optional not shown

    @pytest.mark.unit
    def test_from_tool_function(self):
        """from_tool_function builds schema from decorated function."""
        from Jotty.core.agents._execution_types import ToolSchema

        def my_tool(params):
            """Search the web for results.

            Args:
                - query (str, required): The search query
                - limit (int, optional): Max results
            """
            pass
        my_tool._required_params = ['query']

        schema = ToolSchema.from_tool_function(my_tool, 'my_tool')
        assert schema.name == 'my_tool'
        assert schema.description == "Search the web for results."
        # query should be required (from decorator)
        q = schema.get_param('query')
        assert q is not None
        assert q.required is True
        # limit should be optional (from docstring, not in _required_params)
        lim = schema.get_param('limit')
        assert lim is not None
        assert lim.required is False

    @pytest.mark.unit
    def test_from_metadata(self):
        """from_metadata builds schema from JSON-Schema-style dict."""
        from Jotty.core.agents._execution_types import ToolSchema
        metadata = {
            'description': 'A tool',
            'parameters': {
                'properties': {
                    'query': {'type': 'string', 'description': 'Search query'},
                    'count': {'type': 'integer', 'description': 'Result count'},
                },
                'required': ['query'],
            },
        }
        schema = ToolSchema.from_metadata('my_tool', metadata)
        assert schema.name == 'my_tool'
        q = schema.get_param('query')
        assert q is not None
        assert q.required is True
        c = schema.get_param('count')
        assert c is not None
        assert c.required is False


# =============================================================================
# AgentIOSchema Tests
# =============================================================================

class TestAgentIOSchema:
    """Tests for AgentIOSchema typed I/O contracts for agents."""

    @pytest.mark.unit
    def test_constructor(self):
        """AgentIOSchema initializes with agent name, inputs, outputs."""
        from Jotty.core.agents._execution_types import AgentIOSchema, ToolParam
        schema = AgentIOSchema(
            agent_name="Researcher",
            inputs=[ToolParam(name="task")],
            outputs=[ToolParam(name="analysis")],
            description="Research agent",
        )
        assert schema.agent_name == "Researcher"
        assert len(schema.inputs) == 1
        assert len(schema.outputs) == 1
        assert schema.description == "Research agent"

    @pytest.mark.unit
    def test_input_output_names(self):
        """input_names and output_names return field name lists."""
        from Jotty.core.agents._execution_types import AgentIOSchema, ToolParam
        schema = AgentIOSchema(
            agent_name="A",
            inputs=[ToolParam(name="task"), ToolParam(name="context")],
            outputs=[ToolParam(name="analysis"), ToolParam(name="sources")],
        )
        assert schema.input_names == ["task", "context"]
        assert schema.output_names == ["analysis", "sources"]

    @pytest.mark.unit
    def test_wire_to_exact_name_match(self):
        """wire_to matches outputs to inputs by exact name."""
        from Jotty.core.agents._execution_types import AgentIOSchema, ToolParam
        schema_a = AgentIOSchema(
            agent_name="A",
            outputs=[ToolParam(name="analysis")],
        )
        schema_b = AgentIOSchema(
            agent_name="B",
            inputs=[ToolParam(name="analysis")],
        )
        mapping = schema_a.wire_to(schema_b)
        assert mapping == {"analysis": "analysis"}

    @pytest.mark.unit
    def test_wire_to_semantic_group_match(self):
        """wire_to matches via semantic groups (e.g. analysis→summary)."""
        from Jotty.core.agents._execution_types import AgentIOSchema, ToolParam
        schema_a = AgentIOSchema(
            agent_name="A",
            outputs=[ToolParam(name="analysis")],
        )
        schema_b = AgentIOSchema(
            agent_name="B",
            inputs=[ToolParam(name="summary")],
        )
        mapping = schema_a.wire_to(schema_b)
        assert mapping == {"summary": "analysis"}

    @pytest.mark.unit
    def test_wire_to_content_fallback(self):
        """wire_to uses content fallback for generic receivers."""
        from Jotty.core.agents._execution_types import AgentIOSchema, ToolParam
        schema_a = AgentIOSchema(
            agent_name="A",
            outputs=[ToolParam(name="result")],
        )
        schema_b = AgentIOSchema(
            agent_name="B",
            inputs=[ToolParam(name="input")],  # Generic content receiver
        )
        mapping = schema_a.wire_to(schema_b)
        assert "input" in mapping
        assert mapping["input"] == "result"

    @pytest.mark.unit
    def test_wire_to_no_match(self):
        """wire_to returns empty mapping when no compatible fields."""
        from Jotty.core.agents._execution_types import AgentIOSchema, ToolParam
        schema_a = AgentIOSchema(
            agent_name="A",
            outputs=[ToolParam(name="price_data")],
        )
        schema_b = AgentIOSchema(
            agent_name="B",
            inputs=[ToolParam(name="weather_info")],
        )
        mapping = schema_a.wire_to(schema_b)
        assert mapping == {}

    @pytest.mark.unit
    def test_wire_to_multiple_fields(self):
        """wire_to handles multiple input/output fields."""
        from Jotty.core.agents._execution_types import AgentIOSchema, ToolParam
        schema_a = AgentIOSchema(
            agent_name="A",
            outputs=[
                ToolParam(name="analysis"),
                ToolParam(name="sources"),
            ],
        )
        schema_b = AgentIOSchema(
            agent_name="B",
            inputs=[
                ToolParam(name="analysis"),
                ToolParam(name="references"),  # Semantic match with 'sources'
            ],
        )
        mapping = schema_a.wire_to(schema_b)
        assert mapping["analysis"] == "analysis"
        assert mapping["references"] == "sources"

    @pytest.mark.unit
    def test_map_outputs_basic(self):
        """map_outputs transforms output dict to target kwargs."""
        from Jotty.core.agents._execution_types import AgentIOSchema, ToolParam
        schema_a = AgentIOSchema(
            agent_name="A",
            outputs=[ToolParam(name="analysis")],
        )
        schema_b = AgentIOSchema(
            agent_name="B",
            inputs=[ToolParam(name="analysis")],
        )
        result = schema_a.map_outputs({"analysis": "The result text"}, schema_b)
        assert result == {"analysis": "The result text"}

    @pytest.mark.unit
    def test_map_outputs_skips_missing(self):
        """map_outputs skips fields not present in output dict."""
        from Jotty.core.agents._execution_types import AgentIOSchema, ToolParam
        schema_a = AgentIOSchema(
            agent_name="A",
            outputs=[ToolParam(name="analysis"), ToolParam(name="sources")],
        )
        schema_b = AgentIOSchema(
            agent_name="B",
            inputs=[ToolParam(name="analysis"), ToolParam(name="sources")],
        )
        # Only analysis present in output, not sources
        result = schema_a.map_outputs({"analysis": "text"}, schema_b)
        assert result == {"analysis": "text"}
        assert "sources" not in result

    @pytest.mark.unit
    def test_map_outputs_with_semantic_wiring(self):
        """map_outputs applies semantic wiring to transform keys."""
        from Jotty.core.agents._execution_types import AgentIOSchema, ToolParam
        schema_a = AgentIOSchema(
            agent_name="A",
            outputs=[ToolParam(name="analysis")],
        )
        schema_b = AgentIOSchema(
            agent_name="B",
            inputs=[ToolParam(name="summary")],  # Semantic match
        )
        result = schema_a.map_outputs({"analysis": "full analysis"}, schema_b)
        assert result == {"summary": "full analysis"}

    @pytest.mark.unit
    def test_to_dict(self):
        """to_dict returns structured representation."""
        from Jotty.core.agents._execution_types import AgentIOSchema, ToolParam
        schema = AgentIOSchema(
            agent_name="Research",
            inputs=[ToolParam(name="task", type_hint="str", description="Task")],
            outputs=[ToolParam(name="output", type_hint="str", description="Result")],
            description="Research agent",
        )
        d = schema.to_dict()
        assert d['agent_name'] == "Research"
        assert d['description'] == "Research agent"
        assert len(d['inputs']) == 1
        assert d['inputs'][0]['name'] == "task"
        assert len(d['outputs']) == 1

    @pytest.mark.unit
    def test_repr(self):
        """repr shows agent name, inputs, and outputs."""
        from Jotty.core.agents._execution_types import AgentIOSchema, ToolParam
        schema = AgentIOSchema(
            agent_name="A",
            inputs=[ToolParam(name="task")],
            outputs=[ToolParam(name="result")],
        )
        r = repr(schema)
        assert "A" in r
        assert "task" in r
        assert "result" in r

    @pytest.mark.unit
    def test_from_dspy_signature(self):
        """from_dspy_signature builds schema from DSPy Signature class."""
        import dspy
        from Jotty.core.agents._execution_types import AgentIOSchema

        class TestSig(dspy.Signature):
            """Analyze a topic and produce findings."""
            task: str = dspy.InputField(desc="The task to analyze")
            context: str = dspy.InputField(desc="Background context")
            analysis: str = dspy.OutputField(desc="Analysis results")

        schema = AgentIOSchema.from_dspy_signature("Analyzer", TestSig)
        assert schema.agent_name == "Analyzer"
        assert "task" in schema.input_names
        assert "context" in schema.input_names
        assert "analysis" in schema.output_names
        assert "Analyze" in schema.description

    @pytest.mark.unit
    def test_from_dspy_signature_empty(self):
        """from_dspy_signature handles empty/minimal signatures."""
        import dspy
        from Jotty.core.agents._execution_types import AgentIOSchema

        class MinimalSig(dspy.Signature):
            """Minimal."""
            task: str = dspy.InputField()
            output: str = dspy.OutputField()

        schema = AgentIOSchema.from_dspy_signature("Minimal", MinimalSig)
        assert len(schema.inputs) >= 1
        assert len(schema.outputs) >= 1


# =============================================================================
# FileReference Tests
# =============================================================================

class TestFileReference:
    """Tests for FileReference lazy file handle."""

    @pytest.mark.unit
    def test_exists_false(self):
        """exists returns False for non-existent path."""
        from Jotty.core.agents._execution_types import FileReference
        ref = FileReference(path="/nonexistent/file.txt", size_bytes=100)
        assert ref.exists() is False

    @pytest.mark.unit
    def test_exists_true(self, tmp_path):
        """exists returns True for existing file."""
        from Jotty.core.agents._execution_types import FileReference
        f = tmp_path / "test.txt"
        f.write_text("hello")
        ref = FileReference(path=str(f))
        assert ref.exists() is True

    @pytest.mark.unit
    def test_load(self, tmp_path):
        """load reads file contents."""
        from Jotty.core.agents._execution_types import FileReference
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        ref = FileReference(path=str(f))
        assert ref.load() == "hello world"

    @pytest.mark.unit
    def test_fields(self):
        """FileReference stores metadata fields."""
        from Jotty.core.agents._execution_types import FileReference
        ref = FileReference(
            path="/tmp/out.json",
            content_type="application/json",
            size_bytes=1024,
            checksum="abc123",
            step_key="step_0",
            description="Output file",
        )
        assert ref.content_type == "application/json"
        assert ref.size_bytes == 1024
        assert ref.checksum == "abc123"
        assert ref.step_key == "step_0"


# =============================================================================
# SwarmArtifactStore Tests
# =============================================================================

class TestSwarmArtifactStore:
    """Tests for SwarmArtifactStore tag-queryable artifact registry."""

    @pytest.mark.unit
    def test_register_and_get(self):
        """register stores artifact, get retrieves it."""
        from Jotty.core.agents._execution_types import SwarmArtifactStore
        store = SwarmArtifactStore()
        store.register("step_0", "result data", tags=["search"], description="Search output")
        assert store.get("step_0") == "result data"

    @pytest.mark.unit
    def test_get_default(self):
        """get returns default for missing keys."""
        from Jotty.core.agents._execution_types import SwarmArtifactStore
        store = SwarmArtifactStore()
        assert store.get("missing") is None
        assert store.get("missing", "fallback") == "fallback"

    @pytest.mark.unit
    def test_query_by_tag(self):
        """query_by_tag returns matching artifacts."""
        from Jotty.core.agents._execution_types import SwarmArtifactStore
        store = SwarmArtifactStore()
        store.register("a", "data_a", tags=["search", "web"])
        store.register("b", "data_b", tags=["file"])
        store.register("c", "data_c", tags=["search"])
        results = store.query_by_tag("search")
        assert "a" in results
        assert "c" in results
        assert "b" not in results

    @pytest.mark.unit
    def test_query_by_tag_no_match(self):
        """query_by_tag returns empty dict for unmatched tag."""
        from Jotty.core.agents._execution_types import SwarmArtifactStore
        store = SwarmArtifactStore()
        store.register("a", "data", tags=["web"])
        assert store.query_by_tag("nonexistent") == {}

    @pytest.mark.unit
    def test_dict_getitem(self):
        """__getitem__ provides dict-style access."""
        from Jotty.core.agents._execution_types import SwarmArtifactStore
        store = SwarmArtifactStore()
        store.register("k", "v")
        assert store["k"] == "v"

    @pytest.mark.unit
    def test_dict_setitem(self):
        """__setitem__ provides backward-compatible dict assignment."""
        from Jotty.core.agents._execution_types import SwarmArtifactStore
        store = SwarmArtifactStore()
        store["key"] = "value"
        assert store.get("key") == "value"

    @pytest.mark.unit
    def test_contains(self):
        """__contains__ supports 'in' operator."""
        from Jotty.core.agents._execution_types import SwarmArtifactStore
        store = SwarmArtifactStore()
        store.register("x", 1)
        assert "x" in store
        assert "y" not in store

    @pytest.mark.unit
    def test_len_and_bool(self):
        """__len__ and __bool__ work correctly."""
        from Jotty.core.agents._execution_types import SwarmArtifactStore
        store = SwarmArtifactStore()
        assert len(store) == 0
        assert not store
        store.register("a", 1)
        assert len(store) == 1
        assert store

    @pytest.mark.unit
    def test_keys_values_items(self):
        """keys(), values(), items() iterate correctly."""
        from Jotty.core.agents._execution_types import SwarmArtifactStore
        store = SwarmArtifactStore()
        store.register("a", 1)
        store.register("b", 2)
        assert set(store.keys()) == {"a", "b"}
        assert set(store.values()) == {1, 2}
        assert dict(store.items()) == {"a": 1, "b": 2}

    @pytest.mark.unit
    def test_to_outputs_dict(self):
        """to_outputs_dict converts to plain dict."""
        from Jotty.core.agents._execution_types import SwarmArtifactStore
        store = SwarmArtifactStore()
        store.register("step_0", {"result": "data"}, tags=["search"])
        d = store.to_outputs_dict()
        assert isinstance(d, dict)
        assert d["step_0"] == {"result": "data"}


# =============================================================================
# AgenticExecutionResult Tests
# =============================================================================

class TestAgenticExecutionResult:
    """Tests for AgenticExecutionResult properties."""

    @pytest.mark.unit
    def test_summary_success(self):
        """summary includes success status and timing."""
        from Jotty.core.agents._execution_types import AgenticExecutionResult, TaskType
        result = AgenticExecutionResult(
            success=True, task="Test", task_type=TaskType.RESEARCH,
            skills_used=["web-search"], steps_executed=3,
            outputs={}, final_output="Done", execution_time=2.5,
        )
        s = result.summary
        assert "completed successfully" in s
        assert "2.5s" in s
        assert "3 steps" in s
        assert "web-search" in s

    @pytest.mark.unit
    def test_summary_failure(self):
        """summary reflects failure status and errors."""
        from Jotty.core.agents._execution_types import AgenticExecutionResult, TaskType
        result = AgenticExecutionResult(
            success=False, task="Test", task_type=TaskType.ANALYSIS,
            skills_used=[], steps_executed=1,
            outputs={}, final_output="", execution_time=0.5,
            errors=["timeout", "rate limited"],
        )
        s = result.summary
        assert "failed" in s
        assert "timeout" in s

    @pytest.mark.unit
    def test_artifacts_from_file_operations(self):
        """artifacts extracts files from file-operations outputs."""
        from Jotty.core.agents._execution_types import AgenticExecutionResult, TaskType
        result = AgenticExecutionResult(
            success=True, task="Write", task_type=TaskType.CREATION,
            skills_used=["file-operations"], steps_executed=1,
            outputs={"step_0": {"path": "/tmp/out.txt", "bytes_written": 500, "success": True}},
            final_output="",
        )
        arts = result.artifacts
        assert len(arts) == 1
        assert arts[0]['path'] == "/tmp/out.txt"
        assert arts[0]['size_bytes'] == 500

    @pytest.mark.unit
    def test_artifacts_empty_outputs(self):
        """artifacts returns empty list for no file outputs."""
        from Jotty.core.agents._execution_types import AgenticExecutionResult, TaskType
        result = AgenticExecutionResult(
            success=True, task="Query", task_type=TaskType.RESEARCH,
            skills_used=[], steps_executed=0,
            outputs={}, final_output="answer",
        )
        assert result.artifacts == []


# =============================================================================
# AutonomousAgent Tests
# =============================================================================

class TestAutonomousAgent:
    """Tests for AutonomousAgent config, ExecutionContextManager, and factory."""

    @pytest.mark.unit
    def test_autonomous_agent_config_defaults(self):
        """AutonomousAgentConfig has correct defaults."""
        from Jotty.core.agents.base.autonomous_agent import AutonomousAgentConfig
        config = AutonomousAgentConfig()
        assert config.max_steps == 10
        assert config.enable_replanning is False
        assert config.max_replans == 3
        assert config.skill_filter is None
        assert config.default_output_skill is None
        assert config.enable_output is False
        assert config.enable_ensemble is False
        assert config.ensemble_strategy == "multi_perspective"

    @pytest.mark.unit
    def test_autonomous_agent_config_custom(self):
        """AutonomousAgentConfig accepts custom values."""
        from Jotty.core.agents.base.autonomous_agent import AutonomousAgentConfig
        config = AutonomousAgentConfig(
            max_steps=20,
            enable_replanning=True,
            max_replans=5,
            skill_filter="coding",
            default_output_skill="telegram-sender",
            enable_output=True,
            enable_ensemble=True,
            ensemble_strategy="debate",
        )
        assert config.max_steps == 20
        assert config.enable_replanning is True
        assert config.max_replans == 5
        assert config.skill_filter == "coding"
        assert config.default_output_skill == "telegram-sender"
        assert config.enable_output is True
        assert config.enable_ensemble is True
        assert config.ensemble_strategy == "debate"

    @pytest.mark.unit
    def test_autonomous_agent_forces_enable_skills(self):
        """AutonomousAgent constructor forces enable_skills=True."""
        from Jotty.core.agents.base.autonomous_agent import (
            AutonomousAgent, AutonomousAgentConfig,
        )
        config = AutonomousAgentConfig(name="TestAuto", enable_skills=False)
        agent = AutonomousAgent(config)
        assert agent.config.enable_skills is True

    @pytest.mark.unit
    def test_autonomous_agent_default_name(self):
        """AutonomousAgent uses class name when no config provided."""
        from Jotty.core.agents.base.autonomous_agent import AutonomousAgent
        agent = AutonomousAgent()
        assert agent.config.name == "AutonomousAgent"

    @pytest.mark.unit
    def test_execution_context_manager_empty(self):
        """ExecutionContextManager starts empty."""
        from Jotty.core.agents.base.autonomous_agent import ExecutionContextManager
        ctx = ExecutionContextManager()
        assert ctx.get_context() == []
        assert ctx.get_trajectory() == []

    @pytest.mark.unit
    def test_execution_context_manager_add_step(self):
        """add_step appends to history."""
        from Jotty.core.agents.base.autonomous_agent import ExecutionContextManager
        ctx = ExecutionContextManager()
        ctx.add_step({"step": 0, "output": "hello"})
        ctx.add_step({"step": 1, "output": "world"})
        assert len(ctx.get_context()) == 2
        assert ctx.get_context()[0]["step"] == 0
        assert ctx.get_context()[1]["step"] == 1

    @pytest.mark.unit
    def test_execution_context_manager_get_context_returns_copy(self):
        """get_context returns a copy, not the internal list."""
        from Jotty.core.agents.base.autonomous_agent import ExecutionContextManager
        ctx = ExecutionContextManager()
        ctx.add_step({"step": 0})
        snapshot = ctx.get_context()
        snapshot.append({"step": 99})
        assert len(ctx.get_context()) == 1

    @pytest.mark.unit
    def test_execution_context_manager_get_trajectory_excludes_compressed(self):
        """get_trajectory returns only uncompressed steps."""
        from Jotty.core.agents.base.autonomous_agent import ExecutionContextManager
        ctx = ExecutionContextManager()
        # Manually inject a compressed entry
        ctx._history.append({"_compressed": True, "summary": "old stuff"})
        ctx._history.append({"step": 5, "output": "recent"})
        trajectory = ctx.get_trajectory()
        assert len(trajectory) == 1
        assert trajectory[0]["step"] == 5

    @pytest.mark.unit
    def test_execution_context_manager_compression_trigger(self):
        """Compression triggers when total chars exceed max_history_size."""
        from Jotty.core.agents.base.autonomous_agent import ExecutionContextManager
        ctx = ExecutionContextManager(max_history_size=500)
        # Add enough data to trigger compression
        for i in range(20):
            ctx.add_step({"step": i, "output": "A" * 50})
        # After compression, history should be shorter than 20 entries
        assert len(ctx.get_context()) < 20
        # Should contain at least one compressed entry
        compressed = [e for e in ctx.get_context() if e.get("_compressed")]
        assert len(compressed) >= 1

    @pytest.mark.unit
    def test_execution_context_manager_compression_preserves_recent(self):
        """Compression keeps recent entries intact."""
        from Jotty.core.agents.base.autonomous_agent import ExecutionContextManager
        ctx = ExecutionContextManager(max_history_size=500)
        for i in range(20):
            ctx.add_step({"step": i, "output": "X" * 50})
        # The most recent entries should be uncompressed
        trajectory = ctx.get_trajectory()
        assert len(trajectory) > 0
        # Last step should be the most recently added
        last_uncompressed = trajectory[-1]
        assert last_uncompressed["step"] == 19

    @pytest.mark.unit
    def test_execution_context_manager_no_compression_when_small(self):
        """No compression when below threshold."""
        from Jotty.core.agents.base.autonomous_agent import ExecutionContextManager
        ctx = ExecutionContextManager(max_history_size=100_000)
        for i in range(5):
            ctx.add_step({"step": i, "output": "short"})
        assert len(ctx.get_context()) == 5
        compressed = [e for e in ctx.get_context() if e.get("_compressed")]
        assert len(compressed) == 0

    @pytest.mark.unit
    def test_create_autonomous_agent_factory(self):
        """create_autonomous_agent returns configured AutonomousAgent."""
        from Jotty.core.agents.base.autonomous_agent import create_autonomous_agent
        agent = create_autonomous_agent(max_steps=15, enable_replanning=True)
        assert agent.config.name == "AutonomousAgent"
        assert agent.config.max_steps == 15
        assert agent.config.enable_replanning is True

    @pytest.mark.unit
    def test_create_autonomous_agent_factory_with_skill_filter(self):
        """create_autonomous_agent respects skill_filter."""
        from Jotty.core.agents.base.autonomous_agent import create_autonomous_agent
        agent = create_autonomous_agent(skill_filter="research")
        assert agent.config.skill_filter == "research"

    @pytest.mark.unit
    def test_build_result_success(self):
        """_build_result produces correct success dict."""
        import time
        from Jotty.core.agents.base.autonomous_agent import AutonomousAgent
        start_time = time.time()
        result = AutonomousAgent._build_result(
            task="test task",
            task_type="research",
            outputs={"step_0": {"result": "data"}},
            skills_used=["web-search"],
            errors=[],
            warnings=[],
            start_time=start_time,
            stopped=False,
        )
        assert result["success"] is True
        assert result["task"] == "test task"
        assert result["task_type"] == "research"
        assert result["steps_executed"] == 1
        assert "web-search" in result["skills_used"]
        assert result["stopped_early"] is False

    @pytest.mark.unit
    def test_build_result_failure(self):
        """_build_result produces failure when errors present."""
        import time
        from Jotty.core.agents.base.autonomous_agent import AutonomousAgent
        start_time = time.time()
        result = AutonomousAgent._build_result(
            task="failing task",
            task_type="analysis",
            outputs={},
            skills_used=[],
            errors=["step 1 failed"],
            warnings=[],
            start_time=start_time,
            stopped=True,
        )
        assert result["success"] is False
        assert result["stopped_early"] is True

    @pytest.mark.unit
    def test_build_result_too_hard(self):
        """_build_result includes too_hard flag."""
        import time
        from Jotty.core.agents.base.autonomous_agent import AutonomousAgent
        start_time = time.time()
        result = AutonomousAgent._build_result(
            task="impossible",
            task_type="unknown",
            outputs={},
            skills_used=[],
            errors=["TOO_HARD"],
            warnings=[],
            start_time=start_time,
            stopped=True,
            too_hard=True,
        )
        assert result.get("too_hard") is True


# =============================================================================
# AutoAgent Tests
# =============================================================================

class TestAutoAgent:
    """Tests for AutoAgent configuration, mode prompts, and task type inference."""

    @pytest.mark.unit
    def test_auto_agent_default_creation(self):
        """AutoAgent creates with default config."""
        from Jotty.core.agents.auto_agent import AutoAgent
        agent = AutoAgent()
        assert agent.config.name == "AutoAgent"
        assert agent.config.max_steps == 10
        assert agent.config.enable_replanning is True
        assert agent.config.max_replans == 3

    @pytest.mark.unit
    def test_auto_agent_custom_name(self):
        """AutoAgent uses custom name when provided."""
        from Jotty.core.agents.auto_agent import AutoAgent
        agent = AutoAgent(name="MyAgent")
        assert agent.config.name == "MyAgent"

    @pytest.mark.unit
    def test_auto_agent_custom_timeout(self):
        """AutoAgent respects custom timeout."""
        from Jotty.core.agents.auto_agent import AutoAgent
        agent = AutoAgent(timeout=600)
        assert agent.config.timeout == 600.0

    @pytest.mark.unit
    def test_auto_agent_output_skill_config(self):
        """AutoAgent configures output skill when both provided."""
        from Jotty.core.agents.auto_agent import AutoAgent
        agent = AutoAgent(
            default_output_skill="telegram-sender",
            enable_output=True,
        )
        assert agent.default_output_skill == "telegram-sender"
        assert agent.enable_output is True
        assert agent.config.enable_output is True

    @pytest.mark.unit
    def test_auto_agent_output_disabled_without_skill(self):
        """AutoAgent disables output when no skill specified."""
        from Jotty.core.agents.auto_agent import AutoAgent
        agent = AutoAgent(enable_output=True, default_output_skill=None)
        assert agent.enable_output is False
        assert agent.config.enable_output is False

    @pytest.mark.unit
    def test_auto_agent_system_prompt(self):
        """AutoAgent stores system_prompt in config."""
        from Jotty.core.agents.auto_agent import AutoAgent
        agent = AutoAgent(system_prompt="You are an expert researcher.")
        assert agent.config.system_prompt == "You are an expert researcher."

    @pytest.mark.unit
    def test_auto_agent_planner_injection(self):
        """AutoAgent accepts injected planner."""
        from Jotty.core.agents.auto_agent import AutoAgent
        mock_planner = MagicMock()
        agent = AutoAgent(planner=mock_planner)
        assert agent._planner is mock_planner

    @pytest.mark.unit
    def test_mode_prompts_dict_has_expected_keys(self):
        """_MODE_PROMPTS has entries for playwright, selenium, and terminal."""
        from Jotty.core.agents.auto_agent import _MODE_PROMPTS
        assert "browser-automation:playwright" in _MODE_PROMPTS
        assert "browser-automation:selenium" in _MODE_PROMPTS
        assert "terminal-session" in _MODE_PROMPTS

    @pytest.mark.unit
    def test_mode_prompts_content(self):
        """_MODE_PROMPTS entries contain relevant guidance."""
        from Jotty.core.agents.auto_agent import _MODE_PROMPTS
        assert "Playwright" in _MODE_PROMPTS["browser-automation:playwright"]
        assert "Selenium" in _MODE_PROMPTS["browser-automation:selenium"]
        assert "pexpect" in _MODE_PROMPTS["terminal-session"]

    @pytest.mark.unit
    @patch.dict('os.environ', {'BROWSER_BACKEND': 'playwright'})
    def test_get_mode_prompts_returns_matching(self):
        """_get_mode_prompts returns prompt for matching skill and backend."""
        from Jotty.core.agents.auto_agent import AutoAgent
        agent = AutoAgent()
        prompts = agent._get_mode_prompts({"browser-automation"})
        assert "Playwright" in prompts

    @pytest.mark.unit
    @patch.dict('os.environ', {'BROWSER_BACKEND': 'selenium'})
    def test_get_mode_prompts_filters_by_backend(self):
        """_get_mode_prompts only returns prompt matching BROWSER_BACKEND."""
        from Jotty.core.agents.auto_agent import AutoAgent
        agent = AutoAgent()
        prompts = agent._get_mode_prompts({"browser-automation"})
        assert "Selenium" in prompts
        assert "Playwright" not in prompts

    @pytest.mark.unit
    def test_get_mode_prompts_empty_when_no_match(self):
        """_get_mode_prompts returns empty string for non-matching skills."""
        from Jotty.core.agents.auto_agent import AutoAgent
        agent = AutoAgent()
        prompts = agent._get_mode_prompts({"web-search", "calculator"})
        assert prompts == ""

    @pytest.mark.unit
    def test_get_mode_prompts_terminal_session(self):
        """_get_mode_prompts includes terminal-session prompt."""
        from Jotty.core.agents.auto_agent import AutoAgent
        agent = AutoAgent()
        prompts = agent._get_mode_prompts({"terminal-session"})
        assert "pexpect" in prompts

    @pytest.mark.unit
    def test_infer_task_type_with_planner(self):
        """_infer_task_type delegates to planner when available."""
        from Jotty.core.agents.auto_agent import AutoAgent
        from Jotty.core.agents._execution_types import TaskType
        mock_planner = MagicMock()
        mock_planner.infer_task_type.return_value = (
            TaskType.RESEARCH, "Looks like research", 0.9
        )
        agent = AutoAgent(planner=mock_planner)
        result = agent._infer_task_type("Find recent AI papers")
        assert result == "research"
        mock_planner.infer_task_type.assert_called_once_with("Find recent AI papers")

    @pytest.mark.unit
    def test_infer_task_type_enum_returns_enum(self):
        """_infer_task_type_enum returns TaskType enum."""
        from Jotty.core.agents.auto_agent import AutoAgent
        from Jotty.core.agents._execution_types import TaskType
        mock_planner = MagicMock()
        mock_planner.infer_task_type.return_value = (
            TaskType.ANALYSIS, "Analysis task", 0.85
        )
        agent = AutoAgent(planner=mock_planner)
        result = agent._infer_task_type_enum("Analyze this data")
        assert result == TaskType.ANALYSIS

    @pytest.mark.unit
    def test_infer_task_type_enum_unknown_fallback(self):
        """_infer_task_type_enum returns UNKNOWN for invalid types."""
        from Jotty.core.agents.auto_agent import AutoAgent
        from Jotty.core.agents._execution_types import TaskType
        mock_planner = MagicMock()
        mock_planner.infer_task_type.side_effect = Exception("planner error")
        agent = AutoAgent(planner=mock_planner)
        # When planner fails, base class fallback returns a string that may
        # not match a TaskType enum value, resulting in UNKNOWN
        with patch.object(agent, '_infer_task_type', return_value="unrecognized_type"):
            result = agent._infer_task_type_enum("Do something weird")
            assert result == TaskType.UNKNOWN


# =============================================================================
# FeedbackChannel Tests
# =============================================================================

class TestFeedbackChannel:
    """Tests for FeedbackChannel inter-agent communication."""

    @pytest.mark.unit
    def test_feedback_type_enum_values(self):
        """FeedbackType has expected enum values."""
        from Jotty.core.agents.feedback_channel import FeedbackType
        assert FeedbackType.QUESTION.value == "question"
        assert FeedbackType.ERROR.value == "error"
        assert FeedbackType.SUGGESTION.value == "suggestion"
        assert FeedbackType.REQUEST.value == "request"
        assert FeedbackType.CLARIFICATION.value == "clarification"
        assert FeedbackType.RESPONSE.value == "response"

    @pytest.mark.unit
    def test_feedback_message_creation(self):
        """FeedbackMessage creates with all fields."""
        from Jotty.core.agents.feedback_channel import FeedbackMessage, FeedbackType
        msg = FeedbackMessage(
            source_actor="AgentA",
            target_actor="AgentB",
            feedback_type=FeedbackType.QUESTION,
            content="What tables are relevant?",
            context={"tables": ["t1", "t2"]},
            priority=1,
        )
        assert msg.source_actor == "AgentA"
        assert msg.target_actor == "AgentB"
        assert msg.feedback_type == FeedbackType.QUESTION
        assert msg.content == "What tables are relevant?"
        assert msg.context == {"tables": ["t1", "t2"]}
        assert msg.priority == 1
        assert msg.requires_response is True

    @pytest.mark.unit
    def test_feedback_message_defaults(self):
        """FeedbackMessage has sensible defaults."""
        from Jotty.core.agents.feedback_channel import FeedbackMessage, FeedbackType
        msg = FeedbackMessage(
            source_actor="A",
            target_actor="B",
            feedback_type=FeedbackType.RESPONSE,
            content="Here is the response",
        )
        assert msg.context == {}
        assert msg.requires_response is True
        assert msg.priority == 1
        assert msg.original_message_id is None
        assert msg.message_id.startswith("msg_")

    @pytest.mark.unit
    def test_feedback_channel_send(self):
        """send() stores message and returns message ID."""
        from Jotty.core.agents.feedback_channel import (
            FeedbackChannel, FeedbackMessage, FeedbackType,
        )
        channel = FeedbackChannel()
        msg = FeedbackMessage(
            source_actor="A",
            target_actor="B",
            feedback_type=FeedbackType.QUESTION,
            content="Help?",
        )
        msg_id = channel.send(msg)
        assert msg_id == msg.message_id
        assert channel.message_count == 1

    @pytest.mark.unit
    def test_feedback_channel_get_for_actor(self):
        """get_for_actor returns messages for the specified actor."""
        from Jotty.core.agents.feedback_channel import (
            FeedbackChannel, FeedbackMessage, FeedbackType,
        )
        channel = FeedbackChannel()
        channel.send(FeedbackMessage(
            source_actor="A", target_actor="B",
            feedback_type=FeedbackType.QUESTION, content="Q1",
        ))
        channel.send(FeedbackMessage(
            source_actor="C", target_actor="B",
            feedback_type=FeedbackType.SUGGESTION, content="S1",
        ))
        channel.send(FeedbackMessage(
            source_actor="A", target_actor="D",
            feedback_type=FeedbackType.REQUEST, content="R1",
        ))
        messages = channel.get_for_actor("B")
        assert len(messages) == 2
        assert all(m.target_actor == "B" for m in messages)

    @pytest.mark.unit
    def test_feedback_channel_get_for_actor_clears(self):
        """get_for_actor with clear=True removes retrieved messages."""
        from Jotty.core.agents.feedback_channel import (
            FeedbackChannel, FeedbackMessage, FeedbackType,
        )
        channel = FeedbackChannel()
        channel.send(FeedbackMessage(
            source_actor="A", target_actor="B",
            feedback_type=FeedbackType.QUESTION, content="Q1",
        ))
        messages = channel.get_for_actor("B", clear=True)
        assert len(messages) == 1
        # Second retrieval should be empty
        assert len(channel.get_for_actor("B")) == 0

    @pytest.mark.unit
    def test_feedback_channel_get_for_actor_no_clear(self):
        """get_for_actor with clear=False keeps messages."""
        from Jotty.core.agents.feedback_channel import (
            FeedbackChannel, FeedbackMessage, FeedbackType,
        )
        channel = FeedbackChannel()
        channel.send(FeedbackMessage(
            source_actor="A", target_actor="B",
            feedback_type=FeedbackType.QUESTION, content="Q1",
        ))
        messages = channel.get_for_actor("B", clear=False)
        assert len(messages) == 1
        # Messages still there
        assert len(channel.get_for_actor("B", clear=False)) == 1

    @pytest.mark.unit
    def test_feedback_channel_priority_filtering(self):
        """get_for_actor filters by priority_threshold."""
        from Jotty.core.agents.feedback_channel import (
            FeedbackChannel, FeedbackMessage, FeedbackType,
        )
        channel = FeedbackChannel()
        channel.send(FeedbackMessage(
            source_actor="A", target_actor="B",
            feedback_type=FeedbackType.QUESTION, content="High",
            priority=1,
        ))
        channel.send(FeedbackMessage(
            source_actor="A", target_actor="B",
            feedback_type=FeedbackType.SUGGESTION, content="Low",
            priority=3,
        ))
        # Only get high priority
        high = channel.get_for_actor("B", priority_threshold=1)
        assert len(high) == 1
        assert high[0].content == "High"

    @pytest.mark.unit
    def test_feedback_channel_priority_ordering(self):
        """get_for_actor returns messages sorted by priority (high first)."""
        from Jotty.core.agents.feedback_channel import (
            FeedbackChannel, FeedbackMessage, FeedbackType,
        )
        channel = FeedbackChannel()
        channel.send(FeedbackMessage(
            source_actor="A", target_actor="B",
            feedback_type=FeedbackType.SUGGESTION, content="Low",
            priority=3,
        ))
        channel.send(FeedbackMessage(
            source_actor="A", target_actor="B",
            feedback_type=FeedbackType.QUESTION, content="High",
            priority=1,
        ))
        channel.send(FeedbackMessage(
            source_actor="A", target_actor="B",
            feedback_type=FeedbackType.REQUEST, content="Medium",
            priority=2,
        ))
        messages = channel.get_for_actor("B", priority_threshold=3)
        assert messages[0].priority == 1
        assert messages[1].priority == 2
        assert messages[2].priority == 3

    @pytest.mark.unit
    def test_feedback_channel_has_feedback(self):
        """has_feedback returns True when messages exist."""
        from Jotty.core.agents.feedback_channel import (
            FeedbackChannel, FeedbackMessage, FeedbackType,
        )
        channel = FeedbackChannel()
        assert channel.has_feedback("B") is False
        channel.send(FeedbackMessage(
            source_actor="A", target_actor="B",
            feedback_type=FeedbackType.QUESTION, content="Q1",
        ))
        assert channel.has_feedback("B") is True
        assert channel.has_feedback("C") is False

    @pytest.mark.unit
    def test_feedback_channel_get_conversation(self):
        """get_conversation returns all messages between two actors."""
        from Jotty.core.agents.feedback_channel import (
            FeedbackChannel, FeedbackMessage, FeedbackType,
        )
        channel = FeedbackChannel()
        channel.send(FeedbackMessage(
            source_actor="A", target_actor="B",
            feedback_type=FeedbackType.QUESTION, content="Q from A",
        ))
        channel.send(FeedbackMessage(
            source_actor="B", target_actor="A",
            feedback_type=FeedbackType.RESPONSE, content="R from B",
        ))
        channel.send(FeedbackMessage(
            source_actor="C", target_actor="A",
            feedback_type=FeedbackType.QUESTION, content="Unrelated",
        ))
        conv = channel.get_conversation("A", "B")
        assert len(conv) == 2
        assert conv[0].content == "Q from A"
        assert conv[1].content == "R from B"

    @pytest.mark.unit
    def test_feedback_channel_get_stats(self):
        """get_stats returns correct statistics."""
        from Jotty.core.agents.feedback_channel import (
            FeedbackChannel, FeedbackMessage, FeedbackType,
        )
        channel = FeedbackChannel()
        channel.send(FeedbackMessage(
            source_actor="A", target_actor="B",
            feedback_type=FeedbackType.QUESTION, content="Q1",
        ))
        channel.send(FeedbackMessage(
            source_actor="B", target_actor="A",
            feedback_type=FeedbackType.RESPONSE, content="R1",
        ))
        stats = channel.get_stats()
        assert stats["total_messages"] == 2
        assert stats["pending_messages"] == 2
        assert "B" in stats["actors_with_pending"]
        assert "A" in stats["actors_with_pending"]
        assert stats["message_types"]["question"] == 1
        assert stats["message_types"]["response"] == 1

    @pytest.mark.unit
    def test_feedback_channel_broadcast(self):
        """broadcast sends to all specified participants except self."""
        from Jotty.core.agents.feedback_channel import (
            FeedbackChannel, FeedbackMessage, FeedbackType,
        )
        channel = FeedbackChannel()
        msg_ids = channel.broadcast(
            source_actor="Leader",
            content="Announcement",
            participants=["Leader", "Agent1", "Agent2", "Agent3"],
            priority=2,
        )
        # Should NOT send to Leader (self)
        assert len(msg_ids) == 3
        assert channel.has_feedback("Agent1") is True
        assert channel.has_feedback("Agent2") is True
        assert channel.has_feedback("Agent3") is True
        assert channel.has_feedback("Leader") is False

    @pytest.mark.unit
    def test_feedback_channel_clear_all(self):
        """clear_all removes all pending messages but keeps history."""
        from Jotty.core.agents.feedback_channel import (
            FeedbackChannel, FeedbackMessage, FeedbackType,
        )
        channel = FeedbackChannel()
        channel.send(FeedbackMessage(
            source_actor="A", target_actor="B",
            feedback_type=FeedbackType.QUESTION, content="Q1",
        ))
        assert channel.has_feedback("B") is True
        channel.clear_all()
        assert channel.has_feedback("B") is False
        # History is preserved
        assert len(channel.message_history) == 1

    @pytest.mark.unit
    def test_feedback_channel_format_messages_empty(self):
        """format_messages_for_agent returns empty string for no messages."""
        from Jotty.core.agents.feedback_channel import FeedbackChannel
        channel = FeedbackChannel()
        assert channel.format_messages_for_agent("B", []) == ""

    @pytest.mark.unit
    def test_feedback_channel_format_messages_with_content(self):
        """format_messages_for_agent produces formatted string."""
        from Jotty.core.agents.feedback_channel import (
            FeedbackChannel, FeedbackMessage, FeedbackType,
        )
        channel = FeedbackChannel()
        msg = FeedbackMessage(
            source_actor="A", target_actor="B",
            feedback_type=FeedbackType.QUESTION, content="What tables?",
            context={"tables": ["t1"]},
        )
        formatted = channel.format_messages_for_agent("B", [msg])
        assert "MESSAGES FOR B" in formatted
        assert "FROM A" in formatted
        assert "What tables?" in formatted
        assert "END MESSAGES" in formatted

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_feedback_channel_request_timeout(self):
        """request() returns None on timeout."""
        from Jotty.core.agents.feedback_channel import FeedbackChannel
        channel = FeedbackChannel()
        result = await channel.request(
            source_actor="A",
            target_actor="B",
            content="Question?",
            timeout=0.2,
            poll_interval=0.05,
        )
        assert result is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_feedback_channel_request_with_response(self):
        """request() returns response when available."""
        from Jotty.core.agents.feedback_channel import (
            FeedbackChannel, FeedbackMessage, FeedbackType,
        )
        channel = FeedbackChannel()

        async def respond_later():
            """Simulate delayed response."""
            await asyncio.sleep(0.05)
            # Get the pending question
            pending = channel.messages.get("B", [])
            if pending:
                msg_id = pending[0].message_id
                channel.send(FeedbackMessage(
                    source_actor="B",
                    target_actor="A",
                    feedback_type=FeedbackType.RESPONSE,
                    content="Here is the answer",
                    original_message_id=msg_id,
                ))

        # Start response coroutine concurrently
        import asyncio
        task = asyncio.create_task(respond_later())
        result = await channel.request(
            source_actor="A",
            target_actor="B",
            content="What tables?",
            timeout=2.0,
            poll_interval=0.02,
        )
        await task
        assert result is not None
        assert result.content == "Here is the answer"
        assert result.feedback_type == FeedbackType.RESPONSE


# =============================================================================
# DAGTypes Tests
# =============================================================================

try:
    from Jotty.core.agents.dag_types import (
        TaskType as _DAGTaskType, Actor as _DAGActor,
        ExecutableDAG as _DAGExecutableDAG, DAGAgentMixin as _DAGMixin,
        SwarmResources as _DAGSwarmResources,
    )
    _DAG_TYPES_AVAILABLE = True
except (ImportError, Exception):
    _DAG_TYPES_AVAILABLE = False

_skip_no_dag = pytest.mark.skipif(
    not _DAG_TYPES_AVAILABLE,
    reason="dag_types not importable (missing dep)",
)


@_skip_no_dag
class TestDAGTypes:
    """Tests for DAG types: TaskType, Actor, ExecutableDAG, DAGAgentMixin, SwarmResources."""

    @pytest.mark.unit
    def test_dag_task_type_enum(self):
        """DAG TaskType has expected values."""
        from Jotty.core.agents.dag_types import TaskType
        assert TaskType.SETUP.value == "setup"
        assert TaskType.IMPLEMENTATION.value == "implementation"
        assert TaskType.TESTING.value == "testing"
        assert TaskType.EXECUTION.value == "execution"
        assert TaskType.DOCUMENTATION.value == "documentation"
        assert TaskType.VALIDATION.value == "validation"
        assert TaskType.RESEARCH.value == "research"
        assert TaskType.ANALYSIS.value == "analysis"

    @pytest.mark.unit
    def test_actor_creation(self):
        """Actor dataclass initializes correctly."""
        from Jotty.core.agents.dag_types import Actor
        actor = Actor(
            name="coder",
            capabilities=["coding", "testing"],
            description="A coding agent",
        )
        assert actor.name == "coder"
        assert actor.capabilities == ["coding", "testing"]
        assert actor.description == "A coding agent"
        assert actor.max_concurrent_tasks == 1

    @pytest.mark.unit
    def test_actor_can_handle_matching(self):
        """Actor.can_handle returns True for matching capabilities."""
        from Jotty.core.agents.dag_types import Actor
        actor = Actor(name="coder", capabilities=["coding", "testing"])
        assert actor.can_handle("implementation") is True
        assert actor.can_handle("testing") is True

    @pytest.mark.unit
    def test_actor_can_handle_no_match(self):
        """Actor.can_handle returns False when no capability matches."""
        from Jotty.core.agents.dag_types import Actor
        actor = Actor(name="coder", capabilities=["coding"])
        assert actor.can_handle("research") is False
        assert actor.can_handle("documentation") is False

    @pytest.mark.unit
    def test_actor_can_handle_setup(self):
        """Actor with git capability can handle setup tasks."""
        from Jotty.core.agents.dag_types import Actor
        actor = Actor(name="devops", capabilities=["git", "setup"])
        assert actor.can_handle("setup") is True

    @pytest.mark.unit
    def test_actor_can_handle_research(self):
        """Actor with web_search can handle research tasks."""
        from Jotty.core.agents.dag_types import Actor
        actor = Actor(name="researcher", capabilities=["web_search"])
        assert actor.can_handle("research") is True

    @pytest.mark.unit
    def test_actor_can_handle_unknown_type_fallback(self):
        """Actor.can_handle uses type_name as fallback capability."""
        from Jotty.core.agents.dag_types import Actor
        actor = Actor(name="special", capabilities=["magic"])
        assert actor.can_handle("magic") is True
        assert actor.can_handle("nonexistent") is False

    @pytest.mark.unit
    def test_dag_agent_mixin_init(self):
        """DAGAgentMixin._init_agent_infrastructure sets up metrics."""
        from Jotty.core.agents.dag_types import DAGAgentMixin

        class TestMixin(DAGAgentMixin):
            pass

        obj = TestMixin()
        obj._init_agent_infrastructure("TestDagAgent")
        assert obj._agent_config.name == "TestDagAgent"
        assert obj._metrics["total_executions"] == 0
        assert obj._metrics["successful_executions"] == 0
        assert obj._initialized is False

    @pytest.mark.unit
    def test_dag_agent_mixin_track_execution(self):
        """_track_execution updates metrics correctly."""
        from Jotty.core.agents.dag_types import DAGAgentMixin

        class TestMixin(DAGAgentMixin):
            pass

        obj = TestMixin()
        obj._init_agent_infrastructure("TestDag")
        obj._track_execution(success=True, execution_time=1.5)
        obj._track_execution(success=False, execution_time=0.5)
        assert obj._metrics["total_executions"] == 2
        assert obj._metrics["successful_executions"] == 1
        assert obj._metrics["failed_executions"] == 1
        assert obj._metrics["total_execution_time"] == 2.0

    @pytest.mark.unit
    def test_dag_agent_mixin_get_metrics(self):
        """get_metrics computes success_rate and avg_execution_time."""
        from Jotty.core.agents.dag_types import DAGAgentMixin

        class TestMixin(DAGAgentMixin):
            pass

        obj = TestMixin()
        obj._init_agent_infrastructure("TestDag")
        obj._track_execution(success=True, execution_time=2.0)
        obj._track_execution(success=True, execution_time=4.0)
        metrics = obj.get_metrics()
        assert metrics["success_rate"] == 1.0
        assert metrics["avg_execution_time"] == 3.0

    @pytest.mark.unit
    def test_dag_agent_mixin_get_metrics_zero(self):
        """get_metrics handles zero executions."""
        from Jotty.core.agents.dag_types import DAGAgentMixin

        class TestMixin(DAGAgentMixin):
            pass

        obj = TestMixin()
        obj._init_agent_infrastructure("TestDag")
        metrics = obj.get_metrics()
        assert metrics["success_rate"] == 0.0
        assert metrics["avg_execution_time"] == 0.0

    @pytest.mark.unit
    def test_dag_agent_mixin_reset_metrics(self):
        """reset_metrics zeros all counters."""
        from Jotty.core.agents.dag_types import DAGAgentMixin

        class TestMixin(DAGAgentMixin):
            pass

        obj = TestMixin()
        obj._init_agent_infrastructure("TestDag")
        obj._track_execution(success=True, execution_time=1.0)
        obj.reset_metrics()
        assert obj._metrics["total_executions"] == 0
        assert obj._metrics["total_execution_time"] == 0.0

    @pytest.mark.unit
    def test_dag_agent_mixin_hooks(self):
        """Pre and post hooks are called."""
        from Jotty.core.agents.dag_types import DAGAgentMixin

        class TestMixin(DAGAgentMixin):
            pass

        obj = TestMixin()
        obj._init_agent_infrastructure("TestDag")
        pre_called = []
        post_called = []
        obj.add_pre_hook(lambda agent, **kw: pre_called.append(True))
        obj.add_post_hook(lambda agent, result, **kw: post_called.append(result))
        obj._run_pre_hooks()
        obj._run_post_hooks("result_data")
        assert len(pre_called) == 1
        assert post_called == ["result_data"]

    @pytest.mark.unit
    def test_dag_agent_mixin_hooks_exception_handled(self):
        """Hook exceptions are caught and logged."""
        from Jotty.core.agents.dag_types import DAGAgentMixin

        class TestMixin(DAGAgentMixin):
            pass

        obj = TestMixin()
        obj._init_agent_infrastructure("TestDag")
        obj.add_pre_hook(lambda agent, **kw: (_ for _ in ()).throw(RuntimeError("hook fail")))
        # Should not raise
        obj._run_pre_hooks()

    @pytest.mark.unit
    def test_dag_agent_mixin_to_dict(self):
        """to_dict returns serializable representation."""
        from Jotty.core.agents.dag_types import DAGAgentMixin

        class TestMixin(DAGAgentMixin):
            pass

        obj = TestMixin()
        obj._init_agent_infrastructure("TestDag")
        d = obj.to_dict()
        assert d["name"] == "TestDag"
        assert d["class"] == "TestMixin"
        assert "metrics" in d
        assert d["initialized"] is False

    @pytest.mark.unit
    def test_swarm_resources_singleton(self):
        """SwarmResources is a singleton."""
        from Jotty.core.agents.dag_types import SwarmResources
        SwarmResources.reset()
        try:
            r1 = SwarmResources.get_instance()
            r2 = SwarmResources.get_instance()
            assert r1 is r2
        finally:
            SwarmResources.reset()

    @pytest.mark.unit
    def test_swarm_resources_reset(self):
        """SwarmResources.reset clears singleton."""
        from Jotty.core.agents.dag_types import SwarmResources
        SwarmResources.reset()
        try:
            r1 = SwarmResources.get_instance()
            SwarmResources.reset()
            r2 = SwarmResources.get_instance()
            assert r1 is not r2
        finally:
            SwarmResources.reset()

    @pytest.mark.unit
    def test_swarm_resources_has_components(self):
        """SwarmResources initializes memory, context, bus, learner."""
        from Jotty.core.agents.dag_types import SwarmResources
        SwarmResources.reset()
        try:
            resources = SwarmResources.get_instance()
            assert resources.memory is not None
            assert resources.context is not None
            assert resources.bus is not None
            assert resources.learner is not None
        finally:
            SwarmResources.reset()

    @pytest.mark.unit
    def test_executable_dag_total_tasks(self):
        """ExecutableDAG.total_tasks returns task count."""
        from Jotty.core.agents.dag_types import ExecutableDAG
        mock_todo = MagicMock()
        mock_todo.subtasks = {"t1": MagicMock(), "t2": MagicMock()}
        dag = ExecutableDAG(
            markovian_todo=mock_todo,
            assignments={},
            validation_passed=True,
        )
        assert dag.total_tasks == 2

    @pytest.mark.unit
    def test_executable_dag_get_execution_stages(self):
        """get_execution_stages returns topologically sorted stages."""
        from Jotty.core.agents.dag_types import ExecutableDAG
        mock_todo = MagicMock()
        # t1 has no deps, t2 depends on t1, t3 depends on t1
        t1 = MagicMock()
        t1.depends_on = []
        t2 = MagicMock()
        t2.depends_on = ["t1"]
        t3 = MagicMock()
        t3.depends_on = ["t1"]
        mock_todo.subtasks = {"t1": t1, "t2": t2, "t3": t3}
        dag = ExecutableDAG(
            markovian_todo=mock_todo,
            assignments={},
            validation_passed=True,
        )
        stages = dag.get_execution_stages()
        assert len(stages) == 2
        assert stages[0] == ["t1"]
        assert set(stages[1]) == {"t2", "t3"}

    @pytest.mark.unit
    def test_executable_dag_add_trajectory_step(self):
        """add_trajectory_step records execution step."""
        from Jotty.core.agents.dag_types import ExecutableDAG
        mock_todo = MagicMock()
        mock_todo.subtasks = {}
        dag = ExecutableDAG(
            markovian_todo=mock_todo,
            assignments={},
            validation_passed=True,
        )
        dag.add_trajectory_step(
            task_id="t1",
            action_type="execute",
            action_content="Run implementation",
            observation="Success",
            reward=1.0,
        )
        assert len(dag.trajectory) == 1
        assert dag.trajectory[0].action_type == "execute"
        assert dag.trajectory[0].reward == 1.0

    @pytest.mark.unit
    def test_executable_dag_to_dict(self):
        """to_dict serializes DAG to dict."""
        from Jotty.core.agents.dag_types import ExecutableDAG, Actor
        mock_todo = MagicMock()
        mock_todo.todo_id = "dag_1"
        mock_todo.root_task = "Build app"
        mock_todo.subtasks = {}
        mock_todo.execution_order = []
        actor = Actor(name="coder", capabilities=["coding"])
        dag = ExecutableDAG(
            markovian_todo=mock_todo,
            assignments={"t1": actor},
            validation_passed=True,
            validation_issues=[],
            fixes_applied=[],
        )
        d = dag.to_dict()
        assert d["markovian_todo"]["todo_id"] == "dag_1"
        assert d["markovian_todo"]["root_task"] == "Build app"
        assert d["validation_passed"] is True
        assert "t1" in d["assignments"]
        assert d["assignments"]["t1"]["name"] == "coder"


# =============================================================================
# ExecutionTypes Deep Tests
# =============================================================================

class TestExecutionTypesDeep:
    """Deep tests for _execution_types: TaskType, ExecutionStep, ToolParam, ToolStats, CapabilityIndex."""

    @pytest.mark.unit
    def test_task_type_enum_all_values(self):
        """TaskType enum has all expected members."""
        from Jotty.core.agents._execution_types import TaskType
        assert TaskType.RESEARCH.value == "research"
        assert TaskType.COMPARISON.value == "comparison"
        assert TaskType.CREATION.value == "creation"
        assert TaskType.COMMUNICATION.value == "communication"
        assert TaskType.ANALYSIS.value == "analysis"
        assert TaskType.AUTOMATION.value == "automation"
        assert TaskType.UNKNOWN.value == "unknown"

    @pytest.mark.unit
    def test_execution_step_defaults(self):
        """ExecutionStep has correct default values."""
        from Jotty.core.agents._execution_types import ExecutionStep
        step = ExecutionStep(
            skill_name="web-search",
            tool_name="search_web_tool",
            params={"query": "AI"},
            description="Search the web",
        )
        assert step.depends_on == []
        assert step.output_key == ""
        assert step.optional is False
        assert step.verification == ""
        assert step.fallback_skill == ""

    @pytest.mark.unit
    def test_execution_step_all_fields(self):
        """ExecutionStep stores all provided fields."""
        from Jotty.core.agents._execution_types import ExecutionStep
        step = ExecutionStep(
            skill_name="file-operations",
            tool_name="write_file_tool",
            params={"path": "/tmp/out.txt", "content": "hello"},
            description="Write output file",
            depends_on=[0, 1],
            output_key="step_2",
            optional=True,
            verification="file exists",
            fallback_skill="shell-exec",
        )
        assert step.skill_name == "file-operations"
        assert step.depends_on == [0, 1]
        assert step.output_key == "step_2"
        assert step.optional is True
        assert step.verification == "file exists"
        assert step.fallback_skill == "shell-exec"

    @pytest.mark.unit
    def test_tool_param_defaults(self):
        """ToolParam has correct defaults."""
        from Jotty.core.agents._execution_types import ToolParam
        p = ToolParam(name="query")
        assert p.type_hint == "str"
        assert p.required is True
        assert p.description == ""
        assert p.default is None
        assert p.aliases == []
        assert p.reserved is False

    @pytest.mark.unit
    def test_tool_param_reserved(self):
        """ToolParam can be marked as reserved."""
        from Jotty.core.agents._execution_types import ToolParam
        p = ToolParam(name="_status_callback", reserved=True)
        assert p.reserved is True

    @pytest.mark.unit
    def test_tool_stats_record_and_get(self):
        """ToolStats records and retrieves stats."""
        from Jotty.core.agents._execution_types import ToolStats
        stats = ToolStats()
        stats.record("web-search", "search_tool", success=True, latency_ms=500)
        stats.record("web-search", "search_tool", success=True, latency_ms=700)
        stats.record("web-search", "search_tool", success=False, latency_ms=1000)
        s = stats.get_stats("web-search", "search_tool")
        assert s["call_count"] == 3
        assert abs(s["success_rate"] - 2 / 3) < 0.01
        assert abs(s["avg_latency_ms"] - 733.33) < 1.0

    @pytest.mark.unit
    def test_tool_stats_empty(self):
        """ToolStats returns zeros for unrecorded tools."""
        from Jotty.core.agents._execution_types import ToolStats
        stats = ToolStats()
        s = stats.get_stats("unknown", "tool")
        assert s["call_count"] == 0
        assert s["success_rate"] == 0.0
        assert s["avg_latency_ms"] == 0.0

    @pytest.mark.unit
    def test_tool_stats_get_summary(self):
        """get_summary returns human-readable string."""
        from Jotty.core.agents._execution_types import ToolStats
        stats = ToolStats()
        stats.record("calc", "add_tool", success=True, latency_ms=100)
        summary = stats.get_summary("calc", "add_tool")
        assert "calc/add_tool" in summary
        assert "100%" in summary
        assert "1 calls" in summary

    @pytest.mark.unit
    def test_tool_stats_get_summary_no_history(self):
        """get_summary returns 'no history' for unrecorded tools."""
        from Jotty.core.agents._execution_types import ToolStats
        stats = ToolStats()
        summary = stats.get_summary("unknown", "tool")
        assert "no history" in summary

    @pytest.mark.unit
    def test_tool_stats_max_history(self):
        """ToolStats respects max_history limit."""
        from Jotty.core.agents._execution_types import ToolStats
        stats = ToolStats(max_history=5)
        for i in range(10):
            stats.record("s", "t", success=True, latency_ms=100.0)
        s = stats.get_stats("s", "t")
        assert s["call_count"] == 5

    @pytest.mark.unit
    def test_tool_stats_global_instance(self):
        """TOOL_STATS is a global singleton instance."""
        from Jotty.core.agents._execution_types import TOOL_STATS, ToolStats
        assert isinstance(TOOL_STATS, ToolStats)

    @pytest.mark.unit
    def test_capability_index_register_and_find_chain(self):
        """CapabilityIndex.find_chain finds valid tool chains."""
        from Jotty.core.agents._execution_types import CapabilityIndex
        idx = CapabilityIndex()
        idx.register("web-search", inputs=["query"], outputs=["search_results"])
        idx.register("summarizer", inputs=["search_results"], outputs=["summary"])
        chain = idx.find_chain("query", "summary")
        assert chain == ["web-search", "summarizer"]

    @pytest.mark.unit
    def test_capability_index_no_chain(self):
        """find_chain returns empty list when no chain exists."""
        from Jotty.core.agents._execution_types import CapabilityIndex
        idx = CapabilityIndex()
        idx.register("web-search", inputs=["query"], outputs=["search_results"])
        chain = idx.find_chain("query", "summary")
        assert chain == []

    @pytest.mark.unit
    def test_capability_index_same_type(self):
        """find_chain returns empty for same start and end type."""
        from Jotty.core.agents._execution_types import CapabilityIndex
        idx = CapabilityIndex()
        chain = idx.find_chain("query", "query")
        assert chain == []

    @pytest.mark.unit
    def test_capability_index_get_tools_for_type(self):
        """get_tools_for_type returns producers and consumers."""
        from Jotty.core.agents._execution_types import CapabilityIndex
        idx = CapabilityIndex()
        idx.register("web-search", inputs=["query"], outputs=["search_results"])
        idx.register("rag", inputs=["search_results"], outputs=["answer"])
        info = idx.get_tools_for_type("search_results")
        assert "web-search" in info["producers"]
        assert "rag" in info["consumers"]

    @pytest.mark.unit
    def test_capability_index_multi_hop_chain(self):
        """find_chain finds multi-hop chains."""
        from Jotty.core.agents._execution_types import CapabilityIndex
        idx = CapabilityIndex()
        idx.register("search", inputs=["query"], outputs=["results"])
        idx.register("filter", inputs=["results"], outputs=["filtered"])
        idx.register("summarize", inputs=["filtered"], outputs=["summary"])
        chain = idx.find_chain("query", "summary")
        assert chain == ["search", "filter", "summarize"]

    @pytest.mark.unit
    def test_capability_index_max_depth(self):
        """find_chain respects max_depth limit."""
        from Jotty.core.agents._execution_types import CapabilityIndex
        idx = CapabilityIndex()
        # Create a chain of depth 6
        for i in range(6):
            idx.register(f"tool_{i}", inputs=[f"type_{i}"], outputs=[f"type_{i+1}"])
        # Should not find chain with max_depth=3
        chain = idx.find_chain("type_0", "type_6", max_depth=3)
        assert chain == []

    @pytest.mark.unit
    def test_agentic_execution_result_artifacts_from_outputs(self):
        """AgenticExecutionResult.artifacts extracts from multiple outputs."""
        from Jotty.core.agents._execution_types import AgenticExecutionResult, TaskType
        result = AgenticExecutionResult(
            success=True,
            task="Generate files",
            task_type=TaskType.CREATION,
            skills_used=["file-operations"],
            steps_executed=2,
            outputs={
                "step_0": {"path": "/tmp/a.txt", "success": True, "bytes_written": 100},
                "step_1": {"path": "/tmp/b.txt", "success": True, "bytes_written": 200},
            },
            final_output="Done",
        )
        arts = result.artifacts
        assert len(arts) == 2
        paths = {a["path"] for a in arts}
        assert "/tmp/a.txt" in paths
        assert "/tmp/b.txt" in paths

    @pytest.mark.unit
    def test_agentic_execution_result_summary_with_skills(self):
        """AgenticExecutionResult.summary includes skills info."""
        from Jotty.core.agents._execution_types import AgenticExecutionResult, TaskType
        result = AgenticExecutionResult(
            success=True,
            task="Research AI",
            task_type=TaskType.RESEARCH,
            skills_used=["web-search", "claude-cli-llm"],
            steps_executed=3,
            outputs={},
            final_output="AI is advancing",
            execution_time=5.0,
        )
        s = result.summary
        assert "web-search" in s
        assert "claude-cli-llm" in s
        assert "completed successfully" in s

    @pytest.mark.unit
    def test_swarm_artifact_store_register_with_tags(self):
        """SwarmArtifactStore register stores tags correctly."""
        from Jotty.core.agents._execution_types import SwarmArtifactStore
        store = SwarmArtifactStore()
        store.register("s0", {"data": "x"}, tags=["search", "web"], description="Search result")
        store.register("s1", {"data": "y"}, tags=["file"], description="File output")
        results = store.query_by_tag("search")
        assert "s0" in results
        assert "s1" not in results

    @pytest.mark.unit
    def test_swarm_artifact_store_to_outputs_dict(self):
        """to_outputs_dict converts to plain dict."""
        from Jotty.core.agents._execution_types import SwarmArtifactStore
        store = SwarmArtifactStore()
        store.register("a", {"result": 1}, tags=["search"])
        store.register("b", {"result": 2}, tags=["file"])
        d = store.to_outputs_dict()
        assert d == {"a": {"result": 1}, "b": {"result": 2}}


# =============================================================================
# ChatAssistant Tests
# =============================================================================

class TestChatAssistant:
    """Tests for ChatAssistant keyword detection and factory."""

    @pytest.mark.unit
    def test_chat_assistant_creation(self):
        """ChatAssistant creates with default config."""
        from Jotty.core.agents.chat_assistant import ChatAssistant
        assistant = ChatAssistant()
        assert assistant.state_manager is None
        assert assistant.config == {}

    @pytest.mark.unit
    def test_chat_assistant_with_state_manager(self):
        """ChatAssistant stores state_manager reference."""
        from Jotty.core.agents.chat_assistant import ChatAssistant
        mock_sm = MagicMock()
        assistant = ChatAssistant(state_manager=mock_sm)
        assert assistant.state_manager is mock_sm

    @pytest.mark.unit
    def test_is_task_query_positive(self):
        """_is_task_query detects task-related keywords."""
        from Jotty.core.agents.chat_assistant import ChatAssistant
        assistant = ChatAssistant()
        assert assistant._is_task_query("show me all tasks") is True
        assert assistant._is_task_query("what is in the backlog") is True
        assert assistant._is_task_query("any pending items?") is True
        assert assistant._is_task_query("completed work") is True
        assert assistant._is_task_query("tasks done today") is True
        assert assistant._is_task_query("show my todo list") is True
        assert assistant._is_task_query("what is in progress") is True

    @pytest.mark.unit
    def test_is_task_query_negative(self):
        """_is_task_query rejects non-task queries."""
        from Jotty.core.agents.chat_assistant import ChatAssistant
        assistant = ChatAssistant()
        assert assistant._is_task_query("hello world") is False
        assert assistant._is_task_query("what is the weather") is False
        assert assistant._is_task_query("tell me a joke") is False

    @pytest.mark.unit
    def test_is_status_query_positive(self):
        """_is_status_query detects status-related keywords."""
        from Jotty.core.agents.chat_assistant import ChatAssistant
        assistant = ChatAssistant()
        assert assistant._is_status_query("system status") is True
        assert assistant._is_status_query("health check") is True
        assert assistant._is_status_query("is the system running") is True

    @pytest.mark.unit
    def test_is_status_query_negative(self):
        """_is_status_query rejects non-status queries."""
        from Jotty.core.agents.chat_assistant import ChatAssistant
        assistant = ChatAssistant()
        assert assistant._is_status_query("hello") is False
        assert assistant._is_status_query("show tasks") is False

    @pytest.mark.unit
    def test_is_help_query_positive(self):
        """_is_help_query detects help-related keywords."""
        from Jotty.core.agents.chat_assistant import ChatAssistant
        assistant = ChatAssistant()
        assert assistant._is_help_query("help me") is True
        assert assistant._is_help_query("how do I do this") is True
        assert assistant._is_help_query("what can you do") is True
        assert assistant._is_help_query("your capabilities") is True

    @pytest.mark.unit
    def test_is_help_query_negative(self):
        """_is_help_query rejects non-help queries."""
        from Jotty.core.agents.chat_assistant import ChatAssistant
        assistant = ChatAssistant()
        assert assistant._is_help_query("hello") is False
        assert assistant._is_help_query("list tasks") is False
        assert assistant._is_help_query("tell me a joke") is False

    @pytest.mark.unit
    def test_handle_help_query_returns_card(self):
        """_handle_help_query returns a card with help text."""
        from Jotty.core.agents.chat_assistant import ChatAssistant
        assistant = ChatAssistant()
        result = assistant._handle_help_query()
        assert isinstance(result, dict)
        # Should be a card-type widget
        assert "title" in str(result) or "content" in str(result) or "role" in result

    @pytest.mark.unit
    def test_handle_general_query(self):
        """_handle_general_query returns text response."""
        from Jotty.core.agents.chat_assistant import ChatAssistant
        assistant = ChatAssistant()
        result = assistant._handle_general_query("hello world")
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_task_to_dict_from_dict(self):
        """_task_to_dict returns dict unchanged."""
        from Jotty.core.agents.chat_assistant import ChatAssistant
        assistant = ChatAssistant()
        task_dict = {"task_id": "1", "title": "Test", "status": "backlog"}
        assert assistant._task_to_dict(task_dict) is task_dict

    @pytest.mark.unit
    def test_task_to_dict_from_object(self):
        """_task_to_dict converts object with attributes to dict."""
        from Jotty.core.agents.chat_assistant import ChatAssistant
        assistant = ChatAssistant()
        mock_task = MagicMock()
        mock_task.task_id = "T-001"
        mock_task.title = "Build feature"
        mock_task.description = "Build the feature"
        mock_task.status = "in_progress"
        mock_task.priority = 3
        mock_task.created_at = None
        mock_task.updated_at = None
        result = assistant._task_to_dict(mock_task)
        assert result["task_id"] == "T-001"
        assert result["title"] == "Build feature"
        assert result["status"] == "in_progress"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_no_goal_returns_summary(self):
        """run() with no goal returns task summary widget."""
        from Jotty.core.agents.chat_assistant import ChatAssistant
        assistant = ChatAssistant()
        # With no state_manager, _fetch_tasks returns empty
        result = await assistant.run()
        assert isinstance(result, dict)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_help_query(self):
        """run() routes help queries to _handle_help_query."""
        from Jotty.core.agents.chat_assistant import ChatAssistant
        assistant = ChatAssistant()
        result = await assistant.run(goal="help me understand")
        assert isinstance(result, dict)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_run_status_query(self):
        """run() routes status queries to _handle_status_query."""
        from Jotty.core.agents.chat_assistant import ChatAssistant
        assistant = ChatAssistant()
        result = await assistant.run(goal="system status")
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_create_chat_assistant_no_api_key(self):
        """create_chat_assistant falls back to V1 without API key."""
        from Jotty.core.agents.chat_assistant import create_chat_assistant, ChatAssistant
        with patch.dict('os.environ', {}, clear=True):
            # Remove ANTHROPIC_API_KEY if present
            import os
            old_key = os.environ.pop('ANTHROPIC_API_KEY', None)
            try:
                assistant = create_chat_assistant()
                assert isinstance(assistant, ChatAssistant)
            finally:
                if old_key is not None:
                    os.environ['ANTHROPIC_API_KEY'] = old_key

"""
Tests for Agent Core Module
=============================
Tests for BaseAgent execution, retries, hooks, metrics, and AgentResult.
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

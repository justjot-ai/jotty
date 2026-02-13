"""
Tests for CompositeAgent, CompositeAgentConfig, and UnifiedResult.

Covers the bridge pattern that unifies Agent and Swarm hierarchies,
including configuration defaults, result conversion, sub-agent management,
and factory methods (from_swarm, compose).
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from Jotty.core.agents.base.composite_agent import (
    CompositeAgent,
    CompositeAgentConfig,
    UnifiedResult,
)
from Jotty.core.swarms.swarm_types import SwarmResult, ExecutionTrace, AgentRole
from Jotty.core.swarms.base.agent_team import CoordinationPattern, MergeStrategy


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

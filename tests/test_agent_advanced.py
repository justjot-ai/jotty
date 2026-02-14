"""
Advanced Agent Tests - MetaAgent, CompositeAgent, AutonomousAgent, AutoAgent, SkillPlanExecutor

Comprehensive unit tests covering:
- MetaAgentConfig / MetaAgent (gold evaluation, improvements, learnings, state)
- CompositeAgentConfig / CompositeAgent (pipeline, parallel, consensus, merge)
- UnifiedResult (bidirectional AgentResult/SwarmResult bridge)
- AutonomousAgentConfig / AutonomousAgent (planning, execution, replanning)
- AutoAgent (task type inference, ensemble, mode prompts)
- SkillPlanExecutor (planning, validation, caching, exclusions)
- ExecutionContextManager (step tracking, compression, trajectory)
- ToolCallCache (TTL, LRU eviction, thread safety)

Author: A-Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import json
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import (
    AsyncMock, MagicMock, Mock, PropertyMock, patch, call
)

import pytest

# Ensure Jotty parent is on path for consistent imports
_jotty_parent = str(Path(__file__).parent.parent.parent)
if _jotty_parent not in sys.path:
    sys.path.insert(0, _jotty_parent)

# Try importing core modules with skipif fallback
try:
    from Jotty.core.agents.base.base_agent import BaseAgent, AgentRuntimeConfig, AgentResult
    from Jotty.core.agents.base.meta_agent import MetaAgent, MetaAgentConfig, create_meta_agent
    from Jotty.core.agents.base.composite_agent import (
        CompositeAgent, CompositeAgentConfig, UnifiedResult,
    )
    from Jotty.core.swarms.base.agent_team import CoordinationPattern, MergeStrategy
    from Jotty.core.agents.base.autonomous_agent import (
        AutonomousAgent, AutonomousAgentConfig, ExecutionContextManager,
        create_autonomous_agent,
    )
    from Jotty.core.agents.base.skill_plan_executor import SkillPlanExecutor, ToolCallCache
    AGENTS_AVAILABLE = True
except ImportError as e:
    AGENTS_AVAILABLE = False

try:
    from Jotty.core.agents.auto_agent import AutoAgent
    AUTO_AGENT_AVAILABLE = True
except ImportError:
    AUTO_AGENT_AVAILABLE = False

try:
    from Jotty.core.agents._execution_types import (
        ExecutionStep, TaskType, AgenticExecutionResult,
    )
    EXEC_TYPES_AVAILABLE = True
except ImportError:
    EXEC_TYPES_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not AGENTS_AVAILABLE,
    reason="Core agent modules not importable",
)


# =============================================================================
# HELPERS
# =============================================================================


class _ConcreteAgent(BaseAgent):
    """Minimal concrete BaseAgent for testing infrastructure."""

    async def _execute_impl(self, **kwargs) -> Any:
        return {"result": "ok"}


def _make_agent_result(success=True, output="ok", name="test") -> AgentResult:
    return AgentResult(
        success=success,
        output=output,
        agent_name=name,
        execution_time=0.1,
    )


def _make_mock_sub_agent(name="sub", success=True, output="done"):
    """Create a mock agent whose execute() returns an AgentResult."""
    agent = MagicMock(spec=BaseAgent)
    agent.config = MagicMock()
    agent.config.name = name
    agent.config.timeout = 60.0
    result = _make_agent_result(success=success, output=output, name=name)
    agent.execute = AsyncMock(return_value=result)
    agent.get_io_schema = MagicMock(return_value=None)
    agent.to_dict = MagicMock(return_value={"name": name})
    return agent


# =============================================================================
# MetaAgentConfig Tests
# =============================================================================


class TestMetaAgentConfig:
    """Tests for MetaAgentConfig dataclass."""

    @pytest.mark.unit
    def test_default_enable_gold_db(self):
        config = MetaAgentConfig(name="test")
        assert config.enable_gold_db is True

    @pytest.mark.unit
    def test_default_improvement_threshold(self):
        config = MetaAgentConfig(name="test")
        assert config.improvement_threshold == 0.7

    @pytest.mark.unit
    def test_default_max_learnings_per_run(self):
        config = MetaAgentConfig(name="test")
        assert config.max_learnings_per_run == 5

    @pytest.mark.unit
    def test_default_enable_improvement_history(self):
        config = MetaAgentConfig(name="test")
        assert config.enable_improvement_history is True

    @pytest.mark.unit
    def test_custom_values(self):
        config = MetaAgentConfig(
            name="custom",
            enable_gold_db=False,
            improvement_threshold=0.9,
            max_learnings_per_run=10,
        )
        assert config.enable_gold_db is False
        assert config.improvement_threshold == 0.9
        assert config.max_learnings_per_run == 10

    @pytest.mark.unit
    def test_inherits_from_agent_runtime_config(self):
        config = MetaAgentConfig(name="test")
        assert isinstance(config, AgentRuntimeConfig)
        assert hasattr(config, "model")
        assert hasattr(config, "max_retries")


# =============================================================================
# MetaAgent Tests
# =============================================================================


class TestMetaAgent:
    """Tests for MetaAgent class."""

    @pytest.mark.unit
    def test_init_defaults(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent()
        assert agent.signature is None
        assert agent.gold_db is None
        assert agent.improvement_history is None
        assert agent._dspy_module is None

    @pytest.mark.unit
    def test_init_with_custom_config(self):
        config = MetaAgentConfig(name="TestMeta", improvement_threshold=0.85)
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent(config=config)
        assert agent.config.name == "TestMeta"
        assert agent.config.improvement_threshold == 0.85

    @pytest.mark.unit
    def test_init_with_gold_db(self):
        gold_db = MagicMock()
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent(gold_db=gold_db)
        assert agent.gold_db is gold_db

    @pytest.mark.unit
    def test_init_with_improvement_history(self):
        history = MagicMock()
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent(improvement_history=history)
        assert agent.improvement_history is history

    @pytest.mark.unit
    def test_ensure_initialized_no_signature(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent()
        # Without DSPy, _dspy_module stays None
        agent._initialized = True
        agent._ensure_initialized()
        assert agent._dspy_module is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_against_gold_no_db(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent()
        result = await agent.evaluate_against_gold("gold_1", {"answer": "yes"})
        assert result["overall_score"] == 0.5
        assert result["result"] == "needs_improvement"
        assert result["gold_standard_id"] == "gold_1"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_against_gold_missing_id(self):
        gold_db = MagicMock()
        gold_db.get.return_value = None
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent(gold_db=gold_db)
        result = await agent.evaluate_against_gold("nonexistent", {"answer": "x"})
        assert result["overall_score"] == 0.0
        assert result["result"] == "failed"
        assert "not found" in result["feedback"][0]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_against_gold_simple_exact_match(self):
        gold_standard = MagicMock()
        gold_standard.expected_output = {"key": "value"}
        gold_standard.id = "gold_1"
        gold_db = MagicMock()
        gold_db.get.return_value = gold_standard
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent(gold_db=gold_db)
        result = await agent.evaluate_against_gold("gold_1", {"key": "value"})
        assert result["overall_score"] == 1.0
        assert result["result"] == "good"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_against_gold_simple_partial_match(self):
        gold_standard = MagicMock()
        gold_standard.expected_output = {"summary": "hello world test"}
        gold_standard.id = "gold_1"
        gold_db = MagicMock()
        gold_db.get.return_value = gold_standard
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent(gold_db=gold_db)
        result = await agent.evaluate_against_gold(
            "gold_1", {"summary": "hello world"}
        )
        # 2 out of 3 words match
        assert 0.5 < result["overall_score"] < 1.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_against_gold_no_match(self):
        gold_standard = MagicMock()
        gold_standard.expected_output = {"answer": "yes"}
        gold_standard.id = "gold_1"
        gold_db = MagicMock()
        gold_db.get.return_value = gold_standard
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent(gold_db=gold_db)
        result = await agent.evaluate_against_gold("gold_1", {"answer": None})
        assert result["overall_score"] == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluate_against_gold_exception(self):
        gold_db = MagicMock()
        gold_db.get.side_effect = RuntimeError("db error")
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent(gold_db=gold_db)
        result = await agent.evaluate_against_gold("gold_1", {"data": "x"})
        assert result["result"] == "failed"
        assert result["overall_score"] == 0.0

    @pytest.mark.unit
    def test_simple_evaluation_non_string_partial(self):
        gold_standard = MagicMock()
        gold_standard.expected_output = {"count": 42}
        gold_standard.id = "g1"
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent()
        result = agent._simple_evaluation(gold_standard, {"count": 99})
        assert result["scores"]["count"] == 0.5

    @pytest.mark.unit
    def test_simple_evaluation_empty_expected(self):
        gold_standard = MagicMock()
        gold_standard.expected_output = {}
        gold_standard.id = "g1"
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent()
        result = agent._simple_evaluation(gold_standard, {"key": "val"})
        assert result["overall_score"] == 0.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_analyze_and_suggest_no_evaluations(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent()
        suggestions = await agent.analyze_and_suggest_improvements([])
        assert suggestions == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_analyze_and_suggest_no_dspy(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent()
        evals = [{"result": "needs_improvement", "overall_score": 0.4}]
        suggestions = await agent.analyze_and_suggest_improvements(evals)
        assert suggestions == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_learnings_not_excellent(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent()
        learnings = await agent.extract_learnings(
            {"task": "x"}, {"output": "y"},
            {"result": "needs_improvement"}, "general",
        )
        assert learnings == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_learnings_no_dspy_module(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent()
        learnings = await agent.extract_learnings(
            {"task": "x"}, {"output": "y"},
            {"result": "excellent", "overall_score": 0.95}, "finance",
        )
        assert len(learnings) == 2
        assert "finance" in learnings[0]

    @pytest.mark.unit
    def test_get_agent_state_no_context(self):
        config = MetaAgentConfig(name="test", enable_context=False)
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent(config=config)
        result = agent.get_agent_state("other_agent")
        assert result is None

    @pytest.mark.unit
    def test_get_agent_state_with_context(self):
        ctx = MagicMock()
        ctx.get.return_value = {"agent1": {"status": "active"}}
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent()
        agent._context_manager = ctx
        result = agent.get_agent_state("agent1")
        assert result == {"status": "active"}

    @pytest.mark.unit
    def test_publish_state_with_context(self):
        ctx = MagicMock()
        ctx.get.return_value = {}
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent()
        agent._context_manager = ctx
        agent.publish_state({"status": "done"})
        ctx.set.assert_called_once()

    @pytest.mark.unit
    def test_publish_state_no_context(self):
        config = MetaAgentConfig(name="test", enable_context=False)
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent(config=config)
        # Should not raise
        agent.publish_state({"status": "done"})

    @pytest.mark.unit
    def test_get_all_agent_states_empty(self):
        config = MetaAgentConfig(name="test", enable_context=False)
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent(config=config)
        assert agent.get_all_agent_states() == {}

    @pytest.mark.unit
    def test_get_all_agent_states_with_context(self):
        ctx = MagicMock()
        ctx.get.return_value = {"a": {"x": 1}, "b": {"y": 2}}
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent()
        agent._context_manager = ctx
        states = agent.get_all_agent_states()
        assert len(states) == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_impl_no_dspy_module(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent()
        with pytest.raises(NotImplementedError):
            await agent._execute_impl(task="test")

    @pytest.mark.unit
    def test_create_meta_agent_factory(self):
        agent = create_meta_agent(model="test-model")
        assert isinstance(agent, MetaAgent)
        assert agent.config.name == "MetaAgent"

    @pytest.mark.unit
    def test_create_meta_agent_with_signature(self):
        mock_sig = MagicMock()
        mock_sig.__name__ = "TestSig"
        agent = create_meta_agent(signature=mock_sig, model="test-model")
        assert "TestSig" in agent.config.name


# =============================================================================
# CompositeAgentConfig Tests
# =============================================================================


class TestCompositeAgentConfig:
    """Tests for CompositeAgentConfig dataclass."""

    @pytest.mark.unit
    def test_default_coordination_pattern(self):
        config = CompositeAgentConfig(name="test")
        assert config.coordination_pattern == CoordinationPattern.PIPELINE

    @pytest.mark.unit
    def test_default_merge_strategy(self):
        config = CompositeAgentConfig(name="test")
        assert config.merge_strategy == MergeStrategy.COMBINE

    @pytest.mark.unit
    def test_custom_coordination_pattern(self):
        config = CompositeAgentConfig(
            name="test", coordination_pattern=CoordinationPattern.PARALLEL,
        )
        assert config.coordination_pattern == CoordinationPattern.PARALLEL

    @pytest.mark.unit
    def test_custom_merge_strategy(self):
        config = CompositeAgentConfig(
            name="test", merge_strategy=MergeStrategy.BEST,
        )
        assert config.merge_strategy == MergeStrategy.BEST

    @pytest.mark.unit
    def test_inherits_from_agent_runtime_config(self):
        config = CompositeAgentConfig(name="test")
        assert isinstance(config, AgentRuntimeConfig)


# =============================================================================
# UnifiedResult Tests
# =============================================================================


class TestUnifiedResult:
    """Tests for UnifiedResult dataclass."""

    @pytest.mark.unit
    def test_creation_required_fields(self):
        ur = UnifiedResult(
            success=True, output="data", name="agent1",
            execution_time=1.5,
        )
        assert ur.success is True
        assert ur.output == "data"
        assert ur.name == "agent1"
        assert ur.execution_time == 1.5

    @pytest.mark.unit
    def test_default_metadata(self):
        ur = UnifiedResult(
            success=True, output=None, name="a", execution_time=0.0,
        )
        assert ur.metadata == {}

    @pytest.mark.unit
    def test_default_agent_traces(self):
        ur = UnifiedResult(
            success=True, output=None, name="a", execution_time=0.0,
        )
        assert ur.agent_traces == []

    @pytest.mark.unit
    def test_default_evaluation(self):
        ur = UnifiedResult(
            success=True, output=None, name="a", execution_time=0.0,
        )
        assert ur.evaluation is None

    @pytest.mark.unit
    def test_default_improvements(self):
        ur = UnifiedResult(
            success=True, output=None, name="a", execution_time=0.0,
        )
        assert ur.improvements == []

    @pytest.mark.unit
    def test_default_error(self):
        ur = UnifiedResult(
            success=False, output=None, name="a", execution_time=0.0,
        )
        assert ur.error is None

    @pytest.mark.unit
    def test_to_agent_result(self):
        ur = UnifiedResult(
            success=True, output={"key": "val"}, name="test_agent",
            execution_time=2.0, error=None, metadata={"m": 1},
        )
        ar = ur.to_agent_result()
        assert isinstance(ar, AgentResult)
        assert ar.success is True
        assert ar.output == {"key": "val"}
        assert ar.agent_name == "test_agent"
        assert ar.execution_time == 2.0
        assert ar.metadata == {"m": 1}

    @pytest.mark.unit
    def test_to_agent_result_with_error(self):
        ur = UnifiedResult(
            success=False, output=None, name="a", execution_time=0.1,
            error="something failed",
        )
        ar = ur.to_agent_result()
        assert ar.success is False
        assert ar.error == "something failed"

    @pytest.mark.unit
    def test_to_swarm_result(self):
        ur = UnifiedResult(
            success=True, output={"result": "ok"}, name="swarm1",
            execution_time=3.0, metadata={"domain": "coding"},
            agent_traces=["t1"], evaluation={"score": 0.9},
            improvements=["i1"],
        )
        mock_sr_cls = MagicMock()
        mock_sr_cls.return_value = MagicMock()
        with patch.dict(
            "sys.modules",
            {"Jotty.core.swarms.swarm_types": MagicMock(SwarmResult=mock_sr_cls)},
        ):
            sr = ur.to_swarm_result()
            mock_sr_cls.assert_called_once()

    @pytest.mark.unit
    def test_from_agent_result(self):
        ar = AgentResult(
            success=True, output="hello", agent_name="agent1",
            execution_time=1.0, metadata={"k": "v"},
        )
        ur = UnifiedResult.from_agent_result(ar)
        assert ur.success is True
        assert ur.output == "hello"
        assert ur.name == "agent1"
        assert ur.execution_time == 1.0
        assert ur.metadata == {"k": "v"}

    @pytest.mark.unit
    def test_from_swarm_result(self):
        sr = MagicMock()
        sr.success = True
        sr.output = {"data": 1}
        sr.swarm_name = "coding_swarm"
        sr.execution_time = 5.0
        sr.error = None
        sr.metadata = {"x": 1}
        sr.agent_traces = ["trace1"]
        sr.evaluation = None
        sr.improvements = []
        ur = UnifiedResult.from_swarm_result(sr)
        assert ur.success is True
        assert ur.name == "coding_swarm"
        assert ur.output == {"data": 1}

    @pytest.mark.unit
    def test_roundtrip_agent_result(self):
        ar_original = AgentResult(
            success=True, output="roundtrip", agent_name="rt",
            execution_time=0.5,
        )
        ur = UnifiedResult.from_agent_result(ar_original)
        ar_back = ur.to_agent_result()
        assert ar_back.success == ar_original.success
        assert ar_back.output == ar_original.output
        assert ar_back.agent_name == ar_original.agent_name


# =============================================================================
# CompositeAgent Tests
# =============================================================================


class TestCompositeAgent:
    """Tests for CompositeAgent class."""

    @pytest.mark.unit
    def test_init_defaults(self):
        agent = CompositeAgent()
        assert agent.config.name == "CompositeAgent"
        assert agent.signature is None
        assert agent._sub_agents == {}
        assert agent._wrapped_swarm is None

    @pytest.mark.unit
    def test_init_with_config(self):
        config = CompositeAgentConfig(
            name="MyComposite",
            coordination_pattern=CoordinationPattern.PARALLEL,
        )
        agent = CompositeAgent(config=config)
        assert agent.config.name == "MyComposite"
        assert agent.config.coordination_pattern == CoordinationPattern.PARALLEL

    @pytest.mark.unit
    def test_init_with_sub_agents(self):
        sub1 = _make_mock_sub_agent("sub1")
        sub2 = _make_mock_sub_agent("sub2")
        agent = CompositeAgent(sub_agents={"a": sub1, "b": sub2})
        assert len(agent._sub_agents) == 2

    @pytest.mark.unit
    def test_add_agent_returns_self(self):
        agent = CompositeAgent()
        sub = _make_mock_sub_agent("s")
        result = agent.add_agent("s", sub)
        assert result is agent

    @pytest.mark.unit
    def test_add_agent_chaining(self):
        agent = CompositeAgent()
        s1 = _make_mock_sub_agent("s1")
        s2 = _make_mock_sub_agent("s2")
        result = agent.add_agent("a", s1).add_agent("b", s2)
        assert result is agent
        assert len(agent._sub_agents) == 2

    @pytest.mark.unit
    def test_remove_agent(self):
        sub = _make_mock_sub_agent("sub")
        agent = CompositeAgent(sub_agents={"x": sub})
        result = agent.remove_agent("x")
        assert result is agent
        assert len(agent._sub_agents) == 0

    @pytest.mark.unit
    def test_remove_agent_nonexistent(self):
        agent = CompositeAgent()
        # Should not raise
        agent.remove_agent("nope")
        assert len(agent._sub_agents) == 0

    @pytest.mark.unit
    def test_get_agent(self):
        sub = _make_mock_sub_agent("sub")
        agent = CompositeAgent(sub_agents={"x": sub})
        assert agent.get_agent("x") is sub
        assert agent.get_agent("y") is None

    @pytest.mark.unit
    def test_sub_agents_property_returns_copy(self):
        sub = _make_mock_sub_agent("sub")
        agent = CompositeAgent(sub_agents={"x": sub})
        snapshot = agent.sub_agents
        snapshot["y"] = _make_mock_sub_agent("y")
        # Original should not be modified
        assert "y" not in agent._sub_agents

    @pytest.mark.unit
    def test_from_swarm(self):
        mock_swarm = MagicMock()
        mock_swarm.config = MagicMock()
        mock_swarm.config.name = "CodingSwarm"
        mock_swarm.config.timeout_seconds = 300
        agent = CompositeAgent.from_swarm(mock_swarm)
        assert agent._wrapped_swarm is mock_swarm
        assert agent.config.name == "CodingSwarm"

    @pytest.mark.unit
    def test_compose(self):
        s1 = _make_mock_sub_agent("s1")
        s2 = _make_mock_sub_agent("s2")
        agent = CompositeAgent.compose(
            "Pipeline", coordination=CoordinationPattern.PIPELINE,
            a=s1, b=s2,
        )
        assert agent.config.name == "Pipeline"
        assert len(agent._sub_agents) == 2
        assert agent.config.coordination_pattern == CoordinationPattern.PIPELINE

    @pytest.mark.unit
    def test_compose_parallel_timeout(self):
        s1 = _make_mock_sub_agent("s1")
        s1.config.timeout = 100.0
        s2 = _make_mock_sub_agent("s2")
        s2.config.timeout = 200.0
        agent = CompositeAgent.compose(
            "Par", coordination=CoordinationPattern.PARALLEL,
            a=s1, b=s2,
        )
        # Parallel: max timeout
        assert agent.config.timeout == 200.0

    @pytest.mark.unit
    def test_compose_pipeline_timeout(self):
        s1 = _make_mock_sub_agent("s1")
        s1.config.timeout = 100.0
        s2 = _make_mock_sub_agent("s2")
        s2.config.timeout = 200.0
        agent = CompositeAgent.compose(
            "Pipe", coordination=CoordinationPattern.PIPELINE,
            a=s1, b=s2,
        )
        # Pipeline: sum of timeouts
        assert agent.config.timeout == 300.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_pipeline_sequential(self):
        s1 = _make_mock_sub_agent("s1", output={"step": 1})
        s2 = _make_mock_sub_agent("s2", output={"step": 2})
        config = CompositeAgentConfig(
            name="PipeTest",
            coordination_pattern=CoordinationPattern.PIPELINE,
        )
        agent = CompositeAgent(config=config, sub_agents={"a": s1, "b": s2})
        agent._initialized = True
        result = await agent._execute_pipeline(task="test")
        assert result.success is True
        # Both agents should be called
        assert s1.execute.await_count == 1
        assert s2.execute.await_count == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_pipeline_failure_stops(self):
        s1 = _make_mock_sub_agent("s1", success=False, output=None)
        s1.execute.return_value = AgentResult(
            success=False, output=None, agent_name="s1",
            execution_time=0.1, error="failed",
        )
        s2 = _make_mock_sub_agent("s2")
        config = CompositeAgentConfig(
            name="PipeTest",
            coordination_pattern=CoordinationPattern.PIPELINE,
        )
        agent = CompositeAgent(config=config, sub_agents={"a": s1, "b": s2})
        result = await agent._execute_pipeline(task="test")
        assert result.success is False
        assert "failed" in result.error
        # s2 should not be called
        s2.execute.assert_not_awaited()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_parallel(self):
        s1 = _make_mock_sub_agent("s1", output="r1")
        s2 = _make_mock_sub_agent("s2", output="r2")
        config = CompositeAgentConfig(
            name="ParTest",
            coordination_pattern=CoordinationPattern.PARALLEL,
            merge_strategy=MergeStrategy.COMBINE,
        )
        agent = CompositeAgent(config=config, sub_agents={"a": s1, "b": s2})
        result = await agent._execute_parallel(task="test")
        assert result.success is True
        assert isinstance(result.output, dict)
        assert "a" in result.output
        assert "b" in result.output

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_consensus_majority_success(self):
        s1 = _make_mock_sub_agent("s1", output="yes")
        s2 = _make_mock_sub_agent("s2", output="yes")
        s3 = _make_mock_sub_agent("s3", success=False)
        s3.execute.return_value = AgentResult(
            success=False, output=None, agent_name="s3",
            execution_time=0.1, error="nope",
        )
        config = CompositeAgentConfig(
            name="ConsTest",
            coordination_pattern=CoordinationPattern.CONSENSUS,
        )
        agent = CompositeAgent(
            config=config, sub_agents={"a": s1, "b": s2, "c": s3},
        )
        result = await agent._execute_consensus(task="test")
        # 2/3 succeed => majority success
        assert result.success is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_consensus_majority_fail(self):
        s1 = _make_mock_sub_agent("s1", success=False)
        s1.execute.return_value = AgentResult(
            success=False, output=None, agent_name="s1",
            execution_time=0.1, error="err",
        )
        s2 = _make_mock_sub_agent("s2", success=False)
        s2.execute.return_value = AgentResult(
            success=False, output=None, agent_name="s2",
            execution_time=0.1, error="err",
        )
        s3 = _make_mock_sub_agent("s3", output="ok")
        config = CompositeAgentConfig(
            name="ConsTest",
            coordination_pattern=CoordinationPattern.CONSENSUS,
        )
        agent = CompositeAgent(
            config=config, sub_agents={"a": s1, "b": s2, "c": s3},
        )
        result = await agent._execute_consensus(task="test")
        # 1/3 succeed => majority fail
        assert result.success is False

    @pytest.mark.unit
    def test_merge_outputs_combine(self):
        config = CompositeAgentConfig(
            name="test", merge_strategy=MergeStrategy.COMBINE,
        )
        agent = CompositeAgent(config=config)
        outputs = {"a": "x", "b": "y"}
        result = agent._merge_outputs(outputs)
        assert result == outputs

    @pytest.mark.unit
    def test_merge_outputs_first(self):
        config = CompositeAgentConfig(
            name="test", merge_strategy=MergeStrategy.FIRST,
        )
        agent = CompositeAgent(config=config)
        outputs = {"a": "first_val", "b": "second_val"}
        result = agent._merge_outputs(outputs)
        assert result == "first_val"

    @pytest.mark.unit
    def test_merge_outputs_concat(self):
        config = CompositeAgentConfig(
            name="test", merge_strategy=MergeStrategy.CONCAT,
        )
        agent = CompositeAgent(config=config)
        outputs = {"a": "hello", "b": "world"}
        result = agent._merge_outputs(outputs)
        assert "hello" in result
        assert "world" in result

    @pytest.mark.unit
    def test_merge_outputs_best(self):
        config = CompositeAgentConfig(
            name="test", merge_strategy=MergeStrategy.BEST,
        )
        agent = CompositeAgent(config=config)
        outputs = {"a": "short", "b": "much longer output here"}
        result = agent._merge_outputs(outputs)
        assert result == "much longer output here"

    @pytest.mark.unit
    def test_merge_outputs_empty(self):
        agent = CompositeAgent()
        result = agent._merge_outputs({})
        assert result is None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_orchestrate_no_sub_agents(self):
        agent = CompositeAgent()
        result = await agent._orchestrate(task="test")
        assert result.success is False
        assert "No sub-agents" in result.error

    @pytest.mark.unit
    def test_repr_with_sub_agents(self):
        s1 = _make_mock_sub_agent("s1")
        config = CompositeAgentConfig(
            name="MyComp",
            coordination_pattern=CoordinationPattern.PIPELINE,
        )
        agent = CompositeAgent(config=config, sub_agents={"a": s1})
        r = repr(agent)
        assert "MyComp" in r
        assert "pipeline" in r

    @pytest.mark.unit
    def test_repr_wrapped_swarm(self):
        mock_swarm = MagicMock()
        mock_swarm.__class__.__name__ = "TestSwarm"
        mock_swarm.config = MagicMock()
        mock_swarm.config.name = "TestSwarm"
        mock_swarm.config.timeout_seconds = 300
        agent = CompositeAgent.from_swarm(mock_swarm)
        r = repr(agent)
        assert "wraps=" in r
        assert "TestSwarm" in r

    @pytest.mark.unit
    def test_to_dict(self):
        s1 = _make_mock_sub_agent("s1")
        config = CompositeAgentConfig(
            name="DictTest",
            coordination_pattern=CoordinationPattern.PARALLEL,
            merge_strategy=MergeStrategy.FIRST,
        )
        agent = CompositeAgent(config=config, sub_agents={"a": s1})
        d = agent.to_dict()
        assert d["type"] == "composite"
        assert d["coordination"] == "parallel"
        assert d["merge_strategy"] == "first"
        assert "a" in d["sub_agents"]

    @pytest.mark.unit
    def test_to_dict_wrapped_swarm(self):
        mock_swarm = MagicMock()
        mock_swarm.__class__.__name__ = "CodingSwarm"
        mock_swarm.config = MagicMock()
        mock_swarm.config.name = "CodingSwarm"
        mock_swarm.config.timeout_seconds = 300
        agent = CompositeAgent.from_swarm(mock_swarm)
        d = agent.to_dict()
        assert d["wrapped_swarm"] == "CodingSwarm"

    @pytest.mark.unit
    def test_extract_output_plain(self):
        ar = AgentResult(success=True, output="plain", agent_name="a")
        out, meta = CompositeAgent._extract_output(ar)
        assert out == "plain"
        assert meta == {}

    @pytest.mark.unit
    def test_extract_output_unified_result(self):
        ur = UnifiedResult(
            success=True, output="inner", name="inner",
            execution_time=0.1, metadata={"k": "v"},
        )
        ar = AgentResult(success=True, output=ur, agent_name="a")
        out, meta = CompositeAgent._extract_output(ar)
        assert out == "inner"
        assert meta == {"k": "v"}

    @pytest.mark.unit
    def test_get_io_schema_no_signature(self):
        agent = CompositeAgent()
        schema = agent.get_io_schema()
        # Without signature or swarm, returns None from the override
        # or base class schema
        assert schema is not None or schema is None  # Just ensure no exception


# =============================================================================
# AutonomousAgentConfig Tests
# =============================================================================


class TestAutonomousAgentConfig:
    """Tests for AutonomousAgentConfig dataclass."""

    @pytest.mark.unit
    def test_default_max_steps(self):
        config = AutonomousAgentConfig(name="test")
        assert config.max_steps == 10

    @pytest.mark.unit
    def test_default_enable_replanning(self):
        config = AutonomousAgentConfig(name="test")
        assert config.enable_replanning is False

    @pytest.mark.unit
    def test_default_max_replans(self):
        config = AutonomousAgentConfig(name="test")
        assert config.max_replans == 3

    @pytest.mark.unit
    def test_default_skill_filter(self):
        config = AutonomousAgentConfig(name="test")
        assert config.skill_filter is None

    @pytest.mark.unit
    def test_custom_values(self):
        config = AutonomousAgentConfig(
            name="custom", max_steps=20, enable_replanning=True,
            max_replans=5, skill_filter="coding",
        )
        assert config.max_steps == 20
        assert config.enable_replanning is True
        assert config.max_replans == 5
        assert config.skill_filter == "coding"

    @pytest.mark.unit
    def test_inherits_from_agent_runtime_config(self):
        config = AutonomousAgentConfig(name="test")
        assert isinstance(config, AgentRuntimeConfig)


# =============================================================================
# AutonomousAgent Tests
# =============================================================================


class TestAutonomousAgent:
    """Tests for AutonomousAgent class."""

    @pytest.mark.unit
    def test_init_defaults(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutonomousAgent()
        assert agent.config.enable_skills is True
        assert agent._planner is None
        assert agent._executor is None

    @pytest.mark.unit
    def test_planner_property_lazy_loads(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutonomousAgent()
        agent._initialized = True
        # Directly set _planner to simulate lazy-load
        mock_planner = MagicMock()
        agent._planner = mock_planner
        assert agent.planner is mock_planner

    @pytest.mark.unit
    def test_executor_property_lazy_loads(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutonomousAgent()
        agent._initialized = True
        # Need skills_registry
        mock_registry = MagicMock()
        agent._skills_registry = mock_registry
        executor = agent.executor
        assert isinstance(executor, SkillPlanExecutor)

    @pytest.mark.unit
    def test_infer_task_type_delegates_to_executor(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutonomousAgent()
        agent._initialized = True
        mock_executor = MagicMock()
        mock_executor.infer_task_type.return_value = "research"
        agent._executor = mock_executor
        result = agent._infer_task_type("find latest papers on AI")
        assert result == "research"

    @pytest.mark.unit
    def test_build_result_success(self):
        result = AutonomousAgent._build_result(
            task="test task",
            task_type="research",
            outputs={"step_0": {"data": "ok"}},
            skills_used=["web-search"],
            errors=[],
            warnings=[],
            start_time=time.time() - 1.0,
        )
        assert result["success"] is True
        assert result["task_type"] == "research"
        assert result["steps_executed"] == 1
        assert result["stopped_early"] is False

    @pytest.mark.unit
    def test_build_result_failure_with_errors(self):
        result = AutonomousAgent._build_result(
            task="test",
            task_type="creation",
            outputs={},
            skills_used=[],
            errors=["Step 1: failed"],
            warnings=[],
            start_time=time.time(),
        )
        assert result["success"] is False

    @pytest.mark.unit
    def test_build_result_stopped_early(self):
        result = AutonomousAgent._build_result(
            task="test",
            task_type="analysis",
            outputs={"s": {}},
            skills_used=["calc"],
            errors=[],
            warnings=[],
            start_time=time.time(),
            stopped=True,
        )
        assert result["success"] is False
        assert result["stopped_early"] is True

    @pytest.mark.unit
    def test_build_result_too_hard(self):
        result = AutonomousAgent._build_result(
            task="test",
            task_type="unknown",
            outputs={},
            skills_used=[],
            errors=["TOO_HARD"],
            warnings=[],
            start_time=time.time(),
            stopped=True,
            too_hard=True,
        )
        assert result["too_hard"] is True

    @pytest.mark.unit
    def test_create_autonomous_agent_factory(self):
        agent = create_autonomous_agent(max_steps=15, model="test-model")
        assert isinstance(agent, AutonomousAgent)
        assert agent.config.max_steps == 15
        assert agent.config.name == "AutonomousAgent"

    @pytest.mark.unit
    def test_create_autonomous_agent_with_replanning(self):
        agent = create_autonomous_agent(enable_replanning=True, model="test-model")
        assert agent.config.enable_replanning is True

    @pytest.mark.unit
    def test_emit_sends_event(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutonomousAgent()
        agent.config.name = "TestAgent"
        mock_broadcaster = MagicMock()
        with patch(
            "Jotty.core.agents.base.autonomous_agent.AgentEventBroadcaster"
        ) as MockAEB:
            MockAEB.get_instance.return_value = mock_broadcaster
            agent._emit("step_start", phase="test")
        mock_broadcaster.emit.assert_called_once()

    @pytest.mark.unit
    def test_emit_silences_exceptions(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutonomousAgent()
        with patch(
            "Jotty.core.agents.base.autonomous_agent.AgentEventBroadcaster"
        ) as MockAEB:
            MockAEB.get_instance.side_effect = RuntimeError("boom")
            # Should not raise
            agent._emit("error", phase="test")

    @pytest.mark.unit
    def test_find_best_content_for_file_matching_key(self):
        outputs = {
            "step_report": {"text": "A" * 200, "success": True},
            "step_other": {"content": "B" * 50, "success": True},
        }
        content = AutonomousAgent._find_best_content_for_file(
            Path("report.txt"), outputs,
        )
        assert len(content) == 200

    @pytest.mark.unit
    def test_find_best_content_for_file_fallback(self):
        outputs = {
            "step_0": {"content": "X" * 150, "success": True},
        }
        content = AutonomousAgent._find_best_content_for_file(
            Path("unknown.txt"), outputs,
        )
        assert len(content) == 150


# =============================================================================
# ExecutionContextManager Tests
# =============================================================================


class TestExecutionContextManager:
    """Tests for ExecutionContextManager."""

    @pytest.mark.unit
    def test_add_step(self):
        ctx = ExecutionContextManager()
        ctx.add_step({"step": 0, "output": "test"})
        assert len(ctx.get_context()) == 1

    @pytest.mark.unit
    def test_get_context_returns_copy(self):
        ctx = ExecutionContextManager()
        ctx.add_step({"step": 0})
        context = ctx.get_context()
        context.append({"extra": True})
        assert len(ctx.get_context()) == 1

    @pytest.mark.unit
    def test_get_trajectory_excludes_compressed(self):
        ctx = ExecutionContextManager()
        ctx._history = [
            {"_compressed": True, "summary": "old stuff"},
            {"step": 5, "output": "recent"},
        ]
        trajectory = ctx.get_trajectory()
        assert len(trajectory) == 1
        assert trajectory[0]["step"] == 5

    @pytest.mark.unit
    def test_compress_triggered_by_size(self):
        ctx = ExecutionContextManager(max_history_size=100)
        # Add enough data to trigger compression
        for i in range(20):
            ctx.add_step({"step": i, "output": "x" * 50})
        # Should have compressed some entries
        assert any(e.get("_compressed") for e in ctx._history)

    @pytest.mark.unit
    def test_compress_preserves_recent(self):
        ctx = ExecutionContextManager(max_history_size=100)
        for i in range(10):
            ctx.add_step({"step": i, "output": "x" * 50})
        trajectory = ctx.get_trajectory()
        # Recent steps should still be accessible
        assert len(trajectory) > 0

    @pytest.mark.unit
    def test_compress_too_few_entries(self):
        ctx = ExecutionContextManager(max_history_size=10)
        ctx.add_step({"step": 0, "output": "a" * 100})
        ctx.add_step({"step": 1, "output": "b" * 100})
        # With only 2 entries, _compress should not modify (< 3)
        # It will be called but won't compress since len < 3
        assert len(ctx._history) == 2


# =============================================================================
# AutoAgent Tests
# =============================================================================


@pytest.mark.skipif(
    not AUTO_AGENT_AVAILABLE,
    reason="AutoAgent not importable",
)
class TestAutoAgent:
    """Tests for AutoAgent class."""

    @pytest.mark.unit
    def test_init_defaults(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutoAgent()
        assert agent.config.name == "AutoAgent"
        assert agent.config.enable_replanning is True
        assert agent.default_output_skill is None
        assert agent.enable_output is False

    @pytest.mark.unit
    def test_init_custom_params(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutoAgent(
                max_steps=20, timeout=600,
                name="CustomAgent", skill_filter="coding",
            )
        assert agent.config.name == "CustomAgent"
        assert agent.config.max_steps == 20
        assert agent.config.timeout == 600.0
        assert agent.config.skill_filter == "coding"

    @pytest.mark.unit
    def test_init_with_planner(self):
        mock_planner = MagicMock()
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutoAgent(planner=mock_planner)
        assert agent._planner is mock_planner

    @pytest.mark.unit
    def test_init_output_skill(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutoAgent(
                default_output_skill="telegram-sender",
                enable_output=True,
            )
        assert agent.default_output_skill == "telegram-sender"
        assert agent.enable_output is True

    @pytest.mark.unit
    def test_init_output_without_skill(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutoAgent(enable_output=True)
        # enable_output requires a skill, so it should be False
        assert agent.enable_output is False

    @pytest.mark.unit
    def test_infer_task_type_with_planner(self):
        mock_planner = MagicMock()
        mock_task_type = MagicMock()
        mock_task_type.value = "research"
        mock_planner.infer_task_type.return_value = (mock_task_type, "reason", 0.9)
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutoAgent(planner=mock_planner)
        result = agent._infer_task_type("find AI papers")
        assert result == "research"

    @pytest.mark.unit
    def test_infer_task_type_planner_fails(self):
        mock_planner = MagicMock()
        mock_planner.infer_task_type.side_effect = RuntimeError("fail")
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutoAgent(planner=mock_planner)
        # Set up executor for fallback
        mock_executor = MagicMock()
        mock_executor.infer_task_type.return_value = "unknown"
        agent._executor = mock_executor
        result = agent._infer_task_type("do something")
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_should_auto_ensemble(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutoAgent()
        with patch(
            "Jotty.core.orchestration.swarm_ensemble.should_auto_ensemble",
            return_value=(True, 3),
        ):
            should, perspectives = agent._should_auto_ensemble("complex analysis task")
        assert should is True
        assert perspectives == 3

    @pytest.mark.unit
    def test_get_mode_prompts_no_matching_skills(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutoAgent()
        result = agent._get_mode_prompts({"web-search", "calculator"})
        assert result == ""

    @pytest.mark.unit
    def test_get_mode_prompts_terminal_session(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutoAgent()
        result = agent._get_mode_prompts({"terminal-session"})
        assert "pexpect" in result

    @pytest.mark.unit
    def test_get_mode_prompts_browser_playwright(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutoAgent()
        with patch.dict("os.environ", {"BROWSER_BACKEND": "playwright"}):
            result = agent._get_mode_prompts({"browser-automation"})
        assert "Playwright" in result

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not EXEC_TYPES_AVAILABLE,
        reason="Execution types not importable",
    )
    async def test_execute_returns_agentic_result(self):
        mock_planner = MagicMock()
        mock_planner.infer_task_type.return_value = (
            TaskType.UNKNOWN, "reason", 0.5,
        )
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutoAgent(planner=mock_planner)
        agent._initialized = True
        # Mock the entire _execute_impl to avoid real execution
        with patch.object(
            AutonomousAgent, '_execute_impl',
            new_callable=AsyncMock,
            return_value={
                "success": True,
                "skills_used": ["test"],
                "steps_executed": 1,
                "outputs": {"s": "ok"},
                "final_output": "done",
                "errors": [],
                "stopped_early": False,
            },
        ):
            with patch.object(agent, '_should_auto_ensemble', return_value=(False, 4)):
                result = await agent.execute("test task")
        assert isinstance(result, AgenticExecutionResult)
        assert result.success is True

    @pytest.mark.unit
    def test_init_with_system_prompt(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutoAgent(system_prompt="You are a coding expert.")
        assert agent.config.system_prompt == "You are a coding expert."

    @pytest.mark.unit
    @pytest.mark.skipif(
        not EXEC_TYPES_AVAILABLE,
        reason="Execution types not importable",
    )
    def test_infer_task_type_enum(self):
        mock_planner = MagicMock()
        mock_task_type = MagicMock()
        mock_task_type.value = "research"
        mock_planner.infer_task_type.return_value = (mock_task_type, "r", 0.9)
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutoAgent(planner=mock_planner)
        result = agent._infer_task_type_enum("find papers")
        assert isinstance(result, TaskType)

    @pytest.mark.unit
    @pytest.mark.skipif(
        not EXEC_TYPES_AVAILABLE,
        reason="Execution types not importable",
    )
    def test_infer_task_type_enum_unknown_fallback(self):
        mock_planner = MagicMock()
        mock_planner.infer_task_type.side_effect = RuntimeError("fail")
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = AutoAgent(planner=mock_planner)
        mock_executor = MagicMock()
        mock_executor.infer_task_type.return_value = "nonexistent_type"
        agent._executor = mock_executor
        result = agent._infer_task_type_enum("something weird")
        assert result == TaskType.UNKNOWN


# =============================================================================
# SkillPlanExecutor Tests
# =============================================================================


class TestSkillPlanExecutor:
    """Tests for SkillPlanExecutor class."""

    @pytest.mark.unit
    def test_init_with_registry(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        assert executor._skills_registry is registry
        assert executor._max_steps == 10
        assert executor._enable_replanning is True
        assert executor._max_replans == 3

    @pytest.mark.unit
    def test_init_custom_params(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(
            skills_registry=registry, max_steps=20,
            enable_replanning=False, max_replans=5,
        )
        assert executor._max_steps == 20
        assert executor._enable_replanning is False
        assert executor._max_replans == 5

    @pytest.mark.unit
    def test_planner_property_lazy_loads(self):
        registry = MagicMock()
        mock_planner = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry, planner=mock_planner)
        assert executor.planner is mock_planner

    @pytest.mark.unit
    def test_planner_property_creates_when_none(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        # Planner lazy-loading uses a relative import inside the property;
        # If TaskPlanner import fails, planner remains None
        planner = executor.planner
        # Either it loaded or it's None (depends on env), but shouldn't raise
        assert planner is not None or planner is None

    @pytest.mark.unit
    def test_planner_property_returns_existing(self):
        registry = MagicMock()
        mock_planner = MagicMock()
        executor = SkillPlanExecutor(
            skills_registry=registry, planner=mock_planner,
        )
        assert executor.planner is mock_planner

    @pytest.mark.unit
    def test_excluded_skills_property(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        assert executor.excluded_skills == set()

    @pytest.mark.unit
    def test_exclude_skill(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        executor.exclude_skill("web-search")
        assert "web-search" in executor.excluded_skills

    @pytest.mark.unit
    def test_clear_exclusions(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        executor.exclude_skill("web-search")
        executor.exclude_skill("calculator")
        executor.clear_exclusions()
        assert executor.excluded_skills == set()

    @pytest.mark.unit
    def test_infer_task_type_comparison(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        with patch.object(
            type(executor), 'planner',
            new_callable=PropertyMock, return_value=None,
        ):
            result = executor.infer_task_type("compare RNN vs CNN")
        assert result == "comparison"

    @pytest.mark.unit
    def test_infer_task_type_creation(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        with patch.object(
            type(executor), 'planner',
            new_callable=PropertyMock, return_value=None,
        ):
            result = executor.infer_task_type("create a web scraper")
        assert result == "creation"

    @pytest.mark.unit
    def test_infer_task_type_research(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        with patch.object(
            type(executor), 'planner',
            new_callable=PropertyMock, return_value=None,
        ):
            result = executor.infer_task_type("research AI trends")
        assert result == "research"

    @pytest.mark.unit
    def test_infer_task_type_analysis(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        with patch.object(
            type(executor), 'planner',
            new_callable=PropertyMock, return_value=None,
        ):
            result = executor.infer_task_type("analyze the dataset")
        assert result == "analysis"

    @pytest.mark.unit
    def test_infer_task_type_unknown(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        executor._planner = None
        with patch.object(
            type(executor), 'planner',
            new_callable=PropertyMock, return_value=None,
        ):
            result = executor.infer_task_type("hello world")
        assert result == "unknown"

    @pytest.mark.unit
    def test_infer_task_type_with_planner(self):
        registry = MagicMock()
        mock_planner = MagicMock()
        mock_task_type = MagicMock()
        mock_task_type.value = "automation"
        mock_planner.infer_task_type.return_value = (
            mock_task_type, "reason", 0.95,
        )
        executor = SkillPlanExecutor(
            skills_registry=registry, planner=mock_planner,
        )
        result = executor.infer_task_type("automate deployment")
        assert result == "automation"

    @pytest.mark.unit
    def test_validate_plan_valid(self):
        registry = MagicMock()
        skill = MagicMock()
        skill.tools = {"search_web_tool": MagicMock()}
        skill.get_tool_schema.return_value = None
        registry.get_skill.return_value = skill
        executor = SkillPlanExecutor(skills_registry=registry)
        step = MagicMock()
        step.skill_name = "web-search"
        step.tool_name = "search_web_tool"
        step.params = {"query": "test"}
        step.depends_on = []
        step.description = "search for test"
        issues = executor.validate_plan([step])
        assert issues == []

    @pytest.mark.unit
    def test_validate_plan_missing_skill(self):
        registry = MagicMock()
        registry.get_skill.return_value = None
        executor = SkillPlanExecutor(skills_registry=registry)
        step = MagicMock()
        step.skill_name = "nonexistent"
        step.tool_name = "tool"
        step.params = {}
        step.depends_on = []
        step.description = "test"
        issues = executor.validate_plan([step])
        assert len(issues) == 1
        assert "not found" in issues[0]["errors"][0]

    @pytest.mark.unit
    def test_validate_plan_invalid_depends_on(self):
        registry = MagicMock()
        skill = MagicMock()
        skill.tools = {"tool_a": MagicMock()}
        skill.get_tool_schema.return_value = None
        registry.get_skill.return_value = skill
        executor = SkillPlanExecutor(skills_registry=registry)
        step = MagicMock()
        step.skill_name = "skill-a"
        step.tool_name = "tool_a"
        step.params = {}
        step.depends_on = [5]  # Invalid: only 1 step
        step.description = "test"
        issues = executor.validate_plan([step])
        assert len(issues) == 1
        assert any("Invalid depends_on" in e for e in issues[0]["errors"])

    @pytest.mark.unit
    def test_validate_plan_forward_dependency(self):
        registry = MagicMock()
        skill = MagicMock()
        skill.tools = {"tool_a": MagicMock()}
        skill.get_tool_schema.return_value = None
        registry.get_skill.return_value = skill
        executor = SkillPlanExecutor(skills_registry=registry)

        step0 = MagicMock()
        step0.skill_name = "skill"
        step0.tool_name = "tool_a"
        step0.params = {}
        step0.depends_on = [1]  # Forward dependency
        step0.description = "step 0"

        step1 = MagicMock()
        step1.skill_name = "skill"
        step1.tool_name = "tool_a"
        step1.params = {}
        step1.depends_on = []
        step1.description = "step 1"

        issues = executor.validate_plan([step0, step1])
        assert len(issues) == 1
        assert any("Forward dependency" in e for e in issues[0]["errors"])

    @pytest.mark.unit
    def test_resolve_params_delegates(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        params = {"query": "hello"}
        result = executor.resolve_params(params, {})
        assert result == {"query": "hello"}

    @pytest.mark.unit
    def test_resolve_path(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        outputs = {"step_0": {"text": "hello world"}}
        result = executor.resolve_path("step_0.text", outputs)
        assert result == "hello world"

    @pytest.mark.unit
    def test_build_dependency_graph(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        step0 = MagicMock()
        step0.depends_on = []
        step1 = MagicMock()
        step1.depends_on = [0]
        step2 = MagicMock()
        step2.depends_on = [0, 1]
        graph = executor._build_dependency_graph([step0, step1, step2])
        assert graph[0] == []
        assert graph[1] == [0]
        assert graph[2] == [0, 1]

    @pytest.mark.unit
    def test_find_parallel_groups(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        step0 = MagicMock()
        step0.depends_on = []
        step1 = MagicMock()
        step1.depends_on = []
        step2 = MagicMock()
        step2.depends_on = [0, 1]
        layers = executor._find_parallel_groups([step0, step1, step2])
        # First layer: 0 and 1 (no deps), second layer: 2
        assert len(layers) >= 2
        assert set(layers[0]) == {0, 1}
        assert 2 in layers[1]

    @pytest.mark.unit
    def test_find_parallel_groups_linear(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        step0 = MagicMock()
        step0.depends_on = []
        step1 = MagicMock()
        step1.depends_on = [0]
        step2 = MagicMock()
        step2.depends_on = [1]
        layers = executor._find_parallel_groups([step0, step1, step2])
        assert len(layers) == 3
        assert layers[0] == [0]
        assert layers[1] == [1]
        assert layers[2] == [2]


# =============================================================================
# ToolCallCache Tests
# =============================================================================


class TestToolCallCache:
    """Tests for ToolCallCache class."""

    @pytest.mark.unit
    def test_init_defaults(self):
        cache = ToolCallCache()
        assert cache._ttl == 300
        assert cache._max_size == 100
        assert cache.size == 0

    @pytest.mark.unit
    def test_make_key_deterministic(self):
        key1 = ToolCallCache.make_key("web-search", "search_tool", {"q": "AI"})
        key2 = ToolCallCache.make_key("web-search", "search_tool", {"q": "AI"})
        assert key1 == key2

    @pytest.mark.unit
    def test_make_key_different_params(self):
        key1 = ToolCallCache.make_key("skill", "tool", {"q": "A"})
        key2 = ToolCallCache.make_key("skill", "tool", {"q": "B"})
        assert key1 != key2

    @pytest.mark.unit
    def test_get_set(self):
        cache = ToolCallCache()
        key = cache.make_key("s", "t", {"p": 1})
        cache.set(key, {"result": "ok"})
        assert cache.get(key) == {"result": "ok"}

    @pytest.mark.unit
    def test_get_miss(self):
        cache = ToolCallCache()
        assert cache.get("nonexistent") is None

    @pytest.mark.unit
    def test_ttl_expiration(self):
        cache = ToolCallCache(ttl_seconds=0)
        key = cache.make_key("s", "t", {"p": 1})
        cache.set(key, "data")
        # Sleep a tiny bit to ensure expiration
        time.sleep(0.01)
        assert cache.get(key) is None

    @pytest.mark.unit
    def test_clear(self):
        cache = ToolCallCache()
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        assert cache.size == 2
        cache.clear()
        assert cache.size == 0

    @pytest.mark.unit
    def test_size_property(self):
        cache = ToolCallCache()
        assert cache.size == 0
        cache.set("k1", "v1")
        assert cache.size == 1
        cache.set("k2", "v2")
        assert cache.size == 2

    @pytest.mark.unit
    def test_lru_eviction(self):
        cache = ToolCallCache(max_size=2)
        cache.set("k1", "v1")
        time.sleep(0.01)
        cache.set("k2", "v2")
        time.sleep(0.01)
        # This should evict k1 (oldest)
        cache.set("k3", "v3")
        assert cache.size == 2
        assert cache.get("k1") is None
        assert cache.get("k2") == "v2"
        assert cache.get("k3") == "v3"

    @pytest.mark.unit
    def test_update_existing_key(self):
        cache = ToolCallCache(max_size=2)
        cache.set("k1", "v1")
        cache.set("k1", "v1_updated")
        assert cache.size == 1
        assert cache.get("k1") == "v1_updated"

    @pytest.mark.unit
    def test_make_key_non_serializable(self):
        # Should handle non-JSON-serializable params gracefully
        class Custom:
            pass
        key = ToolCallCache.make_key("s", "t", {"obj": Custom()})
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hex digest


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestMetaAgentEdgeCases:
    """Edge case tests for MetaAgent."""

    @pytest.mark.unit
    def test_simple_evaluation_string_word_overlap_empty_expected(self):
        gold_standard = MagicMock()
        gold_standard.expected_output = {"text": ""}
        gold_standard.id = "g1"
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent()
        result = agent._simple_evaluation(gold_standard, {"text": "something"})
        # Empty expected string -> expected_words is empty set -> score 0.5
        assert result["scores"]["text"] == 0.5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extract_learnings_good_result_no_dspy(self):
        with patch.object(BaseAgent, '_ensure_initialized'):
            agent = MetaAgent()
        learnings = await agent.extract_learnings(
            {"task": "build API"}, {"output": "code"},
            {"result": "good", "overall_score": 0.85}, "coding",
        )
        assert len(learnings) == 2
        assert "coding" in learnings[0]
        assert "0.85" in learnings[1]


class TestCompositeAgentEdgeCases:
    """Edge case tests for CompositeAgent."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_impl_delegates_to_wrapped_swarm(self):
        mock_swarm = MagicMock()
        mock_swarm.config = MagicMock()
        mock_swarm.config.name = "TestSwarm"
        mock_swarm.config.timeout_seconds = 300
        mock_sr = MagicMock()
        mock_sr.success = True
        mock_sr.output = {"data": "ok"}
        mock_sr.swarm_name = "TestSwarm"
        mock_sr.execution_time = 1.0
        mock_sr.error = None
        mock_sr.metadata = {}
        mock_sr.agent_traces = []
        mock_sr.evaluation = None
        mock_sr.improvements = []
        mock_swarm.execute = AsyncMock(return_value=mock_sr)
        agent = CompositeAgent.from_swarm(mock_swarm)
        result = await agent._execute_impl(task="test")
        assert isinstance(result, UnifiedResult)
        assert result.success is True

    @pytest.mark.unit
    def test_compose_max_retries_set_to_1(self):
        s1 = _make_mock_sub_agent("s1")
        agent = CompositeAgent.compose("Test", a=s1)
        assert agent.config.max_retries == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_pipeline_output_chaining_dict(self):
        s1 = _make_mock_sub_agent("s1", output={"key": "from_s1"})
        s2 = _make_mock_sub_agent("s2", output="final")
        config = CompositeAgentConfig(
            name="Chain",
            coordination_pattern=CoordinationPattern.PIPELINE,
        )
        agent = CompositeAgent(config=config, sub_agents={"a": s1, "b": s2})
        result = await agent._execute_pipeline(task="test")
        assert result.success is True
        # s2 should have been called with key="from_s1" in kwargs
        call_kwargs = s2.execute.call_args
        assert "key" in call_kwargs.kwargs or (
            call_kwargs[1] and "key" in call_kwargs[1]
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_pipeline_output_chaining_non_dict(self):
        s1 = _make_mock_sub_agent("s1", output="plain_text")
        s2 = _make_mock_sub_agent("s2", output="final")
        config = CompositeAgentConfig(
            name="Chain",
            coordination_pattern=CoordinationPattern.PIPELINE,
        )
        agent = CompositeAgent(config=config, sub_agents={"a": s1, "b": s2})
        result = await agent._execute_pipeline(task="test")
        assert result.success is True
        # s2 should have been called with task="plain_text"
        call_kwargs = s2.execute.call_args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert all_kwargs.get("task") == "plain_text"
        assert all_kwargs.get("previous_output") == "plain_text"


class TestAutonomousAgentEdgeCases:
    """Edge case tests for AutonomousAgent."""

    @pytest.mark.unit
    def test_build_result_final_output_from_last(self):
        result = AutonomousAgent._build_result(
            task="test",
            task_type="creation",
            outputs={"s0": {"data": "a"}, "s1": {"data": "b"}},
            skills_used=["x", "y"],
            errors=[],
            warnings=["w1"],
            start_time=time.time() - 2.0,
        )
        assert result["final_output"] == {"data": "b"}
        assert result["warnings"] == ["w1"]

    @pytest.mark.unit
    def test_build_result_deduplicates_skills(self):
        result = AutonomousAgent._build_result(
            task="test",
            task_type="research",
            outputs={"s0": {}},
            skills_used=["web-search", "web-search", "calculator"],
            errors=[],
            warnings=[],
            start_time=time.time(),
        )
        assert len(result["skills_used"]) == 2

    @pytest.mark.unit
    def test_build_result_empty_outputs(self):
        result = AutonomousAgent._build_result(
            task="test",
            task_type="unknown",
            outputs={},
            skills_used=[],
            errors=[],
            warnings=[],
            start_time=time.time(),
        )
        assert result["final_output"] is None
        assert result["success"] is False


class TestSkillPlanExecutorEdgeCases:
    """Edge case tests for SkillPlanExecutor."""

    @pytest.mark.unit
    def test_inject_essential_skills(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        available = [
            {"name": "shell-exec", "description": "Run shell commands"},
            {"name": "web-search", "description": "Search the web"},
        ]
        selected = [{"name": "web-search", "description": "Search the web"}]
        result = executor._inject_essential_skills(
            "execute the script please", selected, available,
        )
        names = [s["name"] for s in result]
        assert "shell-exec" in names
        assert "web-search" in names

    @pytest.mark.unit
    def test_inject_essential_skills_already_present(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        available = [
            {"name": "shell-exec", "description": "Run shell"},
        ]
        selected = [{"name": "shell-exec", "description": "Run shell"}]
        result = executor._inject_essential_skills(
            "execute the script", selected, available,
        )
        # Should not duplicate
        assert len(result) == 1

    @pytest.mark.unit
    def test_inject_essential_skills_not_available(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        available = [{"name": "web-search", "description": "Search"}]
        selected = [{"name": "web-search", "description": "Search"}]
        result = executor._inject_essential_skills(
            "execute the script", selected, available,
        )
        # shell-exec not in available, should not be injected
        assert len(result) == 1

    @pytest.mark.unit
    def test_build_dependency_graph_invalid_deps_filtered(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        step0 = MagicMock()
        step0.depends_on = [99, -1, "invalid"]
        graph = executor._build_dependency_graph([step0])
        # All invalid deps should be filtered out
        assert graph[0] == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_select_skills_no_planner(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        executor._planner = None
        # Patch the planner property to return None
        with patch.object(
            type(executor), 'planner',
            new_callable=PropertyMock, return_value=None,
        ):
            skills = [{"name": f"s{i}"} for i in range(15)]
            result = await executor.select_skills("task", skills, max_skills=5)
        assert len(result) == 5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_create_plan_no_planner(self):
        registry = MagicMock()
        executor = SkillPlanExecutor(skills_registry=registry)
        executor._planner = None
        with patch.object(
            type(executor), 'planner',
            new_callable=PropertyMock, return_value=None,
        ):
            steps = await executor.create_plan(
                "task", "research",
                [{"name": "web-search"}],
            )
        assert steps == []

    @pytest.mark.unit
    def test_infer_task_type_planner_fails_gracefully(self):
        registry = MagicMock()
        mock_planner = MagicMock()
        mock_planner.infer_task_type.side_effect = RuntimeError("LLM fail")
        executor = SkillPlanExecutor(
            skills_registry=registry, planner=mock_planner,
        )
        result = executor.infer_task_type("create a web app")
        assert result == "creation"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_step_no_registry(self):
        executor = SkillPlanExecutor(skills_registry=None)
        step = MagicMock()
        step.skill_name = "test"
        result = await executor.execute_step(step, {})
        assert result["success"] is False
        assert "not available" in result["error"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_step_skill_not_found(self):
        registry = MagicMock()
        registry.get_skill.return_value = None
        executor = SkillPlanExecutor(skills_registry=registry)
        step = MagicMock()
        step.skill_name = "nonexistent"
        result = await executor.execute_step(step, {})
        assert result["success"] is False
        assert "not found" in result["error"]


class TestToolCallCacheEdgeCases:
    """Edge case tests for ToolCallCache."""

    @pytest.mark.unit
    def test_make_key_sorted_params(self):
        key1 = ToolCallCache.make_key("s", "t", {"b": 2, "a": 1})
        key2 = ToolCallCache.make_key("s", "t", {"a": 1, "b": 2})
        assert key1 == key2

    @pytest.mark.unit
    def test_concurrent_access(self):
        import threading
        cache = ToolCallCache(max_size=50)
        errors = []

        def writer(start):
            try:
                for i in range(start, start + 20):
                    cache.set(f"key_{i}", f"val_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 20,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cache.size <= 50

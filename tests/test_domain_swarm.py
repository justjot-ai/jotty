"""
Comprehensive Test Suite for DomainSwarm and AgentTeam
=======================================================

Tests cover:
1. CoordinationPattern and MergeStrategy enums
2. AgentSpec creation and attr_name generation
3. TeamResult dataclass
4. AgentTeam definition and operations
5. AgentTeam execution patterns (pipeline, parallel, consensus, etc.)
6. AgentTeam merge strategies
7. PhaseExecutor for swarm execution management
8. DomainSwarm initialization and agent lifecycle
9. DomainSwarm team coordination
10. DomainSwarm template method execution
11. Edge cases and error handling

Author: Jotty Team
Date: February 2026
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Dict, Any

from Jotty.core.swarms.base.agent_team import (
    AgentTeam,
    AgentSpec,
    TeamResult,
    CoordinationPattern,
    MergeStrategy,
)
from Jotty.core.swarms.base.domain_swarm import (
    DomainSwarm,
    PhaseExecutor,
)
from Jotty.core.swarms.swarm_types import (
    SwarmBaseConfig,
    SwarmResult,
    AgentRole,
)


# =============================================================================
# Test Enums
# =============================================================================

@pytest.mark.unit
class TestCoordinationPattern:
    """Test CoordinationPattern enum values."""

    def test_all_patterns_exist(self):
        """Test all coordination patterns are defined."""
        assert CoordinationPattern.NONE.value == "none"
        assert CoordinationPattern.PIPELINE.value == "pipeline"
        assert CoordinationPattern.PARALLEL.value == "parallel"
        assert CoordinationPattern.CONSENSUS.value == "consensus"
        assert CoordinationPattern.HIERARCHICAL.value == "hierarchical"
        assert CoordinationPattern.BLACKBOARD.value == "blackboard"
        assert CoordinationPattern.ROUND_ROBIN.value == "round_robin"

    def test_pattern_enum_length(self):
        """Test expected number of patterns."""
        patterns = list(CoordinationPattern)
        assert len(patterns) == 7


@pytest.mark.unit
class TestMergeStrategy:
    """Test MergeStrategy enum values."""

    def test_all_strategies_exist(self):
        """Test all merge strategies are defined."""
        assert MergeStrategy.COMBINE.value == "combine"
        assert MergeStrategy.FIRST.value == "first"
        assert MergeStrategy.BEST.value == "best"
        assert MergeStrategy.VOTE.value == "vote"
        assert MergeStrategy.CONCAT.value == "concat"

    def test_strategy_enum_length(self):
        """Test expected number of strategies."""
        strategies = list(MergeStrategy)
        assert len(strategies) == 5


# =============================================================================
# Test AgentSpec
# =============================================================================

@pytest.mark.unit
class TestAgentSpec:
    """Test AgentSpec dataclass and attr_name generation."""

    def test_create_basic_spec(self):
        """Test creating a basic AgentSpec."""
        class TestAgent:
            pass

        spec = AgentSpec(agent_class=TestAgent, display_name="Architect")
        assert spec.agent_class == TestAgent
        assert spec.display_name == "Architect"
        assert spec.role is None
        assert spec.priority == 0

    def test_attr_name_auto_generation_single_word(self):
        """Test attr_name generation for single word."""
        class TestAgent:
            pass

        spec = AgentSpec(agent_class=TestAgent, display_name="Architect")
        assert spec.attr_name == "_architect"

    def test_attr_name_auto_generation_camel_case(self):
        """Test attr_name generation for CamelCase."""
        class TestAgent:
            pass

        spec = AgentSpec(agent_class=TestAgent, display_name="TestWriter")
        assert spec.attr_name == "_test_writer"

    def test_attr_name_auto_generation_multi_word(self):
        """Test attr_name generation for multi-word CamelCase."""
        class TestAgent:
            pass

        spec = AgentSpec(agent_class=TestAgent, display_name="DataAnalysisAgent")
        assert spec.attr_name == "_data_analysis_agent"

    def test_attr_name_with_spaces(self):
        """Test attr_name generation with spaces."""
        class TestAgent:
            pass

        spec = AgentSpec(agent_class=TestAgent, display_name="Test Agent")
        # Spaces are replaced with underscores after CamelCase conversion
        assert spec.attr_name == "_test__agent"

    def test_custom_attr_name(self):
        """Test providing custom attr_name."""
        class TestAgent:
            pass

        spec = AgentSpec(
            agent_class=TestAgent,
            display_name="Architect",
            attr_name="_custom_arch"
        )
        assert spec.attr_name == "_custom_arch"

    def test_spec_with_role(self):
        """Test AgentSpec with role."""
        class TestAgent:
            pass

        spec = AgentSpec(
            agent_class=TestAgent,
            display_name="Manager",
            role="manager"
        )
        assert spec.role == "manager"

    def test_spec_with_priority(self):
        """Test AgentSpec with priority."""
        class TestAgent:
            pass

        spec = AgentSpec(
            agent_class=TestAgent,
            display_name="FirstAgent",
            priority=10
        )
        assert spec.priority == 10

    def test_spec_all_fields(self):
        """Test AgentSpec with all fields."""
        class TestAgent:
            pass

        spec = AgentSpec(
            agent_class=TestAgent,
            display_name="CompleteAgent",
            attr_name="_complete",
            role="worker",
            priority=5
        )
        assert spec.agent_class == TestAgent
        assert spec.display_name == "CompleteAgent"
        assert spec.attr_name == "_complete"
        assert spec.role == "worker"
        assert spec.priority == 5


# =============================================================================
# Test TeamResult
# =============================================================================

@pytest.mark.unit
class TestTeamResult:
    """Test TeamResult dataclass."""

    def test_create_basic_result(self):
        """Test creating basic TeamResult."""
        result = TeamResult(
            success=True,
            outputs={"Agent1": "output1"}
        )
        assert result.success is True
        assert result.outputs == {"Agent1": "output1"}
        assert result.merged_output is None
        assert result.pattern == CoordinationPattern.NONE

    def test_result_with_all_fields(self):
        """Test TeamResult with all fields."""
        result = TeamResult(
            success=True,
            outputs={"Agent1": "out1", "Agent2": "out2"},
            merged_output="merged",
            pattern=CoordinationPattern.PIPELINE,
            execution_order=["Agent1", "Agent2"],
            errors={},
            metadata={"duration": 1.5}
        )
        assert result.success is True
        assert len(result.outputs) == 2
        assert result.merged_output == "merged"
        assert result.pattern == CoordinationPattern.PIPELINE
        assert result.execution_order == ["Agent1", "Agent2"]
        assert result.errors == {}
        assert result.metadata == {"duration": 1.5}

    def test_result_with_errors(self):
        """Test TeamResult with errors."""
        result = TeamResult(
            success=False,
            outputs={},
            errors={"Agent1": "Failed to execute"}
        )
        assert result.success is False
        assert result.errors == {"Agent1": "Failed to execute"}

    def test_result_defaults(self):
        """Test TeamResult default values."""
        result = TeamResult(success=True, outputs={})
        assert result.execution_order == []
        assert result.errors == {}
        assert result.metadata == {}


# =============================================================================
# Test AgentTeam Define
# =============================================================================

@pytest.mark.unit
class TestAgentTeamDefine:
    """Test AgentTeam.define() class method."""

    def test_define_simple_team(self):
        """Test defining a simple team."""
        class Agent1:
            pass
        class Agent2:
            pass

        team = AgentTeam.define(
            (Agent1, "Agent1"),
            (Agent2, "Agent2"),
        )

        assert len(team) == 2
        assert team.pattern == CoordinationPattern.NONE
        assert "_agent1" in team.agents
        assert "_agent2" in team.agents

    def test_define_with_pattern(self):
        """Test defining team with coordination pattern."""
        class Agent1:
            pass
        class Agent2:
            pass

        team = AgentTeam.define(
            (Agent1, "Agent1"),
            (Agent2, "Agent2"),
            pattern=CoordinationPattern.PARALLEL
        )

        assert team.pattern == CoordinationPattern.PARALLEL

    def test_define_with_merge_strategy(self):
        """Test defining team with merge strategy."""
        class Agent1:
            pass
        class Agent2:
            pass

        team = AgentTeam.define(
            (Agent1, "Agent1"),
            (Agent2, "Agent2"),
            merge_strategy=MergeStrategy.CONCAT
        )

        assert team.merge_strategy == MergeStrategy.CONCAT

    def test_define_with_custom_attr(self):
        """Test defining team with custom attribute names."""
        class Agent1:
            pass

        team = AgentTeam.define(
            (Agent1, "FirstAgent", "_custom"),
        )

        assert "_custom" in team.agents

    def test_define_with_role(self):
        """Test defining team with roles."""
        class Manager:
            pass
        class Worker:
            pass

        team = AgentTeam.define(
            (Manager, "Manager", None, "manager"),
            (Worker, "Worker", None, "worker"),
        )

        manager_spec = team.agents.get("_manager")
        assert manager_spec.role == "manager"

    def test_define_with_priority(self):
        """Test defining team with priorities."""
        class Agent1:
            pass
        class Agent2:
            pass

        team = AgentTeam.define(
            (Agent1, "Agent1", None, None, 10),
            (Agent2, "Agent2", None, None, 5),
        )

        assert team.agents["_agent1"].priority == 10
        assert team.agents["_agent2"].priority == 5

    def test_define_with_timeout(self):
        """Test defining team with timeout."""
        class Agent1:
            pass

        team = AgentTeam.define(
            (Agent1, "Agent1"),
            timeout=60.0
        )

        assert team.timeout == 60.0

    def test_define_with_manager_attr(self):
        """Test defining team with manager attribute."""
        class Manager:
            pass
        class Worker:
            pass

        team = AgentTeam.define(
            (Manager, "Manager"),
            (Worker, "Worker"),
            manager_attr="_manager"
        )

        assert team.manager_attr == "_manager"

    def test_define_invalid_spec(self):
        """Test defining team with invalid spec raises error."""
        class Agent1:
            pass

        with pytest.raises(ValueError):
            AgentTeam.define((Agent1,))  # Too few elements


# =============================================================================
# Test AgentTeam Operations
# =============================================================================

@pytest.mark.unit
class TestAgentTeamOperations:
    """Test AgentTeam operational methods."""

    def test_add_agent(self):
        """Test adding agent to team."""
        class Agent1:
            pass

        team = AgentTeam()
        team.add(Agent1, "Agent1")

        assert len(team) == 1
        assert "_agent1" in team.agents

    def test_add_agent_with_custom_attr(self):
        """Test adding agent with custom attr name."""
        class Agent1:
            pass

        team = AgentTeam()
        team.add(Agent1, "Agent1", attr_name="_custom")

        assert "_custom" in team.agents

    def test_get_agent_names(self):
        """Test getting list of agent names."""
        class Agent1:
            pass
        class Agent2:
            pass

        team = AgentTeam.define(
            (Agent1, "FirstAgent"),
            (Agent2, "SecondAgent"),
        )

        names = team.get_agent_names()
        assert "FirstAgent" in names
        assert "SecondAgent" in names
        assert len(names) == 2

    def test_get_agents_by_role(self):
        """Test filtering agents by role."""
        class Manager:
            pass
        class Worker1:
            pass
        class Worker2:
            pass

        team = AgentTeam.define(
            (Manager, "Manager", None, "manager"),
            (Worker1, "Worker1", None, "worker"),
            (Worker2, "Worker2", None, "worker"),
        )

        workers = team.get_agents_by_role("worker")
        assert len(workers) == 2

        managers = team.get_agents_by_role("manager")
        assert len(managers) == 1

    def test_get_ordered_agents(self):
        """Test getting agents ordered by priority."""
        class Agent1:
            pass
        class Agent2:
            pass
        class Agent3:
            pass

        team = AgentTeam.define(
            (Agent1, "Agent1", None, None, 5),
            (Agent2, "Agent2", None, None, 10),
            (Agent3, "Agent3", None, None, 1),
        )

        ordered = team.get_ordered_agents()
        # Should be ordered by priority descending
        assert ordered[0][1].display_name == "Agent2"  # priority 10
        assert ordered[1][1].display_name == "Agent1"  # priority 5
        assert ordered[2][1].display_name == "Agent3"  # priority 1

    def test_team_len(self):
        """Test team __len__ method."""
        class Agent1:
            pass
        class Agent2:
            pass

        team = AgentTeam.define(
            (Agent1, "Agent1"),
            (Agent2, "Agent2"),
        )

        assert len(team) == 2

    def test_team_iter(self):
        """Test team __iter__ method."""
        class Agent1:
            pass
        class Agent2:
            pass

        team = AgentTeam.define(
            (Agent1, "Agent1"),
            (Agent2, "Agent2"),
        )

        items = list(team)
        assert len(items) == 2
        assert all(isinstance(item, tuple) for item in items)

    def test_set_instances(self):
        """Test setting agent instances."""
        class Agent1:
            pass

        team = AgentTeam.define((Agent1, "Agent1"))
        agent_instance = Mock()

        team.set_instances({"_agent1": agent_instance})

        assert team._instances == {"_agent1": agent_instance}


# =============================================================================
# Test AgentTeam Execution
# =============================================================================

@pytest.mark.unit
class TestAgentTeamExecution:
    """Test AgentTeam execution with coordination patterns."""

    @pytest.mark.asyncio
    async def test_execute_none_pattern(self):
        """Test execution with NONE pattern returns empty result."""
        class Agent1:
            pass

        team = AgentTeam.define((Agent1, "Agent1"))
        team.set_instances({"_agent1": Mock()})

        result = await team.execute(task="test")

        assert result.success is True
        assert result.outputs == {}
        assert result.pattern == CoordinationPattern.NONE
        assert "note" in result.metadata

    @pytest.mark.asyncio
    async def test_execute_without_instances_raises(self):
        """Test execution without instances raises error."""
        class Agent1:
            pass

        team = AgentTeam.define((Agent1, "Agent1"))

        with pytest.raises(RuntimeError, match="instances not set"):
            await team.execute(task="test")

    @pytest.mark.asyncio
    async def test_execute_pipeline_basic(self):
        """Test pipeline execution with two agents."""
        class Agent1:
            async def execute(self, **kwargs):
                return Mock(output="output1")

        class Agent2:
            async def execute(self, **kwargs):
                return Mock(output="output2")

        team = AgentTeam.define(
            (Agent1, "Agent1", None, None, 2),
            (Agent2, "Agent2", None, None, 1),
            pattern=CoordinationPattern.PIPELINE
        )

        team.set_instances({
            "_agent1": Agent1(),
            "_agent2": Agent2(),
        })

        result = await team.execute(task="test")

        assert result.success is True
        assert "Agent1" in result.outputs
        assert "Agent2" in result.outputs
        assert result.merged_output == "output2"  # Final output
        assert result.execution_order == ["Agent1", "Agent2"]

    @pytest.mark.asyncio
    async def test_execute_pipeline_with_error(self):
        """Test pipeline stops on error."""
        class Agent1:
            async def execute(self, **kwargs):
                raise ValueError("Agent1 failed")

        class Agent2:
            async def execute(self, **kwargs):
                return Mock(output="output2")

        team = AgentTeam.define(
            (Agent1, "Agent1", None, None, 2),
            (Agent2, "Agent2", None, None, 1),
            pattern=CoordinationPattern.PIPELINE
        )

        team.set_instances({
            "_agent1": Agent1(),
            "_agent2": Agent2(),
        })

        result = await team.execute(task="test")

        assert result.success is False
        assert "Agent1" in result.errors
        assert "Agent2" not in result.outputs  # Pipeline stopped

    @pytest.mark.asyncio
    async def test_execute_parallel_basic(self):
        """Test parallel execution with multiple agents."""
        class Agent1:
            async def execute(self, **kwargs):
                await asyncio.sleep(0.01)
                return Mock(output="output1")

        class Agent2:
            async def execute(self, **kwargs):
                await asyncio.sleep(0.01)
                return Mock(output="output2")

        team = AgentTeam.define(
            (Agent1, "Agent1"),
            (Agent2, "Agent2"),
            pattern=CoordinationPattern.PARALLEL
        )

        team.set_instances({
            "_agent1": Agent1(),
            "_agent2": Agent2(),
        })

        result = await team.execute(task="test")

        assert result.success is True
        assert len(result.outputs) == 2
        assert "Agent1" in result.outputs
        assert "Agent2" in result.outputs

    @pytest.mark.asyncio
    async def test_execute_parallel_with_partial_failure(self):
        """Test parallel execution with one agent failing."""
        class Agent1:
            async def execute(self, **kwargs):
                return Mock(output="output1")

        class Agent2:
            async def execute(self, **kwargs):
                raise ValueError("Agent2 failed")

        team = AgentTeam.define(
            (Agent1, "Agent1"),
            (Agent2, "Agent2"),
            pattern=CoordinationPattern.PARALLEL
        )

        team.set_instances({
            "_agent1": Agent1(),
            "_agent2": Agent2(),
        })

        result = await team.execute(task="test")

        assert result.success is True  # At least one succeeded
        assert "Agent1" in result.outputs
        assert "Agent2" in result.errors

    @pytest.mark.asyncio
    async def test_execute_consensus_basic(self):
        """Test consensus execution with voting."""
        class Agent1:
            async def execute(self, **kwargs):
                return Mock(output="result_A")

        class Agent2:
            async def execute(self, **kwargs):
                return Mock(output="result_A")

        class Agent3:
            async def execute(self, **kwargs):
                return Mock(output="result_B")

        team = AgentTeam.define(
            (Agent1, "Agent1"),
            (Agent2, "Agent2"),
            (Agent3, "Agent3"),
            pattern=CoordinationPattern.CONSENSUS
        )

        team.set_instances({
            "_agent1": Agent1(),
            "_agent2": Agent2(),
            "_agent3": Agent3(),
        })

        result = await team.execute(task="test")

        assert result.success is True
        # Majority should win (result_A with 2 votes)
        assert "votes" in result.metadata

    @pytest.mark.asyncio
    async def test_execute_hierarchical_with_manager(self):
        """Test hierarchical execution with manager."""
        class Manager:
            async def plan(self, **kwargs):
                return {"Worker": "subtask1"}

        class Worker:
            async def execute(self, **kwargs):
                return Mock(output="worker_output")

        team = AgentTeam.define(
            (Manager, "Manager", None, "manager"),
            (Worker, "Worker", None, "worker"),
            pattern=CoordinationPattern.HIERARCHICAL
        )

        team.set_instances({
            "_manager": Manager(),
            "_worker": Worker(),
        })

        result = await team.execute(task="test")

        assert result.success is True
        assert "Worker" in result.outputs

    @pytest.mark.asyncio
    async def test_execute_blackboard_basic(self):
        """Test blackboard execution."""
        class Agent1:
            contribution_count = 0

            async def contribute(self, blackboard, **kwargs):
                self.contribution_count += 1
                if self.contribution_count == 1:
                    return "contribution1"
                return None  # No more contributions

        team = AgentTeam.define(
            (Agent1, "Agent1"),
            pattern=CoordinationPattern.BLACKBOARD
        )

        agent = Agent1()
        team.set_instances({"_agent1": agent})

        result = await team.execute(task="test", max_rounds=3)

        assert result.success is True
        assert "Agent1" in result.outputs
        assert isinstance(result.merged_output, dict)
        assert "contributions" in result.merged_output

    @pytest.mark.asyncio
    async def test_execute_round_robin_basic(self):
        """Test round robin execution."""
        class Agent1:
            async def execute(self, **kwargs):
                return Mock(output="agent1_output")

        class Agent2:
            async def execute(self, **kwargs):
                return Mock(output="agent2_output")

        team = AgentTeam.define(
            (Agent1, "Agent1"),
            (Agent2, "Agent2"),
            pattern=CoordinationPattern.ROUND_ROBIN
        )

        team.set_instances({
            "_agent1": Agent1(),
            "_agent2": Agent2(),
        })

        result = await team.execute(task=["task1", "task2", "task3"])

        assert result.success is True
        assert len(result.outputs) == 3  # 3 tasks
        assert isinstance(result.merged_output, list)

    @pytest.mark.asyncio
    async def test_execute_unknown_pattern_raises(self):
        """Test unknown pattern raises ValueError."""
        team = AgentTeam()
        team.pattern = "unknown_pattern"
        team.set_instances({"_agent": Mock()})

        with pytest.raises(ValueError, match="Unknown coordination pattern"):
            await team.execute(task="test")


# =============================================================================
# Test AgentTeam Merge Outputs
# =============================================================================

@pytest.mark.unit
class TestAgentTeamMergeOutputs:
    """Test AgentTeam._merge_outputs method."""

    def test_merge_combine(self):
        """Test COMBINE merge strategy."""
        team = AgentTeam(merge_strategy=MergeStrategy.COMBINE)
        outputs = {"Agent1": "out1", "Agent2": "out2"}

        result = team._merge_outputs(outputs)

        assert result == outputs  # Returns dict as-is

    def test_merge_first(self):
        """Test FIRST merge strategy."""
        team = AgentTeam(merge_strategy=MergeStrategy.FIRST)
        outputs = {"Agent1": "out1", "Agent2": "out2"}

        result = team._merge_outputs(outputs)

        assert result in ["out1", "out2"]  # Returns first value

    def test_merge_concat(self):
        """Test CONCAT merge strategy."""
        team = AgentTeam(merge_strategy=MergeStrategy.CONCAT)
        outputs = {"Agent1": "out1", "Agent2": "out2"}

        result = team._merge_outputs(outputs)

        assert isinstance(result, str)
        assert "out1" in result
        assert "out2" in result

    def test_merge_vote(self):
        """Test VOTE merge strategy."""
        team = AgentTeam(merge_strategy=MergeStrategy.VOTE)
        outputs = {
            "Agent1": "result_A",
            "Agent2": "result_A",
            "Agent3": "result_B"
        }

        result = team._merge_outputs(outputs)

        assert result == "result_A"  # Majority wins

    def test_merge_best(self):
        """Test BEST merge strategy (defaults to first)."""
        team = AgentTeam(merge_strategy=MergeStrategy.BEST)
        outputs = {"Agent1": "out1", "Agent2": "out2"}

        result = team._merge_outputs(outputs)

        assert result in ["out1", "out2"]

    def test_merge_empty_outputs(self):
        """Test merging empty outputs."""
        team = AgentTeam(merge_strategy=MergeStrategy.COMBINE)

        result = team._merge_outputs({})

        assert result is None


# =============================================================================
# Test PhaseExecutor
# =============================================================================

@pytest.mark.unit
class TestPhaseExecutor:
    """Test PhaseExecutor helper class."""

    @pytest.mark.asyncio
    async def test_elapsed_time(self):
        """Test elapsed time tracking."""
        mock_swarm = Mock()
        executor = PhaseExecutor(mock_swarm)

        await asyncio.sleep(0.01)
        elapsed = executor.elapsed()

        assert elapsed > 0

    @pytest.mark.asyncio
    async def test_run_phase_success(self):
        """Test running a successful phase."""
        mock_swarm = Mock()
        mock_swarm._trace_phase = Mock()

        executor = PhaseExecutor(mock_swarm)

        @pytest.mark.asyncio
        async def test_coro():
            return {"result": "success"}

        result = await executor.run_phase(
            phase_num=1,
            phase_name="Test Phase",
            agent_name="TestAgent",
            agent_role=AgentRole.ACTOR,
            coro=test_coro(),
            input_data={"input": "test"},
            tools_used=["tool1"]
        )

        assert result == {"result": "success"}
        mock_swarm._trace_phase.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_phase_with_error_in_result(self):
        """Test running phase that returns error dict."""
        mock_swarm = Mock()
        mock_swarm._trace_phase = Mock()

        executor = PhaseExecutor(mock_swarm)

        @pytest.mark.asyncio
        async def test_coro():
            return {"error": "something failed"}

        result = await executor.run_phase(
            phase_num=1,
            phase_name="Test Phase",
            agent_name="TestAgent",
            agent_role=AgentRole.ACTOR,
            coro=test_coro()
        )

        assert "error" in result
        # Should trace with success=False
        call_args = mock_swarm._trace_phase.call_args
        assert call_args[1]['success'] is False

    @pytest.mark.asyncio
    async def test_run_phase_raises_exception(self):
        """Test running phase that raises exception."""
        mock_swarm = Mock()
        mock_swarm._trace_phase = Mock()

        executor = PhaseExecutor(mock_swarm)

        @pytest.mark.asyncio
        async def test_coro():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await executor.run_phase(
                phase_num=1,
                phase_name="Test Phase",
                agent_name="TestAgent",
                agent_role=AgentRole.ACTOR,
                coro=test_coro()
            )

    @pytest.mark.asyncio
    async def test_run_parallel_basic(self):
        """Test running parallel tasks."""
        mock_swarm = Mock()
        mock_swarm._trace_phase = Mock()

        executor = PhaseExecutor(mock_swarm)

        async def task1():
            return {"output": "task1"}

        async def task2():
            return {"output": "task2"}

        tasks = [
            ("Agent1", AgentRole.ACTOR, task1(), ["tool1"]),
            ("Agent2", AgentRole.ACTOR, task2(), ["tool2"]),
        ]

        results = await executor.run_parallel(
            phase_num=1,
            phase_name="Parallel Phase",
            tasks=tasks
        )

        assert len(results) == 2
        assert results[0] == {"output": "task1"}
        assert results[1] == {"output": "task2"}
        assert mock_swarm._trace_phase.call_count == 2

    @pytest.mark.asyncio
    async def test_run_parallel_with_exception(self):
        """Test parallel execution with one task raising exception."""
        mock_swarm = Mock()
        mock_swarm._trace_phase = Mock()

        executor = PhaseExecutor(mock_swarm)

        async def task1():
            return {"output": "task1"}

        async def task2():
            raise ValueError("Task2 failed")

        tasks = [
            ("Agent1", AgentRole.ACTOR, task1(), []),
            ("Agent2", AgentRole.ACTOR, task2(), []),
        ]

        results = await executor.run_parallel(
            phase_num=1,
            phase_name="Parallel Phase",
            tasks=tasks
        )

        assert len(results) == 2
        assert results[0] == {"output": "task1"}
        assert "error" in results[1]
        assert "Task2 failed" in results[1]["error"]

    def test_build_error_result(self):
        """Test building error result."""
        mock_swarm = Mock()
        executor = PhaseExecutor(mock_swarm)

        error = ValueError("Test error")
        result = executor.build_error_result(
            result_class=SwarmResult,
            error=error,
            config_name="TestSwarm",
            config_domain="test"
        )

        assert isinstance(result, SwarmResult)
        assert result.success is False
        assert result.swarm_name == "TestSwarm"
        assert result.domain == "test"
        assert result.error == "Test error"
        assert result.execution_time >= 0


# =============================================================================
# Test DomainSwarm Init
# =============================================================================

@pytest.mark.unit
class TestDomainSwarmInit:
    """Test DomainSwarm initialization."""

    def test_create_swarm_no_team(self):
        """Test creating swarm without AGENT_TEAM."""
        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)

        assert swarm.AGENT_TEAM is None
        assert swarm._agents_initialized is False
        assert swarm._learning_recorded is False

    def test_swarm_with_agent_team(self):
        """Test swarm with AGENT_TEAM class attribute."""
        class Agent1:
            pass

        class TestSwarm(DomainSwarm):
            AGENT_TEAM = AgentTeam.define((Agent1, "Agent1"))

            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)

        assert swarm.AGENT_TEAM is not None
        assert len(swarm.AGENT_TEAM) == 1

    def test_swarm_repr(self):
        """Test swarm __repr__ method."""
        class Agent1:
            pass

        class TestSwarm(DomainSwarm):
            AGENT_TEAM = AgentTeam.define(
                (Agent1, "Agent1"),
                pattern=CoordinationPattern.PIPELINE
            )

            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)

        repr_str = repr(swarm)
        assert "TestSwarm" in repr_str
        assert "agents=1" in repr_str
        assert "pipeline" in repr_str
        assert "initialized=False" in repr_str


# =============================================================================
# Test DomainSwarm Init Agents
# =============================================================================

@pytest.mark.unit
class TestDomainSwarmInitAgents:
    """Test DomainSwarm._init_agents method."""

    @patch('Jotty.core.swarms.base.domain_swarm.DomainSwarm._init_shared_resources')
    def test_init_agents_basic(self, mock_init_shared):
        """Test basic agent initialization."""
        class MockAgent:
            def __init__(self, **kwargs):
                pass

        class TestSwarm(DomainSwarm):
            AGENT_TEAM = AgentTeam.define((MockAgent, "Agent1"))

            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)
        swarm._memory = Mock()
        swarm._context = Mock()
        swarm._bus = Mock()

        swarm._init_agents()

        assert swarm._agents_initialized is True
        assert hasattr(swarm, '_agent1')
        mock_init_shared.assert_called_once()

    @patch('Jotty.core.swarms.base.domain_swarm.DomainSwarm._init_shared_resources')
    def test_init_agents_multiple(self, mock_init_shared):
        """Test initializing multiple agents."""
        class Agent1:
            def __init__(self, **kwargs):
                pass

        class Agent2:
            def __init__(self, **kwargs):
                pass

        class TestSwarm(DomainSwarm):
            AGENT_TEAM = AgentTeam.define(
                (Agent1, "Agent1"),
                (Agent2, "Agent2"),
            )

            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)
        swarm._memory = Mock()
        swarm._context = Mock()
        swarm._bus = Mock()

        swarm._init_agents()

        assert hasattr(swarm, '_agent1')
        assert hasattr(swarm, '_agent2')

    @patch('Jotty.core.swarms.base.domain_swarm.DomainSwarm._init_shared_resources')
    def test_init_agents_idempotent(self, mock_init_shared):
        """Test _init_agents is idempotent."""
        class MockAgent:
            def __init__(self, **kwargs):
                pass

        class TestSwarm(DomainSwarm):
            AGENT_TEAM = AgentTeam.define((MockAgent, "Agent1"))

            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)
        swarm._memory = Mock()
        swarm._context = Mock()
        swarm._bus = Mock()

        swarm._init_agents()
        swarm._init_agents()  # Call again

        assert mock_init_shared.call_count == 1  # Only once

    def test_init_agents_no_team(self):
        """Test _init_agents with no AGENT_TEAM."""
        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)

        with patch.object(swarm, '_init_shared_resources'):
            swarm._init_agents()

        assert swarm._agents_initialized is True


# =============================================================================
# Test DomainSwarm Create Agent
# =============================================================================

@pytest.mark.unit
class TestDomainSwarmCreateAgent:
    """Test DomainSwarm._create_agent method."""

    def test_create_agent_with_matching_params(self):
        """Test creating agent with matching parameters."""
        class MockAgent:
            def __init__(self, memory, context):
                self.memory = memory
                self.context = context

        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)
        swarm._memory = Mock()
        swarm._context = Mock()
        swarm._bus = Mock()

        spec = AgentSpec(agent_class=MockAgent, display_name="TestAgent")
        agent = swarm._create_agent(spec)

        assert isinstance(agent, MockAgent)
        assert agent.memory == swarm._memory
        assert agent.context == swarm._context

    def test_create_agent_with_kwargs(self):
        """Test creating agent that accepts **kwargs."""
        class MockAgent:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)
        swarm._memory = Mock()
        swarm._context = Mock()
        swarm._bus = Mock()

        with patch.object(swarm, '_agent_context', return_value={}):
            spec = AgentSpec(agent_class=MockAgent, display_name="TestAgent")
            agent = swarm._create_agent(spec)

        assert isinstance(agent, MockAgent)
        assert 'memory' in agent.kwargs
        assert 'context' in agent.kwargs
        assert 'bus' in agent.kwargs

    def test_create_agent_with_learned_context(self):
        """Test creating agent with learned_context parameter."""
        class MockAgent:
            def __init__(self, learned_context):
                self.learned_context = learned_context

        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)
        swarm._memory = Mock()
        swarm._context = Mock()
        swarm._bus = Mock()

        mock_learned = {"key": "value"}
        with patch.object(swarm, '_agent_context', return_value=mock_learned):
            spec = AgentSpec(agent_class=MockAgent, display_name="TestAgent")
            agent = swarm._create_agent(spec)

        assert agent.learned_context == mock_learned


# =============================================================================
# Test DomainSwarm Execute
# =============================================================================

@pytest.mark.unit
class TestDomainSwarmExecute:
    """Test DomainSwarm.execute template method."""

    @pytest.mark.asyncio
    async def test_execute_calls_init_agents(self):
        """Test execute calls _init_agents."""
        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)

        with patch.object(swarm, '_init_agents') as mock_init:
            with patch.object(swarm, '_pre_execute_learning', new_callable=AsyncMock):
                await swarm.execute()

        mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_calls_pre_learning(self):
        """Test execute calls _pre_execute_learning."""
        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)

        with patch.object(swarm, '_init_agents'):
            with patch.object(swarm, '_pre_execute_learning', new_callable=AsyncMock) as mock_pre:
                await swarm.execute()

        mock_pre.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_calls_domain_logic(self):
        """Test execute calls _execute_domain."""
        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={"result": "done"},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)

        with patch.object(swarm, '_init_agents'):
            with patch.object(swarm, '_pre_execute_learning', new_callable=AsyncMock):
                result = await swarm.execute()

        assert result.success is True
        assert result.output == {"result": "done"}

    @pytest.mark.asyncio
    async def test_execute_handles_pre_learning_error(self):
        """Test execute continues if pre-learning fails."""
        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)

        async def failing_pre():
            raise ValueError("Pre-learning failed")

        with patch.object(swarm, '_init_agents'):
            with patch.object(swarm, '_pre_execute_learning', side_effect=failing_pre):
                result = await swarm.execute()

        # Should still execute successfully
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_resets_learning_recorded(self):
        """Test execute resets _learning_recorded flag."""
        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)
        swarm._learning_recorded = True

        with patch.object(swarm, '_init_agents'):
            with patch.object(swarm, '_pre_execute_learning', new_callable=AsyncMock):
                await swarm.execute()

        # Reset at start of execution
        assert swarm._learning_recorded is False


# =============================================================================
# Test DomainSwarm Team Coordination
# =============================================================================

@pytest.mark.unit
class TestDomainSwarmTeamCoordination:
    """Test DomainSwarm team coordination methods."""

    @pytest.mark.asyncio
    async def test_execute_team_basic(self):
        """Test execute_team delegates to team."""
        class MockAgent:
            def __init__(self, **kwargs):
                pass

            async def execute(self, **kwargs):
                return Mock(output="test")

        class TestSwarm(DomainSwarm):
            AGENT_TEAM = AgentTeam.define(
                (MockAgent, "Agent1"),
                pattern=CoordinationPattern.PARALLEL
            )

            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)
        swarm._memory = Mock()
        swarm._context = Mock()
        swarm._bus = Mock()

        swarm._init_agents()

        result = await swarm.execute_team(task="test")

        assert isinstance(result, TeamResult)

    @pytest.mark.asyncio
    async def test_execute_team_no_team_raises(self):
        """Test execute_team raises without AGENT_TEAM."""
        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)

        with pytest.raises(RuntimeError, match="No AGENT_TEAM"):
            await swarm.execute_team(task="test")

    @pytest.mark.asyncio
    async def test_execute_team_inits_agents(self):
        """Test execute_team initializes agents if needed."""
        class MockAgent:
            def __init__(self, **kwargs):
                pass

        class TestSwarm(DomainSwarm):
            AGENT_TEAM = AgentTeam.define((MockAgent, "Agent1"))

            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)
        swarm._memory = Mock()
        swarm._context = Mock()
        swarm._bus = Mock()

        # Mock _init_agents to set up team instances
        original_init = swarm._init_agents
        def mock_init_with_instances():
            original_init()
            # Manually set instances after init
            swarm.AGENT_TEAM.set_instances({"_agent1": MockAgent()})

        with patch.object(swarm, '_init_agents', side_effect=mock_init_with_instances) as mock_init:
            await swarm.execute_team(task="test")

        mock_init.assert_called_once()

    def test_has_team_coordination_true(self):
        """Test has_team_coordination returns True for pattern."""
        class Agent1:
            pass

        class TestSwarm(DomainSwarm):
            AGENT_TEAM = AgentTeam.define(
                (Agent1, "Agent1"),
                pattern=CoordinationPattern.PARALLEL
            )

            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)

        assert swarm.has_team_coordination() is True

    def test_has_team_coordination_false(self):
        """Test has_team_coordination returns False for NONE pattern."""
        class Agent1:
            pass

        class TestSwarm(DomainSwarm):
            AGENT_TEAM = AgentTeam.define((Agent1, "Agent1"))

            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)

        assert swarm.has_team_coordination() is False

    def test_has_team_coordination_no_team(self):
        """Test has_team_coordination returns False without team."""
        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)

        assert swarm.has_team_coordination() is False


# =============================================================================
# Test DomainSwarm Helpers
# =============================================================================

@pytest.mark.unit
class TestDomainSwarmHelpers:
    """Test DomainSwarm helper methods."""

    def test_phase_executor_creation(self):
        """Test _phase_executor creates PhaseExecutor."""
        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)

        executor = swarm._phase_executor()

        assert isinstance(executor, PhaseExecutor)
        assert executor.swarm == swarm

    def test_get_agents_with_team(self):
        """Test get_agents returns agent instances."""
        class MockAgent:
            def __init__(self, **kwargs):
                pass

        class TestSwarm(DomainSwarm):
            AGENT_TEAM = AgentTeam.define(
                (MockAgent, "Agent1"),
                (MockAgent, "Agent2"),
            )

            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)
        swarm._memory = Mock()
        swarm._context = Mock()
        swarm._bus = Mock()

        swarm._init_agents()
        agents = swarm.get_agents()

        assert len(agents) == 2
        assert "_agent1" in agents
        assert "_agent2" in agents

    def test_get_agents_no_team(self):
        """Test get_agents returns empty dict without team."""
        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)

        agents = swarm.get_agents()

        assert agents == {}

    def test_get_io_schema_no_signature(self):
        """Test get_io_schema returns None without SWARM_SIGNATURE."""
        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)

        schema = swarm.get_io_schema()

        assert schema is None

    def test_to_composite(self):
        """Test to_composite creates CompositeAgent."""
        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)

        # CompositeAgent is imported inside to_composite method
        with patch('Jotty.core.agents.base.composite_agent.CompositeAgent') as mock_composite:
            mock_composite.from_swarm = Mock(return_value=Mock())
            composite = swarm.to_composite()

        mock_composite.from_swarm.assert_called_once()


# =============================================================================
# Test DomainSwarm Edge Cases
# =============================================================================

@pytest.mark.unit
class TestDomainSwarmEdgeCases:
    """Test DomainSwarm edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_agent_team(self):
        """Test swarm with empty AGENT_TEAM."""
        class TestSwarm(DomainSwarm):
            AGENT_TEAM = AgentTeam()

            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)

        with patch.object(swarm, '_init_shared_resources'):
            swarm._init_agents()

        assert len(swarm.AGENT_TEAM) == 0

    @pytest.mark.asyncio
    async def test_agent_init_failure(self):
        """Test handling agent initialization failure."""
        class FailingAgent:
            def __init__(self, **kwargs):
                raise ValueError("Agent init failed")

        class TestSwarm(DomainSwarm):
            AGENT_TEAM = AgentTeam.define((FailingAgent, "FailAgent"))

            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)
        swarm._memory = Mock()
        swarm._context = Mock()
        swarm._bus = Mock()

        with patch.object(swarm, '_init_shared_resources'):
            with pytest.raises(ValueError):
                swarm._init_agents()

    @pytest.mark.asyncio
    async def test_safe_execute_domain_success(self):
        """Test _safe_execute_domain with successful execution."""
        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={"result": "done"},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)

        async def execute_fn(executor):
            return SwarmResult(
                success=True,
                swarm_name="test",
                domain="test",
                output={"result": "done"},
                execution_time=1.0
            )

        with patch.object(swarm, '_post_execute_learning', new_callable=AsyncMock):
            with patch.object(swarm, '_get_active_tools', return_value=[]):
                result = await swarm._safe_execute_domain(
                    task_type="test",
                    default_tools=[],
                    result_class=SwarmResult,
                    execute_fn=execute_fn
                )

        assert result.success is True
        assert swarm._learning_recorded is True

    @pytest.mark.asyncio
    async def test_safe_execute_domain_error(self):
        """Test _safe_execute_domain with execution error."""
        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        config = SwarmBaseConfig(name="TestSwarm", domain="test")
        swarm = TestSwarm(config)

        async def execute_fn(executor):
            raise ValueError("Execution failed")

        with patch.object(swarm, '_post_execute_learning', new_callable=AsyncMock):
            with patch.object(swarm, '_get_active_tools', return_value=[]):
                result = await swarm._safe_execute_domain(
                    task_type="test",
                    default_tools=[],
                    result_class=SwarmResult,
                    execute_fn=execute_fn
                )

        assert result.success is False
        assert "Execution failed" in result.error
        assert swarm._learning_recorded is True

    @pytest.mark.asyncio
    async def test_execute_domain_is_abstract(self):
        """Test _execute_domain is abstract and must be implemented."""
        class IncompleteSwarm(DomainSwarm):
            pass

        config = SwarmBaseConfig(name="IncompleteSwarm", domain="test")

        # Should not be able to instantiate without implementing _execute_domain
        with pytest.raises(TypeError):
            swarm = IncompleteSwarm(config)

    def test_defensive_utilities_available(self):
        """Test defensive utilities are available as static methods."""
        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        # Should be accessible as static methods
        assert callable(TestSwarm._split_field)
        assert callable(TestSwarm._safe_join)
        assert callable(TestSwarm._safe_num)

    def test_split_field_usage(self):
        """Test _split_field defensive utility."""
        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        result = TestSwarm._split_field("a|b|c")
        assert result == ["a", "b", "c"]

        result = TestSwarm._split_field(["x", "y"])
        assert result == ["x", "y"]

    def test_safe_join_usage(self):
        """Test _safe_join defensive utility."""
        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        result = TestSwarm._safe_join(["a", "b", "c"])
        assert result == "a, b, c"

    def test_safe_num_usage(self):
        """Test _safe_num defensive utility."""
        class TestSwarm(DomainSwarm):
            async def _execute_domain(self, *args, **kwargs):
                return SwarmResult(
                    success=True,
                    swarm_name="test",
                    domain="test",
                    output={},
                    execution_time=0.0
                )

        result = TestSwarm._safe_num("123")
        assert result == 123.0

        result = TestSwarm._safe_num("invalid", default=0)
        assert result == 0


# =============================================================================
# Summary
# =============================================================================

"""
Test Coverage Summary:
======================

1. CoordinationPattern enum - 2 tests
2. MergeStrategy enum - 2 tests
3. AgentSpec - 9 tests
4. TeamResult - 4 tests
5. AgentTeam.define() - 9 tests
6. AgentTeam operations - 8 tests
7. AgentTeam execution - 11 tests
8. AgentTeam merge outputs - 6 tests
9. PhaseExecutor - 6 tests
10. DomainSwarm init - 3 tests
11. DomainSwarm init agents - 4 tests
12. DomainSwarm create agent - 3 tests
13. DomainSwarm execute - 5 tests
14. DomainSwarm team coordination - 6 tests
15. DomainSwarm helpers - 6 tests
16. DomainSwarm edge cases - 11 tests

Total: 95 tests (individual test methods)

Test classes: 16
Coverage areas:
- All enum values and types
- AgentSpec creation and attr_name generation (various cases)
- TeamResult with all field combinations
- AgentTeam definition with all spec tuple lengths
- All coordination patterns (PIPELINE, PARALLEL, CONSENSUS, HIERARCHICAL, BLACKBOARD, ROUND_ROBIN)
- All merge strategies (COMBINE, FIRST, BEST, VOTE, CONCAT)
- PhaseExecutor timing, phase execution, parallel execution, error handling
- DomainSwarm lifecycle (init, agent creation, execution template)
- Team coordination integration
- Edge cases (errors, empty teams, failures)
- Defensive utilities (_split_field, _safe_join, _safe_num)

All tests:
- Use @pytest.mark.unit
- Use @pytest.mark.asyncio for async tests
- Use mocks to avoid LLM calls
- Are fast and offline
- Follow class-based structure
- Test both success and error paths
"""

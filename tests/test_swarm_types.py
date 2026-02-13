"""
Tests for Jotty swarm types and data classes.

Covers: AgentRole, EvaluationResult, ImprovementType enums,
        SwarmAgentConfig, SwarmBaseConfig, SwarmResult, ExecutionTrace dataclasses.
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from Jotty.core.swarms.swarm_types import (
    AgentRole, EvaluationResult, ImprovementType,
    SwarmAgentConfig, SwarmBaseConfig, SwarmResult, ExecutionTrace,
    GoldStandard, Evaluation, ImprovementSuggestion,
)


@pytest.mark.unit
class TestSwarmAgentConfigDefaults:
    """SwarmAgentConfig sentinel defaults are resolved via __post_init__."""

    def test_sentinel_defaults_resolved(self):
        """Field sentinels (model='', temperature=0.0, max_tokens=0) are replaced by config_defaults."""
        config = SwarmAgentConfig(role=AgentRole.ACTOR, name="test-agent")
        # __post_init__ resolves "" -> DEFAULTS.DEFAULT_MODEL_ALIAS
        assert config.model != ""
        assert config.model == "sonnet"
        # __post_init__ resolves 0.0 -> DEFAULTS.LLM_TEMPERATURE
        assert config.temperature == 0.7
        # __post_init__ resolves 0 -> DEFAULTS.LLM_MAX_OUTPUT_TOKENS
        assert config.max_tokens == 8192

    def test_non_sentinel_values_preserved(self):
        """Explicit non-sentinel values are NOT overridden by __post_init__."""
        config = SwarmAgentConfig(
            role=AgentRole.PLANNER,
            name="custom-agent",
            model="opus",
            temperature=0.5,
            max_tokens=4096,
        )
        assert config.model == "opus"
        assert config.temperature == 0.5
        assert config.max_tokens == 4096


@pytest.mark.unit
class TestSwarmAgentConfigRoleAndName:
    """SwarmAgentConfig stores role and name correctly."""

    def test_role_and_name(self):
        config = SwarmAgentConfig(role=AgentRole.REVIEWER, name="reviewer-1")
        assert config.role == AgentRole.REVIEWER
        assert config.name == "reviewer-1"

    def test_optional_fields_defaults(self):
        config = SwarmAgentConfig(role=AgentRole.EXPERT, name="expert-1")
        assert config.system_prompt == ""
        assert config.tools == []
        assert config.parameters == {}
        assert config.version == 1


@pytest.mark.unit
class TestSwarmBaseConfigDefaults:
    """SwarmBaseConfig defaults match expected values."""

    def test_identity_defaults(self):
        config = SwarmBaseConfig()
        assert config.name == "BaseSwarm"
        assert config.domain == "general"
        assert config.version == "1.0.0"

    def test_execution_defaults(self):
        config = SwarmBaseConfig()
        assert config.max_retries == 3
        assert config.timeout_seconds == 300
        assert config.parallel_execution is True
        assert config.improvement_threshold == 0.7
        assert config.gold_standard_max_version == 3

    def test_output_dir_defaults_to_home(self):
        config = SwarmBaseConfig()
        expected = str(Path.home() / "jotty" / "swarm_outputs")
        assert config.output_dir == expected


@pytest.mark.unit
class TestSwarmBaseConfigLearningFlags:
    """SwarmBaseConfig learning-related flags default to True."""

    def test_enable_learning_default_true(self):
        config = SwarmBaseConfig()
        assert config.enable_learning is True

    def test_enable_self_improvement_default_true(self):
        config = SwarmBaseConfig()
        assert config.enable_self_improvement is True

    def test_gold_standard_path_default_none(self):
        config = SwarmBaseConfig()
        assert config.gold_standard_path is None


@pytest.mark.unit
class TestSwarmResultSuccess:
    """SwarmResult creation for successful execution."""

    def test_success_with_required_fields(self):
        result = SwarmResult(
            success=True,
            swarm_name="CodingSwarm",
            domain="coding",
            output={"code": "print('hello')"},
            execution_time=2.5,
        )
        assert result.success is True
        assert result.swarm_name == "CodingSwarm"
        assert result.domain == "coding"
        assert result.output == {"code": "print('hello')"}
        assert result.execution_time == 2.5


@pytest.mark.unit
class TestSwarmResultFailure:
    """SwarmResult with error information."""

    def test_failure_with_error_string(self):
        result = SwarmResult(
            success=False,
            swarm_name="TestSwarm",
            domain="testing",
            output={},
            execution_time=0.1,
            error="LLM timeout after 120s",
        )
        assert result.success is False
        assert result.error == "LLM timeout after 120s"
        assert result.output == {}


@pytest.mark.unit
class TestSwarmResultDefaults:
    """SwarmResult optional fields default correctly."""

    def test_agent_traces_defaults_empty(self):
        result = SwarmResult(
            success=True,
            swarm_name="Swarm",
            domain="general",
            output={"status": "ok"},
            execution_time=1.0,
        )
        assert result.agent_traces == []
        assert result.evaluation is None
        assert result.improvements == []
        assert result.error is None
        assert result.metadata == {}


@pytest.mark.unit
class TestExecutionTraceCreation:
    """ExecutionTrace captures agent execution details."""

    def test_creation_with_required_fields(self):
        trace = ExecutionTrace(
            agent_name="planner-agent",
            agent_role=AgentRole.PLANNER,
            input_data={"task": "plan deployment"},
            output_data={"steps": ["build", "test", "deploy"]},
            execution_time=1.2,
            success=True,
        )
        assert trace.agent_name == "planner-agent"
        assert trace.agent_role == AgentRole.PLANNER
        assert trace.input_data == {"task": "plan deployment"}
        assert trace.output_data == {"steps": ["build", "test", "deploy"]}
        assert trace.execution_time == 1.2
        assert trace.success is True
        assert trace.error is None
        assert trace.metadata == {}


@pytest.mark.unit
class TestExecutionTraceTimestamp:
    """ExecutionTrace auto-sets timestamp to a datetime instance."""

    def test_timestamp_auto_set(self):
        before = datetime.now()
        trace = ExecutionTrace(
            agent_name="actor-agent",
            agent_role=AgentRole.ACTOR,
            input_data={},
            output_data={},
            execution_time=0.5,
            success=True,
        )
        after = datetime.now()
        assert isinstance(trace.timestamp, datetime)
        assert before <= trace.timestamp <= after


@pytest.mark.unit
class TestEnumValues:
    """AgentRole, EvaluationResult, and ImprovementType enum members."""

    def test_agent_role_values(self):
        assert AgentRole.EXPERT.value == "expert"
        assert AgentRole.REVIEWER.value == "reviewer"
        assert AgentRole.PLANNER.value == "planner"
        assert AgentRole.ACTOR.value == "actor"
        assert AgentRole.ORCHESTRATOR.value == "orchestrator"
        assert AgentRole.AUDITOR.value == "auditor"
        assert AgentRole.LEARNER.value == "learner"

    def test_evaluation_result_values(self):
        assert EvaluationResult.EXCELLENT.value == "excellent"
        assert EvaluationResult.GOOD.value == "good"
        assert EvaluationResult.ACCEPTABLE.value == "acceptable"
        assert EvaluationResult.NEEDS_IMPROVEMENT.value == "needs_improvement"
        assert EvaluationResult.FAILED.value == "failed"

    def test_improvement_type_values(self):
        assert ImprovementType.PROMPT_REFINEMENT.value == "prompt_refinement"
        assert ImprovementType.PARAMETER_TUNING.value == "parameter_tuning"
        assert ImprovementType.WORKFLOW_CHANGE.value == "workflow_change"
        assert ImprovementType.AGENT_REPLACEMENT.value == "agent_replacement"
        assert ImprovementType.TRAINING_DATA.value == "training_data"

    def test_enum_membership(self):
        """Enum lookup by value works correctly."""
        assert AgentRole("expert") is AgentRole.EXPERT
        assert EvaluationResult("failed") is EvaluationResult.FAILED
        assert ImprovementType("workflow_change") is ImprovementType.WORKFLOW_CHANGE

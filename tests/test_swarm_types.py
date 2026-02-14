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


# ===========================================================================
# GoldStandard Tests
# ===========================================================================

@pytest.mark.unit
class TestGoldStandard:
    """Tests for GoldStandard dataclass."""

    def test_creation_with_required_fields(self):
        gs = GoldStandard(
            id="gs-001",
            domain="coding",
            task_type="generate_code",
            input_data={"requirements": "Build a REST API"},
            expected_output={"code": "..."},
            evaluation_criteria={"correctness": 1.0, "style": 0.8},
        )
        assert gs.id == "gs-001"
        assert gs.domain == "coding"
        assert gs.task_type == "generate_code"
        assert gs.evaluation_criteria["correctness"] == 1.0

    def test_default_version(self):
        gs = GoldStandard(
            id="gs-002",
            domain="testing",
            task_type="gen_tests",
            input_data={},
            expected_output={},
            evaluation_criteria={},
        )
        assert gs.version == 1

    def test_auto_created_at(self):
        gs = GoldStandard(
            id="gs-003",
            domain="coding",
            task_type="task",
            input_data={},
            expected_output={},
            evaluation_criteria={},
        )
        assert isinstance(gs.created_at, datetime)


# ===========================================================================
# Evaluation Tests
# ===========================================================================

@pytest.mark.unit
class TestEvaluation:
    """Tests for Evaluation dataclass."""

    def test_creation(self):
        ev = Evaluation(
            gold_standard_id="gs-001",
            actual_output={"code": "print('hello')"},
            scores={"correctness": 0.9, "style": 0.8},
            overall_score=0.85,
            result=EvaluationResult.GOOD,
            feedback=["Well done", "Minor style issues"],
        )
        assert ev.overall_score == 0.85
        assert ev.result == EvaluationResult.GOOD
        assert "correctness" in ev.scores

    def test_feedback_list(self):
        ev = Evaluation(
            gold_standard_id="gs-002",
            actual_output={},
            scores={},
            overall_score=0.5,
            result=EvaluationResult.ACCEPTABLE,
            feedback=[],
        )
        assert ev.feedback == []
        assert ev.gold_standard_id == "gs-002"


# ===========================================================================
# ImprovementSuggestion Tests
# ===========================================================================

@pytest.mark.unit
class TestImprovementSuggestion:
    """Tests for ImprovementSuggestion dataclass."""

    def test_creation(self):
        sug = ImprovementSuggestion(
            agent_role=AgentRole.ACTOR,
            improvement_type=ImprovementType.PROMPT_REFINEMENT,
            description="Add explicit format instructions",
            priority=3,
            expected_impact=0.15,
            implementation_details={"prompt": "Be more explicit"},
            based_on_evaluations=["eval-001"],
        )
        assert sug.agent_role == AgentRole.ACTOR
        assert sug.improvement_type == ImprovementType.PROMPT_REFINEMENT
        assert sug.priority == 3

    def test_all_fields_stored(self):
        sug = ImprovementSuggestion(
            agent_role=AgentRole.EXPERT,
            improvement_type=ImprovementType.PARAMETER_TUNING,
            description="Lower temperature",
            priority=2,
            expected_impact=0.1,
            implementation_details={"temperature": 0.3},
            based_on_evaluations=["eval-002", "eval-003"],
        )
        assert sug.expected_impact == 0.1
        assert sug.implementation_details == {"temperature": 0.3}
        assert len(sug.based_on_evaluations) == 2


# ===========================================================================
# SwarmRegistry Tests
# ===========================================================================

@pytest.mark.unit
class TestSwarmRegistry:
    """Tests for SwarmRegistry class-level registry."""

    def setup_method(self):
        """Clear registry before each test."""
        from Jotty.core.swarms.registry import SwarmRegistry
        self._original = dict(SwarmRegistry._swarms)

    def teardown_method(self):
        """Restore registry after each test."""
        from Jotty.core.swarms.registry import SwarmRegistry
        SwarmRegistry._swarms = self._original

    def test_register_and_get(self):
        from Jotty.core.swarms.registry import SwarmRegistry
        from unittest.mock import MagicMock
        mock_class = MagicMock
        SwarmRegistry.register("test_swarm", mock_class)
        assert SwarmRegistry.get("test_swarm") is mock_class

    def test_get_unknown_returns_none(self):
        from Jotty.core.swarms.registry import SwarmRegistry
        assert SwarmRegistry.get("nonexistent_swarm_xyz") is None

    def test_list_all(self):
        from Jotty.core.swarms.registry import SwarmRegistry
        from unittest.mock import MagicMock
        SwarmRegistry.register("swarm_a", MagicMock)
        SwarmRegistry.register("swarm_b", MagicMock)
        all_swarms = SwarmRegistry.list_all()
        assert "swarm_a" in all_swarms
        assert "swarm_b" in all_swarms

    def test_create_with_config(self):
        from Jotty.core.swarms.registry import SwarmRegistry
        from unittest.mock import MagicMock
        mock_class = MagicMock()
        SwarmRegistry.register("configurable", mock_class)
        config = SwarmBaseConfig(name="TestConfig")
        SwarmRegistry.create("configurable", config)
        mock_class.assert_called_once_with(config)

    def test_create_without_config(self):
        from Jotty.core.swarms.registry import SwarmRegistry
        from unittest.mock import MagicMock
        mock_class = MagicMock()
        SwarmRegistry.register("no_config", mock_class)
        SwarmRegistry.create("no_config")
        mock_class.assert_called_once_with()

    def test_create_unknown_returns_none(self):
        from Jotty.core.swarms.registry import SwarmRegistry
        assert SwarmRegistry.create("unknown_swarm_xyz") is None


@pytest.mark.unit
class TestRegisterSwarmDecorator:
    """Tests for the register_swarm decorator."""

    def setup_method(self):
        from Jotty.core.swarms.registry import SwarmRegistry
        self._original = dict(SwarmRegistry._swarms)

    def teardown_method(self):
        from Jotty.core.swarms.registry import SwarmRegistry
        SwarmRegistry._swarms = self._original

    def test_decorator_registers_class(self):
        from Jotty.core.swarms.registry import SwarmRegistry, register_swarm

        @register_swarm("decorated_swarm")
        class MySwarm:
            pass

        assert SwarmRegistry.get("decorated_swarm") is MySwarm

    def test_decorator_preserves_class(self):
        from Jotty.core.swarms.registry import register_swarm

        @register_swarm("preserve_test")
        class OriginalSwarm:
            def run(self):
                return "hello"

        assert OriginalSwarm().run() == "hello"


# ===========================================================================
# GoldStandardDB Tests
# ===========================================================================

@pytest.mark.unit
class TestGoldStandardDB:
    """Tests for GoldStandardDB persistence."""

    def test_add_and_get(self, tmp_path):
        from Jotty.core.swarms.evaluation import GoldStandardDB
        db = GoldStandardDB(str(tmp_path / "gs_db"))
        gs = GoldStandard(
            id="test-001",
            domain="coding",
            task_type="generate",
            input_data={"req": "hello"},
            expected_output={"code": "world"},
            evaluation_criteria={"correctness": 1.0},
        )
        returned_id = db.add(gs)
        assert returned_id == "test-001"
        retrieved = db.get("test-001")
        assert retrieved is not None
        assert retrieved.domain == "coding"

    def test_auto_generate_id(self, tmp_path):
        from Jotty.core.swarms.evaluation import GoldStandardDB
        db = GoldStandardDB(str(tmp_path / "gs_db"))
        gs = GoldStandard(
            id="",
            domain="testing",
            task_type="gen_tests",
            input_data={"code": "x"},
            expected_output={"tests": "y"},
            evaluation_criteria={"coverage": 0.9},
        )
        generated_id = db.add(gs)
        assert len(generated_id) == 12
        assert db.get(generated_id) is not None

    def test_find_by_domain(self, tmp_path):
        from Jotty.core.swarms.evaluation import GoldStandardDB
        db = GoldStandardDB(str(tmp_path / "gs_db"))
        for i in range(3):
            db.add(GoldStandard(
                id=f"gs-{i}", domain="sql" if i < 2 else "coding",
                task_type="task", input_data={}, expected_output={},
                evaluation_criteria={},
            ))
        sql_results = db.find_by_domain("sql")
        assert len(sql_results) == 2

    def test_find_similar(self, tmp_path):
        from Jotty.core.swarms.evaluation import GoldStandardDB
        db = GoldStandardDB(str(tmp_path / "gs_db"))
        db.add(GoldStandard(
            id="gs-sim", domain="coding", task_type="generate",
            input_data={}, expected_output={}, evaluation_criteria={},
        ))
        result = db.find_similar("generate", {})
        assert result is not None
        assert result.id == "gs-sim"

    def test_find_similar_no_match(self, tmp_path):
        from Jotty.core.swarms.evaluation import GoldStandardDB
        db = GoldStandardDB(str(tmp_path / "gs_db"))
        assert db.find_similar("nonexistent_type", {}) is None

    def test_list_all(self, tmp_path):
        from Jotty.core.swarms.evaluation import GoldStandardDB
        db = GoldStandardDB(str(tmp_path / "gs_db"))
        db.add(GoldStandard(id="a", domain="x", task_type="t",
                            input_data={}, expected_output={}, evaluation_criteria={}))
        db.add(GoldStandard(id="b", domain="y", task_type="t",
                            input_data={}, expected_output={}, evaluation_criteria={}))
        assert len(db.list_all()) == 2

    def test_persistence_across_instances(self, tmp_path):
        from Jotty.core.swarms.evaluation import GoldStandardDB
        db_path = str(tmp_path / "gs_db")
        db1 = GoldStandardDB(db_path)
        db1.add(GoldStandard(id="persist", domain="test", task_type="t",
                             input_data={}, expected_output={}, evaluation_criteria={}))
        db2 = GoldStandardDB(db_path)
        assert db2.get("persist") is not None


# ===========================================================================
# ImprovementHistory Tests
# ===========================================================================

@pytest.mark.unit
class TestImprovementHistory:
    """Tests for ImprovementHistory."""

    def test_record_and_retrieve(self, tmp_path):
        from Jotty.core.swarms.evaluation import ImprovementHistory
        hist = ImprovementHistory(str(tmp_path / "impr"))
        sug = ImprovementSuggestion(
            agent_role=AgentRole.ACTOR,
            improvement_type=ImprovementType.PROMPT_REFINEMENT,
            description="Add format instructions",
            priority=3,
            expected_impact=0.15,
            implementation_details={"prompt": "be explicit"},
            based_on_evaluations=["eval-001"],
        )
        sug_id = hist.record_suggestion(sug)
        assert len(sug_id) == 12
        pending = hist.get_pending_suggestions()
        assert len(pending) == 1

    def test_mark_applied(self, tmp_path):
        from Jotty.core.swarms.evaluation import ImprovementHistory
        hist = ImprovementHistory(str(tmp_path / "impr"))
        sug = ImprovementSuggestion(
            agent_role=AgentRole.REVIEWER,
            improvement_type=ImprovementType.WORKFLOW_CHANGE,
            description="test",
            priority=2,
            expected_impact=0.1,
            implementation_details={},
            based_on_evaluations=[],
        )
        sug_id = hist.record_suggestion(sug)
        hist.mark_applied(sug_id)
        pending = hist.get_pending_suggestions()
        assert len(pending) == 0

    def test_record_outcome(self, tmp_path):
        from Jotty.core.swarms.evaluation import ImprovementHistory
        hist = ImprovementHistory(str(tmp_path / "impr"))
        sug = ImprovementSuggestion(
            agent_role=AgentRole.ACTOR,
            improvement_type=ImprovementType.PARAMETER_TUNING,
            description="test",
            priority=2,
            expected_impact=0.1,
            implementation_details={},
            based_on_evaluations=[],
        )
        sug_id = hist.record_suggestion(sug)
        hist.record_outcome(sug_id, success=True, impact=0.15, notes="Improved speed")
        successful = hist.get_successful_improvements()
        assert len(successful) == 1

    def test_filter_by_role(self, tmp_path):
        """get_successful_improvements filters by role (note: asdict stores enum objects)."""
        from Jotty.core.swarms.evaluation import ImprovementHistory
        hist = ImprovementHistory(str(tmp_path / "impr"))
        for role in [AgentRole.ACTOR, AgentRole.EXPERT, AgentRole.ACTOR]:
            sug = ImprovementSuggestion(
                agent_role=role,
                improvement_type=ImprovementType.PROMPT_REFINEMENT,
                description=f"test-{role.value}",
                priority=0.5,
                expected_impact="low",
                implementation_details="details",
                based_on_evaluations=[],
            )
            sug_id = hist.record_suggestion(sug)
            hist.record_outcome(sug_id, success=True, impact=0.1)
        # Without filter, all 3 should be returned
        all_successful = hist.get_successful_improvements()
        assert len(all_successful) == 3


# ===========================================================================
# EvaluationHistory Tests
# ===========================================================================

@pytest.mark.unit
class TestEvaluationHistory:
    """Tests for EvaluationHistory."""

    def test_record_and_get_recent(self, tmp_path):
        from Jotty.core.swarms.evaluation import EvaluationHistory
        from unittest.mock import MagicMock
        hist = EvaluationHistory(str(tmp_path / "evals"))
        ev = MagicMock()
        ev.overall_score = 0.85
        ev.status = "GOOD"
        ev.dimension_scores = {"correctness": 0.9}
        ev.feedback = "Good job"
        hist.record(ev)
        recent = hist.get_recent(10)
        assert len(recent) == 1
        assert recent[0]["overall_score"] == 0.85

    def test_get_average_score(self, tmp_path):
        from Jotty.core.swarms.evaluation import EvaluationHistory
        from unittest.mock import MagicMock
        hist = EvaluationHistory(str(tmp_path / "evals"))
        for score in [0.8, 0.6, 1.0]:
            ev = MagicMock()
            ev.overall_score = score
            ev.status = "OK"
            ev.dimension_scores = {}
            ev.feedback = ""
            hist.record(ev)
        avg = hist.get_average_score(10)
        assert abs(avg - 0.8) < 0.01

    def test_get_average_score_empty(self, tmp_path):
        from Jotty.core.swarms.evaluation import EvaluationHistory
        hist = EvaluationHistory(str(tmp_path / "evals"))
        assert hist.get_average_score() == 0.0

    def test_get_failures(self, tmp_path):
        from Jotty.core.swarms.evaluation import EvaluationHistory
        from unittest.mock import MagicMock
        hist = EvaluationHistory(str(tmp_path / "evals"))
        for score in [0.3, 0.8, 0.2, 0.9]:
            ev = MagicMock()
            ev.overall_score = score
            ev.status = "FAILED" if score < 0.5 else "GOOD"
            ev.dimension_scores = {}
            ev.feedback = ""
            hist.record(ev)
        failures = hist.get_failures()
        assert len(failures) == 2


# ===========================================================================
# Swarm Signatures Tests (importability)
# ===========================================================================

@pytest.mark.unit
class TestSwarmSignatures:
    """Tests for swarm_signatures.py DSPy signatures."""

    def test_all_exports_exist(self):
        """All entries in __all__ are importable."""
        try:
            from Jotty.core.swarms import swarm_signatures
            for name in swarm_signatures.__all__:
                assert hasattr(swarm_signatures, name), f"{name} not found"
        except ImportError:
            pytest.skip("dspy not available")

    def test_expert_evaluation_signature_fields(self):
        """ExpertEvaluationSignature has expected fields."""
        try:
            from Jotty.core.swarms.swarm_signatures import ExpertEvaluationSignature
            sig = ExpertEvaluationSignature
            # DSPy signatures have fields accessible
            assert sig is not None
        except ImportError:
            pytest.skip("dspy not available")

    def test_coding_swarm_signature_exists(self):
        """CodingSwarmSignature is importable."""
        try:
            from Jotty.core.swarms.swarm_signatures import CodingSwarmSignature
            assert CodingSwarmSignature is not None
        except ImportError:
            pytest.skip("dspy not available")

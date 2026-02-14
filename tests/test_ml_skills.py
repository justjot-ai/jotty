"""
ML Skills Tests - Phase 7
=========================

Comprehensive tests for the ML skills infrastructure and individual skills
from core/skills/ml/.

Tests cover:
- SkillCategory enum values
- SkillResult dataclass creation, defaults, serialization
- MLSkill abstract base class (instantiation, init, validate_inputs, helpers)
- SkillPipeline (sequential execution, results aggregation)
- SkillRegistry (register, get, list, category filter)
- Individual skill metadata (EDA, FeatureEngineering, ModelSelection, Ensemble)
- Backtest data classes (BacktestMetrics, TradeStatistics, TransactionCosts)
- Integration patterns (cross-skill results, pipeline composition)

All sklearn/lightgbm/xgboost calls are mocked -- no real ML needed.
"""

import asyncio
from dataclasses import fields
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, MagicMock, patch

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Safe imports with pandas/sklearn guards
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from Jotty.core.skills.ml.base import (
        SkillCategory,
        SkillResult,
        MLSkill,
        SkillPipeline,
        SkillRegistry,
    )
    HAS_BASE = True
except ImportError:
    HAS_BASE = False

try:
    from Jotty.core.skills.ml.eda import EDASkill
    HAS_EDA = True
except ImportError:
    HAS_EDA = False

try:
    from Jotty.core.skills.ml.feature_engineering import FeatureEngineeringSkill
    HAS_FE = True
except ImportError:
    HAS_FE = False

try:
    from Jotty.core.skills.ml.model_selection import ModelSelectionSkill
    HAS_MS = True
except ImportError:
    HAS_MS = False

try:
    from Jotty.core.skills.ml.ensemble import EnsembleSkill
    HAS_ENS = True
except ImportError:
    HAS_ENS = False

try:
    from Jotty.core.skills.ml.backtest_report import (
        BacktestMetrics,
        TradeStatistics,
        ModelResults,
        BacktestResult,
    )
    HAS_BT_REPORT = True
except ImportError:
    HAS_BT_REPORT = False

try:
    from Jotty.core.skills.ml.backtest_engine import (
        TransactionCosts,
        RiskMetrics,
        StatisticalTests,
        RegimeAnalysis,
        FactorExposure,
        WalkForwardResult,
        MonteCarloResult,
        PositionSizing,
        ComprehensiveBacktestResult,
    )
    HAS_BT_ENGINE = True
except ImportError:
    HAS_BT_ENGINE = False


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------
skip_no_pandas = pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
skip_no_base = pytest.mark.skipif(not HAS_BASE, reason="ML base module not importable")
skip_no_eda = pytest.mark.skipif(not HAS_EDA, reason="EDA skill not importable")
skip_no_fe = pytest.mark.skipif(not HAS_FE, reason="FeatureEngineering skill not importable")
skip_no_ms = pytest.mark.skipif(not HAS_MS, reason="ModelSelection skill not importable")
skip_no_ens = pytest.mark.skipif(not HAS_ENS, reason="Ensemble skill not importable")
skip_no_bt_report = pytest.mark.skipif(not HAS_BT_REPORT, reason="Backtest report not importable")
skip_no_bt_engine = pytest.mark.skipif(not HAS_BT_ENGINE, reason="Backtest engine not importable")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_df(rows=10, cols=5, empty=False):
    """Create a mock pandas DataFrame for testing."""
    if HAS_PANDAS:
        if empty:
            return pd.DataFrame()
        data = {f"col_{i}": np.random.randn(rows) for i in range(cols)}
        return pd.DataFrame(data)
    # Fallback mock
    mock = Mock()
    mock.empty = empty
    mock.shape = (0, 0) if empty else (rows, cols)
    return mock


def _make_concrete_skill(name="test_skill", category=None):
    """Create a concrete MLSkill subclass for testing the abstract base."""
    cat = category or SkillCategory.DATA_PROFILING

    async def _execute(self, X, y=None, **ctx):
        return self._create_result(success=True, data=X)

    # Build the class with execute defined in the namespace so
    # ABCMeta sees the override and allows instantiation.
    cls = type(
        "ConcreteSkill",
        (MLSkill,),
        {
            "name": name,
            "category": cat,
            "execute": _execute,
        },
    )
    return cls


# =============================================================================
# TestSkillCategory
# =============================================================================

@pytest.mark.unit
@skip_no_base
class TestSkillCategory:
    """Tests for SkillCategory enum."""

    def test_data_profiling_value(self):
        assert SkillCategory.DATA_PROFILING.value == "data_profiling"

    def test_data_cleaning_value(self):
        assert SkillCategory.DATA_CLEANING.value == "data_cleaning"

    def test_feature_engineering_value(self):
        assert SkillCategory.FEATURE_ENGINEERING.value == "feature_engineering"

    def test_feature_selection_value(self):
        assert SkillCategory.FEATURE_SELECTION.value == "feature_selection"

    def test_model_selection_value(self):
        assert SkillCategory.MODEL_SELECTION.value == "model_selection"

    def test_hyperparameter_optimization_value(self):
        assert SkillCategory.HYPERPARAMETER_OPTIMIZATION.value == "hyperparameter_optimization"

    def test_ensemble_value(self):
        assert SkillCategory.ENSEMBLE.value == "ensemble"

    def test_evaluation_value(self):
        assert SkillCategory.EVALUATION.value == "evaluation"

    def test_explanation_value(self):
        assert SkillCategory.EXPLANATION.value == "explanation"

    def test_llm_reasoning_value(self):
        assert SkillCategory.LLM_REASONING.value == "llm_reasoning"

    def test_total_category_count(self):
        """All 10 categories should exist."""
        assert len(SkillCategory) == 10

    def test_category_is_enum(self):
        assert isinstance(SkillCategory.DATA_PROFILING, SkillCategory)


# =============================================================================
# TestSkillResult
# =============================================================================

@pytest.mark.unit
@skip_no_base
class TestSkillResult:
    """Tests for SkillResult dataclass."""

    def test_creation_minimal(self):
        result = SkillResult(
            skill_name="test",
            category=SkillCategory.DATA_PROFILING,
            success=True,
        )
        assert result.skill_name == "test"
        assert result.category == SkillCategory.DATA_PROFILING
        assert result.success is True

    def test_defaults_data_none(self):
        result = SkillResult(
            skill_name="test",
            category=SkillCategory.EVALUATION,
            success=False,
        )
        assert result.data is None

    def test_defaults_metrics_empty(self):
        result = SkillResult(
            skill_name="test",
            category=SkillCategory.EVALUATION,
            success=False,
        )
        assert result.metrics == {}

    def test_defaults_metadata_empty(self):
        result = SkillResult(
            skill_name="test",
            category=SkillCategory.EVALUATION,
            success=False,
        )
        assert result.metadata == {}

    def test_defaults_error_none(self):
        result = SkillResult(
            skill_name="test",
            category=SkillCategory.EVALUATION,
            success=False,
        )
        assert result.error is None

    def test_defaults_execution_time_zero(self):
        result = SkillResult(
            skill_name="test",
            category=SkillCategory.EVALUATION,
            success=False,
        )
        assert result.execution_time_seconds == 0.0

    def test_creation_with_all_fields(self):
        result = SkillResult(
            skill_name="full",
            category=SkillCategory.MODEL_SELECTION,
            success=True,
            data={"key": "value"},
            metrics={"score": 0.95},
            metadata={"model": "lgb"},
            error=None,
            execution_time_seconds=1.5,
        )
        assert result.data == {"key": "value"}
        assert result.metrics["score"] == 0.95
        assert result.metadata["model"] == "lgb"
        assert result.execution_time_seconds == 1.5

    def test_to_dict_basic(self):
        result = SkillResult(
            skill_name="test",
            category=SkillCategory.DATA_PROFILING,
            success=True,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["skill_name"] == "test"
        assert d["category"] == "data_profiling"
        assert d["success"] is True

    def test_to_dict_contains_all_expected_keys(self):
        result = SkillResult(
            skill_name="test",
            category=SkillCategory.DATA_PROFILING,
            success=True,
        )
        d = result.to_dict()
        expected_keys = {"skill_name", "category", "success", "metrics", "metadata", "error", "execution_time_seconds"}
        assert expected_keys.issubset(set(d.keys()))

    def test_to_dict_category_is_string_value(self):
        result = SkillResult(
            skill_name="test",
            category=SkillCategory.ENSEMBLE,
            success=True,
        )
        d = result.to_dict()
        assert d["category"] == "ensemble"

    def test_success_pattern(self):
        result = SkillResult(
            skill_name="good",
            category=SkillCategory.DATA_PROFILING,
            success=True,
            data=[1, 2, 3],
            metrics={"accuracy": 0.9},
        )
        assert result.success is True
        assert result.error is None
        assert result.data is not None

    def test_failure_pattern(self):
        result = SkillResult(
            skill_name="bad",
            category=SkillCategory.DATA_PROFILING,
            success=False,
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.data is None

    def test_to_dict_with_metrics(self):
        result = SkillResult(
            skill_name="test",
            category=SkillCategory.FEATURE_ENGINEERING,
            success=True,
            metrics={"n_features": 42, "score": 0.88},
        )
        d = result.to_dict()
        assert d["metrics"]["n_features"] == 42
        assert d["metrics"]["score"] == 0.88


# =============================================================================
# TestMLSkillAbstract
# =============================================================================

@pytest.mark.unit
@skip_no_base
class TestMLSkillAbstract:
    """Tests for the MLSkill abstract base class."""

    def test_cannot_instantiate_directly(self):
        """MLSkill is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            MLSkill()

    def test_concrete_subclass_instantiates(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        assert skill is not None
        assert skill.name == "test_skill"

    def test_concrete_subclass_has_execute(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        assert hasattr(skill, "execute")
        assert callable(skill.execute)

    def test_default_config_empty_dict(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        assert skill.config == {}

    def test_config_passed_through(self):
        ConcreteClass = _make_concrete_skill()
        cfg = {"lr": 0.01, "epochs": 10}
        skill = ConcreteClass(config=cfg)
        assert skill.config == cfg
        assert skill.config["lr"] == 0.01


# =============================================================================
# TestMLSkillInit
# =============================================================================

@pytest.mark.unit
@skip_no_base
class TestMLSkillInit:
    """Tests for MLSkill initialization lifecycle."""

    def test_initialized_flag_false_by_default(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        assert skill._initialized is False

    @pytest.mark.asyncio
    async def test_init_sets_initialized_true(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        await skill.init()
        assert skill._initialized is True

    @pytest.mark.asyncio
    async def test_init_can_be_called_multiple_times(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        await skill.init()
        await skill.init()
        assert skill._initialized is True

    def test_init_not_called_in_constructor(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        assert skill._initialized is False


# =============================================================================
# TestMLSkillValidateInputs
# =============================================================================

@pytest.mark.unit
@skip_no_base
@skip_no_pandas
class TestMLSkillValidateInputs:
    """Tests for MLSkill.validate_inputs()."""

    def test_valid_dataframe(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        df = _make_mock_df(rows=10, cols=5)
        assert skill.validate_inputs(df) is True

    def test_none_input_returns_false(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        assert skill.validate_inputs(None) is False

    def test_numpy_array_is_valid(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        X = np.random.randn(10, 5)
        assert skill.validate_inputs(X) is True

    def test_string_input_returns_false(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        assert skill.validate_inputs("not a dataframe") is False

    def test_with_y_valid(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        df = _make_mock_df(10, 5)
        y = pd.Series(np.random.randint(0, 2, 10))
        assert skill.validate_inputs(df, y=y) is True

    def test_list_input_returns_false(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        assert skill.validate_inputs([1, 2, 3]) is False


# =============================================================================
# TestMLSkillCreateResult
# =============================================================================

@pytest.mark.unit
@skip_no_base
class TestMLSkillCreateResult:
    """Tests for MLSkill._create_result() helper."""

    def test_create_success_result(self):
        ConcreteClass = _make_concrete_skill("my_skill")
        skill = ConcreteClass()
        result = skill._create_result(success=True, data="output")
        assert result.success is True
        assert result.data == "output"
        assert result.skill_name == "my_skill"
        assert result.category == SkillCategory.DATA_PROFILING

    def test_create_result_with_metrics(self):
        ConcreteClass = _make_concrete_skill("my_skill")
        skill = ConcreteClass()
        metrics = {"accuracy": 0.95, "f1": 0.92}
        result = skill._create_result(success=True, metrics=metrics)
        assert result.metrics == {"accuracy": 0.95, "f1": 0.92}

    def test_create_result_with_metadata(self):
        ConcreteClass = _make_concrete_skill("my_skill")
        skill = ConcreteClass()
        metadata = {"model": "lgb", "version": "3.0"}
        result = skill._create_result(success=True, metadata=metadata)
        assert result.metadata == {"model": "lgb", "version": "3.0"}

    def test_create_result_with_error(self):
        ConcreteClass = _make_concrete_skill("my_skill")
        skill = ConcreteClass()
        result = skill._create_result(success=False, error="failed")
        assert result.success is False
        assert result.error == "failed"

    def test_create_result_with_execution_time(self):
        ConcreteClass = _make_concrete_skill("my_skill")
        skill = ConcreteClass()
        result = skill._create_result(success=True, execution_time=2.5)
        assert result.execution_time_seconds == 2.5

    def test_create_result_default_metrics_empty(self):
        ConcreteClass = _make_concrete_skill("my_skill")
        skill = ConcreteClass()
        result = skill._create_result(success=True)
        assert result.metrics == {}


# =============================================================================
# TestMLSkillCreateErrorResult
# =============================================================================

@pytest.mark.unit
@skip_no_base
class TestMLSkillCreateErrorResult:
    """Tests for MLSkill._create_error_result() helper."""

    def test_error_result_has_error_message(self):
        ConcreteClass = _make_concrete_skill("err_skill")
        skill = ConcreteClass()
        result = skill._create_error_result("boom")
        assert result.error == "boom"

    def test_error_result_success_is_false(self):
        ConcreteClass = _make_concrete_skill("err_skill")
        skill = ConcreteClass()
        result = skill._create_error_result("boom")
        assert result.success is False

    def test_error_result_skill_name_set(self):
        ConcreteClass = _make_concrete_skill("err_skill")
        skill = ConcreteClass()
        result = skill._create_error_result("boom")
        assert result.skill_name == "err_skill"

    def test_error_result_data_is_none(self):
        ConcreteClass = _make_concrete_skill("err_skill")
        skill = ConcreteClass()
        result = skill._create_error_result("boom")
        assert result.data is None


# =============================================================================
# TestMLSkillGetInfo
# =============================================================================

@pytest.mark.unit
@skip_no_base
class TestMLSkillGetInfo:
    """Tests for MLSkill.get_info()."""

    def test_get_info_returns_dict(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        info = skill.get_info()
        assert isinstance(info, dict)

    def test_get_info_contains_name(self):
        ConcreteClass = _make_concrete_skill("info_skill")
        skill = ConcreteClass()
        info = skill.get_info()
        assert info["name"] == "info_skill"

    def test_get_info_contains_version(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        info = skill.get_info()
        assert "version" in info
        assert info["version"] == "1.0.0"

    def test_get_info_category_is_string(self):
        ConcreteClass = _make_concrete_skill(category=SkillCategory.ENSEMBLE)
        skill = ConcreteClass()
        info = skill.get_info()
        assert info["category"] == "ensemble"

    def test_get_info_contains_required_inputs(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        info = skill.get_info()
        assert "required_inputs" in info

    def test_get_info_contains_outputs(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        info = skill.get_info()
        assert "outputs" in info

    def test_get_info_contains_requires_llm(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        info = skill.get_info()
        assert "requires_llm" in info

    def test_get_info_contains_requires_gpu(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        info = skill.get_info()
        assert "requires_gpu" in info


# =============================================================================
# TestSkillPipeline
# =============================================================================

@pytest.mark.unit
@skip_no_base
@skip_no_pandas
class TestSkillPipeline:
    """Tests for SkillPipeline sequential execution."""

    def test_create_empty_pipeline(self):
        pipeline = SkillPipeline(skills=[])
        assert len(pipeline.skills) == 0

    def test_add_skills_to_pipeline(self):
        ConcreteClass = _make_concrete_skill("s1")
        s1 = ConcreteClass()
        s2 = ConcreteClass()
        pipeline = SkillPipeline(skills=[s1, s2])
        assert len(pipeline.skills) == 2

    @pytest.mark.asyncio
    async def test_execute_empty_pipeline(self):
        pipeline = SkillPipeline(skills=[])
        df = _make_mock_df(10, 5)
        results = await pipeline.execute(df)
        assert results == []

    @pytest.mark.asyncio
    async def test_execute_single_skill(self):
        ConcreteClass = _make_concrete_skill("single")
        skill = ConcreteClass()
        pipeline = SkillPipeline(skills=[skill])
        df = _make_mock_df(10, 5)
        results = await pipeline.execute(df)
        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_execute_sequential_two_skills(self):
        C1 = _make_concrete_skill("step1")
        C2 = _make_concrete_skill("step2")
        s1, s2 = C1(), C2()
        pipeline = SkillPipeline(skills=[s1, s2])
        df = _make_mock_df(10, 5)
        results = await pipeline.execute(df)
        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_get_final_result_empty(self):
        pipeline = SkillPipeline(skills=[])
        assert pipeline.get_final_result() is None

    @pytest.mark.asyncio
    async def test_get_final_result_after_execute(self):
        ConcreteClass = _make_concrete_skill("final")
        skill = ConcreteClass()
        pipeline = SkillPipeline(skills=[skill])
        df = _make_mock_df(10, 5)
        await pipeline.execute(df)
        final = pipeline.get_final_result()
        assert final is not None
        assert final.skill_name == "final"

    @pytest.mark.asyncio
    async def test_get_all_metrics_empty(self):
        pipeline = SkillPipeline(skills=[])
        df = _make_mock_df(10, 5)
        await pipeline.execute(df)
        assert pipeline.get_all_metrics() == {}

    @pytest.mark.asyncio
    async def test_get_all_metrics_with_results(self):
        """Metrics from successful skills should be aggregated."""
        ConcreteClass = _make_concrete_skill("m_skill")
        skill = ConcreteClass()
        pipeline = SkillPipeline(skills=[skill])
        df = _make_mock_df(10, 5)
        await pipeline.execute(df)
        metrics = pipeline.get_all_metrics()
        assert "m_skill" in metrics

    @pytest.mark.asyncio
    async def test_pipeline_initializes_skills(self):
        ConcreteClass = _make_concrete_skill("init_check")
        skill = ConcreteClass()
        assert skill._initialized is False
        pipeline = SkillPipeline(skills=[skill])
        df = _make_mock_df(10, 5)
        await pipeline.execute(df)
        assert skill._initialized is True


# =============================================================================
# TestSkillRegistry
# =============================================================================

@pytest.mark.unit
@skip_no_base
class TestSkillRegistry:
    """Tests for SkillRegistry class-level registry."""

    def setup_method(self):
        """Clean registry before each test."""
        SkillRegistry._skills = {}

    def test_register_skill(self):
        ConcreteClass = _make_concrete_skill("reg_skill")
        SkillRegistry.register(ConcreteClass)
        assert "reg_skill" in SkillRegistry._skills

    def test_get_registered_skill(self):
        ConcreteClass = _make_concrete_skill("get_skill")
        SkillRegistry.register(ConcreteClass)
        skill = SkillRegistry.get("get_skill")
        assert skill.name == "get_skill"
        assert isinstance(skill, MLSkill)

    def test_get_unknown_skill_raises_keyerror(self):
        with pytest.raises(KeyError):
            SkillRegistry.get("nonexistent")

    def test_get_with_config(self):
        ConcreteClass = _make_concrete_skill("cfg_skill")
        SkillRegistry.register(ConcreteClass)
        skill = SkillRegistry.get("cfg_skill", config={"lr": 0.01})
        assert skill.config == {"lr": 0.01}

    def test_list_skills_empty(self):
        result = SkillRegistry.list_skills()
        assert result == []

    def test_list_skills_returns_info_dicts(self):
        ConcreteClass = _make_concrete_skill("list_skill")
        SkillRegistry.register(ConcreteClass)
        skill_list = SkillRegistry.list_skills()
        assert len(skill_list) == 1
        assert skill_list[0]["name"] == "list_skill"

    def test_get_by_category(self):
        C1 = _make_concrete_skill("cat_a", SkillCategory.DATA_PROFILING)
        C2 = _make_concrete_skill("cat_b", SkillCategory.ENSEMBLE)
        SkillRegistry.register(C1)
        SkillRegistry.register(C2)
        profiling = SkillRegistry.get_by_category(SkillCategory.DATA_PROFILING)
        assert "cat_a" in profiling
        assert "cat_b" not in profiling

    def test_get_by_category_empty(self):
        result = SkillRegistry.get_by_category(SkillCategory.EXPLANATION)
        assert result == []

    def test_register_multiple_skills(self):
        C1 = _make_concrete_skill("multi_1")
        C2 = _make_concrete_skill("multi_2")
        C3 = _make_concrete_skill("multi_3")
        SkillRegistry.register(C1)
        SkillRegistry.register(C2)
        SkillRegistry.register(C3)
        assert len(SkillRegistry._skills) == 3

    def test_register_overwrites_same_name(self):
        C1 = _make_concrete_skill("dupe")
        C2 = _make_concrete_skill("dupe")
        SkillRegistry.register(C1)
        SkillRegistry.register(C2)
        assert len(SkillRegistry._skills) == 1


# =============================================================================
# TestEDASkillMetadata
# =============================================================================

@pytest.mark.unit
@skip_no_eda
class TestEDASkillMetadata:
    """Tests for EDASkill metadata and validation (no execute)."""

    def test_name(self):
        skill = EDASkill()
        assert skill.name == "eda_analysis"

    def test_category(self):
        skill = EDASkill()
        assert skill.category == SkillCategory.DATA_PROFILING

    def test_version(self):
        skill = EDASkill()
        assert skill.version == "1.0.0"

    def test_required_inputs(self):
        skill = EDASkill()
        assert "X" in skill.required_inputs

    def test_optional_inputs_include_y(self):
        skill = EDASkill()
        assert "y" in skill.optional_inputs

    def test_outputs_include_insights(self):
        skill = EDASkill()
        assert "insights" in skill.outputs

    @skip_no_pandas
    def test_validate_inputs_valid_df(self):
        skill = EDASkill()
        df = _make_mock_df(10, 5)
        assert skill.validate_inputs(df) is True

    def test_validate_inputs_none(self):
        skill = EDASkill()
        assert skill.validate_inputs(None) is False

    def test_get_info_structure(self):
        skill = EDASkill()
        info = skill.get_info()
        assert info["name"] == "eda_analysis"
        assert info["category"] == "data_profiling"

    def test_create_error_result(self):
        skill = EDASkill()
        result = skill._create_error_result("EDA failed")
        assert result.success is False
        assert result.error == "EDA failed"
        assert result.skill_name == "eda_analysis"


# =============================================================================
# TestFeatureEngineeringSkillMetadata
# =============================================================================

@pytest.mark.unit
@skip_no_fe
class TestFeatureEngineeringSkillMetadata:
    """Tests for FeatureEngineeringSkill metadata (no execute)."""

    def test_name(self):
        skill = FeatureEngineeringSkill()
        assert skill.name == "feature_engineering"

    def test_category(self):
        skill = FeatureEngineeringSkill()
        assert skill.category == SkillCategory.FEATURE_ENGINEERING

    def test_version(self):
        skill = FeatureEngineeringSkill()
        assert skill.version == "2.0.0"

    def test_required_inputs(self):
        skill = FeatureEngineeringSkill()
        assert "X" in skill.required_inputs

    def test_optional_inputs_include_y(self):
        skill = FeatureEngineeringSkill()
        assert "y" in skill.optional_inputs

    def test_outputs_include_X_engineered(self):
        skill = FeatureEngineeringSkill()
        assert "X_engineered" in skill.outputs

    @skip_no_pandas
    def test_validate_inputs_valid_df(self):
        skill = FeatureEngineeringSkill()
        df = _make_mock_df(10, 5)
        assert skill.validate_inputs(df) is True

    def test_validate_inputs_none(self):
        skill = FeatureEngineeringSkill()
        assert skill.validate_inputs(None) is False

    def test_get_info_structure(self):
        skill = FeatureEngineeringSkill()
        info = skill.get_info()
        assert info["name"] == "feature_engineering"
        assert info["category"] == "feature_engineering"

    def test_create_error_result(self):
        skill = FeatureEngineeringSkill()
        result = skill._create_error_result("FE failed")
        assert result.success is False
        assert result.error == "FE failed"
        assert result.skill_name == "feature_engineering"

    def test_techniques_used_initialized_empty(self):
        skill = FeatureEngineeringSkill()
        assert skill._techniques_used == []


# =============================================================================
# TestModelSelectionSkillMetadata
# =============================================================================

@pytest.mark.unit
@skip_no_ms
class TestModelSelectionSkillMetadata:
    """Tests for ModelSelectionSkill metadata (no execute)."""

    def test_name(self):
        skill = ModelSelectionSkill()
        assert skill.name == "model_selection"

    def test_category(self):
        skill = ModelSelectionSkill()
        assert skill.category == SkillCategory.MODEL_SELECTION

    def test_version(self):
        skill = ModelSelectionSkill()
        assert skill.version == "2.0.0"

    def test_required_inputs_include_X(self):
        skill = ModelSelectionSkill()
        assert "X" in skill.required_inputs

    def test_required_inputs_include_y(self):
        skill = ModelSelectionSkill()
        assert "y" in skill.required_inputs

    def test_outputs_include_best_model(self):
        skill = ModelSelectionSkill()
        assert "best_model" in skill.outputs

    @skip_no_pandas
    def test_validate_inputs_valid_df(self):
        skill = ModelSelectionSkill()
        df = _make_mock_df(10, 5)
        assert skill.validate_inputs(df) is True

    def test_validate_inputs_none(self):
        skill = ModelSelectionSkill()
        assert skill.validate_inputs(None) is False

    def test_get_info_structure(self):
        skill = ModelSelectionSkill()
        info = skill.get_info()
        assert info["name"] == "model_selection"
        assert info["category"] == "model_selection"

    def test_create_error_result(self):
        skill = ModelSelectionSkill()
        result = skill._create_error_result("MS failed")
        assert result.success is False
        assert result.error == "MS failed"
        assert result.skill_name == "model_selection"


# =============================================================================
# TestEnsembleSkillMetadata
# =============================================================================

@pytest.mark.unit
@skip_no_ens
class TestEnsembleSkillMetadata:
    """Tests for EnsembleSkill metadata (no execute)."""

    def test_name(self):
        skill = EnsembleSkill()
        assert skill.name == "ensemble"

    def test_category(self):
        skill = EnsembleSkill()
        assert skill.category == SkillCategory.ENSEMBLE

    def test_version(self):
        skill = EnsembleSkill()
        assert skill.version == "2.0.0"

    def test_required_inputs_include_X(self):
        skill = EnsembleSkill()
        assert "X" in skill.required_inputs

    def test_required_inputs_include_y(self):
        skill = EnsembleSkill()
        assert "y" in skill.required_inputs

    def test_outputs_include_final_model(self):
        skill = EnsembleSkill()
        assert "final_model" in skill.outputs

    @skip_no_pandas
    def test_validate_inputs_valid_df(self):
        skill = EnsembleSkill()
        df = _make_mock_df(10, 5)
        assert skill.validate_inputs(df) is True

    def test_validate_inputs_none(self):
        skill = EnsembleSkill()
        assert skill.validate_inputs(None) is False

    def test_get_info_structure(self):
        skill = EnsembleSkill()
        info = skill.get_info()
        assert info["name"] == "ensemble"
        assert info["category"] == "ensemble"

    def test_create_error_result(self):
        skill = EnsembleSkill()
        result = skill._create_error_result("Ens failed")
        assert result.success is False
        assert result.error == "Ens failed"
        assert result.skill_name == "ensemble"


# =============================================================================
# TestBacktestMetrics
# =============================================================================

@pytest.mark.unit
@skip_no_bt_report
class TestBacktestMetrics:
    """Tests for BacktestMetrics dataclass."""

    def test_creation_defaults(self):
        m = BacktestMetrics()
        assert m.total_return == 0.0
        assert m.annual_return == 0.0
        assert m.volatility == 0.0
        assert m.sharpe_ratio == 0.0

    def test_sortino_ratio_default(self):
        m = BacktestMetrics()
        assert m.sortino_ratio == 0.0

    def test_calmar_ratio_default(self):
        m = BacktestMetrics()
        assert m.calmar_ratio == 0.0

    def test_max_drawdown_default(self):
        m = BacktestMetrics()
        assert m.max_drawdown == 0.0

    def test_romad_default(self):
        m = BacktestMetrics()
        assert m.romad == 0.0

    def test_win_rate_default(self):
        m = BacktestMetrics()
        assert m.win_rate == 0.0

    def test_profit_factor_default(self):
        m = BacktestMetrics()
        assert m.profit_factor == 0.0

    def test_sqn_default(self):
        m = BacktestMetrics()
        assert m.sqn == 0.0

    def test_creation_custom_values(self):
        m = BacktestMetrics(
            total_return=25.5,
            annual_return=12.0,
            volatility=18.0,
            sharpe_ratio=1.5,
            max_drawdown=-10.0,
            win_rate=55.0,
        )
        assert m.total_return == 25.5
        assert m.annual_return == 12.0
        assert m.volatility == 18.0
        assert m.sharpe_ratio == 1.5
        assert m.max_drawdown == -10.0
        assert m.win_rate == 55.0

    def test_all_fields_are_float(self):
        m = BacktestMetrics()
        for f in fields(m):
            if f.name not in ("max_drawdown_duration",):
                assert isinstance(getattr(m, f.name), float), f"Field {f.name} should be float"


# =============================================================================
# TestTradeStatistics
# =============================================================================

@pytest.mark.unit
@skip_no_bt_report
class TestTradeStatistics:
    """Tests for TradeStatistics dataclass."""

    def test_creation_defaults(self):
        ts = TradeStatistics()
        assert ts.total_trades == 0
        assert ts.winning_trades == 0

    def test_avg_win_default(self):
        ts = TradeStatistics()
        assert ts.avg_win == 0.0

    def test_avg_loss_default(self):
        ts = TradeStatistics()
        assert ts.avg_loss == 0.0

    def test_largest_win_default(self):
        ts = TradeStatistics()
        assert ts.largest_win == 0.0

    def test_largest_loss_default(self):
        ts = TradeStatistics()
        assert ts.largest_loss == 0.0

    def test_consecutive_wins_default(self):
        ts = TradeStatistics()
        assert ts.consecutive_wins == 0

    def test_consecutive_losses_default(self):
        ts = TradeStatistics()
        assert ts.consecutive_losses == 0

    def test_expectancy_default(self):
        ts = TradeStatistics()
        assert ts.expectancy == 0.0

    def test_creation_custom_values(self):
        ts = TradeStatistics(
            total_trades=100,
            winning_trades=55,
            avg_win=2.5,
            avg_loss=-1.5,
            largest_win=15.0,
            largest_loss=-8.0,
            consecutive_wins=7,
            consecutive_losses=4,
            expectancy=0.65,
        )
        assert ts.total_trades == 100
        assert ts.winning_trades == 55
        assert ts.avg_win == 2.5
        assert ts.avg_loss == -1.5
        assert ts.largest_win == 15.0
        assert ts.largest_loss == -8.0
        assert ts.consecutive_wins == 7
        assert ts.consecutive_losses == 4
        assert ts.expectancy == 0.65

    def test_losing_trades_default(self):
        ts = TradeStatistics()
        assert ts.losing_trades == 0


# =============================================================================
# TestTransactionCosts
# =============================================================================

@pytest.mark.unit
@skip_no_bt_engine
class TestTransactionCosts:
    """Tests for TransactionCosts dataclass."""

    def test_default_commission(self):
        tc = TransactionCosts()
        assert tc.commission_pct == 0.001

    def test_default_slippage(self):
        tc = TransactionCosts()
        assert tc.slippage_pct == 0.001

    def test_default_market_impact(self):
        tc = TransactionCosts()
        assert tc.market_impact_pct == 0.0005

    def test_default_min_commission(self):
        tc = TransactionCosts()
        assert tc.min_commission == 0.0

    def test_total_cost_pct_default(self):
        tc = TransactionCosts()
        # Round-trip cost = 2 * (0.001 + 0.001 + 0.0005) = 2 * 0.0025 = 0.005
        expected = 2 * (0.001 + 0.001 + 0.0005)
        assert abs(tc.total_cost_pct() - expected) < 1e-10

    def test_total_cost_pct_custom(self):
        tc = TransactionCosts(
            commission_pct=0.002,
            slippage_pct=0.003,
            market_impact_pct=0.001,
        )
        expected = 2 * (0.002 + 0.003 + 0.001)
        assert abs(tc.total_cost_pct() - expected) < 1e-10

    def test_total_cost_pct_zero(self):
        tc = TransactionCosts(
            commission_pct=0.0,
            slippage_pct=0.0,
            market_impact_pct=0.0,
        )
        assert tc.total_cost_pct() == 0.0

    def test_apply_costs(self):
        tc = TransactionCosts(
            commission_pct=0.001,
            slippage_pct=0.001,
            market_impact_pct=0.0,
        )
        returns = np.array([0.01, 0.02, -0.01])
        trades = np.array([1.0, 0.0, 1.0])
        adjusted = tc.apply_costs(returns, trades)
        assert len(adjusted) == 3
        # Where trades == 0, cost should be 0, returns unchanged
        assert adjusted[1] == returns[1]
        # Where trades != 0, returns should be reduced
        assert adjusted[0] < returns[0]

    def test_custom_creation(self):
        tc = TransactionCosts(
            commission_pct=0.005,
            slippage_pct=0.002,
            market_impact_pct=0.001,
            min_commission=1.0,
        )
        assert tc.commission_pct == 0.005
        assert tc.slippage_pct == 0.002
        assert tc.market_impact_pct == 0.001
        assert tc.min_commission == 1.0


# =============================================================================
# TestRiskMetrics
# =============================================================================

@pytest.mark.unit
@skip_no_bt_engine
class TestRiskMetrics:
    """Tests for RiskMetrics dataclass."""

    def test_creation_defaults(self):
        rm = RiskMetrics()
        assert rm.volatility == 0.0
        assert rm.max_drawdown == 0.0
        assert rm.var_95 == 0.0
        assert rm.var_99 == 0.0

    def test_cvar_defaults(self):
        rm = RiskMetrics()
        assert rm.cvar_95 == 0.0
        assert rm.cvar_99 == 0.0

    def test_tail_risk_defaults(self):
        rm = RiskMetrics()
        assert rm.skewness == 0.0
        assert rm.kurtosis == 0.0
        assert rm.tail_ratio == 0.0

    def test_custom_values(self):
        rm = RiskMetrics(
            volatility=22.5,
            max_drawdown=-12.5,
            var_95=-2.5,
            cvar_95=-3.2,
        )
        assert rm.volatility == 22.5
        assert rm.max_drawdown == -12.5
        assert rm.var_95 == -2.5
        assert rm.cvar_95 == -3.2


# =============================================================================
# TestModelResults
# =============================================================================

@pytest.mark.unit
@skip_no_bt_report
class TestModelResults:
    """Tests for ModelResults dataclass."""

    def test_creation_defaults(self):
        mr = ModelResults()
        assert mr.name == ""
        assert mr.accuracy == 0.0
        assert mr.f1_score == 0.0
        assert mr.auc == 0.0
        assert mr.is_best is False

    def test_custom_values(self):
        mr = ModelResults(
            name="lightgbm",
            accuracy=0.85,
            f1_score=0.82,
            auc=0.90,
            is_best=True,
            feature_importance={"col_0": 0.5, "col_1": 0.3},
        )
        assert mr.name == "lightgbm"
        assert mr.accuracy == 0.85
        assert mr.is_best is True
        assert mr.feature_importance["col_0"] == 0.5


# =============================================================================
# TestBacktestResult
# =============================================================================

@pytest.mark.unit
@skip_no_bt_report
class TestBacktestResult:
    """Tests for BacktestResult dataclass."""

    def test_creation_defaults(self):
        br = BacktestResult()
        assert br.symbol == ""
        assert br.problem_type == "classification"
        assert br.target_days == 1

    def test_default_strategy_metrics(self):
        br = BacktestResult()
        assert isinstance(br.strategy_metrics, BacktestMetrics)
        assert br.strategy_metrics.total_return == 0.0

    def test_default_trade_stats(self):
        br = BacktestResult()
        assert isinstance(br.trade_stats, TradeStatistics)
        assert br.trade_stats.total_trades == 0

    def test_custom_symbol(self):
        br = BacktestResult(symbol="RELIANCE")
        assert br.symbol == "RELIANCE"


# =============================================================================
# TestComprehensiveBacktestResult
# =============================================================================

@pytest.mark.unit
@skip_no_bt_engine
class TestComprehensiveBacktestResult:
    """Tests for ComprehensiveBacktestResult dataclass."""

    def test_creation_defaults(self):
        cbr = ComprehensiveBacktestResult()
        assert cbr.symbol == ""
        assert cbr.total_return == 0.0
        assert cbr.sharpe_ratio == 0.0

    def test_risk_metrics_default(self):
        cbr = ComprehensiveBacktestResult()
        assert isinstance(cbr.risk_metrics, RiskMetrics)

    def test_transaction_costs_default(self):
        cbr = ComprehensiveBacktestResult()
        assert isinstance(cbr.transaction_costs, TransactionCosts)

    def test_monte_carlo_default(self):
        cbr = ComprehensiveBacktestResult()
        assert isinstance(cbr.monte_carlo, MonteCarloResult)

    def test_custom_values(self):
        cbr = ComprehensiveBacktestResult(
            symbol="TCS",
            total_return=45.5,
            sharpe_ratio=1.45,
        )
        assert cbr.symbol == "TCS"
        assert cbr.total_return == 45.5
        assert cbr.sharpe_ratio == 1.45


# =============================================================================
# TestWalkForwardResult
# =============================================================================

@pytest.mark.unit
@skip_no_bt_engine
class TestWalkForwardResult:
    """Tests for WalkForwardResult dataclass."""

    def test_creation_defaults(self):
        wf = WalkForwardResult()
        assert wf.train_start == ""
        assert wf.is_return == 0.0
        assert wf.oos_return == 0.0
        assert wf.sharpe_degradation == 0.0

    def test_custom_values(self):
        wf = WalkForwardResult(
            train_start="2022-01-01",
            train_end="2023-01-01",
            test_start="2023-01-01",
            test_end="2023-06-01",
            is_return=15.0,
            oos_return=8.0,
            sharpe_degradation=25.0,
        )
        assert wf.is_return == 15.0
        assert wf.oos_return == 8.0


# =============================================================================
# TestMonteCarloResult
# =============================================================================

@pytest.mark.unit
@skip_no_bt_engine
class TestMonteCarloResult:
    """Tests for MonteCarloResult dataclass."""

    def test_creation_defaults(self):
        mc = MonteCarloResult()
        assert mc.n_simulations == 0
        assert mc.mean_return == 0.0
        assert mc.prob_positive == 0.0

    def test_custom_values(self):
        mc = MonteCarloResult(
            n_simulations=1000,
            mean_return=44.2,
            prob_positive=0.92,
            prob_sharpe_above_1=0.78,
        )
        assert mc.n_simulations == 1000
        assert mc.mean_return == 44.2
        assert mc.prob_positive == 0.92


# =============================================================================
# TestPositionSizing
# =============================================================================

@pytest.mark.unit
@skip_no_bt_engine
class TestPositionSizing:
    """Tests for PositionSizing dataclass."""

    def test_creation_defaults(self):
        ps = PositionSizing()
        assert ps.kelly_fraction == 0.0
        assert ps.half_kelly == 0.0
        assert ps.target_volatility == 0.10
        assert ps.vol_scalar == 1.0

    def test_custom_values(self):
        ps = PositionSizing(
            kelly_fraction=0.25,
            half_kelly=0.125,
            target_volatility=0.15,
        )
        assert ps.kelly_fraction == 0.25
        assert ps.half_kelly == 0.125
        assert ps.target_volatility == 0.15


# =============================================================================
# TestFactorExposure
# =============================================================================

@pytest.mark.unit
@skip_no_bt_engine
class TestFactorExposure:
    """Tests for FactorExposure dataclass."""

    def test_creation_defaults(self):
        fe = FactorExposure()
        assert fe.alpha == 0.0
        assert fe.market_beta == 0.0
        assert fe.r_squared == 0.0

    def test_custom_values(self):
        fe = FactorExposure(
            alpha=5.0,
            market_beta=0.85,
            r_squared=0.72,
            information_ratio=1.2,
        )
        assert fe.alpha == 5.0
        assert fe.market_beta == 0.85
        assert fe.information_ratio == 1.2


# =============================================================================
# TestStatisticalTests
# =============================================================================

@pytest.mark.unit
@skip_no_bt_engine
class TestStatisticalTests:
    """Tests for StatisticalTests dataclass."""

    def test_creation_defaults(self):
        st = StatisticalTests()
        assert st.t_statistic == 0.0
        assert st.p_value == 0.0
        assert st.deflated_sharpe == 0.0

    def test_custom_values(self):
        st = StatisticalTests(
            t_statistic=2.85,
            p_value=0.002,
            prob_sharpe_zero=0.98,
        )
        assert st.t_statistic == 2.85
        assert st.p_value == 0.002


# =============================================================================
# TestRegimeAnalysis
# =============================================================================

@pytest.mark.unit
@skip_no_bt_engine
class TestRegimeAnalysis:
    """Tests for RegimeAnalysis dataclass."""

    def test_creation_defaults(self):
        ra = RegimeAnalysis()
        assert ra.current_regime == "normal"
        assert ra.regime_changes == 0
        assert ra.time_in_bull == 0.0

    def test_custom_values(self):
        ra = RegimeAnalysis(
            current_regime="bull",
            time_in_bull=60.0,
            time_in_bear=25.0,
            time_in_crisis=15.0,
        )
        assert ra.current_regime == "bull"
        assert ra.time_in_bull == 60.0


# =============================================================================
# TestSkillResultIntegration
# =============================================================================

@pytest.mark.unit
@skip_no_base
class TestSkillResultIntegration:
    """Integration tests for SkillResult creation across different skill categories."""

    def test_create_results_from_different_categories(self):
        """Create SkillResults with different categories and compare."""
        r1 = SkillResult(
            skill_name="eda",
            category=SkillCategory.DATA_PROFILING,
            success=True,
            metrics={"n_features": 10},
        )
        r2 = SkillResult(
            skill_name="model_sel",
            category=SkillCategory.MODEL_SELECTION,
            success=True,
            metrics={"best_score": 0.95},
        )
        assert r1.category != r2.category
        assert r1.skill_name != r2.skill_name

    def test_results_independent_metrics(self):
        r1 = SkillResult(
            skill_name="a",
            category=SkillCategory.DATA_PROFILING,
            success=True,
            metrics={"x": 1},
        )
        r2 = SkillResult(
            skill_name="b",
            category=SkillCategory.DATA_PROFILING,
            success=True,
            metrics={"y": 2},
        )
        # Ensure each result has its own metrics dict
        assert "x" not in r2.metrics
        assert "y" not in r1.metrics

    def test_success_and_failure_pattern(self):
        success = SkillResult(
            skill_name="ok",
            category=SkillCategory.EVALUATION,
            success=True,
            data={"score": 0.9},
        )
        failure = SkillResult(
            skill_name="fail",
            category=SkillCategory.EVALUATION,
            success=False,
            error="out of memory",
        )
        assert success.success is True
        assert failure.success is False
        assert success.error is None
        assert failure.error is not None

    def test_to_dict_serialization_consistency(self):
        result = SkillResult(
            skill_name="serial",
            category=SkillCategory.FEATURE_SELECTION,
            success=True,
            metrics={"k": 5},
            metadata={"method": "shap"},
        )
        d = result.to_dict()
        assert d["skill_name"] == "serial"
        assert d["category"] == "feature_selection"
        assert d["metrics"]["k"] == 5
        assert d["metadata"]["method"] == "shap"

    def test_result_with_complex_data(self):
        data = {"features": ["a", "b", "c"], "scores": [0.9, 0.8, 0.7]}
        result = SkillResult(
            skill_name="complex",
            category=SkillCategory.FEATURE_ENGINEERING,
            success=True,
            data=data,
        )
        assert result.data["features"] == ["a", "b", "c"]

    def test_multiple_results_different_times(self):
        r1 = SkillResult(
            skill_name="fast",
            category=SkillCategory.DATA_PROFILING,
            success=True,
            execution_time_seconds=0.1,
        )
        r2 = SkillResult(
            skill_name="slow",
            category=SkillCategory.MODEL_SELECTION,
            success=True,
            execution_time_seconds=30.0,
        )
        assert r2.execution_time_seconds > r1.execution_time_seconds


# =============================================================================
# TestSkillPipelineIntegration
# =============================================================================

@pytest.mark.unit
@skip_no_base
@skip_no_pandas
class TestSkillPipelineIntegration:
    """Integration tests for SkillPipeline with mocked skills."""

    @pytest.mark.asyncio
    async def test_pipeline_three_skills(self):
        """Pipeline of three sequential skills should produce three results."""
        C1 = _make_concrete_skill("step_a")
        C2 = _make_concrete_skill("step_b")
        C3 = _make_concrete_skill("step_c")
        pipeline = SkillPipeline(skills=[C1(), C2(), C3()])
        df = _make_mock_df(10, 5)
        results = await pipeline.execute(df)
        assert len(results) == 3
        names = [r.skill_name for r in results]
        assert names == ["step_a", "step_b", "step_c"]

    @pytest.mark.asyncio
    async def test_pipeline_final_result_is_last(self):
        C1 = _make_concrete_skill("first")
        C2 = _make_concrete_skill("last")
        pipeline = SkillPipeline(skills=[C1(), C2()])
        df = _make_mock_df(10, 5)
        await pipeline.execute(df)
        final = pipeline.get_final_result()
        assert final.skill_name == "last"

    @pytest.mark.asyncio
    async def test_pipeline_all_succeed(self):
        skills = [_make_concrete_skill(f"s{i}")() for i in range(4)]
        pipeline = SkillPipeline(skills=skills)
        df = _make_mock_df(10, 5)
        results = await pipeline.execute(df)
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_pipeline_metrics_aggregation(self):
        C1 = _make_concrete_skill("m1")
        C2 = _make_concrete_skill("m2")
        pipeline = SkillPipeline(skills=[C1(), C2()])
        df = _make_mock_df(10, 5)
        await pipeline.execute(df)
        all_metrics = pipeline.get_all_metrics()
        assert "m1" in all_metrics
        assert "m2" in all_metrics

    @pytest.mark.asyncio
    async def test_pipeline_with_failing_skill(self):
        """A skill that returns failure should still be in results."""
        class FailSkill(MLSkill):
            name = "fail_skill"
            category = SkillCategory.EVALUATION

            async def execute(self, X, y=None, **ctx):
                return self._create_error_result("intentional failure")

        ConcreteOK = _make_concrete_skill("ok_skill")
        pipeline = SkillPipeline(skills=[ConcreteOK(), FailSkill()])
        df = _make_mock_df(10, 5)
        results = await pipeline.execute(df)
        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
        assert results[1].error == "intentional failure"

    @pytest.mark.asyncio
    async def test_pipeline_failing_skill_excluded_from_metrics(self):
        """get_all_metrics only includes successful skills."""
        class FailSkill(MLSkill):
            name = "fail_metrics"
            category = SkillCategory.EVALUATION

            async def execute(self, X, y=None, **ctx):
                return self._create_error_result("boom")

        ConcreteOK = _make_concrete_skill("ok_metrics")
        pipeline = SkillPipeline(skills=[ConcreteOK(), FailSkill()])
        df = _make_mock_df(10, 5)
        await pipeline.execute(df)
        all_metrics = pipeline.get_all_metrics()
        assert "ok_metrics" in all_metrics
        assert "fail_metrics" not in all_metrics


# =============================================================================
# TestSkillResourceHints
# =============================================================================

@pytest.mark.unit
@skip_no_base
class TestSkillResourceHints:
    """Tests for MLSkill resource hint attributes."""

    def test_default_requires_llm_false(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        assert skill.requires_llm is False

    def test_default_requires_gpu_false(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        assert skill.requires_gpu is False

    def test_default_estimated_memory(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        assert skill.estimated_memory_gb == 1.0

    def test_get_info_includes_resource_hints(self):
        ConcreteClass = _make_concrete_skill()
        skill = ConcreteClass()
        info = skill.get_info()
        assert info["requires_llm"] is False
        assert info["requires_gpu"] is False


# =============================================================================
# TestSkillDescriptions
# =============================================================================

@pytest.mark.unit
@skip_no_base
class TestSkillDescriptions:
    """Tests for skill description metadata across all skills."""

    @skip_no_eda
    def test_eda_description(self):
        skill = EDASkill()
        assert len(skill.description) > 0

    @skip_no_fe
    def test_fe_description(self):
        skill = FeatureEngineeringSkill()
        assert len(skill.description) > 0

    @skip_no_ms
    def test_ms_description(self):
        skill = ModelSelectionSkill()
        assert len(skill.description) > 0

    @skip_no_ens
    def test_ensemble_description(self):
        skill = EnsembleSkill()
        assert len(skill.description) > 0


# =============================================================================
# TestBacktestMetricsFieldCount
# =============================================================================

@pytest.mark.unit
@skip_no_bt_report
class TestBacktestMetricsFieldCount:
    """Tests for BacktestMetrics field completeness."""

    def test_has_expectancy_field(self):
        m = BacktestMetrics()
        assert hasattr(m, "expectancy")
        assert m.expectancy == 0.0

    def test_has_avg_drawdown_field(self):
        m = BacktestMetrics()
        assert hasattr(m, "avg_drawdown")
        assert m.avg_drawdown == 0.0

    def test_has_max_drawdown_duration_field(self):
        m = BacktestMetrics()
        assert hasattr(m, "max_drawdown_duration")
        assert m.max_drawdown_duration == 0


# =============================================================================
# TestTradeStatisticsFieldCount
# =============================================================================

@pytest.mark.unit
@skip_no_bt_report
class TestTradeStatisticsFieldCount:
    """Tests for TradeStatistics field completeness."""

    def test_has_avg_trade_duration(self):
        ts = TradeStatistics()
        assert hasattr(ts, "avg_trade_duration")
        assert ts.avg_trade_duration == 0.0

    def test_has_losing_trades(self):
        ts = TradeStatistics()
        assert hasattr(ts, "losing_trades")
        assert ts.losing_trades == 0


# =============================================================================
# TestCrossSkillCreateResult
# =============================================================================

@pytest.mark.unit
@skip_no_base
class TestCrossSkillCreateResult:
    """Test _create_result and _create_error_result across skill types."""

    @skip_no_eda
    def test_eda_create_result_success(self):
        skill = EDASkill()
        r = skill._create_result(
            success=True,
            data={"insights": "analysis done"},
            metrics={"n_features": 20},
        )
        assert r.skill_name == "eda_analysis"
        assert r.category == SkillCategory.DATA_PROFILING
        assert r.success is True

    @skip_no_fe
    def test_fe_create_result_success(self):
        skill = FeatureEngineeringSkill()
        r = skill._create_result(
            success=True,
            metrics={"engineered_features": 50},
        )
        assert r.skill_name == "feature_engineering"
        assert r.category == SkillCategory.FEATURE_ENGINEERING

    @skip_no_ms
    def test_ms_create_error_result(self):
        skill = ModelSelectionSkill()
        r = skill._create_error_result("No target variable")
        assert r.success is False
        assert r.error == "No target variable"
        assert r.skill_name == "model_selection"

    @skip_no_ens
    def test_ensemble_create_error_result(self):
        skill = EnsembleSkill()
        r = skill._create_error_result("Ensemble failed")
        assert r.success is False
        assert r.skill_name == "ensemble"
        assert r.category == SkillCategory.ENSEMBLE

"""
Test Evaluation Framework

Tests reproducibility, benchmarks, evaluation protocol, ablation studies,
and ConfigTuner hyperparameter tuning.
"""
import sys
import random
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add Jotty to path
jotty_path = Path(__file__).parent.parent
sys.path.insert(0, str(jotty_path))

from core.evaluation import (
    ReproducibilityConfig,
    set_reproducible_seeds,
    CustomBenchmark,
    EvaluationProtocol,
    EvaluationReport,
    AblationStudy,
    ComponentType,
    ConfigTuner,
    ConfigSearchGroup,
    ConfigTrialResult,
    TuningResult,
    DEFAULT_SEARCH_GROUPS,
)
from core.evaluation.benchmark import BenchmarkMetrics
from core.foundation.data_structures import SwarmConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_benchmark(tasks=None):
    """Create a simple benchmark for testing."""
    if tasks is None:
        tasks = [{"id": "t1", "question": "2+2?", "answer": "4"}]
    return CustomBenchmark(name="test", tasks=tasks)


def _make_agent_factory(answer="4"):
    """Return a factory that builds a trivial agent returning *answer*."""
    def factory(config):
        class _Agent:
            def run(self, question):
                return answer
        return _Agent()
    return factory


def _make_report(pass_rate=0.8, cost=0.5, time=1.0, n_runs=3):
    """Build a synthetic EvaluationReport for unit-testing score math."""
    return EvaluationReport(
        benchmark_name="test",
        n_runs=n_runs,
        mean_pass_rate=pass_rate,
        std_pass_rate=0.01,
        mean_cost=cost,
        std_cost=0.01,
        mean_execution_time=time,
        std_execution_time=0.1,
    )


# ---------------------------------------------------------------------------
# Existing tests (converted to pytest style)
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Tests for reproducibility framework."""

    @pytest.mark.unit
    def test_same_seed_same_value(self):
        """Same seed produces same random value."""
        set_reproducible_seeds(random_seed=42)
        v1 = random.random()
        set_reproducible_seeds(random_seed=42)
        v2 = random.random()
        assert v1 == v2

    @pytest.mark.unit
    def test_config_stores_seed(self):
        """ReproducibilityConfig stores the given seed."""
        config = ReproducibilityConfig(random_seed=42)
        assert config.random_seed == 42


class TestCustomBenchmark:
    """Tests for CustomBenchmark."""

    @pytest.mark.unit
    def test_perfect_pass_rate(self):
        """Agent that returns correct answer gets 100% pass rate."""
        benchmark = _make_benchmark()

        class Agent:
            def run(self, question):
                return "4"

        metrics = benchmark.evaluate(Agent())
        assert metrics.total_tasks == 1
        assert metrics.successful_tasks == 1
        assert metrics.pass_rate == 1.0


class TestEvaluationProtocol:
    """Tests for EvaluationProtocol."""

    @pytest.mark.unit
    def test_multi_run(self):
        """Protocol runs the benchmark n_runs times."""
        benchmark = _make_benchmark()

        class Agent:
            def run(self, question):
                return "4"

        protocol = EvaluationProtocol(benchmark=benchmark, n_runs=3, random_seed=42)
        report = protocol.evaluate(Agent(), save_results=False)

        assert report.n_runs == 3
        assert report.mean_pass_rate > 0
        assert len(report.runs) == 3


class TestAblationStudy:
    """Tests for AblationStudy."""

    @pytest.mark.unit
    def test_component_contribution(self):
        """Ablation study detects component contribution."""
        benchmark = _make_benchmark()

        def create_agent(config):
            class Agent:
                def run(self, question):
                    if hasattr(config, 'enable_rl') and config.enable_rl:
                        return "4"
                    return "5"
            return Agent()

        baseline_config = SwarmConfig(enable_rl=True)
        components = [
            {
                "name": "learning",
                "type": ComponentType.FEATURE,
                "disable": lambda c: setattr(c, 'enable_rl', False),
            },
        ]

        study = AblationStudy(
            benchmark=benchmark,
            agent_factory=create_agent,
            components=components,
            n_runs=2,
            random_seed=42,
            baseline_config=baseline_config,
        )
        result = study.run()

        assert len(result.component_contributions) == 1
        assert result.baseline_report.mean_pass_rate > 0


# ---------------------------------------------------------------------------
# ConfigTuner tests
# ---------------------------------------------------------------------------

class TestConfigSearchGroup:
    """Tests for ConfigSearchGroup dataclass."""

    @pytest.mark.unit
    def test_to_dict(self):
        """to_dict serializes all fields."""
        group = ConfigSearchGroup(
            name="test_group",
            params={"gamma": [0.9, 0.99]},
            target_metric="pass_rate",
        )
        d = group.to_dict()
        assert d["name"] == "test_group"
        assert "gamma" in d["params"]
        assert d["target_metric"] == "pass_rate"


class TestConfigTrialResult:
    """Tests for ConfigTrialResult dataclass."""

    @pytest.mark.unit
    def test_to_dict(self):
        """to_dict includes all fields."""
        report = _make_report()
        trial = ConfigTrialResult(
            config_overrides={"gamma": 0.95},
            group_name="td_lambda_core",
            report=report,
            composite_score=0.42,
        )
        d = trial.to_dict()
        assert d["config_overrides"] == {"gamma": 0.95}
        assert d["group_name"] == "td_lambda_core"
        assert d["composite_score"] == 0.42
        assert "report" in d


class TestTuningResult:
    """Tests for TuningResult dataclass."""

    @pytest.mark.unit
    def test_to_dict_no_best(self):
        """to_dict handles best_trial=None."""
        report = _make_report()
        result = TuningResult(
            baseline_report=report,
            trials=[],
            best_trial=None,
            recommendations=["No improvement found."],
        )
        d = result.to_dict()
        assert d["best_trial"] is None
        assert len(d["recommendations"]) == 1

    @pytest.mark.unit
    def test_to_dict_with_best(self):
        """to_dict serializes best_trial when present."""
        report = _make_report()
        trial = ConfigTrialResult(
            config_overrides={"gamma": 0.95},
            group_name="g",
            report=report,
            composite_score=0.5,
        )
        result = TuningResult(
            baseline_report=report,
            trials=[trial],
            best_trial=trial,
        )
        d = result.to_dict()
        assert d["best_trial"]["composite_score"] == 0.5


class TestConfigTuner:
    """Tests for ConfigTuner hyperparameter tuning."""

    def _simple_tuner(self, groups=None, strategy="grid", max_trials=50):
        """Build a ConfigTuner with a trivial always-correct agent."""
        benchmark = _make_benchmark()
        return ConfigTuner(
            benchmark=benchmark,
            agent_factory=_make_agent_factory("4"),
            search_groups=groups or [
                ConfigSearchGroup(
                    name="small",
                    params={"gamma": [0.9, 0.99]},
                ),
            ],
            n_runs=1,
            strategy=strategy,
            max_trials=max_trials,
            random_seed=42,
        )

    # -- constraint validation --

    @pytest.mark.unit
    def test_constraint_rejects_invalid(self):
        """Constraints filter out invalid combos (alpha_min > alpha_max)."""
        group = ConfigSearchGroup(
            name="constrained",
            params={"alpha_min": [0.1, 0.001], "alpha_max": [0.05, 0.2]},
            constraints=[lambda c: c["alpha_min"] < c["alpha_max"]],
        )
        trials = ConfigTuner._check_constraints(
            {"alpha_min": 0.1, "alpha_max": 0.05},
            group.constraints,
        )
        assert trials is False

    @pytest.mark.unit
    def test_constraint_accepts_valid(self):
        """Valid overrides pass constraint checks."""
        constraints = [lambda c: c["alpha_min"] < c["alpha_max"]]
        assert ConfigTuner._check_constraints(
            {"alpha_min": 0.001, "alpha_max": 0.2}, constraints
        )

    # -- trial generation --

    @pytest.mark.unit
    def test_grid_generates_cartesian_product(self):
        """Grid strategy produces full cartesian product."""
        tuner = self._simple_tuner(
            groups=[
                ConfigSearchGroup(
                    name="grid_test",
                    params={"a": [1, 2], "b": [10, 20, 30]},
                ),
            ],
            strategy="grid",
        )
        group = tuner.search_groups[0]
        trials = tuner._generate_trials(group)
        assert len(trials) == 2 * 3  # 6 combos

    @pytest.mark.unit
    def test_grid_with_constraints_filters(self):
        """Grid strategy filters combos that violate constraints."""
        group = ConfigSearchGroup(
            name="grid_constrained",
            params={"lo": [1, 5, 10], "hi": [3, 8]},
            constraints=[lambda c: c["lo"] < c["hi"]],
        )
        tuner = self._simple_tuner(groups=[group], strategy="grid")
        trials = tuner._generate_trials(group)
        for t in trials:
            assert t["lo"] < t["hi"]
        # (1,3),(1,8),(5,8) = 3 valid out of 6 total
        assert len(trials) == 3

    @pytest.mark.unit
    def test_random_generates_max_trials(self):
        """Random strategy respects max_trials cap."""
        tuner = self._simple_tuner(
            groups=[
                ConfigSearchGroup(
                    name="rand_test",
                    params={"x": list(range(100))},
                ),
            ],
            strategy="random",
            max_trials=10,
        )
        group = tuner.search_groups[0]
        trials = tuner._generate_trials(group)
        assert len(trials) == 10

    # -- composite score --

    @pytest.mark.unit
    def test_composite_score_formula(self):
        """Composite score matches documented formula."""
        tuner = self._simple_tuner()
        baseline = _make_report(pass_rate=0.5, cost=1.0, time=2.0)
        trial_report = _make_report(pass_rate=0.8, cost=0.5, time=1.0)

        score = tuner._compute_composite_score(trial_report, baseline)

        # w_pass=0.6, w_cost=0.25, w_speed=0.15
        # norm_cost = 0.5/1.0 = 0.5, norm_time = 1.0/2.0 = 0.5
        expected = (0.6 * 0.8) - (0.25 * 0.5) - (0.15 * 0.5)
        assert abs(score - expected) < 1e-9

    @pytest.mark.unit
    def test_composite_score_zero_baseline(self):
        """Composite score handles zero-cost/zero-time baseline safely."""
        tuner = self._simple_tuner()
        baseline = _make_report(pass_rate=0.5, cost=0.0, time=0.0)
        trial_report = _make_report(pass_rate=0.9, cost=0.1, time=0.5)

        score = tuner._compute_composite_score(trial_report, baseline)
        # norm_cost=0, norm_time=0 -> score = 0.6*0.9 = 0.54
        assert abs(score - 0.6 * 0.9) < 1e-9

    # -- tune_group --

    @pytest.mark.unit
    def test_tune_group_selects_best(self):
        """tune_group returns results sorted by composite_score descending."""
        tuner = self._simple_tuner()
        group = tuner.search_groups[0]
        results = tuner.tune_group(group)

        assert len(results) >= 1
        scores = [r.composite_score for r in results]
        assert scores == sorted(scores, reverse=True)

    # -- full tune --

    @pytest.mark.unit
    def test_tune_returns_tuning_result(self):
        """Full tune() returns a well-formed TuningResult."""
        tuner = self._simple_tuner()
        result = tuner.tune()

        assert isinstance(result, TuningResult)
        assert result.baseline_report is not None
        assert len(result.trials) >= 1
        assert result.best_trial is not None
        assert len(result.recommendations) >= 1

    # -- sequential strategy --

    @pytest.mark.unit
    def test_sequential_carries_best_forward(self):
        """Sequential strategy accumulates best overrides across groups."""
        groups = [
            ConfigSearchGroup(name="g1", params={"gamma": [0.9, 0.99]}),
            ConfigSearchGroup(name="g2", params={"alpha": [0.01, 0.05]}),
        ]
        tuner = self._simple_tuner(groups=groups, strategy="sequential")
        result = tuner.tune()

        # g2 trials should have both gamma AND alpha keys
        g2_trials = [t for t in result.trials if t.group_name == "g2"]
        assert len(g2_trials) >= 1
        for trial in g2_trials:
            assert "gamma" in trial.config_overrides
            assert "alpha" in trial.config_overrides

    # -- default search groups --

    @pytest.mark.unit
    def test_default_search_groups_count(self):
        """There are exactly 8 default search groups."""
        assert len(DEFAULT_SEARCH_GROUPS) == 8

    @pytest.mark.unit
    def test_default_search_groups_names(self):
        """Default groups cover the 8 documented parameter families."""
        expected_names = {
            "td_lambda_core", "adaptive_lr", "exploration",
            "trust_adaptation", "stall_detection", "validation_quality",
            "memory_budget", "execution_budget",
        }
        actual_names = {g.name for g in DEFAULT_SEARCH_GROUPS}
        assert actual_names == expected_names

    @pytest.mark.unit
    def test_default_groups_params_exist_on_swarm_config(self):
        """Every param in default search groups is a valid SwarmConfig field."""
        config = SwarmConfig()
        for group in DEFAULT_SEARCH_GROUPS:
            for param_name in group.params:
                assert hasattr(config, param_name), (
                    f"SwarmConfig missing field '{param_name}' from group '{group.name}'"
                )

    # -- build_config --

    @pytest.mark.unit
    def test_build_config_applies_overrides(self):
        """_build_config applies overrides to SwarmConfig."""
        tuner = self._simple_tuner()
        config = tuner._build_config({"gamma": 0.95, "alpha": 0.05})
        assert config.gamma == 0.95
        assert config.alpha == 0.05

    @pytest.mark.unit
    def test_build_config_from_baseline(self):
        """_build_config deep-copies baseline before applying overrides."""
        baseline = SwarmConfig(gamma=0.8)
        tuner = ConfigTuner(
            benchmark=_make_benchmark(),
            agent_factory=_make_agent_factory(),
            baseline_config=baseline,
            n_runs=1,
        )
        config = tuner._build_config({"alpha": 0.02})
        assert config.gamma == 0.8
        assert config.alpha == 0.02
        # Original baseline unchanged
        assert baseline.alpha != 0.02

    # -- recommendations --

    @pytest.mark.unit
    def test_recommendations_on_improvement(self):
        """Recommendations mention improvement when best > baseline."""
        tuner = self._simple_tuner()
        baseline = _make_report(pass_rate=0.5, cost=1.0, time=2.0)
        better = _make_report(pass_rate=0.9, cost=0.5, time=1.0)
        best_trial = ConfigTrialResult(
            config_overrides={"gamma": 0.95},
            group_name="td_lambda_core",
            report=better,
            composite_score=tuner._compute_composite_score(better, baseline),
        )
        recs = tuner._generate_recommendations(baseline, [best_trial], best_trial)
        assert any("improved" in r.lower() for r in recs)

    @pytest.mark.unit
    def test_recommendations_no_improvement(self):
        """Recommendations say optimal when no trial beats baseline."""
        tuner = self._simple_tuner()
        baseline = _make_report(pass_rate=1.0, cost=0.0, time=0.0)
        worse = _make_report(pass_rate=0.5, cost=1.0, time=2.0)
        worst_trial = ConfigTrialResult(
            config_overrides={"gamma": 0.9},
            group_name="td_lambda_core",
            report=worse,
            composite_score=tuner._compute_composite_score(worse, baseline),
        )
        recs = tuner._generate_recommendations(baseline, [worst_trial], worst_trial)
        assert any("optimal" in r.lower() for r in recs)

    # -- custom objective weights --

    @pytest.mark.unit
    def test_custom_objective_weights(self):
        """Custom objective_weights affect composite score."""
        baseline = _make_report(pass_rate=0.5, cost=1.0, time=1.0)
        trial_report = _make_report(pass_rate=0.8, cost=1.0, time=1.0)

        # Only care about pass_rate
        tuner_pass = ConfigTuner(
            benchmark=_make_benchmark(),
            agent_factory=_make_agent_factory(),
            n_runs=1,
            objective_weights={"pass_rate": 1.0, "cost_efficiency": 0.0, "speed": 0.0},
        )
        score_pass = tuner_pass._compute_composite_score(trial_report, baseline)
        assert abs(score_pass - 0.8) < 1e-9

        # Only care about cost
        tuner_cost = ConfigTuner(
            benchmark=_make_benchmark(),
            agent_factory=_make_agent_factory(),
            n_runs=1,
            objective_weights={"pass_rate": 0.0, "cost_efficiency": 1.0, "speed": 0.0},
        )
        score_cost = tuner_cost._compute_composite_score(trial_report, baseline)
        # norm_cost = 1.0/1.0 = 1.0 -> score = -1.0
        assert abs(score_cost - (-1.0)) < 1e-9


# ---------------------------------------------------------------------------
# CLI runner (kept for backwards compat)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

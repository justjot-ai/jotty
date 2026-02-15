"""
Ablation Study Framework

Systematic evaluation of component contributions and hyperparameter tuning.
Based on OAgents empirical validation approach.
"""
import copy
import itertools
import logging
import random
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from .benchmark import Benchmark, BenchmarkMetrics
from .evaluation_protocol import EvaluationProtocol, EvaluationReport

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Type of component for ablation."""
    FEATURE = "feature"  # Optional feature (can be disabled)
    MODULE = "module"  # Module/component (can be removed)
    CONFIG = "config"  # Configuration option (can be changed)


@dataclass
class ComponentContribution:
    """Contribution of a component to performance."""
    component_name: str
    component_type: ComponentType
    baseline_pass_rate: float
    ablated_pass_rate: float
    contribution: float  # Difference (baseline - ablated)
    contribution_percent: float  # Percentage change
    cost_impact: float  # Cost difference
    execution_time_impact: float  # Execution time difference
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component_name": self.component_name,
            "component_type": self.component_type.value,
            "baseline_pass_rate": self.baseline_pass_rate,
            "ablated_pass_rate": self.ablated_pass_rate,
            "contribution": self.contribution,
            "contribution_percent": self.contribution_percent,
            "cost_impact": self.cost_impact,
            "execution_time_impact": self.execution_time_impact,
        }


@dataclass
class AblationResult:
    """Result of ablation study."""
    study_name: str
    baseline_report: EvaluationReport
    component_contributions: List[ComponentContribution]
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "study_name": self.study_name,
            "baseline_report": self.baseline_report.to_dict(),
            "component_contributions": [c.to_dict() for c in self.component_contributions],
            "recommendations": self.recommendations,
        }


class AblationStudy:
    """
    Systematic ablation study framework.
    
    Tests each component's contribution by:
    1. Running baseline (all components enabled)
    2. Running with component disabled/removed
    3. Comparing results
    
    Usage:
        study = AblationStudy(
            benchmark=benchmark,
            agent_factory=lambda config: create_agent(config),
            components=[
                {"name": "learning", "disable": lambda c: setattr(c, 'enable_rl', False)},
                {"name": "memory", "disable": lambda c: setattr(c, 'enable_memory', False)},
            ]
        )
        
        result = study.run()
        print(f"Learning contribution: {result.component_contributions[0].contribution:.2%}")
    """
    
    def __init__(self, benchmark: Benchmark, agent_factory: Callable[[Any], Any], components: List[Dict[str, Any]], n_runs: int = 5, random_seed: int = 42, baseline_config: Optional[Any] = None) -> None:
        """
        Initialize ablation study.
        
        Args:
            benchmark: Benchmark to use
            agent_factory: Function that creates agent from config
            components: List of component definitions, each with:
                - name: Component name
                - disable: Function to disable component (modifies config)
                - type: ComponentType (default: FEATURE)
            n_runs: Number of runs per evaluation
            random_seed: Random seed for reproducibility
            baseline_config: Baseline configuration (default: create new)
        """
        self.benchmark = benchmark
        self.agent_factory = agent_factory
        self.components = components
        self.n_runs = n_runs
        self.random_seed = random_seed
        self.baseline_config = baseline_config
    
    def run(self) -> AblationResult:
        """
        Run ablation study.
        
        Returns:
            AblationResult with component contributions
        """
        logger.info(f"Starting ablation study on {self.benchmark.name}")
        
        # Run baseline
        logger.info("Running baseline (all components enabled)...")
        baseline_agent = self.agent_factory(self.baseline_config)
        baseline_protocol = EvaluationProtocol(
            benchmark=self.benchmark,
            n_runs=self.n_runs,
            random_seed=self.random_seed
        )
        baseline_report = baseline_protocol.evaluate(baseline_agent, save_results=False)
        
        logger.info(f"Baseline pass rate: {baseline_report.mean_pass_rate:.2%}")
        
        # Test each component
        contributions: List[ComponentContribution] = []
        
        for component in self.components:
            component_name = component['name']
            disable_func = component['disable']
            component_type = component.get('type', ComponentType.FEATURE)
            
            logger.info(f"Testing component: {component_name}")
            
            # Create ablated config
            ablated_config = self._create_ablated_config(disable_func)
            
            # Run ablated evaluation
            ablated_agent = self.agent_factory(ablated_config)
            ablated_protocol = EvaluationProtocol(
                benchmark=self.benchmark,
                n_runs=self.n_runs,
                random_seed=self.random_seed
            )
            ablated_report = ablated_protocol.evaluate(ablated_agent, save_results=False)
            
            # Calculate contribution
            contribution = baseline_report.mean_pass_rate - ablated_report.mean_pass_rate
            contribution_percent = (contribution / baseline_report.mean_pass_rate * 100) if baseline_report.mean_pass_rate > 0 else 0.0
            
            cost_impact = ablated_report.mean_cost - baseline_report.mean_cost
            execution_time_impact = ablated_report.mean_execution_time - baseline_report.mean_execution_time
            
            contrib = ComponentContribution(
                component_name=component_name,
                component_type=component_type,
                baseline_pass_rate=baseline_report.mean_pass_rate,
                ablated_pass_rate=ablated_report.mean_pass_rate,
                contribution=contribution,
                contribution_percent=contribution_percent,
                cost_impact=cost_impact,
                execution_time_impact=execution_time_impact
            )
            
            contributions.append(contrib)
            
            logger.info(
                f"Component {component_name}: "
                f"contribution={contribution:.2%}, "
                f"cost_impact=${cost_impact:.4f}"
            )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(contributions, baseline_report)
        
        return AblationResult(
            study_name=f"{self.benchmark.name}_ablation",
            baseline_report=baseline_report,
            component_contributions=contributions,
            recommendations=recommendations
        )
    
    def _create_ablated_config(self, disable_func: Callable) -> Any:
        """Create ablated configuration."""
        # Create copy of baseline config
        if self.baseline_config is None:
            # Create default config
            from ..foundation.data_structures import SwarmLearningConfig
            config = SwarmConfig()
        else:
            # Copy config (simple copy for now)
            import copy
            config = copy.deepcopy(self.baseline_config)
        
        # Apply disable function
        disable_func(config)
        
        return config
    
    def _generate_recommendations(
        self,
        contributions: List[ComponentContribution],
        baseline_report: EvaluationReport
    ) -> List[str]:
        """Generate recommendations based on ablation results."""
        recommendations = []
        
        # Find components with negative contribution (hurt performance)
        negative_contribs = [c for c in contributions if c.contribution < -0.01]
        if negative_contribs:
            recommendations.append(
                f"Consider disabling {len(negative_contribs)} component(s) that hurt performance: "
                f"{', '.join(c.component_name for c in negative_contribs)}"
            )
        
        # Find components with minimal contribution (< 1%)
        minimal_contribs = [c for c in contributions if abs(c.contribution) < 0.01]
        if minimal_contribs:
            recommendations.append(
                f"{len(minimal_contribs)} component(s) have minimal impact (<1%): "
                f"{', '.join(c.component_name for c in minimal_contribs)}. "
                f"Consider removing for simplicity."
            )
        
        # Find high-cost components
        high_cost = [c for c in contributions if c.cost_impact > 0.1]
        if high_cost:
            recommendations.append(
                f"{len(high_cost)} component(s) significantly increase cost: "
                f"{', '.join(c.component_name for c in high_cost)}. "
                f"Consider optimizing or disabling if not critical."
            )
        
        # Find critical components (>5% contribution)
        critical = [c for c in contributions if c.contribution > 0.05]
        if critical:
            recommendations.append(
                f"Critical components (>5% contribution): "
                f"{', '.join(c.component_name for c in critical)}. "
                f"Keep these enabled."
            )

        return recommendations


# ---------------------------------------------------------------------------
# ConfigTuner — Hyperparameter Tuning for MAS/Swarm Config
# ---------------------------------------------------------------------------


@dataclass
class ConfigSearchGroup:
    """Defines one tunable parameter group for hyperparameter sweeps."""
    name: str
    params: Dict[str, List[Any]]
    constraints: List[Callable] = field(default_factory=list)
    target_metric: str = "pass_rate"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "params": {k: [str(v) for v in vals] for k, vals in self.params.items()},
            "target_metric": self.target_metric,
        }


@dataclass
class ConfigTrialResult:
    """Result from evaluating one config variant."""
    config_overrides: Dict[str, Any]
    group_name: str
    report: EvaluationReport
    composite_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config_overrides": self.config_overrides,
            "group_name": self.group_name,
            "report": self.report.to_dict(),
            "composite_score": self.composite_score,
        }


@dataclass
class TuningResult:
    """Full tuning study result."""
    baseline_report: EvaluationReport
    trials: List[ConfigTrialResult]
    best_trial: Optional[ConfigTrialResult]
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "baseline_report": self.baseline_report.to_dict(),
            "trials": [t.to_dict() for t in self.trials],
            "best_trial": self.best_trial.to_dict() if self.best_trial else None,
            "recommendations": self.recommendations,
        }


# 8 default search groups covering the key SwarmConfig parameter families
DEFAULT_SEARCH_GROUPS: List[ConfigSearchGroup] = [
    ConfigSearchGroup(
        name="td_lambda_core",
        params={
            "gamma": [0.9, 0.95, 0.99],
            "lambda_trace": [0.8, 0.9, 0.95],
            "alpha": [0.005, 0.01, 0.05],
            "alpha_min": [0.0005, 0.001, 0.005],
            "alpha_max": [0.05, 0.1, 0.2],
        },
        constraints=[lambda c: c["alpha_min"] < c["alpha_max"]],
        target_metric="pass_rate",
    ),
    ConfigSearchGroup(
        name="adaptive_lr",
        params={
            "alpha_adaptation_rate": [0.05, 0.1, 0.2],
            "adaptive_window_size": [25, 50, 100],
            "instability_threshold_multiplier": [1.0, 1.5, 2.0],
            "slow_learning_threshold": [0.005, 0.01, 0.02],
            "learning_boost_factor": [1.5, 2.0, 3.0],
        },
        target_metric="pass_rate",
    ),
    ConfigSearchGroup(
        name="exploration",
        params={
            "epsilon_start": [0.2, 0.3, 0.5],
            "epsilon_end": [0.01, 0.05, 0.1],
            "epsilon_decay_episodes": [200, 500, 1000],
            "ucb_coefficient": [1.0, 2.0, 3.0],
            "exploration_boost_on_stall": [0.05, 0.1, 0.2],
        },
        constraints=[lambda c: c["epsilon_start"] > c["epsilon_end"]],
        target_metric="pass_rate",
    ),
    ConfigSearchGroup(
        name="trust_adaptation",
        params={
            "trust_decrease_on_struggle": [0.05, 0.1, 0.2],
            "trust_increase_on_excel": [0.02, 0.05, 0.1],
            "trust_min": [0.05, 0.1, 0.2],
            "adaptation_interval": [3, 5, 10],
            "adaptation_struggle_threshold": [0.2, 0.3, 0.4],
            "adaptation_excel_threshold": [0.7, 0.8, 0.9],
        },
        constraints=[
            lambda c: c["adaptation_struggle_threshold"] < c["adaptation_excel_threshold"]
        ],
        target_metric="pass_rate",
    ),
    ConfigSearchGroup(
        name="stall_detection",
        params={
            "stall_detection_window": [50, 100, 200],
            "stall_threshold": [0.0005, 0.001, 0.005],
            "max_exploration_iterations": [5, 10, 20],
        },
        target_metric="pass_rate",
    ),
    ConfigSearchGroup(
        name="validation_quality",
        params={
            "refinement_on_low_confidence": [0.4, 0.6, 0.8],
            "max_validation_rounds": [2, 3, 5],
            "max_refinement_rounds": [1, 2, 3],
            "max_eval_retries": [2, 3, 5],
        },
        target_metric="pass_rate",
    ),
    ConfigSearchGroup(
        name="memory_budget",
        params={
            "min_memory_budget": [5000, 10000, 20000],
            "max_memory_budget": [30000, 60000, 120000],
            "max_context_tokens": [50000, 100000, 200000],
        },
        constraints=[lambda c: c["min_memory_budget"] < c["max_memory_budget"]],
        target_metric="pass_rate",
    ),
    ConfigSearchGroup(
        name="execution_budget",
        params={
            "max_llm_calls_per_episode": [50, 100, 200],
            "max_llm_calls_per_agent": [25, 50, 100],
            "max_actor_iters": [25, 50, 100],
            "llm_timeout_seconds": [60.0, 180.0, 300.0],
        },
        constraints=[
            lambda c: c["max_llm_calls_per_agent"] <= c["max_llm_calls_per_episode"]
        ],
        target_metric="pass_rate",
    ),
]


class ConfigTuner:
    """
    Hyperparameter tuner for SwarmConfig.

    Sweeps parameter values across functionally-grouped search spaces,
    evaluates each variant using EvaluationProtocol, and ranks results
    by a composite objective.

    Usage:
        tuner = ConfigTuner(
            benchmark=benchmark,
            agent_factory=lambda config: create_agent(config),
            search_groups=DEFAULT_SEARCH_GROUPS,
        )
        result = tuner.tune()
        print(f"Best score: {result.best_trial.composite_score:.4f}")
    """

    DEFAULT_OBJECTIVE_WEIGHTS = {
        "pass_rate": 0.6,
        "cost_efficiency": 0.25,
        "speed": 0.15,
    }

    def __init__(self, benchmark: Benchmark, agent_factory: Callable[[Any], Any], search_groups: Optional[List[ConfigSearchGroup]] = None, n_runs: int = 3, strategy: str = 'random', max_trials: int = 50, objective_weights: Optional[Dict[str, float]] = None, baseline_config: Optional[Any] = None, random_seed: int = 42) -> None:
        """
        Initialize ConfigTuner.

        Args:
            benchmark: Benchmark to evaluate against
            agent_factory: Creates an agent from a SwarmConfig
            search_groups: Parameter groups to sweep (default: DEFAULT_SEARCH_GROUPS)
            n_runs: Evaluation runs per trial for variance reduction
            strategy: Search strategy — "grid", "random", or "sequential"
            max_trials: Max trials per group for random strategy
            objective_weights: Composite score weights (keys: pass_rate, cost_efficiency, speed)
            baseline_config: Baseline SwarmConfig (default: SwarmConfig())
            random_seed: Seed for reproducibility
        """
        self.benchmark = benchmark
        self.agent_factory = agent_factory
        self.search_groups = search_groups or DEFAULT_SEARCH_GROUPS
        self.n_runs = n_runs
        self.strategy = strategy
        self.max_trials = max_trials
        self.objective_weights = objective_weights or self.DEFAULT_OBJECTIVE_WEIGHTS
        self.baseline_config = baseline_config
        self.random_seed = random_seed

    def tune(self) -> TuningResult:
        """
        Run full tuning sweep across all search groups.

        Returns:
            TuningResult with baseline, all trials, best trial, and recommendations
        """
        logger.info(
            f"Starting config tuning: {len(self.search_groups)} groups, "
            f"strategy={self.strategy}"
        )

        # Evaluate baseline
        baseline_report = self._evaluate_config({})

        all_trials: List[ConfigTrialResult] = []

        if self.strategy == "sequential":
            all_trials = self._tune_sequential(baseline_report)
        else:
            for group in self.search_groups:
                group_trials = self.tune_group(group, baseline_report)
                all_trials.extend(group_trials)

        # Find best trial
        best_trial = max(all_trials, key=lambda t: t.composite_score) if all_trials else None

        # Generate recommendations
        recommendations = self._generate_recommendations(
            baseline_report, all_trials, best_trial
        )

        return TuningResult(
            baseline_report=baseline_report,
            trials=all_trials,
            best_trial=best_trial,
            recommendations=recommendations,
        )

    def tune_group(
        self,
        group: ConfigSearchGroup,
        baseline_report: Optional[EvaluationReport] = None,
    ) -> List[ConfigTrialResult]:
        """
        Sweep one parameter group.

        Args:
            group: The ConfigSearchGroup to sweep
            baseline_report: Pre-computed baseline (evaluated if None)

        Returns:
            List of ConfigTrialResult for this group, sorted by composite_score desc
        """
        if baseline_report is None:
            baseline_report = self._evaluate_config({})

        trial_overrides = self._generate_trials(group)
        results: List[ConfigTrialResult] = []

        for overrides in trial_overrides:
            report = self._evaluate_config(overrides)
            score = self._compute_composite_score(report, baseline_report)
            results.append(
                ConfigTrialResult(
                    config_overrides=overrides,
                    group_name=group.name,
                    report=report,
                    composite_score=score,
                )
            )

        results.sort(key=lambda t: t.composite_score, reverse=True)
        return results

    def _tune_sequential(
        self, baseline_report: EvaluationReport
    ) -> List[ConfigTrialResult]:
        """
        Tune groups sequentially — carry best overrides forward.

        Each group is tuned independently; the best overrides from earlier
        groups are merged into the baseline for subsequent groups.
        """
        all_trials: List[ConfigTrialResult] = []
        accumulated_overrides: Dict[str, Any] = {}
        current_baseline = baseline_report

        for group in self.search_groups:
            # Build an augmented group whose trials include accumulated overrides
            group_trials = self._generate_trials(group)
            results: List[ConfigTrialResult] = []

            for overrides in group_trials:
                merged = {**accumulated_overrides, **overrides}
                report = self._evaluate_config(merged)
                score = self._compute_composite_score(report, current_baseline)
                results.append(
                    ConfigTrialResult(
                        config_overrides=merged,
                        group_name=group.name,
                        report=report,
                        composite_score=score,
                    )
                )

            if results:
                results.sort(key=lambda t: t.composite_score, reverse=True)
                best = results[0]
                accumulated_overrides = dict(best.config_overrides)
                current_baseline = best.report

            all_trials.extend(results)

        return all_trials

    def _generate_trials(self, group: ConfigSearchGroup) -> List[Dict[str, Any]]:
        """
        Generate config override dicts for a search group.

        Applies the configured strategy (grid / random) and filters
        by constraints.
        """
        param_names = list(group.params.keys())
        param_values = list(group.params.values())

        if self.strategy == "grid":
            combos = [
                dict(zip(param_names, vals))
                for vals in itertools.product(*param_values)
            ]
        else:
            # random (also used as inner sampler for sequential)
            rng = random.Random(self.random_seed)
            combos = []
            for _ in range(self.max_trials):
                combo = {
                    name: rng.choice(values)
                    for name, values in zip(param_names, param_values)
                }
                combos.append(combo)

        # Apply constraints
        valid = [c for c in combos if self._check_constraints(c, group.constraints)]
        return valid

    def _evaluate_config(self, overrides: Dict[str, Any]) -> EvaluationReport:
        """Evaluate a single config variant."""
        config = self._build_config(overrides)
        agent = self.agent_factory(config)
        protocol = EvaluationProtocol(
            benchmark=self.benchmark,
            n_runs=self.n_runs,
            random_seed=self.random_seed,
        )
        return protocol.evaluate(agent, save_results=False)

    def _build_config(self, overrides: Dict[str, Any]) -> Any:
        """Build a SwarmConfig with the given overrides applied."""
        if self.baseline_config is None:
            from ..foundation.data_structures import SwarmLearningConfig
            config = SwarmConfig()
        else:
            config = copy.deepcopy(self.baseline_config)

        for key, value in overrides.items():
            setattr(config, key, value)

        return config

    def _compute_composite_score(
        self, report: EvaluationReport, baseline_report: EvaluationReport
    ) -> float:
        """
        Compute weighted composite score.

        score = (w_pass * pass_rate) - (w_cost * norm_cost) - (w_speed * norm_time)

        Cost and time are normalized relative to the baseline so that
        improvements are positive and regressions are negative.
        """
        w_pass = self.objective_weights.get("pass_rate", 0.6)
        w_cost = self.objective_weights.get("cost_efficiency", 0.25)
        w_speed = self.objective_weights.get("speed", 0.15)

        # Normalize cost relative to baseline (0 = same, positive = cheaper)
        if baseline_report.mean_cost > 0:
            norm_cost = report.mean_cost / baseline_report.mean_cost
        else:
            norm_cost = 0.0

        # Normalize time relative to baseline
        if baseline_report.mean_execution_time > 0:
            norm_time = report.mean_execution_time / baseline_report.mean_execution_time
        else:
            norm_time = 0.0

        return (w_pass * report.mean_pass_rate) - (w_cost * norm_cost) - (w_speed * norm_time)

    @staticmethod
    def _check_constraints(
        overrides: Dict[str, Any], constraints: List[Callable]
    ) -> bool:
        """Check whether a config override dict satisfies all constraints."""
        for constraint in constraints:
            try:
                if not constraint(overrides):
                    return False
            except (KeyError, TypeError):
                return False
        return True

    def _generate_recommendations(
        self,
        baseline_report: EvaluationReport,
        trials: List[ConfigTrialResult],
        best_trial: Optional[ConfigTrialResult],
    ) -> List[str]:
        """Generate human-readable recommendations from tuning results."""
        recommendations: List[str] = []

        if not trials:
            recommendations.append("No trials completed — check search groups and constraints.")
            return recommendations

        baseline_score = self._compute_composite_score(baseline_report, baseline_report)

        if best_trial and best_trial.composite_score > baseline_score:
            improvement = best_trial.composite_score - baseline_score
            recommendations.append(
                f"Best trial ({best_trial.group_name}) improved composite score "
                f"by {improvement:.4f} over baseline."
            )
            recommendations.append(
                f"Recommended overrides: {best_trial.config_overrides}"
            )
        else:
            recommendations.append(
                "No trial improved over baseline. Current defaults may already be optimal."
            )

        # Per-group bests
        groups_seen: Dict[str, ConfigTrialResult] = {}
        for trial in trials:
            if trial.group_name not in groups_seen or trial.composite_score > groups_seen[trial.group_name].composite_score:
                groups_seen[trial.group_name] = trial

        for group_name, trial in groups_seen.items():
            if trial.composite_score > baseline_score:
                recommendations.append(
                    f"Group '{group_name}': best score {trial.composite_score:.4f} "
                    f"(pass_rate={trial.report.mean_pass_rate:.2%})."
                )

        return recommendations

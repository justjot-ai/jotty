"""
Evaluation Protocol

Standardized evaluation protocol for reproducible agent evaluation.
Based on OAgents evaluation approach.
"""

import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .benchmark import Benchmark, BenchmarkMetrics
from .reproducibility import ReproducibilityConfig, set_reproducible_seeds

logger = logging.getLogger(__name__)


@dataclass
class EvaluationRun:
    """Single evaluation run."""

    run_id: int
    seed: int
    metrics: BenchmarkMetrics
    timestamp: float = field(default_factory=time.time)
    config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "metrics": self.metrics.to_dict(),
            "timestamp": self.timestamp,
            "config": self.config,
        }


@dataclass
class EvaluationReport:
    """Aggregated evaluation report across multiple runs."""

    benchmark_name: str
    n_runs: int
    mean_pass_rate: float
    std_pass_rate: float
    mean_cost: float
    std_cost: float
    mean_execution_time: float
    std_execution_time: float
    runs: List[EvaluationRun] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "n_runs": self.n_runs,
            "mean_pass_rate": self.mean_pass_rate,
            "std_pass_rate": self.std_pass_rate,
            "mean_cost": self.mean_cost,
            "std_cost": self.std_cost,
            "mean_execution_time": self.mean_execution_time,
            "std_execution_time": self.std_execution_time,
            "runs": [r.to_dict() for r in self.runs],
        }


class EvaluationProtocol:
    """
    Standardized evaluation protocol for reproducible agent evaluation.

    Ensures:
    - Fixed seeds for reproducibility
    - Multiple runs for variance tracking
    - Standardized metrics
    - Variance analysis

    Usage:
        protocol = EvaluationProtocol(
            benchmark=benchmark,
            n_runs=5,
            random_seed=42
        )

        report = protocol.evaluate(agent)
        print(f"Pass rate: {report.mean_pass_rate:.2%} ± {report.std_pass_rate:.2%}")
    """

    def __init__(
        self,
        benchmark: Benchmark,
        n_runs: int = 5,
        random_seed: int = 42,
        reproducibility_config: Optional[ReproducibilityConfig] = None,
        task_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize evaluation protocol.

        Args:
            benchmark: Benchmark instance
            n_runs: Number of evaluation runs
            random_seed: Base random seed (each run uses seed + run_id)
            reproducibility_config: Optional reproducibility config
            task_ids: Optional list of task IDs to evaluate
        """
        self.benchmark = benchmark
        self.n_runs = n_runs
        self.random_seed = random_seed
        self.reproducibility_config = reproducibility_config
        self.task_ids = task_ids

    def evaluate(
        self, agent: Any, save_results: bool = True, output_dir: Optional[str] = None
    ) -> EvaluationReport:
        """
        Run standardized evaluation.

        Args:
            agent: Agent to evaluate
            save_results: Whether to save results to file
            output_dir: Directory to save results (default: ./evaluation_results)

        Returns:
            EvaluationReport with aggregated metrics
        """
        runs: List[EvaluationRun] = []

        logger.info(f"Starting evaluation: {self.n_runs} runs on {self.benchmark.name}")

        for run_id in range(self.n_runs):
            # Set seed for this run
            seed = self.random_seed + run_id
            set_reproducible_seeds(random_seed=seed)

            logger.info(f"Run {run_id + 1}/{self.n_runs} (seed={seed})")

            # Run evaluation
            metrics = self.benchmark.evaluate(agent=agent, task_ids=self.task_ids)

            # Create run record
            run = EvaluationRun(
                run_id=run_id + 1,
                seed=seed,
                metrics=metrics,
                config={
                    "benchmark": self.benchmark.name,
                    "n_runs": self.n_runs,
                    "random_seed": self.random_seed,
                },
            )
            runs.append(run)

            logger.info(
                f"Run {run_id + 1} complete: "
                f"pass_rate={metrics.pass_rate:.2%}, "
                f"cost=${metrics.total_cost:.4f}"
            )

        # Calculate aggregated metrics
        report = self._aggregate_results(runs)

        # Save results if requested
        if save_results:
            self._save_results(report, output_dir)

        return report

    def _aggregate_results(self, runs: List[EvaluationRun]) -> EvaluationReport:
        """Aggregate results across runs."""
        pass_rates = [r.metrics.pass_rate for r in runs]
        costs = [r.metrics.total_cost for r in runs]
        execution_times = [r.metrics.avg_execution_time for r in runs]

        mean_pass_rate = statistics.mean(pass_rates)
        std_pass_rate = statistics.stdev(pass_rates) if len(pass_rates) > 1 else 0.0

        mean_cost = statistics.mean(costs)
        std_cost = statistics.stdev(costs) if len(costs) > 1 else 0.0

        mean_execution_time = statistics.mean(execution_times)
        std_execution_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0

        return EvaluationReport(
            benchmark_name=self.benchmark.name,
            n_runs=self.n_runs,
            mean_pass_rate=mean_pass_rate,
            std_pass_rate=std_pass_rate,
            mean_cost=mean_cost,
            std_cost=std_cost,
            mean_execution_time=mean_execution_time,
            std_execution_time=std_execution_time,
            runs=runs,
        )

    def _save_results(self, report: EvaluationReport, output_dir: Optional[str]) -> Any:
        """Save evaluation results to file."""
        if output_dir is None:
            output_dir = "./evaluation_results"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save report
        report_file = output_path / f"{self.benchmark.name}_evaluation_report.json"
        with open(report_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Saved evaluation report to {report_file}")

        # Save summary
        summary_file = output_path / f"{self.benchmark.name}_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Evaluation Report: {self.benchmark.name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Number of runs: {report.n_runs}\n")
            f.write(f"Pass rate: {report.mean_pass_rate:.2%} ± {report.std_pass_rate:.2%}\n")
            f.write(f"Mean cost: ${report.mean_cost:.4f} ± ${report.std_cost:.4f}\n")
            f.write(
                f"Mean execution time: {report.mean_execution_time:.2f}s ± {report.std_execution_time:.2f}s\n"
            )

        logger.info(f"Saved summary to {summary_file}")

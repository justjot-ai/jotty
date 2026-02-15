"""
Efficiency Metrics Module

Calculates efficiency metrics based on cost and performance.
Based on OAgents cost efficiency research.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .cost_tracker import CostMetrics


@dataclass
class EfficiencyReport:
    """Efficiency report with cost and performance metrics."""

    cost_per_success: float
    efficiency_score: float
    success_rate: float
    total_cost: float
    success_count: int
    total_attempts: int
    cost_reduction_potential: Optional[float] = None  # Potential cost reduction %
    performance_retention: Optional[float] = None  # Performance retention %

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cost_per_success": self.cost_per_success,
            "efficiency_score": self.efficiency_score,
            "success_rate": self.success_rate,
            "total_cost": self.total_cost,
            "success_count": self.success_count,
            "total_attempts": self.total_attempts,
            "cost_reduction_potential": self.cost_reduction_potential,
            "performance_retention": self.performance_retention,
        }


class EfficiencyMetrics:
    """
    Calculates efficiency metrics based on cost and performance.

    Based on OAgents research:
    - Cost-per-pass metric
    - Efficiency score (performance/cost ratio)
    - Cost reduction potential
    - Performance retention

    Usage:
        metrics = EfficiencyMetrics()

        report = metrics.calculate_efficiency(
            cost_metrics=cost_metrics,
            success_count=10,
            total_attempts=12
        )

        print(f"Cost per success: ${report.cost_per_success:.4f}")
        print(f"Efficiency score: {report.efficiency_score:.4f}")
    """

    @staticmethod
    def calculate_efficiency(
        cost_metrics: CostMetrics,
        success_count: int,
        total_attempts: Optional[int] = None,
        baseline_cost_per_success: Optional[float] = None,
    ) -> EfficiencyReport:
        """
        Calculate efficiency metrics.

        Args:
            cost_metrics: Cost metrics from CostTracker
            success_count: Number of successful tasks/episodes
            total_attempts: Total number of attempts (defaults to cost_metrics.total_calls)
            baseline_cost_per_success: Baseline cost per success for comparison

        Returns:
            EfficiencyReport with efficiency metrics
        """
        total_attempts = total_attempts or cost_metrics.total_calls

        # Cost per success
        cost_per_success = cost_metrics.total_cost / max(success_count, 1)

        # Success rate
        success_rate = success_count / max(total_attempts, 1)

        # Efficiency score: inverse of cost-per-success (higher is better)
        # Normalized to 0-1 scale (assuming $1 per success is baseline)
        # Score = 1 / (cost_per_success / baseline)
        baseline = baseline_cost_per_success or 1.0
        efficiency_score = min(1.0, baseline / max(cost_per_success, 0.001))

        # Cost reduction potential (if baseline provided)
        cost_reduction_potential = None
        if baseline_cost_per_success is not None:
            reduction = (
                (baseline_cost_per_success - cost_per_success) / baseline_cost_per_success
            ) * 100
            cost_reduction_potential = max(0.0, reduction)

        return EfficiencyReport(
            cost_per_success=cost_per_success,
            efficiency_score=efficiency_score,
            success_rate=success_rate,
            total_cost=cost_metrics.total_cost,
            success_count=success_count,
            total_attempts=total_attempts,
            cost_reduction_potential=cost_reduction_potential,
        )

    @staticmethod
    def compare_efficiency(
        current_cost_per_success: float,
        baseline_cost_per_success: float,
        current_performance: float,
        baseline_performance: float,
    ) -> Dict[str, Any]:
        """
        Compare efficiency between two configurations.

        Args:
            current_cost_per_success: Current cost per success
            baseline_cost_per_success: Baseline cost per success
            current_performance: Current performance (e.g., pass rate)
            baseline_performance: Baseline performance

        Returns:
            Dictionary with comparison metrics
        """
        # Cost reduction
        cost_reduction = (
            (baseline_cost_per_success - current_cost_per_success) / baseline_cost_per_success
        ) * 100

        # Performance retention
        performance_retention = (
            (current_performance / baseline_performance) * 100 if baseline_performance > 0 else 0.0
        )

        # Efficiency improvement
        current_efficiency = baseline_performance / max(current_cost_per_success, 0.001)
        baseline_efficiency = baseline_performance / max(baseline_cost_per_success, 0.001)
        efficiency_improvement = (
            (current_efficiency - baseline_efficiency) / baseline_efficiency
        ) * 100

        return {
            "cost_reduction_percent": cost_reduction,
            "performance_retention_percent": performance_retention,
            "efficiency_improvement_percent": efficiency_improvement,
            "cost_per_success_improvement": baseline_cost_per_success - current_cost_per_success,
        }

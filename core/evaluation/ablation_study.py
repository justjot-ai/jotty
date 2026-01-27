"""
Ablation Study Framework

Systematic evaluation of component contributions.
Based on OAgents empirical validation approach.
"""
import logging
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
    
    def __init__(
        self,
        benchmark: Benchmark,
        agent_factory: Callable[[Any], Any],
        components: List[Dict[str, Any]],
        n_runs: int = 5,
        random_seed: int = 42,
        baseline_config: Optional[Any] = None
    ):
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
            from ..foundation.data_structures import SwarmConfig
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

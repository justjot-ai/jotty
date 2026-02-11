"""
Learning Swarm - World-Class Self-Improving Meta-Learning
==========================================================

Production-grade swarm for:
- Swarm performance evaluation
- Agent optimization
- Gold standard management
- Continuous improvement
- Meta-learning across domains

This is the META-SWARM that improves all other swarms.

Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│                        LEARNING SWARM (META)                             │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │   Performance  │  │   Gold         │  │   Pattern      │            │
│  │   Evaluator    │  │   Curator      │  │   Miner        │            │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                   │                   │                      │
│          └───────────────────┼───────────────────┘                      │
│                              ▼                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │   Prompt       │  │   Workflow     │  │   Parameter    │            │
│  │   Optimizer    │  │   Optimizer    │  │   Tuner        │            │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                   │                   │                      │
│          └───────────────────┼───────────────────┘                      │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     META LEARNER                                 │   │
│  │   Synthesizes learnings and applies improvements across swarms   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘

Usage:
    from core.swarms.learning_swarm import LearningSwarm, improve_swarm

    # Full swarm
    meta = LearningSwarm()
    result = await meta.evaluate_and_improve("coding")

    # Continuous improvement loop
    await meta.run_improvement_cycle()

Author: Jotty Team
Date: February 2026
"""

import asyncio
import logging
import json
import dspy
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum

from .base_swarm import (
    BaseSwarm, SwarmConfig, SwarmResult, AgentRole,
    register_swarm, ExecutionTrace, GoldStandard, GoldStandardDB,
    ImprovementHistory, ImprovementSuggestion, ImprovementType,
    Evaluation, EvaluationResult, SwarmRegistry
)
from .base import DomainSwarm, AgentTeam
from ..agents.base import DomainAgent, DomainAgentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class LearningMode(Enum):
    EVALUATE = "evaluate"
    OPTIMIZE = "optimize"
    CURATE = "curate"
    FULL_CYCLE = "full_cycle"


class OptimizationType(Enum):
    PROMPT = "prompt"
    WORKFLOW = "workflow"
    PARAMETERS = "parameters"
    ALL = "all"


@dataclass
class LearningConfig(SwarmConfig):
    """Configuration for LearningSwarm."""
    learning_mode: LearningMode = LearningMode.FULL_CYCLE
    optimization_types: List[OptimizationType] = field(default_factory=lambda: [OptimizationType.ALL])
    evaluation_samples: int = 10
    improvement_threshold: float = 0.7
    min_samples_for_learning: int = 5
    auto_apply_improvements: bool = False
    gold_standard_path: str = None
    history_path: str = None

    def __post_init__(self):
        self.name = "LearningSwarm"
        self.domain = "meta_learning"


@dataclass
class SwarmPerformance:
    """Performance metrics for a swarm."""
    swarm_name: str
    domain: str
    avg_score: float
    success_rate: float
    avg_execution_time: float
    total_executions: int
    evaluations: List[Evaluation]
    weaknesses: List[str]
    strengths: List[str]


@dataclass
class OptimizationResult:
    """Result of optimization."""
    optimization_type: OptimizationType
    original_value: str
    optimized_value: str
    expected_improvement: float
    rationale: str


@dataclass
class LearningResult(SwarmResult):
    """Result from LearningSwarm."""
    swarm_evaluated: str = ""
    performance: Optional[SwarmPerformance] = None
    optimizations: List[OptimizationResult] = field(default_factory=list)
    gold_standards_created: int = 0
    improvements_suggested: int = 0
    improvements_applied: int = 0
    cross_domain_insights: List[str] = field(default_factory=list)


# =============================================================================
# DSPy SIGNATURES
# =============================================================================

class PerformanceAnalysisSignature(dspy.Signature):
    """Analyze swarm performance.

    You are a PERFORMANCE ANALYST for AI swarms. Analyze:
    1. Success patterns
    2. Failure modes
    3. Efficiency bottlenecks
    4. Quality trends
    5. Improvement opportunities

    Be data-driven and specific.
    """
    evaluations: str = dspy.InputField(desc="JSON list of evaluation results")
    execution_traces: str = dspy.InputField(desc="JSON list of execution traces")
    swarm_config: str = dspy.InputField(desc="Current swarm configuration")

    strengths: str = dspy.OutputField(desc="Swarm strengths, separated by |")
    weaknesses: str = dspy.OutputField(desc="Swarm weaknesses, separated by |")
    patterns: str = dspy.OutputField(desc="Performance patterns identified, separated by |")
    improvement_areas: str = dspy.OutputField(desc="Areas for improvement, separated by |")
    overall_health: float = dspy.OutputField(desc="Overall swarm health score 0-100")


class GoldStandardCreationSignature(dspy.Signature):
    """Create gold standards from successful executions.

    You are a QUALITY CURATOR. Create gold standards that:
    1. Represent ideal outputs
    2. Have clear evaluation criteria
    3. Cover diverse scenarios
    4. Are reproducible
    5. Set high but achievable bars

    Quality over quantity.
    """
    successful_outputs: str = dspy.InputField(desc="JSON list of successful execution outputs")
    domain: str = dspy.InputField(desc="Domain of the swarm")
    task_types: str = dspy.InputField(desc="Types of tasks performed")

    gold_standards: str = dspy.OutputField(desc="JSON list of gold standard definitions")
    evaluation_criteria: str = dspy.OutputField(desc="Evaluation criteria for each standard, separated by |")
    coverage_assessment: str = dspy.OutputField(desc="Assessment of scenario coverage")


class PromptOptimizationSignature(dspy.Signature):
    """Optimize agent prompts.

    You are a PROMPT ENGINEER. Optimize prompts to:
    1. Improve clarity
    2. Reduce ambiguity
    3. Add relevant examples
    4. Strengthen constraints
    5. Enhance output quality

    Make prompts more effective.
    """
    current_prompt: str = dspy.InputField(desc="Current prompt")
    feedback: str = dspy.InputField(desc="Feedback from evaluations")
    failure_cases: str = dspy.InputField(desc="Cases where the prompt failed")

    optimized_prompt: str = dspy.OutputField(desc="Optimized prompt")
    changes_made: str = dspy.OutputField(desc="Changes made, separated by |")
    expected_improvement: float = dspy.OutputField(desc="Expected improvement 0-100")
    rationale: str = dspy.OutputField(desc="Rationale for changes")


class WorkflowOptimizationSignature(dspy.Signature):
    """Optimize swarm workflow.

    You are a WORKFLOW OPTIMIZER. Improve workflows to:
    1. Reduce unnecessary steps
    2. Increase parallelization
    3. Better error handling
    4. Improved data flow
    5. Faster execution

    Maintain quality while improving efficiency.
    """
    current_workflow: str = dspy.InputField(desc="Current workflow description")
    performance_data: str = dspy.InputField(desc="Performance metrics")
    bottlenecks: str = dspy.InputField(desc="Identified bottlenecks")

    optimized_workflow: str = dspy.OutputField(desc="Optimized workflow description")
    changes: str = dspy.OutputField(desc="Workflow changes, separated by |")
    expected_speedup: float = dspy.OutputField(desc="Expected speedup factor")
    trade_offs: str = dspy.OutputField(desc="Trade-offs to consider, separated by |")


class ParameterTuningSignature(dspy.Signature):
    """Tune agent parameters.

    You are a PARAMETER TUNER. Optimize parameters for:
    1. Model selection
    2. Temperature settings
    3. Token limits
    4. Retry policies
    5. Timeout values

    Find the sweet spot for performance.
    """
    current_params: str = dspy.InputField(desc="Current parameters JSON")
    performance_correlation: str = dspy.InputField(desc="How parameters correlate with performance")
    constraints: str = dspy.InputField(desc="Parameter constraints")

    optimized_params: str = dspy.OutputField(desc="Optimized parameters JSON")
    tuning_rationale: str = dspy.OutputField(desc="Rationale for each tuning, separated by |")
    expected_impact: str = dspy.OutputField(desc="Expected impact on performance")


class MetaLearningSignature(dspy.Signature):
    """Extract meta-learnings across swarms.

    You are a META-LEARNER. Identify patterns across domains:
    1. Universal best practices
    2. Cross-domain optimizations
    3. Reusable components
    4. Common failure modes
    5. Transferable improvements

    Build collective intelligence.
    """
    domain_performances: str = dspy.InputField(desc="Performance data from multiple domains")
    improvement_history: str = dspy.InputField(desc="History of improvements and their outcomes")
    swarm_architectures: str = dspy.InputField(desc="Architecture of different swarms")

    universal_insights: str = dspy.OutputField(desc="Insights applicable across domains, separated by |")
    transferable_patterns: str = dspy.OutputField(desc="Patterns that can transfer between swarms, separated by |")
    anti_patterns: str = dspy.OutputField(desc="Anti-patterns to avoid, separated by |")
    architectural_recommendations: str = dspy.OutputField(desc="Architecture recommendations, separated by |")


# =============================================================================
# AGENTS
# =============================================================================

class BaseLearningAgent(DomainAgent):
    """Base class for learning agents. Inherits from DomainAgent for unified infrastructure."""

    def __init__(self, memory=None, context=None, bus=None, signature=None):
        config = DomainAgentConfig(
            name=self.__class__.__name__,
            enable_memory=memory is not None,
            enable_context=context is not None,
        )
        super().__init__(signature=signature, config=config)

        # Ensure LM is configured before child classes create DSPy modules
        self._ensure_initialized()

        if memory is not None:
            self._memory = memory
        if context is not None:
            self._context_manager = context
        self.bus = bus

    def _broadcast(self, event: str, data: Dict[str, Any]):
        """Broadcast event to other agents."""
        if self.bus:
            try:
                from ..agents.axon import Message
                msg = Message(
                    sender=self.__class__.__name__,
                    receiver="broadcast",
                    content={'event': event, **data}
                )
                self.bus.publish(msg)
            except Exception:
                pass


class PerformanceEvaluator(BaseLearningAgent):
    """Evaluates swarm performance."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._analyzer = dspy.ChainOfThought(PerformanceAnalysisSignature)

    async def evaluate(
        self,
        evaluations: List[Evaluation],
        traces: List[ExecutionTrace],
        config: Dict[str, Any]
    ) -> SwarmPerformance:
        """Evaluate swarm performance."""
        try:
            evaluations_str = json.dumps([asdict(e) for e in evaluations], default=str)
            if self.learned_context:
                evaluations_str = f"[Learned Context]: {self.learned_context}\n\n{evaluations_str}"
            result = self._analyzer(
                evaluations=evaluations_str,
                execution_traces=json.dumps([asdict(t) for t in traces], default=str),
                swarm_config=json.dumps(config, default=str)
            )

            strengths = [s.strip() for s in str(result.strengths).split('|') if s.strip()]
            weaknesses = [w.strip() for w in str(result.weaknesses).split('|') if w.strip()]

            # Calculate metrics
            avg_score = sum(e.overall_score for e in evaluations) / len(evaluations) if evaluations else 0
            success_count = sum(1 for e in evaluations if e.result in [EvaluationResult.EXCELLENT, EvaluationResult.GOOD])
            success_rate = success_count / len(evaluations) if evaluations else 0
            avg_time = sum(t.execution_time for t in traces) / len(traces) if traces else 0

            self._broadcast("performance_evaluated", {
                'avg_score': avg_score,
                'success_rate': success_rate
            })

            return SwarmPerformance(
                swarm_name=config.get('name', 'unknown'),
                domain=config.get('domain', 'unknown'),
                avg_score=avg_score,
                success_rate=success_rate,
                avg_execution_time=avg_time,
                total_executions=len(traces),
                evaluations=evaluations,
                weaknesses=weaknesses,
                strengths=strengths
            )

        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            return SwarmPerformance(
                swarm_name="unknown",
                domain="unknown",
                avg_score=0,
                success_rate=0,
                avg_execution_time=0,
                total_executions=0,
                evaluations=[],
                weaknesses=[str(e)],
                strengths=[]
            )


class GoldCurator(BaseLearningAgent):
    """Curates gold standards."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._curator = dspy.ChainOfThought(GoldStandardCreationSignature)

    async def curate(
        self,
        successful_outputs: List[Dict[str, Any]],
        domain: str,
        task_types: List[str]
    ) -> List[GoldStandard]:
        """Create gold standards from successful outputs."""
        try:
            outputs_str = json.dumps(successful_outputs)
            if self.learned_context:
                outputs_str = f"[Learned Context]: {self.learned_context}\n\n{outputs_str}"
            result = self._curator(
                successful_outputs=outputs_str,
                domain=domain,
                task_types=",".join(task_types)
            )

            try:
                gold_data = json.loads(result.gold_standards)
            except Exception:
                gold_data = []

            criteria_list = [c.strip() for c in str(result.evaluation_criteria).split('|') if c.strip()]

            gold_standards = []
            for i, gs in enumerate(gold_data):
                gold_standards.append(GoldStandard(
                    id=gs.get('id', f"gs_{domain}_{i}"),
                    domain=domain,
                    task_type=gs.get('task_type', 'general'),
                    input_data=gs.get('input_data', {}),
                    expected_output=gs.get('expected_output', {}),
                    evaluation_criteria=gs.get('criteria', {'quality': 1.0})
                ))

            self._broadcast("gold_standards_created", {
                'domain': domain,
                'count': len(gold_standards)
            })

            return gold_standards

        except Exception as e:
            logger.error(f"Gold curation failed: {e}")
            return []


class PromptOptimizer(BaseLearningAgent):
    """Optimizes agent prompts."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._optimizer = dspy.ChainOfThought(PromptOptimizationSignature)

    async def optimize(
        self,
        current_prompt: str,
        feedback: List[str],
        failure_cases: List[Dict[str, Any]]
    ) -> OptimizationResult:
        """Optimize a prompt."""
        try:
            prompt_str = current_prompt
            if self.learned_context:
                prompt_str = f"[Learned Context]: {self.learned_context}\n\n{current_prompt}"
            result = self._optimizer(
                current_prompt=prompt_str,
                feedback="\n".join(feedback),
                failure_cases=json.dumps(failure_cases)
            )

            changes = [c.strip() for c in str(result.changes_made).split('|') if c.strip()]

            self._broadcast("prompt_optimized", {
                'changes': len(changes),
                'expected_improvement': float(result.expected_improvement) if result.expected_improvement else 0
            })

            return OptimizationResult(
                optimization_type=OptimizationType.PROMPT,
                original_value=current_prompt,
                optimized_value=str(result.optimized_prompt),
                expected_improvement=float(result.expected_improvement) if result.expected_improvement else 0,
                rationale=str(result.rationale)
            )

        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}")
            return OptimizationResult(
                optimization_type=OptimizationType.PROMPT,
                original_value=current_prompt,
                optimized_value=current_prompt,
                expected_improvement=0,
                rationale=str(e)
            )


class WorkflowOptimizer(BaseLearningAgent):
    """Optimizes swarm workflows."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._optimizer = dspy.ChainOfThought(WorkflowOptimizationSignature)

    async def optimize(
        self,
        current_workflow: str,
        performance_data: Dict[str, Any],
        bottlenecks: List[str]
    ) -> OptimizationResult:
        """Optimize workflow."""
        try:
            workflow_str = current_workflow
            if self.learned_context:
                workflow_str = f"[Learned Context]: {self.learned_context}\n\n{current_workflow}"
            result = self._optimizer(
                current_workflow=workflow_str,
                performance_data=json.dumps(performance_data),
                bottlenecks="\n".join(bottlenecks)
            )

            self._broadcast("workflow_optimized", {
                'expected_speedup': float(result.expected_speedup) if result.expected_speedup else 1.0
            })

            return OptimizationResult(
                optimization_type=OptimizationType.WORKFLOW,
                original_value=current_workflow,
                optimized_value=str(result.optimized_workflow),
                expected_improvement=float(result.expected_speedup) if result.expected_speedup else 0,
                rationale=str(result.changes)
            )

        except Exception as e:
            logger.error(f"Workflow optimization failed: {e}")
            return OptimizationResult(
                optimization_type=OptimizationType.WORKFLOW,
                original_value=current_workflow,
                optimized_value=current_workflow,
                expected_improvement=0,
                rationale=str(e)
            )


class ParameterTuner(BaseLearningAgent):
    """Tunes agent parameters."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._tuner = dspy.ChainOfThought(ParameterTuningSignature)

    async def tune(
        self,
        current_params: Dict[str, Any],
        correlations: Dict[str, float],
        constraints: Dict[str, Any]
    ) -> OptimizationResult:
        """Tune parameters."""
        try:
            params_str = json.dumps(current_params)
            if self.learned_context:
                params_str = f"[Learned Context]: {self.learned_context}\n\n{params_str}"
            result = self._tuner(
                current_params=params_str,
                performance_correlation=json.dumps(correlations),
                constraints=json.dumps(constraints)
            )

            try:
                optimized = json.loads(result.optimized_params)
            except Exception:
                optimized = current_params

            self._broadcast("parameters_tuned", {
                'params_changed': len(optimized)
            })

            return OptimizationResult(
                optimization_type=OptimizationType.PARAMETERS,
                original_value=json.dumps(current_params),
                optimized_value=json.dumps(optimized),
                expected_improvement=0.1,
                rationale=str(result.tuning_rationale)
            )

        except Exception as e:
            logger.error(f"Parameter tuning failed: {e}")
            return OptimizationResult(
                optimization_type=OptimizationType.PARAMETERS,
                original_value=json.dumps(current_params),
                optimized_value=json.dumps(current_params),
                expected_improvement=0,
                rationale=str(e)
            )


class MetaLearner(BaseLearningAgent):
    """Extracts cross-domain meta-learnings."""

    def __init__(self, memory=None, context=None, bus=None, learned_context: str = ""):
        super().__init__(memory, context, bus, learned_context)
        self._learner = dspy.ChainOfThought(MetaLearningSignature)

    async def learn(
        self,
        domain_performances: Dict[str, SwarmPerformance],
        improvement_history: List[Dict[str, Any]],
        architectures: Dict[str, str]
    ) -> Dict[str, Any]:
        """Extract meta-learnings."""
        try:
            perf_data = {
                domain: {
                    'avg_score': p.avg_score,
                    'success_rate': p.success_rate,
                    'strengths': p.strengths,
                    'weaknesses': p.weaknesses
                }
                for domain, p in domain_performances.items()
            }

            perf_data_str = json.dumps(perf_data)
            if self.learned_context:
                perf_data_str = f"[Learned Context]: {self.learned_context}\n\n{perf_data_str}"
            result = self._learner(
                domain_performances=perf_data_str,
                improvement_history=json.dumps(improvement_history, default=str),
                swarm_architectures=json.dumps(architectures)
            )

            insights = [i.strip() for i in str(result.universal_insights).split('|') if i.strip()]
            patterns = [p.strip() for p in str(result.transferable_patterns).split('|') if p.strip()]
            anti_patterns = [a.strip() for a in str(result.anti_patterns).split('|') if a.strip()]
            recommendations = [r.strip() for r in str(result.architectural_recommendations).split('|') if r.strip()]

            self._broadcast("meta_learning_completed", {
                'insights': len(insights),
                'patterns': len(patterns)
            })

            return {
                'universal_insights': insights,
                'transferable_patterns': patterns,
                'anti_patterns': anti_patterns,
                'architectural_recommendations': recommendations
            }

        except Exception as e:
            logger.error(f"Meta-learning failed: {e}")
            return {
                'universal_insights': [],
                'transferable_patterns': [],
                'anti_patterns': [],
                'architectural_recommendations': []
            }


# =============================================================================
# LEARNING SWARM
# =============================================================================

@register_swarm("learning")
class LearningSwarm(DomainSwarm):
    """
    World-Class Learning Swarm (Meta-Swarm).

    Continuously improves all swarms through:
    - Performance evaluation
    - Gold standard curation
    - Prompt optimization
    - Workflow optimization
    - Parameter tuning
    - Cross-domain meta-learning
    """

    AGENT_TEAM = AgentTeam.define(
        (PerformanceEvaluator, "PerformanceEvaluator", "_evaluator"),
        (GoldCurator, "GoldCurator", "_curator"),
        (PromptOptimizer, "PromptOptimizer", "_prompt_optimizer"),
        (WorkflowOptimizer, "WorkflowOptimizer", "_workflow_optimizer"),
        (ParameterTuner, "ParameterTuner", "_parameter_tuner"),
        (MetaLearner, "MetaLearner", "_meta_learner"),
    )

    def __init__(self, config: LearningConfig = None):
        super().__init__(config or LearningConfig())

        # Data stores
        self._all_gold_dbs: Dict[str, GoldStandardDB] = {}
        self._all_histories: Dict[str, ImprovementHistory] = {}

    async def _execute_domain(
        self,
        swarm_name: str,
        **kwargs
    ) -> LearningResult:
        """
        Execute learning cycle.

        This is the domain-specific execution method called by DomainSwarm.execute().
        Agent initialization and pre/post learning hooks are handled by the parent class.
        """
        return await self._evaluate_and_improve_internal(swarm_name, **kwargs)

    async def evaluate_and_improve(
        self,
        swarm_name: str,
        evaluations: List[Evaluation] = None,
        traces: List[ExecutionTrace] = None
    ) -> LearningResult:
        """
        Evaluate and improve a swarm.

        Public method that calls execute() to ensure proper lifecycle hooks.

        Args:
            swarm_name: Name of the swarm to improve
            evaluations: Evaluation results (optional)
            traces: Execution traces (optional)

        Returns:
            LearningResult with improvements
        """
        return await self.execute(swarm_name, evaluations=evaluations, traces=traces)

    async def _evaluate_and_improve_internal(
        self,
        swarm_name: str,
        evaluations: List[Evaluation] = None,
        traces: List[ExecutionTrace] = None
    ) -> LearningResult:
        """
        Internal implementation of evaluate and improve.

        Called by _execute_domain() after agents are initialized
        and pre-learning hooks have run. Delegates to _safe_execute_domain
        which handles try/except, timing, and post-execute learning.

        Args:
            swarm_name: Name of the swarm to improve
            evaluations: Evaluation results (optional)
            traces: Execution traces (optional)

        Returns:
            LearningResult with improvements
        """
        evaluations = evaluations or []
        traces = traces or []

        return await self._safe_execute_domain(
            task_type='swarm_learning',
            default_tools=['performance_evaluate', 'gold_curate', 'prompt_optimize'],
            result_class=LearningResult,
            execute_fn=lambda executor: self._execute_phases(
                executor, swarm_name, evaluations, traces
            ),
            output_data_fn=lambda result: {
                'optimizations_count': len(result.optimizations),
                'gold_standards_created': result.gold_standards_created,
                'insights_count': len(result.cross_domain_insights),
                'avg_score': result.performance.avg_score if result.performance else 0,
            },
            input_data_fn=lambda: {'swarm_name': swarm_name},
        )

    async def _execute_phases(
        self,
        executor,
        swarm_name: str,
        evaluations: List[Evaluation],
        traces: List[ExecutionTrace],
    ) -> LearningResult:
        """
        Domain-specific phase logic using PhaseExecutor.

        Runs the four learning phases:
        1. Performance evaluation
        2. Gold standard curation
        3. Optimization (parallel)
        4. Meta-learning

        Args:
            executor: PhaseExecutor instance for tracing and timing
            swarm_name: Name of the swarm to improve
            evaluations: Evaluation results
            traces: Execution traces

        Returns:
            LearningResult with improvements
        """
        config = self.config

        logger.info(f"LearningSwarm starting: improving {swarm_name}")

        # =================================================================
        # PHASE 1: PERFORMANCE EVALUATION
        # =================================================================
        performance = await executor.run_phase(
            1, "Performance Evaluation", "PerformanceEvaluator", AgentRole.EXPERT,
            self._evaluator.evaluate(
                evaluations,
                traces,
                {'name': swarm_name, 'domain': swarm_name}
            ),
            input_data={'swarm_name': swarm_name, 'evaluations': len(evaluations)},
            tools_used=['performance_evaluate'],
        )

        # =================================================================
        # PHASE 2: GOLD STANDARD CURATION
        # =================================================================
        gold_standards_created = 0

        if config.learning_mode in [LearningMode.CURATE, LearningMode.FULL_CYCLE]:
            # Get successful outputs
            successful_outputs = [
                t.output_data for t in traces
                if t.success
            ][:config.evaluation_samples]

            if successful_outputs:
                gold_standards = await executor.run_phase(
                    2, "Gold Standard Curation", "GoldCurator", AgentRole.ACTOR,
                    self._curator.curate(
                        successful_outputs,
                        swarm_name,
                        ['general']
                    ),
                    input_data={'swarm_name': swarm_name},
                    tools_used=['gold_curate'],
                )
                gold_standards_created = len(gold_standards)

        # =================================================================
        # PHASE 3: OPTIMIZATION (parallel)
        # =================================================================
        optimizations = []

        if config.learning_mode in [LearningMode.OPTIMIZE, LearningMode.FULL_CYCLE]:
            parallel_tasks = []

            if OptimizationType.PROMPT in config.optimization_types or OptimizationType.ALL in config.optimization_types:
                parallel_tasks.append((
                    "PromptOptimizer", AgentRole.ACTOR,
                    self._prompt_optimizer.optimize(
                        "Current prompt placeholder",
                        performance.weaknesses,
                        []
                    ),
                    ['prompt_optimize'],
                ))

            if OptimizationType.WORKFLOW in config.optimization_types or OptimizationType.ALL in config.optimization_types:
                parallel_tasks.append((
                    "WorkflowOptimizer", AgentRole.ACTOR,
                    self._workflow_optimizer.optimize(
                        "Sequential workflow",
                        {'avg_time': performance.avg_execution_time},
                        performance.weaknesses
                    ),
                    ['workflow_optimize'],
                ))

            if OptimizationType.PARAMETERS in config.optimization_types or OptimizationType.ALL in config.optimization_types:
                parallel_tasks.append((
                    "ParameterTuner", AgentRole.ACTOR,
                    self._parameter_tuner.tune(
                        {'temperature': 0.7, 'max_tokens': 4096},
                        {},
                        {}
                    ),
                    ['param_tune'],
                ))

            if parallel_tasks:
                opt_results = await executor.run_parallel(
                    3, "Optimization", parallel_tasks
                )
                for result in opt_results:
                    if isinstance(result, OptimizationResult):
                        optimizations.append(result)

        # =================================================================
        # PHASE 4: META-LEARNING
        # =================================================================
        cross_domain_insights = []

        if config.learning_mode == LearningMode.FULL_CYCLE:
            meta_result = await executor.run_phase(
                4, "Meta-Learning", "MetaLearner", AgentRole.EXPERT,
                self._meta_learner.learn(
                    {swarm_name: performance},
                    [],
                    {swarm_name: "Standard multi-agent architecture"}
                ),
                input_data={'full_cycle': True},
                tools_used=['meta_learn'],
            )

            cross_domain_insights = meta_result.get('universal_insights', [])

        # =================================================================
        # BUILD RESULT
        # =================================================================
        result = LearningResult(
            success=True,
            swarm_name=self.config.name,
            domain=self.config.domain,
            output={'swarm_evaluated': swarm_name},
            execution_time=executor.elapsed(),
            swarm_evaluated=swarm_name,
            performance=performance,
            optimizations=optimizations,
            gold_standards_created=gold_standards_created,
            improvements_suggested=len(optimizations),
            improvements_applied=0,
            cross_domain_insights=cross_domain_insights
        )

        logger.info(f"LearningSwarm complete: {len(optimizations)} optimizations, {gold_standards_created} gold standards")

        return result

    async def run_improvement_cycle(
        self,
        swarm_names: List[str] = None
    ) -> Dict[str, LearningResult]:
        """
        Run improvement cycle for multiple swarms.

        Args:
            swarm_names: Swarms to improve (default: all registered)

        Returns:
            Dictionary of swarm name to learning result
        """
        if not swarm_names:
            swarm_names = SwarmRegistry.list_all()

        results = {}

        for name in swarm_names:
            logger.info(f"🔄 Running improvement cycle for {name}...")
            results[name] = await self.evaluate_and_improve(name)

        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def improve_swarm(swarm_name: str, **kwargs) -> LearningResult:
    """
    One-liner swarm improvement.

    Usage:
        from core.swarms.learning_swarm import improve_swarm
        result = await improve_swarm("coding")
    """
    swarm = LearningSwarm()
    return await swarm.evaluate_and_improve(swarm_name, **kwargs)


def improve_swarm_sync(swarm_name: str, **kwargs) -> LearningResult:
    """Synchronous swarm improvement."""
    return asyncio.run(improve_swarm(swarm_name, **kwargs))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'LearningSwarm',
    'LearningConfig',
    'LearningResult',
    'SwarmPerformance',
    'OptimizationResult',
    'LearningMode',
    'OptimizationType',
    'improve_swarm',
    'improve_swarm_sync',
    # Agents
    'PerformanceEvaluator',
    'GoldCurator',
    'PromptOptimizer',
    'WorkflowOptimizer',
    'ParameterTuner',
    'MetaLearner',
]

"""
Self-Improving Agents
======================

The six agents that form the self-improving feedback loop:
- ExpertAgent: Evaluates outputs against gold standards
- ReviewerAgent: Analyzes patterns and suggests improvements
- PlannerAgent: Plans task execution with optimization
- ActorAgent: Executes tasks with learned improvements
- AuditorAgent: Verifies evaluation quality
- LearnerAgent: Extracts reusable learnings

All agents inherit from MetaAgent for unified infrastructure.

Extracted from base_swarm.py for modularity.
Refactored to use BaseAgent hierarchy (Feb 2026).
"""

import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import asdict

import dspy

from .swarm_types import (
    AgentConfig as SwarmAgentConfig, AgentRole, Evaluation, EvaluationResult,
    ImprovementSuggestion, ImprovementType
)
from .swarm_signatures import (
    ExpertEvaluationSignature, ReviewerAnalysisSignature,
    PlannerOptimizationSignature, ActorExecutionSignature,
    AuditorVerificationSignature, LearnerExtractionSignature,
)
from .evaluation import GoldStandardDB, ImprovementHistory

# Import base class
from ..agents.base import MetaAgent, MetaAgentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# SELF-IMPROVING AGENTS (Refactored to use MetaAgent)
# =============================================================================

class ExpertAgent(MetaAgent):
    """
    Expert Agent - Evaluates outputs against gold standards.

    The Expert is the quality gatekeeper. It:
    1. Compares actual output to expected output
    2. Scores each evaluation criterion
    3. Provides specific feedback
    4. Determines if improvement is needed

    Inherits from MetaAgent for:
    - DSPy module auto-initialization
    - Gold standard database integration
    - Memory and context helpers
    - Retry logic and error handling
    """

    def __init__(self, config: SwarmAgentConfig, gold_db: GoldStandardDB):
        """
        Initialize ExpertAgent.

        Args:
            config: Swarm agent configuration
            gold_db: Gold standard database for evaluation
        """
        # Convert swarm config to MetaAgentConfig
        meta_config = MetaAgentConfig(
            name=config.name or "ExpertAgent",
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        super().__init__(
            signature=ExpertEvaluationSignature,
            config=meta_config,
            gold_db=gold_db,
        )
        self.swarm_config = config

    async def evaluate(
        self,
        gold_standard_id: str,
        actual_output: Dict[str, Any],
        context: str = ""
    ) -> Evaluation:
        """
        Evaluate output against gold standard.

        Uses MetaAgent's evaluate_against_gold for core logic,
        then converts to Evaluation dataclass.
        """
        # Use base class evaluation
        result = await self.evaluate_against_gold(
            gold_id=gold_standard_id,
            output=actual_output,
            context=context
        )

        # Parse result enum
        result_enum = EvaluationResult.NEEDS_IMPROVEMENT
        result_str = str(result.get('result', 'needs_improvement')).upper().replace(' ', '_')
        for er in EvaluationResult:
            if er.value.upper() == result_str:
                result_enum = er
                break

        return Evaluation(
            gold_standard_id=gold_standard_id,
            actual_output=actual_output,
            scores=result.get('scores', {}),
            overall_score=result.get('overall_score', 0.0),
            result=result_enum,
            feedback=result.get('feedback', [])
        )


class ReviewerAgent(MetaAgent):
    """
    Reviewer Agent - Analyzes patterns and suggests improvements.

    The Reviewer is the strategist. It:
    1. Analyzes multiple evaluations for patterns
    2. Identifies systematic issues
    3. Suggests concrete improvements
    4. Prioritizes based on impact

    Inherits from MetaAgent for unified infrastructure.
    """

    def __init__(self, config: SwarmAgentConfig, history: ImprovementHistory):
        """
        Initialize ReviewerAgent.

        Args:
            config: Swarm agent configuration
            history: Improvement history tracker
        """
        meta_config = MetaAgentConfig(
            name=config.name or "ReviewerAgent",
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        super().__init__(
            signature=ReviewerAnalysisSignature,
            config=meta_config,
            improvement_history=history,
        )
        self.swarm_config = config
        self.history = history

    async def analyze_and_suggest(
        self,
        evaluations: List[Evaluation],
        agent_configs: Dict[AgentRole, SwarmAgentConfig]
    ) -> List[ImprovementSuggestion]:
        """Analyze evaluations and suggest improvements."""
        if not evaluations:
            return []

        # Use base class method
        result_dicts = await self.analyze_and_suggest_improvements(
            evaluations=[asdict(e) for e in evaluations],
            agent_configs={k.value: asdict(v) for k, v in agent_configs.items()}
        )

        # Convert to ImprovementSuggestion objects
        suggestions = []
        for s in result_dicts:
            try:
                suggestions.append(ImprovementSuggestion(
                    agent_role=AgentRole(s.get('agent_role', 'actor')),
                    improvement_type=ImprovementType(s.get('improvement_type', 'prompt_refinement')),
                    description=s.get('description', ''),
                    priority=int(s.get('priority', 3)),
                    expected_impact=float(s.get('expected_impact', 0.5)),
                    implementation_details=s.get('implementation_details', {}),
                    based_on_evaluations=[e.gold_standard_id for e in evaluations]
                ))
            except Exception as e:
                logger.debug(f"Failed to parse suggestion: {e}")

        return suggestions


class PlannerAgent(MetaAgent):
    """
    Planner Agent - Plans task execution with optimization.

    The Planner is the architect. It:
    1. Breaks down tasks into subtasks
    2. Optimizes based on past performance
    3. Identifies parallelization opportunities
    4. Plans error recovery

    Inherits from MetaAgent for unified infrastructure.
    """

    def __init__(self, config: SwarmAgentConfig, history: ImprovementHistory):
        """
        Initialize PlannerAgent.

        Args:
            config: Swarm agent configuration
            history: Improvement history tracker
        """
        meta_config = MetaAgentConfig(
            name=config.name or "PlannerAgent",
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        super().__init__(
            signature=PlannerOptimizationSignature,
            config=meta_config,
            improvement_history=history,
        )
        self.swarm_config = config
        self.history = history

    async def create_plan(
        self,
        task_description: str,
        relevant_improvements: List[ImprovementSuggestion] = None,
        past_plans: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create an optimized execution plan."""
        try:
            # Execute using base class infrastructure
            result = await self.execute(
                task_description=task_description,
                improvement_suggestions=json.dumps(
                    [asdict(s) for s in (relevant_improvements or [])],
                    default=str
                ),
                past_plans=json.dumps(past_plans or [], default=str)
            )

            if not result.success:
                return {
                    'plan': {'tasks': [{'name': 'execute', 'description': task_description}]},
                    'rationale': f'Fallback plan due to error: {result.error}',
                    'risks': [result.error or 'Unknown error']
                }

            output = result.output or {}

            # Parse plan
            try:
                plan = json.loads(output.get('optimized_plan', '{}'))
            except (json.JSONDecodeError, TypeError):
                plan = {'tasks': [{'name': 'execute', 'description': task_description}]}

            # Parse risk mitigations
            risks_str = str(output.get('risk_mitigations', ''))
            risks = [r.strip() for r in risks_str.split('|') if r.strip()]

            return {
                'plan': plan,
                'rationale': str(output.get('rationale', '')),
                'risks': risks
            }

        except Exception as e:
            logger.error(f"Planner failed: {e}")
            return {
                'plan': {'tasks': [{'name': 'execute', 'description': task_description}]},
                'rationale': 'Fallback plan due to error',
                'risks': [str(e)]
            }


class ActorAgent(MetaAgent):
    """
    Actor Agent - Executes tasks with learned improvements.

    The Actor is the executor. It:
    1. Executes individual tasks
    2. Applies learnings from past feedback
    3. Adapts based on improvement suggestions
    4. Reports confidence levels

    Inherits from MetaAgent for unified infrastructure.
    """

    def __init__(self, config: SwarmAgentConfig, history: ImprovementHistory):
        """
        Initialize ActorAgent.

        Args:
            config: Swarm agent configuration
            history: Improvement history tracker
        """
        meta_config = MetaAgentConfig(
            name=config.name or "ActorAgent",
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        super().__init__(
            signature=ActorExecutionSignature,
            config=meta_config,
            improvement_history=history,
        )
        self.swarm_config = config
        self.history = history

    async def execute_task(
        self,
        task: str,
        context: Dict[str, Any],
        learnings: List[str] = None
    ) -> Tuple[Dict[str, Any], float, List[str]]:
        """
        Execute a task and return output, confidence, and applied learnings.

        Note: Renamed from execute() to avoid conflict with BaseAgent.execute()
        """
        try:
            # Get relevant successful improvements
            successful = self.history.get_successful_improvements(AgentRole.ACTOR)
            all_learnings = (learnings or []) + [
                s['suggestion']['description'] for s in successful[:5]
            ]

            # Execute using base class
            result = await self.execute(
                task=task,
                context=json.dumps(context, default=str),
                learnings="\n".join(all_learnings) if all_learnings else "No prior learnings"
            )

            if not result.success:
                return {'error': result.error}, 0.0, []

            output = result.output or {}

            # Parse output
            try:
                parsed_output = json.loads(output.get('output', '{}'))
            except (json.JSONDecodeError, TypeError):
                parsed_output = {'result': str(output.get('output', ''))}

            confidence = float(output.get('confidence', 0.5))
            applied_str = str(output.get('applied_learnings', ''))
            applied = [l.strip() for l in applied_str.split('|') if l.strip()]

            return parsed_output, confidence, applied

        except Exception as e:
            logger.error(f"Actor execution failed: {e}")
            return {'error': str(e)}, 0.0, []


class AuditorAgent(MetaAgent):
    """
    Auditor Agent - Verifies evaluation quality and consistency.

    The Auditor is the trust verifier. It:
    1. Checks evaluation scores for consistency
    2. Verifies reasoning matches scores
    3. Detects potential evaluation errors
    4. Non-blocking: defaults to passed=True on any failure

    Inherits from MetaAgent for unified infrastructure.
    """

    def __init__(self, config: SwarmAgentConfig):
        """
        Initialize AuditorAgent.

        Args:
            config: Swarm agent configuration
        """
        meta_config = MetaAgentConfig(
            name=config.name or "AuditorAgent",
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        super().__init__(
            signature=AuditorVerificationSignature,
            config=meta_config,
        )
        self.swarm_config = config

    async def audit_evaluation(
        self,
        evaluation: Dict[str, Any],
        output_data: Dict[str, Any],
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Audit an evaluation for consistency and quality.

        Non-blocking: catches all exceptions and defaults to passed=True.
        """
        try:
            result = await self.execute(
                evaluation_data=json.dumps(evaluation, default=str),
                output_data=json.dumps(output_data, default=str),
                context=context or "No additional context"
            )

            if not result.success:
                return {
                    'passed': True,
                    'reasoning': f"Audit skipped (non-blocking): {result.error}"
                }

            output = result.output or {}
            passed = bool(output.get('audit_passed', True))
            reasoning = str(output.get('reasoning', 'Audit completed'))

            return {
                'passed': passed,
                'reasoning': reasoning,
                'confidence': float(output.get('confidence', 0.5))
            }

        except Exception as e:
            logger.debug(f"Auditor defaulting to passed (non-blocking): {e}")
            return {
                'passed': True,
                'reasoning': f"Audit skipped (non-blocking): {str(e)}"
            }


class LearnerAgent(MetaAgent):
    """
    Learner Agent - Extracts reusable learnings from excellent executions.

    The Learner is the knowledge distiller. It:
    1. Analyzes excellent execution outputs
    2. Extracts reusable patterns and quality factors
    3. Provides domain-specific insights
    4. Scores reusability for prioritization

    Inherits from MetaAgent for unified infrastructure.
    """

    def __init__(self, config: SwarmAgentConfig):
        """
        Initialize LearnerAgent.

        Args:
            config: Swarm agent configuration
        """
        meta_config = MetaAgentConfig(
            name=config.name or "LearnerAgent",
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        super().__init__(
            signature=LearnerExtractionSignature,
            config=meta_config,
        )
        self.swarm_config = config

    async def extract_learnings_from_execution(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        evaluation: Dict[str, Any],
        domain: str = "general"
    ) -> List[str]:
        """
        Extract reusable learnings from an excellent execution.

        Note: Renamed from extract_learnings() to be more specific.
        """
        # Use base class method
        return await self.extract_learnings(
            input_data=input_data,
            output_data=output_data,
            evaluation=evaluation,
            domain=domain
        )


__all__ = [
    'ExpertAgent',
    'ReviewerAgent',
    'PlannerAgent',
    'ActorAgent',
    'AuditorAgent',
    'LearnerAgent',
]

"""
Self-Improving Agents
======================

The four agents that form the self-improving feedback loop:
- ExpertAgent: Evaluates outputs against gold standards
- ReviewerAgent: Analyzes patterns and suggests improvements
- PlannerAgent: Plans task execution with optimization
- ActorAgent: Executes tasks with learned improvements

Extracted from base_swarm.py for modularity.
"""

import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import asdict

import dspy

from .swarm_types import (
    AgentConfig, AgentRole, Evaluation, EvaluationResult,
    ImprovementSuggestion, ImprovementType
)
from .swarm_signatures import (
    ExpertEvaluationSignature, ReviewerAnalysisSignature,
    PlannerOptimizationSignature, ActorExecutionSignature,
    AuditorVerificationSignature, LearnerExtractionSignature,
)
from .evaluation import GoldStandardDB, ImprovementHistory

logger = logging.getLogger(__name__)


# =============================================================================
# SELF-IMPROVING AGENTS
# =============================================================================

class ExpertAgent:
    """
    Expert Agent - Evaluates outputs against gold standards.

    The Expert is the quality gatekeeper. It:
    1. Compares actual output to expected output
    2. Scores each evaluation criterion
    3. Provides specific feedback
    4. Determines if improvement is needed
    """

    def __init__(self, config: AgentConfig, gold_db: GoldStandardDB):
        self.config = config
        self.gold_db = gold_db
        self._evaluator = dspy.ChainOfThought(ExpertEvaluationSignature)

    async def evaluate(
        self,
        gold_standard_id: str,
        actual_output: Dict[str, Any],
        context: str = ""
    ) -> Evaluation:
        """Evaluate output against gold standard."""
        gold_standard = self.gold_db.get(gold_standard_id)
        if not gold_standard:
            raise ValueError(f"Gold standard not found: {gold_standard_id}")

        try:
            result = self._evaluator(
                gold_standard=json.dumps({
                    'expected_output': gold_standard.expected_output,
                    'criteria': gold_standard.evaluation_criteria
                }),
                actual_output=json.dumps(actual_output),
                context=context
            )

            # Parse scores
            try:
                scores = json.loads(result.scores)
            except:
                scores = {}

            # Parse result enum
            result_enum = EvaluationResult.NEEDS_IMPROVEMENT
            result_str = str(result.result).upper().replace(' ', '_')
            for er in EvaluationResult:
                if er.value.upper() == result_str:
                    result_enum = er
                    break

            # Parse feedback
            feedback = [f.strip() for f in str(result.feedback).split('|') if f.strip()]

            return Evaluation(
                gold_standard_id=gold_standard_id,
                actual_output=actual_output,
                scores=scores,
                overall_score=float(result.overall_score) if result.overall_score else 0.0,
                result=result_enum,
                feedback=feedback
            )

        except Exception as e:
            logger.error(f"Expert evaluation failed: {e}")
            return Evaluation(
                gold_standard_id=gold_standard_id,
                actual_output=actual_output,
                scores={},
                overall_score=0.0,
                result=EvaluationResult.FAILED,
                feedback=[f"Evaluation error: {str(e)}"]
            )


class ReviewerAgent:
    """
    Reviewer Agent - Analyzes patterns and suggests improvements.

    The Reviewer is the strategist. It:
    1. Analyzes multiple evaluations for patterns
    2. Identifies systematic issues
    3. Suggests concrete improvements
    4. Prioritizes based on impact
    """

    def __init__(self, config: AgentConfig, history: ImprovementHistory):
        self.config = config
        self.history = history
        self._analyzer = dspy.ChainOfThought(ReviewerAnalysisSignature)

    async def analyze_and_suggest(
        self,
        evaluations: List[Evaluation],
        agent_configs: Dict[AgentRole, AgentConfig]
    ) -> List[ImprovementSuggestion]:
        """Analyze evaluations and suggest improvements."""
        if not evaluations:
            return []

        try:
            # Get improvement history
            past_improvements = self.history.get_successful_improvements()

            result = self._analyzer(
                evaluations=json.dumps([asdict(e) for e in evaluations], default=str),
                agent_configs=json.dumps({k.value: asdict(v) for k, v in agent_configs.items()}, default=str),
                improvement_history=json.dumps(past_improvements, default=str)
            )

            # Parse suggestions
            try:
                suggestions_data = json.loads(result.suggestions)
            except:
                suggestions_data = []

            suggestions = []
            for s in suggestions_data:
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

        except Exception as e:
            logger.error(f"Reviewer analysis failed: {e}")
            return []


class PlannerAgent:
    """
    Planner Agent - Plans task execution with optimization.

    The Planner is the architect. It:
    1. Breaks down tasks into subtasks
    2. Optimizes based on past performance
    3. Identifies parallelization opportunities
    4. Plans error recovery
    """

    def __init__(self, config: AgentConfig, history: ImprovementHistory):
        self.config = config
        self.history = history
        self._optimizer = dspy.ChainOfThought(PlannerOptimizationSignature)

    async def create_plan(
        self,
        task_description: str,
        relevant_improvements: List[ImprovementSuggestion] = None,
        past_plans: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create an optimized execution plan."""
        try:
            result = self._optimizer(
                task_description=task_description,
                improvement_suggestions=json.dumps(
                    [asdict(s) for s in (relevant_improvements or [])],
                    default=str
                ),
                past_plans=json.dumps(past_plans or [], default=str)
            )

            # Parse plan
            try:
                plan = json.loads(result.optimized_plan)
            except:
                plan = {'tasks': [{'name': 'execute', 'description': task_description}]}

            # Parse risk mitigations
            risks = [r.strip() for r in str(result.risk_mitigations).split('|') if r.strip()]

            return {
                'plan': plan,
                'rationale': str(result.rationale),
                'risks': risks
            }

        except Exception as e:
            logger.error(f"Planner failed: {e}")
            return {
                'plan': {'tasks': [{'name': 'execute', 'description': task_description}]},
                'rationale': 'Fallback plan due to error',
                'risks': [str(e)]
            }


class ActorAgent:
    """
    Actor Agent - Executes tasks with learned improvements.

    The Actor is the executor. It:
    1. Executes individual tasks
    2. Applies learnings from past feedback
    3. Adapts based on improvement suggestions
    4. Reports confidence levels
    """

    def __init__(self, config: AgentConfig, history: ImprovementHistory):
        self.config = config
        self.history = history
        self._executor = dspy.ChainOfThought(ActorExecutionSignature)

    async def execute(
        self,
        task: str,
        context: Dict[str, Any],
        learnings: List[str] = None
    ) -> Tuple[Dict[str, Any], float, List[str]]:
        """Execute a task and return output, confidence, and applied learnings."""
        try:
            # Get relevant successful improvements
            successful = self.history.get_successful_improvements(AgentRole.ACTOR)
            all_learnings = (learnings or []) + [
                s['suggestion']['description'] for s in successful[:5]
            ]

            result = self._executor(
                task=task,
                context=json.dumps(context, default=str),
                learnings="\n".join(all_learnings) if all_learnings else "No prior learnings"
            )

            # Parse output
            try:
                output = json.loads(result.output)
            except:
                output = {'result': str(result.output)}

            confidence = float(result.confidence) if result.confidence else 0.5
            applied = [l.strip() for l in str(result.applied_learnings).split('|') if l.strip()]

            return output, confidence, applied

        except Exception as e:
            logger.error(f"Actor execution failed: {e}")
            return {'error': str(e)}, 0.0, []


class AuditorAgent:
    """
    Auditor Agent - Verifies evaluation quality and consistency.

    The Auditor is the trust verifier. It:
    1. Checks evaluation scores for consistency
    2. Verifies reasoning matches scores
    3. Detects potential evaluation errors
    4. Non-blocking: defaults to passed=True on any failure
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._verifier = dspy.ChainOfThought(AuditorVerificationSignature)

    async def audit_evaluation(
        self,
        evaluation: Dict[str, Any],
        output_data: Dict[str, Any],
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Audit an evaluation for consistency and quality.

        Non-blocking: catches all exceptions and defaults to passed=True.

        Args:
            evaluation: Evaluation data dict (scores, feedback, etc.)
            output_data: The output that was evaluated
            context: Additional context about the task

        Returns:
            Dict with 'passed' (bool) and 'reasoning' (str)
        """
        try:
            result = self._verifier(
                evaluation_data=json.dumps(evaluation, default=str),
                output_data=json.dumps(output_data, default=str),
                context=context or "No additional context"
            )

            passed = bool(result.audit_passed) if result.audit_passed is not None else True
            reasoning = str(result.reasoning) if result.reasoning else "Audit completed"

            return {
                'passed': passed,
                'reasoning': reasoning,
                'confidence': float(result.confidence) if result.confidence else 0.5
            }

        except Exception as e:
            logger.debug(f"Auditor defaulting to passed (non-blocking): {e}")
            return {
                'passed': True,
                'reasoning': f"Audit skipped (non-blocking): {str(e)}"
            }


class LearnerAgent:
    """
    Learner Agent - Extracts reusable learnings from excellent executions.

    The Learner is the knowledge distiller. It:
    1. Analyzes excellent execution outputs
    2. Extracts reusable patterns and quality factors
    3. Provides domain-specific insights
    4. Scores reusability for prioritization
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._extractor = dspy.ChainOfThought(LearnerExtractionSignature)

    async def extract_learnings(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        evaluation: Dict[str, Any],
        domain: str = "general"
    ) -> List[str]:
        """
        Extract reusable learnings from an excellent execution.

        Args:
            input_data: Task input parameters
            output_data: Task output
            evaluation: Evaluation data (scores, feedback)
            domain: Domain of the task

        Returns:
            List of learning strings
        """
        try:
            result = self._extractor(
                input_data=json.dumps(input_data, default=str),
                output_data=json.dumps(output_data, default=str),
                evaluation_data=json.dumps(evaluation, default=str),
                domain=domain
            )

            learnings = [
                l.strip() for l in str(result.learnings).split('|')
                if l.strip()
            ]

            return learnings

        except Exception as e:
            logger.debug(f"Learner extraction failed: {e}")
            return []


__all__ = [
    'ExpertAgent',
    'ReviewerAgent',
    'PlannerAgent',
    'ActorAgent',
    'AuditorAgent',
    'LearnerAgent',
]

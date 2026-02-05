"""
DSPy Signatures for Self-Improving Loop
=========================================

DSPy signature definitions for the Expert/Reviewer/Planner/Actor loop:
- ExpertEvaluationSignature: Evaluate output against gold standard
- ReviewerAnalysisSignature: Analyze evaluations and suggest improvements
- PlannerOptimizationSignature: Optimize task planning
- ActorExecutionSignature: Execute tasks with learned improvements

Extracted from base_swarm.py for modularity.
"""

import dspy


class ExpertEvaluationSignature(dspy.Signature):
    """Evaluate output against gold standard.

    You are an EXPERT EVALUATOR. Compare actual output against expected output
    using the provided evaluation criteria. Be STRICT but FAIR.

    Score each criterion from 0.0 to 1.0:
    - 1.0: Perfect match or exceeds expectations
    - 0.8: Minor deviations, still excellent
    - 0.6: Acceptable with some issues
    - 0.4: Significant gaps
    - 0.2: Major problems
    - 0.0: Complete failure

    Provide specific, actionable feedback.
    """
    gold_standard: str = dspy.InputField(desc="JSON of expected output and criteria")
    actual_output: str = dspy.InputField(desc="JSON of actual output to evaluate")
    context: str = dspy.InputField(desc="Additional context about the task")

    scores: str = dspy.OutputField(desc="JSON dict of criterion -> score (0.0-1.0)")
    overall_score: float = dspy.OutputField(desc="Weighted overall score 0.0-1.0")
    result: str = dspy.OutputField(desc="EXCELLENT, GOOD, ACCEPTABLE, NEEDS_IMPROVEMENT, or FAILED")
    feedback: str = dspy.OutputField(desc="Specific feedback items separated by |")


class ReviewerAnalysisSignature(dspy.Signature):
    """Analyze evaluations and suggest improvements.

    You are a SENIOR REVIEWER analyzing patterns in agent performance.
    Based on multiple evaluations, identify systematic issues and suggest
    concrete improvements.

    Focus on:
    1. Recurring failure patterns
    2. Prompt engineering improvements
    3. Workflow optimizations
    4. Parameter tuning opportunities
    """
    evaluations: str = dspy.InputField(desc="JSON list of recent evaluations")
    agent_configs: str = dspy.InputField(desc="JSON of current agent configurations")
    improvement_history: str = dspy.InputField(desc="JSON of past improvements and their outcomes")

    analysis: str = dspy.OutputField(desc="Analysis of patterns and issues")
    suggestions: str = dspy.OutputField(desc="JSON list of ImprovementSuggestion objects")
    priority_actions: str = dspy.OutputField(desc="Top 3 priority actions separated by |")


class PlannerOptimizationSignature(dspy.Signature):
    """Optimize task planning based on feedback.

    You are a PLANNING EXPERT. Given improvement suggestions and past performance,
    optimize how tasks should be broken down and sequenced.

    Consider:
    1. Task granularity
    2. Dependency ordering
    3. Parallel execution opportunities
    4. Error recovery strategies
    """
    task_description: str = dspy.InputField(desc="Description of the task to plan")
    improvement_suggestions: str = dspy.InputField(desc="JSON of relevant improvements")
    past_plans: str = dspy.InputField(desc="JSON of similar past plans and their success rates")

    optimized_plan: str = dspy.OutputField(desc="JSON of optimized task breakdown")
    rationale: str = dspy.OutputField(desc="Explanation of planning decisions")
    risk_mitigations: str = dspy.OutputField(desc="Identified risks and mitigations separated by |")


class ActorExecutionSignature(dspy.Signature):
    """Execute a task with self-awareness of improvement areas.

    You are an EXPERT ACTOR. Execute the task while being mindful of
    past feedback and improvement suggestions.

    Apply learnings from:
    1. Previous evaluation feedback
    2. Reviewer suggestions
    3. Quality standards
    """
    task: str = dspy.InputField(desc="Task to execute")
    context: str = dspy.InputField(desc="Execution context and requirements")
    learnings: str = dspy.InputField(desc="Relevant learnings from past executions")

    output: str = dspy.OutputField(desc="Task output")
    confidence: float = dspy.OutputField(desc="Confidence in output 0.0-1.0")
    applied_learnings: str = dspy.OutputField(desc="Which learnings were applied, separated by |")


class AuditorVerificationSignature(dspy.Signature):
    """Verify evaluation quality and check for inconsistencies.

    You are an AUDITOR verifying the quality of evaluations.
    Check for:
    1. Consistency between scores and feedback
    2. Reasonable score ranges for the given output
    3. No contradictions in the evaluation reasoning
    4. Proper application of evaluation criteria
    """
    evaluation_data: str = dspy.InputField(desc="JSON of evaluation scores and feedback")
    output_data: str = dspy.InputField(desc="JSON of the output that was evaluated")
    context: str = dspy.InputField(desc="Additional context about the task and evaluation")

    audit_passed: bool = dspy.OutputField(desc="True if evaluation is consistent and trustworthy")
    reasoning: str = dspy.OutputField(desc="Explanation of audit findings")
    confidence: float = dspy.OutputField(desc="Confidence in audit result 0.0-1.0")


class LearnerExtractionSignature(dspy.Signature):
    """Extract reusable learnings from successful executions.

    You are a LEARNING EXTRACTOR. Analyze excellent executions to identify
    reusable patterns, quality factors, and domain-specific insights.

    Focus on:
    1. What made this execution excellent
    2. Patterns that can be applied to future tasks
    3. Domain-specific quality factors
    4. Reusable techniques or approaches
    """
    input_data: str = dspy.InputField(desc="JSON of task input parameters")
    output_data: str = dspy.InputField(desc="JSON of task output")
    evaluation_data: str = dspy.InputField(desc="JSON of evaluation scores and feedback")
    domain: str = dspy.InputField(desc="Domain of the task")

    learnings: str = dspy.OutputField(desc="Extracted learnings separated by |")
    quality_factors: str = dspy.OutputField(desc="Key quality factors separated by |")
    reusability_score: float = dspy.OutputField(desc="How reusable these learnings are 0.0-1.0")


__all__ = [
    'ExpertEvaluationSignature',
    'ReviewerAnalysisSignature',
    'PlannerOptimizationSignature',
    'ActorExecutionSignature',
    'AuditorVerificationSignature',
    'LearnerExtractionSignature',
]

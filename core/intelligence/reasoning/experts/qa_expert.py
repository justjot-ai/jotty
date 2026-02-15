"""
QA Engineer Expert Agent

Evaluates test strategy for:
- Test coverage (unit, integration, e2e)
- Test cases and scenarios
- Performance and load testing
- Quality metrics
"""

from typing import Any, Dict, List, Optional

import dspy

from .base_expert import BaseExpert


class TestStrategyGenerator(dspy.Signature):
    """Generate comprehensive test strategy with test cases."""

    backend_architecture: str = dspy.InputField(desc="Backend architecture from backend developer")
    previous_feedback: str = dspy.InputField(desc="Feedback from previous iterations")
    test_strategy: str = dspy.OutputField(
        desc="Test strategy with unit tests, integration tests, e2e tests, test cases"
    )


class QAExpertAgent(BaseExpert):
    """Expert in quality assurance and testing strategies."""

    @property
    def domain(self) -> str:
        return "quality_assurance"

    @property
    def description(self) -> str:
        return "Expert in test strategies, test cases, and quality metrics"

    def _create_domain_agent(self, improvements: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Create QA Engineer agent."""
        return dspy.ChainOfThought(TestStrategyGenerator)

    def _create_domain_teacher(self) -> Any:
        """Create teacher agent (not used for QA)."""
        return None

    async def _evaluate_domain(
        self, output: Any, gold_standard: str, task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate test strategy quality."""

        score = 0.0
        issues = []

        # Check for key components
        has_unit_tests = (
            "unit test" in output.lower() or "jest" in output.lower() or "vitest" in output.lower()
        )
        has_integration = "integration test" in output.lower() or "api test" in output.lower()
        has_e2e = (
            "e2e" in output.lower()
            or "end-to-end" in output.lower()
            or "playwright" in output.lower()
            or "cypress" in output.lower()
        )
        has_test_cases = (
            "test case" in output.lower()
            or "scenario" in output.lower()
            or "given" in output.lower()
        )
        has_coverage = (
            "coverage" in output.lower()
            or "metric" in output.lower()
            or "threshold" in output.lower()
        )

        # Scoring
        if has_unit_tests:
            score += 0.25
        else:
            issues.append("Missing unit test strategy")

        if has_integration:
            score += 0.2
        else:
            issues.append("Missing integration test approach")

        if has_e2e:
            score += 0.2
        else:
            issues.append("Missing e2e test strategy")

        if has_test_cases:
            score += 0.2
        else:
            issues.append("Missing specific test cases/scenarios")

        if has_coverage:
            score += 0.15
        else:
            issues.append("Missing coverage metrics/thresholds")

        # Length check
        if len(output) < 700:
            score *= 0.8
            issues.append("Test strategy too brief (< 700 chars)")

        status = (
            "EXCELLENT"
            if score >= 0.9
            else "GOOD" if score >= 0.7 else "NEEDS_IMPROVEMENT" if score >= 0.5 else "POOR"
        )

        suggestions = ""
        if score < 0.9:
            suggestions = "Add: " + ", ".join(issues[:3]) if issues else "Expand test coverage"

        return {"score": score, "status": status, "issues": issues, "suggestions": suggestions}

    def _get_default_training_cases(self) -> List[Dict[str, Any]]:
        return []

    def _get_default_validation_cases(self) -> List[Dict[str, Any]]:
        return []

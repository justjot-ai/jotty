"""
Product Manager Expert Agent

Evaluates product requirements for:
- Clarity and completeness
- User stories and acceptance criteria
- Business value and metrics
- Technical feasibility
"""

from typing import Any, Dict, List, Optional

import dspy

from .base_expert import BaseExpert


class ProductRequirementsGenerator(dspy.Signature):
    """Generate product requirements with user stories."""

    feature_description: str = dspy.InputField()
    previous_feedback: str = dspy.InputField(desc="Feedback from previous iterations")
    requirements: str = dspy.OutputField(
        desc="Complete PRD with user stories, acceptance criteria, metrics"
    )


class ProductManagerExpertAgent(BaseExpert):
    """Expert in product management and requirements definition."""

    @property
    def domain(self) -> str:
        return "product_management"

    @property
    def description(self) -> str:
        return "Expert in defining product requirements, user stories, and success metrics"

    def _create_domain_agent(self, improvements: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Create Product Manager agent."""
        return dspy.ChainOfThought(ProductRequirementsGenerator)

    def _create_domain_teacher(self) -> Any:
        """Create teacher agent (not used for PM)."""
        return None

    async def _evaluate_domain(
        self, output: Any, gold_standard: str, task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate product requirements quality."""

        score = 0.0
        issues = []

        # Check for key components
        has_user_stories = "user story" in output.lower() or "as a" in output.lower()
        has_acceptance_criteria = (
            "acceptance criteria" in output.lower() or "given" in output.lower()
        )
        has_metrics = (
            "metric" in output.lower() or "kpi" in output.lower() or "measure" in output.lower()
        )
        has_business_value = "value" in output.lower() or "benefit" in output.lower()

        # Scoring
        if has_user_stories:
            score += 0.25
        else:
            issues.append("Missing user stories (As a... I want... So that...)")

        if has_acceptance_criteria:
            score += 0.25
        else:
            issues.append("Missing acceptance criteria (Given/When/Then)")

        if has_metrics:
            score += 0.25
        else:
            issues.append("Missing success metrics/KPIs")

        if has_business_value:
            score += 0.25
        else:
            issues.append("Missing business value justification")

        # Length check (should be comprehensive)
        if len(output) < 500:
            score *= 0.8
            issues.append("Requirements too brief (< 500 chars)")

        status = (
            "EXCELLENT"
            if score >= 0.9
            else "GOOD" if score >= 0.7 else "NEEDS_IMPROVEMENT" if score >= 0.5 else "POOR"
        )

        suggestions = ""
        if score < 0.9:
            suggestions = (
                "Add more detail to: " + ", ".join(issues[:3])
                if issues
                else "Expand on current requirements"
            )

        return {"score": score, "status": status, "issues": issues, "suggestions": suggestions}

    def _get_default_training_cases(self) -> List[Dict[str, Any]]:
        return []

    def _get_default_validation_cases(self) -> List[Dict[str, Any]]:
        return []

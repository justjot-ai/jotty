"""
UX Researcher Expert Agent

Evaluates user research for:
- User personas and demographics
- Pain points and needs
- User journey mapping
- Research methodology
"""

import dspy
from typing import Dict, Any, List, Optional
from .base_expert import BaseExpert


class UXResearchGenerator(dspy.Signature):
    """Generate UX research with personas and user journeys."""
    requirements: str = dspy.InputField(desc="Product requirements from PM")
    previous_feedback: str = dspy.InputField(desc="Feedback from previous iterations")
    research: str = dspy.OutputField(desc="User research with personas, pain points, journey maps")


class UXResearcherExpertAgent(BaseExpert):
    """Expert in UX research and user understanding."""

    @property
    def domain(self) -> str:
        return "ux_research"

    @property
    def description(self) -> str:
        return "Expert in user research, personas, and journey mapping"

    def _create_domain_agent(self, improvements: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Create UX Researcher agent."""
        return dspy.ChainOfThought(UXResearchGenerator)

    def _create_domain_teacher(self) -> Any:
        """Create teacher agent (not used for UX)."""
        return None

    async def _evaluate_domain(self, output: Any, gold_standard: str, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate UX research quality."""

        score = 0.0
        issues = []

        # Check for key components
        has_personas = "persona" in output.lower() or "user profile" in output.lower()
        has_pain_points = "pain point" in output.lower() or "frustration" in output.lower() or "challenge" in output.lower()
        has_journey = "journey" in output.lower() or "workflow" in output.lower() or "flow" in output.lower()
        has_goals = "goal" in output.lower() or "objective" in output.lower() or "need" in output.lower()

        # Scoring
        if has_personas:
            score += 0.3
        else:
            issues.append("Missing user personas")

        if has_pain_points:
            score += 0.25
        else:
            issues.append("Missing pain points analysis")

        if has_journey:
            score += 0.25
        else:
            issues.append("Missing user journey/workflow")

        if has_goals:
            score += 0.2
        else:
            issues.append("Missing user goals/needs")

        # Length check
        if len(output) < 600:
            score *= 0.8
            issues.append("Research too brief (< 600 chars)")

        status = "EXCELLENT" if score >= 0.9 else "GOOD" if score >= 0.7 else "NEEDS_IMPROVEMENT" if score >= 0.5 else "POOR"

        suggestions = ""
        if score < 0.9:
            suggestions = "Add: " + ", ".join(issues[:3]) if issues else "Expand research depth"

        return {
            'score': score,
            'status': status,
            'issues': issues,
            'suggestions': suggestions
        }

    def _get_default_training_cases(self) -> List[Dict[str, Any]]:
        return []

    def _get_default_validation_cases(self) -> List[Dict[str, Any]]:
        return []

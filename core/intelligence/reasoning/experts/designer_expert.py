"""
Designer Expert Agent

Evaluates UI/UX design for:
- Visual hierarchy and layout
- Component structure
- Accessibility and usability
- Design system consistency
"""

import dspy
from typing import Dict, Any, List, Optional
from .base_expert import BaseExpert


class DesignGenerator(dspy.Signature):
    """Generate UI/UX design with wireframes and component structure."""
    ux_research: str = dspy.InputField(desc="UX research from researcher")
    previous_feedback: str = dspy.InputField(desc="Feedback from previous iterations")
    design: str = dspy.OutputField(desc="UI/UX design with Mermaid wireframes, component hierarchy, design tokens")


class DesignerExpertAgent(BaseExpert):
    """Expert in UI/UX design and visual systems."""

    @property
    def domain(self) -> str:
        return "ui_ux_design"

    @property
    def description(self) -> str:
        return "Expert in UI/UX design, wireframes, and design systems"

    def _create_domain_agent(self, improvements: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Create Designer agent."""
        return dspy.ChainOfThought(DesignGenerator)

    def _create_domain_teacher(self) -> Any:
        """Create teacher agent (not used for Design)."""
        return None

    async def _evaluate_domain(self, output: Any, gold_standard: str, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate design quality."""

        score = 0.0
        issues = []

        # Check for key components
        has_wireframes = "wireframe" in output.lower() or "mermaid" in output.lower() or "layout" in output.lower()
        has_components = "component" in output.lower() or "element" in output.lower()
        has_hierarchy = "hierarchy" in output.lower() or "structure" in output.lower() or "navigation" in output.lower()
        has_accessibility = "accessibility" in output.lower() or "a11y" in output.lower() or "aria" in output.lower()
        has_responsive = "responsive" in output.lower() or "mobile" in output.lower() or "breakpoint" in output.lower()

        # Scoring
        if has_wireframes:
            score += 0.25
        else:
            issues.append("Missing wireframes/layout diagrams")

        if has_components:
            score += 0.2
        else:
            issues.append("Missing component structure")

        if has_hierarchy:
            score += 0.2
        else:
            issues.append("Missing visual hierarchy/navigation")

        if has_accessibility:
            score += 0.2
        else:
            issues.append("Missing accessibility considerations")

        if has_responsive:
            score += 0.15
        else:
            issues.append("Missing responsive design approach")

        # Length check
        if len(output) < 700:
            score *= 0.8
            issues.append("Design spec too brief (< 700 chars)")

        status = "EXCELLENT" if score >= 0.9 else "GOOD" if score >= 0.7 else "NEEDS_IMPROVEMENT" if score >= 0.5 else "POOR"

        suggestions = ""
        if score < 0.9:
            suggestions = "Enhance: " + ", ".join(issues[:3]) if issues else "Add more design detail"

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

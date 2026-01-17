"""
Frontend Developer Expert Agent

Evaluates frontend architecture for:
- Component architecture (React)
- State management approach
- API integration patterns
- Performance and best practices
"""

import dspy
from typing import Dict, Any, List, Optional
from .base_expert import BaseExpert


class FrontendArchitectureGenerator(dspy.Signature):
    """Generate frontend architecture with React components and state management."""
    design: str = dspy.InputField(desc="UI/UX design from designer")
    previous_feedback: str = dspy.InputField(desc="Feedback from previous iterations")
    architecture: str = dspy.OutputField(desc="Frontend architecture with React components, state management, API integration")


class FrontendExpertAgent(BaseExpert):
    """Expert in frontend development and React architecture."""

    @property
    def domain(self) -> str:
        return "frontend_development"

    @property
    def description(self) -> str:
        return "Expert in React architecture, state management, and frontend best practices"

    def _create_domain_agent(self, improvements: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Create Frontend Developer agent."""
        return dspy.ChainOfThought(FrontendArchitectureGenerator)

    def _create_domain_teacher(self) -> Any:
        """Create teacher agent (not used for Frontend)."""
        return None

    async def _evaluate_domain(self, output: Any, gold_standard: str, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate frontend architecture quality."""

        score = 0.0
        issues = []

        # Check for key components
        has_components = "component" in output.lower() or "react" in output.lower()
        has_state_mgmt = "state" in output.lower() or "redux" in output.lower() or "context" in output.lower() or "zustand" in output.lower()
        has_hooks = "hook" in output.lower() or "usestate" in output.lower() or "useeffect" in output.lower()
        has_api = "api" in output.lower() or "fetch" in output.lower() or "axios" in output.lower()
        has_props = "props" in output.lower() or "interface" in output.lower() or "type" in output.lower()

        # Scoring
        if has_components:
            score += 0.25
        else:
            issues.append("Missing React component structure")

        if has_state_mgmt:
            score += 0.2
        else:
            issues.append("Missing state management approach")

        if has_hooks:
            score += 0.2
        else:
            issues.append("Missing React hooks usage")

        if has_api:
            score += 0.2
        else:
            issues.append("Missing API integration patterns")

        if has_props:
            score += 0.15
        else:
            issues.append("Missing TypeScript types/interfaces")

        # Length check
        if len(output) < 800:
            score *= 0.8
            issues.append("Architecture spec too brief (< 800 chars)")

        status = "EXCELLENT" if score >= 0.9 else "GOOD" if score >= 0.7 else "NEEDS_IMPROVEMENT" if score >= 0.5 else "POOR"

        suggestions = ""
        if score < 0.9:
            suggestions = "Add: " + ", ".join(issues[:3]) if issues else "Expand architecture details"

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

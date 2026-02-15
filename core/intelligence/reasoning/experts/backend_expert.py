"""
Backend Developer Expert Agent

Evaluates backend architecture for:
- API design (REST/GraphQL)
- Data models and database schema
- Authentication and authorization
- Performance and scalability
"""

from typing import Any, Dict, List, Optional

import dspy

from .base_expert import BaseExpert


class BackendArchitectureGenerator(dspy.Signature):
    """Generate backend architecture with API endpoints and data models."""

    frontend_architecture: str = dspy.InputField(
        desc="Frontend architecture from frontend developer"
    )
    previous_feedback: str = dspy.InputField(desc="Feedback from previous iterations")
    architecture: str = dspy.OutputField(
        desc="Backend architecture with API endpoints, data models, auth, database schema"
    )


class BackendExpertAgent(BaseExpert):
    """Expert in backend development and API design."""

    @property
    def domain(self) -> str:
        return "backend_development"

    @property
    def description(self) -> str:
        return "Expert in API design, database schema, and backend architecture"

    def _create_domain_agent(self, improvements: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Create Backend Developer agent."""
        return dspy.ChainOfThought(BackendArchitectureGenerator)

    def _create_domain_teacher(self) -> Any:
        """Create teacher agent (not used for Backend)."""
        return None

    async def _evaluate_domain(
        self, output: Any, gold_standard: str, task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate backend architecture quality."""

        score = 0.0
        issues = []

        # Check for key components
        has_api = (
            "api" in output.lower()
            or "endpoint" in output.lower()
            or "rest" in output.lower()
            or "graphql" in output.lower()
        )
        has_data_models = (
            "model" in output.lower() or "schema" in output.lower() or "entity" in output.lower()
        )
        has_database = (
            "database" in output.lower()
            or "postgres" in output.lower()
            or "mongodb" in output.lower()
            or "mysql" in output.lower()
        )
        has_auth = (
            "auth" in output.lower()
            or "jwt" in output.lower()
            or "session" in output.lower()
            or "oauth" in output.lower()
        )
        has_validation = (
            "validation" in output.lower()
            or "middleware" in output.lower()
            or "error" in output.lower()
        )

        # Scoring
        if has_api:
            score += 0.25
        else:
            issues.append("Missing API endpoint definitions")

        if has_data_models:
            score += 0.25
        else:
            issues.append("Missing data models/schema")

        if has_database:
            score += 0.2
        else:
            issues.append("Missing database design")

        if has_auth:
            score += 0.15
        else:
            issues.append("Missing authentication/authorization")

        if has_validation:
            score += 0.15
        else:
            issues.append("Missing validation/error handling")

        # Length check
        if len(output) < 800:
            score *= 0.8
            issues.append("Architecture spec too brief (< 800 chars)")

        status = (
            "EXCELLENT"
            if score >= 0.9
            else "GOOD" if score >= 0.7 else "NEEDS_IMPROVEMENT" if score >= 0.5 else "POOR"
        )

        suggestions = ""
        if score < 0.9:
            suggestions = (
                "Include: " + ", ".join(issues[:3]) if issues else "Expand backend architecture"
            )

        return {"score": score, "status": status, "issues": issues, "suggestions": suggestions}

    def _get_default_training_cases(self) -> List[Dict[str, Any]]:
        return []

    def _get_default_validation_cases(self) -> List[Dict[str, Any]]:
        return []

#!/usr/bin/env python3
"""
Smart Swarm Registry
====================

Automatically selects appropriate swarms based on task type.
Maps stage types to actual Jotty swarms with best practices built-in.
"""

from typing import Dict, List, Any, Optional, Type
from dataclasses import dataclass
from enum import Enum


class StageType(Enum):
    """Common stage types in workflows."""
    # Analysis & Research
    REQUIREMENTS_ANALYSIS = "requirements_analysis"
    MARKET_RESEARCH = "market_research"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    RESEARCH = "research"

    # Design
    ARCHITECTURE_DESIGN = "architecture_design"
    SYSTEM_DESIGN = "system_design"
    DATABASE_DESIGN = "database_design"

    # Development
    CODE_GENERATION = "code_generation"
    TEST_GENERATION = "test_generation"
    DOCUMENTATION = "documentation"

    # Quality & Review
    CODE_REVIEW = "code_review"
    SECURITY_AUDIT = "security_audit"
    TESTING = "testing"

    # Deployment
    DEPLOYMENT = "deployment"
    INFRASTRUCTURE = "infrastructure"

    # Validation
    VALIDATION = "validation"
    INTEGRATION_TEST = "integration_test"


@dataclass
class SwarmConfig:
    """Configuration for a swarm."""
    swarm_type: str
    description: str
    default_prompts: List[str]
    merge_strategy: str = "BEST_OF_N"
    best_practices: List[str] = None

    def __post_init__(self) -> None:
        if self.best_practices is None:
            self.best_practices = []


class SmartSwarmRegistry:
    """
    Smart registry that maps stage types to appropriate swarms.

    Usage:
        registry = SmartSwarmRegistry()
        swarms = registry.get_swarms_for_stage(
            StageType.CODE_GENERATION,
            context={"framework": "fastapi", "features": ["auth", "crud"]}
        )
    """

    def __init__(self) -> None:
        self._registry: Dict[StageType, SwarmConfig] = {}
        self._initialize_default_mappings()

    def _initialize_default_mappings(self) -> Any:
        """Initialize default stage type → swarm mappings."""

        # Requirements & Analysis
        self._registry[StageType.REQUIREMENTS_ANALYSIS] = SwarmConfig(
            swarm_type="requirements",
            description="Analyze and document requirements",
            default_prompts=[
                "Analyze functional and non-functional requirements",
                "Define user stories and acceptance criteria",
                "Identify constraints and dependencies"
            ],
            merge_strategy="CONCATENATE",
            best_practices=[
                "Be specific and measurable",
                "Include edge cases",
                "Consider security and performance"
            ]
        )

        self._registry[StageType.MARKET_RESEARCH] = SwarmConfig(
            swarm_type="research",
            description="Conduct market research",
            default_prompts=[
                "Research market size and trends",
                "Analyze competitive landscape",
                "Identify opportunities and threats"
            ],
            merge_strategy="CONCATENATE"
        )

        # Design
        self._registry[StageType.ARCHITECTURE_DESIGN] = SwarmConfig(
            swarm_type="architecture",
            description="Design system architecture",
            default_prompts=[
                "Design scalable system architecture",
                "Select appropriate tech stack",
                "Define API contracts and data models",
                "Consider deployment and infrastructure"
            ],
            merge_strategy="BEST_OF_N",
            best_practices=[
                "Use proven patterns",
                "Design for scalability",
                "Consider security from the start"
            ]
        )

        # Development
        self._registry[StageType.CODE_GENERATION] = SwarmConfig(
            swarm_type="coding",
            description="Generate production-ready code",
            default_prompts=[
                "Generate production-ready, well-structured code",
                "Include comprehensive error handling",
                "Follow language/framework best practices",
                "Make code runnable and complete",
                "Include necessary imports and dependencies",
                "Add inline comments for complex logic"
            ],
            merge_strategy="BEST_OF_N",
            best_practices=[
                "Production-ready (not prototype)",
                "Complete (no placeholders or TODOs)",
                "Runnable (all imports included)",
                "Error handling included",
                "Security considerations applied",
                "Output only code in ```language blocks"
            ]
        )

        self._registry[StageType.TEST_GENERATION] = SwarmConfig(
            swarm_type="testing",
            description="Generate comprehensive test suite",
            default_prompts=[
                "Generate complete test suite with fixtures",
                "Include unit tests for all functions",
                "Add integration tests for workflows",
                "Test edge cases and error conditions",
                "Make tests runnable and independent"
            ],
            merge_strategy="BEST_OF_N",
            best_practices=[
                "Use proper test framework (pytest, unittest)",
                "Tests are independent and isolated",
                "Good coverage (>80%)",
                "Test both success and failure paths"
            ]
        )

        self._registry[StageType.DOCUMENTATION] = SwarmConfig(
            swarm_type="documentation",
            description="Generate comprehensive documentation",
            default_prompts=[
                "Create clear, comprehensive documentation",
                "Include setup and installation guide",
                "Document API endpoints with examples",
                "Add troubleshooting section",
                "Include configuration options"
            ],
            merge_strategy="BEST_OF_N"
        )

        # Quality & Review
        self._registry[StageType.CODE_REVIEW] = SwarmConfig(
            swarm_type="review",
            description="Review code for quality and security",
            default_prompts=[
                "Review code for security vulnerabilities",
                "Check for performance issues",
                "Verify best practices are followed",
                "Identify bugs and edge cases",
                "Provide actionable feedback"
            ],
            merge_strategy="VOTING",
            best_practices=[
                "Focus on critical issues first",
                "Provide specific examples",
                "Suggest concrete improvements"
            ]
        )

        self._registry[StageType.SECURITY_AUDIT] = SwarmConfig(
            swarm_type="security",
            description="Audit security vulnerabilities",
            default_prompts=[
                "Audit for OWASP Top 10 vulnerabilities",
                "Check authentication and authorization",
                "Review input validation and sanitization",
                "Identify data exposure risks",
                "Provide risk rating and remediation"
            ],
            merge_strategy="VOTING"
        )

        # Deployment
        self._registry[StageType.DEPLOYMENT] = SwarmConfig(
            swarm_type="deployment",
            description="Generate deployment scripts and configs",
            default_prompts=[
                "Generate production-ready deployment scripts",
                "Include Dockerfile and docker-compose",
                "Add CI/CD configuration",
                "Include environment configuration",
                "Make scripts runnable and complete"
            ],
            merge_strategy="BEST_OF_N"
        )

        # Validation
        self._registry[StageType.VALIDATION] = SwarmConfig(
            swarm_type="validation",
            description="Validate completeness and readiness",
            default_prompts=[
                "Validate all requirements are met",
                "Check completeness of deliverables",
                "Assess production readiness",
                "Identify missing or incomplete items",
                "Provide go/no-go recommendation"
            ],
            merge_strategy="VOTING"
        )

    def get_swarms_for_stage(
        self,
        stage_type: StageType,
        context: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        Get appropriate swarms for a stage type.

        Args:
            stage_type: Type of stage
            context: Additional context (framework, features, etc.)

        Returns:
            List of (name, prompt) tuples for SwarmAdapter.quick_swarms()
        """
        if stage_type not in self._registry:
            raise ValueError(f"Unknown stage type: {stage_type}")

        config = self._registry[stage_type]
        context = context or {}

        # Build enriched prompts with context and best practices
        swarms = []

        for i, base_prompt in enumerate(config.default_prompts):
            # Add context if provided
            context_str = self._build_context_string(context)

            # Build full prompt
            full_prompt = base_prompt

            if context_str:
                full_prompt = f"{base_prompt}\n\nContext:\n{context_str}"

            # Add best practices
            if config.best_practices:
                practices = "\n".join(f"- {p}" for p in config.best_practices)
                full_prompt = f"{full_prompt}\n\nBest Practices:\n{practices}"

            # Create swarm name
            swarm_name = f"{config.swarm_type.title()} {i+1}" if len(config.default_prompts) > 1 else config.swarm_type.title()

            swarms.append((swarm_name, full_prompt))

        return swarms

    def _build_context_string(self, context: Dict[str, Any]) -> str:
        """Build context string from dictionary."""
        if not context:
            return ""

        parts = []
        for key, value in context.items():
            if isinstance(value, list):
                parts.append(f"{key}: {', '.join(str(v) for v in value)}")
            else:
                parts.append(f"{key}: {value}")

        return "\n".join(parts)

    def get_merge_strategy(self, stage_type: StageType) -> str:
        """Get recommended merge strategy for stage type."""
        if stage_type not in self._registry:
            return "BEST_OF_N"
        return self._registry[stage_type].merge_strategy

    def register_custom_stage(self, stage_type: str, config: SwarmConfig) -> Any:
        """Register a custom stage type."""
        # Convert string to enum if needed
        custom_type = StageType(stage_type) if isinstance(stage_type, str) else stage_type
        self._registry[custom_type] = config

    def list_stage_types(self) -> List[StageType]:
        """List all available stage types."""
        return list(self._registry.keys())


# Singleton instance
_registry = None


def get_smart_registry() -> SmartSwarmRegistry:
    """Get singleton smart swarm registry."""
    global _registry
    if _registry is None:
        _registry = SmartSwarmRegistry()
    return _registry

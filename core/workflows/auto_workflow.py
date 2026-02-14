#!/usr/bin/env python3
"""
AutoWorkflow - Intent-Based Workflow Execution
===============================================

Automatically decomposes high-level goals into stages and executes them
using the smart swarm registry.

User provides intent, system figures out the rest.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from ..orchestration import (
    MultiStagePipeline,
    SwarmAdapter,
    MergeStrategy,
    PipelineResult,
    extract_code_from_markdown
)
from .smart_swarm_registry import StageType, get_smart_registry


@dataclass
class WorkflowIntent:
    """High-level workflow intent."""
    goal: str
    project_type: Optional[str] = None  # "rest_api", "trading_strategy", "ml_model", etc.
    deliverables: Optional[List[str]] = None  # ["code", "tests", "docs", "deployment"]
    tech_stack: Optional[List[str]] = None  # ["fastapi", "postgresql", "redis"]
    features: Optional[List[str]] = None  # ["authentication", "crud", "caching"]
    requirements: Optional[List[str]] = None  # Custom requirements


class AutoWorkflow:
    """
    Automatically decomposes goals into stages and executes them.

    Usage:
        # Simplest - just provide goal
        result = await AutoWorkflow.execute(
            goal="Build todo API with authentication"
        )

        # With guidance
        result = await AutoWorkflow.execute(
            goal="Build trading strategy for NVDA",
            project_type="trading_strategy",
            deliverables=["code", "backtest", "docs"]
        )

        # With custom config
        workflow = AutoWorkflow(intent)
        workflow.customize_stage("code", max_tokens=3000)
        result = await workflow.run()
    """

    def __init__(self, intent: WorkflowIntent):
        """
        Initialize auto workflow.

        Args:
            intent: High-level workflow intent
        """
        self.intent = intent
        self.registry = get_smart_registry()
        self.pipeline = None
        self.stage_configs = {}

    @classmethod
    async def execute(
        cls,
        goal: str,
        project_type: Optional[str] = None,
        deliverables: Optional[List[str]] = None,
        tech_stack: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        requirements: Optional[List[str]] = None,
        verbose: bool = True
    ) -> PipelineResult:
        """
        Execute workflow from high-level intent (simplest API).

        Args:
            goal: What you want to build/achieve
            project_type: Type of project (optional)
            deliverables: What to generate (optional)
            tech_stack: Technologies to use (optional)
            features: Features to include (optional)
            requirements: Custom requirements (optional)
            verbose: Print progress

        Returns:
            PipelineResult with all outputs
        """
        intent = WorkflowIntent(
            goal=goal,
            project_type=project_type,
            deliverables=deliverables,
            tech_stack=tech_stack,
            features=features,
            requirements=requirements
        )

        workflow = cls(intent)
        workflow.build_pipeline()
        return await workflow.run(verbose=verbose)

    def build_pipeline(self):
        """Build pipeline from intent by decomposing into stages."""
        # Determine deliverables
        deliverables = self.intent.deliverables or self._infer_deliverables()

        # Create pipeline
        self.pipeline = MultiStagePipeline(task=self.intent.goal)

        # Add stages based on deliverables
        self._add_stages_for_deliverables(deliverables)

    def _infer_deliverables(self) -> List[str]:
        """Infer deliverables from goal and project type."""
        goal_lower = self.intent.goal.lower()

        # Default deliverables for common project types
        if self.intent.project_type == "rest_api" or "api" in goal_lower:
            return ["requirements", "architecture", "code", "tests", "docs", "deployment"]

        elif self.intent.project_type == "trading_strategy" or "trading" in goal_lower or "strategy" in goal_lower:
            return ["research", "strategy", "code", "validation"]

        elif "research" in goal_lower or "analyze" in goal_lower:
            return ["research", "analysis", "synthesis"]

        # Default: full software development
        return ["requirements", "code", "tests", "docs"]

    def _add_stages_for_deliverables(self, deliverables: List[str]):
        """Add pipeline stages based on deliverables."""

        # Map deliverable names to stage types
        deliverable_mapping = {
            "requirements": StageType.REQUIREMENTS_ANALYSIS,
            "research": StageType.MARKET_RESEARCH,
            "architecture": StageType.ARCHITECTURE_DESIGN,
            "design": StageType.SYSTEM_DESIGN,
            "code": StageType.CODE_GENERATION,
            "tests": StageType.TEST_GENERATION,
            "docs": StageType.DOCUMENTATION,
            "documentation": StageType.DOCUMENTATION,
            "review": StageType.CODE_REVIEW,
            "security": StageType.SECURITY_AUDIT,
            "deployment": StageType.DEPLOYMENT,
            "validation": StageType.VALIDATION,
            "strategy": StageType.ARCHITECTURE_DESIGN,  # For trading strategies
            "analysis": StageType.MARKET_RESEARCH,
            "synthesis": StageType.VALIDATION,
        }

        # Build context dict for all stages
        context = self._build_stage_context()

        # Track previous stages for context chaining
        previous_stages = []

        for deliverable in deliverables:
            if deliverable not in deliverable_mapping:
                print(f"⚠️  Unknown deliverable: {deliverable}, skipping")
                continue

            stage_type = deliverable_mapping[deliverable]

            # Get swarms from registry
            swarms_config = self.registry.get_swarms_for_stage(stage_type, context)
            merge_strategy_str = self.registry.get_merge_strategy(stage_type)

            # Convert string to MergeStrategy enum
            merge_strategy = getattr(MergeStrategy, merge_strategy_str, MergeStrategy.BEST_OF_N)

            # Create swarms
            swarms = SwarmAdapter.quick_swarms(
                swarms_config,
                model=self.stage_configs.get(deliverable, {}).get("model", "claude-3-5-haiku-20241022"),
                max_tokens=self.stage_configs.get(deliverable, {}).get("max_tokens", 800)
            )

            # Determine context sources (use all previous stages)
            context_from = previous_stages.copy() if previous_stages else None

            # Add stage
            self.pipeline.add_stage(
                name=deliverable,
                swarms=swarms,
                merge_strategy=merge_strategy,
                context_from=context_from,
                max_context_chars=1500
            )

            previous_stages.append(deliverable)

    def _build_stage_context(self) -> Dict[str, Any]:
        """Build context dict for stage prompts."""
        context = {}

        if self.intent.tech_stack:
            context["tech_stack"] = self.intent.tech_stack

        if self.intent.features:
            context["features"] = self.intent.features

        if self.intent.requirements:
            context["requirements"] = self.intent.requirements

        if self.intent.project_type:
            context["project_type"] = self.intent.project_type

        return context

    def customize_stage(
        self,
        stage_name: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Customize a specific stage.

        Args:
            stage_name: Name of stage (e.g., "code", "tests")
            model: Model to use
            max_tokens: Max tokens for this stage
        """
        if stage_name not in self.stage_configs:
            self.stage_configs[stage_name] = {}

        if model:
            self.stage_configs[stage_name]["model"] = model
        if max_tokens:
            self.stage_configs[stage_name]["max_tokens"] = max_tokens

    async def run(self, verbose: bool = True) -> PipelineResult:
        """Execute the workflow pipeline."""
        if self.pipeline is None:
            self.build_pipeline()

        return await self.pipeline.execute(auto_trace=True, verbose=verbose)


# Convenience functions
async def build(
    goal: str,
    **kwargs
) -> PipelineResult:
    """
    Simplest API - just build something from a goal.

    Usage:
        result = await build("Todo API with authentication")
        result = await build("Trading strategy for NVDA", deliverables=["code", "backtest"])
    """
    return await AutoWorkflow.execute(goal, **kwargs)


async def research(
    topic: str,
    depth: str = "comprehensive",
    **kwargs
) -> PipelineResult:
    """
    Research a topic automatically.

    Usage:
        result = await research("AI safety challenges in 2026")
    """
    return await AutoWorkflow.execute(
        goal=f"Research {topic} ({depth} analysis)",
        project_type="research",
        deliverables=["research", "analysis", "synthesis"],
        **kwargs
    )


async def develop(
    description: str,
    tech_stack: Optional[List[str]] = None,
    **kwargs
) -> PipelineResult:
    """
    Develop software from description.

    Usage:
        result = await develop(
            "REST API for todo management",
            tech_stack=["fastapi", "postgresql"]
        )
    """
    return await AutoWorkflow.execute(
        goal=f"Build {description}",
        project_type="rest_api",
        deliverables=["requirements", "architecture", "code", "tests", "docs", "deployment"],
        tech_stack=tech_stack,
        **kwargs
    )

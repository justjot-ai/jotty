#!/usr/bin/env python3
"""
AutoWorkflow - Intent-Based Workflow Execution
===============================================

Automatically decomposes high-level goals into stages and executes them
using the smart swarm registry.

User provides intent, system figures out the rest.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..orchestration import MergeStrategy, MultiStagePipeline, PipelineResult, SwarmAdapter
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

    def __init__(self, intent: WorkflowIntent) -> None:
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
    def from_intent(
        cls,
        goal: str,
        project_type: Optional[str] = None,
        deliverables: Optional[List[str]] = None,
        tech_stack: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        requirements: Optional[List[str]] = None,
    ) -> "AutoWorkflow":
        """
        Create workflow from intent (without executing).
        Allows inspection and customization before execution.

        Args:
            goal: What you want to build/achieve
            project_type: Type of project (optional)
            deliverables: What to generate (optional)
            tech_stack: Technologies to use (optional)
            features: Features to include (optional)
            requirements: Custom requirements (optional)

        Returns:
            AutoWorkflow instance (not yet executed)
        """
        intent = WorkflowIntent(
            goal=goal,
            project_type=project_type,
            deliverables=deliverables,
            tech_stack=tech_stack,
            features=features,
            requirements=requirements,
        )
        return cls(intent)

    @classmethod
    async def execute(
        cls,
        goal: str,
        project_type: Optional[str] = None,
        deliverables: Optional[List[str]] = None,
        tech_stack: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        requirements: Optional[List[str]] = None,
        verbose: bool = True,
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
            requirements=requirements,
        )

        workflow = cls(intent)
        workflow.build_pipeline()
        return await workflow.run(verbose=verbose)

    def build_pipeline(self) -> Any:
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

        elif (
            self.intent.project_type == "trading_strategy"
            or "trading" in goal_lower
            or "strategy" in goal_lower
        ):
            return ["research", "strategy", "code", "validation"]

        elif "research" in goal_lower or "analyze" in goal_lower:
            return ["research", "analysis", "synthesis"]

        # Default: full software development
        return ["requirements", "code", "tests", "docs"]

    def _add_stages_for_deliverables(self, deliverables: List[str]) -> Any:
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
            stage_config = self.stage_configs.get(deliverable, {})

            # Check if stage is completely replaced
            if stage_config.get("replace"):
                # Use custom swarms
                swarms = stage_config["custom_swarms"]
                merge_strategy = stage_config.get("merge_strategy", MergeStrategy.BEST_OF_N)
                context_from = stage_config.get(
                    "context_from", previous_stages.copy() if previous_stages else None
                )

            else:
                # Auto-generate with SmartRegistry (possibly customized)
                if deliverable not in deliverable_mapping:
                    print(f"⚠️  Unknown deliverable: {deliverable}, skipping")
                    continue

                stage_type = deliverable_mapping[deliverable]

                # Get swarms from registry
                swarms_config = self.registry.get_swarms_for_stage(stage_type, context)

                # Apply customizations (additional prompts, context, etc.)
                if stage_config.get("custom_prompts"):
                    for custom_prompt in stage_config["custom_prompts"]:
                        swarms_config.append((f"{deliverable.title()} Custom", custom_prompt))

                if stage_config.get("additional_context"):
                    # Add additional context to each swarm prompt
                    swarms_config = [
                        (
                            name,
                            f"{prompt}\n\nAdditional Context:\n{stage_config['additional_context']}",
                        )
                        for name, prompt in swarms_config
                    ]

                # Get merge strategy (customized or default)
                if stage_config.get("merge_strategy"):
                    merge_strategy = stage_config["merge_strategy"]
                else:
                    merge_strategy_str = self.registry.get_merge_strategy(stage_type)
                    merge_strategy = getattr(
                        MergeStrategy, merge_strategy_str, MergeStrategy.BEST_OF_N
                    )

                # Create swarms
                swarms = SwarmAdapter.quick_swarms(
                    swarms_config,
                    model=stage_config.get("model", "claude-3-5-haiku-20241022"),
                    max_tokens=stage_config.get("max_tokens", 800),
                )

                # Determine context sources (use all previous stages)
                context_from = previous_stages.copy() if previous_stages else None

            # Add stage
            self.pipeline.add_stage(
                name=deliverable,
                swarms=swarms,
                merge_strategy=merge_strategy,
                context_from=context_from,
                max_context_chars=1500,
            )

            previous_stages.append(deliverable)

        # Add any custom stages
        for stage_name, config in self.stage_configs.items():
            if config.get("custom_stage"):
                # This is a completely new stage to insert
                swarms = config["swarms"]
                merge_strategy = config.get("merge_strategy", MergeStrategy.BEST_OF_N)
                context_from = config.get("context_from")

                self.pipeline.add_stage(
                    name=stage_name,
                    swarms=swarms,
                    merge_strategy=merge_strategy,
                    context_from=context_from,
                    max_context_chars=1500,
                )

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
        max_tokens: Optional[int] = None,
        merge_strategy: Optional[MergeStrategy] = None,
        additional_context: Optional[str] = None,
        custom_prompts: Optional[List[str]] = None,
    ) -> Any:
        """
        Customize a specific stage (keeps auto-generation but tweaks it).

        Args:
            stage_name: Name of stage (e.g., "code", "tests")
            model: Model to use
            max_tokens: Max tokens for this stage
            merge_strategy: Override merge strategy
            additional_context: Additional context to inject
            custom_prompts: Additional prompts to add to SmartRegistry prompts
        """
        if stage_name not in self.stage_configs:
            self.stage_configs[stage_name] = {}

        if model:
            self.stage_configs[stage_name]["model"] = model
        if max_tokens:
            self.stage_configs[stage_name]["max_tokens"] = max_tokens
        if merge_strategy:
            self.stage_configs[stage_name]["merge_strategy"] = merge_strategy
        if additional_context:
            self.stage_configs[stage_name]["additional_context"] = additional_context
        if custom_prompts:
            self.stage_configs[stage_name]["custom_prompts"] = custom_prompts

    def replace_stage(
        self,
        stage_name: str,
        swarms: List[Any],
        merge_strategy: Optional[MergeStrategy] = None,
        context_from: Optional[List[str]] = None,
    ) -> Any:
        """
        Completely replace a stage with custom swarms (full control).

        Args:
            stage_name: Name of stage to replace
            swarms: Custom swarms to use
            merge_strategy: Merge strategy
            context_from: Which stages to use as context
        """
        if stage_name not in self.stage_configs:
            self.stage_configs[stage_name] = {}

        self.stage_configs[stage_name]["replace"] = True
        self.stage_configs[stage_name]["custom_swarms"] = swarms
        if merge_strategy:
            self.stage_configs[stage_name]["merge_strategy"] = merge_strategy
        if context_from:
            self.stage_configs[stage_name]["context_from"] = context_from

    def add_custom_stage(
        self,
        stage_name: str,
        swarms: List[Any],
        position: Optional[int] = None,
        merge_strategy: MergeStrategy = MergeStrategy.BEST_OF_N,
        context_from: Optional[List[str]] = None,
    ) -> Any:
        """
        Add a completely custom stage (insert new stage).

        Args:
            stage_name: Name for the new stage
            swarms: Swarms to execute
            position: Where to insert (None = append at end)
            merge_strategy: How to merge results
            context_from: Which stages to use as context
        """
        self.stage_configs[stage_name] = {
            "custom_stage": True,
            "swarms": swarms,
            "position": position,
            "merge_strategy": merge_strategy,
            "context_from": context_from,
        }

    def inspect_pipeline(self) -> Dict[str, Any]:
        """
        Inspect what the auto-generated pipeline will do.

        Returns:
            Dict with pipeline structure before execution
        """
        deliverables = self.intent.deliverables or self._infer_deliverables()

        inspection = {"goal": self.intent.goal, "deliverables": deliverables, "stages": []}

        for deliverable in deliverables:
            stage_info = {
                "name": deliverable,
                "stage_type": deliverable,
                "customized": deliverable in self.stage_configs,
                "replaced": self.stage_configs.get(deliverable, {}).get("replace", False),
            }
            inspection["stages"].append(stage_info)

        return inspection

    def show_pipeline(self, verbose: bool = True) -> Any:
        """
        Print pipeline structure (useful before executing).
        """
        inspection = self.inspect_pipeline()

        if verbose:
            print("\n" + "=" * 70)
            print("PIPELINE INSPECTION")
            print("=" * 70)
            print(f"\n🎯 Goal: {inspection['goal']}")
            print(f"📊 Stages: {len(inspection['stages'])}")
            print()

            for i, stage in enumerate(inspection["stages"], 1):
                status = ""
                if stage["replaced"]:
                    status = "🔧 REPLACED (custom swarms)"
                elif stage["customized"]:
                    status = "⚙️  CUSTOMIZED (tweaked)"
                else:
                    status = "🤖 AUTO (SmartRegistry)"

                print(f"{i}. {stage['name']:<20} {status}")

            print("\n" + "=" * 70 + "\n")

    async def run(self, verbose: bool = True) -> PipelineResult:
        """Execute the workflow pipeline."""
        if self.pipeline is None:
            self.build_pipeline()

        return await self.pipeline.execute(auto_trace=True, verbose=verbose)


# Convenience functions
async def build(goal: str, **kwargs: Any) -> PipelineResult:
    """
    Simplest API - just build something from a goal.

    Usage:
        result = await build("Todo API with authentication")
        result = await build("Trading strategy for NVDA", deliverables=["code", "backtest"])
    """
    return await AutoWorkflow.execute(goal, **kwargs)


async def research(topic: str, depth: str = "comprehensive", **kwargs: Any) -> PipelineResult:
    """
    Research a topic automatically.

    Usage:
        result = await research("AI safety challenges in 2026")
    """
    return await AutoWorkflow.execute(
        goal=f"Research {topic} ({depth} analysis)",
        project_type="research",
        deliverables=["research", "analysis", "synthesis"],
        **kwargs,
    )


async def develop(
    description: str, tech_stack: Optional[List[str]] = None, **kwargs: Any
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
        **kwargs,
    )

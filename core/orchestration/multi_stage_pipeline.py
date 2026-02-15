#!/usr/bin/env python3
"""
Multi-Stage Pipeline Orchestrator
==================================

High-level utility for chaining multiple swarm executions with automatic
context passing and result aggregation.

Reduces boilerplate for complex multi-stage workflows.
"""

from __future__ import annotations  # Enable forward references
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import time
from .multi_swarm_coordinator import MultiSwarmCoordinator, SwarmResult, MergeStrategy
from ..observability import get_distributed_tracer


@dataclass
class StageConfig:
    """Configuration for a single pipeline stage."""
    name: str
    swarms: List[Any]
    merge_strategy: MergeStrategy = MergeStrategy.BEST_OF_N
    context_from: List[str] = field(default_factory=list)
    context_template: Optional[str] = None
    max_context_chars: int = 1500

    def get_context_prompt(self, previous_results: Dict[str, StageResult]) -> str:
        """Build context prompt from previous stage results."""
        if not self.context_from:
            return ""

        context_parts = []
        for stage_name in self.context_from:
            if stage_name in previous_results:
                result = previous_results[stage_name]
                output = result.result.output[:self.max_context_chars]
                context_parts.append(f"[{stage_name.upper()}]\n{output}")

        context = "\n\n".join(context_parts)

        if self.context_template:
            return self.context_template.format(context=context)
        else:
            return f"Context from previous stages:\n\n{context}\n\n"


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage_name: str
    result: SwarmResult
    execution_time: float
    cost: float
    trace_id: Optional[str] = None


@dataclass
class PipelineResult:
    """Result from complete pipeline execution."""
    task: str
    stages: List[StageResult]
    total_cost: float
    total_time: float
    final_result: StageResult

    def get_stage(self, name: str) -> Optional[StageResult]:
        """Get result from specific stage by name."""
        for stage in self.stages:
            if stage.stage_name == name:
                return stage
        return None

    def print_summary(self, verbose: bool = True) -> Any:
        """Print formatted pipeline summary."""
        print("\n" + "="*80)
        print("MULTI-STAGE PIPELINE RESULTS")
        print("="*80 + "\n")

        print(f"📋 Task: {self.task[:80]}...")
        print(f"📊 Total Stages: {len(self.stages)}")
        print(f"💰 Total Cost: ${self.total_cost:.6f}")
        print(f"⚡ Total Time: {self.total_time:.2f}s")
        print()

        print("🔄 Pipeline Execution:")
        for i, stage in enumerate(self.stages, 1):
            print(f"\n{i}. {stage.stage_name.upper()}")
            print(f"   Success: {stage.result.success}")
            print(f"   Confidence: {stage.result.confidence:.2f}")
            print(f"   Cost: ${stage.cost:.6f}")
            print(f"   Time: {stage.execution_time:.2f}s")

            if verbose:
                preview = stage.result.output[:150] if stage.result.output else "No output"
                print(f"   Output: {preview}...")

        print(f"\n🎯 Final Result: {self.final_result.stage_name}")
        print(f"   Confidence: {self.final_result.result.confidence:.2f}")
        print()


class MultiStagePipeline:
    """
    High-level multi-stage pipeline orchestrator.

    Usage:
        pipeline = MultiStagePipeline(task="Build trading strategy")

        # Add stages
        pipeline.add_stage("research", swarms=[...],
                          merge_strategy=MergeStrategy.CONCATENATE)

        pipeline.add_stage("strategy", swarms=[...],
                          context_from=["research"],
                          merge_strategy=MergeStrategy.BEST_OF_N)

        # Execute
        result = await pipeline.execute(auto_trace=True)
        result.print_summary()
    """

    def __init__(self, task: str, coordinator: Optional[MultiSwarmCoordinator] = None) -> None:
        """
        Initialize pipeline.

        Args:
            task: Overall task description
            coordinator: Optional coordinator instance
        """
        self.task = task
        self.coordinator = coordinator or MultiSwarmCoordinator()
        self.stages: List[StageConfig] = []

    def add_stage(
        self,
        name: str,
        swarms: List[Any],
        merge_strategy: MergeStrategy = MergeStrategy.BEST_OF_N,
        context_from: Optional[List[str]] = None,
        context_template: Optional[str] = None,
        max_context_chars: int = 1500
    ) -> "MultiStagePipeline":
        """
        Add a stage to the pipeline.

        Args:
            name: Stage name (used for context passing)
            swarms: List of swarms to execute in this stage
            merge_strategy: How to merge results
            context_from: List of previous stage names to use as context
            context_template: Optional template for context prompt
            max_context_chars: Max characters of context to pass

        Returns:
            Self for chaining
        """
        self.stages.append(StageConfig(
            name=name,
            swarms=swarms,
            merge_strategy=merge_strategy,
            context_from=context_from or [],
            context_template=context_template,
            max_context_chars=max_context_chars
        ))
        return self

    async def execute(
        self,
        auto_trace: bool = True,
        verbose: bool = True
    ) -> PipelineResult:
        """
        Execute the complete pipeline.

        Args:
            auto_trace: Automatically use distributed tracing
            verbose: Print progress messages

        Returns:
            PipelineResult with all stage results
        """
        tracer = get_distributed_tracer("pipeline") if auto_trace else None

        if verbose:
            print(f"\n🚀 Starting pipeline: {self.task}")
            print(f"   Stages: {len(self.stages)}")
            print(f"   Auto-trace: {auto_trace}")
            print()

        results: List[StageResult] = []
        results_by_name: Dict[str, StageResult] = {}
        total_cost = 0.0
        start_time = time.time()

        # Main trace context
        trace_ctx = tracer.trace(f"pipeline_{self.task[:30]}") if tracer else None
        main_trace_id = trace_ctx.__enter__() if trace_ctx else None

        try:
            for i, stage in enumerate(self.stages, 1):
                if verbose:
                    print(f"▶ Stage {i}/{len(self.stages)}: {stage.name}")

                # Build task with context from previous stages
                stage_task = self.task
                if stage.context_from:
                    context_prompt = stage.get_context_prompt(results_by_name)
                    stage_task = context_prompt + "\n" + stage_task

                # Nested trace for stage
                stage_trace_ctx = tracer.trace(f"stage_{stage.name}") if tracer else None
                stage_trace_id = stage_trace_ctx.__enter__() if stage_trace_ctx else None

                try:
                    stage_start = time.time()

                    # Execute stage
                    result = await self.coordinator.execute_parallel(
                        swarms=stage.swarms,
                        task=stage_task,
                        merge_strategy=stage.merge_strategy
                    )

                    execution_time = time.time() - stage_start
                    cost = result.metadata.get('cost_usd', 0.0)
                    total_cost += cost

                    stage_result = StageResult(
                        stage_name=stage.name,
                        result=result,
                        execution_time=execution_time,
                        cost=cost,
                        trace_id=stage_trace_id
                    )

                    results.append(stage_result)
                    results_by_name[stage.name] = stage_result

                    if verbose:
                        print(f"  ✓ {stage.name}: {result.confidence:.2f} confidence, "
                              f"${cost:.6f}, {execution_time:.2f}s\n")

                finally:
                    if stage_trace_ctx:
                        stage_trace_ctx.__exit__(None, None, None)

        finally:
            if trace_ctx:
                trace_ctx.__exit__(None, None, None)

        total_time = time.time() - start_time

        return PipelineResult(
            task=self.task,
            stages=results,
            total_cost=total_cost,
            total_time=total_time,
            final_result=results[-1] if results else None
        )


# Utility function for code extraction
def extract_code_from_markdown(text: str, language: str = "python") -> Optional[str]:
    """
    Extract code from markdown code blocks.

    Args:
        text: Text containing markdown code blocks
        language: Expected language (e.g., "python", "javascript")

    Returns:
        Extracted code or None if not found
    """
    # Try with language specifier
    marker = f"```{language}"
    if marker in text:
        code_start = text.find(marker) + len(marker)
        code_end = text.find("```", code_start)
        if code_end != -1:
            return text[code_start:code_end].strip()

    # Try generic code block
    if "```" in text:
        code_start = text.find("```") + 3
        # Skip language identifier if present
        newline = text.find("\n", code_start)
        if newline != -1:
            code_start = newline + 1
        code_end = text.find("```", code_start)
        if code_end != -1:
            return text[code_start:code_end].strip()

    return None


# Facade function
def create_pipeline(task: str, **kwargs: Any) -> MultiStagePipeline:
    """
    Quick function to create a pipeline.

    Usage:
        pipeline = create_pipeline("Build trading strategy")
        pipeline.add_stage("research", swarms=[...])
        pipeline.add_stage("strategy", swarms=[...], context_from=["research"])
        result = await pipeline.execute()
    """
    return MultiStagePipeline(task, **kwargs)

"""
Jotty V3 - Main Entry Point
============================

Simple, unified API for all execution tiers.

Usage:
    from Jotty import Jotty

    jotty = Jotty()
    result = await jotty.run("Research AI trends")
"""

import logging
from typing import Optional, Callable, Any
from pathlib import Path

from .core.execution import (
    UnifiedExecutor,
    ExecutionConfig,
    ExecutionTier,
    ExecutionResult,
    TierDetector,
)

logger = logging.getLogger(__name__)


class Jotty:
    """
    Jotty V3 - Unified AI Agent Framework

    Progressive complexity tiers:
    - Tier 1 (DIRECT): Fast single LLM call
    - Tier 2 (AGENTIC): Planning + orchestration (default)
    - Tier 3 (LEARNING): Memory + validation
    - Tier 4 (RESEARCH): Full V2 features

    Examples:
        # Simple usage (auto-detects tier)
        jotty = Jotty()
        result = await jotty.run("What is 2+2?")  # Tier 1

        result = await jotty.run("Research AI and create report")  # Tier 2

        # Explicit tier
        result = await jotty.run(
            "Analyze sales data",
            tier=ExecutionTier.LEARNING
        )

        # Full config
        from Jotty.core.execution import ExecutionConfig

        config = ExecutionConfig(
            tier=ExecutionTier.LEARNING,
            memory_backend="json",
            enable_validation=True,
        )
        result = await jotty.run("Task...", config=config)
    """

    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        log_level: str = "INFO",
    ):
        """
        Initialize Jotty.

        Args:
            config: Default execution config
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Create executor
        self.config = config or ExecutionConfig()
        self.executor = UnifiedExecutor(config=self.config)
        self.detector = TierDetector()

        logger.info("Jotty V3 initialized")

    async def run(
        self,
        goal: str,
        tier: Optional[ExecutionTier] = None,
        config: Optional[ExecutionConfig] = None,
        status_callback: Optional[Callable] = None,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute a task.

        Args:
            goal: Task description (natural language)
            tier: Override auto-detection with explicit tier
            config: Override default config
            status_callback: Optional callback(stage, detail) for progress
            **kwargs: Additional arguments

        Returns:
            ExecutionResult with output and metadata

        Examples:
            # Auto-detect tier
            result = await jotty.run("What is 2+2?")

            # Explicit tier
            result = await jotty.run("Task...", tier=ExecutionTier.DIRECT)

            # With callback
            def callback(stage, detail):
                print(f"[{stage}] {detail}")

            result = await jotty.run("Task...", status_callback=callback)
        """
        # Override tier if specified
        exec_config = config or self.config
        if tier:
            exec_config = ExecutionConfig(**{
                **exec_config.__dict__,
                'tier': tier,
            })

        # Execute
        result = await self.executor.execute(
            goal=goal,
            config=exec_config,
            status_callback=status_callback,
            **kwargs
        )

        return result

    def explain_tier(self, goal: str) -> str:
        """
        Explain which tier would be used for a goal.

        Args:
            goal: Task description

        Returns:
            Explanation string

        Example:
            explanation = jotty.explain_tier("What is 2+2?")
            print(explanation)
            # Tier 1 (DIRECT) selected:
            #   1. Contains direct query keywords
            #   2. Short query (≤10 words)
        """
        return self.detector.explain_detection(goal)

    async def chat(
        self,
        message: str,
        tier: ExecutionTier = ExecutionTier.DIRECT,
        **kwargs
    ) -> str:
        """
        Chat mode - simple question/answer.

        Uses Tier 1 (DIRECT) by default for fast responses.

        Args:
            message: User message
            tier: Execution tier (default: DIRECT)
            **kwargs: Additional arguments

        Returns:
            String response

        Example:
            response = await jotty.chat("What is the capital of France?")
            print(response)  # "Paris"
        """
        result = await self.run(message, tier=tier, **kwargs)
        return str(result.output)

    async def plan(self, goal: str) -> ExecutionResult:
        """
        Create and execute a plan for complex task.

        Forces Tier 2 (AGENTIC) execution with planning.

        Args:
            goal: Task description

        Returns:
            ExecutionResult with plan and steps

        Example:
            result = await jotty.plan("Research AI and create report")
            print(f"Plan: {len(result.steps)} steps")
            for step in result.steps:
                print(f"  - {step.description}")
        """
        return await self.run(goal, tier=ExecutionTier.AGENTIC)

    async def learn(
        self,
        goal: str,
        memory_backend: str = "json",
        enable_validation: bool = True,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute with learning (memory + validation).

        Forces Tier 3 (LEARNING) execution.

        Args:
            goal: Task description
            memory_backend: "json" | "redis" | "none"
            enable_validation: Enable output validation
            **kwargs: Additional arguments

        Returns:
            ExecutionResult with validation and memory data

        Example:
            result = await jotty.learn("Analyze sales data")
            print(f"Validated: {result.validation.success}")
            print(f"Used memory: {result.used_memory}")
        """
        config = ExecutionConfig(
            tier=ExecutionTier.LEARNING,
            memory_backend=memory_backend,
            enable_validation=enable_validation,
            **kwargs
        )
        return await self.run(goal, config=config)

    async def research(
        self,
        goal: str,
        enable_td_lambda: bool = True,
        enable_hierarchical_memory: bool = True,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute with full research features (V2).

        Forces Tier 4 (RESEARCH) execution.

        Args:
            goal: Task description
            enable_td_lambda: Enable TD-Lambda learning
            enable_hierarchical_memory: Enable 5-level memory
            **kwargs: Additional arguments

        Returns:
            ExecutionResult with full V2 data

        Example:
            result = await jotty.research("Optimize agent performance")
            print(f"V2 episode: {result.v2_episode}")
        """
        config = ExecutionConfig(
            tier=ExecutionTier.RESEARCH,
            enable_td_lambda=enable_td_lambda,
            enable_hierarchical_memory=enable_hierarchical_memory,
            **kwargs
        )
        return await self.run(goal, config=config)

    def set_default_tier(self, tier: ExecutionTier):
        """
        Set default execution tier.

        Args:
            tier: Default tier for run()

        Example:
            jotty.set_default_tier(ExecutionTier.LEARNING)
            result = await jotty.run("Task...")  # Uses LEARNING
        """
        self.config.tier = tier
        self.executor.config.tier = tier
        logger.info(f"Default tier set to: {tier.name}")

    def get_stats(self) -> dict:
        """
        Get execution statistics.

        Returns:
            Dict with stats

        Example:
            stats = jotty.get_stats()
            print(stats)
        """
        return {
            'default_tier': self.config.tier.name if self.config.tier else 'AUTO',
            'memory_backend': self.config.memory_backend,
            'validation_enabled': self.config.enable_validation,
        }


# Convenience functions for quick usage
async def run(goal: str, **kwargs) -> ExecutionResult:
    """
    Quick run without creating Jotty instance.

    Example:
        from Jotty import run

        result = await run("What is 2+2?")
        print(result.output)
    """
    jotty = Jotty()
    return await jotty.run(goal, **kwargs)


async def chat(message: str, **kwargs) -> str:
    """
    Quick chat without creating Jotty instance.

    Example:
        from Jotty import chat

        response = await chat("What is the capital of France?")
        print(response)  # "Paris"
    """
    jotty = Jotty()
    return await jotty.chat(message, **kwargs)

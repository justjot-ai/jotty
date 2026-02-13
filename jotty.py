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
from typing import AsyncGenerator, Optional, Callable, Any, Dict, List
from pathlib import Path

from .core.execution import (
    TierExecutor,
    ExecutionConfig,
    ExecutionTier,
    ExecutionResult,
    TierDetector,
)
from .core.execution.types import StreamEvent, StreamEventType

logger = logging.getLogger(__name__)


class Jotty:
    """
    Jotty V3 - Unified AI Agent Framework

    Progressive complexity tiers:
    - Tier 1 (DIRECT): Fast single LLM call
    - Tier 2 (AGENTIC): Planning + orchestration (default)
    - Tier 3 (LEARNING): Memory + validation
    - Tier 4 (RESEARCH): Domain swarm execution
    - Tier 5 (AUTONOMOUS): Sandbox + coalition + full features

    Examples:
        # Simple usage (auto-detects tier)
        jotty = Jotty()
        result = await jotty.run("What is 2+2?")  # Tier 1

        result = await jotty.run("Research AI and create report")  # Tier 2

        # Specific swarm
        result = await jotty.swarm("Build REST API", swarm_name="coding")

        # Full autonomous
        result = await jotty.autonomous("Analyze untrusted code in sandbox")

        # Explicit tier
        result = await jotty.run(
            "Analyze sales data",
            tier=ExecutionTier.LEARNING
        )
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
        self.executor = TierExecutor(config=self.config)
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
        if tier is not None:
            resolved_tier = ExecutionTier(tier) if isinstance(tier, int) else tier
            exec_config = ExecutionConfig(**{
                **exec_config.__dict__,
                'tier': resolved_tier,
            })

        # Execute
        result = await self.executor.execute(
            goal=goal,
            config=exec_config,
            status_callback=status_callback,
            **kwargs
        )

        return result

    async def stream(
        self,
        goal: str,
        tier: Optional[ExecutionTier] = None,
        config: Optional[ExecutionConfig] = None,
        **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream execution events as an async generator.

        Yields events as execution progresses:
        - STATUS: phase changes (planning, executing, validating...)
        - TOKEN: individual LLM tokens (Tier 1 streaming)
        - STEP_COMPLETE: step finished with partial output
        - RESULT: final ExecutionResult
        - ERROR: execution error

        Args:
            goal: Task description
            tier: Override auto-detection with explicit tier
            config: Override default config
            **kwargs: Additional arguments

        Yields:
            StreamEvent objects

        Examples:
            # Token streaming (Tier 1)
            async for event in jotty.stream("What is 2+2?"):
                if event.type == StreamEventType.TOKEN:
                    print(event.data, end="", flush=True)
                elif event.type == StreamEventType.RESULT:
                    print(f"\\nDone: {event.data.cost_usd}")

            # Progress streaming (any tier)
            async for event in jotty.stream("Research AI trends"):
                if event.type == StreamEventType.STATUS:
                    print(f"[{event.data['stage']}] {event.data['detail']}")
        """
        exec_config = config or self.config
        if tier:
            exec_config = ExecutionConfig(**{
                **exec_config.__dict__,
                'tier': tier,
            })

        async for event in self.executor.stream(goal, config=exec_config, **kwargs):
            yield event

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
            print(f"Episode: {result.episode}")
        """
        config = ExecutionConfig(
            tier=ExecutionTier.RESEARCH,
            enable_td_lambda=enable_td_lambda,
            enable_hierarchical_memory=enable_hierarchical_memory,
            **kwargs
        )
        return await self.run(goal, config=config)

    async def swarm(self, goal: str, swarm_name: str, **kwargs) -> ExecutionResult:
        """
        Run a specific domain swarm directly (Tier 4).

        Args:
            goal: Task description
            swarm_name: Swarm to use (e.g. "coding", "research", "testing")
            **kwargs: Additional arguments

        Returns:
            ExecutionResult from the domain swarm

        Example:
            result = await jotty.swarm("Build REST API", swarm_name="coding")
        """
        config = ExecutionConfig(tier=ExecutionTier.RESEARCH, swarm_name=swarm_name)
        return await self.run(goal, config=config, **kwargs)

    async def autonomous(self, goal: str, sandbox: bool = True, **kwargs) -> ExecutionResult:
        """
        Run with full autonomous features — sandbox, coalition (Tier 5).

        Args:
            goal: Task description
            sandbox: Enable sandbox execution (default True)
            **kwargs: Additional arguments

        Returns:
            ExecutionResult with autonomous metadata

        Example:
            result = await jotty.autonomous("Execute untrusted code safely")
        """
        config = ExecutionConfig(
            tier=ExecutionTier.AUTONOMOUS,
            enable_sandbox=sandbox,
        )
        return await self.run(goal, config=config, **kwargs)

    @staticmethod
    def list_swarms() -> List[str]:
        """List all available domain swarms.

        Returns:
            List of registered swarm names

        Example:
            swarms = Jotty.list_swarms()
            print(swarms)  # ['coding', 'research', 'testing', ...]
        """
        from Jotty.core.swarms.registry import SwarmRegistry
        return SwarmRegistry.list_all()

    @staticmethod
    def list_tiers() -> Dict[int, str]:
        """List all execution tiers with descriptions.

        Returns:
            Dict mapping tier number to description

        Example:
            for num, desc in Jotty.list_tiers().items():
                print(f"  {num}. {desc}")
        """
        return {
            1: "DIRECT — Single LLM call (1-2s, ~$0.01)",
            2: "AGENTIC — Planning + orchestration (3-5s, ~$0.03)",
            3: "LEARNING — Memory + validation (5-10s, ~$0.06)",
            4: "RESEARCH — Domain swarm execution (10-30s, ~$0.15)",
            5: "AUTONOMOUS — Sandbox + coalition + full features (30-120s, ~$0.50)",
        }

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
        Get execution statistics with real metrics from observability.

        Returns:
            Dict with config, session, global, per_agent, and task_types data

        Example:
            stats = jotty.get_stats()
            print(stats['global']['total_executions'])
        """
        from Jotty.core.observability import get_metrics
        summary = get_metrics().get_summary()
        return {
            'config': {
                'default_tier': self.config.tier.name if self.config.tier else 'AUTO',
                'memory_backend': self.config.memory_backend,
                'validation_enabled': self.config.enable_validation,
            },
            **summary,
        }

    def get_cost_breakdown(self) -> dict:
        """
        Get cost breakdown by agent and model.

        Returns:
            Dict with total_cost_usd, by_agent, by_model

        Example:
            costs = jotty.get_cost_breakdown()
            print(f"Total: ${costs['total_cost_usd']:.4f}")
        """
        from Jotty.core.observability import get_metrics
        return get_metrics().get_cost_breakdown()

    def get_recent_errors(self, limit: int = 10) -> list:
        """
        Get recent execution errors.

        Args:
            limit: Maximum number of errors to return

        Returns:
            List of error dicts with agent, task_type, error, timestamp

        Example:
            errors = jotty.get_recent_errors()
            for e in errors:
                print(f"{e['agent']}: {e['error']}")
        """
        from Jotty.core.observability import get_metrics
        return get_metrics().recent_errors(limit)

    def save_metrics(self, path: str = None) -> str:
        """
        Save metrics to a JSON file.

        Args:
            path: File path (defaults to ~/jotty/v3_metrics/session.json)

        Returns:
            Path where metrics were saved

        Example:
            saved = jotty.save_metrics()
            print(f"Saved to {saved}")
        """
        import json
        from Jotty.core.observability import get_metrics

        path = path or str(Path.home() / "jotty" / "v3_metrics" / "session.json")
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = get_metrics().get_summary()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return path


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

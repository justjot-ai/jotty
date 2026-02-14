"""
Jotty GAIA Adapter — sync→async bridge for GAIA benchmark evaluation.

Wraps the async Jotty.run() behind the sync agent.run(question) interface
that GAIABenchmark.evaluate_task() expects.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

logger = logging.getLogger(__name__)

# GAIA-optimized system prompt: forces concise answers for exact-match scoring.
GAIA_SYSTEM_PROMPT = (
    "You are answering a benchmark question. "
    "Answer concisely. Give ONLY the final answer with no explanation. "
    "Do not include units unless the question explicitly asks for them."
)


class JottyGAIAAdapter:
    """
    Sync adapter wrapping Jotty for GAIA evaluation.

    GAIABenchmark.evaluate_task() calls adapter.run(question) synchronously.
    This adapter translates that into an async Jotty.run() call.

    After each call, `last_result` holds the full ExecutionResult so the
    runner can extract cost/tokens/latency.

    Usage:
        adapter = JottyGAIAAdapter(tier="DIRECT")
        result = benchmark.evaluate_task(task, adapter)
        cost = adapter.last_result.cost_usd
    """

    def __init__(
        self,
        tier: Optional[str] = None,
        model: Optional[str] = None,
        dry_run: bool = False,
    ):
        """
        Args:
            tier: Execution tier name (DIRECT, AGENTIC, etc.). None = auto-detect.
            model: Model override (e.g. 'claude-sonnet-4-20250514').
            dry_run: If True, skip Jotty entirely and return placeholder.
        """
        self.tier = tier
        self.model = model
        self.dry_run = dry_run
        self.last_result = None  # ExecutionResult from last run
        self._jotty = None  # Lazy-initialized

    def _get_jotty(self):
        """Lazy-initialize the Jotty instance."""
        if self._jotty is None:
            from Jotty.jotty import Jotty
            self._jotty = Jotty()
        return self._jotty

    def _build_prompt(self, question: str) -> str:
        """Wrap question with GAIA-optimized prompt."""
        return f"{GAIA_SYSTEM_PROMPT}\n\nQuestion: {question}"

    def run(self, question: str, **kwargs) -> str:
        """
        Synchronous entry point for GAIABenchmark.evaluate_task().

        Returns the agent's answer as a string.
        """
        if self.dry_run:
            self.last_result = None
            return "[DRY RUN]"

        prompt = self._build_prompt(question)

        # Build kwargs for Jotty.run()
        run_kwargs = {}

        if self.tier:
            from Jotty.core.execution.types import ExecutionTier
            tier_map = {t.name: t for t in ExecutionTier}
            tier_enum = tier_map.get(self.tier.upper())
            if tier_enum:
                run_kwargs['tier'] = tier_enum

        if self.model:
            from Jotty.core.execution.types import ExecutionConfig
            run_kwargs['config'] = ExecutionConfig(model=self.model)

        result = self._run_async(prompt, **run_kwargs)
        self.last_result = result

        # Extract the text output
        output = result.output if result else ""
        return str(output) if output else ""

    def _run_async(self, prompt: str, **kwargs):
        """Run the async Jotty.run() from sync context."""
        jotty = self._get_jotty()

        async def _exec():
            return await jotty.run(prompt, **kwargs)

        # If there's already an event loop running (e.g. Jupyter),
        # use a thread pool to avoid "cannot run nested event loop"
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _exec())
                return future.result()
        else:
            return asyncio.run(_exec())

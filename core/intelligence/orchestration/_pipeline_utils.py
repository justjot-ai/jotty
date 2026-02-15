"""
Pipeline utility functions (AgentScope-inspired convenience).

Extracted from __init__.py to keep the lazy-import init clean.
"""

import asyncio
from typing import List, Any

from .agent_runner import AgentRunner
from Jotty.core.infrastructure.foundation.data_structures import EpisodeResult


async def sequential_pipeline(runners: List[AgentRunner], goal: str, **kwargs: Any) -> EpisodeResult:
    """
    Run agents sequentially, chaining output.

    Each agent receives the previous agent's output as additional context.
    Useful for: research -> summarize -> format pipelines.

    DRY: Reuses AgentRunner.run() for each step.
    """
    result = None
    for runner in runners:
        enriched = goal
        if result and result.output:
            enriched = f"{goal}\n\nPrevious output:\n{str(result.output)[:2000]}"
        result = await runner.run(goal=enriched, **kwargs)
        if not result.success:
            break
    return result


async def fanout_pipeline(runners: List[AgentRunner], goal: str, **kwargs: Any) -> List[EpisodeResult]:
    """
    Run agents in parallel on the same input.

    Useful for: getting multiple perspectives / ensemble approaches.

    DRY: Reuses AgentRunner.run() and asyncio.gather.
    """
    return await asyncio.gather(
        *(r.run(goal=goal, **kwargs) for r in runners)
    )

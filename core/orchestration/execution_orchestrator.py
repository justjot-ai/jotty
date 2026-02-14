"""
ExecutionOrchestrator - Extracted from Orchestrator
====================================================

Handles the actual execution dispatch logic for single-agent and multi-agent modes.
This separates "how to run agents" from "how to configure/manage agents" (Orchestrator).

Architecture:
    Orchestrator owns ExecutionOrchestrator
    Orchestrator.run() delegates to ExecutionOrchestrator.execute()
    ExecutionOrchestrator delegates paradigm dispatch to paradigm methods

This file is intentionally thin - it holds the dispatch logic that was previously
inlined in Orchestrator.run() and _execute_multi_agent(). The actual agent execution
still goes through AgentRunner.run().
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .swarm_manager import Orchestrator

logger = logging.getLogger(__name__)


class ExecutionOrchestrator:
    """Handles agent execution dispatch for Orchestrator.

    Extracted to reduce Orchestrator's cognitive load from ~2700 lines.
    Owns the logic for:
    - Single vs multi-agent routing
    - Discussion paradigm selection (fanout, relay, debate, refinement)
    - Coalition formation for multi-agent tasks
    - Handoff coordination for relay paradigm
    - Efficiency metrics collection

    Does NOT own:
    - Agent/runner creation (Orchestrator)
    - Learning pipelines (SwarmLearningPipeline)
    - Ensemble management (EnsembleManager)
    - Provider management (ProviderManager)
    """

    def __init__(self, manager: 'Orchestrator'):
        """Initialize with reference to owning Orchestrator.

        Args:
            manager: The Orchestrator instance that owns this orchestrator
        """
        self._mgr = manager
        self._efficiency_stats: Dict[str, Any] = {}

    async def execute_single(self, goal: str, **kwargs) -> Any:
        """Execute in single-agent mode.

        Delegates to Orchestrator._execute_single_agent for now,
        but provides a clean seam for future extraction.
        """
        return await self._mgr._execute_single_agent(goal, **kwargs)

    async def execute_multi(self, goal: str, **kwargs) -> Any:
        """Execute in multi-agent mode with paradigm dispatch.

        Delegates to Orchestrator._execute_multi_agent for now,
        but provides a clean seam for future extraction.
        """
        return await self._mgr._execute_multi_agent(goal, **kwargs)

    def select_paradigm(self, goal: str, default: str = 'fanout') -> str:
        """Select the best discussion paradigm based on learning data.

        Args:
            goal: Task goal
            default: Fallback paradigm

        Returns:
            Paradigm name: 'fanout', 'relay', 'debate', 'refinement'
        """
        try:
            lp = self._mgr.learning
            task_type = lp.transfer_learning.extractor.extract_task_type(goal)
            paradigm = lp.recommend_paradigm(task_type)
            logger.info(f"Auto paradigm: selected '{paradigm}' for task_type='{task_type}'")
            return paradigm
        except Exception:
            return default

    def wire_coordination(self, paradigm: str, goal: str, agent_names: List[str]) -> None:
        """Wire coordination protocols based on paradigm.

        Args:
            paradigm: Discussion paradigm being used
            goal: Task goal
            agent_names: List of agent names participating
        """
        si = self._mgr.swarm_intelligence
        if not si or len(agent_names) < 2:
            return

        try:
            if paradigm == 'relay' and len(agent_names) >= 2:
                si.initiate_handoff(
                    from_agent=agent_names[0],
                    to_agent=agent_names[1],
                    task=goal,
                    context={'paradigm': 'relay', 'agents': agent_names}
                )
            elif paradigm == 'fanout':
                task_type = self._mgr.learning.transfer_learning.extractor.extract_task_type(goal)
                si.form_coalition(
                    task_type=task_type,
                    min_agents=min(2, len(agent_names)),
                    available_agents=agent_names
                )
                logger.info(f"Coalition formed for '{task_type}' with {len(agent_names)} agents")
        except Exception as e:
            logger.debug(f"Coordination wiring skipped: {e}")

    def compute_efficiency(self, start_time: float, result, ensemble_time: float = 0) -> Dict[str, float]:
        """Compute execution efficiency metrics.

        Args:
            start_time: When execution started
            result: EpisodeResult
            ensemble_time: Time spent on ensemble (if any)

        Returns:
            Dict with total_time, overhead_pct, etc.
        """
        total_elapsed = time.time() - start_time
        exec_t = getattr(result, 'execution_time', total_elapsed - ensemble_time)
        overhead = max(0, total_elapsed - exec_t)
        overhead_pct = (overhead / total_elapsed * 100) if total_elapsed > 0 else 0

        stats = {
            'total_time': total_elapsed,
            'execution_time': exec_t,
            'ensemble_time': ensemble_time,
            'overhead_time': overhead,
            'overhead_pct': overhead_pct,
        }
        self._efficiency_stats = stats
        return stats

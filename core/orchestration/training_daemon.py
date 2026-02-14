"""
TrainingDaemon - Background self-improvement loop
==================================================

Extracted from Orchestrator for decomposition.
Manages curriculum-based training: task queue, daemon lifecycle,
convergence detection.

Usage:
    daemon = TrainingDaemon(orchestrator)
    daemon.start(max_tasks=10)
    status = daemon.status()
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List

from Jotty.core.foundation.data_structures import EpisodeResult

logger = logging.getLogger(__name__)


class TrainingDaemon:
    """
    Background training daemon and self-improvement loop.

    Composed by Orchestrator. Holds a reference to the Orchestrator
    for access to learning pipeline and run() method.
    """

    def __init__(self, manager: Any) -> None:
        self._manager = manager
        self._daemon_task: Optional[asyncio.Task] = None
        self._results: List[EpisodeResult] = []

    async def run_training_task(self) -> Optional[EpisodeResult]:
        """
        Pop and execute the next queued training task.

        Returns None if no training tasks are pending.
        """
        sm = self._manager
        task = sm.learning.pop_training_task()
        if task is None:
            return None

        logger.info(
            f" Running training task: {task.description[:60]} "
            f"(difficulty={task.difficulty:.2f})"
        )
        try:
            result = await sm.run(
                goal=task.description,
                skip_autonomous_setup=True,
                skip_validation=True,
            )
            logger.info(
                f" Training task {'passed' if result.success else 'failed'}: "
                f"{task.description[:40]}"
            )
            return result
        except Exception as e:
            logger.warning(f"Training task failed: {e}")
            return None

    @property
    def pending_count(self) -> int:
        """Number of curriculum-generated training tasks waiting."""
        try:
            return self._manager.learning.pending_training_count()
        except Exception as e:
            logger.debug(f"Pending training count unavailable: {e}")
            return 0

    async def start_training_loop(
        self,
        max_tasks: int = 5,
        interval_seconds: float = 0.0,
        stop_on_convergence: bool = True,
    ) -> list:
        """
        Autonomous self-improvement loop.

        Drains the curriculum queue until empty, max_tasks reached,
        or adaptive learning detects convergence.
        """
        sm = self._manager
        results = []
        for i in range(max_tasks):
            if stop_on_convergence:
                try:
                    al = sm.learning.adaptive_learning
                    if al.state.is_converging and al.should_stop_early():
                        logger.info(
                            f" Training loop: converged after {i} tasks, stopping"
                        )
                        break
                except Exception as e:
                    logger.debug(f"Convergence check failed: {e}")

            result = await self.run_training_task()
            if result is None:
                logger.info(f" Training loop: queue empty after {i} tasks")
                break

            results.append(result)

            if interval_seconds > 0:
                await asyncio.sleep(interval_seconds)

        logger.info(
            f" Training loop complete: {len(results)} tasks, "
            f"{sum(1 for r in results if r.success)}/{len(results)} succeeded"
        )
        return results

    def start(
        self,
        max_tasks: int = 10,
        interval_seconds: float = 2.0,
        stop_on_convergence: bool = True,
    ) -> bool:
        """
        Start a background training daemon as an asyncio.Task.

        Returns True if started, False if already running.
        """
        if self._daemon_task and not self._daemon_task.done():
            logger.info(" Training daemon already running")
            return False

        self._results = []

        async def _daemon() -> Any:
            try:
                results = await self.start_training_loop(
                    max_tasks=max_tasks,
                    interval_seconds=interval_seconds,
                    stop_on_convergence=stop_on_convergence,
                )
                self._results = results
            except asyncio.CancelledError:
                logger.info(" Training daemon cancelled")
            except Exception as e:
                logger.warning(f" Training daemon error: {e}")

        self._daemon_task = asyncio.ensure_future(_daemon())
        logger.info(
            f" Training daemon started (max_tasks={max_tasks}, "
            f"interval={interval_seconds}s)"
        )
        return True

    def stop(self) -> bool:
        """
        Cancel the background training daemon.

        Returns True if cancelled, False if not running.
        """
        if not self._daemon_task or self._daemon_task.done():
            return False
        self._daemon_task.cancel()
        logger.info(" Training daemon stop requested")
        return True

    def status(self) -> Dict[str, Any]:
        """Get training daemon status."""
        running = (
            self._daemon_task is not None
            and not self._daemon_task.done()
        )
        results = self._results or []
        succeeded = sum(1 for r in results if r and r.success)

        return {
            'running': running,
            'completed': len(results),
            'succeeded': succeeded,
            'success_rate': succeeded / len(results) if results else 0.0,
            'pending_tasks': self.pending_count,
        }

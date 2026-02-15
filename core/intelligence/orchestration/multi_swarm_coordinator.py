"""
Multi-Swarm Coordination
=========================

Run multiple swarms in parallel and merge results intelligently.

KISS PRINCIPLE: Simple asyncio.gather (no complex orchestration).
DRY PRINCIPLE: Reuses existing BaseSwarm infrastructure.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)


class MergeStrategy(Enum):
    """Strategies for merging multi-swarm results."""
    VOTING = "voting"              # Majority vote
    ENSEMBLE = "ensemble"          # Average/weighted combination
    BEST_OF_N = "best_of_n"       # Highest confidence
    CONCATENATE = "concatenate"    # Combine all outputs
    FIRST_SUCCESS = "first_success"  # Return first successful result


@dataclass
class SwarmResult:
    """Result from a single swarm execution."""
    swarm_name: str
    output: str
    success: bool
    confidence: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class MultiSwarmCoordinator:
    """
    Execute multiple swarms in parallel and merge results.

    USE CASES:
    1. Parallel decomposition: "Research AI in healthcare AND education"
       → Spawn 2 ResearchSwarms, merge via concatenation

    2. Consensus voting: "Is this text positive or negative?"
       → Spawn 3 SentimentSwarms, merge via voting (2/3 agree)

    3. Ensemble: "Predict stock price"
       → Spawn 5 AnalysisSwarms, merge via averaging

    BENEFITS:
    - 2-5x faster (parallel execution)
    - Higher accuracy (ensemble/voting)
    - Fault tolerance (if one swarm fails, others succeed)
    """

    def __init__(self) -> None:
        self.execution_count = 0
        self.merge_stats: Dict[str, int] = {}

        logger.info("🤝 MultiSwarmCoordinator initialized")

    async def execute_parallel(
        self,
        swarms: List[Any],  # List[BaseSwarm]
        task: str,
        merge_strategy: MergeStrategy = MergeStrategy.VOTING,
        timeout_per_swarm: float = 60.0,
    ) -> SwarmResult:
        """
        Execute multiple swarms in parallel and merge results.

        Args:
            swarms: List of swarm instances to execute
            task: Task description
            merge_strategy: How to merge results
            timeout_per_swarm: Timeout per swarm (seconds)

        Returns:
            Merged SwarmResult
        """
        self.execution_count += 1

        logger.info(
            f"🚀 Executing {len(swarms)} swarms in parallel "
            f"(strategy={merge_strategy.value})"
        )

        # Execute all swarms concurrently (KISS - simple gather)
        tasks = [
            self._execute_with_timeout(swarm, task, timeout_per_swarm)
            for swarm in swarms
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failures
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Swarm {i} failed: {result}")
            elif isinstance(result, SwarmResult) and result.success:
                valid_results.append(result)

        if not valid_results:
            logger.error("❌ All swarms failed")
            return SwarmResult(
                swarm_name="multi_swarm",
                output="All swarms failed",
                success=False,
            )

        # Merge results based on strategy
        merged = self._merge_results(valid_results, merge_strategy)

        # Track stats
        self.merge_stats[merge_strategy.value] =             self.merge_stats.get(merge_strategy.value, 0) + 1

        logger.info(
            f"✅ Merged {len(valid_results)} results "
            f"(strategy={merge_strategy.value})"
        )

        return merged

    async def _execute_with_timeout(
        self,
        swarm: Any,
        task: str,
        timeout: float
    ) -> SwarmResult:
        """Execute single swarm with timeout."""
        try:
            # Call swarm.execute() if available, else swarm.run()
            if hasattr(swarm, 'execute'):
                result = await asyncio.wait_for(
                    swarm.execute(task),
                    timeout=timeout
                )
            elif hasattr(swarm, 'run'):
                result = await asyncio.wait_for(
                    swarm.run(goal=task),
                    timeout=timeout
                )
            else:
                raise ValueError(f"Swarm {swarm} has no execute() or run() method")

            # Convert to SwarmResult
            if isinstance(result, SwarmResult):
                return result
            elif isinstance(result, dict):
                return SwarmResult(
                    swarm_name=getattr(swarm, 'name', 'unknown'),
                    output=result.get('output', str(result)),
                    success=result.get('success', True),
                    confidence=result.get('confidence', 0.8),
                    metadata=result,
                )
            else:
                return SwarmResult(
                    swarm_name=getattr(swarm, 'name', 'unknown'),
                    output=str(result),
                    success=True,
                    confidence=0.7,
                )

        except asyncio.TimeoutError:
            logger.warning(f"Swarm timeout after {timeout}s")
            return SwarmResult(
                swarm_name=getattr(swarm, 'name', 'unknown'),
                output="Timeout",
                success=False,
            )
        except Exception as e:
            logger.error(f"Swarm execution failed: {e}")
            raise

    def _merge_results(
        self,
        results: List[SwarmResult],
        strategy: MergeStrategy
    ) -> SwarmResult:
        """Merge results using specified strategy."""

        if strategy == MergeStrategy.VOTING:
            return self._merge_voting(results)
        elif strategy == MergeStrategy.ENSEMBLE:
            return self._merge_ensemble(results)
        elif strategy == MergeStrategy.BEST_OF_N:
            return self._merge_best_of_n(results)
        elif strategy == MergeStrategy.CONCATENATE:
            return self._merge_concatenate(results)
        elif strategy == MergeStrategy.FIRST_SUCCESS:
            return results[0]  # Already filtered for success
        else:
            logger.warning(f"Unknown strategy {strategy}, using BEST_OF_N")
            return self._merge_best_of_n(results)

    def _merge_voting(self, results: List[SwarmResult]) -> SwarmResult:
        """Merge via majority voting (KISS - simple Counter)."""
        outputs = [r.output.strip() for r in results]
        counter = Counter(outputs)
        most_common_output, votes = counter.most_common(1)[0]

        # Find a result with the winning output
        winning_result = next(
            (r for r in results if r.output.strip() == most_common_output),
            results[0]
        )

        return SwarmResult(
            swarm_name="multi_swarm_voting",
            output=most_common_output,
            success=True,
            confidence=votes / len(results),  # Vote ratio as confidence
            metadata={
                'votes': votes,
                'total_swarms': len(results),
                'vote_distribution': dict(counter),
            }
        )

    def _merge_ensemble(self, results: List[SwarmResult]) -> SwarmResult:
        """Merge via weighted averaging (for numeric outputs)."""
        # Try to parse outputs as numbers
        try:
            values = [float(r.output) for r in results]
            weights = [r.confidence for r in results]

            # Weighted average
            weighted_sum = sum(v * w for v, w in zip(values, weights))
            total_weight = sum(weights)
            avg = weighted_sum / total_weight if total_weight > 0 else 0.0

            return SwarmResult(
                swarm_name="multi_swarm_ensemble",
                output=str(avg),
                success=True,
                confidence=sum(weights) / len(weights),
                metadata={
                    'method': 'weighted_average',
                    'individual_values': values,
                    'weights': weights,
                }
            )
        except ValueError:
            # Fall back to concatenation for non-numeric
            logger.debug("Ensemble merge failed (non-numeric), using concatenation")
            return self._merge_concatenate(results)

    def _merge_best_of_n(self, results: List[SwarmResult]) -> SwarmResult:
        """Return highest-confidence result."""
        best = max(results, key=lambda r: r.confidence)
        best.swarm_name = f"multi_swarm_best_of_{len(results)}"
        best.metadata = best.metadata or {}
        best.metadata['selection_reason'] = 'highest_confidence'
        best.metadata['total_swarms'] = len(results)
        return best

    def _merge_concatenate(self, results: List[SwarmResult]) -> SwarmResult:
        """Concatenate all outputs."""
        combined_output = "\n\n".join([
            f"=== {r.swarm_name} ===\n{r.output}"
            for r in results
        ])

        avg_confidence = sum(r.confidence for r in results) / len(results)

        return SwarmResult(
            swarm_name="multi_swarm_concatenated",
            output=combined_output,
            success=True,
            confidence=avg_confidence,
            metadata={
                'method': 'concatenation',
                'swarm_count': len(results),
            }
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get coordination statistics."""
        return {
            'total_executions': self.execution_count,
            'merge_strategy_usage': self.merge_stats,
        }


# Singleton
_coordinator = None


def get_multi_swarm_coordinator() -> MultiSwarmCoordinator:
    """Get or create multi-swarm coordinator singleton."""
    global _coordinator
    if _coordinator is None:
        _coordinator = MultiSwarmCoordinator()
    return _coordinator


def reset_multi_swarm_coordinator() -> None:
    """Reset the singleton coordinator (used in tests)."""
    global _coordinator
    _coordinator = None


__all__ = [
    'MultiSwarmCoordinator',
    'MergeStrategy',
    'SwarmResult',
    'get_multi_swarm_coordinator',
    'reset_multi_swarm_coordinator',
]

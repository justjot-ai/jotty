"""
ExecutionManager - Manages actor execution.

Extracted from conductor.py to improve maintainability and testability.
All execution-related logic is centralized here.
"""
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of actor execution."""
    success: bool
    output: Any
    duration: float = 0.0
    error: Optional[str] = None


class ExecutionManager:
    """
    Centralized execution management for actors.

    Responsibilities:
    - Actor execution coordination
    - Output collection
    - State updates
    - Execution statistics

    This manager coordinates actor execution (delegates complex logic to conductor).
    """

    def __init__(self, config):
        """
        Initialize execution manager.

        Args:
            config: JottyConfig with execution parameters
        """
        self.config = config
        self.execution_count = 0
        self.success_count = 0
        self.total_duration = 0.0

        logger.info("⚙️  ExecutionManager initialized")

    def record_execution(
        self,
        actor_name: str,
        success: bool,
        duration: float
    ):
        """
        Record execution statistics.

        Args:
            actor_name: Name of the actor
            success: Whether execution succeeded
            duration: Execution duration in seconds
        """
        self.execution_count += 1
        if success:
            self.success_count += 1
        self.total_duration += duration

        logger.debug(f"📊 Execution recorded: {actor_name} ({'success' if success else 'failed'}, {duration:.2f}s)")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dict with execution metrics
        """
        success_rate = (self.success_count / self.execution_count) if self.execution_count > 0 else 0.0
        avg_duration = (self.total_duration / self.execution_count) if self.execution_count > 0 else 0.0

        return {
            "total_executions": self.execution_count,
            "successes": self.success_count,
            "success_rate": success_rate,
            "total_duration": self.total_duration,
            "avg_duration": avg_duration
        }

    def reset_stats(self):
        """Reset execution statistics."""
        self.execution_count = 0
        self.success_count = 0
        self.total_duration = 0.0
        logger.debug("ExecutionManager stats reset")

    # NOTE: The actual _execute_actor logic remains in conductor.py for now
    # This manager provides statistics tracking and future extension points
    # Future enhancement: Move full execution logic here from conductor.py

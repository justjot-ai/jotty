"""
BaseManager - Abstract base class for all orchestration managers.

Enforces consistent interface across all manager implementations.
Provides common patterns for statistics tracking and lifecycle management.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseManager(ABC):
    """
    Abstract base class for all orchestration managers.

    All managers should inherit from this class to ensure:
    - Consistent statistics interface
    - Proper initialization patterns
    - Standardized lifecycle management
    """

    def __init__(self, config):
        """
        Initialize base manager.

        Args:
            config: JottyConfig instance
        """
        self.config = config
        self._initialized = True

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get manager statistics.

        Returns:
            Dict with manager-specific metrics

        Note:
            Each manager should return different stats based on its domain:
            - LearningManager: Q-values, learning rate, episodes
            - ValidationManager: Validation counts, confidence scores
            - ExecutionManager: Execution counts, success rates
            - etc.
        """
        pass

    @abstractmethod
    def reset_stats(self):
        """
        Reset manager statistics.

        Note:
            Each manager should reset its domain-specific counters
            while preserving configuration and state.
        """
        pass

    def is_initialized(self) -> bool:
        """
        Check if manager is initialized.

        Returns:
            True if manager is ready for use
        """
        return getattr(self, '_initialized', False)

    def __repr__(self) -> str:
        """String representation of manager."""
        return f"{self.__class__.__name__}(initialized={self.is_initialized()})"


class StatelessManager(BaseManager):
    """
    Base class for stateless managers that don't track statistics.

    Use this for managers that are purely functional (no state to track).
    """

    def get_stats(self) -> Dict[str, Any]:
        """
        Get minimal stats for stateless manager.

        Returns:
            Dict with just initialization status
        """
        return {
            "manager_type": self.__class__.__name__,
            "initialized": self.is_initialized()
        }

    def reset_stats(self):
        """
        No-op for stateless managers.

        Stateless managers have no statistics to reset.
        """
        logger.debug(f"{self.__class__.__name__} has no stats to reset (stateless)")


class StatefulManager(BaseManager):
    """
    Base class for stateful managers that track operations.

    Use this for managers that maintain counters, history, or state.
    Provides common patterns for stats tracking.
    """

    def __init__(self, config):
        """
        Initialize stateful manager with stats tracking.

        Args:
            config: JottyConfig instance
        """
        super().__init__(config)
        self._operation_count = 0
        self._error_count = 0
        self._last_operation_time = None

    def _increment_operation_count(self):
        """Increment operation counter."""
        self._operation_count += 1

    def _increment_error_count(self):
        """Increment error counter."""
        self._error_count += 1

    def get_base_stats(self) -> Dict[str, Any]:
        """
        Get common base statistics for all stateful managers.

        Returns:
            Dict with common metrics
        """
        return {
            "manager_type": self.__class__.__name__,
            "initialized": self.is_initialized(),
            "total_operations": self._operation_count,
            "total_errors": self._error_count,
            "last_operation": self._last_operation_time
        }

    def reset_base_stats(self):
        """Reset base statistics (call from subclass reset_stats)."""
        self._operation_count = 0
        self._error_count = 0
        self._last_operation_time = None
        logger.debug(f"{self.__class__.__name__} base stats reset")

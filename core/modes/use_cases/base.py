"""
Base Use Case Interface

Defines the common interface for all use cases in Jotty.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class UseCaseType(Enum):
    """Types of use cases supported."""

    CHAT = "chat"
    WORKFLOW = "workflow"


@dataclass
class UseCaseConfig:
    """Configuration for a use case."""

    use_case_type: UseCaseType
    max_iterations: int = 100
    enable_learning: bool = True
    enable_memory: bool = True
    enable_streaming: bool = True
    timeout: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class UseCaseResult:
    """Result from executing a use case."""

    success: bool
    output: Any
    metadata: Dict[str, Any]
    execution_time: float
    use_case_type: UseCaseType

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "output": (
                str(self.output) if not isinstance(self.output, (dict, list)) else self.output
            ),
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "use_case_type": self.use_case_type.value,
        }


class BaseUseCase(ABC):
    """
    Base class for all use cases.

    Each use case implements:
    - execute(): Synchronous execution
    - stream(): Streaming execution
    - enqueue(): Asynchronous execution (if supported)
    """

    def __init__(self, conductor: Any, config: Optional[UseCaseConfig] = None) -> None:
        """
        Initialize use case.

        Args:
            conductor: Jotty Conductor instance
            config: Use case configuration
        """
        self.conductor = conductor
        self.config = config or UseCaseConfig(use_case_type=self._get_use_case_type())
        self._validate_config()

    @abstractmethod
    def _get_use_case_type(self) -> UseCaseType:
        """Return the use case type."""
        pass

    def _validate_config(self) -> Any:
        """Validate configuration."""
        if self.config.use_case_type != self._get_use_case_type():
            raise ValueError(
                f"Config use_case_type ({self.config.use_case_type}) "
                f"doesn't match use case type ({self._get_use_case_type()})"
            )

    @abstractmethod
    async def execute(
        self, goal: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> UseCaseResult:
        """
        Execute use case synchronously.

        Args:
            goal: Goal or message to process
            context: Additional context
            **kwargs: Additional arguments

        Returns:
            UseCaseResult with execution results
        """
        pass

    @abstractmethod
    async def stream(
        self, goal: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute use case with streaming.

        Args:
            goal: Goal or message to process
            context: Additional context
            **kwargs: Additional arguments

        Yields:
            Event dictionaries
        """
        pass

    async def enqueue(
        self, goal: str, context: Optional[Dict[str, Any]] = None, priority: int = 3, **kwargs: Any
    ) -> str:
        """
        Enqueue task for asynchronous execution.

        Args:
            goal: Goal or message to process
            context: Additional context
            priority: Task priority (1-5, higher is more important)
            **kwargs: Additional arguments

        Returns:
            Task ID

        Raises:
            NotImplementedError: If async execution not supported
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support async execution")

    def _create_result(
        self,
        success: bool,
        output: Any,
        metadata: Optional[Dict[str, Any]] = None,
        execution_time: Optional[float] = None,
    ) -> UseCaseResult:
        """Create a UseCaseResult."""
        return UseCaseResult(
            success=success,
            output=output,
            metadata=metadata or {},
            execution_time=execution_time or 0.0,
            use_case_type=self._get_use_case_type(),
        )

    def _get_execution_time(self, start_time: float) -> float:
        """Calculate execution time."""
        return time.time() - start_time

    # DRY: Common execution wrapper with error handling and timing
    async def _execute_with_error_handling(
        self, executor_method: Callable[..., Awaitable[Dict[str, Any]]], **kwargs: Any
    ) -> UseCaseResult:
        """
        DRY wrapper for execute with timing and error handling.

        Eliminates duplicate try/except/timing code across use cases.

        Args:
            executor_method: Async method to execute (e.g., self.executor.execute)
            **kwargs: Arguments to pass to executor_method

        Returns:
            UseCaseResult with success/failure details
        """
        start_time = time.time()
        try:
            result = await executor_method(**kwargs)
            execution_time = time.time() - start_time

            return self._create_result(
                success=result.get("success", False),
                output=self._extract_output(result),
                metadata=self._extract_metadata(result, execution_time),
                execution_time=execution_time,
            )
        except Exception as e:
            logger.error(f"{self.__class__.__name__} execution failed: {e}", exc_info=True)
            execution_time = time.time() - start_time

            return self._create_result(
                success=False,
                output=self._error_output(e),
                metadata={"error": str(e)},
                execution_time=execution_time,
            )

    def _extract_output(self, result: Dict[str, Any]) -> Any:
        """Extract output from executor result. Override per use case."""
        return result.get("output") or result.get("message") or result.get("result")

    def _extract_metadata(self, result: Dict[str, Any], execution_time: float) -> Dict[str, Any]:
        """Extract metadata from executor result. Override per use case."""
        metadata = result.get("metadata", {}).copy()
        metadata["execution_time"] = result.get("execution_time", execution_time)
        return metadata

    def _error_output(self, error: Exception) -> Any:
        """Format error output. Override per use case."""
        return f"Error: {str(error)}"

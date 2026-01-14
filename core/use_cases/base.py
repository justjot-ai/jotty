"""
Base Use Case Interface

Defines the common interface for all use cases in Jotty.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, AsyncIterator
from dataclasses import dataclass
from enum import Enum
import time
import logging

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
            "output": str(self.output) if not isinstance(self.output, (dict, list)) else self.output,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "use_case_type": self.use_case_type.value
        }


class BaseUseCase(ABC):
    """
    Base class for all use cases.
    
    Each use case implements:
    - execute(): Synchronous execution
    - stream(): Streaming execution
    - enqueue(): Asynchronous execution (if supported)
    """
    
    def __init__(
        self,
        conductor: Any,  # Conductor instance
        config: Optional[UseCaseConfig] = None
    ):
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
    
    def _validate_config(self):
        """Validate configuration."""
        if self.config.use_case_type != self._get_use_case_type():
            raise ValueError(
                f"Config use_case_type ({self.config.use_case_type}) "
                f"doesn't match use case type ({self._get_use_case_type()})"
            )
    
    @abstractmethod
    async def execute(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
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
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
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
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        priority: int = 3,
        **kwargs
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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async execution"
        )
    
    def _create_result(
        self,
        success: bool,
        output: Any,
        metadata: Optional[Dict[str, Any]] = None,
        execution_time: Optional[float] = None
    ) -> UseCaseResult:
        """Create a UseCaseResult."""
        return UseCaseResult(
            success=success,
            output=output,
            metadata=metadata or {},
            execution_time=execution_time or 0.0,
            use_case_type=self._get_use_case_type()
        )
    
    def _get_execution_time(self, start_time: float) -> float:
        """Calculate execution time."""
        return time.time() - start_time

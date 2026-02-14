"""
Timeout Management and Circuit Breakers for ReVal.

Provides production-ready timeout handling, circuit breakers, and dead letter queues
to prevent cascading failures and ensure system resilience.

A-Team Design:
- Alex Shannon: Robust error handling and recovery
- Dr. Chen: Circuit breaker state machine
- Marcus Rodriguez: Scalable dead letter queue
- Prof. Manning: Adaptive timeout strategy
"""
import time
import asyncio
import functools
import logging
from typing import Optional, Callable, Any, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import signal

logger = logging.getLogger(__name__)


# =============================================================================
# CIRCUIT BREAKER STATE MACHINE
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit tripped, rejecting requests
    HALF_OPEN = "half_open"  # Testing if system recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout: float = 60.0  # Seconds before trying half-open
    name: str = "default"


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.
    
    Pattern:
    - CLOSED: Allows all requests, counts failures
    - OPEN: Rejects all requests immediately (fail fast)
    - HALF_OPEN: Allows test requests to check recovery
    
    Example:
        breaker = CircuitBreaker(name="llm_calls", failure_threshold=3)
        
        @breaker.protect
        async def call_llm():
            return await llm.generate(...)
    """
    
    def __init__(self, config: CircuitBreakerConfig) -> None:
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.total_calls = 0
        self.total_failures = 0
        
    def can_request(self) -> Tuple[bool, str]:
        """Check if request should be allowed."""
        self.total_calls += 1
        
        if self.state == CircuitState.CLOSED:
            return True, "Circuit closed, allowing request"
        
        if self.state == CircuitState.OPEN:
            # Check if timeout expired
            if self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.config.timeout:
                    # Try half-open
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f" [{self.config.name}] Circuit HALF_OPEN (testing recovery)")
                    return True, "Circuit half-open, testing recovery"
            
            return False, f"Circuit open, rejecting request (wait {self.config.timeout}s)"
        
        if self.state == CircuitState.HALF_OPEN:
            return True, "Circuit half-open, allowing test request"
        
        return False, "Unknown circuit state"
    
    def record_success(self) -> None:
        """Record successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f" [{self.config.name}] Circuit CLOSED (system recovered)")
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self, error: Exception) -> None:
        """Record failed call."""
        self.failure_count += 1
        self.total_failures += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.error(f" [{self.config.name}] Circuit OPEN! {self.failure_count} consecutive failures")
                logger.error(f"   Last error: {error}")
        
        elif self.state == CircuitState.HALF_OPEN:
            # Failed during testing, back to open
            self.state = CircuitState.OPEN
            logger.warning(f" [{self.config.name}] Circuit back to OPEN (recovery test failed)")
    
    def protect(self, func: Callable) -> Callable:
        """Decorator to protect a function with circuit breaker."""
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                can_call, reason = self.can_request()
                if not can_call:
                    raise CircuitOpenError(f"[{self.config.name}] {reason}")
                
                try:
                    result = await func(*args, **kwargs)
                    self.record_success()
                    return result
                except Exception as e:
                    self.record_failure(e)
                    raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                can_call, reason = self.can_request()
                if not can_call:
                    raise CircuitOpenError(f"[{self.config.name}] {reason}")
                
                try:
                    result = func(*args, **kwargs)
                    self.record_success()
                    return result
                except Exception as e:
                    self.record_failure(e)
                    raise
            return sync_wrapper
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            'name': self.config.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'total_calls': self.total_calls,
            'total_failures': self.total_failures,
            'failure_rate': self.total_failures / self.total_calls if self.total_calls > 0 else 0.0,
            'last_failure_time': self.last_failure_time
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# =============================================================================
# TIMEOUT DECORATORS
# =============================================================================

from Jotty.core.foundation.exceptions import TimeoutError  # noqa: F811


def timeout(seconds: float, error_message: str = 'Operation timed out') -> Any:
    """
    Timeout decorator for synchronous functions.
    
    Uses signal.alarm (Unix only) for true timeout without threads.
    
    Args:
        seconds: Timeout in seconds
        error_message: Custom error message
    
    Example:
        @timeout(30, "LLM call timed out")
        def call_llm():
            return llm.generate(...)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # For async functions, use async_timeout instead
            if asyncio.iscoroutinefunction(func):
                raise ValueError(f"Use async_timeout for async function {func.__name__}")
            
            def timeout_handler(signum: Any, frame: Any) -> None:
                raise TimeoutError(f"{error_message} (after {seconds}s)")
            
            # Set timeout (Unix only)
            try:
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(seconds))
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)  # Cancel alarm
                    signal.signal(signal.SIGALRM, old_handler)
                
                return result
            except AttributeError:
                # Windows doesn't have signal.SIGALRM
                logger.warning(f" Timeout not supported on this platform, running without timeout")
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def async_timeout(seconds: float, error_message: str = 'Operation timed out') -> Any:
    """
    Timeout decorator for async functions.
    
    Uses asyncio.wait_for for proper async timeout.
    
    Args:
        seconds: Timeout in seconds
        error_message: Custom error message
    
    Example:
        @async_timeout(30, "LLM call timed out")
        async def call_llm():
            return await llm.generate(...)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"{error_message} (after {seconds}s)")
        
        return wrapper
    return decorator


# =============================================================================
# DEAD LETTER QUEUE
# =============================================================================

@dataclass
class FailedOperation:
    """Record of a failed operation."""
    operation_name: str
    args: Tuple
    kwargs: Dict
    error: Exception
    timestamp: datetime
    retry_count: int = 0
    last_retry: Optional[datetime] = None


class DeadLetterQueue:
    """
    Queue for failed operations that can be retried later.
    
    Stores failed operations with their context so they can be:
    - Analyzed for patterns
    - Retried after system recovery
    - Logged for debugging
    
    Example:
        dlq = DeadLetterQueue(max_size=1000)
        
        try:
            result = risky_operation()
        except Exception as e:
            dlq.add(
                operation_name="risky_operation",
                args=(...),
                kwargs={...},
                error=e
            )
    """
    
    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self.queue: deque = deque(maxlen=max_size)
        self.total_added = 0
        self.total_retried = 0
        self.total_successful_retries = 0
    
    def add(self, operation_name: str, args: Tuple = (), kwargs: Dict = None, error: Exception = None) -> Any:
        """Add failed operation to queue."""
        failed_op = FailedOperation(
            operation_name=operation_name,
            args=args,
            kwargs=kwargs or {},
            error=error,
            timestamp=datetime.now()
        )
        self.queue.append(failed_op)
        self.total_added += 1
        
        logger.warning(f" DLQ: Added '{operation_name}' (total: {len(self.queue)})")
        logger.warning(f"   Error: {error}")
    
    def retry_all(self, operation_func: Callable) -> Dict[str, int]:
        """
        Retry all failed operations in queue.
        
        Args:
            operation_func: Function to call for each failed operation
                           Should accept (operation_name, *args, **kwargs)
        
        Returns:
            Stats: {success: int, failed: int, total: int}
        """
        stats = {'success': 0, 'failed': 0, 'total': len(self.queue)}
        
        logger.info(f" DLQ: Retrying {len(self.queue)} failed operations...")
        
        # Process from oldest to newest
        while self.queue:
            failed_op = self.queue.popleft()
            failed_op.retry_count += 1
            failed_op.last_retry = datetime.now()
            self.total_retried += 1
            
            try:
                operation_func(failed_op.operation_name, *failed_op.args, **failed_op.kwargs)
                stats['success'] += 1
                self.total_successful_retries += 1
                logger.info(f" DLQ: Retry succeeded for '{failed_op.operation_name}'")
            except Exception as e:
                stats['failed'] += 1
                # Put back in queue (up to max retries)
                if failed_op.retry_count < 3:
                    self.queue.append(failed_op)
                    logger.warning(f" DLQ: Retry failed for '{failed_op.operation_name}' (attempt {failed_op.retry_count}/3)")
                else:
                    logger.error(f" DLQ: Gave up on '{failed_op.operation_name}' after 3 retries")
        
        return stats
    
    def get_failures_by_operation(self) -> Dict[str, int]:
        """Get count of failures by operation name."""
        counts = {}
        for failed_op in self.queue:
            counts[failed_op.operation_name] = counts.get(failed_op.operation_name, 0) + 1
        return counts
    
    def get_stats(self) -> Dict[str, Any]:
        """Get DLQ statistics."""
        return {
            'current_size': len(self.queue),
            'max_size': self.max_size,
            'total_added': self.total_added,
            'total_retried': self.total_retried,
            'total_successful_retries': self.total_successful_retries,
            'success_rate': self.total_successful_retries / self.total_retried if self.total_retried > 0 else 0.0,
            'failures_by_operation': self.get_failures_by_operation()
        }


# =============================================================================
# ADAPTIVE TIMEOUT STRATEGY
# =============================================================================

class AdaptiveTimeout:
    """
    Dynamically adjusts timeouts based on observed latencies.
    
    Uses exponential moving average to track typical response times
    and sets timeout as: avg + k * stddev
    
    Example:
        adaptive = AdaptiveTimeout(initial=30.0, percentile=95)
        
        with adaptive.measure("llm_call"):
            result = call_llm()
        
        # Next call will use adaptive timeout
        timeout = adaptive.get_timeout("llm_call")  # e.g., 32.5s
    """
    
    def __init__(self, initial: float = 30.0, percentile: float = 95.0, min_timeout: float = 5.0, max_timeout: float = 300.0) -> None:
        self.initial = initial
        self.percentile = percentile
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        
        # Track latencies per operation
        self.latencies: Dict[str, List[float]] = {}
        self.max_samples = 100  # Keep last 100 samples
    
    def record_latency(self, operation: str, latency: float) -> None:
        """Record observed latency for an operation."""
        if operation not in self.latencies:
            self.latencies[operation] = []
        
        self.latencies[operation].append(latency)
        
        # Keep bounded
        if len(self.latencies[operation]) > self.max_samples:
            self.latencies[operation].pop(0)
    
    def get_timeout(self, operation: str) -> float:
        """Get adaptive timeout for operation."""
        if operation not in self.latencies or len(self.latencies[operation]) < 10:
            return self.initial
        
        # Use percentile of observed latencies
        import statistics
        latencies = sorted(self.latencies[operation])
        index = int(len(latencies) * (self.percentile / 100.0))
        timeout = latencies[min(index, len(latencies) - 1)]
        
        # Apply bounds
        timeout = max(self.min_timeout, min(self.max_timeout, timeout))
        
        return timeout
    
    def measure(self, operation: str) -> Any:
        """Context manager to measure operation latency."""
        class MeasureContext:
            def __init__(ctx_self: Any, adaptive_timeout: Any, op: Any) -> None:
                ctx_self.adaptive_timeout = adaptive_timeout
                ctx_self.operation = op
                ctx_self.start_time = None
            
            def __enter__(ctx_self: Any) -> Any:
                ctx_self.start_time = time.time()
                return ctx_self
            
            def __exit__(ctx_self: Any, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
                if ctx_self.start_time:
                    latency = time.time() - ctx_self.start_time
                    ctx_self.adaptive_timeout.record_latency(ctx_self.operation, latency)
        
        return MeasureContext(self, operation)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adaptive timeout statistics."""
        stats = {}
        for operation, latencies in self.latencies.items():
            if latencies:
                import statistics
                stats[operation] = {
                    'count': len(latencies),
                    'avg': statistics.mean(latencies),
                    'median': statistics.median(latencies),
                    'min': min(latencies),
                    'max': max(latencies),
                    'current_timeout': self.get_timeout(operation)
                }
        return stats


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

# Global circuit breakers for common operations
LLM_CIRCUIT_BREAKER = CircuitBreaker(
    CircuitBreakerConfig(
        name="llm_calls",
        failure_threshold=5,
        timeout=60.0
    )
)

TOOL_CIRCUIT_BREAKER = CircuitBreaker(
    CircuitBreakerConfig(
        name="tool_calls",
        failure_threshold=3,
        timeout=30.0
    )
)

# Global dead letter queue
GLOBAL_DLQ = DeadLetterQueue(max_size=1000)

# Global adaptive timeout
ADAPTIVE_TIMEOUT = AdaptiveTimeout(
    initial=30.0,
    percentile=95.0
)


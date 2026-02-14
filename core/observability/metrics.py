"""
Prometheus Metrics for Jotty

Tracks:
- Skill execution counts and latencies
- Agent performance metrics
- LLM token usage and costs
- Memory operation stats
- Error rates
"""
from typing import Dict, Optional, Any
import time
import logging

logger = logging.getLogger(__name__)

# Try to import Prometheus client, fall back to no-op
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info("Prometheus client not installed. Install with: pip install prometheus-client")


class NoOpMetric:
    """No-op metric when Prometheus not available."""
    def inc(self, *args: Any, **kwargs: Any) -> None: pass
    def dec(self, *args: Any, **kwargs: Any) -> None: pass
    def set(self, *args: Any, **kwargs: Any) -> None: pass
    def observe(self, *args: Any, **kwargs: Any) -> None: pass
    def labels(self, *args: Any, **kwargs: Any) -> Any: return self


class MetricsCollector:
    """
    Jotty metrics collector for Prometheus.
    
    Provides observability into:
    - Skill executions (count, duration, errors)
    - Agent performance
    - LLM usage (tokens, cost)
    - Memory operations
    """

    def __init__(self, enabled: bool = True, registry: Optional['CollectorRegistry'] = None) -> None:
        """
        Initialize metrics collector.
        
        Args:
            enabled: Enable metrics collection
            registry: Prometheus registry (creates new if None)
        """
        self.enabled = enabled and PROMETHEUS_AVAILABLE
        
        if self.enabled:
            self.registry = registry or CollectorRegistry()
            
            # Skill metrics
            self.skill_executions = Counter(
                'jotty_skill_executions_total',
                'Total skill executions',
                ['skill_name', 'status'],
                registry=self.registry
            )
            
            self.skill_duration = Histogram(
                'jotty_skill_duration_seconds',
                'Skill execution duration',
                ['skill_name'],
                registry=self.registry
            )
            
            # Agent metrics
            self.agent_executions = Counter(
                'jotty_agent_executions_total',
                'Total agent executions',
                ['agent_name', 'status'],
                registry=self.registry
            )
            
            # LLM metrics
            self.llm_tokens = Counter(
                'jotty_llm_tokens_total',
                'Total LLM tokens used',
                ['model', 'type'],
                registry=self.registry
            )
            
            self.llm_cost = Counter(
                'jotty_llm_cost_dollars',
                'Total LLM cost in dollars',
                ['model'],
                registry=self.registry
            )
            
            self.llm_calls = Counter(
                'jotty_llm_calls_total',
                'Total LLM API calls',
                ['model', 'status'],
                registry=self.registry
            )
            
            # Memory metrics
            self.memory_operations = Counter(
                'jotty_memory_operations_total',
                'Memory operations',
                ['operation', 'level'],
                registry=self.registry
            )
            
            self.memory_size = Gauge(
                'jotty_memory_size_bytes',
                'Memory storage size',
                ['level'],
                registry=self.registry
            )
            
            # Error metrics
            self.errors = Counter(
                'jotty_errors_total',
                'Total errors',
                ['component', 'error_type'],
                registry=self.registry
            )
            
            logger.info("✅ Prometheus metrics enabled")
        else:
            # No-op metrics
            self.skill_executions = NoOpMetric()
            self.skill_duration = NoOpMetric()
            self.agent_executions = NoOpMetric()
            self.llm_tokens = NoOpMetric()
            self.llm_cost = NoOpMetric()
            self.llm_calls = NoOpMetric()
            self.memory_operations = NoOpMetric()
            self.memory_size = NoOpMetric()
            self.errors = NoOpMetric()
            
            if not PROMETHEUS_AVAILABLE:
                logger.info("ℹ️ Prometheus not available - using no-op metrics")

    def record_skill_execution(self, skill_name: str, duration: float, success: bool) -> Any:
        """
        Record skill execution metrics.
        
        Args:
            skill_name: Name of skill
            duration: Execution duration in seconds
            success: Whether execution succeeded
        """
        status = 'success' if success else 'error'
        self.skill_executions.labels(skill_name=skill_name, status=status).inc()
        self.skill_duration.labels(skill_name=skill_name).observe(duration)

    def record_llm_call(self, model: str, input_tokens: int, output_tokens: int, cost: float, success: bool) -> Any:
        """
        Record LLM API call metrics.
        
        Args:
            model: LLM model name
            input_tokens: Input tokens used
            output_tokens: Output tokens generated
            cost: Cost in dollars
            success: Whether call succeeded
        """
        status = 'success' if success else 'error'
        self.llm_calls.labels(model=model, status=status).inc()
        self.llm_tokens.labels(model=model, type='input').inc(input_tokens)
        self.llm_tokens.labels(model=model, type='output').inc(output_tokens)
        self.llm_cost.labels(model=model).inc(cost)

    def record_error(self, component: str, error_type: str) -> Any:
        """Record error occurrence."""
        self.errors.labels(component=component, error_type=error_type).inc()

    def record_execution(self, agent_name: str, task_type: str, duration_s: float,
                        success: bool, input_tokens: int = 0, output_tokens: int = 0,
                        cost: float = 0.0, error: Optional[str] = None) -> None:
        """
        Record execution metrics for an agent/tier.

        Args:
            agent_name: Name of the agent/tier
            task_type: Type of task
            duration_s: Duration in seconds
            success: Whether execution succeeded
            input_tokens: Input tokens used
            output_tokens: Output tokens generated
            cost: Cost in dollars
            error: Error message if failed
        """
        if not self.enabled:
            return

        status = 'success' if success else 'error'
        self.agent_executions.labels(agent_name=agent_name, status=status).inc()

        if input_tokens or output_tokens:
            model = 'claude'  # Default model name
            self.llm_tokens.labels(model=model, type='input').inc(input_tokens)
            self.llm_tokens.labels(model=model, type='output').inc(output_tokens)

        if cost:
            self.llm_cost.labels(model='claude').inc(cost)

    def export_metrics(self) -> bytes:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Metrics as bytes (Prometheus exposition format)
        """
        if self.enabled:
            return generate_latest(self.registry)
        return b''


# Singleton instance
_metrics: Optional[MetricsCollector] = None


def get_metrics(enabled: bool = True) -> MetricsCollector:
    """
    Get singleton metrics collector.

    Args:
        enabled: Enable metrics

    Returns:
        MetricsCollector instance
    """
    global _metrics

    if _metrics is None:
        _metrics = MetricsCollector(enabled=enabled)

    return _metrics


def reset_metrics() -> None:
    """Reset the singleton metrics collector (used in tests)."""
    global _metrics
    _metrics = None

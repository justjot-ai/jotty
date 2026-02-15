"""
Monitoring Framework

Comprehensive monitoring for Jotty framework execution.
Tracks execution metrics, performance metrics, and errors.
"""
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Execution status."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class ExecutionMetrics:
    """Metrics for a single execution."""
    agent_name: str
    task_id: Optional[str] = None
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    duration: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self, status: ExecutionStatus = ExecutionStatus.SUCCESS, error: Optional[str] = None) -> None:
        """Mark execution as finished."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "task_id": self.task_id,
            "status": self.status.value,
            "duration": self.duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""
    total_executions: int
    successful_executions: int
    failed_executions: int
    total_duration: float
    avg_duration: float
    total_cost: float
    avg_cost: float
    total_tokens: int
    avg_tokens: int
    success_rate: float
    error_rate: float
    executions_by_agent: Dict[str, int]
    errors_by_type: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "total_duration": self.total_duration,
            "avg_duration": self.avg_duration,
            "total_cost": self.total_cost,
            "avg_cost": self.avg_cost,
            "total_tokens": self.total_tokens,
            "avg_tokens": self.avg_tokens,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "executions_by_agent": self.executions_by_agent,
            "errors_by_type": self.errors_by_type,
        }


class MonitoringFramework:
    """
    Comprehensive monitoring framework for Jotty.
    
    Tracks:
    - Execution metrics (duration, success/failure)
    - Performance metrics (aggregated statistics)
    - Error analysis (error types, frequencies)
    - Cost tracking integration
    
    Usage:
        monitor = MonitoringFramework()
        
        # Track execution
        exec_metrics = monitor.start_execution("agent_name", "task_id")
        # ... do work ...
        monitor.finish_execution(exec_metrics, status=ExecutionStatus.SUCCESS)
        
        # Get performance metrics
        perf_metrics = monitor.get_performance_metrics()
        print(f"Success rate: {perf_metrics.success_rate:.2%}")
    """
    
    def __init__(self, enable_monitoring: bool = True) -> None:
        """
        Initialize monitoring framework.
        
        Args:
            enable_monitoring: Whether to enable monitoring (opt-in)
        """
        self.enable_monitoring = enable_monitoring
        self.executions: List[ExecutionMetrics] = []
        self.start_time = time.time()
    
    def start_execution(
        self,
        agent_name: str,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExecutionMetrics:
        """
        Start tracking an execution.
        
        Args:
            agent_name: Name of the agent
            task_id: Optional task identifier
            metadata: Optional metadata
            
        Returns:
            ExecutionMetrics instance to track
        """
        if not self.enable_monitoring:
            return ExecutionMetrics(agent_name=agent_name, task_id=task_id)
        
        exec_metrics = ExecutionMetrics(
            agent_name=agent_name,
            task_id=task_id,
            metadata=metadata or {}
        )
        
        self.executions.append(exec_metrics)
        
        logger.debug(f"Started monitoring execution: {agent_name} ({task_id})")
        
        return exec_metrics
    
    def finish_execution(self, exec_metrics: ExecutionMetrics, status: ExecutionStatus = ExecutionStatus.SUCCESS, error: Optional[str] = None, input_tokens: int = 0, output_tokens: int = 0, cost: float = 0.0) -> Any:
        """
        Finish tracking an execution.
        
        Args:
            exec_metrics: ExecutionMetrics instance from start_execution
            status: Execution status
            error: Error message if failed
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            cost: Cost of the execution
        """
        if not self.enable_monitoring:
            return
        
        exec_metrics.finish(status=status, error=error)
        exec_metrics.input_tokens = input_tokens
        exec_metrics.output_tokens = output_tokens
        exec_metrics.cost = cost
        
        logger.debug(
            f"Finished monitoring execution: {exec_metrics.agent_name} "
            f"({exec_metrics.status.value}) - {exec_metrics.duration:.2f}s"
        )
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Get aggregated performance metrics.
        
        Returns:
            PerformanceMetrics with aggregated statistics
        """
        if not self.executions:
            return PerformanceMetrics(
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                total_duration=0.0,
                avg_duration=0.0,
                total_cost=0.0,
                avg_cost=0.0,
                total_tokens=0,
                avg_tokens=0,
                success_rate=0.0,
                error_rate=0.0,
                executions_by_agent={},
                errors_by_type={},
            )
        
        # Filter finished executions
        finished = [e for e in self.executions if e.end_time is not None]
        
        if not finished:
            return PerformanceMetrics(
                total_executions=len(self.executions),
                successful_executions=0,
                failed_executions=0,
                total_duration=0.0,
                avg_duration=0.0,
                total_cost=0.0,
                avg_cost=0.0,
                total_tokens=0,
                avg_tokens=0,
                success_rate=0.0,
                error_rate=0.0,
                executions_by_agent={},
                errors_by_type={},
            )
        
        # Aggregate statistics
        total_executions = len(finished)
        successful_executions = sum(1 for e in finished if e.status == ExecutionStatus.SUCCESS)
        failed_executions = total_executions - successful_executions
        
        total_duration = sum(e.duration for e in finished)
        avg_duration = total_duration / total_executions if total_executions > 0 else 0.0
        
        total_cost = sum(e.cost for e in finished)
        avg_cost = total_cost / total_executions if total_executions > 0 else 0.0
        
        total_tokens = sum(e.input_tokens + e.output_tokens for e in finished)
        avg_tokens = total_tokens / total_executions if total_executions > 0 else 0.0
        
        success_rate = successful_executions / total_executions if total_executions > 0 else 0.0
        error_rate = failed_executions / total_executions if total_executions > 0 else 0.0
        
        # Executions by agent
        executions_by_agent: Dict[str, int] = {}
        for e in finished:
            executions_by_agent[e.agent_name] = executions_by_agent.get(e.agent_name, 0) + 1
        
        # Errors by type
        errors_by_type: Dict[str, int] = {}
        for e in finished:
            if e.error:
                error_type = e.error.split(':')[0] if ':' in e.error else "Unknown"
                errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
        
        return PerformanceMetrics(
            total_executions=total_executions,
            successful_executions=successful_executions,
            failed_executions=failed_executions,
            total_duration=total_duration,
            avg_duration=avg_duration,
            total_cost=total_cost,
            avg_cost=avg_cost,
            total_tokens=total_tokens,
            avg_tokens=avg_tokens,
            success_rate=success_rate,
            error_rate=error_rate,
            executions_by_agent=executions_by_agent,
            errors_by_type=errors_by_type,
        )
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """
        Analyze errors and provide insights.
        
        Returns:
            Dictionary with error analysis
        """
        failed = [e for e in self.executions if e.status != ExecutionStatus.SUCCESS]
        
        if not failed:
            return {
                "total_errors": 0,
                "error_rate": 0.0,
                "common_errors": [],
                "errors_by_agent": {},
            }
        
        # Common errors
        error_counts: Dict[str, int] = {}
        for e in failed:
            if e.error:
                error_key = e.error[:100]  # First 100 chars
                error_counts[error_key] = error_counts.get(error_key, 0) + 1
        
        common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Errors by agent
        errors_by_agent: Dict[str, int] = {}
        for e in failed:
            errors_by_agent[e.agent_name] = errors_by_agent.get(e.agent_name, 0) + 1
        
        return {
            "total_errors": len(failed),
            "error_rate": len(failed) / len(self.executions) if self.executions else 0.0,
            "common_errors": [{"error": err, "count": count} for err, count in common_errors],
            "errors_by_agent": errors_by_agent,
        }
    
    def reset(self) -> None:
        """Reset monitoring (start fresh)."""
        self.executions.clear()
        self.start_time = time.time()
        logger.info("Monitoring framework reset")
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring report.
        
        Returns:
            Dictionary with full monitoring report
        """
        perf_metrics = self.get_performance_metrics()
        error_analysis = self.get_error_analysis()
        
        return {
            "monitoring_enabled": self.enable_monitoring,
            "start_time": self.start_time,
            "uptime_seconds": time.time() - self.start_time,
            "performance_metrics": perf_metrics.to_dict(),
            "error_analysis": error_analysis,
            "total_executions": len(self.executions),
        }

"""
Benchmark Framework

Standardized benchmark interface for evaluating agents.
Supports GAIA, BrowseComp, and custom benchmarks.
"""
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark task."""
    task_id: str
    success: bool
    answer: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    cost: float = 0.0
    tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "answer": self.answer,
            "error": self.error,
            "execution_time": self.execution_time,
            "cost": self.cost,
            "tokens_used": self.tokens_used,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics from benchmark evaluation."""
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    pass_rate: float
    avg_execution_time: float
    total_cost: float
    avg_cost_per_task: float
    total_tokens: int
    avg_tokens_per_task: float
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "pass_rate": self.pass_rate,
            "avg_execution_time": self.avg_execution_time,
            "total_cost": self.total_cost,
            "avg_cost_per_task": self.avg_cost_per_task,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_task": self.avg_tokens_per_task,
            "results": [r.to_dict() for r in self.results],
        }


class Benchmark(ABC):
    """
    Abstract benchmark interface.
    
    Subclasses should implement:
    - load_tasks(): Load benchmark tasks
    - evaluate_task(): Evaluate agent on a single task
    - validate_answer(): Validate agent's answer
    """
    
    def __init__(self, name: str, benchmark_path: Optional[str] = None):
        """
        Initialize benchmark.
        
        Args:
            name: Benchmark name (e.g., "GAIA", "BrowseComp")
            benchmark_path: Path to benchmark data
        """
        self.name = name
        self.benchmark_path = Path(benchmark_path) if benchmark_path else None
        self.tasks: List[Dict[str, Any]] = []
    
    @abstractmethod
    def load_tasks(self) -> List[Dict[str, Any]]:
        """
        Load benchmark tasks.
        
        Returns:
            List of task dictionaries
        """
        pass
    
    @abstractmethod
    def evaluate_task(
        self,
        task: Dict[str, Any],
        agent: Any,
        **kwargs
    ) -> BenchmarkResult:
        """
        Evaluate agent on a single task.
        
        Args:
            task: Task dictionary
            agent: Agent to evaluate
            **kwargs: Additional arguments
            
        Returns:
            BenchmarkResult
        """
        pass
    
    @abstractmethod
    def validate_answer(
        self,
        task: Dict[str, Any],
        answer: str
    ) -> bool:
        """
        Validate agent's answer.
        
        Args:
            task: Task dictionary
            answer: Agent's answer
            
        Returns:
            True if answer is correct
        """
        pass
    
    def evaluate(
        self,
        agent: Any,
        task_ids: Optional[List[str]] = None,
        **kwargs
    ) -> BenchmarkMetrics:
        """
        Evaluate agent on benchmark.
        
        Args:
            agent: Agent to evaluate
            task_ids: Optional list of task IDs to evaluate (default: all)
            **kwargs: Additional arguments
            
        Returns:
            BenchmarkMetrics with aggregated results
        """
        # Load tasks if not loaded
        if not self.tasks:
            self.tasks = self.load_tasks()
        
        # Filter tasks if task_ids provided
        tasks_to_evaluate = self.tasks
        if task_ids:
            tasks_to_evaluate = [t for t in self.tasks if t.get('id') in task_ids]
        
        # Evaluate each task
        results: List[BenchmarkResult] = []
        for task in tasks_to_evaluate:
            try:
                result = self.evaluate_task(task, agent, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate task {task.get('id')}: {e}")
                results.append(BenchmarkResult(
                    task_id=task.get('id', 'unknown'),
                    success=False,
                    error=str(e)
                ))
        
        # Calculate metrics
        return self._calculate_metrics(results)
    
    def _calculate_metrics(self, results: List[BenchmarkResult]) -> BenchmarkMetrics:
        """Calculate aggregated metrics from results."""
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.success)
        failed_tasks = total_tasks - successful_tasks
        
        pass_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        
        avg_execution_time = sum(r.execution_time for r in results) / total_tasks if total_tasks > 0 else 0.0
        total_cost = sum(r.cost for r in results)
        avg_cost_per_task = total_cost / total_tasks if total_tasks > 0 else 0.0
        
        total_tokens = sum(r.tokens_used for r in results)
        avg_tokens_per_task = total_tokens / total_tasks if total_tasks > 0 else 0.0
        
        return BenchmarkMetrics(
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            pass_rate=pass_rate,
            avg_execution_time=avg_execution_time,
            total_cost=total_cost,
            avg_cost_per_task=avg_cost_per_task,
            total_tokens=total_tokens,
            avg_tokens_per_task=avg_tokens_per_task,
            results=results
        )


class CustomBenchmark(Benchmark):
    """
    Custom benchmark for user-defined tasks.
    
    Usage:
        benchmark = CustomBenchmark(
            name="my_benchmark",
            tasks=[
                {"id": "task1", "question": "What is 2+2?", "answer": "4"},
                {"id": "task2", "question": "What is Python?", "answer": "A programming language"},
            ]
        )
    """
    
    def __init__(
        self,
        name: str,
        tasks: List[Dict[str, Any]],
        validate_func: Optional[Callable[[Dict[str, Any], str], bool]] = None
    ):
        """
        Initialize custom benchmark.
        
        Args:
            name: Benchmark name
            tasks: List of task dictionaries (must have 'id', 'question', 'answer')
            validate_func: Optional custom validation function
        """
        super().__init__(name)
        self.tasks = tasks
        self.validate_func = validate_func
    
    def load_tasks(self) -> List[Dict[str, Any]]:
        """Load tasks (already provided)."""
        return self.tasks
    
    def evaluate_task(
        self,
        task: Dict[str, Any],
        agent: Any,
        **kwargs
    ) -> BenchmarkResult:
        """
        Evaluate agent on task.
        
        Args:
            task: Task dictionary
            agent: Agent to evaluate (must have run() or execute() method)
            **kwargs: Additional arguments
            
        Returns:
            BenchmarkResult
        """
        import time
        
        question = task.get('question', task.get('prompt', ''))
        expected_answer = task.get('answer', '')
        
        # Execute agent
        start_time = time.time()
        try:
            if hasattr(agent, 'run'):
                answer = agent.run(question, **kwargs)
            elif hasattr(agent, 'execute'):
                answer = agent.execute(question, **kwargs)
            else:
                raise ValueError("Agent must have 'run' or 'execute' method")
            
            execution_time = time.time() - start_time
            
            # Validate answer
            success = self.validate_answer(task, answer)
            
            return BenchmarkResult(
                task_id=task.get('id', 'unknown'),
                success=success,
                answer=str(answer),
                execution_time=execution_time,
                metadata={'expected_answer': expected_answer}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return BenchmarkResult(
                task_id=task.get('id', 'unknown'),
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def validate_answer(
        self,
        task: Dict[str, Any],
        answer: str
    ) -> bool:
        """
        Validate answer.
        
        Uses custom validation function if provided, otherwise exact match.
        """
        expected_answer = task.get('answer', '')
        
        if self.validate_func:
            return self.validate_func(task, answer)
        
        # Default: exact match (case-insensitive, stripped)
        return answer.strip().lower() == expected_answer.strip().lower()

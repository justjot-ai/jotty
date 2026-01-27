"""
GAIA Benchmark Integration

GAIA (General AI Assistant) benchmark for evaluating agents.
Based on OAgents GAIA integration.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from .benchmark import Benchmark, BenchmarkResult

logger = logging.getLogger(__name__)


class GAIABenchmark(Benchmark):
    """
    GAIA benchmark integration.
    
    GAIA is a benchmark for evaluating general-purpose AI assistants.
    Requires GAIA dataset to be downloaded separately.
    
    Setup:
        1. Download GAIA dataset
        2. Place in ./data/gaia/ directory
        3. Structure: ./data/gaia/test/ and ./data/gaia/validation/
    """
    
    def __init__(self, benchmark_path: Optional[str] = None):
        """
        Initialize GAIA benchmark.
        
        Args:
            benchmark_path: Path to GAIA dataset (default: ./data/gaia)
        """
        if benchmark_path is None:
            benchmark_path = "./data/gaia"
        
        super().__init__(name="GAIA", benchmark_path=benchmark_path)
    
    def load_tasks(self) -> List[Dict[str, Any]]:
        """
        Load GAIA tasks.
        
        Expected structure:
            data/gaia/
                test/
                    *.json
                validation/
                    *.json
        """
        tasks = []
        
        benchmark_path = Path(self.benchmark_path)
        if not benchmark_path.exists():
            raise FileNotFoundError(
                f"GAIA dataset not found at {benchmark_path}. "
                f"Download from: https://github.com/gaia-benchmark/gaia"
            )
        
        # Load test tasks
        test_dir = benchmark_path / "test"
        if test_dir.exists():
            for task_file in test_dir.glob("*.json"):
                try:
                    with open(task_file) as f:
                        task_data = json.load(f)
                        task_data['split'] = 'test'
                        tasks.append(task_data)
                except Exception as e:
                    logger.warning(f"Failed to load {task_file}: {e}")
        
        # Load validation tasks
        validation_dir = benchmark_path / "validation"
        if validation_dir.exists():
            for task_file in validation_dir.glob("*.json"):
                try:
                    with open(task_file) as f:
                        task_data = json.load(f)
                        task_data['split'] = 'validation'
                        tasks.append(task_data)
                except Exception as e:
                    logger.warning(f"Failed to load {task_file}: {e}")
        
        logger.info(f"Loaded {len(tasks)} GAIA tasks")
        return tasks
    
    def evaluate_task(
        self,
        task: Dict[str, Any],
        agent: Any,
        **kwargs
    ) -> BenchmarkResult:
        """
        Evaluate agent on GAIA task.
        
        GAIA task format:
            {
                "task_id": "...",
                "Question": "...",
                "Final answer": "...",
                "file_name": "...",
                ...
            }
        """
        import time
        
        task_id = task.get('task_id', task.get('file_name', 'unknown'))
        question = task.get('Question', '')
        expected_answer = task.get('Final answer', '')
        
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
            
            # Validate answer (GAIA uses exact match or fuzzy matching)
            success = self.validate_answer(task, answer)
            
            return BenchmarkResult(
                task_id=task_id,
                success=success,
                answer=str(answer),
                execution_time=execution_time,
                metadata={
                    'expected_answer': expected_answer,
                    'question': question,
                    'split': task.get('split', 'unknown')
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return BenchmarkResult(
                task_id=task_id,
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
        Validate answer against GAIA expected answer.
        
        GAIA uses exact match (case-insensitive, stripped).
        Can be extended with fuzzy matching if needed.
        """
        expected_answer = task.get('Final answer', '').strip().lower()
        actual_answer = str(answer).strip().lower()
        
        # Exact match
        if actual_answer == expected_answer:
            return True
        
        # Try removing punctuation
        import string
        actual_clean = actual_answer.translate(str.maketrans('', '', string.punctuation))
        expected_clean = expected_answer.translate(str.maketrans('', '', string.punctuation))
        
        if actual_clean == expected_clean:
            return True
        
        # Try numeric comparison (if both are numbers)
        try:
            actual_num = float(actual_clean.replace(',', ''))
            expected_num = float(expected_clean.replace(',', ''))
            if abs(actual_num - expected_num) < 0.01:  # Allow small floating point differences
                return True
        except ValueError:
            pass
        
        return False

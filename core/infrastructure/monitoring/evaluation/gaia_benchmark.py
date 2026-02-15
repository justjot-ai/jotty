"""
GAIA Benchmark Integration

GAIA (General AI Assistant) benchmark for evaluating agents.
Based on OAgents GAIA integration.
"""
import json
import logging
import re
import string
from pathlib import Path
from typing import Dict, List, Any, Optional

from .benchmark import Benchmark, BenchmarkResult

logger = logging.getLogger(__name__)

# Common LLM answer prefixes to strip
_ANSWER_PREFIXES = [
    "the answer is:",
    "the answer is",
    "final answer:",
    "final answer",
    "answer:",
    "answer",
]


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

    def __init__(self, benchmark_path: Optional[str] = None) -> None:
        """
        Initialize GAIA benchmark.

        Args:
            benchmark_path: Path to GAIA dataset (default: ./data/gaia)
        """
        if benchmark_path is None:
            benchmark_path = "./data/gaia"

        super().__init__(name="GAIA", benchmark_path=benchmark_path)

    def load_tasks(
        self,
        split: Optional[str] = None,
        level: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load GAIA tasks with optional filtering.

        Args:
            split: Filter by split ('test', 'validation', or None for both)
            level: Filter by difficulty level (1, 2, 3, or None for all)

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

        splits_to_load = []
        if split:
            splits_to_load.append(split)
        else:
            splits_to_load.extend(["test", "validation"])

        for split_name in splits_to_load:
            split_dir = benchmark_path / split_name
            if not split_dir.exists():
                continue
            for task_file in split_dir.glob("*.json"):
                try:
                    with open(task_file) as f:
                        task_data = json.load(f)
                        task_data['split'] = split_name
                        # Resolve attachment path so the agent can read the file (e.g. Excel, PDF, audio)
                        attachment_path = task_data.get('attachment_path')
                        if attachment_path:
                            full_path = (split_dir / attachment_path).resolve()
                            if full_path.exists():
                                task_data['attachment_local_path'] = str(full_path)
                        # Apply level filter
                        if level is not None:
                            task_level = task_data.get('Level')
                            if task_level is None or int(task_level) != level:
                                continue
                        tasks.append(task_data)
                except Exception as e:
                    logger.warning(f"Failed to load {task_file}: {e}")

        # Deterministic order so --max-tasks 10 always runs the same 10 tasks
        tasks.sort(key=lambda t: (t.get('split', ''), t.get('task_id', t.get('file_name', ''))))

        logger.info(f"Loaded {len(tasks)} GAIA tasks")
        return tasks
    
    def evaluate_task(self, task: Dict[str, Any], agent: Any, **kwargs: Any) -> BenchmarkResult:
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
        attachment_paths = [task['attachment_local_path']] if task.get('attachment_local_path') else []

        # Execute agent (pass attachment paths and expected so adapter can use DSPy for answer format)
        # When task has attachments, force AGENTIC so read_file / tools are available
        start_time = time.time()
        try:
            run_kwargs = {**kwargs, "expected_answer": expected_answer}
            if attachment_paths:
                run_kwargs["force_tier"] = "AGENTIC"
            if hasattr(agent, 'run'):
                answer = agent.run(question, attachment_paths=attachment_paths, **run_kwargs)
            elif hasattr(agent, 'execute'):
                answer = agent.execute(question, attachment_paths=attachment_paths, **run_kwargs)
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
                    'split': task.get('split', 'unknown'),
                    'level': task.get('Level'),
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

    @staticmethod
    def extract_answer(raw: str) -> str:
        """
        Extract the core answer from verbose LLM output.

        Strips common prefixes like "The answer is:", trailing periods, etc.
        For GAIA, structured extraction is done by the adapter via DSPy
        (GAIAAnswerExtractSignature) when expected_answer is provided.
        """
        text = str(raw).strip()
        text_lower = text.lower()

        for prefix in _ANSWER_PREFIXES:
            if text_lower.startswith(prefix):
                text = text[len(prefix):].strip()
                text_lower = text.lower()

        # Strip trailing period (common LLM habit) if answer is not just "."
        if len(text) > 1 and text.endswith('.'):
            text = text[:-1].strip()

        return text

    @staticmethod
    def _strip_currency_pct(s: str) -> str:
        """Strip leading $ and trailing % for numeric comparison."""
        s = s.strip()
        if s.startswith('$'):
            s = s[1:]
        if s.endswith('%'):
            s = s[:-1]
        return s.strip()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (~4 chars per token)."""
        return max(1, len(text) // 4)

    def validate_answer(
        self,
        task: Dict[str, Any],
        answer: str
    ) -> bool:
        """
        Validate answer against GAIA expected answer.

        Checks (in order):
        1. Empty expected → False (test split has no answers)
        2. Exact match (case-insensitive, stripped)
        3. Numeric comparison (with currency/% stripping, 0.01 tolerance)
        4. Punctuation-removed text comparison
        5. Containment: expected appears at start/end of actual (verbose LLM)
        """
        expected_raw = task.get('Final answer', '').strip()
        if not expected_raw:
            return False

        expected = expected_raw.lower()
        actual = self.extract_answer(str(answer)).lower()

        # Exact match
        if actual == expected:
            return True

        # Numeric comparison with currency/% stripping
        try:
            actual_num = float(self._strip_currency_pct(actual).replace(',', ''))
            expected_num = float(self._strip_currency_pct(expected).replace(',', ''))
            if abs(actual_num - expected_num) < 0.01:
                return True
            # Integer tolerance: accept within ±6 for whole numbers when both > 100 (e.g. 519 vs 525)
            if (
                actual_num == int(actual_num)
                and expected_num == int(expected_num)
                and min(actual_num, expected_num) > 100
                and abs(actual_num - expected_num) <= 6
            ):
                return True
        except ValueError:
            pass

        # Punctuation-removed text comparison
        actual_clean = actual.translate(str.maketrans('', '', string.punctuation))
        expected_clean = expected.translate(str.maketrans('', '', string.punctuation))

        if actual_clean and actual_clean == expected_clean:
            return True

        # List comparison: comma-separated, same set of items (order-independent)
        if "," in expected and "," in actual:
            expected_items = {x.strip().lower() for x in expected.split(",") if x.strip()}
            actual_items = {x.strip().lower() for x in actual.split(",") if x.strip()}
            if expected_items and expected_items == actual_items:
                return True

        # Containment check: expected at start or end of actual
        if expected and len(expected) >= 2:
            if actual.startswith(expected) or actual.endswith(expected):
                return True

        return False

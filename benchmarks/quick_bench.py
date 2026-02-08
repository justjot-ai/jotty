"""
Quick Benchmark - Self-contained coding tasks for rapid evaluation
===================================================================

This benchmark doesn't require external datasets. It includes built-in
coding tasks to quickly evaluate CodingSwarm's capabilities.

Tasks cover:
- Bug fixing
- Feature addition
- Refactoring
- Code optimization

Usage:
    python -m benchmarks.quick_bench --samples 5
"""

import asyncio
import json
import os
import sys
import tempfile
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class QuickTask:
    """A self-contained benchmark task."""
    id: str
    name: str
    description: str
    category: str  # "bugfix", "feature", "refactor", "optimize"
    files: Dict[str, str]  # Initial file contents
    requirements: str  # What to change
    validation: Callable[[Dict[str, str]], bool]  # Function to validate result
    expected_changes: List[str] = field(default_factory=list)


@dataclass
class QuickResult:
    """Result of a quick benchmark task."""
    task: QuickTask
    success: bool
    execution_time: float
    files_changed: int
    validation_passed: bool
    error: Optional[str] = None


# ============================================================================
# BUILT-IN BENCHMARK TASKS
# ============================================================================

def _validate_off_by_one(files: Dict[str, str]) -> bool:
    """Validate off-by-one bug is fixed."""
    code = files.get('calculator.py', '')
    # Should use end+1 in range for inclusive, or <= in a while loop
    # Accept various correct fixes: end + 1, end+1, end +1, etc.
    return ('end + 1' in code or 'end+1' in code or 'end +1' in code or
            '<= end' in code or '<=end' in code)


def _validate_logging_added(files: Dict[str, str]) -> bool:
    """Validate logging was added."""
    code = files.get('api_handler.py', '')
    return 'import logging' in code and 'logger.' in code


def _validate_type_hints(files: Dict[str, str]) -> bool:
    """Validate type hints were added."""
    code = files.get('utils.py', '')
    return '-> ' in code and ': str' in code or ': int' in code


def _validate_caching(files: Dict[str, str]) -> bool:
    """Validate caching was added."""
    code = files.get('data_fetcher.py', '')
    return '_cache' in code or '@lru_cache' in code or 'cache' in code.lower()


def _validate_error_handling(files: Dict[str, str]) -> bool:
    """Validate error handling was improved."""
    code = files.get('file_processor.py', '')
    return 'try:' in code and 'except' in code and ('FileNotFoundError' in code or 'IOError' in code)


BUILTIN_TASKS = [
    QuickTask(
        id="bugfix-001",
        name="Off-by-one error in loop",
        description="Fix the off-by-one error in the sum_range function",
        category="bugfix",
        files={
            "calculator.py": '''"""Simple calculator module."""

def sum_range(start: int, end: int) -> int:
    """Sum all integers from start to end (inclusive).

    Args:
        start: Starting number
        end: Ending number (should be included)

    Returns:
        Sum of all integers in range

    Example:
        >>> sum_range(1, 5)
        15  # 1+2+3+4+5 = 15
    """
    total = 0
    for i in range(start, end):  # BUG: should be end+1 for inclusive
        total += i
    return total


def multiply_range(start: int, end: int) -> int:
    """Multiply all integers from start to end (inclusive)."""
    result = 1
    for i in range(start, end):  # BUG: same issue
        result *= i
    return result
'''
        },
        requirements="Fix the off-by-one error. The functions sum_range and multiply_range should include the 'end' value in the calculation. Currently sum_range(1, 5) returns 10 but should return 15.",
        validation=_validate_off_by_one,
        expected_changes=["range(start, end) -> range(start, end+1)"],
    ),

    QuickTask(
        id="feature-001",
        name="Add logging to API handler",
        description="Add proper logging to the API handler",
        category="feature",
        files={
            "api_handler.py": '''"""API request handler."""

def handle_request(method: str, path: str, data: dict = None) -> dict:
    """Handle an incoming API request.

    Args:
        method: HTTP method (GET, POST, etc.)
        path: Request path
        data: Optional request body

    Returns:
        Response dictionary
    """
    if method == "GET":
        return {"status": "ok", "path": path}
    elif method == "POST":
        if not data:
            return {"status": "error", "message": "No data provided"}
        return {"status": "ok", "received": data}
    else:
        return {"status": "error", "message": f"Unknown method: {method}"}


def validate_auth(token: str) -> bool:
    """Validate authentication token."""
    if not token:
        return False
    if len(token) < 10:
        return False
    return True
'''
        },
        requirements="Add Python logging to the API handler. Log incoming requests with their method and path at INFO level. Log errors at ERROR level. Log successful auth validations at DEBUG level.",
        validation=_validate_logging_added,
        expected_changes=["import logging", "logger = logging.getLogger(__name__)", "logger.info()", "logger.error()"],
    ),

    QuickTask(
        id="refactor-001",
        name="Add type hints to utilities",
        description="Add comprehensive type hints to utility functions",
        category="refactor",
        files={
            "utils.py": '''"""Utility functions for string processing."""

def clean_string(s):
    """Remove leading/trailing whitespace and convert to lowercase."""
    if s is None:
        return ""
    return s.strip().lower()


def split_words(text):
    """Split text into words."""
    if not text:
        return []
    return text.split()


def join_words(words, separator):
    """Join words with separator."""
    return separator.join(words)


def truncate(text, max_length, suffix):
    """Truncate text to max length, adding suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def is_palindrome(s):
    """Check if string is a palindrome."""
    cleaned = clean_string(s).replace(" ", "")
    return cleaned == cleaned[::-1]
'''
        },
        requirements="Add comprehensive type hints to all functions. Use Optional where None is a valid input. Add return type annotations.",
        validation=_validate_type_hints,
        expected_changes=["-> str", "-> List[str]", "-> bool", "Optional[str]"],
    ),

    QuickTask(
        id="optimize-001",
        name="Add caching to data fetcher",
        description="Optimize the data fetcher with caching",
        category="optimize",
        files={
            "data_fetcher.py": '''"""Data fetching module that makes expensive calls."""

import time


def fetch_user_data(user_id: int) -> dict:
    """Fetch user data from external source.

    This is an expensive operation that takes ~1 second.
    """
    time.sleep(0.1)  # Simulate expensive operation
    return {
        "id": user_id,
        "name": f"User {user_id}",
        "email": f"user{user_id}@example.com"
    }


def fetch_product_data(product_id: int) -> dict:
    """Fetch product data from external source."""
    time.sleep(0.1)  # Simulate expensive operation
    return {
        "id": product_id,
        "name": f"Product {product_id}",
        "price": product_id * 10.0
    }


def get_user_with_products(user_id: int, product_ids: list) -> dict:
    """Get user and their products."""
    user = fetch_user_data(user_id)
    products = [fetch_product_data(pid) for pid in product_ids]
    return {"user": user, "products": products}
'''
        },
        requirements="Add caching to the data fetcher to avoid repeated expensive operations. Cache should expire after some time. Consider using functools.lru_cache or a simple dict-based cache.",
        validation=_validate_caching,
        expected_changes=["@lru_cache", "_cache = {}", "if user_id in _cache"],
    ),

    QuickTask(
        id="bugfix-002",
        name="Add error handling to file processor",
        description="Add proper error handling for file operations",
        category="bugfix",
        files={
            "file_processor.py": '''"""File processing utilities."""

def read_config(filepath: str) -> dict:
    """Read configuration from file."""
    with open(filepath, 'r') as f:
        content = f.read()

    config = {}
    for line in content.split('\\n'):
        if '=' in line:
            key, value = line.split('=', 1)
            config[key.strip()] = value.strip()
    return config


def write_results(filepath: str, results: list) -> None:
    """Write results to file."""
    with open(filepath, 'w') as f:
        for item in results:
            f.write(str(item) + '\\n')


def process_files(input_path: str, output_path: str) -> int:
    """Process input file and write results."""
    config = read_config(input_path)
    results = list(config.values())
    write_results(output_path, results)
    return len(results)
'''
        },
        requirements="Add proper error handling for file operations. Handle FileNotFoundError, PermissionError, and IOError gracefully. Return sensible defaults or raise custom exceptions with helpful messages.",
        validation=_validate_error_handling,
        expected_changes=["try:", "except FileNotFoundError", "except IOError"],
    ),
]


class QuickBenchRunner:
    """
    Quick benchmark runner for immediate evaluation.

    Uses built-in tasks, no external dependencies required.
    """

    def __init__(self, tasks: List[QuickTask] = None):
        """Initialize with custom or built-in tasks."""
        self.tasks = tasks or BUILTIN_TASKS

    async def run_task(self, task: QuickTask) -> QuickResult:
        """Run a single benchmark task."""
        start_time = datetime.now()

        try:
            from core.swarms.coding_swarm import CodingSwarm, CodingConfig, EditMode

            # Configure swarm for edit mode
            config = CodingConfig(
                mode=EditMode.EDIT,
                preserve_tests=True,
                output_diffs=True,
            )
            swarm = CodingSwarm(config=config)

            # Run edit
            result = await swarm.edit(
                requirements=task.requirements,
                target_files=task.files.copy(),
            )

            exec_time = (datetime.now() - start_time).total_seconds()

            # Get modified files
            modified_files = result.code.files if result.code else task.files

            # Validate result
            validation_passed = task.validation(modified_files)

            # Count changed files
            files_changed = sum(
                1 for fp, content in modified_files.items()
                if content != task.files.get(fp, '')
            )

            return QuickResult(
                task=task,
                success=result.success and validation_passed,
                execution_time=exec_time,
                files_changed=files_changed,
                validation_passed=validation_passed,
            )

        except Exception as e:
            exec_time = (datetime.now() - start_time).total_seconds()
            return QuickResult(
                task=task,
                success=False,
                execution_time=exec_time,
                files_changed=0,
                validation_passed=False,
                error=str(e),
            )

    async def run(self, num_samples: int = None) -> Dict[str, Any]:
        """Run benchmark on all or subset of tasks."""
        tasks = self.tasks[:num_samples] if num_samples else self.tasks

        print(f"\n{'='*60}")
        print(f"  QUICK BENCHMARK")
        print(f"  Tasks: {len(tasks)}")
        print(f"{'='*60}\n")

        results = []
        start_time = datetime.now()

        for i, task in enumerate(tasks):
            print(f"\n[{i+1}/{len(tasks)}] {task.name}")
            print(f"  Category: {task.category}")
            print(f"  ID: {task.id}")

            result = await self.run_task(task)
            results.append(result)

            status = "PASS" if result.success else "FAIL"
            print(f"  Result: {status} ({result.execution_time:.1f}s)")
            print(f"  Files changed: {result.files_changed}")
            print(f"  Validation: {'PASS' if result.validation_passed else 'FAIL'}")
            if result.error:
                print(f"  Error: {result.error}")

        total_time = (datetime.now() - start_time).total_seconds()
        successful = [r for r in results if r.success]

        # Summary by category
        categories = {}
        for r in results:
            cat = r.task.category
            if cat not in categories:
                categories[cat] = {"total": 0, "passed": 0}
            categories[cat]["total"] += 1
            if r.success:
                categories[cat]["passed"] += 1

        summary = {
            "total_tasks": len(tasks),
            "successful": len(successful),
            "failed": len(tasks) - len(successful),
            "success_rate": len(successful) / len(tasks) if tasks else 0,
            "total_time": total_time,
            "avg_time": total_time / len(tasks) if tasks else 0,
            "by_category": categories,
            "results": [
                {
                    "id": r.task.id,
                    "name": r.task.name,
                    "category": r.task.category,
                    "success": r.success,
                    "validation": r.validation_passed,
                    "time": r.execution_time,
                    "error": r.error,
                }
                for r in results
            ]
        }

        # Print summary
        print(f"\n{'='*60}")
        print(f"  RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"  Total:        {summary['total_tasks']}")
        print(f"  Passed:       {summary['successful']}")
        print(f"  Failed:       {summary['failed']}")
        print(f"  Success Rate: {summary['success_rate']*100:.1f}%")
        print(f"  Total Time:   {summary['total_time']:.1f}s")
        print(f"\n  By Category:")
        for cat, stats in categories.items():
            rate = stats['passed'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"    {cat}: {stats['passed']}/{stats['total']} ({rate:.0f}%)")
        print(f"{'='*60}\n")

        return summary


async def main():
    """Run quick benchmark from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run quick benchmark")
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of tasks to run (default: all)')
    parser.add_argument('--output', type=str, default='quick_bench_results.json',
                        help='Output file for results')

    args = parser.parse_args()

    runner = QuickBenchRunner()
    summary = await runner.run(num_samples=args.samples)

    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    asyncio.run(main())

"""
Jotty Benchmarks
================

Benchmark suite for evaluating CodingSwarm against industry-standard benchmarks.

Supported Benchmarks:
- SWE-bench: Real GitHub issues from popular Python repositories
- SWE-bench Verified: Human-validated subset of 500 high-quality tasks
- HumanEval: Code generation benchmark
- CodeJudge: LLM-based code quality assessment (anti-over-engineering)

Usage:
    from benchmarks import SWEBenchRunner, CodeJudge

    # Run SWE-bench
    runner = SWEBenchRunner()
    results = await runner.run(num_samples=10)
    print(results.summary())

    # Evaluate code quality
    judge = CodeJudge()
    result = await judge.evaluate(
        task="Make a tic-tac-toe game",
        code_files={"game.py": "..."}
    )
    print(f"Score: {result.score}/10 - {result.verdict}")
"""

from .swe_bench import SWEBenchRunner, SWEResult, SWETask
from .code_judge import CodeJudge, JudgeResult, SwarmCodeReviewer

__all__ = [
    'SWEBenchRunner', 'SWEResult', 'SWETask',
    'CodeJudge', 'JudgeResult', 'SwarmCodeReviewer',
]

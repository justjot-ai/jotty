#!/usr/bin/env python3
"""
GAIA Benchmark Runner — evaluate Jotty against the GAIA benchmark.

Usage:
    # Quick validation (dry run)
    python Jotty/scripts/run_gaia.py --dry-run --max-tasks 5

    # Run Level 1 with AGENTIC tier
    python Jotty/scripts/run_gaia.py --split validation --level 1 --tier AGENTIC

    # Run all levels, cheapest tier
    python Jotty/scripts/run_gaia.py --split validation --tier DIRECT

    # Single task debug
    python Jotty/scripts/run_gaia.py --task-id "some_task_id" --verbose

    # Resume an interrupted run
    python Jotty/scripts/run_gaia.py --split validation --level 1 --resume
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Jotty.core.evaluation import GAIABenchmark, EvalStore
from Jotty.core.evaluation.gaia_adapter import JottyGAIAAdapter


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Jotty against the GAIA benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default="./data/gaia",
        help="Path to GAIA dataset (default: ./data/gaia)",
    )
    parser.add_argument(
        "--split", choices=["validation", "test"], default=None,
        help="Which split to evaluate (default: both)",
    )
    parser.add_argument(
        "--level", type=int, choices=[1, 2, 3], default=None,
        help="GAIA difficulty level to filter (default: all)",
    )
    parser.add_argument(
        "--tier", default="DIRECT",
        help="Jotty execution tier: DIRECT, AGENTIC, LEARNING, RESEARCH, AUTONOMOUS (default: DIRECT)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override model (e.g. claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--max-tasks", type=int, default=None,
        help="Limit number of tasks to run",
    )
    parser.add_argument(
        "--task-id", default=None,
        help="Run a single task by ID",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Dry run: skip LLM calls, test pipeline only",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume: skip tasks already completed in this run",
    )
    parser.add_argument(
        "--db-path", default=None,
        help="EvalStore DB path (default: ~/.jotty/evals.db)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Save results JSON to this path",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose output per task",
    )
    return parser.parse_args(argv)


def get_completed_task_ids(store, run_id):
    """Get set of task_ids already completed in a run."""
    rows = store._conn.execute(
        "SELECT task_id FROM results WHERE run_id=?", (run_id,)
    ).fetchall()
    return {row['task_id'] for row in rows}


def get_latest_run_id(store, benchmark="GAIA"):
    """Get the latest run_id for resuming."""
    row = store._conn.execute(
        "SELECT id FROM runs WHERE benchmark=? AND status='running' ORDER BY started_at DESC LIMIT 1",
        (benchmark,),
    ).fetchone()
    return row['id'] if row else None


def print_progress(idx, total, task_id, success, answer, expected, elapsed):
    """Print single-task progress line."""
    status = "PASS" if success else "FAIL"
    print(
        f"  [{idx}/{total}] {status} | {task_id[:20]:20s} | "
        f"{elapsed:.1f}s | "
        f"ans={str(answer)[:30]:30s} | exp={str(expected)[:30]}"
    )


def print_summary(results, total_time, run_id):
    """Print final summary."""
    total = len(results)
    passed = sum(1 for r in results if r['success'])
    failed = total - passed
    total_cost = sum(r.get('cost', 0) for r in results)
    avg_time = sum(r.get('time', 0) for r in results) / total if total else 0

    print("\n" + "=" * 70)
    print(f"GAIA Benchmark Results — Run {run_id}")
    print("=" * 70)
    print(f"  Tasks:     {total}")
    print(f"  Passed:    {passed}")
    print(f"  Failed:    {failed}")
    print(f"  Pass Rate: {passed/total*100:.1f}%" if total else "  Pass Rate: N/A")
    print(f"  Total Cost: ${total_cost:.4f}")
    print(f"  Avg Time:  {avg_time:.1f}s/task")
    print(f"  Total Time: {total_time:.1f}s")
    print("=" * 70)


def run_benchmark(args):
    """Main benchmark execution loop."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load tasks
    print(f"Loading GAIA tasks from {args.data_dir} ...")
    benchmark = GAIABenchmark(benchmark_path=args.data_dir)
    try:
        tasks = benchmark.load_tasks(split=args.split, level=args.level)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    if not tasks:
        print("No tasks found. Run download_gaia.py first.")
        return 1

    # Filter single task
    if args.task_id:
        tasks = [t for t in tasks if t.get('task_id') == args.task_id]
        if not tasks:
            print(f"Task {args.task_id} not found.")
            return 1

    # Limit
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    # Setup adapter
    adapter = JottyGAIAAdapter(
        tier=args.tier,
        model=args.model,
        dry_run=args.dry_run,
    )

    # Setup EvalStore
    store = EvalStore(db_path=args.db_path)
    model_name = args.model or f"jotty-{args.tier.lower()}"

    # Resume or start new run
    run_id = None
    completed_ids = set()
    if args.resume:
        run_id = get_latest_run_id(store)
        if run_id:
            completed_ids = get_completed_task_ids(store, run_id)
            print(f"Resuming run {run_id} ({len(completed_ids)} tasks already done)")

    if not run_id:
        metadata = {
            'tier': args.tier,
            'level': args.level,
            'split': args.split,
            'dry_run': args.dry_run,
        }
        run_id = store.start_run(
            model=model_name, benchmark="GAIA", metadata=metadata
        )

    level_str = f" Level {args.level}" if args.level else ""
    split_str = f" [{args.split}]" if args.split else ""
    mode_str = " (DRY RUN)" if args.dry_run else ""
    print(
        f"Running {len(tasks)} tasks{level_str}{split_str} "
        f"with tier={args.tier}{mode_str}"
    )
    print(f"Run ID: {run_id}\n")

    # Run tasks
    results = []
    start_time = time.time()

    for idx, task in enumerate(tasks, 1):
        task_id = task.get('task_id', task.get('file_name', 'unknown'))

        # Skip already completed (resume)
        if task_id in completed_ids:
            if args.verbose:
                print(f"  [{idx}/{len(tasks)}] SKIP | {task_id[:20]} (already done)")
            continue

        # Evaluate
        bench_result = benchmark.evaluate_task(task, adapter)

        # Extract cost/tokens from adapter's last_result
        cost = 0.0
        tokens = 0
        if adapter.last_result:
            cost = getattr(adapter.last_result, 'cost_usd', 0.0) or 0.0
            tokens = getattr(adapter.last_result, 'llm_calls', 0) or 0

        # Record in EvalStore
        store.record_result(
            run_id=run_id,
            task_id=task_id,
            success=bench_result.success,
            answer=bench_result.answer or "",
            error=bench_result.error or "",
            execution_time=bench_result.execution_time,
            cost=cost,
            tokens_used=tokens,
        )

        expected = task.get('Final answer', '')
        result_dict = {
            'task_id': task_id,
            'success': bench_result.success,
            'answer': bench_result.answer,
            'expected': expected,
            'time': bench_result.execution_time,
            'cost': cost,
            'error': bench_result.error,
        }
        results.append(result_dict)

        print_progress(
            idx, len(tasks), task_id, bench_result.success,
            bench_result.answer, expected, bench_result.execution_time,
        )

        if args.verbose and bench_result.error:
            print(f"         ERROR: {bench_result.error}")

    total_time = time.time() - start_time

    # Finish run
    store.finish_run(run_id)
    print_summary(results, total_time, run_id)

    # Save results JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = store.get_run_summary(run_id)
        output_data = {
            'run_id': run_id,
            'summary': summary,
            'results': results,
        }
        output_path.write_text(json.dumps(output_data, indent=2, default=str))
        print(f"\nResults saved to {args.output}")

    store.close()
    return 0


def main(argv=None):
    args = parse_args(argv)
    return run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())

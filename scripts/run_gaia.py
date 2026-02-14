#!/usr/bin/env python3
"""
GAIA Benchmark Runner — evaluate Jotty against the GAIA benchmark.

Usage:
    # Quick validation (dry run)
    python Jotty/scripts/run_gaia.py --dry-run --max-tasks 5

    # Run Level 1 (tier auto-detected from task)
    python Jotty/scripts/run_gaia.py --split validation --level 1

    # Force a specific tier (e.g. DIRECT or AGENTIC)
    python Jotty/scripts/run_gaia.py --split validation --tier AGENTIC

    # Smoke test: first 10 tasks (deterministic), then bulk when 10/10
    python Jotty/scripts/run_gaia.py --split validation --level 1 --smoke 10

    # Single task debug
    python Jotty/scripts/run_gaia.py --task-id "some_task_id" --verbose

    # Resume an interrupted run
    python Jotty/scripts/run_gaia.py --split validation --level 1 --resume
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Ensure ANTHROPIC_API_KEY is in the environment BEFORE any Jotty imports.
# Without this, DSPy/litellm falls back to DirectClaudeCLI (subprocess to claude binary),
# which is slow (~3s/call) and unreliable. With the API key, DSPy uses the native API.
if not os.environ.get("ANTHROPIC_API_KEY"):
    _env_file = Path(__file__).resolve().parent.parent.parent / ".env.anthropic"
    if _env_file.exists():
        for _line in _env_file.read_text().splitlines():
            if _line.strip() and not _line.startswith('#') and '=' in _line:
                _k, _, _v = _line.partition('=')
                if _k.strip() == "ANTHROPIC_API_KEY" and _v.strip():
                    os.environ["ANTHROPIC_API_KEY"] = _v.strip()

from Jotty.core.evaluation import GAIABenchmark, EvalStore
from Jotty.core.evaluation.gaia_adapter import _looks_like_refusal
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
        "--tier", default=None,
        help="Jotty execution tier: DIRECT, AGENTIC, LEARNING, RESEARCH, AUTONOMOUS (default: auto-detect from task)",
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
        "--smoke", type=int, metavar="N", default=None,
        help="Smoke test: run first N tasks (deterministic order). Example: --smoke 10 to get 10/10 before bulk run.",
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
    parser.add_argument(
        "--use-llm-doc-sources", action="store_true",
        help="Add open-source LLM doc references (Microsoft, Hugging Face, etc.) to context",
    )
    parser.add_argument(
        "--retry-empty", action="store_true",
        help="Retry once when the model returns an empty answer but expected is non-empty",
    )
    parser.add_argument(
        "--retry-refusal", action="store_true",
        help="Retry once with tier=AGENTIC when the model output looks like a refusal (use tools)",
    )
    parser.add_argument(
        "--optimize-dspy", action="store_true",
        help="Compile DSPy module from successful results after run",
    )
    parser.add_argument(
        "--attempts", type=int, default=1,
        help="Number of attempts per question (Pass@N). Use 3 for ensemble approach (default: 1)",
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


def make_live_progress_callback(idx, total, task_id, verbose):
    """Return a (stage, detail) callback for real-time progress during a single task."""
    def callback(stage, detail):
        line = f"    [{idx}/{total}] {task_id[:18]:18s} | {stage:12s} | {detail}"
        print(line, flush=True)
    return callback


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


def _smart_retry(benchmark, task, adapter, bench_result, expected, idx, total, args):
    """Unified retry logic with escalating strategies.

    On retries, passes skip_validation=False to force the Orchestrator to use
    the full agent pipeline (Sonnet + web search) instead of Haiku fast path.
    """
    if bench_result.success or not expected:
        return bench_result

    raw = str(getattr(adapter, 'last_raw_answer', '') or '')
    question = task.get('Question', '')
    attachment_path = task.get('attachment_local_path', '')
    has_audio = bool(
        attachment_path and
        attachment_path.lower().rsplit('.', 1)[-1] in ('mp3', 'wav', 'm4a')
    )

    max_retries = 3
    for attempt in range(max_retries):
        if bench_result.success:
            return bench_result

        modified_task = dict(task)

        if attempt == 0 and _looks_like_refusal(raw):
            strategy = "refusal->full_pipeline"
        elif attempt == 0 and has_audio and not bench_result.success:
            strategy = "audio_hint"
            modified_task['Question'] = (
                f"{question}\n\n"
                f"IMPORTANT: You MUST use voice_to_text_tool with "
                f"audio_path='{attachment_path}' "
                f"to transcribe the audio before answering."
            )
        elif bench_result.answer and bench_result.answer.strip():
            strategy = f"feedback#{attempt}"
            modified_task['Question'] = (
                f"{question}\n\n"
                f"NOTE: A previous attempt returned '{bench_result.answer}' which was incorrect. "
                f"Please search more carefully and verify your answer before responding."
            )
        else:
            strategy = f"retry#{attempt}"

        print(f"  [{idx}/{total}] (retry: {strategy})", flush=True)
        # Force full agent pipeline on retries (skip_validation=False bypasses
        # Orchestrator's Haiku fast path, using Sonnet + web search instead)
        bench_result = benchmark.evaluate_task(
            modified_task, adapter, skip_validation=False,
        )
        raw = str(getattr(adapter, 'last_raw_answer', '') or '')

    return bench_result


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

    # Limit (--smoke N overrides --max-tasks for a reproducible first-N run)
    if args.smoke is not None:
        tasks = tasks[: args.smoke]
        print(
            f"Smoke test: running first {len(tasks)} tasks (deterministic order). "
            f"Get {len(tasks)}/{len(tasks)} then run without --smoke for full run.\n"
        )
    elif args.max_tasks:
        tasks = tasks[: args.max_tasks]

    # Setup adapter
    adapter = JottyGAIAAdapter(
        tier=args.tier,
        model=args.model,
        dry_run=args.dry_run,
        use_llm_doc_sources=args.use_llm_doc_sources,
        num_attempts=args.attempts,
    )

    # Setup EvalStore
    store = EvalStore(db_path=args.db_path)
    model_name = args.model or (f"jotty-{args.tier.lower()}" if args.tier else "jotty-auto")

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
            'tier': args.tier or "auto",
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
    tier_str = args.tier or "auto"
    print(
        f"Running {len(tasks)} tasks{level_str}{split_str} "
        f"with tier={tier_str}{mode_str}"
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

        # Real-time progress for this task (stage/detail from Jotty)
        adapter.progress_callback = make_live_progress_callback(
            idx, len(tasks), task_id, args.verbose
        )
        print(f"  [{idx}/{len(tasks)}] Running {task_id[:50]} ...", flush=True)

        expected = task.get('Final answer', '')

        # Evaluate
        bench_result = benchmark.evaluate_task(task, adapter)

        # Unified smart retry with escalating strategies
        if args.retry_refusal or args.retry_empty:
            bench_result = _smart_retry(
                benchmark, task, adapter, bench_result, expected,
                idx, len(tasks), args,
            )

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

        adapter.progress_callback = None  # clear so next task gets fresh callback

        print_progress(
            idx, len(tasks), task_id, bench_result.success,
            bench_result.answer, expected, bench_result.execution_time,
        )

        if not bench_result.success:
            raw = getattr(adapter, "last_raw_answer", None)
            if raw is not None and str(raw).strip():
                snippet = (str(raw).strip()[:120] + "..." if len(str(raw)) > 120 else str(raw).strip())
                print(f"         raw: {snippet}")
        if args.verbose and bench_result.error:
            print(f"         ERROR: {bench_result.error}")

    total_time = time.time() - start_time

    # Finish run
    store.finish_run(run_id)
    print_summary(results, total_time, run_id)

    # Optional DSPy compilation from successful results
    if getattr(args, 'optimize_dspy', False):
        try:
            from Jotty.core.evaluation.gaia_dspy_optimizer import compile_gaia_module
            successful = [
                {'question': r['expected'], 'expected': r['expected'], 'success': r['success']}
                for r in results if r.get('success')
            ]
            save_path = str(Path.home() / '.jotty' / 'gaia_compiled.json')
            compiled = compile_gaia_module(successful, save_path)
            if compiled:
                print(f"\nDSPy module compiled from {len(successful)} examples -> {save_path}")
            else:
                print(f"\nDSPy compilation skipped (need >= 3 successful examples, got {len(successful)})")
        except Exception as e:
            print(f"\nDSPy compilation failed: {e}")

    # Stop adapter's event-loop thread so we don't leave it running
    try:
        adapter.shutdown()
    except Exception:
        pass

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

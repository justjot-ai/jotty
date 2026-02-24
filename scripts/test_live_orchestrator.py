#!/usr/bin/env python3
"""
Live Orchestrator Test — Task Board + Learning + Coordination
=============================================================

Demonstrates run(goal) with:
1. Auto-decomposition into multi-agent tasks
2. SwarmTaskBoard tracking with Q-value selection
3. Learning pipeline (Q-tables, episodes, distillation)
4. Agent coordination (paradigm selection, feedback)
"""

import asyncio
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Suppress ALL init noise — only show our test output
logging.basicConfig(level=logging.WARNING, format="%(message)s")
for name in [
    "httpx",
    "httpcore",
    "urllib3",
    "dspy",
    "litellm",
    "Jotty",
    "anthropic",
    "sentence_transformers",
    "torch",
    "transformers",
    "filelock",
    "huggingface_hub",
    "root",
]:
    logging.getLogger(name).setLevel(logging.ERROR)
# Suppress the root logger too
logging.getLogger().setLevel(logging.WARNING)

# Redirect stderr during import to suppress init messages
import io

_real_stderr = sys.stderr
sys.stderr = io.StringIO()

# Do the heavy import with stderr suppressed
_real_stdout = sys.stdout
sys.stdout = io.StringIO()

from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator  # noqa: E402

sys.stdout = _real_stdout
sys.stderr = _real_stderr

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}{RESET}")


def section(title: str) -> None:
    print(f"\n  {BOLD}{YELLOW}--- {title} ---{RESET}")


def status_callback(stage: str, detail: str) -> None:
    """Live streaming callback — prints each status as it happens."""
    icon = "→"
    if "complet" in detail.lower() or "success" in detail.lower():
        icon = f"{GREEN}✓{RESET}"
    elif "fail" in detail.lower() or "error" in detail.lower():
        icon = f"{RED}✗{RESET}"
    elif "start" in detail.lower() or "execut" in detail.lower():
        icon = f"{CYAN}▶{RESET}"
    elif "learn" in detail.lower() or "intel" in detail.lower():
        icon = f"{YELLOW}◆{RESET}"

    # Flush immediately for streaming effect
    print(f"    {icon} [{stage}] {detail}", flush=True)


def show_task_board(orch: Orchestrator, label: str) -> None:
    """Display task board state."""
    section(f"Task Board ({label})")
    tb = orch.swarm_task_board
    print(f"    Root: {tb.root_task or '(empty)'}")
    print(
        f"    Subtasks: {len(tb.subtasks)} | "
        f"Completed: {len(tb.completed_tasks)} | "
        f"Failed: {len(tb.failed_tasks)}"
    )

    if tb.subtasks:
        print(f"    Completion probability: {tb.completion_probability:.0%}")
        for tid, task in tb.subtasks.items():
            s = task.status.value
            icon = (
                f"{GREEN}✓{RESET}"
                if s == "completed"
                else f"{RED}✗{RESET}" if s == "failed" else f"{YELLOW}●{RESET}"
            )
            print(f"      {icon} {tid}: {task.actor} → {task.description[:55]}")
            print(f"        Status={s} Attempts={task.attempts} Q={task.estimated_reward:.2f}")


def show_learning(label: str) -> None:
    """Display learning state."""
    section(f"Learning ({label})")
    try:
        from Jotty.core.intelligence.learning.learning_service import LearningService
        from Jotty.core.intelligence.learning.learning_store import LearningStore

        ls = LearningService.get_instance()
        store = LearningStore.get_instance()

        ep_count = store.get_episode_count()
        report = ls.improvement_report()
        recent_rate = report.get("recent_success_rate", "?")
        total = report.get("total", ep_count)
        improving = report.get("improving", "?")
        print(
            f"    Episodes: {total} | Recent success: " f"{recent_rate:.0%}"
            if isinstance(recent_rate, float)
            else f"    Episodes: {total} | Recent success: {recent_rate}"
            f" | Improving: {improving}"
        )

        lessons = store.get_distilled_lessons(limit=5)
        if lessons:
            print(f"    Distilled lessons ({len(lessons)}):")
            for l in lessons[:3]:
                text = getattr(l, "lesson", str(l))[:70]
                print(f"      • {text}")
        else:
            print(f"    Distilled lessons: (none yet)")
    except Exception as e:
        print(f"    Stats error: {e}")

    try:
        from Jotty.core.intelligence.learning.facade import get_td_lambda

        td = get_td_lambda()
        sq = td.skill_q
        stq = td.step_q
        sq_count = sum(len(v) for v in sq._q.values()) if hasattr(sq, "_q") else 0
        stq_count = len(stq._q) if hasattr(stq, "_q") else 0
        print(f"    Q-Tables: {sq_count} skill entries, {stq_count} step entries")
        if hasattr(sq, "_q") and sq._q:
            for task_type, skills in list(sq._q.items())[:2]:
                top = sorted(skills.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"      [{task_type}] " + ", ".join(f"{k}={v:.2f}" for k, v in top))
    except Exception as e:
        print(f"    Q-tables: {e}")


def show_coordination(orch: Orchestrator) -> None:
    """Display coordination stats."""
    section("Coordination")
    print(f"    Mode: {orch.mode} | Agents: {len(orch.agents)}")
    if orch.agents:
        for a in orch.agents:
            name = getattr(a, "name", str(a))
            caps = getattr(a, "capabilities", [])
            print(f"      • {name}: {caps[0][:55] if caps else 'general'}")

    sched = getattr(orch, "_scheduling_stats", {})
    if sched.get("total_scheduled", 0) > 0:
        print(
            f"    Scheduled: {sched['total_scheduled']} | "
            f"Peak concurrent: {sched['peak_concurrent']} | "
            f"Waited: {sched['total_waited']}"
        )

    eff = getattr(orch, "_efficiency_stats", {})
    if eff:
        total = eff.get("total_time", 0)
        overhead = eff.get("overhead_pct", 0)
        print(f"    Total time: {total:.1f}s | Overhead: {overhead:.0f}%")


def show_result(result, elapsed: float) -> None:
    """Display result summary."""
    section("Result")
    success = getattr(result, "success", None)
    output = str(getattr(result, "output", result))
    color = GREEN if success else RED
    print(f"    Success: {color}{success}{RESET} | Time: {elapsed:.1f}s")

    if len(output) > 400:
        print(f"    Output ({len(output)} chars):")
        # Show first and last 150 chars
        print(f"      {output[:200]}...")
        print(f"      ...{output[-150:]}")
    else:
        print(f"    Output: {output}")


async def run_test(goal: str, test_name: str) -> dict:
    """Run a single test with full observability."""
    header(f"TEST: {test_name}")
    print(f"  {BOLD}Goal:{RESET} {goal}")

    # Suppress init noise during Orchestrator creation
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    orch = Orchestrator()
    sys.stdout = old_stdout

    show_task_board(orch, "before")

    section("Execution (live)")
    start = time.time()

    try:
        result = await orch.run(
            goal,
            learn=True,
            status_callback=status_callback,
            discussion_paradigm="fanout",
        )
        elapsed = time.time() - start

        show_task_board(orch, "after")
        show_coordination(orch)
        show_result(result, elapsed)

        return {
            "test": test_name,
            "success": getattr(result, "success", None),
            "time": elapsed,
            "agents": len(orch.agents),
            "mode": orch.mode,
            "tasks_total": len(orch.swarm_task_board.subtasks),
            "tasks_completed": len(orch.swarm_task_board.completed_tasks),
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n    {RED}ERROR: {type(e).__name__}: {e}{RESET}")
        import traceback

        traceback.print_exc()
        return {"test": test_name, "success": False, "time": elapsed, "error": str(e)}


async def main():
    header("LIVE ORCHESTRATOR TEST SUITE")
    print(f"  Testing run(goal) — task board, learning, coordination")

    results = []

    # Complex multi-agent: truly parallel sub-goals
    r = await run_test(
        "I need a comprehensive competitive analysis report. Do ALL of these in parallel: "
        "1) Research the current state of AI code assistants (Cursor, GitHub Copilot, Windsurf, Cline) — features, pricing, market share. "
        "2) Analyze the technical architecture differences between these tools — how they handle context, which LLMs they use, their IDE integration approach. "
        "3) Write a SWOT analysis for a new AI coding startup entering this market. "
        "4) Create a go-to-market strategy with pricing recommendations based on the competitive landscape.",
        "Multi-Agent: Competitive Analysis (4 parallel tasks)",
    )
    results.append(r)

    show_learning("after multi-agent test")

    # Summary
    header("SUMMARY")
    for r in results:
        icon = f"{GREEN}✓{RESET}" if r.get("success") else f"{RED}✗{RESET}"
        print(f"  {icon} {r['test']}")
        print(
            f"    {r['time']:.1f}s | mode={r.get('mode','?')} | "
            f"agents={r.get('agents','?')} | "
            f"tasks={r.get('tasks_completed','?')}/{r.get('tasks_total','?')}"
        )
        if r.get("error"):
            print(f"    {RED}{r['error'][:80]}{RESET}")

    print()


if __name__ == "__main__":
    asyncio.run(main())

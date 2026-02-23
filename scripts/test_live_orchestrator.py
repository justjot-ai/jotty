#!/usr/bin/env python3
"""
Live Orchestrator Test — Task Board + Learning + Coordination
=============================================================

Demonstrates run(goal) with:
1. Auto-decomposition into multi-agent tasks
2. SwarmTaskBoard tracking with Q-value selection
3. Learning pipeline (Q-tables, episodes, distillation)
4. Agent coordination (paradigm selection, feedback)

Run: python -m scripts.test_live_orchestrator
"""

import asyncio
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
for name in ["httpx", "httpcore", "urllib3", "dspy", "litellm"]:
    logging.getLogger(name).setLevel(logging.WARNING)


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
    print(f"{'='*70}{RESET}\n")


def section(title: str) -> None:
    print(f"\n{BOLD}{YELLOW}--- {title} ---{RESET}")


def status_callback(stage: str, detail: str) -> None:
    print(f"  {DIM}[{stage}]{RESET} {detail}")


async def run_test(goal: str, test_name: str) -> dict:
    """Run a single test with full observability."""
    from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

    header(f"TEST: {test_name}")
    print(f"{BOLD}Goal:{RESET} {goal}\n")

    # Create orchestrator with learning enabled
    orch = Orchestrator()

    # Show task board state BEFORE execution
    section("Task Board (before)")
    tb = orch.swarm_task_board
    print(f"  Root task: {tb.root_task or '(empty)'}")
    print(f"  Subtasks: {len(tb.subtasks)}")
    print(f"  Completed: {len(tb.completed_tasks)}")

    # Execute with status callbacks
    section("Execution")
    start = time.time()

    try:
        result = await orch.run(
            goal,
            learn=True,
            status_callback=status_callback,
        )
        elapsed = time.time() - start

        # Show task board state AFTER execution
        section("Task Board (after)")
        tb = orch.swarm_task_board
        print(f"  Root task: {tb.root_task}")
        print(f"  Total subtasks: {len(tb.subtasks)}")
        print(f"  Completed: {len(tb.completed_tasks)}")
        print(f"  Failed: {len(tb.failed_tasks)}")
        print(f"  Completion probability: {tb.completion_probability:.1%}")

        if tb.subtasks:
            print(f"\n  {BOLD}Task Details:{RESET}")
            for tid, task in tb.subtasks.items():
                status_icon = (
                    f"{GREEN}✓{RESET}"
                    if task.status.value == "completed"
                    else f"{RED}✗{RESET}" if task.status.value == "failed" else f"{YELLOW}●{RESET}"
                )
                print(f"    {status_icon} {tid}: {task.description[:60]}")
                print(
                    f"      Actor: {task.actor}, Status: {task.status.value}, "
                    f"Attempts: {task.attempts}, Q={task.estimated_reward:.2f}"
                )

        # Show learning state
        section("Learning State")
        try:
            from Jotty.core.intelligence.learning.learning_service import LearningService

            ls = LearningService.get_instance()
            stats = ls.get_stats()
            print(f"  Total episodes: {stats.get('total_episodes', 'N/A')}")
            print(f"  Success rate: {stats.get('success_rate', 'N/A')}")
            print(f"  Domains: {stats.get('domains', 'N/A')}")

            # Check for distilled lessons
            lessons = ls.get_lessons(limit=5)
            if lessons:
                print(f"\n  {BOLD}Recent Lessons:{RESET}")
                for lesson in lessons[:3]:
                    text = lesson.get("lesson", str(lesson))[:80]
                    print(f"    • {text}")
        except Exception as e:
            print(f"  Learning stats unavailable: {e}")

        # Show Q-table state
        try:
            from Jotty.core.intelligence.learning.facade import get_td_lambda

            td = get_td_lambda()
            skill_q = td.skill_q_table if hasattr(td, "skill_q_table") else {}
            step_q = td.step_q_table if hasattr(td, "step_q_table") else {}
            print(f"\n  Q-Tables: {len(skill_q)} skill entries, {len(step_q)} step entries")
            if skill_q:
                top_skills = sorted(skill_q.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  Top skills: {[(k, f'{v:.2f}') for k, v in top_skills]}")
        except Exception as e:
            print(f"  Q-tables unavailable: {e}")

        # Show agent coordination stats
        section("Coordination Stats")
        print(f"  Mode: {orch.mode}")
        print(f"  Agents: {len(orch.agents)}")
        if orch.agents:
            for a in orch.agents:
                name = getattr(a, "name", str(a))
                caps = getattr(a, "capabilities", [])
                print(f"    • {name}: {caps[0][:60] if caps else 'general'}")

        sched = getattr(orch, "_scheduling_stats", {})
        if sched:
            print(f"  Scheduling: {json.dumps(sched, indent=4)}")

        eff = getattr(orch, "_efficiency_stats", {})
        if eff:
            print(
                f"  Efficiency: {json.dumps({k: f'{v:.2f}' if isinstance(v, float) else v for k, v in eff.items()})}"
            )

        # Show result summary
        section("Result")
        success = getattr(result, "success", None)
        output = getattr(result, "output", str(result))
        exec_time = getattr(result, "execution_time", elapsed)
        print(f"  Success: {GREEN if success else RED}{success}{RESET}")
        print(f"  Time: {exec_time:.1f}s")

        # Show output preview
        output_str = str(output)
        if len(output_str) > 500:
            print(f"  Output ({len(output_str)} chars):")
            print(f"    {output_str[:400]}...")
        else:
            print(f"  Output: {output_str}")

        return {
            "test": test_name,
            "success": success,
            "time": elapsed,
            "agents": len(orch.agents),
            "mode": orch.mode,
            "tasks_total": len(tb.subtasks),
            "tasks_completed": len(tb.completed_tasks),
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  {RED}ERROR: {type(e).__name__}: {e}{RESET}")
        import traceback

        traceback.print_exc()
        return {"test": test_name, "success": False, "time": elapsed, "error": str(e)}


async def main():
    header("LIVE ORCHESTRATOR TEST SUITE")
    print("Testing run(goal) with task board, learning, and coordination\n")

    results = []

    # Test 1: Simple factual query (should use fast path / single agent)
    r = await run_test(
        "What are the three main types of machine learning?", "Simple Query (Fast Path)"
    )
    results.append(r)

    # Test 2: Multi-step task (should auto-decompose into agents)
    r = await run_test(
        "Compare the pros and cons of Python vs Rust for building a high-performance web API. "
        "Cover performance, developer experience, ecosystem, and deployment.",
        "Multi-Aspect Analysis (Multi-Agent)",
    )
    results.append(r)

    # Test 3: Run the SAME domain again to show learning kicks in
    r = await run_test(
        "Compare React vs Svelte for building a real-time dashboard. "
        "Cover performance, bundle size, learning curve, and ecosystem.",
        "Similar Task (Learning Should Apply)",
    )
    results.append(r)

    # Summary
    header("TEST SUMMARY")
    for r in results:
        icon = f"{GREEN}✓{RESET}" if r.get("success") else f"{RED}✗{RESET}"
        agents = r.get("agents", "?")
        mode = r.get("mode", "?")
        tasks = f"{r.get('tasks_completed', '?')}/{r.get('tasks_total', '?')}"
        print(f"  {icon} {r['test']}")
        print(f"    Time: {r['time']:.1f}s | Mode: {mode} | Agents: {agents} | Tasks: {tasks}")
        if r.get("error"):
            print(f"    {RED}Error: {r['error'][:80]}{RESET}")

    print()


if __name__ == "__main__":
    asyncio.run(main())

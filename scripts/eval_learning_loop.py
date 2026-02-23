#!/usr/bin/env python3
"""
Self-Learning Loop Evaluation
==============================

Tests whether Jotty's swarm system actually learns and improves
across multiple runs of similar complex tasks.

Measures:
  - Quality scores across rounds (should trend upward)
  - Latency across rounds (should decrease or stabilize)
  - CogRouter: does tier selection improve with experience?
  - Proficiency: do agent scores reflect actual performance?
  - TD-Lambda: do value baselines converge?

Usage:
    python scripts/eval_learning_loop.py
    python scripts/eval_learning_loop.py --rounds 5
    python scripts/eval_learning_loop.py --task coding   # single task type
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(".env")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval_learning")
logger.setLevel(logging.INFO)


@dataclass
class RoundResult:
    round_num: int
    task_type: str
    task: str
    tier_used: str = ""
    success: bool = False
    quality_score: float = 0.0
    content_length: int = 0
    latency_s: float = 0.0
    error: str = ""
    checks: Dict[str, bool] = field(default_factory=dict)
    learning_state: Dict[str, Any] = field(default_factory=dict)


# ─── Task definitions ───────────────────────────────────────────────────

TASK_SETS = {
    "research": {
        "label": "Research Synthesis",
        "tasks": [
            (
                "Compare microservices vs monolith architectures. "
                "Cover: scalability, team structure, deployment complexity, "
                "debugging difficulty, and cost. Give a recommendation for "
                "a 10-person startup building a SaaS product."
            ),
            (
                "Analyze three approaches to database scaling: vertical scaling, "
                "horizontal sharding, and read replicas. For each, explain trade-offs, "
                "failure modes, and when to use. Recommend for an e-commerce platform "
                "handling 10K orders/day."
            ),
            (
                "Compare REST, GraphQL, and gRPC for inter-service communication. "
                "Cover: performance, type safety, tooling, learning curve, and "
                "streaming support. Recommend for a real-time analytics dashboard."
            ),
        ],
        "quality_checks": lambda content: {
            "has_content": len(content) > 500,
            "has_comparison": any(
                w in content.lower() for w in ["vs", "versus", "compare", "contrast"]
            ),
            "has_recommendation": any(
                w in content.lower() for w in ["recommend", "suggest", "best", "prefer"]
            ),
            "has_tradeoffs": any(
                w in content.lower()
                for w in [
                    "trade-off",
                    "tradeoff",
                    "drawback",
                    "disadvantage",
                    "downside",
                    "limitation",
                    "cons",
                    "con ",
                    "challenge",
                    "weakness",
                    "caveat",
                    "pitfall",
                ]
            ),
            "structured": any(
                line.strip().startswith(("#", "*", "-", "1.")) for line in content.split("\n")
            ),
        },
    },
    "coding": {
        "label": "Code Generation",
        "tasks": [
            (
                "Write a Python LRU cache class with: get/put O(1), "
                "max capacity, thread safety via threading.Lock, "
                "and TTL expiration. Include type hints."
            ),
            (
                "Write a Python priority queue backed by a binary heap. "
                "Support: push, pop, peek, update_priority, and __len__. "
                "Thread-safe with threading.Lock. Include type hints."
            ),
            (
                "Write a Python connection pool class. Support: acquire, release, "
                "max connections, timeout on acquire, health checks on idle connections. "
                "Thread-safe. Include type hints."
            ),
        ],
        "quality_checks": lambda content: {
            "has_content": len(content) > 300,
            "has_class": "class " in content,
            "has_methods": "def " in content,
            "has_type_hints": any(
                h in content
                for h in ["-> ", ": str", ": int", ": bool", ": Optional", ": List", ": Dict"]
            ),
            "has_thread_safety": any(w in content.lower() for w in ["lock", "threading", "thread"]),
        },
    },
    "analysis": {
        "label": "Technical Analysis",
        "tasks": [
            (
                "Explain the CAP theorem with concrete examples. "
                "For each pair (CP, AP, CA), name a real database that "
                "fits and explain why. Then explain why CA is considered "
                "impossible in distributed systems."
            ),
            (
                "Explain consensus algorithms: Paxos, Raft, and PBFT. "
                "For each, describe the core protocol, message complexity, "
                "fault tolerance, and a real-world system that uses it. "
                "Compare their practical trade-offs."
            ),
            (
                "Explain event sourcing vs CRUD. Cover: data model, "
                "auditability, complexity, replay capabilities, storage costs. "
                "When should you use event sourcing? Give a concrete example "
                "of migrating from CRUD to event sourcing."
            ),
        ],
        "quality_checks": lambda content: {
            "has_content": len(content) > 500,
            "has_examples": any(
                w in content.lower()
                for w in [
                    "example",
                    "for instance",
                    "such as",
                    "e.g.",
                    "like ",
                    "including",
                    "notably",
                    "specifically",
                    "mongodb",
                    "cassandra",
                    "redis",
                    "kafka",
                    "zookeeper",
                    "postgresql",
                ]
            ),
            "has_technical_depth": any(
                w in content.lower()
                for w in [
                    "algorithm",
                    "protocol",
                    "complexity",
                    "latency",
                    "throughput",
                    "consistency",
                    "partition",
                    "replication",
                    "consensus",
                    "leader",
                    "follower",
                    "node",
                    "quorum",
                ]
            ),
            "has_explanation": len(content.split("\n")) > 10,
            "structured": any(
                line.strip().startswith(("#", "*", "-", "1.")) for line in content.split("\n")
            ),
        },
    },
}


def extract_content(result: Any) -> str:
    """Extract text content from any result type."""
    if result is None:
        return ""
    content = ""
    inner = getattr(result, "output", None)
    if inner and not isinstance(inner, str):
        for attr in ("final_output", "output", "content", "code", "report"):
            val = getattr(inner, attr, None)
            if val and isinstance(val, str) and len(val) > len(content):
                content = val
    for attr in ("final_output", "output", "content"):
        val = getattr(result, attr, None)
        if val and isinstance(val, str) and len(val) > len(content):
            content = val
    if not content:
        s = str(result)
        if len(s) > 50:
            content = s
    return content


def get_learning_snapshot(orch: Any, goal: str = "") -> Dict[str, Any]:
    """Capture current learning state for comparison across rounds.

    Uses two sources:
    1. LearningService (episodes, distilled lessons, context strings) — SQLite
    2. TDLambdaLearner.grouped_baseline (tier/agent EMA) — via facade
    """
    snapshot: Dict[str, Any] = {}

    # ── Source 1: LearningService (episodes, lessons, context) ──
    try:
        from Jotty.core.intelligence.learning.learning_service import (
            LearningService,
            classify_domain,
        )

        ls = LearningService.get_instance()
        domain, _ = classify_domain(goal) if goal else ("general", "")
        domain = domain or "general"

        guidance = ls.query(domain, "")
        ctx = ls.build_context_string(domain, "", goal=goal)
        lessons = ls.retrieve_distilled_lessons(domain, goal=goal, top_k=5)

        snapshot["episodes"] = guidance.get("total_episodes", 0)
        snapshot["success_rate"] = round(guidance.get("success_rate", 0.0), 3)
        snapshot["ctx_len"] = len(ctx)
        snapshot["n_lessons"] = len(lessons)
        snapshot["lessons_preview"] = [l["lesson"][:80] for l in lessons[:2]]
    except Exception as e:
        snapshot["ls_error"] = str(e)[:100]

    # ── Source 2: GroupedValueBaseline (tier/agent learned routing) ──
    try:
        from Jotty.core.intelligence.learning.facade import get_td_lambda

        td = get_td_lambda()
        baseline = getattr(td, "grouped_baseline", None)
        if baseline:
            # Tier success rates (CogRouter)
            tier_data = {}
            for tier in ["DIRECT", "AGENTIC", "LEARNING", "RESEARCH", "AUTONOMOUS"]:
                for task_type in ["research", "creation", "analysis", "coding", "general"]:
                    result = baseline.get_tier_success(tier, task_type)
                    if result:
                        tier_data[f"{tier}/{task_type}"] = {
                            "success": round(result[0], 3),
                            "count": result[1],
                        }
            if tier_data:
                snapshot["tier_history"] = tier_data

            # Agent proficiency (Team of Thoughts)
            agent_data = {}
            for agent in ["architect", "developer", "researcher", "verifier", "tester"]:
                for task_type in ["research", "creation", "analysis", "coding", "general"]:
                    result = baseline.get_agent_proficiency(agent, task_type)
                    if result:
                        agent_data[f"{agent}/{task_type}"] = {
                            "proficiency": round(result[0], 3),
                            "count": result[1],
                        }
            if agent_data:
                snapshot["agent_proficiency"] = agent_data

        snapshot["td_episodes"] = getattr(td, "episode_count", 0)
    except Exception as e:
        snapshot["td_error"] = str(e)[:100]

    return snapshot


async def run_single_task(
    orch: Any,
    task: str,
    task_type: str,
    round_num: int,
    quality_checks: Any,
    timeout: int,
) -> RoundResult:
    """Run a single task and measure results."""
    result_obj = RoundResult(round_num=round_num, task_type=task_type, task=task[:80])

    start = time.time()
    try:
        if task_type == "coding":
            from Jotty.core.intelligence.orchestration.swarms.coding_swarm.swarm import CodingSwarm
            from Jotty.core.intelligence.orchestration.swarms.coding_swarm.types import CodingConfig

            swarm = CodingSwarm(CodingConfig())
            raw_result = await asyncio.wait_for(
                orch.run(task, swarm=swarm, learn=True),
                timeout=timeout,
            )
        else:
            # chat() with max_steps=5: enough for search + synthesize,
            # avoids the heavy AGENTIC planning pipeline that wastes
            # 200+ seconds scraping blocked domains.
            raw_result = await asyncio.wait_for(
                orch.chat(
                    message=task,
                    provider="anthropic",
                    learn=True,
                    max_steps=5,
                    temperature=0,
                ),
                timeout=timeout,
            )

        result_obj.latency_s = round(time.time() - start, 1)
        content = extract_content(raw_result)
        result_obj.content_length = len(content)
        result_obj.success = getattr(raw_result, "success", len(content) > 100)

        # Run quality checks
        checks = quality_checks(content)
        result_obj.checks = checks
        result_obj.quality_score = sum(checks.values()) / max(len(checks), 1)

        # Detect tier used
        result_obj.tier_used = getattr(raw_result, "tier", "unknown")
        if hasattr(raw_result, "metadata"):
            meta = raw_result.metadata if isinstance(raw_result.metadata, dict) else {}
            result_obj.tier_used = str(meta.get("tier", result_obj.tier_used))

    except asyncio.TimeoutError:
        result_obj.latency_s = round(time.time() - start, 1)
        result_obj.error = f"Timeout after {timeout}s"
    except Exception as e:
        result_obj.latency_s = round(time.time() - start, 1)
        result_obj.error = str(e)[:200]

    # Capture learning state after this task
    result_obj.learning_state = get_learning_snapshot(orch, goal=task)

    return result_obj


async def run_eval(task_types: List[str], rounds: int, timeout: int):
    """Run the full multi-round learning evaluation."""
    from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

    print(f"\n{'=' * 72}")
    print(f"  SELF-LEARNING LOOP EVALUATION")
    print(f"{'=' * 72}")
    print(f"  Date:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Rounds:   {rounds}")
    print(f"  Tasks:    {', '.join(task_types)}")
    print(f"  Timeout:  {timeout}s per task")
    print(f"  API Key:  {'present' if os.getenv('ANTHROPIC_API_KEY') else 'MISSING'}")
    print(f"{'=' * 72}")

    all_results: List[RoundResult] = []

    for round_num in range(1, rounds + 1):
        print(f"\n{'━' * 72}")
        print(f"  ROUND {round_num}/{rounds}")
        print(f"{'━' * 72}")

        # Fresh Orchestrator each round to prevent accumulated state
        # (web search cache, executor state) from degrading responses.
        # Learning persists across rounds via SQLite — that's the signal
        # we're testing, not in-memory state.
        orch = Orchestrator()

        for task_type in task_types:
            task_set = TASK_SETS[task_type]
            # Use the SAME task across all rounds to measure learning.
            # (Previous cycling made quality differences look like learning
            #  regressions when they were just different-task variance.)
            task_idx = 0
            task = task_set["tasks"][task_idx]

            print(f"\n  [{task_set['label']}] Round {round_num}: {task[:65]}...")

            result = await run_single_task(
                orch,
                task,
                task_type,
                round_num,
                task_set["quality_checks"],
                timeout,
            )
            all_results.append(result)

            # Print result
            status = "✅" if result.success and result.quality_score >= 0.6 else "❌"
            print(
                f"  {status} Quality: {result.quality_score:.0%} | "
                f"Latency: {result.latency_s}s | "
                f"Content: {result.content_length} chars"
            )
            if result.error:
                print(f"     ERROR: {result.error[:100]}")
            for k, v in result.checks.items():
                mark = "✓" if v else "✗"
                print(f"     {mark} {k}")
            # Show learning state growth
            ls = result.learning_state
            if ls:
                ep = ls.get("episodes", 0)
                sr = ls.get("success_rate", 0)
                nl = ls.get("n_lessons", 0)
                cl = ls.get("ctx_len", 0)
                print(f"     DB: episodes={ep} success_rate={sr:.0%} lessons={nl} ctx={cl}ch")

    # ─── ANALYSIS ────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  LEARNING ANALYSIS")
    print(f"{'=' * 72}")

    for task_type in task_types:
        type_results = [r for r in all_results if r.task_type == task_type]
        if not type_results:
            continue

        label = TASK_SETS[task_type]["label"]
        print(f"\n  {label}:")

        # Quality trend
        qualities = [r.quality_score for r in type_results]
        latencies = [r.latency_s for r in type_results]

        first_half = qualities[: len(qualities) // 2] if len(qualities) > 1 else qualities
        second_half = qualities[len(qualities) // 2 :] if len(qualities) > 1 else qualities

        avg_first = sum(first_half) / max(len(first_half), 1)
        avg_second = sum(second_half) / max(len(second_half), 1)

        trend = (
            "↑ IMPROVING"
            if avg_second > avg_first + 0.05
            else ("↓ DEGRADING" if avg_second < avg_first - 0.05 else "→ STABLE")
        )

        print(f"    Quality: {' → '.join(f'{q:.0%}' for q in qualities)}")
        print(
            f"    Trend:   {trend} (first half avg: {avg_first:.0%}, second half avg: {avg_second:.0%})"
        )
        print(f"    Latency: {' → '.join(f'{l:.0f}s' for l in latencies)}")

        # Success rate
        successes = sum(1 for r in type_results if r.success and r.quality_score >= 0.6)
        print(
            f"    Success: {successes}/{len(type_results)} ({successes/max(len(type_results),1):.0%})"
        )

        # Learning DB growth across rounds
        episodes = [r.learning_state.get("episodes", 0) for r in type_results]
        lessons = [r.learning_state.get("n_lessons", 0) for r in type_results]
        ctx_lens = [r.learning_state.get("ctx_len", 0) for r in type_results]
        if any(episodes) or any(lessons):
            print(f"    Episodes: {' → '.join(str(e) for e in episodes)}")
            print(f"    Lessons:  {' → '.join(str(l) for l in lessons)}")
            print(f"    Context:  {' → '.join(f'{c}ch' for c in ctx_lens)}")

    # Learning state from last result
    last_state = all_results[-1].learning_state if all_results else {}
    if last_state.get("tier_history"):
        print(f"\n  CogRouter Tier History (learned):")
        for key, val in sorted(last_state["tier_history"].items()):
            print(f"    {key}: success={val['success']:.2f} count={val['count']}")

    if last_state.get("agent_proficiency"):
        print(f"\n  Agent Proficiency (learned):")
        for key, val in sorted(last_state["agent_proficiency"].items()):
            print(f"    {key}: proficiency={val['proficiency']:.2f} count={val['count']}")

    if last_state.get("td_episodes"):
        print(f"\n  TD-Lambda: {last_state['td_episodes']} episodes")

    # ─── GRADE ───────────────────────────────────────────────────────────
    total = len(all_results)
    passed = sum(1 for r in all_results if r.success and r.quality_score >= 0.6)
    pct = passed / max(total, 1)

    # Bonus for improvement trend
    all_qualities = [r.quality_score for r in all_results]
    if len(all_qualities) > 2:
        first_q = sum(all_qualities[: len(all_qualities) // 2]) / max(len(all_qualities) // 2, 1)
        last_q = sum(all_qualities[len(all_qualities) // 2 :]) / max(
            len(all_qualities) - len(all_qualities) // 2, 1
        )
        learning_bonus = 0.05 if last_q > first_q else 0.0
    else:
        learning_bonus = 0.0

    final_pct = min(1.0, pct + learning_bonus)
    if final_pct >= 0.9:
        grade = "A"
    elif final_pct >= 0.75:
        grade = "B"
    elif final_pct >= 0.6:
        grade = "C"
    elif final_pct >= 0.4:
        grade = "D"
    else:
        grade = "F"

    print(f"\n{'=' * 72}")
    print(
        f"  FINAL: {passed}/{total} passed ({pct:.0%}) + learning bonus {learning_bonus:.0%} = Grade {grade}"
    )
    print(f"  Total time: {sum(r.latency_s for r in all_results):.0f}s")
    print(f"{'=' * 72}")

    # Save results
    results_dir = Path("scripts/eval_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = results_dir / f"{ts}_learning_loop.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "rounds": rounds,
        "task_types": task_types,
        "total": total,
        "passed": passed,
        "grade": grade,
        "learning_bonus": learning_bonus,
        "results": [asdict(r) for r in all_results],
        "final_learning_state": last_state,
    }
    outfile.write_text(json.dumps(data, indent=2, default=str))
    print(f"\n  Saved: {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Learning Loop Evaluation")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds per task type")
    parser.add_argument(
        "--task", type=str, default=None, help="Single task type: research, coding, analysis"
    )
    parser.add_argument("--timeout", type=int, default=240, help="Timeout per task in seconds")
    args = parser.parse_args()

    task_types = [args.task] if args.task else ["research", "coding", "analysis"]
    asyncio.run(run_eval(task_types, args.rounds, args.timeout))

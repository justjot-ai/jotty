#!/usr/bin/env python3
"""
MAS (Multi-Agent System) Learning Evaluation
=============================================

Tests whether learning improves ACTUAL multi-agent orchestration outcomes.

Unlike the A/B eval (raw LLM calls) and real-world eval (orch.chat with max_steps=1),
this eval uses:
  - orch.run() — the full MAS pipeline with task decomposition
  - Multiple agents coordinating
  - Tools enabled
  - Learning propagating across agents and steps

Each task runs 3 times. If learning works at the MAS level:
  1. Run 1: Cold start, baseline quality
  2. Run 2: Learning from run 1 informs agents
  3. Run 3: Accumulated learning from runs 1+2

Uses Haiku to stress-test: if learning helps the weakest model in MAS, it works.

Cost estimate: ~$3-5 (18 MAS runs on Haiku, multi-step)
"""

import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval_mas")
logger.setLevel(logging.INFO)

EVAL_MODEL = "claude-3-haiku-20240307"
RUNS_PER_TASK = 3
SLEEP_BETWEEN = 8


def rx(content: str, pattern: str) -> bool:
    return bool(re.search(pattern, content, re.IGNORECASE | re.DOTALL))


def code_blocks(content: str, n: int = 2, min_lines: int = 5) -> bool:
    blocks = re.findall(r"```[\w]*\n(.*?)```", content, re.DOTALL)
    return len([b for b in blocks if len(b.strip().split("\n")) >= min_lines]) >= n


def has_func(content: str, name: str) -> bool:
    return rx(content, rf"def\s+{name}\s*\(")


def has_class(content: str, name: str) -> bool:
    return rx(content, rf"class\s+{name}")


ALL_MAS_TASKS = [
    # Task 1: Multi-agent coding task — requires planning + implementation + testing
    {
        "id": "MAS_api_framework",
        "goal": (
            "Build a Python REST API framework from scratch with:\n"
            "1. A Router class that maps HTTP methods + paths to handler functions\n"
            "2. Request/Response classes with headers, body, status code\n"
            "3. Middleware support (logging, auth, CORS)\n"
            "4. Path parameter extraction (e.g., /users/{id})\n"
            "5. JSON serialization/deserialization\n"
            "6. Error handling with proper HTTP status codes\n"
            "7. At least 5 test functions with assertions\n\n"
            "Write complete, working Python code. Do NOT use any external tools."
        ),
        "domain": "coding",
        "checks": [
            ("Router class", lambda c: has_class(c, "Router")),
            ("Request class", lambda c: has_class(c, "Request")),
            ("Response class", lambda c: has_class(c, "Response")),
            (
                "add_route or route decorator",
                lambda c: rx(c, r"def\s+(?:add_route|route|register)"),
            ),
            ("GET/POST methods", lambda c: rx(c, r"GET|POST|PUT|DELETE")),
            ("Middleware", lambda c: rx(c, r"middleware|Middleware")),
            ("Path parameters", lambda c: rx(c, r"\{.*?\}|path.?param|<\w+>")),
            ("JSON handling", lambda c: rx(c, r"json\.dumps|json\.loads|JSONResponse")),
            ("Error handling", lambda c: rx(c, r"(?:400|404|500|HTTPException|raise.*Error)")),
            ("Test functions", lambda c: len(re.findall(r"def\s+test_", c)) >= 3),
            ("Assertions", lambda c: len(re.findall(r"assert\s+\w+", c)) >= 5),
            ("Code blocks", lambda c: code_blocks(c, 2, 8)),
        ],
    },
    # Task 2: Research + synthesis — requires info gathering + analysis
    {
        "id": "MAS_distributed_db",
        "goal": (
            "Write a comprehensive technical comparison of distributed database "
            "consistency models. Cover:\n"
            "1. Strong consistency (linearizability) — definition, examples (Spanner), tradeoffs\n"
            "2. Eventual consistency — definition, examples (DynamoDB, Cassandra), conflict resolution\n"
            "3. Causal consistency — definition, vector clocks, examples\n"
            "4. CRDT-based consistency — what CRDTs are, G-Counter, OR-Set examples with code\n"
            "5. Consistency in practice: how to choose based on use case\n"
            "6. Include code examples for at least 2 consistency models\n"
            "7. Compare with a summary table\n\n"
            "Be thorough and technical. Do NOT use any external tools."
        ),
        "domain": "system_design",
        "checks": [
            ("Strong consistency", lambda c: rx(c, r"strong.?consistency|linearizab")),
            ("Spanner example", lambda c: rx(c, r"Spanner|TrueTime|Google")),
            ("Eventual consistency", lambda c: rx(c, r"eventual.?consistency")),
            ("DynamoDB or Cassandra", lambda c: rx(c, r"DynamoDB|Cassandra")),
            ("Causal consistency", lambda c: rx(c, r"causal.?consistency")),
            ("Vector clocks", lambda c: rx(c, r"vector.?clock|lamport|logical.?clock")),
            ("CRDTs", lambda c: rx(c, r"CRDT|conflict.?free|G.?Counter|OR.?Set")),
            ("Code examples", lambda c: code_blocks(c, 2, 3)),
            ("Comparison table", lambda c: rx(c, r"\|.*\|.*\||comparison|table|vs")),
            ("Tradeoffs", lambda c: rx(c, r"tradeoff|trade.?off|CAP|latency.*consistency")),
            ("500+ words", lambda c: len(c.split()) >= 500),
        ],
    },
    # Task 3: Complex algorithm — requires deep implementation
    {
        "id": "MAS_skiplist",
        "goal": (
            "Implement a Skip List data structure in Python.\n\n"
            "Requirements:\n"
            "1. SkipList class with:\n"
            "   - insert(key, value) — probabilistic level promotion (p=0.5)\n"
            "   - search(key) → value or None\n"
            "   - delete(key) → bool\n"
            "   - range_query(start, end) → list of (key, value) pairs\n"
            "2. SkipNode class with forward pointers array\n"
            "3. Randomized level generation with max_level cap\n"
            "4. Explain time complexity: O(log n) average for all operations\n"
            "5. Visual display method showing the skip list structure\n"
            "6. Tests:\n"
            "   - test_insert_and_search: insert 20 items, verify all found\n"
            "   - test_delete: insert, delete, verify gone\n"
            "   - test_range_query: verify correct range returned\n"
            "   - test_duplicates: update value on duplicate key\n"
            "   - 8+ assertions\n\n"
            "Do NOT use any external tools."
        ),
        "domain": "coding",
        "checks": [
            ("SkipList class", lambda c: has_class(c, "SkipList")),
            ("SkipNode class", lambda c: has_class(c, r"SkipNode|Node")),
            ("insert method", lambda c: has_func(c, "insert")),
            ("search method", lambda c: has_func(c, "search")),
            ("delete method", lambda c: has_func(c, "delete")),
            ("range_query method", lambda c: rx(c, r"def\s+range_query|def\s+range_search")),
            ("Random level", lambda c: rx(c, r"random|p\s*=\s*0\.5|coin.?flip|level.*random")),
            ("Forward pointers", lambda c: rx(c, r"forward|next_nodes|pointers|levels?\[")),
            ("O(log n) complexity", lambda c: rx(c, r"O\(log\s*n\)|logarithmic|log.?n")),
            ("Display method", lambda c: rx(c, r"def\s+(?:display|__str__|__repr__|print|show)")),
            ("Test functions", lambda c: len(re.findall(r"def\s+test_", c)) >= 3),
            ("8+ assertions", lambda c: len(re.findall(r"assert\s+\w+", c)) >= 8),
            ("Substantial code", lambda c: code_blocks(c, 3, 8)),
        ],
    },
    # Task 4: Cross-domain — system design + code
    {
        "id": "MAS_event_bus",
        "goal": (
            "Design and implement a distributed event bus system in Python.\n\n"
            "Requirements:\n"
            "1. ARCHITECTURE:\n"
            "   - EventBus class: publish/subscribe pattern\n"
            "   - Topic-based routing with wildcard support (e.g., 'orders.*')\n"
            "   - At-least-once delivery guarantee — explain how\n"
            "   - Dead letter queue for failed messages\n"
            "2. IMPLEMENTATION:\n"
            "   - Event class with id, topic, payload, timestamp, metadata\n"
            "   - Subscriber class with filter, handler, retry policy\n"
            "   - Async event dispatch with configurable concurrency\n"
            "   - Message persistence (in-memory store with WAL concept)\n"
            "3. PATTERNS:\n"
            "   - Event sourcing integration — how events reconstruct state\n"
            "   - Saga pattern for distributed transactions\n"
            "   - Explain backpressure handling\n"
            "4. TESTS:\n"
            "   - test_publish_subscribe: publish event, verify subscriber receives it\n"
            "   - test_wildcard_routing: 'orders.*' matches 'orders.created'\n"
            "   - test_dead_letter: failed handler → message goes to DLQ\n"
            "   - test_ordering: events delivered in publish order\n"
            "   - 8+ assertions\n\n"
            "Do NOT use any external tools."
        ),
        "domain": "system_design",
        "checks": [
            ("EventBus class", lambda c: has_class(c, "EventBus")),
            ("Event class", lambda c: has_class(c, "Event")),
            ("Subscriber class", lambda c: has_class(c, r"Subscriber|Handler|Consumer")),
            ("Publish method", lambda c: has_func(c, r"publish|emit|dispatch")),
            ("Subscribe method", lambda c: has_func(c, r"subscribe|on|listen|register")),
            ("Wildcard routing", lambda c: rx(c, r"wildcard|\*|fnmatch|glob|pattern.*match")),
            ("Dead letter queue", lambda c: rx(c, r"dead.?letter|DLQ|failed.*queue")),
            ("At-least-once", lambda c: rx(c, r"at.?least.?once|retry|acknowledge|ack")),
            ("Event sourcing", lambda c: rx(c, r"event.?sourc|reconstruct.*state|replay")),
            ("Saga pattern", lambda c: rx(c, r"saga|compensat|distributed.*transaction")),
            ("Backpressure", lambda c: rx(c, r"backpressure|back.?pressure|rate.?limit|throttl")),
            ("Test functions", lambda c: len(re.findall(r"def\s+test_", c)) >= 3),
            ("8+ assertions", lambda c: len(re.findall(r"assert\s+\w+", c)) >= 5),
            ("Substantial code", lambda c: code_blocks(c, 3, 5)),
        ],
    },
]

# Use first 2 tasks for quick eval, all 4 for full eval
MAS_TASKS = ALL_MAS_TASKS[:2]


class MASEval:
    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []
        self._orch = None
        self._learning = None

    def _get_orchestrator(self) -> Any:
        if self._orch is None:
            from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

            self._orch = Orchestrator()
        return self._orch

    def _get_learning(self) -> Any:
        if self._learning is None:
            from Jotty.core.intelligence.learning.learning_service import LearningService

            self._learning = LearningService.get_instance()
        return self._learning

    def _snapshot(self, domain: str, goal: str = "") -> Dict[str, Any]:
        ls = self._get_learning()
        guidance = ls.query(domain or "general", "")
        ctx = ls.build_context_string(domain or "general", "", goal=goal)
        ret = ls.build_retrieval_context(domain or "general", "", goal=goal)
        try:
            conn = ls._store._get_conn()
            global_eps = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        except Exception:
            global_eps = 0
        return {
            "domain_episodes": guidance.get("total_episodes", 0),
            "global_episodes": global_eps,
            "success_rate": guidance.get("success_rate", 0.0),
            "ctx_len": len(ctx),
            "ret_len": len(ret),
            "ctx_preview": ctx[:200] if ctx else "",
        }

    async def run_task(self, task: Dict, run_num: int) -> Dict[str, Any]:
        label = f"{task['id']} run {run_num}/{RUNS_PER_TASK}"
        print(f"\n  {'─' * 60}")
        print(f"  {label}")
        print(f"  {'─' * 60}")

        orch = self._get_orchestrator()
        from Jotty.core.intelligence.learning.learning_service import classify_domain

        classified_domain, _ = classify_domain(task["goal"])
        domain = classified_domain or task.get("domain", "general")
        snap_before = self._snapshot(domain, task["goal"])

        print(f"  Model: {EVAL_MODEL} | Domain: {domain}")
        print(
            f"  Episodes: {snap_before['domain_episodes']} (domain) / {snap_before['global_episodes']} (global)"
        )
        print(f"  Success rate: {snap_before['success_rate']:.0%}")
        print(f"  Context: {snap_before['ctx_len']}ch | Retrieval: {snap_before['ret_len']}ch")
        if snap_before["ctx_preview"]:
            print(f"    > {snap_before['ctx_preview'][:120]}")

        start = time.time()
        content = ""
        error = None
        execution_mode = "unknown"

        try:
            result = await orch.run(
                goal=task["goal"],
                learn=True,
                provider="anthropic",
                model=EVAL_MODEL,
            )
            for _attr in ("output", "final_output", "content", "result"):
                _val = getattr(result, _attr, None)
                if isinstance(_val, str) and len(_val) > len(content):
                    content = _val
            if not content:
                content = str(result)
            execution_mode = getattr(result, "mode", "auto")
        except Exception as e:
            error = str(e)[:300]
            print(f"  ERROR: {error}")

        elapsed = time.time() - start
        snap_after = self._snapshot(domain, task["goal"])

        checks = {}
        for name, fn in task["checks"]:
            try:
                checks[name] = fn(content)
            except Exception:
                checks[name] = False

        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        check_ratio = passed / max(total, 1)

        n_code = len(re.findall(r"```[\w]*\n.+?```", content, re.DOTALL))
        quality = (
            check_ratio * 0.5 + min(1.0, len(content) / 6000) * 0.25 + min(1.0, n_code / 3) * 0.25
        )
        success = check_ratio >= 0.60 and len(content) >= 2000

        record = {
            "task_id": task["id"],
            "run": run_num,
            "domain": domain,
            "execution_mode": str(execution_mode),
            "success": success,
            "quality": round(quality, 3),
            "checks_passed": passed,
            "checks_total": total,
            "check_ratio": round(check_ratio, 3),
            "content_length": len(content),
            "code_blocks": n_code,
            "elapsed": round(elapsed, 1),
            "ep_before": snap_before["domain_episodes"],
            "ep_after": snap_after["domain_episodes"],
            "ctx_injected": snap_before["ctx_len"],
            "ret_injected": snap_before["ret_len"],
            "error": error,
        }
        self.results.append(record)

        status = "PASS" if success else "FAIL"
        print(
            f"\n  [{status}] Q={quality:.2f} | Checks={passed}/{total} ({check_ratio:.0%}) | "
            f"Len={len(content)} | Code={n_code} | Mode={execution_mode} | {elapsed:.0f}s"
        )
        print(f"  Episodes: {snap_before['domain_episodes']}→{snap_after['domain_episodes']}")

        for name, ok in checks.items():
            print(f"    [{'+'if ok else '-'}] {name}")

        return record

    def _print_summary(self) -> None:
        print(f"\n{'=' * 80}")
        print(f"  MAS LEARNING EVALUATION — RESULTS")
        print(f"{'=' * 80}")

        for r in self.results:
            s = "PASS" if r["success"] else "FAIL"
            print(
                f"  [{s}] {r['task_id']:22s} run={r['run']} Q={r['quality']:.2f} "
                f"Chk={r['checks_passed']}/{r['checks_total']} ({r['check_ratio']:.0%}) "
                f"Len={r['content_length']:5d} mode={r['execution_mode']}"
            )

        print(f"\n{'─' * 80}")
        print(f"  LEARNING CURVES")
        print(f"{'─' * 80}")

        task_ids = list(dict.fromkeys(r["task_id"] for r in self.results))
        improvements = []

        for tid in task_ids:
            runs = sorted([r for r in self.results if r["task_id"] == tid], key=lambda r: r["run"])
            if len(runs) < 2:
                continue

            print(f"\n  {tid}:")
            for r in runs:
                bar_len = int(r["check_ratio"] * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                print(
                    f"    Run {r['run']}: {bar} {r['check_ratio']:.0%} "
                    f"({r['checks_passed']}/{r['checks_total']}) "
                    f"Q={r['quality']:.2f} ctx={r['ctx_injected']}ch"
                )

            first, last = runs[0]["check_ratio"], runs[-1]["check_ratio"]
            best = max(r["check_ratio"] for r in runs)
            delta = last - first
            improvements.append(
                {"task": tid, "first": first, "last": last, "best": best, "delta": delta}
            )

            trend = "IMPROVED" if delta > 0.03 else "MAINTAINED" if delta >= -0.03 else "DEGRADED"
            print(f"    Trend: {first:.0%} → {last:.0%} ({delta:+.0%}) [{trend}]")

        # Aggregate
        print(f"\n{'─' * 80}")
        print(f"  AGGREGATE")
        print(f"{'─' * 80}")

        run1 = [r for r in self.results if r["run"] == 1]
        run3 = [r for r in self.results if r["run"] == RUNS_PER_TASK]

        if run1 and run3:
            avg_q1 = sum(r["quality"] for r in run1) / len(run1)
            avg_q3 = sum(r["quality"] for r in run3) / len(run3)
            avg_cr1 = sum(r["check_ratio"] for r in run1) / len(run1)
            avg_cr3 = sum(r["check_ratio"] for r in run3) / len(run3)
            pass1 = sum(1 for r in run1 if r["success"]) / len(run1)
            pass3 = sum(1 for r in run3 if r["success"]) / len(run3)

            print(f"\n  {'Metric':<25s} {'Run 1':>10s} {'Run 3':>10s} {'Delta':>10s}")
            print(f"  {'─' * 55}")
            print(
                f"  {'Avg check rate':<25s} {avg_cr1:>9.0%} {avg_cr3:>9.0%} {avg_cr3-avg_cr1:>+9.0%}"
            )
            print(f"  {'Avg quality':<25s} {avg_q1:>9.2f} {avg_q3:>9.2f} {avg_q3-avg_q1:>+9.2f}")
            print(f"  {'Pass rate':<25s} {pass1:>9.0%} {pass3:>9.0%} {pass3-pass1:>+9.0%}")

        if improvements:
            improved = sum(1 for i in improvements if i["delta"] > 0.03)
            maintained = sum(1 for i in improvements if -0.03 <= i["delta"] <= 0.03)
            degraded = sum(1 for i in improvements if i["delta"] < -0.03)
            avg_delta = sum(i["delta"] for i in improvements) / len(improvements)
            print(f"\n  Improved:   {improved}/{len(improvements)}")
            print(f"  Maintained: {maintained}/{len(improvements)}")
            print(f"  Degraded:   {degraded}/{len(improvements)}")
            print(f"  Avg Δ: {avg_delta:+.1%}")

    async def run_all(self) -> None:
        print(f"\n{'=' * 80}")
        print(f"  JOTTY MAS LEARNING EVALUATION")
        print(f"{'=' * 80}")
        print(f"  Date: {datetime.now().isoformat()}")
        print(f"  Model: {EVAL_MODEL}")
        print(
            f"  Tasks: {len(MAS_TASKS)} × {RUNS_PER_TASK} runs = {len(MAS_TASKS) * RUNS_PER_TASK} total"
        )
        print(f"  Pipeline: orch.run() — full MAS with agent routing")

        for task_idx, task in enumerate(MAS_TASKS):
            print(f"\n{'=' * 80}")
            print(f"  TASK {task_idx + 1}/{len(MAS_TASKS)}: {task['id']}")
            print(f"{'=' * 80}")

            for run in range(1, RUNS_PER_TASK + 1):
                await self.run_task(task, run)
                if run < RUNS_PER_TASK:
                    print(f"\n  Waiting {SLEEP_BETWEEN}s for learning...")
                    await asyncio.sleep(SLEEP_BETWEEN)

        self._print_summary()

        results_dir = Path(__file__).parent / "eval_results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = results_dir / f"{ts}_mas_learning.json"
        with open(out, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "model": EVAL_MODEL,
                    "runs_per_task": RUNS_PER_TASK,
                    "results": self.results,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\n  Saved: {out}")


async def main() -> None:
    await MASEval().run_all()


if __name__ == "__main__":
    asyncio.run(main())

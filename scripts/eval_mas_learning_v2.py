#!/usr/bin/env python3
"""
MAS Learning Eval V2 — Tests orch.run() learning over repeated tasks
=====================================================================

Tests whether the FULL multi-agent pipeline (orch.run) improves over
repeated runs of the same task. Uses DB isolation so each eval starts clean.

Key difference from eval_learning_v2.py:
  - This uses orch.run() (full MAS: agent routing, tools, decomposition)
  - eval_learning_v2.py uses direct LLM calls (isolates learning signal)

For each task: runs 5 times sequentially. Learning from run N should
improve run N+1. Tracks embeddings stored, distilled lessons, and
context injected at each run.

Cost estimate: ~$5-10 (Haiku, 2 tasks × 5 MAS runs)
"""

import asyncio
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval_mas_v2")
logger.setLevel(logging.INFO)

# Disable DSPy disk cache so each run makes a fresh LLM call.
# Without this, DSPy returns stale responses from ~/.dspy_cache
# and we cannot measure the effect of new learning context.
try:
    import dspy

    dspy.cache.enable_disk_cache = False
    dspy.cache.enable_memory_cache = False
    logger.info("DSPy cache disabled for eval")
except Exception:
    pass

EVAL_MODEL = "claude-3-haiku-20240307"
RUNS_PER_TASK = 5
SLEEP_BETWEEN = 8


# ─── Check Helpers ──────────────────────────────────────────────────────


def rx(content: str, pattern: str) -> bool:
    return bool(re.search(pattern, content, re.IGNORECASE | re.DOTALL))


def code_blocks(content: str, n: int = 2, min_lines: int = 5) -> bool:
    """Detect substantial code — fenced blocks OR inline code patterns."""
    # 1. Fenced markdown blocks (```python ... ```)
    fenced = re.findall(r"```[\w]*\n(.*?)```", content, re.DOTALL)
    fenced_count = len([b for b in fenced if len(b.strip().split("\n")) >= min_lines])
    if fenced_count >= n:
        return True
    # 2. Count total code-like lines (class/def/import/self/assert/return)
    code_line_count = len(
        re.findall(
            r"^\s*(class |def |import |from .+ import |self\.|assert |return )",
            content,
            re.MULTILINE,
        )
    )
    threshold = n * min_lines
    return code_line_count >= threshold


def has_func(content: str, name: str) -> bool:
    return rx(content, rf"def\s+{name}\s*\(")


def has_class(content: str, name: str) -> bool:
    return rx(content, rf"class\s+{name}")


# ─── Tasks ──────────────────────────────────────────────────────────────

MAS_TASKS = [
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
            ("Route method", lambda c: rx(c, r"def\s+(?:add_route|route|register)")),
            ("HTTP methods", lambda c: rx(c, r"GET|POST|PUT|DELETE")),
            ("Middleware", lambda c: rx(c, r"middleware|Middleware")),
            ("Path params", lambda c: rx(c, r"\{.*?\}|path.?param|<\w+>")),
            ("JSON handling", lambda c: rx(c, r"json\.dumps|json\.loads|JSONResponse")),
            ("Error handling", lambda c: rx(c, r"(?:400|404|500|HTTPException|raise.*Error)")),
            ("Test functions", lambda c: len(re.findall(r"def\s+test_", c)) >= 3),
            ("Assertions", lambda c: len(re.findall(r"assert\s+\w+", c)) >= 5),
            ("Substantial code", lambda c: code_blocks(c, 2, 8)),
        ],
    },
    {
        "id": "MAS_distributed_db",
        "goal": (
            "Write a comprehensive technical comparison of distributed database "
            "consistency models. Cover:\n"
            "1. Strong consistency (linearizability) — definition, examples (Spanner)\n"
            "2. Eventual consistency — definition, examples (DynamoDB, Cassandra)\n"
            "3. Causal consistency — vector clocks, examples\n"
            "4. CRDTs — G-Counter, OR-Set with code examples\n"
            "5. How to choose based on use case\n"
            "6. Code examples for 2+ models\n"
            "7. Comparison table\n\n"
            "Be thorough and technical. Do NOT use any external tools."
        ),
        "domain": "system_design",
        "checks": [
            ("Strong consistency", lambda c: rx(c, r"strong.?consistency|linearizab")),
            ("Spanner", lambda c: rx(c, r"Spanner|TrueTime|Google")),
            ("Eventual consistency", lambda c: rx(c, r"eventual.?consistency")),
            ("DynamoDB/Cassandra", lambda c: rx(c, r"DynamoDB|Cassandra")),
            ("Causal consistency", lambda c: rx(c, r"causal.?consistency")),
            ("Vector clocks", lambda c: rx(c, r"vector.?clock|lamport|logical.?clock")),
            ("CRDTs", lambda c: rx(c, r"CRDT|conflict.?free|G.?Counter|OR.?Set")),
            ("Code examples", lambda c: code_blocks(c, 2, 3)),
            ("Comparison table", lambda c: rx(c, r"\|.*\|.*\||comparison|table|vs")),
            ("Tradeoffs", lambda c: rx(c, r"tradeoff|trade.?off|CAP|latency.*consistency")),
            ("500+ words", lambda c: len(c.split()) >= 500),
        ],
    },
]


# ─── DB Isolation ───────────────────────────────────────────────────────


class IsolatedLearningDB:
    """Redirects LearningStore to a fresh temp database."""

    def __init__(self, label: str = "mas_eval"):
        self.label = label
        self.tmp_dir: Optional[str] = None
        self._orig_store_instances: Dict = {}
        self._orig_ls: Any = None

    def __enter__(self):
        self.tmp_dir = tempfile.mkdtemp(prefix=f"jotty_mas_{self.label}_")
        db_path = os.path.join(self.tmp_dir, "learning.db")

        from Jotty.core.intelligence.learning.learning_store import LearningStore

        self._orig_store_instances = dict(LearningStore._instances)
        LearningStore._instances.clear()
        self.store = LearningStore(db_path)
        LearningStore._instances["default"] = self.store

        from Jotty.core.intelligence.learning.learning_service import LearningService

        self._orig_ls = LearningService._instance
        LearningService._instance = None

        return self

    def __exit__(self, *exc):
        from Jotty.core.intelligence.learning.learning_store import LearningStore
        from Jotty.core.intelligence.learning.learning_service import LearningService

        LearningStore._instances.clear()
        LearningStore._instances.update(self._orig_store_instances)
        LearningService._instance = self._orig_ls

        if self.tmp_dir:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def get_stats(self) -> Dict[str, Any]:
        conn = self.store._get_conn()
        ep_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        emb_count = conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        try:
            lesson_count = conn.execute("SELECT COUNT(*) FROM distilled_lessons").fetchone()[0]
        except Exception:
            lesson_count = 0
        return {"episodes": ep_count, "embeddings": emb_count, "lessons": lesson_count}


# ─── Eval Engine ────────────────────────────────────────────────────────


class MASEvalV2:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self._orch = None

    def _get_orchestrator(self) -> Any:
        if self._orch is None:
            from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

            self._orch = Orchestrator()
        return self._orch

    def _snapshot(self, domain: str, goal: str) -> Dict[str, Any]:
        from Jotty.core.intelligence.learning.learning_service import LearningService

        ls = LearningService.get_instance()
        guidance = ls.query(domain, "")
        ctx = ls.build_context_string(domain, "", goal=goal)
        ret = ls.build_retrieval_context(domain, "", goal=goal)
        lessons = ls.retrieve_distilled_lessons(domain, goal=goal, top_k=5)
        return {
            "episodes": guidance.get("total_episodes", 0),
            "success_rate": guidance.get("success_rate", 0.0),
            "ctx_len": len(ctx),
            "ret_len": len(ret),
            "n_lessons": len(lessons),
            "ctx_preview": ctx[:150] if ctx else "",
            "lessons_preview": [l["lesson"][:80] for l in lessons[:2]],
        }

    async def run_task(
        self, task: Dict, run_num: int, iso_db: IsolatedLearningDB
    ) -> Dict[str, Any]:
        from Jotty.core.intelligence.learning.learning_service import classify_domain

        classified_domain, _ = classify_domain(task["goal"])
        domain = classified_domain or task.get("domain", "general")
        snap = self._snapshot(domain, task["goal"])
        db_stats = iso_db.get_stats()

        print(f"\n  {'─' * 60}")
        print(f"  {task['id']} run {run_num}/{RUNS_PER_TASK}")
        print(f"  {'─' * 60}")
        print(
            f"  Domain: {domain} | Episodes: {snap['episodes']} | Success: {snap['success_rate']:.0%}"
        )
        print(
            f"  Context: {snap['ctx_len']}ch | Retrieval: {snap['ret_len']}ch | Lessons: {snap['n_lessons']}"
        )
        print(
            f"  DB: ep={db_stats['episodes']} emb={db_stats['embeddings']} les={db_stats['lessons']}"
        )
        if snap["lessons_preview"]:
            for lp in snap["lessons_preview"][:2]:
                print(f"    lesson> {lp}")
        elif snap["ctx_preview"]:
            print(f"    ctx> {snap['ctx_preview'][:120]}")

        start = time.time()
        content = ""
        error = None

        # Clear DSPy cache before each run so fresh prompts hit the API
        try:
            import dspy as _dspy

            _dspy.cache.enable_disk_cache = False
            _dspy.cache.enable_memory_cache = False
            _dspy.cache.memory_cache.clear()
        except Exception:
            pass

        try:
            orch = self._get_orchestrator()
            result = await orch.run(
                goal=task["goal"],
                learn=True,
                provider="anthropic",
                model=EVAL_MODEL,
            )
            # Extract content from result (handle both str and dict outputs)
            for _attr in ("output", "final_output", "content", "result"):
                _val = getattr(result, _attr, None)
                if isinstance(_val, str) and len(_val) > len(content):
                    content = _val
                elif isinstance(_val, dict):
                    # Multi-agent dict: join per-agent outputs
                    parts = [str(v).strip() for v in _val.values() if v]
                    joined = "\n\n---\n\n".join(parts)
                    if len(joined) > len(content):
                        content = joined
            if not content:
                content = str(getattr(result, "output", result))
        except Exception as e:
            error = str(e)[:300]
            print(f"  ERROR: {error}")

        elapsed = time.time() - start

        # Score
        checks = {}
        for name, fn in task["checks"]:
            try:
                checks[name] = fn(content)
            except Exception:
                checks[name] = False

        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        ratio = passed / max(total, 1)
        n_fenced = len(re.findall(r"```[\w]*\n.+?```", content, re.DOTALL))
        n_code = n_fenced or (1 if code_blocks(content, 1, 5) else 0)
        quality = ratio * 0.5 + min(1.0, len(content) / 6000) * 0.25 + min(1.0, n_code / 3) * 0.25

        snap_after = self._snapshot(domain, task["goal"])
        db_after = iso_db.get_stats()

        record = {
            "task_id": task["id"],
            "run": run_num,
            "domain": domain,
            "success": ratio >= 0.60 and len(content) >= 2000,
            "quality": round(quality, 3),
            "check_ratio": round(ratio, 3),
            "passed": passed,
            "total": total,
            "content_len": len(content),
            "code_blocks": n_code,
            "elapsed": round(elapsed, 1),
            "ctx_injected": snap["ctx_len"],
            "ret_injected": snap["ret_len"],
            "n_lessons_before": snap["n_lessons"],
            "db_episodes": db_after["episodes"],
            "db_embeddings": db_after["embeddings"],
            "db_lessons": db_after["lessons"],
            "error": error,
        }
        self.results.append(record)

        status = "PASS" if record["success"] else "FAIL"
        print(
            f"\n  [{status}] Q={quality:.2f} | Chk={passed}/{total} ({ratio:.0%}) | "
            f"Len={len(content)} | Code={n_code} | {elapsed:.0f}s"
        )
        print(
            f"  DB after: ep={db_after['episodes']} emb={db_after['embeddings']} "
            f"les={db_after['lessons']}"
        )
        for name, ok in checks.items():
            print(f"    [{'+'if ok else '-'}] {name}")

        return record

    def _print_summary(self) -> None:
        print(f"\n{'=' * 80}")
        print(f"  MAS LEARNING EVAL V2 — RESULTS")
        print(f"{'=' * 80}")

        task_ids = list(dict.fromkeys(r["task_id"] for r in self.results))

        for tid in task_ids:
            runs = sorted([r for r in self.results if r["task_id"] == tid], key=lambda x: x["run"])
            if len(runs) < 2:
                continue

            print(f"\n  {tid} ({runs[0]['domain']})")
            print(f"  {'─' * 60}")

            for r in runs:
                bar_len = int(r["check_ratio"] * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                print(
                    f"    Run {r['run']}: {bar} {r['check_ratio']:.0%} "
                    f"({r['passed']}/{r['total']}) Q={r['quality']:.2f} "
                    f"ctx={r['ctx_injected']:4d}ch les={r['db_lessons']} "
                    f"emb={r['db_embeddings']} {r['elapsed']:.0f}s"
                )

            first, last = runs[0]["check_ratio"], runs[-1]["check_ratio"]
            best = max(r["check_ratio"] for r in runs)
            delta = last - first
            trend = "IMPROVED" if delta > 0.03 else "MAINTAINED" if delta >= -0.03 else "DEGRADED"
            print(f"    Trend: {first:.0%} → {last:.0%} (best={best:.0%}) [{trend}]")

        # Aggregate
        print(f"\n{'─' * 80}")
        print(f"  AGGREGATE")
        print(f"{'─' * 80}")

        run1 = [r for r in self.results if r["run"] == 1]
        run_last = [r for r in self.results if r["run"] == RUNS_PER_TASK]

        if run1 and run_last:
            avg_cr1 = sum(r["check_ratio"] for r in run1) / len(run1)
            avg_cr_last = sum(r["check_ratio"] for r in run_last) / len(run_last)
            avg_q1 = sum(r["quality"] for r in run1) / len(run1)
            avg_q_last = sum(r["quality"] for r in run_last) / len(run_last)

            print(
                f"\n  {'Metric':<25s} {'Run 1':>10s} {'Run {}'.format(RUNS_PER_TASK):>10s} {'Delta':>10s}"
            )
            print(f"  {'─' * 55}")
            print(
                f"  {'Avg check rate':<25s} {avg_cr1:>9.0%} {avg_cr_last:>9.0%} {avg_cr_last-avg_cr1:>+9.0%}"
            )
            print(
                f"  {'Avg quality':<25s} {avg_q1:>9.2f} {avg_q_last:>9.2f} {avg_q_last-avg_q1:>+9.2f}"
            )

            # Infrastructure
            if run_last:
                avg_emb = sum(r["db_embeddings"] for r in run_last) / len(run_last)
                avg_les = sum(r["db_lessons"] for r in run_last) / len(run_last)
                print(f"\n  Infrastructure (after {RUNS_PER_TASK} runs):")
                print(f"    Avg embeddings: {avg_emb:.0f}")
                print(f"    Avg distilled lessons: {avg_les:.0f}")

    async def run_all(self) -> None:
        print(f"\n{'=' * 80}")
        print(f"  JOTTY MAS LEARNING EVAL V2 — orch.run()")
        print(f"{'=' * 80}")
        print(f"  Date: {datetime.now().isoformat()}")
        print(f"  Model: {EVAL_MODEL}")
        print(f"  Tasks: {len(MAS_TASKS)} × {RUNS_PER_TASK} runs")
        print(f"  Pipeline: orch.run() — full MAS with agent routing, tools, learning")
        print(f"  DB: ISOLATED (fresh per task)")

        for task_idx, task in enumerate(MAS_TASKS):
            print(f"\n{'=' * 80}")
            print(f"  TASK {task_idx + 1}/{len(MAS_TASKS)}: {task['id']}")
            print(f"{'=' * 80}")

            with IsolatedLearningDB(f"mas_{task['id']}") as iso:
                for run in range(1, RUNS_PER_TASK + 1):
                    await self.run_task(task, run, iso)
                    if run < RUNS_PER_TASK:
                        print(f"\n  Waiting {SLEEP_BETWEEN}s for learning pipeline...")
                        await asyncio.sleep(SLEEP_BETWEEN)

        self._print_summary()

        results_dir = Path(__file__).parent / "eval_results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = results_dir / f"{ts}_mas_learning_v2.json"
        with open(out, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "model": EVAL_MODEL,
                    "runs_per_task": RUNS_PER_TASK,
                    "pipeline": "orch.run()",
                    "features": [
                        "vector_embeddings",
                        "fact_distillation",
                        "db_isolation",
                        "per_agent",
                    ],
                    "results": self.results,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\n  Saved: {out}")


async def main() -> None:
    await MASEvalV2().run_all()


if __name__ == "__main__":
    asyncio.run(main())

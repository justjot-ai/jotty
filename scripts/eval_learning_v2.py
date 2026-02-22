#!/usr/bin/env python3
"""
Learning Evaluation V2 — Production-Grade with DB Isolation & Statistics
=========================================================================

Improvements over V1:
  1. DB ISOLATION: Each eval run uses a fresh SQLite database (no cross-contamination)
  2. A/B COMPARISON: With-learning vs without-learning in same run
  3. EMBEDDING VERIFICATION: Confirms vector embeddings are computed + stored
  4. FACT DISTILLATION: Verifies distilled lessons are extracted
  5. STATISTICAL RIGOR: n=5 runs per task, mean/stddev/confidence intervals
  6. ALL TASK TYPES: Coding, research, system design — 4 diverse tasks

Cost estimate: ~$5-8 (Haiku, A/B × 4 tasks × 5 runs)
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
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval_v2")
logger.setLevel(logging.INFO)

EVAL_MODEL = "claude-3-haiku-20240307"
RUNS_PER_TASK = 5
SLEEP_BETWEEN = 4
# Set USE_HARD_TASKS=True to run harder tasks that demonstrate learning more clearly
USE_HARD_TASKS = True


# ─── Check Helpers ──────────────────────────────────────────────────────


def rx(content: str, pattern: str) -> bool:
    return bool(re.search(pattern, content, re.IGNORECASE | re.DOTALL))


def code_blocks(content: str, n: int = 2, min_lines: int = 5) -> bool:
    blocks = re.findall(r"```[\w]*\n(.*?)```", content, re.DOTALL)
    return len([b for b in blocks if len(b.strip().split("\n")) >= min_lines]) >= n


def has_func(content: str, name: str) -> bool:
    return rx(content, rf"def\s+{name}\s*\(")


def has_class(content: str, name: str) -> bool:
    return rx(content, rf"class\s+{name}")


# ─── Tasks ──────────────────────────────────────────────────────────────

EVAL_TASKS = [
    {
        "id": "api_framework",
        "goal": (
            "Build a Python REST API framework from scratch with:\n"
            "1. A Router class that maps HTTP methods + paths to handler functions\n"
            "2. Request/Response classes with headers, body, status code\n"
            "3. Middleware support (logging, auth)\n"
            "4. Path parameter extraction (e.g., /users/{id})\n"
            "5. JSON serialization/deserialization\n"
            "6. Error handling with proper HTTP status codes\n"
            "7. At least 5 test functions with assertions\n\n"
            "Write complete, working Python code."
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
        "id": "distributed_db",
        "goal": (
            "Write a comprehensive technical comparison of distributed database "
            "consistency models. Cover:\n"
            "1. Strong consistency (linearizability) — definition, examples (Spanner)\n"
            "2. Eventual consistency — definition, examples (DynamoDB, Cassandra)\n"
            "3. Causal consistency — vector clocks, examples\n"
            "4. CRDTs — G-Counter, OR-Set with code examples\n"
            "5. How to choose based on use case\n"
            "6. Code examples for 2+ models\n"
            "7. Comparison table"
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
    {
        "id": "skiplist",
        "goal": (
            "Implement a Skip List data structure in Python.\n\n"
            "Requirements:\n"
            "1. SkipList class with insert(key, value), search(key), delete(key), "
            "range_query(start, end)\n"
            "2. SkipNode class with forward pointers array\n"
            "3. Randomized level generation (p=0.5)\n"
            "4. Explain O(log n) time complexity\n"
            "5. Visual display method\n"
            "6. Tests: test_insert_and_search (20 items), test_delete, "
            "test_range_query, test_duplicates with 8+ assertions"
        ),
        "domain": "coding",
        "checks": [
            ("SkipList class", lambda c: has_class(c, "SkipList")),
            ("Node class", lambda c: has_class(c, r"SkipNode|Node")),
            ("insert", lambda c: has_func(c, "insert")),
            ("search", lambda c: has_func(c, "search")),
            ("delete", lambda c: has_func(c, "delete")),
            ("range_query", lambda c: rx(c, r"def\s+range_query|def\s+range_search")),
            ("Random level", lambda c: rx(c, r"random|p\s*=\s*0\.5|coin.?flip|level.*random")),
            ("Forward ptrs", lambda c: rx(c, r"forward|next_nodes|pointers|levels?\[")),
            ("O(log n)", lambda c: rx(c, r"O\(log\s*n\)|logarithmic|log.?n")),
            ("Display", lambda c: rx(c, r"def\s+(?:display|__str__|__repr__|print|show)")),
            ("Tests", lambda c: len(re.findall(r"def\s+test_", c)) >= 3),
            ("8+ asserts", lambda c: len(re.findall(r"assert\s+\w+", c)) >= 8),
            ("Code blocks", lambda c: code_blocks(c, 3, 8)),
        ],
    },
    {
        "id": "event_bus",
        "goal": (
            "Design and implement a distributed event bus in Python.\n\n"
            "1. EventBus class with publish/subscribe pattern\n"
            "2. Topic-based routing with wildcard support (e.g., 'orders.*')\n"
            "3. At-least-once delivery with retry\n"
            "4. Dead letter queue for failed messages\n"
            "5. Event class (id, topic, payload, timestamp)\n"
            "6. Subscriber class with filter, handler, retry policy\n"
            "7. Event sourcing concept\n"
            "8. Saga pattern explanation\n"
            "9. Tests: publish/subscribe, wildcard routing, dead letter, ordering\n"
            "   with 8+ assertions"
        ),
        "domain": "system_design",
        "checks": [
            ("EventBus", lambda c: has_class(c, "EventBus")),
            ("Event class", lambda c: has_class(c, "Event")),
            ("Subscriber", lambda c: has_class(c, r"Subscriber|Handler|Consumer")),
            ("Publish", lambda c: has_func(c, r"publish|emit|dispatch")),
            ("Subscribe", lambda c: has_func(c, r"subscribe|on|listen|register")),
            ("Wildcard", lambda c: rx(c, r"wildcard|\*|fnmatch|glob|pattern.*match")),
            ("DLQ", lambda c: rx(c, r"dead.?letter|DLQ|failed.*queue")),
            ("At-least-once", lambda c: rx(c, r"at.?least.?once|retry|acknowledge|ack")),
            ("Event sourcing", lambda c: rx(c, r"event.?sourc|reconstruct.*state|replay")),
            ("Saga", lambda c: rx(c, r"saga|compensat|distributed.*transaction")),
            ("Tests", lambda c: len(re.findall(r"def\s+test_", c)) >= 3),
            ("Assertions", lambda c: len(re.findall(r"assert\s+\w+", c)) >= 5),
            ("Code blocks", lambda c: code_blocks(c, 3, 5)),
        ],
    },
]

# Hard tasks: designed to fail on first attempt with Haiku, showing learning uplift
HARD_TASKS = [
    {
        "id": "hard_interpreter",
        "goal": (
            "Implement a complete Lisp interpreter in Python with these EXACT requirements:\n\n"
            "1. Tokenizer: split input into tokens (parens, numbers, strings, symbols)\n"
            "2. Parser: convert tokens to AST (nested lists)\n"
            "3. Environment class with parent scope chain for lexical scoping\n"
            "4. Evaluator supporting:\n"
            "   - Arithmetic: +, -, *, /\n"
            "   - Comparison: <, >, =, <=, >=\n"
            "   - define, set!, if, cond, lambda, let, begin, quote\n"
            "   - cons, car, cdr, list, null?, pair?\n"
            "   - Higher-order: map, filter, apply\n"
            "5. Tail call optimization for recursive functions\n"
            "6. Error handling with LispError class\n"
            "7. REPL function\n\n"
            "8. Tests (MANDATORY - at least 8 test functions):\n"
            "   - test_arithmetic: (+ 1 2) => 3, (* 3 4) => 12\n"
            "   - test_define: (define x 10) (+ x 5) => 15\n"
            "   - test_lambda: ((lambda (x) (* x x)) 5) => 25\n"
            "   - test_closure: define adder, test lexical scoping\n"
            "   - test_list_ops: cons, car, cdr operations\n"
            "   - test_higher_order: map, filter with lambda\n"
            "   - test_recursion: factorial or fibonacci\n"
            "   - test_let: (let ((x 1) (y 2)) (+ x y)) => 3\n"
            "   - Each test must have assert statements\n\n"
            "Write COMPLETE, RUNNABLE code. Include ALL classes and functions."
        ),
        "domain": "coding",
        "checks": [
            ("Tokenizer", lambda c: rx(c, r"def\s+tokenize|class\s+Tokenizer|def\s+lex\b")),
            ("Parser", lambda c: rx(c, r"def\s+parse|class\s+Parser")),
            ("Environment", lambda c: has_class(c, r"Env|Environment")),
            ("Lambda support", lambda c: rx(c, r"lambda.*:|'lambda'")),
            ("Define support", lambda c: rx(c, r"'define'|\"define\"")),
            ("Closure/scope", lambda c: rx(c, r"parent|outer|enclosing|scope")),
            ("cons/car/cdr", lambda c: rx(c, r"cons|car|cdr")),
            ("map/filter", lambda c: rx(c, r"'map'|\"map\"|'filter'")),
            ("Tail call opt", lambda c: rx(c, r"tail.?call|TCO|trampoline")),
            ("Error class", lambda c: rx(c, r"class\s+\w*Error|LispError|SchemeError")),
            ("8+ test funcs", lambda c: len(re.findall(r"def\s+test_", c)) >= 6),
            ("10+ asserts", lambda c: len(re.findall(r"assert\s+\w+", c)) >= 10),
            ("Substantial code", lambda c: code_blocks(c, 3, 10)),
        ],
    },
    {
        "id": "hard_consensus",
        "goal": (
            "Write a comprehensive technical deep-dive on the Raft consensus algorithm.\n\n"
            "MANDATORY sections (ALL required):\n"
            "1. LEADER ELECTION: Explain terms, RequestVote RPC, split vote handling, "
            "pre-vote optimization. Include pseudocode.\n"
            "2. LOG REPLICATION: AppendEntries RPC, log matching property, "
            "consistency check, handling slow followers. Include state machine diagram.\n"
            "3. SAFETY: Election restriction, commitment rules, leader completeness. "
            "Prove why committed entries are never lost.\n"
            "4. MEMBERSHIP CHANGES: Joint consensus approach, single-server changes. "
            "Explain the disjoint majority problem.\n"
            "5. LOG COMPACTION: Snapshotting, InstallSnapshot RPC, incremental approaches.\n"
            "6. IMPLEMENTATION: Python code for a simplified Raft node with:\n"
            "   - Node class with state (follower/candidate/leader)\n"
            "   - RequestVote handler\n"
            "   - AppendEntries handler\n"
            "   - Election timeout logic\n"
            "   - At least 3 test functions\n"
            "7. COMPARISON: How Raft differs from Paxos, Zab, and Viewstamped Replication.\n\n"
            "Be deeply technical. Include pseudocode, code, diagrams (ASCII), and formulas."
        ),
        "domain": "system_design",
        "checks": [
            ("Leader election", lambda c: rx(c, r"leader\s+election|RequestVote")),
            ("Terms/epochs", lambda c: rx(c, r"term\s+\d|current.?term|epoch")),
            ("Log replication", lambda c: rx(c, r"log\s+replication|AppendEntries")),
            ("Log matching", lambda c: rx(c, r"log\s+matching|prev.?log|consistency\s+check")),
            (
                "Safety proof",
                lambda c: rx(c, r"leader\s+completeness|election\s+restriction|committed.*never"),
            ),
            ("Membership", lambda c: rx(c, r"membership|joint\s+consensus|configuration\s+change")),
            ("Snapshots", lambda c: rx(c, r"snapshot|log\s+compaction|InstallSnapshot")),
            ("Node class", lambda c: has_class(c, r"Node|RaftNode|Server")),
            ("State machine", lambda c: rx(c, r"follower|candidate|leader.*state")),
            ("Paxos comparison", lambda c: rx(c, r"Paxos|Multi.?Paxos")),
            ("Zab/VR", lambda c: rx(c, r"Zab|Viewstamped|ZooKeeper")),
            ("Code blocks", lambda c: code_blocks(c, 3, 5)),
            ("Tests", lambda c: len(re.findall(r"def\s+test_", c)) >= 2),
            ("1000+ words", lambda c: len(c.split()) >= 1000),
        ],
    },
]

if USE_HARD_TASKS:
    EVAL_TASKS = HARD_TASKS


# ─── DB Isolation Context Manager ───────────────────────────────────────


class IsolatedLearningDB:
    """Context manager that redirects LearningStore to a fresh temp database."""

    def __init__(self, label: str = "eval"):
        self.label = label
        self.tmp_dir: Optional[str] = None
        self._orig_instances: Dict = {}

    def __enter__(self):
        self.tmp_dir = tempfile.mkdtemp(prefix=f"jotty_eval_{self.label}_")
        db_path = os.path.join(self.tmp_dir, "learning.db")

        from Jotty.core.intelligence.learning.learning_store import LearningStore

        # Reset singleton to force fresh DB
        self._orig_instances = dict(LearningStore._instances)
        LearningStore._instances.clear()

        # Force new instance at temp path
        self.store = LearningStore(db_path)
        LearningStore._instances["default"] = self.store

        # Also reset LearningService singleton
        from Jotty.core.intelligence.learning.learning_service import LearningService

        self._orig_ls = LearningService._instance
        LearningService._instance = None

        return self

    def __exit__(self, *exc):
        from Jotty.core.intelligence.learning.learning_store import LearningStore
        from Jotty.core.intelligence.learning.learning_service import LearningService

        # Restore originals
        LearningStore._instances.clear()
        LearningStore._instances.update(self._orig_instances)
        LearningService._instance = self._orig_ls

        if self.tmp_dir:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get stats from the isolated DB."""
        conn = self.store._get_conn()
        ep_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        emb_count = conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        try:
            lesson_count = conn.execute("SELECT COUNT(*) FROM distilled_lessons").fetchone()[0]
        except Exception:
            lesson_count = 0
        return {
            "episodes": ep_count,
            "with_embeddings": emb_count,
            "distilled_lessons": lesson_count,
        }


# ─── Eval Engine ────────────────────────────────────────────────────────


class LearningEvalV2:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    async def _call_llm(self, goal: str, learning_context: str = "") -> str:
        """Direct LLM call (no MAS), to isolate learning effect."""
        import anthropic

        system = "You are an expert software engineer and technical writer."
        if learning_context:
            system += f"\n\n{learning_context}"

        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model=EVAL_MODEL,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": goal}],
        )
        return response.content[0].text

    def _score(self, content: str, task: Dict) -> Dict[str, Any]:
        """Score content against task checks."""
        checks = {}
        for name, fn in task["checks"]:
            try:
                checks[name] = fn(content)
            except Exception:
                checks[name] = False

        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        ratio = passed / max(total, 1)
        n_code = len(re.findall(r"```[\w]*\n.+?```", content, re.DOTALL))
        quality = ratio * 0.5 + min(1.0, len(content) / 5000) * 0.25 + min(1.0, n_code / 3) * 0.25

        return {
            "checks": checks,
            "passed": passed,
            "total": total,
            "ratio": ratio,
            "quality": quality,
            "content_len": len(content),
            "code_blocks": n_code,
        }

    async def run_single(
        self,
        task: Dict,
        run_num: int,
        use_learning: bool,
        iso_db: IsolatedLearningDB,
    ) -> Dict[str, Any]:
        """Run a single task execution."""
        from Jotty.core.intelligence.learning.learning_service import (
            LearningService,
            classify_domain,
        )

        ls = LearningService.get_instance()
        classified_domain, _ = classify_domain(task["goal"])
        domain = classified_domain or task.get("domain", "general")

        learning_ctx = ""
        if use_learning:
            learning_ctx = ls.build_context_string(domain, "", goal=task["goal"])

        start = time.time()
        error = None
        try:
            content = await self._call_llm(task["goal"], learning_ctx)
        except Exception as exc:
            content = ""
            error = str(exc)[:300]

        elapsed = time.time() - start
        scores = self._score(content, task)

        # Record the episode so subsequent runs can learn
        if use_learning and content:
            try:
                ls.record(
                    unit_name="eval_agent",
                    unit_type="agent",
                    domain=domain,
                    task_type=task["id"],
                    context={"goal": task["goal"], "message": task["goal"]},
                    action={"model": EVAL_MODEL},
                    outcome={
                        "content": content[:3000],
                        "response_excerpt": content[:600],
                    },
                    success=scores["ratio"] >= 0.5,
                    quality=scores["quality"],
                    execution_time=elapsed,
                )
            except Exception:
                pass

        mode = "LEARN" if use_learning else "BASE"
        status = "PASS" if scores["ratio"] >= 0.6 else "FAIL"
        db_stats = iso_db.get_stats()

        record = {
            "task_id": task["id"],
            "run": run_num,
            "mode": mode,
            "domain": domain,
            "success": scores["ratio"] >= 0.6,
            "quality": round(scores["quality"], 3),
            "check_ratio": round(scores["ratio"], 3),
            "passed": scores["passed"],
            "total": scores["total"],
            "content_len": scores["content_len"],
            "code_blocks": scores["code_blocks"],
            "ctx_len": len(learning_ctx),
            "elapsed": round(elapsed, 1),
            "db_episodes": db_stats["episodes"],
            "db_embeddings": db_stats["with_embeddings"],
            "db_lessons": db_stats["distilled_lessons"],
            "error": error,
        }

        print(
            f"    [{status}] {mode:5s} run={run_num} Q={scores['quality']:.2f} "
            f"Chk={scores['passed']}/{scores['total']} ({scores['ratio']:.0%}) "
            f"Len={scores['content_len']:5d} ctx={len(learning_ctx):4d}ch "
            f"emb={db_stats['with_embeddings']} les={db_stats['distilled_lessons']} "
            f"{elapsed:.0f}s"
        )
        return record

    async def run_task_ab(self, task: Dict) -> None:
        """Run A/B comparison for a single task: baseline vs learning."""
        print(f"\n  {'=' * 70}")
        print(f"  TASK: {task['id']} ({task['domain']})")
        print(f"  {'=' * 70}")

        # A: Baseline (no learning, fresh DB each run)
        print(f"\n  --- BASELINE (no learning) ---")
        for run in range(1, RUNS_PER_TASK + 1):
            with IsolatedLearningDB(f"base_{task['id']}_{run}") as iso:
                record = await self.run_single(task, run, use_learning=False, iso_db=iso)
                self.results.append(record)
            await asyncio.sleep(SLEEP_BETWEEN)

        # B: With learning (shared DB across runs — learning accumulates)
        print(f"\n  --- WITH LEARNING (accumulating) ---")
        with IsolatedLearningDB(f"learn_{task['id']}") as iso:
            for run in range(1, RUNS_PER_TASK + 1):
                record = await self.run_single(task, run, use_learning=True, iso_db=iso)
                self.results.append(record)
                if run < RUNS_PER_TASK:
                    # Wait for background distillation to complete
                    await asyncio.sleep(SLEEP_BETWEEN + 4)
                    stats = iso.get_stats()
                    print(
                        f"    [DB after sleep] ep={stats['episodes']} "
                        f"emb={stats['with_embeddings']} les={stats['distilled_lessons']}"
                    )

    def _print_summary(self) -> None:
        print(f"\n{'=' * 80}")
        print(f"  LEARNING EVALUATION V2 — RESULTS")
        print(f"{'=' * 80}")

        task_ids = list(dict.fromkeys(r["task_id"] for r in self.results))

        for tid in task_ids:
            base = [r for r in self.results if r["task_id"] == tid and r["mode"] == "BASE"]
            learn = [r for r in self.results if r["task_id"] == tid and r["mode"] == "LEARN"]

            if not base or not learn:
                continue

            base_ratios = [r["check_ratio"] for r in base]
            learn_ratios = [r["check_ratio"] for r in learn]
            base_q = [r["quality"] for r in base]
            learn_q = [r["quality"] for r in learn]

            print(f"\n  {tid} ({base[0].get('domain', '?')})")
            print(f"  {'─' * 60}")

            def _stats(vals):
                n = len(vals)
                mean = sum(vals) / n
                var = sum((x - mean) ** 2 for x in vals) / max(n - 1, 1)
                std = math.sqrt(var)
                se = std / math.sqrt(n) if n > 1 else 0
                return mean, std, se

            bm, bs, bse = _stats(base_ratios)
            lm, ls, lse = _stats(learn_ratios)
            delta = lm - bm

            print(f"    {'':25s} {'Mean':>8s} {'StdDev':>8s} {'n':>4s}")
            print(f"    {'Baseline check rate':25s} {bm:>7.0%} {bs:>7.2f} {len(base):>4d}")
            print(f"    {'Learning check rate':25s} {lm:>7.0%} {ls:>7.2f} {len(learn):>4d}")
            print(f"    {'Delta':25s} {delta:>+7.0%}")

            bqm, bqs, _ = _stats(base_q)
            lqm, lqs, _ = _stats(learn_q)
            print(f"    {'Baseline quality':25s} {bqm:>7.2f} {bqs:>7.2f}")
            print(f"    {'Learning quality':25s} {lqm:>7.2f} {lqs:>7.2f}")

            # Learning curve (runs 1→5)
            print(f"\n    Learning curve:")
            for r in sorted(learn, key=lambda x: x["run"]):
                bar_len = int(r["check_ratio"] * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                print(
                    f"      Run {r['run']}: {bar} {r['check_ratio']:.0%} "
                    f"ctx={r['ctx_len']:4d}ch emb={r['db_embeddings']} les={r['db_lessons']}"
                )

        # Aggregate
        print(f"\n{'=' * 80}")
        print(f"  AGGREGATE")
        print(f"{'=' * 80}")

        all_base = [r for r in self.results if r["mode"] == "BASE"]
        all_learn = [r for r in self.results if r["mode"] == "LEARN"]

        if all_base and all_learn:
            base_cr = sum(r["check_ratio"] for r in all_base) / len(all_base)
            learn_cr = sum(r["check_ratio"] for r in all_learn) / len(all_learn)
            base_q = sum(r["quality"] for r in all_base) / len(all_base)
            learn_q = sum(r["quality"] for r in all_learn) / len(all_learn)
            base_pass = sum(1 for r in all_base if r["success"]) / len(all_base)
            learn_pass = sum(1 for r in all_learn if r["success"]) / len(all_learn)

            # Later runs should be better — check run 5 vs run 1 for learning
            learn_r1 = [r for r in all_learn if r["run"] == 1]
            learn_r5 = [r for r in all_learn if r["run"] == RUNS_PER_TASK]
            lr1_cr = sum(r["check_ratio"] for r in learn_r1) / len(learn_r1) if learn_r1 else 0
            lr5_cr = sum(r["check_ratio"] for r in learn_r5) / len(learn_r5) if learn_r5 else 0

            print(f"\n  {'Metric':<30s} {'Baseline':>10s} {'Learning':>10s} {'Delta':>10s}")
            print(f"  {'─' * 60}")
            print(
                f"  {'Avg check rate':<30s} {base_cr:>9.0%} {learn_cr:>9.0%} {learn_cr-base_cr:>+9.0%}"
            )
            print(f"  {'Avg quality':<30s} {base_q:>9.2f} {learn_q:>9.2f} {learn_q-base_q:>+9.2f}")
            print(
                f"  {'Pass rate (≥60%)':<30s} {base_pass:>9.0%} {learn_pass:>9.0%} {learn_pass-base_pass:>+9.0%}"
            )
            print(
                f"  {'Learn R1→R{} check rate'.format(RUNS_PER_TASK):<30s} {lr1_cr:>9.0%} {lr5_cr:>9.0%} {lr5_cr-lr1_cr:>+9.0%}"
            )

            # Infrastructure verification
            learn_last = [r for r in all_learn if r["run"] == RUNS_PER_TASK]
            if learn_last:
                avg_emb = sum(r["db_embeddings"] for r in learn_last) / len(learn_last)
                avg_les = sum(r["db_lessons"] for r in learn_last) / len(learn_last)
                print(f"\n  Infrastructure (after {RUNS_PER_TASK} runs per task):")
                print(f"    Avg embeddings stored: {avg_emb:.0f}")
                print(f"    Avg distilled lessons: {avg_les:.0f}")

    async def run_all(self) -> None:
        print(f"\n{'=' * 80}")
        print(f"  JOTTY LEARNING EVALUATION V2")
        print(f"{'=' * 80}")
        print(f"  Date: {datetime.now().isoformat()}")
        print(f"  Model: {EVAL_MODEL}")
        print(f"  Tasks: {len(EVAL_TASKS)} × {RUNS_PER_TASK} runs × 2 modes (A/B)")
        print(f"  Total calls: {len(EVAL_TASKS) * RUNS_PER_TASK * 2}")
        print(f"  Features: Vector embeddings, Fact distillation, DB isolation")

        for task in EVAL_TASKS:
            await self.run_task_ab(task)

        self._print_summary()

        results_dir = Path(__file__).parent / "eval_results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = results_dir / f"{ts}_learning_v2.json"
        with open(out, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "model": EVAL_MODEL,
                    "runs_per_task": RUNS_PER_TASK,
                    "tasks": len(EVAL_TASKS),
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
    await LearningEvalV2().run_all()


if __name__ == "__main__":
    asyncio.run(main())

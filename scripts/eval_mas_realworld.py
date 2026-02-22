#!/usr/bin/env python3
"""
Real-World Multi-Agent System (MAS) Evaluation
================================================

Tests every multi-agent path in Jotty with real LLM calls.
Each test validates: instantiation, execution, output quality, learning.

Usage:
    python scripts/eval_mas_realworld.py              # all tests
    python scripts/eval_mas_realworld.py --test coding # single test
    python scripts/eval_mas_realworld.py --fast        # shorter timeouts
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

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("eval_mas")
logger.setLevel(logging.INFO)


# ─── Result Types ────────────────────────────────────────────────────────────


@dataclass
class TestResult:
    name: str
    swarm: str
    passed: bool
    elapsed_s: float
    output_len: int = 0
    agents_detected: List[str] = field(default_factory=list)
    learning_recorded: bool = False
    error: str = ""
    quality_notes: List[str] = field(default_factory=list)
    score: float = 0.0  # 0-10


@dataclass
class EvalReport:
    timestamp: str
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    total_time_s: float = 0.0
    results: List[Dict[str, Any]] = field(default_factory=list)
    overall_score: float = 0.0


# ─── Helpers ─────────────────────────────────────────────────────────────────


def extract_content(result: Any) -> str:
    """Extract the best text content from any result type."""
    if result is None:
        return ""

    content = ""
    # Try nested output first (EpisodeResult wrapping AgenticExecutionResult)
    inner = getattr(result, "output", None)
    if inner and not isinstance(inner, str):
        for attr in ("final_output", "output", "content", "code", "report"):
            val = getattr(inner, attr, None)
            if val and isinstance(val, str) and len(val) > len(content):
                content = val

    # Try direct attributes
    for attr in ("final_output", "output", "content", "code", "report", "analysis"):
        val = getattr(result, attr, None)
        if val and isinstance(val, str) and len(val) > len(content):
            content = val

    # Try dict-style output
    out_dict = getattr(result, "output", None)
    if isinstance(out_dict, dict):
        for key in ("code", "report", "content", "analysis", "architecture"):
            val = out_dict.get(key, "")
            if isinstance(val, str) and len(val) > len(content):
                content = val

    if not content:
        s = str(result)
        if len(s) > 50:
            content = s

    return content


def score_output(content: str, *, min_len: int = 200, keywords: List[str] = None) -> tuple:
    """Score output quality. Returns (score: float, notes: list[str])."""
    notes = []
    score = 0.0

    if not content:
        return 0.0, ["Empty output"]

    # Length scoring (0-3 points)
    if len(content) >= min_len * 3:
        score += 3.0
    elif len(content) >= min_len:
        score += 2.0
        notes.append(f"Output adequate ({len(content)} chars)")
    elif len(content) >= 50:
        score += 1.0
        notes.append(f"Output short ({len(content)} chars)")
    else:
        notes.append(f"Output too short ({len(content)} chars)")

    # Structure scoring (0-3 points)
    has_headings = any(line.startswith("#") for line in content.split("\n"))
    has_code_blocks = "```" in content
    has_bullets = any(line.strip().startswith(("- ", "* ", "• ")) for line in content.split("\n"))
    has_numbered = any(
        line.strip()[:2].rstrip(".").isdigit() for line in content.split("\n") if line.strip()
    )
    structure_score = sum([has_headings, has_code_blocks, has_bullets, has_numbered])
    score += min(structure_score, 3)
    if structure_score == 0:
        notes.append("No structured formatting")

    # Keyword relevance (0-2 points)
    if keywords:
        lower_content = content.lower()
        matched = sum(1 for kw in keywords if kw.lower() in lower_content)
        relevance = matched / len(keywords)
        score += relevance * 2
        if relevance < 0.3:
            notes.append(f"Low keyword relevance ({matched}/{len(keywords)})")

    # Coherence (0-2 points) — no repeated blocks, reasonable sentence structure
    substantive_lines = [line.strip() for line in content.split("\n") if len(line.strip()) > 30]
    unique_substantive = set(substantive_lines)
    if len(substantive_lines) > 5 and len(unique_substantive) < len(substantive_lines) * 0.7:
        notes.append(
            f"Repetition detected ({len(unique_substantive)}/{len(substantive_lines)} unique)"
        )
        score += 1.0
    else:
        score += 2.0

    if not notes:
        notes.append("Strong output quality")

    return min(score, 10.0), notes


# ─── Test Functions ──────────────────────────────────────────────────────────


async def test_coding_swarm(timeout: int) -> TestResult:
    """CodingSwarm: Build a real data structure with tests."""
    from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator
    from Jotty.core.intelligence.orchestration.swarms.coding_swarm.swarm import CodingSwarm
    from Jotty.core.intelligence.orchestration.swarms.coding_swarm.types import CodingConfig

    orch = Orchestrator()
    swarm = CodingSwarm(CodingConfig())

    start = time.time()
    result = await asyncio.wait_for(
        orch.run(
            "Build a Python LRU cache class with get(), put(), and evict() methods. "
            "Include type hints, docstrings, and 5 unit tests using pytest.",
            swarm=swarm,
            learn=True,
        ),
        timeout=timeout,
    )
    elapsed = time.time() - start

    content = extract_content(result)
    score, notes = score_output(
        content,
        min_len=500,
        keywords=["class", "def get", "def put", "pytest", "assert", "def test_"],
    )

    return TestResult(
        name="CodingSwarm: LRU Cache",
        swarm="coding",
        passed=len(content) > 300 and score >= 4.0,
        elapsed_s=elapsed,
        output_len=len(content),
        learning_recorded=getattr(result, "success", False),
        quality_notes=notes,
        score=score,
    )


async def test_review_swarm(timeout: int) -> TestResult:
    """ReviewSwarm: Review a piece of code for quality issues."""
    from Jotty.core.intelligence.orchestration.swarms.templates.review_swarm import ReviewSwarm

    swarm = ReviewSwarm()

    code = """
import sqlite3

def get_user(user_id):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
    result = cursor.fetchone()
    return result

def delete_all_users():
    conn = sqlite3.connect("users.db")
    conn.execute("DELETE FROM users")
    conn.commit()

def save_password(username, password):
    conn = sqlite3.connect("users.db")
    conn.execute(f"INSERT INTO users VALUES ('{username}', '{password}')")
    conn.commit()
"""

    start = time.time()
    result = await asyncio.wait_for(
        swarm.execute(task=code, context="Production user management module", language="python"),
        timeout=timeout,
    )
    elapsed = time.time() - start

    content = extract_content(result)
    score, notes = score_output(
        content,
        min_len=300,
        keywords=["sql injection", "password", "hash", "connection", "security", "vulnerability"],
    )

    return TestResult(
        name="ReviewSwarm: Security Audit",
        swarm="review",
        passed=len(content) > 200 and score >= 3.0,
        elapsed_s=elapsed,
        output_len=len(content),
        quality_notes=notes,
        score=score,
    )


async def test_testing_swarm(timeout: int) -> TestResult:
    """TestingSwarm: Generate tests for a function."""
    from Jotty.core.intelligence.orchestration.swarms.templates.testing_swarm import TestingSwarm

    swarm = TestingSwarm()

    code = """
from datetime import datetime, timedelta
from typing import List, Optional

class Scheduler:
    def __init__(self):
        self._events: List[dict] = []

    def add_event(self, name: str, start: datetime, duration_minutes: int) -> bool:
        end = start + timedelta(minutes=duration_minutes)
        for evt in self._events:
            if start < evt["end"] and end > evt["start"]:
                return False  # conflict
        self._events.append({"name": name, "start": start, "end": end})
        return True

    def get_events(self, date: datetime) -> List[dict]:
        return [e for e in self._events if e["start"].date() == date.date()]

    def cancel_event(self, name: str) -> bool:
        for i, evt in enumerate(self._events):
            if evt["name"] == name:
                self._events.pop(i)
                return True
        return False

    def next_available_slot(self, date: datetime, duration_minutes: int) -> Optional[datetime]:
        day_events = sorted(self.get_events(date), key=lambda e: e["start"])
        current = date.replace(hour=9, minute=0, second=0, microsecond=0)
        end_of_day = date.replace(hour=17, minute=0, second=0, microsecond=0)
        for evt in day_events:
            if current + timedelta(minutes=duration_minutes) <= evt["start"]:
                return current
            current = max(current, evt["end"])
        if current + timedelta(minutes=duration_minutes) <= end_of_day:
            return current
        return None
"""

    start = time.time()
    result = await asyncio.wait_for(
        swarm.execute(task=code, language="python"),
        timeout=timeout,
    )
    elapsed = time.time() - start

    content = extract_content(result)
    score, notes = score_output(
        content,
        min_len=400,
        keywords=["def test_", "assert", "Scheduler", "add_event", "conflict", "cancel", "edge"],
    )

    return TestResult(
        name="TestingSwarm: Scheduler Tests",
        swarm="testing",
        passed=len(content) > 200 and score >= 3.0,
        elapsed_s=elapsed,
        output_len=len(content),
        quality_notes=notes,
        score=score,
    )


async def test_devops_swarm(timeout: int) -> TestResult:
    """DevOpsSwarm: Design infrastructure for a real-world app."""
    from Jotty.core.intelligence.orchestration.swarms.templates.devops_swarm import DevOpsSwarm

    swarm = DevOpsSwarm()

    start = time.time()
    result = await asyncio.wait_for(
        swarm.execute(
            task="payment-gateway",
            app_type="api",
            language="python",
            requirements="PCI-DSS compliant payment processing API with Stripe integration, "
            "Redis caching, PostgreSQL, rate limiting, webhook handling, "
            "and auto-scaling to handle 10K TPS.",
            scale="large",
        ),
        timeout=timeout,
    )
    elapsed = time.time() - start

    content = extract_content(result)
    score, notes = score_output(
        content,
        min_len=500,
        keywords=[
            "docker",
            "kubernetes",
            "postgres",
            "redis",
            "scaling",
            "monitoring",
            "CI",
            "deploy",
        ],
    )

    return TestResult(
        name="DevOpsSwarm: Payment Gateway Infra",
        swarm="devops",
        passed=len(content) > 300 and score >= 3.0,
        elapsed_s=elapsed,
        output_len=len(content),
        quality_notes=notes,
        score=score,
    )


async def test_idea_writer_swarm(timeout: int) -> TestResult:
    """IdeaWriterSwarm: Write a technical blog post."""
    from Jotty.core.intelligence.orchestration.swarms.templates.idea_writer_swarm import (
        IdeaWriterSwarm,
    )

    swarm = IdeaWriterSwarm()

    start = time.time()
    result = await asyncio.wait_for(
        swarm.execute(
            task="Why Every Engineering Team Needs a Chaos Engineering Practice: "
            "Lessons from Netflix, Amazon, and Google",
        ),
        timeout=timeout,
    )
    elapsed = time.time() - start

    content = extract_content(result)
    score, notes = score_output(
        content,
        min_len=800,
        keywords=["chaos", "netflix", "resilience", "failure", "production", "testing", "blast"],
    )

    return TestResult(
        name="IdeaWriterSwarm: Chaos Engineering Blog",
        swarm="idea_writer",
        passed=len(content) > 500 and score >= 4.0,
        elapsed_s=elapsed,
        output_len=len(content),
        quality_notes=notes,
        score=score,
    )


async def test_auto_detect_coding(timeout: int) -> TestResult:
    """Auto-detect: Should route a coding task through the agent/LLM path."""
    from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

    orch = Orchestrator()

    start = time.time()
    result = await asyncio.wait_for(
        orch.run(
            "Write a Python decorator called @retry that retries a function up to N times "
            "with exponential backoff. Include type hints and 3 unit tests.",
            learn=True,
        ),
        timeout=timeout,
    )
    elapsed = time.time() - start

    content = extract_content(result)
    score, notes = score_output(
        content,
        min_len=300,
        keywords=["def retry", "decorator", "backoff", "def test_", "assert", "except"],
    )

    return TestResult(
        name="Auto-Detect: Retry Decorator",
        swarm="auto",
        passed=len(content) > 200 and score >= 3.0,
        elapsed_s=elapsed,
        output_len=len(content),
        learning_recorded=getattr(result, "success", False),
        quality_notes=notes,
        score=score,
    )


async def test_auto_detect_analysis(timeout: int) -> TestResult:
    """Auto-detect: Should route an analysis task through the agent/LLM path."""
    from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

    orch = Orchestrator()

    start = time.time()
    result = await asyncio.wait_for(
        orch.run(
            "Compare microservices vs monolithic architecture across 6 dimensions: "
            "scalability, deployment complexity, team autonomy, data consistency, "
            "operational cost, and debugging difficulty. For each, give a concrete "
            "example from a real company.",
            learn=True,
        ),
        timeout=timeout,
    )
    elapsed = time.time() - start

    content = extract_content(result)
    score, notes = score_output(
        content,
        min_len=500,
        keywords=[
            "scalability",
            "deployment",
            "microservice",
            "monolith",
            "consistency",
            "debugging",
        ],
    )

    return TestResult(
        name="Auto-Detect: Architecture Analysis",
        swarm="auto",
        passed=len(content) > 300 and score >= 3.0,
        elapsed_s=elapsed,
        output_len=len(content),
        learning_recorded=getattr(result, "success", False),
        quality_notes=notes,
        score=score,
    )


async def test_orchestrator_chat(timeout: int) -> TestResult:
    """Orchestrator.chat(): Conversational multi-turn path."""
    from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

    orch = Orchestrator()

    start = time.time()
    r1 = await asyncio.wait_for(
        orch.chat("What are the SOLID principles in software engineering?"),
        timeout=timeout,
    )

    c1 = extract_content(r1)
    if len(c1) < 100:
        return TestResult(
            name="Orchestrator.chat(): Multi-turn",
            swarm="chat",
            passed=False,
            elapsed_s=time.time() - start,
            output_len=len(c1),
            error="First turn too short",
            score=0.0,
        )

    r2 = await asyncio.wait_for(
        orch.chat(
            "Now give me a Python code example that violates the Open/Closed Principle, "
            "then refactor it to follow it."
        ),
        timeout=timeout,
    )
    elapsed = time.time() - start

    c2 = extract_content(r2)
    combined = c1 + "\n" + c2

    score, notes = score_output(
        combined,
        min_len=500,
        keywords=["SOLID", "open", "closed", "class", "def", "refactor"],
    )

    return TestResult(
        name="Orchestrator.chat(): Multi-turn",
        swarm="chat",
        passed=len(c2) > 200 and score >= 3.0,
        elapsed_s=elapsed,
        output_len=len(combined),
        quality_notes=notes,
        score=score,
    )


async def test_learning_persistence(timeout: int) -> TestResult:
    """Verify learning system actually records and retrieves episodes."""
    from Jotty.core.intelligence.learning.learning_service import LearningService

    learning = LearningService.get_instance()
    start = time.time()
    notes = []

    # Record a synthetic episode via the real API
    try:
        episode_id = learning.record(
            unit_name="eval_test_agent",
            unit_type="agent",
            domain="eval_test",
            task_type="coding",
            context={"goal": "Build a rate limiter"},
            action={"tool": "code_generation"},
            outcome={"output_length": 5000, "has_code": True},
            success=True,
            quality=0.85,
            execution_time=30.0,
            metadata={"test": True, "eval_run": True},
        )
        notes.append(f"Episode recorded: {episode_id}")
    except Exception as e:
        return TestResult(
            name="Learning: Episode Persistence",
            swarm="learning",
            passed=False,
            elapsed_s=time.time() - start,
            error=f"record() failed: {e}",
            score=0.0,
        )

    # Retrieve context
    try:
        ctx = learning.build_context_string(
            domain="eval_test", task_type="coding", goal="Build a cache"
        )
        if ctx:
            notes.append(f"Context retrieved ({len(ctx)} chars)")
        else:
            notes.append("No context returned (may need more episodes)")
    except Exception as e:
        notes.append(f"Context retrieval failed: {e}")

    elapsed = time.time() - start
    has_record = any("Episode recorded" in n for n in notes)

    return TestResult(
        name="Learning: Episode Persistence",
        swarm="learning",
        passed=has_record,
        elapsed_s=elapsed,
        learning_recorded=has_record,
        quality_notes=notes,
        score=8.0 if has_record else 2.0,
    )


async def test_circuit_breaker_adaptive(timeout: int) -> TestResult:
    """Verify circuit breaker defers unhealthy providers."""
    from Jotty.core.infrastructure.utils.provider_health import ProviderHealthManager

    health = ProviderHealthManager()
    start = time.time()
    notes = []

    # Fresh manager: should be healthy
    assert health.is_healthy("test_provider"), "Fresh provider should be healthy"
    notes.append("Fresh provider is healthy")

    # Record 2 failures (threshold)
    health.record_failure("test_provider", Exception("simulated 400"))
    health.record_failure("test_provider", Exception("simulated 400"))

    if not health.is_healthy("test_provider"):
        notes.append("Circuit opened after 2 failures")
    else:
        return TestResult(
            name="Circuit Breaker: Adaptive Deferral",
            swarm="infra",
            passed=False,
            elapsed_s=time.time() - start,
            error="Circuit breaker did not open after threshold failures",
            score=0.0,
        )

    # Verify stats
    stats = health.get_stats()
    tp = stats.get("test_provider", {})
    notes.append(
        f"Backoff: level {tp.get('backoff_multiplier')}, timeout {tp.get('current_timeout')}s"
    )

    elapsed = time.time() - start

    return TestResult(
        name="Circuit Breaker: Adaptive Deferral",
        swarm="infra",
        passed=True,
        elapsed_s=elapsed,
        quality_notes=notes,
        score=10.0,
    )


# ─── Test Registry ───────────────────────────────────────────────────────────

ALL_TESTS = {
    "coding": ("1. CodingSwarm (explicit)", test_coding_swarm),
    "review": ("2. ReviewSwarm (code audit)", test_review_swarm),
    "testing": ("3. TestingSwarm (test gen)", test_testing_swarm),
    "devops": ("4. DevOpsSwarm (infra design)", test_devops_swarm),
    "writer": ("5. IdeaWriterSwarm (blog)", test_idea_writer_swarm),
    "auto_code": ("6. Auto-Detect → coding", test_auto_detect_coding),
    "auto_analysis": ("7. Auto-Detect → analysis", test_auto_detect_analysis),
    "chat": ("8. Orchestrator.chat() multi-turn", test_orchestrator_chat),
    "learning": ("9. Learning persistence", test_learning_persistence),
    "circuit": ("10. Circuit breaker adaptive", test_circuit_breaker_adaptive),
}


# ─── Runner ──────────────────────────────────────────────────────────────────


def print_banner():
    print("\n" + "=" * 72)
    print("  JOTTY MAS REAL-WORLD EVALUATION")
    print("  Multi-Agent System End-to-End Tests with Real LLM Calls")
    print("=" * 72)


def print_result(idx: int, tr: TestResult):
    status = "✅ PASS" if tr.passed else "❌ FAIL"
    print(f"\n{'─' * 60}")
    print(f"  [{idx}] {tr.name}")
    print(f"  Status: {status}  |  Score: {tr.score:.1f}/10  |  Time: {tr.elapsed_s:.1f}s")
    print(f"  Output: {tr.output_len:,} chars  |  Swarm: {tr.swarm}")
    if tr.error:
        print(f"  Error: {tr.error[:200]}")
    for note in tr.quality_notes[:5]:
        print(f"    • {note}")


async def run_eval(test_filter: Optional[str], fast: bool):
    print_banner()

    timeout = 120 if fast else 300

    tests_to_run = {}
    if test_filter:
        for key in test_filter.split(","):
            key = key.strip()
            if key in ALL_TESTS:
                tests_to_run[key] = ALL_TESTS[key]
            else:
                print(f"Unknown test: {key}. Available: {', '.join(ALL_TESTS.keys())}")
                return
    else:
        tests_to_run = ALL_TESTS

    print(
        f"\n  Tests: {len(tests_to_run)}  |  Timeout: {timeout}s  |  Mode: {'fast' if fast else 'full'}"
    )
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results: List[TestResult] = []
    eval_start = time.time()

    for idx, (key, (label, test_fn)) in enumerate(tests_to_run.items(), 1):
        print(f"\n{'━' * 60}")
        print(f"  Running {label} ...")

        try:
            tr = await test_fn(timeout)
        except asyncio.TimeoutError:
            tr = TestResult(
                name=label,
                swarm=key,
                passed=False,
                elapsed_s=timeout,
                error=f"Timed out after {timeout}s",
                score=0.0,
            )
        except Exception as e:
            tr = TestResult(
                name=label,
                swarm=key,
                passed=False,
                elapsed_s=0.0,
                error=f"{type(e).__name__}: {e}",
                score=0.0,
            )
            traceback.print_exc()

        results.append(tr)
        print_result(idx, tr)

    total_time = time.time() - eval_start

    # ── Summary ──
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    avg_score = sum(r.score for r in results) / len(results) if results else 0

    print(f"\n{'═' * 72}")
    print(f"  FINAL RESULTS")
    print(f"{'═' * 72}")
    print(f"  Passed: {passed}/{len(results)}  |  Failed: {failed}")
    print(f"  Average Score: {avg_score:.1f}/10")
    print(f"  Total Time: {total_time:.1f}s")
    print()

    for r in results:
        status = "✅" if r.passed else "❌"
        print(f"  {status} {r.name:<45} {r.score:.1f}/10  {r.elapsed_s:.0f}s")

    print(f"\n{'═' * 72}")

    # ── Save results ──
    report = EvalReport(
        timestamp=datetime.now().isoformat(),
        total_tests=len(results),
        passed=passed,
        failed=failed,
        total_time_s=total_time,
        overall_score=avg_score,
        results=[asdict(r) for r in results],
    )

    results_dir = Path("scripts/eval_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    outfile = results_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_mas_realworld.json"
    outfile.write_text(json.dumps(asdict(report), indent=2, default=str))
    print(f"\n  Results saved: {outfile}")

    return report


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jotty MAS Real-World Evaluation")
    parser.add_argument("--test", type=str, default=None, help="Comma-separated test keys to run")
    parser.add_argument("--fast", action="store_true", help="Use shorter timeouts (120s)")
    args = parser.parse_args()

    report = asyncio.run(run_eval(args.test, args.fast))
    sys.exit(0 if report and report.failed == 0 else 1)

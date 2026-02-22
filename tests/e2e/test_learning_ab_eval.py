#!/usr/bin/env python3
"""
A/B Evaluation: Does learning context actually improve LLM output quality?

Methodology:
  Phase 1: SEED — Record 30 episodes to build real learning patterns
  Phase 2: BASELINE — Run 10 tasks WITHOUT learning context
  Phase 3: TREATMENT — Run the SAME 10 tasks WITH learning context
  Phase 4: COMPARE — Measure quality on 8 dimensions per task
  Phase 5: STATISTICAL SUMMARY — Which approach wins, by how much

Each task runs twice (baseline + treatment) on the same model (Haiku).
analyze_response() scores both outputs on identical dimensions.

Cost estimate: ~$1.50 (30 seed + 20 eval = 50 Haiku calls)

Run:  python tests/e2e/test_learning_ab_eval.py
"""

import json
import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pathlib import Path

env_file = Path(__file__).parents[2] / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if v and k not in os.environ:
                    os.environ[k] = v

import anthropic

from Jotty.core.intelligence.learning.learning_service import LearningService, analyze_response
from Jotty.core.intelligence.learning.learning_store import LearningStore


def banner(text: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def run_llm(prompt: str, system: str = "", max_tokens: int = 2048) -> str:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return getattr(resp.content[0], "text", "")


# ═════════════════════════════════════════════════════════════════════
# SEED TASKS — Build learning patterns from diverse coding tasks
# ═════════════════════════════════════════════════════════════════════

SEED_TASKS = [
    # High-quality coding tasks to build strong patterns
    "Write a Python linked list with insert, delete, search, and reverse methods",
    "Implement a Python priority queue using a binary heap with push/pop/peek",
    "Write a Python decorator that retries a function N times on exception",
    "Create a Python context manager for database transactions with rollback",
    "Implement a Python LRU cache from scratch using OrderedDict",
    "Write a Python async rate limiter using token bucket algorithm",
    "Implement a Python trie data structure for autocomplete",
    "Write a Python graph class with BFS, DFS, and shortest path",
    "Create a Python command pattern implementation with undo/redo",
    "Write a Python observer pattern for event-driven systems",
    # Research tasks
    "Explain how consensus algorithms work in distributed systems (Raft, Paxos)",
    "Compare microservices vs monolith architectures with real-world tradeoffs",
    "Explain CAP theorem with concrete examples of each tradeoff",
    "Describe how garbage collection works in different languages",
    "Explain how database indexing works (B-trees, hash indexes, bitmap)",
    # System design tasks
    "Design a URL shortener system like bit.ly with all components",
    "Design a rate limiter for an API gateway",
    "Design a real-time chat system architecture",
    "Design a distributed cache system like Redis",
    "Design a notification system that handles millions of users",
    # Lower quality (to create contrast)
    "What is a variable?",
    "What is a list?",
    "Define a function",
    "What is an object?",
    "What is a class?",
    # Failures
    "Write a quantum computer simulator in 10 lines",
    "Build a full operating system kernel in Python",
    "Create a complete 3D game engine in one function",
    "Implement GPT-4 from scratch in 20 lines",
    "Build a self-driving car AI in pure Python",
]

# ═════════════════════════════════════════════════════════════════════
# EVAL TASKS — These run twice: baseline (no context) and treatment (with context)
# ═════════════════════════════════════════════════════════════════════

EVAL_TASKS = [
    {
        "prompt": "Write a Python binary search tree with insert, search, delete, and in-order traversal",
        "domain": "coding",
        "task_type": "implementation",
        "system_base": "You are a Python programmer.",
    },
    {
        "prompt": "Implement a Python hash map from scratch with collision handling using chaining",
        "domain": "coding",
        "task_type": "implementation",
        "system_base": "You are a Python programmer.",
    },
    {
        "prompt": "Write a Python merge sort with both recursive and iterative implementations",
        "domain": "coding",
        "task_type": "algorithm",
        "system_base": "You are a Python programmer.",
    },
    {
        "prompt": "Implement a Python state machine for parsing a simple arithmetic expression",
        "domain": "coding",
        "task_type": "implementation",
        "system_base": "You are a Python programmer.",
    },
    {
        "prompt": "Write a Python thread pool executor from scratch using threading primitives",
        "domain": "coding",
        "task_type": "implementation",
        "system_base": "You are a Python programmer.",
    },
    {
        "prompt": "Explain how B-tree indexing works in databases with insertion and search algorithms",
        "domain": "research",
        "task_type": "explanation",
        "system_base": "You are a technical writer.",
    },
    {
        "prompt": "Compare eventual consistency vs strong consistency in distributed databases",
        "domain": "research",
        "task_type": "comparison",
        "system_base": "You are a technical writer.",
    },
    {
        "prompt": "Design a distributed task queue system like Celery with architecture diagram",
        "domain": "system_design",
        "task_type": "design",
        "system_base": "You are a system architect.",
    },
    {
        "prompt": "Write a Python decorator library with memoize, retry, timeout, and validate",
        "domain": "coding",
        "task_type": "implementation",
        "system_base": "You are a Python programmer.",
    },
    {
        "prompt": "Implement a Python event sourcing system with event store, projections, and snapshots",
        "domain": "coding",
        "task_type": "implementation",
        "system_base": "You are a Python programmer.",
    },
]


def score_response(output: str, goal: str) -> dict:
    """Score a response on multiple dimensions using analyze_response."""
    analysis = analyze_response(output, goal)
    return {
        "quality_score": analysis.get("quality_score", 0),
        "structure_score": analysis.get("structure_score", 0),
        "goal_coverage": analysis.get("goal_coverage", 0),
        "word_count": analysis.get("word_count", 0),
        "code_lines": analysis.get("code_lines", 0),
        "code_block_count": analysis.get("code_block_count", 0),
        "has_code": analysis.get("has_code", False),
        "has_headings": analysis.get("has_headings", False),
        "has_tests": analysis.get("has_tests", False),
        "has_class": analysis.get("has_class", False),
        "has_citations": analysis.get("has_citations", False),
        "assertion_count": analysis.get("assertion_count", 0),
        "reasoning_density": analysis.get("reasoning_density", 0),
        "output_length": len(output),
    }


def main() -> None:
    tmpdir = tempfile.mkdtemp(prefix="jotty_ab_eval_")
    db_path = os.path.join(tmpdir, "ab_eval.db")
    print(f"DB: {db_path}")

    store = LearningStore(db_path)
    LearningService.reset_instance()
    svc = LearningService.get_instance(store)
    svc._pattern_extraction_interval = 1
    svc._auto_transfer_interval = 10
    svc._auto_reflect_interval = 5

    total_cost = 0.0

    # ── PHASE 1: SEED — Build learning patterns ──────────────────────
    banner("PHASE 1: Seeding Learning System (30 tasks)")

    for i, task in enumerate(SEED_TASKS):
        domain = (
            "coding"
            if i < 10
            else ("research" if i < 15 else ("system_design" if i < 20 else "coding"))
        )
        is_failure = i >= 25
        is_low = 20 <= i < 25

        system = (
            "You are an expert Python programmer. Write complete, well-documented "
            "code with type hints, docstrings, tests (def test_*), and assertions. "
            "Use ## headings to organize. Analyze time/space complexity."
            if domain == "coding"
            else "You are a technical expert. Provide thorough analysis with headings, "
            "examples, and citations where appropriate."
        )

        t0 = time.time()
        output = run_llm(task, system=system)
        elapsed = time.time() - t0
        cost = 0.04
        total_cost += cost

        quality = 0.3 if is_low else (0.2 if is_failure else 0.88)
        success = not is_failure

        svc.record(
            unit_name=f"seed-{i}",
            unit_type="agent",
            domain=domain,
            task_type="implementation" if domain == "coding" else "analysis",
            context={"goal": task},
            action={
                "model": "claude-haiku-4-5-20251001",
                "tools": ["code_gen", "test_runner"] if domain == "coding" else ["web_search"],
                "temperature": 0.3,
            },
            outcome={"content": output},
            success=success,
            quality=quality,
            execution_time=elapsed,
            cost=cost,
        )

        status = "FAIL" if is_failure else ("LOW" if is_low else "OK")
        print(f"  [{status}] {i+1:2d}/30  {task[:55]}...  {len(output):4d} chars  {elapsed:.1f}s")

    # Show what was learned
    print(f"\n  Episodes: {store.get_episode_count()}")
    for d in ["coding", "research", "system_design"]:
        pats = store.get_patterns(domain=d)
        sr, total = store.get_success_rate(domain=d)
        print(f"  {d:18s}  patterns={len(pats):2d}  success={sr:.0%} ({total} eps)")

    # Show patterns
    coding_pats = store.get_patterns(domain="coding")
    print(f"\n  Learned coding patterns:")
    for p in coding_pats[:8]:
        print(f"    [{p.pattern_type}] {p.description[:70]}")

    # Show what context would be injected
    ctx_preview = svc.build_context_string("coding", "implementation", "eval-agent")
    print(f"\n  Context that will be injected ({len(ctx_preview)} chars):")
    for line in ctx_preview.split("\n")[:20]:
        print(f"    {line}")

    # ── PHASE 2 & 3: A/B EVALUATION ──────────────────────────────────
    banner("PHASE 2 & 3: A/B Evaluation (10 tasks x 2 runs each)")

    results = []

    for i, task in enumerate(EVAL_TASKS):
        print(f"\n  Task {i+1}/10: {task['prompt'][:60]}...")

        # ── BASELINE: No learning context ──
        t0 = time.time()
        baseline_output = run_llm(task["prompt"], system=task["system_base"])
        baseline_time = time.time() - t0
        total_cost += 0.04

        baseline_scores = score_response(baseline_output, task["prompt"])
        print(
            f"    BASELINE: {len(baseline_output):4d} chars  "
            f"Q={baseline_scores['quality_score']:.2f}  "
            f"struct={baseline_scores['structure_score']:.1f}  "
            f"coverage={baseline_scores['goal_coverage']:.0%}  "
            f"code={baseline_scores['code_lines']}L  "
            f"{baseline_time:.1f}s"
        )

        # ── TREATMENT: With learning context ──
        learning_ctx = svc.build_context_string(
            task["domain"], task["task_type"], "eval-agent", goal=task["prompt"]
        )

        treatment_system = task["system_base"]
        if learning_ctx:
            treatment_system += (
                "\n\n=== LEARNING FROM PAST EXPERIENCE ===\n"
                + learning_ctx
                + "\n=== END LEARNING ==="
            )

        t0 = time.time()
        treatment_output = run_llm(task["prompt"], system=treatment_system)
        treatment_time = time.time() - t0
        total_cost += 0.04

        treatment_scores = score_response(treatment_output, task["prompt"])
        print(
            f"    LEARNED:  {len(treatment_output):4d} chars  "
            f"Q={treatment_scores['quality_score']:.2f}  "
            f"struct={treatment_scores['structure_score']:.1f}  "
            f"coverage={treatment_scores['goal_coverage']:.0%}  "
            f"code={treatment_scores['code_lines']}L  "
            f"{treatment_time:.1f}s"
        )

        # Record treatment to keep learning
        svc.record(
            unit_name="eval-agent",
            unit_type="agent",
            domain=task["domain"],
            task_type=task["task_type"],
            context={"goal": task["prompt"]},
            action={"model": "claude-haiku-4-5-20251001", "tools": ["code_gen"]},
            outcome={"content": treatment_output},
            success=True,
            quality=treatment_scores["quality_score"],
            execution_time=treatment_time,
        )

        # Compute deltas
        deltas = {}
        for metric in [
            "quality_score",
            "structure_score",
            "goal_coverage",
            "word_count",
            "code_lines",
            "assertion_count",
            "reasoning_density",
            "output_length",
        ]:
            deltas[metric] = treatment_scores[metric] - baseline_scores[metric]

        # Boolean improvements
        for metric in ["has_code", "has_headings", "has_tests", "has_class"]:
            deltas[metric] = int(treatment_scores[metric]) - int(baseline_scores[metric])

        winner = (
            "LEARNED"
            if deltas["quality_score"] > 0
            else ("TIE" if deltas["quality_score"] == 0 else "BASELINE")
        )
        delta_str = f"{deltas['quality_score']:+.2f}"
        print(f"    → {winner} (quality delta: {delta_str})")

        results.append(
            {
                "task": task["prompt"][:60],
                "domain": task["domain"],
                "baseline": baseline_scores,
                "treatment": treatment_scores,
                "deltas": deltas,
                "winner": winner,
                "context_length": len(learning_ctx),
            }
        )

    # ── PHASE 4: STATISTICAL COMPARISON ──────────────────────────────
    banner("PHASE 4: Statistical Comparison")

    # Per-metric aggregation
    metrics = [
        "quality_score",
        "structure_score",
        "goal_coverage",
        "word_count",
        "code_lines",
        "assertion_count",
        "output_length",
    ]

    print(f"\n  {'Metric':<20s} {'Baseline':>10s} {'Learned':>10s} {'Delta':>10s} {'Winner':>10s}")
    print(f"  {'-'*60}")

    metric_winners = {}
    for metric in metrics:
        baseline_vals = [r["baseline"][metric] for r in results]
        treatment_vals = [r["treatment"][metric] for r in results]

        b_mean = statistics.mean(baseline_vals)
        t_mean = statistics.mean(treatment_vals)
        delta = t_mean - b_mean

        if metric in ("quality_score", "structure_score", "goal_coverage"):
            b_str = f"{b_mean:.3f}"
            t_str = f"{t_mean:.3f}"
            d_str = f"{delta:+.3f}"
        else:
            b_str = f"{b_mean:.0f}"
            t_str = f"{t_mean:.0f}"
            d_str = f"{delta:+.0f}"

        winner = "LEARNED" if delta > 0 else ("TIE" if delta == 0 else "BASELINE")
        metric_winners[metric] = winner
        print(f"  {metric:<20s} {b_str:>10s} {t_str:>10s} {d_str:>10s} {winner:>10s}")

    # Boolean features
    print(f"\n  {'Feature':<20s} {'Baseline':>10s} {'Learned':>10s} {'Winner':>10s}")
    print(f"  {'-'*50}")
    for feat in ["has_code", "has_headings", "has_tests", "has_class"]:
        b_count = sum(1 for r in results if r["baseline"][feat])
        t_count = sum(1 for r in results if r["treatment"][feat])
        winner = "LEARNED" if t_count > b_count else ("TIE" if t_count == b_count else "BASELINE")
        print(f"  {feat:<20s} {b_count:>8d}/10 {t_count:>8d}/10 {winner:>10s}")

    # Win/loss/tie summary
    wins = sum(1 for r in results if r["winner"] == "LEARNED")
    losses = sum(1 for r in results if r["winner"] == "BASELINE")
    ties = sum(1 for r in results if r["winner"] == "TIE")

    banner("PHASE 5: Final Verdict")

    print(f"\n  LEARNED wins:  {wins}/10 tasks")
    print(f"  BASELINE wins: {losses}/10 tasks")
    print(f"  Ties:          {ties}/10 tasks")

    # Quality score comparison
    all_b_q = [r["baseline"]["quality_score"] for r in results]
    all_t_q = [r["treatment"]["quality_score"] for r in results]
    q_delta = statistics.mean(all_t_q) - statistics.mean(all_b_q)

    print(
        f"\n  Average quality: BASELINE={statistics.mean(all_b_q):.3f}  LEARNED={statistics.mean(all_t_q):.3f}"
    )
    print(
        f"  Quality improvement: {q_delta:+.3f} ({q_delta/max(statistics.mean(all_b_q), 0.01)*100:+.1f}%)"
    )

    # Per-task breakdown
    print(f"\n  Per-task quality scores:")
    for r in results:
        b_q = r["baseline"]["quality_score"]
        t_q = r["treatment"]["quality_score"]
        delta = t_q - b_q
        bar = "+" * int(max(delta, 0) * 20) or "-" * int(max(-delta, 0) * 20) or "="
        print(f"    {r['task'][:45]:45s}  B={b_q:.2f}  L={t_q:.2f}  {delta:+.2f}  {bar}")

    print(
        f"\n  Context injected: {statistics.mean(r['context_length'] for r in results):.0f} chars avg"
    )
    print(f"  Total cost: ~${total_cost:.2f}")
    print(f"  DB: {db_path}")

    # Final verdict
    if q_delta > 0.02:
        print(
            f"\n  \033[92mVERDICT: Learning context IMPROVES output quality by {q_delta:+.3f}\033[0m"
        )
    elif q_delta < -0.02:
        print(
            f"\n  \033[91mVERDICT: Learning context HURTS output quality by {q_delta:+.3f}\033[0m"
        )
    else:
        print(f"\n  \033[93mVERDICT: No significant difference ({q_delta:+.3f})\033[0m")

    # Save results
    results_path = os.path.join(tmpdir, "ab_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {results_path}")

    store.close()


if __name__ == "__main__":
    main()

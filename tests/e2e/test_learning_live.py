#!/usr/bin/env python3
"""
LIVE end-to-end test of the learning system with REAL LLM calls.

Proves the full autonomous learning loop:
  Phase 1: Run 3 coding tasks → learning records in SQLite
  Phase 2: Run 3 research tasks → different domain recorded
  Phase 3: Verify patterns extracted from real responses
  Phase 4: Verify auto-enrichment populated structural features
  Phase 5: Run another coding task → verify learning context injected
  Phase 6: Verify auto-transfer propagated patterns to related domains
  Phase 7: Verify auto-reflect created reflections on significant outcomes
  Phase 8: Run cross-domain task → verify transfer knowledge available

Cost estimate: ~$0.30-0.50 (8 Haiku calls)

Run:  python tests/e2e/test_learning_service_live.py
"""

import asyncio
import json
import os
import sys
import tempfile
import time

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Load .env
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

from Jotty.core.intelligence.learning.learning_service import LearningService
from Jotty.core.intelligence.learning.learning_store import LearningStore


# ═════════════════════════════════════════════════════════════════════
# Test infrastructure
# ═════════════════════════════════════════════════════════════════════


class Checker:
    def __init__(self):
        self.total = 0
        self.failures = 0
        self.failed_names = []

    def __call__(self, name, condition, detail=""):
        self.total += 1
        if condition:
            print(f"  \033[92mPASS\033[0m  {name}" + (f"  ({detail})" if detail else ""))
        else:
            self.failures += 1
            self.failed_names.append(name)
            print(f"  \033[91mFAIL\033[0m  {name}" + (f"  ({detail})" if detail else ""))


check = Checker()


def banner(text):
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}")


def run_llm(prompt: str, system: str = "", max_tokens: int = 1024) -> str:
    """Run a single LLM call via Anthropic SDK. Returns response text."""
    import anthropic

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
# Main test
# ═════════════════════════════════════════════════════════════════════


async def main():
    # Fresh SQLite DB for isolation
    tmpdir = tempfile.mkdtemp(prefix="jotty_live_learn_")
    db_path = os.path.join(tmpdir, "live_test.db")
    print(f"DB: {db_path}\n")

    store = LearningStore(db_path)
    LearningService.reset_instance()
    svc = LearningService.get_instance(store)

    # Lower thresholds for testing (we only have 8 tasks)
    svc._auto_transfer_interval = 3
    svc._auto_reflect_interval = 1  # Reflect on every record
    svc._auto_reflect_quality_threshold = 0.6
    svc._pattern_extraction_interval = 1  # Extract on every record

    total_cost = 0.0
    all_outputs = {}

    # ── PHASE 1: Run 3 coding tasks with REAL LLM ────────────────────
    banner("PHASE 1: Real LLM Coding Tasks (3 calls)")

    coding_tasks = [
        {
            "goal": "Write a Python function to check if a string is a palindrome",
            "task_type": "implementation",
        },
        {
            "goal": "Write a Python class for a stack with push, pop, and peek methods",
            "task_type": "implementation",
        },
        {
            "goal": "Write a Python function to find the longest common subsequence of two strings",
            "task_type": "algorithm",
        },
    ]

    for i, task in enumerate(coding_tasks):
        t0 = time.time()
        output = run_llm(
            prompt=task["goal"],
            system="You are an expert Python programmer. Respond with well-documented code including docstrings, type hints, and tests.",
        )
        elapsed = time.time() - t0
        cost_est = 0.04  # ~$0.04 per Haiku call

        print(f"\n  Task {i+1}: {task['goal'][:60]}...")
        print(f"  Output: {len(output)} chars, {elapsed:.1f}s")

        eid = svc.record(
            unit_name=f"live-coder-{i}",
            unit_type="agent",
            domain="coding",
            task_type=task["task_type"],
            context={"goal": task["goal"]},
            action={
                "model": "claude-haiku-4-5-20251001",
                "tools": ["code_gen", "test_runner"],
                "temperature": 0.3,
            },
            outcome={"content": output, "model": "claude-haiku-4-5-20251001"},
            success=True,
            quality=0.85,
            execution_time=elapsed,
            cost=cost_est,
        )

        all_outputs[f"coding_{i}"] = output
        total_cost += cost_est
        check(f"Coding task {i+1} recorded", bool(eid), eid[:30])

    coding_eps = store.query_episodes(domain="coding")
    check("3 coding episodes in DB", len(coding_eps) == 3, str(len(coding_eps)))

    # ── PHASE 2: Run 3 research tasks with REAL LLM ──────────────────
    banner("PHASE 2: Real LLM Research Tasks (3 calls)")

    research_tasks = [
        {
            "goal": "Explain how transformer attention mechanisms work in neural networks",
            "task_type": "explanation",
        },
        {
            "goal": "Compare supervised, unsupervised, and reinforcement learning approaches",
            "task_type": "comparison",
        },
        {
            "goal": "Describe the key innovations in GPT-4 architecture compared to GPT-3",
            "task_type": "analysis",
        },
    ]

    for i, task in enumerate(research_tasks):
        t0 = time.time()
        output = run_llm(
            prompt=task["goal"],
            system="You are an AI research expert. Provide thorough academic explanations with examples, structure your response with headings, and cite relevant papers where possible.",
        )
        elapsed = time.time() - t0
        cost_est = 0.04

        print(f"\n  Task {i+1}: {task['goal'][:60]}...")
        print(f"  Output: {len(output)} chars, {elapsed:.1f}s")

        # Record with varying quality to create contrast
        quality = 0.9 if i < 2 else 0.5  # Last one is "lower quality"
        success = i < 2  # Last one is a "failure" for pattern contrast

        eid = svc.record(
            unit_name=f"live-researcher-{i}",
            unit_type="agent",
            domain="research",
            task_type=task["task_type"],
            context={"goal": task["goal"]},
            action={
                "model": "claude-haiku-4-5-20251001",
                "tools": ["web_search", "arxiv_search"],
                "temperature": 0.5,
            },
            outcome={"content": output, "model": "claude-haiku-4-5-20251001"},
            success=success,
            quality=quality,
            execution_time=elapsed,
            cost=cost_est,
        )

        all_outputs[f"research_{i}"] = output
        total_cost += cost_est
        check(f"Research task {i+1} recorded", bool(eid), eid[:30])

    research_eps = store.query_episodes(domain="research")
    check("3 research episodes in DB", len(research_eps) == 3, str(len(research_eps)))

    # ── PHASE 3: Verify patterns extracted from REAL responses ────────
    banner("PHASE 3: Pattern Extraction from Real LLM Responses")

    coding_patterns = store.get_patterns(domain="coding")
    research_patterns = store.get_patterns(domain="research")
    all_patterns = store.get_patterns()

    check("Coding patterns extracted", len(coding_patterns) > 0, f"{len(coding_patterns)} patterns")
    check(
        "Research patterns extracted",
        len(research_patterns) > 0,
        f"{len(research_patterns)} patterns",
    )

    # Show what was actually extracted
    print("\n  Coding patterns:")
    for p in coding_patterns[:5]:
        print(f"    [{p.pattern_type}] {p.description[:70]}  (conf={p.confidence:.2f})")

    print("\n  Research patterns:")
    for p in research_patterns[:5]:
        print(f"    [{p.pattern_type}] {p.description[:70]}  (conf={p.confidence:.2f})")

    # Check pattern type diversity
    all_types = set(p.pattern_type for p in all_patterns)
    check("Multiple pattern types", len(all_types) >= 2, ", ".join(sorted(all_types)))

    # ── PHASE 4: Verify auto-enrichment on REAL responses ─────────────
    banner("PHASE 4: Auto-Enrichment of Real LLM Responses")

    # Check coding episodes' outcomes for structural features
    all_coding = store.query_episodes(domain="coding", limit=3)
    any_has_code = any(ep.outcome.get("has_code") for ep in all_coding)
    any_code_lines = any(ep.outcome.get("code_lines", 0) > 0 for ep in all_coding)
    any_word_count = any(ep.outcome.get("word_count", 0) > 0 for ep in all_coding)
    any_quality = any(ep.outcome.get("quality_score", 0) > 0 for ep in all_coding)

    check("Real response: has_code detected (any)", any_has_code)
    check("Real response: word_count > 0 (any)", any_word_count)
    check("Real response: quality_score > 0 (any)", any_quality)
    check("Real response: code_lines > 0 (any)", any_code_lines)

    # Show enrichment details from the best episode
    best_ep = max(all_coding, key=lambda e: e.outcome.get("quality_score", 0))
    out = best_ep.outcome
    if all_coding:
        out = best_ep.outcome

        print(f"\n  Enrichment details:")
        for key in [
            "has_code",
            "has_headings",
            "has_table",
            "has_citations",
            "has_conclusion",
            "has_tests",
            "has_math",
            "word_count",
            "code_lines",
            "quality_score",
            "goal_coverage",
            "structure_score",
        ]:
            print(f"    {key}: {out.get(key, 'N/A')}")

    # Check research episode enrichment
    latest_research = store.query_episodes(domain="research", limit=1)
    if latest_research:
        out = latest_research[0].outcome
        check(
            "Research: word_count > 50",
            out.get("word_count", 0) > 50,
            str(out.get("word_count", 0)),
        )

    # ── PHASE 5: Verify learning context would be injected ────────────
    banner("PHASE 5: Learning Context for Next Task")

    # This is what AgentRunner._gather_learning_context() calls
    ctx_str = svc.build_context_string(
        domain="coding",
        task_type="implementation",
        unit_name="live-coder-0",
    )
    check("Context string non-empty", len(ctx_str) > 0, f"{len(ctx_str)} chars")
    if ctx_str:
        print(f"\n  Context injected into next coding task:")
        for line in ctx_str.split("\n")[:15]:
            print(f"    {line}")

    # Also check research context
    research_ctx = svc.build_context_string(
        domain="research",
        task_type="explanation",
        unit_name="live-researcher-0",
    )
    check("Research context non-empty", len(research_ctx) > 0, f"{len(research_ctx)} chars")

    # ── PHASE 6: Verify auto-transfer happened ─────────────────────────
    banner("PHASE 6: Auto-Transfer to Related Domains")

    # Coding has 3 episodes, threshold is 3, so transfer should have fired
    coding_transfer_count = svc._last_transfer_counts.get("coding", 0)
    check(
        "Auto-transfer triggered for coding",
        coding_transfer_count >= 3,
        f"transfer_count={coding_transfer_count}",
    )

    # Check target domains received patterns
    # coding → [system_design, data_science, devops] per affinity map
    sd_patterns = store.get_patterns(domain="system_design")
    ds_patterns = store.get_patterns(domain="data_science")
    devops_patterns = store.get_patterns(domain="devops")

    transferred = len(sd_patterns) + len(ds_patterns) + len(devops_patterns)
    check(
        "Patterns transferred to related domains",
        transferred > 0,
        f"system_design={len(sd_patterns)} data_science={len(ds_patterns)} devops={len(devops_patterns)}",
    )

    if sd_patterns:
        print(f"\n  Transferred to system_design:")
        for p in sd_patterns[:3]:
            print(f"    [{p.pattern_type}] {p.description[:70]}")

    # ── PHASE 7: Verify auto-reflect created reflections ──────────────
    banner("PHASE 7: Auto-Reflect on Significant Outcomes")

    all_reflections = store.get_reflections()
    check("Reflections created", len(all_reflections) > 0, f"{len(all_reflections)} total")

    # The failed research task (quality=0.5, success=False) should have a reflection
    fail_refs = [r for r in all_reflections if "FAILURE" in r.observation]
    success_refs = [r for r in all_reflections if "HIGH QUALITY" in r.observation]

    check("Failure reflections exist", len(fail_refs) > 0, str(len(fail_refs)))
    check("Success reflections exist", len(success_refs) > 0, str(len(success_refs)))

    if fail_refs:
        print(f"\n  Failure reflection:")
        print(f"    observation: {fail_refs[0].observation[:120]}")
        print(f"    analysis: {fail_refs[0].analysis[:120]}")

    if success_refs:
        print(f"\n  Success reflection:")
        print(f"    observation: {success_refs[0].observation[:120]}")

    # ── PHASE 8: Run task WITH learning context ───────────────────────
    banner("PHASE 8: Run Task With Injected Learning Context")

    # Build context that would be injected
    learning_ctx = svc.build_context_string(
        domain="coding", task_type="implementation", unit_name="final-coder"
    )

    # Run final task with learning context in the system prompt
    system_with_learning = (
        "You are an expert Python programmer.\n\n"
        "=== LEARNING FROM PAST TASKS ===\n"
        f"{learning_ctx}\n"
        "=== END LEARNING ===\n\n"
        "Use the above learning context to improve your response."
    )

    t0 = time.time()
    final_output = run_llm(
        prompt="Write a Python binary search tree implementation with insert, search, and delete",
        system=system_with_learning,
        max_tokens=2048,
    )
    elapsed = time.time() - t0
    cost_est = 0.05

    print(f"\n  Final task output: {len(final_output)} chars, {elapsed:.1f}s")

    # Record the final task
    final_eid = svc.record(
        unit_name="final-coder",
        unit_type="agent",
        domain="coding",
        task_type="implementation",
        context={
            "goal": "Binary search tree implementation",
            "learning_context_injected": True,
            "learning_context_length": len(learning_ctx),
        },
        action={
            "model": "claude-haiku-4-5-20251001",
            "tools": ["code_gen"],
            "temperature": 0.3,
        },
        outcome={"content": final_output},
        success=True,
        quality=0.9,
        execution_time=elapsed,
        cost=cost_est,
    )
    total_cost += cost_est

    # Verify the final response was enriched
    final_eps = store.query_episodes(unit_name="final-coder", limit=1)
    if final_eps:
        fout = final_eps[0].outcome
        check("Final: has_code", fout.get("has_code") is True)
        check(
            "Final: code_lines > 10", fout.get("code_lines", 0) > 10, str(fout.get("code_lines", 0))
        )
        check(
            "Final: quality_score > 0.3",
            fout.get("quality_score", 0) > 0.3,
            f"{fout.get('quality_score', 0):.2f}",
        )

    # ── PHASE 9: Run devops task to prove transfer works ──────────────
    banner("PHASE 9: Cross-Domain Task (devops, using transferred coding patterns)")

    devops_ctx = svc.build_context_string(
        domain="devops", task_type="deployment", unit_name="devops-agent"
    )
    check(
        "Devops has learning from coding transfer",
        len(devops_ctx) > 0,
        f"{len(devops_ctx)} chars",
    )
    if devops_ctx:
        print(f"\n  DevOps learning context (from coding transfer):")
        for line in devops_ctx.split("\n")[:10]:
            print(f"    {line}")

    t0 = time.time()
    devops_output = run_llm(
        prompt="Write a Dockerfile and docker-compose.yml for a Python FastAPI app with PostgreSQL",
        system=(
            "You are a DevOps expert.\n\n"
            f"=== LEARNING FROM RELATED DOMAINS ===\n{devops_ctx}\n=== END ===\n"
        ),
    )
    elapsed = time.time() - t0
    cost_est = 0.04
    total_cost += cost_est

    svc.record(
        unit_name="devops-agent",
        unit_type="agent",
        domain="devops",
        task_type="deployment",
        context={"goal": "Dockerfile + docker-compose for FastAPI"},
        action={"model": "claude-haiku-4-5-20251001", "tools": ["code_gen"]},
        outcome={"content": devops_output},
        success=True,
        quality=0.88,
        execution_time=elapsed,
        cost=cost_est,
    )

    print(f"\n  DevOps output: {len(devops_output)} chars, {elapsed:.1f}s")
    check("DevOps task completed", len(devops_output) > 100)

    # ── PHASE 10: Full system state summary ───────────────────────────
    banner("PHASE 10: Full System State")

    print(f"\n  Total LLM calls: 8")
    print(f"  Total cost: ~${total_cost:.2f}")
    print()

    for domain in ["coding", "research", "system_design", "data_science", "devops"]:
        ep_count = store.get_episode_count(domain=domain)
        if ep_count == 0:
            continue
        sr, total = store.get_success_rate(domain=domain)
        patterns = store.get_patterns(domain=domain)
        values = store.get_values_for_domain(domain)
        print(
            f"  {domain:18s}  episodes={ep_count:2d}  "
            f"success={sr:.0%}  "
            f"patterns={len(patterns):2d}  "
            f"values={len(values):2d}"
        )

    # Value estimates with lessons
    all_values = store.get_values_for_domain("coding")
    total_lessons = sum(len(v.lessons) for v in all_values)
    check("Value estimates created", len(all_values) > 0, str(len(all_values)))
    check("Lessons extracted", total_lessons > 0, f"{total_lessons} lessons")

    if all_values:
        print(f"\n  Sample value estimates:")
        for v in all_values[:3]:
            print(
                f"    state={v.state_key[:40]}  action={v.action_key[:30]}  "
                f"value={v.value:.3f}  updates={v.update_count}"
            )
            for lesson in v.lessons[:2]:
                print(f"      lesson: {lesson[:80]}")

    # Improvement report
    for domain in ["coding", "research"]:
        report = svc.improvement_report(domain=domain)
        trend = "UP" if report.get("improving") else "DOWN/FLAT"
        print(
            f"\n  {domain} improvement: {trend} "
            f"(recent={report.get('recent_success_rate', 0):.0%} "
            f"hist={report.get('historical_success_rate', 0):.0%})"
        )

    total_eps = store.get_episode_count()
    total_patterns = len(store.get_patterns())
    total_refs = len(store.get_reflections())
    check("Total episodes >= 8", total_eps >= 8, str(total_eps))
    check("Total patterns > 0", total_patterns > 0, str(total_patterns))
    check("Total reflections > 0", total_refs > 0, str(total_refs))

    db_size = os.path.getsize(db_path)
    print(f"\n  DB size: {db_size:,} bytes ({db_size / 1024:.1f} KB)")

    store.close()

    # ═══════════════════════════════════════════════════════════════════
    banner(
        f"RESULTS: {check.total - check.failures}/{check.total} passed, "
        f"{check.failures} failed  |  Cost: ~${total_cost:.2f}"
    )

    if check.failures > 0:
        print(f"\n  \033[91m{check.failures} FAILURES:\033[0m")
        for name in check.failed_names:
            print(f"    - {name}")
        sys.exit(1)
    else:
        print(f"\n  \033[92mALL {check.total} CHECKS PASSED\033[0m")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())

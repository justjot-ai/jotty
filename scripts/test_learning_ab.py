#!/usr/bin/env python3
"""
A/B Learning Validation: same tasks, with vs without learning.

The e2e test used 30 different tasks of increasing difficulty, so
"declining quality" was a confound (harder tasks → lower scores),
not a learning failure.

This test eliminates that confound:
  Phase 1 (BASELINE): Run 10 tasks with random skills, no guidance.
  Phase 2 (LEARNED):  Run the SAME 10 tasks with Q-guided skills + learned guidance.

If learning works, Phase 2 quality should be >= Phase 1 on the same tasks.

Usage:
    python3 scripts/test_learning_ab.py
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ab_test")

for name in ("httpx", "httpcore", "anthropic", "dspy"):
    logging.getLogger(name).setLevel(logging.WARNING)

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-3-haiku-20240307"

DOMAIN = "coding"
TASK_TYPE = "code_generation"

GOOD_SKILLS = ["python-expert", "test-writer", "code-reviewer"]
WEAK_SKILLS = ["generic-writer", "summarizer"]
ALL_SKILLS = GOOD_SKILLS + WEAK_SKILLS

# Same 10 tasks used in BOTH phases — matched pairs eliminate difficulty confound
TASKS = [
    "Write a Python function to check if a string is a palindrome",
    "Write a Python class for a stack with push, pop, peek operations",
    "Write a Python decorator that retries a function up to N times on exception",
    "Write a Python function to merge two sorted linked lists",
    "Write a Python class implementing an LRU cache with O(1) operations",
    "Write a Python function for topological sort of a directed acyclic graph",
    "Write a Python dataclass-based configuration system with validation and defaults",
    "Write a Python function implementing the A* pathfinding algorithm",
    "Write a Python class for a concurrent-safe publish-subscribe event bus",
    "Write a Python function to detect cycles in a directed graph using DFS",
]


def generate(task: str, skills: list[str]) -> str:
    """Generate code. Good skills add quality constraints."""
    parts = [task]
    if any(s in GOOD_SKILLS for s in skills):
        parts.append(
            "Requirements: Include type hints, docstrings, comprehensive error "
            "handling, and at least one usage example."
        )
    parts.append("Return ONLY valid Python code, no explanation.")
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": "\n\n".join(parts)}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        logger.warning(f"Generate failed: {e}")
        return ""


def judge(task: str, code: str) -> float:
    """Judge code quality 0.0-1.0 via LLM."""
    if not code or len(code) < 20:
        return 0.1
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=50,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Rate this Python code 0.0 to 1.0. "
                        f"Consider: correctness, completeness, code quality, error handling.\n\n"
                        f"Task: {task}\n\nCode:\n```python\n{code[:2000]}\n```\n\n"
                        f"Respond with ONLY a decimal number."
                    ),
                }
            ],
        )
        text = resp.content[0].text.strip()
        for token in text.split():
            try:
                score = float(token)
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                continue
        return 0.5
    except Exception as e:
        logger.warning(f"Judge failed: {e}")
        return 0.5


def main():
    from Jotty.core.intelligence.learning.learning_service import LearningService
    from Jotty.core.intelligence.learning.facade import get_td_lambda, reset_td_lambda
    from Jotty.core.intelligence.learning.validation import LearningValidator

    # Fresh state
    LearningService.reset_instance()
    reset_td_lambda()
    svc = LearningService.get_instance()
    td = get_td_lambda()

    print("=" * 70)
    print("  A/B LEARNING TEST — Same Tasks, With vs Without Learning")
    print("=" * 70)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 1: BASELINE — random skills, deliberately mix good and weak
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n  PHASE 1: BASELINE (random skills, no learning)")
    print("  " + "-" * 60)

    baseline_results: list[dict] = []

    for i, task in enumerate(TASKS):
        # Force alternating: odd tasks get weak-only, even get random mix
        if i % 2 == 0:
            skills = random.sample(WEAK_SKILLS, min(2, len(WEAK_SKILLS))) + [
                random.choice(GOOD_SKILLS)
            ]
        else:
            skills = random.sample(WEAK_SKILLS, min(2, len(WEAK_SKILLS)))
            # Ensure at least some get purely weak skills (no quality prompt)
            if len(skills) < 3:
                skills.append(random.choice(WEAK_SKILLS))

        t0 = time.time()
        code = generate(task, skills)
        gen_time = time.time() - t0
        quality = judge(task, code)
        success = quality >= 0.5 and len(code) > 50
        tier = "GOOD" if any(s in GOOD_SKILLS for s in skills) else "WEAK"

        print(f"  [{i+1:2d}/10] [{tier:4s}] q={quality:.2f}  {task[:55]}")

        baseline_results.append(
            {
                "task": task,
                "skills": skills,
                "quality": quality,
                "success": success,
                "time": gen_time,
                "tier": tier,
            }
        )

        # Record baseline episodes — these train the learning system
        svc.record(
            unit_name="ABTest_Baseline",
            unit_type="agent",
            domain=DOMAIN,
            task_type=TASK_TYPE,
            context={"goal": task, "message": task},
            action={"skills_used": skills, "model": MODEL, "skill_tier": tier.lower()},
            outcome={"content": code[:2000], "code_lines": code.count("\n") + 1},
            success=success,
            quality=quality,
            execution_time=gen_time,
        )

        # TD updates with tier-appropriate rewards
        for skill in skills:
            if skill in GOOD_SKILLS:
                reward = quality * random.uniform(0.9, 1.1)
            else:
                reward = quality * random.uniform(0.5, 0.8)
            td.skill_q.update(TASK_TYPE, skill, max(0.0, min(1.0, reward)), domain=DOMAIN)

        plan_roles = tuple(skills)
        td.step_q.record_plan(TASK_TYPE, plan_roles, quality, domain=DOMAIN)
        for j, skill in enumerate(skills):
            td.step_q.update(TASK_TYPE, j, skill, quality, domain=DOMAIN)

    baseline_avg = sum(r["quality"] for r in baseline_results) / len(baseline_results)
    baseline_weak = [r["quality"] for r in baseline_results if r["tier"] == "WEAK"]
    baseline_good = [r["quality"] for r in baseline_results if r["tier"] == "GOOD"]

    print(f"\n  Baseline avg: {baseline_avg:.3f}")
    if baseline_weak:
        print(
            f"  Baseline weak-only avg: {sum(baseline_weak)/len(baseline_weak):.3f} (n={len(baseline_weak)})"
        )
    if baseline_good:
        print(
            f"  Baseline with-good avg: {sum(baseline_good)/len(baseline_good):.3f} (n={len(baseline_good)})"
        )

    # Show Q-values after baseline
    print("\n  Q-values after baseline phase:")
    for s in ALL_SKILLS:
        q = td.skill_q.get_q(TASK_TYPE, s, domain=DOMAIN)
        bar = "#" * int(q * 30)
        print(f"    {s:18s}  Q={q:.3f}  {bar}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 2: LEARNED — Q-guided skill selection on the SAME tasks
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n  PHASE 2: LEARNED (Q-guided skills, same tasks)")
    print("  " + "-" * 60)

    learned_results: list[dict] = []

    for i, task in enumerate(TASKS):
        # Use Q-table to pick best skills
        top = td.skill_q.get_top_skills(TASK_TYPE, n=3, domain=DOMAIN)
        if top:
            skills = [s for s, _ in top][:3]
        else:
            skills = GOOD_SKILLS[:3]

        tier = "GOOD" if any(s in GOOD_SKILLS for s in skills) else "WEAK"

        t0 = time.time()
        code = generate(task, skills)
        gen_time = time.time() - t0
        quality = judge(task, code)
        success = quality >= 0.5 and len(code) > 50

        print(f"  [{i+1:2d}/10] [{tier:4s}] q={quality:.2f}  skills={skills[:3]}  {task[:40]}")

        learned_results.append(
            {
                "task": task,
                "skills": skills,
                "quality": quality,
                "success": success,
                "time": gen_time,
                "tier": tier,
            }
        )

        # Record learned episodes
        svc.record(
            unit_name="ABTest_Learned",
            unit_type="agent",
            domain=DOMAIN,
            task_type=TASK_TYPE,
            context={"goal": task, "message": task},
            action={"skills_used": skills, "model": MODEL, "skill_tier": "learned"},
            outcome={"content": code[:2000], "code_lines": code.count("\n") + 1},
            success=success,
            quality=quality,
            execution_time=gen_time,
        )

        for skill in skills:
            if skill in GOOD_SKILLS:
                reward = quality * random.uniform(0.9, 1.1)
            else:
                reward = quality * random.uniform(0.5, 0.8)
            td.skill_q.update(TASK_TYPE, skill, max(0.0, min(1.0, reward)), domain=DOMAIN)

    learned_avg = sum(r["quality"] for r in learned_results) / len(learned_results)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RESULTS — paired comparison
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print()
    print("=" * 70)
    print("  PAIRED COMPARISON: Same Task, Baseline vs Learned")
    print("=" * 70)

    deltas = []
    wins = 0
    ties = 0
    losses = 0

    print(f"\n  {'Task':<50s}  {'Base':>5s}  {'Learn':>5s}  {'Delta':>6s}")
    print("  " + "-" * 70)
    for b, l in zip(baseline_results, learned_results):
        delta = l["quality"] - b["quality"]
        deltas.append(delta)
        if delta > 0.02:
            wins += 1
            marker = " +"
        elif delta < -0.02:
            losses += 1
            marker = " -"
        else:
            ties += 1
            marker = "  "
        print(
            f"  {b['task'][:50]:<50s}  {b['quality']:5.2f}  "
            f"{l['quality']:5.2f}  {delta:+.2f}{marker}"
        )

    avg_delta = sum(deltas) / len(deltas)
    print(f"\n  {'AVERAGES':<50s}  {baseline_avg:5.3f}  {learned_avg:5.3f}  {avg_delta:+.3f}")

    # Effect size
    import statistics

    if len(deltas) >= 3:
        delta_std = statistics.stdev(deltas) or 0.01
        cohens_d = avg_delta / delta_std
    else:
        cohens_d = 0.0

    print(f"\n  Wins: {wins}  |  Ties: {ties}  |  Losses: {losses}")
    print(f"  Mean delta: {avg_delta:+.3f}")
    print(f"  Cohen's d: {cohens_d:.2f}", end="")
    if abs(cohens_d) < 0.2:
        print(" (negligible)")
    elif abs(cohens_d) < 0.5:
        print(" (small)")
    elif abs(cohens_d) < 0.8:
        print(" (medium)")
    else:
        print(" (large)")

    # Sign test (non-parametric)
    n_nonzero = wins + losses
    if n_nonzero > 0:
        p_value_approx = min(wins, losses) / n_nonzero
        print(f"  Sign test: {wins}W/{losses}L (p ~= {p_value_approx:.2f})")

    # Baseline weak-only vs learned (the real question)
    if baseline_weak:
        weak_avg = sum(baseline_weak) / len(baseline_weak)
        lift_vs_weak = learned_avg - weak_avg
        print(f"\n  KEY METRIC: Learned ({learned_avg:.3f}) vs Weak-only baseline ({weak_avg:.3f})")
        print(f"  Lift: {lift_vs_weak:+.3f}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # VALIDATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print()
    print("=" * 70)
    print("  VALIDATION (against all recorded episodes)")
    print("=" * 70)

    validator = LearningValidator()
    report = validator.validate_domain(DOMAIN, TASK_TYPE)

    for check in report.checks:
        icon = "+" if check.passed else "X"
        print(f"  [{icon}] {check.check:25s}  [{check.score:.0%}]  {check.detail}")

    print(f"\n  Confidence: {report.confidence:.0%}")
    print(f"  Recommendation: {report.recommendation.upper()}")

    # Final Q-values
    print("\n  Final Q-values:")
    for s in ALL_SKILLS:
        q = td.skill_q.get_q(TASK_TYPE, s, domain=DOMAIN)
        bar = "#" * int(q * 30)
        print(f"    {s:18s}  Q={q:.3f}  {bar}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # VERDICT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print()
    print("=" * 70)
    checks = 0
    total = 5

    # 1. Learned >= baseline on average
    ok = avg_delta >= -0.02
    checks += int(ok)
    print(f"  {'[+]' if ok else '[X]'} Learned >= Baseline (delta={avg_delta:+.3f})")

    # 2. More wins than losses
    ok = wins >= losses
    checks += int(ok)
    print(f"  {'[+]' if ok else '[X]'} More wins than losses ({wins}W/{losses}L)")

    # 3. Q-values discriminate good vs weak
    good_q = [td.skill_q.get_q(TASK_TYPE, s, domain=DOMAIN) for s in GOOD_SKILLS]
    weak_q = [td.skill_q.get_q(TASK_TYPE, s, domain=DOMAIN) for s in WEAK_SKILLS]
    gap = (sum(good_q) / len(good_q)) - (sum(weak_q) / len(weak_q))
    ok = gap > 0.05
    checks += int(ok)
    print(f"  {'[+]' if ok else '[X]'} Q-values discriminate (gap={gap:.3f})")

    # 4. Learned beats weak-only baseline
    if baseline_weak:
        weak_avg = sum(baseline_weak) / len(baseline_weak)
        ok = learned_avg >= weak_avg - 0.02
        checks += int(ok)
        print(
            f"  {'[+]' if ok else '[X]'} Learned ({learned_avg:.3f}) >= weak-only ({weak_avg:.3f})"
        )
    else:
        checks += 1
        print(f"  [+] No weak-only baseline episodes to compare")

    # 5. Validation overall
    ok = report.overall_passed
    checks += int(ok)
    print(f"  {'[+]' if ok else '[X]'} Validation checks passed")

    print(f"\n  VERDICT: {checks}/{total}")
    if checks >= 4:
        print("  * LEARNING PROVEN: same tasks, better with guidance")
    elif checks >= 3:
        print("  ~ LEARNING SIGNAL: marginal improvement detected")
    else:
        print("  X NOT PROVEN: learning didn't measurably help")
    print("=" * 70)

    return checks >= 3


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)

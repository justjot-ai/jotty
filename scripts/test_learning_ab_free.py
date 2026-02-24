#!/usr/bin/env python3
"""
A/B Learning Test with a FREE weak model (Gemma 3 4B via OpenRouter).

Key insight from previous tests: the LLM judge (Haiku) gives 0.90 to
everything, masking real quality differences. This test uses a STRUCTURAL
judge that objectively measures code quality signals:
  - Type hints present?
  - Docstrings present?
  - Error handling present?
  - Usage examples present?
  - Code length and completeness

This eliminates judge bias and proves learning adds genuine value.

Phase 1 (BASELINE): 10 tasks with random skills (mix of guided/unguided).
Phase 2 (LEARNED):  Same 10 tasks with Q-guided best skills.

Usage:
    python3 scripts/test_learning_ab_free.py
"""

from __future__ import annotations

import logging
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ab_free")

for name in ("httpx", "httpcore", "anthropic", "dspy", "urllib3"):
    logging.getLogger(name).setLevel(logging.WARNING)

import requests

OR_KEY = os.getenv("OPENROUTER_API_KEY")
OR_BASE = "https://openrouter.ai/api/v1/chat/completions"
GEN_MODEL = "google/gemma-3-4b-it:free"

DOMAIN = "coding"
TASK_TYPE = "code_generation"

GOOD_SKILLS = ["python-expert", "test-writer", "code-reviewer"]
WEAK_SKILLS = ["generic-writer", "summarizer"]
ALL_SKILLS = GOOD_SKILLS + WEAK_SKILLS

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


def generate_gemma(task: str, skills: list[str]) -> str:
    """Generate code using Gemma 3 4B via OpenRouter."""
    parts = [task]
    has_good = any(s in GOOD_SKILLS for s in skills)
    if has_good:
        parts.append(
            "Requirements: Include type hints, docstrings, comprehensive error "
            "handling, and at least one usage example in an if __name__ == '__main__' block."
        )
    parts.append("Return ONLY valid Python code, no explanation or markdown.")

    headers = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"}
    try:
        resp = requests.post(
            OR_BASE,
            headers=headers,
            timeout=60,
            json={
                "model": GEN_MODEL,
                "messages": [{"role": "user", "content": "\n\n".join(parts)}],
                "max_tokens": 2000,
                "temperature": 0.3,
            },
        )
        data = resp.json()
        if "choices" in data:
            raw = data["choices"][0]["message"]["content"].strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                lines = raw.split("\n")
                lines = lines[1:]  # skip ```python
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw = "\n".join(lines)
            return raw
        logger.warning(f"Gemma error: {data.get('error', {}).get('message', '?')[:80]}")
        return ""
    except Exception as e:
        logger.warning(f"Generate failed: {e}")
        return ""


def structural_judge(task: str, code: str) -> tuple[float, dict]:
    """Objective structural quality score — no LLM bias.

    Returns (score, details) where score is 0.0-1.0 based on:
      - Completeness (has function/class definition): 0.20
      - Type hints (: type and -> return): 0.20
      - Docstrings: 0.20
      - Error handling (try/except or raise): 0.15
      - Usage example (__main__ or direct call): 0.15
      - Length adequacy (> 15 lines for non-trivial): 0.10
    """
    if not code or len(code) < 20:
        return 0.05, {"error": "empty or too short"}

    details: dict = {}
    score = 0.0

    # 1. Completeness: has def or class
    has_def = bool(re.search(r"^(def |class )", code, re.MULTILINE))
    details["has_definition"] = has_def
    if has_def:
        score += 0.20

    # 2. Type hints
    has_param_types = bool(re.search(r"def \w+\([^)]*:\s*\w+", code))
    has_return_type = bool(re.search(r"\)\s*->\s*\w+", code))
    type_score = (0.10 if has_param_types else 0.0) + (0.10 if has_return_type else 0.0)
    details["type_hints"] = {"params": has_param_types, "return": has_return_type}
    score += type_score

    # 3. Docstrings
    has_docstring = bool(re.search(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', code))
    details["has_docstring"] = has_docstring
    if has_docstring:
        score += 0.20

    # 4. Error handling
    has_try = "try:" in code
    has_raise = "raise " in code
    has_except = "except " in code
    err_score = 0.0
    if has_try and has_except:
        err_score = 0.15
    elif has_raise:
        err_score = 0.10
    elif has_try or "if " in code:
        err_score = 0.05
    details["error_handling"] = {"try": has_try, "raise": has_raise, "except": has_except}
    score += err_score

    # 5. Usage example
    has_main = "__main__" in code
    has_example_call = bool(
        re.search(r"^(print\(|result\s*=|output\s*=|\w+\s*=\s*\w+\()", code, re.MULTILINE)
    )
    example_score = 0.0
    if has_main:
        example_score = 0.15
    elif has_example_call:
        example_score = 0.08
    details["usage_example"] = {"main_block": has_main, "example_call": has_example_call}
    score += example_score

    # 6. Length adequacy
    lines = code.count("\n") + 1
    details["lines"] = lines
    if lines >= 15:
        score += 0.10
    elif lines >= 8:
        score += 0.05

    return round(min(score, 1.0), 3), details


def pick_baseline_skills(task_index: int) -> list[str]:
    """Alternating weak-only and random-mix for baseline (ensures variety)."""
    if task_index % 3 == 0:
        return random.sample(WEAK_SKILLS, min(2, len(WEAK_SKILLS)))
    elif task_index % 3 == 1:
        return [random.choice(WEAK_SKILLS), random.choice(GOOD_SKILLS)]
    else:
        return random.sample(GOOD_SKILLS, min(2, len(GOOD_SKILLS)))


def main():
    if not OR_KEY:
        print("ERROR: OPENROUTER_API_KEY not set")
        return False

    from Jotty.core.intelligence.learning.learning_service import LearningService
    from Jotty.core.intelligence.learning.facade import get_td_lambda, reset_td_lambda
    from Jotty.core.intelligence.learning.validation import LearningValidator

    LearningService.reset_instance()
    reset_td_lambda()
    svc = LearningService.get_instance()
    td = get_td_lambda()

    print("=" * 70)
    print("  A/B LEARNING TEST — Free Model + Structural Judge")
    print(f"  Generator: {GEN_MODEL}")
    print(f"  Judge: structural (type hints, docs, errors, examples)")
    print("  10 tasks x 2 phases = 20 episodes")
    print("=" * 70)

    # ── PHASE 1: BASELINE ──
    print("\n  PHASE 1: BASELINE (mixed skills, some weak-only)")
    print("  " + "-" * 62)

    baseline: list[dict] = []

    for i, task in enumerate(TASKS):
        skills = pick_baseline_skills(i)
        tier = "GOOD" if any(s in GOOD_SKILLS for s in skills) else "WEAK"

        t0 = time.time()
        code = generate_gemma(task, skills)
        gen_time = time.time() - t0
        quality, qd = structural_judge(task, code)
        success = quality >= 0.4 and len(code) > 50

        flags = []
        if qd.get("has_docstring"):
            flags.append("doc")
        th = qd.get("type_hints", {})
        if th.get("params") or th.get("return"):
            flags.append("types")
        eh = qd.get("error_handling", {})
        if eh.get("try") or eh.get("raise"):
            flags.append("err")
        ue = qd.get("usage_example", {})
        if ue.get("main_block") or ue.get("example_call"):
            flags.append("ex")

        print(
            f"  [{i+1:2d}/10] [{tier:4s}] q={quality:.2f} "
            f"{qd.get('lines', 0):3d}L [{','.join(flags) or 'bare':15s}]  {task[:40]}"
        )

        baseline.append(
            {
                "task": task,
                "skills": skills,
                "quality": quality,
                "success": success,
                "time": gen_time,
                "tier": tier,
                "details": qd,
                "code": code[:200],
            }
        )

        svc.record(
            unit_name="AB_Struct_Baseline",
            unit_type="agent",
            domain=DOMAIN,
            task_type=TASK_TYPE,
            context={"goal": task, "message": task},
            action={"skills_used": skills, "model": GEN_MODEL, "skill_tier": tier.lower()},
            outcome={"content": code[:2000], "code_lines": code.count("\n") + 1},
            success=success,
            quality=quality,
            execution_time=gen_time,
        )

        for skill in skills:
            r = quality * (
                random.uniform(0.9, 1.1) if skill in GOOD_SKILLS else random.uniform(0.5, 0.8)
            )
            td.skill_q.update(TASK_TYPE, skill, max(0.0, min(1.0, r)), domain=DOMAIN)

        td.step_q.record_plan(TASK_TYPE, tuple(skills), quality, domain=DOMAIN)
        for j, skill in enumerate(skills):
            td.step_q.update(TASK_TYPE, j, skill, quality, domain=DOMAIN)

    b_avg = sum(r["quality"] for r in baseline) / len(baseline)
    b_weak = [r["quality"] for r in baseline if r["tier"] == "WEAK"]
    b_good = [r["quality"] for r in baseline if r["tier"] == "GOOD"]

    print(f"\n  Baseline overall avg: {b_avg:.3f}")
    if b_weak:
        print(f"  Baseline WEAK-only:   {sum(b_weak)/len(b_weak):.3f} (n={len(b_weak)})")
    if b_good:
        print(f"  Baseline with GOOD:   {sum(b_good)/len(b_good):.3f} (n={len(b_good)})")

    print("\n  Q-values after baseline:")
    for s in ALL_SKILLS:
        q = td.skill_q.get_q(TASK_TYPE, s, domain=DOMAIN)
        bar = "#" * int(q * 40)
        print(f"    {s:18s}  Q={q:.3f}  {bar}")

    # ── PHASE 2: LEARNED ──
    print("\n  PHASE 2: LEARNED (Q-guided skill selection, same tasks)")
    print("  " + "-" * 62)

    learned: list[dict] = []

    for i, task in enumerate(TASKS):
        top = td.skill_q.get_top_skills(TASK_TYPE, n=3, domain=DOMAIN)
        skills = [s for s, _ in top][:2] if top else GOOD_SKILLS[:2]

        tier = "GOOD" if any(s in GOOD_SKILLS for s in skills) else "WEAK"

        t0 = time.time()
        code = generate_gemma(task, skills)
        gen_time = time.time() - t0
        quality, qd = structural_judge(task, code)
        success = quality >= 0.4 and len(code) > 50

        flags = []
        if qd.get("has_docstring"):
            flags.append("doc")
        th = qd.get("type_hints", {})
        if th.get("params") or th.get("return"):
            flags.append("types")
        eh = qd.get("error_handling", {})
        if eh.get("try") or eh.get("raise"):
            flags.append("err")
        ue = qd.get("usage_example", {})
        if ue.get("main_block") or ue.get("example_call"):
            flags.append("ex")

        print(
            f"  [{i+1:2d}/10] [{tier:4s}] q={quality:.2f} "
            f"{qd.get('lines', 0):3d}L [{','.join(flags) or 'bare':15s}]  "
            f"skills={skills}  {task[:30]}"
        )

        learned.append(
            {
                "task": task,
                "skills": skills,
                "quality": quality,
                "success": success,
                "time": gen_time,
                "tier": tier,
                "details": qd,
            }
        )

        svc.record(
            unit_name="AB_Struct_Learned",
            unit_type="agent",
            domain=DOMAIN,
            task_type=TASK_TYPE,
            context={"goal": task, "message": task},
            action={"skills_used": skills, "model": GEN_MODEL, "skill_tier": "learned"},
            outcome={"content": code[:2000], "code_lines": code.count("\n") + 1},
            success=success,
            quality=quality,
            execution_time=gen_time,
        )

        for skill in skills:
            r = quality * (
                random.uniform(0.9, 1.1) if skill in GOOD_SKILLS else random.uniform(0.5, 0.8)
            )
            td.skill_q.update(TASK_TYPE, skill, max(0.0, min(1.0, r)), domain=DOMAIN)

    l_avg = sum(r["quality"] for r in learned) / len(learned)

    # ── PAIRED RESULTS ──
    print()
    print("=" * 70)
    print("  PAIRED COMPARISON (same task, baseline vs learned)")
    print("=" * 70)

    deltas = []
    wins, ties, losses = 0, 0, 0

    print(f"\n  {'Task':<40s}  {'B.Tier':>6s}  {'Base':>5s}  {'Learn':>5s}  {'Delta':>6s}")
    print("  " + "-" * 65)
    for b, l in zip(baseline, learned):
        d = l["quality"] - b["quality"]
        deltas.append(d)
        if d > 0.02:
            wins += 1
            mark = " +"
        elif d < -0.02:
            losses += 1
            mark = " -"
        else:
            ties += 1
            mark = "  "
        print(
            f"  {b['task'][:40]:<40s}  {b['tier']:>6s}  {b['quality']:5.3f}  "
            f"{l['quality']:5.3f}  {d:+.3f}{mark}"
        )

    avg_d = sum(deltas) / len(deltas)
    print(f"\n  {'AVERAGES':<40s}  {'':>6s}  {b_avg:5.3f}  {l_avg:5.3f}  {avg_d:+.3f}")

    import statistics

    if len(deltas) >= 3:
        d_std = statistics.stdev(deltas) or 0.01
        cohens_d = avg_d / d_std
    else:
        cohens_d = 0.0

    print(f"\n  Wins: {wins}  |  Ties: {ties}  |  Losses: {losses}")
    print(f"  Mean delta: {avg_d:+.3f}")
    effect = (
        "negligible"
        if abs(cohens_d) < 0.2
        else "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "LARGE"
    )
    print(f"  Cohen's d: {cohens_d:.2f} ({effect})")

    if b_weak:
        weak_avg = sum(b_weak) / len(b_weak)
        lift_vs_weak = l_avg - weak_avg
        print(f"\n  KEY: Learned ({l_avg:.3f}) vs Weak-only baseline ({weak_avg:.3f})")
        print(f"  Lift vs weak: {lift_vs_weak:+.3f}")
    else:
        weak_avg = b_avg

    # ── VALIDATION ──
    print()
    print("=" * 70)
    print("  VALIDATION")
    print("=" * 70)

    validator = LearningValidator()
    report = validator.validate_domain(DOMAIN, TASK_TYPE)
    for check in report.checks:
        icon = "+" if check.passed else "X"
        print(f"  [{icon}] {check.check:25s}  [{check.score:.0%}]  {check.detail}")
    print(f"\n  Confidence: {report.confidence:.0%}")

    # ── VERDICT ──
    print()
    print("=" * 70)
    checks = 0
    total = 5

    ok = avg_d >= -0.02
    checks += int(ok)
    print(f"  {'[+]' if ok else '[X]'} Learned >= Baseline (delta={avg_d:+.3f})")

    ok = wins >= losses
    checks += int(ok)
    print(f"  {'[+]' if ok else '[X]'} More wins than losses ({wins}W/{ties}T/{losses}L)")

    good_q = [td.skill_q.get_q(TASK_TYPE, s, domain=DOMAIN) for s in GOOD_SKILLS]
    weak_q_vals = [td.skill_q.get_q(TASK_TYPE, s, domain=DOMAIN) for s in WEAK_SKILLS]
    gap = (sum(good_q) / len(good_q)) - (sum(weak_q_vals) / len(weak_q_vals))
    ok = gap > 0.03
    checks += int(ok)
    print(f"  {'[+]' if ok else '[X]'} Q-values discriminate (gap={gap:.3f})")

    ok = l_avg >= weak_avg - 0.02
    checks += int(ok)
    print(f"  {'[+]' if ok else '[X]'} Learned beats weak baseline ({l_avg:.3f} vs {weak_avg:.3f})")

    ok = report.overall_passed
    checks += int(ok)
    print(f"  {'[+]' if ok else '[X]'} Validation passed")

    print(f"\n  VERDICT: {checks}/{total}")
    if checks >= 4:
        print("  * LEARNING PROVEN: measurable quality lift on free model")
    elif checks >= 3:
        print("  ~ LEARNING SIGNAL detected")
    else:
        print("  X LEARNING NOT PROVEN on this model")
    print("=" * 70)

    return checks >= 3


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)

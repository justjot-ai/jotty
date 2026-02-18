"""
Jotty Deep Evaluation: Complex Tasks + Learning Over Time
==========================================================

Runs increasingly complex tasks through Orchestrator.run() with real Anthropic LLM,
then measures whether the system learns and improves over repeated runs.

Evaluates:
  1. Task execution quality (correctness, depth, structure)
  2. Learning accumulation (episodes recorded, patterns extracted)
  3. Guidance injection (does prior learning improve subsequent runs?)
  4. Cost efficiency (tokens, timing)
  5. Cross-domain transfer (does coding knowledge help research?)

Usage:
  python scripts/eval_jotty_learning.py
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

RESULTS_DIR = PROJECT_ROOT / "scripts" / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)


def separator(title: str) -> None:
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def subsep(title: str) -> None:
    print(f"\n  --- {title} ---\n")


# =============================================================================
# TASK DEFINITIONS — Progressively harder, multi-domain
# =============================================================================

TASKS = [
    # --- ROUND 1: Warm-up (establishes baseline learning) ---
    {
        "id": "R1_coding_basics",
        "domain": "coding",
        "goal": (
            "Write a Python function that implements a thread-safe LRU cache with TTL "
            "(time-to-live) expiration. Requirements:\n"
            "1. O(1) get and put operations\n"
            "2. Thread-safe with minimal locking (use RLock)\n"
            "3. TTL-based expiration checked lazily on access\n"
            "4. max_size parameter with LRU eviction\n"
            "5. Include type hints and docstrings\n"
            "6. Write 3 unit tests proving correctness\n\n"
            "Return the complete implementation. Do NOT use any tools."
        ),
        "checks": ["class", "def get", "def put", "threading", "test"],
        "min_length": 800,
    },
    {
        "id": "R1_research_basics",
        "domain": "research",
        "goal": (
            "Analyze the economic implications of Large Language Models on the labor market. "
            "Structure your analysis as:\n"
            "1. DIRECT EFFECTS: Which jobs are most at risk? Quantify with percentages.\n"
            "2. INDIRECT EFFECTS: New job categories created. Name at least 5.\n"
            "3. MACRO IMPACT: Effect on GDP, productivity, inequality (cite Acemoglu, Autor).\n"
            "4. POLICY RECOMMENDATIONS: 3 specific policy proposals with cost estimates.\n"
            "5. TIMELINE: When do each of these effects materialize? (2024-2035)\n\n"
            "Be specific with numbers, not vague. Do NOT use any tools."
        ),
        "checks": ["direct", "indirect", "gdp", "policy", "2024"],
        "min_length": 1200,
    },
    # --- ROUND 2: Harder tasks (should benefit from R1 learning) ---
    {
        "id": "R2_coding_advanced",
        "domain": "coding",
        "goal": (
            "Design and implement a distributed rate limiter in Python that works across "
            "multiple server instances. Requirements:\n"
            "1. Token bucket algorithm with sliding window fallback\n"
            "2. Uses Redis for cross-instance state (show the Redis commands)\n"
            "3. Handles Redis failures gracefully (local fallback)\n"
            "4. Supports multiple rate limit tiers (free: 100/hr, pro: 1000/hr, enterprise: 10000/hr)\n"
            "5. Lua script for atomic Redis operations to prevent race conditions\n"
            "6. Include monitoring hooks (prometheus-style counters)\n"
            "7. Write integration test pseudocode\n\n"
            "This must be production-grade. Do NOT use any tools."
        ),
        "checks": ["token bucket", "redis", "lua", "fallback", "class"],
        "min_length": 1500,
    },
    {
        "id": "R2_cross_domain",
        "domain": "cross_domain",
        "goal": (
            "You are a systems architect. Design a real-time fraud detection system for "
            "a payment processor handling 50,000 transactions per second. Your design must:\n\n"
            "A. DATA PIPELINE:\n"
            "   - Kafka ingestion with exactly-once semantics\n"
            "   - Feature engineering (velocity, geolocation, device fingerprint, behavioral)\n"
            "   - Real-time feature store (Redis) + batch feature store (Hive/Iceberg)\n\n"
            "B. ML MODELS:\n"
            "   - Ensemble: XGBoost (fast) + neural network (deep patterns) + rules engine\n"
            "   - Online learning: model updates every 5 minutes without downtime\n"
            "   - Explain decisions (SHAP values for compliance)\n\n"
            "C. SYSTEM DESIGN:\n"
            "   - P99 latency < 50ms for scoring\n"
            "   - 99.99% availability with multi-region failover\n"
            "   - Capacity planning numbers (CPU, memory, network)\n\n"
            "D. ADVERSARIAL ROBUSTNESS:\n"
            "   - How do you handle concept drift?\n"
            "   - Adversarial attacks on the model (data poisoning, evasion)\n"
            "   - Human-in-the-loop escalation for borderline cases\n\n"
            "Be extremely specific with technology choices and numbers. "
            "Do NOT use any tools."
        ),
        "checks": ["kafka", "xgboost", "latency", "shap", "drift", "failover"],
        "min_length": 2000,
    },
    # --- ROUND 3: Ultra-complex (maximum difficulty, tests deep learning) ---
    {
        "id": "R3_synthesis",
        "domain": "synthesis",
        "goal": (
            "NOVEL RESEARCH PROPOSAL: Design a new approach to continual learning in "
            "neural networks that draws from three domains:\n\n"
            "1. NEUROSCIENCE: Complementary Learning Systems theory (hippocampus + neocortex), "
            "   synaptic consolidation (Kirkpatrick et al., 2017), neurogenesis\n\n"
            "2. INFORMATION THEORY: Minimum Description Length, Rate-Distortion theory, "
            "   Information Bottleneck (Tishby), Kolmogorov complexity\n\n"
            "3. EVOLUTIONARY BIOLOGY: Baldwin effect, genetic assimilation, "
            "   punctuated equilibrium, niche construction\n\n"
            "Your proposal must include:\n"
            "A. A novel architecture (with diagram described in text)\n"
            "B. Mathematical formulation of the loss function\n"
            "C. Theoretical analysis: prove it avoids catastrophic forgetting under assumptions X\n"
            "D. Experimental design: 3 benchmarks, baselines, expected results\n"
            "E. Computational complexity analysis (training and inference)\n"
            "F. Comparison to EWC, PackNet, Progressive Networks, and LoRA\n\n"
            "This should be publishable-quality. Do NOT use any tools."
        ),
        "checks": ["hippocampus", "consolidation", "loss", "theorem", "complexity", "ewc"],
        "min_length": 2500,
    },
]


# =============================================================================
# EVALUATION ENGINE
# =============================================================================


class JottyEvaluator:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.learning_snapshots: List[Dict[str, Any]] = []

    def _get_orchestrator(self):
        from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

        orch = object.__new__(Orchestrator)
        orch.config = type(
            "C",
            (),
            {
                "domain": "general",
                "base_path": None,
                "learning_wait_timeout_seconds": 0,
            },
        )()
        orch.agents = []
        orch.mode = "single"
        orch.runners = {}
        orch._runners_built = False
        orch._efficiency_stats = {}
        orch._intelligence_metrics = {}
        orch._engine = None
        orch._learning_ready = asyncio.Event()
        orch._learning_ready.set()
        return orch

    def _get_learning_service(self):
        from Jotty.core.intelligence.learning.learning_service import LearningService

        return LearningService.get_instance()

    def _snapshot_learning(self, label: str) -> Dict[str, Any]:
        ls = self._get_learning_service()
        count = ls._store.get_episode_count()

        guidance_general = ls.query(domain="general", task_type="run")
        guidance_coding = ls.query(domain="coding", task_type="run")
        guidance_research = ls.query(domain="research", task_type="run")

        ctx_str = ls.build_context_string(domain="general", task_type="run")

        snapshot = {
            "label": label,
            "timestamp": time.time(),
            "episode_count": count,
            "guidance_general_keys": list(guidance_general.keys()),
            "guidance_general_has_learning": guidance_general.get("has_learning", False),
            "guidance_coding_has_learning": guidance_coding.get("has_learning", False),
            "guidance_research_has_learning": guidance_research.get("has_learning", False),
            "context_string_length": len(ctx_str),
            "context_string_preview": ctx_str[:300] if ctx_str else "(empty)",
        }
        self.learning_snapshots.append(snapshot)
        return snapshot

    async def run_task(self, task: Dict[str, Any], task_num: int, total: int) -> Dict[str, Any]:
        subsep(f"Task {task_num}/{total}: {task['id']}")

        orch = self._get_orchestrator()

        before_snap = self._snapshot_learning(f"before_{task['id']}")
        print(
            f"  Learning state: {before_snap['episode_count']} episodes, "
            f"has_learning={before_snap['guidance_general_has_learning']}"
        )

        start = time.time()
        error_msg = None
        content = ""

        try:
            result = await orch.chat(
                message=task["goal"],
                provider="anthropic",
                learn=True,
            )
            content = getattr(result, "content", str(result))
        except Exception as e:
            error_msg = str(e)
            print(f"  ERROR: {error_msg[:200]}")

        elapsed = time.time() - start

        after_snap = self._snapshot_learning(f"after_{task['id']}")
        episodes_added = after_snap["episode_count"] - before_snap["episode_count"]

        # Quality checks
        content_lower = content.lower() if content else ""
        checks_passed = 0
        checks_detail = {}
        for check in task.get("checks", []):
            found = check.lower() in content_lower
            checks_detail[check] = found
            if found:
                checks_passed += 1

        total_checks = len(task.get("checks", []))
        length_ok = len(content) >= task.get("min_length", 0)

        quality_score = 0.0
        if content and not error_msg:
            check_ratio = checks_passed / total_checks if total_checks else 1.0
            length_ratio = min(len(content) / max(task.get("min_length", 1), 1), 2.0) / 2.0
            quality_score = check_ratio * 0.6 + length_ratio * 0.3 + (0.1 if length_ok else 0.0)

        result_record = {
            "task_id": task["id"],
            "domain": task.get("domain", "general"),
            "success": error_msg is None and len(content) > 100,
            "error": error_msg,
            "content_length": len(content),
            "min_length": task.get("min_length", 0),
            "length_ok": length_ok,
            "checks_passed": checks_passed,
            "checks_total": total_checks,
            "checks_detail": checks_detail,
            "quality_score": round(quality_score, 3),
            "elapsed_seconds": round(elapsed, 2),
            "episodes_before": before_snap["episode_count"],
            "episodes_after": after_snap["episode_count"],
            "episodes_added": episodes_added,
            "learning_injected": before_snap["context_string_length"] > 0,
            "content_preview": content[:500] if content else "(empty)",
        }
        self.results.append(result_record)

        status = "PASS" if result_record["success"] else "FAIL"
        print(
            f"  [{status}] Quality={quality_score:.2f} | "
            f"Checks={checks_passed}/{total_checks} | "
            f"Length={len(content)}/{task.get('min_length', 0)} | "
            f"Time={elapsed:.1f}s"
        )
        print(
            f"  Learning: +{episodes_added} episodes | "
            f"Guidance injected: {result_record['learning_injected']}"
        )

        if content:
            print(f"  Preview: {content[:150].replace(chr(10), ' ')}...")

        return result_record

    async def run_all(self):
        separator("JOTTY DEEP EVALUATION — Complex Tasks + Learning Over Time")
        print(f"  Date: {datetime.now().isoformat()}")
        print(f"  Tasks: {len(TASKS)}")
        print(f"  Provider: Anthropic (real LLM)")
        print(f"  Learning: ON (learn=True)")

        initial_snap = self._snapshot_learning("initial")
        print(f"\n  Initial learning state: {initial_snap['episode_count']} episodes")

        for i, task in enumerate(TASKS, 1):
            await self.run_task(task, i, len(TASKS))

        final_snap = self._snapshot_learning("final")

        separator("EVALUATION RESULTS")
        self._print_results(initial_snap, final_snap)
        self._save_report(initial_snap, final_snap)

    def _print_results(self, initial_snap, final_snap):
        total = len(self.results)
        successes = sum(1 for r in self.results if r["success"])
        avg_quality = sum(r["quality_score"] for r in self.results) / total if total else 0
        avg_time = sum(r["elapsed_seconds"] for r in self.results) / total if total else 0
        total_episodes = final_snap["episode_count"] - initial_snap["episode_count"]

        print(f"  Tasks Run:        {total}")
        print(f"  Successes:        {successes}/{total} ({successes/total*100:.0f}%)")
        print(f"  Avg Quality:      {avg_quality:.3f}")
        print(f"  Avg Time:         {avg_time:.1f}s")
        print(f"  Episodes Added:   {total_episodes}")
        print(f"  Final Episodes:   {final_snap['episode_count']}")

        subsep("Per-Task Breakdown")
        for r in self.results:
            status = "PASS" if r["success"] else "FAIL"
            learning_marker = "L" if r["learning_injected"] else "-"
            print(
                f"  [{status}][{learning_marker}] {r['task_id']:30s} "
                f"Q={r['quality_score']:.2f}  "
                f"Checks={r['checks_passed']}/{r['checks_total']}  "
                f"Len={r['content_length']:>5d}  "
                f"Time={r['elapsed_seconds']:>5.1f}s  "
                f"+{r['episodes_added']} ep"
            )

        subsep("Learning Progression")
        prev_count = initial_snap["episode_count"]
        for snap in self.learning_snapshots:
            if snap["label"].startswith("after_"):
                task_id = snap["label"].replace("after_", "")
                delta = snap["episode_count"] - prev_count
                has_learning = snap["guidance_general_has_learning"]
                ctx_len = snap["context_string_length"]
                print(
                    f"  After {task_id:30s}: "
                    f"episodes={snap['episode_count']:>3d} (+{delta})  "
                    f"has_learning={has_learning}  "
                    f"context_len={ctx_len}"
                )
                prev_count = snap["episode_count"]

        subsep("Cross-Domain Transfer Check")
        coding_results = [r for r in self.results if r["domain"] == "coding"]
        if len(coding_results) >= 2:
            r1, r2 = coding_results[0], coding_results[1]
            print(
                f"  Coding R1 → R2: quality {r1['quality_score']:.2f} → {r2['quality_score']:.2f}"
            )
            if r2["learning_injected"]:
                print(f"  Learning WAS injected into R2 (good: prior coding experience used)")
            else:
                print(f"  Learning was NOT injected into R2 (needs more episodes)")

        cross_domain = [r for r in self.results if r["domain"] == "cross_domain"]
        if cross_domain:
            cd = cross_domain[0]
            print(
                f"  Cross-domain task: quality={cd['quality_score']:.2f}, "
                f"learning_injected={cd['learning_injected']}"
            )

        # --- FINAL RATING ---
        subsep("JOTTY RATING")

        scores = {
            "Execution Success": successes / total if total else 0,
            "Output Quality": avg_quality,
            "Learning Recording": min(total_episodes / total, 1.0) if total else 0,
            "Learning Injection": (
                sum(1 for r in self.results if r["learning_injected"]) / total if total else 0
            ),
            "Speed (sub-30s avg)": max(0, 1.0 - (avg_time - 10) / 50) if avg_time else 0,
            "Depth (check pass rate)": (
                sum(r["checks_passed"] for r in self.results)
                / sum(r["checks_total"] for r in self.results)
                if sum(r["checks_total"] for r in self.results) > 0
                else 0
            ),
        }

        for name, score in scores.items():
            bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
            print(f"  {name:30s} [{bar}] {score*100:5.1f}%")

        overall = sum(scores.values()) / len(scores)
        stars = int(overall * 5)
        star_str = "*" * stars + "." * (5 - stars)
        print(f"\n  OVERALL RATING: [{star_str}] {overall*100:.1f}% ({stars}/5 stars)")

        if overall >= 0.8:
            print("  VERDICT: Excellent — Jotty executes complex tasks well and learns.")
        elif overall >= 0.6:
            print("  VERDICT: Good — Core execution works, learning needs more iterations.")
        elif overall >= 0.4:
            print("  VERDICT: Fair — Execution works but learning pipeline needs tuning.")
        else:
            print("  VERDICT: Needs work — Significant issues in execution or learning.")

    def _save_report(self, initial_snap, final_snap):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = RESULTS_DIR / f"{ts}_jotty_evaluation.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "tasks_count": len(TASKS),
            "results": self.results,
            "learning_snapshots": self.learning_snapshots,
            "initial_episodes": initial_snap["episode_count"],
            "final_episodes": final_snap["episode_count"],
        }

        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\n  Report saved: {report_path}")

        md_path = RESULTS_DIR / f"{ts}_jotty_evaluation.md"
        md_lines = [
            f"# Jotty Deep Evaluation — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"**Tasks:** {len(TASKS)} | **Provider:** Anthropic",
            "",
            "## Results",
            "",
            "| Task | Domain | Quality | Checks | Length | Time | Learning |",
            "|------|--------|---------|--------|--------|------|----------|",
        ]
        for r in self.results:
            status = "Pass" if r["success"] else "FAIL"
            md_lines.append(
                f"| {r['task_id']} | {r['domain']} | {r['quality_score']:.2f} | "
                f"{r['checks_passed']}/{r['checks_total']} | {r['content_length']} | "
                f"{r['elapsed_seconds']:.1f}s | {'+' + str(r['episodes_added']) + ' ep'} |"
            )

        md_lines.extend(["", "## Full Outputs", ""])
        for r in self.results:
            md_lines.append(f"### {r['task_id']}")
            md_lines.append("")
            md_lines.append(r.get("content_preview", "(no content)"))
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")

        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"  Markdown report: {md_path}")


# =============================================================================
# MAIN
# =============================================================================


async def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Load .env first.")
        sys.exit(1)

    print(f"ANTHROPIC_API_KEY: ...{os.environ['ANTHROPIC_API_KEY'][-8:]}")

    evaluator = JottyEvaluator()
    await evaluator.run_all()


if __name__ == "__main__":
    asyncio.run(main())

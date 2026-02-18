"""
Jotty Deep Evaluation v2: Learning Improvement + Auto-Curriculum
=================================================================

Tests whether Jotty's learning system genuinely improves over time by:
  1. Running tasks that are HARD enough to show quality variance
  2. Running similar tasks TWICE so we can measure improvement
  3. Auto-executing curriculum for weak domains
  4. Verifying Thompson Sampling, causal patterns, and cross-domain transfer

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
# TASK DEFINITIONS — Hard enough to show quality variance + repeated domains
# =============================================================================

ROUND_1_TASKS = [
    {
        "id": "R1_coding",
        "domain": "coding",
        "goal": (
            "Implement a lock-free concurrent skip list in Python that supports "
            "insert, delete, and search operations. Requirements:\n"
            "1. Use atomic CAS (Compare-And-Swap) via ctypes or threading primitives\n"
            "2. Probabilistic leveling with p=0.5\n"
            "3. Memory-safe: no dangling references during concurrent delete\n"
            "4. Linearizable: concurrent operations must appear atomic\n"
            "5. Prove correctness informally for each operation\n"
            "6. Include 3 concurrent stress tests using threading\n"
            "7. Analyze time complexity for each operation (expected and worst case)\n\n"
            "This is extremely hard. Most implementations get the delete wrong. Do NOT use any tools."
        ),
        "checks": ["class", "insert", "delete", "search", "threading", "cas", "test"],
        "min_length": 2000,
    },
    {
        "id": "R1_economics",
        "domain": "economics",
        "goal": (
            "Construct a formal economic model of AI-induced labor market disruption "
            "using the Acemoglu & Restrepo (2018) task-based framework. Your model must:\n\n"
            "1. FORMAL MODEL: Write the production function Y = F(A_L * L, A_K * K) with "
            "   task allocation threshold I*. Derive the equilibrium wage equation.\n"
            "2. CALIBRATION: Use BLS data to estimate displacement rates for 5 specific "
            "   occupations. Provide exact numbers (e.g., 'accountants: 34% task displacement').\n"
            "3. DYNAMICS: Solve the transition path using Bellman equations. Show how adjustment "
            "   costs create a J-curve in aggregate output.\n"
            "4. WELFARE ANALYSIS: Compute Gini coefficient change using Lorenz curves. "
            "   Show the math for both short-run and long-run equilibria.\n"
            "5. POLICY: Design an optimal tax-and-transfer scheme using Mirrlees framework. "
            "   Prove it's incentive compatible.\n\n"
            "Include ALL mathematical derivations. Cite specific papers with years. "
            "Do NOT use any tools."
        ),
        "checks": ["acemoglu", "production function", "bellman", "gini", "mirrlees", "equilibrium"],
        "min_length": 2500,
    },
    {
        "id": "R1_system_design",
        "domain": "system_design",
        "goal": (
            "Design a globally distributed CRDT-based collaborative text editor "
            "(like Google Docs) that supports 10M concurrent users. Must include:\n\n"
            "1. CRDT ALGORITHM: Implement RGA (Replicated Growable Array) with tombstone GC.\n"
            "   Show the data structure, merge function, and prove commutativity.\n"
            "2. NETWORK: Causal broadcast protocol with vector clocks. Handle partitions.\n"
            "   Calculate bandwidth: with 10M users, 5 chars/sec each = ? messages/sec.\n"
            "3. STORAGE: LSM-tree based operation log with compaction. Calculate storage "
            "   for 1B operations/day.\n"
            "4. PRESENCE: Who's online, cursor positions, selections. Heartbeat protocol.\n"
            "5. CONFLICT RESOLUTION: Last-writer-wins for formatting, operational transform "
            "   fallback for concurrent inserts at same position.\n"
            "6. LATENCY BUDGET: P50 < 100ms, P99 < 500ms. Show the breakdown.\n\n"
            "Include actual capacity planning numbers. Do NOT use any tools."
        ),
        "checks": ["crdt", "rga", "vector clock", "tombstone", "bandwidth", "latency"],
        "min_length": 2000,
    },
]

ROUND_2_TASKS = [
    {
        "id": "R2_coding",
        "domain": "coding",
        "goal": (
            "Implement a work-stealing thread pool in Python with the following:\n"
            "1. Each worker has a local deque (double-ended queue) for tasks\n"
            "2. When a worker's deque is empty, it steals from another worker's deque\n"
            "3. Use lock-free or fine-grained locking on deque operations\n"
            "4. Support task dependencies (DAG scheduling)\n"
            "5. Adaptive thread count based on CPU utilization\n"
            "6. Include a ForkJoinTask abstraction with fork() and join() methods\n"
            "7. Write 3 tests: fibonacci fork-join, matrix multiply, and steal verification\n"
            "8. Prove the scheduler is deadlock-free\n\n"
            "This should be production-quality. Do NOT use any tools."
        ),
        "checks": ["class", "deque", "steal", "fork", "join", "test", "deadlock"],
        "min_length": 2000,
    },
    {
        "id": "R2_economics",
        "domain": "economics",
        "goal": (
            "Build a DSGE (Dynamic Stochastic General Equilibrium) model of monetary policy "
            "in the age of Central Bank Digital Currencies (CBDC). Requirements:\n\n"
            "1. MODEL: Three-sector New Keynesian model (households, firms, central bank) with "
            "   CBDC as a third form of money alongside cash and deposits.\n"
            "2. EQUATIONS: Write the full system — Euler equation, Phillips curve, Taylor rule "
            "   modified for CBDC interest rate. Show steady-state derivation.\n"
            "3. CALIBRATION: Quarterly parameters from US data. Cite Christiano, Eichenbaum, Evans.\n"
            "4. SIMULATION: Impulse responses to: (a) CBDC adoption shock, (b) digital bank run, "
            "   (c) negative CBDC rate. Describe what the IRFs should look like.\n"
            "5. FINANCIAL STABILITY: Prove conditions under which CBDC prevents bank runs "
            "   (Diamond-Dybvig framework extension).\n\n"
            "All equations must be fully derived, not just stated. Do NOT use any tools."
        ),
        "checks": ["euler", "phillips", "taylor", "cbdc", "steady state", "diamond"],
        "min_length": 2500,
    },
    {
        "id": "R2_system_design",
        "domain": "system_design",
        "goal": (
            "Design a real-time multiplayer game server infrastructure supporting 1M concurrent "
            "players with authoritative server model. Include:\n\n"
            "1. NETCODE: Client-side prediction, server reconciliation, and entity interpolation.\n"
            "   Show the state buffer, rollback logic, and calculate bandwidth per player.\n"
            "2. SPATIAL: Distributed spatial hashing with interest management. Players only receive "
            "   updates within their Area of Interest (AoI). Calculate AoI message reduction.\n"
            "3. MATCHMAKING: ELO/Glicko-2 with skill-based matchmaking. Queue time vs match quality "
            "   tradeoff. Show the math for queue wait time as function of pool size.\n"
            "4. ANTI-CHEAT: Statistical anomaly detection on player actions. Speed hack, aim assist, "
            "   and wallhack detection algorithms with false positive analysis.\n"
            "5. INFRASTRUCTURE: Global server mesh with Agones/Kubernetes. Auto-scaling from "
            "   1K to 1M players. Calculate server costs at each scale point.\n"
            "6. TICK RATE: 60Hz server, 128Hz client. Jitter buffer design. Show timing diagram.\n\n"
            "Include capacity planning numbers for AWS. Do NOT use any tools."
        ),
        "checks": ["prediction", "reconciliation", "interpolation", "elo", "spatial", "tick"],
        "min_length": 2500,
    },
]


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

    def _snapshot_learning(self, label: str, domain: str = "") -> Dict[str, Any]:
        ls = self._get_learning_service()
        count = ls._store.get_episode_count()

        snap_domain = domain or "general"
        ctx_str = ls.build_context_string(domain=snap_domain, task_type="")

        snapshot = {
            "label": label,
            "timestamp": time.time(),
            "episode_count": count,
            "context_string_length": len(ctx_str),
            "domain": snap_domain,
        }
        self.learning_snapshots.append(snapshot)
        return snapshot

    async def run_task(
        self, task: Dict[str, Any], task_num: int, total: int, learn: bool = True
    ) -> Dict[str, Any]:
        subsep(f"Task {task_num}/{total}: {task['id']}{'' if learn else ' [NO LEARNING]'}")

        orch = self._get_orchestrator()
        ls = self._get_learning_service()

        from Jotty.core.intelligence.learning.learning_service import classify_domain

        det_domain, det_task = classify_domain(task["goal"])
        before_snap = self._snapshot_learning(f"before_{task['id']}", domain=det_domain)
        optimal = ls.get_optimal_execution_params(det_domain, det_task, task["goal"])
        retrieval = ls.retrieve_similar_responses(det_domain, det_task, task["goal"])

        print(f"  Learning state: {before_snap['episode_count']} episodes")
        print(f"  Domain: {det_domain}/{det_task}")
        print(
            f"  Thompson: strategy={optimal.get('strategy')}, "
            f"model={optimal.get('model') or 'default'}, "
            f"temp={optimal.get('temperature') or 'default'}, "
            f"paradigm={optimal.get('paradigm') or 'default'}, "
            f"explore={optimal.get('exploration')}"
        )
        if optimal.get("exploration_reason"):
            print(f"  Explore reason: {optimal['exploration_reason']}")
        if optimal.get("tools_hint"):
            print(f"  Tools hint: {optimal['tools_hint']}")
        if retrieval:
            print(
                f"  Retrieved {len(retrieval)} prior examples "
                f"(best quality={retrieval[0].get('quality', 0):.2f})"
            )
        else:
            print(f"  No prior examples (cold start)")

        start = time.time()
        error_msg = None
        content = ""

        try:
            result = await orch.chat(
                message=task["goal"],
                provider="anthropic",
                learn=learn,
            )
            content = getattr(result, "content", str(result))
        except Exception as e:
            error_msg = str(e)
            print(f"  ERROR: {error_msg[:200]}")

        elapsed = time.time() - start

        after_snap = self._snapshot_learning(f"after_{task['id']}", domain=det_domain)
        episodes_added = after_snap["episode_count"] - before_snap["episode_count"]

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
            from Jotty.core.intelligence.learning.learning_service import analyze_response

            ra = analyze_response(content, task["goal"])
            heuristic_q = ra.get("quality_score", 0.5)

            check_ratio = checks_passed / total_checks if total_checks else 1.0
            depth_target = task.get("min_length", 2000) * 3
            depth_ratio = min(len(content) / max(depth_target, 1), 1.0)
            quality_score = (
                check_ratio * 0.35
                + heuristic_q * 0.35
                + depth_ratio * 0.20
                + (0.10 if length_ok else 0.0)
            )

        result_record = {
            "task_id": task["id"],
            "domain": task.get("domain", "general"),
            "detected_domain": det_domain,
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
            "thompson_strategy": optimal.get("strategy", "default"),
            "thompson_paradigm": optimal.get("paradigm") or "default",
            "thompson_exploration": optimal.get("exploration", False),
            "thompson_reason": optimal.get("exploration_reason", ""),
            "retrieval_examples": len(retrieval),
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
            f"  Learning: +{episodes_added} ep | "
            f"thompson={result_record['thompson_strategy']} | "
            f"paradigm={result_record['thompson_paradigm']} | "
            f"retrieval={result_record['retrieval_examples']}"
        )

        if content:
            print(f"  Preview: {content[:120].replace(chr(10), ' ')}...")

        return result_record

    async def run_all(self):
        separator("JOTTY DEEP EVALUATION v2 — Learning Improvement + A/B Baseline")
        print(f"  Date: {datetime.now().isoformat()}")
        print(f"  A/B Baseline: 1 task WITHOUT learning (control)")
        print(f"  Round 1 Tasks: {len(ROUND_1_TASKS)} (cold start with learning)")
        print(f"  Round 2 Tasks: {len(ROUND_2_TASKS)} (with accumulated learning)")
        print(f"  Provider: Anthropic (real LLM)")

        initial_snap = self._snapshot_learning("initial")
        print(f"\n  Initial learning state: {initial_snap['episode_count']} episodes")

        # ── A/B BASELINE: Run one task WITHOUT learning ──
        separator("A/B BASELINE — Same task, NO learning context")
        baseline_task = {
            "id": "BASELINE_coding",
            "domain": "coding",
            "goal": ROUND_2_TASKS[0]["goal"],
            "checks": ROUND_2_TASKS[0]["checks"],
            "min_length": ROUND_2_TASKS[0]["min_length"],
        }
        await self.run_task(baseline_task, 1, 1, learn=False)

        # ── ROUND 1: Baseline ──
        separator("ROUND 1 — Cold start (learning ON, no prior data)")
        for i, task in enumerate(ROUND_1_TASKS, 1):
            await self.run_task(task, i, len(ROUND_1_TASKS))

        mid_snap = self._snapshot_learning("mid_round")

        # ── AUTO-CURRICULUM: Practice weak domains ──
        separator("AUTO-CURRICULUM — Practice weak domains")
        ls = self._get_learning_service()
        weak = ls.identify_weak_domains(min_episodes=1)
        if weak:
            print(f"  Weak domains: {len(weak)}")
            for w in weak:
                print(f"    {w['domain']:15s}: quality={w['avg_quality']:.3f} — {w['needs']}")

            print("\n  Running auto-curriculum...")
            curriculum_results = await ls.run_curriculum(
                max_tasks=2, provider="anthropic", min_episodes=1
            )
            for cr in curriculum_results:
                status = "IMPROVED" if cr.get("improved") else "needs work"
                print(
                    f"    [{status}] {cr['domain']}: quality={cr['quality']:.3f} "
                    f"(target={cr['target_quality']:.2f})"
                )
        else:
            print("  No weak domains detected.")
            curriculum_results = []

        curriculum_snap = self._snapshot_learning("post_curriculum")

        # ── ROUND 2: Should benefit from learning ──
        separator("ROUND 2 — With Learning (same domains, harder tasks)")
        for i, task in enumerate(ROUND_2_TASKS, 1):
            await self.run_task(
                task, i + len(ROUND_1_TASKS), len(ROUND_1_TASKS) + len(ROUND_2_TASKS)
            )

        final_snap = self._snapshot_learning("final")

        separator("EVALUATION RESULTS")
        self._print_results(initial_snap, mid_snap, final_snap, curriculum_results, weak)
        self._save_report(initial_snap, final_snap)

    def _print_results(self, initial_snap, mid_snap, final_snap, curriculum_results, weak_domains):
        total = len(self.results)
        successes = sum(1 for r in self.results if r["success"])
        avg_quality = sum(r["quality_score"] for r in self.results) / total if total else 0
        avg_time = sum(r["elapsed_seconds"] for r in self.results) / total if total else 0

        # A/B Comparison
        baseline = [r for r in self.results if r["task_id"].startswith("BASELINE_")]
        learning = [r for r in self.results if not r["task_id"].startswith("BASELINE_")]

        if baseline:
            subsep("A/B COMPARISON — Learning vs No-Learning (same task)")
            for br in baseline:
                domain = br["domain"]
                lr = [r for r in self.results if r["task_id"] == f"R2_{domain}"]
                if lr:
                    lr = lr[0]
                    delta = lr["quality_score"] - br["quality_score"]
                    len_delta = lr["content_length"] - br["content_length"]
                    marker = "+" if delta > 0 else ("-" if delta < 0 else "=")
                    print(
                        f"  {domain:15s}: NO_LEARN={br['quality_score']:.2f} "
                        f"vs LEARNED={lr['quality_score']:.2f} "
                        f"({marker}{abs(delta):.2f})  "
                        f"depth: {br['content_length']}→{lr['content_length']} "
                        f"({'+' if len_delta > 0 else ''}{len_delta})"
                    )
                else:
                    print(
                        f"  {domain:15s}: NO_LEARN={br['quality_score']:.2f} "
                        f"(no matching R2 task)"
                    )

        total_learning = len(learning)
        successes_learning = sum(1 for r in learning if r["success"])
        avg_quality_learning = (
            sum(r["quality_score"] for r in learning) / total_learning if total_learning else 0
        )

        print(
            f"\n  Tasks Run:        {total} ({len(baseline)} baseline + {total_learning} learning)"
        )
        print(
            f"  Successes:        {successes_learning}/{total_learning} "
            f"({successes_learning/total_learning*100:.0f}%)"
        )
        print(f"  Avg Quality:      {avg_quality_learning:.3f}")
        print(f"  Avg Time:         {avg_time:.1f}s")
        print(
            f"  Episodes:         {initial_snap['episode_count']} → {final_snap['episode_count']}"
        )

        # ── Round comparison: did R2 improve over R1? ──
        subsep("Round-over-Round Improvement (KEY METRIC)")
        r1 = [r for r in self.results if r["task_id"].startswith("R1_")]
        r2 = [r for r in self.results if r["task_id"].startswith("R2_")]

        domains_tested = set(r["domain"] for r in r1) & set(r["domain"] for r in r2)
        improvement_count = 0
        for domain in sorted(domains_tested):
            r1_d = [r for r in r1 if r["domain"] == domain]
            r2_d = [r for r in r2 if r["domain"] == domain]
            r1_q = sum(r["quality_score"] for r in r1_d) / len(r1_d)
            r2_q = sum(r["quality_score"] for r in r2_d) / len(r2_d)
            r1_checks = (
                sum(r["checks_passed"] for r in r1_d) / sum(r["checks_total"] for r in r1_d)
                if sum(r["checks_total"] for r in r1_d)
                else 0
            )
            r2_checks = (
                sum(r["checks_passed"] for r in r2_d) / sum(r["checks_total"] for r in r2_d)
                if sum(r["checks_total"] for r in r2_d)
                else 0
            )
            delta = r2_q - r1_q
            check_delta = r2_checks - r1_checks
            improved = delta > -0.05 and r2_q > 0.6
            if improved:
                improvement_count += 1
            marker = "+" if delta > 0 else ("-" if delta < 0 else "=")
            print(
                f"  {domain:15s}: R1={r1_q:.2f} → R2={r2_q:.2f} ({marker}{abs(delta):.2f})  "
                f"checks: {r1_checks:.0%} → {r2_checks:.0%} ({marker}{abs(check_delta):.0%})"
            )

        if domains_tested:
            print(f"\n  Domains improved or maintained: {improvement_count}/{len(domains_tested)}")

        subsep("Per-Task Breakdown")
        for r in self.results:
            status = "PASS" if r["success"] else "FAIL"
            print(
                f"  [{status}] {r['task_id']:25s} "
                f"Q={r['quality_score']:.2f}  "
                f"Checks={r['checks_passed']}/{r['checks_total']}  "
                f"Len={r['content_length']:>6d}  "
                f"Time={r['elapsed_seconds']:>5.1f}s  "
                f"thompson={r['thompson_strategy']:20s}  "
                f"paradigm={r['thompson_paradigm']}"
            )

        # ── LLM Judge Verification ──
        subsep("LLM Judge Verification")
        ls = self._get_learning_service()
        conn = ls._store._get_conn()
        llm_judged = conn.execute(
            "SELECT episode_id, domain, quality, outcome FROM episodes "
            "WHERE outcome LIKE '%llm_judged%' ORDER BY timestamp"
        ).fetchall()
        print(f"  LLM-judged episodes: {len(llm_judged)}")
        for row in llm_judged:
            out = json.loads(row["outcome"]) if row["outcome"] else {}
            print(
                f"    {row['domain']:15s} blended={row['quality']:.3f}  "
                f"llm={out.get('llm_score', '?')}  heuristic={out.get('heuristic_quality', '?')}"
            )

        # ── Thompson Sampling arms ──
        subsep("Thompson Sampling Arms")
        for domain in sorted(domains_tested):
            arms = ls._get_arm_stats(domain)
            if arms:
                print(f"  {domain}:")
                for key, stats in sorted(arms.items()):
                    mean = stats["alpha"] / (stats["alpha"] + stats["beta"])
                    print(
                        f"    {key:30s} α={stats['alpha']:.2f} β={stats['beta']:.2f} "
                        f"mean={mean:.3f} n={stats['n']}"
                    )

        # ── Pattern types ──
        subsep("Pattern Extraction Detail")
        pattern_rows = conn.execute(
            "SELECT pattern_type, COUNT(*) as cnt FROM patterns GROUP BY pattern_type "
            "ORDER BY cnt DESC"
        ).fetchall()
        total_patterns = sum(r["cnt"] for r in pattern_rows)
        print(f"  Total patterns: {total_patterns}")
        for row in pattern_rows:
            print(f"    {row['pattern_type']:25s}: {row['cnt']} patterns")

        # Show causal patterns specifically
        causal = conn.execute(
            "SELECT source_domain, description, recommendation FROM patterns "
            "WHERE pattern_type = 'causal' ORDER BY source_domain"
        ).fetchall()
        if causal:
            print(f"\n  Causal patterns:")
            for p in causal:
                print(f"    [{p['source_domain']:12s}] {p['description'][:80]}")
                print(f"                  → {p['recommendation']}")

        # Show transfer patterns
        transfers = conn.execute(
            "SELECT source_domain, description, recommendation FROM patterns "
            "WHERE pattern_type = 'cross_domain_transfer'"
        ).fetchall()
        if transfers:
            print(f"\n  Cross-domain transfer patterns:")
            for p in transfers:
                print(f"    {p['description'][:90]}")
                print(f"    → {p['recommendation']}")

        # ── Curriculum results ──
        subsep("Auto-Curriculum Results")
        if curriculum_results:
            for cr in curriculum_results:
                status = "IMPROVED" if cr.get("improved") else "needs work"
                print(
                    f"  [{status}] {cr['domain']:15s}: "
                    f"quality={cr['quality']:.3f} target={cr['target_quality']:.2f} "
                    f"len={cr.get('content_length', 0)}"
                )
        else:
            print("  No curriculum tasks executed.")

        # ── FINAL RATING ──
        subsep("JOTTY RATING v2")

        r1_avg = sum(r["quality_score"] for r in r1) / len(r1) if r1 else 0
        r2_avg = sum(r["quality_score"] for r in r2) / len(r2) if r2 else 0
        improvement_delta = r2_avg - r1_avg

        r1_len = sum(r["content_length"] for r in r1) / len(r1) if r1 else 0
        r2_len = sum(r["content_length"] for r in r2) / len(r2) if r2 else 0
        depth_improvement = min(1.0, r2_len / max(r1_len, 1))

        r2_retrieval = sum(1 for r in r2 if r.get("retrieval_examples", 0) > 0)

        # R1→R2: maintaining quality on harder tasks + deeper responses = improvement
        quality_maintained = (
            1.0 if improvement_delta >= -0.05 else max(0, 0.5 + improvement_delta * 5)
        )
        r1r2_score = (
            quality_maintained * 0.5
            + depth_improvement * 0.3
            + (r2_retrieval / max(len(r2), 1)) * 0.2
        )

        # A/B delta: learning vs no-learning on same task
        ab_delta = 0.5
        if baseline:
            for br in baseline:
                lr = [r for r in self.results if r["task_id"] == f"R2_{br['domain']}"]
                if lr:
                    d = lr[0]["quality_score"] - br["quality_score"]
                    # Scoring: 0.5 = neutral (delta within noise band ±0.03)
                    # Positive delta → up to 1.0, negative → down to 0.0
                    if abs(d) <= 0.03:
                        ab_delta = 0.5
                    elif d > 0:
                        ab_delta = min(1.0, 0.5 + d * 5)
                    else:
                        ab_delta = max(0, 0.5 + d * 5)

        scores = {
            "Execution Success": successes_learning / total_learning if total_learning else 0,
            "Output Quality": avg_quality_learning,
            "Learning Recording": (
                1.0 if final_snap["episode_count"] > initial_snap["episode_count"] else 0
            ),
            "A/B: Learn vs No-Learn": ab_delta,
            "R1→R2 Quality+Depth": min(1.0, r1r2_score),
            "Thompson Active": (
                sum(1 for r in self.results if r["thompson_strategy"] != "default") / total
                if total
                else 0
            ),
            "LLM Judge Coverage": min(1.0, len(llm_judged) / max(total, 1)),
            "Retrieval Active": min(
                1.0,
                sum(1 for r in self.results if r.get("retrieval_examples", 0) > 0) / max(total, 1),
            ),
            "Pattern Richness": min(1.0, total_patterns / 15),
            "Causal Patterns": (
                min(1.0, len(causal) / 3) if causal else (0.5 if total_patterns >= 10 else 0)
            ),
            "Auto-Curriculum": (
                sum(
                    (
                        1.0
                        if cr.get("improved")
                        else max(0, cr.get("quality", 0) / max(cr.get("target_quality", 1), 0.01))
                    )
                    for cr in curriculum_results
                )
                / max(len(curriculum_results), 1)
                if curriculum_results
                else (1.0 if not weak_domains else 0)
            ),
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
            print(
                "  VERDICT: World-class — Genuine learning improvement with principled exploration."
            )
        elif overall >= 0.6:
            print("  VERDICT: Excellent — Strong execution with active learning.")
        elif overall >= 0.4:
            print("  VERDICT: Good — Learning works but improvement not yet proven.")
        else:
            print("  VERDICT: Needs work.")

    def _save_report(self, initial_snap, final_snap):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = RESULTS_DIR / f"{ts}_jotty_evaluation_v2.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "version": 2,
            "results": self.results,
            "learning_snapshots": self.learning_snapshots,
            "initial_episodes": initial_snap["episode_count"],
            "final_episodes": final_snap["episode_count"],
        }

        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\n  Report saved: {report_path}")


async def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    print(f"ANTHROPIC_API_KEY: ...{os.environ['ANTHROPIC_API_KEY'][-8:]}")

    evaluator = JottyEvaluator()
    await evaluator.run_all()


if __name__ == "__main__":
    asyncio.run(main())

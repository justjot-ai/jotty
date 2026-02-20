#!/usr/bin/env python3
"""
Jotty RETRY Learning Evaluation
=================================

The definitive test: run the SAME hard task TWICE.
1st attempt → likely FAIL (recorded as failure, quality < 0.5)
2nd attempt → learning injects corrective context from the failure
Compare: did the learning system help?

Also runs an A/B test on the second attempt:
- Baseline (learn=False): no learning context
- Learning (learn=True): full learning context from prior failure
"""

import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval_retry")
logger.setLevel(logging.INFO)

EVAL_MODEL = "claude-3-haiku-20240307"


def sep(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def subsep(title: str) -> None:
    print(f"\n  --- {title} ---\n")


# =============================================================================
# CHECKS
# =============================================================================

def check_regex(content: str, pattern: str, flags: int = re.IGNORECASE) -> bool:
    return bool(re.search(pattern, content, flags))


def check_code_blocks(content: str, min_blocks: int = 3, min_lines: int = 8) -> bool:
    blocks = re.findall(r'```[\w]*\n(.*?)```', content, re.DOTALL)
    substantial = [b for b in blocks if len(b.strip().split('\n')) >= min_lines]
    return len(substantial) >= min_blocks


def check_test_assertions(content: str, min_count: int = 5) -> bool:
    return len(re.findall(r'assert\s+\w+', content)) >= min_count


def check_math(content: str, min_count: int = 2) -> bool:
    formulas = re.findall(
        r'O\([^)]+\)|Θ\([^)]+\)|n\s*(?:log|lg)\s*n|2\^n|n\^[23k]|≤|≥|∀|∃|⟹|→',
        content,
    )
    return len(formulas) >= min_count


# =============================================================================
# TASKS — chosen to be hard enough for Haiku to fail, but verifiable
# =============================================================================

# Task A: Run this first as a "warm-up" that should pass
WARMUP_TASK = {
    "id": "W1_vector_clock",
    "goal": (
        "Implement vector clocks with causal broadcast.\n\n"
        "1. Define happens-before (→): same process order, send→receive, transitivity.\n"
        "2. Vector clock rules: local V[i][i]++, send attaches V, receive merges max(V,T) then V[i][i]++.\n"
        "3. Give 3-process example (P0,P1,P2) with 6+ events showing causal chain and concurrent events.\n"
        "4. Implement VectorClock class with increment(), merge(), happens_before(), concurrent().\n"
        "5. Implement CausalBroadcast with delivery condition: T(m)[i]==V[j][i]+1 AND T(m)[k]<=V[j][k] for k≠i.\n"
        "6. Write test_happens_before, test_concurrent, test_causal_delivery_order (buffer and reorder).\n"
    ),
    "checks": [
        ("Happens-before definition", lambda c: check_regex(c, r'happens.before.*(?:send|receive|transit)')),
        ("Vector clock update rules", lambda c: check_regex(c, r'V\[i\].*(?:max|merge|\+\+)')),
        ("3-process example", lambda c: check_regex(c, r'P[012].*(?:\[[\d,\s]+\]|\(\d+,\s*\d+)')),
        ("VectorClock class", lambda c: check_regex(c, r'class\s+VectorClock')),
        ("increment method", lambda c: check_regex(c, r'def\s+increment')),
        ("merge method", lambda c: check_regex(c, r'def\s+merge')),
        ("happens_before method", lambda c: check_regex(c, r'def\s+happens_before')),
        ("CausalBroadcast", lambda c: check_regex(c, r'(?:Causal|causal).*(?:Broadcast|broadcast|deliver)')),
        ("Delivery condition", lambda c: check_regex(c, r'(?:T\[.*\]|timestamp).*(?:==|<=).*(?:V\[|clock)')),
        ("Buffer/pending mechanism", lambda c: check_regex(c, r'(?:buffer|pending|queue).*(?:deliver|ready|wait)')),
        ("Code blocks", lambda c: check_code_blocks(c, 2, 6)),
        ("Test functions", lambda c: check_regex(c, r'def\s+test_\w+')),
        ("Assertions", lambda c: check_test_assertions(c, 3)),
    ],
    "domain": "system_design",
}

# Task B: The HARD task — Haiku will likely fail this
HARD_TASK = {
    "id": "PBFT",
    "goal": (
        "Implement Practical Byzantine Fault Tolerance (PBFT) with cryptographic verification.\n\n"
        "PART A — PROTOCOL (must be exact):\n"
        "1. Define Byzantine generals: up to f faulty of 3f+1 total. Prove 3f+1 minimum with "
        "the exact 3-node impossibility scenario.\n"
        "2. PBFT THREE phases:\n"
        "   - Pre-prepare: primary sends ⟨PRE-PREPARE, v, n, d⟩\n"
        "   - Prepare: replica sends ⟨PREPARE, v, n, d, i⟩ — need 2f matching\n"
        "   - Commit: replica sends ⟨COMMIT, v, n, d, i⟩ — need 2f+1 matching\n"
        "3. View changes: trigger, VIEW-CHANGE message, NEW-VIEW construction.\n"
        "4. Message complexity: O(n²) per request. Why is this a bottleneck? "
        "How does HotStuff reduce to O(n)?\n\n"
        "PART B — IMPLEMENTATION:\n"
        "- PBFTNode: state machine with handle_pre_prepare, handle_prepare, handle_commit\n"
        "- MessageLog: tracks messages per (view, sequence)\n"
        "- CryptoVerifier: sign_message(msg, key) + verify_signature using HMAC-SHA256\n"
        "- ViewChangeManager: detect timeout, broadcast VIEW-CHANGE, construct NEW-VIEW\n"
        "- CheckpointManager: periodic stable checkpoints, garbage collect old logs\n"
        "Handle: duplicate messages, out-of-order delivery, primary failure.\n\n"
        "PART C — TESTS:\n"
        "- test_normal_case: 4 nodes (f=1), verify all honest nodes reply same result\n"
        "- test_byzantine_primary: conflicting pre-prepares, detect and trigger view change\n"
        "- test_byzantine_replica: 1 faulty of 4, protocol still commits\n"
        "- test_view_change: kill primary, verify new primary and pending re-processed\n"
        "- test_message_counts: verify (n-1) prepares and n commits per request\n"
        "- test_checkpoint: 100 requests → checkpoint, old logs pruned"
    ),
    "checks": [
        ("3f+1 minimum proof", lambda c: check_regex(c, r'3\s*f\s*\+\s*1.*(?:minimum|require|need)', re.DOTALL)),
        ("3-node impossibility", lambda c: check_regex(c, r'(?:3\s*node|three\s*node|1\s*byzantine).*(?:conflict|split|impossible)', re.DOTALL)),
        ("PRE-PREPARE format", lambda c: check_regex(c, r'PRE.?PREPARE.*(?:view|sequence|digest)', re.DOTALL)),
        ("2f prepare threshold", lambda c: check_regex(c, r'2\s*f.*(?:prepare|matching|quorum)', re.DOTALL)),
        ("2f+1 commit threshold", lambda c: check_regex(c, r'2\s*f\s*\+\s*1.*(?:commit|matching)', re.DOTALL)),
        ("View change mechanism", lambda c: check_regex(c, r'(?:timeout|view.?change).*(?:trigger|detect|suspect)', re.DOTALL)),
        ("O(n²) complexity", lambda c: check_regex(c, r'O\(\s*n\s*[²2\^]\s*\)', re.DOTALL)),
        ("HotStuff comparison", lambda c: check_regex(c, r'HotStuff.*O\(\s*n\s*\)', re.DOTALL)),
        ("PBFTNode class", lambda c: check_regex(c, r'class\s+PBFTNode.*def\s+(?:handle|process|on)_', re.DOTALL)),
        ("MessageLog tracking", lambda c: check_regex(c, r'(?:view|sequence).*(?:log|dict|map|track)', re.DOTALL)),
        ("HMAC/crypto signing", lambda c: check_regex(c, r'(?:hmac|HMAC|sign|SHA|sha256|hashlib)', re.DOTALL)),
        ("ViewChange class/logic", lambda c: check_regex(c, r'(?:view.?change|ViewChange).*(?:class|def|broadcast)', re.DOTALL)),
        ("Checkpoint management", lambda c: check_regex(c, r'checkpoint.*(?:2f|proof|stable|garbage|prune)', re.DOTALL)),
        ("Duplicate handling", lambda c: check_regex(c, r'(?:duplicate|already|seen).*(?:message|request|ignore)', re.DOTALL)),
        ("Code blocks (3+)", lambda c: check_code_blocks(c, 3, 8)),
        ("test_normal_case", lambda c: check_regex(c, r'def\s+test_\w*normal', re.DOTALL)),
        ("test_byzantine", lambda c: check_regex(c, r'def\s+test_\w*byzantine', re.DOTALL)),
        ("test_view_change", lambda c: check_regex(c, r'def\s+test_\w*view', re.DOTALL)),
        ("5+ assertions", lambda c: check_test_assertions(c, 5)),
        ("Math reasoning", lambda c: check_math(c, 2)),
    ],
    "domain": "system_design",
}


# =============================================================================
# EVALUATOR
# =============================================================================

class RetryEval:
    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []
        self._orch = None
        self._learning = None

    def _get_orch(self) -> Any:
        if self._orch is None:
            from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator
            self._orch = Orchestrator()
        return self._orch

    def _get_learning(self) -> Any:
        if self._learning is None:
            from Jotty.core.intelligence.learning.learning_service import LearningService
            self._learning = LearningService.get_instance()
        return self._learning

    def _snapshot(self, domain: str = "system_design") -> Dict[str, Any]:
        ls = self._get_learning()
        g = ls.query(domain, "")
        ctx = ls.build_context_string(domain, "")
        ret = ls.build_retrieval_context(domain, "")
        return {
            "episodes": g.get("total_episodes", 0),
            "success_rate": g.get("success_rate", 0.0),
            "failures": len(g.get("failure_analysis", [])),
            "context_str": ctx,
            "context_len": len(ctx),
            "retrieval_len": len(ret),
        }

    def _run_checks(self, content: str, checks: List) -> Dict[str, bool]:
        out = {}
        for name, fn in checks:
            try:
                out[name] = fn(content)
            except Exception:
                out[name] = False
        return out

    async def run_task(
        self, task: Dict, learn: bool = True, tag: str = ""
    ) -> Dict[str, Any]:
        label = f"{task['id']}"
        if tag:
            label += f" ({tag})"
        if not learn:
            label += " [BASELINE]"
        subsep(label)

        orch = self._get_orch()
        snap_before = self._snapshot()

        print(f"  Model: {EVAL_MODEL}")
        print(f"  Episodes: {snap_before['episodes']} | Rate: {snap_before['success_rate']:.0%} | Failures: {snap_before['failures']}")
        print(f"  Context injected: {snap_before['context_len']} chars")
        if snap_before['context_str']:
            for line in snap_before['context_str'].split('\n')[:5]:
                print(f"    > {line[:120]}")
        print(f"  Retrieval: {snap_before['retrieval_len']} chars")

        start = time.time()
        content = ""
        error = None

        try:
            result = await orch.chat(
                message=task["goal"],
                provider="anthropic",
                model=EVAL_MODEL,
                learn=learn,
                enabled_tools=[],
                max_steps=1,
                temperature=0,
            )
            content = getattr(result, "content", str(result))
        except Exception as e:
            error = str(e)
            print(f"  ERROR: {error[:200]}")

        elapsed = time.time() - start

        # Wait for episode to be recorded
        await asyncio.sleep(3)
        snap_after = self._snapshot()

        check_results = self._run_checks(content, task["checks"])
        passed = sum(1 for v in check_results.values() if v)
        total_checks = len(check_results)
        check_ratio = passed / max(total_checks, 1)

        code_blocks = len(re.findall(r'```[\w]*\n.+?```', content, re.DOTALL))
        success = check_ratio >= 0.70 and len(content) >= 3000

        record = {
            "task_id": task["id"],
            "tag": tag,
            "learn": learn,
            "success": success,
            "checks_passed": passed,
            "checks_total": total_checks,
            "check_ratio": round(check_ratio, 3),
            "content_length": len(content),
            "code_blocks": code_blocks,
            "elapsed": round(elapsed, 1),
            "ep_before": snap_before["episodes"],
            "ep_after": snap_after["episodes"],
            "rate_before": snap_before["success_rate"],
            "rate_after": snap_after["success_rate"],
            "context_len": snap_before["context_len"],
            "retrieval_len": snap_before["retrieval_len"],
            "error": error,
        }
        self.results.append(record)

        status = "PASS" if success else "FAIL"
        print(f"\n  [{status}] Checks={passed}/{total_checks} ({check_ratio:.0%}) | "
              f"Len={len(content)} | Code={code_blocks} | {elapsed:.0f}s")
        print(f"  Episodes: {snap_before['episodes']}→{snap_after['episodes']} | "
              f"Rate: {snap_before['success_rate']:.0%}→{snap_after['success_rate']:.0%}")

        for name, ok in check_results.items():
            print(f"    [{'+'if ok else '-'}] {name}")

        return record

    async def run_all(self) -> None:
        sep("JOTTY RETRY LEARNING EVALUATION")
        print(f"  Date: {datetime.now().isoformat()}")
        print(f"  Model: {EVAL_MODEL}")
        print(f"  Strategy: Run hard task → fail → retry with learning context")
        snap = self._snapshot()
        print(f"  Initial state: {snap['episodes']} episodes, {snap['success_rate']:.0%} rate")

        # ─────────────────────────────────────────────────────
        # PHASE 1: Warm-up (easy task → should PASS → success episode)
        # ─────────────────────────────────────────────────────
        sep("PHASE 1: WARM-UP (should pass)")
        await self.run_task(WARMUP_TASK, learn=True, tag="warmup")
        print("\n  Waiting for learning to process...")
        await asyncio.sleep(15)

        # ─────────────────────────────────────────────────────
        # PHASE 2: Hard task FIRST attempt (likely FAIL)
        # ─────────────────────────────────────────────────────
        sep("PHASE 2: HARD TASK — 1st ATTEMPT (expecting failure)")
        r1 = await self.run_task(HARD_TASK, learn=True, tag="attempt_1")
        print("\n  Waiting for failure to be recorded and learned from...")
        await asyncio.sleep(15)

        # ─────────────────────────────────────────────────────
        # PHASE 3: Check learning state after failure
        # ─────────────────────────────────────────────────────
        sep("PHASE 3: LEARNING STATE CHECK")
        snap = self._snapshot()
        print(f"  Episodes: {snap['episodes']}")
        print(f"  Success rate: {snap['success_rate']:.0%}")
        print(f"  Failures recorded: {snap['failures']}")
        print(f"  Context length: {snap['context_len']} chars")
        if snap['context_str']:
            print(f"  Context content:")
            for line in snap['context_str'].split('\n'):
                print(f"    > {line[:120]}")
        else:
            print(f"  Context: EMPTY (gate closed)")
        print(f"  Retrieval length: {snap['retrieval_len']} chars")

        # ─────────────────────────────────────────────────────
        # PHASE 4: A/B TEST — same hard task
        # ─────────────────────────────────────────────────────
        sep("PHASE 4: A/B TEST — SAME HARD TASK")
        print("  Running BASELINE (no learning context)...")
        r_baseline = await self.run_task(
            {**HARD_TASK, "id": "PBFT_baseline"}, learn=False, tag="A/B baseline"
        )
        await asyncio.sleep(5)

        print("\n  Running WITH LEARNING (corrective context from failure)...")
        r_learning = await self.run_task(
            {**HARD_TASK, "id": "PBFT_learning"}, learn=True, tag="A/B learning"
        )
        await asyncio.sleep(15)

        # ─────────────────────────────────────────────────────
        # PHASE 5: SECOND RETRY (more episodes now)
        # ─────────────────────────────────────────────────────
        sep("PHASE 5: RETRY — 3rd attempt with accumulated learning")
        r3 = await self.run_task(HARD_TASK, learn=True, tag="attempt_3")
        await asyncio.sleep(10)

        # ─────────────────────────────────────────────────────
        # RESULTS
        # ─────────────────────────────────────────────────────
        sep("FINAL RESULTS")

        for r in self.results:
            s = "PASS" if r["success"] else "FAIL"
            ctx_tag = f" ctx={r['context_len']}" if r["context_len"] > 0 else ""
            learn_tag = "" if r["learn"] else " [BASELINE]"
            print(
                f"  [{s}] {r['task_id']:20s} ({r['tag']:15s}) "
                f"Chk={r['checks_passed']}/{r['checks_total']} ({r['check_ratio']:.0%}) "
                f"Len={r['content_length']:5d} "
                f"Ep={r['ep_before']}→{r['ep_after']} Rate={r['rate_before']:.0%}→{r['rate_after']:.0%}"
                f"{learn_tag}{ctx_tag}"
            )

        # Retry improvement
        subsep("RETRY IMPROVEMENT (same task across attempts)")
        pbft_attempts = [r for r in self.results if r["task_id"] == "PBFT" and r["learn"]]
        if len(pbft_attempts) >= 2:
            for i, r in enumerate(pbft_attempts):
                bar_len = int(r["check_ratio"] * 40)
                bar = "█" * bar_len + "░" * (40 - bar_len)
                print(f"  Attempt {i+1}: {bar} {r['check_ratio']:.0%} "
                      f"({r['checks_passed']}/{r['checks_total']}) ctx={r['context_len']}ch")
            first = pbft_attempts[0]["check_ratio"]
            last = pbft_attempts[-1]["check_ratio"]
            delta = last - first
            print(f"\n  Improvement: {first:.0%} → {last:.0%} ({delta:+.0%})")
            if delta > 0.05:
                print(f"  ✓ LEARNING HELPED — {delta:+.0%} improvement on retry")
            elif delta > -0.05:
                print(f"  ~ NEUTRAL — no significant change")
            else:
                print(f"  ✗ REGRESSION — learning hurt performance")

        # A/B comparison
        subsep("A/B TEST (same task, with vs without learning)")
        ab_base = next((r for r in self.results if r["task_id"] == "PBFT_baseline"), None)
        ab_learn = next((r for r in self.results if r["task_id"] == "PBFT_learning"), None)
        if ab_base and ab_learn:
            delta = ab_learn["check_ratio"] - ab_base["check_ratio"]
            print(f"  Baseline:  {ab_base['check_ratio']:.0%} ({ab_base['checks_passed']}/{ab_base['checks_total']}) ctx=0")
            print(f"  Learning:  {ab_learn['check_ratio']:.0%} ({ab_learn['checks_passed']}/{ab_learn['checks_total']}) ctx={ab_learn['context_len']}")
            print(f"  Delta: {delta:+.0%}")
            if delta > 0.05:
                print(f"  ✓ LEARNING WINS")
            elif delta > -0.05:
                print(f"  ~ NEUTRAL")
            else:
                print(f"  ✗ BASELINE WINS")

        # Save results
        results_dir = Path(__file__).parent / "eval_results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = results_dir / f"{ts}_retry_learning.json"
        with open(out_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "model": EVAL_MODEL,
                "results": self.results,
            }, f, indent=2, default=str)
        print(f"\n  Saved: {out_path}")


async def main() -> None:
    await RetryEval().run_all()

if __name__ == "__main__":
    asyncio.run(main())

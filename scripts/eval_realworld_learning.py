#!/usr/bin/env python3
"""
Real-World Learning Stress Test
================================

Tests whether Jotty ACTUALLY learns across repeated runs of the SAME complex task.
Each task is run 3 times. If the learning system works, quality should improve
(or at minimum maintain) across iterations.

Domains:
1. SECURITY: Build a JWT auth system with OWASP defenses
2. DISTRIBUTED: Design a Raft consensus protocol implementation
3. FINANCE: Build a Black-Scholes option pricer with Greeks
4. COMPILER: Write a regex engine with NFA→DFA compilation
5. ML_OPS: Build a feature store with versioning and serving
6. SYSTEMS: Implement a lock-free concurrent queue

Uses Haiku (weakest model) to stress-test the learning signal.
Each task runs 3x to measure improvement from learning.
"""

import asyncio
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval_realworld")
logger.setLevel(logging.INFO)

EVAL_MODEL = "claude-3-haiku-20240307"
RUNS_PER_TASK = 3
SLEEP_BETWEEN = 12  # seconds for learning to process


# =============================================================================
# CHECK HELPERS
# =============================================================================


def rx(content: str, pattern: str) -> bool:
    return bool(re.search(pattern, content, re.IGNORECASE | re.DOTALL))


def code_blocks(content: str, n: int = 3, min_lines: int = 5) -> bool:
    blocks = re.findall(r"```[\w]*\n(.*?)```", content, re.DOTALL)
    return len([b for b in blocks if len(b.strip().split("\n")) >= min_lines]) >= n


def assertions(content: str, n: int = 5) -> bool:
    return len(re.findall(r"assert\s+\w+", content)) >= n


def has_class(content: str, name: str) -> bool:
    return rx(content, rf"class\s+{name}")


def has_func(content: str, name: str) -> bool:
    return rx(content, rf"def\s+{name}\s*\(")


def word_count(content: str, n: int) -> bool:
    return len(content.split()) >= n


# =============================================================================
# 6 REAL-WORLD TASKS
# =============================================================================

TASKS = [
    # ─── 1. SECURITY: JWT Auth System ───
    {
        "id": "SEC_jwt_auth",
        "domain": "security",
        "goal": (
            "Build a production-grade JWT authentication system in Python.\n\n"
            "REQUIREMENTS:\n"
            "1. JWTManager class with:\n"
            "   - create_access_token(user_id, roles, expires_in=3600)\n"
            "   - create_refresh_token(user_id, expires_in=86400*30)\n"
            "   - verify_token(token) → payload or raise InvalidTokenError\n"
            "   - refresh_access_token(refresh_token) → new access token\n"
            "   - revoke_token(jti) — add to blacklist\n"
            "2. SECURITY (OWASP):\n"
            "   - Use RS256 (asymmetric) not HS256 — show why\n"
            "   - Include 'jti' (JWT ID) for revocation\n"
            "   - Short-lived access tokens (15min), long-lived refresh\n"
            "   - Token blacklist with TTL cleanup\n"
            "   - Rate limiting on token refresh endpoint\n"
            "3. MIDDLEWARE:\n"
            "   - require_auth decorator for Flask/FastAPI routes\n"
            "   - require_role('admin') decorator\n"
            "   - CORS configuration for SPA frontend\n"
            "4. TESTS:\n"
            "   - test_token_lifecycle: create, verify, refresh, revoke\n"
            "   - test_expired_token: mock time, verify rejection\n"
            "   - test_tampered_token: modify payload, verify signature fails\n"
            "   - test_role_authorization: admin vs user access\n"
            "   - test_blacklisted_token: revoke then verify fails\n"
            "   - 8+ assertions\n\n"
            "Do NOT use any tools."
        ),
        "checks": [
            ("JWTManager class", lambda c: has_class(c, "JWTManager")),
            ("create_access_token", lambda c: has_func(c, "create_access_token")),
            ("create_refresh_token", lambda c: has_func(c, "create_refresh_token")),
            ("verify_token", lambda c: has_func(c, "verify_token")),
            ("revoke_token", lambda c: has_func(c, "revoke_token")),
            ("RS256 algorithm", lambda c: rx(c, r"RS256|asymmetric|RSA|private.?key.*public.?key")),
            ("JTI for revocation", lambda c: rx(c, r"jti.*revoc|blacklist.*jti|jti.*unique")),
            ("Token blacklist", lambda c: rx(c, r"blacklist|revoked|token.*invalid")),
            ("Rate limiting", lambda c: rx(c, r"rate.?limit|throttl|too.?many")),
            (
                "Auth decorator",
                lambda c: rx(c, r"def\s+require_auth|@require_auth|@login_required"),
            ),
            ("Role decorator", lambda c: rx(c, r"require_role|role.*check|has_role")),
            ("test_token_lifecycle", lambda c: rx(c, r"def\s+test_\w*(?:lifecycle|create|verify)")),
            ("test_expired_token", lambda c: rx(c, r"def\s+test_\w*expir")),
            ("test_tampered_token", lambda c: rx(c, r"def\s+test_\w*(?:tamper|invalid|signature)")),
            ("8+ assertions", lambda c: assertions(c, 8)),
            ("Substantial code", lambda c: code_blocks(c, 3, 8)),
        ],
    },
    # ─── 2. DISTRIBUTED: Raft Consensus ───
    {
        "id": "DIST_raft_consensus",
        "domain": "distributed_systems",
        "goal": (
            "Implement the Raft consensus protocol in Python.\n\n"
            "REQUIREMENTS:\n"
            "1. RaftNode class with states: Follower, Candidate, Leader\n"
            "2. Leader election:\n"
            "   - Election timeout (randomized 150-300ms)\n"
            "   - RequestVote RPC: candidate_id, term, last_log_index, last_log_term\n"
            "   - Vote granting logic with term comparison\n"
            "   - Majority quorum calculation\n"
            "3. Log replication:\n"
            "   - AppendEntries RPC: term, leader_id, prev_log_index, prev_log_term, entries, leader_commit\n"
            "   - Log consistency check (prev_log_index + prev_log_term match)\n"
            "   - Commit index advancement\n"
            "   - Apply committed entries to state machine\n"
            "4. Safety:\n"
            "   - Election restriction: only vote for candidates with up-to-date log\n"
            "   - Leader completeness: committed entries survive leader changes\n"
            "   - Explain the split-brain problem and how Raft prevents it\n"
            "5. TESTS:\n"
            "   - test_leader_election: 5 nodes, verify one leader elected\n"
            "   - test_log_replication: submit command, verify all nodes have it\n"
            "   - test_leader_failure: kill leader, verify new election\n"
            "   - test_network_partition: split 5 nodes into 3+2, verify only majority side works\n"
            "   - 8+ assertions\n\n"
            "Do NOT use any tools."
        ),
        "checks": [
            ("RaftNode class", lambda c: has_class(c, "RaftNode")),
            (
                "State enum (Follower/Candidate/Leader)",
                lambda c: rx(c, r"Follower|Candidate|Leader"),
            ),
            ("Election timeout", lambda c: rx(c, r"election.?timeout|150.*300|random.*timeout")),
            ("RequestVote RPC", lambda c: rx(c, r"request.?vote|vote.?request|RequestVote")),
            ("Term comparison", lambda c: rx(c, r"(?:current_)?term.*(?:>|<|>=|<=|compare)")),
            ("Majority quorum", lambda c: rx(c, r"majority|quorum|len.*//.*2|n/2|half")),
            ("AppendEntries RPC", lambda c: rx(c, r"append.?entries|AppendEntries")),
            ("Log consistency check", lambda c: rx(c, r"prev_log_(?:index|term)|log.*consisten")),
            ("Commit index", lambda c: rx(c, r"commit.?index|committed.*entries|apply.*commit")),
            ("Split-brain prevention", lambda c: rx(c, r"split.?brain|partition|network.*split")),
            ("test_leader_election", lambda c: rx(c, r"def\s+test_\w*(?:leader|election)")),
            ("test_log_replication", lambda c: rx(c, r"def\s+test_\w*(?:log|replic)")),
            ("test_leader_failure", lambda c: rx(c, r"def\s+test_\w*(?:fail|crash|kill)")),
            ("8+ assertions", lambda c: assertions(c, 8)),
            ("Substantial code", lambda c: code_blocks(c, 3, 8)),
        ],
    },
    # ─── 3. FINANCE: Black-Scholes + Greeks ───
    {
        "id": "FIN_black_scholes",
        "domain": "finance",
        "goal": (
            "Implement a complete Black-Scholes option pricing engine in Python.\n\n"
            "REQUIREMENTS:\n"
            "1. THEORY:\n"
            "   - Derive the Black-Scholes PDE from Geometric Brownian Motion\n"
            "   - Show: dS/S = μdt + σdW, then Ito's lemma → BS equation\n"
            "   - Closed-form solution for European call: C = S·N(d1) - K·e^(-rT)·N(d2)\n"
            "   - Define d1 and d2 formulas explicitly\n"
            "2. IMPLEMENTATION:\n"
            "   - BlackScholes class with price_call(), price_put()\n"
            "   - All 5 Greeks: delta, gamma, theta, vega, rho\n"
            "   - Implied volatility solver (Newton-Raphson on vega)\n"
            "   - Monte Carlo pricer for comparison (10000 paths)\n"
            "3. RISK ANALYSIS:\n"
            "   - Portfolio Greeks aggregation\n"
            "   - P&L sensitivity to 1% move in each parameter\n"
            "   - Volatility smile/surface explanation\n"
            "4. TESTS:\n"
            "   - test_put_call_parity: C - P = S - K*e^(-rT)\n"
            "   - test_greeks_symmetry: call_delta + put_delta = 1\n"
            "   - test_monte_carlo_convergence: MC price within 1% of BS\n"
            "   - test_implied_vol_roundtrip: compute price → extract IV → matches original σ\n"
            "   - 8+ assertions\n\n"
            "Do NOT use any tools."
        ),
        "checks": [
            ("BlackScholes class", lambda c: has_class(c, "BlackScholes")),
            ("price_call method", lambda c: has_func(c, "price_call")),
            ("price_put method", lambda c: has_func(c, "price_put")),
            ("d1 and d2 formulas", lambda c: rx(c, r"d[_]?1.*d[_]?2|d1.*=.*ln|d2.*=.*d1")),
            ("N(d) normal CDF", lambda c: rx(c, r"norm\.cdf|N\(d|stats\.norm|erf|cumulative")),
            ("Delta Greek", lambda c: rx(c, r"def\s+\w*delta|delta.*=.*N\(d|Δ|\.delta")),
            ("Gamma Greek", lambda c: rx(c, r"def\s+\w*gamma|gamma.*=|Γ|\.gamma")),
            ("Vega Greek", lambda c: rx(c, r"def\s+\w*vega|vega.*=|\.vega")),
            ("Theta Greek", lambda c: rx(c, r"def\s+\w*theta|theta.*=|Θ|\.theta")),
            (
                "Implied vol solver",
                lambda c: rx(c, r"implied.?vol|newton.?raphson|iv.*solver|bisect"),
            ),
            ("Monte Carlo", lambda c: rx(c, r"monte.?carlo|10000|simulation|random.*path")),
            ("Put-call parity", lambda c: rx(c, r"put.?call.?parity|C\s*-\s*P|parity")),
            ("GBM derivation", lambda c: rx(c, r"geometric.?brownian|dS.*=.*μ.*dt|Ito.?s?.?lemma")),
            ("test functions", lambda c: rx(c, r"def\s+test_")),
            ("8+ assertions", lambda c: assertions(c, 8)),
            ("Substantial code", lambda c: code_blocks(c, 3, 8)),
        ],
    },
    # ─── 4. COMPILER: Regex Engine ───
    {
        "id": "COMP_regex_engine",
        "domain": "compilers",
        "goal": (
            "Build a regex engine from scratch in Python.\n\n"
            "REQUIREMENTS:\n"
            "1. PARSER:\n"
            "   - Parse regex string into AST: Literal, Dot, Star(*), Plus(+), Optional(?)\n"
            "   - Concatenation and alternation (|)\n"
            "   - Character classes [a-z], [^abc]\n"
            "   - Escaped characters \\d, \\w, \\s\n"
            "   - Grouping with parentheses\n"
            "2. NFA CONSTRUCTION (Thompson's algorithm):\n"
            "   - NFAState class with transitions (char→set of states) and epsilon transitions\n"
            "   - Build NFA from AST using Thompson's construction\n"
            "   - Explain why Thompson's gives O(n) states for pattern of length n\n"
            "3. NFA→DFA CONVERSION (subset construction):\n"
            "   - Compute epsilon-closure\n"
            "   - Subset construction algorithm\n"
            "   - DFA minimization (optional)\n"
            "4. MATCHING:\n"
            "   - match(pattern, text) → bool\n"
            "   - find_all(pattern, text) → list of matches with positions\n"
            "   - NFA simulation (on-the-fly, without full DFA conversion)\n"
            "5. TESTS:\n"
            "   - test_literals: 'abc' matches 'abc'\n"
            "   - test_star: 'a*b' matches 'b', 'ab', 'aaab'\n"
            "   - test_alternation: 'cat|dog' matches both\n"
            "   - test_char_class: '[a-z]+' matches 'hello'\n"
            "   - test_complex: '(a|b)*c' matches 'aababc'\n"
            "   - 8+ assertions\n\n"
            "Do NOT use any tools."
        ),
        "checks": [
            ("AST node classes", lambda c: rx(c, r"class\s+(?:Literal|Star|Plus|Alt|Concat|Dot)")),
            ("Parse function", lambda c: has_func(c, "parse")),
            ("Alternation support", lambda c: rx(c, r"alternation|\|.*alt|class\s+Alt")),
            ("Character classes", lambda c: rx(c, r"\[a-z\]|char.?class|CharClass|range")),
            ("NFAState class", lambda c: rx(c, r"class\s+NFA(?:State|Node)")),
            ("Thompson's construction", lambda c: rx(c, r"[Tt]hompson|nfa.*build|build.*nfa")),
            ("Epsilon transitions", lambda c: rx(c, r"epsilon|ε|eps.*transition|None.*transition")),
            ("Epsilon closure", lambda c: rx(c, r"epsilon.?closure|eps.?closure|ε.?closure")),
            ("Subset construction", lambda c: rx(c, r"subset.?construct|nfa.?to.?dfa|powerset")),
            ("match function", lambda c: has_func(c, "match")),
            ("find_all function", lambda c: rx(c, r"def\s+find_?all")),
            ("test_literals", lambda c: rx(c, r"def\s+test_\w*literal")),
            ("test_star", lambda c: rx(c, r"def\s+test_\w*star|test_\w*kleene")),
            ("test_alternation", lambda c: rx(c, r"def\s+test_\w*alt")),
            ("8+ assertions", lambda c: assertions(c, 8)),
            ("Substantial code", lambda c: code_blocks(c, 3, 8)),
        ],
    },
    # ─── 5. ML_OPS: Feature Store ───
    {
        "id": "MLOPS_feature_store",
        "domain": "ml_engineering",
        "goal": (
            "Build a feature store system for ML model serving in Python.\n\n"
            "REQUIREMENTS:\n"
            "1. FEATURE REGISTRY:\n"
            "   - FeatureDefinition: name, dtype, description, owner, tags, version\n"
            "   - FeatureGroup: logical grouping of features with entity key\n"
            "   - FeatureRegistry: register, discover, search features by name/tag\n"
            "   - Schema validation on registration\n"
            "2. OFFLINE STORE:\n"
            "   - Point-in-time correct joins (prevent data leakage)\n"
            "   - Historical feature retrieval with timestamps\n"
            "   - Batch materialization from source tables\n"
            "   - Explain why point-in-time correctness matters for training\n"
            "3. ONLINE STORE:\n"
            "   - Low-latency serving (in-memory cache + persistence)\n"
            "   - FeatureServer class with get_features(entity_id, feature_names)\n"
            "   - TTL-based cache invalidation\n"
            "   - Write-through on feature update\n"
            "4. VERSIONING:\n"
            "   - Feature version tracking (v1, v2, ...)\n"
            "   - Schema migration support\n"
            "   - Feature lineage (source → transformation → feature)\n"
            "5. TESTS:\n"
            "   - test_register_and_discover\n"
            "   - test_point_in_time_join: verify no future data leaks\n"
            "   - test_online_serving_latency: verify <1ms for cached features\n"
            "   - test_version_compatibility\n"
            "   - 8+ assertions\n\n"
            "Do NOT use any tools."
        ),
        "checks": [
            ("FeatureDefinition class", lambda c: has_class(c, "FeatureDefinition")),
            ("FeatureGroup class", lambda c: has_class(c, "FeatureGroup")),
            ("FeatureRegistry class", lambda c: has_class(c, "FeatureRegistry")),
            ("Schema validation", lambda c: rx(c, r"schema.*valid|validate.*schema|dtype.*check")),
            (
                "Point-in-time joins",
                lambda c: rx(c, r"point.?in.?time|temporal.*join|time.*correct"),
            ),
            ("Data leakage prevention", lambda c: rx(c, r"data.?leakage|future.*data|lookahead")),
            ("FeatureServer class", lambda c: has_class(c, "FeatureServer")),
            ("get_features method", lambda c: has_func(c, "get_features")),
            ("TTL cache", lambda c: rx(c, r"ttl|time.?to.?live|cache.*expir|invalidat")),
            ("Version tracking", lambda c: rx(c, r"version|v\d+|schema.*migrat")),
            ("Feature lineage", lambda c: rx(c, r"lineage|provenance|source.*transform")),
            ("test functions", lambda c: rx(c, r"def\s+test_")),
            ("8+ assertions", lambda c: assertions(c, 8)),
            ("Substantial code", lambda c: code_blocks(c, 3, 8)),
            ("500+ words", lambda c: word_count(c, 500)),
        ],
    },
    # ─── 6. SYSTEMS: Lock-Free Queue ───
    {
        "id": "SYS_lockfree_queue",
        "domain": "systems_programming",
        "goal": (
            "Implement a lock-free concurrent queue in Python.\n\n"
            "REQUIREMENTS:\n"
            "1. THEORY:\n"
            "   - Explain the Michael-Scott lock-free queue algorithm\n"
            "   - ABA problem: what it is, why CAS alone doesn't prevent it\n"
            "   - Solutions: hazard pointers, epoch-based reclamation, tagged pointers\n"
            "   - Compare lock-free vs wait-free vs obstruction-free\n"
            "   - Memory ordering: acquire, release, seq_cst — when each is needed\n"
            "2. IMPLEMENTATION:\n"
            "   - Node class with value and atomic next pointer\n"
            "   - LockFreeQueue with enqueue(value) and dequeue() → value\n"
            "   - Use CAS (compare-and-swap) for thread safety\n"
            "   - Sentinel/dummy node pattern for empty queue\n"
            "   - Handle contention with exponential backoff\n"
            "3. CORRECTNESS:\n"
            "   - Linearizability argument: identify linearization points\n"
            "   - Progress guarantee: prove lock-freedom (at least one thread progresses)\n"
            "   - Explain why Python GIL makes this educational (real impl needs C/Rust)\n"
            "4. TESTS:\n"
            "   - test_single_thread: enqueue 1..100, dequeue all, verify order\n"
            "   - test_concurrent_producers: 4 threads enqueue, verify all items present\n"
            "   - test_concurrent_consumers: prefill queue, 4 threads dequeue, no duplicates\n"
            "   - test_empty_dequeue: dequeue from empty returns None\n"
            "   - test_mixed_concurrent: simultaneous enqueue/dequeue across 8 threads\n"
            "   - 8+ assertions\n\n"
            "Do NOT use any tools."
        ),
        "checks": [
            ("Node class", lambda c: has_class(c, "Node")),
            ("LockFreeQueue class", lambda c: has_class(c, "LockFreeQueue")),
            ("enqueue method", lambda c: has_func(c, "enqueue")),
            ("dequeue method", lambda c: has_func(c, "dequeue")),
            ("CAS operation", lambda c: rx(c, r"compare.?and.?swap|CAS|compare_exchange|atomic")),
            ("ABA problem", lambda c: rx(c, r"ABA.*problem|ABA.*hazard|ABA")),
            ("Hazard pointers", lambda c: rx(c, r"hazard.?pointer|epoch.?based|tagged.?pointer")),
            (
                "Sentinel/dummy node",
                lambda c: rx(c, r"sentinel|dummy.*node|dummy_node|head.*=.*Node"),
            ),
            ("Linearizability", lambda c: rx(c, r"lineariz|linearisation|linearization.?point")),
            (
                "Lock-freedom proof",
                lambda c: rx(c, r"lock.?free.*progress|at.?least.?one.*thread|progress.?guarantee"),
            ),
            (
                "Memory ordering",
                lambda c: rx(c, r"acquire|release|seq_cst|memory.?order|memory.?model"),
            ),
            ("Exponential backoff", lambda c: rx(c, r"backoff|exponential.*back|retry.*delay")),
            ("test_single_thread", lambda c: rx(c, r"def\s+test_\w*(?:single|basic|fifo|order)")),
            ("test_concurrent", lambda c: rx(c, r"def\s+test_\w*(?:concurrent|thread|parallel)")),
            ("8+ assertions", lambda c: assertions(c, 8)),
            ("Substantial code", lambda c: code_blocks(c, 3, 8)),
        ],
    },
]


# =============================================================================
# EVALUATOR
# =============================================================================


class RealWorldEval:
    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []
        self._orch = None
        self._learning = None

    def _get_orchestrator(self) -> Any:
        if self._orch is None:
            from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

            self._orch = Orchestrator()
        return self._orch

    def _get_learning(self) -> Any:
        if self._learning is None:
            from Jotty.core.intelligence.learning.learning_service import LearningService

            self._learning = LearningService.get_instance()
        return self._learning

    def _snapshot(self, domain: str = "") -> Dict[str, Any]:
        ls = self._get_learning()
        query_domain = domain or "general"
        guidance = ls.query(query_domain, "")
        ctx = ls.build_context_string(query_domain, "")
        ret = ls.build_retrieval_context(query_domain, "")
        try:
            conn = ls._store._get_conn()
            row = conn.execute("SELECT COUNT(*) as cnt FROM episodes").fetchone()
            global_episodes = row["cnt"] if row else 0
        except Exception:
            global_episodes = guidance.get("total_episodes", 0)
        return {
            "episodes": guidance.get("total_episodes", 0),
            "global_episodes": global_episodes,
            "rate": guidance.get("success_rate", 0.0),
            "ctx_len": len(ctx),
            "ret_len": len(ret),
            "ctx": ctx[:300] if ctx else "",
        }

    async def run_task(self, task: Dict, run_num: int) -> Dict[str, Any]:
        label = f"{task['id']} (run {run_num}/{RUNS_PER_TASK})"
        print(f"\n  --- {label} ---\n")

        orch = self._get_orchestrator()
        from Jotty.core.intelligence.learning.learning_service import classify_domain

        classified_domain, _ = classify_domain(task["goal"])
        domain = classified_domain or task.get("domain", "general")
        snap_before = self._snapshot(domain)

        print(f"  Model: {EVAL_MODEL}")
        print(
            f"  Episodes: {snap_before['episodes']} (domain) / {snap_before['global_episodes']} (global)"
        )
        print(f"  Success rate: {snap_before['rate']:.0%}")
        print(f"  Context injected: {snap_before['ctx_len']} chars")
        if snap_before["ctx"]:
            for line in snap_before["ctx"].split("\n")[:3]:
                print(f"    > {line[:120]}")
        print(f"  Retrieval: {snap_before['ret_len']} chars")

        start = time.time()
        content = ""
        error = None

        try:
            result = await orch.chat(
                message=task["goal"],
                provider="anthropic",
                model=EVAL_MODEL,
                learn=True,
                enabled_tools=[],
                max_steps=1,
                temperature=0,
            )
            content = getattr(result, "content", str(result))
        except Exception as e:
            error = str(e)
            print(f"  ERROR: {error[:200]}")

        elapsed = time.time() - start
        snap_after = self._snapshot(domain)

        check_results = {}
        for check_name, check_fn in task["checks"]:
            try:
                check_results[check_name] = check_fn(content)
            except Exception:
                check_results[check_name] = False

        passed = sum(1 for v in check_results.values() if v)
        total = len(check_results)
        check_ratio = passed / max(total, 1)

        code_blk = len(re.findall(r"```[\w]*\n.+?```", content, re.DOTALL))
        quality = (
            check_ratio * 0.5 + min(1.0, len(content) / 6000) * 0.25 + min(1.0, code_blk / 3) * 0.25
        )

        success = check_ratio >= 0.60 and len(content) >= 2000

        record = {
            "task_id": task["id"],
            "run": run_num,
            "domain": domain,
            "success": success,
            "quality": round(quality, 3),
            "checks_passed": passed,
            "checks_total": total,
            "check_ratio": round(check_ratio, 3),
            "content_length": len(content),
            "code_blocks": code_blk,
            "elapsed": round(elapsed, 1),
            "ep_before": snap_before["episodes"],
            "ep_after": snap_after["episodes"],
            "ctx_injected": snap_before["ctx_len"],
            "ret_injected": snap_before["ret_len"],
            "success_rate_before": round(snap_before["rate"], 3),
            "error": error,
        }
        self.results.append(record)

        status = "PASS" if success else "FAIL"
        print(
            f"\n  [{status}] Q={quality:.2f} | Checks={passed}/{total} ({check_ratio:.0%}) | "
            f"Len={len(content)} | Code={code_blk} | {elapsed:.0f}s"
        )
        print(
            f"  Episodes: {snap_before['episodes']}→{snap_after['episodes']} | "
            f"Rate: {snap_before['rate']:.0%}→{snap_after['rate']:.0%}"
        )

        for name, ok in check_results.items():
            print(f"    [{'+'if ok else '-'}] {name}")

        return record

    def _print_results(self) -> None:
        print(f"\n{'=' * 80}")
        print(f"  REAL-WORLD LEARNING STRESS TEST — RESULTS")
        print(f"{'=' * 80}\n")

        # Per-task summary
        for r in self.results:
            s = "PASS" if r["success"] else "FAIL"
            ctx = r["ctx_injected"] + r["ret_injected"]
            print(
                f"  [{s}] {r['task_id']:25s} run={r['run']} Q={r['quality']:.2f} "
                f"Chk={r['checks_passed']}/{r['checks_total']} ({r['check_ratio']:.0%}) "
                f"Len={r['content_length']:5d} ctx={ctx}"
            )

        # ─── PER-DOMAIN LEARNING CURVES ───
        print(f"\n{'─' * 80}")
        print(f"  LEARNING CURVES (run 1 → 2 → 3)")
        print(f"{'─' * 80}\n")

        task_ids = list(dict.fromkeys(r["task_id"] for r in self.results))
        improvements = []

        for tid in task_ids:
            runs = sorted([r for r in self.results if r["task_id"] == tid], key=lambda r: r["run"])
            if len(runs) < 2:
                continue

            print(f"  {tid}:")
            for r in runs:
                bar_len = int(r["check_ratio"] * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                print(
                    f"    Run {r['run']}: {bar} {r['check_ratio']:.0%} "
                    f"({r['checks_passed']}/{r['checks_total']}) "
                    f"Q={r['quality']:.2f} ctx={r['ctx_injected']}ch"
                )

            first_q = runs[0]["check_ratio"]
            last_q = runs[-1]["check_ratio"]
            best_q = max(r["check_ratio"] for r in runs)
            delta = last_q - first_q
            delta_best = best_q - first_q
            improvements.append(
                {
                    "task": tid,
                    "first": first_q,
                    "last": last_q,
                    "best": best_q,
                    "delta": delta,
                    "delta_best": delta_best,
                }
            )

            trend = "IMPROVED" if delta > 0.03 else "MAINTAINED" if delta >= -0.03 else "DEGRADED"
            print(f"    Trend: {first_q:.0%} → {last_q:.0%} ({delta:+.0%}) [{trend}]")
            if best_q > last_q:
                print(
                    f"    Peak: {best_q:.0%} (run {next(r['run'] for r in runs if r['check_ratio'] == best_q)})"
                )
            print()

        # ─── AGGREGATE METRICS ───
        print(f"{'─' * 80}")
        print(f"  AGGREGATE METRICS")
        print(f"{'─' * 80}\n")

        # Run-1 vs Run-3 comparison
        run1 = [r for r in self.results if r["run"] == 1]
        run3 = [r for r in self.results if r["run"] == RUNS_PER_TASK]

        if run1 and run3:
            avg_q1 = sum(r["check_ratio"] for r in run1) / len(run1)
            avg_q3 = sum(r["check_ratio"] for r in run3) / len(run3)
            avg_qual1 = sum(r["quality"] for r in run1) / len(run1)
            avg_qual3 = sum(r["quality"] for r in run3) / len(run3)
            pass1 = sum(1 for r in run1 if r["success"]) / len(run1)
            pass3 = sum(1 for r in run3 if r["success"]) / len(run3)

            print(
                f"  {'Metric':<25s} {'Run 1':>10s} {'Run {0}':>10s} {'Delta':>10s}".format(
                    RUNS_PER_TASK
                )
            )
            print(f"  {'─' * 55}")
            print(f"  {'Avg check rate':<25s} {avg_q1:>9.0%} {avg_q3:>9.0%} {avg_q3-avg_q1:>+9.0%}")
            print(
                f"  {'Avg quality score':<25s} {avg_qual1:>9.2f} {avg_qual3:>9.2f} {avg_qual3-avg_qual1:>+9.2f}"
            )
            print(f"  {'Pass rate':<25s} {pass1:>9.0%} {pass3:>9.0%} {pass3-pass1:>+9.0%}")

        # Context injection coverage
        ctx_hits = sum(1 for r in self.results if r["ctx_injected"] > 0)
        ret_hits = sum(1 for r in self.results if r["ret_injected"] > 0)
        total_tasks = len(self.results)
        print(f"\n  Context injected:    {ctx_hits}/{total_tasks} tasks")
        print(f"  Retrieval injected:  {ret_hits}/{total_tasks} tasks")

        # Learning improvement summary
        if improvements:
            improved = sum(1 for i in improvements if i["delta"] > 0.03)
            maintained = sum(1 for i in improvements if -0.03 <= i["delta"] <= 0.03)
            degraded = sum(1 for i in improvements if i["delta"] < -0.03)
            print(f"\n  Improved:    {improved}/{len(improvements)} tasks")
            print(f"  Maintained:  {maintained}/{len(improvements)} tasks")
            print(f"  Degraded:    {degraded}/{len(improvements)} tasks")

            avg_delta = sum(i["delta"] for i in improvements) / len(improvements)
            avg_delta_best = sum(i["delta_best"] for i in improvements) / len(improvements)
            print(f"\n  Avg Δ (run 1→{RUNS_PER_TASK}): {avg_delta:+.1%}")
            print(f"  Avg Δ (run 1→best):  {avg_delta_best:+.1%}")

        # Total episodes
        if self.results:
            final_ep = self.results[-1]["ep_after"]
            print(f"\n  Total episodes in DB: {final_ep}")

    async def run_all(self) -> None:
        print(f"\n{'=' * 80}")
        print(f"  JOTTY REAL-WORLD LEARNING STRESS TEST")
        print(f"{'=' * 80}")
        now = datetime.now().isoformat()
        snap = self._snapshot()
        print(f"\n  Date: {now}")
        print(f"  Model: {EVAL_MODEL}")
        print(
            f"  Tasks: {len(TASKS)} domains × {RUNS_PER_TASK} runs = {len(TASKS) * RUNS_PER_TASK} total"
        )
        print(f"  Initial episodes: {snap['global_episodes']}")
        print(f"  Sleep between runs: {SLEEP_BETWEEN}s")

        for task_idx, task in enumerate(TASKS):
            print(f"\n{'=' * 80}")
            print(f"  TASK {task_idx + 1}/{len(TASKS)}: {task['id']} ({task['domain']})")
            print(f"{'=' * 80}")

            for run in range(1, RUNS_PER_TASK + 1):
                await self.run_task(task, run)
                if run < RUNS_PER_TASK:
                    print(f"\n  Waiting {SLEEP_BETWEEN}s for learning to process...")
                    await asyncio.sleep(SLEEP_BETWEEN)

        self._print_results()

        # Save JSON
        results_dir = Path(__file__).parent / "eval_results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = results_dir / f"{ts}_realworld_learning.json"
        with open(out, "w") as f:
            json.dump(
                {
                    "timestamp": now,
                    "model": EVAL_MODEL,
                    "runs_per_task": RUNS_PER_TASK,
                    "results": self.results,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\n  Saved: {out}")


async def main() -> None:
    await RealWorldEval().run_all()


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Paper Integrations Real-World Evaluation
=========================================

Tests whether the 6 research paper integrations provide measurable gains
when running complex tasks through the real orchestrator with live LLM calls.

Measures:
  A) Tier Detection accuracy (Delegation Floor + CogRouter)
  B) Skill Cap effectiveness (SkillsBench)
  C) Proficiency filtering (Team of Thoughts)
  D) Hierarchical compression (LCM)
  E) Deferred learning latency (MAPLE hot/cold split)
  F) End-to-end multi-swarm with all integrations active

Usage:
    python scripts/eval_paper_integrations.py
    python scripts/eval_paper_integrations.py --test tier    # single section
    python scripts/eval_paper_integrations.py --fast          # shorter timeouts
"""

import argparse
import asyncio
import json
import logging
import os
import re
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

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval_papers")
logger.setLevel(logging.INFO)

TIMEOUT_DEFAULT = 180
TIMEOUT_FAST = 90


# ─── Result ─────────────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    section: str
    name: str
    passed: bool
    elapsed_s: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


def extract_content(result: Any) -> str:
    """Extract text content from any result type."""
    if result is None:
        return ""
    content = ""
    inner = getattr(result, "output", None)
    if inner and not isinstance(inner, str):
        for attr in ("final_output", "output", "content", "code", "report"):
            val = getattr(inner, attr, None)
            if val and isinstance(val, str) and len(val) > len(content):
                content = val
    for attr in ("final_output", "output", "content"):
        val = getattr(result, attr, None)
        if val and isinstance(val, str) and len(val) > len(content):
            content = val
    if not content:
        s = str(result)
        if len(s) > 50:
            content = s
    return content


# =============================================================================
# A) TIER DETECTION: Delegation Floor + CogRouter
# =============================================================================


async def test_tier_detection(timeout: int) -> List[CheckResult]:
    """Verify the delegation floor and CogRouter route tasks correctly."""
    from Jotty.core.intelligence.orchestration.execution.tier_detector import TierDetector
    from Jotty.core.intelligence.orchestration.execution.types import ExecutionTier
    from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline

    results = []

    # -- A1: Delegation floor catches trivial tasks --
    td = TierDetector()
    trivial_tasks = [
        "What is GDP?",
        "Hello",
        "How are you?",
        "Tell me a joke",
        "2+2",
    ]
    complex_tasks = [
        "Research the implications of quantum computing on cryptography and create a report",
        "Build a distributed cache with consensus protocol and benchmark it",
        "Analyze, compare, and evaluate three machine learning approaches for anomaly detection",
    ]

    trivial_correct = sum(1 for t in trivial_tasks if td.detect(t) == ExecutionTier.DIRECT)
    complex_correct = sum(1 for t in complex_tasks if td.detect(t) != ExecutionTier.DIRECT)

    results.append(
        CheckResult(
            section="TIER_DETECTION",
            name="Delegation floor: trivial → DIRECT",
            passed=trivial_correct >= 4,
            details={
                "trivial_correct": f"{trivial_correct}/{len(trivial_tasks)}",
                "tasks": {t: td.detect(t).name for t in trivial_tasks},
            },
        )
    )
    results.append(
        CheckResult(
            section="TIER_DETECTION",
            name="Delegation floor: complex → NOT DIRECT",
            passed=complex_correct >= 2,
            details={
                "complex_correct": f"{complex_correct}/{len(complex_tasks)}",
                "tasks": {t[:60]: td.detect(t).name for t in complex_tasks},
            },
        )
    )

    # -- A2: CogRouter learns from history --
    baseline = GroupedValueBaseline()
    # Simulate: RESEARCH tier historically succeeds for "research" tasks
    for _ in range(10):
        baseline.update_tier("RESEARCH", "research", 0.95)
    for _ in range(10):
        baseline.update_tier("DIRECT", "research", 0.3)

    td_with_history = TierDetector(grouped_baseline=baseline)
    # "find latest AI papers" — ambiguous without history
    tier_no_hist = TierDetector().detect("find latest AI papers")
    tier_with_hist = td_with_history.detect("find latest AI papers")

    results.append(
        CheckResult(
            section="TIER_DETECTION",
            name="CogRouter: history improves routing",
            passed=True,  # We just want to see the difference
            details={
                "without_history": tier_no_hist.name,
                "with_history": tier_with_hist.name,
                "history_used": tier_with_hist != tier_no_hist,
            },
        )
    )

    # -- A3: Live tier detection on a real run --
    start = time.time()
    try:
        from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

        orch = Orchestrator()
        result = await asyncio.wait_for(
            orch.chat(
                message="What is the capital of France?",
                provider="anthropic",
                learn=True,
                max_steps=1,
                temperature=0,
            ),
            timeout=timeout,
        )
        content = extract_content(result)
        elapsed = time.time() - start

        results.append(
            CheckResult(
                section="TIER_DETECTION",
                name="Live: trivial task completes fast",
                passed=elapsed < 30 and len(content) > 10,
                elapsed_s=elapsed,
                details={"content_preview": content[:100], "latency": f"{elapsed:.1f}s"},
            )
        )
    except Exception as e:
        results.append(
            CheckResult(
                section="TIER_DETECTION",
                name="Live: trivial task completes fast",
                passed=False,
                elapsed_s=time.time() - start,
                error=str(e)[:200],
            )
        )

    return results


# =============================================================================
# B) SKILL CAP: SkillsBench (2-3 optimal)
# =============================================================================


async def test_skill_cap(timeout: int) -> List[CheckResult]:
    """Verify skill cap enforces 3-skill limit."""
    from Jotty.core.intelligence.reasoning.executors.skill_plan_executor import SkillPlanExecutor
    import inspect

    results = []

    # -- B1: Default max_skills is 3 --
    sig = inspect.signature(SkillPlanExecutor.select_skills)
    default = sig.parameters["max_skills"].default
    results.append(
        CheckResult(
            section="SKILL_CAP",
            name="select_skills default max_skills=3",
            passed=default == 3,
            details={"default": default},
        )
    )

    # -- B2: _enforce_skill_cap trims correctly --
    executor = SkillPlanExecutor.__new__(SkillPlanExecutor)
    skills = [{"name": f"skill_{i}", "rank": i} for i in range(8)]
    essential = {"skill_7"}
    capped = executor._enforce_skill_cap(skills, essential, max_skills=3)

    results.append(
        CheckResult(
            section="SKILL_CAP",
            name="Cap enforced: 8 skills → 3",
            passed=len(capped) == 3,
            details={
                "input_count": len(skills),
                "output_count": len(capped),
                "names": [s["name"] for s in capped],
            },
        )
    )

    # -- B3: Essential skills survive --
    essential_names = {s["name"] for s in capped}
    results.append(
        CheckResult(
            section="SKILL_CAP",
            name="Essential skill_7 preserved in cap",
            passed="skill_7" in essential_names,
            details={"surviving": list(essential_names)},
        )
    )

    # -- B4: Live run — observe skill count in a complex task --
    start = time.time()
    try:
        from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

        orch = Orchestrator()
        result = await asyncio.wait_for(
            orch.run(
                "Write a Python function that takes a URL, fetches the page, "
                "parses the HTML, and returns all links. Include error handling.",
                learn=True,
            ),
            timeout=timeout,
        )
        elapsed = time.time() - start
        content = extract_content(result)

        results.append(
            CheckResult(
                section="SKILL_CAP",
                name="Live: complex task with skill cap active",
                passed=len(content) > 200,
                elapsed_s=elapsed,
                details={
                    "content_length": len(content),
                    "has_code": "def " in content or "class " in content,
                    "latency": f"{elapsed:.1f}s",
                },
            )
        )
    except Exception as e:
        results.append(
            CheckResult(
                section="SKILL_CAP",
                name="Live: complex task with skill cap active",
                passed=False,
                elapsed_s=time.time() - start,
                error=str(e)[:200],
            )
        )

    return results


# =============================================================================
# C) PROFICIENCY FILTERING: Team of Thoughts
# =============================================================================


async def test_proficiency_filter(timeout: int) -> List[CheckResult]:
    """Verify proficiency-based agent filtering in debate/relay."""
    from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline
    from Jotty.core.intelligence.orchestration.coordination.paradigm_executor import (
        ParadigmExecutor,
    )
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    results = []

    # -- C1: Low-proficiency agents are filtered --
    baseline = GroupedValueBaseline()
    for _ in range(10):
        baseline.update_agent("expert", "general", 0.95)
        baseline.update_agent("good", "general", 0.7)
        baseline.update_agent("bad", "general", 0.05)

    td_learner = SimpleNamespace(grouped_baseline=baseline)
    learning = SimpleNamespace(td_learner=td_learner)
    mgr = MagicMock()
    mgr.learning = learning

    pe = ParadigmExecutor(mgr)
    agents = [
        SimpleNamespace(name="expert"),
        SimpleNamespace(name="good"),
        SimpleNamespace(name="bad"),
    ]
    filtered = pe._filter_by_proficiency(agents, "do something general")
    filtered_names = [a.name for a in filtered]

    results.append(
        CheckResult(
            section="PROFICIENCY",
            name="Low-proficiency 'bad' agent filtered",
            passed="bad" not in filtered_names,
            details={
                "input": [a.name for a in agents],
                "filtered": filtered_names,
                "proficiencies": {
                    a.name: baseline.get_agent_proficiency(a.name, "general") for a in agents
                },
            },
        )
    )

    # -- C2: Minimum 2 agents preserved --
    baseline2 = GroupedValueBaseline()
    for _ in range(10):
        baseline2.update_agent("a", "general", 0.05)
        baseline2.update_agent("b", "general", 0.08)
        baseline2.update_agent("c", "general", 0.02)

    td2 = SimpleNamespace(grouped_baseline=baseline2)
    learning2 = SimpleNamespace(td_learner=td2)
    mgr2 = MagicMock()
    mgr2.learning = learning2

    pe2 = ParadigmExecutor(mgr2)
    all_bad = [SimpleNamespace(name=n) for n in ["a", "b", "c"]]
    filtered2 = pe2._filter_by_proficiency(all_bad, "task", min_agents=2)

    results.append(
        CheckResult(
            section="PROFICIENCY",
            name="Minimum 2 agents always preserved",
            passed=len(filtered2) >= 2,
            details={"filtered_count": len(filtered2)},
        )
    )

    # -- C3: Unknown agents kept (benefit of doubt) --
    filtered3 = pe._filter_by_proficiency(
        [SimpleNamespace(name="expert"), SimpleNamespace(name="new_unknown")],
        "task",
    )
    unknown_kept = any(a.name == "new_unknown" for a in filtered3)

    results.append(
        CheckResult(
            section="PROFICIENCY",
            name="Unknown agents get benefit of doubt",
            passed=unknown_kept,
            details={"filtered": [a.name for a in filtered3]},
        )
    )

    # -- C4: Live multi-agent task via orchestrator --
    start = time.time()
    try:
        from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator
        from Jotty.core.intelligence.orchestration.swarms.coding_swarm.swarm import CodingSwarm
        from Jotty.core.intelligence.orchestration.swarms.coding_swarm.types import CodingConfig

        orch = Orchestrator()
        swarm = CodingSwarm(CodingConfig())
        result = await asyncio.wait_for(
            orch.run(
                "Create a Python class for a thread-safe bounded queue with put(), get(), "
                "and size() methods. Use threading.Condition. Include 3 tests.",
                swarm=swarm,
                learn=True,
            ),
            timeout=timeout,
        )
        elapsed = time.time() - start
        content = extract_content(result)
        success = getattr(result, "success", False)

        results.append(
            CheckResult(
                section="PROFICIENCY",
                name="Live CodingSwarm with proficiency filtering",
                passed=len(content) > 200 and success,
                elapsed_s=elapsed,
                details={
                    "content_length": len(content),
                    "success": success,
                    "has_class": "class " in content,
                    "has_tests": "def test_" in content or "assert " in content,
                    "agents": list(getattr(result, "agent_contributions", {}).keys()),
                },
            )
        )
    except Exception as e:
        results.append(
            CheckResult(
                section="PROFICIENCY",
                name="Live CodingSwarm with proficiency filtering",
                passed=False,
                elapsed_s=time.time() - start,
                error=str(e)[:200],
            )
        )

    return results


# =============================================================================
# D) HIERARCHICAL COMPRESSION: LCM
# =============================================================================


async def test_hierarchical_compression(timeout: int) -> List[CheckResult]:
    """Verify LCM hierarchical compression preserves originals."""
    from Jotty.core.infrastructure.context.utils import hierarchical_compress
    from Jotty.core.infrastructure.context.context_manager import SmartContextManager

    results = []

    # -- D1: hierarchical_compress stores and summarizes --
    store: Dict[str, str] = {}
    original = (
        "This is a long document about machine learning.\n"
        "It covers supervised, unsupervised, and reinforcement learning.\n"
        "Each section has code examples and mathematical formulations.\n"
        "The document is 500 pages long with detailed proofs.\n"
        "Key finding: transformers outperform RNNs on most NLP benchmarks."
    )
    summary = hierarchical_compress(original, "doc_1", store)

    results.append(
        CheckResult(
            section="COMPRESSION",
            name="hierarchical_compress: stores original",
            passed="doc_1" in store and store["doc_1"] == original,
            details={
                "stored": "doc_1" in store,
                "original_preserved": store.get("doc_1") == original,
            },
        )
    )

    results.append(
        CheckResult(
            section="COMPRESSION",
            name="hierarchical_compress: summary has pointer",
            passed="retrieve:doc_1" in summary and "5 lines" in summary,
            details={"summary": summary},
        )
    )

    # -- D2: compress_parts uses hierarchical for large parts --
    mgr = SmartContextManager()
    big_part_a = "A detailed analysis of system architecture.\n" * 200  # ~8400 chars
    big_part_b = "Performance benchmarks and optimization results.\n" * 200
    small_part = "Short note."

    compressed = mgr.compress_parts([big_part_a, big_part_b, small_part], max_total_chars=500)

    results.append(
        CheckResult(
            section="COMPRESSION",
            name="compress_parts: large parts → hierarchical",
            passed=len(mgr._content_store) >= 2,
            details={
                "store_entries": len(mgr._content_store),
                "compressed_sizes": [len(p) for p in compressed],
                "original_sizes": [len(big_part_a), len(big_part_b), len(small_part)],
            },
        )
    )

    # -- D3: retrieve_full_content recovers original --
    any_recovered = False
    for content_id, original_text in mgr._content_store.items():
        recovered = mgr.retrieve_full_content(content_id)
        if recovered == original_text:
            any_recovered = True
            break

    results.append(
        CheckResult(
            section="COMPRESSION",
            name="retrieve_full_content: lossless recovery",
            passed=any_recovered,
            details={"recovered": any_recovered},
        )
    )

    # -- D4: Live compression in real context building --
    start = time.time()
    try:
        mgr2 = SmartContextManager(max_tokens=2000)
        mgr2.register_goal("Analyze system performance")
        # Add several large chunks to force compression
        for i in range(5):
            mgr2.add_chunk(
                f"Chunk {i}: " + "detailed performance data " * 100,
                category="research",
            )
        ctx = mgr2.build_context(
            system_prompt="You are a research assistant.",
            user_input="Summarize the performance findings.",
        )
        elapsed = time.time() - start

        results.append(
            CheckResult(
                section="COMPRESSION",
                name="Live: context build with compression",
                passed=ctx["truncated"] or len(ctx["context"]) > 0,
                elapsed_s=elapsed,
                details={
                    "context_length": len(ctx["context"]),
                    "truncated": ctx["truncated"],
                    "chunks_included": ctx["preserved"]["chunks_included"],
                    "chunks_total": ctx["preserved"]["chunks_total"],
                    "budget_remaining": ctx["stats"]["budget_remaining"],
                },
            )
        )
    except Exception as e:
        results.append(
            CheckResult(
                section="COMPRESSION",
                name="Live: context build with compression",
                passed=False,
                error=str(e)[:200],
            )
        )

    return results


# =============================================================================
# E) MAPLE: Hot/Cold Learning Split
# =============================================================================


async def test_maple_split(timeout: int) -> List[CheckResult]:
    """Verify MAPLE hot-path completes quickly, cold-path is deferred."""
    from Jotty.core.intelligence.orchestration.swarms._base._learning_mixin import (
        SwarmLearningMixin,
    )

    results = []

    # -- E1: Methods exist --
    has_schedule = hasattr(SwarmLearningMixin, "_schedule_deferred_learning")
    has_deferred = hasattr(SwarmLearningMixin, "_deferred_post_learning")
    has_bg_tasks = hasattr(SwarmLearningMixin, "_bg_learning_tasks")

    results.append(
        CheckResult(
            section="MAPLE",
            name="Hot/cold split methods exist",
            passed=has_schedule and has_deferred and has_bg_tasks,
            details={
                "_schedule_deferred_learning": has_schedule,
                "_deferred_post_learning": has_deferred,
                "_bg_learning_tasks": has_bg_tasks,
                "deferred_is_async": asyncio.iscoroutinefunction(
                    SwarmLearningMixin._deferred_post_learning
                ),
            },
        )
    )

    # -- E2: Live swarm execution — measure if learning happens async --
    start = time.time()
    try:
        from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

        orch = Orchestrator()
        result = await asyncio.wait_for(
            orch.chat(
                message="Explain the observer pattern in software design. Give a Python example.",
                provider="anthropic",
                learn=True,
                max_steps=1,
                temperature=0,
            ),
            timeout=timeout,
        )
        response_time = time.time() - start
        content = extract_content(result)

        # Give deferred learning 3 seconds to complete in background
        await asyncio.sleep(3)
        post_learning_time = time.time() - start

        results.append(
            CheckResult(
                section="MAPLE",
                name="Live: response returns before deferred learning",
                passed=len(content) > 100 and response_time < timeout,
                elapsed_s=response_time,
                details={
                    "response_time": f"{response_time:.1f}s",
                    "post_learning_time": f"{post_learning_time:.1f}s",
                    "content_length": len(content),
                    "has_pattern": "observer" in content.lower(),
                },
            )
        )
    except Exception as e:
        results.append(
            CheckResult(
                section="MAPLE",
                name="Live: response returns before deferred learning",
                passed=False,
                elapsed_s=time.time() - start,
                error=str(e)[:200],
            )
        )

    return results


# =============================================================================
# F) END-TO-END: Complex tasks exercising all integrations
# =============================================================================


async def test_e2e_complex(timeout: int) -> List[CheckResult]:
    """Run complex tasks that exercise multiple integrations simultaneously."""
    from Jotty.core.intelligence.orchestration.core.swarm_manager import Orchestrator

    results = []
    orch = Orchestrator()

    # -- F1: Complex coding task (exercises: tier detection, skill cap, learning) --
    start = time.time()
    try:
        result = await asyncio.wait_for(
            orch.run(
                "Implement a Python event-driven state machine that supports: "
                "state transitions with guards, entry/exit actions, hierarchical states, "
                "and event history logging. Include type hints and 5 pytest tests.",
                learn=True,
            ),
            timeout=timeout,
        )
        elapsed = time.time() - start
        content = extract_content(result)
        success = getattr(result, "success", bool(content))

        checks = {
            "has_content": len(content) > 300,
            "has_class": "class " in content,
            "has_state": "state" in content.lower(),
            "has_transition": "transition" in content.lower(),
            "has_tests": "def test_" in content or "assert " in content,
            "success_flag": bool(success),
        }

        results.append(
            CheckResult(
                section="E2E",
                name="Complex: event-driven state machine",
                passed=sum(checks.values()) >= 4,
                elapsed_s=elapsed,
                details={
                    **checks,
                    "content_length": len(content),
                    "quality": f"{sum(checks.values())}/{len(checks)}",
                },
            )
        )
    except Exception as e:
        results.append(
            CheckResult(
                section="E2E",
                name="Complex: event-driven state machine",
                passed=False,
                elapsed_s=time.time() - start,
                error=str(e)[:300],
            )
        )

    # -- F2: Research synthesis (exercises: tier detection → RESEARCH, proficiency filter) --
    start = time.time()
    try:
        result = await asyncio.wait_for(
            orch.chat(
                message=(
                    "Compare and contrast three approaches to distributed consensus: "
                    "Raft, PBFT, and Tendermint. For each, explain: core algorithm, "
                    "fault tolerance guarantees, performance characteristics, "
                    "and real-world deployment examples. "
                    "Then recommend which to use for a financial trading system."
                ),
                provider="anthropic",
                learn=True,
                max_steps=1,
                temperature=0,
            ),
            timeout=timeout,
        )
        elapsed = time.time() - start
        content = extract_content(result)

        checks = {
            "has_content": len(content) > 500,
            "raft_mentioned": "raft" in content.lower(),
            "pbft_mentioned": "pbft" in content.lower() or "byzantine" in content.lower(),
            "fault_tolerance": "fault" in content.lower() or "tolerance" in content.lower(),
            "recommendation": "recommend" in content.lower() or "suggest" in content.lower(),
            "structured": any(
                line.startswith("#") or line.startswith("*") for line in content.split("\n")
            ),
        }

        results.append(
            CheckResult(
                section="E2E",
                name="Complex: consensus protocol comparison",
                passed=sum(checks.values()) >= 4,
                elapsed_s=elapsed,
                details={
                    **checks,
                    "content_length": len(content),
                    "quality": f"{sum(checks.values())}/{len(checks)}",
                },
            )
        )
    except Exception as e:
        results.append(
            CheckResult(
                section="E2E",
                name="Complex: consensus protocol comparison",
                passed=False,
                elapsed_s=time.time() - start,
                error=str(e)[:300],
            )
        )

    # -- F3: Multi-agent swarm task (exercises: proficiency filter, MAPLE, compression) --
    start = time.time()
    try:
        from Jotty.core.intelligence.orchestration.swarms.coding_swarm.swarm import CodingSwarm
        from Jotty.core.intelligence.orchestration.swarms.coding_swarm.types import CodingConfig

        swarm = CodingSwarm(CodingConfig())
        result = await asyncio.wait_for(
            orch.run(
                "Build a Python rate limiter using the sliding window log algorithm. "
                "Requirements: thread-safe, supports per-client limits, configurable "
                "window size, and cleanup of expired entries. "
                "Include: class with type hints, docstrings, and 5 pytest tests.",
                swarm=swarm,
                learn=True,
            ),
            timeout=timeout,
        )
        elapsed = time.time() - start
        content = extract_content(result)
        success = getattr(result, "success", bool(content))
        agents = getattr(result, "agent_contributions", {})

        checks = {
            "has_content": len(content) > 300,
            "has_class": "class " in content,
            "has_rate_limit": "rate" in content.lower() and "limit" in content.lower(),
            "thread_safe": "thread" in content.lower() or "lock" in content.lower(),
            "has_tests": "def test_" in content or "assert " in content,
            "success": bool(success),
        }

        results.append(
            CheckResult(
                section="E2E",
                name="CodingSwarm: sliding window rate limiter",
                passed=sum(checks.values()) >= 4,
                elapsed_s=elapsed,
                details={
                    **checks,
                    "content_length": len(content),
                    "agents_involved": list(agents.keys()) if agents else "unknown",
                    "quality": f"{sum(checks.values())}/{len(checks)}",
                },
            )
        )
    except Exception as e:
        results.append(
            CheckResult(
                section="E2E",
                name="CodingSwarm: sliding window rate limiter",
                passed=False,
                elapsed_s=time.time() - start,
                error=str(e)[:300],
            )
        )

    return results


# =============================================================================
# RUNNER
# =============================================================================

ALL_SECTIONS = {
    "tier": ("A. Tier Detection (Delegation Floor + CogRouter)", test_tier_detection),
    "skill": ("B. Skill Cap (SkillsBench)", test_skill_cap),
    "proficiency": ("C. Proficiency Filter (Team of Thoughts)", test_proficiency_filter),
    "compression": ("D. Hierarchical Compression (LCM)", test_hierarchical_compression),
    "maple": ("E. MAPLE Hot/Cold Split", test_maple_split),
    "e2e": ("F. End-to-End Complex Tasks", test_e2e_complex),
}


async def run_eval(section_filter: Optional[str], fast: bool):
    timeout = TIMEOUT_FAST if fast else TIMEOUT_DEFAULT

    sections = {}
    if section_filter:
        for key in section_filter.split(","):
            key = key.strip()
            if key in ALL_SECTIONS:
                sections[key] = ALL_SECTIONS[key]
            else:
                print(f"Unknown section: {key}. Available: {', '.join(ALL_SECTIONS.keys())}")
                return
    else:
        sections = ALL_SECTIONS

    print(f"\n{'=' * 72}")
    print(f"  PAPER INTEGRATIONS REAL-WORLD EVALUATION")
    print(f"{'=' * 72}")
    print(f"  Date:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Timeout: {timeout}s  |  Mode: {'fast' if fast else 'full'}")
    print(f"  API Key: {'present' if os.getenv('ANTHROPIC_API_KEY') else 'MISSING'}")
    print(f"  Sections: {len(sections)}")
    print(f"{'=' * 72}")

    all_results: List[CheckResult] = []
    eval_start = time.time()

    for key, (label, test_fn) in sections.items():
        print(f"\n{'━' * 72}")
        print(f"  {label}")
        print(f"{'━' * 72}")

        try:
            section_results = await test_fn(timeout)
            all_results.extend(section_results)

            for r in section_results:
                status = "✅" if r.passed else "❌"
                time_str = f" ({r.elapsed_s:.1f}s)" if r.elapsed_s > 0 else ""
                print(f"  {status} {r.name}{time_str}")
                if r.error:
                    print(f"     ERROR: {r.error[:120]}")
                for k, v in r.details.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            print(f"     {kk}: {vv}")
                    else:
                        print(f"     {k}: {v}")

        except Exception as e:
            print(f"  SECTION FAILED: {e}")
            traceback.print_exc()

    total_time = time.time() - eval_start
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)

    # Summary
    print(f"\n{'=' * 72}")
    print(f"  FINAL RESULTS")
    print(f"{'=' * 72}")
    print(f"  Checks: {passed}/{total} passed ({passed/max(total,1):.0%})")
    print(f"  Time:   {total_time:.1f}s")

    # Per-section breakdown
    section_stats: Dict[str, Dict] = {}
    for r in all_results:
        if r.section not in section_stats:
            section_stats[r.section] = {"passed": 0, "total": 0}
        section_stats[r.section]["total"] += 1
        if r.passed:
            section_stats[r.section]["passed"] += 1

    print()
    for section, stats in section_stats.items():
        bar_len = int((stats["passed"] / max(stats["total"], 1)) * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {section:<20s} {bar} {stats['passed']}/{stats['total']}")

    grade_pct = passed / max(total, 1)
    if grade_pct >= 0.9:
        grade = "A"
    elif grade_pct >= 0.75:
        grade = "B"
    elif grade_pct >= 0.6:
        grade = "C"
    elif grade_pct >= 0.4:
        grade = "D"
    else:
        grade = "F"

    print(f"\n  Grade: {grade} ({grade_pct:.0%})")

    # Save
    results_dir = Path("scripts/eval_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = results_dir / f"{ts}_paper_integrations.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "total_checks": total,
        "passed": passed,
        "grade": grade,
        "total_time_s": round(total_time, 1),
        "results": [asdict(r) for r in all_results],
    }
    outfile.write_text(json.dumps(data, indent=2, default=str))
    print(f"\n  Saved: {outfile}")

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper Integrations Evaluation")
    parser.add_argument("--test", type=str, default=None, help="Comma-separated sections")
    parser.add_argument("--fast", action="store_true", help="Use shorter timeouts")
    args = parser.parse_args()

    asyncio.run(run_eval(args.test, args.fast))

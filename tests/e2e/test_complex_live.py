#!/usr/bin/env python3
"""Complex real-world use cases through Jotty orchestrator with real LLM."""

import asyncio
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path

env_file = Path(__file__).parents[2] / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip() and k.strip() not in os.environ:
                    os.environ[k.strip()] = v.strip()

import logging

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


async def run_test(name, coro):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    start = time.time()
    try:
        result = await coro
        elapsed = time.time() - start
        print(f"  Time: {elapsed:.1f}s")
        return result, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"  FAILED ({elapsed:.1f}s): {e}")
        traceback.print_exc()
        return None, elapsed


async def main():
    from Jotty import Jotty
    from Jotty.core.intelligence.orchestration.execution import ExecutionTier, ExecutionConfig

    jotty = Jotty(log_level="WARNING")
    results = {}

    # ── 1. Multi-step research with synthesis ──────────────────────────
    async def test_research_synthesis():
        r = await jotty.run(
            "Compare microservices vs monolith architecture for a fintech startup "
            "processing 10,000 transactions/day. Consider: latency requirements, "
            "team size of 5 developers, regulatory compliance (PCI-DSS), and "
            "deployment complexity. Provide a concrete recommendation with trade-offs.",
            tier=ExecutionTier.AGENTIC,
        )
        out = str(r.output)
        print(f"  Success: {r.success} | Output: {len(out)} chars")
        print(f"  Preview: {out[:300]}")
        assert r.success and len(out) > 100
        return True

    r, t = await run_test(
        "1. ARCHITECTURE DECISION — multi-factor analysis", test_research_synthesis()
    )
    results["1_arch_decision"] = "PASS" if r else "FAIL"

    # ── 2. Code generation with explanation ────────────────────────────
    async def test_code_gen():
        config = ExecutionConfig(tier=ExecutionTier.AGENTIC)
        r = await jotty.run(
            "Write a Python class `RateLimiter` that implements token bucket algorithm. "
            "It should support: configurable rate and burst size, thread-safety, "
            "async `acquire()` method that waits if no tokens available, and a "
            "`try_acquire()` that returns immediately. Include docstrings and type hints. "
            "Then explain the time complexity of each method.",
            config=config,
        )
        out = str(r.output)
        print(f"  Success: {r.success} | Output: {len(out)} chars")
        print(f"  Preview: {out[:300]}")
        assert r.success and len(out) > 200
        # Should contain code artifacts
        has_class = "class" in out.lower() or "ratelimiter" in out.lower() or "rate" in out.lower()
        print(f"  Has code content: {has_class}")
        return True

    r, t = await run_test("2. CODE GENERATION — token bucket rate limiter", test_code_gen())
    results["2_code_gen"] = "PASS" if r else "FAIL"

    # ── 3. Data analysis reasoning chain ───────────────────────────────
    async def test_data_analysis():
        r = await jotty.run(
            "A SaaS company has these metrics over 6 months:\n"
            "Month 1: MRR $50K, Churn 3.2%, CAC $450, LTV $5400\n"
            "Month 2: MRR $58K, Churn 2.8%, CAC $420, LTV $5800\n"
            "Month 3: MRR $63K, Churn 3.5%, CAC $480, LTV $5200\n"
            "Month 4: MRR $61K, Churn 4.1%, CAC $510, LTV $4900\n"
            "Month 5: MRR $59K, Churn 4.8%, CAC $550, LTV $4500\n"
            "Month 6: MRR $56K, Churn 5.2%, CAC $580, LTV $4200\n\n"
            "Identify the inflection point, root cause hypothesis, and "
            "3 specific actions to reverse the trend. Use the LTV:CAC ratio "
            "to support your analysis.",
            tier=ExecutionTier.AGENTIC,
        )
        out = str(r.output)
        print(f"  Success: {r.success} | Output: {len(out)} chars")
        print(f"  Preview: {out[:300]}")
        assert r.success and len(out) > 100
        return True

    r, t = await run_test("3. DATA ANALYSIS — SaaS metrics trend reversal", test_data_analysis())
    results["3_data_analysis"] = "PASS" if r else "FAIL"

    # ── 4. Tier 4 RESEARCH swarm — deep domain task ───────────────────
    async def test_research_swarm():
        config = ExecutionConfig(
            tier=ExecutionTier.RESEARCH,
            swarm_name="research",
            timeout_seconds=180,
        )
        r = await jotty.run(
            "Research the impact of the EU AI Act on open-source AI development. "
            "Cover: classification of AI systems by risk level, compliance requirements "
            "for foundation model providers, specific exemptions for open-source, "
            "and how this compares to US executive orders on AI. "
            "Cite specific articles from the regulation where possible.",
            config=config,
        )
        out = str(r.output)
        print(f"  Success: {r.success} | Output: {len(out)} chars")
        if isinstance(r.output, dict):
            for k, v in r.output.items():
                if v:
                    print(f"    {k}: {str(v)[:100]}")
        else:
            print(f"  Preview: {out[:400]}")
        assert r.success
        return True

    r, t = await run_test("4. RESEARCH SWARM — EU AI Act analysis", test_research_swarm())
    results["4_research_swarm"] = "PASS" if r else "FAIL"

    # ── 5. Multi-step planning with constraints ────────────────────────
    async def test_constrained_planning():
        r = await jotty.run(
            "Design a database schema for a multi-tenant project management tool like Jira. "
            "Requirements:\n"
            "- Tenant isolation (row-level security)\n"
            "- Projects, epics, stories, tasks hierarchy\n"
            "- Custom fields per tenant\n"
            "- Audit trail for all changes\n"
            "- Efficient queries for kanban boards (filter by status, assignee, sprint)\n"
            "- Support 10M+ issues across all tenants\n\n"
            "Provide: the SQL DDL for core tables, key indexes, and explain "
            "partitioning strategy for scale.",
            tier=ExecutionTier.AGENTIC,
        )
        out = str(r.output)
        print(f"  Success: {r.success} | Output: {len(out)} chars")
        print(f"  Preview: {out[:300]}")
        assert r.success and len(out) > 200
        return True

    r, t = await run_test(
        "5. SCHEMA DESIGN — multi-tenant project management", test_constrained_planning()
    )
    results["5_schema_design"] = "PASS" if r else "FAIL"

    # ── 6. Tier 3 LEARNING — with memory and validation ────────────────
    async def test_learning_tier():
        config = ExecutionConfig(
            tier=ExecutionTier.LEARNING,
            memory_backend="json",
            enable_validation=True,
            timeout_seconds=120,
        )
        r = await jotty.run(
            "Explain how B-trees work in database indexes. Cover: node structure, "
            "search algorithm, insertion with splits, deletion with merging, "
            "and why B+ trees are preferred for disk-based storage. "
            "Include Big-O complexity for each operation.",
            config=config,
        )
        out = str(r.output)
        print(f"  Success: {r.success} | Output: {len(out)} chars")
        print(f"  Preview: {out[:300]}")
        assert r.success and len(out) > 100
        return True

    r, t = await run_test("6. LEARNING TIER — B-tree index internals", test_learning_tier())
    results["6_learning_tier"] = "PASS" if r else "FAIL"

    # ── 7. Coding swarm — real implementation task ─────────────────────
    async def test_coding_swarm():
        config = ExecutionConfig(
            tier=ExecutionTier.RESEARCH,
            swarm_name="coding",
            timeout_seconds=180,
        )
        r = await jotty.run(
            "Implement a Python LRU cache from scratch (no functools) that supports:\n"
            "- O(1) get and put operations\n"
            "- Maximum capacity with eviction\n"
            "- Thread-safe operations\n"
            "- TTL (time-to-live) per entry\n"
            "- Statistics (hits, misses, evictions)\n"
            "Include comprehensive unit tests using pytest.",
            config=config,
        )
        out = str(r.output)
        print(f"  Success: {r.success} | Output: {len(out)} chars")
        if isinstance(r.output, dict):
            for k, v in r.output.items():
                if v:
                    print(f"    {k}: {str(v)[:100]}")
        else:
            print(f"  Preview: {out[:400]}")
        assert r.success
        return True

    r, t = await run_test("7. CODING SWARM — LRU cache implementation", test_coding_swarm())
    results["7_coding_swarm"] = "PASS" if r else "FAIL"

    # ── 8. Complex multi-domain reasoning ──────────────────────────────
    async def test_multi_domain():
        r = await jotty.run(
            "A startup is choosing between these tech stacks for a real-time "
            "collaborative document editor (like Google Docs):\n\n"
            "Option A: React + Yjs CRDT + WebSocket + PostgreSQL + Redis\n"
            "Option B: Next.js + Automerge CRDT + WebSocket + MongoDB + Redis\n"
            "Option C: SolidJS + Diamond Types OT + gRPC + ScyllaDB + NATS\n\n"
            "For each option analyze: conflict resolution approach, "
            "offline support capability, operational complexity, "
            "latency at 10K concurrent editors, and hiring difficulty. "
            "Then make a final recommendation considering the team has "
            "3 senior engineers and needs to ship MVP in 8 weeks.",
            tier=ExecutionTier.AGENTIC,
        )
        out = str(r.output)
        print(f"  Success: {r.success} | Output: {len(out)} chars")
        print(f"  Preview: {out[:300]}")
        assert r.success and len(out) > 200
        return True

    r, t = await run_test("8. MULTI-DOMAIN — collaborative editor tech stack", test_multi_domain())
    results["8_multi_domain"] = "PASS" if r else "FAIL"

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    passed = sum(1 for v in results.values() if v == "PASS")
    for name, status in results.items():
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {name}: {status}")
    print(f"\n  {passed}/{len(results)} passed")
    print(f"{'='*70}")
    return passed == len(results)


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)

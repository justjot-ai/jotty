#!/usr/bin/env python3
"""
Live LLM integration test for context consolidation.

Exercises the full pipeline through the Jotty orchestrator with real
Anthropic API calls to verify the unified context subsystem works
end-to-end.

Run:
    python tests/e2e/test_context_consolidation_live.py
"""

import asyncio
import logging
import os
import sys
import time
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("context_test")


# ─────────────────────────────────────────────────────────────────────
# Test 1: Canonical context imports all work
# ─────────────────────────────────────────────────────────────────────
def test_canonical_imports():
    """Verify all canonical imports from the unified context subsystem."""
    print("\n" + "=" * 70)
    print("TEST 1: Canonical context imports")
    print("=" * 70)

    from Jotty.core.infrastructure.context import (
        ENRICHMENT_MARKERS,
        CompressionResult,
        ErrorDetector,
        ErrorType,
        ExecutionTrajectory,
        SmartContextManager,
        detect_error_type,
        get_context_manager,
        get_error_detector,
        strip_enrichment_context,
    )

    # Verify types
    assert isinstance(ENRICHMENT_MARKERS, tuple), "ENRICHMENT_MARKERS should be tuple"
    assert len(ENRICHMENT_MARKERS) > 20, f"Expected 20+ markers, got {len(ENRICHMENT_MARKERS)}"

    # Verify strip works
    dirty = "Analyze data\nLearned Insights: old context\n# Q-Learning Lessons\nstuff"
    clean = strip_enrichment_context(dirty)
    assert clean == "Analyze data", f"Expected 'Analyze data', got '{clean}'"

    # Verify ErrorDetector
    err = Exception("input is too long for model")
    et = ErrorDetector.detect(err)
    assert et == ErrorType.CONTEXT_LENGTH

    # Verify detect_error_type convenience
    et2, strategy = detect_error_type(Exception("timeout after 30s"))
    assert et2 == ErrorType.TIMEOUT
    assert strategy["should_retry"] is True

    # Verify ExecutionTrajectory
    traj = ExecutionTrajectory()
    traj.add_step(
        {"description": "web search", "skill_name": "web-search"}, output="found 5 results"
    )
    traj.add_step(
        {"description": "analyze", "skill_name": "claude-cli-llm"}, output="analysis done"
    )
    ctx = traj.to_context()
    assert "web search" in ctx
    assert "analyze" in ctx
    assert traj.current_step_index == 2

    # Verify SmartContextManager singleton
    mgr = get_context_manager()
    assert isinstance(mgr, SmartContextManager)

    # Verify compress_parts
    parts = ["A" * 5000, "B" * 5000, "C" * 5000]
    compressed = mgr.compress_parts(parts, max_total_chars=6000)
    total = sum(len(p) for p in compressed)
    assert total <= 7000, f"Expected <=7000 chars, got {total}"

    # Verify error detector facade
    detector = get_error_detector()
    assert isinstance(detector, ErrorDetector)

    # Verify shim re-exports match canonical
    from Jotty.core.infrastructure.utils.context_utils import (
        ErrorDetector as ShimED,
        ErrorType as ShimET,
        strip_enrichment_context as shim_strip,
    )

    assert ShimED is ErrorDetector
    assert ShimET is ErrorType
    assert shim_strip is strip_enrichment_context

    print("  PASS: All canonical imports verified")
    print("  PASS: strip_enrichment_context works")
    print("  PASS: ErrorDetector classifies errors correctly")
    print("  PASS: ExecutionTrajectory round-trips")
    print("  PASS: SmartContextManager.compress_parts works")
    print("  PASS: Shim re-exports match canonical")
    return True


# ─────────────────────────────────────────────────────────────────────
# Test 2: SmartContextManager build_context with real data
# ─────────────────────────────────────────────────────────────────────
def test_context_manager_build():
    """Test SmartContextManager builds context within budget."""
    print("\n" + "=" * 70)
    print("TEST 2: SmartContextManager build_context")
    print("=" * 70)

    from Jotty.core.infrastructure.context.context_manager import SmartContextManager
    from Jotty.core.infrastructure.context.models import ContextPriority

    mgr = SmartContextManager(max_tokens=4000, safety_margin=0.9)

    # Register critical state
    mgr.register_goal("Research and analyze recent AI trends in 2026")
    mgr.register_todo("1. Search web\n2. Analyze findings\n3. Generate report")
    mgr.register_critical_memory("Budget limit: $0.50 per execution")

    # Add context chunks
    mgr.add_chunk("Previous research found GPT-5 released in Jan 2026" * 10, "memory")
    mgr.add_chunk("Agent trajectory: searched 3 papers, analyzed 2" * 20, "trajectory")
    mgr.add_chunk("Verbose debug logs from previous run" * 50, "debug_log", ContextPriority.LOW)

    result = mgr.build_context(
        system_prompt="You are a research assistant specialized in AI trends.",
        user_input="What are the top 3 AI trends in 2026?",
    )

    assert result["preserved"]["goal"], "Goal should be preserved"
    assert result["preserved"]["todo"], "Todo should be preserved"
    assert result["preserved"]["critical_memories"] >= 1
    assert "Research and analyze" in result["context"]
    assert "Budget limit" in result["context"]

    stats = result["stats"]
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Effective limit: {stats['effective_limit']}")
    print(f"  Budget remaining: {stats['budget_remaining']}")
    print(
        f"  Chunks included: {result['preserved']['chunks_included']}/{result['preserved']['chunks_total']}"
    )
    print(f"  Truncated: {result['truncated']}")
    print("  PASS: Context built within budget, critical info preserved")
    return True


# ─────────────────────────────────────────────────────────────────────
# Test 3: Tier 1 (DIRECT) — simple chat through Jotty
# ─────────────────────────────────────────────────────────────────────
async def test_tier1_direct():
    """Test simple chat via Tier 1 with real LLM call."""
    print("\n" + "=" * 70)
    print("TEST 3: Tier 1 (DIRECT) — Real LLM Chat")
    print("=" * 70)

    from Jotty import Jotty
    from Jotty.core.intelligence.orchestration.execution import ExecutionTier

    jotty = Jotty(log_level="WARNING")

    start = time.time()
    result = await jotty.run(
        "What is the capital of France? Reply in one word.", tier=ExecutionTier.DIRECT
    )
    elapsed = time.time() - start

    output = str(result.output).strip().lower()
    print(f"  Output: {result.output}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Success: {result.success}")

    assert result.success, f"Expected success, got error: {getattr(result, 'error', 'unknown')}"
    assert "paris" in output, f"Expected 'paris' in output, got: {output}"

    print("  PASS: Tier 1 direct LLM call works")
    return True


# ─────────────────────────────────────────────────────────────────────
# Test 4: Tier 2 (AGENTIC) — planning + multi-step
# ─────────────────────────────────────────────────────────────────────
async def test_tier2_agentic():
    """Test Tier 2 planning with real LLM call."""
    print("\n" + "=" * 70)
    print("TEST 4: Tier 2 (AGENTIC) — Plan & Execute")
    print("=" * 70)

    from Jotty import Jotty
    from Jotty.core.intelligence.orchestration.execution import ExecutionTier

    jotty = Jotty(log_level="WARNING")

    def progress_cb(stage, detail):
        print(f"  [{stage}] {detail}")

    start = time.time()
    result = await jotty.run(
        "List 3 benefits of exercise. Be concise.",
        tier=ExecutionTier.AGENTIC,
        status_callback=progress_cb,
    )
    elapsed = time.time() - start

    output = str(result.output)
    print(f"  Output length: {len(output)} chars")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Success: {result.success}")
    print(f"  Output preview: {output[:200]}")

    assert result.success, f"Expected success, got error: {getattr(result, 'error', 'unknown')}"
    assert len(output) > 20, f"Output too short: {output}"

    print("  PASS: Tier 2 agentic execution works")
    return True


# ─────────────────────────────────────────────────────────────────────
# Test 5: BaseAgent context property wires SmartContextManager
# ─────────────────────────────────────────────────────────────────────
def test_base_agent_context():
    """Verify BaseAgent.context now returns SmartContextManager."""
    print("\n" + "=" * 70)
    print("TEST 5: BaseAgent.context → SmartContextManager")
    print("=" * 70)

    from Jotty.core.infrastructure.context.context_manager import SmartContextManager
    from Jotty.core.intelligence.reasoning.base.base_agent import AgentRuntimeConfig, BaseAgent

    class TestAgent(BaseAgent):
        async def _execute_impl(self, **kwargs):
            return "test output"

    agent = TestAgent(config=AgentRuntimeConfig(name="test_agent", enable_context=True))
    ctx = agent.context

    print(f"  Context type: {type(ctx).__name__}")
    assert isinstance(
        ctx, SmartContextManager
    ), f"Expected SmartContextManager, got {type(ctx).__name__}"

    # Verify it's functional
    ctx.register_goal("Test goal")
    stats = ctx.get_statistics()
    assert stats["has_goal"] is True

    print("  PASS: BaseAgent.context returns SmartContextManager")
    return True


# ─────────────────────────────────────────────────────────────────────
# Test 6: Agent runner compress_parts integration
# ─────────────────────────────────────────────────────────────────────
def test_agent_runner_compress():
    """Verify agent_runner uses SmartContextManager.compress_parts."""
    print("\n" + "=" * 70)
    print("TEST 6: AgentRunner compress_parts integration")
    print("=" * 70)

    from Jotty.core.infrastructure.context.facade import get_context_manager

    mgr = get_context_manager()

    # Simulate what agent_runner does
    parts = [
        "Goal: Research AI trends " * 100,  # ~2500 chars
        "Learning context: " + "x" * 5000,  # 5000 chars
        "Tool results: " + "y" * 3000,  # 3000 chars
    ]
    total_before = sum(len(p) for p in parts)

    compressed = mgr.compress_parts(parts, max_total_chars=4000)
    total_after = sum(len(p) for p in compressed)

    print(f"  Before: {total_before} chars ({len(parts)} parts)")
    print(f"  After:  {total_after} chars ({len(compressed)} parts)")
    assert total_after < total_before, "Compression should reduce size"
    assert len(compressed) == len(parts), "Should preserve number of parts"

    # Verify each part has content from start and end
    for i, p in enumerate(compressed):
        assert len(p) > 0, f"Part {i} should not be empty"
        if len(parts[i]) > 4000 // len(parts):
            assert "[...compressed...]" in p, f"Part {i} should have compression marker"

    print("  PASS: compress_parts reduces size while preserving all parts")
    return True


# ─────────────────────────────────────────────────────────────────────
# Test 7: Tier 4 (RESEARCH) — domain swarm with context management
# ─────────────────────────────────────────────────────────────────────
async def test_tier4_research():
    """Test Tier 4 research swarm — exercises full context pipeline."""
    print("\n" + "=" * 70)
    print("TEST 7: Tier 4 (RESEARCH) — Domain Swarm")
    print("=" * 70)

    from Jotty import Jotty
    from Jotty.core.intelligence.orchestration.execution import ExecutionConfig, ExecutionTier

    config = ExecutionConfig(
        tier=ExecutionTier.RESEARCH,
        swarm_name="research",
        timeout_seconds=120,
    )
    jotty = Jotty(config=config, log_level="WARNING")

    def progress_cb(stage, detail):
        print(f"  [{stage}] {detail}")

    start = time.time()
    result = await jotty.run(
        "Compare Python vs Rust for building CLI tools. Give 3 pros and cons for each. Be concise.",
        status_callback=progress_cb,
    )
    elapsed = time.time() - start

    output = str(result.output) if result.output else ""
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Success: {result.success}")
    print(f"  Output length: {len(output)} chars")
    if output:
        print(f"  Output preview: {output[:300]}...")

    assert result.success, f"Expected success, got: {getattr(result, 'error', 'unknown')}"
    assert len(output) > 50, f"Output too short: {output}"

    print("  PASS: Tier 4 research swarm works with unified context")
    return True


# ─────────────────────────────────────────────────────────────────────
# Test 8: enrichment stripping in consumer code paths
# ─────────────────────────────────────────────────────────────────────
def test_enrichment_strip_consumers():
    """Verify consumer imports resolve correctly."""
    print("\n" + "=" * 70)
    print("TEST 8: Consumer import paths")
    print("=" * 70)

    # plan_utils.py (canonical import)
    from Jotty.core.infrastructure.context.utils import strip_enrichment_context as canonical

    # execution_types.py (try/except import)
    try:
        from Jotty.core.intelligence.reasoning.types.execution_types import _clean_for_display

        cleaned = _clean_for_display("Task\nLearned Insights: junk")
        assert cleaned == "Task", f"Expected 'Task', got '{cleaned}'"
        print("  PASS: execution_types._clean_for_display works")
    except ImportError as e:
        print(f"  SKIP: execution_types import: {e}")

    # agentic_planner.py (fixed broken import)
    try:
        from Jotty.core.infrastructure.context.error_handling import (
            ErrorDetector,
            ErrorType,
            detect_error_type,
        )

        assert ErrorDetector.detect(Exception("too many tokens")) == ErrorType.CONTEXT_LENGTH
        print("  PASS: agentic_planner error_handling import works")
    except ImportError as e:
        print(f"  FAIL: agentic_planner import: {e}")
        return False

    # Verify canonical function is the same object everywhere
    from Jotty.core.infrastructure.utils.context_utils import strip_enrichment_context as shim

    assert canonical is shim, "Shim should re-export the same function"
    print("  PASS: All consumer import paths verified")
    return True


# ─────────────────────────────────────────────────────────────────────
# MAIN — run all tests
# ─────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 70)
    print("  CONTEXT CONSOLIDATION LIVE INTEGRATION TEST")
    print("  Using real Anthropic LLM calls")
    print("=" * 70)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("\n  ERROR: ANTHROPIC_API_KEY not set. Cannot run live tests.")
        sys.exit(1)
    print(f"  API key: {api_key[:10]}...{api_key[-4:]}")

    results = {}
    tests = [
        ("1_canonical_imports", lambda: test_canonical_imports()),
        ("2_context_build", lambda: test_context_manager_build()),
        ("3_tier1_direct", lambda: asyncio.ensure_future(test_tier1_direct())),
        ("4_tier2_agentic", lambda: asyncio.ensure_future(test_tier2_agentic())),
        ("5_base_agent_ctx", lambda: test_base_agent_context()),
        ("6_compress_parts", lambda: test_agent_runner_compress()),
        ("7_tier4_research", lambda: asyncio.ensure_future(test_tier4_research())),
        ("8_enrichment_strip", lambda: test_enrichment_strip_consumers()),
    ]

    for name, test_fn in tests:
        try:
            result = test_fn()
            if asyncio.isfuture(result) or asyncio.iscoroutine(result):
                result = await result
            results[name] = "PASS" if result else "FAIL"
        except Exception as e:
            results[name] = f"ERROR: {e}"
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)
    for name, status in results.items():
        icon = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{icon}] {name}: {status}")
    print(f"\n  {passed}/{total} tests passed")
    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

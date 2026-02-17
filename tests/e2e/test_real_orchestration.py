#!/usr/bin/env python3
"""
Real LLM Orchestration Tests — End-to-End
==========================================

Tests Jotty's full orchestration pipeline with REAL Anthropic API calls.
Covers: ChatExecutor, ModeRouter, AutoAgent, TierExecutor, Skills, Streaming.

Usage:
    # Load .env then run
    python tests/e2e/test_real_orchestration.py

    # Or with pytest
    pytest tests/e2e/test_real_orchestration.py -v -s
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
# Silence noisy loggers
for name in ("httpx", "httpcore", "urllib3", "anthropic", "dspy"):
    logging.getLogger(name).setLevel(logging.ERROR)

logger = logging.getLogger("test_orchestration")
logger.setLevel(logging.INFO)


# ============================================================================
# Test Result Tracking
# ============================================================================


@dataclass
class TestResult:
    name: str
    success: bool
    duration: float
    output_preview: str = ""
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class TestRunner:
    def __init__(self):
        self.results: List[TestResult] = []
        self.total_cost = 0.0

    def record(self, result: TestResult):
        self.results.append(result)

    def print_summary(self):
        print("\n" + "=" * 80)
        print("JOTTY REAL ORCHESTRATION TEST RESULTS")
        print("=" * 80)

        passed = sum(1 for r in self.results if r.success)
        failed = sum(1 for r in self.results if not r.success)
        total_time = sum(r.duration for r in self.results)

        for i, r in enumerate(self.results, 1):
            status = "PASS" if r.success else "FAIL"
            icon = "+" if r.success else "X"
            print(f"\n  [{icon}] {i}. {r.name} ({r.duration:.1f}s) — {status}")
            if r.output_preview:
                # Truncate long output
                preview = r.output_preview[:200].replace("\n", " ")
                print(f"      Output: {preview}...")
            if r.error:
                print(f"      Error: {r.error[:200]}")
            if r.metadata:
                meta_str = ", ".join(f"{k}={v}" for k, v in r.metadata.items())
                print(f"      Meta: {meta_str}")

        print(f"\n{'=' * 80}")
        print(f"  PASSED: {passed}/{len(self.results)}")
        print(f"  FAILED: {failed}/{len(self.results)}")
        print(f"  TOTAL TIME: {total_time:.1f}s")
        print(f"{'=' * 80}")

        return failed == 0


runner = TestRunner()


# ============================================================================
# Test Helpers
# ============================================================================


def check_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key or key == "your-key-here":
        print("ERROR: ANTHROPIC_API_KEY not set. Load .env first.")
        sys.exit(1)
    print(f"  API Key: ...{key[-8:]}")


# ============================================================================
# TEST 1: ChatExecutor — Direct LLM Tool-Calling
# ============================================================================


async def test_chat_executor_simple():
    """Test ChatExecutor with a simple factual question."""
    from Jotty.core.intelligence.orchestration.execution.unified_executor import ChatExecutor

    start = time.time()
    try:
        executor = ChatExecutor(provider="anthropic")
        result = await executor.execute("What are the 3 laws of thermodynamics? Be concise.")

        duration = time.time() - start
        success = result.success and len(result.content) > 50
        runner.record(
            TestResult(
                name="ChatExecutor: Simple factual question",
                success=success,
                duration=duration,
                output_preview=result.content[:300] if result.content else "No content",
                metadata={"steps": result.steps_taken, "tools": len(result.tool_results)},
            )
        )
    except Exception as e:
        runner.record(
            TestResult(
                name="ChatExecutor: Simple factual question",
                success=False,
                duration=time.time() - start,
                error=str(e),
            )
        )


# ============================================================================
# TEST 2: ChatExecutor — Web Search Tool Use
# ============================================================================


async def test_chat_executor_web_search():
    """Test ChatExecutor with a query that requires web search."""
    from Jotty.core.intelligence.orchestration.execution.unified_executor import ChatExecutor

    start = time.time()
    try:
        executor = ChatExecutor(provider="anthropic")
        result = await executor.execute(
            "Search the web for the latest AI news from February 2026. Summarize top 3 stories."
        )

        duration = time.time() - start
        used_tools = [t.tool_name for t in result.tool_results] if result.tool_results else []
        has_search = any(
            "search" in t.lower() or "web" in t.lower() or "fetch" in t.lower() for t in used_tools
        )

        runner.record(
            TestResult(
                name="ChatExecutor: Web search + summarize",
                success=result.success and len(result.content) > 100,
                duration=duration,
                output_preview=result.content[:300] if result.content else "No content",
                metadata={"tools_used": used_tools, "web_search": has_search},
            )
        )
    except Exception as e:
        runner.record(
            TestResult(
                name="ChatExecutor: Web search + summarize",
                success=False,
                duration=time.time() - start,
                error=str(e),
            )
        )


# ============================================================================
# TEST 3: ChatExecutor — Multi-Step Tool Calling (complex)
# ============================================================================


async def test_chat_executor_multi_step():
    """Test ChatExecutor with a complex multi-step task."""
    from Jotty.core.intelligence.orchestration.execution.unified_executor import ChatExecutor

    start = time.time()
    try:
        executor = ChatExecutor(provider="anthropic", max_steps=5)
        result = await executor.execute(
            "Calculate: If I invest $10,000 at 7% annual compound interest for 20 years, "
            "what will be the final amount? Also search the web for the current S&P 500 index value. "
            "Present both results together."
        )

        duration = time.time() - start
        used_tools = [t.tool_name for t in result.tool_results] if result.tool_results else []

        runner.record(
            TestResult(
                name="ChatExecutor: Multi-step (calculate + search)",
                success=result.success and len(result.content) > 50,
                duration=duration,
                output_preview=result.content[:300] if result.content else "No content",
                metadata={
                    "steps": result.steps_taken,
                    "tools": used_tools,
                    "num_tools": len(used_tools),
                },
            )
        )
    except Exception as e:
        runner.record(
            TestResult(
                name="ChatExecutor: Multi-step (calculate + search)",
                success=False,
                duration=time.time() - start,
                error=str(e),
            )
        )


# ============================================================================
# TEST 4: ChatExecutor — Streaming
# ============================================================================


async def test_chat_executor_streaming():
    """Test ChatExecutor with streaming output."""
    from Jotty.core.intelligence.orchestration.execution.unified_executor import ChatExecutor

    start = time.time()
    chunks = []

    def stream_cb(chunk: str):
        chunks.append(chunk)

    try:
        executor = ChatExecutor(
            provider="anthropic",
            stream_callback=stream_cb,
        )
        result = await executor.execute("Write a haiku about artificial intelligence.")

        duration = time.time() - start
        total_streamed = "".join(chunks)

        runner.record(
            TestResult(
                name="ChatExecutor: Streaming response",
                success=result.success and len(chunks) > 0,
                duration=duration,
                output_preview=total_streamed[:300] if total_streamed else result.content[:300],
                metadata={"chunks_received": len(chunks), "total_chars": len(total_streamed)},
            )
        )
    except Exception as e:
        runner.record(
            TestResult(
                name="ChatExecutor: Streaming response",
                success=False,
                duration=time.time() - start,
                error=str(e),
            )
        )


# ============================================================================
# TEST 5: ModeRouter — Intelligent Routing (DIRECT vs FULL)
# ============================================================================


async def test_mode_router():
    """Test ModeRouter with different complexity levels."""
    from Jotty.core.interface.api.mode_router import ModeRouter
    from Jotty.core.infrastructure.foundation.types.sdk_types import (
        ExecutionContext,
        ExecutionMode,
        ChannelType,
    )

    router = ModeRouter()
    start = time.time()

    try:
        ctx = ExecutionContext(
            mode=ExecutionMode.CHAT,
            channel=ChannelType.CLI,
        )

        result = await router.route("What is 2+2?", ctx)

        duration = time.time() - start
        content = str(result.content) if result.content else ""
        validation_mode = (
            result.metadata.get("validation_mode", "unknown") if result.metadata else "unknown"
        )

        runner.record(
            TestResult(
                name="ModeRouter: Simple question (should route DIRECT)",
                success=result.success and "4" in content,
                duration=duration,
                output_preview=content[:300],
                metadata={"validation_mode": validation_mode, "mode": result.mode.value},
            )
        )
    except Exception as e:
        runner.record(
            TestResult(
                name="ModeRouter: Simple question (should route DIRECT)",
                success=False,
                duration=time.time() - start,
                error=str(e),
            )
        )


# ============================================================================
# TEST 6: ModeRouter — Complex Workflow
# ============================================================================


async def test_mode_router_complex():
    """Test ModeRouter with a complex query needing full pipeline."""
    from Jotty.core.interface.api.mode_router import ModeRouter
    from Jotty.core.infrastructure.foundation.types.sdk_types import (
        ExecutionContext,
        ExecutionMode,
        ChannelType,
    )

    router = ModeRouter()
    start = time.time()

    try:
        ctx = ExecutionContext(
            mode=ExecutionMode.CHAT,
            channel=ChannelType.CLI,
        )

        result = await router.route(
            "Compare the pros and cons of Python vs Rust for building web APIs. "
            "Consider performance, developer experience, ecosystem maturity, and deployment.",
            ctx,
        )

        duration = time.time() - start
        content = str(result.content) if result.content else ""

        runner.record(
            TestResult(
                name="ModeRouter: Complex analysis (Python vs Rust)",
                success=result.success and len(content) > 200,
                duration=duration,
                output_preview=content[:300],
                metadata={
                    "validation_mode": (
                        result.metadata.get("validation_mode", "?") if result.metadata else "?"
                    ),
                    "content_length": len(content),
                },
            )
        )
    except Exception as e:
        runner.record(
            TestResult(
                name="ModeRouter: Complex analysis (Python vs Rust)",
                success=False,
                duration=time.time() - start,
                error=str(e),
            )
        )


# ============================================================================
# TEST 7: AutoAgent — Multi-Step Planning + Execution
# ============================================================================


async def test_auto_agent():
    """Test AutoAgent with a task requiring planning."""
    start = time.time()

    try:
        from Jotty.core.intelligence.reasoning.agents.auto_agent import AutoAgent

        agent = AutoAgent(max_steps=5, timeout=120)
        result = await agent.execute(
            "Create a Python function that calculates the Fibonacci sequence "
            "up to n terms using dynamic programming. Include docstring and type hints."
        )

        duration = time.time() - start
        output = ""
        success = False

        if isinstance(result, dict):
            output = result.get("final_output", str(result))
            success = result.get("success", False)
        elif hasattr(result, "final_output"):
            output = str(result.final_output)
            success = getattr(result, "success", True)
        else:
            output = str(result)
            success = len(output) > 50

        has_code = "def " in output or "fibonacci" in output.lower()

        runner.record(
            TestResult(
                name="AutoAgent: Code generation (Fibonacci DP)",
                success=success and has_code,
                duration=duration,
                output_preview=output[:400],
                metadata={"has_code": has_code, "output_len": len(output)},
            )
        )
    except Exception as e:
        runner.record(
            TestResult(
                name="AutoAgent: Code generation (Fibonacci DP)",
                success=False,
                duration=time.time() - start,
                error=str(e),
            )
        )


# ============================================================================
# TEST 8: TierExecutor — Tiered Execution System
# ============================================================================


async def test_tier_executor():
    """Test TierExecutor which auto-detects complexity and routes to appropriate tier."""
    start = time.time()

    try:
        from Jotty.core.intelligence.orchestration.execution.tier_executor import TierExecutor

        executor = TierExecutor()
        result = await executor.execute(goal="Explain quantum entanglement in simple terms")

        duration = time.time() - start
        content = str(result.output) if result.output else ""
        tier_name = result.tier.name if hasattr(result.tier, "name") else str(result.tier)

        runner.record(
            TestResult(
                name="TierExecutor: Auto-tier detection + execution",
                success=result.success and len(content) > 50,
                duration=duration,
                output_preview=content[:300],
                metadata={
                    "tier": tier_name,
                    "llm_calls": result.llm_calls,
                    "cost_usd": f"${result.cost_usd:.4f}",
                },
            )
        )
    except Exception as e:
        runner.record(
            TestResult(
                name="TierExecutor: Auto-tier detection + execution",
                success=False,
                duration=time.time() - start,
                error=str(e),
            )
        )


# ============================================================================
# TEST 9: SDK Client — End-to-End via Jotty SDK
# ============================================================================


async def test_sdk_client():
    """Test the full SDK client (apps layer -> SDK -> core)."""
    start = time.time()

    try:
        from Jotty.sdk import Jotty

        client = Jotty()
        client.use_local()  # Use internal APIs, not HTTP server
        response = await client.chat(
            "List 5 design patterns commonly used in Python with one-line descriptions."
        )

        duration = time.time() - start
        content = response.content if response.content else ""
        success = response.success and len(content) > 100

        mode_val = "?"
        if hasattr(response, "mode") and response.mode is not None:
            mode_val = response.mode.value

        runner.record(
            TestResult(
                name="SDK Client: chat() end-to-end",
                success=success,
                duration=duration,
                output_preview=content[:300] if content else "No content",
                metadata={
                    "mode": mode_val,
                    "content_length": len(content),
                },
            )
        )
    except Exception as e:
        runner.record(
            TestResult(
                name="SDK Client: chat() end-to-end",
                success=False,
                duration=time.time() - start,
                error=str(e),
            )
        )


# ============================================================================
# TEST 10: Skills Registry — Tool Discovery + Execution
# ============================================================================


async def test_skills_integration():
    """Test that skills are discovered and can be listed."""
    start = time.time()

    try:
        from Jotty.core.capabilities.registry.skills_registry import get_skills_registry

        registry = get_skills_registry()
        registry.init()

        # Discover skills with a broad query
        skills = registry.discover(task="general purpose tools")
        total_loaded = len(registry.loaded_skills)

        # Check specific skills load
        web_search = registry.get_skill("web-search")
        ws_tools = list(web_search.tools.keys()) if web_search else []

        math = registry.get_skill("math-toolkit")
        math_tools = list(math.tools.keys()) if math else []

        duration = time.time() - start

        runner.record(
            TestResult(
                name="Skills Registry: Discovery + tool loading",
                success=total_loaded > 100 and web_search is not None and math is not None,
                duration=duration,
                output_preview=f"Loaded: {total_loaded}, discovered: {len(skills)}, web-search: {ws_tools}, math: {math_tools}",
                metadata={
                    "total_loaded": total_loaded,
                    "discovered_for_query": len(skills),
                    "web_search_tools": len(ws_tools),
                    "math_tools": len(math_tools),
                },
            )
        )
    except Exception as e:
        runner.record(
            TestResult(
                name="Skills Registry: Discovery + tool loading",
                success=False,
                duration=time.time() - start,
                error=str(e),
            )
        )


# ============================================================================
# MAIN
# ============================================================================


async def main():
    print("\n" + "=" * 80)
    print("JOTTY REAL ORCHESTRATION TESTS — With Live Anthropic API")
    print("=" * 80)

    check_api_key()

    tests = [
        ("Skills Registry", test_skills_integration),
        ("ChatExecutor Simple", test_chat_executor_simple),
        ("ChatExecutor Streaming", test_chat_executor_streaming),
        ("ChatExecutor Web Search", test_chat_executor_web_search),
        ("ChatExecutor Multi-Step", test_chat_executor_multi_step),
        ("ModeRouter Simple", test_mode_router),
        ("ModeRouter Complex", test_mode_router_complex),
        ("AutoAgent Code Gen", test_auto_agent),
        ("TierExecutor", test_tier_executor),
        ("SDK Client E2E", test_sdk_client),
    ]

    print(f"\n  Running {len(tests)} tests with REAL LLM calls...")
    print(f"  (This will use your Anthropic API key and incur costs)\n")

    for i, (label, test_fn) in enumerate(tests, 1):
        print(f"  [{i}/{len(tests)}] {label}...", end=" ", flush=True)
        try:
            await test_fn()
            latest = runner.results[-1]
            status = "PASS" if latest.success else "FAIL"
            print(f"{status} ({latest.duration:.1f}s)")
        except Exception as e:
            print(f"CRASH: {e}")
            runner.record(
                TestResult(
                    name=label,
                    success=False,
                    duration=0,
                    error=f"Unhandled: {e}",
                )
            )

    all_passed = runner.print_summary()
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

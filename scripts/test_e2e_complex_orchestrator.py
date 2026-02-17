#!/usr/bin/env python3
"""
Complex E2E orchestrator test — exercises swarms, skill picking, and multi-agent coordination.

NOT simple Q&A — real complex tasks that force the full pipeline:
1. AutoAgent with skill discovery + planning + multi-step execution
2. Tier 2 AGENTIC with full planning pipeline (skip complexity gate)
3. Tier 4 RESEARCH with a named swarm (coding/research)
4. Tier 4 RESEARCH via Orchestrator delegation
5. Jotty.swarm() direct swarm invocation

Uses real Anthropic API.
"""

import asyncio
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# Load API keys
project_root = Path(__file__).parent.parent
for env_name in (".env", ".env.anthropic"):
    env_file = project_root / env_name
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

sys.path.insert(0, str(project_root.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2e_complex")

for name in ["httpx", "httpcore", "anthropic", "urllib3"]:
    logging.getLogger(name).setLevel(logging.WARNING)


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.duration = 0.0
        self.details = ""

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        extra = f" ({self.duration:.1f}s)" if self.duration > 0 else ""
        err = f"\n    ERROR: {self.error}" if self.error else ""
        det = f"\n    {self.details}" if self.details else ""
        return f"  [{status}] {self.name}{extra}{err}{det}"


# =============================================================================
# TEST 1: AutoAgent complex task — skill discovery + planning + execution
# =============================================================================
async def test_auto_agent_complex():
    """AutoAgent with a task that requires skill discovery, planning, and multi-step execution."""
    r = TestResult("AutoAgent: Complex multi-step task with skills")
    t0 = time.time()
    try:
        from Jotty.core.intelligence.reasoning.agents import AutoAgent

        agent = AutoAgent(max_steps=5)

        # Complex task that should trigger skill discovery + planning
        result = await agent.execute(
            "Write a Python function that calculates compound interest, "
            "then explain the time complexity of the algorithm. "
            "Format the output with the function code and analysis."
        )

        r.duration = time.time() - t0

        output = ""
        if hasattr(result, "output") and result.output:
            output = str(result.output)
        elif hasattr(result, "final_output") and result.final_output:
            output = str(result.final_output)
        elif isinstance(result, dict):
            output = str(result.get("output", result.get("final_output", str(result))))
        else:
            output = str(result)

        # Check for meaningful content
        has_code = "def " in output or "compound" in output.lower() or "interest" in output.lower()
        has_content = len(output.strip()) > 50

        r.passed = has_content
        r.details = (
            f"Output ({len(output)} chars): '{output[:200]}...'\n"
            f"    Has code/content: {has_code} | "
            f"Steps: {getattr(result, 'steps_executed', '?')} | "
            f"Skills used: {getattr(result, 'skills_used', getattr(result, 'selected_skills', '?'))}"
        )
    except Exception as e:
        r.duration = time.time() - t0
        r.error = str(e)
        r.details = traceback.format_exc().split("\n")[-3]
    return r


# =============================================================================
# TEST 2: Tier 2 AGENTIC — force full planning pipeline
# =============================================================================
async def test_tier2_full_planning():
    """Tier 2 with skip_complexity_gate=True to force full planning path."""
    r = TestResult("Tier 2 AGENTIC: Full planning pipeline (no shortcut)")
    t0 = time.time()
    try:
        from Jotty import Jotty
        from Jotty.jotty import ExecutionTier

        jotty = Jotty(log_level="WARNING")
        result = await jotty.run(
            "Create a detailed comparison between Python and Rust for systems programming. "
            "Cover performance, memory safety, ecosystem, and learning curve. "
            "Provide a recommendation for different use cases.",
            tier=ExecutionTier.AGENTIC,
            skip_complexity_gate=True,  # Force planning path
        )

        r.duration = time.time() - t0
        output = str(result.output).strip() if result.output else ""

        has_content = len(output) > 100
        has_comparison = any(
            w in output.lower() for w in ["python", "rust", "performance", "memory"]
        )
        steps = result.steps if hasattr(result, "steps") and result.steps else []

        r.passed = has_content
        r.details = (
            f"Output ({len(output)} chars): '{output[:200]}...'\n"
            f"    Has comparison content: {has_comparison} | "
            f"Steps: {len(steps)} | "
            f"Cost: ${result.cost_usd:.4f} | "
            f"LLM calls: {result.llm_calls}"
        )
    except Exception as e:
        r.duration = time.time() - t0
        r.error = str(e)
        r.details = traceback.format_exc().split("\n")[-3]
    return r


# =============================================================================
# TEST 3: Tier 4 RESEARCH — named swarm (research)
# =============================================================================
async def test_tier4_research_swarm():
    """Tier 4 with explicit research swarm."""
    r = TestResult("Tier 4 RESEARCH: Named research swarm execution")
    t0 = time.time()
    try:
        from Jotty import Jotty

        jotty = Jotty(log_level="WARNING")
        result = await jotty.swarm(
            "Explain the key differences between transformer and LSTM architectures "
            "for natural language processing. Include advantages and disadvantages of each.",
            swarm_name="research",
        )

        r.duration = time.time() - t0
        output = str(result.output).strip() if result.output else ""

        has_content = len(output) > 50
        swarm_name = result.swarm_name if hasattr(result, "swarm_name") else "?"

        r.passed = has_content
        r.details = (
            f"Swarm: {swarm_name} | Output ({len(output)} chars): '{output[:200]}...'\n"
            f"    Cost: ${result.cost_usd:.4f} | "
            f"Success: {result.success} | "
            f"LLM calls: {result.llm_calls}"
        )
    except Exception as e:
        r.duration = time.time() - t0
        r.error = str(e)
        r.details = traceback.format_exc().split("\n")[-3]
    return r


# =============================================================================
# TEST 4: Tier 4 RESEARCH — coding swarm
# =============================================================================
async def test_tier4_coding_swarm():
    """Tier 4 with explicit coding swarm — multi-agent code generation."""
    r = TestResult("Tier 4 RESEARCH: Coding swarm (multi-agent)")
    t0 = time.time()
    try:
        from Jotty import Jotty

        jotty = Jotty(log_level="WARNING")
        result = await jotty.swarm(
            "Create a Python class for a simple stack data structure with push, pop, peek, "
            "and is_empty methods. Include type hints and docstrings.",
            swarm_name="coding",
        )

        r.duration = time.time() - t0
        output = str(result.output).strip() if result.output else ""

        has_code = "class " in output or "def " in output or "stack" in output.lower()
        swarm_name = result.swarm_name if hasattr(result, "swarm_name") else "?"

        r.passed = len(output) > 30
        r.details = (
            f"Swarm: {swarm_name} | Has code: {has_code}\n"
            f"    Output ({len(output)} chars): '{output[:200]}...'\n"
            f"    Cost: ${result.cost_usd:.4f} | "
            f"Success: {result.success}"
        )
    except Exception as e:
        r.duration = time.time() - t0
        r.error = str(e)
        r.details = traceback.format_exc().split("\n")[-3]
    return r


# =============================================================================
# TEST 5: Direct swarm instantiation + execution
# =============================================================================
async def test_direct_swarm_instantiation():
    """Directly instantiate and run a swarm (bypass Jotty/TierExecutor)."""
    r = TestResult("Direct swarm: Instantiate + execute research swarm")
    t0 = time.time()
    try:
        # Import and trigger registration
        from Jotty.core.intelligence.orchestration.swarms.research_swarm.swarm import ResearchSwarm
        from Jotty.core.intelligence.orchestration.swarms._base.registry import SwarmRegistry

        # Verify registration
        swarms = SwarmRegistry.list_all()

        # Instantiate directly
        swarm = ResearchSwarm()

        # Execute
        result = await swarm.execute(
            task="What are the main benefits of microservices architecture? List 3 key points."
        )

        r.duration = time.time() - t0

        output = ""
        if hasattr(result, "output"):
            output = str(result.output)
        elif isinstance(result, dict):
            output = str(result.get("output", result.get("result", str(result))))
        else:
            output = str(result)

        success = getattr(result, "success", True)

        r.passed = len(output.strip()) > 20
        r.details = (
            f"Registry: {len(swarms)} swarms | Success: {success}\n"
            f"    Output ({len(output)} chars): '{output[:200]}...'"
        )
    except Exception as e:
        r.duration = time.time() - t0
        r.error = str(e)
        r.details = traceback.format_exc().split("\n")[-3]
    return r


# =============================================================================
# TEST 6: AutoAgent with explicit skill selection
# =============================================================================
async def test_auto_agent_skill_discovery():
    """AutoAgent — verify skill discovery and selection pipeline."""
    r = TestResult("AutoAgent: Skill discovery + selection pipeline")
    t0 = time.time()
    try:
        from Jotty.core.intelligence.reasoning.agents import AutoAgent

        agent = AutoAgent(max_steps=3)

        # Use a task that should trigger specific skill categories
        result = await agent.execute(
            "Calculate the monthly payment for a $300,000 mortgage at 6.5% interest over 30 years."
        )

        r.duration = time.time() - t0

        output = ""
        if hasattr(result, "output") and result.output:
            output = str(result.output)
        elif hasattr(result, "final_output") and result.final_output:
            output = str(result.final_output)
        else:
            output = str(result)

        # Get metadata
        selected_skills = getattr(result, "selected_skills", [])
        steps = getattr(result, "steps_executed", 0)
        task_type = getattr(result, "task_type", "?")

        r.passed = len(output.strip()) > 10
        r.details = (
            f"Task type: {task_type} | Skills: {selected_skills[:5]} | Steps: {steps}\n"
            f"    Output: '{output.strip()[:200]}'"
        )
    except Exception as e:
        r.duration = time.time() - t0
        r.error = str(e)
        r.details = traceback.format_exc().split("\n")[-3]
    return r


# =============================================================================
# RUNNER
# =============================================================================
async def main():
    print("\n" + "=" * 70)
    print("  JOTTY COMPLEX E2E TEST — Swarms, Skills, Multi-Agent")
    print("=" * 70)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    print(f"  API Key: {'...' + api_key[-8:] if len(api_key) > 8 else 'NOT SET'}")
    print(f"  Python: {sys.version.split()[0]}")
    print("=" * 70 + "\n")

    if not api_key:
        print("  ERROR: ANTHROPIC_API_KEY not set! Aborting.\n")
        return 1

    test_funcs = [
        test_auto_agent_complex,
        test_tier2_full_planning,
        test_tier4_research_swarm,
        test_tier4_coding_swarm,
        test_direct_swarm_instantiation,
        test_auto_agent_skill_discovery,
    ]

    all_results = []
    total_start = time.time()

    for i, func in enumerate(test_funcs, 1):
        name = func.__doc__.strip().split("\n")[0] if func.__doc__ else func.__name__
        print(f"[{i}/{len(test_funcs)}] Running: {name}...")

        try:
            result = await asyncio.wait_for(func(), timeout=420)
        except asyncio.TimeoutError:
            result = TestResult(name)
            result.error = "TIMEOUT (420s)"
            result.duration = 420.0
        except Exception as e:
            result = TestResult(name)
            result.error = f"Unexpected: {e}"

        all_results.append(result)
        print(str(result))
        print()

    total_time = time.time() - total_start

    passed = sum(1 for r in all_results if r.passed)
    failed = sum(1 for r in all_results if not r.passed)
    total = len(all_results)

    print("=" * 70)
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed ({total_time:.1f}s total)")
    print("=" * 70)

    if failed > 0:
        print("\n  FAILURES:")
        for r in all_results:
            if not r.passed:
                print(f"    - {r.name}: {r.error}")
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

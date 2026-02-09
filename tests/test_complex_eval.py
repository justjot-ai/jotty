#!/usr/bin/env python3
"""
JOTTY V2 — COMPLEX USE CASE EVALUATION (Post-Fix)
====================================================

Tests 4 genuinely complex, real-world scenarios after applying:
  Fix 1: Faster web scraping (8s timeout, 20s budget)
  Fix 2: Skip web-search for knowledge tasks
  Fix 3: Total execution budget with graceful degradation (240s)
  Fix 4: Code gen saves .py not .html
  Fix 5: image-generator lazy imports (no crash)
  Fix 6: Parallel task-type + skill discovery

Each scenario has a 300s timeout and reports:
  - Success/failure, latency, output length
  - Quality metrics (structure, keywords, depth)
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# ── Load API keys ──
for env_file in [Path(__file__).parents[2] / '.env.anthropic',
                 Path(__file__).parents[1] / '.env.anthropic',
                 Path(__file__).parents[1] / '.env']:
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    k, v = k.strip(), v.strip()
                    if v and k not in os.environ:
                        os.environ[k] = v

import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)-8s %(name)s: %(message)s')
logger = logging.getLogger('complex_eval')
logger.setLevel(logging.INFO)

B = '\033[1m'; G = '\033[92m'; R = '\033[91m'; Y = '\033[93m'; D = '\033[2m'; E = '\033[0m'


# ══════════════════════════════════════════════════════════════════
# SCENARIO 1: Code Generation (FastAPI REST API)
# Tests: planner routing, file extension fix, code quality
# ══════════════════════════════════════════════════════════════════
async def scenario_1_codegen():
    from Jotty.core.orchestration.v2.swarm_manager import SwarmManager

    print(f"\n{B}SCENARIO 1: FastAPI REST API Code Generation{E}")
    print(f"{D}  Goal: Generate a complete FastAPI CRUD API with models{E}")

    sm = SwarmManager(enable_lotus=False, enable_zero_config=False)
    trail = []
    t0 = time.time()

    result = await sm.run(
        "Create a complete FastAPI REST API with CRUD operations for a 'User' model. "
        "Include: models with Pydantic, router with GET/POST/PUT/DELETE endpoints, "
        "in-memory storage, and error handling. Save as app.py",
        skip_autonomous_setup=True,
        ensemble=False,
        status_callback=lambda s, d: trail.append((time.time() - t0, s, d)),
    )

    elapsed = time.time() - t0
    output = str(result.output) if result.output else ""
    out_len = len(output)

    # Quality checks
    has_fastapi = 'fastapi' in output.lower() or 'FastAPI' in output
    has_crud = sum(1 for kw in ['get', 'post', 'put', 'delete'] if kw in output.lower()) >= 3
    has_pydantic = 'BaseModel' in output or 'pydantic' in output.lower()
    has_code = 'def ' in output or 'async def' in output
    saved_py = any('app.py' in str(d) for _, s, d in trail if 'file' in s.lower() or 'write' in s.lower() or 'done' in s.lower())

    quality = sum([has_fastapi, has_crud, has_pydantic, has_code]) / 4

    print(f"  {'PASS' if result.success else 'FAIL'} | {elapsed:.0f}s | {out_len:,} chars | quality={quality:.0%}")
    print(f"  FastAPI={has_fastapi} CRUD={has_crud} Pydantic={has_pydantic} Code={has_code}")
    for ts, s, d in trail[-5:]:
        print(f"  {D}  [{ts:.1f}s] {s}: {d[:80]}{E}")

    return result.success, elapsed, out_len, quality


# ══════════════════════════════════════════════════════════════════
# SCENARIO 2: Architecture Design (Knowledge Task - NO web needed)
# Tests: Fix 2 (skip web-search), knowledge tasks use LLM directly
# This was the one that TIMED OUT before the fix.
# ══════════════════════════════════════════════════════════════════
async def scenario_2_architecture():
    from Jotty.core.orchestration.v2.swarm_manager import SwarmManager

    print(f"\n{B}SCENARIO 2: Event-Driven Architecture Design{E}")
    print(f"{D}  Goal: Design document WITHOUT web scraping (LLM knowledge only){E}")

    sm = SwarmManager(enable_lotus=False, enable_zero_config=False)
    trail = []
    t0 = time.time()

    result = await sm.run(
        "Design an event-driven microservices architecture for an e-commerce platform. "
        "Cover: service decomposition (order, inventory, payment, notification), "
        "event bus design (Kafka vs RabbitMQ trade-offs), saga pattern for distributed "
        "transactions, CQRS for read/write separation, and failure handling strategies. "
        "Include concrete code patterns in Python.",
        skip_autonomous_setup=True,
        ensemble=False,
        status_callback=lambda s, d: trail.append((time.time() - t0, s, d)),
    )

    elapsed = time.time() - t0
    output = str(result.output) if result.output else ""
    out_len = len(output)

    # Quality checks — should have deep architecture content
    has_services = sum(1 for kw in ['order', 'inventory', 'payment', 'notification'] if kw in output.lower()) >= 3
    has_patterns = sum(1 for kw in ['saga', 'cqrs', 'event', 'kafka', 'rabbitmq'] if kw in output.lower()) >= 3
    has_code = 'def ' in output or 'class ' in output or 'async' in output
    has_depth = out_len > 2000  # Architecture doc should be substantial
    no_web_used = not any('web-search' in str(d) or 'scrape' in str(d) for _, s, d in trail)

    quality = sum([has_services, has_patterns, has_code, has_depth]) / 4

    print(f"  {'PASS' if result.success else 'FAIL'} | {elapsed:.0f}s | {out_len:,} chars | quality={quality:.0%}")
    print(f"  Services={has_services} Patterns={has_patterns} Code={has_code} Depth={has_depth}")
    print(f"  {G}Web skipped={no_web_used}{E}" if no_web_used else f"  {R}Web used (should have been skipped){E}")
    for ts, s, d in trail[-5:]:
        print(f"  {D}  [{ts:.1f}s] {s}: {d[:80]}{E}")

    return result.success, elapsed, out_len, quality


# ══════════════════════════════════════════════════════════════════
# SCENARIO 3: Multi-Agent Market Analysis
# Tests: multi-agent coordination, parallel execution, synthesis
# ══════════════════════════════════════════════════════════════════
async def scenario_3_multi_agent():
    from Jotty.core.orchestration.v2.swarm_manager import SwarmManager
    from Jotty.core.agents.auto_agent import AutoAgent
    from Jotty.core.foundation.agent_config import AgentConfig

    print(f"\n{B}SCENARIO 3: Multi-Agent AI Market Analysis{E}")
    print(f"{D}  Goal: 3 specialized agents produce coordinated analysis{E}")

    # Wrap AutoAgent instances in AgentConfig (SwarmManager expects AgentConfig list)
    agents = [
        AgentConfig(
            name="TechAnalyst",
            agent=AutoAgent(name="TechAnalyst", system_prompt="You are a technical analyst specializing in AI/ML technology trends. Focus on architectures, benchmarks, and capabilities."),
        ),
        AgentConfig(
            name="MarketAnalyst",
            agent=AutoAgent(name="MarketAnalyst", system_prompt="You are a market analyst specializing in AI industry dynamics. Focus on market size, growth rates, key players, and competitive landscape."),
        ),
        AgentConfig(
            name="StrategyAdvisor",
            agent=AutoAgent(name="StrategyAdvisor", system_prompt="You are a strategy advisor. Synthesize technical and market insights into actionable business recommendations."),
        ),
    ]

    sm = SwarmManager(
        agents=agents,
        enable_lotus=False,
        enable_zero_config=False,
    )
    trail = []
    t0 = time.time()

    result = await sm.run(
        "Analyze the current state of the AI agent framework market in 2026. "
        "Compare LangChain, CrewAI, AutoGen, and custom frameworks. "
        "Provide technology assessment, market positioning, and strategic recommendations.",
        skip_autonomous_setup=True,
        ensemble=False,
        status_callback=lambda s, d: trail.append((time.time() - t0, s, d)),
    )

    elapsed = time.time() - t0
    output = str(result.output) if result.output else ""
    out_len = len(output)

    has_frameworks = sum(1 for kw in ['langchain', 'crewai', 'autogen'] if kw in output.lower()) >= 2
    has_analysis = any(kw in output.lower() for kw in ['market', 'strategy', 'recommendation', 'competitive'])
    has_depth = out_len > 3000
    multi_agent_used = any('agents' in str(d) for _, s, d in trail)

    quality = sum([has_frameworks, has_analysis, has_depth, multi_agent_used]) / 4

    print(f"  {'PASS' if result.success else 'FAIL'} | {elapsed:.0f}s | {out_len:,} chars | quality={quality:.0%}")
    print(f"  Frameworks={has_frameworks} Analysis={has_analysis} Depth={has_depth} MultiAgent={multi_agent_used}")
    for ts, s, d in trail[-5:]:
        print(f"  {D}  [{ts:.1f}s] {s}: {d[:80]}{E}")

    return result.success, elapsed, out_len, quality


# ══════════════════════════════════════════════════════════════════
# SCENARIO 4: Code + Execute (Stock Simulation)
# Tests: code gen + file write + execution pipeline
# ══════════════════════════════════════════════════════════════════
async def scenario_4_code_execute():
    from Jotty.core.orchestration.v2.swarm_manager import SwarmManager

    print(f"\n{B}SCENARIO 4: Stock Monte Carlo Simulation + Execution{E}")
    print(f"{D}  Goal: Generate Python script, save as .py, and execute{E}")

    sm = SwarmManager(enable_lotus=False, enable_zero_config=False)
    trail = []
    t0 = time.time()

    result = await sm.run(
        "Write a Python script that performs a Monte Carlo simulation for stock "
        "price prediction. Use geometric Brownian motion with parameters: "
        "S0=100, mu=0.08, sigma=0.2, T=1 year, 1000 simulations. "
        "Calculate and print: mean final price, 5th/95th percentile, "
        "probability of profit. Save the script as monte_carlo.py and run it.",
        skip_autonomous_setup=True,
        ensemble=False,
        status_callback=lambda s, d: trail.append((time.time() - t0, s, d)),
    )

    elapsed = time.time() - t0
    output = str(result.output) if result.output else ""
    out_len = len(output)

    has_monte_carlo = 'monte carlo' in output.lower() or 'simulation' in output.lower()
    has_numbers = any(c.isdigit() for c in output) and ('$' in output or 'price' in output.lower() or '%' in output)
    has_code = 'numpy' in output.lower() or 'np.' in output or 'import' in output
    saved_py = any('.py' in str(d) for _, s, d in trail)

    quality = sum([has_monte_carlo, has_numbers, has_code, saved_py]) / 4

    print(f"  {'PASS' if result.success else 'FAIL'} | {elapsed:.0f}s | {out_len:,} chars | quality={quality:.0%}")
    print(f"  MonteCarlo={has_monte_carlo} Numbers={has_numbers} Code={has_code} SavedPy={saved_py}")
    for ts, s, d in trail[-5:]:
        print(f"  {D}  [{ts:.1f}s] {s}: {d[:80]}{E}")

    return result.success, elapsed, out_len, quality


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
async def main():
    print(f"\n{'='*70}")
    print(f"{B}  JOTTY V2 — COMPLEX USE CASE EVALUATION (Post-Fix){E}")
    print(f"{D}  Testing 4 complex real-world scenarios with all 6 fixes applied{E}")
    print(f"{D}  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}{E}")
    print(f"{'='*70}")

    scenarios = [
        ("FastAPI REST API Code Gen",           scenario_1_codegen),
        ("Architecture Design (no web)",        scenario_2_architecture),
        ("Multi-Agent Market Analysis",         scenario_3_multi_agent),
        ("Stock Simulation + Execution",        scenario_4_code_execute),
    ]

    results = []
    for name, fn in scenarios:
        try:
            r = await asyncio.wait_for(fn(), timeout=300)
            results.append((name, *r))
        except asyncio.TimeoutError:
            print(f"  {R}TIMEOUT{E} (>300s)")
            results.append((name, False, 300.0, 0, 0.0))
        except Exception as e:
            print(f"  {R}ERROR: {e}{E}")
            results.append((name, False, 0.0, 0, 0.0))

    # ── SUMMARY ──
    print(f"\n{'='*70}")
    print(f"{B}  SUMMARY{E}")
    print(f"  {'Scenario':<40} {'Status':>6} {'Time':>7} {'Output':>8} {'Quality':>8}")
    print(f"  {'─'*40} {'─'*6} {'─'*7} {'─'*8} {'─'*8}")

    total_pass = 0
    total_time = 0
    total_quality = 0

    for name, success, elapsed, out_len, quality in results:
        st = f"{G}PASS{E}" if success else f"{R}FAIL{E}"
        print(f"  {name:<40} {st} {elapsed:>6.0f}s {out_len:>7,} {quality:>7.0%}")
        total_pass += int(success)
        total_time += elapsed
        total_quality += quality

    avg_quality = total_quality / len(results) if results else 0

    print(f"\n  {B}Result: {total_pass}/{len(results)} passed | "
          f"Total: {total_time:.0f}s | Avg Quality: {avg_quality:.0%}{E}")

    # Grade
    if total_pass == 4 and avg_quality >= 0.7:
        grade = f"{G}A — Production-worthy for complex tasks{E}"
    elif total_pass >= 3 and avg_quality >= 0.5:
        grade = f"{G}B — Solid, minor gaps remain{E}"
    elif total_pass >= 2:
        grade = f"{Y}C — Partially works, needs improvement{E}"
    else:
        grade = f"{R}D — Significant issues{E}"

    print(f"\n  {B}Grade: {grade}{E}\n")
    return total_pass, len(results)


if __name__ == '__main__':
    try:
        passed, total = asyncio.run(main())
        sys.exit(0 if passed >= 2 else 1)
    except KeyboardInterrupt:
        print(f"\n{Y}Interrupted{E}")
        sys.exit(130)

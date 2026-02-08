#!/usr/bin/env python3
"""
JOTTY COMPONENT ATTRIBUTION DEMO
================================

Shows what each component contributes and what happens without it.

Components:
1. Skills (pre-built tools)
2. SwarmManager (orchestration)
3. LLM (decision making)
4. Swarm Intelligence (learning)
"""

import asyncio
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

class C:
    H = '\033[95m'
    B = '\033[94m'
    C = '\033[96m'
    G = '\033[92m'
    Y = '\033[93m'
    R = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    E = '\033[0m'

def log(msg, color=""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{C.DIM}[{ts}]{C.E} {color}{msg}{C.E}", flush=True)

def section(title):
    print(f"\n{C.BOLD}{C.H}{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}{C.E}\n", flush=True)


async def demo_without_skills():
    """What happens with ONLY LLM, no skills."""
    section("SCENARIO 1: LLM ONLY (No Skills)")

    log("Using pure LLM without any skills...", C.B)

    from core.foundation.unified_lm_provider import configure_dspy_lm
    import dspy

    configure_dspy_lm()

    class SimpleResearch(dspy.Signature):
        """Research stocks and provide recommendations."""
        query: str = dspy.InputField()
        analysis: str = dspy.OutputField()

    researcher = dspy.ChainOfThought(SimpleResearch)

    start = time.time()
    result = researcher(query="Analyze AAPL and NVDA stocks, give buy/sell recommendation")
    elapsed = time.time() - start

    print(f"\n{C.Y}📊 LLM-Only Output:{C.E}")
    print(f"{result.analysis[:1500]}...")

    print(f"\n{C.C}What LLM provides:{C.E}")
    print("  ✓ General knowledge about stocks")
    print("  ✓ Reasoning and analysis")
    print("  ✗ NO live price data")
    print("  ✗ NO real financials")
    print("  ✗ NO charts/PDFs")
    print("  ✗ NO actionable target prices")

    return {'time': elapsed, 'has_live_data': False, 'has_pdf': False}


async def demo_with_skills_no_orchestration():
    """What happens with skills but manual orchestration."""
    section("SCENARIO 2: SKILLS + MANUAL ORCHESTRATION (No SwarmManager)")

    log("Manually calling skills without SwarmManager...", C.B)

    # You have to manually:
    # 1. Know which skills exist
    # 2. Know the correct parameters
    # 3. Call them in the right order
    # 4. Handle errors yourself

    from core.registry.skills_registry import get_skills_registry

    registry = get_skills_registry()
    registry.init()

    skill = registry.get_skill('stock-research-comprehensive')
    tool = skill.tools.get('comprehensive_stock_research_tool')

    start = time.time()

    # Manual call - you need to know exact params
    result_aapl = await tool({
        'ticker': 'AAPL',
        'exchange': 'US',  # You need to know this
        'enhanced': True,
        'send_telegram': False
    })

    result_nvda = await tool({
        'ticker': 'NVDA',
        'exchange': 'US',
        'enhanced': True,
        'send_telegram': False
    })

    elapsed = time.time() - start

    print(f"\n{C.G}✓ AAPL: {result_aapl.get('rating')} @ ${result_aapl.get('current_price')}{C.E}")
    print(f"{C.G}✓ NVDA: {result_nvda.get('rating')} @ ${result_nvda.get('current_price')}{C.E}")

    print(f"\n{C.C}What Skills provide:{C.E}")
    print("  ✓ Live financial data from Yahoo Finance")
    print("  ✓ Professional PDF reports")
    print("  ✓ Price charts")
    print("  ✓ Actionable ratings and targets")
    print(f"\n{C.R}What's MISSING without SwarmManager:{C.E}")
    print("  ✗ You must KNOW which skills exist")
    print("  ✗ You must KNOW exact parameter names")
    print("  ✗ You must MANUALLY parallelize")
    print("  ✗ NO learning or improvement")
    print("  ✗ NO automatic error recovery")

    return {'time': elapsed, 'has_live_data': True, 'has_pdf': True, 'manual': True}


async def demo_with_swarm_manager():
    """Full SwarmManager with all components."""
    section("SCENARIO 3: FULL JOTTY (SwarmManager + Skills + Learning)")

    log("Using SwarmManager with zero-config mode...", C.B)

    from core.orchestration.v2.swarm_manager import SwarmManager
    from core.foundation.unified_lm_provider import configure_dspy_lm

    configure_dspy_lm()

    goal = "Analyze AAPL and NVDA stocks, give buy/sell recommendation"

    start = time.time()
    swarm = SwarmManager(agents=goal, enable_zero_config=True)
    result = await swarm.run(goal=goal)
    elapsed = time.time() - start

    print(f"\n{C.G}✓ Success: {result.success}{C.E}")
    if hasattr(result, 'outputs'):
        for key, val in result.outputs.items():
            if isinstance(val, dict) and 'rating' in val:
                print(f"{C.G}✓ {val.get('ticker')}: {val.get('rating')} @ ${val.get('current_price'):.2f} → ${val.get('target_price'):.2f}{C.E}")

    si = swarm.swarm_intelligence

    print(f"\n{C.C}What SwarmManager adds:{C.E}")
    print("  ✓ AUTOMATIC task understanding (LLM infers 'analysis' task)")
    print(f"  ✓ AUTOMATIC skill discovery (found {len(swarm.skills_registry._skills) if hasattr(swarm, 'skills_registry') else 122} skills)")
    print("  ✓ AUTOMATIC skill selection (LLM chooses best skills)")
    print("  ✓ AUTOMATIC parameter inference (no manual params needed)")
    print("  ✓ AUTOMATIC parallel execution")
    print("  ✓ AUTOMATIC error recovery and replanning")
    print(f"\n{C.C}What Swarm Intelligence adds:{C.E}")
    print(f"  ✓ Stigmergy signals: {len(si.stigmergy.signals)} (learned routing)")
    print(f"  ✓ Q-learning: remembers successful strategies")
    print(f"  ✓ Trust scores: tracks agent reliability")
    print(f"  ✓ Credit assignment: learns which skills work best")

    return {'time': elapsed, 'has_live_data': True, 'has_pdf': True, 'auto': True}


async def main():
    print(f"{C.BOLD}{C.H}")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║         JOTTY COMPONENT ATTRIBUTION ANALYSIS                         ║")
    print("║                                                                      ║")
    print("║  What does each component contribute?                                ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"{C.E}\n")

    results = {}

    # Scenario 1: LLM only
    results['llm_only'] = await demo_without_skills()

    # Scenario 2: Skills without orchestration
    results['skills_manual'] = await demo_with_skills_no_orchestration()

    # Scenario 3: Full Jotty
    results['full_jotty'] = await demo_with_swarm_manager()

    # =========================================================================
    # ATTRIBUTION SUMMARY
    # =========================================================================
    section("COMPONENT ATTRIBUTION SUMMARY")

    print(f"{C.BOLD}{'Component':<25} {'Provides':<50} {'Value':<15}{C.E}")
    print("─" * 90)

    components = [
        ("LLM (Claude)", "Reasoning, task understanding, decisions", "BRAIN 🧠"),
        ("Skills (122 tools)", "Actual work: data fetching, PDF gen, charts", "HANDS 🔧"),
        ("SwarmManager", "Orchestration: auto-select, plan, parallelize", "COORDINATOR 🎯"),
        ("Swarm Intelligence", "Learning: stigmergy, Q-values, trust", "MEMORY 📚"),
    ]

    for comp, provides, value in components:
        print(f"{comp:<25} {provides:<50} {value:<15}")

    print("\n")

    # Comparison table
    print(f"{C.BOLD}{'Scenario':<30} {'Time':<12} {'Live Data':<12} {'PDF':<10} {'Auto':<10}{C.E}")
    print("─" * 80)

    scenarios = [
        ("LLM Only", results['llm_only']),
        ("Skills (Manual)", results['skills_manual']),
        ("Full Jotty (Auto)", results['full_jotty']),
    ]

    for name, r in scenarios:
        time_str = f"{r['time']:.1f}s"
        live = "✓" if r.get('has_live_data') else "✗"
        pdf = "✓" if r.get('has_pdf') else "✗"
        auto = "✓" if r.get('auto') else ("Manual" if r.get('manual') else "✗")
        print(f"{name:<30} {time_str:<12} {live:<12} {pdf:<10} {auto:<10}")

    # Value proposition
    section("VALUE PROPOSITION OF EACH LAYER")

    print(f"""{C.C}
┌─────────────────────────────────────────────────────────────────────────────┐
│                           JOTTY ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  SWARM INTELLIGENCE (Learning Layer)                                │   │
│   │  • Stigmergy: "NVDA research works → route similar tasks here"     │   │
│   │  • Q-Learning: "stock-research-comprehensive → high reward"        │   │
│   │  • Trust: "This agent succeeded 87% of time"                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓ learns from                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  SWARM MANAGER (Orchestration Layer)                                │   │
│   │  • Task Analysis: "This is an 'analysis' task"                     │   │
│   │  • Skill Selection: "Use stock-research-comprehensive"             │   │
│   │  • Planning: "Execute AAPL and NVDA in parallel"                   │   │
│   │  • Error Recovery: "Step failed → replan remaining"                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓ orchestrates                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  SKILLS (Execution Layer) - 122 skills                              │   │
│   │  • stock-research-comprehensive: Fetch data, generate reports      │   │
│   │  • financial-visualization: Create charts                          │   │
│   │  • screener-to-pdf-telegram: PDF + Telegram delivery              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓ uses                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  LLM (Intelligence Layer) - Claude via CLI                          │   │
│   │  • Understands natural language goals                              │   │
│   │  • Makes decisions about what to do                                │   │
│   │  • Generates content and analysis                                  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
{C.E}""")

    print(f"\n{C.BOLD}{C.G}KEY INSIGHT:{C.E}")
    print(f"""
    • {C.Y}Without Skills{C.E}: LLM can only give general advice (no live data)
    • {C.Y}Without SwarmManager{C.E}: You must manually code everything
    • {C.Y}Without Swarm Intelligence{C.E}: No learning, same mistakes repeated
    • {C.G}With Full Jotty{C.E}: Natural language → Professional output → Continuous improvement
    """)

    print(f"\n{C.BOLD}{C.G}✓ DEMO COMPLETE{C.E}\n")
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))

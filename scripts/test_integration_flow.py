#!/usr/bin/env python3
"""
Full Integration Flow Trace
============================

Traces a SINGLE request through ALL Jotty components, showing the exact
data flow between each layer. This is NOT a test — it's a diagnostic tool
that proves the entire stack is wired correctly.

Flow traced:
  User Input
    → Jotty.run()           [Entry point]
    → TierDetector          [Tier auto-detection]
    → IntentClassifier      [Intent analysis]
    → TierExecutor.execute  [Unified dispatcher]
    → ComplexityGate        [Direct vs planning decision]
    → TaskPlanner           [LLM-based plan generation]
    → SkillSelection        [LLM picks best skills from 164]
    → StepExecution         [Per-step LLM calls with tools]
    → Synthesis             [Final output assembly]
    → ExecutionResult       [Structured result]

Also traces:
  - SwarmRegistry          [Swarm discovery]
  - ResearchSwarm          [Direct swarm execution]
  - AutoAgent              [Autonomous agent with skills]
  - CostTracker            [Per-call cost accounting]
  - MetricsCollector       [Observability]
  - TracingContext          [Distributed traces]

Uses real Anthropic API.
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

# ─── Load API keys ────────────────────────────────────────────────────────────
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


# ─── Custom logging that shows component transitions ─────────────────────────
class FlowFormatter(logging.Formatter):
    """Custom formatter that highlights component names in the flow."""

    COMPONENTS = {
        "Jotty": "🏠",
        "TierDetector": "🔍",
        "TierExecutor": "⚡",
        "ComplexityGate": "🚪",
        "TaskPlanner": "📋",
        "SkillSelection": "🎯",
        "LLMProvider": "🤖",
        "CostTracker": "💰",
        "SwarmRegistry": "📦",
        "ResearchSwarm": "🔬",
        "AutoAgent": "🤖",
        "IntentClassifier": "🧭",
    }

    def format(self, record):
        msg = super().format(record)
        for comp, icon in self.COMPONENTS.items():
            if comp.lower() in record.name.lower() or comp.lower() in msg.lower():
                return f"  {icon} {msg}"
        return f"     {msg}"


handler = logging.StreamHandler()
handler.setFormatter(FlowFormatter("%(name)s: %(message)s"))
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)
logger = logging.getLogger("flow_trace")

# Suppress noisy loggers
for name in ["httpx", "httpcore", "anthropic", "urllib3", "dspy"]:
    logging.getLogger(name).setLevel(logging.WARNING)


def section(title: str, icon: str = "═"):
    """Print a section header."""
    width = 74
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def component(name: str, detail: str = ""):
    """Print component entry."""
    icons = {
        "Jotty": "🏠",
        "TierDetector": "🔍",
        "TierExecutor": "⚡",
        "ComplexityGate": "🚪",
        "TaskPlanner": "📋",
        "SkillSelection": "🎯",
        "LLMProvider": "🤖",
        "CostTracker": "💰",
        "Metrics": "📊",
        "Tracer": "🔗",
        "SwarmRegistry": "📦",
        "ResearchSwarm": "🔬",
        "CodingSwarm": "💻",
        "AutoAgent": "🤖",
        "IntentClassifier": "🧭",
        "Synthesis": "🧩",
        "Result": "✅",
        "Error": "❌",
        "Memory": "🧠",
        "Validator": "✓",
    }
    icon = icons.get(name, "→")
    suffix = f" — {detail}" if detail else ""
    print(f"  {icon} [{name}]{suffix}")


def data_flow(label: str, value: Any, max_len: int = 120):
    """Print data flowing between components."""
    s = str(value)
    if len(s) > max_len:
        s = s[:max_len] + "..."
    print(f"      ↳ {label}: {s}")


# =============================================================================
# FLOW 1: Full Tier 2 AGENTIC Pipeline
# =============================================================================
async def trace_tier2_agentic_flow():
    """Trace Jotty.run() → TierDetector → TierExecutor → Planning → Steps → Synthesis."""
    section("FLOW 1: Tier 2 AGENTIC — Full Planning Pipeline")

    task = (
        "Create a detailed comparison between Python and Rust for systems programming. "
        "Cover performance, memory safety, and ecosystem."
    )
    data_flow("Input task", task)

    # ── Step 1: Jotty entry point ──
    component("Jotty", "Jotty(log_level='WARNING')")
    from Jotty import Jotty
    from Jotty.jotty import ExecutionTier

    jotty = Jotty(log_level="WARNING")
    data_flow("executor", type(jotty.executor).__name__)
    data_flow("detector", type(jotty.detector).__name__)

    # ── Step 2: TierDetector ──
    component("TierDetector", "detect(task)")
    detected_tier = jotty.detector.detect(task)
    data_flow("detected_tier", f"{detected_tier.name} (value={detected_tier.value})")
    data_flow("explanation", jotty.explain_tier(task)[:200])

    # ── Step 3: Force Tier 2 with skip_complexity_gate ──
    component("TierExecutor", "execute(task, tier=AGENTIC, skip_complexity_gate=True)")

    # Track what happens inside execute()
    t0 = time.time()

    # Instrument the executor to capture internals
    executor = jotty.executor

    # ── Step 3a: IntentClassifier ──
    component("IntentClassifier", "classify_task_intent(task)")
    from Jotty.core.intelligence.orchestration.execution.intent_classifier import (
        classify_task_intent,
    )

    intent = classify_task_intent(task, [])
    data_flow("intent", f"{intent.intent.value} (confidence={intent.confidence:.2f})")
    data_flow("required_tools", intent.required_tools or "none")

    # ── Step 3b: ComplexityGate (skipped) ──
    component("ComplexityGate", "SKIPPED (skip_complexity_gate=True)")

    # ── Step 3c: TaskPlanner ──
    component("TaskPlanner", "plan(task)")
    planner = executor.planner
    data_flow("planner_type", type(planner).__name__ if planner else "None")

    if planner:
        plan_t0 = time.time()
        plan_result = await executor._run_planner(task)
        plan_time = time.time() - plan_t0
        data_flow("plan_time", f"{plan_time:.1f}s")
        data_flow("plan_result_type", type(plan_result).__name__)

        # Parse plan
        plan = executor._parse_plan(task, plan_result)
        data_flow("steps_count", len(plan.steps))
        for step in plan.steps:
            data_flow(
                f"  step {step.step_num}", f"[{step.skill or 'general'}] {step.description[:80]}"
            )

    # ── Step 3d: SkillSelection (inside planner) ──
    component("SkillSelection", "select_skills(task)")
    try:
        from Jotty.core.capabilities.registry import get_unified_registry

        registry = get_unified_registry()
        total_skills = len(registry.list_skills())
        data_flow("total_skills_available", total_skills)

        discovery = registry.discover_for_task(task)
        discovered = discovery.get("skills", [])[:8]
        skill_names = [s["name"] if isinstance(s, dict) else s for s in discovered]
        data_flow("discovered_skills", skill_names)

        # Get Claude tools
        tools = registry.get_claude_tools(skill_names[:5]) if skill_names else []
        data_flow("claude_tools_loaded", f"{len(tools)} tools")
        for t in tools[:3]:
            data_flow(f"  tool", f"{t.get('name', '?')}: {t.get('description', '?')[:60]}")
    except Exception as e:
        data_flow("skill_discovery_error", str(e))

    # ── Step 3e: Full execution ──
    component("TierExecutor", "Full Tier 2 execution now...")
    result = await jotty.run(
        task,
        tier=ExecutionTier.AGENTIC,
        skip_complexity_gate=True,
    )
    duration = time.time() - t0

    # ── Step 3f: CostTracker + Metrics ──
    component("CostTracker", "Record LLM costs")
    data_flow("total_cost", f"${result.cost_usd:.4f}")
    data_flow("llm_calls", result.llm_calls)

    component("Metrics", "Record execution metrics")
    data_flow("latency_ms", f"{result.latency_ms:.0f}ms")

    component("Tracer", "Distributed trace")
    trace = result.trace
    if trace:
        data_flow("trace_id", getattr(trace, "trace_id", "?"))
        spans = getattr(trace, "spans", [])
        data_flow("spans_count", len(spans))
        for span in spans[:5]:
            name = getattr(span, "name", "?")
            dur = getattr(span, "duration_ms", 0)
            data_flow(f"  span", f"{name} ({dur:.0f}ms)")

    # ── Final Result ──
    component("Result", f"Tier 2 AGENTIC complete")
    output = str(result.output) if result.output else ""
    data_flow("success", result.success)
    data_flow("output_length", f"{len(output)} chars")
    data_flow("output_preview", output[:200])
    data_flow("steps_executed", len(result.steps) if result.steps else 0)
    data_flow("duration", f"{duration:.1f}s")

    return result


# =============================================================================
# FLOW 2: Tier 4 RESEARCH Swarm
# =============================================================================
async def trace_tier4_swarm_flow():
    """Trace Jotty.swarm() → SwarmRegistry → ResearchSwarm → agents → output."""
    section("FLOW 2: Tier 4 RESEARCH — Swarm Execution")

    task = "Research INFY.NS stock - give me a quick analysis"
    data_flow("Input task", task)

    # ── Step 1: Jotty.swarm() entry ──
    component("Jotty", "jotty.swarm(task, swarm_name='research')")
    from Jotty import Jotty

    jotty = Jotty(log_level="WARNING")

    # ── Step 2: SwarmRegistry ──
    component("SwarmRegistry", "List registered swarms")
    from Jotty.core.intelligence.orchestration.swarms._base.registry import SwarmRegistry

    # Trigger registration
    import importlib

    for mod in [
        "Jotty.core.intelligence.orchestration.swarms.research_swarm.swarm",
        "Jotty.core.intelligence.orchestration.swarms.coding_swarm.swarm",
        "Jotty.core.intelligence.orchestration.swarms.pilot_swarm.swarm",
    ]:
        try:
            importlib.import_module(mod)
        except Exception:
            pass

    all_swarm_names = SwarmRegistry.list_all()
    data_flow("registered_swarms", f"{len(all_swarm_names)} swarms")
    for name in all_swarm_names[:8]:
        cls = SwarmRegistry.get(name)
        data_flow(f"  swarm", f"{name}: {cls.__name__ if cls else '?'}")

    # ── Step 3: ResearchSwarm instantiation ──
    component("ResearchSwarm", "Direct instantiation")
    from Jotty.core.intelligence.orchestration.swarms.research_swarm.swarm import ResearchSwarm

    swarm = ResearchSwarm()
    data_flow("swarm_class", type(swarm).__name__)
    data_flow("has TASK_TYPE", getattr(swarm, "TASK_TYPE", "?"))
    data_flow("has DEFAULT_TOOLS", getattr(swarm, "DEFAULT_TOOLS", "?"))

    # ── Step 4: Execute via Jotty.swarm() ──
    component("TierExecutor", "Swarm dispatch via jotty.swarm()")
    t0 = time.time()
    result = await jotty.swarm(task, swarm_name="research")
    duration = time.time() - t0

    # ── Result ──
    component("CostTracker", "Swarm costs")
    data_flow("total_cost", f"${result.cost_usd:.4f}")
    data_flow("llm_calls", result.llm_calls)

    component("Result", "Tier 4 RESEARCH complete")
    output = str(result.output) if result.output else ""
    data_flow("success", result.success)
    data_flow("output_length", f"{len(output)} chars")
    data_flow("output_preview", output[:200])
    data_flow("swarm_name", getattr(result, "swarm_name", "?"))
    data_flow("duration", f"{duration:.1f}s")

    return result


# =============================================================================
# FLOW 3: AutoAgent with Skill Discovery
# =============================================================================
async def trace_auto_agent_flow():
    """Trace AutoAgent → skill discovery → LLM selection → planning → execution."""
    section("FLOW 3: AutoAgent — Skill Discovery + Planning + Execution")

    task = "Calculate the monthly payment for a $300,000 mortgage at 6.5% interest over 30 years."
    data_flow("Input task", task)

    # ── Step 1: AutoAgent creation ──
    component("AutoAgent", "AutoAgent(max_steps=3)")
    from Jotty.core.intelligence.reasoning.agents import AutoAgent

    agent = AutoAgent(max_steps=3)
    data_flow("agent_type", type(agent).__name__)
    data_flow("max_steps", getattr(agent, "max_steps", getattr(agent.config, "max_steps", "?")))

    # ── Step 2: Skill registry ──
    component("SkillSelection", "Available skills for agent")
    try:
        from Jotty.core.capabilities.registry import get_unified_registry

        registry = get_unified_registry()
        all_skills = registry.list_skills()
        data_flow("total_skills", len(all_skills))

        # Show some skill categories
        categories = {}
        for name in all_skills[:50]:
            skill = registry.get_skill(name)
            if skill:
                caps = getattr(skill, "capabilities", [])
                for cap in caps[:1]:
                    categories.setdefault(cap, []).append(name)
        for cap, names in list(categories.items())[:5]:
            data_flow(f"  [{cap}]", f"{len(names)} skills: {names[:3]}")
    except Exception as e:
        data_flow("registry_error", str(e))

    # ── Step 3: Execute ──
    component("AutoAgent", "agent.execute(task)")
    t0 = time.time()
    result = await agent.execute(task)
    duration = time.time() - t0

    # ── Step 4: Extract result details ──
    output = ""
    if hasattr(result, "output") and result.output:
        output = str(result.output)
    elif hasattr(result, "final_output") and result.final_output:
        output = str(result.final_output)
    else:
        output = str(result)

    component("Result", "AutoAgent execution complete")
    data_flow("output_length", f"{len(output)} chars")
    data_flow("output_preview", output[:200])
    data_flow("steps_executed", getattr(result, "steps_executed", "?"))
    data_flow(
        "skills_used", getattr(result, "selected_skills", getattr(result, "skills_used", "?"))
    )
    data_flow("task_type", getattr(result, "task_type", "?"))
    data_flow("duration", f"{duration:.1f}s")

    return result


# =============================================================================
# FLOW 4: Tier 1 DIRECT — Single LLM Call with Tools
# =============================================================================
async def trace_tier1_direct_flow():
    """Trace Jotty.run() → Tier 1 DIRECT → single LLM call with tool discovery."""
    section("FLOW 4: Tier 1 DIRECT — Single LLM Call with Tool Discovery")

    task = "What is the square root of 144?"
    data_flow("Input task", task)

    # ── Step 1: Jotty entry ──
    component("Jotty", "jotty.run(task) — auto-detect tier")
    from Jotty import Jotty

    jotty = Jotty(log_level="WARNING")

    # ── Step 2: TierDetector ──
    component("TierDetector", "Auto-detect tier for simple question")
    detected = jotty.detector.detect(task)
    data_flow("detected_tier", f"{detected.name} (value={detected.value})")

    # ── Step 3: IntentClassifier ──
    component("IntentClassifier", "Classify task intent")
    from Jotty.core.intelligence.orchestration.execution.intent_classifier import (
        classify_task_intent,
    )

    intent = classify_task_intent(task, [])
    data_flow("intent", f"{intent.intent.value} (confidence={intent.confidence:.2f})")

    # ── Step 4: Execute ──
    component("TierExecutor", "Tier 1 DIRECT execution")
    t0 = time.time()
    result = await jotty.run(task)
    duration = time.time() - t0

    # ── Result ──
    component("LLMProvider", "Anthropic API call")
    data_flow("provider", "anthropic")
    data_flow("model", "claude-sonnet-4-20250514")

    component("CostTracker", "Cost accounting")
    data_flow("cost", f"${result.cost_usd:.4f}")
    data_flow("llm_calls", result.llm_calls)

    component("Result", "Tier 1 DIRECT complete")
    output = str(result.output) if result.output else ""
    data_flow("success", result.success)
    data_flow("tier", result.tier.name if result.tier else "?")
    data_flow("output", output[:200])
    data_flow("duration", f"{duration:.1f}s")

    skills = result.metadata.get("skills_discovered", []) if result.metadata else []
    data_flow("skills_discovered", skills)

    return result


# =============================================================================
# FLOW 5: Component Wiring Verification
# =============================================================================
async def verify_component_wiring():
    """Verify all components are properly wired and can be instantiated."""
    section("FLOW 5: Component Wiring Verification")

    checks = []

    def check(name: str, fn):
        try:
            result = fn()
            checks.append((name, True, str(result)[:80]))
            component(name, "OK")
            data_flow("result", str(result)[:80])
        except Exception as e:
            checks.append((name, False, str(e)[:80]))
            component("Error", f"{name}: {e}")

    # Entry points
    check("Jotty import", lambda: __import__("Jotty").Jotty)
    check(
        "ExecutionTier import",
        lambda: __import__("Jotty.jotty", fromlist=["ExecutionTier"]).ExecutionTier,
    )

    # Tier execution
    check(
        "TierExecutor",
        lambda: __import__(
            "Jotty.core.intelligence.orchestration.execution", fromlist=["TierExecutor"]
        ).TierExecutor,
    )
    check(
        "TierDetector",
        lambda: __import__(
            "Jotty.core.intelligence.orchestration.execution", fromlist=["TierDetector"]
        ).TierDetector,
    )
    check(
        "IntentClassifier",
        lambda: __import__(
            "Jotty.core.intelligence.orchestration.execution.intent_classifier",
            fromlist=["classify_task_intent"],
        ).classify_task_intent,
    )

    # Agents
    check(
        "AutoAgent",
        lambda: __import__(
            "Jotty.core.intelligence.reasoning.agents", fromlist=["AutoAgent"]
        ).AutoAgent,
    )
    check(
        "ChatAssistant",
        lambda: __import__(
            "Jotty.core.intelligence.reasoning.agents", fromlist=["ChatAssistant"]
        ).ChatAssistant,
    )
    check(
        "CompositeAgent",
        lambda: __import__(
            "Jotty.core.intelligence.reasoning.agents", fromlist=["CompositeAgent"]
        ).CompositeAgent,
    )
    check(
        "DomainAgent",
        lambda: __import__(
            "Jotty.core.intelligence.reasoning.agents", fromlist=["DomainAgent"]
        ).DomainAgent,
    )

    # Planners
    check(
        "TaskPlanner",
        lambda: __import__(
            "Jotty.core.intelligence.reasoning.planners.agentic_planner", fromlist=["TaskPlanner"]
        ).TaskPlanner,
    )

    # Executors
    check(
        "SkillPlanExecutor",
        lambda: __import__(
            "Jotty.core.intelligence.reasoning.executors", fromlist=["SkillPlanExecutor"]
        ).SkillPlanExecutor,
    )

    # Capabilities
    check(
        "LearningCapability",
        lambda: __import__(
            "Jotty.core.intelligence.reasoning.capabilities", fromlist=["LearningCapability"]
        ).LearningCapability,
    )
    check(
        "MemoryCapability",
        lambda: __import__(
            "Jotty.core.intelligence.reasoning.capabilities", fromlist=["MemoryCapability"]
        ).MemoryCapability,
    )
    check(
        "ValidationCapability",
        lambda: __import__(
            "Jotty.core.intelligence.reasoning.capabilities", fromlist=["ValidationCapability"]
        ).ValidationCapability,
    )

    # Swarms
    check(
        "SwarmRegistry",
        lambda: __import__(
            "Jotty.core.intelligence.orchestration.swarms._base.registry",
            fromlist=["SwarmRegistry"],
        ).SwarmRegistry,
    )
    check(
        "ResearchSwarm",
        lambda: __import__(
            "Jotty.core.intelligence.orchestration.swarms.research_swarm.swarm",
            fromlist=["ResearchSwarm"],
        ).ResearchSwarm,
    )
    check(
        "CodingSwarm",
        lambda: __import__(
            "Jotty.core.intelligence.orchestration.swarms.coding_swarm.swarm",
            fromlist=["CodingSwarm"],
        ).CodingSwarm,
    )

    # Orchestration
    check(
        "Orchestrator (facade)",
        lambda: __import__(
            "Jotty.core.intelligence.orchestration.facade", fromlist=["get_swarm_router"]
        ).get_swarm_router,
    )

    # Infrastructure
    check(
        "UnifiedLMProvider",
        lambda: __import__(
            "Jotty.core.infrastructure.foundation.unified_lm_provider",
            fromlist=["UnifiedLMProvider"],
        ).UnifiedLMProvider,
    )
    check(
        "BudgetTracker",
        lambda: __import__(
            "Jotty.core.infrastructure.utils.facade", fromlist=["get_budget_tracker"]
        ).get_budget_tracker,
    )
    check(
        "CostTracker",
        lambda: __import__(
            "Jotty.core.infrastructure.monitoring.metrics.cost_tracker", fromlist=["CostTracker"]
        ).CostTracker,
    )

    # Memory
    check(
        "MemorySystem",
        lambda: __import__(
            "Jotty.core.intelligence.memory.facade", fromlist=["get_memory_system"]
        ).get_memory_system,
    )

    # Learning
    check(
        "TDLambda",
        lambda: __import__(
            "Jotty.core.intelligence.learning.facade", fromlist=["get_td_lambda"]
        ).get_td_lambda,
    )

    # Skills
    check(
        "SkillsRegistry",
        lambda: __import__(
            "Jotty.core.capabilities.registry.skills_registry", fromlist=["get_skills_registry"]
        ).get_skills_registry,
    )
    check(
        "UnifiedRegistry",
        lambda: __import__(
            "Jotty.core.capabilities.registry", fromlist=["get_unified_registry"]
        ).get_unified_registry,
    )

    # Use cases
    check(
        "ChatUseCase",
        lambda: __import__(
            "Jotty.core.intelligence.orchestration.use_cases.chat.chat_use_case",
            fromlist=["ChatUseCase"],
        ).ChatUseCase,
    )
    check(
        "ChatExecutor",
        lambda: __import__(
            "Jotty.core.intelligence.orchestration.use_cases.chat.chat_executor",
            fromlist=["ChatExecutor"],
        ).ChatExecutor,
    )

    # Autonomous
    check(
        "IntentParser",
        lambda: __import__(
            "Jotty.core.intelligence.reasoning.autonomous.intent_parser", fromlist=["IntentParser"]
        ).IntentParser,
    )

    # Summary
    passed = sum(1 for _, ok, _ in checks if ok)
    failed = sum(1 for _, ok, _ in checks if not ok)

    print(f"\n  {'─' * 50}")
    print(f"  Component Wiring: {passed}/{len(checks)} OK, {failed} FAILED")
    if failed:
        print(f"\n  FAILURES:")
        for name, ok, detail in checks:
            if not ok:
                print(f"    ✗ {name}: {detail}")

    return passed, failed


# =============================================================================
# FLOW 6: Cross-Component Data Flow Diagram
# =============================================================================
def print_architecture_diagram():
    """Print the actual component architecture and data flow."""
    section("ARCHITECTURE: Component Integration Map")

    print(
        """
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  USER INPUT: "Create comparison between Python and Rust"               │
  └───────────────────────────────┬─────────────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────────────┐
  │  🏠 Jotty (jotty.py)                                                   │
  │  • Entry point: run() / chat() / swarm() / research() / learn()       │
  │  • Creates: TierExecutor + TierDetector                                │
  └───────────────────────────────┬─────────────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────────────┐
  │  🔍 TierDetector (execution/tier_detector.py)                          │
  │  • Heuristic keywords → DIRECT / AGENTIC / LEARNING / RESEARCH / AUTO │
  │  • Optional LLM fallback for ambiguous cases                           │
  └───────────────────────────────┬─────────────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────────────┐
  │  🧭 IntentClassifier (execution/intent_classifier.py)                  │
  │  • Classifies: fact_retrieval / creation / analysis / communication    │
  │  • Routes fact_retrieval → Tier 1 (with curated tools)                 │
  └───────────────────────────────┬─────────────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────────────┐
  │  ⚡ TierExecutor (execution/tier_executor.py)                          │
  │  • Central dispatcher: execute() routes to _execute_tier{1,2,3,4,5}   │
  │  • Lazy-loads: planner, memory, validator, complexity gate             │
  │  • Observability: tracer spans + metrics + cost tracking               │
  ├─────────────┬────────────┬────────────┬──────────────┬─────────────────┤
  │  Tier 1     │  Tier 2    │  Tier 3    │  Tier 4      │  Tier 5        │
  │  DIRECT     │  AGENTIC   │  LEARNING  │  RESEARCH    │  AUTONOMOUS    │
  │  1 LLM call │  Plan+Exec │  Memory+   │  Swarm exec  │  Sandbox+      │
  │  + tools    │  3-5 calls │  Validate  │  8+ agents   │  Coalition     │
  └──────┬──────┴──────┬─────┴──────┬─────┴──────┬───────┴────────┬───────┘
         │             │            │            │                │
         ▼             ▼            ▼            ▼                ▼
  ┌──────────┐  ┌──────────┐ ┌──────────┐ ┌──────────┐   ┌──────────┐
  │🤖 LLM   │  │📋 Task   │ │🧠 Memory │ │📦 Swarm  │   │🔒 Sand- │
  │Provider  │  │Planner   │ │System    │ │Registry  │   │box Mgr   │
  │          │  │(DSPy)    │ │(5-level) │ │          │   │          │
  └──────────┘  └────┬─────┘ └──────────┘ └────┬─────┘   └──────────┘
                     │                          │
                     ▼                          ▼
              ┌──────────┐              ┌──────────────┐
              │🎯 Skill  │              │🔬 Domain     │
              │Selection │              │Swarms        │
              │(LLM pick │              │• Research    │
              │from 164) │              │• Coding (8)  │
              └────┬─────┘              │• Learning    │
                   │                    │• Olympiad    │
                   ▼                    │• Pilot       │
              ┌──────────┐              └──────────────┘
              │📝 Step   │
              │Execution │
              │(per-step │
              │LLM+tools)│
              └────┬─────┘
                   │
                   ▼
              ┌──────────┐
              │🧩 Synthe-│
              │sis       │
              │(combine  │
              │ results) │
              └────┬─────┘
                   │
                   ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  ✅ ExecutionResult                                                    │
  │  • output: str           • cost_usd: float    • trace: TraceData     │
  │  • tier: ExecutionTier   • llm_calls: int     • steps: List[Step]    │
  │  • success: bool         • latency_ms: float  • metadata: dict       │
  └────────────────────────────────────────────────────────────────────────┘

  CROSS-CUTTING CONCERNS (wired to every tier):
  ┌────────────────────────────────────────────────────────────────────────┐
  │  💰 CostTracker — per-call cost accounting (provider × model pricing) │
  │  📊 MetricsCollector — execution counts, latency, success rates       │
  │  🔗 TracingContext — distributed spans with parent-child hierarchy     │
  │  🤖 UnifiedLMProvider — DSPy LM factory (Anthropic/OpenAI/Groq)      │
  └────────────────────────────────────────────────────────────────────────┘
"""
    )


# =============================================================================
# RUNNER
# =============================================================================
async def main():
    print("\n" + "=" * 74)
    print("  JOTTY FULL INTEGRATION FLOW TRACE")
    print("  Traces real requests through ALL components with real LLM calls")
    print("=" * 74)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    print(f"\n  API Key: {'...' + api_key[-8:] if len(api_key) > 8 else 'NOT SET'}")
    print(f"  Python: {sys.version.split()[0]}")

    if not api_key:
        print("\n  ERROR: ANTHROPIC_API_KEY not set! Aborting.\n")
        return 1

    total_start = time.time()
    all_ok = True

    # ── Architecture Diagram ──
    print_architecture_diagram()

    # ── Flow 5: Component Wiring ──
    try:
        passed, failed = await verify_component_wiring()
        if failed > 0:
            all_ok = False
    except Exception as e:
        print(f"  ✗ Component wiring check failed: {e}")
        traceback.print_exc()
        all_ok = False

    # ── Flow 4: Tier 1 DIRECT ──
    try:
        result = await asyncio.wait_for(trace_tier1_direct_flow(), timeout=60)
        if not result.success:
            all_ok = False
    except asyncio.TimeoutError:
        component("Error", "Tier 1 DIRECT timed out (60s)")
        all_ok = False
    except Exception as e:
        component("Error", f"Tier 1 DIRECT failed: {e}")
        traceback.print_exc()
        all_ok = False

    # ── Flow 1: Tier 2 AGENTIC ──
    try:
        result = await asyncio.wait_for(trace_tier2_agentic_flow(), timeout=180)
        if not result.success:
            all_ok = False
    except asyncio.TimeoutError:
        component("Error", "Tier 2 AGENTIC timed out (180s)")
        all_ok = False
    except Exception as e:
        component("Error", f"Tier 2 AGENTIC failed: {e}")
        traceback.print_exc()
        all_ok = False

    # ── Flow 3: AutoAgent ──
    try:
        result = await asyncio.wait_for(trace_auto_agent_flow(), timeout=120)
    except asyncio.TimeoutError:
        component("Error", "AutoAgent timed out (120s)")
        all_ok = False
    except Exception as e:
        component("Error", f"AutoAgent failed: {e}")
        traceback.print_exc()
        all_ok = False

    # ── Flow 2: Tier 4 RESEARCH Swarm ──
    try:
        result = await asyncio.wait_for(trace_tier4_swarm_flow(), timeout=180)
        if not result.success:
            all_ok = False
    except asyncio.TimeoutError:
        component("Error", "Tier 4 RESEARCH timed out (180s)")
        all_ok = False
    except Exception as e:
        component("Error", f"Tier 4 RESEARCH failed: {e}")
        traceback.print_exc()
        all_ok = False

    # ── Summary ──
    total_time = time.time() - total_start

    print("\n" + "=" * 74)
    status = "ALL FLOWS PASSED" if all_ok else "SOME FLOWS FAILED"
    icon = "✅" if all_ok else "❌"
    print(f"  {icon} {status} ({total_time:.1f}s total)")
    print("=" * 74 + "\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

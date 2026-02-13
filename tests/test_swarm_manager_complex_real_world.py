#!/usr/bin/env python3
"""
Orchestrator — Most Complicated Real-World Multi-Agent Use Cases
================================================================

Tests the full multi-agent pipeline with:
  - 4-agent relay: Research → Analyze → Write → Edit (sequential handoffs)
  - Fanout: parallel decomposition with 3 agents on distinct sub-goals
  - Real-world goals: research, comparison, structured output, optional file write
  - Learning loop: post_episode runs; effectiveness and TD(λ) are exercised

Run:
  pytest tests/test_swarm_manager_complex_real_world.py -v -s
  pytest tests/test_swarm_manager_complex_real_world.py -v -s -k relay
  python tests/test_swarm_manager_complex_real_world.py
"""

import asyncio
import os
import time
from pathlib import Path

# Load env from project root or tests dir
for env_file in [
    Path(__file__).resolve().parents[1] / ".env",
    Path(__file__).resolve().parents[1] / ".env.anthropic",
    Path(__file__).resolve().parents[0] / ".env",
]:
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if v and k not in os.environ:
                        os.environ[k] = v

import logging
import pytest

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger("swarm_complex_real")
logger.setLevel(logging.INFO)


def _make_four_agent_relay_swarm():
    """4 agents in relay order: Researcher → Analyst → Writer → Editor."""
    from Jotty.core.orchestration.swarm_manager import Orchestrator
    from Jotty.core.foundation.agent_config import AgentConfig
    from Jotty.core.agents.auto_agent import AutoAgent
    from Jotty.core.foundation.data_structures import SwarmConfig

    agents = [
        AgentConfig(
            name="Researcher",
            agent=AutoAgent(),
            capabilities=[
                "Research and list current AI code assistants (e.g. GitHub Copilot, Cursor, Codeium, Tabnine). "
                "Summarize each with: what it does, main features, and one pro/con. Use web search if needed."
            ],
        ),
        AgentConfig(
            name="Analyst",
            agent=AutoAgent(),
            capabilities=[
                "Compare the AI code assistants from the previous step on: pricing model, IDE support, and developer adoption. "
                "Identify the top 3 risks for teams adopting these tools. Output a structured comparison and risk list."
            ],
        ),
        AgentConfig(
            name="Writer",
            agent=AutoAgent(),
            capabilities=[
                "Turn the comparison and risk list from the previous agent into a 1-page executive summary in markdown. "
                "Include: title, 3 sections (Overview, Comparison, Top 3 Risks), and a short recommendation. Use clear headings."
            ],
        ),
        AgentConfig(
            name="Editor",
            agent=AutoAgent(),
            capabilities=[
                "Review the executive summary from the previous agent. Improve clarity and fix any inconsistencies. "
                "Output the final markdown. If possible, save it to a file named ai_assistants_brief.md."
            ],
        ),
    ]

    config = SwarmConfig()
    sm = Orchestrator(
        agents=agents,
        config=config,
        enable_zero_config=False,
        enable_lotus=False,
        max_concurrent_agents=2,
    )
    return sm


def _make_three_agent_fanout_swarm():
    """3 agents for parallel fanout: each gets a distinct sub-goal."""
    from Jotty.core.orchestration.swarm_manager import Orchestrator
    from Jotty.core.foundation.agent_config import AgentConfig
    from Jotty.core.agents.auto_agent import AutoAgent
    from Jotty.core.foundation.data_structures import SwarmConfig

    agents = [
        AgentConfig(
            name="PerformanceExpert",
            agent=AutoAgent(),
            capabilities=[
                "Focus only on performance: compare FastAPI, Flask, and Django on throughput, latency, and async support. "
                "Output 3–5 bullet points with concrete trade-offs."
            ],
        ),
        AgentConfig(
            name="DocsExpert",
            agent=AutoAgent(),
            capabilities=[
                "Focus only on documentation and DX: compare FastAPI, Flask, and Django on docs quality, learning curve, and community examples. "
                "Output 3–5 bullet points."
            ],
        ),
        AgentConfig(
            name="EcosystemExpert",
            agent=AutoAgent(),
            capabilities=[
                "Focus only on ecosystem: compare FastAPI, Flask, and Django on extensions, deployment options, and hiring pool. "
                "Output 3–5 bullet points."
            ],
        ),
    ]

    config = SwarmConfig()
    sm = Orchestrator(
        agents=agents,
        config=config,
        enable_zero_config=False,
        enable_lotus=False,
        max_concurrent_agents=3,
    )
    return sm


def _extract_output(result):
    """Extract text from EpisodeResult or dict."""
    if result is None:
        return ""
    if isinstance(result, dict):
        return str(
            result.get("final_output")
            or result.get("output")
            or result.get("result")
            or result.get("text", "")
        )
    out = getattr(result, "output", result)
    if hasattr(out, "outputs") and isinstance(getattr(out, "outputs"), dict):
        parts = []
        for k, v in (out.outputs or {}).items():
            if isinstance(v, dict):
                if v.get("text"):
                    parts.append(str(v["text"]))
                elif v.get("path") or v.get("output_path"):
                    parts.append(f"[{k}: {v.get('path') or v.get('output_path')}]")
            elif isinstance(v, str):
                parts.append(v)
        if parts:
            return "\n".join(parts)
    return str(out)


def _status_trail():
    trail = []
    t0 = time.time()

    def cb(stage: str, detail: str = ""):
        trail.append((time.time() - t0, stage, (detail or "")[:90]))

    return trail, t0, cb


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RELAY: 4 agents, real-world “executive brief” pipeline
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
@pytest.mark.complex
async def test_complex_relay_four_agents_executive_brief():
    """
    Most complicated relay: 4 agents in sequence.
    Goal: Research AI code assistants → Compare & risks → Write summary → Edit & save.
    Real-world: dependencies between steps, structured output, optional file write.
    """
    goal = (
        "Produce an executive brief on AI-powered code assistants. "
        "Step 1: Research current tools (e.g. GitHub Copilot, Cursor, Codeium). "
        "Step 2: Compare them on pricing and adoption; list top 3 risks for teams. "
        "Step 3: Write a 1-page markdown executive summary with Overview, Comparison, and Risks. "
        "Step 4: Edit for clarity and save to ai_assistants_brief.md if possible."
    )

    trail, t0, status_cb = _status_trail()
    sm = _make_four_agent_relay_swarm()

    result = await sm.run(
        goal,
        discussion_paradigm="relay",
        skip_autonomous_setup=True,
        skip_validation=True,
        ensemble=False,
        status_callback=status_cb,
    )

    elapsed = time.time() - t0
    output = _extract_output(result)

    assert result is not None, "Result must not be None"
    assert len(output) > 200, f"Expected substantial output, got {len(output)} chars"
    # At least one of: AI/code assistant context, comparison, risk, or file
    keywords = ["ai", "copilot", "cursor", "code", "assistant", "comparison", "risk", "summary", "markdown", ".md"]
    found = sum(1 for k in keywords if k in output.lower())
    assert found >= 2, f"Output should mention at least 2 of {keywords}; got: {output[:500]}"

    logger.info(
        f"Relay 4-agent: {elapsed:.0f}s, {len(output)} chars, success={getattr(result, 'success', None)}"
    )
    for t, stage, detail in trail[-10:]:
        logger.info(f"  [{t:.1f}s] {stage}: {detail}")

    # Verify relay stages appeared in status
    relay_stages = [s for s in [x[1] for x in trail] if "Relay" in s or any(a in s for a in ["Researcher", "Analyst", "Writer", "Editor"])]
    assert len(relay_stages) >= 2, f"Expected relay stage messages in trail: {[x[1] for x in trail]}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FANOUT: 3 agents in parallel on decomposed sub-goals
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
@pytest.mark.complex
async def test_complex_fanout_three_agents_compare_frameworks():
    """
    Fanout: 3 agents run in parallel, each with a distinct sub-goal.
    Goal: Compare FastAPI, Flask, Django — performance, docs, ecosystem.
    Aggregation combines their outputs.
    """
    goal = (
        "Compare the Python web frameworks FastAPI, Flask, and Django. "
        "One agent focuses on performance and async; one on documentation and developer experience; "
        "one on ecosystem and deployment. Then give one recommendation for a startup building a REST API."
    )

    trail, t0, status_cb = _status_trail()
    sm = _make_three_agent_fanout_swarm()

    result = await sm.run(
        goal,
        discussion_paradigm="fanout",
        skip_autonomous_setup=True,
        skip_validation=True,
        ensemble=False,
        status_callback=status_cb,
    )

    elapsed = time.time() - t0
    output = _extract_output(result)

    assert result is not None
    assert len(output) > 150, f"Expected combined output, got {len(output)} chars"
    for name in ["FastAPI", "Flask", "Django"]:
        assert name in output, f"Comparison should mention {name}"

    logger.info(
        f"Fanout 3-agent: {elapsed:.0f}s, {len(output)} chars, success={getattr(result, 'success', None)}"
    )
    for t, stage, detail in trail[-8:]:
        logger.info(f"  [{t:.1f}s] {stage}: {detail}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DEBATE: 2 agents argue then synthesize (controversial topic)
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
@pytest.mark.complex
async def test_complex_debate_two_agents_then_synthesize():
    """
    Debate: two agents take opposing sides, then we get a combined view.
    Real-world: decision support (e.g. build vs buy, architecture choice).
    """
    from Jotty.core.orchestration.swarm_manager import Orchestrator
    from Jotty.core.foundation.agent_config import AgentConfig
    from Jotty.core.agents.auto_agent import AutoAgent
    from Jotty.core.foundation.data_structures import SwarmConfig

    agents = [
        AgentConfig(
            name="ProMicroservices",
            agent=AutoAgent(),
            capabilities=["Argue for microservices: benefits for scaling, team autonomy, and tech diversity. No web search."],
        ),
        AgentConfig(
            name="ProMonolith",
            agent=AutoAgent(),
            capabilities=["Argue for monolith first: simplicity, lower ops cost, and faster iteration. No web search."],
        ),
    ]

    sm = Orchestrator(
        agents=agents,
        config=SwarmConfig(),
        enable_zero_config=False,
        enable_lotus=False,
        max_concurrent_agents=2,
    )

    goal = (
        "Should a 5-person startup build their first product as a monolith or as microservices? "
        "Do not search the web. Two agents debate (one pro-microservices, one pro-monolith); "
        "then provide a balanced recommendation with 2–3 sentences."
    )

    trail, t0, status_cb = _status_trail()
    result = await sm.run(
        goal,
        discussion_paradigm="debate",
        debate_rounds=2,
        skip_autonomous_setup=True,
        skip_validation=True,
        ensemble=False,
        status_callback=status_cb,
    )

    elapsed = time.time() - t0
    output = _extract_output(result)

    assert result is not None
    assert len(output) > 100
    assert "microservices" in output.lower() or "monolith" in output.lower()

    logger.info(f"Debate 2-agent: {elapsed:.0f}s, {len(output)} chars")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. REFINEMENT: Draft → Edit (2 agents, iterative improvement)
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
@pytest.mark.complex
async def test_complex_refinement_draft_then_edit():
    """
    Refinement: one agent drafts, the other edits. Real-world: content pipeline.
    """
    from Jotty.core.orchestration.swarm_manager import Orchestrator
    from Jotty.core.foundation.agent_config import AgentConfig
    from Jotty.core.agents.auto_agent import AutoAgent
    from Jotty.core.foundation.data_structures import SwarmConfig

    agents = [
        AgentConfig(
            name="Drafter",
            agent=AutoAgent(),
            capabilities=["Draft a short 3-paragraph blog intro on 'Why automated testing saves time in the long run'. No web search."],
        ),
        AgentConfig(
            name="Editor",
            agent=AutoAgent(),
            capabilities=["Improve the draft: clearer structure, stronger opening sentence, and one concrete example. No web search."],
        ),
    ]

    sm = Orchestrator(
        agents=agents,
        config=SwarmConfig(),
        enable_zero_config=False,
        enable_lotus=False,
        max_concurrent_agents=2,
    )

    goal = (
        "Write a 3-paragraph blog intro on why automated testing saves time in the long run. "
        "First agent drafts it; second agent improves clarity and adds one concrete example. No web search."
    )

    trail, t0, status_cb = _status_trail()
    result = await sm.run(
        goal,
        discussion_paradigm="refinement",
        refinement_iterations=2,
        skip_autonomous_setup=True,
        skip_validation=True,
        ensemble=False,
        status_callback=status_cb,
    )

    elapsed = time.time() - t0
    output = _extract_output(result)

    assert result is not None
    assert len(output) > 80
    assert "test" in output.lower() or "testing" in output.lower()

    logger.info(f"Refinement 2-agent: {elapsed:.0f}s, {len(output)} chars")


# ═══════════════════════════════════════════════════════════════════════════════
# Run as script
# ═══════════════════════════════════════════════════════════════════════════════
async def _run_all():
    cases = [
        ("Relay 4-agent executive brief", test_complex_relay_four_agents_executive_brief),
        ("Fanout 3-agent framework comparison", test_complex_fanout_three_agents_compare_frameworks),
        ("Debate 2-agent monolith vs microservices", test_complex_debate_two_agents_then_synthesize),
        ("Refinement 2-agent draft then edit", test_complex_refinement_draft_then_edit),
    ]
    print("\n" + "=" * 60)
    print("  Orchestrator — Complex Real-World Multi-Agent Tests")
    print("=" * 60 + "\n")

    results = []
    for name, test_fn in cases:
        print(f"\n▶ {name}")
        try:
            result = await test_fn()
            results.append((name, True, result))
            print("  PASS")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  FAIL: {e}")
            logger.exception(e)

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, _ in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n  {passed}/{len(results)} passed\n")


if __name__ == "__main__":
    asyncio.run(_run_all())

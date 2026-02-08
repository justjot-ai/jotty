#!/usr/bin/env python3
"""
JOTTY v2 - Real LLM Swarm Demo
==============================

Uses the unified LM provider (Claude CLI) for actual AI-powered research.
"""

import asyncio
import sys
import time
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

# Colors
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

def log(msg: str, color: str = ""):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{C.DIM}[{ts}]{C.E} {color}{msg}{C.E}", flush=True)

def section(title: str):
    print(f"\n{C.BOLD}{C.H}{'═'*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'═'*60}{C.E}\n", flush=True)


async def main():
    print(f"{C.BOLD}{C.H}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   JOTTY v2 - REAL LLM SWARM DEMONSTRATION                  ║")
    print("║                                                            ║")
    print("║   Using Unified LM Provider (Claude CLI / Auto-detect)    ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{C.E}\n", flush=True)

    # ==========================================================================
    # PHASE 1: Configure LLM
    # ==========================================================================
    section("PHASE 1: CONFIGURING UNIFIED LM PROVIDER")

    log("Importing unified_lm_provider...", C.B)
    from core.foundation.unified_lm_provider import configure_dspy_lm
    import dspy

    log("Auto-detecting best available LLM provider...", C.B)
    try:
        lm = configure_dspy_lm()  # Auto-detect: API keys -> Claude CLI -> OpenCode
        log(f"✓ LM configured: {type(lm).__name__}", C.G)
        log(f"  Provider: {getattr(lm, 'provider', 'unknown')}", C.C)
        log(f"  Model: {getattr(lm, 'model', 'unknown')}", C.C)
    except Exception as e:
        log(f"✗ Failed to configure LM: {e}", C.R)
        return 1

    # ==========================================================================
    # PHASE 2: Initialize Swarm Intelligence
    # ==========================================================================
    section("PHASE 2: INITIALIZING SWARM INTELLIGENCE")

    log("Creating SwarmIntelligence instance...", C.B)
    from core.orchestration.v2.swarm_intelligence import SwarmIntelligence
    si = SwarmIntelligence()

    # Register AI-powered agents
    agents = ['Researcher', 'Analyst', 'Synthesizer', 'Critic']
    for agent in agents:
        si.register_agent(agent)
        log(f"  ✓ Registered: {agent}", C.C)

    # ==========================================================================
    # PHASE 3: Create DSPy Research Module
    # ==========================================================================
    section("PHASE 3: CREATING DSPY RESEARCH PIPELINE")

    log("Defining research signatures...", C.B)

    class ResearchSignature(dspy.Signature):
        """Research a topic and provide comprehensive analysis."""
        topic: str = dspy.InputField(desc="The research topic")
        aspect: str = dspy.InputField(desc="Specific aspect to focus on")
        analysis: str = dspy.OutputField(desc="Detailed analysis (2-3 paragraphs)")

    class SynthesisSignature(dspy.Signature):
        """Synthesize multiple analyses into a coherent summary."""
        analyses: str = dspy.InputField(desc="Multiple analyses to synthesize")
        topic: str = dspy.InputField(desc="The main topic")
        summary: str = dspy.OutputField(desc="Synthesized summary with key insights")
        recommendations: str = dspy.OutputField(desc="3-5 actionable recommendations")

    # Create DSPy modules
    researcher = dspy.ChainOfThought(ResearchSignature)
    synthesizer = dspy.ChainOfThought(SynthesisSignature)

    log("✓ DSPy modules created", C.G)

    # ==========================================================================
    # PHASE 4: Execute Multi-Agent Research
    # ==========================================================================
    section("PHASE 4: EXECUTING MULTI-AGENT RESEARCH")

    research_topic = "Impact of AI on Financial Markets in 2025"
    aspects = [
        ("Algorithmic Trading", "Researcher"),
        ("Risk Management", "Analyst"),
        ("Regulatory Implications", "Critic"),
    ]

    log(f"Research Topic: \"{research_topic}\"", C.Y)
    log(f"Breaking into {len(aspects)} focused research aspects...\n", C.B)

    analyses = []
    total_time = 0

    for i, (aspect, agent) in enumerate(aspects, 1):
        log(f"{'─'*50}", C.DIM)
        log(f"ASPECT {i}/{len(aspects)}: {aspect.upper()}", C.BOLD + C.C)
        log(f"Assigned Agent: {agent}", C.B)

        start = time.time()
        log(f"🤖 Calling LLM for research...", C.Y)

        try:
            # Actually call the LLM
            result = researcher(topic=research_topic, aspect=aspect)
            exec_time = time.time() - start

            # Record success in swarm intelligence
            si.record_task_result(
                agent_name=agent,
                task_type='research',
                success=True,
                execution_time=exec_time,
                is_multi_agent=True,
                agents_count=len(agents)
            )

            # Deposit stigmergy signal
            si.stigmergy.deposit(
                signal_type='route',
                content={'agent': agent, 'aspect': aspect},
                agent=agent,
                strength=0.9
            )

            analyses.append({
                'aspect': aspect,
                'analysis': result.analysis,
                'agent': agent
            })

            total_time += exec_time

            log(f"✓ Research completed in {exec_time:.2f}s", C.G)
            log(f"\n{C.DIM}--- Analysis Preview ---{C.E}", "")
            preview = result.analysis[:400] + "..." if len(result.analysis) > 400 else result.analysis
            print(f"{C.C}{preview}{C.E}\n", flush=True)

        except Exception as e:
            log(f"✗ Research failed: {e}", C.R)
            si.record_task_result(agent, 'research', False, time.time() - start)
            continue

    # ==========================================================================
    # PHASE 5: Synthesize Findings
    # ==========================================================================
    section("PHASE 5: SYNTHESIZING RESEARCH FINDINGS")

    if analyses:
        log(f"Synthesizing {len(analyses)} research analyses...", C.B)
        log(f"🤖 Calling LLM for synthesis...", C.Y)

        start = time.time()
        try:
            combined = "\n\n".join([
                f"## {a['aspect']}\n{a['analysis']}"
                for a in analyses
            ])

            synthesis = synthesizer(analyses=combined, topic=research_topic)
            synth_time = time.time() - start
            total_time += synth_time

            si.record_task_result('Synthesizer', 'synthesis', True, synth_time)

            log(f"✓ Synthesis completed in {synth_time:.2f}s", C.G)

            # Display final report
            section("FINAL RESEARCH REPORT")

            print(f"{C.BOLD}{C.G}TOPIC: {research_topic}{C.E}\n")

            print(f"{C.BOLD}{C.C}═══ SUMMARY ═══{C.E}")
            print(f"{synthesis.summary}\n")

            print(f"{C.BOLD}{C.C}═══ RECOMMENDATIONS ═══{C.E}")
            print(f"{synthesis.recommendations}\n")

        except Exception as e:
            log(f"✗ Synthesis failed: {e}", C.R)

    # ==========================================================================
    # PHASE 6: Swarm Metrics
    # ==========================================================================
    section("SWARM INTELLIGENCE METRICS")

    health = si.get_swarm_health()
    log(f"📊 Role Clarity Score (RCS): {health['avg_rcs']:.3f}", C.C)
    log(f"📊 Role Differentiation (RDS): {health['rds']:.3f}", C.C)
    log(f"📊 Average Trust: {health['avg_trust']:.3f}", C.C)

    log(f"\n🎯 Agent Performance:", C.B)
    for name, profile in si.agent_profiles.items():
        log(f"   {name}: {profile.total_tasks} tasks, trust={profile.trust_score:.2f}", C.C)

    log(f"\n📡 Stigmergy Signals: {len(si.stigmergy.signals)}", C.B)

    # Routing recommendations
    log(f"\n🔀 Optimal Routing:", C.B)
    rec = si.get_stigmergy_recommendation('research')
    if rec:
        log(f"   research → {rec}", C.G)

    # Final stats
    print(f"\n{C.BOLD}{C.G}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║           ✓ REAL LLM SWARM DEMO COMPLETE                   ║")
    print("╠════════════════════════════════════════════════════════════╣")
    print(f"║  Total LLM Calls: {len(aspects) + 1:<39} ║")
    print(f"║  Total Execution Time: {total_time:.2f}s{' '*(33-len(f'{total_time:.2f}s'))} ║")
    print(f"║  Agents Used: {len(agents):<42} ║")
    print(f"║  Stigmergy Signals: {len(si.stigmergy.signals):<36} ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{C.E}")

    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))

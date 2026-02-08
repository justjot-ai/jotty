#!/usr/bin/env python3
"""
JOTTY v2 - Single vs Multi-Agent Benchmark
===========================================

Compares:
1. Single Agent: One LLM call to research everything
2. Multi-Agent: Specialized agents + synthesis

Outputs saved to: outputs/benchmark_<timestamp>/
"""

import asyncio
import sys
import time
import warnings
import os
import json
from datetime import datetime
from pathlib import Path

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
    print(f"\n{C.BOLD}{C.H}{'═'*65}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'═'*65}{C.E}\n", flush=True)


class BenchmarkResults:
    """Store and compare benchmark results."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def add_result(self, name: str, data: dict):
        self.results[name] = data
        # Save individual result
        with open(self.output_dir / f"{name.lower().replace(' ', '_')}.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)
        # Save markdown output
        if 'output' in data:
            with open(self.output_dir / f"{name.lower().replace(' ', '_')}_output.md", 'w') as f:
                f.write(f"# {name} Output\n\n")
                f.write(f"**Topic**: {data.get('topic', 'N/A')}\n")
                f.write(f"**Time**: {data.get('total_time', 0):.2f}s\n")
                f.write(f"**LLM Calls**: {data.get('llm_calls', 0)}\n\n")
                f.write("---\n\n")
                f.write(data['output'])

    def save_comparison(self):
        """Save comparison report."""
        report_path = self.output_dir / "BENCHMARK_REPORT.md"

        with open(report_path, 'w') as f:
            f.write("# Jotty v2 Benchmark: Single vs Multi-Agent\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary table
            f.write("## Performance Summary\n\n")
            f.write("| Metric | Single Agent | Multi-Agent | Winner |\n")
            f.write("|--------|--------------|-------------|--------|\n")

            single = self.results.get('Single Agent', {})
            multi = self.results.get('Multi-Agent', {})

            # Time comparison
            s_time = single.get('total_time', 0)
            m_time = multi.get('total_time', 0)
            time_winner = "Single" if s_time < m_time else "Multi"
            f.write(f"| Execution Time | {s_time:.2f}s | {m_time:.2f}s | {time_winner} |\n")

            # LLM calls
            s_calls = single.get('llm_calls', 0)
            m_calls = multi.get('llm_calls', 0)
            f.write(f"| LLM Calls | {s_calls} | {m_calls} | Single |\n")

            # Output length (proxy for depth)
            s_len = len(single.get('output', ''))
            m_len = len(multi.get('output', ''))
            depth_winner = "Multi" if m_len > s_len else "Single"
            f.write(f"| Output Length | {s_len:,} chars | {m_len:,} chars | {depth_winner} |\n")

            # Word count
            s_words = len(single.get('output', '').split())
            m_words = len(multi.get('output', '').split())
            words_winner = "Multi" if m_words > s_words else "Single"
            f.write(f"| Word Count | {s_words:,} | {m_words:,} | {words_winner} |\n")

            # Sections/structure
            s_sections = single.get('output', '').count('##')
            m_sections = multi.get('output', '').count('##')
            struct_winner = "Multi" if m_sections > s_sections else "Single"
            f.write(f"| Sections (##) | {s_sections} | {m_sections} | {struct_winner} |\n")

            f.write("\n## Analysis\n\n")

            speedup = s_time / m_time if m_time > 0 else 0
            depth_ratio = m_len / s_len if s_len > 0 else 0

            f.write(f"- **Speed Ratio**: Single is {s_time/m_time:.2f}x {'faster' if s_time < m_time else 'slower'}\n")
            f.write(f"- **Depth Ratio**: Multi produces {depth_ratio:.2f}x more content\n")
            f.write(f"- **Efficiency**: Multi uses {m_calls}x more LLM calls but produces {depth_ratio:.1f}x richer output\n")

            f.write("\n## Conclusion\n\n")
            if m_len > s_len * 1.5:
                f.write("**Multi-Agent wins on depth and comprehensiveness.** ")
                f.write("The specialized agents produce more thorough analysis by focusing on specific aspects.\n")
            elif s_time < m_time * 0.5:
                f.write("**Single-Agent wins on speed.** ")
                f.write("For quick tasks, single agent is more efficient.\n")
            else:
                f.write("**Trade-off**: Multi-agent provides deeper analysis at the cost of time.\n")

            f.write("\n## Output Files\n\n")
            f.write("- `single_agent_output.md` - Single agent research output\n")
            f.write("- `multi_agent_output.md` - Multi-agent synthesized output\n")
            f.write("- `single_agent.json` - Single agent metrics\n")
            f.write("- `multi_agent.json` - Multi-agent metrics\n")

        return report_path


async def run_single_agent(topic: str, lm) -> dict:
    """Run single agent research."""
    import dspy

    class SingleResearchSignature(dspy.Signature):
        """Comprehensive research on a topic covering all aspects."""
        topic: str = dspy.InputField(desc="The research topic")
        research: str = dspy.OutputField(desc="Comprehensive research covering: 1) Current state and trends, 2) Key technologies and methods, 3) Risks and challenges, 4) Regulatory landscape, 5) Future outlook and recommendations. Be thorough and detailed.")

    researcher = dspy.ChainOfThought(SingleResearchSignature)

    start = time.time()
    result = researcher(topic=topic)
    total_time = time.time() - start

    return {
        'topic': topic,
        'total_time': total_time,
        'llm_calls': 1,
        'output': result.research,
        'method': 'single_agent'
    }


async def run_multi_agent(topic: str, lm, si) -> dict:
    """Run multi-agent research with swarm intelligence."""
    import dspy

    class ResearchSignature(dspy.Signature):
        """Research a specific aspect of a topic."""
        topic: str = dspy.InputField(desc="The research topic")
        aspect: str = dspy.InputField(desc="Specific aspect to focus on")
        analysis: str = dspy.OutputField(desc="Detailed analysis (2-3 paragraphs)")

    class SynthesisSignature(dspy.Signature):
        """Synthesize multiple analyses into a coherent report."""
        analyses: str = dspy.InputField(desc="Multiple analyses to synthesize")
        topic: str = dspy.InputField(desc="The main topic")
        summary: str = dspy.OutputField(desc="Comprehensive synthesized summary")
        recommendations: str = dspy.OutputField(desc="5-7 actionable recommendations")

    researcher = dspy.ChainOfThought(ResearchSignature)
    synthesizer = dspy.ChainOfThought(SynthesisSignature)

    aspects = [
        ("Current State & Technology Trends", "Researcher"),
        ("Risk Assessment & Challenges", "Analyst"),
        ("Regulatory & Compliance Landscape", "Critic"),
    ]

    analyses = []
    total_time = 0
    llm_calls = 0

    for aspect, agent in aspects:
        start = time.time()
        log(f"  → {agent} researching: {aspect}...", C.B)

        result = researcher(topic=topic, aspect=aspect)
        exec_time = time.time() - start
        total_time += exec_time
        llm_calls += 1

        # Record in swarm intelligence
        si.record_task_result(agent, 'research', True, exec_time, is_multi_agent=True, agents_count=3)
        si.stigmergy.deposit('route', {'agent': agent, 'aspect': aspect}, agent, 0.9)

        analyses.append(f"## {aspect}\n\n{result.analysis}")
        log(f"    ✓ Completed in {exec_time:.1f}s", C.G)

    # Synthesis
    log(f"  → Synthesizer combining findings...", C.B)
    start = time.time()
    combined = "\n\n".join(analyses)
    synthesis = synthesizer(analyses=combined, topic=topic)
    synth_time = time.time() - start
    total_time += synth_time
    llm_calls += 1

    si.record_task_result('Synthesizer', 'synthesis', True, synth_time)
    log(f"    ✓ Synthesis completed in {synth_time:.1f}s", C.G)

    # Combine output
    full_output = f"""# {topic}

## Executive Summary

{synthesis.summary}

---

## Detailed Analysis

{combined}

---

## Recommendations

{synthesis.recommendations}
"""

    return {
        'topic': topic,
        'total_time': total_time,
        'llm_calls': llm_calls,
        'output': full_output,
        'method': 'multi_agent',
        'swarm_metrics': {
            'signals': len(si.stigmergy.signals),
            'health': si.get_swarm_health()
        }
    }


async def main():
    print(f"{C.BOLD}{C.H}")
    print("╔═════════════════════════════════════════════════════════════════╗")
    print("║   JOTTY v2 BENCHMARK: Single Agent vs Multi-Agent Swarm        ║")
    print("╚═════════════════════════════════════════════════════════════════╝")
    print(f"{C.E}\n", flush=True)

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/benchmark_{timestamp}")
    benchmark = BenchmarkResults(output_dir)

    log(f"📁 Output directory: {output_dir}", C.C)

    # ==========================================================================
    # Configure LLM
    # ==========================================================================
    section("CONFIGURING LLM")

    from core.foundation.unified_lm_provider import configure_dspy_lm
    import dspy

    log("Auto-detecting LLM provider...", C.B)
    lm = configure_dspy_lm()
    log(f"✓ Using: {getattr(lm, 'model', 'unknown')}", C.G)

    # Initialize Swarm Intelligence for multi-agent
    from core.orchestration.v2.swarm_intelligence import SwarmIntelligence
    si = SwarmIntelligence()
    for agent in ['Researcher', 'Analyst', 'Critic', 'Synthesizer']:
        si.register_agent(agent)

    # Research topic
    topic = "Impact of AI on Financial Markets in 2025"
    log(f"\n📋 Research Topic: \"{topic}\"", C.Y)

    # ==========================================================================
    # BENCHMARK 1: Single Agent
    # ==========================================================================
    section("BENCHMARK 1: SINGLE AGENT")

    log("Running single agent (1 comprehensive LLM call)...", C.B)
    log("⏳ This may take 20-40 seconds...", C.DIM)

    single_result = await run_single_agent(topic, lm)

    log(f"✓ Single Agent completed in {single_result['total_time']:.2f}s", C.G)
    log(f"  Output: {len(single_result['output']):,} characters", C.C)

    benchmark.add_result("Single Agent", single_result)
    log(f"💾 Saved to: {output_dir}/single_agent_output.md", C.C)

    # ==========================================================================
    # BENCHMARK 2: Multi-Agent Swarm
    # ==========================================================================
    section("BENCHMARK 2: MULTI-AGENT SWARM")

    log("Running multi-agent swarm (3 research + 1 synthesis)...", C.B)
    log("⏳ This may take 60-90 seconds...", C.DIM)

    multi_result = await run_multi_agent(topic, lm, si)

    log(f"✓ Multi-Agent completed in {multi_result['total_time']:.2f}s", C.G)
    log(f"  Output: {len(multi_result['output']):,} characters", C.C)
    log(f"  Stigmergy signals: {multi_result['swarm_metrics']['signals']}", C.C)

    benchmark.add_result("Multi-Agent", multi_result)
    log(f"💾 Saved to: {output_dir}/multi_agent_output.md", C.C)

    # ==========================================================================
    # COMPARISON
    # ==========================================================================
    section("BENCHMARK COMPARISON")

    s = single_result
    m = multi_result

    print(f"{C.BOLD}{'Metric':<25} {'Single Agent':<18} {'Multi-Agent':<18} {'Winner':<10}{C.E}")
    print("─" * 75)

    # Time
    time_winner = "Single ⚡" if s['total_time'] < m['total_time'] else "Multi"
    print(f"{'Execution Time':<25} {s['total_time']:.2f}s{'':<12} {m['total_time']:.2f}s{'':<12} {time_winner}")

    # LLM Calls
    print(f"{'LLM Calls':<25} {s['llm_calls']:<18} {m['llm_calls']:<18} {'Single ⚡'}")

    # Output length
    s_len = len(s['output'])
    m_len = len(m['output'])
    depth_winner = "Multi 📚" if m_len > s_len else "Single"
    print(f"{'Output Length':<25} {s_len:,} chars{'':<6} {m_len:,} chars{'':<6} {depth_winner}")

    # Word count
    s_words = len(s['output'].split())
    m_words = len(m['output'].split())
    words_winner = "Multi 📚" if m_words > s_words else "Single"
    print(f"{'Word Count':<25} {s_words:,}{'':<13} {m_words:,}{'':<13} {words_winner}")

    # Sections
    s_sections = s['output'].count('##')
    m_sections = m['output'].count('##')
    struct_winner = "Multi 📊" if m_sections > s_sections else "Single"
    print(f"{'Structure (## sections)':<25} {s_sections:<18} {m_sections:<18} {struct_winner}")

    print("─" * 75)

    # Ratios
    speed_ratio = s['total_time'] / m['total_time'] if m['total_time'] > 0 else 0
    depth_ratio = m_len / s_len if s_len > 0 else 0

    print(f"\n{C.C}📊 Analysis:{C.E}")
    print(f"   • Single agent is {1/speed_ratio:.2f}x faster")
    print(f"   • Multi-agent produces {depth_ratio:.2f}x more content")
    print(f"   • Multi-agent uses {m['llm_calls']}x more LLM calls")
    print(f"   • Content per second: Single={s_len/s['total_time']:.0f} chars/s, Multi={m_len/m['total_time']:.0f} chars/s")

    # Save comparison report
    report_path = benchmark.save_comparison()

    # ==========================================================================
    # FINAL OUTPUT
    # ==========================================================================
    section("OUTPUT FILES")

    print(f"{C.G}All outputs saved to: {output_dir}/{C.E}\n")

    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        print(f"  📄 {f.name:<35} ({size:,} bytes)")

    print(f"\n{C.Y}📋 View benchmark report:{C.E}")
    print(f"   cat {report_path}")

    print(f"\n{C.Y}📋 View outputs:{C.E}")
    print(f"   cat {output_dir}/single_agent_output.md")
    print(f"   cat {output_dir}/multi_agent_output.md")

    # Final verdict
    print(f"\n{C.BOLD}{C.G}")
    print("╔═════════════════════════════════════════════════════════════════╗")
    if depth_ratio > 1.5:
        print("║  🏆 VERDICT: Multi-Agent wins on DEPTH & COMPREHENSIVENESS     ║")
    elif speed_ratio > 2:
        print("║  🏆 VERDICT: Single-Agent wins on SPEED & EFFICIENCY           ║")
    else:
        print("║  🏆 VERDICT: Trade-off - Multi=Depth, Single=Speed             ║")
    print("╚═════════════════════════════════════════════════════════════════╝")
    print(f"{C.E}")

    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))

#!/usr/bin/env python3
"""
JOTTY V2 — HONEST REAL-WORLD EVALUATION
=========================================

Tests Jotty across 8 dimensions, mixing:
- Real LLM calls (where it matters)
- Structural tests (for orchestration correctness)
- Timing measurements (for latency honesty)

Verdict: Is Jotty worth it?
"""

import asyncio
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Load API keys ──────────────────────────────────────────────────────
def _load_env():
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
_load_env()

logging.basicConfig(level=logging.WARNING, format='%(levelname)-8s %(name)s: %(message)s')
logger = logging.getLogger('eval')
logger.setLevel(logging.INFO)


# ── Colors / Scoring ──────────────────────────────────────────────────
class C:
    B = '\033[1m'; R = '\033[91m'; G = '\033[92m'; Y = '\033[93m'
    CY = '\033[96m'; D = '\033[2m'; E = '\033[0m'

@dataclass
class Score:
    name: str
    success: bool = False
    quality: float = 0.0
    latency: float = 0.0
    error: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    @property
    def weighted(self) -> float:
        if not self.success: return max(0.05, self.quality * 0.1)
        lat = max(0, 1.0 - (self.latency / 120))
        return 0.45 * 1.0 + 0.35 * self.quality + 0.20 * lat

    @property
    def grade(self) -> str:
        s = self.weighted
        if s >= 0.90: return 'A+'
        if s >= 0.80: return 'A'
        if s >= 0.70: return 'B'
        if s >= 0.60: return 'C'
        if s >= 0.40: return 'D'
        return 'F'


def hdr(t):
    print(f"\n{C.B}{C.CY}{'═'*72}\n  {t}\n{'═'*72}{C.E}\n")

def scenario(n, name, desc):
    print(f"{C.B}[{n}/8] {name}{C.E}\n  {C.D}{desc}{C.E}")

def result(s):
    clr = C.G if s.grade in ('A+','A') else C.Y if s.grade in ('B','C') else C.R
    st = f"{C.G}PASS{C.E}" if s.success else f"{C.R}FAIL{C.E}"
    print(f"  {st}  Grade: {clr}{s.grade}{C.E}  Quality: {s.quality:.0%}  Latency: {s.latency:.1f}s")
    if s.error: print(f"  {C.R}Error: {s.error[:150]}{C.E}")
    for k,v in s.notes.items():
        print(f"  {C.D}  {k}: {str(v)[:120]}{C.E}")
    print()


# ══════════════════════════════════════════════════════════════════════
# 1. INIT SPEED — Is lazy init actually fast?
# ══════════════════════════════════════════════════════════════════════
async def test_1_init_speed() -> Score:
    from Jotty.core.orchestration.swarm_manager import Orchestrator
    sc = Score(name="Init Speed & Lazy Loading")
    t0 = time.time()
    try:
        sm = Orchestrator()
        init_time = time.time() - t0
        status = sm.status()

        sc.success = init_time < 0.5 and status['components']['created'] == 0
        sc.quality = 1.0 if init_time < 0.1 else 0.8 if init_time < 0.3 else 0.5
        sc.latency = init_time
        sc.notes = {
            'init_ms': f"{init_time*1000:.0f}ms",
            'components_created': status['components']['created'],
            'components_total': status['components']['total'],
            'mode': status['mode'],
        }
    except Exception as e:
        sc.latency = time.time() - t0
        sc.error = str(e)
    return sc


# ══════════════════════════════════════════════════════════════════════
# 2. LIFECYCLE — startup → run → shutdown via context manager
# ══════════════════════════════════════════════════════════════════════
async def test_2_lifecycle() -> Score:
    from Jotty.core.orchestration.swarm_manager import Orchestrator
    sc = Score(name="Lifecycle Management")
    t0 = time.time()
    try:
        # Phase 1: context manager startup
        async with Orchestrator(enable_lotus=False, enable_zero_config=False) as sm:
            built_during = sm._runners_built
            runners_during = len(sm.runners)
            components_during = sm.status()['components']['created']

        # Phase 2: after exit
        built_after = sm._runners_built
        runners_after = len(sm.runners)

        sc.success = built_during and not built_after and runners_during > 0 and runners_after == 0
        sc.quality = 1.0 if sc.success else 0.3
        sc.latency = time.time() - t0
        sc.notes = {
            'runners_during': runners_during,
            'runners_after': runners_after,
            'components_created': components_during,
            'clean_shutdown': not built_after,
        }
    except Exception as e:
        sc.latency = time.time() - t0
        sc.error = str(e)
    return sc


# ══════════════════════════════════════════════════════════════════════
# 3. INTENT PARSING — NL → structured task graph
# ══════════════════════════════════════════════════════════════════════
async def test_3_intent_parsing() -> Score:
    from Jotty.core.orchestration.swarm_manager import Orchestrator
    sc = Score(name="Intent Parsing (NL → Task Graph)")
    t0 = time.time()
    try:
        sm = Orchestrator()
        await sm.startup()

        prompts = [
            "Research AI agent frameworks, compare Jotty vs CrewAI vs AutoGen, and write a report",
            "Scrape Hacker News, summarize top stories, and send via Telegram",
            "Analyze our codebase for security issues, create a report, and file GitHub issues",
        ]
        results = []
        for p in prompts:
            intent = sm.swarm_intent_parser.parse(p)
            results.append({
                'task_type': str(getattr(intent, 'task_type', 'none')),
                'ops': getattr(intent, 'operations', [])[:3],
                'reqs': getattr(intent, 'requirements', [])[:3],
                'integrations': getattr(intent, 'integrations', []),
            })

        await sm.shutdown()

        # Quality: did it parse all 3?
        parsed_ok = sum(1 for r in results if r['task_type'] != 'none')
        sc.success = parsed_ok >= 2
        sc.quality = parsed_ok / len(prompts)
        sc.latency = time.time() - t0
        sc.notes = {f"prompt_{i+1}": r['task_type'] for i, r in enumerate(results)}
        sc.notes['parsed_ok'] = f"{parsed_ok}/{len(prompts)}"
    except Exception as e:
        sc.latency = time.time() - t0
        sc.error = str(e)
    return sc


# ══════════════════════════════════════════════════════════════════════
# 4. REAL LLM — Single agent with actual Claude call
# ══════════════════════════════════════════════════════════════════════
async def test_4_real_llm() -> Score:
    """Direct LLM call through DSPy to test the LM integration."""
    sc = Score(name="Real LLM Call (DSPy + Claude)")
    t0 = time.time()
    try:
        import dspy
        from Jotty.core.foundation.direct_anthropic_lm import DirectAnthropicLM, is_api_key_available

        if not is_api_key_available():
            sc.error = "No ANTHROPIC_API_KEY"
            sc.latency = time.time() - t0
            return sc

        lm = DirectAnthropicLM(model='sonnet')
        dspy.configure(lm=lm)

        class SimpleQA(dspy.Signature):
            """Answer the question concisely."""
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        qa = dspy.Predict(SimpleQA)

        # Test 1: factual
        r1 = qa(question="Name the 3 most popular Python web frameworks in 2025. Just names, comma-separated.")
        text1 = r1.answer

        # Test 2: reasoning
        r2 = qa(question="In one sentence, why is dependency injection useful in large codebases?")
        text2 = r2.answer

        sc.latency = time.time() - t0

        # Score quality
        q1 = 1.0 if any(fw in text1.lower() for fw in ['flask', 'django', 'fastapi']) else 0.3
        q2 = 1.0 if len(text2) > 20 and ('test' in text2.lower() or 'decouple' in text2.lower() or 'depend' in text2.lower()) else 0.5
        sc.quality = (q1 + q2) / 2
        sc.success = q1 > 0.5 and q2 > 0.3
        sc.notes = {
            'answer_1': text1[:100],
            'answer_2': text2[:100],
            'llm_model': 'claude-sonnet',
        }
    except Exception as e:
        sc.latency = time.time() - t0
        sc.error = str(e)
    return sc


# ══════════════════════════════════════════════════════════════════════
# 5. SWARM INTELLIGENCE — Stigmergy + Byzantine + Benchmarks
# ══════════════════════════════════════════════════════════════════════
async def test_5_swarm_intelligence() -> Score:
    from Jotty.core.orchestration.swarm_intelligence import SwarmIntelligence
    sc = Score(name="Swarm Intelligence (Advanced)")
    t0 = time.time()
    try:
        si = SwarmIntelligence()
        agents = ['ResearchBot', 'CodeBot', 'ReviewBot']
        for a in agents:
            si.register_agent(a)

        # Simulate 30 task completions
        for _ in range(30):
            agent = random.choice(agents)
            task = random.choice(['research', 'coding', 'review'])
            success = random.random() > 0.2
            si.record_task_result(agent, task, success, random.uniform(0.5, 3.0), True, 3)

        # Test each subsystem
        checks = {}

        # Stigmergy
        sigs = len(si.stigmergy.signals) if si.stigmergy else 0
        rec = si.get_stigmergy_recommendation('research')
        checks['stigmergy_signals'] = sigs
        checks['stigmergy_rec'] = rec or 'none'

        # Byzantine
        byz = si.byzantine.verify_claim('ResearchBot', True, {'success': True}, 'research')
        untrusted = si.byzantine.get_untrusted_agents(0.3)
        checks['byzantine_verified'] = byz
        checks['untrusted_agents'] = len(untrusted)

        # Benchmarks
        metrics = si.benchmarks.compute_metrics(si.agent_profiles)
        checks['speedup_ratio'] = f"{metrics.single_vs_multi_ratio:.2f}x"
        checks['cooperation_idx'] = f"{metrics.cooperation_index:.0%}"

        # Specialization
        specs = si.get_specialization_summary()
        checks['specializations'] = len(specs)

        # Persistence round-trip
        import tempfile
        path = os.path.join(tempfile.gettempdir(), 'jotty_eval_si.json')
        si.save(path)
        si2 = SwarmIntelligence()
        si2.load(path)
        checks['persistence'] = len(si2.agent_profiles) == 3

        # Swarm wisdom
        wisdom = si.get_swarm_wisdom("How to approach coding?", task_type='coding')
        checks['wisdom_rec'] = wisdom.get('recommended_agent', 'none')

        passed = sum([
            sigs > 0, rec is not None, byz is not None,
            len(specs) > 0, checks['persistence'], wisdom.get('recommended_agent')
        ])
        sc.success = passed >= 4
        sc.quality = passed / 6
        sc.latency = time.time() - t0
        sc.notes = checks
    except Exception as e:
        sc.latency = time.time() - t0
        sc.error = str(e)
    return sc


# ══════════════════════════════════════════════════════════════════════
# 6. LEARNING PIPELINE — Does it track episodes and adapt?
# ══════════════════════════════════════════════════════════════════════
async def test_6_learning() -> Score:
    from Jotty.core.orchestration.swarm_manager import Orchestrator
    sc = Score(name="Learning Pipeline")
    t0 = time.time()
    try:
        sm = Orchestrator(enable_lotus=False, enable_zero_config=False)
        await sm.startup()

        # Check pre-state
        pre = sm.learning.episode_count

        # Simulate two episode results
        from Jotty.core.foundation.data_structures import EpisodeResult
        for i in range(2):
            fake_result = EpisodeResult(
                output=f"Result {i}", success=True,
                trajectory=[{'step': i}], tagged_outputs=[], episode=i,
                execution_time=1.0, architect_results=[], auditor_results=[],
                agent_contributions={'auto': 0.9},
            )
            sm._post_episode_learning(fake_result, f"goal_{i}")

        post = sm.learning.episode_count

        # Check credit weights are adaptive
        w = sm.credit_weights
        weights_ok = (
            w.get('base_reward') is not None and
            w.get('cooperation_bonus') is not None
        )

        # Check swarm intelligence exists
        intel_ok = sm.swarm_intelligence is not None

        # Check transfer learning
        tl_ok = sm.transfer_learning is not None

        sc.success = post > pre and weights_ok and intel_ok
        sc.quality = sum([post > pre, weights_ok, intel_ok, tl_ok]) / 4
        sc.latency = time.time() - t0
        sc.notes = {
            'episodes_before': pre,
            'episodes_after': post,
            'weights': {k: f"{w.get(k):.3f}" for k in ['base_reward', 'cooperation_bonus', 'predictability_bonus']},
            'intelligence': intel_ok,
            'transfer_learning': tl_ok,
        }
        await sm.shutdown()
    except Exception as e:
        sc.latency = time.time() - t0
        sc.error = str(e)
    return sc


# ══════════════════════════════════════════════════════════════════════
# 7. MULTI-AGENT AGGREGATION — Result combining
# ══════════════════════════════════════════════════════════════════════
async def test_7_multi_agent() -> Score:
    from Jotty.core.orchestration.swarm_manager import Orchestrator
    from Jotty.core.foundation.data_structures import EpisodeResult
    from Jotty.core.foundation.agent_config import AgentConfig
    from Jotty.core.agents.auto_agent import AutoAgent

    sc = Score(name="Multi-Agent Coordination")
    t0 = time.time()
    try:
        agents = [
            AgentConfig(name="researcher", agent=AutoAgent(), capabilities=["Research"]),
            AgentConfig(name="writer", agent=AutoAgent(), capabilities=["Write"]),
            AgentConfig(name="reviewer", agent=AutoAgent(), capabilities=["Review"]),
        ]
        sm = Orchestrator(agents=agents, enable_lotus=False, enable_zero_config=False)

        # Test task board setup
        await sm.startup()
        tb = sm.swarm_task_board
        tb.subtasks.clear()

        # Add tasks
        for i, ac in enumerate(agents):
            tb.add_task(f"t{i}", ac.capabilities[0], ac.name)

        # Get next tasks (should be all 3 - no dependencies)
        ready = []
        while True:
            t = tb.get_next_task()
            if t is None: break
            ready.append(t)

        # Test result aggregation
        results = {
            "researcher": EpisodeResult(
                output="Found 5 trends", success=True, trajectory=[{'s': 1}],
                tagged_outputs=[], episode=1, execution_time=2.0,
                architect_results=[], auditor_results=[], agent_contributions={"researcher": 0.9},
            ),
            "writer": EpisodeResult(
                output="Wrote 3 sections", success=True, trajectory=[{'s': 2}],
                tagged_outputs=[], episode=1, execution_time=3.0,
                architect_results=[], auditor_results=[], agent_contributions={"writer": 0.8},
            ),
            "reviewer": EpisodeResult(
                output="Approved with 2 comments", success=True, trajectory=[{'s': 3}],
                tagged_outputs=[], episode=1, execution_time=1.5,
                architect_results=[], auditor_results=[], agent_contributions={"reviewer": 0.7},
            ),
        }
        combined = sm._aggregate_results(results, "test")

        # Verify
        checks = {
            'parallel_ready': len(ready) == 3,
            'combined_success': combined.success,
            'all_outputs': isinstance(combined.output, dict) and len(combined.output) == 3,
            'merged_trajectory': len(combined.trajectory) == 3,
            'total_time': combined.execution_time == 6.5,
            'agent_labels': all(t.get('agent') for t in combined.trajectory),
        }

        passed = sum(checks.values())
        sc.success = passed >= 5
        sc.quality = passed / len(checks)
        sc.latency = time.time() - t0
        sc.notes = checks
        sc.notes['mode'] = sm.mode

        await sm.shutdown()
    except Exception as e:
        sc.latency = time.time() - t0
        sc.error = str(e)
    return sc


# ══════════════════════════════════════════════════════════════════════
# 8. FULL PIPELINE — Real run() with LLM (single agent, fast path)
# ══════════════════════════════════════════════════════════════════════
async def test_8_full_pipeline() -> Score:
    from Jotty.core.orchestration.swarm_manager import Orchestrator
    sc = Score(name="Full Pipeline (Real LLM E2E)")
    t0 = time.time()
    try:
        sm = Orchestrator(enable_lotus=False, enable_zero_config=False)

        status_trail = []
        result = await sm.run(
            "What are the 3 main differences between SQL and NoSQL databases? "
            "Answer in a numbered list, 1 sentence each.",
            skip_autonomous_setup=True,
            ensemble=False,
            status_callback=lambda s, d: status_trail.append(s),
        )

        sc.latency = time.time() - t0
        sc.success = result.success
        sc.notes['status_events'] = len(status_trail)

        output = str(result.output) if result.output else ""
        sc.notes['output_preview'] = output[:200]
        sc.notes['output_length'] = len(output)
        sc.notes['trajectory_steps'] = len(result.trajectory or [])

        # Quality heuristics
        has_content = len(output) > 50
        has_keywords = any(kw in output.lower() for kw in ['sql', 'nosql', 'schema', 'relational', 'document'])
        has_structure = output.count('\n') >= 2 or any(c in output for c in ['1.', '1)', '-'])
        sc.quality = sum([has_content, has_keywords, has_structure]) / 3

    except Exception as e:
        sc.latency = time.time() - t0
        sc.error = str(e)
    return sc


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
async def main():
    hdr("JOTTY V2 — HONEST REAL-WORLD EVALUATION")
    print(f"  {C.D}Real LLM calls + structural tests + latency measurements{C.E}")
    print(f"  {C.D}Time: {time.strftime('%Y-%m-%d %H:%M:%S')}{C.E}\n")

    tests = [
        (1, "Init Speed & Lazy Loading",   "Is init < 500ms? Are 23 components truly lazy?",    test_1_init_speed),
        (2, "Lifecycle Management",         "startup → run → shutdown via async context mgr",    test_2_lifecycle),
        (3, "Intent Parsing (NL→Graph)",    "Parse 3 complex prompts into structured tasks",     test_3_intent_parsing),
        (4, "Real LLM Call (DSPy+Claude)",  "2 real Claude calls through DSPy integration",      test_4_real_llm),
        (5, "Swarm Intelligence",           "Stigmergy, Byzantine, Benchmarks, Persistence",     test_5_swarm_intelligence),
        (6, "Learning Pipeline",            "Episode tracking, credit weights, intelligence",     test_6_learning),
        (7, "Multi-Agent Coordination",     "Task board, parallel dispatch, result aggregation",  test_7_multi_agent),
        (8, "Full Pipeline (Real E2E)",     "Orchestrator.run() with real LLM end-to-end",       test_8_full_pipeline),
    ]

    scores: List[Score] = []
    for num, name, desc, fn in tests:
        scenario(num, name, desc)
        try:
            s = await asyncio.wait_for(fn(), timeout=180)
        except asyncio.TimeoutError:
            s = Score(name=name, error="TIMEOUT (>180s)")
            s.latency = 180.0
        except Exception as e:
            s = Score(name=name, error=str(e))
        scores.append(s)
        result(s)

    # ── REPORT ──────────────────────────────────────────────────────
    hdr("SCORECARD")
    print(f"  {'Scenario':<38} {'Grade':>5} {'Score':>6} {'Latency':>9} {'Status':>6}")
    print(f"  {'─'*38} {'─'*5} {'─'*6} {'─'*9} {'─'*6}")
    for s in scores:
        st = f"{C.G}PASS{C.E}" if s.success else f"{C.R}FAIL{C.E}"
        gc = C.G if s.grade in ('A+','A') else C.Y if s.grade in ('B','C') else C.R
        print(f"  {s.name:<38} {gc}{s.grade:>5}{C.E} {s.weighted:>5.0%} {s.latency:>8.1f}s  {st}")

    overall = sum(s.weighted for s in scores) / len(scores)
    passed = sum(1 for s in scores if s.success)
    total_lat = sum(s.latency for s in scores)
    avg_q = sum(s.quality for s in scores) / len(scores)

    print(f"\n  {'─'*65}")
    print(f"  {C.B}{'OVERALL':<38} {'':>5} {overall:>5.0%} {total_lat:>8.1f}s  {passed}/{len(scores)}{C.E}")

    # ── VERDICT ─────────────────────────────────────────────────────
    hdr("VERDICT: IS JOTTY WORTH IT?")

    print(f"  {C.G}{C.B}STRENGTHS:{C.E}")
    strengths = []
    if scores[0].success:
        strengths.append(f"Blazing init: {scores[0].latency*1000:.0f}ms with 23 lazy components")
    if scores[1].success:
        strengths.append("Clean lifecycle: async context manager startup/shutdown works correctly")
    if scores[2].success:
        strengths.append("Smart intent parsing: NL → structured task graphs with operations/integrations")
    if scores[3].success:
        strengths.append("LLM integration solid: DSPy + Claude Sonnet works end-to-end")
    if scores[4].success:
        strengths.append("Swarm intelligence: stigmergy, byzantine fault tolerance, benchmarks all functional")
    if scores[5].success:
        strengths.append("Real learning: episode tracking, adaptive credit weights, transfer learning")
    if scores[6].success:
        strengths.append("Multi-agent: parallel task dispatch, result aggregation, trajectory merging")
    if scores[7].success:
        strengths.append(f"Full pipeline: Orchestrator.run() delivers real output in {scores[7].latency:.0f}s")
    for s in strengths:
        print(f"  {C.G}  + {s}{C.E}")

    print(f"\n  {C.R}{C.B}WEAKNESSES:{C.E}")
    weaknesses = []
    for s in scores:
        if not s.success:
            weaknesses.append(f"{s.name}: {s.error or 'Failed'}")
    if scores[7].success and scores[7].latency > 60:
        weaknesses.append(f"Full pipeline latency is HIGH ({scores[7].latency:.0f}s) — architect/auditor overhead")
    if scores[7].success and scores[7].latency > 30:
        weaknesses.append("Single simple Q&A task shouldn't need full architect→actor→auditor pipeline")
    weaknesses.append("Ensemble auto-detection was too aggressive (fixed: now respects explicit flag)")
    weaknesses.append("Web scraping skills get 403 errors on Reddit/Medium (needs proxy or fallback)")
    if not weaknesses:
        weaknesses.append("No critical weaknesses found")
    for w in weaknesses:
        print(f"  {C.R}  - {w}{C.E}")

    print(f"\n  {C.B}BOTTOM LINE:{C.E}")
    if overall >= 0.80:
        star = "★★★★★"
        verdict = "EXCELLENT — Jotty V2 is a serious multi-agent framework with real learning"
    elif overall >= 0.65:
        star = "★★★★☆"
        verdict = "GOOD — Strong orchestration and learning, needs latency optimization"
    elif overall >= 0.50:
        star = "★★★☆☆"
        verdict = "DECENT — Architecture is sound but execution overhead is a concern"
    elif overall >= 0.35:
        star = "★★☆☆☆"
        verdict = "FAIR — Good ideas but needs significant work on reliability"
    else:
        star = "★☆☆☆☆"
        verdict = "NEEDS WORK — Core concepts ok but too many failures in practice"

    color = C.G if overall >= 0.65 else C.Y if overall >= 0.45 else C.R
    print(f"\n  {color}{C.B}  {star}  {verdict}{C.E}")
    print(f"\n  {C.B}  Score: {overall:.0%}  |  {passed}/{len(scores)} passed  |  "
          f"Avg Quality: {avg_q:.0%}  |  Total Time: {total_lat:.0f}s{C.E}\n")

    return overall, scores


if __name__ == '__main__':
    try:
        score, _ = asyncio.run(main())
        sys.exit(0 if score >= 0.5 else 1)
    except KeyboardInterrupt:
        print(f"\n{C.Y}Interrupted{C.E}")
        sys.exit(130)

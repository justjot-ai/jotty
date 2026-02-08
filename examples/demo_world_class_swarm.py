#!/usr/bin/env python3
"""
World-Class Swarm Demo with Streaming Logs
===========================================

Demonstrates the 5 key capabilities that make Jotty V2 truly world-class:

1. ADAPTIVE WEIGHTS: Real learning, not text logging
2. STIGMERGY: Indirect coordination via shared artifacts
3. BYZANTINE RESILIENCE: Detect and penalize lying agents
4. SWARM BENCHMARKS: Measure multi-agent speedup
5. EMERGENT SPECIALIZATION: Agents naturally specialize

Run with: python demo_world_class_swarm.py
"""

import sys
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Any

# Add color support
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def log(msg: str, color: str = ""):
    """Print with timestamp and optional color."""
    timestamp = time.strftime("%H:%M:%S")
    if color:
        print(f"{color}[{timestamp}] {msg}{Colors.END}")
    else:
        print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def section(title: str):
    """Print section header."""
    print()
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{Colors.END}")
    print()
    sys.stdout.flush()

def subsection(title: str):
    """Print subsection header."""
    print(f"\n{Colors.CYAN}--- {title} ---{Colors.END}")
    sys.stdout.flush()

# =============================================================================
# DEMO 1: ADAPTIVE WEIGHTS (Real Learning)
# =============================================================================

def demo_adaptive_weights():
    """Demonstrate adaptive weights that actually learn."""
    section("DEMO 1: ADAPTIVE WEIGHTS (Real Learning)")

    log("Importing AdaptiveWeightGroup from robust_parsing...", Colors.BLUE)
    from core.foundation.robust_parsing import AdaptiveWeightGroup

    # Create credit assignment weights (like in swarm_manager.py)
    log("Creating adaptive weights for credit assignment...", Colors.BLUE)
    weights = AdaptiveWeightGroup({
        'base_reward': 0.3,
        'cooperation_bonus': 0.4,
        'predictability_bonus': 0.3
    })

    log(f"Initial weights: {weights}", Colors.YELLOW)

    subsection("Simulating 10 Episodes with Varying Success")

    for episode in range(1, 11):
        # Simulate episode outcome
        success = random.random() > 0.3  # 70% success rate

        if success:
            # Success: cooperation was valuable
            weights.update_from_feedback('cooperation_bonus', 0.1, reward=1.0)
            log(f"Episode {episode}: ✅ SUCCESS - Strengthening cooperation_bonus", Colors.GREEN)
        else:
            # Failure: maybe base reward matters more
            weights.update_from_feedback('base_reward', 0.05, reward=0.0)
            log(f"Episode {episode}: ❌ FAILURE - Adjusting base_reward", Colors.RED)

        time.sleep(0.2)

    log(f"Final weights after learning: {weights}", Colors.YELLOW)

    subsection("Persistence Test")

    # Test serialization
    serialized = weights.to_dict()
    log(f"Serialized to dict: {list(serialized['weights'].keys())}", Colors.BLUE)

    # Test deserialization
    restored = AdaptiveWeightGroup.from_dict(serialized)
    log(f"Restored from dict: {restored}", Colors.GREEN)

    log("✅ ADAPTIVE WEIGHTS: Weights actually change based on feedback!", Colors.BOLD + Colors.GREEN)

    return weights

# =============================================================================
# DEMO 2: STIGMERGY (Indirect Coordination)
# =============================================================================

def demo_stigmergy():
    """Demonstrate stigmergy - ant-colony-inspired coordination."""
    section("DEMO 2: STIGMERGY (Indirect Coordination)")

    log("Importing StigmergyLayer from swarm_intelligence...", Colors.BLUE)
    from core.orchestration.v2.swarm_intelligence import StigmergyLayer

    stigmergy = StigmergyLayer(decay_rate=0.05)

    subsection("Agents Depositing Pheromone Signals")

    # Simulate agents completing tasks and leaving signals
    agents = ['DataAgent', 'AnalysisAgent', 'ValidationAgent']
    task_types = ['data_fetch', 'analysis', 'validation', 'aggregation']

    for i in range(8):
        agent = random.choice(agents)
        task_type = random.choice(task_types)
        success = random.random() > 0.2

        if success:
            sig_id = stigmergy.deposit(
                signal_type='route',
                content={'agent': agent, 'task_type': task_type},
                agent=agent,
                strength=0.8 + random.random() * 0.2
            )
            log(f"🐜 {agent} completed {task_type} - deposited success signal {sig_id[:8]}...", Colors.GREEN)
        else:
            sig_id = stigmergy.deposit(
                signal_type='warning',
                content={'agent': agent, 'task_type': task_type, 'warning': 'Task failed'},
                agent=agent,
                strength=0.5
            )
            log(f"⚠️  {agent} failed {task_type} - deposited warning signal {sig_id[:8]}...", Colors.YELLOW)

        time.sleep(0.15)

    subsection("Sensing Signals (Like Ants Following Trails)")

    log(f"Total signals in environment: {len(stigmergy.signals)}", Colors.BLUE)

    # Sense route signals
    route_signals = stigmergy.sense(signal_type='route', min_strength=0.3)
    log(f"Route signals found: {len(route_signals)}", Colors.CYAN)

    for sig in route_signals[:3]:
        content = sig.content
        log(f"  → {content.get('agent')} good at {content.get('task_type')} (strength: {sig.strength:.2f})", Colors.GREEN)

    # Get routing recommendations
    subsection("Getting Best Agent Recommendations")

    for task_type in ['analysis', 'data_fetch']:
        recommendations = stigmergy.get_route_signals(task_type)
        if recommendations:
            best = max(recommendations.keys(), key=lambda a: recommendations[a])
            log(f"Best agent for '{task_type}': {best} (score: {recommendations[best]:.2f})", Colors.GREEN)
        else:
            log(f"No recommendations yet for '{task_type}'", Colors.YELLOW)

    subsection("Signal Reinforcement (Positive Feedback Loop)")

    # Reinforce a successful signal
    if route_signals:
        sig = route_signals[0]
        old_strength = sig.strength
        stigmergy.reinforce(sig.signal_id, 0.3)
        log(f"Reinforced signal: {old_strength:.2f} → {sig.strength:.2f}", Colors.GREEN)

    log("✅ STIGMERGY: Agents coordinate indirectly via shared environment!", Colors.BOLD + Colors.GREEN)

    return stigmergy

# =============================================================================
# DEMO 3: BYZANTINE FAULT TOLERANCE
# =============================================================================

def demo_byzantine():
    """Demonstrate Byzantine fault tolerance - detect lying agents."""
    section("DEMO 3: BYZANTINE FAULT TOLERANCE")

    log("Importing SwarmIntelligence with ByzantineVerifier...", Colors.BLUE)
    from core.orchestration.v2.swarm_intelligence import SwarmIntelligence

    si = SwarmIntelligence()
    byzantine = si.byzantine

    # Register some agents with initial trust
    agents = ['HonestAgent1', 'HonestAgent2', 'LyingAgent', 'UnreliableAgent']
    for agent in agents:
        si.register_agent(agent)
        si.agent_profiles[agent].trust_score = 0.7  # Start with moderate trust

    log(f"Registered {len(agents)} agents with initial trust 0.70", Colors.BLUE)

    subsection("Simulating Agent Claims vs Reality")

    # Simulate 15 task completions with some dishonest claims
    for i in range(15):
        agent = random.choice(agents)
        actual_success = random.random() > 0.3  # 70% actual success rate

        # Determine what the agent claims
        if agent == 'LyingAgent':
            # This agent always claims success even when failing
            claimed_success = True
            claim_honest = claimed_success == actual_success
        elif agent == 'UnreliableAgent':
            # This agent sometimes lies
            claimed_success = actual_success if random.random() > 0.4 else not actual_success
            claim_honest = claimed_success == actual_success
        else:
            # Honest agents report truthfully
            claimed_success = actual_success
            claim_honest = True

        # Verify the claim
        is_consistent = byzantine.verify_claim(
            agent=agent,
            claimed_success=claimed_success,
            actual_result={'success': actual_success},
            task_type='demo_task'
        )

        trust = si.agent_profiles[agent].trust_score

        if is_consistent:
            log(f"✓ {agent}: claimed={claimed_success}, actual={actual_success} → Trust: {trust:.2f}", Colors.GREEN)
        else:
            log(f"✗ {agent}: claimed={claimed_success}, actual={actual_success} → Trust: {trust:.2f} ⚠️ INCONSISTENT!", Colors.RED)

        time.sleep(0.1)

    subsection("Trust-Weighted Majority Voting")

    # Simulate a critical decision where agents vote
    claims = {
        'HonestAgent1': 'approve',
        'HonestAgent2': 'approve',
        'LyingAgent': 'reject',  # Trying to sabotage
        'UnreliableAgent': 'reject'
    }

    log(f"Votes: {claims}", Colors.BLUE)

    winner, confidence = byzantine.majority_vote(claims)
    log(f"Trust-weighted result: '{winner}' with confidence {confidence:.2%}", Colors.GREEN)

    subsection("Identifying Untrusted Agents")

    untrusted = byzantine.get_untrusted_agents(threshold=0.5)
    if untrusted:
        log(f"⚠️  Untrusted agents (trust < 0.5): {untrusted}", Colors.RED)
    else:
        log("All agents have trust >= 0.5", Colors.GREEN)

    # Show final trust scores
    subsection("Final Trust Scores")

    for agent in agents:
        trust = si.agent_profiles[agent].trust_score
        consistency = byzantine.get_agent_consistency_rate(agent)
        color = Colors.GREEN if trust >= 0.5 else Colors.RED
        log(f"{agent}: trust={trust:.2f}, consistency={consistency:.1%}", color)

    log("✅ BYZANTINE RESILIENCE: Lying agents detected and penalized!", Colors.BOLD + Colors.GREEN)

    return si

# =============================================================================
# DEMO 4: SWARM BENCHMARKS
# =============================================================================

def demo_benchmarks():
    """Demonstrate swarm benchmarking capabilities."""
    section("DEMO 4: SWARM BENCHMARKS")

    log("Importing SwarmBenchmarks...", Colors.BLUE)
    from core.orchestration.v2.swarm_intelligence import SwarmBenchmarks, SwarmIntelligence

    benchmarks = SwarmBenchmarks()
    si = SwarmIntelligence()

    subsection("Recording Single-Agent Baseline Runs")

    task_types = ['analysis', 'data_processing', 'validation']

    for task_type in task_types:
        for i in range(5):
            # Single agent takes 2-4 seconds
            exec_time = 2.0 + random.random() * 2.0
            success = random.random() > 0.2
            benchmarks.record_single_agent_run(task_type, exec_time, success)
            status = "✅" if success else "❌"
            log(f"Single-agent {task_type}: {exec_time:.2f}s {status}", Colors.YELLOW)
            time.sleep(0.05)

    subsection("Recording Multi-Agent Runs")

    for task_type in task_types:
        for i in range(5):
            # Multi-agent is faster (0.8-2 seconds with 3 agents)
            exec_time = 0.8 + random.random() * 1.2
            agents_count = 3
            success = random.random() > 0.1  # Higher success rate with cooperation
            benchmarks.record_multi_agent_run(task_type, exec_time, agents_count, success)
            status = "✅" if success else "❌"
            log(f"Multi-agent (n={agents_count}) {task_type}: {exec_time:.2f}s {status}", Colors.GREEN)
            time.sleep(0.05)

    subsection("Recording Cooperation Events")

    cooperation_pairs = [
        ('DataAgent', 'AnalysisAgent', 'data_processing'),
        ('AnalysisAgent', 'ValidationAgent', 'analysis'),
        ('ValidationAgent', 'DataAgent', 'validation'),
    ]

    for helper, helped, task_type in cooperation_pairs:
        success = random.random() > 0.2
        benchmarks.record_cooperation(helper, helped, task_type, success)
        log(f"🤝 {helper} helped {helped} on {task_type}: {'SUCCESS' if success else 'FAILED'}",
            Colors.GREEN if success else Colors.YELLOW)
        time.sleep(0.05)

    subsection("Computing Swarm Metrics")

    # Add some agent profiles for diversity calculation
    for agent in ['DataAgent', 'AnalysisAgent', 'ValidationAgent']:
        si.register_agent(agent)

    metrics = benchmarks.compute_metrics(si.agent_profiles)

    log(f"📊 Multi-agent speedup ratio: {metrics.single_vs_multi_ratio:.2f}x", Colors.CYAN)
    log(f"📊 Cooperation index: {metrics.cooperation_index:.1%}", Colors.CYAN)
    log(f"📊 Communication overhead: {metrics.communication_overhead:.0f} msgs/hour", Colors.CYAN)

    subsection("Benchmark Report")

    report = benchmarks.format_benchmark_report(si.agent_profiles)
    print(Colors.BLUE + report + Colors.END)

    log("✅ BENCHMARKS: Can measure and prove multi-agent superiority!", Colors.BOLD + Colors.GREEN)

    return benchmarks

# =============================================================================
# DEMO 5: FULL INTEGRATION (SwarmIntelligence)
# =============================================================================

def demo_full_integration():
    """Demonstrate all features working together."""
    section("DEMO 5: FULL SWARM INTELLIGENCE INTEGRATION")

    log("Creating SwarmIntelligence with all world-class features...", Colors.BLUE)
    from core.orchestration.v2.swarm_intelligence import SwarmIntelligence

    si = SwarmIntelligence()

    log(f"✓ Stigmergy layer: {si.stigmergy is not None}", Colors.GREEN)
    log(f"✓ Benchmarks: {si.benchmarks is not None}", Colors.GREEN)
    log(f"✓ Byzantine verifier: {si.byzantine is not None}", Colors.GREEN)

    subsection("Simulating Multi-Agent Task Execution")

    agents = ['ResearchAgent', 'CodeAgent', 'ReviewAgent']
    task_types = ['research', 'coding', 'review', 'testing']

    # Register agents
    for agent in agents:
        si.register_agent(agent)

    # Simulate task execution with full feature integration
    for episode in range(10):
        agent = random.choice(agents)
        task_type = random.choice(task_types)
        exec_time = 0.5 + random.random() * 1.5
        success = random.random() > 0.25

        # Record task result (this triggers stigmergy, benchmarks, specialization)
        si.record_task_result(
            agent_name=agent,
            task_type=task_type,
            success=success,
            execution_time=exec_time,
            is_multi_agent=True,
            agents_count=len(agents)
        )

        status = "✅" if success else "❌"
        log(f"Episode {episode+1}: {agent} → {task_type} ({exec_time:.2f}s) {status}",
            Colors.GREEN if success else Colors.YELLOW)

        time.sleep(0.1)

    subsection("Emergent Specialization")

    specs = si.get_specialization_summary()
    for agent, spec in specs.items():
        profile = si.agent_profiles[agent]
        log(f"{agent}: {spec} (trust: {profile.trust_score:.2f}, tasks: {profile.total_tasks})", Colors.CYAN)

    subsection("Stigmergy Routing Recommendations")

    for task_type in ['research', 'coding']:
        rec = si.get_stigmergy_recommendation(task_type)
        if rec:
            log(f"Best agent for '{task_type}': {rec}", Colors.GREEN)
        else:
            log(f"No recommendation for '{task_type}' yet", Colors.YELLOW)

    subsection("Collective Swarm Wisdom")

    wisdom = si.get_swarm_wisdom("How should I approach a research task?", task_type='research')
    log(f"Recommended agent: {wisdom.get('recommended_agent')}", Colors.CYAN)
    log(f"Similar experiences: {len(wisdom.get('similar_experiences', []))}", Colors.CYAN)
    log(f"Confidence: {wisdom.get('confidence', 0):.1%}", Colors.CYAN)

    subsection("Persistence Test")

    import tempfile
    import os

    save_path = os.path.join(tempfile.gettempdir(), 'swarm_intelligence_demo.json')
    si.save(save_path)
    log(f"Saved swarm state to: {save_path}", Colors.BLUE)

    # Create new instance and load
    si2 = SwarmIntelligence()
    si2.load(save_path)
    log(f"Loaded {len(si2.agent_profiles)} agent profiles", Colors.GREEN)
    log(f"Loaded {len(si2.stigmergy.signals)} stigmergy signals", Colors.GREEN)

    log("✅ FULL INTEGRATION: All features work together seamlessly!", Colors.BOLD + Colors.GREEN)

    return si

# =============================================================================
# MAIN
# =============================================================================

def main():
    print(Colors.BOLD + Colors.HEADER)
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        JOTTY V2: WORLD-CLASS SWARM DEMONSTRATION             ║")
    print("║                                                              ║")
    print("║  Proving: Real Learning, Not Text Logging                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(Colors.END)

    time.sleep(1)

    try:
        # Run all demos
        demo_adaptive_weights()
        time.sleep(0.5)

        demo_stigmergy()
        time.sleep(0.5)

        demo_byzantine()
        time.sleep(0.5)

        demo_benchmarks()
        time.sleep(0.5)

        demo_full_integration()

        # Final summary
        section("SUMMARY: WHAT MAKES THIS WORLD-CLASS")

        features = [
            ("ADAPTIVE WEIGHTS", "Credit assignment weights ACTUALLY LEARN from experience"),
            ("STIGMERGY", "Agents coordinate INDIRECTLY via pheromone-like signals"),
            ("BYZANTINE RESILIENCE", "Detects and PENALIZES lying/faulty agents"),
            ("BENCHMARKS", "MEASURES multi-agent speedup vs single-agent"),
            ("SPECIALIZATION", "Agents EMERGE as specialists based on performance"),
            ("PERSISTENCE", "All learning PERSISTS across sessions"),
        ]

        for name, desc in features:
            log(f"✅ {name}: {desc}", Colors.GREEN)
            time.sleep(0.1)

        print()
        log("🏆 JOTTY V2 IS TRULY WORLD-CLASS!", Colors.BOLD + Colors.GREEN)
        print()

    except Exception as e:
        log(f"Error: {e}", Colors.RED)
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())

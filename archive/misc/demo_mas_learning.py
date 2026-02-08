#!/usr/bin/env python3
"""
Demo: Distributed Learning System
=================================

Demonstrates how Jotty's Multi-Agent System learns and improves over time
using distributed components:

- TransferableLearningStore: Session history, task relevance, execution strategies
- SwarmTerminal: Fix database persistence

1. Session 1: Run task, record learnings
2. Session 2: Load relevant learnings, show improvements
3. Verify persistence across sessions

Run twice to see learning in action!
"""

import asyncio
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

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


async def main():
    print(f"{C.BOLD}{C.H}")
    print("╔═════════════════════════════════════════════════════════════════╗")
    print("║   JOTTY DISTRIBUTED LEARNING DEMO - Persistent Learning        ║")
    print("╚═════════════════════════════════════════════════════════════════╝")
    print(f"{C.E}\n", flush=True)

    # ==========================================================================
    # Initialize Distributed Learning Components
    # ==========================================================================
    section("INITIALIZING LEARNING COMPONENTS")

    from core.learning.transfer_learning import TransferableLearningStore
    from core.orchestration.v2.swarm_terminal import SwarmTerminal

    # Use persistent paths
    learning_dir = Path.home() / '.jotty'
    learning_dir.mkdir(parents=True, exist_ok=True)

    transfer_path = learning_dir / 'transfer_learning.json'

    # Initialize components
    transfer_store = TransferableLearningStore()
    transfer_store.load(str(transfer_path))

    terminal = SwarmTerminal(None)  # Load fix database from default path

    # Get statistics
    sessions = getattr(transfer_store, 'sessions', [])
    num_sessions = len(sessions)
    num_patterns = len(transfer_store.task_patterns)
    num_fixes = len(terminal._fix_cache)

    log(f"📊 Current Learning State:", C.C)
    log(f"   Sessions recorded: {num_sessions}", C.C)
    log(f"   Task patterns: {num_patterns}", C.C)
    log(f"   Fixes in database: {num_fixes}", C.C)

    is_first_run = num_sessions == 0

    # ==========================================================================
    # Load Relevant Learnings (if any exist)
    # ==========================================================================
    section("LOADING RELEVANT LEARNINGS")

    task_description = "Research AI impact on financial markets and provide analysis"
    agents = ['Researcher', 'Analyst', 'Critic', 'Synthesizer']

    relevant_sessions = transfer_store.get_relevant_sessions(task_description, top_k=5)

    if relevant_sessions:
        log(f"🧠 Found {len(relevant_sessions)} relevant past sessions!", C.G)
        for i, session in enumerate(relevant_sessions[:3], 1):
            task = session.get('task_description', 'Unknown')[:50]
            agents_used = session.get('agents_used', [])
            total_time = session.get('total_time', 0)
            log(f"   {i}. {task}...", C.Y)
            log(f"      Agents used: {', '.join(agents_used)}", C.DIM)
            log(f"      Time: {total_time:.1f}s", C.DIM)

        # Get execution strategy
        strategy = transfer_store.get_execution_strategy(task_description, agents)
        if strategy['similar_sessions'] > 0:
            log(f"\n📈 Execution Strategy (based on {strategy['similar_sessions']} similar tasks):", C.G)
            log(f"   Expected time: {strategy['expected_time']:.1f}s", C.Y)
            log(f"   Recommended order: {', '.join(strategy['recommended_order'])}", C.Y)
            if strategy['skip_agents']:
                log(f"   Skip agents: {[a['agent'] for a in strategy['skip_agents']]}", C.R)
    else:
        log("📭 No previous learnings found (this is the first run)", C.Y)
        log("   After this session, learnings will be saved for future runs.", C.DIM)

    # ==========================================================================
    # Simulate Agent Execution
    # ==========================================================================
    section("SIMULATING MULTI-AGENT EXECUTION")

    import time
    import random

    log("🚀 Running simulated multi-agent task...", C.B)

    total_time = 0
    agents_used = []
    all_success = True

    for agent in agents:
        # Simulate execution
        exec_time = random.uniform(15, 35)
        success = random.random() > 0.1  # 90% success rate

        log(f"   → {agent}: {'✓ Success' if success else '✗ Failed'} in {exec_time:.1f}s",
            C.G if success else C.R)

        # Record experience to TransferLearning
        transfer_store.record_experience(
            query=task_description,
            agent=agent,
            action='execute',
            reward=1.0 if success else -0.5,
            success=success,
            error=None if success else "Simulated failure",
            context={'time': exec_time}
        )

        agents_used.append(agent)
        total_time += exec_time
        if not success:
            all_success = False

    # Simulate a fix being applied
    if random.random() > 0.7:
        log(f"\n🔧 Simulating error fix...", C.Y)
        terminal.record_fix(
            error="ModuleNotFoundError: No module named 'financial_data'",
            commands=["pip install financial-data-toolkit"],
            description="Install missing financial data package",
            source='pattern',
            success=True
        )
        log("   Fix recorded and saved for future use", C.G)

    # ==========================================================================
    # Record Session
    # ==========================================================================
    section("RECORDING SESSION LEARNINGS")

    transfer_store.record_session(
        task_description=task_description,
        agents_used=agents_used,
        total_time=total_time,
        success=all_success,
        stigmergy_signals=random.randint(5, 15),
        output_quality=random.uniform(0.7, 0.95) if all_success else 0.3
    )

    log(f"📝 Session recorded:", C.G)
    log(f"   Task: {task_description[:50]}...", C.C)
    log(f"   Agents: {', '.join(agents_used)}", C.C)
    log(f"   Total time: {total_time:.1f}s", C.C)
    log(f"   Success: {'Yes' if all_success else 'No'}", C.G if all_success else C.R)

    # ==========================================================================
    # Save All Learnings
    # ==========================================================================
    section("SAVING PERSISTENT LEARNINGS")

    transfer_store.save(str(transfer_path))
    terminal.save_fix_database()

    # Verify what was saved
    new_sessions = getattr(transfer_store, 'sessions', [])
    new_num_sessions = len(new_sessions)
    new_num_patterns = len(transfer_store.task_patterns)
    new_num_fixes = len(terminal._fix_cache)

    log(f"💾 Learnings saved to: {learning_dir}", C.G)
    log(f"\n📊 Updated Learning State:", C.C)
    log(f"   Sessions recorded: {new_num_sessions} (+{new_num_sessions - num_sessions})", C.C)
    log(f"   Task patterns: {new_num_patterns}", C.C)
    log(f"   Fixes in database: {new_num_fixes}", C.C)

    # Show role profiles if any emerged
    if transfer_store.role_profiles:
        log(f"\n🎯 Role Profiles (learned):", C.Y)
        for role, profile in transfer_store.role_profiles.items():
            strengths = ', '.join(profile.strengths) if profile.strengths else 'none yet'
            log(f"   {role}: strengths={strengths}", C.C)

    # ==========================================================================
    # Show what will be available next run
    # ==========================================================================
    section("WHAT HAPPENS NEXT RUN")

    log("When you run this demo again, Jotty will:", C.B)
    log("   1. Load all previous session learnings (TransferLearning)", C.C)
    log("   2. Match current task to similar past tasks (topic scoring)", C.C)
    log("   3. Suggest best agents based on performance history", C.C)
    log("   4. Provide execution strategy (order, skip, expected time)", C.C)
    log("   5. Auto-apply known fixes for common errors (SwarmTerminal)", C.C)

    # Final message
    print(f"\n{C.BOLD}{C.G}")
    if is_first_run:
        print("╔═════════════════════════════════════════════════════════════════╗")
        print("║  ✅ FIRST RUN COMPLETE - Run again to see learning in action!  ║")
        print("╚═════════════════════════════════════════════════════════════════╝")
    else:
        print("╔═════════════════════════════════════════════════════════════════╗")
        print(f"║  🧠 JOTTY IS LEARNING! Session #{new_num_sessions} recorded                  ║")
        print("╚═════════════════════════════════════════════════════════════════╝")
    print(f"{C.E}")

    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))

#!/usr/bin/env python3
"""
HYBRID MINIMAL TASK RUNNER - P2P Discovery + Sequential Delivery

Just provide:
1. Goal (what to build)
2. Data location (where data is)

Jotty will:
Phase 1 (P2P): Agents explore data in PARALLEL
Phase 2 (Sequential): Agents build system in ORDER using discoveries

NO prescriptive requirements!
"""

import asyncio
import dspy
import logging
from pathlib import Path
from datetime import datetime

from core.persistence.shared_context import SharedContext
from core.foundation.types.agent_types import SharedScratchpad, AgentMessage, CommunicationType
from core.persistence.scratchpad_persistence import ScratchpadPersistence
from core.integration.direct_claude_cli_lm import DirectClaudeCLI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiscoveryAgent(dspy.Signature):
    """P2P discovery agent that explores data in parallel."""
    goal: str = dspy.InputField(desc="Overall goal")
    data_location: str = dspy.InputField(desc="Where data is located")
    agent_role: str = dspy.InputField(desc="Your discovery focus area")
    other_discoveries: str = dspy.InputField(desc="Discoveries from other agents")
    discovery: str = dspy.OutputField(desc="Your discovery findings")


class DeliveryAgent(dspy.Signature):
    """Sequential delivery agent that builds using discoveries."""
    goal: str = dspy.InputField(desc="Overall goal")
    all_discoveries: str = dspy.InputField(desc="ALL findings from P2P phase")
    agent_role: str = dspy.InputField(desc="Your delivery role")
    previous_deliverable: str = dspy.InputField(desc="Output from previous agent")
    deliverable: str = dspy.OutputField(desc="Your deliverable (code/docs/tests)")


async def p2p_discovery_phase(
    goal: str,
    data_location: str,
    shared_context: SharedContext,
    scratchpad: SharedScratchpad,
    persistence: ScratchpadPersistence,
    session_file: Path
):
    """Phase 1: P2P Discovery - Agents explore data in PARALLEL."""

    print(f"\n{'='*90}")
    print(f"PHASE 1: P2P DISCOVERY - PARALLEL EXPLORATION")
    print(f"{'='*90}\n")

    # Discovery roles (will figure out their own focus!)
    discovery_roles = [
        "Explore the data files and identify key metrics",
        "Analyze data quality and define cleaning requirements",
        "Research screening criteria and recommend algorithms"
    ]

    async def run_discovery(agent_num: int, role: str):
        agent_name = f"Discovery Agent {agent_num}"

        print(f"🔍 {agent_name} starting...")
        print(f"   Focus: {role}\n")

        # Read other discoveries from scratchpad (P2P collaboration!)
        other_discoveries = "\n\n".join([
            f"{msg.sender}: {msg.content.get('summary', '')[:200]}"
            for msg in scratchpad.messages
            if msg.sender != agent_name and msg.message_type == CommunicationType.INSIGHT
        ]) if scratchpad.messages else "No discoveries yet (you're first!)"

        # Run discovery
        agent = dspy.ChainOfThought(DiscoveryAgent)

        start = datetime.now()
        result = agent(
            goal=goal,
            data_location=data_location,
            agent_role=role,
            other_discoveries=other_discoveries
        )
        discovery = result.discovery
        elapsed = (datetime.now() - start).total_seconds()

        print(f"✅ {agent_name} completed in {elapsed:.1f}s")
        print(f"   Generated: {len(discovery)} chars\n")

        # Post to scratchpad (for other P2P agents to see!)
        message = AgentMessage(
            sender=agent_name,
            receiver="*",  # Broadcast to all
            message_type=CommunicationType.INSIGHT,
            content={'summary': discovery[:200], 'full_discovery': discovery},
            insight=f"{agent_name} completed discovery"
        )
        scratchpad.add_message(message)
        persistence.save_message(session_file, message)

        # Store in shared context
        shared_context.set(f'discovery_{agent_num}', discovery)

        return {
            'agent': agent_name,
            'discovery': discovery,
            'time': elapsed
        }

    # Run all discoveries in PARALLEL (P2P!)
    print("🚀 Launching 3 discovery agents in PARALLEL...\n")
    tasks = [run_discovery(i, role) for i, role in enumerate(discovery_roles, 1)]
    results = await asyncio.gather(*tasks)

    discoveries = {r['agent']: r['discovery'] for r in results}
    total_time = max(r['time'] for r in results)

    print(f"\n{'='*90}")
    print(f"✅ PHASE 1 COMPLETE (P2P Discovery)")
    print(f"{'='*90}")
    print(f"  Discoveries: {len(discoveries)}")
    print(f"  Total time: {total_time:.1f}s (parallel execution!)")
    print(f"  Messages: {len(scratchpad.messages)}\n")

    return discoveries


async def sequential_delivery_phase(
    goal: str,
    discoveries: dict,
    shared_context: SharedContext,
    scratchpad: SharedScratchpad,
    persistence: ScratchpadPersistence,
    session_file: Path
):
    """Phase 2: Sequential Delivery - Agents build in ORDER."""

    print(f"\n{'='*90}")
    print(f"PHASE 2: SEQUENTIAL DELIVERY - ORDERED BUILD")
    print(f"{'='*90}\n")

    # Consolidate ALL discoveries
    all_discoveries = "\n\n".join([
        f"## {name}\n{findings}"
        for name, findings in discoveries.items()
    ])

    # Delivery roles (will figure out what to build!)
    delivery_roles = [
        "Implement data loading and cleaning code",
        "Implement screening algorithm and scoring",
        "Write tests and documentation"
    ]

    deliverables = {}
    previous_deliverable = "None (you're first!)"

    for i, role in enumerate(delivery_roles, 1):
        agent_name = f"Delivery Agent {i}"

        print(f"\n📋 {agent_name} ({i}/{len(delivery_roles)})")
        print(f"   Role: {role}\n")

        # Run delivery
        agent = dspy.ChainOfThought(DeliveryAgent)

        start = datetime.now()
        result = agent(
            goal=goal,
            all_discoveries=all_discoveries,  # ALL Phase 1 findings!
            agent_role=role,
            previous_deliverable=previous_deliverable
        )
        deliverable = result.deliverable
        elapsed = (datetime.now() - start).total_seconds()

        print(f"✅ {agent_name} completed in {elapsed:.1f}s")
        print(f"   Generated: {len(deliverable)} chars\n")

        # Post to scratchpad
        message = AgentMessage(
            sender=agent_name,
            receiver="*",
            message_type=CommunicationType.INSIGHT,
            content={'summary': deliverable[:200], 'sequence': i},
            insight=f"{agent_name} completed deliverable {i}"
        )
        scratchpad.add_message(message)
        persistence.save_message(session_file, message)

        # Store in shared context
        shared_context.set(f'deliverable_{i}', deliverable)

        deliverables[agent_name] = deliverable
        previous_deliverable = deliverable  # Pass to next agent!

    print(f"\n{'='*90}")
    print(f"✅ PHASE 2 COMPLETE (Sequential Delivery)")
    print(f"{'='*90}")
    print(f"  Deliverables: {len(deliverables)}\n")

    return deliverables


async def run_hybrid_minimal_task(
    goal: str,
    data_location: str
):
    """
    Run task with HYBRID workflow:
    - Phase 1: P2P discovery (parallel)
    - Phase 2: Sequential delivery (ordered)

    Args:
        goal: What to build (e.g., "Build a stock screener")
        data_location: Where data is
    """

    print("=" * 90)
    print("HYBRID MINIMAL TASK RUNNER - P2P + SEQUENTIAL")
    print("=" * 90)
    print(f"\nGoal: {goal}")
    print(f"Data: {data_location}")
    print("\nPhase 1: P2P Discovery (3 agents in parallel)")
    print("Phase 2: Sequential Delivery (3 agents in order)\n")

    # Configure
    lm = DirectClaudeCLI(model='sonnet')
    dspy.configure(lm=lm)

    print("✅ Claude CLI configured")

    # Infrastructure
    shared_context = SharedContext()
    scratchpad = SharedScratchpad()
    persistence = ScratchpadPersistence()

    session_name = f"hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_file = persistence.create_session(session_name)

    print("✅ Collaboration infrastructure initialized")
    print(f"   Session: {session_file}")
    print("-" * 90)

    # Store goal and data location
    shared_context.set('goal', goal)
    shared_context.set('data_location', data_location)

    # PHASE 1: P2P DISCOVERY
    discoveries = await p2p_discovery_phase(
        goal=goal,
        data_location=data_location,
        shared_context=shared_context,
        scratchpad=scratchpad,
        persistence=persistence,
        session_file=session_file
    )

    # PHASE 2: SEQUENTIAL DELIVERY
    deliverables = await sequential_delivery_phase(
        goal=goal,
        discoveries=discoveries,
        shared_context=shared_context,
        scratchpad=scratchpad,
        persistence=persistence,
        session_file=session_file
    )

    # Save final state
    persistence.save_scratchpad(session_file, scratchpad)

    # Generate report
    print(f"\n{'='*90}")
    print("TASK COMPLETE - HYBRID WORKFLOW")
    print("=" * 90)

    print(f"\n📊 Summary:")
    print(f"  Phase 1 (P2P): {len(discoveries)} discoveries")
    print(f"  Phase 2 (Sequential): {len(deliverables)} deliverables")
    print(f"  Total Messages: {len(scratchpad.messages)}")
    print(f"  Session: {session_file}")

    # Save outputs
    output_file = Path(f"HYBRID_TASK_OUTPUT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    doc = f"""# Hybrid Task Output: {goal}

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Data**: {data_location}
**Workflow**: Hybrid (P2P Discovery + Sequential Delivery)
**Session**: {session_file.name}

---

## Phase 1: P2P Discoveries (Parallel Exploration)

"""

    for agent_name, discovery in discoveries.items():
        doc += f"""### {agent_name}

{discovery}

---

"""

    doc += """
## Phase 2: Sequential Deliverables (Ordered Build)

"""

    for agent_name, deliverable in deliverables.items():
        doc += f"""### {agent_name}

{deliverable}

---

"""

    doc += f"""
## Collaboration Summary

**Total Messages**: {len(scratchpad.messages)}
**Session File**: {session_file}

---

## What This Demonstrates

### ✅ Hybrid Workflow Pattern
- **Phase 1 (P2P)**: 3 agents explored data in PARALLEL
- **Phase 2 (Sequential)**: 3 agents built system in ORDER using ALL discoveries
- **Best of both worlds!**

### ✅ True Collaboration
- P2P agents read from SharedScratchpad (not isolated!)
- Sequential agents build on previous deliverables
- All agents have access to ALL Phase 1 findings

### ✅ Minimal Specification
- Input: Just goal + data_location (2 lines!)
- Agents discovered: What to explore, what to build, how to implement
- NO prescriptive requirements!

**This is Jotty with hybrid workflow!** 🚀
"""

    output_file.write_text(doc)

    print(f"\n📄 Output saved: {output_file}")

    # Export session
    markdown = persistence.export_session(session_file, format='markdown')
    export_file = Path(f"{session_file.stem}_session.md")
    export_file.write_text(markdown)

    print(f"📄 Session exported: {export_file}")

    print("\n" + "=" * 90)

    return True


async def main():
    """Run with minimal specification."""

    # MINIMAL INPUT - Just 2 lines!
    goal = "Build a stock market screening system to find undervalued growth stocks"
    data_location = "/var/www/sites/personal/stock_market/common/Data/FUNDAMENTALS"

    # Let Jotty figure out the rest with HYBRID workflow!
    await run_hybrid_minimal_task(
        goal=goal,
        data_location=data_location
    )


if __name__ == "__main__":
    print("\n🚀 Hybrid Minimal Task Runner")
    print("P2P Discovery (parallel) + Sequential Delivery (ordered)")
    print("Give Jotty a goal and data location - it figures out the rest!\n")

    response = input("Ready? (y/n): ")
    if response.lower() == 'y':
        asyncio.run(main())
    else:
        print("Cancelled")

#!/usr/bin/env python3
"""
Stock Screener Task Executor

Runs the stock screener task using Jotty's hybrid workflow:
- Phase 1 (P2P Discovery): 4 agents research in parallel
- Phase 2 (Sequential Delivery): 6 agents build in order

This demonstrates Jotty building a complete system!
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


class StockScreenerDiscovery(dspy.Signature):
    """Discovery research for stock screener."""
    role: str = dspy.InputField(desc="Agent role (Financial Analyst, Ratio Expert, etc.)")
    data_location: str = dspy.InputField(desc="Path to financial data")
    other_findings: str = dspy.InputField(desc="Findings from other agents")
    findings: str = dspy.OutputField(desc="Discovery findings and recommendations")


class StockScreenerDelivery(dspy.Signature):
    """Deliverable for stock screener system."""
    role: str = dspy.InputField(desc="Agent role (Requirements Engineer, Data Engineer, etc.)")
    discoveries: str = dspy.InputField(desc="Findings from Phase 1")
    previous_deliverable: str = dspy.InputField(desc="Output from previous agent")
    deliverable: str = dspy.OutputField(desc="Code, documentation, or tests")


async def phase1_discovery(
    shared_context: SharedContext,
    scratchpad: SharedScratchpad,
    persistence: ScratchpadPersistence,
    session_file: Path
):
    """Phase 1: P2P Discovery - 4 agents research in parallel."""

    print(f"\n{'='*90}")
    print(f"PHASE 1: P2P DISCOVERY - PARALLEL RESEARCH")
    print(f"{'='*90}\n")

    data_location = "/var/www/sites/personal/stock_market/common/Data/FUNDAMENTALS/"

    # Define discovery agents
    discovery_roles = [
        {
            'name': 'Financial Data Analyst',
            'focus': 'Analyze BalanceSheet, PnL, Cashflow data. Identify key metrics for screening.'
        },
        {
            'name': 'Ratio & Valuation Expert',
            'focus': 'Define undervaluation criteria (P/E, P/B, etc.) and growth indicators from Ratio_data.xlsx'
        },
        {
            'name': 'Technical Analyst',
            'focus': 'Analyze Technical.csv for momentum indicators and filters to avoid value traps'
        },
        {
            'name': 'System Architect',
            'focus': 'Design data pipeline architecture, recommend Python libraries, plan scalability'
        }
    ]

    async def run_discovery(role_config):
        agent_name = role_config['name']
        focus = role_config['focus']

        print(f"🔍 {agent_name} - Starting discovery...")
        print(f"   Focus: {focus}")

        # Read other findings
        other_findings = []
        for msg in scratchpad.messages:
            if msg.sender != agent_name and msg.message_type == CommunicationType.INSIGHT:
                other_findings.append(f"{msg.sender}: {msg.content.get('summary', '')}")

        other_findings_text = "\n".join(other_findings) if other_findings else "No findings yet (first agent)"

        # Run discovery
        agent = dspy.ChainOfThought(StockScreenerDiscovery)

        start = datetime.now()
        result = agent(
            role=f"{agent_name}: {focus}",
            data_location=data_location,
            other_findings=other_findings_text
        )
        findings = result.findings
        elapsed = (datetime.now() - start).total_seconds()

        print(f"✅ {agent_name} - Completed in {elapsed:.1f}s ({len(findings)} chars)")

        # Post to scratchpad
        message = AgentMessage(
            sender=agent_name,
            receiver="*",
            message_type=CommunicationType.INSIGHT,
            content={
                'summary': findings[:200],
                'full_findings': findings,
                'focus_area': focus
            },
            insight=f"{agent_name} completed discovery"
        )
        scratchpad.add_message(message)
        persistence.save_message(session_file, message)

        # Store in shared context
        context_key = f"{agent_name.lower().replace(' ', '_')}_findings"
        shared_context.set(context_key, findings)

        return {
            'agent': agent_name,
            'findings': findings,
            'time': elapsed
        }

    # Run all discoveries in parallel
    tasks = [run_discovery(role) for role in discovery_roles]
    results = await asyncio.gather(*tasks)

    discoveries = {r['agent']: r['findings'] for r in results}

    print(f"\n{'='*90}")
    print(f"✅ PHASE 1 COMPLETE")
    print(f"{'='*90}")
    print(f"  Discoveries: {len(discoveries)}")
    print(f"  Messages: {len(scratchpad.messages)}")

    return discoveries


async def phase2_delivery(
    discoveries: dict,
    shared_context: SharedContext,
    scratchpad: SharedScratchpad,
    persistence: ScratchpadPersistence,
    session_file: Path
):
    """Phase 2: Sequential Delivery - 6 agents build in order."""

    print(f"\n{'='*90}")
    print(f"PHASE 2: SEQUENTIAL DELIVERY - ORDERED BUILD")
    print(f"{'='*90}\n")

    # Consolidate discoveries
    all_discoveries = "\n\n".join([
        f"## {name}\n{findings}"
        for name, findings in discoveries.items()
    ])

    # Define delivery agents (sequential order matters!)
    delivery_roles = [
        {
            'name': 'Requirements Engineer',
            'role': 'Write complete PRD: user stories, acceptance criteria, API/CLI spec, success metrics'
        },
        {
            'name': 'Data Engineer',
            'role': 'Create data_loader.py: Load Excel files, clean data, calculate features, validate quality'
        },
        {
            'name': 'Screening Engine Developer',
            'role': 'Create screening_engine.py: Multi-criteria filter, scoring algorithm, ranking system'
        },
        {
            'name': 'Backend Developer',
            'role': 'Create api.py: CLI interface, result formatting, CSV/JSON export'
        },
        {
            'name': 'Test Engineer',
            'role': 'Create test files: Unit tests, integration tests, backtesting framework'
        },
        {
            'name': 'Documentation Writer',
            'role': 'Create README: Usage guide, methodology, examples, deployment instructions'
        }
    ]

    deliverables = {}
    previous_deliverable = "None (first deliverable)"

    for i, role_config in enumerate(delivery_roles):
        agent_name = role_config['name']
        role_desc = role_config['role']

        print(f"\n📋 {agent_name} ({i+1}/{len(delivery_roles)})")
        print(f"   Role: {role_desc}")

        # Run delivery
        agent = dspy.ChainOfThought(StockScreenerDelivery)

        start = datetime.now()
        result = agent(
            role=f"{agent_name}: {role_desc}",
            discoveries=all_discoveries,
            previous_deliverable=previous_deliverable
        )
        deliverable = result.deliverable
        elapsed = (datetime.now() - start).total_seconds()

        print(f"✅ Generated in {elapsed:.1f}s ({len(deliverable)} chars)")

        # Post to scratchpad
        message = AgentMessage(
            sender=agent_name,
            receiver="*",
            message_type=CommunicationType.INSIGHT,
            content={
                'summary': deliverable[:200],
                'deliverable_type': role_desc.split(':')[0],
                'sequence': i + 1
            },
            insight=f"{agent_name} completed deliverable {i+1}"
        )
        scratchpad.add_message(message)
        persistence.save_message(session_file, message)

        # Store in shared context
        context_key = f"{agent_name.lower().replace(' ', '_')}_deliverable"
        shared_context.set(context_key, deliverable)

        deliverables[agent_name] = deliverable
        previous_deliverable = deliverable  # Pass to next agent

    print(f"\n{'='*90}")
    print(f"✅ PHASE 2 COMPLETE")
    print(f"{'='*90}")
    print(f"  Deliverables: {len(deliverables)}")

    return deliverables


async def run_stock_screener_task():
    """Execute the stock screener task with hybrid workflow."""

    print("=" * 90)
    print("STOCK SCREENER TASK - JOTTY BUILDING A SYSTEM!")
    print("=" * 90)
    print("\nUsing Hybrid Workflow: P2P Discovery + Sequential Delivery\n")

    # Configure Claude CLI
    lm = DirectClaudeCLI(model='sonnet')
    dspy.configure(lm=lm)

    print("✅ Claude 3.5 Sonnet configured")

    # Initialize collaboration infrastructure
    shared_context = SharedContext()
    scratchpad = SharedScratchpad()
    persistence = ScratchpadPersistence()

    # Create session
    session_name = f"stock_screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_file = persistence.create_session(session_name)

    print("✅ Collaboration infrastructure initialized")
    print(f"   Session: {session_file}")
    print("-" * 90)

    # PHASE 1: P2P DISCOVERY
    discoveries = await phase1_discovery(
        shared_context=shared_context,
        scratchpad=scratchpad,
        persistence=persistence,
        session_file=session_file
    )

    # PHASE 2: SEQUENTIAL DELIVERY
    deliverables = await phase2_delivery(
        discoveries=discoveries,
        shared_context=shared_context,
        scratchpad=scratchpad,
        persistence=persistence,
        session_file=session_file
    )

    # Save final state
    persistence.save_scratchpad(session_file, scratchpad)

    # Generate final report
    print("\n" + "=" * 90)
    print("TASK COMPLETE - STOCK SCREENER BUILT!")
    print("=" * 90)

    print(f"\n📊 Summary:")
    print(f"  Phase 1 (Discovery): {len(discoveries)} research findings")
    print(f"  Phase 2 (Delivery): {len(deliverables)} deliverables")
    print(f"  Total Messages: {len(scratchpad.messages)}")
    print(f"  Session: {session_file}")

    # Export full session
    markdown_export = persistence.export_session(session_file, format='markdown')
    export_file = Path(f"STOCK_SCREENER_BUILD_{datetime.now().strftime('%Y%m%d')}.md")
    export_file.write_text(markdown_export)

    print(f"  Export: {export_file}")

    # Save deliverables
    deliverables_file = Path("STOCK_SCREENER_DELIVERABLES.md")
    doc = f"""# Stock Screener System - Built by Jotty

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Workflow**: Hybrid (P2P Discovery + Sequential Delivery)
**Session**: {session_file.name}

---

## Phase 1: Discovery Findings

"""
    for agent_name, findings in discoveries.items():
        doc += f"### {agent_name}\n\n{findings}\n\n---\n\n"

    doc += "## Phase 2: Deliverables\n\n"

    for agent_name, deliverable in deliverables.items():
        doc += f"### {agent_name}\n\n{deliverable}\n\n---\n\n"

    doc += """
## What This Demonstrates

### ✅ Jotty Building a Complete System
- 4 discovery agents researched in parallel
- 6 delivery agents built sequentially
- Real collaboration via SharedScratchpad
- Real persistence to disk
- Complete stock screener system generated!

### ✅ Hybrid Workflow Pattern
- Phase 1 (P2P): Parallel exploration of problem space
- Phase 2 (Sequential): Ordered build using discoveries
- Best of both worlds!

### ✅ Meta-System
- Multi-agent system building another system
- Agents designed the architecture
- Agents wrote the code
- Agents created tests and docs

**This is Jotty in action!** 🚀
"""

    deliverables_file.write_text(doc)

    print(f"  Deliverables: {deliverables_file}")

    print("\n" + "=" * 90)

    return True


async def main():
    try:
        success = await run_stock_screener_task()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        exit(130)
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    print("\n🚀 Stock Screener Task Executor")
    print("Jotty will build a complete stock screening system using hybrid workflow\n")
    print("This will take ~60-90 minutes with real Claude CLI\n")

    response = input("Ready to let Jotty build the stock screener? (y/n): ")
    if response.lower() == 'y':
        asyncio.run(main())
    else:
        print("Cancelled")

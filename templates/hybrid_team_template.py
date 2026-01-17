#!/usr/bin/env python3
"""
HYBRID TEAM TEMPLATE - P2P Discovery + Sequential Delivery

Combines best of both patterns:
1. **P2P Phase (Discovery)**: Multiple agents explore/research in parallel
2. **Sequential Phase (Delivery)**: Use discoveries to build in ordered steps

Use this when:
- Problem needs exploration before solution (unknown requirements)
- Multiple perspectives valuable (research, competitive analysis, user needs)
- After discovery, clear build order emerges (design → implement → test)

Example Workflows:

1. **Product Development**:
   - P2P Discovery: Market researcher, Competitor analyst, User researcher (parallel)
   - Sequential Delivery: PM → UX → Designer → Frontend → Backend → QA

2. **Security Audit**:
   - P2P Discovery: Auth expert, API expert, Data expert, Infrastructure expert (parallel)
   - Sequential Delivery: Prioritizer → Fixer → Tester → Documenter

3. **ML Model Development**:
   - P2P Discovery: Data explorer, Feature engineer, Algorithm researcher (parallel)
   - Sequential Delivery: Data prep → Model training → Evaluation → Deployment

4. **Content Creation**:
   - P2P Discovery: Topic researchers, Keyword analyzers, Trend watchers (parallel)
   - Sequential Delivery: Outliner → Writer → Editor → Publisher

Pattern:
    Phase 1 (P2P):     Agent A ↘
                       Agent B → SharedScratchpad → Insights
                       Agent C ↗

    Phase 2 (Sequential): Insights → Agent D → Agent E → Agent F → Final Output

Benefits:
+ Fast discovery (parallel exploration)
+ Comprehensive insights (multiple perspectives)
+ Ordered delivery (no chaos in implementation)
+ Best of both worlds!
"""

import asyncio
import dspy
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from core.persistence.shared_context import SharedContext
from core.foundation.types.agent_types import SharedScratchpad, AgentMessage, CommunicationType
from core.persistence.scratchpad_persistence import ScratchpadPersistence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def p2p_discovery_phase(
    agents_config: List[Dict[str, Any]],
    task: str,
    shared_context: SharedContext,
    scratchpad: SharedScratchpad,
    persistence: ScratchpadPersistence,
    session_file: Path
) -> Dict[str, Any]:
    """
    Phase 1: P2P Discovery - All agents explore in parallel.

    Args:
        agents_config: List of agent configs [{name, agent, expert}, ...]
        task: Discovery task
        shared_context: Shared data store
        scratchpad: Message passing workspace
        persistence: Scratchpad persistence manager
        session_file: Session file for persistence

    Returns:
        {
            'discoveries': {agent_name: output, ...},
            'total_insights': count,
            'total_messages': count
        }
    """

    print(f"\n{'='*90}")
    print(f"PHASE 1: P2P DISCOVERY - PARALLEL EXPLORATION")
    print(f"{'='*90}\n")

    print(f"📋 Discovery Task: {task[:100]}...")
    print(f"👥 Agents: {len(agents_config)}")
    print()

    # Store task in shared context
    shared_context.set('discovery_task', task)

    async def run_discovery_agent(config: Dict[str, Any]) -> Dict[str, Any]:
        """Run single discovery agent."""

        agent_name = config['name']
        agent = config['agent']
        expert = config.get('expert')

        print(f"🔍 {agent_name} - Starting discovery...")

        # Read other discoveries from scratchpad
        other_discoveries = []
        for msg in scratchpad.messages:
            if msg.sender != agent_name and msg.message_type == CommunicationType.INSIGHT:
                other_discoveries.append(f"{msg.sender}: {msg.content.get('summary', '')}")

        context_text = "\n".join(other_discoveries) if other_discoveries else "No discoveries yet"

        # Generate discovery
        start = datetime.now()
        result = agent(
            task=task,
            context=context_text
        )
        output = result.output  # Adjust based on signature
        elapsed = (datetime.now() - start).total_seconds()

        print(f"✅ {agent_name} - Discovered in {elapsed:.1f}s ({len(output)} chars)")

        # Post to scratchpad
        message = AgentMessage(
            sender=agent_name,
            receiver="*",
            message_type=CommunicationType.INSIGHT,
            content={'summary': output[:200], 'full_text': output},
            insight=f"{agent_name} completed discovery"
        )
        scratchpad.add_message(message)
        scratchpad.shared_insights.append(f"{agent_name}: {output[:100]}")

        # Save to persistence
        persistence.save_message(session_file, message)

        # Store in shared context
        context_key = f"{agent_name.lower().replace(' ', '_')}_discovery"
        shared_context.set(context_key, output)

        return {
            'agent': agent_name,
            'output': output,
            'time': elapsed
        }

    # Run all discovery agents in parallel
    tasks = [run_discovery_agent(config) for config in agents_config]
    results = await asyncio.gather(*tasks)

    # Consolidate discoveries
    discoveries = {r['agent']: r['output'] for r in results}

    print(f"\n{'='*90}")
    print(f"✅ PHASE 1 COMPLETE - P2P DISCOVERY")
    print(f"{'='*90}")
    print(f"  Discoveries: {len(discoveries)}")
    print(f"  Insights: {len(scratchpad.shared_insights)}")
    print(f"  Messages: {len(scratchpad.messages)}")

    return {
        'discoveries': discoveries,
        'total_insights': len(scratchpad.shared_insights),
        'total_messages': len(scratchpad.messages),
        'results': results
    }


async def sequential_delivery_phase(
    agents_config: List[Dict[str, Any]],
    discoveries: Dict[str, str],
    shared_context: SharedContext,
    scratchpad: SharedScratchpad,
    persistence: ScratchpadPersistence,
    session_file: Path
) -> Dict[str, Any]:
    """
    Phase 2: Sequential Delivery - Build in order using discoveries.

    Args:
        agents_config: List of agent configs for delivery
        discoveries: Discoveries from Phase 1
        shared_context: Shared data store
        scratchpad: Message passing workspace
        persistence: Scratchpad persistence manager
        session_file: Session file for persistence

    Returns:
        {
            'deliverables': {agent_name: output, ...},
            'total_iterations': count
        }
    """

    print(f"\n{'='*90}")
    print(f"PHASE 2: SEQUENTIAL DELIVERY - ORDERED BUILD")
    print(f"{'='*90}\n")

    print(f"📦 Deliverables: {len(agents_config)}")
    print(f"🔗 Using discoveries from Phase 1")
    print()

    deliverables = {}
    previous_output = None

    for i, config in enumerate(agents_config):
        agent_name = config['name']
        agent = config['agent']

        print(f"\n📋 {agent_name} ({i+1}/{len(agents_config)})")

        # Build context from discoveries + previous deliverable
        all_discoveries = "\n\n".join([f"{name}: {text[:500]}" for name, text in discoveries.items()])

        if previous_output:
            context = f"""DISCOVERIES FROM PHASE 1:
{all_discoveries}

PREVIOUS DELIVERABLE:
{previous_output}
"""
        else:
            context = f"""DISCOVERIES FROM PHASE 1:
{all_discoveries}
"""

        # Generate deliverable
        start = datetime.now()
        result = agent(
            context=context,
            previous_output=previous_output or "First deliverable"
        )
        output = result.output
        elapsed = (datetime.now() - start).total_seconds()

        print(f"✅ Generated in {elapsed:.1f}s ({len(output)} chars)")

        # Post to scratchpad
        message = AgentMessage(
            sender=agent_name,
            receiver="*",
            message_type=CommunicationType.INSIGHT,
            content={'summary': output[:200], 'deliverable': True},
            insight=f"{agent_name} completed deliverable {i+1}"
        )
        scratchpad.add_message(message)

        # Save to persistence
        persistence.save_message(session_file, message)

        # Store in shared context
        context_key = f"{agent_name.lower().replace(' ', '_')}_deliverable"
        shared_context.set(context_key, output)

        deliverables[agent_name] = output
        previous_output = output  # Pass to next agent

    print(f"\n{'='*90}")
    print(f"✅ PHASE 2 COMPLETE - SEQUENTIAL DELIVERY")
    print(f"{'='*90}")
    print(f"  Deliverables: {len(deliverables)}")

    return {
        'deliverables': deliverables,
        'total_iterations': len(agents_config)
    }


async def hybrid_workflow():
    """
    Hybrid workflow: P2P Discovery → Sequential Delivery

    Example:
    - Phase 1 (P2P): 3 researchers explore in parallel
    - Phase 2 (Sequential): Designer → Developer → Tester build in order
    """

    print("=" * 90)
    print("HYBRID WORKFLOW - P2P DISCOVERY + SEQUENTIAL DELIVERY")
    print("=" * 90)
    print("\nPhase 1: Parallel exploration | Phase 2: Ordered build\n")

    # Configure LLM
    from core.integration.direct_claude_cli_lm import DirectClaudeCLI

    lm = DirectClaudeCLI(model='sonnet')
    dspy.configure(lm=lm)

    print("✅ LLM configured")

    # Initialize collaboration infrastructure
    shared_context = SharedContext()
    scratchpad = SharedScratchpad()
    persistence = ScratchpadPersistence()

    # Create session
    session_file = persistence.create_session("hybrid_workflow")

    print("✅ Collaboration infrastructure initialized")
    print(f"   Session: {session_file}")
    print("-" * 90)

    # Define task
    task = """
    Build a stock market screening system that identifies undervalued growth stocks.

    Discovery needed:
    - What metrics indicate undervaluation?
    - What defines growth potential?
    - What data sources are available?

    Delivery needed:
    - System design
    - Implementation plan
    - Testing strategy
    """

    # PHASE 1: P2P DISCOVERY AGENTS
    # TODO: Replace with your actual discovery agents
    class DiscoverySignature(dspy.Signature):
        """Discovery research."""
        task: str = dspy.InputField()
        context: str = dspy.InputField()
        output: str = dspy.OutputField(desc="Discovery findings")

    discovery_agents = [
        {'name': 'Market Researcher', 'agent': dspy.ChainOfThought(DiscoverySignature)},
        {'name': 'Data Analyst', 'agent': dspy.ChainOfThought(DiscoverySignature)},
        {'name': 'Domain Expert', 'agent': dspy.ChainOfThought(DiscoverySignature)},
    ]

    # Run Phase 1
    phase1_results = await p2p_discovery_phase(
        agents_config=discovery_agents,
        task=task,
        shared_context=shared_context,
        scratchpad=scratchpad,
        persistence=persistence,
        session_file=session_file
    )

    # PHASE 2: SEQUENTIAL DELIVERY AGENTS
    class DeliverySignature(dspy.Signature):
        """Sequential delivery."""
        context: str = dspy.InputField()
        previous_output: str = dspy.InputField()
        output: str = dspy.OutputField(desc="Deliverable")

    delivery_agents = [
        {'name': 'System Designer', 'agent': dspy.ChainOfThought(DeliverySignature)},
        {'name': 'Implementation Planner', 'agent': dspy.ChainOfThought(DeliverySignature)},
        {'name': 'Test Strategist', 'agent': dspy.ChainOfThought(DeliverySignature)},
    ]

    # Run Phase 2
    phase2_results = await sequential_delivery_phase(
        agents_config=delivery_agents,
        discoveries=phase1_results['discoveries'],
        shared_context=shared_context,
        scratchpad=scratchpad,
        persistence=persistence,
        session_file=session_file
    )

    # Save final scratchpad state
    persistence.save_scratchpad(session_file, scratchpad)

    # Generate report
    print("\n" + "=" * 90)
    print("HYBRID WORKFLOW COMPLETE")
    print("=" * 90)

    print(f"\n📊 Summary:")
    print(f"  Phase 1 (P2P Discovery): {len(phase1_results['discoveries'])} discoveries")
    print(f"  Phase 2 (Sequential Delivery): {len(phase2_results['deliverables'])} deliverables")
    print(f"  Total Messages: {len(scratchpad.messages)}")
    print(f"  Session File: {session_file}")

    # Export session
    markdown_export = persistence.export_session(session_file, format='markdown')
    export_file = Path(f"{session_file.stem}_export.md")
    export_file.write_text(markdown_export)

    print(f"  Exported: {export_file}")

    print("=" * 90)

    return True


async def main():
    try:
        success = await hybrid_workflow()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        exit(130)
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    print("\n🚀 Hybrid Template: P2P Discovery + Sequential Delivery")
    print("Best of both worlds: Parallel exploration → Ordered build\n")

    response = input("Ready to run? (y/n): ")
    if response.lower() == 'y':
        asyncio.run(main())
    else:
        print("Cancelled")

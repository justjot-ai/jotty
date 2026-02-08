#!/usr/bin/env python3
"""
MINIMAL TASK RUNNER - Let Jotty Figure It Out!

Just provide:
1. Goal (what to build)
2. Data location (where data is)
3. Number of agents

Jotty will:
- Explore the data
- Decide what to build
- Build it
- Test it

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


class MinimalAgent(dspy.Signature):
    """Generic agent that figures out what to do."""
    goal: str = dspy.InputField(desc="Overall goal")
    data_location: str = dspy.InputField(desc="Where data is located")
    agent_number: int = dspy.InputField(desc="Agent number (1, 2, 3...)")
    total_agents: int = dspy.InputField(desc="Total number of agents")
    previous_outputs: str = dspy.InputField(desc="Outputs from previous agents")
    scratchpad_messages: str = dspy.InputField(desc="Messages from other agents")
    output: str = dspy.OutputField(desc="Your contribution")


async def run_minimal_task(
    goal: str,
    data_location: str,
    num_agents: int = 6
):
    """
    Run task with minimal specification.

    Args:
        goal: What to build (e.g., "Build a stock screener")
        data_location: Where data is
        num_agents: How many agents to use (default 6)
    """

    print("=" * 90)
    print("MINIMAL TASK RUNNER - LET JOTTY FIGURE IT OUT!")
    print("=" * 90)
    print(f"\nGoal: {goal}")
    print(f"Data: {data_location}")
    print(f"Agents: {num_agents}")
    print("\nLetting agents discover what to build...\n")

    # Configure
    lm = DirectClaudeCLI(model='sonnet')
    dspy.configure(lm=lm)

    # Infrastructure
    shared_context = SharedContext()
    scratchpad = SharedScratchpad()
    persistence = ScratchpadPersistence()

    session_name = f"minimal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_file = persistence.create_session(session_name)

    print("✅ Infrastructure ready")
    print(f"   Session: {session_file}")
    print("-" * 90)

    # Store goal and data location
    shared_context.set('goal', goal)
    shared_context.set('data_location', data_location)

    # Run agents sequentially (they decide what phase they're in!)
    agent = dspy.ChainOfThought(MinimalAgent)
    outputs = []

    for i in range(1, num_agents + 1):
        print(f"\n{'='*90}")
        print(f"AGENT {i}/{num_agents}")
        print(f"{'='*90}\n")

        # Collect previous outputs
        previous = "\n\n".join([
            f"Agent {j}: {out}"
            for j, out in enumerate(outputs, 1)
        ]) if outputs else "None (you're first!)"

        # Collect scratchpad messages
        messages = "\n".join([
            f"{msg.sender}: {msg.content.get('summary', str(msg.content)[:200])}"
            for msg in scratchpad.messages[-5:]  # Last 5 messages
        ]) if scratchpad.messages else "No messages yet"

        print(f"📖 Context available:")
        print(f"   Previous agents: {len(outputs)}")
        print(f"   Scratchpad messages: {len(scratchpad.messages)}")
        print()

        # Let agent decide what to do!
        print(f"🤖 Agent {i} deciding what to contribute...")

        start = datetime.now()
        result = agent(
            goal=goal,
            data_location=data_location,
            agent_number=i,
            total_agents=num_agents,
            previous_outputs=previous,
            scratchpad_messages=messages
        )
        output = result.output
        elapsed = (datetime.now() - start).total_seconds()

        print(f"✅ Completed in {elapsed:.1f}s")
        print(f"   Generated: {len(output)} chars")
        print(f"   Preview: {output[:150]}...")

        # Save to scratchpad
        message = AgentMessage(
            sender=f"Agent {i}",
            receiver="*",
            message_type=CommunicationType.INSIGHT,
            content={'summary': output[:200], 'full_output': output},
            insight=f"Agent {i} contributed"
        )
        scratchpad.add_message(message)
        persistence.save_message(session_file, message)

        # Store in context
        shared_context.set(f'agent_{i}_output', output)

        outputs.append(output)

        print(f"\n{'='*90}")

    # Save final state
    persistence.save_scratchpad(session_file, scratchpad)

    # Generate report
    print("\n" + "=" * 90)
    print("TASK COMPLETE")
    print("=" * 90)

    print(f"\n📊 Summary:")
    print(f"  Agents: {num_agents}")
    print(f"  Messages: {len(scratchpad.messages)}")
    print(f"  Session: {session_file}")

    # Save outputs
    output_file = Path(f"TASK_OUTPUT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    doc = f"""# Task Output: {goal}

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Data**: {data_location}
**Agents**: {num_agents}
**Session**: {session_file.name}

---

## Agent Outputs

"""

    for i, out in enumerate(outputs, 1):
        doc += f"""### Agent {i}

{out}

---

"""

    doc += f"""
## Collaboration

**Total Messages**: {len(scratchpad.messages)}
**Session File**: {session_file}

---

*Generated by Minimal Task Runner - Jotty figured out what to build!*
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

    # Let Jotty figure out the rest!
    await run_minimal_task(
        goal=goal,
        data_location=data_location,
        num_agents=6  # How many agents to use
    )


if __name__ == "__main__":
    print("\n🚀 Minimal Task Runner")
    print("Give Jotty a goal and data location - it figures out the rest!\n")

    response = input("Ready? (y/n): ")
    if response.lower() == 'y':
        asyncio.run(main())
    else:
        print("Cancelled")

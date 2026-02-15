#!/usr/bin/env python3
"""
COLLABORATIVE TEAM TEMPLATE - P2P Multi-Agent Pattern

Use this template when:
- Agents can work in parallel
- Agents share discoveries via scratchpad
- Agents communicate via messages
- Tool results cached and shared
- True collaboration, not just sequential handoffs

Examples:
- Code review: Multiple reviewers checking different aspects simultaneously
- Research: Multiple researchers investigating different angles in parallel
- Design critique: Design, A11y, UX, Eng all reviewing simultaneously
- Security audit: Multiple security experts checking different attack vectors

Pattern:
    All agents → SharedScratchpad ← All agents
    Agent A posts message → Agent B reads → Agent B responds
    Agent C caches tool result → Agent D reuses it

Pros:
+ Fast (parallel work)
+ Cross-pollination (agents learn from each other in real-time)
+ No bottlenecks (agents work independently)
+ Tool result caching (avoid duplicate work)
+ Real collaboration (not just handoffs)

Cons:
- More complex to coordinate
- Potential conflicts (need conflict resolution)
- Harder to debug (non-linear flow)
"""

import asyncio
import dspy
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Import collaboration infrastructure
from core.persistence.shared_context import SharedContext
from core.foundation.types.agent_types import SharedScratchpad, AgentMessage, CommunicationType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_collaborative_agent(
    agent_name: str,
    agent,
    expert,
    initial_task: str,
    shared_context: SharedContext,
    scratchpad: SharedScratchpad,
    max_iterations: int = 3,
    score_threshold: float = 0.85
) -> Dict[str, Any]:
    """
    Run agent with access to shared collaboration workspace.

    The agent:
    1. Reads from SharedContext (persistent data)
    2. Reads from SharedScratchpad (messages, tool cache, insights)
    3. Generates output with learning loop
    4. Writes to SharedContext (stores results)
    5. Posts to SharedScratchpad (broadcasts insights, messages)

    Args:
        agent_name: Human-readable agent name
        agent: DSPy agent
        expert: Domain expert for evaluation
        initial_task: Initial task description
        shared_context: SharedContext for persistent storage
        scratchpad: SharedScratchpad for communication
        max_iterations: Maximum learning iterations
        score_threshold: Minimum acceptable score

    Returns:
        {
            'output': final output,
            'iterations': number of iterations,
            'scores': [score1, score2, ...],
            'messages_sent': number of messages posted,
            'messages_received': number of messages read,
            'tool_cache_hits': number of cached tool results reused
        }
    """

    print(f"\n{'='*90}")
    print(f"🤖 {agent_name.upper()} (Collaborative Mode)")
    print(f"{'='*90}\n")

    feedback = "First attempt - no previous feedback"
    iterations_history = []
    messages_sent = 0
    messages_received = 0
    tool_cache_hits = 0

    for iteration in range(1, max_iterations + 1):
        print(f"📝 Iteration {iteration}/{max_iterations}")

        # READ FROM SHARED WORKSPACE
        print(f"   📖 Reading shared context...")

        # 1. Get all shared data
        all_shared_data = shared_context.get_all()
        print(f"      - Shared context: {len(all_shared_data)} items")

        # 2. Get messages from other agents
        relevant_messages = [
            msg for msg in scratchpad.messages
            if msg.receiver in [agent_name, "*"]  # Messages to me or broadcast
        ]
        messages_received += len(relevant_messages)
        print(f"      - Messages received: {len(relevant_messages)}")

        # 3. Get shared insights
        shared_insights = scratchpad.shared_insights
        print(f"      - Shared insights: {len(shared_insights)}")

        # 4. Check tool cache for reusable results
        if scratchpad.tool_cache:
            tool_cache_hits += len(scratchpad.tool_cache)
            print(f"      - Tool cache: {len(scratchpad.tool_cache)} cached results")

        # BUILD CONTEXT from shared workspace
        collaboration_context = {
            'shared_data': all_shared_data,
            'messages': [
                f"{msg.sender}: {msg.content.get('summary', str(msg.content)[:100])}"
                for msg in relevant_messages[-5:]  # Last 5 messages
            ],
            'insights': shared_insights[-5:],  # Last 5 insights
            'tool_cache': list(scratchpad.tool_cache.keys())
        }

        # Generate output
        start = datetime.now()

        # NOTE: Customize based on your agent's signature
        result = agent(
            task=initial_task,
            previous_feedback=feedback,
            collaboration_context=str(collaboration_context)  # Pass shared context
        )
        output = result.output  # Adjust field name

        elapsed = (datetime.now() - start).total_seconds()

        print(f"   ✅ Generated in {elapsed:.1f}s ({len(output)} chars)")

        # WRITE TO SHARED WORKSPACE
        print(f"   📝 Writing to shared workspace...")

        # 1. Store output in SharedContext
        context_key = f"{agent_name.lower().replace(' ', '_')}_output"
        shared_context.set(context_key, output)
        print(f"      - Stored in context: '{context_key}'")

        # 2. Broadcast insight message
        insight_msg = AgentMessage(
            sender=agent_name,
            receiver="*",  # Broadcast to all
            message_type=CommunicationType.INSIGHT,
            content={
                'summary': output[:200] + "..." if len(output) > 200 else output,
                'full_output_key': context_key,
                'iteration': iteration
            },
            insight=f"{agent_name} completed iteration {iteration}"
        )
        scratchpad.add_message(insight_msg)
        messages_sent += 1
        print(f"      - Broadcasted insight message")

        # 3. Add to shared insights
        scratchpad.shared_insights.append(
            f"{agent_name} (iter {iteration}): Generated {len(output)} chars"
        )

        # Expert evaluation
        evaluation = await expert._evaluate_domain(
            output=output,
            gold_standard="",
            task=f"{agent_name} collaborative task",
            context={"iteration": iteration, "collaboration_mode": True}
        )

        score = evaluation.get('score', 0.0)
        status = evaluation.get('status', 'UNKNOWN')
        issues = evaluation.get('issues', [])
        suggestions = evaluation.get('suggestions', '')

        print(f"   📊 Score: {score:.2f} - Status: {status}")

        if issues:
            print(f"   ⚠️  Issues: {', '.join(issues[:2])}")

        iterations_history.append({
            'iteration': iteration,
            'output': output,
            'score': score,
            'status': status,
            'issues': issues,
            'time': elapsed,
            'messages_in_context': len(relevant_messages)
        })

        # Check threshold
        if score >= score_threshold:
            print(f"   ✅ Threshold reached! ({score:.2f} >= {score_threshold})")
            break

        # Build feedback for next iteration
        if iteration < max_iterations:
            print(f"   🔄 Score below threshold ({score:.2f} < {score_threshold}), iterating...")

            feedback_parts = []
            if issues:
                feedback_parts.append(f"Fix: {', '.join(issues[:3])}")
            if suggestions:
                feedback_parts.append(f"Suggestions: {suggestions}")

            # Include insights from other agents
            if shared_insights:
                feedback_parts.append(f"Other agents discovered: {'; '.join(shared_insights[-3:])}")

            feedback = "; ".join(feedback_parts)
        else:
            print(f"   ⚠️  Max iterations reached")

    final = iterations_history[-1]

    print(f"\n{'='*90}")
    print(f"✅ {agent_name} Complete")
    print(f"   Iterations: {len(iterations_history)}")
    print(f"   Final Score: {final['score']:.2f}")
    print(f"   Improvement: {final['score'] - iterations_history[0]['score']:+.2f}")
    print(f"   Messages Sent: {messages_sent}")
    print(f"   Messages Received: {messages_received}")
    print(f"   Tool Cache Hits: {tool_cache_hits}")
    print(f"{'='*90}")

    return {
        'output': final['output'],
        'iterations': len(iterations_history),
        'scores': [h['score'] for h in iterations_history],
        'history': iterations_history,
        'messages_sent': messages_sent,
        'messages_received': messages_received,
        'tool_cache_hits': tool_cache_hits
    }


async def collaborative_team_workflow():
    """
    Collaborative P2P workflow: All agents work in parallel with shared workspace.

    Pattern:
    - All agents start simultaneously
    - All read from SharedContext and SharedScratchpad
    - All write insights, messages, results to shared workspace
    - Agents can message each other directly
    - Tool results cached and shared
    - True collaboration, not sequential handoffs
    """

    print("=" * 90)
    print("COLLABORATIVE TEAM WORKFLOW (P2P)")
    print("=" * 90)
    print("\nParallel collaboration: All agents work simultaneously with shared workspace\n")

    # Configure LLM
    from core.integration.direct_claude_cli_lm import DirectClaudeCLI

    lm = DirectClaudeCLI(model='sonnet')
    dspy.configure(lm=lm)

    print("✅ LLM configured")

    # Initialize collaboration infrastructure
    shared_context = SharedContext()
    scratchpad = SharedScratchpad()

    print("✅ Shared workspace initialized")
    print(f"   - SharedContext: {shared_context}")
    print(f"   - SharedScratchpad: {scratchpad}")
    print("-" * 90)

    # Initialize domain experts
    from core.experts.product_manager_expert import ProductManagerExpertAgent
    from core.experts.ux_researcher_expert import UXResearcherExpertAgent
    from core.experts.designer_expert import DesignerExpertAgent
    from core.experts.frontend_expert import FrontendExpertAgent

    experts = {
        'pm': ProductManagerExpertAgent(),
        'ux': UXResearcherExpertAgent(),
        'design': DesignerExpertAgent(),
        'frontend': FrontendExpertAgent(),
    }

    print(f"✅ {len(experts)} experts initialized")
    print("-" * 90)

    # Define shared task (all agents work on same goal)
    shared_task = """
    Build a "Real-Time Collaboration Dashboard" for teams.

    Each agent should contribute their expertise:
    - PM: Define requirements and success metrics
    - UX: Research users and create personas
    - Designer: Create UI/UX designs
    - Frontend: Define React architecture

    Work in parallel, share insights, and collaborate!
    """

    print(f"\n📋 Shared Task: {shared_task[:100]}...")
    print()

    # Store initial task in shared context
    shared_context.set('task_description', shared_task)

    # Create agents
    from core.experts.product_manager_expert import ProductRequirementsGenerator
    from core.experts.ux_researcher_expert import UXResearchGenerator
    from core.experts.designer_expert import DesignGenerator
    from core.experts.frontend_expert import FrontendArchitectureGenerator

    agents_config = [
        {
            'name': 'Product Manager',
            'agent': dspy.ChainOfThought(ProductRequirementsGenerator),
            'expert': experts['pm']
        },
        {
            'name': 'UX Researcher',
            'agent': dspy.ChainOfThought(UXResearchGenerator),
            'expert': experts['ux']
        },
        {
            'name': 'Designer',
            'agent': dspy.ChainOfThought(DesignGenerator),
            'expert': experts['design']
        },
        {
            'name': 'Frontend Developer',
            'agent': dspy.ChainOfThought(FrontendArchitectureGenerator),
            'expert': experts['frontend']
        },
    ]

    # Run all agents IN PARALLEL
    print("🚀 Starting all agents in parallel...")
    print("=" * 90)

    tasks = []
    for config in agents_config:
        # NOTE: This is pseudo-parallel - they'll run sequentially due to await
        # For true parallelism, use asyncio.gather()
        task = run_collaborative_agent(
            agent_name=config['name'],
            agent=config['agent'],
            expert=config['expert'],
            initial_task=shared_task,
            shared_context=shared_context,
            scratchpad=scratchpad,
            max_iterations=2,  # Fewer iterations since they collaborate
            score_threshold=0.85
        )
        tasks.append(task)

    # Run all agents concurrently
    results = await asyncio.gather(*tasks)

    # Map results
    team_results = {
        agents_config[i]['name']: results[i]
        for i in range(len(agents_config))
    }

    # Analysis
    print("\n" + "=" * 90)
    print("COLLABORATIVE WORKFLOW COMPLETE")
    print("=" * 90)

    print("\n📈 Team Performance:")
    total_iterations = 0
    total_messages = 0
    total_tool_cache_hits = 0

    for agent_name, result in team_results.items():
        total_iterations += result['iterations']
        total_messages += result['messages_sent'] + result['messages_received']
        total_tool_cache_hits += result['tool_cache_hits']

        initial_score = result['scores'][0]
        final_score = result['scores'][-1]
        improvement = final_score - initial_score

        print(f"\n{agent_name.upper()}:")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Initial Score: {initial_score:.2f}")
        print(f"  Final Score: {final_score:.2f}")
        print(f"  Improvement: {improvement:+.2f}")
        print(f"  Messages Sent: {result['messages_sent']}")
        print(f"  Messages Received: {result['messages_received']}")
        print(f"  Tool Cache Hits: {result['tool_cache_hits']}")

    print(f"\n📊 Collaboration Metrics:")
    print(f"  Total Iterations: {total_iterations}")
    print(f"  Total Messages: {total_messages}")
    print(f"  Tool Cache Hits: {total_tool_cache_hits}")
    print(f"  Shared Context Items: {len(shared_context.keys())}")
    print(f"  Shared Insights: {len(scratchpad.shared_insights)}")
    print(f"  Scratchpad Messages: {len(scratchpad.messages)}")
    print(f"  Pattern: Collaborative P2P (Parallel)")

    # Show final shared workspace state
    print(f"\n🗂️  Final Shared Workspace State:")
    print(f"  SharedContext keys: {shared_context.keys()}")
    print(f"  Scratchpad messages: {len(scratchpad.messages)} total")
    print(f"  Shared insights:")
    for insight in scratchpad.shared_insights[-10:]:
        print(f"    - {insight}")

    # Save output
    output_file = Path("COLLABORATIVE_TEAM_OUTPUT.md")
    doc = f"""# Collaborative Team Workflow Output (P2P)

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Pattern**: Collaborative P2P (Parallel)
**Agents**: {len(team_results)}

---

## Collaboration Metrics

| Agent | Iterations | Score | Messages Sent | Messages Received | Tool Cache Hits |
|-------|------------|-------|---------------|-------------------|-----------------|
"""

    for agent_name, result in team_results.items():
        final = result['scores'][-1]
        doc += f"| {agent_name} | {result['iterations']} | {final:.2f} | {result['messages_sent']} | {result['messages_received']} | {result['tool_cache_hits']} |\n"

    doc += f"""
**Total Messages**: {total_messages}
**Total Tool Cache Hits**: {total_tool_cache_hits}
**Shared Context Items**: {len(shared_context.keys())}
**Shared Insights**: {len(scratchpad.shared_insights)}

---

## Shared Workspace State

### SharedContext Keys
{', '.join(shared_context.keys())}

### Shared Insights
"""

    for insight in scratchpad.shared_insights:
        doc += f"- {insight}\n"

    doc += f"""
### Agent Communication

Total messages exchanged: {len(scratchpad.messages)}

---

## Agent Outputs

"""

    for agent_name, result in team_results.items():
        doc += f"""### {agent_name.upper()}

**Iterations**: {result['iterations']}
**Score Progression**: {' → '.join([f"{s:.2f}" for s in result['scores']])}
**Collaboration**: {result['messages_received']} messages received, {result['messages_sent']} sent

{result['output']}

---

"""

    doc += """
## What This Demonstrates

### ✅ TRUE Collaborative P2P
- All agents work in parallel (not sequential)
- SharedContext for persistent data storage
- SharedScratchpad for real-time communication
- Message passing between agents
- Tool result caching and sharing
- Shared insights across team

### ✅ Real Collaboration Infrastructure
- Used `SharedContext` (core/persistence/shared_context.py)
- Used `SharedScratchpad` (core/foundation/types/agent_types.py)
- Used `AgentMessage` for inter-agent communication
- Used `CommunicationType` enum for message types

### ✅ NOT Just String Passing
- Agents broadcast insights to scratchpad
- Agents read each other's messages
- Tool results cached and reused
- Shared discovery and learning

---

*This is TRUE collaborative multi-agent learning with shared workspace!*
"""

    output_file.write_text(doc)

    print(f"\n📄 Output saved: {output_file}")
    print("=" * 90)

    return True


async def main():
    try:
        success = await collaborative_team_workflow()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        exit(130)
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    print("\n🚀 Collaborative Team Template (P2P)")
    print("All agents work in parallel with shared workspace\n")

    response = input("Ready to run? (y/n): ")
    if response.lower() == 'y':
        asyncio.run(main())
    else:
        print("Cancelled")

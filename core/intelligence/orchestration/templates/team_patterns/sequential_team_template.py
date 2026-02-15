#!/usr/bin/env python3
"""
SEQUENTIAL TEAM TEMPLATE - Waterfall Multi-Agent Pattern

Use this template when:
- Agents must work in strict order (A → B → C → D)
- Each agent depends on previous agent's complete output
- No parallel work possible (dependencies are linear)
- Simple handoff pattern (like assembly line)

Examples:
- Product development: PM → UX → Design → Frontend → Backend → QA
- Content creation: Research → Outline → Draft → Edit → Review → Publish
- Data pipeline: Extract → Transform → Validate → Load → Index → Query

Pattern:
    Agent 1 completes → Agent 2 uses output → Agent 3 uses Agent 2's output → ...

Pros:
+ Simple to understand and debug
+ Clear dependencies and order
+ Easy to track progress (linear)

Cons:
- Slow (no parallelism)
- Bottlenecks (one slow agent blocks everyone)
- No cross-pollination (agents can't learn from each other in real-time)
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import dspy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_agent_with_learning(
    agent_name: str,
    agent,
    expert,
    input_context: str,
    max_iterations: int = 3,
    score_threshold: float = 0.85,
) -> Dict[str, Any]:
    """
    Run a single agent with learning loop.

    The agent iterates until:
    - Score >= threshold, OR
    - Max iterations reached

    Args:
        agent_name: Human-readable agent name (e.g., "Product Manager")
        agent: DSPy agent (ChainOfThought signature)
        expert: Domain expert for evaluation
        input_context: Input data (from previous agent or initial task)
        max_iterations: Maximum learning iterations
        score_threshold: Minimum acceptable score (0.0-1.0)

    Returns:
        {
            'output': final output string,
            'iterations': number of iterations,
            'scores': [score1, score2, ...],
            'history': [iteration details...]
        }
    """

    print(f"\n{'='*90}")
    print(f"🤖 {agent_name.upper()}")
    print(f"{'='*90}\n")

    feedback = "First attempt - no previous feedback"
    iterations_history = []

    for iteration in range(1, max_iterations + 1):
        print(f"📝 Iteration {iteration}/{max_iterations}")
        print(
            f"   Feedback: {feedback[:100]}..."
            if len(feedback) > 100
            else f"   Feedback: {feedback}"
        )

        # Generate output
        start = datetime.now()

        # NOTE: Customize this based on your agent's signature
        # This example assumes agent has 'input' and 'previous_feedback' fields
        result = agent(input=input_context, previous_feedback=feedback)
        output = result.output  # Adjust field name based on your signature

        elapsed = (datetime.now() - start).total_seconds()

        print(f"   ✅ Generated in {elapsed:.1f}s ({len(output)} chars)")

        # Expert evaluation
        evaluation = await expert._evaluate_domain(
            output=output,
            gold_standard="",  # Optional: provide gold standard for comparison
            task=f"{agent_name} task",
            context={"iteration": iteration},
        )

        score = evaluation.get("score", 0.0)
        status = evaluation.get("status", "UNKNOWN")
        issues = evaluation.get("issues", [])
        suggestions = evaluation.get("suggestions", "")

        print(f"   📊 Score: {score:.2f} - Status: {status}")

        if issues:
            print(f"   ⚠️  Issues: {', '.join(issues[:2])}")

        iterations_history.append(
            {
                "iteration": iteration,
                "output": output,
                "score": score,
                "status": status,
                "issues": issues,
                "time": elapsed,
            }
        )

        # Check if threshold reached
        if score >= score_threshold:
            print(f"   ✅ Threshold reached! ({score:.2f} >= {score_threshold})")
            break

        # Build feedback for next iteration
        if iteration < max_iterations:
            print(f"   🔄 Score below threshold ({score:.2f} < {score_threshold}), iterating...")

            feedback_parts = []
            if issues:
                feedback_parts.append(f"Fix these issues: {', '.join(issues[:3])}")
            if suggestions:
                feedback_parts.append(f"Suggestions: {suggestions}")
            feedback_parts.append(f"Previous score: {score:.2f}, aim for {score_threshold:.2f}+")

            feedback = "; ".join(feedback_parts)
        else:
            print("   ⚠️  Max iterations reached")

    final = iterations_history[-1]

    print(f"\n{'='*90}")
    print(f"✅ {agent_name} Complete")
    print(f"   Iterations: {len(iterations_history)}")
    print(f"   Final Score: {final['score']:.2f}")
    print(f"   Improvement: {final['score'] - iterations_history[0]['score']:+.2f}")
    print(f"{'='*90}")

    return {
        "output": final["output"],
        "iterations": len(iterations_history),
        "scores": [h["score"] for h in iterations_history],
        "history": iterations_history,
    }


async def sequential_team_workflow():
    """
    Sequential team workflow: Agent 1 → Agent 2 → Agent 3 → ...

    Each agent:
    1. Receives output from previous agent (or initial task)
    2. Generates output with learning loop
    3. Passes output to next agent

    This is a WATERFALL pattern - strict sequential order.
    """

    print("=" * 90)
    print("SEQUENTIAL TEAM WORKFLOW")
    print("=" * 90)
    print("\nWaterfall pattern: Each agent waits for previous to complete\n")

    # Configure LLM (adjust to your setup)
    from core.integration.direct_claude_cli_lm import DirectClaudeCLI

    lm = DirectClaudeCLI(model="sonnet")
    dspy.configure(lm=lm)

    print("✅ LLM configured")
    print("-" * 90)

    # Initialize your domain experts
    # IMPLEMENT: Replace with your actual experts
    from core.experts.product_manager_expert import ProductManagerExpertAgent
    from core.experts.ux_researcher_expert import UXResearcherExpertAgent

    # ... add more as needed

    pm_expert = ProductManagerExpertAgent()
    ux_expert = UXResearcherExpertAgent()
    # ... initialize other experts

    print("✅ Experts initialized")
    print("-" * 90)

    # Define initial task
    initial_task = """
    Your initial task description here.
    This is what the first agent will work on.
    """

    print(f"\n📋 Task: {initial_task[:100]}...")
    print()

    # Track all results
    team_results = {}

    # AGENT 1: First Agent (Foundation)
    # IMPLEMENT: Replace with your actual agent signature
    from core.experts.product_manager_expert import ProductRequirementsGenerator

    agent1 = dspy.ChainOfThought(ProductRequirementsGenerator)

    result1 = await run_agent_with_learning(
        agent_name="Agent 1",
        agent=agent1,
        expert=pm_expert,
        input_context=initial_task,
        max_iterations=3,
        score_threshold=0.85,
    )
    team_results["agent1"] = result1

    # AGENT 2: Second Agent (Builds on Agent 1)
    # IMPLEMENT: Replace with your actual agent signature
    from core.experts.ux_researcher_expert import UXResearchGenerator

    agent2 = dspy.ChainOfThought(UXResearchGenerator)

    result2 = await run_agent_with_learning(
        agent_name="Agent 2",
        agent=agent2,
        expert=ux_expert,
        input_context=result1["output"],  # ← SEQUENTIAL: Uses Agent 1's output
        max_iterations=3,
        score_threshold=0.85,
    )
    team_results["agent2"] = result2

    # AGENT 3, 4, 5... Continue pattern
    # Each agent receives previous agent's output

    # Analysis
    print("\n" + "=" * 90)
    print("SEQUENTIAL WORKFLOW COMPLETE")
    print("=" * 90)

    print("\n📈 Team Performance:")
    total_iterations = 0
    for agent_id, result in team_results.items():
        total_iterations += result["iterations"]
        initial_score = result["scores"][0]
        final_score = result["scores"][-1]
        improvement = final_score - initial_score

        print(f"\n{agent_id.upper()}:")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Initial Score: {initial_score:.2f}")
        print(f"  Final Score: {final_score:.2f}")
        print(f"  Improvement: {improvement:+.2f}")
        print(f"  Score progression: {' → '.join([f'{s:.2f}' for s in result['scores']])}")

    print("\n📊 Overall Metrics:")
    print(f"  Total Iterations: {total_iterations}")
    print(f"  Agents: {len(team_results)}")
    print("  Pattern: Sequential (Waterfall)")
    print(f"  Coordination Events: {len(team_results) - 1} (each agent built on previous)")

    # Save output
    output_file = Path("SEQUENTIAL_TEAM_OUTPUT.md")
    doc = f"""# Sequential Team Workflow Output

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Pattern**: Sequential (Waterfall)
**Agents**: {len(team_results)}

---

## Team Metrics

| Agent | Iterations | Initial Score | Final Score | Improvement |
|-------|------------|---------------|-------------|-------------|
"""

    for agent_id, result in team_results.items():
        initial = result["scores"][0]
        final = result["scores"][-1]
        improvement = final - initial
        doc += f"| {agent_id} | {result['iterations']} | {initial:.2f} | {final:.2f} | {improvement:+.2f} |\n"

    doc += f"""
**Total Iterations**: {total_iterations}
**Coordination**: Sequential handoffs

---

## Agent Outputs

"""

    for agent_id, result in team_results.items():
        doc += f"""### {agent_id.upper()}

**Iterations**: {result['iterations']}
**Score Progression**: {' → '.join([f"{s:.2f}" for s in result['scores']])}

{result['output']}

---

"""

    output_file.write_text(doc)

    print(f"\n📄 Output saved: {output_file}")
    print("=" * 90)

    return True


async def main():
    try:
        success = await sequential_team_workflow()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        exit(130)
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    print("\n🚀 Sequential Team Template")
    print("Waterfall pattern: Each agent waits for previous to complete\n")

    response = input("Ready to run? (y/n): ")
    if response.lower() == "y":
        asyncio.run(main())
    else:
        print("Cancelled")

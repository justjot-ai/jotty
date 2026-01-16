"""
End-to-End Phase 8 Tests - Real Execution with DSPy/Claude
===========================================================

Tests SingleAgentOrchestrator and MultiAgentsOrchestrator with actual LLM execution:
1. SAS without expert (regular agent)
2. SAS with expert (gold standard learning)
3. MAS without team templates (manual coordination)
4. MAS with team templates (pre-configured teams)
"""

import sys
import os
import asyncio
import logging

# Add Jotty to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import dspy
from core.orchestration import SingleAgentOrchestrator, MultiAgentsOrchestrator
from core.foundation import JottyConfig, AgentConfig as ActorConfig
from core.experts.expert_templates import create_mermaid_expert, create_sql_expert
from core.orchestration.team_templates import create_diagram_team, create_custom_team


# =============================================================================
# TEST SETUP
# =============================================================================

def setup_dspy():
    """Configure DSPy with Claude or OpenAI via unified LM interface."""
    try:
        # Try to configure with Claude via LiteLLM
        # Format: "provider/model" - uses LiteLLM under the hood
        lm = dspy.LM(model="anthropic/claude-3-5-sonnet-20241022", max_tokens=1000)
        dspy.configure(lm=lm)
        logger.info("✓ DSPy configured with Claude (via LiteLLM)")
        return True
    except Exception as e:
        logger.warning(f"⚠️  Could not configure Claude: {e}")
        try:
            # Fallback to OpenAI if available
            lm = dspy.LM(model="openai/gpt-4", max_tokens=1000)
            dspy.configure(lm=lm)
            logger.info("✓ DSPy configured with OpenAI (fallback, via LiteLLM)")
            return True
        except Exception as e2:
            logger.error(f"❌ Could not configure DSPy: {e2}")
            logger.info("⚠️  Skipping E2E tests (no LLM available)")
            return False


# =============================================================================
# TEST 1: SAS WITHOUT EXPERT (Regular Agent)
# =============================================================================

async def test_sas_regular_agent():
    """Test SingleAgentOrchestrator as regular agent (no gold standards)."""

    print("\n" + "="*70)
    print("TEST 1: SingleAgentOrchestrator - Regular Agent (No Expert)")
    print("="*70)

    # Define a simple signature
    class SimpleQASignature(dspy.Signature):
        """Answer simple questions."""
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    # Create minimal config
    config = JottyConfig(
        base_path="/tmp/jotty_test_regular",
        enable_validation=False,  # Disable validation for simple test
        enable_rl=False  # Disable RL for simple test
    )

    # Create regular agent (NO gold standard learning)
    agent = SingleAgentOrchestrator(
        agent=dspy.ChainOfThought(SimpleQASignature),
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[],
        config=config,
        # No gold standard learning
        enable_gold_standard_learning=False
    )

    print("\n📋 Configuration:")
    print(f"   - Agent Type: Regular (no expert features)")
    print(f"   - Gold Standard Learning: {agent.enable_gold_standard_learning}")
    print(f"   - Optimization Pipeline: {agent.optimization_pipeline}")

    # Test execution
    print("\n🚀 Running agent...")
    try:
        result = await agent.arun(question="What is 2+2?")

        print("\n✅ Execution Result:")
        print(f"   - Success: {result.success}")
        print(f"   - Output: {result.output}")
        print(f"   - Trajectory Steps: {len(result.trajectory)}")

        return True
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        return False


# =============================================================================
# TEST 2: SAS WITH EXPERT (Gold Standard Learning)
# =============================================================================

async def test_sas_expert_agent():
    """Test SingleAgentOrchestrator as expert agent (with gold standards)."""

    print("\n" + "="*70)
    print("TEST 2: SingleAgentOrchestrator - Expert Agent (Gold Standards)")
    print("="*70)

    # Define validation function
    def validate_mermaid(output: str) -> bool:
        """Simple validator - check if output contains mermaid keywords."""
        keywords = ["graph", "sequenceDiagram", "classDiagram", "flowchart", "%%"]
        return any(keyword in output for keyword in keywords)

    # Gold standard examples
    gold_standards = [
        {
            "input": {"description": "Simple flowchart", "diagram_type": "flowchart"},
            "expected_output": "flowchart TD\n    A[Start] --> B[End]"
        },
        {
            "input": {"description": "Sequence diagram", "diagram_type": "sequence"},
            "expected_output": "sequenceDiagram\n    User->>System: Request"
        }
    ]

    # Create config
    config = JottyConfig(
        base_path="/tmp/jotty_test_expert",
        enable_validation=False,  # Disable validation for simple test
        enable_rl=False  # Disable RL for simple test
    )

    # Define signature
    class MermaidSignature(dspy.Signature):
        """Generate Mermaid diagram code."""
        description: str = dspy.InputField(desc="Diagram description")
        diagram_type: str = dspy.InputField(desc="Type of diagram")
        mermaid_code: str = dspy.OutputField(desc="Mermaid code")

    # Create expert agent (WITH gold standard learning)
    agent = SingleAgentOrchestrator(
        agent=dspy.ChainOfThought(MermaidSignature),
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[],
        config=config,

        # 🎓 Expert features
        enable_gold_standard_learning=True,
        gold_standards=gold_standards,
        domain="mermaid",
        domain_validator=validate_mermaid,
        max_training_iterations=3
    )

    print("\n📋 Configuration:")
    print(f"   - Agent Type: Expert (with gold standards)")
    print(f"   - Gold Standard Learning: {agent.enable_gold_standard_learning}")
    print(f"   - Domain: {agent.domain}")
    print(f"   - Gold Standards: {len(agent.gold_standards)} examples")
    print(f"   - Optimization Pipeline: {'Initialized' if agent.optimization_pipeline else 'Failed (expected)'}")

    # Test execution
    print("\n🚀 Running expert agent...")
    try:
        result = await agent.arun(
            description="User login process",
            diagram_type="flowchart"
        )

        print("\n✅ Execution Result:")
        print(f"   - Success: {result.success}")
        print(f"   - Output: {str(result.output)[:100]}...")
        print(f"   - Validated: {validate_mermaid(str(result.output))}")

        return True
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST 3: MAS WITHOUT TEAM TEMPLATES (Manual Multi-Agent)
# =============================================================================

async def test_mas_manual_coordination():
    """Test MultiAgentsOrchestrator without team templates (manual setup)."""

    print("\n" + "="*70)
    print("TEST 3: MultiAgentsOrchestrator - Manual Coordination")
    print("="*70)

    # Create config
    config = JottyConfig(
        base_path="/tmp/jotty_test_mas_manual",
        enable_validation=False,
        enable_rl=False
    )

    # Define signatures
    class Agent1Signature(dspy.Signature):
        """First agent - data analysis."""
        question: str = dspy.InputField()
        analysis: str = dspy.OutputField()

    class Agent2Signature(dspy.Signature):
        """Second agent - visualization recommendation."""
        analysis: str = dspy.InputField()
        visualization: str = dspy.OutputField()

    # Create agents manually
    agent1 = SingleAgentOrchestrator(
        agent=dspy.ChainOfThought(Agent1Signature),
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[],
        config=config
    )

    agent2 = SingleAgentOrchestrator(
        agent=dspy.ChainOfThought(Agent2Signature),
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[],
        config=config
    )

    # Create actor configs manually
    actors = [
        ActorConfig(
            name="Analyst",
            agent=agent1,
            enable_architect=False,
            enable_auditor=False,
            metadata={"description": "Data analyst"}
        ),
        ActorConfig(
            name="Visualizer",
            agent=agent2,
            enable_architect=False,
            enable_auditor=False,
            metadata={"description": "Visualization expert"}
        )
    ]

    print("\n📋 Configuration:")
    print(f"   - Team Type: Manual (no team template)")
    print(f"   - Actors: {len(actors)}")
    for actor in actors:
        desc = actor.metadata.get("description", "No description") if actor.metadata else "No description"
        print(f"     - {actor.name}: {desc}")

    # Create orchestrator manually
    try:
        orchestrator = MultiAgentsOrchestrator(
            actors=actors,
            metadata_provider=None,
            config=config
        )

        print("\n🚀 Running multi-agent orchestration...")
        result = await orchestrator.run(
            goal="Analyze sales data and suggest visualizations"
        )

        print("\n✅ Orchestration Result:")
        print(f"   - Success: {result.success}")
        print(f"   - Actor Outputs: {len(result.actor_outputs)}")
        print(f"   - Actors: {result.list_actors()}")

        return True
    except Exception as e:
        print(f"\n❌ Orchestration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# TEST 4: MAS WITH TEAM TEMPLATES (Pre-configured Team)
# =============================================================================

async def test_mas_team_templates():
    """Test MultiAgentsOrchestrator with team templates (pre-configured)."""

    print("\n" + "="*70)
    print("TEST 4: MultiAgentsOrchestrator - Team Templates")
    print("="*70)

    # Create config
    config = JottyConfig(
        base_path="/tmp/jotty_test_mas_team",
        enable_validation=False,
        enable_rl=False
    )

    print("\n📋 Creating team using template...")

    try:
        # Use team template (pre-configured)
        team = create_custom_team(
            actors=[
                ActorConfig(
                    name="MermaidExpert",
                    agent=create_mermaid_expert(config=config),
                    enable_architect=False,
                    enable_auditor=False,
                    metadata={"description": "Expert in Mermaid diagrams"}
                )
            ],
            config=config,
            metadata_provider=None
        )

        print("\n📋 Configuration:")
        print(f"   - Team Type: Pre-configured (team template)")
        print(f"   - Team Factory: create_custom_team()")
        print(f"   - Expert Agents: 1 (Mermaid)")
        print(f"   - Team Actors: {len(team.actors)}")

        print("\n🚀 Running team orchestration...")
        result = await team.run(
            goal="Generate a sequence diagram for user authentication"
        )

        print("\n✅ Team Result:")
        print(f"   - Success: {result.success}")
        print(f"   - Actor Outputs: {len(result.actor_outputs)}")
        print(f"   - Actors: {result.list_actors()}")

        return True
    except Exception as e:
        print(f"\n❌ Team execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

async def run_all_tests():
    """Run all end-to-end tests."""

    print("\n" + "="*70)
    print("Phase 8 End-to-End Tests - Real Execution")
    print("="*70)

    # Setup DSPy
    if not setup_dspy():
        print("\n⚠️  Skipping E2E tests (LLM not available)")
        print("   Configure ANTHROPIC_API_KEY or OPENAI_API_KEY to run tests")
        return

    results = {}

    # Test 1: SAS without expert
    print("\n" + "-"*70)
    results['sas_regular'] = await test_sas_regular_agent()

    # Test 2: SAS with expert
    print("\n" + "-"*70)
    results['sas_expert'] = await test_sas_expert_agent()

    # Test 3: MAS without team templates
    print("\n" + "-"*70)
    results['mas_manual'] = await test_mas_manual_coordination()

    # Test 4: MAS with team templates
    print("\n" + "-"*70)
    results['mas_team'] = await test_mas_team_templates()

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All end-to-end tests passed!")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed")


if __name__ == "__main__":
    # Run tests
    asyncio.run(run_all_tests())

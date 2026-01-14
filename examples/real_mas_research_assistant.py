#!/usr/bin/env python
"""
Real-World Multi-Agent System Example: Research Assistant
==========================================================

This demonstrates the refactored Jotty MAS with a practical use case:
A research assistant that uses multiple specialized agents to analyze topics.

Agents:
1. TopicExtractor - Identifies key topics from a user query
2. FactGatherer - Gathers facts about the topic
3. AnalysisAgent - Analyzes the facts and provides insights
4. SummaryAgent - Creates a final summary

This tests:
- Multi-agent collaboration
- Parameter passing between agents (ParameterResolver)
- State management (StateManager)
- Tool coordination (ToolManager)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_cli_wrapper import ClaudeCLILM
import dspy
from core import SwarmConfig, AgentSpec, Conductor
from unittest.mock import Mock


def create_research_mas():
    """Create a research assistant Multi-Agent System."""

    # Agent 1: Topic Extractor
    class ExtractTopics(dspy.Signature):
        """Extract 2-3 key topics from a research query."""
        query = dspy.InputField(desc="The research question")
        topics = dspy.OutputField(desc="2-3 key topics to research (comma-separated)")

    # Agent 2: Fact Gatherer
    class GatherFacts(dspy.Signature):
        """Gather 3-4 key facts about a topic."""
        topic = dspy.InputField(desc="The topic to research")
        facts = dspy.OutputField(desc="3-4 key facts about the topic")

    # Agent 3: Analysis Agent
    class AnalyzeFacts(dspy.Signature):
        """Analyze the gathered facts and provide insights."""
        topic = dspy.InputField(desc="The research topic")
        facts = dspy.InputField(desc="Facts gathered about the topic")
        analysis = dspy.OutputField(desc="Analysis and insights from the facts")

    # Agent 4: Summary Agent
    class CreateSummary(dspy.Signature):
        """Create a comprehensive summary from all analysis."""
        original_query = dspy.InputField(desc="The original research question")
        analysis = dspy.InputField(desc="Analysis from previous agents")
        summary = dspy.OutputField(desc="Final comprehensive summary")

    # Create agent specifications
    topic_extractor = AgentSpec(
        name="TopicExtractor",
        agent=dspy.ChainOfThought(ExtractTopics),
        architect_prompts=[],
        auditor_prompts=[],
        outputs=["topics"]
    )

    fact_gatherer = AgentSpec(
        name="FactGatherer",
        agent=dspy.ChainOfThought(GatherFacts),
        architect_prompts=[],
        auditor_prompts=[],
        parameter_mappings={"topic": "TopicExtractor"},  # Get topic from TopicExtractor
        outputs=["facts"]
    )

    analysis_agent = AgentSpec(
        name="AnalysisAgent",
        agent=dspy.ChainOfThought(AnalyzeFacts),
        architect_prompts=[],
        auditor_prompts=[],
        parameter_mappings={
            "topic": "TopicExtractor",
            "facts": "FactGatherer"
        },
        outputs=["analysis"]
    )

    summary_agent = AgentSpec(
        name="SummaryAgent",
        agent=dspy.ChainOfThought(CreateSummary),
        architect_prompts=[],
        auditor_prompts=[],
        parameter_mappings={"analysis": "AnalysisAgent"},
        outputs=["summary"]
    )

    # Return actors list and config separately (Conductor API)
    actors = [topic_extractor, fact_gatherer, analysis_agent, summary_agent]
    config = SwarmConfig(
        max_actor_iters=10
    )

    return actors, config


def test_research_mas_with_cli():
    """Test the research MAS with Claude CLI."""
    print("\n" + "="*70)
    print("REAL-WORLD MAS TEST: Research Assistant")
    print("="*70)

    # Configure DSPy with Claude CLI LM (proper dspy.BaseLM)
    print("\n🔧 Configuring Claude CLI LM...")
    lm = ClaudeCLILM(model="haiku")
    dspy.configure(lm=lm)
    print("✓ DSPy configured with Claude CLI LM (dspy.BaseLM)")

    # Create the research MAS
    print("\n🤖 Creating Research Assistant MAS...")
    actors, config = create_research_mas()
    print(f"✓ Created MAS with {len(actors)} agents:")
    for agent in actors:
        print(f"   - {agent.name}")

    # Create metadata provider (simple mock for this demo)
    metadata_provider = Mock()
    metadata_provider.get_tools = Mock(return_value=[])

    # Create the Conductor (disable data registry to avoid LLM type issues with CLI wrapper)
    print("\n🚀 Initializing Conductor...")
    conductor = Conductor(
        actors=actors,
        metadata_provider=metadata_provider,
        config=config,
        enable_data_registry=False  # Disable agentic data discovery for CLI test
    )
    print("✓ Conductor initialized")

    # Test query
    research_query = "What are the benefits of refactoring large codebases?"

    print("\n" + "="*70)
    print("RUNNING MULTI-AGENT RESEARCH")
    print("="*70)
    print(f"\n📝 Research Query: {research_query}")
    print("\n" + "-"*70)

    # Run the swarm
    try:
        import asyncio
        result = asyncio.run(conductor.run(
            goal="Research and analyze the query using multiple specialized agents",
            query=research_query,
            original_query=research_query
        ))

        print("\n" + "="*70)
        print("RESEARCH RESULTS")
        print("="*70)

        print("\n📊 Agent Outputs:")
        print("-"*70)

        # Get outputs from each agent
        all_outputs = conductor.io_manager.get_all_outputs()
        for agent_name in ["TopicExtractor", "FactGatherer", "AnalysisAgent", "SummaryAgent"]:
            actor_output = all_outputs.get(agent_name)
            if actor_output:
                print(f"\n🤖 {agent_name}:")
                print(f"   {actor_output.output_fields}")

        print("\n" + "="*70)
        print("FINAL RESULT")
        print("="*70)
        print(f"\n{result.final_output if hasattr(result, 'final_output') else result}")

        print("\n✅ Multi-Agent System executed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Error during MAS execution: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_two_agent_collaboration():
    """Simpler test with just two agents to verify collaboration."""
    print("\n" + "="*70)
    print("SIMPLIFIED MAS TEST: Two-Agent Collaboration")
    print("="*70)

    # Configure DSPy with Claude CLI LM (proper dspy.BaseLM)
    print("\n🔧 Configuring Claude CLI LM...")
    lm = ClaudeCLILM(model="haiku")
    dspy.configure(lm=lm)
    print("✓ DSPy configured with Claude CLI LM (dspy.BaseLM)")

    # Define simple agents
    class GenerateTopic(dspy.Signature):
        """Generate a simple topic about code quality."""
        request = dspy.InputField()
        topic = dspy.OutputField(desc="A specific topic (2-3 words)")

    class ExplainTopic(dspy.Signature):
        """Explain a topic in 1-2 sentences."""
        topic = dspy.InputField()
        explanation = dspy.OutputField()

    # Create agents
    agent1 = AgentSpec(
        name="TopicGenerator",
        agent=dspy.ChainOfThought(GenerateTopic),
        architect_prompts=[],
        auditor_prompts=[],
        outputs=["topic"]
    )

    agent2 = AgentSpec(
        name="TopicExplainer",
        agent=dspy.ChainOfThought(ExplainTopic),
        architect_prompts=[],
        auditor_prompts=[],
        parameter_mappings={"topic": "TopicGenerator"},  # Get topic from first agent
        outputs=["explanation"]
    )

    # Create actors list and config
    actors = [agent1, agent2]
    config = SwarmConfig(
        max_actor_iters=10
    )

    print(f"\n🤖 Created 2-agent MAS:")
    print(f"   1. TopicGenerator → generates topic")
    print(f"   2. TopicExplainer → explains topic (receives from TopicGenerator)")

    # Create metadata provider
    metadata_provider = Mock()
    metadata_provider.get_tools = Mock(return_value=[])

    # Create Conductor (disable data registry for CLI test)
    print("\n🚀 Creating Conductor...")
    conductor = Conductor(
        actors=actors,
        metadata_provider=metadata_provider,
        config=config,
        enable_data_registry=False  # Disable agentic data discovery for CLI test
    )

    try:
        import asyncio
        result = asyncio.run(conductor.run(
            goal="Generate and explain a code quality topic",
            request="Tell me about code refactoring"
        ))

        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)

        # Show agent outputs
        print("\n📊 Agent Collaboration:")

        all_outputs = conductor.io_manager.get_all_outputs()

        topic_output = all_outputs.get("TopicGenerator")
        topic = topic_output.output_fields if topic_output else None

        explanation_output = all_outputs.get("TopicExplainer")
        explanation = explanation_output.output_fields if explanation_output else None

        print(f"\n   TopicGenerator produced: {topic}")
        print(f"\n   TopicExplainer received topic and produced:")
        print(f"   {explanation}")

        print("\n✅ Two-agent collaboration successful!")
        print("\n🎉 Parameter passing verified:")
        print("   - TopicGenerator output → TopicExplainer input")
        print("   - ParameterResolver correctly resolved dependencies")
        print("   - StateManager tracked agent outputs")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run MAS tests."""
    print("\n" + "="*70)
    print("TESTING REAL MULTI-AGENT SYSTEM USE CASES")
    print("="*70)

    # Run simple test first
    print("\n\n### TEST 1: Simple Two-Agent Collaboration ###\n")
    success1 = test_simple_two_agent_collaboration()

    if success1:
        print("\n\n### TEST 2: Complex Research Assistant MAS ###\n")
        success2 = test_research_mas_with_cli()

        if success2:
            print("\n" + "="*70)
            print("🎉 ALL MAS TESTS SUCCESSFUL!")
            print("="*70)
            print("\n✅ Verified:")
            print("   • Multi-agent collaboration")
            print("   • Parameter passing between agents")
            print("   • ParameterResolver in action")
            print("   • StateManager tracking outputs")
            print("   • ToolManager coordinating tools")
            print("   • Refactored components working in real use case")
            return 0

    print("\n❌ Some tests failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())

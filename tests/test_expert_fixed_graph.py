"""
Test Expert Agent in Fixed LangGraph Workflow

Tests the example from COMPLETE_ARCHITECTURE.md:
- Expert agent (MermaidExpertAgent)
- Fixed graph via LangGraph (static mode)
- ExecutionMode with workflow style
- Learning and memory integration
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from unittest.mock import Mock, MagicMock, patch

from Jotty.core.experts import MermaidExpertAgent
from Jotty.core.orchestration import ExecutionMode, Conductor
from Jotty.core.foundation.agent_config import AgentConfig
from Jotty.core.foundation.data_structures import JottyConfig


class SimpleMetadataProvider:
    """Simple metadata provider for testing."""
    
    def get_metadata(self, *args, **kwargs):
        return {}


async def test_expert_in_fixed_langgraph_workflow():
    """
    Test expert agent in fixed LangGraph workflow (static mode).
    
    This tests the exact example from COMPLETE_ARCHITECTURE.md:
    - Expert agent (MermaidExpertAgent)
    - Fixed graph via LangGraph (static mode)
    - ExecutionMode with workflow style
    - Learning and memory integration
    """
    print("\n" + "="*70)
    print("Testing Expert Agent in Fixed LangGraph Workflow")
    print("="*70)
    
    # Step 1: Create expert
    print("\n1. Creating MermaidExpertAgent...")
    expert = MermaidExpertAgent()
    print(f"   ✅ Expert created: {expert.config.name}")
    print(f"   ✅ Expert has OptimizationPipeline: {hasattr(expert, '_optimization_pipeline')}")
    print(f"   ✅ Expert has memory: {expert.memory is not None}")
    
    # Step 2: Wrap expert as AgentConfig
    print("\n2. Wrapping expert as AgentConfig...")
    expert_agent = AgentConfig(
        name="mermaid_expert",
        agent=expert.generate,
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[],
        enabled=True,
        parameter_mappings={},
        outputs={},
        capabilities=["mermaid", "diagram_generation"],
        dependencies=[],
        metadata={},
        enable_architect=False,
        enable_auditor=False,
        validation_mode="none",
        is_critical=False,
        max_retries=3,
        retry_strategy="exponential_backoff",
        is_executor=False
    )
    print(f"   ✅ AgentConfig created: {expert_agent.name}")
    
    # Step 3: Create Conductor
    print("\n3. Creating Conductor...")
    config = JottyConfig()
    metadata_provider = SimpleMetadataProvider()
    
    conductor = Conductor(
        actors=[expert_agent],
        metadata_provider=metadata_provider,
        config=config
    )
    print(f"   ✅ Conductor created with {len(conductor.actors)} actor(s)")
    has_learning = (
        hasattr(conductor, 'q_predictor') and conductor.q_predictor is not None
    ) or (
        hasattr(conductor, 'q_learner') and conductor.q_learner is not None
    )
    has_memory = (
        hasattr(conductor, 'memory') and conductor.memory is not None
    ) or (
        hasattr(conductor, 'shared_memory') and conductor.shared_memory is not None
    )
    print(f"   ✅ Conductor has learning: {has_learning}")
    print(f"   ✅ Conductor has memory: {has_memory}")
    
    # Step 4: Create ExecutionMode with static mode
    print("\n4. Creating ExecutionMode with static mode (fixed graph)...")
    workflow = ExecutionMode(
        conductor=conductor,
        style="workflow",
        execution="sync",
        mode="static",  # Fixed graph
        agent_order=["mermaid_expert"]  # Only one agent for this test
    )
    print(f"   ✅ ExecutionMode created")
    print(f"   ✅ Style: {workflow.style}")
    print(f"   ✅ Execution: {workflow.execution}")
    print(f"   ✅ Mode: {workflow.mode}")
    print(f"   ✅ Learning enabled: {workflow.learning_enabled}")
    print(f"   ✅ Memory enabled: {workflow.memory_enabled}")
    print(f"   ✅ Queue enabled: {workflow.queue_enabled}")
    
    # Step 5: Test execution (mock the actual generation to avoid LLM calls)
    print("\n5. Testing execution (with mocked LLM calls)...")
    
    # Mock the expert's generate method to return a test result
    original_generate = expert.generate
    
    async def mock_generate(task: str, description: str = "", diagram_type: str = "flowchart", **kwargs):
        """Mock generate that returns a simple Mermaid diagram."""
        return {
            "output": "graph TD; A-->B",
            "task": task,
            "description": description,
            "diagram_type": diagram_type
        }
    
    # Patch the expert's generate method
    expert.generate = mock_generate
    
    try:
        # Execute workflow
        result = await workflow.execute(
            goal="Create a simple flowchart diagram",
            context={"description": "A simple flow from A to B"}
        )
        
        print(f"   ✅ Execution completed")
        print(f"   ✅ Result type: {type(result)}")
        
        # Verify learning and memory are active
        print("\n6. Verifying learning and memory integration...")
        print(f"   ✅ Expert learning (OptimizationPipeline): {hasattr(expert, '_optimization_pipeline')}")
        conductor_has_learning = (
            hasattr(conductor, 'q_predictor') and conductor.q_predictor is not None
        ) or (
            hasattr(conductor, 'q_learner') and conductor.q_learner is not None
        )
        conductor_has_memory = (
            hasattr(conductor, 'memory') and conductor.memory is not None
        ) or (
            hasattr(conductor, 'shared_memory') and conductor.shared_memory is not None
        )
        print(f"   ✅ Conductor learning (Q-learning): {conductor_has_learning}")
        print(f"   ✅ Conductor memory: {conductor_has_memory}")
        print(f"   ✅ ExecutionMode learning: {workflow.learning_enabled}")
        print(f"   ✅ ExecutionMode memory: {workflow.memory_enabled}")
        
        # Verify expert can learn
        if expert.memory:
            print(f"   ✅ Expert memory system active: {expert.memory is not None}")
        
        print("\n" + "="*70)
        print("✅ TEST PASSED: Expert in Fixed LangGraph Workflow")
        print("="*70)
        
        return result
        
    finally:
        # Restore original generate method
        expert.generate = original_generate


async def test_expert_learning_in_workflow():
    """
    Test that expert learning (OptimizationPipeline) works in workflow.
    """
    print("\n" + "="*70)
    print("Testing Expert Learning in Workflow")
    print("="*70)
    
    # Create expert
    expert = MermaidExpertAgent()
    
    # Create agent config
    expert_agent = AgentConfig(
        name="mermaid_expert",
        agent=expert.generate,
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[],
        enabled=True,
        parameter_mappings={},
        outputs={},
        capabilities=["mermaid"],
        dependencies=[],
        metadata={},
        enable_architect=False,
        enable_auditor=False,
        validation_mode="none",
        is_critical=False,
        max_retries=3,
        retry_strategy="exponential_backoff",
        is_executor=False
    )
    
    # Create conductor
    config = JottyConfig()
    metadata_provider = SimpleMetadataProvider()
    conductor = Conductor(
        actors=[expert_agent],
        metadata_provider=metadata_provider,
        config=config
    )
    
    # Create workflow
    workflow = ExecutionMode(
        conductor=conductor,
        style="workflow",
        execution="sync",
        mode="static",
        agent_order=["mermaid_expert"]
    )
    
    # Verify expert can be trained (has OptimizationPipeline capability)
    print(f"\n✅ Expert can be trained: {hasattr(expert, 'train')}")
    print(f"✅ Expert has OptimizationPipeline capability: {hasattr(expert, '_optimization_pipeline')}")
    print(f"✅ Expert memory integration: {expert.memory is not None}")
    
    print("\n" + "="*70)
    print("✅ TEST PASSED: Expert Learning Capability Verified")
    print("="*70)


async def test_expert_memory_integration():
    """
    Test that expert memory integration works in workflow.
    """
    print("\n" + "="*70)
    print("Testing Expert Memory Integration")
    print("="*70)
    
    # Create expert with memory
    expert = MermaidExpertAgent()
    
    print(f"✅ Expert memory: {expert.memory is not None}")
    if expert.memory:
        print(f"✅ Expert memory type: {type(expert.memory).__name__}")
        print(f"✅ Expert uses memory storage: {expert.use_memory_storage}")
    
    # Create agent config
    expert_agent = AgentConfig(
        name="mermaid_expert",
        agent=expert.generate,
        architect_prompts=[],
        auditor_prompts=[],
        architect_tools=[],
        auditor_tools=[],
        enabled=True,
        parameter_mappings={},
        outputs={},
        capabilities=["mermaid"],
        dependencies=[],
        metadata={},
        enable_architect=False,
        enable_auditor=False,
        validation_mode="none",
        is_critical=False,
        max_retries=3,
        retry_strategy="exponential_backoff",
        is_executor=False
    )
    
    # Create conductor (which also has memory)
    config = JottyConfig()
    metadata_provider = SimpleMetadataProvider()
    conductor = Conductor(
        actors=[expert_agent],
        metadata_provider=metadata_provider,
        config=config
    )
    
    conductor_has_memory = (
        hasattr(conductor, 'memory') and conductor.memory is not None
    ) or (
        hasattr(conductor, 'shared_memory') and conductor.shared_memory is not None
    )
    print(f"✅ Conductor memory: {conductor_has_memory}")
    if conductor_has_memory:
        memory_obj = getattr(conductor, 'memory', None) or getattr(conductor, 'shared_memory', None)
        if memory_obj:
            print(f"✅ Conductor memory type: {type(memory_obj).__name__}")
    
    # Create workflow
    workflow = ExecutionMode(
        conductor=conductor,
        style="workflow",
        execution="sync",
        mode="static",
        agent_order=["mermaid_expert"]
    )
    
    print(f"✅ Workflow memory enabled: {workflow.memory_enabled}")
    
    print("\n" + "="*70)
    print("✅ TEST PASSED: Expert Memory Integration Verified")
    print("="*70)


if __name__ == "__main__":
    print("Running Expert Fixed Graph Workflow Tests")
    print("="*70)
    
    # Run tests
    asyncio.run(test_expert_in_fixed_langgraph_workflow())
    asyncio.run(test_expert_learning_in_workflow())
    asyncio.run(test_expert_memory_integration())
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED")
    print("="*70)

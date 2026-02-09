"""
Test Expert Agent Real Mermaid Generation

Tests actual Mermaid diagram generation (requires LLM).
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Jotty.core.experts import MermaidExpertAgent
from Jotty.core.orchestration import ExecutionMode, Conductor
from Jotty.core.foundation.agent_config import AgentConfig
from Jotty.core.foundation.data_structures import JottyConfig


class SimpleMetadataProvider:
    """Simple metadata provider for testing."""
    
    def get_metadata(self, *args, **kwargs):
        return {}


async def test_expert_real_mermaid_generation():
    """
    Test expert agent actually generating a Mermaid diagram.
    """
    print("\n" + "="*70)
    print("Testing Expert Agent Real Mermaid Generation")
    print("="*70)
    
    # Step 1: Create expert
    print("\n1. Creating MermaidExpertAgent...")
    expert = MermaidExpertAgent()
    print(f"   ✅ Expert created: {expert.config.name}")
    
    # Step 2: Test standalone generation
    print("\n2. Testing standalone expert generation...")
    try:
        result = await expert.generate(
            task="Create a simple flowchart",
            description="A simple flow from Start to End",
            diagram_type="flowchart"
        )
        print(f"   ✅ Generation completed")
        print(f"   ✅ Result type: {type(result)}")
        
        # Extract the actual Mermaid code
        if isinstance(result, dict):
            mermaid_code = result.get("output", result.get("mermaid", str(result)))
        else:
            mermaid_code = str(result)
        
        print(f"\n   📊 Generated Mermaid Diagram:")
        print("   " + "-"*66)
        print("   " + "\n   ".join(mermaid_code.split("\n")))
        print("   " + "-"*66)
        
        # Verify it looks like Mermaid
        if "graph" in mermaid_code.lower() or "flowchart" in mermaid_code.lower() or "-->" in mermaid_code:
            print("   ✅ Contains Mermaid syntax")
        else:
            print("   ⚠️  May not be valid Mermaid syntax")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_expert_in_workflow_real_generation():
    """
    Test expert agent generating Mermaid in workflow.
    """
    print("\n" + "="*70)
    print("Testing Expert in Workflow - Real Generation")
    print("="*70)
    
    # Step 1: Create expert
    print("\n1. Creating MermaidExpertAgent...")
    expert = MermaidExpertAgent()
    
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
    
    # Step 3: Create Conductor
    print("\n3. Creating Conductor...")
    config = JottyConfig()
    metadata_provider = SimpleMetadataProvider()
    
    conductor = Conductor(
        actors=[expert_agent],
        metadata_provider=metadata_provider,
        config=config
    )
    
    # Step 4: Create ExecutionMode with static mode
    print("\n4. Creating ExecutionMode with static mode...")
    workflow = ExecutionMode(
        conductor=conductor,
        style="workflow",
        execution="sync",
        mode="static",
        agent_order=["mermaid_expert"]
    )
    
    # Step 5: Execute workflow
    print("\n5. Executing workflow to generate Mermaid diagram...")
    try:
        result = await workflow.execute(
            goal="Create a simple flowchart diagram",
            context={
                "description": "A simple flow from Start to End",
                "diagram_type": "flowchart"
            }
        )
        
        print(f"   ✅ Execution completed")
        print(f"   ✅ Result type: {type(result)}")
        
        # Extract Mermaid code from result
        mermaid_code = None
        if isinstance(result, dict):
            # Try different possible keys
            mermaid_code = (
                result.get("output") or 
                result.get("mermaid") or 
                result.get("result") or
                result.get("diagram") or
                str(result)
            )
        else:
            mermaid_code = str(result)
        
        print(f"\n   📊 Generated Mermaid Diagram:")
        print("   " + "-"*66)
        if isinstance(mermaid_code, str):
            print("   " + "\n   ".join(mermaid_code.split("\n")))
        else:
            print(f"   {mermaid_code}")
        print("   " + "-"*66)
        
        # Verify it looks like Mermaid
        if isinstance(mermaid_code, str):
            if "graph" in mermaid_code.lower() or "flowchart" in mermaid_code.lower() or "-->" in mermaid_code:
                print("   ✅ Contains Mermaid syntax")
            else:
                print("   ⚠️  May not be valid Mermaid syntax")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("Running Expert Real Mermaid Generation Tests")
    print("="*70)
    print("\n⚠️  NOTE: These tests require LLM access and may take time.")
    print("="*70)
    
    # Run tests
    print("\n" + "="*70)
    print("Test 1: Standalone Expert Generation")
    print("="*70)
    result1 = asyncio.run(test_expert_real_mermaid_generation())
    
    print("\n" + "="*70)
    print("Test 2: Expert in Workflow Generation")
    print("="*70)
    result2 = asyncio.run(test_expert_in_workflow_real_generation())
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED")
    print("="*70)
    
    if result1:
        print("\n✅ Standalone generation: SUCCESS")
    else:
        print("\n❌ Standalone generation: FAILED")
    
    if result2:
        print("✅ Workflow generation: SUCCESS")
    else:
        print("❌ Workflow generation: FAILED")

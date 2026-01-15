"""
Test Expert Agent in All Execution Modes

Tests real Mermaid generation in:
- Workflow mode (sync)
- Workflow mode (async with queue)
- Chat mode
"""

import asyncio
import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import dspy
import subprocess

# Configure DSPy with Claude CLI wrapper
try:
    result = subprocess.run(["which", "claude"], capture_output=True, text=True)
    if result.returncode == 0:
        sys.path.insert(0, str(project_root / "JustJot.ai" / "supervisor"))
        from claude_cli_wrapper_enhanced import EnhancedClaudeCLILM
        dspy.configure(lm=EnhancedClaudeCLILM(model="sonnet"))
        print("✅ DSPy configured with Claude CLI wrapper")
    else:
        print("⚠️  Claude CLI not found")
except Exception as e:
    print(f"⚠️  Claude CLI wrapper configuration failed: {e}")

from Jotty.core.experts import MermaidExpertAgent
from Jotty.core.orchestration import ExecutionMode, Conductor
from Jotty.core.foundation.agent_config import AgentConfig
from Jotty.core.foundation.data_structures import JottyConfig
from Jotty.core.queue import SQLiteTaskQueue, Task, TaskStatus, TaskPriority


class SimpleMetadataProvider:
    """Simple metadata provider for testing."""
    
    def get_metadata(self, *args, **kwargs):
        return {}


async def test_expert_workflow_sync():
    """
    Test expert in Workflow mode (synchronous).
    """
    print("\n" + "="*70)
    print("Test 1: Expert in Workflow Mode (Sync)")
    print("="*70)
    
    # Step 1: Create expert
    print("\n1. Creating MermaidExpertAgent...")
    expert = MermaidExpertAgent()
    print(f"   ✅ Expert created: {expert.config.name}")
    
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
    
    # Step 4: Create ExecutionMode - Workflow Sync
    print("\n4. Creating ExecutionMode (workflow, sync, static)...")
    workflow = ExecutionMode(
        conductor=conductor,
        style="workflow",
        execution="sync",
        mode="static",
        agent_order=["mermaid_expert"]
    )
    print(f"   ✅ Style: {workflow.style}")
    print(f"   ✅ Execution: {workflow.execution}")
    print(f"   ✅ Mode: {workflow.mode}")
    print(f"   ✅ Learning enabled: {workflow.learning_enabled}")
    print(f"   ✅ Memory enabled: {workflow.memory_enabled}")
    
    # Step 5: Execute workflow
    print("\n5. Executing workflow (real generation)...")
    try:
        result = await workflow.execute(
            goal="Create a simple flowchart diagram",
            context={
                "description": "A workflow test: Start -> Process -> End",
                "diagram_type": "flowchart"
            }
        )
        
        print(f"   ✅ Execution completed")
        print(f"   ✅ Result type: {type(result)}")
        
        # Extract Mermaid code
        mermaid_code = None
        if isinstance(result, dict):
            mermaid_code = (
                result.get("final_output") or
                result.get("output") or
                result.get("result") or
                str(result)
            )
        else:
            mermaid_code = str(result)
        
        # Display Mermaid
        if mermaid_code and isinstance(mermaid_code, str):
            print(f"\n   📊 Generated Mermaid Diagram:")
            print("   " + "-"*66)
            display = mermaid_code[:300] + ("..." if len(mermaid_code) > 300 else "")
            print("   " + "\n   ".join(display.split("\n")))
            print("   " + "-"*66)
            
            if "flowchart" in mermaid_code.lower() or "-->" in mermaid_code:
                print("   ✅ Contains Mermaid syntax")
        
        return {"success": True, "result": result}
        
    except Exception as e:
        print(f"   ❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


async def test_expert_workflow_async():
    """
    Test expert in Workflow mode (asynchronous with queue).
    """
    print("\n" + "="*70)
    print("Test 2: Expert in Workflow Mode (Async with Queue)")
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
    
    # Step 3: Create Conductor
    print("\n3. Creating Conductor...")
    config = JottyConfig()
    metadata_provider = SimpleMetadataProvider()
    conductor = Conductor(
        actors=[expert_agent],
        metadata_provider=metadata_provider,
        config=config
    )
    
    # Step 4: Create queue
    print("\n4. Creating SQLiteTaskQueue...")
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    queue = SQLiteTaskQueue(db_path=db_path, init_schema=True)
    print(f"   ✅ Queue created: {db_path}")
    
    # Step 5: Create ExecutionMode - Workflow Async
    print("\n5. Creating ExecutionMode (workflow, async, static)...")
    workflow = ExecutionMode(
        conductor=conductor,
        style="workflow",
        execution="async",
        mode="static",
        agent_order=["mermaid_expert"],
        queue=queue
    )
    print(f"   ✅ Style: {workflow.style}")
    print(f"   ✅ Execution: {workflow.execution}")
    print(f"   ✅ Queue enabled: {workflow.queue_enabled}")
    
    # Step 6: Enqueue task
    print("\n6. Enqueuing task...")
    try:
        task_id = await workflow.enqueue_task(
            goal="Create a queue test flowchart",
            context={
                "description": "Queue test: Start -> Queue -> Process -> End",
                "diagram_type": "flowchart"
            }
        )
        print(f"   ✅ Task enqueued: {task_id}")
        
        # Step 7: Process queue
        print("\n7. Processing queue (real generation)...")
        print("   ⚠️  Note: Queue processing may take time with real LLM calls")
        # Process with timeout
        try:
            await asyncio.wait_for(workflow.process_queue(), timeout=120)
        except asyncio.TimeoutError:
            print("   ⚠️  Queue processing timed out (LLM calls take time)")
            # Check task status anyway
            task = await queue.get_task_by_id(task_id)
            if task:
                print(f"   Task status: {task.status}")
                return {"success": task.status == TaskStatus.COMPLETED, "task_id": task_id, "task": task}
            return {"success": False, "error": "Timeout"}
        
        # Step 8: Check task status
        print("\n8. Checking task status...")
        task = await queue.get_task_by_id(task_id)
        if task:
            print(f"   ✅ Task status: {task.status}")
            print(f"   ✅ Task result: {task.result[:200] if task.result else 'None'}...")
            
            if task.status == TaskStatus.COMPLETED:
                print("   ✅ Task completed successfully")
                return {"success": True, "task_id": task_id, "task": task}
            else:
                print(f"   ⚠️  Task status: {task.status}")
                return {"success": False, "task_id": task_id, "status": task.status}
        else:
            print("   ❌ Task not found")
            return {"success": False, "error": "Task not found"}
            
    except Exception as e:
        print(f"   ❌ Queue operation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
    finally:
        # Cleanup
        try:
            os.unlink(db_path)
        except:
            pass


async def test_expert_chat_mode():
    """
    Test expert in Chat mode (conversational).
    """
    print("\n" + "="*70)
    print("Test 3: Expert in Chat Mode")
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
    
    # Step 3: Create Conductor
    print("\n3. Creating Conductor...")
    config = JottyConfig()
    metadata_provider = SimpleMetadataProvider()
    conductor = Conductor(
        actors=[expert_agent],
        metadata_provider=metadata_provider,
        config=config
    )
    
    # Step 4: Create ExecutionMode - Chat
    print("\n4. Creating ExecutionMode (chat, sync)...")
    chat = ExecutionMode(
        conductor=conductor,
        style="chat",
        execution="sync",
        agent_id="mermaid_expert"  # Single agent chat
    )
    print(f"   ✅ Style: {chat.style}")
    print(f"   ✅ Execution: {chat.execution}")
    print(f"   ✅ Agent ID: {chat.agent_id}")
    print(f"   ✅ Learning enabled: {chat.learning_enabled}")
    print(f"   ✅ Memory enabled: {chat.memory_enabled}")
    
    # Step 5: Stream chat
    print("\n5. Streaming chat (real generation)...")
    try:
        events = []
        async for event in chat.stream(
            goal="Create a simple flowchart showing a chat conversation flow",
            history=[]
        ):
            events.append(event)
            if event.get("type") == "agent_complete":
                print(f"   ✅ Agent completed: {event.get('agent')}")
                result = event.get("result")
                if result:
                    result_str = str(result)[:200]
                    print(f"   📊 Result preview: {result_str}...")
        
        print(f"   ✅ Received {len(events)} events")
        
        # Extract Mermaid from events
        mermaid_code = None
        for event in events:
            if event.get("type") == "agent_complete":
                result = event.get("result")
                if result:
                    mermaid_code = str(result)
                    break
        
        if mermaid_code:
            print(f"\n   📊 Generated Mermaid Diagram:")
            print("   " + "-"*66)
            display = mermaid_code[:300] + ("..." if len(mermaid_code) > 300 else "")
            print("   " + "\n   ".join(display.split("\n")))
            print("   " + "-"*66)
            
            if "flowchart" in mermaid_code.lower() or "-->" in mermaid_code:
                print("   ✅ Contains Mermaid syntax")
        
        return {"success": True, "events": events}
        
    except Exception as e:
        print(f"   ❌ Chat failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


async def test_expert_workflow_dynamic():
    """
    Test expert in Workflow mode (dynamic routing).
    """
    print("\n" + "="*70)
    print("Test 4: Expert in Workflow Mode (Dynamic)")
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
    
    # Step 3: Create Conductor
    print("\n3. Creating Conductor...")
    config = JottyConfig()
    metadata_provider = SimpleMetadataProvider()
    conductor = Conductor(
        actors=[expert_agent],
        metadata_provider=metadata_provider,
        config=config
    )
    
    # Step 4: Create ExecutionMode - Workflow Dynamic
    print("\n4. Creating ExecutionMode (workflow, sync, dynamic)...")
    workflow = ExecutionMode(
        conductor=conductor,
        style="workflow",
        execution="sync",
        mode="dynamic"  # Dynamic routing
    )
    print(f"   ✅ Style: {workflow.style}")
    print(f"   ✅ Execution: {workflow.execution}")
    print(f"   ✅ Mode: {workflow.mode} (dynamic routing)")
    
    # Step 5: Execute workflow
    print("\n5. Executing workflow with dynamic routing (real generation)...")
    try:
        result = await workflow.execute(
            goal="Create a dynamic routing test flowchart",
            context={
                "description": "Dynamic test: Start -> Route -> Process -> End",
                "diagram_type": "flowchart"
            }
        )
        
        print(f"   ✅ Execution completed")
        print(f"   ✅ Result type: {type(result)}")
        
        # Extract Mermaid code
        mermaid_code = None
        if isinstance(result, dict):
            mermaid_code = (
                result.get("final_output") or
                result.get("output") or
                result.get("result") or
                str(result)
            )
        else:
            mermaid_code = str(result)
        
        if mermaid_code and isinstance(mermaid_code, str):
            print(f"\n   📊 Generated Mermaid Diagram:")
            print("   " + "-"*66)
            display = mermaid_code[:300] + ("..." if len(mermaid_code) > 300 else "")
            print("   " + "\n   ".join(display.split("\n")))
            print("   " + "-"*66)
            
            if "flowchart" in mermaid_code.lower() or "-->" in mermaid_code:
                print("   ✅ Contains Mermaid syntax")
        
        return {"success": True, "result": result}
        
    except Exception as e:
        print(f"   ❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test expert in different modes")
    parser.add_argument("--mode", choices=["all", "workflow_sync", "workflow_async", "chat", "workflow_dynamic"], 
                       default="all", help="Which test to run")
    args = parser.parse_args()
    
    print("="*70)
    print("Testing Expert Agent in All Execution Modes")
    print("="*70)
    print("\n⚠️  NOTE: Using Claude CLI wrapper for real generation")
    print("⚠️  NOTE: Tests may take time due to real LLM calls")
    print("="*70)
    
    results = {}
    
    if args.mode in ["all", "workflow_sync"]:
        print("\n" + "="*70)
        print("Running Test 1: Workflow Sync")
        print("="*70)
        results["workflow_sync"] = asyncio.run(test_expert_workflow_sync())
    
    if args.mode in ["all", "workflow_async"]:
        print("\n" + "="*70)
        print("Running Test 2: Workflow Async")
        print("="*70)
        results["workflow_async"] = asyncio.run(test_expert_workflow_async())
    
    if args.mode in ["all", "chat"]:
        print("\n" + "="*70)
        print("Running Test 3: Chat Mode")
        print("="*70)
        results["chat"] = asyncio.run(test_expert_chat_mode())
    
    if args.mode in ["all", "workflow_dynamic"]:
        print("\n" + "="*70)
        print("Running Test 4: Workflow Dynamic")
        print("="*70)
        results["workflow_dynamic"] = asyncio.run(test_expert_workflow_dynamic())
    
    # Summary
    print("\n" + "="*70)
    print("✅ TESTS COMPLETED")
    print("="*70)
    
    if results:
        print("\n📊 Test Results Summary:")
        for test_name, result in results.items():
            status = "✅ PASSED" if result.get("success") else "❌ FAILED"
            print(f"   {test_name:20s}: {status}")
            if not result.get("success"):
                print(f"      Error: {result.get('error', 'Unknown')}")
    
    print("\n" + "="*70)

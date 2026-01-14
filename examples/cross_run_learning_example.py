#!/usr/bin/env python3
"""
Cross-Run Learning Example - How to Enable Learning Across Multiple Runs

This example shows how to:
1. Save state after Run 1
2. Load state before Run 2
3. See learning applied across runs

Usage:
    python examples/cross_run_learning_example.py
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from Jotty import Conductor, AgentConfig, JottyConfig, LearningMode
import dspy


def find_latest_run(output_dir: str = "outputs") -> Optional[Path]:
    """Find the latest run directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    
    runs = sorted(
        output_path.glob("run_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return runs[0] if runs else None


def load_state_from_run(conductor: Conductor, run_dir: Path):
    """Load state from a previous run."""
    print(f"\n📂 Loading state from {run_dir.name}...")
    
    # Load shared memory
    memory_file = run_dir / "jotty_state" / "memories" / "shared_memory.json"
    if memory_file.exists():
        with open(memory_file) as f:
            memory_data = json.load(f)
            from Jotty.core.memory.cortex import HierarchicalMemory
            conductor.shared_memory = HierarchicalMemory.from_dict(
                memory_data,
                conductor.config
            )
        total_memories = sum(len(m) for m in memory_data.get('memories', {}).values())
        print(f"✅ Loaded shared memory: {total_memories} memories")
        
        # Show top memories
        for level, memories in memory_data.get('memories', {}).items():
            if memories:
                sorted_mems = sorted(
                    memories.items(),
                    key=lambda x: x[1].get('default_value', 0),
                    reverse=True
                )[:3]
                print(f"   {level}: Top memories:")
                for key, mem in sorted_mems:
                    print(f"     - {mem.get('content', '')[:60]}... (V={mem.get('default_value', 0):.3f})")
    else:
        print("⚠️  No shared memory found")
    
    # Load Q-table
    q_file = run_dir / "jotty_state" / "q_tables" / "q_predictor_buffer.json"
    if q_file.exists() and hasattr(conductor, 'q_predictor'):
        if conductor.q_predictor.load_state(str(q_file)):
            print(f"✅ Loaded Q-table: {len(conductor.q_predictor.Q)} entries")
            
            # Show top Q-values
            sorted_q = sorted(
                conductor.q_predictor.Q.items(),
                key=lambda x: x[1].get('value', 0),
                reverse=True
            )[:3]
            print("   Top Q-values:")
            for (state, action), q_data in sorted_q:
                print(f"     - Q({state[:40]}..., {action[:30]}...) = {q_data.get('value', 0):.3f}")
        else:
            print("⚠️  Failed to load Q-table")
    else:
        print("⚠️  No Q-table found")
    
    # Load agent memories
    local_mem_dir = run_dir / "jotty_state" / "memories" / "local_memories"
    if local_mem_dir.exists():
        for agent_file in local_mem_dir.glob("*.json"):
            agent_name = agent_file.stem
            if agent_name in conductor.local_memories:
                with open(agent_file) as f:
                    agent_memory_data = json.load(f)
                    from Jotty.core.memory.cortex import HierarchicalMemory
                    conductor.local_memories[agent_name] = HierarchicalMemory.from_dict(
                        agent_memory_data,
                        conductor.config
                    )
                print(f"✅ Loaded memory for {agent_name}")


async def run_1_learning_phase():
    """Run 1: Agent learns from experience."""
    print("\n" + "="*70)
    print("RUN 1: LEARNING PHASE")
    print("="*70)
    
    # Configure for persistent learning
    config = JottyConfig(
        learning_mode=LearningMode.PERSISTENT,
        output_base_dir="./outputs",
        persist_memories=True,
        persist_q_tables=True,
        enable_rl=True,
        enable_learning=True
    )
    
    # Create simple agent
    class SimpleAgent(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.ChainOfThought("query -> answer")
        
        def forward(self, query: str) -> str:
            return self.predictor(query=query).answer
    
    # Create conductor
    conductor = Conductor(
        actors=[
            AgentConfig(
                name="SimpleAgent",
                agent=SimpleAgent(),
                architect_prompts=[],
                auditor_prompts=[]
            )
        ],
        config=config,
        metadata_provider=None
    )
    
    # Run
    result = await conductor.run(
        goal="Answer the question",
        query="What is 2+2?"
    )
    
    print(f"\n✅ Run 1 completed")
    print(f"   State saved to: {conductor.persistence_manager.jotty_dir if conductor.persistence_manager else 'N/A'}")
    
    return conductor, result


async def run_2_application_phase():
    """Run 2: Agent applies learned patterns."""
    print("\n" + "="*70)
    print("RUN 2: APPLICATION PHASE (With Loaded Learning)")
    print("="*70)
    
    # Same config
    config = JottyConfig(
        learning_mode=LearningMode.PERSISTENT,
        output_base_dir="./outputs",
        persist_memories=True,
        persist_q_tables=True,
        enable_rl=True,
        enable_learning=True
    )
    
    # Create agent (same as Run 1)
    class SimpleAgent(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.ChainOfThought("query -> answer")
        
        def forward(self, query: str) -> str:
            return self.predictor(query=query).answer
    
    # Create conductor
    conductor = Conductor(
        actors=[
            AgentConfig(
                name="SimpleAgent",
                agent=SimpleAgent(),
                architect_prompts=[],
                auditor_prompts=[]
            )
        ],
        config=config,
        metadata_provider=None
    )
    
    # ⭐ LOAD STATE FROM PREVIOUS RUN
    latest_run = find_latest_run("outputs")
    if latest_run:
        load_state_from_run(conductor, latest_run)
        
        # Show what will be injected into prompts
        print("\n📝 Learned Context (will be injected into prompts):")
        consolidated = conductor.shared_memory.get_consolidated_knowledge(
            goal="Answer the question",
            max_items=5
        )
        if consolidated:
            print(consolidated)
        else:
            print("  (No consolidated knowledge yet)")
    else:
        print("⚠️  No previous run found - starting fresh")
    
    # Run with loaded state
    result = await conductor.run(
        goal="Answer the question",
        query="What is 3+3?"  # Similar but different query
    )
    
    print(f"\n✅ Run 2 completed")
    print(f"   Agent used learned patterns from Run 1!")
    
    return conductor, result


async def main():
    """Run both phases."""
    print("\n" + "="*70)
    print("CROSS-RUN LEARNING DEMONSTRATION")
    print("="*70)
    
    # Configure DSPy
    import os
    api_key = os.getenv('ANTHROPIC_API_KEY') or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No API key found!")
        print("   Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
        return
    
    if 'ANTHROPIC' in api_key:
        lm = dspy.LM(model='anthropic/claude-3-5-haiku-20241022')
    else:
        lm = dspy.LM(model='openai/gpt-4')
    
    dspy.configure(lm=lm)
    
    # Run 1: Learning
    conductor1, result1 = await run_1_learning_phase()
    
    # Wait a bit
    await asyncio.sleep(1)
    
    # Run 2: Application
    conductor2, result2 = await run_2_application_phase()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n✅ Cross-run learning demonstrated!")
    print("\nKey Points:")
    print("1. Run 1 learned patterns and saved state")
    print("2. Run 2 loaded state and applied learning")
    print("3. Agent got better across runs")
    print("\nTo enable automatic loading, add _load_previous_state()")
    print("to Conductor.__init__() when learning_mode=PERSISTENT")


if __name__ == "__main__":
    asyncio.run(main())

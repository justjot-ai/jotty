#!/usr/bin/env python3
"""
Jotty State Inspector - View memories, Q-table, and prompts

Usage:
    python inspect_jotty.py [command] [run_dir]
    
Commands:
    memories  - View agent memories
    qtable    - View Q-learning table
    prompts   - View execution log (if available)
    all       - View everything
    latest    - View latest run (default)

Examples:
    python inspect_jotty.py all
    python inspect_jotty.py memories outputs/run_20260106_114212
    python inspect_jotty.py qtable
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def find_latest_run() -> Optional[Path]:
    """Find the latest run directory."""
    output_dir = Path("outputs")
    if not output_dir.exists():
        return None
    
    runs = sorted(
        output_dir.glob("run_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return runs[0] if runs else None


def inspect_memories(run_dir: Optional[Path] = None):
    """Inspect saved memories from a run."""
    if run_dir is None:
        run_dir = find_latest_run()
    
    if run_dir is None:
        print("❌ No runs found in outputs/")
        return
    
    memory_dir = run_dir / "jotty_state" / "memories"
    
    if not memory_dir.exists():
        print(f"❌ Memory directory not found: {memory_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"📚 MEMORIES: {run_dir.name}")
    print('='*70)
    
    # Shared memory
    shared_file = memory_dir / "shared_memory.json"
    if shared_file.exists():
        print("\n🔗 SHARED MEMORY")
        print("-"*70)
        with open(shared_file) as f:
            data = json.load(f)
            total = sum(len(m) for m in data.get('memories', {}).values())
            print(f"Total memories: {total}")
            
            for level, memories in data.get('memories', {}).items():
                if not memories:
                    continue
                print(f"\n{level}: {len(memories)} memories")
                
                # Show top 5 by value
                sorted_mems = sorted(
                    memories.items(),
                    key=lambda x: x[1].get('default_value', 0),
                    reverse=True
                )[:5]
                
                for key, mem in sorted_mems:
                    content = mem.get('content', '')[:80]
                    value = mem.get('default_value', 0)
                    accesses = mem.get('access_count', 0)
                    print(f"  • {content}...")
                    print(f"    Value: {value:.3f} | Accesses: {accesses}")
    
    # Agent memories
    local_dir = memory_dir / "local_memories"
    if local_dir.exists():
        print("\n\n👤 AGENT MEMORIES")
        print("-"*70)
        agent_files = sorted(local_dir.glob("*.json"))
        
        if not agent_files:
            print("  (No agent-specific memories)")
        else:
            for agent_file in agent_files:
                print(f"\n{agent_file.stem}:")
                with open(agent_file) as f:
                    data = json.load(f)
                    total = sum(len(m) for m in data.get('memories', {}).values())
                    print(f"  Total: {total} memories")
                    
                    for level, memories in data.get('memories', {}).items():
                        if memories:
                            print(f"    {level}: {len(memories)}")
                            
                            # Show top 3
                            sorted_mems = sorted(
                                memories.items(),
                                key=lambda x: x[1].get('default_value', 0),
                                reverse=True
                            )[:3]
                            
                            for key, mem in sorted_mems:
                                content = mem.get('content', '')[:60]
                                value = mem.get('default_value', 0)
                                print(f"      - {content}... (V={value:.3f})")


def inspect_q_table(run_dir: Optional[Path] = None):
    """Inspect Q-learning table."""
    if run_dir is None:
        run_dir = find_latest_run()
    
    if run_dir is None:
        print("❌ No runs found")
        return
    
    q_file = run_dir / "jotty_state" / "q_tables" / "q_predictor_buffer.json"
    
    if not q_file.exists():
        print(f"❌ Q-table not found: {q_file}")
        print("   (Q-table is only saved if learning is enabled)")
        return
    
    print(f"\n{'='*70}")
    print(f"📊 Q-TABLE: {run_dir.name}")
    print('='*70)
    
    with open(q_file) as f:
        data = json.load(f)
        
        q_table = data.get('q_table', {})
        print(f"\nTotal Q-values: {len(q_table)}")
        
        if not q_table:
            print("  (Q-table is empty)")
            return
        
        # Show top 10 by value
        sorted_q = sorted(
            q_table.items(),
            key=lambda x: x[1].get('value', 0),
            reverse=True
        )[:10]
        
        print("\nTop 10 Q-values:")
        print("-"*70)
        
        for i, ((state, action), q_data) in enumerate(sorted_q, 1):
            value = q_data.get('value', 0)
            visits = q_data.get('visit_count', 0)
            lessons = q_data.get('learned_lessons', [])
            
            print(f"\n{i}. Q(state, action) = {value:.3f}")
            print(f"   State: {state[:60]}...")
            print(f"   Action: {action[:60]}...")
            print(f"   Visits: {visits}")
            
            if lessons:
                print(f"   Lessons:")
                for lesson in lessons[:2]:  # Show first 2 lessons
                    print(f"     • {lesson[:80]}...")


def inspect_prompts(run_dir: Optional[Path] = None):
    """Inspect prompts from execution log."""
    if run_dir is None:
        run_dir = find_latest_run()
    
    if run_dir is None:
        print("❌ No runs found")
        return
    
    log_file = run_dir / "beautified" / "execution_log.md"
    
    print(f"\n{'='*70}")
    print(f"📝 EXECUTION LOG: {run_dir.name}")
    print('='*70)
    
    if not log_file.exists():
        print("\n⚠️  Execution log not found")
        print("   Prompts are not saved by default.")
        print("\n   To see prompts:")
        print("   1. Enable debug logging:")
        print("      import logging")
        print("      logging.basicConfig(level=logging.DEBUG)")
        print("   2. Check DSPy LM history:")
        print("      lm = dspy.settings.lm")
        print("      last_prompt = lm.history[-1] if lm.history else None")
        return
    
    print("\nExecution log (first 2000 chars):")
    print("-"*70)
    with open(log_file) as f:
        content = f.read()
        print(content[:2000])
        if len(content) > 2000:
            print(f"\n... (truncated, full log: {len(content)} chars)")


def inspect_all(run_dir: Optional[Path] = None):
    """Inspect everything."""
    inspect_memories(run_dir)
    inspect_q_table(run_dir)
    inspect_prompts(run_dir)


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        run_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    else:
        command = "latest"
        run_dir = None
    
    if command == "memories":
        inspect_memories(run_dir)
    elif command == "qtable":
        inspect_q_table(run_dir)
    elif command == "prompts":
        inspect_prompts(run_dir)
    elif command in ["all", "latest"]:
        inspect_all(run_dir)
    else:
        print(f"❌ Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

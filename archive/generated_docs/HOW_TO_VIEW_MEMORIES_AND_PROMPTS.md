# 📖 How to View Final Agent Memories and Prompts in Jotty

This guide shows you how to access and view:
1. **Agent memories** (what agents learned)
2. **Final prompts** (what was sent to LLMs)
3. **Q-table lessons** (learned patterns)
4. **Consolidated knowledge** (memory consolidation results)

---

## 🗂️ Method 1: View Saved Memory Files

### Location
After running Jotty, memories are saved to:
```
outputs/run_YYYYMMDD_HHMMSS/jotty_state/memories/
├── shared_memory.json          # Shared memories across all agents
└── local_memories/
    ├── AgentName1.json         # Per-agent memories
    ├── AgentName2.json
    └── ...
```

### View Memory Files

```python
import json
from pathlib import Path

# Find the latest run directory
output_dir = Path("outputs")
runs = sorted(output_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
latest_run = runs[0] if runs else None

if latest_run:
    memory_dir = latest_run / "jotty_state" / "memories"
    
    # View shared memory
    shared_memory_file = memory_dir / "shared_memory.json"
    if shared_memory_file.exists():
        with open(shared_memory_file) as f:
            shared_memory = json.load(f)
            print(json.dumps(shared_memory, indent=2))
    
    # View agent-specific memories
    local_memories_dir = memory_dir / "local_memories"
    for agent_file in local_memories_dir.glob("*.json"):
        print(f"\n{'='*60}")
        print(f"Agent: {agent_file.stem}")
        print('='*60)
        with open(agent_file) as f:
            agent_memory = json.load(f)
            # Print memory statistics
            print(f"Total memories: {sum(len(mems) for mems in agent_memory.get('memories', {}).values())}")
            for level, memories in agent_memory.get('memories', {}).items():
                print(f"  {level}: {len(memories)} memories")
                # Show top 3 memories by value
                sorted_mems = sorted(
                    memories.items(),
                    key=lambda x: x[1].get('default_value', 0),
                    reverse=True
                )[:3]
                for key, mem in sorted_mems:
                    print(f"    - {mem['content'][:100]}... (V={mem['default_value']:.3f})")
```

---

## 🐍 Method 2: Access Memories Programmatically

### After Running Conductor

```python
from Jotty import Conductor, AgentConfig

# ... create and run conductor ...
conductor = Conductor(actors=[...])
result = await conductor.run(goal="...")

# Access memories from conductor
for agent_config in conductor.actors:
    agent_name = agent_config.name
    
    # Get agent's memory system
    # (Note: This depends on how JottyCore stores memories)
    # Check conductor.jotty_cores or similar
    
    # Method 1: Export to dict
    if hasattr(conductor, 'shared_memory'):
        memory_dict = conductor.shared_memory.to_dict()
        print(f"\nShared Memory:")
        print(json.dumps(memory_dict, indent=2, default=str))
    
    # Method 2: Get consolidated knowledge (what gets injected into prompts)
    if hasattr(conductor, 'shared_memory'):
        consolidated = conductor.shared_memory.get_consolidated_knowledge(
            goal="Your goal here",
            max_items=10
        )
        print(f"\nConsolidated Knowledge (for prompts):")
        print(consolidated)
```

---

## 📝 Method 3: View Learned Context (What Gets Injected into Prompts)

### TD(λ) Learned Context

```python
from Jotty.core.learning.learning import TDLambdaLearner

# After episode ends
td_learner = conductor.td_learner  # Or however you access it

# Get learned context (this is what gets injected into prompts)
learned_context = td_learner.get_learned_context(
    memories=conductor.shared_memory.memories,
    goal="Your goal"
)

print("TD(λ) Learned Context:")
print(learned_context)
```

**Output format:**
```
# TD(λ) Learned Values:

## High-Value Patterns (Learned from Success):
- Use partition columns for date filters... (V=0.800)
- Check table metadata before querying... (V=0.750)

## Low-Value Patterns (Learned from Failure):
- AVOID: Using transaction_category for P2P queries... (V=0.200)

## Recently Updated Understanding:
- ↑ Use partition columns... (V=0.800, Δ=+0.300)
```

### Q-Learning Learned Context

```python
from Jotty.core.learning.q_learning import LLMQPredictor

# Access Q-learning predictor
q_predictor = conductor.q_predictor  # Or however you access it

# Get learned context
learned_context = q_predictor.get_learned_context(
    state={"query": "Count P2P transactions"},
    action={"actor": "SQLGenerator"}
)

print("Q-Learning Learned Context:")
print(learned_context)
```

**Output format:**
```
# Q-Learning Lessons (Learned from Experience):
1. ✅ LEARNED: Using partition columns → SUCCESS (reward=0.8)
2. ❌ AVOID: Using transaction_category → FAILED (reward=0.2)
3. 📈 DISCOVERY: Checking metadata first performed better than expected

# Expected Value: Q(state, action) = 0.750
```

---

## 🔍 Method 4: View Final Prompts (What Was Sent to LLM)

### Option A: Enable Debug Logging

Add this to see prompts in logs:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("Jotty")

# Now when you run conductor, prompts will be logged
conductor = Conductor(...)
result = await conductor.run(...)
```

### Option B: Access Prompt Construction Directly

```python
# In JottyCore, prompts are constructed in _build_architect_prompt and _build_auditor_prompt
# You can add a method to expose them:

class JottyCore:
    def get_final_prompt(self, agent_type: str = "actor") -> str:
        """
        Get the final prompt that was sent to the LLM.
        
        Args:
            agent_type: "actor", "architect", or "auditor"
        """
        if agent_type == "architect":
            return self._last_architect_prompt
        elif agent_type == "auditor":
            return self._last_auditor_prompt
        else:
            return self._last_actor_prompt
    
    # You'd need to store these in the execution methods:
    def _execute_architect(self, ...):
        prompt = self._build_architect_prompt(...)
        self._last_architect_prompt = prompt  # Store it
        # ... rest of execution
```

### Option C: Hook into DSPy

```python
import dspy

# DSPy stores the last prompt in the LM's history
# After running an agent:

# Get last prompt sent to LLM
lm = dspy.settings.lm
if hasattr(lm, 'history'):
    last_call = lm.history[-1] if lm.history else None
    if last_call:
        print("Last prompt sent to LLM:")
        print(last_call.get('messages', []))
        print("\nLast response:")
        print(last_call.get('response', ''))
```

---

## 📊 Method 5: View Complete State

### Get All State Information

```python
# After running conductor
state = {
    # Memories
    'shared_memory': conductor.shared_memory.to_dict(),
    'agent_memories': {
        agent.name: agent.memory.to_dict() 
        for agent in conductor.actors
    },
    
    # Q-table
    'q_table': conductor.q_predictor.get_state() if hasattr(conductor, 'q_predictor') else {},
    
    # Roadmap/TODO state
    'roadmap': conductor.todo.get_state_summary() if hasattr(conductor, 'todo') else {},
    
    # Learning statistics
    'learning_stats': conductor.td_learner.get_statistics() if hasattr(conductor, 'td_learner') else {},
    
    # Episode history
    'episode_count': conductor.episode_count,
    'trajectory': conductor.trajectory[-10:] if hasattr(conductor, 'trajectory') else []
}

# Save to file
import json
with open("jotty_complete_state.json", "w") as f:
    json.dump(state, f, indent=2, default=str)

print("Complete state saved to jotty_complete_state.json")
```

---

## 🎯 Method 6: Create a State Inspector Utility

Create a helper script:

```python
#!/usr/bin/env python3
"""
Jotty State Inspector - View memories and prompts
"""

import json
import sys
from pathlib import Path
from typing import Optional

def inspect_memories(run_dir: Optional[str] = None):
    """Inspect saved memories from a run."""
    if run_dir:
        base = Path(run_dir)
    else:
        # Find latest run
        output_dir = Path("outputs")
        runs = sorted(output_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            print("❌ No runs found in outputs/")
            return
        base = runs[0]
    
    memory_dir = base / "jotty_state" / "memories"
    
    if not memory_dir.exists():
        print(f"❌ Memory directory not found: {memory_dir}")
        return
    
    print(f"\n📂 Inspecting: {base.name}\n")
    
    # Shared memory
    shared_file = memory_dir / "shared_memory.json"
    if shared_file.exists():
        print("="*70)
        print("SHARED MEMORY")
        print("="*70)
        with open(shared_file) as f:
            data = json.load(f)
            print(f"Total memories: {sum(len(m) for m in data.get('memories', {}).values())}")
            for level, memories in data.get('memories', {}).items():
                print(f"\n{level}: {len(memories)} memories")
                # Show top 5
                sorted_mems = sorted(
                    memories.items(),
                    key=lambda x: x[1].get('default_value', 0),
                    reverse=True
                )[:5]
                for key, mem in sorted_mems:
                    print(f"  • {mem['content'][:80]}...")
                    print(f"    Value: {mem.get('default_value', 0):.3f}, Accesses: {mem.get('access_count', 0)}")
    
    # Agent memories
    local_dir = memory_dir / "local_memories"
    if local_dir.exists():
        print("\n" + "="*70)
        print("AGENT MEMORIES")
        print("="*70)
        for agent_file in sorted(local_dir.glob("*.json")):
            print(f"\n{agent_file.stem}:")
            with open(agent_file) as f:
                data = json.load(f)
                total = sum(len(m) for m in data.get('memories', {}).values())
                print(f"  Total: {total} memories")
                for level, memories in data.get('memories', {}).items():
                    if memories:
                        print(f"    {level}: {len(memories)}")

def inspect_q_table(run_dir: Optional[str] = None):
    """Inspect Q-table."""
    if run_dir:
        base = Path(run_dir)
    else:
        output_dir = Path("outputs")
        runs = sorted(output_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            print("❌ No runs found")
            return
        base = runs[0]
    
    q_file = base / "jotty_state" / "q_tables" / "q_predictor_buffer.json"
    
    if not q_file.exists():
        print(f"❌ Q-table not found: {q_file}")
        return
    
    print(f"\n📊 Q-TABLE: {base.name}\n")
    with open(q_file) as f:
        data = json.load(f)
        
        q_table = data.get('q_table', {})
        print(f"Total Q-values: {len(q_table)}")
        
        # Show top 10 by value
        sorted_q = sorted(
            q_table.items(),
            key=lambda x: x[1].get('value', 0),
            reverse=True
        )[:10]
        
        print("\nTop 10 Q-values:")
        for (state, action), q_data in sorted_q:
            value = q_data.get('value', 0)
            visits = q_data.get('visit_count', 0)
            lessons = q_data.get('learned_lessons', [])
            print(f"\n  Q({state[:50]}..., {action[:30]}...) = {value:.3f}")
            print(f"    Visits: {visits}")
            if lessons:
                print(f"    Lessons: {lessons[0][:80]}...")

def inspect_prompts(run_dir: Optional[str] = None):
    """Inspect prompts from execution log."""
    if run_dir:
        base = Path(run_dir)
    else:
        output_dir = Path("outputs")
        runs = sorted(output_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            print("❌ No runs found")
            return
        base = runs[0]
    
    log_file = base / "beautified" / "execution_log.md"
    
    if not log_file.exists():
        print(f"⚠️  Execution log not found: {log_file}")
        print("   Prompts are not saved by default. Enable debug logging to see them.")
        return
    
    print(f"\n📝 EXECUTION LOG: {base.name}\n")
    with open(log_file) as f:
        content = f.read()
        # Extract prompt sections (if they exist in the log)
        # This depends on how logging is configured
        print(content[:2000])  # Show first 2000 chars

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        run_dir = sys.argv[2] if len(sys.argv) > 2 else None
        
        if command == "memories":
            inspect_memories(run_dir)
        elif command == "qtable":
            inspect_q_table(run_dir)
        elif command == "prompts":
            inspect_prompts(run_dir)
        elif command == "all":
            inspect_memories(run_dir)
            inspect_q_table(run_dir)
            inspect_prompts(run_dir)
        else:
            print(f"Unknown command: {command}")
            print("Usage: python inspect_jotty.py [memories|qtable|prompts|all] [run_dir]")
    else:
        # Default: show all
        inspect_memories()
        inspect_q_table()
        inspect_prompts()
```

**Usage:**
```bash
# View all
python inspect_jotty.py all

# View specific
python inspect_jotty.py memories
python inspect_jotty.py qtable
python inspect_jotty.py prompts

# View specific run
python inspect_jotty.py all outputs/run_20260106_114212
```

---

## 🔧 Method 7: Add Inspection Methods to Conductor

Add these methods to your Conductor class:

```python
class Conductor:
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of all memories."""
        return {
            'shared_memory': {
                'stats': self.shared_memory.get_statistics(),
                'consolidated': self.shared_memory.get_consolidated_knowledge(max_items=20)
            },
            'agent_memories': {
                agent.name: {
                    'stats': agent.memory.get_statistics(),
                    'consolidated': agent.memory.get_consolidated_knowledge(max_items=10)
                }
                for agent in self.actors
            }
        }
    
    def get_learned_context_summary(self, goal: str) -> str:
        """Get all learned context that would be injected into prompts."""
        context_parts = []
        
        # TD(λ) learned values
        if hasattr(self, 'td_learner'):
            td_context = self.td_learner.get_learned_context(
                memories=self.shared_memory.memories,
                goal=goal
            )
            if td_context:
                context_parts.append(td_context)
        
        # Q-learning lessons
        if hasattr(self, 'q_predictor'):
            q_context = self.q_predictor.get_learned_context(
                state={"goal": goal},
                action=None
            )
            if q_context:
                context_parts.append(q_context)
        
        # Consolidated knowledge
        consolidated = self.shared_memory.get_consolidated_knowledge(goal=goal)
        if consolidated:
            context_parts.append(consolidated)
        
        return "\n\n".join(context_parts)
    
    def export_state(self, filepath: str):
        """Export complete state to JSON file."""
        state = {
            'memories': self.get_memory_summary(),
            'q_table': self.q_predictor.get_state() if hasattr(self, 'q_predictor') else {},
            'roadmap': self.todo.get_state_summary() if hasattr(self, 'todo') else {},
            'episode_count': self.episode_count,
            'trajectory': self.trajectory[-20:] if hasattr(self, 'trajectory') else []
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        print(f"✅ State exported to {filepath}")
```

**Usage:**
```python
conductor = Conductor(...)
result = await conductor.run(goal="...")

# Get memory summary
memory_summary = conductor.get_memory_summary()
print(json.dumps(memory_summary, indent=2))

# Get learned context (what gets injected into prompts)
learned_context = conductor.get_learned_context_summary(goal="...")
print("\nLearned Context (for prompts):")
print(learned_context)

# Export everything
conductor.export_state("jotty_state_export.json")
```

---

## 📋 Quick Reference

### Memory File Locations
```
outputs/run_YYYYMMDD_HHMMSS/jotty_state/
├── memories/
│   ├── shared_memory.json
│   └── local_memories/
│       └── AgentName.json
├── q_tables/
│   └── q_predictor_buffer.json
├── markovian_todos/
│   ├── todo_state.json
│   └── todo_display.md
└── episode_history/
    └── episode_N.json
```

### Key Methods to Access Memories

```python
# From HierarchicalMemory (cortex.py)
memory.to_dict()                          # Export to dict
memory.get_statistics()                   # Get stats
memory.get_consolidated_knowledge(goal)   # Get prompt-ready knowledge

# From TDLambdaLearner (learning.py)
td_learner.get_learned_context(memories, goal)  # TD(λ) learned values

# From LLMQPredictor (q_learning.py)
q_predictor.get_learned_context(state, action)   # Q-learning lessons
q_predictor.get_state()                          # Export Q-table

# From Conductor
conductor.shared_memory                       # Shared memory system
conductor.td_learner                          # TD(λ) learner
conductor.q_predictor                         # Q-learning predictor
```

### Viewing Prompts

Prompts are **not saved by default**. To see them:

1. **Enable debug logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Hook into DSPy:**
   ```python
   # After agent execution
   lm = dspy.settings.lm
   last_prompt = lm.history[-1] if lm.history else None
   ```

3. **Add prompt storage to JottyCore:**
   ```python
   # Store prompts in execution methods
   self._last_architect_prompt = prompt
   self._last_auditor_prompt = prompt
   self._last_actor_prompt = prompt
   ```

---

## 🎯 Example: Complete Inspection Script

```python
#!/usr/bin/env python3
"""Complete Jotty state inspection"""

import json
from pathlib import Path
from Jotty import Conductor, AgentConfig

async def inspect_jotty_run(conductor: Conductor, goal: str):
    """Inspect a Jotty run completely."""
    
    print("="*70)
    print("JOTTY STATE INSPECTION")
    print("="*70)
    
    # 1. Memory Summary
    print("\n📚 MEMORY SUMMARY")
    print("-"*70)
    memory_stats = conductor.shared_memory.get_statistics()
    print(json.dumps(memory_stats, indent=2))
    
    # 2. Consolidated Knowledge
    print("\n🧠 CONSOLIDATED KNOWLEDGE (What gets injected into prompts)")
    print("-"*70)
    consolidated = conductor.shared_memory.get_consolidated_knowledge(goal=goal)
    print(consolidated)
    
    # 3. TD(λ) Learned Context
    print("\n📈 TD(λ) LEARNED VALUES")
    print("-"*70)
    if hasattr(conductor, 'td_learner'):
        td_context = conductor.td_learner.get_learned_context(
            memories=conductor.shared_memory.memories,
            goal=goal
        )
        print(td_context)
    
    # 4. Q-Learning Lessons
    print("\n🎯 Q-LEARNING LESSONS")
    print("-"*70)
    if hasattr(conductor, 'q_predictor'):
        q_context = conductor.q_predictor.get_learned_context(
            state={"goal": goal},
            action=None
        )
        print(q_context)
        
        # Q-table stats
        q_state = conductor.q_predictor.get_state()
        print(f"\nQ-table size: {len(q_state.get('q_table', {}))} entries")
    
    # 5. Export to file
    export_path = "jotty_inspection.json"
    state = {
        'memory_stats': memory_stats,
        'consolidated_knowledge': consolidated,
        'td_learned_context': td_context if hasattr(conductor, 'td_learner') else None,
        'q_learned_context': q_context if hasattr(conductor, 'q_predictor') else None,
        'q_table_state': conductor.q_predictor.get_state() if hasattr(conductor, 'q_predictor') else None
    }
    
    with open(export_path, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    
    print(f"\n✅ Complete state exported to {export_path}")

# Usage
if __name__ == "__main__":
    # After running conductor
    # result = await conductor.run(goal="...")
    # await inspect_jotty_run(conductor, goal="...")
    pass
```

---

## 🚀 Quick Commands

```bash
# View latest memories
python -c "
from pathlib import Path
import json
runs = sorted(Path('outputs').glob('run_*'), key=lambda p: p.stat().st_mtime, reverse=True)
if runs:
    mem_file = runs[0] / 'jotty_state' / 'memories' / 'shared_memory.json'
    if mem_file.exists():
        with open(mem_file) as f:
            data = json.load(f)
            print(json.dumps(data, indent=2))
"

# View Q-table
python -c "
from pathlib import Path
import json
runs = sorted(Path('outputs').glob('run_*'), key=lambda p: p.stat().st_mtime, reverse=True)
if runs:
    q_file = runs[0] / 'jotty_state' / 'q_tables' / 'q_predictor_buffer.json'
    if q_file.exists():
        with open(q_file) as f:
            data = json.load(f)
            print(f'Q-table entries: {len(data.get(\"q_table\", {}))}')
            for (s, a), q_data in list(data.get('q_table', {}).items())[:5]:
                print(f'Q({s[:50]}..., {a[:30]}...) = {q_data.get(\"value\", 0):.3f}')
"
```

---

*For more details, see:*
- `core/memory/cortex.py` - `to_dict()`, `get_consolidated_knowledge()`
- `core/learning/learning.py` - `get_learned_context()`
- `core/learning/q_learning.py` - `get_learned_context()`, `get_state()`
- `core/persistence/persistence.py` - `save_memory()`, `load_memory()`

# 🔄 How Learning Persists Across Runs in Jotty

**Understanding Cross-Run Learning with Persistent Memories**

---

## 🎯 The Problem

When memories are saved to `outputs/run_*/jotty_state/memories/`, how does the **next run** learn from them? How do agents get better over multiple runs?

---

## 📋 Learning Modes

Jotty has **3 learning modes** that control persistence:

### 1. **DISABLED** - No Learning
```python
learning_mode = LearningMode.DISABLED
```
- No memory updates
- No Q-learning
- No state persistence
- **Use case**: Production inference with fixed behavior

### 2. **CONTEXTUAL** - Session-Only Learning
```python
learning_mode = LearningMode.CONTEXTUAL
```
- Updates memory and Q-values **during the run**
- **Forgets after session ends** (not saved to disk)
- **Use case**: Testing, debugging, single-run scenarios

### 3. **PERSISTENT** - Cross-Run Learning ⭐
```python
learning_mode = LearningMode.PERSISTENT
```
- Saves Q-tables, memories, brain state **to disk**
- **Auto-loads on next run**
- **Use case**: Training over multiple runs, continuous improvement

---

## 🔄 How It Works: The Complete Flow

### Run 1: First Execution

```
1. Conductor initialized
   ├── Creates NEW memories (empty)
   ├── Creates NEW Q-table (empty)
   └── LearningMode.PERSISTENT enabled

2. Episode execution
   ├── Agent tries approach A → FAILS
   ├── Memory stored: "AVOID approach A" (V=0.2)
   ├── Q-table updated: Q("query", "approach A") = 0.2
   └── Agent tries approach B → SUCCEEDS
       ├── Memory stored: "USE approach B" (V=0.8)
       └── Q-table updated: Q("query", "approach B") = 0.8

3. Episode ends
   ├── TD(λ) updates memory values
   ├── Q-learning updates Q-values
   └── State saved to disk:
       ├── outputs/run_1/jotty_state/memories/shared_memory.json
       ├── outputs/run_1/jotty_state/q_tables/q_predictor_buffer.json
       └── outputs/run_1/jotty_state/brain_state/consolidated_memories.json
```

### Run 2: Second Execution (With Learning)

```
1. Conductor initialized
   ├── LearningMode.PERSISTENT enabled
   ├── LOADS memories from outputs/latest/jotty_state/memories/
   ├── LOADS Q-table from outputs/latest/jotty_state/q_tables/
   └── Memories now contain:
       - "AVOID approach A" (V=0.2)
       - "USE approach B" (V=0.8)

2. Episode execution
   ├── Similar query arrives
   ├── Memory retrieval:
   │   ├── Gets "USE approach B" (high value: 0.8)
   │   └── Injects into prompt: "High-value pattern: USE approach B"
   ├── Q-value prediction:
   │   ├── Q("similar query", "approach A") = 0.2 (low!)
   │   └── Q("similar query", "approach B") = 0.8 (high!)
   └── Agent immediately uses approach B → SUCCEEDS FASTER!

3. Episode ends
   ├── Memory values updated further (V=0.8 → 0.9)
   ├── Q-values updated (Q=0.8 → 0.9)
   └── State saved to NEW run folder:
       └── outputs/run_2/jotty_state/...
```

---

## 🔧 How to Enable Cross-Run Learning

### Option 1: Use LearningMode.PERSISTENT (Recommended)

```python
from Jotty import Conductor, AgentConfig, JottyConfig, LearningMode

config = JottyConfig(
    learning_mode=LearningMode.PERSISTENT,  # ⭐ Enable persistence
    output_base_dir="./outputs",            # Where to save state
    auto_load_on_start=True,                # Auto-load previous state
    persist_memories=True,                  # Save memories
    persist_q_tables=True,                  # Save Q-tables
    persist_brain_state=True                # Save brain state
)

conductor = Conductor(
    actors=[...],
    config=config
)

# Run 1: Learns and saves
result1 = await conductor.run(goal="...")

# Run 2: Loads previous learning automatically
result2 = await conductor.run(goal="...")  # Uses learned patterns!
```

### Option 2: Manual State Loading

```python
from pathlib import Path
import json
from Jotty.core.memory.cortex import HierarchicalMemory
from Jotty.core.learning.q_learning import LLMQPredictor

# Create conductor
conductor = Conductor(actors=[...], config=config)

# Find latest run
output_dir = Path("outputs")
runs = sorted(output_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
latest_run = runs[0] if runs else None

if latest_run:
    # Load memories
    memory_file = latest_run / "jotty_state" / "memories" / "shared_memory.json"
    if memory_file.exists():
        with open(memory_file) as f:
            memory_data = json.load(f)
            conductor.shared_memory = HierarchicalMemory.from_dict(
                memory_data, 
                config
            )
        print(f"✅ Loaded {sum(len(m) for m in memory_data.get('memories', {}).values())} memories")
    
    # Load Q-table
    q_file = latest_run / "jotty_state" / "q_tables" / "q_predictor_buffer.json"
    if q_file.exists():
        conductor.q_predictor.load_state(str(q_file))
        print(f"✅ Loaded Q-table")

# Now run with loaded state
result = await conductor.run(goal="...")
```

---

## 📂 State Persistence Structure

### What Gets Saved

```
outputs/run_YYYYMMDD_HHMMSS/jotty_state/
├── memories/
│   ├── shared_memory.json          # Shared memories (all agents)
│   └── local_memories/
│       ├── AgentName1.json         # Per-agent memories
│       └── AgentName2.json
├── q_tables/
│   └── q_predictor_buffer.json     # Q-learning state
├── brain_state/
│   └── consolidated_memories.json  # Brain consolidation state
└── markovian_todos/
    └── todo_state.json              # Task planning state
```

### What Gets Loaded

When `LearningMode.PERSISTENT` is enabled:

1. **Memories** (`shared_memory.json` + `local_memories/*.json`)
   - All memory levels (Episodic, Semantic, Procedural, Meta, Causal)
   - Goal-conditioned values
   - Access counts
   - Causal links

2. **Q-Table** (`q_predictor_buffer.json`)
   - Natural language Q-values
   - Experience buffer
   - Learned lessons
   - Tiered memory (Tier 1, 2, 3)

3. **Brain State** (`consolidated_memories.json`)
   - Consolidated patterns
   - Hippocampal memories
   - Neocortex patterns

---

## 🔍 Current Implementation Status

### ✅ What Works

1. **State Saving**: ✅ Fully implemented
   - `Vault.save_all()` saves everything
   - Called at end of `conductor.run()`
   - Saves to `outputs/run_*/jotty_state/`

2. **State Loading**: ⚠️ **Partially Implemented**
   - `SessionManager.load_previous_state()` exists
   - `HierarchicalMemory.from_dict()` exists
   - `LLMQPredictor.load_state()` exists
   - **BUT**: Not automatically called in `Conductor.__init__()`

### ⚠️ What's Missing

**Auto-loading on initialization is NOT currently implemented!**

The code has the infrastructure but doesn't automatically load state. You need to:

1. **Manually load state** (see Option 2 above), OR
2. **Add auto-loading to Conductor.__init__()**

---

## 🛠️ How to Add Auto-Loading

### Add to Conductor.__init__()

```python
class Conductor:
    def __init__(self, ...):
        # ... existing initialization ...
        
        # Initialize persistence manager
        if output_dir:
            self.persistence_manager = Vault(output_dir)
            
            # ⭐ NEW: Auto-load previous state if PERSISTENT mode
            if self.config.learning_mode == LearningMode.PERSISTENT:
                self._load_previous_state()
    
    def _load_previous_state(self):
        """Load state from latest run."""
        if not self.persistence_manager:
            return
        
        # Find latest run
        output_dir = Path(self.config.output_base_dir)
        runs = sorted(
            output_dir.glob("run_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not runs:
            logger.info("📭 No previous state found - starting fresh")
            return
        
        latest_run = runs[0]
        logger.info(f"📂 Loading state from {latest_run.name}")
        
        # Load memories
        memory_file = latest_run / "jotty_state" / "memories" / "shared_memory.json"
        if memory_file.exists():
            with open(memory_file) as f:
                memory_data = json.load(f)
                self.shared_memory = HierarchicalMemory.from_dict(
                    memory_data,
                    self.config
                )
            logger.info(f"✅ Loaded shared memory: {sum(len(m) for m in memory_data.get('memories', {}).values())} memories")
        
        # Load agent memories
        local_mem_dir = latest_run / "jotty_state" / "memories" / "local_memories"
        if local_mem_dir.exists():
            for agent_file in local_mem_dir.glob("*.json"):
                agent_name = agent_file.stem
                if agent_name in self.local_memories:
                    with open(agent_file) as f:
                        agent_memory_data = json.load(f)
                        self.local_memories[agent_name] = HierarchicalMemory.from_dict(
                            agent_memory_data,
                            self.config
                        )
                    logger.info(f"✅ Loaded memory for {agent_name}")
        
        # Load Q-table
        q_file = latest_run / "jotty_state" / "q_tables" / "q_predictor_buffer.json"
        if q_file.exists() and hasattr(self, 'q_predictor'):
            if self.q_predictor.load_state(str(q_file)):
                logger.info(f"✅ Loaded Q-table: {len(self.q_predictor.Q)} entries")
        
        # Load brain state
        brain_file = latest_run / "jotty_state" / "brain_state" / "consolidated_memories.json"
        if brain_file.exists() and hasattr(self, 'brain'):
            with open(brain_file) as f:
                brain_data = json.load(f)
                # Restore brain state (implementation depends on brain type)
                logger.info("✅ Loaded brain state")
```

---

## 🎯 How Learning Manifests Across Runs

### Memory Values → Prompt Injection

**Run 1:**
```
Memory: "Use partition columns" → V=0.3 (low, just learned)
```

**Run 2 (after loading):**
```
Memory loaded: "Use partition columns" → V=0.7 (high, from Run 1)
Prompt injection: "High-value pattern: Use partition columns (V=0.700)"
Agent sees this → Uses partition columns immediately
```

### Q-Values → Action Selection

**Run 1:**
```
Q("Count P2P", "Use transaction_category") = 0.3 (failed)
Q("Count P2P", "Use partition column") = 0.8 (succeeded)
```

**Run 2 (after loading):**
```
Q-table loaded with previous values
Similar query arrives
Q-predictor predicts: Q("Count P2P", "Use partition column") = 0.8
Agent chooses partition column (high Q-value)
```

### Consolidated Knowledge → Context

**Run 1:**
```
Episodic memories → Consolidated → Semantic patterns
Pattern: "For date filters, use partition columns"
```

**Run 2 (after loading):**
```
Consolidated knowledge loaded
get_consolidated_knowledge() returns:
  "## Learned Patterns:
   - For date filters, use partition columns"
Agent sees this in prompt → Applies pattern
```

---

## 📊 Example: Cross-Run Learning Flow

### Run 1: Learning Phase

```python
config = JottyConfig(
    learning_mode=LearningMode.PERSISTENT,
    output_base_dir="./outputs"
)

conductor = Conductor(actors=[...], config=config)

# Run 1: Agent learns
result1 = await conductor.run(goal="Count P2P transactions")

# What happened:
# - Agent tried: transaction_category → FAILED
# - Memory stored: "AVOID: transaction_category for P2P" (V=0.2)
# - Agent tried: partition column → SUCCEEDED
# - Memory stored: "USE: partition column for date filters" (V=0.8)
# - Q-table updated: Q("Count P2P", "partition column") = 0.8
# - State saved to: outputs/run_20260106_120000/jotty_state/
```

### Run 2: Application Phase

```python
# Same config (PERSISTENT mode)
conductor2 = Conductor(actors=[...], config=config)

# ⭐ State automatically loaded (if auto-loading implemented)
# OR manually load:
# conductor2._load_previous_state()

# Run 2: Agent applies learning
result2 = await conductor2.run(goal="Get P2P count yesterday")

# What happens:
# 1. Memory retrieval:
#    - Finds "USE: partition column" (V=0.8, high value)
#    - Injects: "High-value pattern: USE partition column (V=0.800)"
#
# 2. Q-value prediction:
#    - Q("Get P2P count", "partition column") = 0.8 (from Run 1)
#    - Q("Get P2P count", "transaction_category") = 0.2 (from Run 1)
#    - Chooses partition column (high Q-value)
#
# 3. Agent execution:
#    - Immediately uses partition column
#    - SUCCEEDS faster than Run 1!
#
# 4. Learning continues:
#    - Memory value increases: V=0.8 → 0.9
#    - Q-value increases: Q=0.8 → 0.9
#    - State saved to: outputs/run_20260106_120500/jotty_state/
```

### Run 3: Mastery Phase

```python
conductor3 = Conductor(actors=[...], config=config)

result3 = await conductor3.run(goal="Count peer-to-peer transactions")

# What happens:
# - Memory: "USE partition column" (V=0.9, very high)
# - Q-value: Q("Count P2P", "partition column") = 0.9
# - Agent: Immediately uses partition column
# - Result: SUCCEEDS instantly (no trial and error!)
```

---

## 🔧 Configuration for Cross-Run Learning

### Full Configuration

```python
from Jotty import JottyConfig, LearningMode

config = JottyConfig(
    # Learning mode
    learning_mode=LearningMode.PERSISTENT,  # ⭐ Enable persistence
    
    # Output directory
    output_base_dir="./outputs",           # Where to save state
    create_run_folder=True,                # Create timestamped folders
    
    # Persistence settings
    persist_memories=True,                 # Save memories
    persist_q_tables=True,                 # Save Q-tables
    persist_brain_state=True,              # Save brain state
    persist_todos=True,                    # Save TODO state
    
    # Auto-loading (if implemented)
    auto_load_on_start=True,               # Load previous state
    auto_save_interval=10,                 # Auto-save every N iterations
    
    # Learning parameters
    enable_rl=True,                        # Enable reinforcement learning
    enable_learning=True,                  # Enable learning updates
    
    # Memory settings
    episodic_capacity=10000,               # Max episodic memories
    semantic_capacity=5000,                # Max semantic memories
    consolidation_interval=3,              # Episodes between consolidation
)

conductor = Conductor(actors=[...], config=config)
```

---

## 🐛 Current Issue: Auto-Loading Not Implemented

### The Problem

**Auto-loading is NOT currently implemented in `Conductor.__init__()`**

Even with `LearningMode.PERSISTENT`, state is **saved** but **not automatically loaded** on the next run.

### The Solution

You have two options:

#### Option A: Manual Loading (Current Workaround)

```python
# After creating conductor
conductor = Conductor(...)

# Manually load state
latest_run = find_latest_run("outputs")
if latest_run:
    # Load memories
    memory_file = latest_run / "jotty_state" / "memories" / "shared_memory.json"
    if memory_file.exists():
        with open(memory_file) as f:
            memory_data = json.load(f)
            conductor.shared_memory = HierarchicalMemory.from_dict(
                memory_data, conductor.config
            )
    
    # Load Q-table
    q_file = latest_run / "jotty_state" / "q_tables" / "q_predictor_buffer.json"
    if q_file.exists():
        conductor.q_predictor.load_state(str(q_file))
```

#### Option B: Add Auto-Loading (Recommended Fix)

Add the `_load_previous_state()` method to `Conductor` (see code above).

---

## 📈 Learning Progression Example

### Run 1: Exploration
```
Query: "Count P2P transactions"
Attempts:
  1. transaction_category → FAILED (error: column not found)
  2. partition column → SUCCEEDED
  
Memories saved:
  - "AVOID: transaction_category" (V=0.2)
  - "USE: partition column" (V=0.8)
  
Q-table saved:
  - Q("Count P2P", "transaction_category") = 0.2
  - Q("Count P2P", "partition column") = 0.8
```

### Run 2: Application (After Loading)
```
Query: "Get P2P count"
Loaded memories:
  - "USE: partition column" (V=0.8) → Injected into prompt
  - "AVOID: transaction_category" (V=0.2) → Injected into prompt
  
Loaded Q-values:
  - Q("Get P2P count", "partition column") = 0.8 (high!)
  
Agent behavior:
  - Immediately uses partition column (no trial and error)
  - SUCCEEDS on first attempt
  
Updated:
  - V=0.8 → 0.85 (value increased)
  - Q=0.8 → 0.85 (Q-value increased)
```

### Run 3: Mastery (After Loading)
```
Query: "Count peer-to-peer transactions"
Loaded memories:
  - "USE: partition column" (V=0.85) → High confidence
  
Agent behavior:
  - Instantly recognizes pattern
  - Uses partition column immediately
  - SUCCEEDS instantly
  
Updated:
  - V=0.85 → 0.9 (approaching mastery)
  - Q=0.85 → 0.9
```

---

## 🎯 Key Methods for Cross-Run Learning

### Saving State

```python
# Automatic (at end of run)
conductor.persistence_manager.save_all(conductor)

# Manual
conductor.persistence_manager.save_memory(memory, "shared")
conductor.q_predictor.save_state("path/to/q_table.json")
```

### Loading State

```python
# Load memories
memory = HierarchicalMemory.from_dict(memory_data, config)

# Load Q-table
q_predictor.load_state("path/to/q_table.json")

# Load brain state
brain.load_state("path/to/brain_state.json")
```

### Accessing Learned Knowledge

```python
# Get consolidated knowledge (what gets injected into prompts)
consolidated = conductor.shared_memory.get_consolidated_knowledge(
    goal="Your goal",
    max_items=10
)

# Get TD(λ) learned context
td_context = conductor.td_learner.get_learned_context(
    memories=conductor.shared_memory.memories,
    goal="Your goal"
)

# Get Q-learning lessons
q_context = conductor.q_predictor.get_learned_context(
    state={"goal": "Your goal"},
    action=None
)
```

---

## 🔍 Debugging Cross-Run Learning

### Check if State is Saved

```python
from pathlib import Path

run_dir = Path("outputs/run_20260106_120000")
memory_file = run_dir / "jotty_state" / "memories" / "shared_memory.json"

if memory_file.exists():
    print("✅ Memories saved")
    with open(memory_file) as f:
        data = json.load(f)
        print(f"   Total: {sum(len(m) for m in data.get('memories', {}).values())} memories")
else:
    print("❌ Memories NOT saved")
```

### Check if State is Loaded

```python
# After conductor initialization
print(f"Shared memory size: {len(conductor.shared_memory.memories)}")
print(f"Q-table size: {len(conductor.q_predictor.Q) if hasattr(conductor, 'q_predictor') else 0}")

# If sizes are 0, state wasn't loaded!
```

### Verify Learning is Applied

```python
# Get learned context that would be injected
learned_context = conductor.shared_memory.get_consolidated_knowledge(
    goal="Your goal"
)

print("Learned context (for prompts):")
print(learned_context)

# If empty, no learning from previous runs!
```

---

## 🚀 Quick Start: Enable Cross-Run Learning

### Step 1: Configure for Persistence

```python
from Jotty import Conductor, AgentConfig, JottyConfig, LearningMode

config = JottyConfig(
    learning_mode=LearningMode.PERSISTENT,  # ⭐ Enable persistence
    output_base_dir="./outputs",
    persist_memories=True,
    persist_q_tables=True
)
```

### Step 2: Add Auto-Loading (if not implemented)

```python
# Add to Conductor.__init__() or call manually:
conductor = Conductor(...)

# Manually load previous state
latest_run = find_latest_run("outputs")
if latest_run:
    load_state_from_run(conductor, latest_run)
```

### Step 3: Run Multiple Times

```python
# Run 1: Learns
result1 = await conductor.run(goal="...")

# Run 2: Applies learning (if state loaded)
result2 = await conductor.run(goal="...")

# Run 3: Masters (if state loaded)
result3 = await conductor.run(goal="...")
```

---

## 📝 Summary

### How Learning Persists:

1. **Run 1**: Learns → Saves to `outputs/run_1/jotty_state/`
2. **Run 2**: Loads from `outputs/run_1/` → Applies learning → Saves to `outputs/run_2/`
3. **Run 3**: Loads from `outputs/run_2/` → Applies learning → Saves to `outputs/run_3/`

### Current Status:

- ✅ **State saving**: Fully implemented
- ✅ **State loading methods**: Exist (`from_dict()`, `load_state()`)
- ⚠️ **Auto-loading**: NOT automatically called in `Conductor.__init__()`
- ✅ **Manual loading**: Works perfectly

### To Enable Cross-Run Learning:

1. Set `learning_mode=LearningMode.PERSISTENT`
2. **Manually load state** before running (workaround), OR
3. **Add auto-loading** to `Conductor.__init__()` (recommended fix)

---

*For implementation details, see:*
- `core/persistence/persistence.py` - `Vault.save_all()`
- `core/persistence/session_manager.py` - `load_previous_state()`
- `core/memory/cortex.py` - `from_dict()`
- `core/learning/q_learning.py` - `load_state()`

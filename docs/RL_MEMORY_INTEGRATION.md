# RL Layer Memory Integration

## Current State

### ✅ RL Layer DOES Use Memory

The RL layer (`core/learning/`) **does use memory**, but there's a **disconnect**:

1. **TDLambdaLearner** (`learning.py`):
   - Uses `MemoryEntry` objects directly
   - Calls `record_access(memory: MemoryEntry)` 
   - Calls `end_episode(memories: Dict[str, MemoryEntry])`
   - Updates memory values: `memory.goal_values[goal].value = new_value`

2. **Q-Learning** (`q_learning.py`):
   - Has its own tiered memory system (Tier 1, 2, 3)
   - Stores experiences in `experience_buffer`
   - Uses natural language Q-table

3. **Conductor** (`conductor.py`):
   - Uses `HierarchicalMemory` (the unified memory system)
   - `self.shared_memory = HierarchicalMemory(...)`
   - `self.local_memories: Dict[str, HierarchicalMemory] = {}`

### ⚠️ The Problem

**RL Layer receives `Dict[str, MemoryEntry]` but doesn't directly use `HierarchicalMemory`**

This means:
- RL layer updates memory values
- But it doesn't use HierarchicalMemory's retrieval methods
- RL layer has its own memory management (Q-learning tiered memory)
- Conductor uses HierarchicalMemory but RL layer operates on raw MemoryEntry dicts

---

## Should RL Layer Use HierarchicalMemory?

### ✅ YES - Here's Why:

1. **Unified Memory System**: Everything should use the same memory package
2. **Consistency**: RL updates should go through HierarchicalMemory
3. **Features**: HierarchicalMemory has:
   - Retrieval methods (`retrieve()`, `retrieve_async()`)
   - Consolidation (EPISODIC → SEMANTIC)
   - Deduplication
   - Goal-conditioned values (which RL needs!)
   - Memory levels (EPISODIC, SEMANTIC, PROCEDURAL, META, CAUSAL)

4. **Integration**: RL layer should:
   - Read from HierarchicalMemory
   - Update values in HierarchicalMemory
   - Use HierarchicalMemory's retrieval for context

---

## How RL Layer Should Integrate

### Current Flow (Disconnected):

```
Conductor
  ↓
HierarchicalMemory (stores memories)
  ↓
Extract MemoryEntry dict → Pass to RL
  ↓
TDLambdaLearner (updates MemoryEntry values)
  ↓
??? (values updated but not synced back?)
```

### Proposed Flow (Integrated):

```
Conductor
  ↓
HierarchicalMemory (unified memory system)
  ↓
TDLambdaLearner (uses HierarchicalMemory directly)
  ↓
Updates values in HierarchicalMemory
  ↓
HierarchicalMemory provides context to agents
```

---

## Integration Points

### 1. **TDLambdaLearner Should Accept HierarchicalMemory**

**Current:**
```python
def record_access(self, memory: MemoryEntry, step_reward: float = 0.0)
def end_episode(self, memories: Dict[str, MemoryEntry], ...)
```

**Should be:**
```python
def record_access(self, memory: HierarchicalMemory, memory_key: str, level: MemoryLevel, step_reward: float = 0.0)
def end_episode(self, memory: HierarchicalMemory, goal: str, ...)
```

### 2. **RL Should Use HierarchicalMemory Retrieval**

**Current:**
```python
# RL layer receives pre-retrieved memories
memories = {...}  # Dict[str, MemoryEntry]
learner.end_episode(memories, ...)
```

**Should be:**
```python
# RL layer retrieves from HierarchicalMemory
memories = hierarchical_memory.retrieve(goal=goal, top_k=50)
learner.end_episode(hierarchical_memory, goal=goal, ...)
```

### 3. **Q-Learning Should Store in HierarchicalMemory**

**Current:**
```python
# Q-learning has its own tiered memory
self.tier1_working = []
self.tier2_clusters = {}
self.tier3_archive = []
```

**Should be:**
```python
# Q-learning uses HierarchicalMemory levels
# Tier 1 → PROCEDURAL (working memory)
# Tier 2 → SEMANTIC (clusters)
# Tier 3 → EPISODIC (archive)
```

---

## Benefits of Integration

### ✅ Unified Memory System

- **Single source of truth**: All memory goes through HierarchicalMemory
- **Consistent API**: Same retrieval/update methods everywhere
- **Better features**: Consolidation, deduplication, semantic search

### ✅ RL Updates Persist Properly

- **Current**: RL updates MemoryEntry values, but may not sync back to HierarchicalMemory
- **After**: RL updates go directly to HierarchicalMemory, automatically persisted

### ✅ Better Context Retrieval

- **Current**: RL gets pre-retrieved memories (may miss relevant ones)
- **After**: RL can use HierarchicalMemory's smart retrieval (semantic search, goal-conditioned)

### ✅ Memory Consolidation Works

- **Current**: RL has separate memory management
- **After**: RL updates trigger HierarchicalMemory consolidation automatically

---

## Implementation Plan

### Phase 1: Refactor TDLambdaLearner

1. Change `record_access()` to accept `HierarchicalMemory` + key
2. Change `end_episode()` to accept `HierarchicalMemory` + goal
3. Update values directly in HierarchicalMemory
4. Use HierarchicalMemory's retrieval methods

### Phase 2: Refactor Q-Learning

1. Remove custom tiered memory (Tier 1, 2, 3)
2. Map tiers to HierarchicalMemory levels:
   - Tier 1 (working) → PROCEDURAL
   - Tier 2 (clusters) → SEMANTIC  
   - Tier 3 (archive) → EPISODIC
3. Use HierarchicalMemory for storage/retrieval

### Phase 3: Update Conductor Integration

1. Pass `HierarchicalMemory` directly to RL components
2. Remove intermediate `Dict[str, MemoryEntry]` conversion
3. RL components use HierarchicalMemory API directly

---

## Current Code Locations

### RL Layer Memory Usage:

- `core/learning/learning.py`:
  - `TDLambdaLearner.record_access(memory: MemoryEntry)`
  - `TDLambdaLearner.end_episode(memories: Dict[str, MemoryEntry])`
  - `TDLambdaLearner.get_learned_context(memories: Dict[str, MemoryEntry])`

- `core/learning/q_learning.py`:
  - Custom tiered memory system
  - `self.tier1_working`, `self.tier2_clusters`, `self.tier3_archive`

- `core/learning/offline_learning.py`:
  - ✅ Already uses `HierarchicalMemory`!
  - `agent_memories: Dict[str, HierarchicalMemory]`

### Conductor Memory Usage:

- `core/orchestration/conductor.py`:
  - `self.shared_memory = HierarchicalMemory(...)`
  - `self.local_memories: Dict[str, HierarchicalMemory] = {}`
  - But RL components not directly integrated

---

## Summary

**Current State:**
- ✅ RL layer uses memory (MemoryEntry objects)
- ⚠️ RL layer doesn't use HierarchicalMemory directly
- ⚠️ Disconnect between RL updates and HierarchicalMemory

**Should RL Use HierarchicalMemory?**
- ✅ **YES** - For unified memory system, better integration, persistence

**Next Steps:**
1. Refactor TDLambdaLearner to use HierarchicalMemory
2. Refactor Q-Learning to use HierarchicalMemory levels
3. Update Conductor to pass HierarchicalMemory directly to RL

**Result:**
- Single unified memory system
- RL updates persist properly
- Better context retrieval
- Automatic consolidation

# Expert Agents - Memory System Integration

## Overview

Expert agent improvements are now integrated with **Jotty's HierarchicalMemory system** instead of just file-based storage. This provides:

- ✅ **Semantic search** via LLM
- ✅ **Automatic deduplication**
- ✅ **Goal-conditioned retrieval**
- ✅ **Memory consolidation**
- ✅ **Persistent across runs**

## Architecture

```
Expert Agent
    │
    ├─ HierarchicalMemory (Jotty Memory System)
    │   ├─ PROCEDURAL Level: Specific improvement patterns
    │   └─ META Level: General learning wisdom
    │
    └─ File Storage (Fallback)
        └─ improvements.json
```

## Memory Levels Used

### PROCEDURAL Level
**For**: Specific improvement patterns (how to generate correct outputs)

**Example:**
```python
{
    "task": "Generate simple flowchart",
    "learned_pattern": "When task is 'Generate simple flowchart', use 'graph TD...'"
}
```

**Why PROCEDURAL**: These are action sequences - "how to do X correctly"

### META Level
**For**: General learning wisdom (when to use what patterns)

**Example:**
```python
{
    "wisdom": "Simple flowcharts should use minimal nodes",
    "applicability": "When task mentions 'simple' or 'basic'"
}
```

**Why META**: This is wisdom about learning itself - "when to apply what knowledge"

## Integration Points

### 1. Expert Agent Initialization

```python
from core.memory.cortex import HierarchicalMemory
from core.experts import MermaidExpertAgent

# Create memory system
memory = HierarchicalMemory(agent_name="mermaid_expert", config=config)

# Create expert with memory
expert = MermaidExpertAgent(memory=memory)
```

### 2. Loading Improvements

```python
# ExpertAgent._load_improvements()
# 1. Try memory system first
if self.use_memory_storage and self.memory:
    memory_entries = self.memory.retrieve(
        query="expert agent improvements mermaid",
        goal="expert_mermaid_improvements",
        levels=[MemoryLevel.PROCEDURAL, MemoryLevel.META]
    )
    
# 2. Fallback to file if memory empty
if not improvements:
    improvements = load_from_json_file()
```

### 3. Storing Improvements

```python
# optimization_pipeline.py _record_improvement()
# 1. Save to file (always)
save_to_json_file(improvement)

# 2. Store to memory (if available)
if memory_system:
    store_improvement_to_memory(
        memory=memory_system,
        improvement=improvement,
        expert_name=expert_name,
        domain=domain
    )
```

## Memory Storage Functions

### `store_improvement_to_memory()`

Stores a single improvement to HierarchicalMemory.

```python
from core.experts.memory_integration import store_improvement_to_memory

entry = store_improvement_to_memory(
    memory=memory,
    improvement={
        "learned_pattern": "...",
        "task": "...",
        ...
    },
    expert_name="mermaid_expert",
    domain="mermaid"
)
```

**Storage Details:**
- **Level**: PROCEDURAL (for corrections) or META (for wisdom)
- **Content**: JSON string of improvement
- **Context**: Expert name, domain, task, iteration
- **Goal**: `expert_{domain}_improvements`
- **Value**: 1.0 (high value for learned patterns)

### `retrieve_improvements_from_memory()`

Retrieves improvements from HierarchicalMemory.

```python
from core.experts.memory_integration import retrieve_improvements_from_memory

improvements = retrieve_improvements_from_memory(
    memory=memory,
    expert_name="mermaid_expert",
    domain="mermaid",
    task="Generate flowchart",  # Optional filter
    max_results=20
)
```

**Retrieval Details:**
- **Query**: "expert agent {name} domain {domain} improvements"
- **Goal**: `expert_{domain}_improvements`
- **Levels**: PROCEDURAL, META
- **Method**: LLM-based semantic search

### `sync_improvements_to_memory()`

Syncs a list of improvements to memory (e.g., from file).

```python
from core.experts.memory_integration import sync_improvements_to_memory

count = sync_improvements_to_memory(
    memory=memory,
    improvements=[...],  # List of improvements
    expert_name="mermaid_expert",
    domain="mermaid"
)
```

## Benefits Over File Storage

### ✅ Semantic Search

**File-based**: Exact string matching  
**Memory-based**: LLM semantic understanding

```python
# Can find improvements even with different wording
memory.retrieve(
    query="how to generate simple diagrams",
    # Finds improvements about "simple flowcharts"
)
```

### ✅ Automatic Deduplication

**File-based**: Manual deduplication needed  
**Memory-based**: Automatic duplicate detection

```python
# Same improvement stored twice → automatically merged
memory.store(...)  # First time
memory.store(...)  # Second time → merged with first
```

### ✅ Goal-Conditioned Retrieval

**File-based**: All improvements loaded  
**Memory-based**: Only relevant improvements retrieved

```python
# Only retrieves improvements for specific goal
memory.retrieve(
    goal="expert_mermaid_improvements",
    # Only returns mermaid improvements
)
```

### ✅ Memory Consolidation

**File-based**: Static storage  
**Memory-based**: Automatic consolidation and pruning

```python
# Old, unused improvements automatically pruned
# Important patterns strengthened
# Related patterns consolidated
```

## Usage Examples

### Example 1: Expert with Memory

```python
from core.memory.cortex import HierarchicalMemory
from core.experts import MermaidExpertAgent
from core.foundation.data_structures import JottyConfig

# Create memory
memory_config = JottyConfig()
memory = HierarchicalMemory(
    agent_name="mermaid_expert",
    config=memory_config
)

# Create expert with memory
expert = MermaidExpertAgent(memory=memory)

# Train (improvements stored to memory)
await expert.train()

# Generate (improvements loaded from memory)
diagram = await expert.generate_mermaid(...)
```

### Example 2: Integration with Conductor

```python
from core.orchestration.conductor import Conductor
from core.experts import MermaidExpertAgent

# Conductor has shared_memory
conductor = Conductor(...)

# Create expert with conductor's memory
expert = MermaidExpertAgent(
    memory=conductor.shared_memory
)

# Improvements stored to shared memory
# Available to all agents in conductor
```

### Example 3: Manual Memory Operations

```python
from core.experts.memory_integration import (
    store_improvement_to_memory,
    retrieve_improvements_from_memory
)

# Store improvement
store_improvement_to_memory(
    memory=memory,
    improvement={
        "learned_pattern": "...",
        "task": "..."
    },
    expert_name="mermaid_expert",
    domain="mermaid"
)

# Retrieve improvements
improvements = retrieve_improvements_from_memory(
    memory=memory,
    expert_name="mermaid_expert",
    domain="mermaid"
)
```

## Configuration

### Enable Memory Storage

```python
config = ExpertAgentConfig(
    name="mermaid_expert",
    domain="mermaid",
    use_memory_storage=True,  # Enable memory integration
    ...
)

expert = MermaidExpertAgent(config=config, memory=memory)
```

### Disable Memory Storage (Use Files Only)

```python
config = ExpertAgentConfig(
    name="mermaid_expert",
    domain="mermaid",
    use_memory_storage=False,  # Use file storage only
    ...
)

expert = MermaidExpertAgent(config=config)
```

## Migration from Files

### Sync Existing Improvements

```python
from core.experts.memory_integration import sync_improvements_to_memory
import json

# Load from file
with open("expert_data/mermaid/improvements.json", 'r') as f:
    improvements = json.load(f)

# Sync to memory
sync_improvements_to_memory(
    memory=memory,
    improvements=improvements,
    expert_name="mermaid_expert",
    domain="mermaid"
)
```

## Comparison

| Feature | File Storage | Memory System |
|---------|-------------|---------------|
| **Storage** | JSON file | HierarchicalMemory |
| **Search** | Exact match | Semantic (LLM) |
| **Deduplication** | Manual | Automatic |
| **Retrieval** | Load all | Goal-conditioned |
| **Consolidation** | None | Automatic |
| **Persistence** | File system | Memory + optional persistence |

## Files Created

- `core/experts/memory_integration.py` - Memory integration utilities
- `docs/EXPERT_AGENTS_MEMORY_INTEGRATION.md` - This document

## Conclusion

✅ **Improvements integrated with Jotty's HierarchicalMemory**  
✅ **Semantic search via LLM**  
✅ **Automatic deduplication**  
✅ **Goal-conditioned retrieval**  
✅ **Memory consolidation**  
✅ **Backward compatible** (falls back to files if no memory)

**Expert agents now use Jotty's memory system for intelligent improvement storage and retrieval!** 🎉

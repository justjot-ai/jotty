# Expert Agents - Memory Consolidation Integration

## Overview

Expert agent improvements now support **memory consolidation and summarization** using Jotty's HierarchicalMemory synthesis capabilities. This provides:

- ✅ **Consolidated improvements** - Multiple improvements synthesized into patterns
- ✅ **Semantic patterns** - PROCEDURAL improvements promoted to SEMANTIC level
- ✅ **Brain-inspired synthesis** - LLM-based pattern extraction and summarization
- ✅ **Preference extraction** - Automatic identification of best practices

## Memory Consolidation Flow

```
Raw Improvements (PROCEDURAL)
    ↓
Consolidation Cycle
    ↓
Synthesized Patterns (SEMANTIC)
    ↓
Extracted Preferences (META)
```

## New Functions

### `retrieve_synthesized_improvements()`

Retrieves improvements using memory synthesis to create consolidated wisdom.

```python
from core.experts.memory_integration import retrieve_synthesized_improvements

synthesized = retrieve_synthesized_improvements(
    memory=memory,
    expert_name="mermaid_expert",
    domain="mermaid"
)

# Returns: Consolidated text summary of all improvements
```

**Benefits:**
- Finds emergent patterns across improvements
- Resolves contradictions
- Creates coherent wisdom
- More efficient than loading all raw improvements

### `consolidate_improvements()`

Consolidates PROCEDURAL improvements into SEMANTIC patterns.

```python
from core.experts.memory_integration import consolidate_improvements

result = consolidate_improvements(
    memory=memory,
    expert_name="mermaid_expert",
    domain="mermaid"
)

# Returns: {"consolidated": 1, "preferences": 0, "consolidated_entry_key": "..."}
```

**Process:**
1. Retrieves all PROCEDURAL improvements
2. Uses synthesis to extract common patterns
3. Stores consolidated pattern as SEMANTIC memory
4. Returns consolidation results

### `run_improvement_consolidation_cycle()`

Runs full consolidation cycle (similar to memory consolidation).

```python
from core.experts.memory_integration import run_improvement_consolidation_cycle

result = await run_improvement_consolidation_cycle(
    memory=memory,
    expert_name="mermaid_expert",
    domain="mermaid"
)

# Returns: {"consolidated": 1, "preferences": 0}
```

**Cycle:**
1. Consolidate PROCEDURAL → SEMANTIC
2. Extract preferences/patterns
3. Promote important patterns to META

## Usage Examples

### Example 1: Use Synthesized Improvements

```python
from core.experts import MermaidExpertAgent, ExpertAgentConfig
from core.memory.cortex import HierarchicalMemory

# Create expert with synthesis enabled
config = ExpertAgentConfig(
    name="mermaid_expert",
    domain="mermaid",
    use_memory_storage=True,
    use_memory_synthesis=True  # Enable synthesis
)

memory = HierarchicalMemory(...)
expert = MermaidExpertAgent(config=config, memory=memory)

# Expert loads synthesized improvements automatically
# Instead of raw improvements, gets consolidated patterns
```

### Example 2: Manual Consolidation

```python
from core.experts.memory_integration import (
    consolidate_improvements,
    retrieve_synthesized_improvements
)

# Consolidate improvements
result = consolidate_improvements(memory, "mermaid_expert", "mermaid")
print(f"Consolidated {result['consolidated']} patterns")

# Retrieve synthesized improvements
synthesized = retrieve_synthesized_improvements(
    memory, "mermaid_expert", "mermaid"
)
print(synthesized)
```

### Example 3: Periodic Consolidation

```python
from core.experts.memory_integration import run_improvement_consolidation_cycle

# Run consolidation cycle periodically (e.g., after training)
async def after_training():
    result = await run_improvement_consolidation_cycle(
        memory=memory,
        expert_name="mermaid_expert",
        domain="mermaid"
    )
    print(f"Consolidation: {result['consolidated']} patterns, {result['preferences']} preferences")
```

## Configuration

### Enable Memory Synthesis

```python
config = ExpertAgentConfig(
    name="mermaid_expert",
    domain="mermaid",
    use_memory_storage=True,
    use_memory_synthesis=True  # Use synthesized improvements
)
```

### Use Raw Improvements (Default)

```python
config = ExpertAgentConfig(
    name="mermaid_expert",
    domain="mermaid",
    use_memory_storage=True,
    use_memory_synthesis=False  # Use raw improvements
)
```

## Memory Levels Used

### PROCEDURAL Level
**Raw improvements** - Specific correction patterns
- "When task is X, use Y instead of Z"
- Stored as individual improvements

### SEMANTIC Level
**Consolidated patterns** - Synthesized from PROCEDURAL
- "General pattern: Use flowchart TD for flowcharts"
- Created by consolidation cycle

### META Level
**Learning wisdom** - When to use what patterns
- "Simple diagrams should use minimal nodes"
- Promoted from SEMANTIC after consolidation

## Benefits

### ✅ Efficiency
- **Raw**: Load all improvements (potentially many)
- **Synthesized**: Load consolidated summary (one text)

### ✅ Intelligence
- **Raw**: Individual patterns
- **Synthesized**: Emergent patterns, contradictions resolved

### ✅ Scalability
- **Raw**: Grows linearly with improvements
- **Synthesized**: Stays compact, extracts essence

## Comparison

| Feature | Raw Improvements | Synthesized Improvements |
|---------|------------------|-------------------------|
| **Storage** | PROCEDURAL level | SEMANTIC level |
| **Format** | List of dicts | Single text summary |
| **Size** | Grows with count | Compact summary |
| **Patterns** | Individual | Emergent |
| **Contradictions** | May exist | Resolved |
| **Use Case** | Detailed learning | Quick reference |

## Integration Points

### ExpertAgent._load_improvements()

```python
def _load_improvements(self, use_synthesis: bool = False):
    if use_synthesis:
        # Use synthesis for consolidated improvements
        synthesized = retrieve_synthesized_improvements(...)
        return [{"learned_pattern": synthesized, "is_synthesized": True}]
    else:
        # Use raw improvements
        return retrieve_improvements_from_memory(...)
```

### After Training

```python
# After training, run consolidation
await run_improvement_consolidation_cycle(memory, expert_name, domain)
```

## Files Updated

- `core/experts/memory_integration.py` - Added consolidation functions
- `core/experts/expert_agent.py` - Added synthesis support
- `core/experts/__init__.py` - Exported new functions
- `docs/EXPERT_AGENTS_MEMORY_CONSOLIDATION.md` - This document

## Conclusion

✅ **Memory consolidation integrated**  
✅ **Synthesized improvements available**  
✅ **Consolidation cycle supported**  
✅ **Backward compatible** (raw improvements still default)

**Expert agents now support brain-inspired memory consolidation!** 🧠✨

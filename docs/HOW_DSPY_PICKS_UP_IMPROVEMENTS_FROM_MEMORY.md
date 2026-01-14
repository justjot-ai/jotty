# How DSPy Picks Up Improvements from Memory System

## Complete Flow

```
Training → Store in Memory → Load from Memory → Inject into DSPy → Use in Generation
```

## Step-by-Step Process

### 1. Storage (Training Time)

**Location**: Jotty's HierarchicalMemory System

```python
# optimization_pipeline.py _record_improvement()
improvement = {
    "learned_pattern": "When task is 'Generate simple flowchart', use 'graph TD...'"
}

# Store to memory
store_improvement_to_memory(
    memory=memory_system,
    improvement=improvement,
    expert_name="mermaid_expert",
    domain="mermaid"
)

# Stored as:
# - PROCEDURAL level memory
# - Goal: "expert_mermaid_improvements"
# - Content: JSON string of improvement
```

### 2. Loading (Next Run)

**Location**: ExpertAgent initialization

```python
# expert_agent.py __init__()
self.memory = memory  # HierarchicalMemory instance
self.improvements = self._load_improvements()

# _load_improvements() does:
# 1. Query memory: "expert agent mermaid_expert improvements mermaid"
# 2. Retrieve from PROCEDURAL and META levels
# 3. Parse JSON from memory entries
# 4. Return as list of improvements
```

### 3. Integration (Agent Creation)

**Location**: ExpertAgent._create_agents()

```python
# Improvements injected into DSPy signature
agent = self._create_mermaid_agent(improvements=self.improvements)

# Signature docstring now includes:
"""
Generate Mermaid diagram...

## Learned Patterns:
- When task is 'Generate simple flowchart', use 'graph TD...'
"""
```

### 4. Application (Generation)

**Location**: ExpertAgent.generate()

```python
# Apply improvements to module
apply_improvements_to_dspy_module(agent, self.improvements)

# Pass as input field
improvements_str = create_improvements_context(self.improvements)
result = agent(learned_improvements=improvements_str, ...)
```

### 5. Usage (DSPy/LLM)

**Location**: DSPy module execution

```python
# DSPy sees improvements in:
# 1. Signature docstring (LLM reads this)
# 2. learned_improvements input field
# 3. Module instructions (if available)

# LLM generates using learned patterns!
```

## Memory System Benefits

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

Memory system automatically detects and merges duplicates.

### ✅ Goal-Conditioned Retrieval

Only retrieves improvements relevant to specific goal.

### ✅ Memory Consolidation

Old patterns consolidated, important patterns strengthened.

## How DSPy Uses Memory-Stored Improvements

### Method 1: Signature Docstring

```python
# Improvements injected into signature
class MermaidGenerationSignature(dspy.Signature):
    """Generate Mermaid diagram...
    
    ## Learned Patterns (from memory):
    - When task is 'Generate simple flowchart', use 'graph TD...'
    """
```

**Why it works**: LLM reads docstring as part of prompt.

### Method 2: Input Field

```python
# Improvements passed as input
agent(
    task="Generate flowchart",
    learned_improvements="When task is 'Generate simple flowchart'..."
)
```

**Why it works**: LLM sees improvements as explicit input.

### Method 3: Module Instructions

```python
# Improvements added to module
module.instructions.append("When task is 'Generate simple flowchart'...")
```

**Why it works**: Module uses instructions during generation.

## Complete Example

```python
from core.memory.cortex import HierarchicalMemory
from core.experts import MermaidExpertAgent

# 1. Create memory system
memory = HierarchicalMemory(agent_name="mermaid_expert", config=config)

# 2. Create expert with memory
expert = MermaidExpertAgent(memory=memory)

# 3. Train (stores improvements to memory)
await expert.train()
# → Improvements stored to HierarchicalMemory (PROCEDURAL level)

# 4. Next run - expert loads from memory
expert2 = MermaidExpertAgent(memory=memory)
# → Loads improvements from memory via semantic search

# 5. Generate (uses improvements from memory)
diagram = await expert2.generate_mermaid(...)
# → DSPy sees improvements in signature docstring + input field
# → LLM generates using learned patterns!
```

## Test Results

```
✅ Improvement stored to memory (PROCEDURAL level)
✅ Improvement retrieved from memory
✅ Expert loaded improvements from memory
✅ DSPy signature updated with improvements
✅ Generated output uses learned patterns
```

## Conclusion

**DSPy picks up improvements from memory system through:**

1. ✅ **Memory Storage**: Improvements stored to HierarchicalMemory
2. ✅ **Memory Retrieval**: Expert loads improvements via semantic search
3. ✅ **Signature Injection**: Improvements added to DSPy signature docstring
4. ✅ **Input Field**: Improvements passed as explicit input
5. ✅ **Module Instructions**: Improvements added to module (if available)

**All three methods ensure DSPy uses stored improvements!** 🎉

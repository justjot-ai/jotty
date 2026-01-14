# How DSPy Fetches Improvements from Memory and Consolidation

## Complete Flow Diagram

```
ExpertAgent.__init__()
    ↓
_load_improvements(use_synthesis=False/True)
    ↓
┌─────────────────────────────────────────┐
│ OPTION 1: Raw Improvements             │
│ memory.retrieve()                       │
│ → PROCEDURAL, META, SEMANTIC levels    │
│ → Returns List[MemoryEntry]             │
│ → Parsed to List[Dict]                  │
└─────────────────────────────────────────┘
    OR
┌─────────────────────────────────────────┐
│ OPTION 2: Synthesized Improvements     │
│ memory.retrieve_and_synthesize()        │
│ → LLM synthesizes all improvements      │
│ → Returns consolidated text             │
│ → Converted to single improvement dict │
└─────────────────────────────────────────┘
    ↓
self.improvements = List[Dict]
    ↓
ExpertAgent.generate()
    ↓
_create_agents() → Creates DSPy module
    ↓
apply_improvements_to_dspy_module()
    ↓
┌─────────────────────────────────────────┐
│ METHOD 1: Signature Docstring          │
│ inject_improvements_into_signature()    │
│ → Updates signature.__doc__             │
│ → LLM reads docstring during generation │
└─────────────────────────────────────────┘
    +
┌─────────────────────────────────────────┐
│ METHOD 2: Module Instructions           │
│ apply_improvements_to_dspy_module()     │
│ → Updates module.instructions           │
│ → DSPy uses instructions                │
└─────────────────────────────────────────┘
    +
┌─────────────────────────────────────────┐
│ METHOD 3: Input Field                    │
│ create_improvements_context()           │
│ → Formats improvements as string         │
│ → Passed as learned_improvements param  │
│ → LLM sees as explicit input            │
└─────────────────────────────────────────┘
    ↓
DSPy Module Execution
    ↓
LLM generates using all three methods
```

## Step-by-Step Flow

### Step 1: Expert Agent Initialization

**Location**: `ExpertAgent.__init__()`

```python
# expert_agent.py line 107
self.improvements = self._load_improvements(use_synthesis=config.use_memory_synthesis)
```

**What happens:**
- Calls `_load_improvements()` with synthesis flag
- Stores improvements in `self.improvements`

### Step 2: Loading Improvements from Memory

**Location**: `ExpertAgent._load_improvements()`

#### Option A: Raw Improvements (Default)

```python
# expert_agent.py lines 149-175
memory_entries = self.memory.retrieve(
    query=f"expert agent improvements {self.config.domain} {self.config.name}",
    goal=f"expert_{self.config.domain}_improvements",
    budget_tokens=10000,
    levels=[MemoryLevel.PROCEDURAL, MemoryLevel.META, MemoryLevel.SEMANTIC]
)

# Convert memory entries to improvement format
for entry in memory_entries:
    improvement_data = json.loads(entry.content)
    improvements.append(improvement_data)
```

**Memory System Flow:**
1. `memory.retrieve()` called
2. HierarchicalMemory searches PROCEDURAL, META, SEMANTIC levels
3. Uses LLM-based RAG to find relevant memories
4. Returns `List[MemoryEntry]`
5. Each entry's `content` is JSON string of improvement
6. Parsed to `List[Dict]`

#### Option B: Synthesized Improvements

```python
# expert_agent.py lines 128-147
synthesized = retrieve_synthesized_improvements(
    memory=self.memory,
    expert_name=self.config.name,
    domain=self.config.domain
)

# Convert to improvement format
improvements.append({
    "learned_pattern": synthesized,
    "is_synthesized": True
})
```

**Memory System Flow:**
1. `retrieve_synthesized_improvements()` called
2. Calls `memory.retrieve_and_synthesize()`
3. HierarchicalMemory retrieves all relevant memories
4. **LLM synthesizes** all memories into coherent text
5. Returns single synthesized string
6. Converted to single improvement dict

**Memory Consolidation:**
- `memory.retrieve_and_synthesize()` uses `LLMRAGRetriever.retrieve_and_synthesize()`
- LLM analyzes all memories and creates consolidated summary
- Finds patterns, resolves contradictions, extracts wisdom

### Step 3: Applying Improvements to DSPy Module

**Location**: `ExpertAgent.generate()`

```python
# expert_agent.py lines 401-405
if self.improvements:
    from .dspy_improvements import apply_improvements_to_dspy_module
    apply_improvements_to_dspy_module(agent, self.improvements)
```

**What happens:**
- Gets improvements from `self.improvements` (loaded in Step 2)
- Applies to DSPy module using three methods

### Step 4: Three Methods of Injection

#### Method 1: Signature Docstring Injection

**Location**: `MermaidExpertAgent._create_mermaid_agent()`

```python
# mermaid_expert.py lines 86-89
if improvements:
    from .dspy_improvements import inject_improvements_into_signature
    signature_class = inject_improvements_into_signature(MermaidGenerationSignature, improvements)
```

**What happens:**
```python
# dspy_improvements.py lines 127-178
# Creates new signature class with improvements in docstring
new_doc = original_doc + "\n\n## Learned Patterns:\n" + patterns
new_signature = type(signature_class.__name__, (signature_class,), {'__doc__': new_doc})
```

**How DSPy uses it:**
- DSPy signature docstring becomes part of LLM prompt
- LLM reads docstring during generation
- Improvements are in the prompt context

#### Method 2: Module Instructions

**Location**: `apply_improvements_to_dspy_module()`

```python
# dspy_improvements.py lines 77-84
if hasattr(module, 'instructions'):
    if isinstance(module.instructions, list):
        module.instructions.extend(patterns)
    elif isinstance(module.instructions, str):
        module.instructions = f"{module.instructions}\n\n## Learned Patterns\n\n{instructions_text}"
```

**What happens:**
- Updates `module.instructions` attribute
- DSPy modules use instructions during generation
- Instructions guide LLM behavior

#### Method 3: Input Field

**Location**: `ExpertAgent.generate()`

```python
# expert_agent.py lines 377-395
improvements_str = create_improvements_context(self.improvements)
result = self._call_dspy_agent(
    agent,
    task=task,
    learned_improvements=improvements_str,
    **context
)
```

**What happens:**
```python
# dspy_improvements.py lines 96-124
def create_improvements_context(improvements):
    patterns = [f"- {imp.get('learned_pattern', '')}" for imp in improvements[-5:]]
    return "\n".join(["## Previously Learned Patterns:", *patterns])
```

**How DSPy uses it:**
- Improvements passed as `learned_improvements` input field
- LLM sees as explicit input parameter
- Direct context for generation

### Step 5: DSPy Module Execution

**Location**: `ExpertAgent._call_dspy_agent()`

```python
# expert_agent.py lines 482-490
if self._is_dspy_module(agent):
    # Call DSPy module directly
    result = agent(**kwargs)  # Includes learned_improvements
else:
    result = agent.forward(**kwargs)
```

**What happens:**
- DSPy module called with all parameters
- LLM receives:
  1. Signature docstring (with improvements)
  2. Module instructions (with improvements)
  3. `learned_improvements` input field (with improvements)
- LLM generates using all three sources

## Memory Consolidation Flow

### When Consolidation Happens

**Option 1: Manual Consolidation**

```python
from core.experts.memory_integration import consolidate_improvements

result = consolidate_improvements(memory, "mermaid_expert", "mermaid")
```

**Flow:**
1. Retrieves all PROCEDURAL improvements
2. Uses `memory.retrieve_and_synthesize()` to consolidate
3. Stores consolidated pattern as SEMANTIC memory
4. Future retrievals include SEMANTIC level

**Option 2: Automatic via Synthesis**

```python
config = ExpertAgentConfig(
    use_memory_synthesis=True  # Enable synthesis
)
```

**Flow:**
1. `_load_improvements(use_synthesis=True)` called
2. Calls `retrieve_synthesized_improvements()`
3. Uses `memory.retrieve_and_synthesize()` to get consolidated text
4. Returns single synthesized improvement
5. DSPy uses synthesized improvement (all three methods)

### Consolidation Process

**Location**: `consolidate_improvements()`

```python
# memory_integration.py lines 263-320
# 1. Retrieve PROCEDURAL improvements
procedural_improvements = memory.retrieve(
    query=f"expert agent {expert_name} domain {domain} improvements",
    goal=f"expert_{domain}_improvements",
    levels=[MemoryLevel.PROCEDURAL]
)

# 2. Synthesize using LLM
synthesized = memory.retrieve_and_synthesize(
    query=f"Consolidate and extract patterns from expert agent improvements",
    goal=f"expert_{domain}_improvements",
    levels=[MemoryLevel.PROCEDURAL],
    context_hints="Extract common patterns, best practices, and general rules"
)

# 3. Store as SEMANTIC memory
consolidated_entry = memory.store(
    content=synthesized,
    level=MemoryLevel.SEMANTIC,
    ...
)
```

**Memory System Internals:**
- `memory.retrieve_and_synthesize()` → `LLMRAGRetriever.retrieve_and_synthesize()`
- LLM analyzes all memories
- Creates consolidated summary
- Returns synthesized text

## Complete Example Flow

### Example: Mermaid Expert with Synthesized Improvements

```python
# 1. Create expert with synthesis
config = ExpertAgentConfig(
    name="mermaid_expert",
    domain="mermaid",
    use_memory_synthesis=True
)
expert = MermaidExpertAgent(config=config, memory=memory)

# 2. Expert loads improvements (happens in __init__)
# → _load_improvements(use_synthesis=True)
# → retrieve_synthesized_improvements()
# → memory.retrieve_and_synthesize()
# → LLM synthesizes all improvements
# → Returns: "When generating flowcharts, use 'flowchart TD'..."
# → Stored in: expert.improvements = [{"learned_pattern": synthesized, ...}]

# 3. Generate diagram
diagram = await expert.generate_mermaid("Create a flowchart")

# 4. Expert.generate() called
# → _create_agents() creates DSPy module
# → apply_improvements_to_dspy_module(agent, expert.improvements)
#    → Updates signature docstring
#    → Updates module instructions
# → create_improvements_context(expert.improvements)
#    → Returns formatted string
# → agent(learned_improvements=improvements_str, ...)

# 5. DSPy module execution
# → LLM sees:
#    - Signature docstring: "Generate Mermaid diagram...\n## Learned Patterns:\n..."
#    - Module instructions: ["When generating flowcharts, use 'flowchart TD'..."]
#    - Input field: learned_improvements="## Previously Learned Patterns:\n- ..."
# → LLM generates using all three sources
```

## Key Points

### Where Improvements Come From

1. **Memory Storage**: HierarchicalMemory (PROCEDURAL, META, SEMANTIC levels)
2. **Memory Retrieval**: `memory.retrieve()` or `memory.retrieve_and_synthesize()`
3. **Memory Consolidation**: LLM synthesizes multiple improvements into patterns

### How DSPy Gets Them

1. **Signature Docstring**: Injected during agent creation
2. **Module Instructions**: Applied during generation
3. **Input Field**: Passed as `learned_improvements` parameter

### When Consolidation Happens

1. **Manual**: Call `consolidate_improvements()` or `run_improvement_consolidation_cycle()`
2. **Automatic**: Use `use_memory_synthesis=True` in config
3. **Memory System**: Uses `retrieve_and_synthesize()` for LLM-based consolidation

## Files Involved

- `core/experts/expert_agent.py` - Loads improvements, applies to DSPy
- `core/experts/memory_integration.py` - Memory retrieval and consolidation
- `core/experts/dspy_improvements.py` - DSPy injection methods
- `core/memory/cortex.py` - HierarchicalMemory retrieval/synthesis
- `core/memory/llm_rag.py` - LLM-based RAG and synthesis

## Conclusion

**DSPy fetches improvements from memory through:**

1. ✅ **ExpertAgent initialization** → Loads from memory (raw or synthesized)
2. ✅ **Memory retrieval** → `memory.retrieve()` or `memory.retrieve_and_synthesize()`
3. ✅ **DSPy injection** → Three methods (signature, instructions, input field)
4. ✅ **LLM generation** → Uses all three sources during generation

**Memory consolidation happens:**

1. ✅ **Manual consolidation** → `consolidate_improvements()` creates SEMANTIC patterns
2. ✅ **Synthesis retrieval** → `retrieve_synthesized_improvements()` uses LLM synthesis
3. ✅ **Automatic** → `use_memory_synthesis=True` enables synthesis by default

**All improvements flow from memory → ExpertAgent → DSPy → LLM!** 🎉

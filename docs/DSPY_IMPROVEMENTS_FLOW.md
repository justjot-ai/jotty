# DSPy Improvements Flow - Visual Guide

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                            │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  OptimizationPipeline            │
        │  - Agent generates output        │
        │  - Teacher provides correction   │
        │  - Improvement recorded          │
        └─────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  _record_improvement()           │
        │  - Creates improvement dict      │
        │  - Saves to JSON file            │
        └─────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  STORAGE: improvements.json                                 │
│  {                                                           │
│    "iteration": 1,                                           │
│    "task": "Generate flowchart",                            │
│    "student_output": "...",                                 │
│    "teacher_output": "...",                                 │
│    "learned_pattern": "When task is '...', use '...'"      │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘

                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    NEXT RUN                                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  ExpertAgent.__init__()          │
        │  - Loads improvements.json      │
        │  - Stores in self.improvements  │
        └─────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  INTEGRATION PHASE                                          │
└─────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
┌──────────────────┐            ┌──────────────────┐
│ Method 1:        │            │ Method 2:        │
│ Signature        │            │ Module            │
│ Docstring        │            │ Instructions      │
│                  │            │                  │
│ Inject into      │            │ Apply to          │
│ __doc__          │            │ instructions      │
└──────────────────┘            └──────────────────┘
        │                                   │
        └─────────────────┬─────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  Method 3: Input Field           │
        │  - Pass as learned_improvements │
        │  - LLM sees as explicit input    │
        └─────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    GENERATION PHASE                         │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  DSPy Module                    │
        │  - Has improvements in docstring│
        │  - Has improvements in input     │
        │  - Has improvements in instr.    │
        └─────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  LLM (Claude/Cursor)            │
        │  - Reads signature docstring    │
        │  - Sees learned_improvements     │
        │  - Uses learned patterns        │
        └─────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │  Generated Output               │
        │  - Uses learned patterns        │
        │  - Matches previous corrections │
        └─────────────────────────────────┘
```

## Code Flow

### 1. Storage (Training)

```python
# optimization_pipeline.py
def _record_improvement(...):
    improvement = {
        "learned_pattern": "...",
        ...
    }
    # Save to JSON
    with open(self.improvements_file, 'w') as f:
        json.dump(existing + [improvement], f)
```

### 2. Loading (Next Run)

```python
# expert_agent.py
def __init__(self, config):
    self.improvements_file = self.data_dir / "improvements.json"
    self.improvements = self._load_improvements()  # ← Loads from JSON
```

### 3. Integration (Agent Creation)

```python
# expert_agent.py
def _create_agents(self):
    agent = self._create_default_agent(improvements=self.improvements)
    # Improvements passed to agent creation

# mermaid_expert.py
def _create_mermaid_agent(improvements=None):
    if improvements:
        signature_class = inject_improvements_into_signature(
            MermaidGenerationSignature,
            improvements
        )
    # Signature now has improvements in docstring
```

### 4. Application (Generation)

```python
# expert_agent.py
def generate(self, task, context):
    # Apply improvements to module
    apply_improvements_to_dspy_module(agent, self.improvements)
    
    # Pass as input
    improvements_str = create_improvements_context(self.improvements)
    result = agent(learned_improvements=improvements_str, ...)
```

### 5. Usage (DSPy/LLM)

```python
# DSPy sees:
# 1. Signature docstring with learned patterns
# 2. learned_improvements input field
# 3. Module instructions (if available)

# LLM generates using all three!
```

## Key Files

| File | Purpose |
|------|---------|
| `core/orchestration/optimization_pipeline.py` | Records improvements to JSON |
| `core/experts/expert_agent.py` | Loads improvements, applies to DSPy |
| `core/experts/dspy_improvements.py` | Utilities to inject improvements into DSPy |
| `core/experts/mermaid_expert.py` | Creates agent with improvements |

## Test Verification

Run:
```bash
python tests/test_dspy_improvements_integration.py
```

**Expected:**
- ✅ Improvements loaded from JSON
- ✅ Improvements injected into signature
- ✅ Improvements applied to module
- ✅ Generated output uses learned patterns

## Conclusion

**DSPy picks up improvements through:**
1. ✅ **Signature docstring** - LLM reads this
2. ✅ **Input field** - Explicit context
3. ✅ **Module instructions** - If available

**All three methods work together** to ensure improvements are used! 🎉

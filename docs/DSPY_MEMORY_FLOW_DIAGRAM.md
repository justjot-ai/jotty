# DSPy Memory Flow - Visual Diagram

## Complete Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERT AGENT INITIALIZATION                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ExpertAgent.__init__()                                        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ self.improvements = _load_improvements()                 │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────┴─────────────────────┐
        │                                             │
        ▼                                             ▼
┌───────────────────────┐              ┌──────────────────────────┐
│ RAW IMPROVEMENTS      │              │ SYNTHESIZED IMPROVEMENTS │
│ (Default)             │              │ (use_memory_synthesis)   │
└───────────────────────┘              └──────────────────────────┘
        │                                             │
        ▼                                             ▼
┌───────────────────────┐              ┌──────────────────────────┐
│ memory.retrieve()     │              │ retrieve_synthesized_    │
│                       │              │   improvements()          │
│ Levels:               │              │                          │
│ - PROCEDURAL          │              │ memory.retrieve_and_     │
│ - META                │              │   synthesize()            │
│ - SEMANTIC            │              │                          │
└───────────────────────┘              │ LLM synthesizes all      │
        │                               │ improvements → text      │
        ▼                               └──────────────────────────┘
┌───────────────────────┐                          │
│ Returns:              │                          │
│ List[MemoryEntry]     │                          │
│                       │                          │
│ Each entry.content:   │                          │
│ JSON string of        │                          │
│ improvement           │                          │
└───────────────────────┘                          │
        │                                             │
        ▼                                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Parse JSON → List[Dict]                                       │
│  self.improvements = [                                          │
│    {"learned_pattern": "...", ...},                            │
│    {"learned_pattern": "...", ...},                            │
│    ...                                                          │
│  ]                                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERT AGENT GENERATION                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  ExpertAgent.generate()                                        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ agents = _create_agents()                                 │ │
│  │ agent = agents[0].agent  # DSPy module                   │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  apply_improvements_to_dspy_module(agent, self.improvements)   │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ METHOD 1:     │    │ METHOD 2:     │    │ METHOD 3:     │
│ Signature     │    │ Instructions  │    │ Input Field   │
│ Docstring     │    │               │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ inject_       │    │ module.       │    │ create_       │
│ improvements_ │    │ instructions  │    │ improvements_ │
│ into_         │    │ = patterns   │    │ context()      │
│ signature()   │    │               │    │               │
│               │    │               │    │ Returns:      │
│ Updates:      │    │ Updates:      │    │ "## Learned  │
│ signature.    │    │ module.       │    │ Patterns:     │
│ __doc__       │    │ instructions  │    │ - pattern1    │
│               │    │               │    │ - pattern2"    │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  _call_dspy_agent(agent, learned_improvements=..., ...)        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ result = agent(                                            │ │
│  │     task=task,                                             │ │
│  │     learned_improvements=improvements_str,  # Method 3     │ │
│  │     description=...,                                       │ │
│  │     diagram_type=...                                        │ │
│  │ )                                                           │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DSPY MODULE EXECUTION                        │
│                                                                 │
│  LLM receives:                                                  │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ 1. Signature Docstring (Method 1)                       │ │
│  │    "Generate Mermaid diagram...                          │ │
│  │     ## Learned Patterns:                                 │ │
│  │     - When generating flowcharts, use 'flowchart TD'..." │ │
│  └──────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ 2. Module Instructions (Method 2)                        │ │
│  │    ["When generating flowcharts, use 'flowchart TD'..."] │ │
│  └──────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ 3. Input Field (Method 3)                                 │ │
│  │    learned_improvements="## Previously Learned Patterns: │ │
│  │    - When generating flowcharts, use 'flowchart TD'..." │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM GENERATION                                │
│                                                                 │
│  LLM uses all three sources to generate output:                 │
│  - Reads signature docstring                                    │
│  - Follows module instructions                                  │
│  - Considers learned_improvements input                          │
│                                                                 │
│  Generates: Mermaid diagram using learned patterns              │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Consolidation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    MANUAL CONSOLIDATION                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  consolidate_improvements(memory, expert_name, domain)           │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ 1. Retrieve PROCEDURAL improvements                       │ │
│  │    procedural = memory.retrieve(levels=[PROCEDURAL])     │ │
│  └───────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ 2. Synthesize using LLM                                   │ │
│  │    synthesized = memory.retrieve_and_synthesize()        │ │
│  │    → LLM analyzes all improvements                       │ │
│  │    → Extracts patterns, resolves contradictions          │ │
│  │    → Returns consolidated text                          │ │
│  └───────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ 3. Store as SEMANTIC memory                               │ │
│  │    memory.store(level=SEMANTIC, content=synthesized)    │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Future retrievals include SEMANTIC level                       │
│  → Consolidated patterns available                              │
└─────────────────────────────────────────────────────────────────┘
```

## Key Points

1. **Memory → ExpertAgent**: `_load_improvements()` retrieves from memory
2. **ExpertAgent → DSPy**: Three methods inject improvements
3. **DSPy → LLM**: All three methods feed into LLM prompt
4. **Consolidation**: LLM synthesizes multiple improvements into patterns

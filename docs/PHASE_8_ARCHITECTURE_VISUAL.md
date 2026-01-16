# Phase 8 Architecture Visual - Expert System Integration

## Current Architecture (Phase 7 - Fragmented)

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER CODE                                     │
└─────────────────────────────────────────────────────────────────┘
              ↓                                  ↓
    ┌──────────────────────┐        ┌──────────────────────┐
    │ SingleAgent          │        │ ExpertAgent          │
    │ Orchestrator         │        │ (Separate System)    │
    ├──────────────────────┤        ├──────────────────────┤
    │ • Architect          │        │ • Optimization       │
    │ • Agent Execute      │        │   Pipeline           │
    │ • Auditor            │        │ • Gold Standards     │
    │ • TD-lambda          │        │ • Validation         │
    │ • Q-learning         │        │ • Memory Storage     │
    │ • Memory             │        │ • Improvements       │
    └──────────────────────┘        └──────────────────────┘
              ↓                                  ↓
    ┌──────────────────────────────────────────────────────┐
    │      MultiAgentsOrchestrator                         │
    │      ❌ Can't coordinate experts properly            │
    └──────────────────────────────────────────────────────┘
```

**Problems:**
- ❌ Two separate systems for agents
- ❌ Duplication of validation, learning, memory
- ❌ Experts can't use SingleAgent features
- ❌ No team coordination for experts

---

## Proposed Architecture (Phase 8 - Unified)

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER CODE                                │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│              SingleAgentOrchestrator (UNIVERSAL BASE)            │
├─────────────────────────────────────────────────────────────────┤
│  CORE FEATURES (always present):                                │
│  ├── Architect → Agent → Auditor validation                     │
│  ├── TD-lambda learning                                         │
│  ├── Q-learning                                                 │
│  ├── Credit assignment                                          │
│  ├── Hierarchical memory                                        │
│  └── Episode management                                         │
│                                                                  │
│  🆕 OPTIONAL: Gold Standard Learning (enable_gold_standard_learning=True)
│  ├── OptimizationPipeline integration                           │
│  ├── Gold standard examples                                     │
│  ├── Validation cases                                           │
│  └── Continuous improvement                                     │
└─────────────────────────────────────────────────────────────────┘
                ↓                                    ↓
    ┌───────────────────────┐        ┌───────────────────────┐
    │  Regular Agent        │        │  Expert Agent         │
    │  (no gold standards)  │        │  (with gold standards)│
    ├───────────────────────┤        ├───────────────────────┤
    │ SingleAgent           │        │ SingleAgent           │
    │ Orchestrator(         │        │ Orchestrator(         │
    │   agent=...,          │        │   agent=...,          │
    │   architect_prompts,  │        │   architect_prompts,  │
    │   auditor_prompts     │        │   auditor_prompts,    │
    │ )                     │        │   enable_gold=True,   │
    │                       │        │   gold_standards=[...]│
    │                       │        │   domain="mermaid"    │
    │                       │        │ )                     │
    └───────────────────────┘        └───────────────────────┘
                ↓                                    ↓
┌─────────────────────────────────────────────────────────────────┐
│           MultiAgentsOrchestrator (TEAM COORDINATION)            │
├─────────────────────────────────────────────────────────────────┤
│  Coordinates team of SingleAgentOrchestrator instances:         │
│  ├── Mix of experts and non-experts                             │
│  ├── All share same execution path                              │
│  ├── Team-level learning                                        │
│  └── Gold standard sharing across team                          │
└─────────────────────────────────────────────────────────────────┘
```

**Benefits:**
- ✅ Single execution path for all agents
- ✅ No code duplication
- ✅ Experts get all SingleAgent features
- ✅ Team coordination works for everyone

---

## Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: SingleAgentOrchestrator (Universal Base)              │
│           └── Base for ALL agents (expert or not)               │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: Expert Templates (Domain-Specific Factories)          │
│           ├── create_mermaid_expert()                           │
│           ├── create_sql_expert()                               │
│           ├── create_plantuml_expert()                          │
│           ├── create_data_analysis_expert()                     │
│           └── create_custom_expert(domain, gold_standards)      │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: MultiAgentsOrchestrator (Team Coordination)           │
│           └── Coordinates multiple SingleAgent instances        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Feature Composition

### Regular Agent (Base Features Only)

```python
agent = SingleAgentOrchestrator(
    agent=my_dspy_module,
    architect_prompts=["planner.md"],
    auditor_prompts=["validator.md"]
)
```

**Features:**
- ✅ Architect validation
- ✅ Agent execution
- ✅ Auditor validation
- ✅ TD-lambda learning
- ✅ Memory storage
- ❌ Gold standard learning (disabled)

### Expert Agent (Base + Gold Standards)

```python
expert = SingleAgentOrchestrator(
    agent=my_dspy_module,
    architect_prompts=["planner.md"],
    auditor_prompts=["validator.md"],
    # 🆕 Expert features
    enable_gold_standard_learning=True,
    gold_standards=[...],
    domain="mermaid"
)
```

**Features:**
- ✅ Architect validation
- ✅ Agent execution
- ✅ Auditor validation
- ✅ TD-lambda learning
- ✅ Memory storage
- ✅ Gold standard learning (enabled)
- ✅ OptimizationPipeline
- ✅ Validation cases
- ✅ Continuous improvement

---

## Expert Templates Pattern

### Creating Experts (Easy!)

```python
# Option 1: Use pre-built template
from Jotty.core.experts.expert_templates import create_mermaid_expert

expert = create_mermaid_expert(config=JottyConfig())
```

### What Templates Do

```python
def create_mermaid_expert(
    config: JottyConfig = None,
    gold_standards: List[Dict] = None
) -> SingleAgentOrchestrator:
    """Factory for Mermaid expert agent."""

    # Load defaults
    if gold_standards is None:
        gold_standards = load_mermaid_gold_standards()

    # Domain-specific configuration
    return SingleAgentOrchestrator(
        agent=dspy.ChainOfThought(MermaidSignature),
        architect_prompts=[
            "prompts/experts/mermaid/planning.md",
            "prompts/experts/mermaid/diagram_types.md"
        ],
        auditor_prompts=[
            "prompts/experts/mermaid/validation.md",
            "prompts/experts/mermaid/syntax_check.md"
        ],
        # Expert configuration
        enable_gold_standard_learning=True,
        gold_standards=gold_standards,
        validation_cases=load_mermaid_validation_cases(),
        domain="mermaid",
        domain_validator=MermaidValidator().validate
    )
```

---

## Team Composition

### Mixed Team (Experts + Non-Experts)

```python
orchestrator = MultiAgentsOrchestrator(
    actors=[
        # Expert 1: Mermaid diagrams
        ActorConfig(
            name="MermaidExpert",
            agent=create_mermaid_expert(config)  # ← Expert
        ),

        # Expert 2: SQL queries
        ActorConfig(
            name="SQLExpert",
            agent=create_sql_expert(config)  # ← Expert
        ),

        # Non-Expert: Data fetcher
        ActorConfig(
            name="DataFetcher",
            agent=SingleAgentOrchestrator(  # ← Regular agent
                agent=dspy.ReAct(FetchSignature),
                architect_prompts=["planner.md"],
                auditor_prompts=["validator.md"]
            )
        ),
    ],
    config=config
)
```

**Team Workflow:**
```
User Goal: "Fetch data, analyze it, create SQL query, generate diagram"
    ↓
MultiAgentsOrchestrator coordinates:
    ↓
1. DataFetcher (regular agent) → fetches data
    ↓
2. SQLExpert (expert) → generates optimized SQL query (uses gold standards)
    ↓
3. MermaidExpert (expert) → creates diagram (uses gold standards)
    ↓
Result: All agents work together seamlessly!
```

---

## Execution Flow Comparison

### Before Phase 8 (Two Separate Paths)

```
Path 1 (Regular Agent):
User → SingleAgentOrchestrator → Architect → Agent → Auditor → Result

Path 2 (Expert Agent):
User → ExpertAgent → OptimizationPipeline → Validation → Result
       ↑
       Separate system, can't use Architect/Auditor
```

### After Phase 8 (One Unified Path)

```
User → SingleAgentOrchestrator
       ├→ [Optional] Gold Standard Optimization
       ├→ Architect Validation
       ├→ Agent Execution
       ├→ Auditor Validation
       ├→ Learning (TD-lambda, Q-learning)
       └→ [Optional] Store as New Gold Standard
       → Result

All agents (expert or not) follow the same path!
```

---

## Migration Impact

### Files Changed

**New files:**
- `core/experts/expert_templates.py` - Factory functions for experts

**Modified files:**
- `core/orchestration/single_agent_orchestrator.py` - Add gold standard learning
- `core/experts/expert_agent.py` - Deprecate, make factory wrapper

**Unchanged files:**
- `core/orchestration/conductor.py` (MultiAgentsOrchestrator) - No changes needed!
- All existing validation, learning, memory code - Works as-is

### Backward Compatibility

**Old Code (ExpertAgent):**
```python
config = ExpertAgentConfig(name="Expert", domain="mermaid")
expert = ExpertAgent(config)  # Shows deprecation warning
```

**New Code (SingleAgentOrchestrator):**
```python
expert = create_mermaid_expert(config=JottyConfig())
```

**Both work!** Old code shows deprecation warning but continues to function.

---

## Summary Table

| Aspect | Before Phase 8 | After Phase 8 |
|--------|----------------|---------------|
| **Base for all agents** | SingleAgentOrchestrator | SingleAgentOrchestrator |
| **Expert agents** | Separate ExpertAgent class | SingleAgentOrchestrator + flag |
| **Gold standard learning** | Only in ExpertAgent | Optional in any SingleAgent |
| **Code duplication** | ❌ Yes (validation, memory) | ✅ No (shared codebase) |
| **Team coordination** | ❌ Doesn't work with experts | ✅ Works with all agents |
| **Feature composition** | ❌ Can't combine features | ✅ Experts get all features |
| **Execution paths** | 2 separate paths | 1 unified path |
| **Creating experts** | Manual ExpertAgent setup | Easy templates |
| **Custom experts** | Complex | Simple factory function |
| **Backward compatibility** | N/A | 100% (deprecated wrapper) |

---

## Visual: Expert Template Usage

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Import Template                                        │
│  from Jotty.core.experts.expert_templates import (              │
│      create_mermaid_expert,                                     │
│      create_sql_expert,                                         │
│      create_plantuml_expert                                     │
│  )                                                              │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Create Expert Instance                                 │
│  expert = create_mermaid_expert(                                │
│      config=JottyConfig(),                                      │
│      gold_standards=[...]  # Optional: custom examples          │
│  )                                                              │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Use Like Any Agent                                     │
│  result = await expert.arun(                                    │
│      question="Generate sequence diagram for user login"        │
│  )                                                              │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  Behind the Scenes (Automatic):                                 │
│  1. Check gold standards for similar examples                   │
│  2. Run OptimizationPipeline if needed                          │
│  3. Architect validation                                        │
│  4. Agent execution                                             │
│  5. Auditor validation                                          │
│  6. Store successful result as new gold standard                │
│  7. Return validated result                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Insight

**The genius of Phase 8:**

Every agent is just a `SingleAgentOrchestrator` with different configuration!

- Regular agent = SingleAgentOrchestrator(enable_gold=False)
- Expert agent = SingleAgentOrchestrator(enable_gold=True, gold_standards=[...])

**No separate classes needed!** Just configuration differences.

This is the Unix philosophy applied to AI agents:
- **Single base class** that does one thing well
- **Composition** of features through configuration
- **Templates** for common patterns
- **Teams** coordinate any combination

**Result:** Clean, flexible, maintainable architecture! 🎉

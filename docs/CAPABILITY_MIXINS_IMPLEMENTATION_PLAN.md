# Capability Mixins Implementation Plan

**Date:** 2026-02-16
**Status:** Implementation Ready

---

## Architecture Decision: Where Things Go

### 1. Capability Mixins → `core/intelligence/reasoning/capabilities/`

**Rationale:**
- Capabilities are core agent functionality
- Lives with BaseAgent (single agent hierarchy)
- Reusable by ALL agents (swarm agents, experts, domain agents)
- Clear namespace: `from core.intelligence.reasoning.capabilities import LearningCapability`

**Structure:**
```
core/intelligence/reasoning/
├── base/
│   ├── base_agent.py          # Core BaseAgent
│   ├── domain_agent.py         # DSPy-based agents
│   └── ...
├── capabilities/               # ✅ NEW - Capability mixins
│   ├── __init__.py            # Public exports
│   ├── learning_capability.py  # Gold standard learning
│   ├── validation_capability.py # Domain validation
│   ├── memory_capability.py    # Enhanced memory integration
│   └── optimization_capability.py # Optimization pipeline
├── agents/
│   ├── swarm_agent.py
│   └── ...
└── types/
    └── ...
```

---

### 2. Expert Classes → REFACTOR in `core/intelligence/reasoning/experts/`

**Rationale:**
- Experts are domain specialists (Mermaid, PlantUML, LaTeX, etc.)
- Already organized by domain
- Keep existing location BUT refactor to use mixins
- Gradual migration (no big bang)

**Strategy:**
```
core/intelligence/reasoning/experts/
├── base_expert.py             # ⚠️ DEPRECATE - Replace with mixins
├── mermaid_expert.py          # ✅ REFACTOR to use LearningCapability
├── plantuml_expert.py         # ✅ REFACTOR to use LearningCapability
├── math_latex_expert.py       # ✅ REFACTOR to use LearningCapability
├── backend_expert.py          # ✅ REFACTOR to use LearningCapability
└── ...
```

**After refactoring:**
- Each expert becomes: `BaseAgent + LearningCapability + ValidationCapability`
- Remove dependency on `BaseExpert`
- Use standard agent interface (`_execute_impl()`)

---

### 3. Training Data → Centralized in `core/intelligence/knowledge/`

**Rationale:**
- Training data is KNOWLEDGE, not code
- Separate data from code (clean architecture)
- Centralized location for all domain knowledge
- Easy to version control, share, and maintain
- Can be loaded from multiple sources (files, DB, memory)

**New Structure:**
```
core/intelligence/knowledge/              # ✅ NEW - Centralized knowledge
├── __init__.py
├── gold_standards/                       # Gold standard examples
│   ├── mermaid/
│   │   ├── diagrams.json                # Mermaid training examples
│   │   └── validation_cases.json        # Validation cases
│   ├── plantuml/
│   │   ├── diagrams.json
│   │   └── github_examples.json
│   ├── latex/
│   │   ├── equations.json
│   │   └── validation_cases.json
│   ├── coding/
│   │   ├── python_examples.json
│   │   └── best_practices.json
│   └── research/
│       ├── analysis_examples.json
│       └── report_templates.json
├── learned_improvements/                 # ⚠️ DEPRECATED - Use memory instead
│   └── README.md                        # "Use SwarmMemory instead"
└── loaders/
    ├── __init__.py
    ├── base_loader.py                   # Base class for data loaders
    ├── json_loader.py                   # Load from JSON files
    ├── github_loader.py                 # Load from GitHub repos
    └── memory_loader.py                 # Load from SwarmMemory
```

**Benefits:**
- ✅ Clear separation: code vs data
- ✅ Domain-organized training data
- ✅ Easy to add new domains
- ✅ Centralized knowledge base
- ✅ Versioned with git
- ✅ Can share across agents

---

### 4. Learned Improvements → SwarmMemory (Not Files!)

**Rationale:**
- Improvements are learned knowledge → belongs in memory
- Memory system already handles: retrieval, ranking, consolidation
- No file I/O overhead
- Automatic deduplication and synthesis
- Persists across sessions
- Can be queried by relevance

**Memory Levels:**
- **PROCEDURAL**: Specific patterns (how to fix diagram syntax)
- **META**: General wisdom (when to use which pattern)

**Example:**
```python
# Store improvement to memory (not file)
memory.store(
    content=json.dumps(improvement),
    level=MemoryLevel.PROCEDURAL,
    context={
        "expert": "mermaid",
        "domain": "diagram_generation",
        "task": task,
        "improvement_type": "syntax_correction"
    },
    goal="mermaid_diagram_improvements",
    initial_value=1.0  # High value for learned patterns
)

# Retrieve improvements
improvements = memory.retrieve(
    query=task,
    goal="mermaid_diagram_improvements",
    level=MemoryLevel.PROCEDURAL,
    top_k=10
)
```

**Migration:**
- Existing file-based improvements → Bulk load into memory
- Going forward → Always use memory
- Delete old improvement files after migration

---

## Implementation Plan

### Phase 1: Create Capability Mixins (Week 1)

**Files to create:**

1. **`core/intelligence/reasoning/capabilities/__init__.py`**
```python
"""
Agent Capability Mixins

Reusable capabilities that can be mixed into any BaseAgent:
- LearningCapability: Gold standard learning
- ValidationCapability: Domain validation
- MemoryCapability: Enhanced memory integration
- OptimizationCapability: Optimization pipeline
"""

from .learning_capability import LearningCapability
from .validation_capability import ValidationCapability
from .memory_capability import MemoryCapability
from .optimization_capability import OptimizationCapability

__all__ = [
    "LearningCapability",
    "ValidationCapability",
    "MemoryCapability",
    "OptimizationCapability",
]
```

2. **`core/intelligence/reasoning/capabilities/learning_capability.py`** (Full implementation below)

3. **`core/intelligence/reasoning/capabilities/validation_capability.py`**

4. **`core/intelligence/reasoning/capabilities/memory_capability.py`**

5. **`core/intelligence/reasoning/capabilities/optimization_capability.py`**

### Phase 2: Create Knowledge Repository (Week 1)

**Files to create:**

1. **`core/intelligence/knowledge/__init__.py`**
```python
"""
Knowledge Repository for Jotty

Centralized location for:
- Gold standard training examples
- Validation cases
- Domain-specific knowledge
- Best practices

Training data is loaded dynamically, not imported.
"""

from .loaders import (
    load_gold_standards,
    load_validation_cases,
    GoldStandardLoader,
)

__all__ = [
    "load_gold_standards",
    "load_validation_cases",
    "GoldStandardLoader",
]
```

2. **`core/intelligence/knowledge/loaders/base_loader.py`**

3. **Migrate existing data:**
```bash
# Move expert data to centralized knowledge
mkdir -p core/intelligence/knowledge/gold_standards/{mermaid,plantuml,latex}

# Move existing training data
mv core/intelligence/reasoning/experts/data/plantuml_expert/github_training_examples.json \
   core/intelligence/knowledge/gold_standards/plantuml/examples.json
```

### Phase 3: Refactor 2-3 Experts as Proof-of-Concept (Week 2)

**Target experts:**
1. MermaidExpert → MermaidAgent (with LearningCapability)
2. PlantUMLExpert → PlantUMLAgent (with LearningCapability)
3. LatexExpert → LatexAgent (with LearningCapability + ValidationCapability)

**Migration pattern:**
```python
# OLD (BaseExpert)
class MermaidExpert(BaseExpert):
    @property
    def domain(self):
        return "mermaid"

    def _create_domain_agent(self, improvements=None):
        return MermaidModule()

# NEW (BaseAgent + mixins)
from core.intelligence.reasoning.base import BaseAgent
from core.intelligence.reasoning.capabilities import LearningCapability, ValidationCapability
from core.intelligence.knowledge import load_gold_standards

class MermaidAgent(BaseAgent, LearningCapability, ValidationCapability):
    def __init__(self, config=None, enable_learning=True):
        # Initialize base agent
        BaseAgent.__init__(self, config or AgentRuntimeConfig(name="Mermaid"))

        # Load gold standards from centralized knowledge
        if enable_learning:
            gold_standards = load_gold_standards("mermaid")
            LearningCapability.__init__(
                self,
                gold_standards=gold_standards,
                domain_validator=self._validate_mermaid
            )

        # Initialize validation
        ValidationCapability.__init__(self, domain="mermaid")

    async def _execute_impl(self, task: str, **kwargs):
        """Generate Mermaid diagram."""
        # Generate diagram
        diagram = await self._generate_diagram(task)

        # Validate syntax
        diagram = await self.validate(diagram, **kwargs)

        # Learn and improve (if enabled)
        if hasattr(self, 'learn_from_gold_standards'):
            diagram = await self.learn_from_gold_standards(task, diagram)

        return {"diagram": diagram, "syntax_valid": True}

    async def _generate_diagram(self, task: str) -> str:
        """Generate Mermaid diagram from task description."""
        # DSPy generation logic
        pass

    async def _validate_mermaid(self, output, expected, task, context):
        """Validate Mermaid diagram."""
        # Validation logic
        pass
```

### Phase 4: Migrate Remaining Experts (Week 3-4)

**Remaining experts:**
- BackendExpert → BackendAgent
- FrontendExpert → FrontendAgent
- DesignerExpert → DesignerAgent
- PipelineExpert → PipelineAgent
- QAExpert → QAAgent
- UXResearcherExpert → UXResearcherAgent
- ProductManagerExpert → ProductManagerAgent

### Phase 5: Enable Learning in Swarm Agents (Week 5)

**Add learning to existing swarm agents:**
```python
# core/intelligence/swarms/research_swarm/agents.py

from core.intelligence.reasoning.capabilities import LearningCapability
from core.intelligence.knowledge import load_gold_standards

class DataFetcherAgent(BaseAgent, LearningCapability):
    """Fetches financial data with optional learning."""

    def __init__(self, config=None, enable_learning=False):
        BaseAgent.__init__(self, config)

        # Optional learning from gold standards
        if enable_learning:
            gold_standards = load_gold_standards("research/data_fetching")
            LearningCapability.__init__(self, gold_standards)
```

### Phase 6: Deprecate BaseExpert (Week 6)

1. Add deprecation warnings to `base_expert.py`
2. Update all imports to use new agents
3. Update documentation
4. Mark for removal in next major version

---

## File Organization Summary

### Final Structure

```
core/
├── modes/agent/
│   ├── base/
│   │   └── base_agent.py                    # Core agent
│   ├── capabilities/                         # ✅ NEW - Reusable mixins
│   │   ├── __init__.py
│   │   ├── learning_capability.py
│   │   ├── validation_capability.py
│   │   ├── memory_capability.py
│   │   └── optimization_capability.py
│   └── agents/
│       └── swarm_agent.py
│
├── intelligence/
│   ├── knowledge/                            # ✅ NEW - Centralized knowledge
│   │   ├── __init__.py
│   │   ├── gold_standards/                  # Training data by domain
│   │   │   ├── mermaid/
│   │   │   ├── plantuml/
│   │   │   ├── latex/
│   │   │   ├── coding/
│   │   │   └── research/
│   │   └── loaders/                         # Data loaders
│   │       ├── __init__.py
│   │       ├── json_loader.py
│   │       └── github_loader.py
│   │
│   ├── reasoning/experts/                    # Refactored experts
│   │   ├── base_expert.py                   # ⚠️ DEPRECATED
│   │   ├── mermaid_agent.py                 # ✅ Uses mixins
│   │   ├── plantuml_agent.py                # ✅ Uses mixins
│   │   └── ...
│   │
│   ├── memory/
│   │   └── cortex.py                        # Stores learned improvements
│   │
│   └── swarms/
│       └── research_swarm/
│           └── agents.py                    # Can use LearningCapability
```

---

## Migration Checklist

### Week 1: Foundation
- [ ] Create `core/intelligence/reasoning/capabilities/` directory
- [ ] Implement `LearningCapability` mixin
- [ ] Implement `ValidationCapability` mixin
- [ ] Implement `MemoryCapability` mixin
- [ ] Implement `OptimizationCapability` mixin
- [ ] Create `core/intelligence/knowledge/` directory
- [ ] Create knowledge loaders
- [ ] Move existing training data to knowledge repo

### Week 2: Proof of Concept
- [ ] Refactor MermaidExpert → MermaidAgent
- [ ] Refactor PlantUMLExpert → PlantUMLAgent
- [ ] Refactor LatexExpert → LatexAgent
- [ ] Test with real workflows
- [ ] Validate learning works
- [ ] Verify memory integration

### Week 3-4: Full Migration
- [ ] Migrate BackendExpert
- [ ] Migrate FrontendExpert
- [ ] Migrate DesignerExpert
- [ ] Migrate PipelineExpert
- [ ] Migrate QAExpert
- [ ] Migrate UXResearcherExpert
- [ ] Migrate ProductManagerExpert

### Week 5: Enable in Swarms
- [ ] Add LearningCapability to DataFetcherAgent
- [ ] Add LearningCapability to WebSearchAgent
- [ ] Add LearningCapability to ArchitectAgent
- [ ] Add LearningCapability to DeveloperAgent
- [ ] Create gold standards for each agent type
- [ ] Test end-to-end learning

### Week 6: Cleanup
- [ ] Add deprecation warnings to BaseExpert
- [ ] Update documentation
- [ ] Update CLAUDE.md
- [ ] Update examples
- [ ] Remove old improvement files
- [ ] Mark BaseExpert for removal

---

## Benefits of This Organization

### Code Benefits
✅ **Single agent hierarchy** (BaseAgent only)
✅ **Reusable capabilities** (mix-and-match)
✅ **No code duplication** (DRY)
✅ **Clear responsibilities** (separation of concerns)

### Data Benefits
✅ **Centralized knowledge** (easy to find/maintain)
✅ **Domain-organized** (clear structure)
✅ **Version controlled** (git)
✅ **Shareable** (across agents)

### Memory Benefits
✅ **Learned improvements in memory** (not files)
✅ **Automatic consolidation** (memory system)
✅ **Relevance-based retrieval** (smart)
✅ **Persists across sessions** (durable)

### Developer Benefits
✅ **Clear where to add new agents** (modes/agent/agents/)
✅ **Clear where to add training data** (intelligence/knowledge/gold_standards/)
✅ **Clear how to enable learning** (add LearningCapability)
✅ **Consistent patterns** (one way to do things)

---

## Next Steps

1. **Approve this plan** ✅
2. **Start with Phase 1** (create capability mixins)
3. **Implement proof-of-concept** (2-3 experts)
4. **Iterate based on learnings**
5. **Full migration**

Ready to implement?

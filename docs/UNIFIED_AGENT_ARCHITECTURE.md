# Unified Agent Architecture - Final Design

**Date:** 2026-02-16
**Status:** Approved Architecture

---

## Core Principle: One Agent Hierarchy

**All agents live in `core/modes/agent/`** - No exceptions!

The distinction between "agents" and "experts" is eliminated:
- **Agent** = BaseAgent
- **Expert** = BaseAgent + LearningCapability
- **Learnable Agent** = BaseAgent + LearningCapability
- **Domain Specialist** = BaseAgent + LearningCapability + ValidationCapability

---

## Directory Structure

```
core/
├── modes/agent/                          # SINGLE agent hierarchy
│   ├── __init__.py
│   │
│   ├── base/                             # Base classes
│   │   ├── __init__.py
│   │   ├── base_agent.py                # Core BaseAgent
│   │   ├── domain_agent.py              # DSPy-based agent
│   │   └── types.py                     # AgentRuntimeConfig, AgentResult
│   │
│   ├── capabilities/                     # Reusable mixins
│   │   ├── __init__.py
│   │   ├── learning_capability.py       # Gold standard learning
│   │   ├── validation_capability.py     # Domain validation
│   │   ├── memory_capability.py         # Enhanced memory
│   │   └── optimization_capability.py   # Optimization pipeline
│   │
│   └── agents/                           # ALL agents here
│       ├── __init__.py
│       │
│       ├── domain/                       # Domain specialists (ex-experts)
│       │   ├── __init__.py
│       │   ├── mermaid_agent.py         # Mermaid diagrams
│       │   ├── plantuml_agent.py        # PlantUML diagrams
│       │   ├── latex_agent.py           # LaTeX equations
│       │   ├── backend_agent.py         # Backend code generation
│       │   ├── frontend_agent.py        # Frontend code generation
│       │   ├── designer_agent.py        # Design work
│       │   ├── pipeline_agent.py        # Pipeline definitions
│       │   ├── qa_agent.py              # QA testing
│       │   └── ux_researcher_agent.py   # UX research
│       │
│       ├── swarm/                        # Swarm-specific agents
│       │   ├── __init__.py
│       │   ├── swarm_agent.py           # Base swarm agent
│       │   └── learning_agent.py        # SwarmLearningAgent
│       │
│       └── research/                     # Research domain (optional)
│           ├── __init__.py
│           ├── data_fetcher.py          # Data fetching
│           ├── web_search.py            # Web search
│           └── sentiment.py             # Sentiment analysis
│
├── intelligence/
│   ├── knowledge/                        # Training data repository
│   │   ├── __init__.py
│   │   ├── gold_standards/              # Gold standards by domain
│   │   │   ├── mermaid/
│   │   │   │   ├── diagrams.json
│   │   │   │   └── validation_cases.json
│   │   │   ├── plantuml/
│   │   │   ├── latex/
│   │   │   ├── coding/
│   │   │   └── research/
│   │   └── loaders/                     # Data loaders
│   │       ├── __init__.py
│   │       ├── json_loader.py
│   │       └── github_loader.py
│   │
│   ├── reasoning/
│   │   └── experts/                     # ⚠️ DEPRECATED - Remove after migration
│   │       └── README.md                # "Moved to core/modes/agent/agents/domain/"
│   │
│   └── memory/
│       └── cortex.py                    # Stores learned improvements
```

---

## Import Patterns

### Clean, Consistent Imports

```python
# Base agent
from core.modes.agent.base import BaseAgent, AgentRuntimeConfig

# Capabilities
from core.modes.agent.capabilities import (
    LearningCapability,
    ValidationCapability,
    MemoryCapability,
)

# Domain specialists
from core.modes.agent.agents.domain import (
    MermaidAgent,
    PlantUMLAgent,
    LatexAgent,
)

# Swarm agents
from core.modes.agent.agents.swarm import SwarmAgent

# Gold standards
from core.intelligence.knowledge import load_gold_standards
```

### No More Confusion!

```python
# ❌ OLD (confusing):
from core.intelligence.reasoning.experts import MermaidExpert
from core.modes.agent.agents import SwarmAgent
# Why are similar things in different places?

# ✅ NEW (clear):
from core.modes.agent.agents.domain import MermaidAgent
from core.modes.agent.agents.swarm import SwarmAgent
# All agents in one hierarchy!
```

---

## Agent Creation Patterns

### Simple Agent (No Learning)

```python
from core.modes.agent.base import BaseAgent, AgentRuntimeConfig

class SimpleAgent(BaseAgent):
    """Basic agent without learning."""

    async def _execute_impl(self, **kwargs):
        return {"result": "done"}
```

### Domain Specialist (With Learning)

```python
from core.modes.agent.base import BaseAgent, AgentRuntimeConfig
from core.modes.agent.capabilities import LearningCapability, ValidationCapability
from core.intelligence.knowledge import load_gold_standards

class MermaidAgent(BaseAgent, LearningCapability, ValidationCapability):
    """Mermaid diagram generation with learning and validation."""

    def __init__(self, config=None, enable_learning=True):
        # Base agent initialization
        BaseAgent.__init__(self, config or AgentRuntimeConfig(name="Mermaid"))

        # Add learning capability
        if enable_learning:
            gold_standards = load_gold_standards("mermaid")
            LearningCapability.__init__(
                self,
                gold_standards=gold_standards,
                domain_validator=self._validate_diagram
            )

        # Add validation capability
        ValidationCapability.__init__(self, domain="mermaid")

    async def _execute_impl(self, task: str, **kwargs):
        """Generate Mermaid diagram."""
        diagram = await self._generate_diagram(task)
        diagram = await self.validate(diagram, **kwargs)

        # Learn and improve if capability enabled
        if hasattr(self, 'learn_from_gold_standards'):
            diagram = await self.learn_from_gold_standards(task, diagram)

        return {"diagram": diagram, "valid": True}

    async def _generate_diagram(self, task: str) -> str:
        """Generate diagram using DSPy."""
        pass

    async def _validate_diagram(self, output, expected, task, context):
        """Validate diagram syntax."""
        pass
```

---

## Migration Plan

### Phase 1: Create New Structure (Week 1)

**Create directories:**
```bash
mkdir -p core/modes/agent/capabilities
mkdir -p core/modes/agent/agents/domain
mkdir -p core/modes/agent/agents/swarm
mkdir -p core/modes/agent/agents/research
mkdir -p core/intelligence/knowledge/gold_standards/{mermaid,plantuml,latex,coding}
```

**Create capability mixins:**
- `core/modes/agent/capabilities/learning_capability.py`
- `core/modes/agent/capabilities/validation_capability.py`
- `core/modes/agent/capabilities/memory_capability.py`

**Create knowledge loaders:**
- `core/intelligence/knowledge/loaders/json_loader.py`
- `core/intelligence/knowledge/loaders/github_loader.py`

### Phase 2: Migrate Domain Specialists (Week 2-3)

**Move and refactor experts:**

```bash
# Mermaid
mv core/intelligence/reasoning/experts/mermaid_expert.py \
   core/modes/agent/agents/domain/mermaid_agent.py

# PlantUML
mv core/intelligence/reasoning/experts/plantuml_expert.py \
   core/modes/agent/agents/domain/plantuml_agent.py

# LaTeX
mv core/intelligence/reasoning/experts/math_latex_expert.py \
   core/modes/agent/agents/domain/latex_agent.py

# Backend
mv core/intelligence/reasoning/experts/backend_expert.py \
   core/modes/agent/agents/domain/backend_agent.py

# Frontend
mv core/intelligence/reasoning/experts/frontend_expert.py \
   core/modes/agent/agents/domain/frontend_agent.py

# Designer
mv core/intelligence/reasoning/experts/designer_expert.py \
   core/modes/agent/agents/domain/designer_agent.py
```

**Refactor each file:**
- Change base class from `BaseExpert` to `BaseAgent + LearningCapability`
- Update imports
- Update class names (Expert → Agent)
- Use standard `_execute_impl()` pattern

### Phase 3: Update All Imports (Week 3-4)

**Find and replace:**
```bash
# Update imports across codebase
grep -r "from.*intelligence.reasoning.experts import" . | cut -d: -f1 | sort -u

# Update to new location
# FROM: from core.intelligence.reasoning.experts import MermaidExpert
# TO:   from core.modes.agent.agents.domain import MermaidAgent
```

### Phase 4: Move Training Data (Week 4)

**Centralize knowledge:**
```bash
# Move existing training data
mv core/intelligence/reasoning/experts/data/plantuml_expert/*.json \
   core/intelligence/knowledge/gold_standards/plantuml/

# Create gold standards for other domains
# (manual work - curate high-quality examples)
```

### Phase 5: Cleanup (Week 5)

**Remove deprecated code:**
```bash
# Remove old experts folder (after all migrations)
rm -rf core/intelligence/reasoning/experts/

# Add README explaining migration
echo "DEPRECATED: Moved to core/modes/agent/agents/domain/" > \
  core/intelligence/reasoning/experts/README.md
```

---

## Benefits of Unified Structure

### Developer Benefits

✅ **One place to look**: All agents in `modes/agent/agents/`
✅ **Clear organization**: domain/, swarm/, research/ subdirs
✅ **Consistent imports**: All from `core.modes.agent`
✅ **Easy discovery**: Browse `agents/` to see all available agents
✅ **No confusion**: No more "is this an agent or expert?"

### Architectural Benefits

✅ **Single hierarchy**: One BaseAgent, multiple capabilities
✅ **DRY**: No duplication between agents and experts
✅ **Extensible**: Add new capabilities without changing structure
✅ **Clear separation**: Code (agents/) vs Data (knowledge/)
✅ **Standard patterns**: All agents use same interface

### Maintenance Benefits

✅ **One codebase**: Not split across intelligence/ and modes/
✅ **Easy refactoring**: All in one place
✅ **Clear ownership**: Agent team owns modes/agent/
✅ **Simple testing**: Test all agents from one location

---

## Comparison: Old vs New

### Old Structure (Fragmented)

```
modes/agent/agents/          # Some agents
intelligence/reasoning/experts/  # Other agents
# Why are they separate? Confusing!
```

### New Structure (Unified)

```
modes/agent/agents/
├── domain/     # Domain specialists (with learning)
├── swarm/      # Swarm agents
└── research/   # Research agents

# Everything in one place! Clear!
```

---

## Decision: APPROVED ✅

**Final Structure:**
- ALL agents → `core/modes/agent/agents/`
- Training data → `core/intelligence/knowledge/gold_standards/`
- Learned improvements → `SwarmMemory` (not files)

**Rationale:**
- True unification (not half-measures)
- Clear, consistent organization
- Easy to find and use
- Follows user's insight: "why not keep them together?"

**The user was right!** 🎯

---

## Next Steps

1. ✅ Approve this unified architecture
2. Create capability mixins
3. Create knowledge repository
4. Migrate domain specialists to `modes/agent/agents/domain/`
5. Update all imports
6. Remove deprecated experts folder

Ready to implement?

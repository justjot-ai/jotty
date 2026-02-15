# Experts vs Agents - Architectural Analysis & Unification Plan

**Date:** 2026-02-16
**Status:** Analysis & Recommendation

---

## Current Architecture

### 1. BaseAgent (`core/modes/agent/base/base_agent.py`)

**Purpose:** Generic agent framework for all task execution

**Features:**
- ✅ `_execute_impl()` for execution
- ✅ Retry logic with exponential backoff
- ✅ Error handling and recovery
- ✅ Memory access (lazy-loaded)
- ✅ Cost tracking and metrics
- ✅ Skills registry access
- ❌ No built-in learning from gold standards
- ❌ No domain-specific training
- ❌ No optimization pipeline

**Used by:**
- ResearchAgent, CodingAgent, DataFetcherAgent
- WebSearchAgent, SentimentAgent, etc.
- All swarm agents (14+ agents)

### 2. BaseExpert (`core/intelligence/reasoning/experts/base_expert.py`)

**Purpose:** Domain-specific expert framework with learning

**Features:**
- ✅ Training gold standards
- ✅ Validation cases
- ✅ Learns from improvements via OptimizationPipeline
- ✅ Memory integration for learned patterns (PROCEDURAL/META)
- ✅ Domain-specific evaluation
- ✅ Improvement injection into DSPy signatures
- ❌ No direct execution (needs ExpertAgent wrapper)
- ❌ No retry logic
- ❌ No standard agent interface

**Used by:**
- MermaidExpert, PlantUMLExpert, LatexExpert
- BackendExpert, FrontendExpert, DesignerExpert
- PipelineExpert, QAExpert, UXResearcherExpert
- Total: 10+ domain experts

### 3. ExpertAgent (**DEPRECATED in Phase 8**)

**Status:** ⚠️ **DEPRECATED** - Backward compatibility only

**Deprecation Notice:**
```python
# Old (deprecated):
from experts import ExpertAgent, ExpertAgentConfig
config = ExpertAgentConfig(name="Expert", domain="mermaid")
expert = ExpertAgent(config)

# New (recommended):
from orchestration import SingleAgentOrchestrator
expert = SingleAgentOrchestrator(
    agent=my_agent,
    enable_gold_standard_learning=True,
    gold_standards=[...],
    domain="my_domain"
)
```

---

## Key Differences

| Feature | BaseAgent | BaseExpert | Ideal (Unified) |
|---------|-----------|------------|-----------------|
| **Execution** | ✅ _execute_impl() | ❌ No direct exec | ✅ _execute_impl() |
| **Retry Logic** | ✅ Exponential backoff | ❌ No | ✅ Yes |
| **Error Handling** | ✅ Built-in | ❌ Basic | ✅ Advanced |
| **Training Data** | ❌ No | ✅ Gold standards | ✅ Optional |
| **Learning** | ❌ No | ✅ OptimizationPipeline | ✅ Optional flag |
| **Memory Integration** | ⚠️ Basic | ✅ PROCEDURAL/META | ✅ All levels |
| **Domain Validation** | ❌ No | ✅ Domain-specific | ✅ Optional |
| **Improvement Injection** | ❌ No | ✅ DSPy signature | ✅ Optional |
| **Skills Access** | ✅ Registry | ❌ No | ✅ Registry |
| **Cost Tracking** | ✅ Built-in | ❌ No | ✅ Built-in |
| **Use Case** | General tasks | Domain expertise | **Both!** |

---

## Problems with Current Architecture

1. **Code Duplication**: BaseAgent and BaseExpert have overlapping functionality
2. **Confusion**: Users don't know when to use Agent vs Expert
3. **Feature Gap**: Agents can't learn, Experts can't use skills
4. **Maintenance**: Two separate hierarchies to maintain
5. **Inconsistency**: Different patterns for similar functionality

---

## Unification Options

### Option A: Add Learning to BaseAgent (Simplest)

**Approach:** Extend BaseAgent with optional learning capabilities

```python
class BaseAgent(ABC):
    def __init__(
        self,
        config: AgentRuntimeConfig = None,
        enable_learning: bool = False,  # ✅ NEW
        gold_standards: List[Dict] = None,  # ✅ NEW
        domain_validator: Callable = None,  # ✅ NEW
    ):
        self.config = config or AgentRuntimeConfig(name=self.__class__.__name__)

        # Learning configuration
        self.enable_learning = enable_learning
        self.gold_standards = gold_standards or []
        self.domain_validator = domain_validator
        self._optimization_pipeline = None  # Lazy-loaded

        # Existing: memory, context, metrics, etc.
        ...

    async def execute(self, **kwargs):
        """Execute with optional learning."""
        result = await self._execute_with_retry(**kwargs)

        # If learning enabled, run optimization pipeline
        if self.enable_learning and self.gold_standards:
            result = await self._optimize_with_learning(result, **kwargs)

        return result
```

**Pros:**
- ✅ Simple to implement
- ✅ Backward compatible (learning is opt-in)
- ✅ Single agent hierarchy
- ✅ All agents can learn if needed

**Cons:**
- ⚠️ BaseAgent becomes larger
- ⚠️ Mixing concerns (execution + learning)

---

### Option B: Keep Separate (Current - NOT RECOMMENDED)

**Approach:** Maintain BaseAgent and BaseExpert as separate hierarchies

**Pros:**
- ✅ Separation of concerns
- ✅ No changes needed

**Cons:**
- ❌ Code duplication
- ❌ User confusion
- ❌ Feature gaps
- ❌ Maintenance burden

---

### Option C: Capability Mixins (RECOMMENDED ⭐)

**Approach:** BaseAgent is core, add optional capability mixins

```python
# Core base agent (unchanged)
class BaseAgent(ABC):
    """Core agent with execution, retry, memory, skills."""
    async def _execute_impl(self, **kwargs):
        pass

# Learning capability mixin
class LearningCapability:
    """Adds gold standard learning and optimization."""

    def __init__(self, gold_standards=None, domain_validator=None):
        self.gold_standards = gold_standards or []
        self.domain_validator = domain_validator
        self._optimization_pipeline = None

    async def _optimize_with_learning(self, result, **kwargs):
        """Run optimization pipeline to improve output."""
        if not self._optimization_pipeline:
            self._optimization_pipeline = create_optimization_pipeline(...)
        return await self._optimization_pipeline.run(result, **kwargs)

# Domain validation capability
class ValidationCapability:
    """Adds domain-specific validation."""

    async def validate(self, output, expected):
        """Validate output against expected."""
        pass

# Example: Learnable agent
class MermaidAgent(BaseAgent, LearningCapability, ValidationCapability):
    """Mermaid diagram agent with learning and validation."""

    def __init__(self, config, gold_standards=None):
        BaseAgent.__init__(self, config)
        LearningCapability.__init__(self, gold_standards)
        ValidationCapability.__init__(self)

    async def _execute_impl(self, **kwargs):
        # Generate mermaid diagram
        result = await self._generate_diagram(**kwargs)

        # Validate if capability enabled
        if hasattr(self, 'validate'):
            result = await self.validate(result, kwargs.get('expected'))

        # Learn if capability enabled
        if hasattr(self, '_optimize_with_learning'):
            result = await self._optimize_with_learning(result, **kwargs)

        return result

# Example: Simple agent without learning
class SimpleAgent(BaseAgent):
    """Basic agent without learning."""

    async def _execute_impl(self, **kwargs):
        return {"result": "done"}
```

**Pros:**
- ✅ Clean separation of concerns
- ✅ Mix-and-match capabilities
- ✅ Backward compatible
- ✅ No code duplication
- ✅ Easy to extend (add new capabilities)
- ✅ Single agent hierarchy

**Cons:**
- ⚠️ Slightly more complex initialization
- ⚠️ Multiple inheritance (Python handles well)

---

## Recommended Implementation Plan

### Phase 1: Create Capability Mixins

**Files to create:**
1. `core/modes/agent/capabilities/learning_capability.py`
2. `core/modes/agent/capabilities/validation_capability.py`
3. `core/modes/agent/capabilities/memory_capability.py`
4. `core/modes/agent/capabilities/__init__.py`

**Example: LearningCapability**
```python
# core/modes/agent/capabilities/learning_capability.py
from typing import Any, Callable, Dict, List, Optional

class LearningCapability:
    """
    Adds gold standard learning to agents.

    Usage:
        class MyAgent(BaseAgent, LearningCapability):
            def __init__(self, config, gold_standards=None):
                BaseAgent.__init__(self, config)
                LearningCapability.__init__(self, gold_standards)
    """

    def __init__(
        self,
        gold_standards: Optional[List[Dict[str, Any]]] = None,
        domain_validator: Optional[Callable] = None,
        enable_optimization: bool = True,
    ):
        self.gold_standards = gold_standards or []
        self.domain_validator = domain_validator
        self.enable_optimization = enable_optimization
        self._optimization_pipeline = None
        self._learned_improvements = []

    async def learn_from_gold_standards(self, task: str, output: Any):
        """Learn from gold standards and improve."""
        if not self.enable_optimization or not self.gold_standards:
            return output

        # Lazy-load optimization pipeline
        if self._optimization_pipeline is None:
            from Jotty.core.intelligence.orchestration import create_optimization_pipeline
            self._optimization_pipeline = create_optimization_pipeline(
                agent=self,
                gold_standards=self.gold_standards,
                evaluator=self.domain_validator,
            )

        # Run optimization
        improved_output, improvements = await self._optimization_pipeline.optimize(
            task=task,
            output=output,
        )

        # Store improvements
        self._learned_improvements.extend(improvements)

        return improved_output

    def get_learned_improvements(self) -> List[Dict[str, Any]]:
        """Get all learned improvements."""
        return self._learned_improvements
```

### Phase 2: Migrate Experts to Use Mixins

**Convert existing experts:**
```python
# OLD (BaseExpert):
class MermaidExpert(BaseExpert):
    @property
    def domain(self):
        return "mermaid"

    def _create_domain_agent(self, improvements=None):
        return MermaidModule()

# NEW (BaseAgent + LearningCapability):
class MermaidAgent(BaseAgent, LearningCapability, ValidationCapability):
    def __init__(self, config=None, gold_standards=None):
        BaseAgent.__init__(self, config or AgentRuntimeConfig(name="MermaidAgent"))
        LearningCapability.__init__(self, gold_standards)
        ValidationCapability.__init__(self)

    async def _execute_impl(self, task: str, **kwargs):
        # Generate diagram
        diagram = await self._generate_diagram(task)

        # Validate
        diagram = await self.validate(diagram, kwargs.get('expected'))

        # Learn and improve
        diagram = await self.learn_from_gold_standards(task, diagram)

        return {"diagram": diagram, "success": True}
```

### Phase 3: Update Templates to Use Learning Agents

**Enable learning in templates:**
```python
# core/intelligence/swarms/templates/research.py
from core.modes.agent.capabilities import LearningCapability

class ResearchAgent(BaseAgent, LearningCapability):
    def __init__(self, config, gold_standards=None):
        BaseAgent.__init__(self, config)

        # Enable learning if gold standards provided
        if gold_standards:
            LearningCapability.__init__(self, gold_standards, enable_optimization=True)

    async def _execute_impl(self, ticker: str, **kwargs):
        result = await self.research(ticker)

        # Learn if capability enabled
        if hasattr(self, 'learn_from_gold_standards'):
            result = await self.learn_from_gold_standards(ticker, result)

        return result
```

### Phase 4: Deprecate BaseExpert (Gradual)

1. Add deprecation warnings to BaseExpert
2. Update documentation to recommend BaseAgent + mixins
3. Migrate all experts to new pattern
4. Remove BaseExpert in future release

---

## Migration Examples

### Example 1: Simple Agent (No Learning)

```python
# Current
from core.modes.agent.base import BaseAgent

class SimpleAgent(BaseAgent):
    async def _execute_impl(self, **kwargs):
        return {"result": "done"}

# ✅ No change needed!
```

### Example 2: Agent with Learning

```python
# Current (doesn't exist - no learning support)
# N/A

# New (with LearningCapability)
from core.modes.agent.base import BaseAgent
from core.modes.agent.capabilities import LearningCapability

class LearnableAgent(BaseAgent, LearningCapability):
    def __init__(self, config, gold_standards=None):
        BaseAgent.__init__(self, config)
        LearningCapability.__init__(self, gold_standards)

    async def _execute_impl(self, task: str, **kwargs):
        result = await self.process(task)
        result = await self.learn_from_gold_standards(task, result)
        return result
```

### Example 3: Domain Expert

```python
# Current (BaseExpert)
from core.intelligence.reasoning.experts.base_expert import BaseExpert

class MermaidExpert(BaseExpert):
    @property
    def domain(self):
        return "mermaid"

    def _create_domain_agent(self, improvements=None):
        return MermaidModule()

# New (BaseAgent + mixins)
from core.modes.agent.base import BaseAgent
from core.modes.agent.capabilities import LearningCapability, ValidationCapability

class MermaidAgent(BaseAgent, LearningCapability, ValidationCapability):
    def __init__(self, config=None, gold_standards=None):
        BaseAgent.__init__(self, config or AgentRuntimeConfig(name="Mermaid"))
        LearningCapability.__init__(self, gold_standards)
        ValidationCapability.__init__(self, domain="mermaid")

    async def _execute_impl(self, task: str, **kwargs):
        diagram = await self._generate_diagram(task)
        diagram = await self.validate(diagram, **kwargs)
        diagram = await self.learn_from_gold_standards(task, diagram)
        return {"diagram": diagram}
```

---

## Benefits of Unification

1. **Single Agent Hierarchy**: One BaseAgent, multiple capabilities
2. **No Code Duplication**: Capabilities are reusable
3. **Clear Intent**: Want learning? Add LearningCapability
4. **Backward Compatible**: Existing agents work unchanged
5. **Easy to Extend**: Add new capabilities without modifying BaseAgent
6. **Mix-and-Match**: Agents can combine capabilities as needed
7. **Consistent Interface**: All agents use _execute_impl()

---

## Next Steps

1. ✅ Create capability mixins (3 files)
2. ✅ Migrate 1-2 experts as proof-of-concept
3. ✅ Test with real workflows
4. ✅ Update documentation
5. ✅ Gradually migrate remaining experts
6. ✅ Add deprecation warnings to BaseExpert
7. ✅ Remove BaseExpert in next major version

---

## Conclusion

**Recommendation:** Implement **Option C (Capability Mixins)** ⭐

This provides:
- Clean architecture with separation of concerns
- No code duplication
- Backward compatibility
- Easy extensibility
- Clear migration path from BaseExpert

The unification will eliminate confusion between Agents and Experts, reduce maintenance burden, and provide a consistent interface for all agent types while preserving the learning capabilities that make experts powerful.

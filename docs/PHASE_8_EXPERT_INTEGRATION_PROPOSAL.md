# Jotty Phase 8: Expert System Integration

**Date:** January 2026
**Status:** Proposed
**Goal:** Integrate expert system (gold standard learning) into SingleAgentOrchestrator architecture

---

## Current Problem

**Fragmented Architecture:**

```
Current State (Disconnected):

SingleAgentOrchestrator          ExpertAgent (Separate)
├── Architect validation         ├── OptimizationPipeline
├── Agent execution              ├── Gold standard training
├── Auditor validation           ├── Validation cases
├── TD-lambda learning           ├── Memory storage
└── Episode management           └── Improvements tracking

MultiAgentsOrchestrator
├── Coordinates actors
└── ❌ No integration with expert system
```

**Issues:**
1. **Duplication**: Expert agents re-implement validation, learning, memory
2. **Inconsistency**: Two separate paths for agent execution
3. **Complexity**: Expert agents can't use SingleAgentOrchestrator features
4. **No Team Learning**: Experts not integrated with MultiAgentsOrchestrator

---

## Proposed Solution

**Unified Architecture:** Expert system as **optional feature** of SingleAgentOrchestrator

```
Proposed Architecture:

SingleAgentOrchestrator (Base)
├── Core Validation: Architect → Agent → Auditor
├── Core Learning: TD-lambda, Q-learning, credit assignment
├── Core Memory: Hierarchical memory integration
└── 🆕 OPTIONAL: Gold Standard Learning
    ├── OptimizationPipeline integration
    ├── Gold standard examples
    ├── Validation cases
    └── Continuous improvement

Expert Agents (Templatized SingleAgentOrchestrator)
├── MermaidExpert = SingleAgentOrchestrator(domain="mermaid", gold_standards=[...])
├── PlantUMLExpert = SingleAgentOrchestrator(domain="plantuml", gold_standards=[...])
├── SQLExpert = SingleAgentOrchestrator(domain="sql", gold_standards=[...])
└── DataAnalysisExpert = SingleAgentOrchestrator(domain="data_analysis", gold_standards=[...])

MultiAgentsOrchestrator (Team Coordination)
├── Coordinates multiple SingleAgentOrchestrator instances
├── Each can be expert or non-expert
├── Team-level learning and memory sharing
└── Gold standard examples propagate across team
```

---

## Key Design Principles

### 1. SingleAgentOrchestrator = Universal Base

**Every agent is a SingleAgentOrchestrator**, whether expert or not:

```python
# Non-expert agent (standard)
agent = SingleAgentOrchestrator(
    agent=my_dspy_module,
    architect_prompts=["planner.md"],
    auditor_prompts=["validator.md"],
    # No gold standards
)

# Expert agent (with gold standard learning)
expert = SingleAgentOrchestrator(
    agent=my_dspy_module,
    architect_prompts=["planner.md"],
    auditor_prompts=["validator.md"],
    # 🆕 Gold standard learning enabled
    enable_gold_standard_learning=True,
    gold_standards=[...],
    validation_cases=[...],
    domain="mermaid"
)
```

### 2. Expert = Templatized SingleAgentOrchestrator

**Expert agents are just SingleAgentOrchestrator instances with domain-specific configuration:**

```python
# Create Mermaid expert (template)
def create_mermaid_expert(
    config: JottyConfig,
    gold_standards: List[Dict[str, Any]] = None
) -> SingleAgentOrchestrator:
    """Factory for Mermaid expert agent."""

    # Load default gold standards if not provided
    if gold_standards is None:
        gold_standards = load_mermaid_gold_standards()

    # Load domain-specific prompts
    architect_prompts = [
        "prompts/experts/mermaid/planning.md",
        "prompts/experts/mermaid/diagram_types.md"
    ]
    auditor_prompts = [
        "prompts/experts/mermaid/validation.md",
        "prompts/experts/mermaid/syntax_check.md"
    ]

    # Domain-specific validation function
    def mermaid_validator(output: str) -> bool:
        from ..experts.domain_validators import MermaidValidator
        validator = MermaidValidator()
        return validator.validate(output)

    # Create SingleAgentOrchestrator with expert features
    return SingleAgentOrchestrator(
        agent=dspy.ChainOfThought(MermaidSignature),
        architect_prompts=architect_prompts,
        auditor_prompts=auditor_prompts,
        architect_tools=[],
        auditor_tools=[],
        config=config,
        # 🆕 Expert features
        enable_gold_standard_learning=True,
        gold_standards=gold_standards,
        validation_cases=load_mermaid_validation_cases(),
        domain="mermaid",
        domain_validator=mermaid_validator,
        max_training_iterations=5
    )
```

### 3. MultiAgentsOrchestrator = Team of SingleAgents

**MultiAgentsOrchestrator coordinates a team**, whether experts or not:

```python
# Create team with mix of experts and non-experts
orchestrator = MultiAgentsOrchestrator(
    actors=[
        # Expert agents (with gold standards)
        ActorConfig(
            name="MermaidExpert",
            agent=create_mermaid_expert(config),
            enable_architect=True,
            enable_auditor=True
        ),
        ActorConfig(
            name="SQLExpert",
            agent=create_sql_expert(config),
            enable_architect=True,
            enable_auditor=True
        ),

        # Non-expert agent (standard)
        ActorConfig(
            name="DataFetcher",
            agent=SingleAgentOrchestrator(
                agent=dspy.ReAct(FetchSignature),
                architect_prompts=["planner.md"],
                auditor_prompts=["validator.md"],
                # No gold standards
            ),
            enable_architect=True,
            enable_auditor=True
        ),
    ],
    metadata_provider=provider,
    config=config
)

# All agents benefit from team-level coordination
result = await orchestrator.run(goal="Generate analytics dashboard")
```

---

## Implementation Plan

### Step 1: Add Gold Standard Learning to SingleAgentOrchestrator

**Extend SingleAgentOrchestrator with optional expert features:**

```python
class SingleAgentOrchestrator:
    """
    Single-agent orchestrator with optional gold standard learning.
    """

    def __init__(self,
                 agent: dspy.Module,
                 architect_prompts: List[str],
                 auditor_prompts: List[str],
                 architect_tools: List[Any],
                 auditor_tools: List[Any],
                 config: JottyConfig = None,
                 agent_config: 'ActorConfig' = None,
                 shared_context: Optional[Dict[str, Any]] = None,

                 # 🆕 Phase 8: Gold Standard Learning (optional)
                 enable_gold_standard_learning: bool = False,
                 gold_standards: Optional[List[Dict[str, Any]]] = None,
                 validation_cases: Optional[List[Dict[str, Any]]] = None,
                 domain: Optional[str] = None,
                 domain_validator: Optional[Callable] = None,
                 max_training_iterations: int = 5,
                 min_validation_score: float = 1.0,

                 # Backward compatibility
                 actor: dspy.Module = None,
                 actor_config: 'ActorConfig' = None):
        """
        Initialize SingleAgentOrchestrator.

        Parameters:
            ... (existing parameters)

            # Gold Standard Learning (Phase 8)
            enable_gold_standard_learning: Enable expert training with gold standards
            gold_standards: List of {input, expected_output} training examples
            validation_cases: List of validation test cases
            domain: Domain name for the expert (e.g., "mermaid", "sql")
            domain_validator: Custom validation function
            max_training_iterations: Max optimization iterations
            min_validation_score: Minimum score to pass validation
        """

        # Existing initialization
        # ...

        # 🆕 Phase 8: Gold Standard Learning
        self.enable_gold_standard_learning = enable_gold_standard_learning
        self.gold_standards = gold_standards or []
        self.validation_cases = validation_cases or []
        self.domain = domain
        self.domain_validator = domain_validator

        if enable_gold_standard_learning:
            # Initialize OptimizationPipeline
            from ..orchestration.optimization_pipeline import create_optimization_pipeline

            self.optimization_pipeline = create_optimization_pipeline(
                agent=self.agent,
                gold_standards=gold_standards,
                validation_function=domain_validator,
                config=OptimizationConfig(
                    max_iterations=max_training_iterations,
                    min_score=min_validation_score,
                    enable_teacher_model=True,
                    save_improvements=True
                )
            )

            logger.info(f"🎓 Gold standard learning enabled for domain: {domain}")
        else:
            self.optimization_pipeline = None

    async def arun(self, **kwargs) -> EpisodeResult:
        """
        Run episode with optional gold standard optimization.
        """

        # 🆕 Phase 8: Pre-execution optimization (if enabled)
        if self.enable_gold_standard_learning and self.optimization_pipeline:
            # Check if agent needs improvement
            if should_optimize(self.agent, kwargs):
                logger.info("🎓 Running optimization pipeline...")
                optimized_result = await self.optimization_pipeline.optimize(
                    task=kwargs,
                    context=kwargs.get('context', {})
                )

                # Update agent with improvements
                self.agent = optimized_result.improved_agent

        # Existing validation workflow (Architect → Agent → Auditor)
        # ...

        # 🆕 Phase 8: Post-execution learning (store examples)
        if self.enable_gold_standard_learning and result.success:
            # Add successful execution as new gold standard
            self.gold_standards.append({
                "input": kwargs,
                "expected_output": result.output,
                "validated": True,
                "timestamp": datetime.now().isoformat()
            })

            # Store in memory
            if self.memory:
                self.memory.store(
                    content=json.dumps(self.gold_standards[-1]),
                    context={
                        "domain": self.domain,
                        "type": "gold_standard",
                        "success": True
                    },
                    level=MemoryLevel.PROCEDURAL
                )

        return result
```

### Step 2: Create Expert Templates

**Create factory functions for common expert types:**

```python
# core/experts/expert_templates.py

from typing import List, Dict, Any
from ..orchestration import SingleAgentOrchestrator
from ..foundation import JottyConfig


def create_mermaid_expert(
    config: JottyConfig = None,
    gold_standards: List[Dict[str, Any]] = None
) -> SingleAgentOrchestrator:
    """Create Mermaid diagram expert."""

    from .domain_validators import MermaidValidator
    from .training_data_loader import TrainingDataLoader

    # Load default gold standards
    if gold_standards is None:
        loader = TrainingDataLoader(domain="mermaid")
        gold_standards = loader.load_from_github_repo(
            repo_url="https://github.com/mermaid-js/mermaid",
            path="packages/mermaid/src/diagrams/",
            file_pattern="*.spec.js",
            max_files=50
        )

    # Create validator
    validator = MermaidValidator()

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
        architect_tools=[],
        auditor_tools=[],
        config=config or JottyConfig(),
        # Expert features
        enable_gold_standard_learning=True,
        gold_standards=gold_standards,
        validation_cases=validator.get_test_cases(),
        domain="mermaid",
        domain_validator=validator.validate,
        max_training_iterations=5,
        min_validation_score=1.0
    )


def create_sql_expert(
    config: JottyConfig = None,
    gold_standards: List[Dict[str, Any]] = None
) -> SingleAgentOrchestrator:
    """Create SQL query expert."""

    from .domain_validators import SQLValidator

    # Load default gold standards
    if gold_standards is None:
        # Load from SQL examples database
        gold_standards = load_sql_gold_standards()

    validator = SQLValidator()

    return SingleAgentOrchestrator(
        agent=dspy.ChainOfThought(SQLSignature),
        architect_prompts=["prompts/experts/sql/planning.md"],
        auditor_prompts=["prompts/experts/sql/validation.md"],
        architect_tools=[],
        auditor_tools=[],
        config=config or JottyConfig(),
        # Expert features
        enable_gold_standard_learning=True,
        gold_standards=gold_standards,
        validation_cases=validator.get_test_cases(),
        domain="sql",
        domain_validator=validator.validate,
        max_training_iterations=5,
        min_validation_score=1.0
    )


def create_plantuml_expert(...) -> SingleAgentOrchestrator:
    """Create PlantUML diagram expert."""
    # Similar pattern
    pass


def create_data_analysis_expert(...) -> SingleAgentOrchestrator:
    """Create data analysis expert."""
    # Similar pattern
    pass
```

### Step 3: Deprecate Standalone ExpertAgent

**Make ExpertAgent a factory for SingleAgentOrchestrator:**

```python
# core/experts/expert_agent.py (Phase 8 - deprecated wrapper)

from ..orchestration import SingleAgentOrchestrator
from .expert_templates import (
    create_mermaid_expert,
    create_sql_expert,
    create_plantuml_expert
)


class ExpertAgent:
    """
    DEPRECATED: Use SingleAgentOrchestrator with enable_gold_standard_learning=True

    This class is now a factory wrapper for backward compatibility.
    """

    def __new__(cls, config: ExpertAgentConfig, memory=None):
        """Create SingleAgentOrchestrator instance based on domain."""

        import warnings
        warnings.warn(
            "ExpertAgent is deprecated. Use SingleAgentOrchestrator with "
            "enable_gold_standard_learning=True instead.",
            DeprecationWarning,
            stacklevel=2
        )

        # Map domain to expert template
        expert_factories = {
            "mermaid": create_mermaid_expert,
            "sql": create_sql_expert,
            "plantuml": create_plantuml_expert,
        }

        factory = expert_factories.get(config.domain)
        if not factory:
            raise ValueError(f"Unknown expert domain: {config.domain}")

        # Convert ExpertAgentConfig to SingleAgentOrchestrator parameters
        return factory(
            config=JottyConfig(),  # Map config
            gold_standards=config.training_gold_standards
        )
```

### Step 4: Update MultiAgentsOrchestrator

**No changes needed!** MultiAgentsOrchestrator already works with any agent, including SingleAgentOrchestrator instances with gold standard learning.

```python
# Example: Team with experts
orchestrator = MultiAgentsOrchestrator(
    actors=[
        ActorConfig(
            name="MermaidExpert",
            agent=create_mermaid_expert(config),  # ← SingleAgentOrchestrator with gold standards
            enable_architect=True,
            enable_auditor=True
        ),
        ActorConfig(
            name="SQLExpert",
            agent=create_sql_expert(config),  # ← SingleAgentOrchestrator with gold standards
            enable_architect=True,
            enable_auditor=True
        ),
    ],
    metadata_provider=provider,
    config=config
)
```

---

## Benefits of This Architecture

### 1. Unified Execution Path

**Before (Phase 7):**
```
Path 1: SingleAgentOrchestrator → Architect → Agent → Auditor → Learning
Path 2: ExpertAgent → OptimizationPipeline → Validation (separate system)
```

**After (Phase 8):**
```
Path: SingleAgentOrchestrator → Architect → Agent → Auditor → Learning
      └── Optional: Gold Standard Optimization
```

### 2. Feature Composition

**Any SingleAgentOrchestrator can be an expert:**
- ✅ Regular agent: Architect + Agent + Auditor + TD-lambda
- ✅ Expert agent: All above + Gold standard learning + Domain validation
- ✅ Mixed teams: Some agents experts, some not

### 3. Code Reuse

**No duplication:**
- Validation logic: One implementation (Architect/Auditor)
- Learning logic: One implementation (TD-lambda, Q-learning)
- Memory integration: One implementation (HierarchicalMemory)
- Gold standards: Optional add-on (not separate system)

### 4. Clearer Abstractions

**Architecture layers:**
```
Layer 1: SingleAgentOrchestrator (base for ALL agents)
Layer 2: Expert Templates (domain-specific factories)
Layer 3: MultiAgentsOrchestrator (team coordination)
```

### 5. Flexibility

**Create custom experts easily:**
```python
# Custom expert for any domain
def create_my_custom_expert(gold_standards):
    return SingleAgentOrchestrator(
        agent=my_dspy_module,
        architect_prompts=["my_prompts.md"],
        auditor_prompts=["my_validation.md"],
        enable_gold_standard_learning=True,
        gold_standards=gold_standards,
        domain="my_domain",
        domain_validator=my_validator
    )
```

---

## Migration Path

### Immediate (Phase 8.1)

1. ✅ Add gold standard learning parameters to SingleAgentOrchestrator
2. ✅ Integrate OptimizationPipeline into SingleAgentOrchestrator.arun()
3. ✅ Create expert_templates.py with factory functions
4. ✅ Test: Create Mermaid expert using new architecture

### Short-term (Phase 8.2)

5. ✅ Migrate existing experts (Mermaid, PlantUML, SQL) to templates
6. ✅ Deprecate standalone ExpertAgent class
7. ✅ Update documentation with new patterns
8. ✅ Test: Team with mix of experts and non-experts

### Long-term (Phase 8.3)

9. ✅ Remove ExpertAgent class (Version 7.0)
10. ✅ Create more expert templates (DataAnalysis, NLP, etc.)
11. ✅ Add team-level gold standard sharing
12. ✅ Continuous improvement system

---

## Code Examples

### Example 1: Create Expert Agent

**Before (current ExpertAgent):**
```python
from Jotty.core.experts import ExpertAgent, ExpertAgentConfig

config = ExpertAgentConfig(
    name="MermaidExpert",
    domain="mermaid",
    description="Mermaid diagram expert",
    training_gold_standards=gold_standards,
    max_training_iterations=5,
    min_validation_score=1.0
)

expert = ExpertAgent(config)
await expert.train()
result = await expert.run(task="Generate sequence diagram")
```

**After (Phase 8 - SingleAgentOrchestrator):**
```python
from Jotty.core.experts.expert_templates import create_mermaid_expert

# Option 1: Use template
expert = create_mermaid_expert(config=JottyConfig())

# Option 2: Create custom expert
expert = SingleAgentOrchestrator(
    agent=dspy.ChainOfThought(MermaidSignature),
    architect_prompts=["planner.md"],
    auditor_prompts=["validator.md"],
    enable_gold_standard_learning=True,
    gold_standards=gold_standards,
    domain="mermaid"
)

# Same interface
result = await expert.arun(question="Generate sequence diagram")
```

### Example 2: Team with Experts

**After (Phase 8):**
```python
from Jotty.core.orchestration import MultiAgentsOrchestrator
from Jotty.core.experts.expert_templates import (
    create_mermaid_expert,
    create_sql_expert
)

# Create team
orchestrator = MultiAgentsOrchestrator(
    actors=[
        # Expert 1: Mermaid
        ActorConfig(
            name="MermaidExpert",
            agent=create_mermaid_expert(config),
            enable_architect=True,
            enable_auditor=True
        ),

        # Expert 2: SQL
        ActorConfig(
            name="SQLExpert",
            agent=create_sql_expert(config),
            enable_architect=True,
            enable_auditor=True
        ),

        # Non-expert: Data fetcher
        ActorConfig(
            name="DataFetcher",
            agent=SingleAgentOrchestrator(
                agent=dspy.ReAct(FetchSignature),
                architect_prompts=["planner.md"],
                auditor_prompts=["validator.md"]
                # No gold standards
            ),
            enable_architect=True,
            enable_auditor=True
        ),
    ],
    metadata_provider=provider,
    config=config
)

# Run team
result = await orchestrator.run(
    goal="Fetch data, analyze it, generate SQL query, and create diagram"
)
```

### Example 3: Custom Expert

**After (Phase 8):**
```python
# Create custom domain expert
def create_custom_expert(domain: str, gold_standards: List[Dict]):
    return SingleAgentOrchestrator(
        agent=dspy.ChainOfThought(f"{domain} -> output"),
        architect_prompts=[f"prompts/{domain}/planning.md"],
        auditor_prompts=[f"prompts/{domain}/validation.md"],
        enable_gold_standard_learning=True,
        gold_standards=gold_standards,
        domain=domain,
        domain_validator=create_validator_for_domain(domain)
    )

# Use custom expert
nlp_expert = create_custom_expert("nlp", nlp_gold_standards)
result = await nlp_expert.arun(text="Analyze this text")
```

---

## Testing Strategy

### Phase 8.1 Tests (Gold Standard Learning Integration)

```python
def test_single_agent_with_gold_standards():
    """SingleAgentOrchestrator accepts gold standard parameters."""

    gold_standards = [
        {"input": "task 1", "expected_output": "result 1"},
        {"input": "task 2", "expected_output": "result 2"}
    ]

    agent = SingleAgentOrchestrator(
        agent=dspy.ChainOfThought("input -> output"),
        architect_prompts=[],
        auditor_prompts=[],
        enable_gold_standard_learning=True,
        gold_standards=gold_standards,
        domain="test"
    )

    assert agent.enable_gold_standard_learning == True
    assert len(agent.gold_standards) == 2
    assert agent.domain == "test"


def test_optimization_pipeline_integration():
    """OptimizationPipeline integrates with SingleAgentOrchestrator."""

    agent = SingleAgentOrchestrator(
        agent=dspy.ChainOfThought("input -> output"),
        architect_prompts=[],
        auditor_prompts=[],
        enable_gold_standard_learning=True,
        gold_standards=[{"input": "test", "expected_output": "output"}],
        domain="test"
    )

    assert agent.optimization_pipeline is not None


def test_expert_template_creation():
    """Expert templates create valid SingleAgentOrchestrator instances."""

    expert = create_mermaid_expert(config=JottyConfig())

    assert isinstance(expert, SingleAgentOrchestrator)
    assert expert.enable_gold_standard_learning == True
    assert expert.domain == "mermaid"
    assert len(expert.gold_standards) > 0
```

### Phase 8.2 Tests (Expert Migration)

```python
def test_expert_agent_deprecated():
    """Old ExpertAgent class shows deprecation warning."""

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        config = ExpertAgentConfig(name="Test", domain="mermaid")
        expert = ExpertAgent(config)

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "Use SingleAgentOrchestrator" in str(w[0].message)


def test_team_with_experts():
    """MultiAgentsOrchestrator works with expert agents."""

    orchestrator = MultiAgentsOrchestrator(
        actors=[
            ActorConfig(
                name="Expert1",
                agent=create_mermaid_expert(config)
            ),
            ActorConfig(
                name="Expert2",
                agent=create_sql_expert(config)
            ),
        ],
        metadata_provider=provider,
        config=config
    )

    assert len(orchestrator.actors) == 2
```

---

## Summary

**Phase 8 Goal:** Unify expert system with SingleAgentOrchestrator architecture

**Key Changes:**
1. ✅ Gold standard learning → optional feature of SingleAgentOrchestrator
2. ✅ Expert agents → templatized SingleAgentOrchestrator instances
3. ✅ MultiAgentsOrchestrator → coordinates team of SingleAgents (expert or not)
4. ✅ ExpertAgent class → deprecated factory wrapper

**Benefits:**
- Unified execution path (one codebase, not two)
- Feature composition (experts get all SingleAgent features)
- Code reuse (no duplication)
- Clearer abstractions (three-layer architecture)
- Flexibility (easy to create custom experts)

**Backward Compatibility:** 100% (ExpertAgent becomes factory wrapper)

**Timeline:**
- Phase 8.1: Add gold standard learning to SingleAgentOrchestrator (2 weeks)
- Phase 8.2: Migrate experts and deprecate ExpertAgent (2 weeks)
- Phase 8.3: Remove ExpertAgent class in Version 7.0 (6 months)

---

## Next Steps

1. **Review & Approve**: Get feedback on architecture proposal
2. **Prototype**: Create proof-of-concept with Mermaid expert
3. **Implement Phase 8.1**: Extend SingleAgentOrchestrator
4. **Test & Validate**: Comprehensive test suite
5. **Migrate Experts**: Move existing experts to new architecture
6. **Document**: Update all documentation with new patterns

**Ready to implement when approved!** 🚀

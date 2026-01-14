# Unified Proposal: LangGraph + DSPy Agent Graph Templates

## Executive Summary

**Two complementary features:**

1. **LangGraph Integration**: Add LangGraph state machine to Jotty for explicit, directed execution flow
2. **DSPy Agent Graph Templates**: Create templatable directed graphs of DSPy agents with explicit ordering

**Together, these enable:**
- ✅ Explicit, debuggable execution flows (LangGraph)
- ✅ Reusable agent workflow templates (Graph Templates)
- ✅ Brain-inspired learning (Jotty's existing capabilities)
- ✅ Visual workflow representation
- ✅ Type-safe agent orchestration

---

## Part 1: LangGraph Integration

### Overview
Add LangGraph state machine as execution engine while preserving Jotty's intelligence layer.

### Architecture
```
LangGraph State Machine (Directed Execution Flow)
    ↓
Jotty Intelligence Layer (Dependency Resolution, Learning, Memory)
```

### Key Components
- `JottyLangGraphAdapter`: Wraps Conductor with LangGraph
- `JottySwarmState`: State schema for LangGraph execution
- Streaming support via LangGraph's async streaming
- Backward compatibility: opt-in via `use_langgraph=True`

**See**: `LANGGRAPH_MERGE_PROPOSAL.md` for full details

---

## Part 2: DSPy Agent Graph Templates

### Overview
Create **independent directed graphs of DSPy agents** that can be **templatized with explicit ordering**.

### Key Concepts

#### 1. Agent Graph Template
```python
AgentGraphTemplate(
    id="research_workflow",
    name="Research → Analyze → Report",
    nodes=[
        AgentNode(id="research", agent_name="ResearchAgent"),
        AgentNode(id="analyze", agent_name="AnalyzeAgent"),
        AgentNode(id="report", agent_name="ReportAgent"),
    ],
    edges=[
        AgentEdge(from_node="research", to_node="analyze"),
        AgentEdge(from_node="analyze", to_node="report"),
    ],
    entry_nodes=["research"],
    exit_nodes=["report"],
)
```

#### 2. Edge Types
- **SEQUENTIAL**: Agent A → Agent B (B waits for A)
- **PARALLEL**: Agent A → (Agent B, Agent C) (B & C run together)
- **CONDITIONAL**: Agent A → Agent B if condition, else Agent C
- **MERGE**: (Agent A, Agent B) → Agent C (C waits for both)

#### 3. Parameter Mapping
```python
AgentNode(
    id="analyze",
    agent_name="AnalyzeAgent",
    input_mapping={
        "query": "context.goal",
        "data": "research.output",  # From previous agent
    },
    output_key="analysis_results",  # Store in context
)
```

### Implementation

#### Core Classes

**AgentGraphTemplate** (`agent_graph_template.py`)
- Defines graph structure (nodes, edges)
- Validates graph (no cycles, all nodes reachable)
- Computes execution order (topological sort)
- Serialization (to_dict/from_dict)

**AgentGraphExecutor** (`agent_graph_executor.py`)
- Executes template using LangGraph
- Resolves parameters from context
- Handles parallel/conditional/merge edges
- Integrates with Conductor for agent execution

**AgentGraphTemplateRegistry** (`template_registry.py`)
- Load/save templates from disk
- Template validation
- Template discovery and listing

### Example Templates

#### Sequential Workflow
```python
research_template = AgentGraphTemplate(
    id="research_analyze_report",
    nodes=[
        AgentNode(id="research", agent_name="ResearchAgent"),
        AgentNode(id="analyze", agent_name="AnalyzeAgent"),
        AgentNode(id="report", agent_name="ReportAgent"),
    ],
    edges=[
        AgentEdge(from_node="research", to_node="analyze"),
        AgentEdge(from_node="analyze", to_node="report"),
    ],
)
```

#### Parallel Workflow
```python
parallel_template = AgentGraphTemplate(
    id="parallel_analysis",
    nodes=[
        AgentNode(id="research", agent_name="ResearchAgent"),
        AgentNode(id="analyze", agent_name="AnalyzeAgent"),
        AgentNode(id="summarize", agent_name="SummarizeAgent"),
        AgentNode(id="merge", agent_name="MergeAgent"),
    ],
    edges=[
        AgentEdge(from_node="research", to_node="analyze"),
        AgentEdge(from_node="research", to_node="summarize"),
        AgentEdge(from_node="analyze", to_node="merge", edge_type=EdgeType.MERGE),
        AgentEdge(from_node="summarize", to_node="merge", edge_type=EdgeType.MERGE),
    ],
)
```

#### Conditional Workflow
```python
conditional_template = AgentGraphTemplate(
    id="conditional_review",
    nodes=[
        AgentNode(id="generate", agent_name="GenerateAgent"),
        AgentNode(id="review", agent_name="ReviewAgent"),
        AgentNode(id="approve", agent_name="ApproveAgent"),
        AgentNode(id="revise", agent_name="ReviseAgent"),
    ],
    edges=[
        AgentEdge(from_node="generate", to_node="review"),
        AgentEdge(
            from_node="review",
            to_node="approve",
            edge_type=EdgeType.CONDITIONAL,
            condition="result.score >= 0.8"
        ),
        AgentEdge(
            from_node="review",
            to_node="revise",
            edge_type=EdgeType.CONDITIONAL,
            condition="result.score < 0.8"
        ),
    ],
)
```

### Integration Points

#### 1. With Conductor
```python
conductor = Conductor(actors=[...], config=config)

# Execute template
result = await conductor.execute_template(
    template_id="research_workflow",
    goal="Research AI trends",
    context={"year": 2024}
)
```

#### 2. With WorkflowPreset (JustJot.ai)
```typescript
// Convert WorkflowPreset to AgentGraphTemplate
const template = convertWorkflowPresetToAgentGraph(workflowPreset);

// Execute via Python bridge
const result = await jottyBridge.executeTemplate(template.id, goal);
```

#### 3. With LangGraph
```python
# AgentGraphExecutor uses LangGraph internally
executor = AgentGraphExecutor(template, conductor)
result = await executor.execute(goal, context)

# LangGraph provides:
# - Explicit state transitions
# - Streaming support
# - Visual graph representation
```

---

## Unified Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              USER INTERFACE (JustJot.ai)                    │
│  • Workflow Builder (Visual)                                │
│  • Template Gallery                                         │
│  • Execution Monitor                                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           AGENT GRAPH TEMPLATE SYSTEM                       │
│  • Template Definition (Nodes/Edges)                        │
│  • Template Registry (Load/Save)                            │
│  • Template Validation                                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              LANGGRAPH STATE MACHINE                        │
│  • Directed Graph Execution                                 │
│  • Explicit State Transitions                               │
│  • Streaming Support                                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              JOTTY INTELLIGENCE LAYER                       │
│  • Dynamic Dependency Resolution                            │
│  • Brain-Inspired Memory                                    │
│  • Q-Learning & Credit Assignment                           │
│  • Architect/Auditor Validation                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              DSPY AGENT EXECUTION                           │
│  • Agent Execution                                          │
│  • Parameter Resolution                                     │
│  • Result Aggregation                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (Weeks 1-2)
- [ ] Implement `AgentGraphTemplate` class
- [ ] Implement graph validation (cycles, reachability)
- [ ] Implement topological sort for execution order
- [ ] Create `AgentGraphTemplateRegistry`

### Phase 2: LangGraph Integration (Weeks 3-4)
- [ ] Implement `AgentGraphExecutor` using LangGraph
- [ ] Add parameter resolution from context
- [ ] Support all edge types (sequential, parallel, conditional, merge)
- [ ] Integrate with Conductor

### Phase 3: Template System (Weeks 5-6)
- [ ] Create example templates
- [ ] Add template persistence (JSON files)
- [ ] Add template validation UI
- [ ] Document template format

### Phase 4: Integration (Weeks 7-8)
- [ ] Integrate with Conductor API
- [ ] Create TypeScript bridge for JustJot.ai
- [ ] Convert WorkflowPreset to AgentGraphTemplate
- [ ] Add streaming support

### Phase 5: UI & Polish (Weeks 9-10)
- [ ] Visual template builder
- [ ] Template gallery
- [ ] Execution monitoring
- [ ] Documentation

---

## Benefits

### 1. Reusability
- Define agent workflows once, use many times
- Share templates across teams
- Version control templates

### 2. Explicit Ordering
- Clear execution sequence
- No ambiguity about agent dependencies
- Predictable execution flow

### 3. Visual Representation
- Can be visualized as directed graph
- Easy to understand workflow
- Debug-friendly

### 4. Type Safety
- Strong typing for nodes/edges
- Validation at template creation time
- Runtime safety checks

### 5. Flexibility
- Supports sequential, parallel, conditional, merge
- Parameter mapping from context
- Conditional execution paths

### 6. Integration
- Works with existing Jotty components
- Compatible with WorkflowPreset
- Uses LangGraph for execution

---

## Example Usage

### Creating a Template
```python
from Jotty.core.orchestration.agent_graph_template import (
    AgentGraphTemplate, AgentNode, AgentEdge, EdgeType
)

# Define template
template = AgentGraphTemplate(
    id="code_review_workflow",
    name="Code Review Workflow",
    description="Generate code → Review → Fix → Test",
    nodes=[
        AgentNode(
            id="generate",
            agent_name="CodeGeneratorAgent",
            input_mapping={"requirement": "context.requirement"},
            output_key="generated_code",
        ),
        AgentNode(
            id="review",
            agent_name="CodeReviewAgent",
            input_mapping={"code": "generate.output"},
            output_key="review_feedback",
        ),
        AgentNode(
            id="fix",
            agent_name="CodeFixAgent",
            input_mapping={
                "code": "generate.output",
                "feedback": "review.output",
            },
            output_key="fixed_code",
        ),
        AgentNode(
            id="test",
            agent_name="TestAgent",
            input_mapping={"code": "fix.output"},
        ),
    ],
    edges=[
        AgentEdge(from_node="generate", to_node="review"),
        AgentEdge(from_node="review", to_node="fix"),
        AgentEdge(from_node="fix", to_node="test"),
    ],
    entry_nodes=["generate"],
    exit_nodes=["test"],
    category="code",
)

# Register template
registry = AgentGraphTemplateRegistry()
registry.register(template)
```

### Executing a Template
```python
from Jotty import Conductor

# Initialize conductor
conductor = Conductor(actors=[...], config=config)

# Execute template
result = await conductor.execute_template(
    template_id="code_review_workflow",
    goal="Create a REST API endpoint",
    context={
        "requirement": "Create GET /users endpoint",
    }
)

print(f"Success: {result['success']}")
print(f"Final output: {result['results']['test'].output}")
```

### Using with LangGraph Streaming
```python
# Execute with streaming
async for event in conductor.execute_template_stream(
    template_id="code_review_workflow",
    goal="Create a REST API endpoint",
):
    print(f"Event: {event.type} - {event.data}")
```

---

## Migration Path

### Option 1: Gradual Migration
1. **Week 1-2**: Implement AgentGraphTemplate system
2. **Week 3-4**: Add LangGraph executor
3. **Week 5-6**: Create example templates
4. **Week 7-8**: Integrate with Conductor
5. **Week 9-10**: UI & documentation

### Option 2: Parallel Development
- Develop AgentGraphTemplate independently
- Integrate with LangGraph later
- Lower risk, faster initial delivery

---

## Success Criteria

- ✅ AgentGraphTemplate class implemented
- ✅ Graph validation working
- ✅ LangGraph executor functional
- ✅ Template registry working
- ✅ Example templates created
- ✅ Integration with Conductor
- ✅ TypeScript bridge functional
- ✅ Documentation complete

---

## Next Steps

1. **Review & Approval**: Get team approval
2. **Create GitHub Issues**: Track implementation
3. **Set up Branch**: `feature/agent-graph-templates`
4. **Implement Phase 1**: Core infrastructure
5. **Test & Iterate**: Continuous testing

---

**Proposed by**: AI Assistant  
**Date**: 2026-01-11  
**Status**: Awaiting Approval

**Related Documents**:
- `LANGGRAPH_MERGE_PROPOSAL.md` - LangGraph integration details
- `DSPY_AGENT_GRAPH_TEMPLATE_PROPOSAL.md` - Template system details

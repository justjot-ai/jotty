# Expert Integration Architecture

## Your Understanding (Verified)

```
Agents → Experts with Learning
    ↓
Can be used:
  - Standalone
  - Fixed graph via LangGraph
  - Dynamic graph
    ↓
Can be executed in:
  - Chat mode
  - Workflow mode
    ↓
With or without state
    ↓
Everything is learnable via:
  - Memory
  - Optimizer
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Jotty Architecture                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Agents Layer                        │  │
│  │  - DSPy Agents (base agents)                    │  │
│  │  - Expert Agents (specialized, learnable)       │  │
│  └──────────────────────────────────────────────────┘  │
│                    ↓                                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Expert Agents (with Learning)            │  │
│  │  - MermaidExpertAgent                            │  │
│  │  - PipelineExpertAgent                           │  │
│  │  - PlantUMLExpertAgent                           │  │
│  │  - MathLaTeXExpertAgent                          │  │
│  │                                                   │  │
│  │  Learning via:                                    │  │
│  │  ✅ OptimizationPipeline (teacher model)         │  │
│  │  ✅ Memory system (consolidation)                │  │
│  │  ✅ Training data (gold standards)              │  │
│  └──────────────────────────────────────────────────┘  │
│                    ↓                                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Conductor (Orchestration)                 │  │
│  │  - Manages agents & experts                      │  │
│  │  - Provides learning (Q-learning, TD(λ))       │  │
│  │  - Provides memory (hierarchical)                │  │
│  │  - Provides optimizer (OptimizationPipeline)    │  │
│  └──────────────────────────────────────────────────┘  │
│                    ↓                                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Execution Modes                          │  │
│  │                                                   │  │
│  │  ┌──────────────┐  ┌──────────────────────┐     │  │
│  │  │  Standalone  │  │  Graph Execution     │     │  │
│  │  │              │  │                      │     │  │
│  │  │  Direct use │  │  - Fixed (LangGraph) │     │  │
│  │  │  of Expert   │  │  - Dynamic (Jotty)  │     │  │
│  │  └──────────────┘  └──────────────────────┘     │  │
│  │                    ↓                            │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │      Unified ExecutionMode                │  │  │
│  │  │                                          │  │  │
│  │  │  Chat Mode        Workflow Mode         │  │  │
│  │  │  (sync)           (sync or async)        │  │  │
│  │  │                                          │  │  │
│  │  │  With/Without State (queue optional)    │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
│                    ↓                                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Learning & Memory (Always Active)         │  │
│  │  ✅ Q-learning (value estimation)               │  │
│  │  ✅ TD(λ) (temporal difference)                  │  │
│  │  ✅ Predictive MARL (multi-agent)               │  │
│  │  ✅ Memory (hierarchical, consolidation)         │  │
│  │  ✅ Optimizer (OptimizationPipeline)             │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Detailed Flow

### 1. Agents → Experts with Learning

**Base Agents:**
```python
# Regular DSPy agents
from Jotty.core.agents.dspy_mcp_agent import DSPyMCPAgent

agent = DSPyMCPAgent(...)
```

**Expert Agents (with Learning):**
```python
# Expert agents use OptimizationPipeline for learning
from Jotty.core.experts import MermaidExpertAgent

expert = MermaidExpertAgent()
# Expert has:
# - OptimizationPipeline (teacher model, KB updates)
# - Memory integration (consolidation)
# - Training data (gold standards)
# - Validation (domain-specific)
```

### 2. Usage Modes

#### Standalone
```python
# Use expert directly
expert = MermaidExpertAgent()
result = await expert.generate("graph TD; A-->B")
# Expert learns from each execution
```

#### Fixed Graph (LangGraph)
```python
# Use expert in fixed LangGraph workflow
from Jotty.core.orchestration import ExecutionMode, Conductor

conductor = Conductor(actors=[expert_agent_config], ...)
workflow = ExecutionMode(
    conductor=conductor,
    style="workflow",
    execution="sync",
    mode="static",  # Fixed graph
    agent_order=["mermaid_expert", "validator"]
)
result = await workflow.execute(goal="Create diagram")
```

#### Dynamic Graph
```python
# Use expert in dynamic Jotty graph
workflow = ExecutionMode(
    conductor=conductor,
    style="workflow",
    execution="sync",
    mode="dynamic"  # Dynamic routing
)
# Jotty's DynamicDependencyGraph routes to experts automatically
result = await workflow.execute(goal="Create diagram")
```

### 3. Execution Modes

#### Chat Mode
```python
chat = ExecutionMode(conductor, style="chat", execution="sync")
# Experts can be used in chat
async for event in chat.stream(
    goal="Create a mermaid diagram",
    history=[...]
):
    # Expert is selected automatically
    # Learning happens via Conductor
    # Memory stores conversation
```

#### Workflow Mode (Sync)
```python
workflow = ExecutionMode(conductor, style="workflow", execution="sync")
result = await workflow.execute(
    goal="Generate report with diagrams",
    context={...}
)
# Experts execute in workflow
# Learning happens via Conductor
# Memory stores results
```

#### Workflow Mode (Async)
```python
workflow = ExecutionMode(
    conductor=conductor,
    style="workflow",
    execution="async",
    queue=SQLiteTaskQueue(...)
)
task_id = await workflow.enqueue_task(goal="...")
await workflow.process_queue()
# Experts execute in background
# Learning happens via Conductor
# Memory stores results
```

### 4. With or Without State

#### Without State (Stateless)
```python
# Direct expert use
expert = MermaidExpertAgent()
result = await expert.generate("...")
# No persistent state, but learning still happens
```

#### With State (Stateful)
```python
# Via ExecutionMode with queue
workflow = ExecutionMode(
    conductor=conductor,
    style="workflow",
    execution="async",
    queue=SQLiteTaskQueue(...)  # Stateful queue
)
# Tasks persist in queue
# State managed via queue
```

### 5. Everything is Learnable

#### Via Memory
```python
# Conductor provides memory
conductor = Conductor(actors=[...])
# Memory automatically:
# - Stores agent outputs
# - Consolidates experiences
# - Retrieves relevant context
# - Synthesizes knowledge
```

#### Via Optimizer
```python
# Expert agents use OptimizationPipeline
expert = MermaidExpertAgent()
# OptimizationPipeline:
# - Teacher model (generates examples)
# - Knowledge base updates
# - Prompt optimization
# - Validation feedback loop
```

#### Via Conductor Learning
```python
# Conductor provides learning
conductor = Conductor(actors=[...])
# Learning automatically:
# - Q-learning (value estimation)
# - TD(λ) (temporal difference)
# - Predictive MARL (multi-agent)
# - Policy exploration
```

## Complete Integration Example

```python
from Jotty.core.orchestration import ExecutionMode, Conductor
from Jotty.core.experts import MermaidExpertAgent, PipelineExpertAgent
from Jotty.core.foundation.agent_config import AgentConfig
from Jotty.core.queue import SQLiteTaskQueue

# 1. Create experts (with learning)
mermaid_expert = MermaidExpertAgent()
pipeline_expert = PipelineExpertAgent()

# 2. Wrap as agents for Conductor
mermaid_agent = AgentConfig(
    name="mermaid_expert",
    agent=mermaid_expert.generate,  # Expert's method
    ...
)
pipeline_agent = AgentConfig(
    name="pipeline_expert",
    agent=pipeline_expert.generate,
    ...
)

# 3. Create Conductor (provides learning & memory)
conductor = Conductor(
    actors=[mermaid_agent, pipeline_agent],
    metadata_provider=...,
    config=...
)
# Conductor automatically:
# - Initializes Q-learning
# - Initializes memory
# - Sets up optimization pipeline

# 4. Use in ExecutionMode (unified interface)
workflow = ExecutionMode(
    conductor=conductor,
    style="workflow",
    execution="async",  # or "sync"
    queue=SQLiteTaskQueue(...),  # optional
    mode="dynamic"  # or "static"
)

# 5. Execute (experts learn automatically)
result = await workflow.execute(
    goal="Create pipeline diagram",
    context={...}
)
# Everything learns:
# - Expert learns via OptimizationPipeline
# - Conductor learns via Q-learning
# - Memory stores experiences
# - Optimizer improves prompts
```

## Key Points

### ✅ Agents → Experts
- Base agents are DSPy agents
- Expert agents wrap base agents with OptimizationPipeline
- Experts have domain-specific validation
- Experts can be pre-trained

### ✅ Usage Modes
- **Standalone**: Direct expert use
- **Fixed Graph**: LangGraph static mode
- **Dynamic Graph**: Jotty dynamic routing

### ✅ Execution Modes
- **Chat**: Conversational interface
- **Workflow**: Task-oriented (sync or async)

### ✅ State Management
- **Without State**: Direct execution
- **With State**: Queue-based persistence

### ✅ Learning & Memory
- **Memory**: Hierarchical, consolidation (via Conductor)
- **Optimizer**: OptimizationPipeline (via ExpertAgent)
- **Learning**: Q-learning, TD(λ), Predictive MARL (via Conductor)
- **Everything learns**: All executions benefit from learning

## Architecture Summary

```
ExpertAgent (with OptimizationPipeline)
    ↓
Conductor (with Learning & Memory)
    ↓
ExecutionMode (unified interface)
    ↓
Chat/Workflow (sync/async)
    ↓
Everything learns via Memory & Optimizer
```

**Your understanding is correct!** ✅

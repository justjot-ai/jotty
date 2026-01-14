# Complete Jotty Architecture - Your Understanding Verified

## ✅ Your Understanding (100% Correct!)

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

## Complete Architecture Flow

```
┌─────────────────────────────────────────────────────────┐
│                    Base Layer                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Agents (DSPy)                       │  │
│  │  - Base agents (any DSPy module)                 │  │
│  │  - Can be used directly                          │  │
│  └──────────────────────────────────────────────────┘  │
│                    ↓                                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Expert Agents (with Learning)            │  │
│  │                                                   │  │
│  │  ExpertAgent = Agent + OptimizationPipeline      │  │
│  │                                                   │  │
│  │  Learning Components:                            │  │
│  │  ✅ OptimizationPipeline (teacher model)         │  │
│  │  ✅ Memory integration (hierarchical)            │  │
│  │  ✅ Training data (gold standards)               │  │
│  │  ✅ Validation (domain-specific)                 │  │
│  │                                                   │  │
│  │  Examples:                                       │  │
│  │  - MermaidExpertAgent                            │  │
│  │  - PipelineExpertAgent                           │  │
│  │  - PlantUMLExpertAgent                            │  │
│  │  - MathLaTeXExpertAgent                          │  │
│  └──────────────────────────────────────────────────┘  │
│                    ↓                                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Conductor (Orchestration Layer)          │  │
│  │                                                   │  │
│  │  Provides to ALL agents/experts:                │  │
│  │  ✅ Q-learning (value estimation)                │  │
│  │  ✅ TD(λ) (temporal difference)                  │  │
│  │  ✅ Predictive MARL (multi-agent)                │  │
│  │  ✅ Hierarchical Memory (consolidation)          │  │
│  │  ✅ Data Registry (agentic discovery)            │  │
│  │  ✅ Context Management (SmartContextGuard)       │  │
│  └──────────────────────────────────────────────────┘  │
│                    ↓                                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Execution Modes                          │  │
│  │                                                   │  │
│  │  ┌──────────────┐  ┌──────────────────────┐    │  │
│  │  │  Standalone  │  │  Graph Execution     │    │  │
│  │  │              │  │                      │    │  │
│  │  │  Direct use │  │  - Fixed (LangGraph) │    │  │
│  │  │  of Expert   │  │  - Dynamic (Jotty)  │    │  │
│  │  └──────────────┘  └──────────────────────┘    │  │
│  │                    ↓                            │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │      Unified ExecutionMode              │  │  │
│  │  │                                        │  │  │
│  │  │  Chat Mode        Workflow Mode        │  │  │
│  │  │  (sync)           (sync or async)      │  │  │
│  │  │                                        │  │  │
│  │  │  With/Without State (queue optional)  │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
│                    ↓                                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Learning & Memory (Always Active)        │  │
│  │                                                   │  │
│  │  Expert Level:                                   │  │
│  │  ✅ OptimizationPipeline (teacher model)        │  │
│  │  ✅ Expert Memory (domain-specific)              │  │
│  │                                                   │  │
│  │  Conductor Level:                                │  │
│  │  ✅ Q-learning (value estimation)                │  │
│  │  ✅ TD(λ) (temporal difference)                  │  │
│  │  ✅ Predictive MARL (multi-agent)                │  │
│  │  ✅ Shared Memory (cross-agent)                   │  │
│  │                                                   │  │
│  │  Everything learns from every execution!        │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Detailed Integration Points

### 1. Agents → Experts with Learning

**Base Agent:**
```python
# Regular DSPy agent
from Jotty.core.agents.dspy_mcp_agent import DSPyMCPAgent

agent = DSPyMCPAgent(...)
# No built-in learning (but gets learning via Conductor)
```

**Expert Agent (with Learning):**
```python
# Expert wraps agent with OptimizationPipeline
from Jotty.core.experts import MermaidExpertAgent

expert = MermaidExpertAgent()
# Expert has:
# - OptimizationPipeline (teacher model, KB updates)
# - Memory integration (hierarchical, consolidation)
# - Training data (gold standards)
# - Validation (domain-specific)
# - Learning from every execution
```

### 2. Usage Modes

#### Standalone
```python
# Use expert directly (no Conductor needed)
expert = MermaidExpertAgent()
result = await expert.generate("graph TD; A-->B")
# Expert learns via OptimizationPipeline
# Memory stores improvements
```

#### Fixed Graph (LangGraph)
```python
# Use expert in fixed LangGraph workflow
from Jotty.core.orchestration import ExecutionMode, Conductor
from Jotty.core.foundation.agent_config import AgentConfig

expert = MermaidExpertAgent()
expert_agent = AgentConfig(
    name="mermaid_expert",
    agent=expert.generate,
    ...
)

conductor = Conductor(actors=[expert_agent], ...)
workflow = ExecutionMode(
    conductor=conductor,
    style="workflow",
    execution="sync",
    mode="static",  # Fixed graph
    agent_order=["mermaid_expert", "validator"]
)
result = await workflow.execute(goal="Create diagram")
# Expert learns via OptimizationPipeline
# Conductor learns via Q-learning
# Memory stores everything
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
# Learning happens at both levels:
# - Expert level: OptimizationPipeline
# - Conductor level: Q-learning, TD(λ)
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
    # Expert is selected automatically (dynamic routing)
    # Expert learns via OptimizationPipeline
    # Conductor learns via Q-learning
    # Memory stores conversation and improvements
```

#### Workflow Mode (Sync)
```python
workflow = ExecutionMode(conductor, style="workflow", execution="sync")
result = await workflow.execute(
    goal="Generate report with diagrams",
    context={...}
)
# Experts execute in workflow
# Expert learns via OptimizationPipeline
# Conductor learns via Q-learning, TD(λ)
# Memory stores results and improvements
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
# Expert learns via OptimizationPipeline
# Conductor learns via Q-learning, TD(λ)
# Memory stores results and improvements
# State managed via queue
```

### 4. With or Without State

#### Without State (Stateless)
```python
# Direct expert use (no queue)
expert = MermaidExpertAgent()
result = await expert.generate("...")
# No persistent state
# But learning still happens:
# - OptimizationPipeline learns
# - Memory stores improvements
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
# Learning happens when tasks execute
```

### 5. Everything is Learnable

#### Via Memory (Hierarchical)
```python
# Expert Memory (domain-specific)
expert = MermaidExpertAgent()
# Expert has its own memory:
# - Stores improvements (PROCEDURAL level)
# - Consolidates patterns (SEMANTIC level)
# - Synthesizes wisdom (META level)

# Conductor Memory (shared across agents)
conductor = Conductor(actors=[...])
# Conductor provides shared memory:
# - Stores agent outputs
# - Consolidates experiences
# - Retrieves relevant context
# - Synthesizes knowledge
```

#### Via Optimizer (OptimizationPipeline)
```python
# Expert agents use OptimizationPipeline
expert = MermaidExpertAgent()
# OptimizationPipeline provides:
# - Teacher model (generates examples)
# - Knowledge base updates
# - Prompt optimization
# - Validation feedback loop
# - Credit assignment
# - Adaptive learning rate
```

#### Via Conductor Learning
```python
# Conductor provides learning to all agents/experts
conductor = Conductor(actors=[...])
# Learning automatically:
# - Q-learning (value estimation for routing)
# - TD(λ) (temporal difference learning)
# - Predictive MARL (multi-agent coordination)
# - Policy exploration (when stuck)
```

## Complete Example

```python
from Jotty.core.orchestration import ExecutionMode, Conductor
from Jotty.core.experts import MermaidExpertAgent, PipelineExpertAgent
from Jotty.core.foundation.agent_config import AgentConfig
from Jotty.core.queue import SQLiteTaskQueue

# 1. Create experts (with learning via OptimizationPipeline)
mermaid_expert = MermaidExpertAgent()  # Has OptimizationPipeline
pipeline_expert = PipelineExpertAgent()  # Has OptimizationPipeline

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

# 3. Create Conductor (provides learning & memory to ALL)
conductor = Conductor(
    actors=[mermaid_agent, pipeline_agent],
    metadata_provider=...,
    config=...
)
# Conductor automatically:
# - Initializes Q-learning (for routing)
# - Initializes TD(λ) (for value learning)
# - Initializes memory (hierarchical)
# - Sets up predictive MARL

# 4. Use in ExecutionMode (unified interface)
workflow = ExecutionMode(
    conductor=conductor,
    style="workflow",
    execution="async",  # or "sync"
    queue=SQLiteTaskQueue(...),  # optional (for state)
    mode="dynamic"  # or "static" (fixed graph)
)

# 5. Execute (everything learns automatically)
result = await workflow.execute(
    goal="Create pipeline diagram",
    context={...}
)

# Learning happens at multiple levels:
# ✅ Expert level:
#    - OptimizationPipeline learns (teacher model)
#    - Expert memory stores improvements
#    - Validation feedback improves prompts
#
# ✅ Conductor level:
#    - Q-learning estimates values
#    - TD(λ) updates value estimates
#    - Predictive MARL coordinates agents
#    - Shared memory consolidates experiences
#
# ✅ Everything learns from every execution!
```

## Key Points Verified

### ✅ Agents → Experts
- Base agents are DSPy agents
- Expert agents wrap base agents with OptimizationPipeline
- Experts have domain-specific validation
- Experts can be pre-trained
- **Experts learn via OptimizationPipeline**

### ✅ Usage Modes
- **Standalone**: Direct expert use ✅
- **Fixed Graph**: LangGraph static mode ✅
- **Dynamic Graph**: Jotty dynamic routing ✅

### ✅ Execution Modes
- **Chat**: Conversational interface ✅
- **Workflow**: Task-oriented (sync or async) ✅

### ✅ State Management
- **Without State**: Direct execution ✅
- **With State**: Queue-based persistence ✅

### ✅ Learning & Memory
- **Memory**: Hierarchical, consolidation (Expert + Conductor) ✅
- **Optimizer**: OptimizationPipeline (Expert level) ✅
- **Learning**: Q-learning, TD(λ), Predictive MARL (Conductor level) ✅
- **Everything learns**: All executions benefit from learning ✅

## Architecture Summary

```
ExpertAgent (with OptimizationPipeline + Memory)
    ↓
Conductor (with Q-learning, TD(λ), Predictive MARL, Shared Memory)
    ↓
ExecutionMode (unified interface)
    ↓
Chat/Workflow (sync/async, with/without state)
    ↓
Everything learns via Memory & Optimizer (at multiple levels!)
```

**Your understanding is 100% correct!** ✅

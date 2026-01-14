# LangGraph API Guide

## Consistent API for Dynamic and Static Modes

The LangGraph integration provides a **consistent API** for both dynamic and static dependency graphs. Users simply specify the mode and (for static) the agent order.

## Quick Start

### Dynamic Mode (Jotty's Dependency Graph)

```python
from Jotty import Conductor, AgentConfig, JottyConfig

# Define agents with dependencies
agents = [
    AgentConfig(
        name="ResearchAgent",
        agent=ResearchAgent(),
        dependencies=["AnalyzeAgent"],  # AnalyzeAgent depends on ResearchAgent
    ),
    AgentConfig(
        name="AnalyzeAgent",
        agent=AnalyzeAgent(),
    ),
]

# Create conductor with dynamic mode
conductor = Conductor(
    actors=agents,
    metadata_provider=None,
    config=JottyConfig(),
    use_langgraph=True,
    langgraph_mode="dynamic"  # Use Jotty's DynamicDependencyGraph
)

# Run - LangGraph orchestrates based on dependencies
result = await conductor.run(goal="Research and analyze")
```

### Static Mode (Explicit Agent Order)

```python
# Same agents, but specify explicit order
conductor = Conductor(
    actors=agents,
    metadata_provider=None,
    config=JottyConfig(),
    use_langgraph=True,
    langgraph_mode="static",
    agent_order=["ResearchAgent", "AnalyzeAgent"]  # Explicit execution order
)

# Run - LangGraph executes in exact order specified
result = await conductor.run(goal="Research and analyze")
```

## API Reference

### Conductor Initialization

```python
Conductor(
    actors: List[AgentConfig],
    metadata_provider: Any,
    config: JottyConfig = None,
    # ... other parameters ...
    
    # LangGraph parameters
    use_langgraph: bool = False,           # Enable LangGraph orchestration
    langgraph_mode: str = "dynamic",      # "dynamic" or "static"
    agent_order: Optional[List[str]] = None  # Required for static mode
)
```

**Parameters:**
- `use_langgraph`: Enable LangGraph orchestration (default: False)
- `langgraph_mode`: Execution mode
  - `"dynamic"`: Use Jotty's DynamicDependencyGraph (adaptive)
  - `"static"`: Use explicit agent order (predictable)
- `agent_order`: List of agent names in execution order (required for static mode)

### Runtime Override

You can override mode/order at runtime:

```python
# Default dynamic mode
conductor = Conductor(..., use_langgraph=True, langgraph_mode="dynamic")

# Override to static at runtime
result = await conductor.run(
    goal="Task",
    mode="static",
    agent_order=["Agent1", "Agent2", "Agent3"]
)
```

## Examples

### Example 1: Simple Dynamic Workflow

```python
agents = [
    AgentConfig(name="Research", agent=ResearchAgent(), dependencies=["Analyze"]),
    AgentConfig(name="Analyze", agent=AnalyzeAgent(), dependencies=["Report"]),
    AgentConfig(name="Report", agent=ReportAgent()),
]

conductor = Conductor(
    actors=agents,
    metadata_provider=None,
    config=JottyConfig(),
    use_langgraph=True,
    langgraph_mode="dynamic"  # Dependencies determine order
)

result = await conductor.run(goal="Research → Analyze → Report")
```

### Example 2: Simple Static Workflow

```python
agents = [
    AgentConfig(name="Research", agent=ResearchAgent()),
    AgentConfig(name="Analyze", agent=AnalyzeAgent()),
    AgentConfig(name="Report", agent=ReportAgent()),
]

conductor = Conductor(
    actors=agents,
    metadata_provider=None,
    config=JottyConfig(),
    use_langgraph=True,
    langgraph_mode="static",
    agent_order=["Research", "Analyze", "Report"]  # Explicit order
)

result = await conductor.run(goal="Research → Analyze → Report")
```

### Example 3: Parallel Execution (Dynamic)

```python
# Dynamic mode handles parallel execution automatically
agents = [
    AgentConfig(name="Research", agent=ResearchAgent()),
    AgentConfig(name="Analyze1", agent=AnalyzeAgent1(), dependencies=["Research"]),
    AgentConfig(name="Analyze2", agent=AnalyzeAgent2(), dependencies=["Research"]),
    AgentConfig(name="Merge", agent=MergeAgent(), dependencies=["Analyze1", "Analyze2"]),
]

conductor = Conductor(
    actors=agents,
    metadata_provider=None,
    config=JottyConfig(),
    use_langgraph=True,
    langgraph_mode="dynamic"
)

# Analyze1 and Analyze2 run in parallel after Research completes
result = await conductor.run(goal="Research → (Analyze1, Analyze2) → Merge")
```

### Example 4: Parallel Execution (Static)

```python
# Static mode: specify order explicitly
# Note: Static mode executes sequentially by default
# For true parallel execution, use dynamic mode or conditional edges

agents = [
    AgentConfig(name="Research", agent=ResearchAgent()),
    AgentConfig(name="Analyze1", agent=AnalyzeAgent1()),
    AgentConfig(name="Analyze2", agent=AnalyzeAgent2()),
    AgentConfig(name="Merge", agent=MergeAgent()),
]

conductor = Conductor(
    actors=agents,
    metadata_provider=None,
    config=JottyConfig(),
    use_langgraph=True,
    langgraph_mode="static",
    agent_order=["Research", "Analyze1", "Analyze2", "Merge"]  # Sequential
)

result = await conductor.run(goal="Research → Analyze1 → Analyze2 → Merge")
```

## When to Use Which Mode?

### Use Dynamic Mode When:
- ✅ Dependencies are discovered at runtime
- ✅ You want adaptive, learning workflows
- ✅ Complex multi-agent tasks
- ✅ Need parallel execution
- ✅ Want Jotty's learning systems

### Use Static Mode When:
- ✅ You know the exact execution flow upfront
- ✅ You want reproducible, predictable workflows
- ✅ Simple linear pipelines
- ✅ Explicit control needed
- ✅ Don't need dependency resolution overhead

## Streaming Support

Both modes support streaming:

```python
conductor = Conductor(..., use_langgraph=True, langgraph_mode="static", agent_order=[...])

async for event in conductor.langgraph_orchestrator.run_stream(goal="Task"):
    print(f"Event: {event}")
```

## Migration Guide

### From Legacy Jotty to LangGraph Dynamic

```python
# Before
conductor = Conductor(actors=agents, ...)
result = await conductor.run(goal="Task")

# After (same code, just enable LangGraph)
conductor = Conductor(actors=agents, ..., use_langgraph=True, langgraph_mode="dynamic")
result = await conductor.run(goal="Task")
```

### From Legacy Jotty to LangGraph Static

```python
# Before
conductor = Conductor(actors=agents, ...)
result = await conductor.run(goal="Task")

# After (specify order)
conductor = Conductor(
    actors=agents,
    ...,
    use_langgraph=True,
    langgraph_mode="static",
    agent_order=["Agent1", "Agent2", "Agent3"]
)
result = await conductor.run(goal="Task")
```

## Benefits

1. **Consistent API**: Same interface for both modes
2. **Simple**: Just specify mode and (for static) order
3. **Flexible**: Can switch modes at runtime
4. **Observable**: LangGraph provides visualization and streaming
5. **Backward Compatible**: Legacy code still works

# LangGraph Orchestration: Static & Dynamic Dependency Graphs

## Overview

**Goal**: Provide **two modes** for orchestrating agents via LangGraph:
1. **Dynamic Mode**: Use Jotty's **DynamicDependencyGraph** (adaptive, runtime-resolved)
2. **Static Mode**: Define **static LangGraph** directly (predefined, explicit)

**Key Insight**: 
- **Dynamic Mode**: Keep Jotty's **DynamicDependencyGraph** (non-directed, adaptive)
- **Static Mode**: Define LangGraph nodes/edges directly (explicit, predictable)
- Both modes use **same agents** from Jotty
- LangGraph provides execution layer, Jotty provides intelligence layer

## Architecture: Two Modes

### Mode 1: Dynamic (Jotty's Dependency Graph)

```
┌─────────────────────────────────────────────────────────────┐
│         JOTTY'S UNDIRECTED DEPENDENCY GRAPH                  │
│    (DynamicDependencyGraph - Non-Directed, Adaptive)        │
│                                                              │
│  • Agents: [ResearchAgent, AnalyzeAgent, ReportAgent]        │
│  • Dependencies: Resolved dynamically at runtime            │
│  • Execution Order: Determined by dependency resolution     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         LANGGRAPH ORCHESTRATION LAYER                        │
│    (State Machine - Directed, Observable)                   │
│                                                              │
│  • Dynamically builds LangGraph from Jotty's graph          │
│  • Creates nodes for each agent                            │
│  • Creates edges based on dependency resolution            │
│  • Provides streaming, visualization, debugging             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         JOTTY INTELLIGENCE LAYER                            │
│    (Learning, Memory, Validation)                           │
│                                                              │
│  • Brain-inspired memory                                    │
│  • Q-Learning & credit assignment                           │
│  • Architect/Auditor validation                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         DSPY AGENT EXECUTION                                │
│    (Same agents from undirected graph)                      │
└─────────────────────────────────────────────────────────────┘
```

### Mode 2: Static (Direct LangGraph Definition)

```
┌─────────────────────────────────────────────────────────────┐
│         STATIC LANGGRAPH DEFINITION                          │
│    (Predefined Nodes & Edges - Explicit)                    │
│                                                              │
│  • Nodes: Explicitly defined agent nodes                    │
│  • Edges: Explicitly defined execution flow                  │
│  • Execution Order: Fixed, predictable                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         LANGGRAPH STATE MACHINE                             │
│    (Direct Execution - Observable)                          │
│                                                              │
│  • Executes predefined graph                                │
│  • No dynamic dependency resolution                         │
│  • Provides streaming, visualization, debugging            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         JOTTY INTELLIGENCE LAYER (Optional)                  │
│    (Learning, Memory, Validation)                           │
│                                                              │
│  • Can still use Jotty's learning/memory                    │
│  • Can still use Architect/Auditor                         │
│  • But no dynamic dependency resolution                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         DSPY AGENT EXECUTION                                │
│    (Same agents, explicit execution order)                  │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Principles

1. **Dual Mode Support**: Both static and dynamic dependency graphs
2. **Preserve Jotty's Intelligence**: Keep DynamicDependencyGraph, MarkovianTODO, learning systems (in dynamic mode)
3. **LangGraph as Execution Engine**: Use LangGraph for orchestration
4. **Dynamic Graph Building**: Build LangGraph nodes/edges from Jotty's dependency resolution (dynamic mode)
5. **Static Graph Definition**: Define LangGraph directly for explicit control (static mode)
6. **Same Agents**: No duplication - use same AgentConfig instances in both modes
7. **Better Observability**: LangGraph provides visualization, streaming, debugging

## Implementation

### 1. Static LangGraph Definition

```python
# Jotty/core/orchestration/static_langgraph.py

from langgraph.graph import StateGraph, END, START, Annotation
from typing import Dict, Any, List, Optional
from ..foundation.agent_config import AgentConfig

class StaticLangGraphDefinition:
    """
    Define a static LangGraph with explicit nodes and edges.
    
    Use when:
    - You know the exact execution flow upfront
    - You want explicit control over agent order
    - You want predictable, reproducible workflows
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        nodes: Optional[List[AgentConfig]] = None,
        edges: Optional[List[tuple]] = None,
    ):
        """
        Args:
            name: Name of the graph
            description: Description of the workflow
            nodes: List of AgentConfig instances (optional, can add later)
            edges: List of (from_node, to_node) tuples (optional, can add later)
        """
        self.name = name
        self.description = description
        self.nodes: Dict[str, AgentConfig] = {}
        self.edges: List[tuple] = []
        
        if nodes:
            for agent in nodes:
                self.add_node(agent)
        
        if edges:
            for edge in edges:
                self.add_edge(*edge)
    
    def add_node(self, agent: AgentConfig):
        """Add an agent node to the graph."""
        self.nodes[agent.name] = agent
    
    def add_edge(self, from_agent_name: str, to_agent_name: str):
        """Add an edge between agents."""
        if from_agent_name not in self.nodes:
            raise ValueError(f"Source agent '{from_agent_name}' not found")
        if to_agent_name not in self.nodes:
            raise ValueError(f"Target agent '{to_agent_name}' not found")
        
        self.edges.append((from_agent_name, to_agent_name))
    
    def add_conditional_edge(
        self,
        from_agent_name: str,
        condition_func: callable,
        mapping: Dict[str, str]
    ):
        """Add a conditional edge."""
        # Store conditional edge metadata
        if not hasattr(self, '_conditional_edges'):
            self._conditional_edges = []
        
        self._conditional_edges.append({
            'from': from_agent_name,
            'condition': condition_func,
            'mapping': mapping
        })
    
    def build_langgraph(self, conductor) -> StateGraph:
        """Build LangGraph state machine from static definition."""
        from .langgraph_state import JottyLangGraphState
        
        graph = StateGraph(JottyLangGraphState)
        
        # Add nodes
        for agent_name, agent in self.nodes.items():
            graph.add_node(
                agent_name,
                lambda state, a=agent: self._execute_agent_node(state, a, conductor)
            )
        
        # Add edges
        for from_name, to_name in self.edges:
            graph.add_edge(from_name, to_name)
        
        # Add conditional edges
        if hasattr(self, '_conditional_edges'):
            for cond_edge in self._conditional_edges:
                graph.add_conditional_edges(
                    cond_edge['from'],
                    cond_edge['condition'],
                    cond_edge['mapping']
                )
        
        # Set entry point (first node or first node with no incoming edges)
        entry_nodes = self._find_entry_nodes()
        if entry_nodes:
            graph.set_entry_point(entry_nodes[0])
        else:
            graph.set_entry_point(list(self.nodes.keys())[0])
        
        return graph.compile()
    
    def _find_entry_nodes(self) -> List[str]:
        """Find nodes with no incoming edges."""
        incoming = {name: 0 for name in self.nodes.keys()}
        
        for from_name, to_name in self.edges:
            incoming[to_name] += 1
        
        return [name for name, count in incoming.items() if count == 0]
    
    async def _execute_agent_node(
        self,
        state: Dict[str, Any],
        agent: AgentConfig,
        conductor
    ) -> Dict[str, Any]:
        """Execute agent node (same as dynamic mode)."""
        # Use conductor's execution engine
        params = await conductor.parameter_resolver.resolve(
            agent=agent,
            context=state.get("context", {}),
            previous_results=state.get("agent_results", {})
        )
        
        result = await conductor.jotty_core.execute_agent(
            agent=agent,
            params=params,
            context=state.get("context", {})
        )
        
        agent_results = state.get("agent_results", {}).copy()
        agent_results[agent.name] = result
        
        return {
            "agent_results": agent_results,
            "completed_agents": state.get("completed_agents", set()) | {agent.name},
        }
```

### 2. LangGraph Adapter for DynamicDependencyGraph

```python
# Jotty/core/orchestration/langgraph_orchestrator.py

from langgraph.graph import StateGraph, END, START, Annotation
from typing import Dict, Any, List, Set, Optional
from .dynamic_dependency_graph import DynamicDependencyGraph
from .conductor import Conductor
from ..foundation.agent_config import AgentConfig

class JottyLangGraphState(TypedDict):
    """State for LangGraph execution of Jotty's undirected graph."""
    
    # Jotty components (preserved)
    conductor: Conductor
    dependency_graph: DynamicDependencyGraph
    roadmap: Any  # MarkovianTODO
    
    # Execution state
    goal: str
    context: Dict[str, Any]
    
    # Agent execution tracking
    completed_agents: Set[str]  # Agent names that completed
    agent_results: Dict[str, Any]  # agent_name -> result
    current_agent: Optional[str]
    
    # LangGraph control
    iteration: int
    max_iterations: int
    should_stop: bool
    error: Optional[str]
    
    # Streaming
    stream_events: List[Dict[str, Any]]

class LangGraphOrchestrator:
    """
    Orchestrates Jotty's undirected dependency graph via LangGraph.
    
    Key Features:
    - Dynamically builds LangGraph from Jotty's dependency resolution
    - Uses same agents from Jotty's graph
    - Provides LangGraph's observability benefits
    """
    
    def __init__(self, conductor: Conductor):
        self.conductor = conductor
        self.graph: Optional[StateGraph] = None
    
    def build_graph_from_dependencies(self) -> StateGraph:
        """
        Dynamically build LangGraph from Jotty's DynamicDependencyGraph.
        
        Process:
        1. Get current state of dependency graph
        2. Create LangGraph node for each agent
        3. Create edges based on dependency resolution
        4. Handle parallel execution groups
        """
        graph = StateGraph(JottyLangGraphState)
        
        # Get agents from conductor
        agents = self.conductor.actors
        agent_map = {agent.name: agent for agent in agents}
        
        # Get dependency graph snapshot
        dag = self.conductor.dependency_graph
        snapshot = dag.get_snapshot()
        
        # Create LangGraph node for each agent
        for agent in agents:
            graph.add_node(
                agent.name,
                lambda state, a=agent: self._execute_agent_node(state, a)
            )
        
        # Build edges from dependency graph
        # For each agent, find its dependencies and create edges
        for agent in agents:
            agent_id = agent.name
            
            # Get agents that this agent depends on
            dependencies = snapshot.dependencies.get(agent_id, [])
            
            if not dependencies:
                # No dependencies - this is an entry point
                # Will be handled by conditional routing
                pass
            else:
                # Create edges from dependencies to this agent
                for dep_id in dependencies:
                    if dep_id in agent_map:
                        # Sequential edge: dep → agent
                        graph.add_edge(dep_id, agent_id)
        
        # Add conditional routing for entry nodes
        graph.add_conditional_edges(
            START,
            self._select_entry_nodes,
            {agent.name: agent.name for agent in agents if not snapshot.dependencies.get(agent.name)}
        )
        
        # Add conditional routing for completion
        for agent in agents:
            graph.add_conditional_edges(
                agent.name,
                lambda state, a=agent: self._should_continue(state, a),
                {
                    "continue": self._get_next_agents,
                    "complete": END
                }
            )
        
        return graph.compile()
    
    async def _execute_agent_node(
        self,
        state: JottyLangGraphState,
        agent: AgentConfig
    ) -> Dict[str, Any]:
        """
        Execute a single agent node.
        Uses Jotty's execution engine (JottyCore) with Architect/Auditor.
        """
        # Resolve parameters using Jotty's parameter resolver
        params = await self.conductor.parameter_resolver.resolve(
            agent=agent,
            context=state["context"],
            previous_results=state["agent_results"]
        )
        
        # Execute via Jotty's JottyCore (includes Architect/Auditor)
        result = await self.conductor.jotty_core.execute_agent(
            agent=agent,
            params=params,
            context=state["context"]
        )
        
        # Update dependency graph
        await self.conductor.dependency_graph.mark_completed(agent.name)
        
        # Store result
        agent_results = state["agent_results"].copy()
        agent_results[agent.name] = result
        
        # Update learning
        await self._update_learning(state, agent, result)
        
        # Emit stream event
        stream_events = state["stream_events"].copy()
        stream_events.append({
            "type": "agent_complete",
            "agent": agent.name,
            "result": result,
            "timestamp": time.time()
        })
        
        return {
            "agent_results": agent_results,
            "completed_agents": state["completed_agents"] | {agent.name},
            "current_agent": None,
            "stream_events": stream_events,
            "iteration": state["iteration"] + 1,
        }
    
    def _select_entry_nodes(self, state: JottyLangGraphState) -> str:
        """Select entry nodes (agents with no dependencies)."""
        snapshot = state["dependency_graph"].get_snapshot()
        
        # Find agents with no dependencies
        entry_agents = [
            agent.name
            for agent in state["conductor"].actors
            if not snapshot.dependencies.get(agent.name)
        ]
        
        if entry_agents:
            # Return first entry agent
            return entry_agents[0]
        else:
            # Fallback: return first agent
            return state["conductor"].actors[0].name
    
    def _should_continue(
        self,
        state: JottyLangGraphState,
        agent: AgentConfig
    ) -> str:
        """Determine if execution should continue or complete."""
        snapshot = state["dependency_graph"].get_snapshot()
        
        # Check if there are more agents to execute
        # Get agents that depend on this agent
        dependents = snapshot.dependents.get(agent.name, [])
        
        # Filter to agents that can now execute (all deps met)
        ready_agents = [
            dep_id
            for dep_id in dependents
            if snapshot.can_execute(dep_id)
        ]
        
        if ready_agents:
            return "continue"
        else:
            # Check if all agents are done
            all_agents = {a.name for a in state["conductor"].actors}
            if state["completed_agents"] == all_agents:
                return "complete"
            else:
                return "continue"
    
    def _get_next_agents(self, state: JottyLangGraphState) -> str:
        """Get next agent(s) to execute based on dependency graph."""
        snapshot = state["dependency_graph"].get_snapshot()
        
        # Get independent tasks (can run in parallel)
        independent = snapshot.get_independent_tasks()
        
        if independent:
            # Return first independent agent
            return independent[0]
        else:
            # No more agents
            return END
    
    async def _update_learning(
        self,
        state: JottyLangGraphState,
        agent: AgentConfig,
        result: Any
    ):
        """Update Jotty's learning systems."""
        # Calculate reward
        reward = self._calculate_reward(result)
        
        # Update Q-learning
        if hasattr(self.conductor, 'q_predictor'):
            await self.conductor.q_predictor.record_outcome(
                state={
                    "agent": agent.name,
                    "completed_agents": list(state["completed_agents"]),
                },
                action={"agent": agent.name},
                reward=reward
            )
        
        # Update memory
        if hasattr(self.conductor, 'memory'):
            await self.conductor.memory.store_episode(
                episode_id=state.get("episode_id"),
                trajectory=[{
                    "agent": agent.name,
                    "result": result,
                }]
            )
    
    def _calculate_reward(self, result: Any) -> float:
        """Calculate reward from agent result."""
        if hasattr(result, 'success') and result.success:
            return 1.0
        elif hasattr(result, 'validation_passed') and result.validation_passed:
            return 0.8
        else:
            return 0.2
    
    async def run(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Run Jotty's undirected graph via LangGraph orchestration.
        
        Process:
        1. Initialize Jotty's dependency graph
        2. Build LangGraph dynamically from dependencies
        3. Execute via LangGraph
        4. Return results
        """
        # Initialize Jotty's systems
        await self.conductor._initialize_episode(goal, context or {})
        
        # Build LangGraph from current dependency state
        self.graph = self.build_graph_from_dependencies()
        
        # Initial state
        initial_state: JottyLangGraphState = {
            "conductor": self.conductor,
            "dependency_graph": self.conductor.dependency_graph,
            "roadmap": self.conductor.roadmap,
            "goal": goal,
            "context": context or {},
            "completed_agents": set(),
            "agent_results": {},
            "current_agent": None,
            "iteration": 0,
            "max_iterations": max_iterations,
            "should_stop": False,
            "error": None,
            "stream_events": [],
        }
        
        # Execute via LangGraph
        final_state = await self.graph.ainvoke(initial_state)
        
        return {
            "success": not final_state["error"],
            "results": final_state["agent_results"],
            "completed_agents": list(final_state["completed_agents"]),
            "stream_events": final_state["stream_events"],
            "error": final_state["error"],
        }
    
    async def run_stream(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Run with streaming events."""
        # Initialize
        await self.conductor._initialize_episode(goal, context or {})
        self.graph = self.build_graph_from_dependencies()
        
        initial_state: JottyLangGraphState = {
            "conductor": self.conductor,
            "dependency_graph": self.conductor.dependency_graph,
            "roadmap": self.conductor.roadmap,
            "goal": goal,
            "context": context or {},
            "completed_agents": set(),
            "agent_results": {},
            "current_agent": None,
            "iteration": 0,
            "max_iterations": 100,
            "should_stop": False,
            "error": None,
            "stream_events": [],
        }
        
        # Stream execution
        async for event in self.graph.astream(initial_state):
            yield event
```

### 3. Unified Orchestrator with Dual Mode Support

```python
# Jotty/core/orchestration/langgraph_orchestrator.py

from enum import Enum
from typing import Optional, Union

class GraphMode(Enum):
    """Graph execution mode."""
    DYNAMIC = "dynamic"  # Use Jotty's DynamicDependencyGraph
    STATIC = "static"    # Use static LangGraph definition
    LEGACY = "legacy"    # Use original Jotty execution (no LangGraph)

class LangGraphOrchestrator:
    """
    Unified orchestrator supporting both static and dynamic modes.
    """
    
    def __init__(
        self,
        conductor: Conductor,
        mode: GraphMode = GraphMode.DYNAMIC,
        static_graph: Optional[StaticLangGraphDefinition] = None
    ):
        self.conductor = conductor
        self.mode = mode
        self.static_graph = static_graph
        
        if mode == GraphMode.STATIC and not static_graph:
            raise ValueError("Static mode requires static_graph definition")
        
        self.graph: Optional[StateGraph] = None
    
    def build_graph(self) -> StateGraph:
        """Build LangGraph based on mode."""
        if self.mode == GraphMode.DYNAMIC:
            return self._build_dynamic_graph()
        elif self.mode == GraphMode.STATIC:
            return self.static_graph.build_langgraph(self.conductor)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
    
    def _build_dynamic_graph(self) -> StateGraph:
        """Build graph from Jotty's DynamicDependencyGraph."""
        # ... existing dynamic graph building code ...
        pass
```

### 4. Integration with Conductor

```python
# Jotty/core/orchestration/conductor.py

class Conductor:
    """Main orchestrator - now with optional LangGraph orchestration."""
    
    def __init__(
        self,
        ...,
        use_langgraph: bool = False,
        langgraph_mode: str = "dynamic",  # "dynamic" or "static"
        static_graph: Optional[StaticLangGraphDefinition] = None
    ):
        # ... existing initialization ...
        
        self.use_langgraph = use_langgraph
        self.langgraph_mode = langgraph_mode
        
        if use_langgraph:
            from .langgraph_orchestrator import LangGraphOrchestrator, GraphMode
            
            mode = GraphMode.DYNAMIC if langgraph_mode == "dynamic" else GraphMode.STATIC
            self.langgraph_orchestrator = LangGraphOrchestrator(
                self,
                mode=mode,
                static_graph=static_graph
            )
        else:
            self.langgraph_orchestrator = None
    
    async def run(self, goal: str, **kwargs):
        """Run with or without LangGraph orchestration."""
        if self.use_langgraph and self.langgraph_orchestrator:
            # Use LangGraph to orchestrate
            return await self.langgraph_orchestrator.run(goal, **kwargs)
        else:
            # Legacy execution (existing code)
            return await self._run_legacy(goal, **kwargs)
    
    async def run_stream(self, goal: str, **kwargs):
        """Run with streaming (requires LangGraph)."""
        if not self.langgraph_orchestrator:
            raise ValueError("Streaming requires LangGraph. Set use_langgraph=True")
        
        async for event in self.langgraph_orchestrator.run_stream(goal, **kwargs):
            yield event
    
    def define_static_graph(
        self,
        name: str,
        description: str = ""
    ) -> StaticLangGraphDefinition:
        """
        Create a static graph definition.
        
        Usage:
            graph = conductor.define_static_graph("My Workflow")
            graph.add_node(research_agent)
            graph.add_node(analyze_agent)
            graph.add_edge("ResearchAgent", "AnalyzeAgent")
            
            conductor.langgraph_mode = "static"
            conductor.static_graph = graph
        """
        from .static_langgraph import StaticLangGraphDefinition
        return StaticLangGraphDefinition(name, description)
```

### 3. Dynamic Graph Updates

```python
# Jotty/core/orchestration/langgraph_orchestrator.py

class LangGraphOrchestrator:
    """Orchestrates Jotty's undirected graph via LangGraph."""
    
    def rebuild_graph_if_needed(self, state: JottyLangGraphState) -> bool:
        """
        Rebuild LangGraph if dependency graph changed.
        
        Jotty's dependency graph can change at runtime:
        - New dependencies discovered
        - Dependencies removed
        - Parallel execution opportunities emerge
        
        This method checks if graph needs rebuilding.
        """
        snapshot = state["dependency_graph"].get_snapshot()
        
        # Check if graph structure changed
        if not hasattr(self, '_last_graph_hash'):
            self._last_graph_hash = None
        
        current_hash = hash(json.dumps({
            "dependencies": snapshot.dependencies,
            "completed": list(snapshot.completed_tasks),
        }))
        
        if current_hash != self._last_graph_hash:
            # Graph changed - rebuild
            self.graph = self.build_graph_from_dependencies()
            self._last_graph_hash = current_hash
            return True
        
        return False
```

## Benefits

### 1. Preserves Jotty's Intelligence
- ✅ Keeps DynamicDependencyGraph (adaptive, non-directed)
- ✅ Keeps learning systems (Q-learning, memory)
- ✅ Keeps validation (Architect/Auditor)
- ✅ Same agents, same execution logic

### 2. Adds LangGraph Benefits
- ✅ **Observability**: Visual graph representation
- ✅ **Streaming**: Real-time execution events
- ✅ **Debugging**: Explicit state transitions
- ✅ **Discoverability**: Can inspect graph structure

### 3. Dynamic & Adaptive
- ✅ Graph rebuilds as dependencies change
- ✅ Handles parallel execution opportunities
- ✅ Adapts to runtime discoveries

## Example Usage

### Dynamic Mode (Jotty's Dependency Graph)

```python
from Jotty import Conductor, AgentConfig, JottyConfig

# Define agents (same as before)
agents = [
    AgentConfig(name="ResearchAgent", agent=ResearchAgent(), ...),
    AgentConfig(name="AnalyzeAgent", agent=AnalyzeAgent(), ...),
    AgentConfig(name="ReportAgent", agent=ReportAgent(), ...),
]

# Create conductor with LangGraph orchestration (dynamic mode)
config = JottyConfig(...)
conductor = Conductor(
    actors=agents,
    config=config,
    use_langgraph=True,
    langgraph_mode="dynamic"  # Use Jotty's DynamicDependencyGraph
)

# Run - LangGraph orchestrates Jotty's undirected graph
result = await conductor.run(
    goal="Research AI trends and create report",
    query="What are the latest AI trends?"
)

# Or with streaming
async for event in conductor.run_stream(goal="...", query="..."):
    print(f"Event: {event}")
```

### Static Mode (Direct LangGraph Definition)

```python
from Jotty import Conductor, AgentConfig, JottyConfig

# Define agents
research_agent = AgentConfig(name="ResearchAgent", agent=ResearchAgent(), ...)
analyze_agent = AgentConfig(name="AnalyzeAgent", agent=AnalyzeAgent(), ...)
report_agent = AgentConfig(name="ReportAgent", agent=ReportAgent(), ...)

agents = [research_agent, analyze_agent, report_agent]

# Create conductor
config = JottyConfig(...)
conductor = Conductor(
    actors=agents,
    config=config,
    use_langgraph=True,
    langgraph_mode="static"  # Use static graph definition
)

# Define static graph with explicit edges
static_graph = conductor.define_static_graph(
    name="Research → Analyze → Report",
    description="Fixed workflow for research tasks"
)

# Add nodes
static_graph.add_node(research_agent)
static_graph.add_node(analyze_agent)
static_graph.add_node(report_agent)

# Add edges (explicit execution order)
static_graph.add_edge("ResearchAgent", "AnalyzeAgent")
static_graph.add_edge("AnalyzeAgent", "ReportAgent")

# Set static graph
conductor.static_graph = static_graph

# Run - LangGraph executes static graph
result = await conductor.run(
    goal="Research AI trends and create report",
    query="What are the latest AI trends?"
)
```

### Hybrid: Static Graph with Conditional Edges

```python
# Define static graph with conditional routing
static_graph = conductor.define_static_graph("Conditional Review")

static_graph.add_node(generate_agent)
static_graph.add_node(review_agent)
static_graph.add_node(approve_agent)
static_graph.add_node(revise_agent)

# Sequential edges
static_graph.add_edge("GenerateAgent", "ReviewAgent")

# Conditional edge: if score >= 0.8, approve; else revise
def check_score(state):
    review_result = state["agent_results"].get("ReviewAgent")
    if review_result and hasattr(review_result, "score"):
        return "approve" if review_result.score >= 0.8 else "revise"
    return "revise"

static_graph.add_conditional_edge(
    "ReviewAgent",
    check_score,
    {
        "approve": "ApproveAgent",
        "revise": "ReviseAgent"
    }
)

conductor.static_graph = static_graph
conductor.langgraph_mode = "static"

result = await conductor.run(goal="Generate and review code")
```

### Switching Between Modes

```python
# Start with dynamic mode
conductor = Conductor(..., use_langgraph=True, langgraph_mode="dynamic")
result1 = await conductor.run(goal="...")  # Uses DynamicDependencyGraph

# Switch to static mode
static_graph = conductor.define_static_graph("My Workflow")
static_graph.add_node(agent1)
static_graph.add_node(agent2)
static_graph.add_edge("Agent1", "Agent2")
conductor.static_graph = static_graph
conductor.langgraph_mode = "static"

result2 = await conductor.run(goal="...")  # Uses static graph

# Switch back to dynamic
conductor.langgraph_mode = "dynamic"
result3 = await conductor.run(goal="...")  # Uses DynamicDependencyGraph again
```

## Comparison: Modes

### Mode 1: Dynamic (Jotty's Dependency Graph)
```
DynamicDependencyGraph (Undirected) - Intelligence Layer
    ↓
LangGraphOrchestrator.build_graph_from_dependencies()
    ↓
LangGraph State Machine (Directed) - Execution Layer
    ↓
Agent Execution (Same agents, adaptive order)
```

**When to Use:**
- Dependencies discovered at runtime
- Adaptive workflows
- Learning from experience
- Complex, evolving tasks

**Benefits:**
- Adaptive to runtime conditions
- Learns optimal execution order
- Handles dynamic dependencies
- Visual graph representation
- Streaming support

### Mode 2: Static (Direct LangGraph)
```
StaticLangGraphDefinition (Explicit Nodes/Edges)
    ↓
LangGraph State Machine (Fixed) - Execution Layer
    ↓
Agent Execution (Same agents, fixed order)
```

**When to Use:**
- Known execution flow upfront
- Reproducible workflows
- Explicit control needed
- Simple, linear pipelines

**Benefits:**
- Predictable execution
- Explicit control
- Easy to understand
- Visual graph representation
- Streaming support
- No dependency resolution overhead

### Mode 3: Legacy (Original Jotty)
```
DynamicDependencyGraph (Undirected)
    ↓
Conductor.run() (Internal execution)
    ↓
Agent Execution
```

**When to Use:**
- Don't need LangGraph features
- Want minimal overhead
- Existing code compatibility

**Limitations:**
- Hard to visualize execution flow
- No streaming support
- Limited observability

## Implementation Plan

### Phase 1: Static Graph Support (Week 1-2)
- [ ] Implement `StaticLangGraphDefinition` class
- [ ] Implement node/edge management
- [ ] Implement conditional edges
- [ ] Build LangGraph from static definition
- [ ] Integrate with Conductor

### Phase 2: Dynamic Graph Orchestration (Week 3-4)
- [ ] Implement `LangGraphOrchestrator` (dynamic mode)
- [ ] Implement `build_graph_from_dependencies()`
- [ ] Implement agent node execution
- [ ] Handle runtime dependency changes
- [ ] Support parallel execution groups

### Phase 3: Unified Orchestrator (Week 5)
- [ ] Create unified `LangGraphOrchestrator` supporting both modes
- [ ] Add mode switching capability
- [ ] Add graph rebuilding for dynamic mode
- [ ] Integrate with Conductor API

### Phase 4: Streaming & Observability (Week 6)
- [ ] Add streaming support (both modes)
- [ ] Add graph visualization
- [ ] Add debugging tools
- [ ] Add mode comparison utilities

### Phase 5: Testing & Polish (Week 7-8)
- [ ] Unit tests (both modes)
- [ ] Integration tests
- [ ] Performance comparison
- [ ] Documentation
- [ ] Examples (dynamic, static, hybrid)

## Success Criteria

- ✅ Static graph mode functional
- ✅ Dynamic graph mode functional
- ✅ Mode switching works
- ✅ Same agents work in both modes
- ✅ Streaming support functional (both modes)
- ✅ Graph visualization working
- ✅ Backward compatible (opt-in)
- ✅ Performance acceptable for both modes

## Decision Matrix: When to Use Which Mode?

| Scenario | Recommended Mode | Reason |
|----------|------------------|--------|
| Known workflow upfront | **Static** | Predictable, explicit control |
| Dependencies change at runtime | **Dynamic** | Adaptive to conditions |
| Learning optimal order | **Dynamic** | Uses Jotty's learning |
| Simple linear pipeline | **Static** | Easier to understand |
| Complex multi-agent task | **Dynamic** | Handles complexity |
| Need reproducibility | **Static** | Fixed execution order |
| Need observability | **Both** | LangGraph provides this |
| Don't need LangGraph features | **Legacy** | Minimal overhead |

---

## Key Insights

1. **Static Mode**: For explicit, predictable workflows. Define once, execute many times.

2. **Dynamic Mode**: For adaptive, learning workflows. Jotty's intelligence determines execution order.

3. **Both Use LangGraph**: LangGraph provides the **execution engine** and **observability layer** for both modes.

4. **Same Agents**: No duplication - use same AgentConfig instances in both modes.

5. **Flexible**: Can switch between modes or even combine (start static, allow dynamic updates).

# DSPy Agent Graph Template System

## Overview

Create an **independent directed graph of DSPy agents** that can be **templatized with explicit ordering**. This allows users to define reusable agent workflows with precise execution sequences.

## Current State

### Existing Systems:
1. **WorkflowPreset** (JustJot.ai): Has nodes/connections but not DSPy-specific
2. **SwarmPreset**: Simple agent list with parallel flag
3. **Jotty Conductor**: Dynamic dependency resolution (non-directed)

### Gap:
- No way to define **templated DSPy agent graphs** with explicit ordering
- No reusable agent workflow templates
- No visual representation of agent execution flow

## Proposed Solution

### 1. DSPy Agent Graph Template

```python
# Jotty/core/orchestration/agent_graph_template.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
import json

class EdgeType(Enum):
    """Types of edges in agent graph."""
    SEQUENTIAL = "sequential"  # Agent A → Agent B (B waits for A)
    PARALLEL = "parallel"       # Agent A → (Agent B, Agent C) (B & C run together)
    CONDITIONAL = "conditional"  # Agent A → Agent B if condition, else Agent C
    MERGE = "merge"             # (Agent A, Agent B) → Agent C (C waits for both)

@dataclass
class AgentNode:
    """A node in the agent graph representing a DSPy agent."""
    id: str
    agent_name: str  # Name of DSPy agent (from AgentConfig)
    agent_config: Optional[Any] = None  # Reference to AgentConfig
    
    # Node metadata
    label: Optional[str] = None
    description: Optional[str] = None
    
    # Execution config
    max_retries: int = 3
    timeout: Optional[float] = None
    required: bool = True  # If False, can skip on failure
    
    # Parameter mapping
    input_mapping: Dict[str, str] = field(default_factory=dict)
    # e.g., {"query": "context.goal", "data": "PreviousAgent.output"}
    
    # Output handling
    output_key: Optional[str] = None  # Key to store output in context
    # e.g., "research_results" → context["research_results"]

@dataclass
class AgentEdge:
    """An edge connecting two agent nodes."""
    id: str
    from_node: str  # Source agent node ID
    to_node: str   # Target agent node ID
    edge_type: EdgeType = EdgeType.SEQUENTIAL
    
    # Conditional execution
    condition: Optional[str] = None  # Python expression, e.g., "result.success == True"
    
    # Parallel execution
    parallel_group: Optional[str] = None  # Groups parallel edges
    
    # Merge strategy (for MERGE edges)
    merge_strategy: Optional[str] = None  # "concatenate", "merge_json", "first", "last"

@dataclass
class AgentGraphTemplate:
    """
    A templated directed graph of DSPy agents with explicit ordering.
    
    Example:
        ResearchAgent → (AnalyzeAgent, SummarizeAgent) → ReportAgent
    """
    id: str
    name: str
    description: str
    
    # Graph structure
    nodes: List[AgentNode] = field(default_factory=list)
    edges: List[AgentEdge] = field(default_factory=list)
    
    # Entry/exit points
    entry_nodes: List[str] = field(default_factory=list)  # Starting nodes
    exit_nodes: List[str] = field(default_factory=list)  # Ending nodes
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    category: Optional[str] = None
    version: str = "1.0.0"
    
    def validate(self) -> List[str]:
        """Validate graph structure. Returns list of errors."""
        errors = []
        
        # Check all edges reference valid nodes
        node_ids = {n.id for n in self.nodes}
        for edge in self.edges:
            if edge.from_node not in node_ids:
                errors.append(f"Edge {edge.id}: from_node '{edge.from_node}' not found")
            if edge.to_node not in node_ids:
                errors.append(f"Edge {edge.id}: to_node '{edge.to_node}' not found")
        
        # Check entry nodes exist
        for entry_id in self.entry_nodes:
            if entry_id not in node_ids:
                errors.append(f"Entry node '{entry_id}' not found")
        
        # Check for cycles (if sequential only)
        if self._has_cycles():
            errors.append("Graph contains cycles (not allowed for sequential execution)")
        
        # Check all nodes are reachable
        unreachable = self._find_unreachable_nodes()
        if unreachable:
            errors.append(f"Unreachable nodes: {unreachable}")
        
        return errors
    
    def _has_cycles(self) -> bool:
        """Check if graph has cycles (DFS)."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for edge in self.edges:
                if edge.from_node == node_id:
                    if edge.to_node not in visited:
                        if has_cycle(edge.to_node):
                            return True
                    elif edge.to_node in rec_stack:
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        for node in self.nodes:
            if node.id not in visited:
                if has_cycle(node.id):
                    return True
        
        return False
    
    def _find_unreachable_nodes(self) -> List[str]:
        """Find nodes not reachable from entry nodes."""
        if not self.entry_nodes:
            return []
        
        reachable = set()
        
        def dfs(node_id: str):
            if node_id in reachable:
                return
            reachable.add(node_id)
            
            for edge in self.edges:
                if edge.from_node == node_id:
                    dfs(edge.to_node)
        
        for entry_id in self.entry_nodes:
            dfs(entry_id)
        
        all_node_ids = {n.id for n in self.nodes}
        return list(all_node_ids - reachable)
    
    def get_execution_order(self) -> List[List[str]]:
        """
        Get topological execution order.
        
        Returns:
            List of execution levels, each level can run in parallel.
            Example: [["A"], ["B", "C"], ["D"]]
        """
        # Build adjacency list
        in_degree = {n.id: 0 for n in self.nodes}
        adj_list = {n.id: [] for n in self.nodes}
        
        for edge in self.edges:
            if edge.edge_type == EdgeType.SEQUENTIAL:
                adj_list[edge.from_node].append(edge.to_node)
                in_degree[edge.to_node] += 1
        
        # Topological sort (Kahn's algorithm)
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result = []
        
        while queue:
            level = []
            next_queue = []
            
            for node_id in queue:
                level.append(node_id)
                for neighbor in adj_list[node_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_queue.append(neighbor)
            
            result.append(level)
            queue = next_queue
        
        return result
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": [
                {
                    "id": n.id,
                    "agent_name": n.agent_name,
                    "label": n.label,
                    "description": n.description,
                    "max_retries": n.max_retries,
                    "timeout": n.timeout,
                    "required": n.required,
                    "input_mapping": n.input_mapping,
                    "output_key": n.output_key,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "id": e.id,
                    "from_node": e.from_node,
                    "to_node": e.to_node,
                    "edge_type": e.edge_type.value,
                    "condition": e.condition,
                    "parallel_group": e.parallel_group,
                    "merge_strategy": e.merge_strategy,
                }
                for e in self.edges
            ],
            "entry_nodes": self.entry_nodes,
            "exit_nodes": self.exit_nodes,
            "tags": self.tags,
            "category": self.category,
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "AgentGraphTemplate":
        """Deserialize from dictionary."""
        nodes = [
            AgentNode(
                id=n["id"],
                agent_name=n["agent_name"],
                label=n.get("label"),
                description=n.get("description"),
                max_retries=n.get("max_retries", 3),
                timeout=n.get("timeout"),
                required=n.get("required", True),
                input_mapping=n.get("input_mapping", {}),
                output_key=n.get("output_key"),
            )
            for n in data["nodes"]
        ]
        
        edges = [
            AgentEdge(
                id=e["id"],
                from_node=e["from_node"],
                to_node=e["to_node"],
                edge_type=EdgeType(e["edge_type"]),
                condition=e.get("condition"),
                parallel_group=e.get("parallel_group"),
                merge_strategy=e.get("merge_strategy"),
            )
            for e in data["edges"]
        ]
        
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            nodes=nodes,
            edges=edges,
            entry_nodes=data.get("entry_nodes", []),
            exit_nodes=data.get("exit_nodes", []),
            tags=data.get("tags", []),
            category=data.get("category"),
            version=data.get("version", "1.0.0"),
        )
```

### 2. LangGraph Executor for Agent Graph Templates

```python
# Jotty/core/orchestration/agent_graph_executor.py

from langgraph.graph import StateGraph, END, START
from typing import Dict, Any, List
from .agent_graph_template import AgentGraphTemplate, EdgeType
from .conductor import Conductor

class AgentGraphState(TypedDict):
    """State for agent graph execution."""
    template: AgentGraphTemplate
    conductor: Conductor
    
    # Execution state
    current_level: int
    execution_order: List[List[str]]
    completed_nodes: Set[str]
    node_results: Dict[str, Any]
    
    # Context
    context: Dict[str, Any]
    goal: str
    
    # Control
    should_stop: bool
    error: Optional[str]

class AgentGraphExecutor:
    """
    Execute AgentGraphTemplate using LangGraph state machine.
    
    Converts template's directed graph into LangGraph execution flow.
    """
    
    def __init__(self, template: AgentGraphTemplate, conductor: Conductor):
        self.template = template
        self.conductor = conductor
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph from template."""
        graph = StateGraph(AgentGraphState)
        
        # Add node for each agent in template
        for node in self.template.nodes:
            graph.add_node(
                node.id,
                lambda state, n=node: self._execute_agent_node(state, n)
            )
        
        # Add edges based on template
        for edge in self.template.edges:
            if edge.edge_type == EdgeType.SEQUENTIAL:
                graph.add_edge(edge.from_node, edge.to_node)
            elif edge.edge_type == EdgeType.PARALLEL:
                # Parallel execution: both target nodes depend on source
                graph.add_edge(edge.from_node, edge.to_node)
            elif edge.edge_type == EdgeType.CONDITIONAL:
                # Conditional: add conditional edge
                graph.add_conditional_edges(
                    edge.from_node,
                    lambda state, e=edge: self._check_condition(state, e),
                    {
                        "true": edge.to_node,
                        "false": END  # Or next conditional branch
                    }
                )
            elif edge.edge_type == EdgeType.MERGE:
                # Merge: target waits for all sources
                # LangGraph handles this automatically with multiple edges
                graph.add_edge(edge.from_node, edge.to_node)
        
        # Set entry point
        if self.template.entry_nodes:
            graph.set_entry_point(self.template.entry_nodes[0])
        else:
            # Use first node in execution order
            exec_order = self.template.get_execution_order()
            if exec_order:
                graph.set_entry_point(exec_order[0][0])
        
        return graph.compile()
    
    async def _execute_agent_node(
        self,
        state: AgentGraphState,
        node: AgentNode
    ) -> Dict[str, Any]:
        """Execute a single agent node."""
        # Resolve input parameters from context
        params = self._resolve_parameters(node, state)
        
        # Execute agent via Conductor
        agent_config = self._get_agent_config(node.agent_name)
        result = await state["conductor"].jotty_core.execute_agent(
            agent=agent_config,
            params=params,
            context=state["context"]
        )
        
        # Store result
        node_results = state["node_results"].copy()
        if node.output_key:
            node_results[node.output_key] = result.output
        node_results[node.id] = result
        
        return {
            "node_results": node_results,
            "completed_nodes": state["completed_nodes"] | {node.id}
        }
    
    def _resolve_parameters(
        self,
        node: AgentNode,
        state: AgentGraphState
    ) -> Dict[str, Any]:
        """Resolve input parameters from context using mapping."""
        params = {}
        
        for param_name, context_path in node.input_mapping.items():
            # Parse context path, e.g., "PreviousAgent.output" or "context.goal"
            value = self._get_context_value(context_path, state)
            params[param_name] = value
        
        return params
    
    def _get_context_value(
        self,
        path: str,
        state: AgentGraphState
    ) -> Any:
        """Get value from context using dot notation."""
        parts = path.split(".")
        
        if parts[0] == "context":
            # Access state["context"]
            value = state["context"]
            for part in parts[1:]:
                value = value.get(part)
        else:
            # Access node result, e.g., "ResearchAgent.output"
            node_id = parts[0]
            node_results = state["node_results"]
            if node_id in node_results:
                result = node_results[node_id]
                if len(parts) > 1 and parts[1] == "output":
                    return result.output if hasattr(result, "output") else result
                return result
        
        return value
    
    async def execute(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the agent graph template."""
        initial_state: AgentGraphState = {
            "template": self.template,
            "conductor": self.conductor,
            "current_level": 0,
            "execution_order": self.template.get_execution_order(),
            "completed_nodes": set(),
            "node_results": {},
            "context": context or {},
            "goal": goal,
            "should_stop": False,
            "error": None,
        }
        
        # Execute via LangGraph
        final_state = await self.graph.ainvoke(initial_state)
        
        return {
            "success": not final_state["error"],
            "results": final_state["node_results"],
            "completed_nodes": list(final_state["completed_nodes"]),
            "error": final_state["error"],
        }
```

### 3. Template Registry & Persistence

```python
# Jotty/core/orchestration/template_registry.py

from typing import Dict, List, Optional
from pathlib import Path
import json
from .agent_graph_template import AgentGraphTemplate

class AgentGraphTemplateRegistry:
    """Registry for agent graph templates."""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or Path("templates/agent_graphs")
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self._templates: Dict[str, AgentGraphTemplate] = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all templates from disk."""
        for template_file in self.templates_dir.glob("*.json"):
            try:
                with open(template_file, "r") as f:
                    data = json.load(f)
                    template = AgentGraphTemplate.from_dict(data)
                    self._templates[template.id] = template
            except Exception as e:
                logger.warning(f"Failed to load template {template_file}: {e}")
    
    def register(self, template: AgentGraphTemplate):
        """Register a template."""
        # Validate
        errors = template.validate()
        if errors:
            raise ValueError(f"Template validation failed: {errors}")
        
        self._templates[template.id] = template
        
        # Persist
        template_file = self.templates_dir / f"{template.id}.json"
        with open(template_file, "w") as f:
            json.dump(template.to_dict(), f, indent=2)
    
    def get(self, template_id: str) -> Optional[AgentGraphTemplate]:
        """Get template by ID."""
        return self._templates.get(template_id)
    
    def list(self, category: Optional[str] = None) -> List[AgentGraphTemplate]:
        """List all templates, optionally filtered by category."""
        templates = list(self._templates.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return templates
    
    def delete(self, template_id: str):
        """Delete a template."""
        if template_id in self._templates:
            del self._templates[template_id]
            template_file = self.templates_dir / f"{template_id}.json"
            if template_file.exists():
                template_file.unlink()
```

### 4. Example Templates

```python
# Example: Research → Analyze → Report workflow

research_template = AgentGraphTemplate(
    id="research_analyze_report",
    name="Research → Analyze → Report",
    description="Research topic, analyze findings, generate report",
    nodes=[
        AgentNode(
            id="research",
            agent_name="ResearchAgent",
            label="Research",
            input_mapping={"query": "context.goal"},
            output_key="research_results",
        ),
        AgentNode(
            id="analyze",
            agent_name="AnalyzeAgent",
            label="Analyze",
            input_mapping={"data": "research_results"},
            output_key="analysis",
        ),
        AgentNode(
            id="report",
            agent_name="ReportAgent",
            label="Generate Report",
            input_mapping={
                "research": "research_results",
                "analysis": "analysis",
            },
        ),
    ],
    edges=[
        AgentEdge(
            id="e1",
            from_node="research",
            to_node="analyze",
            edge_type=EdgeType.SEQUENTIAL,
        ),
        AgentEdge(
            id="e2",
            from_node="analyze",
            to_node="report",
            edge_type=EdgeType.SEQUENTIAL,
        ),
    ],
    entry_nodes=["research"],
    exit_nodes=["report"],
    category="research",
)

# Example: Parallel analysis workflow

parallel_template = AgentGraphTemplate(
    id="parallel_analysis",
    name="Parallel Analysis",
    description="Research → (Analyze + Summarize) → Merge",
    nodes=[
        AgentNode(id="research", agent_name="ResearchAgent", output_key="research"),
        AgentNode(id="analyze", agent_name="AnalyzeAgent", output_key="analysis"),
        AgentNode(id="summarize", agent_name="SummarizeAgent", output_key="summary"),
        AgentNode(id="merge", agent_name="MergeAgent"),
    ],
    edges=[
        AgentEdge(id="e1", from_node="research", to_node="analyze", edge_type=EdgeType.SEQUENTIAL),
        AgentEdge(id="e2", from_node="research", to_node="summarize", edge_type=EdgeType.SEQUENTIAL),
        AgentEdge(id="e3", from_node="analyze", to_node="merge", edge_type=EdgeType.MERGE),
        AgentEdge(id="e4", from_node="summarize", to_node="merge", edge_type=EdgeType.MERGE),
    ],
    entry_nodes=["research"],
    exit_nodes=["merge"],
)
```

## Integration with Existing Systems

### 1. WorkflowPreset Integration

```typescript
// JustJot.ai/src/lib/ai/agents/workflow-template-bridge.ts

export function convertWorkflowPresetToAgentGraph(
  preset: IWorkflowPreset
): AgentGraphTemplate {
  // Convert WorkflowPreset (TypeScript) to AgentGraphTemplate (Python)
  // Bridge between JustJot.ai UI and Jotty execution
}
```

### 2. Conductor Integration

```python
# Jotty/core/orchestration/conductor.py

class Conductor:
    def execute_template(
        self,
        template_id: str,
        goal: str,
        context: Optional[Dict] = None
    ):
        """Execute an agent graph template."""
        template = self.template_registry.get(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        executor = AgentGraphExecutor(template, self)
        return await executor.execute(goal, context)
```

## Benefits

1. **Reusability**: Define once, use many times
2. **Explicit Ordering**: Clear execution sequence
3. **Visual Representation**: Can be visualized as directed graph
4. **Type Safety**: Strong typing for nodes/edges
5. **Validation**: Graph structure validation
6. **Flexibility**: Supports sequential, parallel, conditional, merge

## Next Steps

1. Implement `AgentGraphTemplate` class
2. Implement `AgentGraphExecutor` with LangGraph
3. Create template registry
4. Add template validation
5. Create example templates
6. Add UI for template creation/editing
7. Integrate with Conductor

"""
LangGraph Orchestrator
======================

Unified orchestrator supporting both static and dynamic dependency graphs.
Provides consistent API for end users.
"""

from langgraph.graph import StateGraph, END, START
from typing import Dict, Any, List, Set, Optional, TypedDict
from enum import Enum
from .dynamic_dependency_graph import DynamicDependencyGraph
from .static_langgraph import StaticLangGraphDefinition
from .conductor import Conductor
from ..foundation.agent_config import AgentConfig
import logging
import time
import asyncio

logger = logging.getLogger(__name__)


class GraphMode(Enum):
    """Graph execution mode."""
    DYNAMIC = "dynamic"  # Use Jotty's DynamicDependencyGraph
    STATIC = "static"    # Use static LangGraph definition
    LEGACY = "legacy"    # Use original Jotty execution (no LangGraph)


class JottyLangGraphState(TypedDict):
    """State for LangGraph execution."""
    
    # Jotty components
    conductor: Conductor
    dependency_graph: Optional[DynamicDependencyGraph]
    roadmap: Any  # MarkovianTODO
    
    # Execution state
    goal: str
    context: Dict[str, Any]
    
    # Agent execution tracking
    completed_agents: Set[str]
    agent_results: Dict[str, Any]
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
    Unified orchestrator supporting both static and dynamic modes.
    
    Provides consistent API:
    - Dynamic mode: mode="dynamic" (uses Jotty's DynamicDependencyGraph)
    - Static mode: mode="static", agent_order=["Agent1", "Agent2", ...]
    """
    
    def __init__(
        self,
        conductor: Conductor,
        mode: GraphMode = GraphMode.DYNAMIC,
        agent_order: Optional[List[str]] = None,
        static_graph: Optional[StaticLangGraphDefinition] = None
    ):
        """
        Initialize orchestrator.
        
        Args:
            conductor: Conductor instance
            mode: Execution mode ("dynamic" or "static")
            agent_order: Agent execution order (required for static mode)
            static_graph: Pre-built static graph (optional, will create from agent_order if not provided)
        """
        self.conductor = conductor
        self.mode = mode
        
        if mode == GraphMode.STATIC:
            if agent_order:
                # Create static graph from agent_order
                self.static_graph = StaticLangGraphDefinition(
                    name="Static Workflow",
                    description="Static workflow defined by agent_order",
                    agent_order=agent_order
                )
            elif static_graph:
                self.static_graph = static_graph
            else:
                raise ValueError(
                    "Static mode requires either agent_order or static_graph. "
                    "Example: mode='static', agent_order=['Agent1', 'Agent2', 'Agent3']"
                )
        else:
            self.static_graph = None
        
        self.graph: Optional[StateGraph] = None
    
    def build_graph(self) -> StateGraph:
        """Build LangGraph based on mode."""
        if self.mode == GraphMode.DYNAMIC:
            return self._build_dynamic_graph()
        elif self.mode == GraphMode.STATIC:
            return self.static_graph.build_langgraph(
                self.conductor,
                JottyLangGraphState
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
    
    def _build_dynamic_graph(self) -> StateGraph:
        """Build graph from Jotty's DynamicDependencyGraph."""
        graph = StateGraph(JottyLangGraphState)
        
        # Get agents from conductor
        # Conductor stores actors as dict: self.actors[name] = ActorConfig
        actors_dict = self.conductor.actors
        if isinstance(actors_dict, dict):
            agents = list(actors_dict.values())
        else:
            agents = actors_dict
        
        agent_map = {agent.name: agent for agent in agents}
        
        # Get dependency graph snapshot
        dag = self.conductor.dependency_graph
        if not dag:
            # Initialize if not exists
            from .dynamic_dependency_graph import DynamicDependencyGraph
            dag = DynamicDependencyGraph()
            self.conductor.dependency_graph = dag
        
        snapshot = dag.get_snapshot()
        
        # Create LangGraph node for each agent
        # Need to properly wrap async function - LangGraph handles async automatically
        for agent in agents:
            def make_node_func(a):
                async def node_func(state):
                    return await self._execute_agent_node(state, a)
                return node_func
            
            graph.add_node(
                agent.name,
                make_node_func(agent)
            )
        
        # Build edges from dependency graph
        for agent in agents:
            agent_id = agent.name
            dependencies = snapshot.dependencies.get(agent_id, [])
            
            if dependencies:
                # Create edges from dependencies to this agent
                for dep_id in dependencies:
                    if dep_id in agent_map:
                        graph.add_edge(dep_id, agent_id)
        
        # Find entry nodes (agents with no dependencies)
        entry_agents = [
            agent.name
            for agent in agents
            if not snapshot.dependencies.get(agent.name)
        ]
        
        if entry_agents:
            if len(entry_agents) == 1:
                # Single entry point
                graph.set_entry_point(entry_agents[0])
            else:
                # Multiple entry points - start with first one
                graph.set_entry_point(entry_agents[0])
        else:
            # Fallback: start with first agent
            if agents:
                graph.set_entry_point(agents[0].name)
        
        # Find exit nodes (agents with no dependents)
        # For now, connect all agents that have no dependents to END
        exit_agents = []
        all_agent_names = {agent.name for agent in agents}
        for agent in agents:
            agent_id = agent.name
            dependents = snapshot.dependents.get(agent_id, [])
            # Check if any dependents are in our agent list
            has_dependents = any(dep in all_agent_names for dep in dependents)
            if not has_dependents:
                exit_agents.append(agent_id)
        
        # Connect exit nodes to END
        for exit_agent in exit_agents:
            graph.add_edge(exit_agent, END)
        
        # If no exit nodes found, connect last agent to END
        if not exit_agents and agents:
            graph.add_edge(agents[-1].name, END)
        
        return graph.compile()
    
    async def _execute_agent_node(
        self,
        state: JottyLangGraphState,
        agent: AgentConfig
    ) -> Dict[str, Any]:
        """Execute a single agent node."""
        # Resolve parameters - use simple approach for LangGraph
        # Extract from context and previous results
        params = {}
        
        # Get agent signature to know what parameters it needs
        agent_name = agent.name
        if agent_name in self.conductor.actor_signatures:
            sig = self.conductor.actor_signatures[agent_name]
            if isinstance(sig, dict):
                for param_name in sig.keys():
                    # Try to get from context first
                    if param_name in state["context"]:
                        params[param_name] = state["context"][param_name]
                    # Then try previous results
                    elif state["agent_results"]:
                        for prev_name, prev_result in state["agent_results"].items():
                            if hasattr(prev_result, 'output') and hasattr(prev_result.output, param_name):
                                params[param_name] = getattr(prev_result.output, param_name)
                                break
                            elif isinstance(prev_result, dict) and param_name in prev_result:
                                params[param_name] = prev_result[param_name]
                                break
        
        # If no signature, use context directly
        if not params:
            params = state["context"].copy()
        
        # Execute agent - use wrapped agent from conductor
        # Conductor wraps agents with JottyCore, stored in self.actors[name].agent
        agent_to_execute = agent.agent if hasattr(agent, 'agent') else agent
        
        # Check if it's a JottyCore-wrapped agent
        from .jotty_core import JottyCore
        if isinstance(agent_to_execute, JottyCore):
            # Use JottyCore's arun method
            episode_result = await agent_to_execute.arun(**params)
            
            # Extract output from episode result
            class LangGraphResult:
                def __init__(self, episode_result):
                    self.output = episode_result.final_output if hasattr(episode_result, 'final_output') else str(episode_result)
                    self.success = episode_result.success if hasattr(episode_result, 'success') else True
                    self.validation_passed = episode_result.valid if hasattr(episode_result, 'valid') else True
                    self.episode_result = episode_result
            
            result = LangGraphResult(episode_result)
        else:
            # Direct execution fallback
            try:
                if hasattr(agent_to_execute, 'forward'):
                    if asyncio.iscoroutinefunction(agent_to_execute.forward):
                        output = await agent_to_execute.forward(**params)
                    else:
                        output = agent_to_execute.forward(**params)
                else:
                    output = str(params)
                
                # Create simple result object
                class SimpleResult:
                    def __init__(self, output):
                        self.output = output
                        self.success = True
                        self.validation_passed = True
                
                result = SimpleResult(output)
            except Exception as e:
                class ErrorResult:
                    def __init__(self, error):
                        self.output = ""
                        self.success = False
                        self.validation_passed = False
                        self.error = str(error)
                result = ErrorResult(e)
        
        # Update dependency graph (for dynamic mode)
        if self.mode == GraphMode.DYNAMIC and self.conductor.dependency_graph:
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
    
    def _should_continue(
        self,
        state: JottyLangGraphState,
        agent: AgentConfig
    ) -> str:
        """Determine if execution should continue or complete."""
        if self.mode == GraphMode.STATIC:
            # Static mode: check if all agents completed
            actors_dict = self.conductor.actors
            if isinstance(actors_dict, dict):
                all_agents = set(actors_dict.keys())
            else:
                all_agents = {a.name for a in actors_dict}
            
            if state["completed_agents"] == all_agents:
                return "complete"
            return "continue"
        
        # Dynamic mode: check dependency graph
        if state["dependency_graph"]:
            snapshot = state["dependency_graph"].get_snapshot()
            dependents = snapshot.dependents.get(agent.name, [])
            ready_agents = [
                dep_id
                for dep_id in dependents
                if snapshot.can_execute(dep_id)
            ]
            
            if ready_agents:
                return "continue"
        
        # Check if all agents completed
        actors_dict = self.conductor.actors
        if isinstance(actors_dict, dict):
            all_agents = set(actors_dict.keys())
        else:
            all_agents = {a.name for a in actors_dict}
        
        if state["completed_agents"] == all_agents:
            return "complete"
        return "continue"
    
    def _get_next_agents(self, state: JottyLangGraphState) -> str:
        """Get next agent(s) to execute."""
        if self.mode == GraphMode.STATIC:
            # Static mode: follow agent_order
            completed = state["completed_agents"]
            for agent_name in self.static_graph.agent_order:
                if agent_name not in completed:
                    return agent_name
            return END
        
        # Dynamic mode: get from dependency graph
        if state["dependency_graph"]:
            snapshot = state["dependency_graph"].get_snapshot()
            independent = snapshot.get_independent_tasks()
            
            if independent:
                return independent[0]
        
        # Fallback: check if any agents not completed
        actors_dict = self.conductor.actors
        if isinstance(actors_dict, dict):
            all_agents = set(actors_dict.keys())
        else:
            all_agents = {a.name for a in actors_dict}
        
        completed = state["completed_agents"]
        remaining = all_agents - completed
        if remaining:
            return list(remaining)[0]
        
        return END
    
    async def _update_learning(
        self,
        state: JottyLangGraphState,
        agent: AgentConfig,
        result: Any
    ):
        """Update Jotty's learning systems."""
        reward = self._calculate_reward(result)
        
        # Update Q-learning if available
        q_predictor = getattr(self.conductor, 'q_predictor', None) or getattr(self.conductor, 'q_learner', None)
        if q_predictor:
            try:
                await q_predictor.record_outcome(
                    state={
                        "agent": agent.name,
                        "completed_agents": list(state["completed_agents"]),
                    },
                    action={"agent": agent.name},
                    reward=reward
                )
            except Exception as e:
                logger.debug(f"Q-learning update failed: {e}")
        
        # Update memory if available
        memory = getattr(self.conductor, 'memory', None) or getattr(self.conductor, 'shared_memory', None)
        if memory:
            try:
                await memory.store_episode(
                    episode_id=state.get("episode_id"),
                    trajectory=[{
                        "agent": agent.name,
                        "result": result,
                    }]
                )
            except Exception as e:
                logger.debug(f"Memory update failed: {e}")
    
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
        Run workflow via LangGraph orchestration.
        
        Args:
            goal: Goal/task description
            context: Execution context
            max_iterations: Maximum iterations
        
        Returns:
            Execution results
        """
        # Initialize Jotty's systems (if method exists)
        # Note: Conductor may initialize episode in run() method
        
        # Build LangGraph
        self.graph = self.build_graph()
        
        # Get dependency graph (may be None for static mode)
        dependency_graph = None
        if self.mode == GraphMode.DYNAMIC:
            if hasattr(self.conductor, 'dependency_graph'):
                dependency_graph = self.conductor.dependency_graph
            else:
                # Initialize if not exists
                from .dynamic_dependency_graph import DynamicDependencyGraph
                dependency_graph = DynamicDependencyGraph()
        
        # Get roadmap/todo
        roadmap = getattr(self.conductor, 'roadmap', None) or getattr(self.conductor, 'todo', None)
        
        # Initial state
        initial_state: JottyLangGraphState = {
            "conductor": self.conductor,
            "dependency_graph": dependency_graph,
            "roadmap": roadmap,
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
        
        # Aggregate output from all agent results
        agent_results = final_state["agent_results"]
        aggregated_output = "\n\n".join([
            f"## {name}\n{res.output if hasattr(res, 'output') else str(res)}"
            for name, res in agent_results.items()
        ])
        
        return {
            "success": not final_state["error"],
            "results": agent_results,
            "aggregated_output": aggregated_output,
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
        # Initialize Jotty's systems (if method exists)
        # Note: Conductor may initialize episode in run() method
        self.graph = self.build_graph()
        
        # Get dependency graph
        dependency_graph = None
        if self.mode == GraphMode.DYNAMIC:
            dependency_graph = getattr(self.conductor, 'dependency_graph', None)
            if not dependency_graph:
                from .dynamic_dependency_graph import DynamicDependencyGraph
                dependency_graph = DynamicDependencyGraph()
        
        roadmap = getattr(self.conductor, 'roadmap', None) or getattr(self.conductor, 'todo', None)
        
        initial_state: JottyLangGraphState = {
            "conductor": self.conductor,
            "dependency_graph": dependency_graph,
            "roadmap": roadmap,
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
        
        async for event in self.graph.astream(initial_state):
            yield event

"""
Static LangGraph Definition
===========================

Define static LangGraph workflows with explicit agent ordering.
"""

from langgraph.graph import StateGraph, END, START
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from ..foundation.agent_config import AgentConfig
import logging
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class StaticLangGraphDefinition:
    """
    Define a static LangGraph with explicit nodes and edges.
    
    Usage:
        graph = StaticLangGraphDefinition(
            name="Research Workflow",
            agent_order=["ResearchAgent", "AnalyzeAgent", "ReportAgent"]
        )
    """
    
    name: str
    description: str = ""
    agent_order: List[str] = field(default_factory=list)
    conditional_edges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate agent order."""
        if not self.agent_order:
            raise ValueError("agent_order cannot be empty")
        
        # Check for duplicates
        if len(self.agent_order) != len(set(self.agent_order)):
            raise ValueError("agent_order contains duplicate agents")
    
    def build_langgraph(
        self,
        conductor: Any,
        state_schema: Any
    ) -> StateGraph:
        """
        Build LangGraph state machine from static definition.
        
        Args:
            conductor: Conductor instance for agent execution
            state_schema: TypedDict schema for LangGraph state
        
        Returns:
            Compiled LangGraph
        """
        graph = StateGraph(state_schema)
        
        # Get agent configs from conductor
        # Conductor stores actors as dict: self.actors[name] = ActorConfig
        actors_dict = conductor.actors
        if isinstance(actors_dict, dict):
            agents_list = list(actors_dict.values())
        else:
            agents_list = actors_dict
        
        agent_map = {}
        for agent in agents_list:
            if isinstance(agent, AgentConfig):
                agent_map[agent.name] = agent
            elif hasattr(agent, 'name'):
                agent_map[agent.name] = agent
        
        # Validate all agents exist
        missing = [name for name in self.agent_order if name not in agent_map]
        if missing:
            raise ValueError(f"Agents not found: {missing}")
        
        # Add nodes for each agent in order
        for agent_name in self.agent_order:
            agent = agent_map[agent_name]
            
            def make_node_func(a, c):
                async def node_func(state):
                    return await self._execute_agent_node(state, a, c)
                return node_func
            
            graph.add_node(
                agent_name,
                make_node_func(agent, conductor)
            )
        
        # Add sequential edges (each agent → next agent)
        for i in range(len(self.agent_order) - 1):
            from_agent = self.agent_order[i]
            to_agent = self.agent_order[i + 1]
            graph.add_edge(from_agent, to_agent)
        
        # Add conditional edges if specified
        for from_agent, cond_config in self.conditional_edges.items():
            if from_agent not in self.agent_order:
                raise ValueError(f"Conditional edge from unknown agent: {from_agent}")
            
            condition_func = cond_config.get("condition")
            mapping = cond_config.get("mapping", {})
            
            if condition_func and mapping:
                graph.add_conditional_edges(
                    from_agent,
                    condition_func,
                    mapping
                )
        
        # Set entry point (first agent)
        graph.set_entry_point(self.agent_order[0])
        
        # Set exit point (last agent → END)
        graph.add_edge(self.agent_order[-1], END)
        
        return graph.compile()
    
    async def _execute_agent_node(
        self,
        state: Dict[str, Any],
        agent: AgentConfig,
        conductor: Any
    ) -> Dict[str, Any]:
        """
        Execute agent node.
        
        Uses conductor's execution engine (JottyCore) with Architect/Auditor.
        """
        # Resolve parameters - use simple approach for LangGraph
        # Extract from context and previous results
        params = {}
        
        # Get agent signature to know what parameters it needs
        agent_name = agent.name
        if hasattr(conductor, 'actor_signatures') and agent_name in conductor.actor_signatures:
            sig = conductor.actor_signatures[agent_name]
            if isinstance(sig, dict):
                for param_name in sig.keys():
                    # Try to get from context first
                    context = state.get("context", {})
                    if param_name in context:
                        params[param_name] = context[param_name]
                    # Then try previous results
                    elif state.get("agent_results"):
                        for prev_name, prev_result in state["agent_results"].items():
                            if hasattr(prev_result, 'output') and hasattr(prev_result.output, param_name):
                                params[param_name] = getattr(prev_result.output, param_name)
                                break
                            elif isinstance(prev_result, dict) and param_name in prev_result:
                                params[param_name] = prev_result[param_name]
                                break
        
        # If no signature, use context directly
        if not params:
            params = state.get("context", {}).copy()
        
        # Execute agent - use wrapped agent from conductor
        # Conductor wraps agents with JottyCore, stored in self.actors[name].agent
        agent_to_execute = agent.agent if hasattr(agent, 'agent') else agent
        
        # Check if it's a JottyCore-wrapped agent
        from .jotty_core import JottyCore
        import asyncio
        
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
        
        # Store result
        agent_results = state.get("agent_results", {}).copy()
        agent_results[agent.name] = result
        
        # Update learning (optional)
        q_predictor = getattr(conductor, 'q_predictor', None) or getattr(conductor, 'q_learner', None)
        if q_predictor:
            try:
                reward = self._calculate_reward(result)
                if hasattr(q_predictor, 'record_outcome'):
                    if asyncio.iscoroutinefunction(q_predictor.record_outcome):
                        await q_predictor.record_outcome(
                            state={
                                "agent": agent.name,
                                "completed_agents": list(state.get("completed_agents", set())),
                            },
                            action={"agent": agent.name},
                            reward=reward
                        )
                    else:
                        q_predictor.record_outcome(
                            state={
                                "agent": agent.name,
                                "completed_agents": list(state.get("completed_agents", set())),
                            },
                            action={"agent": agent.name},
                            reward=reward
                        )
            except Exception as e:
                import logging
                logging.debug(f"Q-learning update failed: {e}")
        
        # Emit stream event
        stream_events = state.get("stream_events", []).copy()
        stream_events.append({
            "type": "agent_complete",
            "agent": agent.name,
            "result": result,
            "timestamp": __import__("time").time()
        })
        
        return {
            "agent_results": agent_results,
            "completed_agents": state.get("completed_agents", set()) | {agent.name},
            "stream_events": stream_events,
            "iteration": state.get("iteration", 0) + 1,
        }
    
    def _calculate_reward(self, result: Any) -> float:
        """Calculate reward from agent result."""
        if hasattr(result, 'success') and result.success:
            return 1.0
        elif hasattr(result, 'validation_passed') and result.validation_passed:
            return 0.8
        else:
            return 0.2
    
    def add_conditional_edge(
        self,
        from_agent: str,
        condition_func: Callable,
        mapping: Dict[str, str]
    ):
        """
        Add conditional edge.
        
        Args:
            from_agent: Source agent name
            condition_func: Function that returns key from mapping
            mapping: Dict mapping condition results to target agents
        """
        if from_agent not in self.agent_order:
            raise ValueError(f"Agent '{from_agent}' not in agent_order")
        
        for target in mapping.values():
            if target not in self.agent_order and target != END:
                raise ValueError(f"Target agent '{target}' not in agent_order")
        
        self.conditional_edges[from_agent] = {
            "condition": condition_func,
            "mapping": mapping
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "agent_order": self.agent_order,
            "conditional_edges": {
                k: {
                    "mapping": v["mapping"],
                    # Note: condition_func cannot be serialized
                }
                for k, v in self.conditional_edges.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StaticLangGraphDefinition":
        """Deserialize from dictionary."""
        instance = cls(
            name=data["name"],
            description=data.get("description", ""),
            agent_order=data["agent_order"]
        )
        
        # Note: conditional_edges condition_func must be recreated
        for from_agent, config in data.get("conditional_edges", {}).items():
            # User must provide condition_func when loading
            pass
        
        return instance

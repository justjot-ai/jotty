"""
 AGENTIC FEEDBACK ROUTER
===========================

NO USER CONFIGURATION NEEDED!

Uses:
- Game Theory: Agents decide who can help
- Cooperation: Agents help each other automatically
- Shared Scratchpad: Agents see each other's work
- TODO System: Agents understand dependencies

The swarm ITSELF decides feedback routing based on:
1. Task dependencies (who depends on whom)
2. Output capabilities (who produces what)
3. Error patterns (what kind of error occurred)
4. Agent expertise (what each agent specializes in)
5. Current swarm state (what's been done, what's pending)
"""

import dspy
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
import json
from ..agents.feedback_channel import FeedbackMessage, FeedbackType
import logging

logger = logging.getLogger(__name__)


class AgenticFeedbackSignature(dspy.Signature):
    """Intelligently route feedback to the agent(s) who can best help."""
    
    error_message = dspy.InputField(desc="The error that occurred")
    failing_agent = dspy.InputField(desc="Name of the agent that failed")
    failing_agent_goal = dspy.InputField(desc="What the failing agent was trying to do")
    
    available_agents = dspy.InputField(desc="List of other agents with their capabilities")
    task_dependencies = dspy.InputField(desc="Who depends on whom (task graph)")
    swarm_state = dspy.InputField(desc="Current swarm state (completed, pending tasks)")
    shared_scratchpad = dspy.InputField(desc="What other agents have produced so far")
    
    target_agents = dspy.OutputField(
        desc=(
            "JSON array of agent names who should receive feedback. "
            "Example: [\"BusinessTermResolver\", \"ColumnFilterSelector\"]. "
            "Return [] if no agent should be notified."
        )
    )
    feedback_type = dspy.OutputField(desc="Type of feedback: 'error_correction', 'refinement', 'consultation'")
    reasoning = dspy.OutputField(desc="Why these agents were chosen")
    cooperation_strategy = dspy.OutputField(desc="How agents should cooperate to solve this")


class AgenticFeedbackRouter:
    """
    Routes feedback intelligently using game theory and swarm coordination.
    
    NO hardcoding! Agents decide among themselves who should help.
    """
    
    def __init__(self, lm: Any) -> None:
        self.lm = lm
        self.router = dspy.ChainOfThought(AgenticFeedbackSignature)
        self.routing_history = []  # Learn from past routing decisions
    
    async def route_feedback(
        self,
        error_message: str,
        failing_actor_name: str,
        failing_actor_goal: str,
        available_actors: Dict[str, Dict],  # {name: {capabilities, provides, dependencies}}
        task_dependencies: Dict[str, List[str]],  # {actor: [depends_on, ...]}
        swarm_state: Dict,  # Current TODO, completed tasks, etc.
        shared_scratchpad: Dict[str, any],  # Actor outputs so far
    ) -> List[FeedbackMessage]:
        """
        Route feedback using AGENTIC decision-making.
        
        NO user configuration needed!
        """
        logger.info(f" Agentic feedback routing for {failing_actor_name}...")
        
        # Build context for the routing agent
        available_agents_desc = self._describe_agents(available_actors)
        dependencies_desc = self._describe_dependencies(task_dependencies)
        state_desc = self._describe_swarm_state(swarm_state)
        scratchpad_desc = self._describe_scratchpad(shared_scratchpad)
        
        # Add learning from history
        history_context = self._get_routing_history_context()
        
        # Call the agentic router
        with dspy.context(lm=self.lm):
            result = self.router(
                error_message=error_message,
                failing_agent=failing_actor_name,
                failing_agent_goal=failing_actor_goal,
                available_agents=available_agents_desc + "\n\n" + history_context,
                task_dependencies=dependencies_desc,
                swarm_state=state_desc,
                shared_scratchpad=scratchpad_desc
            )
        
        # Parse target agents
        target_agents = self._parse_target_agents_json(result.target_agents)
        
        logger.info(f" Agentic routing decision:")
        logger.info(f"   Target agents: {target_agents}")
        logger.info(f"   Feedback type: {result.feedback_type}")
        logger.info(f"   Reasoning: {result.reasoning}")
        logger.info(f"   Cooperation strategy: {result.cooperation_strategy}")
        
        # Create feedback messages
        feedback_messages = []
        feedback_type_enum = self._parse_feedback_type(result.feedback_type)
        
        for target_agent in target_agents:
            if target_agent in available_actors:
                message = FeedbackMessage(
                    from_actor=failing_actor_name,
                    to_actor=target_agent,
                    content=f"Error: {error_message}\n\nCooperation Strategy: {result.cooperation_strategy}",
                    feedback_type=feedback_type_enum,
                    context={
                        'reasoning': result.reasoning,
                        'cooperation_strategy': result.cooperation_strategy,
                        'failing_actor_goal': failing_actor_goal,
                    }
                )
                feedback_messages.append(message)
        
        # Record this decision for learning
        self.routing_history.append({
            'failing_actor': failing_actor_name,
            'error': error_message,
            'targets': target_agents,
            'reasoning': result.reasoning,
        })
        
        return feedback_messages
    
    def _describe_agents(self, available_actors: Dict[str, Dict]) -> str:
        """Describe available agents and their capabilities."""
        desc = "## Available Agents\n\n"
        for name, info in available_actors.items():
            desc += f"### {name}\n"
            if 'provides' in info and info['provides']:
                desc += f"- Provides: {', '.join(info['provides'])}\n"
            if 'dependencies' in info and info['dependencies']:
                desc += f"- Depends on: {', '.join(info['dependencies'])}\n"
            if 'goal' in info:
                desc += f"- Goal: {info['goal']}\n"
            desc += "\n"
        return desc
    
    def _describe_dependencies(self, task_dependencies: Dict[str, List[str]]) -> str:
        """Describe task dependency graph."""
        desc = "## Task Dependencies\n\n"
        for actor, deps in task_dependencies.items():
            if deps:
                desc += f"- {actor} depends on: {', '.join(deps)}\n"
        return desc
    
    def _describe_swarm_state(self, swarm_state: Dict) -> str:
        """Describe current swarm state."""
        desc = "## Swarm State\n\n"
        if 'completed_tasks' in swarm_state:
            desc += f"Completed: {', '.join(swarm_state['completed_tasks'])}\n"
        if 'pending_tasks' in swarm_state:
            desc += f"Pending: {', '.join(swarm_state['pending_tasks'])}\n"
        if 'failed_tasks' in swarm_state:
            desc += f"Failed: {', '.join(swarm_state['failed_tasks'])}\n"
        return desc
    
    def _describe_scratchpad(self, shared_scratchpad: Dict[str, any]) -> str:
        """Describe shared scratchpad (what agents have produced)."""
        desc = "## Shared Scratchpad\n\n"
        for actor, output in shared_scratchpad.items():
            desc += f"### {actor} produced:\n"
            # Show preview of output
            output_str = str(output)
            if len(output_str) > 200:
                desc += f"{output_str[:200]}...\n\n"
            else:
                desc += f"{output_str}\n\n"
        return desc
    
    def _get_routing_history_context(self) -> str:
        """Get context from past routing decisions."""
        if not self.routing_history:
            return ""
        
        recent = self.routing_history[-5:]  # Last 5 decisions
        desc = "## Past Routing Decisions (Learn from these)\n\n"
        for i, decision in enumerate(recent, 1):
            desc += f"{i}. When {decision['failing_actor']} had error '{decision['error'][:100]}', "
            desc += f"we routed to {', '.join(decision['targets'])} because: {decision['reasoning'][:200]}\n\n"
        return desc
    
    def _parse_target_agents_json(self, target_agents_json: str) -> List[str]:
        """
        Parse agent names from LLM output.
        
        STRICT POLICY: no regex/fuzzy parsing. The routing agent must return JSON.
        """
        try:
            parsed = json.loads(target_agents_json)
            if isinstance(parsed, list):
                # Keep only non-empty strings, preserve order, de-dupe
                seen = set()
                out: List[str] = []
                for item in parsed:
                    if isinstance(item, str):
                        name = item.strip()
                        if name and name not in seen:
                            out.append(name)
                            seen.add(name)
                return out
            return []
        except Exception:
            logger.warning(" AgenticFeedbackRouter: target_agents was not valid JSON; treating as no targets.")
            return []
    
    def _parse_feedback_type(self, feedback_type_str: str) -> FeedbackType:
        """Parse feedback type from LLM output."""
        ft_lower = feedback_type_str.lower()
        if 'correction' in ft_lower or 'error' in ft_lower:
            return FeedbackType.ERROR_CORRECTION
        elif 'refinement' in ft_lower or 'improve' in ft_lower:
            return FeedbackType.REFINEMENT
        elif 'consultation' in ft_lower or 'consult' in ft_lower:
            return FeedbackType.CONSULTATION
        else:
            return FeedbackType.ERROR_CORRECTION  # Default


"""
Chat Orchestrator

Handles agent selection and routing for chat interactions.
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """
    Orchestrates chat interactions by selecting and routing to appropriate agents.
    """
    
    def __init__(
        self,
        conductor: Any,  # Conductor instance
        agent_id: Optional[str] = None,
        mode: str = "dynamic"
    ):
        """
        Initialize chat orchestrator.
        
        Args:
            conductor: Jotty Conductor instance
            agent_id: Specific agent ID for single-agent chat (optional)
            mode: Orchestration mode ("static" or "dynamic")
        """
        self.conductor = conductor
        self.agent_id = agent_id
        self.mode = mode
        
        # Validate mode
        if mode not in ["static", "dynamic"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'static' or 'dynamic'")
    
    def select_agent(
        self,
        message: str,
        history: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Select agent for handling the message.
        
        Args:
            message: User message
            history: Conversation history
            context: Additional context
            
        Returns:
            Agent ID or name
        """
        # Single-agent chat
        if self.agent_id:
            logger.debug(f"Using single agent: {self.agent_id}")
            return self.agent_id
        
        # Multi-agent routing
        if self.mode == "static":
            # Use first available agent
            if hasattr(self.conductor, 'actors') and self.conductor.actors:
                agent_name = self.conductor.actors[0].name if hasattr(self.conductor.actors[0], 'name') else str(self.conductor.actors[0])
                logger.debug(f"Static mode: Using first agent: {agent_name}")
                return agent_name
        
        # Dynamic routing (delegate to conductor)
        if hasattr(self.conductor, 'select_agent'):
            agent = self.conductor.select_agent(message, history, context)
            logger.debug(f"Dynamic mode: Selected agent: {agent}")
            return agent
        
        # Fallback: use conductor's agent selection logic
        if hasattr(self.conductor, 'actors') and self.conductor.actors:
            # Use Q-predictor if available for intelligent routing
            if hasattr(self.conductor, 'q_predictor') and self.conductor.q_predictor:
                agent = self._select_with_q_predictor(message, history, context)
                if agent:
                    return agent
            
            # Default: first agent
            agent_name = self.conductor.actors[0].name if hasattr(self.conductor.actors[0], 'name') else str(self.conductor.actors[0])
            logger.debug(f"Fallback: Using first agent: {agent_name}")
            return agent_name
        
        raise RuntimeError("No agents available for chat")
    
    def _select_with_q_predictor(
        self,
        message: str,
        history: Optional[List[Any]],
        context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Select agent using Q-predictor."""
        if not hasattr(self.conductor, 'q_predictor') or not self.conductor.q_predictor:
            return None
        
        # Build state description
        state = {
            "message": message,
            "history_length": len(history) if history else 0,
            "context_keys": list(context.keys()) if context else []
        }
        
        # Evaluate each agent
        best_agent = None
        best_q_value = -1.0
        
        for actor in self.conductor.actors:
            agent_name = actor.name if hasattr(actor, 'name') else str(actor)
            action = {
                "agent": agent_name,
                "type": "chat"
            }
            
            try:
                q_value, confidence, _ = self.conductor.q_predictor.predict_q_value(
                    state=state,
                    action=action,
                    goal=message
                )
                
                if q_value and q_value > best_q_value:
                    best_q_value = q_value
                    best_agent = agent_name
            except Exception as e:
                logger.warning(f"Q-prediction failed for {agent_name}: {e}")
        
        return best_agent
    
    def prepare_agent_context(
        self,
        message: str,
        history: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare context for agent execution.
        
        Args:
            message: User message
            history: Conversation history
            context: Additional context
            
        Returns:
            Context dictionary for agent
        """
        agent_context = {
            "query": message,
            "user_message": message,
            **(context or {})
        }
        
        # Add conversation history
        if history:
            agent_context["conversation_history"] = self._format_history(history)
            agent_context["messages"] = [
                msg.to_dict() if hasattr(msg, 'to_dict') else str(msg)
                for msg in history
            ]
        
        return agent_context
    
    def _format_history(self, history: List[Any]) -> str:
        """Format conversation history as string."""
        lines = []
        for msg in history:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                lines.append(f"{msg.role}: {msg.content}")
            else:
                lines.append(str(msg))
        return "\n".join(lines)

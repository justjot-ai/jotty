"""
Jotty Unified Execution Mode
=============================

Comprehensive unified interface for all agent interactions.

Features:
- Style: "chat" (conversational) or "workflow" (task-oriented)
- Execution: "sync" (immediate) or "async" (queue-based)
- Learning: All modes get Q-learning, TD(λ), and predictive MARL
- Memory: All modes get hierarchical memory and consolidation
- Queue: Optional but seamlessly integrated
"""

from enum import Enum
from typing import Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass
import time
import logging

from .langgraph_orchestrator import LangGraphOrchestrator, GraphMode
from .conductor import Conductor

logger = logging.getLogger(__name__)


class ExecutionModeEnum(Enum):
    """Top-level execution modes for Jotty."""
    WORKFLOW = "workflow"  # Task-oriented (goal → result)
    CHAT = "chat"          # Conversational (message → response)


@dataclass
class ChatMessage:
    """Structured chat message."""
    role: str  # user, assistant, system, tool
    content: str
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }


class ExecutionMode:
    """
    Unified comprehensive execution mode for all agent interactions.
    
    Every execution gets access to:
    - ✅ Learning: Q-learning, TD(λ), predictive MARL (via Conductor)
    - ✅ Memory: Hierarchical memory, consolidation (via Conductor)
    - ✅ Queue: Optional but seamlessly integrated
    - ✅ All Jotty capabilities: Context management, data registry, etc.
    
    Supports:
    - Style: "chat" (conversational) or "workflow" (task-oriented)
    - Execution: "sync" (immediate) or "async" (queue-based)
    
    Usage:
        # Chat style (synchronous) - gets learning & memory
        chat = ExecutionMode(conductor, style="chat", execution="sync")
        async for event in chat.stream(message="...", history=[...]):
            ...
        
        # Workflow style (synchronous) - gets learning & memory
        workflow = ExecutionMode(conductor, style="workflow", execution="sync")
        result = await workflow.execute(goal="...", context={...})
        
        # Workflow style (asynchronous) - gets learning & memory + queue
        workflow = ExecutionMode(
            conductor, 
            style="workflow", 
            execution="async",
            queue=SQLiteTaskQueue(...)
        )
        task_id = await workflow.enqueue_task(goal="...")
        await workflow.process_queue()
    """
    
    def __init__(
        self,
        conductor: Conductor,
        style: str = "workflow",  # "chat" or "workflow"
        execution: str = "sync",  # "sync" or "async"
        queue: Optional[Any] = None,  # TaskQueue instance, required for async
        mode: str = "dynamic",  # "static" or "dynamic" (graph mode)
        agent_order: Optional[List[str]] = None,
        agent_id: Optional[str] = None,  # For single-agent chat
        # Async-specific parameters
        max_concurrent: int = 3,
        poll_interval: float = 1.0,
        # Learning and memory are automatically enabled via Conductor
    ):
        """
        Initialize unified execution mode.
        
        Args:
            conductor: Jotty Conductor instance (provides learning, memory, etc.)
            style: "chat" (conversational) or "workflow" (task-oriented)
            execution: "sync" (immediate) or "async" (queue-based)
            queue: TaskQueue instance (required for async execution)
            mode: "static" (predefined order) or "dynamic" (adaptive routing)
            agent_order: Required for static mode
            agent_id: Specific agent for single-agent chat (chat style only)
            max_concurrent: Maximum concurrent tasks (async only)
            poll_interval: Polling interval in seconds (async only)
        
        Note:
            Learning and memory capabilities are automatically available via Conductor.
            All executions benefit from:
            - Q-learning and TD(λ) for value estimation
            - Hierarchical memory for context retention
            - Predictive MARL for multi-agent coordination
            - Data registry for agentic data discovery
        """
        self.conductor = conductor
        self.style = style
        self.execution = execution
        self.mode = mode
        self.agent_id = agent_id
        
        # Validate style
        if style not in ["chat", "workflow"]:
            raise ValueError(f"Invalid style: {style}. Must be 'chat' or 'workflow'")
        
        # Validate execution
        if execution not in ["sync", "async"]:
            raise ValueError(f"Invalid execution: {execution}. Must be 'sync' or 'async'")
        
        # Validate async requirements
        if execution == "async":
            if queue is None:
                raise ValueError("queue is required for async execution")
            if style == "chat":
                raise ValueError("Chat style doesn't support async execution")
            # Configure Conductor with queue
            if not hasattr(conductor, 'task_queue') or conductor.task_queue is None:
                conductor.task_queue = queue
        
        self.queue = queue
        
        # Setup orchestrator
        if style == "chat" and agent_id:
            # Single-agent chat
            graph_mode = GraphMode.STATIC
            agent_order = [agent_id]
        elif mode == "static":
            graph_mode = GraphMode.STATIC
            if agent_order is None:
                raise ValueError("agent_order is required for static mode")
        else:
            graph_mode = GraphMode.DYNAMIC
            agent_order = None
        
        self.orchestrator = LangGraphOrchestrator(
            conductor=conductor,
            mode=graph_mode,
            agent_order=agent_order
        )
        
        # Log capabilities
        logger.info(
            f"✅ ExecutionMode initialized: style={style}, execution={execution}, "
            f"learning={'enabled' if hasattr(conductor, 'q_predictor') else 'via Conductor'}, "
            f"memory={'enabled' if hasattr(conductor, 'memory') else 'via Conductor'}, "
            f"queue={'enabled' if queue else 'disabled'}"
        )
    
    # ========== Capability Access ==========
    
    @property
    def learning_enabled(self) -> bool:
        """Check if learning is enabled (via Conductor)"""
        return (
            hasattr(self.conductor, 'q_predictor') and self.conductor.q_predictor is not None
        ) or (
            hasattr(self.conductor, 'q_learner') and self.conductor.q_learner is not None
        )
    
    @property
    def memory_enabled(self) -> bool:
        """Check if memory is enabled (via Conductor)"""
        return (
            hasattr(self.conductor, 'memory') and self.conductor.memory is not None
        ) or (
            hasattr(self.conductor, 'shared_memory') and self.conductor.shared_memory is not None
        )
    
    @property
    def queue_enabled(self) -> bool:
        """Check if queue is enabled for this ExecutionMode instance"""
        # Queue is only enabled if:
        # 1. We're in async mode AND
        # 2. Queue is configured
        if self.execution != "async":
            return False
        return self.queue is not None or (
            hasattr(self.conductor, 'task_queue') and self.conductor.task_queue is not None
        )
    
    def get_memory_summary(self) -> Optional[Dict[str, Any]]:
        """Get memory summary (if available via Conductor)"""
        if hasattr(self.conductor, 'memory') and self.conductor.memory:
            try:
                return {
                    "enabled": True,
                    "type": type(self.conductor.memory).__name__,
                    "summary": "Memory available via Conductor"
                }
            except:
                pass
        return {"enabled": False}
    
    def get_learning_summary(self) -> Optional[Dict[str, Any]]:
        """Get learning summary (if available via Conductor)"""
        learning_info = {"enabled": False, "components": []}
        
        if hasattr(self.conductor, 'q_predictor') and self.conductor.q_predictor:
            learning_info["enabled"] = True
            learning_info["components"].append("Q-learning")
        
        if hasattr(self.conductor, 'q_learner') and self.conductor.q_learner:
            learning_info["enabled"] = True
            learning_info["components"].append("TD(λ)")
        
        return learning_info
    
    # ========== Synchronous Execution ==========
    
    async def execute(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[ChatMessage]] = None,  # For chat style
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Execute synchronously (immediate execution).
        
        All executions benefit from:
        - Learning: Q-values, TD(λ) updates, predictive MARL
        - Memory: Context retention, consolidation
        - Data registry: Agentic data discovery
        
        For workflow style: Returns workflow result
        For chat style: Returns chat response
        """
        if self.execution != "sync":
            raise ValueError("execute() requires sync execution mode. Use enqueue_task() for async.")
        
        # Prepare context based on style
        if self.style == "chat":
            history = history or []
            conversation_context = self._format_history(history)
            context = {
                "messages": [msg.to_dict() for msg in history],
                "conversation_history": conversation_context,
                "user_message": goal,
                **(context or {})
            }
        else:
            context = context or {}
        
        # Execute via orchestrator (learning and memory happen automatically via Conductor)
        result = await self.orchestrator.run(
            goal=goal,
            context=context,
            max_iterations=max_iterations
        )
        
        # Transform result based on style
        if self.style == "chat":
            return self._transform_to_chat_response(result)
        else:
            if hasattr(result, 'to_dict'):
                return result.to_dict()
            return {
                "success": True,
                "final_output": str(result),
                "actor_outputs": getattr(result, 'actor_outputs', {}),
                "metadata": getattr(result, 'metadata', {}),
                "learning": self.get_learning_summary(),
                "memory": self.get_memory_summary()
            }
    
    async def stream(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[ChatMessage]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream execution events.
        
        All executions benefit from learning and memory (via Conductor).
        
        For workflow style: Raw workflow events
        For chat style: Chat-specific events (reasoning, tool calls, text chunks)
        """
        if self.execution != "sync":
            raise ValueError("stream() requires sync execution mode")
        
        # Prepare context based on style
        if self.style == "chat":
            history = history or []
            conversation_context = self._format_history(history)
            context = {
                "messages": [msg.to_dict() for msg in history],
                "conversation_history": conversation_context,
                "user_message": goal,
                **(context or {})
            }
        else:
            context = context or {}
        
        # Stream via orchestrator (learning and memory happen automatically)
        async for event in self.orchestrator.run_stream(goal=goal, context=context):
            if self.style == "chat":
                # Transform to chat events
                for chat_event in self._transform_to_chat_events(event):
                    yield chat_event
            else:
                # Workflow events as-is
                yield event
    
    # ========== Asynchronous Execution ==========
    
    async def enqueue_task(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        priority: int = 3,
        **kwargs
    ) -> str:
        """
        Enqueue task for asynchronous execution (workflow style only).
        
        Queued tasks also benefit from learning and memory when executed.
        
        Use for:
        - Background jobs
        - Long-running tasks
        - Batch processing
        """
        if self.execution != "async":
            raise ValueError("enqueue_task() requires async execution mode. Use execute() for sync.")
        
        if self.style == "chat":
            raise ValueError("Chat style doesn't support async execution")
        
        # Use Conductor's existing enqueue_goal method
        task_id = await self.conductor.enqueue_goal(
            goal=goal,
            priority=priority,
            **kwargs
        )
        
        if task_id is None:
            raise RuntimeError("Failed to enqueue task. Ensure Conductor has task_queue configured.")
        
        logger.info(f"📥 Enqueued task: {task_id} (will benefit from learning & memory when executed)")
        return task_id
    
    async def process_queue(
        self,
        max_tasks: Optional[int] = None,
        max_concurrent: int = 3
    ):
        """
        Start processing queued tasks (async mode, workflow style only).
        
        All processed tasks benefit from:
        - Learning: Q-learning, TD(λ) updates
        - Memory: Context retention, consolidation
        - Data registry: Agentic data discovery
        
        This uses Conductor's process_queue() method which:
        - Polls queue for pending tasks
        - Executes tasks via Conductor.run() (with learning & memory)
        - Manages concurrency
        - Handles retries and errors
        """
        if self.execution != "async":
            raise ValueError("process_queue() requires async execution mode")
        
        if self.style == "chat":
            raise ValueError("Chat style doesn't support async execution")
        
        logger.info(
            f"🚀 Starting queue processing (learning={'enabled' if self.learning_enabled else 'via Conductor'}, "
            f"memory={'enabled' if self.memory_enabled else 'via Conductor'})"
        )
        
        await self.conductor.process_queue(
            max_tasks=max_tasks,
            max_concurrent=max_concurrent
        )
    
    # ========== Helper Methods ==========
    
    def _format_history(self, history: List[ChatMessage]) -> str:
        """Format message history for conversational context."""
        return "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in history
        ])
    
    def _transform_to_chat_events(
        self,
        event: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Transform workflow events to chat-specific events."""
        chat_events = []
        
        # Only process agent_complete events
        if event.get("type") != "agent_complete":
            return chat_events
        
        result = event.get("result", {})
        agent_name = event.get("agent")
        
        # 1. Agent selection
        chat_events.append({
            "type": "agent_selected",
            "agent": agent_name,
            "timestamp": time.time()
        })
        
        # 2. Reasoning (if present)
        if result.get("reasoning"):
            chat_events.append({
                "type": "reasoning",
                "content": result["reasoning"],
                "timestamp": time.time()
            })
        
        # 3. Tool calls
        for tool_call in result.get("tool_calls", []):
            chat_events.append({
                "type": "tool_call",
                "tool": tool_call.get("name"),
                "args": tool_call.get("arguments"),
                "timestamp": time.time()
            })
        
        # 4. Tool results
        for tool_result in result.get("tool_results", []):
            chat_events.append({
                "type": "tool_result",
                "result": tool_result.get("result"),
                "timestamp": time.time()
            })
        
        # 5. Text response (chunk by sentence for streaming effect)
        response_text = result.get("response", "")
        if response_text:
            # Split into sentences for progressive rendering
            sentences = response_text.split(". ")
            for sentence in sentences:
                if sentence.strip():
                    chat_events.append({
                        "type": "text_chunk",
                        "content": sentence.strip() + ". ",
                        "timestamp": time.time()
                    })
        
        # 6. Done event
        chat_events.append({
            "type": "done",
            "final_message": ChatMessage(
                role="assistant",
                content=response_text,
                timestamp=time.time()
            ).to_dict(),
            "tool_calls": result.get("tool_calls", []),
            "tool_results": result.get("tool_results", []),
            "timestamp": time.time()
        })
        
        return chat_events
    
    def _transform_to_chat_response(self, result) -> Dict[str, Any]:
        """Transform workflow result to chat response."""
        if hasattr(result, 'final_output'):
            return {
                "message": result.final_output,
                "agent_outputs": getattr(result, 'actor_outputs', {}),
                "metadata": getattr(result, 'metadata', {}),
                "learning": self.get_learning_summary(),
                "memory": self.get_memory_summary()
            }
        return {"message": str(result)}


# ========== Backward-Compatible Wrappers ==========

class WorkflowMode:
    """
    Multi-agent task execution mode (backward-compatible wrapper).
    
    DEPRECATED: Use ExecutionMode(style="workflow") instead.
    This class is kept for backward compatibility.
    
    All executions benefit from learning and memory via Conductor.
    """
    
    def __init__(
        self,
        conductor: Conductor,
        mode: str = "dynamic",
        agent_order: Optional[List[str]] = None
    ):
        """Initialize workflow mode (delegates to ExecutionMode)."""
        logger.warning(
            "WorkflowMode is deprecated. Use ExecutionMode(style='workflow', execution='sync') instead. "
            "All executions get learning & memory capabilities automatically."
        )
        self._execution_mode = ExecutionMode(
            conductor=conductor,
            style="workflow",
            execution="sync",
            mode=mode,
            agent_order=agent_order
        )
        self.conductor = conductor
        self.orchestrator = self._execution_mode.orchestrator
    
    async def execute(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """Execute workflow (delegates to ExecutionMode, gets learning & memory)."""
        return await self._execution_mode.execute(
            goal=goal,
            context=context,
            max_iterations=max_iterations
        )
    
    async def execute_stream(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute workflow with streaming (delegates to ExecutionMode, gets learning & memory)."""
        async for event in self._execution_mode.stream(goal=goal, context=context):
            yield event


class ChatMode:
    """
    Conversational interface mode (backward-compatible wrapper).
    
    DEPRECATED: Use ExecutionMode(style="chat") instead.
    This class is kept for backward compatibility.
    
    All executions benefit from learning and memory via Conductor.
    """
    
    def __init__(
        self,
        conductor: Conductor,
        agent_id: Optional[str] = None,
        mode: str = "dynamic"
    ):
        """Initialize chat mode (delegates to ExecutionMode)."""
        logger.warning(
            "ChatMode is deprecated. Use ExecutionMode(style='chat', execution='sync') instead. "
            "All executions get learning & memory capabilities automatically."
        )
        self._execution_mode = ExecutionMode(
            conductor=conductor,
            style="chat",
            execution="sync",
            mode=mode,
            agent_id=agent_id
        )
        self.conductor = conductor
        self.agent_id = agent_id
        self.orchestrator = self._execution_mode.orchestrator
    
    async def stream(
        self,
        message: str,
        history: Optional[List[ChatMessage]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream chat response (delegates to ExecutionMode, gets learning & memory)."""
        async for event in self._execution_mode.stream(
            goal=message,
            history=history
        ):
            yield event
    
    def _format_history(self, history: List[ChatMessage]) -> str:
        """Format message history (delegates to ExecutionMode)."""
        return self._execution_mode._format_history(history)
    
    def _transform_to_chat_events(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transform events (delegates to ExecutionMode)."""
        return self._execution_mode._transform_to_chat_events(event)


# Convenience factory functions
def create_workflow(
    conductor: Conductor,
    mode: str = "dynamic",
    agent_order: Optional[List[str]] = None
) -> WorkflowMode:
    """
    Create workflow mode instance (backward-compatible).
    
    DEPRECATED: Use ExecutionMode(style="workflow") instead.
    All executions get learning & memory capabilities automatically.
    """
    return WorkflowMode(conductor, mode=mode, agent_order=agent_order)


def create_chat(
    conductor: Conductor,
    agent_id: Optional[str] = None,
    mode: str = "dynamic"
) -> ChatMode:
    """
    Create chat mode instance (backward-compatible).
    
    DEPRECATED: Use ExecutionMode(style="chat") instead.
    All executions get learning & memory capabilities automatically.
    """
    return ChatMode(conductor, agent_id=agent_id, mode=mode)

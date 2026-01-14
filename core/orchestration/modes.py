"""
Jotty Execution Modes
======================

Unified interface for different execution patterns.

Modes:
1. WorkflowMode - Multi-agent task execution (batch/background)
2. ChatMode - Conversational interface (interactive/streaming)
"""

from enum import Enum
from typing import Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass
import time

from .langgraph_orchestrator import LangGraphOrchestrator, GraphMode
from .conductor import Conductor
from ..agents.dspy_mcp_agent import DSPyMCPAgent


class ExecutionMode(Enum):
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


class WorkflowMode:
    """
    Multi-agent task execution mode.

    Use for:
    - Background jobs
    - Data pipelines
    - Report generation
    - Multi-step tasks

    Usage:
        workflow = WorkflowMode(
            conductor=conductor,
            mode="dynamic"  # or "static"
        )

        result = await workflow.execute(
            goal="Generate quarterly report",
            context={"quarter": "Q4", "year": 2026}
        )
    """

    def __init__(
        self,
        conductor: Conductor,
        mode: str = "dynamic",
        agent_order: Optional[List[str]] = None
    ):
        """
        Initialize workflow mode.

        Args:
            conductor: Jotty Conductor instance
            mode: "static" (predefined order) or "dynamic" (adaptive routing)
            agent_order: Required for static mode
        """
        self.conductor = conductor

        graph_mode = GraphMode.STATIC if mode == "static" else GraphMode.DYNAMIC

        self.orchestrator = LangGraphOrchestrator(
            conductor=conductor,
            mode=graph_mode,
            agent_order=agent_order
        )

    async def execute(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Execute workflow to achieve goal.

        Args:
            goal: Task description (e.g., "Generate report")
            context: Additional context data
            max_iterations: Max execution steps

        Returns:
            {
                "success": bool,
                "final_output": str,
                "actor_outputs": {...},
                "metadata": {...}
            }
        """
        return await self.orchestrator.run(
            goal=goal,
            context=context,
            max_iterations=max_iterations
        )

    async def execute_stream(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute workflow with streaming events.

        Yields:
            {"type": "agent_start", "agent": "..."}
            {"type": "agent_complete", "agent": "...", "result": {...}}
        """
        async for event in self.orchestrator.run_stream(goal=goal, context=context):
            yield event


class ChatMode:
    """
    Conversational interface mode.

    Use for:
    - Interactive chat
    - User conversations
    - Streaming responses
    - Tool-augmented chat

    Usage:
        chat = ChatMode(
            conductor=conductor,
            agent_id="research-assistant",  # or None for multi-agent
            mode="dynamic"
        )

        async for event in chat.stream(
            message="Create idea about transformers",
            history=[...]
        ):
            if event["type"] == "text_chunk":
                print(event["content"], end="", flush=True)
    """

    def __init__(
        self,
        conductor: Conductor,
        agent_id: Optional[str] = None,
        mode: str = "dynamic"
    ):
        """
        Initialize chat mode.

        Args:
            conductor: Jotty Conductor instance
            agent_id: Specific agent for single-agent chat, or None for multi-agent
            mode: "static" or "dynamic" (for multi-agent chat)
        """
        self.conductor = conductor
        self.agent_id = agent_id

        # Setup orchestrator
        if agent_id:
            # Single-agent chat
            graph_mode = GraphMode.STATIC
            agent_order = [agent_id]
        else:
            # Multi-agent chat
            graph_mode = GraphMode.STATIC if mode == "static" else GraphMode.DYNAMIC
            agent_order = None

        self.orchestrator = LangGraphOrchestrator(
            conductor=conductor,
            mode=graph_mode,
            agent_order=agent_order
        )

    async def stream(
        self,
        message: str,
        history: Optional[List[ChatMessage]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream chat response with chat-specific events.

        Args:
            message: User message
            history: Previous conversation messages

        Yields:
            {"type": "agent_selected", "agent": "research-assistant"}
            {"type": "reasoning", "content": "Let me search..."}
            {"type": "tool_call", "tool": "search", "args": {...}}
            {"type": "tool_result", "result": {...}}
            {"type": "text_chunk", "content": "Based on..."}
            {"type": "done", "final_message": {...}}
        """
        history = history or []

        # Format conversation history for DSPy agent
        conversation_context = self._format_history(history)

        # Prepare context with message history
        context = {
            "messages": [msg.to_dict() for msg in history],
            "conversation_history": conversation_context,
            "user_message": message
        }

        # Stream via orchestrator and transform events
        async for event in self.orchestrator.run_stream(
            goal=message,  # User message becomes the goal
            context=context
        ):
            # Transform generic workflow events to chat events
            for chat_event in self._transform_to_chat_events(event):
                yield chat_event

    def _transform_to_chat_events(
        self,
        event: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Transform LangGraph workflow events to chat-specific events.

        Workflow event:
            {"type": "agent_complete", "agent": "...", "result": {...}}

        Chat events:
            [
                {"type": "agent_selected", "agent": "..."},
                {"type": "reasoning", "content": "..."},
                {"type": "tool_call", "tool": "...", "args": {...}},
                {"type": "tool_result", "result": {...}},
                {"type": "text_chunk", "content": "..."},
                {"type": "done", "final_message": {...}}
            ]
        """
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

    def _format_history(self, history: List[ChatMessage]) -> str:
        """Format message history for DSPy agent."""
        return "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in history
        ])


# Convenience factory functions
def create_workflow(
    conductor: Conductor,
    mode: str = "dynamic",
    agent_order: Optional[List[str]] = None
) -> WorkflowMode:
    """
    Create workflow mode instance.

    Examples:
        # Dynamic workflow (adaptive routing)
        workflow = create_workflow(conductor, mode="dynamic")

        # Static workflow (predefined order)
        workflow = create_workflow(
            conductor,
            mode="static",
            agent_order=["Research", "Writer", "Editor"]
        )
    """
    return WorkflowMode(conductor, mode=mode, agent_order=agent_order)


def create_chat(
    conductor: Conductor,
    agent_id: Optional[str] = None,
    mode: str = "dynamic"
) -> ChatMode:
    """
    Create chat mode instance.

    Examples:
        # Single-agent chat
        chat = create_chat(conductor, agent_id="research-assistant")

        # Multi-agent chat with dynamic routing
        chat = create_chat(conductor, mode="dynamic")

        # Multi-agent chat with static order
        chat = create_chat(
            conductor,
            mode="static",
            agent_order=["Analyst", "Writer"]
        )
    """
    return ChatMode(conductor, agent_id=agent_id, mode=mode)

"""
FeedbackChannel: Inter-Agent Communication System

Enables agents to:
1. Ask questions to other agents
2. Request re-execution with hints
3. Provide error feedback
4. Suggest alternatives
5. Seek clarification

This is critical for intelligent multi-agent systems where agents
can consult each other rather than working in isolation.

Example:
    SQLGenerator has 15 tables but only needs 3-5:
    → Consults BusinessTermResolver: "Which tables are most relevant?"
    → Gets response: ["table1", "table2", "table3"]
    → Uses only those tables in SQL
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback messages between agents."""

    QUESTION = "question"  # Ask another agent for information
    ERROR = "error"  # Report an error to another agent
    SUGGESTION = "suggestion"  # Suggest an alternative approach
    REQUEST = "request"  # Request re-execution
    CLARIFICATION = "clarification"  # Ask for clarification
    RESPONSE = "response"  # Response to a question


@dataclass
class FeedbackMessage:
    """
    Message from one agent to another.

    Attributes:
        source_actor: Agent sending the message
        target_actor: Agent receiving the message
        feedback_type: Type of feedback
        content: The actual message content
        context: Additional context (data, parameters, etc.)
        timestamp: When message was created
        requires_response: Whether this needs a response
        priority: Message priority (1=high, 2=medium, 3=low)
        original_message_id: If this is a response, ID of original message
    """

    source_actor: str
    target_actor: str
    feedback_type: FeedbackType
    content: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    requires_response: bool = True
    priority: int = 1  # 1=high, 2=medium, 3=low
    message_id: str = field(default_factory=lambda: f"msg_{datetime.now().timestamp()}")
    original_message_id: Optional[str] = None


class FeedbackChannel:
    """
    Communication channel between agents.

    Manages message passing, history, and ensures agents can
    communicate effectively without tight coupling.

    Usage:
        channel = FeedbackChannel()

        # Agent A asks Agent B a question
        channel.send(FeedbackMessage(
            source_actor="SQLGenerator",
            target_actor="BusinessTermResolver",
            feedback_type=FeedbackType.QUESTION,
            content="Which data sources are most relevant for {target_concept}?",
            context={"current_tables": ["table1", "table2", ...]}
        ))

        # Agent B checks for messages
        messages = channel.get_for_actor("BusinessTermResolver")

        # Agent B responds
        channel.send(FeedbackMessage(
            source_actor="BusinessTermResolver",
            target_actor="SQLGenerator",
            feedback_type=FeedbackType.RESPONSE,
            content="Most relevant tables: table1, table2, table3",
            original_message_id=messages[0].message_id
        ))
    """

    def __init__(self) -> None:
        """Initialize the feedback channel."""
        self.messages: Dict[str, List[FeedbackMessage]] = defaultdict(list)
        self.message_history: List[FeedbackMessage] = []
        self.message_count = 0
        logger.info(" FeedbackChannel initialized - agents can now communicate!")

    def send(self, message: FeedbackMessage) -> str:
        """
        Send a feedback message to an actor.

        Args:
            message: The feedback message to send

        Returns:
            Message ID for tracking
        """
        self.messages[message.target_actor].append(message)
        self.message_history.append(message)
        self.message_count += 1

        logger.info(
            f" {message.source_actor} → {message.target_actor}: "
            f"{message.feedback_type.value} "
            f"(priority={message.priority}, id={message.message_id})"
        )
        logger.debug(f"   Content: {message.content}...")

        return message.message_id

    def get_for_actor(
        self, actor_name: str, clear: bool = True, priority_threshold: int = 3
    ) -> List[FeedbackMessage]:
        """
        Get messages for a specific actor.

        Args:
            actor_name: Name of the actor
            clear: Whether to clear messages after retrieval
            priority_threshold: Only get messages with priority <= this (1=high, 3=low)

        Returns:
            List of feedback messages for the actor
        """
        if actor_name not in self.messages:
            return []

        # Filter by priority
        messages = [msg for msg in self.messages[actor_name] if msg.priority <= priority_threshold]

        # Sort by priority (high first), then timestamp
        messages.sort(key=lambda m: (m.priority, m.timestamp))

        if clear:
            self.messages[actor_name] = [
                msg for msg in self.messages[actor_name] if msg.priority > priority_threshold
            ]

        if messages:
            logger.info(
                f" {actor_name} has {len(messages)} message(s) "
                f"(priority <={priority_threshold})"
            )

        return messages

    def has_feedback(self, actor_name: str, priority_threshold: int = 3) -> bool:
        """
        Check if an actor has pending messages.

        Args:
            actor_name: Name of the actor
            priority_threshold: Only check messages with priority <= this

        Returns:
            True if actor has pending messages
        """
        if actor_name not in self.messages:
            return False

        return any(msg.priority <= priority_threshold for msg in self.messages[actor_name])

    def get_conversation(self, actor1: str, actor2: str) -> List[FeedbackMessage]:
        """
        Get all messages between two actors.

        Args:
            actor1: First actor name
            actor2: Second actor name

        Returns:
            List of all messages between the two actors, sorted by time
        """
        conversation = [
            msg
            for msg in self.message_history
            if (msg.source_actor == actor1 and msg.target_actor == actor2)
            or (msg.source_actor == actor2 and msg.target_actor == actor1)
        ]
        conversation.sort(key=lambda m: m.timestamp)
        return conversation

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about message passing.

        Returns:
            Dictionary with stats
        """
        return {
            "total_messages": self.message_count,
            "pending_messages": sum(len(msgs) for msgs in self.messages.values()),
            "actors_with_pending": list(self.messages.keys()),
            "message_types": {
                ft.value: sum(1 for msg in self.message_history if msg.feedback_type == ft)
                for ft in FeedbackType
            },
        }

    # =========================================================================
    # MsgHub-inspired: Broadcast to multiple agents at once
    # =========================================================================

    def broadcast(
        self,
        source_actor: str,
        content: str,
        participants: Optional[List[str]] = None,
        feedback_type: FeedbackType = FeedbackType.RESPONSE,
        context: Optional[Dict[str, Any]] = None,
        priority: int = 2,
    ) -> List[str]:
        """
        Broadcast message to all participants (AgentScope MsgHub pattern).

        When an agent produces output that all other agents should see,
        use broadcast instead of N individual sends.

        DRY: Reuses existing send() for each target.

        Args:
            source_actor: Agent sending the broadcast
            content: Message content
            participants: Target agents (default: all known agents)
            feedback_type: Message type (default: RESPONSE)
            context: Additional context dict
            priority: Message priority (default: 2/medium)

        Returns:
            List of message IDs
        """
        # Default: broadcast to all agents that have ever received messages
        targets = participants or list(self.messages.keys())

        msg_ids = []
        for target in targets:
            if target == source_actor:
                continue  # Don't broadcast to self
            msg = FeedbackMessage(
                source_actor=source_actor,
                target_actor=target,
                feedback_type=feedback_type,
                content=content,
                context=context or {},
                requires_response=False,
                priority=priority,
            )
            msg_ids.append(self.send(msg))

        if msg_ids:
            logger.info(f" {source_actor} broadcast to {len(msg_ids)} agents")
        return msg_ids

    # =========================================================================
    # A2A-inspired: Request-reply with async await
    # =========================================================================

    async def request(
        self,
        source_actor: str,
        target_actor: str,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
        poll_interval: float = 0.1,
    ) -> Optional[FeedbackMessage]:
        """
        Send request and await response (AgentScope A2A pattern).

        Enables the original design intent: agents consulting each other.
        Example: SQLGenerator asks BusinessTermResolver for relevant tables.

        DRY: Reuses send() and get_for_actor(). No event system needed.
        KISS: Simple polling with timeout.

        Args:
            source_actor: Agent asking the question
            target_actor: Agent being asked
            content: The question/request
            context: Additional context
            timeout: Max wait time in seconds (default 30s)
            poll_interval: Poll frequency in seconds (default 0.1s)

        Returns:
            Response FeedbackMessage, or None on timeout
        """
        msg = FeedbackMessage(
            source_actor=source_actor,
            target_actor=target_actor,
            feedback_type=FeedbackType.QUESTION,
            content=content,
            context=context or {},
            requires_response=True,
            priority=1,
        )
        msg_id = self.send(msg)

        # Poll for response (KISS: no event/Future infrastructure needed)
        _loop = asyncio.get_running_loop()
        deadline = _loop.time() + timeout
        while _loop.time() < deadline:
            # Check for responses to our message
            pending = self.messages.get(source_actor, [])
            for i, response in enumerate(pending):
                if response.original_message_id == msg_id:
                    # Found our response — remove from queue and return
                    pending.pop(i)
                    logger.info(
                        f" {source_actor} got reply from {target_actor} "
                        f"({(_loop.time() - (deadline - timeout)):.1f}s)"
                    )
                    return response
            await asyncio.sleep(poll_interval)

        logger.warning(f"⏰ {source_actor} request to {target_actor} timed out ({timeout}s)")
        return None

    def clear_all(self) -> None:
        """Clear all pending messages (but keep history)."""
        self.messages.clear()
        logger.info(" FeedbackChannel: All pending messages cleared")

    def format_messages_for_agent(self, actor_name: str, messages: List[FeedbackMessage]) -> str:
        """
        Format messages as a string for injection into agent context.

        Args:
            actor_name: Name of the actor receiving the messages
            messages: List of messages to format

        Returns:
            Formatted string for injection
        """
        if not messages:
            return ""

        formatted = f"\n MESSAGES FOR {actor_name}:\n\n"

        for i, msg in enumerate(messages, 1):
            formatted += f"{i}. FROM {msg.source_actor} ({msg.feedback_type.value}):\n"
            formatted += f"   {msg.content}\n"

            if msg.context:
                formatted += f"   Context: {str(msg.context)}\n"

            if msg.requires_response:
                formatted += " Requires Response\n"

            formatted += "\n"

        formatted += "END MESSAGES\n"
        return formatted

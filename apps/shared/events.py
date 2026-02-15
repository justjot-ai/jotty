"""
Event Processing
================

Processes SDK events and updates chat interface accordingly.
"""

import asyncio
import os
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Deque, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from Jotty.sdk import SDKEvent, SDKEventType

from .interface import ChatInterface
from .models import Error, Message
from .state import ChatState


@dataclass
class ProcessedEvent:
    """Event after processing."""

    event: SDKEvent
    processed_at: datetime
    message: Optional[Message] = None
    state_change: Optional[ChatState] = None


class EventQueue:
    """Thread-safe event queue."""

    def __init__(self, maxlen: int = 1000):
        """
        Initialize event queue.

        Args:
            maxlen: Maximum queue size
        """
        self._queue: Deque[SDKEvent] = deque(maxlen=maxlen)
        self._lock = asyncio.Lock()

    async def put(self, event: SDKEvent) -> None:
        """Add event to queue."""
        async with self._lock:
            self._queue.append(event)

    async def get(self) -> Optional[SDKEvent]:
        """Get event from queue (FIFO)."""
        async with self._lock:
            if self._queue:
                return self._queue.popleft()
        return None

    async def get_all(self) -> list[SDKEvent]:
        """Get all events from queue."""
        async with self._lock:
            events = list(self._queue)
            self._queue.clear()
            return events

    def size(self) -> int:
        """Get queue size."""
        return len(self._queue)


class EventProcessor:
    """
    Processes SDK events and updates ChatInterface.

    Maps SDK events to:
    - State transitions
    - Message updates
    - Status displays
    """

    def __init__(self, chat_interface: ChatInterface):
        """
        Initialize event processor.

        Args:
            chat_interface: Chat interface to update
        """
        self.chat = chat_interface
        self._streaming_message: Optional[Message] = None

        # Event handlers
        self._handlers: Dict[SDKEventType, Callable] = {
            SDKEventType.START: self._handle_start,
            SDKEventType.COMPLETE: self._handle_complete,
            SDKEventType.ERROR: self._handle_error,
            SDKEventType.THINKING: self._handle_thinking,
            SDKEventType.PLANNING: self._handle_planning,
            SDKEventType.SKILL_START: self._handle_skill_start,
            SDKEventType.SKILL_PROGRESS: self._handle_skill_progress,
            SDKEventType.SKILL_COMPLETE: self._handle_skill_complete,
            SDKEventType.STREAM: self._handle_stream,
            SDKEventType.DELTA: self._handle_delta,
            SDKEventType.AGENT_START: self._handle_agent_start,
            SDKEventType.AGENT_COMPLETE: self._handle_agent_complete,
            SDKEventType.MEMORY_RECALL: self._handle_memory_recall,
            SDKEventType.MEMORY_STORE: self._handle_memory_store,
            SDKEventType.VOICE_STT_START: self._handle_voice_stt_start,
            SDKEventType.VOICE_STT_COMPLETE: self._handle_voice_stt_complete,
            SDKEventType.VOICE_TTS_START: self._handle_voice_tts_start,
            SDKEventType.VOICE_TTS_COMPLETE: self._handle_voice_tts_complete,
            SDKEventType.SWARM_AGENT_START: self._handle_swarm_agent_start,
            SDKEventType.SWARM_AGENT_COMPLETE: self._handle_swarm_agent_complete,
            SDKEventType.SWARM_COORDINATION: self._handle_swarm_coordination,
        }

    async def process_event(self, event: SDKEvent) -> ProcessedEvent:
        """
        Process single SDK event.

        Args:
            event: SDK event to process

        Returns:
            Processed event with updates
        """
        # Get handler for event type
        handler = self._handlers.get(event.type)
        if handler:
            await handler(event)

        return ProcessedEvent(
            event=event,
            processed_at=datetime.now(),
        )

    async def process_events(self, events: list[SDKEvent]) -> list[ProcessedEvent]:
        """
        Process multiple events in sequence.

        Args:
            events: List of SDK events

        Returns:
            List of processed events
        """
        processed = []
        for event in events:
            result = await self.process_event(event)
            processed.append(result)
        return processed

    # Event Handlers

    async def _handle_start(self, event: SDKEvent) -> None:
        """Handle START event."""
        self.chat.set_state(ChatState.THINKING)

    async def _handle_complete(self, event: SDKEvent) -> None:
        """Handle COMPLETE event."""
        # Finalize streaming message if any
        if self._streaming_message:
            self._streaming_message = None

        # Return to IDLE
        self.chat.set_state(ChatState.IDLE)

    async def _handle_error(self, event: SDKEvent) -> None:
        """Handle ERROR event."""
        error_msg = event.data.get("error", "Unknown error")
        error = Error(
            message=error_msg,
            error_type=event.data.get("error_type"),
            recoverable=event.data.get("recoverable", True),
        )
        self.chat.show_error(error)

        # Add error message to history
        message = Message(
            role="system",
            content=f"Error: {error_msg}",
            event_type=SDKEventType.ERROR,
        )
        self.chat.add_message(message)

    async def _handle_thinking(self, event: SDKEvent) -> None:
        """Handle THINKING event."""
        self.chat.set_state(ChatState.THINKING)

        # Add ephemeral thinking message
        message = Message(
            role="system",
            content="Thinking...",
            event_type=SDKEventType.THINKING,
            ephemeral=True,
        )
        self.chat.add_message(message)

    async def _handle_planning(self, event: SDKEvent) -> None:
        """Handle PLANNING event."""
        self.chat.set_state(ChatState.PLANNING)

        # Add ephemeral planning message
        plan = event.data.get("plan", "Creating plan...")
        message = Message(
            role="system",
            content=f"Planning: {plan}",
            event_type=SDKEventType.PLANNING,
            ephemeral=True,
        )
        self.chat.add_message(message)

    async def _handle_skill_start(self, event: SDKEvent) -> None:
        """Handle SKILL_START event."""
        skill_name = event.data.get("skill", "unknown")
        self.chat.set_state(ChatState.EXECUTING_SKILL, skill_name=skill_name)

        # Add ephemeral skill execution message
        message = Message(
            role="system",
            content=f"Running skill: {skill_name}",
            event_type=SDKEventType.SKILL_START,
            skill_name=skill_name,
            ephemeral=True,
        )
        self.chat.add_message(message)

    async def _handle_skill_progress(self, event: SDKEvent) -> None:
        """Handle SKILL_PROGRESS event."""
        progress = event.data.get("progress", 0.0)
        skill_name = event.data.get("skill", "unknown")

        self.chat.update_progress(progress, f"{skill_name}: {int(progress * 100)}%")

    async def _handle_skill_complete(self, event: SDKEvent) -> None:
        """Handle SKILL_COMPLETE event."""
        skill_name = event.data.get("skill", "unknown")

        # Add completion message
        message = Message(
            role="system",
            content=f"✓ Completed: {skill_name}",
            event_type=SDKEventType.SKILL_COMPLETE,
            skill_name=skill_name,
            ephemeral=True,
        )
        self.chat.add_message(message)

    async def _handle_stream(self, event: SDKEvent) -> None:
        """Handle STREAM event (text chunk)."""
        chunk = event.data.get("delta", "")

        # If no streaming message, create one
        if not self._streaming_message:
            self._streaming_message = Message(
                role="assistant",
                content=chunk,
                event_type=SDKEventType.STREAM,
            )
            self.chat.add_message(self._streaming_message)
            self.chat.set_state(ChatState.STREAMING)
        else:
            # Append to existing streaming message
            self.chat.update_streaming_message(self._streaming_message, chunk)

    async def _handle_delta(self, event: SDKEvent) -> None:
        """Handle DELTA event (incremental update)."""
        # Same as STREAM
        await self._handle_stream(event)

    async def _handle_agent_start(self, event: SDKEvent) -> None:
        """Handle AGENT_START event."""
        agent_name = event.data.get("agent", "unknown")
        self.chat.set_state(ChatState.EXECUTING_AGENT, agent_name=agent_name)

        message = Message(
            role="system",
            content=f"Agent started: {agent_name}",
            event_type=SDKEventType.AGENT_START,
            agent_name=agent_name,
            ephemeral=True,
        )
        self.chat.add_message(message)

    async def _handle_agent_complete(self, event: SDKEvent) -> None:
        """Handle AGENT_COMPLETE event."""
        agent_name = event.data.get("agent", "unknown")

        message = Message(
            role="system",
            content=f"✓ Agent completed: {agent_name}",
            event_type=SDKEventType.AGENT_COMPLETE,
            agent_name=agent_name,
            ephemeral=True,
        )
        self.chat.add_message(message)

    async def _handle_memory_recall(self, event: SDKEvent) -> None:
        """Handle MEMORY_RECALL event."""
        count = event.data.get("count", 0)

        message = Message(
            role="system",
            content=f"🧠 Recalled {count} relevant memories",
            event_type=SDKEventType.MEMORY_RECALL,
            ephemeral=True,
        )
        self.chat.add_message(message)

    async def _handle_memory_store(self, event: SDKEvent) -> None:
        """Handle MEMORY_STORE event."""
        message = Message(
            role="system",
            content="💾 Stored to memory",
            event_type=SDKEventType.MEMORY_STORE,
            ephemeral=True,
        )
        self.chat.add_message(message)

    async def _handle_voice_stt_start(self, event: SDKEvent) -> None:
        """Handle VOICE_STT_START event."""
        self.chat.set_state(ChatState.TRANSCRIBING)

        message = Message(
            role="system",
            content="🎤 Transcribing audio...",
            event_type=SDKEventType.VOICE_STT_START,
            ephemeral=True,
        )
        self.chat.add_message(message)

    async def _handle_voice_stt_complete(self, event: SDKEvent) -> None:
        """Handle VOICE_STT_COMPLETE event."""
        transcript = event.data.get("transcript", "")

        # Add user message with transcript
        message = Message(
            role="user",
            content=transcript,
            event_type=SDKEventType.VOICE_STT_COMPLETE,
        )
        self.chat.add_message(message)

    async def _handle_voice_tts_start(self, event: SDKEvent) -> None:
        """Handle VOICE_TTS_START event."""
        self.chat.set_state(ChatState.SYNTHESIZING)

        message = Message(
            role="system",
            content="🔊 Generating speech...",
            event_type=SDKEventType.VOICE_TTS_START,
            ephemeral=True,
        )
        self.chat.add_message(message)

    async def _handle_voice_tts_complete(self, event: SDKEvent) -> None:
        """Handle VOICE_TTS_COMPLETE event."""
        message = Message(
            role="system",
            content="✓ Speech ready",
            event_type=SDKEventType.VOICE_TTS_COMPLETE,
            ephemeral=True,
        )
        self.chat.add_message(message)

    async def _handle_swarm_agent_start(self, event: SDKEvent) -> None:
        """Handle SWARM_AGENT_START event."""
        agent_name = event.data.get("agent", "unknown")
        swarm_id = event.data.get("swarm_id", "default")

        self.chat.set_state(
            ChatState.COORDINATING_SWARM,
            agent_name=agent_name,
            swarm_id=swarm_id,
        )

        message = Message(
            role="system",
            content=f"🐝 Swarm agent started: {agent_name}",
            event_type=SDKEventType.SWARM_AGENT_START,
            agent_name=agent_name,
            swarm_id=swarm_id,
            ephemeral=True,
        )
        self.chat.add_message(message)

    async def _handle_swarm_agent_complete(self, event: SDKEvent) -> None:
        """Handle SWARM_AGENT_COMPLETE event."""
        agent_name = event.data.get("agent", "unknown")

        message = Message(
            role="system",
            content=f"✓ Swarm agent completed: {agent_name}",
            event_type=SDKEventType.SWARM_AGENT_COMPLETE,
            agent_name=agent_name,
            ephemeral=True,
        )
        self.chat.add_message(message)

    async def _handle_swarm_coordination(self, event: SDKEvent) -> None:
        """Handle SWARM_COORDINATION event."""
        action = event.data.get("action", "coordinating")

        message = Message(
            role="system",
            content=f"🔀 Swarm: {action}",
            event_type=SDKEventType.SWARM_COORDINATION,
            ephemeral=True,
        )
        self.chat.add_message(message)

"""
Chat State Management
=====================

Finite state machine for chat interface states.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional


class ChatState(Enum):
    """Chat interface states."""

    IDLE = "idle"  # Waiting for user input
    THINKING = "thinking"  # Agent is reasoning
    PLANNING = "planning"  # Creating execution plan
    EXECUTING_SKILL = "executing_skill"  # Running a skill
    EXECUTING_AGENT = "executing_agent"  # Agent executing
    COORDINATING_SWARM = "coordinating_swarm"  # Swarm coordination
    STREAMING = "streaming"  # Streaming response
    TRANSCRIBING = "transcribing"  # Voice STT
    SYNTHESIZING = "synthesizing"  # Voice TTS
    WAITING_INPUT = "waiting_input"  # Waiting for specific input
    VALIDATING = "validating"  # Validating response
    LEARNING = "learning"  # Learning update
    ERROR = "error"  # Error state


class StateTransition(Enum):
    """Valid state transitions."""

    # From IDLE
    START_THINKING = "start_thinking"
    START_PLANNING = "start_planning"
    START_SKILL = "start_skill"
    START_AGENT = "start_agent"
    START_SWARM = "start_swarm"
    START_STREAMING = "start_streaming"
    START_VOICE = "start_voice"

    # From processing states
    FINISH_THINKING = "finish_thinking"
    FINISH_PLANNING = "finish_planning"
    FINISH_SKILL = "finish_skill"
    FINISH_AGENT = "finish_agent"
    FINISH_SWARM = "finish_swarm"
    FINISH_STREAMING = "finish_streaming"
    FINISH_VOICE = "finish_voice"

    # Error handling
    ENCOUNTER_ERROR = "encounter_error"
    RECOVER_FROM_ERROR = "recover_from_error"

    # Return to idle
    RESET = "reset"


@dataclass
class StateContext:
    """Context data for current state."""

    state: ChatState
    entered_at: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)

    # Current execution details
    skill_name: Optional[str] = None
    agent_name: Optional[str] = None
    swarm_id: Optional[str] = None
    progress: float = 0.0
    step_current: int = 0
    step_total: int = 0

    def get_display_text(self) -> str:
        """Get human-readable state description."""
        text_map = {
            ChatState.IDLE: "Ready",
            ChatState.THINKING: "Thinking...",
            ChatState.PLANNING: "Planning...",
            ChatState.EXECUTING_SKILL: f"Running {self.skill_name or 'skill'}...",
            ChatState.EXECUTING_AGENT: f"Agent: {self.agent_name or 'executing'}...",
            ChatState.COORDINATING_SWARM: f"Swarm coordination...",
            ChatState.STREAMING: "Streaming response...",
            ChatState.TRANSCRIBING: "Transcribing audio...",
            ChatState.SYNTHESIZING: "Generating speech...",
            ChatState.WAITING_INPUT: "Waiting for input...",
            ChatState.VALIDATING: "Validating...",
            ChatState.LEARNING: "Learning...",
            ChatState.ERROR: "Error occurred",
        }
        return text_map.get(self.state, str(self.state.value))


class ChatStateMachine:
    """
    Finite state machine for chat interface.

    Manages state transitions and callbacks.
    """

    # Valid transitions (from_state -> [to_states])
    TRANSITIONS = {
        ChatState.IDLE: [
            ChatState.THINKING,
            ChatState.PLANNING,
            ChatState.EXECUTING_SKILL,
            ChatState.EXECUTING_AGENT,
            ChatState.COORDINATING_SWARM,
            ChatState.STREAMING,
            ChatState.TRANSCRIBING,
            ChatState.WAITING_INPUT,
            ChatState.ERROR,
        ],
        ChatState.THINKING: [
            ChatState.PLANNING,
            ChatState.EXECUTING_SKILL,
            ChatState.IDLE,
            ChatState.ERROR,
        ],
        ChatState.PLANNING: [
            ChatState.EXECUTING_SKILL,
            ChatState.EXECUTING_AGENT,
            ChatState.IDLE,
            ChatState.ERROR,
        ],
        ChatState.EXECUTING_SKILL: [
            ChatState.IDLE,
            ChatState.STREAMING,
            ChatState.VALIDATING,
            ChatState.ERROR,
        ],
        ChatState.EXECUTING_AGENT: [ChatState.IDLE, ChatState.STREAMING, ChatState.ERROR],
        ChatState.COORDINATING_SWARM: [ChatState.IDLE, ChatState.ERROR],
        ChatState.STREAMING: [ChatState.IDLE, ChatState.ERROR],
        ChatState.TRANSCRIBING: [ChatState.THINKING, ChatState.IDLE, ChatState.ERROR],
        ChatState.SYNTHESIZING: [ChatState.IDLE, ChatState.ERROR],
        ChatState.WAITING_INPUT: [ChatState.THINKING, ChatState.IDLE, ChatState.ERROR],
        ChatState.VALIDATING: [ChatState.IDLE, ChatState.LEARNING, ChatState.ERROR],
        ChatState.LEARNING: [ChatState.IDLE, ChatState.ERROR],
        ChatState.ERROR: [ChatState.IDLE],
    }

    def __init__(self, initial_state: ChatState = ChatState.IDLE):
        """Initialize state machine."""
        self.context = StateContext(state=initial_state)
        self._callbacks: Dict[ChatState, list] = {}

    def transition(self, new_state: ChatState, **kwargs) -> bool:
        """
        Transition to new state.

        Args:
            new_state: Target state
            **kwargs: Context data for new state

        Returns:
            True if transition succeeded, False if invalid
        """
        # Validate transition
        if new_state not in self.TRANSITIONS.get(self.context.state, []):
            return False

        # Update context
        old_state = self.context.state
        self.context.state = new_state
        self.context.entered_at = datetime.now()

        # Update context data
        for key, value in kwargs.items():
            setattr(self.context, key, value)

        # Trigger callbacks
        self._trigger_callbacks(old_state, new_state)

        return True

    def reset(self) -> None:
        """Reset to IDLE state."""
        self.transition(ChatState.IDLE)
        self.context = StateContext(state=ChatState.IDLE)

    def on_state_enter(self, state: ChatState, callback: Callable) -> None:
        """Register callback for state entry."""
        if state not in self._callbacks:
            self._callbacks[state] = []
        self._callbacks[state].append(callback)

    def _trigger_callbacks(self, old_state: ChatState, new_state: ChatState) -> None:
        """Trigger registered callbacks for state transition."""
        for callback in self._callbacks.get(new_state, []):
            try:
                callback(self.context)
            except Exception as e:
                print(f"State callback error: {e}")

    def get_state(self) -> ChatState:
        """Get current state."""
        return self.context.state

    def get_context(self) -> StateContext:
        """Get full state context."""
        return self.context

    def can_transition(self, new_state: ChatState) -> bool:
        """Check if transition to new state is valid."""
        return new_state in self.TRANSITIONS.get(self.context.state, [])

    def update_progress(self, progress: float) -> None:
        """Update progress (0.0-1.0)."""
        self.context.progress = max(0.0, min(1.0, progress))

    def update_steps(self, current: int, total: int) -> None:
        """Update step progress."""
        self.context.step_current = current
        self.context.step_total = total
        if total > 0:
            self.context.progress = current / total

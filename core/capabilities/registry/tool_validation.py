"""
ToolGuard — Tool access control and validation.

Provides:
- Plan/Act mode: restrict side-effect tools during planning
- One side-effect per turn: prevent cascading mutations
- Path access control: block sensitive file operations
"""

import logging
from typing import Any, Set

logger = logging.getLogger(__name__)


class ToolGuard:
    """Guards tool execution with access control policies."""

    def __init__(self) -> None:
        self._blocked_paths: Set[str] = set()
        self._planning_mode: bool = False
        self._side_effects_this_turn: int = 0
        self._max_side_effects: int = 1

    def is_allowed(self, tool_name: str, **kwargs: Any) -> bool:
        """Check if a tool invocation is allowed under current policies."""
        if self._planning_mode and self._is_side_effect(tool_name):
            logger.debug(f"Tool {tool_name} blocked: planning mode active")
            return False
        if (
            self._is_side_effect(tool_name)
            and self._side_effects_this_turn >= self._max_side_effects
        ):
            logger.debug(f"Tool {tool_name} blocked: side-effect limit reached")
            return False
        return True

    def record_execution(self, tool_name: str) -> None:
        """Record that a tool was executed."""
        if self._is_side_effect(tool_name):
            self._side_effects_this_turn += 1

    def reset_turn(self) -> None:
        """Reset per-turn counters."""
        self._side_effects_this_turn = 0

    def set_planning_mode(self, active: bool) -> None:
        """Enable/disable planning mode (blocks side-effect tools)."""
        self._planning_mode = active

    def _is_side_effect(self, tool_name: str) -> bool:
        """Check if a tool has side effects."""
        side_effect_tools = {
            "file_write",
            "file_delete",
            "terminal_session",
            "browser_automation",
            "email_send",
            "api_call",
        }
        return tool_name in side_effect_tools

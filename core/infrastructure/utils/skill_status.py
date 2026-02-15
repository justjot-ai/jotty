"""
Skill Status Utility
====================

Shared utility for skills to emit status updates during execution.
Skills import this module to get consistent status reporting.

Usage in a skill's tools.py:
    from Jotty.core.infrastructure.utils.skill_status import SkillStatus

    status = SkillStatus("my-skill")

    def my_tool(params: dict) -> dict:
        status.set_callback(params.pop('_status_callback', None))

        status.emit("Searching", " Searching web...")
        results = search_web(query)

        status.emit("Processing", " Processing results...")
        processed = process(results)

        return {"success": True, "data": processed}
"""

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class SkillStatus:
    """
    Status emitter for skills.

    Thread-safe status callback handler that skills can use
    to report progress during execution.
    """

    def __init__(self, skill_name: str) -> None:
        """
        Initialize status emitter for a skill.

        Args:
            skill_name: Name of the skill (for logging)
        """
        self.skill_name = skill_name
        self._callback: Optional[Callable] = None

    def set_callback(self, callback: Optional[Callable]) -> None:
        """
        Set the status callback.

        Called at the start of tool execution with the callback
        from params['_status_callback'].

        Args:
            callback: The status callback function(stage, detail)
        """
        self._callback = callback

    def emit(self, stage: str, detail: str = "") -> None:
        """
        Emit a status update.

        Args:
            stage: Stage name (e.g., "Searching", "Processing")
            detail: Detail message with emoji (e.g., " Searching web...")
        """
        if self._callback:
            try:
                self._callback(stage, detail)
            except Exception:
                pass
        logger.debug(f"[{self.skill_name}] {stage}: {detail}")

    def searching(self, target: str) -> None:
        """Emit searching status."""
        self.emit("Searching", f" Searching {target}...")

    def fetching(self, url: str) -> None:
        """Emit fetching status."""
        # Truncate long URLs
        display_url = url[:60] + "..." if len(url) > 60 else url
        self.emit("Fetching", f" {display_url}")

    def processing(self, item: str = "") -> None:
        """Emit processing status."""
        msg = f" Processing {item}..." if item else " Processing..."
        self.emit("Processing", msg)

    def analyzing(self, item: str = "") -> None:
        """Emit analyzing status."""
        msg = f" Analyzing {item}..." if item else " Analyzing..."
        self.emit("Analyzing", msg)

    def creating(self, item: str) -> None:
        """Emit creating status."""
        self.emit("Creating", f" Creating {item}...")

    def sending(self, destination: str) -> None:
        """Emit sending status."""
        self.emit("Sending", f" Sending to {destination}...")

    def done(self, message: str = "Complete") -> None:
        """Emit completion status."""
        self.emit("Done", f" {message}")

    def error(self, message: str) -> None:
        """Emit error status."""
        self.emit("Error", f" {message}")


# Global instances for common skills (optional convenience)
def get_status(skill_name: str) -> SkillStatus:
    """Get or create a status emitter for a skill."""
    return SkillStatus(skill_name)


__all__ = ["SkillStatus", "get_status"]

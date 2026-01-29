"""
History Manager
===============

Command history management with persistence.
"""

import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


class HistoryManager:
    """
    Manages command history with file persistence.

    Integrates with prompt_toolkit's history system.
    """

    def __init__(
        self,
        history_file: Optional[str] = None,
        max_entries: int = 1000
    ):
        """
        Initialize history manager.

        Args:
            history_file: Path to history file
            max_entries: Maximum history entries to keep
        """
        self.history_file = Path(history_file or "~/.jotty/history").expanduser()
        self.max_entries = max_entries
        self._history: List[str] = []

        # Ensure parent directory exists
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing history
        self._load()

    def _load(self):
        """Load history from file."""
        if not self.history_file.exists():
            return

        try:
            with open(self.history_file, "r") as f:
                lines = f.readlines()
            self._history = [line.strip() for line in lines if line.strip()]
            logger.debug(f"Loaded {len(self._history)} history entries")
        except Exception as e:
            logger.error(f"Failed to load history: {e}")

    def _save(self):
        """Save history to file."""
        try:
            # Trim to max entries
            if len(self._history) > self.max_entries:
                self._history = self._history[-self.max_entries:]

            with open(self.history_file, "w") as f:
                for entry in self._history:
                    f.write(entry + "\n")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def add(self, entry: str):
        """
        Add entry to history.

        Args:
            entry: Command to add
        """
        if not entry or not entry.strip():
            return

        entry = entry.strip()

        # Don't add duplicates of last entry
        if self._history and self._history[-1] == entry:
            return

        self._history.append(entry)
        self._save()

    def get_entries(self, limit: int = None) -> List[str]:
        """
        Get history entries.

        Args:
            limit: Max entries to return

        Returns:
            List of history entries
        """
        if limit:
            return self._history[-limit:]
        return self._history.copy()

    def search(self, prefix: str) -> List[str]:
        """
        Search history for entries starting with prefix.

        Args:
            prefix: Search prefix

        Returns:
            Matching entries
        """
        return [e for e in self._history if e.startswith(prefix)]

    def clear(self):
        """Clear history."""
        self._history.clear()
        self._save()

    def get_prompt_toolkit_history(self):
        """
        Get history object for prompt_toolkit.

        Returns:
            FileHistory instance or None
        """
        try:
            from prompt_toolkit.history import FileHistory
            return FileHistory(str(self.history_file))
        except ImportError:
            return None

    def __len__(self) -> int:
        return len(self._history)

    def __iter__(self):
        return iter(self._history)

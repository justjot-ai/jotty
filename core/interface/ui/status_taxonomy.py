"""
Status Taxonomy - Generic Status Mapping

Provides canonical status mapping for different client systems.
Allows clients to use their own status values while maintaining
consistent internal representation.

Usage:
    from core.ui.status_taxonomy import status_mapper

    # Map client status to kanban column
    column = status_mapper.to_kanban_column('in_progress')
    # Returns: 'in_progress' (matches kanban column ID)

    # Get display label
    label = status_mapper.get_label('in_progress')
    # Returns: 'In Progress'
"""

from typing import Any, Dict, Optional


class StatusTaxonomy:
    """
    Generic status taxonomy for task management.

    Canonical statuses (internal):
    - backlog: Not started, waiting
    - in_progress: Actively being worked on
    - completed: Finished successfully
    - failed: Encountered errors

    Clients can map their own statuses to these canonical values.
    """

    # Canonical status definitions
    CANONICAL_STATUSES = {
        "backlog": {
            "label": "Backlog",
            "description": "Tasks waiting to be started",
            "kanban_column": "backlog",
            "aliases": ["todo", "pending", "not_started", "queue", "waiting"],
        },
        "in_progress": {
            "label": "In Progress",
            "description": "Tasks currently being worked on",
            "kanban_column": "in_progress",
            "aliases": ["active", "doing", "working", "started", "wip"],
        },
        "completed": {
            "label": "Completed",
            "description": "Tasks finished successfully",
            "kanban_column": "completed",
            "aliases": ["done", "finished", "closed", "resolved", "complete"],
        },
        "failed": {
            "label": "Failed",
            "description": "Tasks that encountered errors",
            "kanban_column": "failed",
            "aliases": ["error", "blocked", "cancelled", "rejected"],
        },
    }

    def __init__(self, custom_mapping: Optional[Dict[str, str]] = None) -> None:
        """
        Initialize status taxonomy.

        Args:
            custom_mapping: Optional client-specific status mapping
                           e.g., {'pending': 'backlog', 'active': 'in_progress'}
        """
        self.custom_mapping = custom_mapping or {}

        # Build reverse lookup: alias → canonical
        self._alias_to_canonical = {}
        for canonical, config in self.CANONICAL_STATUSES.items():
            # Canonical status maps to itself
            self._alias_to_canonical[canonical] = canonical
            # Aliases map to canonical
            for alias in config.get("aliases", []):
                self._alias_to_canonical[alias] = canonical

    def normalize(self, status: str) -> str:
        """
        Normalize any status value to canonical status.

        Args:
            status: Status value (can be alias or canonical)

        Returns:
            Canonical status (backlog/in_progress/completed/failed)

        Example:
            >>> taxonomy.normalize('todo')
            'backlog'
            >>> taxonomy.normalize('doing')
            'in_progress'
            >>> taxonomy.normalize('done')
            'completed'
        """
        status_lower = status.lower().replace(" ", "_").replace("-", "_")

        # Check custom mapping first
        if status_lower in self.custom_mapping:
            return self.custom_mapping[status_lower]

        # Check canonical statuses and aliases
        if status_lower in self._alias_to_canonical:
            return self._alias_to_canonical[status_lower]

        # Default to backlog if unknown
        return "backlog"

    def to_kanban_column(self, status: str) -> str:
        """
        Convert status to kanban column ID.

        Args:
            status: Status value

        Returns:
            Kanban column ID

        Example:
            >>> taxonomy.to_kanban_column('done')
            'completed'
        """
        canonical = self.normalize(status)
        return self.CANONICAL_STATUSES[canonical]["kanban_column"]

    def get_label(self, status: str) -> str:
        """
        Get display label for status.

        Args:
            status: Status value

        Returns:
            Human-readable label

        Example:
            >>> taxonomy.get_label('wip')
            'In Progress'
        """
        canonical = self.normalize(status)
        return self.CANONICAL_STATUSES[canonical]["label"]

    def get_all_statuses(self) -> list:
        """Get list of all canonical statuses."""
        return list(self.CANONICAL_STATUSES.keys())

    def create_kanban_columns(self) -> list:
        """
        Generate kanban columns for all canonical statuses.

        Returns:
            List of column definitions

        Example:
            >>> taxonomy.create_kanban_columns()
            [
                {'id': 'backlog', 'title': 'Backlog', 'items': []},
                {'id': 'in_progress', 'title': 'In Progress', 'items': []},
                ...
            ]
        """
        columns = []
        for canonical, config in self.CANONICAL_STATUSES.items():
            columns.append({"id": config["kanban_column"], "title": config["label"], "items": []})
        return columns


# Global instance
status_mapper = StatusTaxonomy()

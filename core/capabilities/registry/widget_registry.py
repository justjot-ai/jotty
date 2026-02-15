"""
Widget Registry - UI Component Types (Section Types)
====================================================

Manages available widget types (section types) that can be used in chat/AI contexts.
Widgets represent different UI components or content types that can be generated.

This is generic and can be extended by any project using Jotty.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WidgetSchema:
    """Schema describing a widget (section type)."""

    value: str  # Unique identifier (e.g., 'text', 'mermaid', 'code')
    label: str  # Human-readable label
    icon: str  # Icon/emoji for UI
    description: str  # What this widget does
    category: str  # Category grouping (e.g., 'Content', 'Diagrams')
    hasOwnUI: bool = False  # Whether widget has custom UI
    contentType: str = "text"  # 'text', 'markdown', 'json', 'code'
    contentSchema: str = ""  # Example/default schema

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "value": self.value,
            "label": self.label,
            "icon": self.icon,
            "description": self.description,
            "category": self.category,
            "hasOwnUI": self.hasOwnUI,
            "contentType": self.contentType,
            "contentSchema": self.contentSchema,
        }


class WidgetRegistry:
    """
    Registry for widgets (section types).

    Generic registry that can be populated by any project using Jotty.
    """

    def __init__(self) -> None:
        self._widgets: Dict[str, WidgetSchema] = {}
        self._by_category: Dict[str, List[str]] = {}
        logger.info(" WidgetRegistry initialized")

    def register(
        self,
        value: str,
        label: str,
        icon: str,
        description: str,
        category: str,
        hasOwnUI: bool = False,
        contentType: str = "text",
        contentSchema: str = "",
    ) -> Any:
        """Register a widget."""
        widget = WidgetSchema(
            value=value,
            label=label,
            icon=icon,
            description=description,
            category=category,
            hasOwnUI=hasOwnUI,
            contentType=contentType,
            contentSchema=contentSchema,
        )

        self._widgets[value] = widget

        # Update category index
        if category not in self._by_category:
            self._by_category[category] = []
        if value not in self._by_category[category]:
            self._by_category[category].append(value)

        logger.debug(f" Registered widget: {value} ({category})")

    def register_batch(self, widgets: List[Dict[str, Any]]) -> None:
        """Register multiple widgets at once."""
        for widget_data in widgets:
            self.register(**widget_data)

    def get(self, value: str) -> Optional[WidgetSchema]:
        """Get widget by value."""
        return self._widgets.get(value)

    def get_all(self) -> List[WidgetSchema]:
        """Get all widgets."""
        return list(self._widgets.values())

    def get_by_category(self, category: str) -> List[WidgetSchema]:
        """Get widgets in a category."""
        values = self._by_category.get(category, [])
        return [self._widgets[v] for v in values if v in self._widgets]

    def get_categories(self) -> List[str]:
        """Get all categories."""
        return sorted(self._by_category.keys())

    def list_values(self) -> List[str]:
        """List all widget values."""
        return list(self._widgets.keys())

    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "available": [w.to_dict() for w in self.get_all()],
            "categories": self.get_categories(),
            "count": len(self._widgets),
        }

    def clear(self) -> None:
        """Clear all widgets (useful for testing)."""
        self._widgets.clear()
        self._by_category.clear()
        logger.info(" WidgetRegistry cleared")


# Global instance (can be extended by projects)
_global_widget_registry = WidgetRegistry()


def get_widget_registry() -> WidgetRegistry:
    """Get the global widget registry instance."""
    return _global_widget_registry

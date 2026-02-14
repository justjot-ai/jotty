"""
AGUI Component Registry
=======================

Registry for client-provided AGUI components and adapters.
Allows projects like JustJot.ai to register their section renderer adapters
with Jotty SDK, enabling agents to use client-specific UI components.

This follows DRY principles:
- Jotty SDK provides the registration infrastructure
- Clients register their components minimally
- Agents automatically get access to client components
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class AGUIComponentAdapter:
    """
    Client-provided AGUI component adapter.

    This represents a section renderer that has been adapted to work
    with AGUI/A2UI protocols.
    """
    section_type: str  # e.g., 'data-table', 'kanban-board'
    label: str  # Human-readable label
    category: str  # Category grouping

    # Serializable adapter functions (JSON strings containing function code or API endpoints)
    to_a2ui: Optional[str] = None  # Function to convert to A2UI blocks
    to_agui: Optional[str] = None  # Function to convert to AGUI component
    from_a2ui: Optional[str] = None  # Function to convert from A2UI
    from_agui: Optional[str] = None  # Function to convert from AGUI

    # Runtime adapter functions (Python callables, not serializable)
    to_a2ui_func: Optional[Callable] = field(default=None, repr=False)
    to_agui_func: Optional[Callable] = field(default=None, repr=False)
    from_a2ui_func: Optional[Callable] = field(default=None, repr=False)
    from_agui_func: Optional[Callable] = field(default=None, repr=False)

    # Metadata
    description: str = ''
    bidirectional: bool = False  # Supports round-trip conversion
    content_type: str = 'json'  # 'json', 'text', 'markdown', 'csv'
    example_input: Optional[str] = None
    example_output: Optional[str] = None

    # Client identification
    client_id: str = 'unknown'  # Which client registered this (e.g., 'justjot')
    version: str = '1.0.0'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'section_type': self.section_type,
            'label': self.label,
            'category': self.category,
            'description': self.description,
            'bidirectional': self.bidirectional,
            'content_type': self.content_type,
            'example_input': self.example_input,
            'example_output': self.example_output,
            'client_id': self.client_id,
            'version': self.version,
            'has_to_a2ui': self.to_a2ui is not None or self.to_a2ui_func is not None,
            'has_to_agui': self.to_agui is not None or self.to_agui_func is not None,
            'has_from_a2ui': self.from_a2ui is not None or self.from_a2ui_func is not None,
            'has_from_agui': self.from_agui is not None or self.from_agui_func is not None,
        }

    def to_json_serializable(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable format (includes function code strings).
        Used for exporting adapters to remote agents.
        """
        return {
            'section_type': self.section_type,
            'label': self.label,
            'category': self.category,
            'to_a2ui': self.to_a2ui,
            'to_agui': self.to_agui,
            'from_a2ui': self.from_a2ui,
            'from_agui': self.from_agui,
            'description': self.description,
            'bidirectional': self.bidirectional,
            'content_type': self.content_type,
            'example_input': self.example_input,
            'example_output': self.example_output,
            'client_id': self.client_id,
            'version': self.version,
        }


class AGUIComponentRegistry:
    """
    Registry for client AGUI component adapters.

    Allows projects to register their section renderer adapters,
    making them available to agents for generating UI components.

    Usage (in client project):
    ```python
    from Jotty.core.registry import get_agui_registry

    registry = get_agui_registry()
    registry.register(
        section_type='data-table',
        label='Data Table',
        category='Data',
        to_a2ui_func=my_data_table_to_a2ui,
        client_id='justjot'
    )
    ```
    """

    def __init__(self):
        self._adapters: Dict[str, AGUIComponentAdapter] = {}
        self._by_category: Dict[str, List[str]] = {}
        self._by_client: Dict[str, List[str]] = {}
        logger.info(" AGUIComponentRegistry initialized")

    def register(
        self,
        section_type: str,
        label: str,
        category: str,
        to_a2ui: Optional[str] = None,
        to_agui: Optional[str] = None,
        from_a2ui: Optional[str] = None,
        from_agui: Optional[str] = None,
        to_a2ui_func: Optional[Callable] = None,
        to_agui_func: Optional[Callable] = None,
        from_a2ui_func: Optional[Callable] = None,
        from_agui_func: Optional[Callable] = None,
        description: str = '',
        bidirectional: bool = False,
        content_type: str = 'json',
        example_input: Optional[str] = None,
        example_output: Optional[str] = None,
        client_id: str = 'unknown',
        version: str = '1.0.0'
    ):
        """
        Register a client AGUI component adapter.

        Args:
            section_type: Section type identifier (e.g., 'data-table')
            label: Human-readable label
            category: Category grouping (e.g., 'Data', 'Visualization')
            to_a2ui: Serialized function code to convert to A2UI
            to_agui: Serialized function code to convert to AGUI
            from_a2ui: Serialized function code to convert from A2UI
            from_agui: Serialized function code to convert from AGUI
            to_a2ui_func: Runtime Python callable to convert to A2UI
            to_agui_func: Runtime Python callable to convert to AGUI
            from_a2ui_func: Runtime Python callable to convert from A2UI
            from_agui_func: Runtime Python callable to convert from AGUI
            description: Description of the adapter
            bidirectional: Whether adapter supports round-trip conversion
            content_type: Type of content ('json', 'text', 'markdown', 'csv')
            example_input: Example input content
            example_output: Example output (A2UI/AGUI)
            client_id: Client identifier (e.g., 'justjot')
            version: Adapter version
        """
        adapter = AGUIComponentAdapter(
            section_type=section_type,
            label=label,
            category=category,
            to_a2ui=to_a2ui,
            to_agui=to_agui,
            from_a2ui=from_a2ui,
            from_agui=from_agui,
            to_a2ui_func=to_a2ui_func,
            to_agui_func=to_agui_func,
            from_a2ui_func=from_a2ui_func,
            from_agui_func=from_agui_func,
            description=description,
            bidirectional=bidirectional,
            content_type=content_type,
            example_input=example_input,
            example_output=example_output,
            client_id=client_id,
            version=version,
        )

        self._adapters[section_type] = adapter

        # Update category index
        if category not in self._by_category:
            self._by_category[category] = []
        if section_type not in self._by_category[category]:
            self._by_category[category].append(section_type)

        # Update client index
        if client_id not in self._by_client:
            self._by_client[client_id] = []
        if section_type not in self._by_client[client_id]:
            self._by_client[client_id].append(section_type)

        logger.debug(f" Registered AGUI adapter: {section_type} from {client_id}")

    def register_batch(self, adapters: List[Dict[str, Any]]) -> None:
        """
        Register multiple adapters at once.

        Usage:
        ```python
        registry.register_batch([
            {
                'section_type': 'data-table',
                'label': 'Data Table',
                'category': 'Data',
                'to_a2ui_func': data_table_to_a2ui,
                'client_id': 'justjot'
            },
            # ... more adapters
        ])
        ```
        """
        for adapter_data in adapters:
            self.register(**adapter_data)
        logger.info(f" Registered {len(adapters)} AGUI adapters in batch")

    def get(self, section_type: str) -> Optional[AGUIComponentAdapter]:
        """Get adapter by section type."""
        return self._adapters.get(section_type)

    def get_all(self) -> List[AGUIComponentAdapter]:
        """Get all adapters."""
        return list(self._adapters.values())

    def get_by_category(self, category: str) -> List[AGUIComponentAdapter]:
        """Get adapters in a category."""
        section_types = self._by_category.get(category, [])
        return [self._adapters[st] for st in section_types if st in self._adapters]

    def get_by_client(self, client_id: str) -> List[AGUIComponentAdapter]:
        """Get adapters registered by a specific client."""
        section_types = self._by_client.get(client_id, [])
        return [self._adapters[st] for st in section_types if st in self._adapters]

    def get_categories(self) -> List[str]:
        """Get all categories."""
        return sorted(self._by_category.keys())

    def get_clients(self) -> List[str]:
        """Get all registered clients."""
        return sorted(self._by_client.keys())

    def list_section_types(self) -> List[str]:
        """List all section types."""
        return list(self._adapters.keys())

    def convert_to_a2ui(self, section_type: str, content: str, **props) -> Optional[List[Dict[str, Any]]]:
        """
        Convert section content to A2UI blocks using registered adapter.

        Args:
            section_type: Section type identifier
            content: Section content
            **props: Additional props to pass to adapter

        Returns:
            A2UI blocks list or None if adapter not found
        """
        adapter = self.get(section_type)
        if not adapter or not adapter.to_a2ui_func:
            logger.warning(f"No A2UI adapter found for section type: {section_type}")
            return None

        try:
            # Call adapter function with content and props
            result = adapter.to_a2ui_func({'content': content, **props})
            return result
        except Exception as e:
            logger.error(f"Error converting {section_type} to A2UI: {e}")
            return None

    def convert_to_agui(self, section_type: str, content: str, **props) -> Optional[Dict[str, Any]]:
        """
        Convert section content to AGUI component using registered adapter.

        Args:
            section_type: Section type identifier
            content: Section content
            **props: Additional props to pass to adapter

        Returns:
            AGUI component or None if adapter not found
        """
        adapter = self.get(section_type)
        if not adapter or not adapter.to_agui_func:
            logger.warning(f"No AGUI adapter found for section type: {section_type}")
            return None

        try:
            result = adapter.to_agui_func({'content': content, **props})
            return result
        except Exception as e:
            logger.error(f"Error converting {section_type} to AGUI: {e}")
            return None

    def to_api_response(self) -> Dict[str, Any]:
        """
        Convert to API response format.

        Returns metadata about all registered adapters without the function code.
        """
        return {
            'available': [adapter.to_dict() for adapter in self.get_all()],
            'categories': self.get_categories(),
            'clients': self.get_clients(),
            'count': len(self._adapters),
        }

    def export_for_remote_agent(self, client_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Export adapters in JSON-serializable format for remote agents.

        Args:
            client_id: Optional filter by client ID

        Returns:
            JSON-serializable dict with adapter definitions
        """
        adapters = self.get_by_client(client_id) if client_id else self.get_all()

        return {
            'version': '1.0.0',
            'client_id': client_id or 'all',
            'adapters': [adapter.to_json_serializable() for adapter in adapters],
            'count': len(adapters),
        }

    def clear(self) -> None:
        """Clear all adapters (useful for testing)."""
        self._adapters.clear()
        self._by_category.clear()
        self._by_client.clear()
        logger.info(" AGUIComponentRegistry cleared")


# Global instance
_global_agui_registry = AGUIComponentRegistry()


def get_agui_registry() -> AGUIComponentRegistry:
    """Get the global AGUI component registry instance."""
    return _global_agui_registry

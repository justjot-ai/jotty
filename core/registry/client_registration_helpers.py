"""
Client Registration Helpers
============================

Helper functions for clients (like JustJot.ai) to register their AGUI
component adapters with Jotty SDK.

This provides high-level convenience functions that make registration simple
and follow DRY principles.
"""

from typing import Dict, Any, Callable, List, Optional
import logging
from .agui_component_registry import get_agui_registry, AGUIComponentAdapter

logger = logging.getLogger(__name__)


def register_agui_adapter_from_registry(adapter_registry: Dict[str, Any], client_id: str = 'unknown', version: str = '1.0.0') -> Any:
    """
    Register adapters from a client's adapter registry dict.

    This is designed to work with JustJot.ai's ADAPTER_REGISTRY pattern.

    Usage:
    ```python
    from my_project.lib.agui_a2ui.adapters import ADAPTER_REGISTRY
    from Jotty.core.registry import register_agui_adapter_from_registry

    register_agui_adapter_from_registry(
        ADAPTER_REGISTRY,
        client_id='justjot',
        version='1.0.0'
    )
    ```

    Args:
        adapter_registry: Dict mapping section_type -> adapter functions
        client_id: Client identifier (e.g., 'justjot')
        version: Adapter version
    """
    registry = get_agui_registry()
    count = 0

    for section_type, adapter_funcs in adapter_registry.items():
        try:
            # Extract adapter functions
            to_a2ui_func = adapter_funcs.get('toA2UI')
            to_agui_func = adapter_funcs.get('toAGUI')
            from_a2ui_func = adapter_funcs.get('fromA2UI')
            from_agui_func = adapter_funcs.get('fromAGUI')

            # Generate label and category from section type
            label = section_type.replace('-', ' ').replace('_', ' ').title()
            category = _guess_category_from_section_type(section_type)

            # Determine content type
            content_type = _guess_content_type(section_type)

            # Check if bidirectional
            bidirectional = from_a2ui_func is not None and from_agui_func is not None

            registry.register(
                section_type=section_type,
                label=label,
                category=category,
                to_a2ui_func=to_a2ui_func,
                to_agui_func=to_agui_func,
                from_a2ui_func=from_a2ui_func,
                from_agui_func=from_agui_func,
                description=f'{label} adapter for AGUI/A2UI protocols',
                bidirectional=bidirectional,
                content_type=content_type,
                client_id=client_id,
                version=version
            )
            count += 1
        except Exception as e:
            logger.error(f"Failed to register adapter for {section_type}: {e}")

    logger.info(f" Registered {count} AGUI adapters from {client_id} registry")
    return count


def register_agui_adapters_from_module(adapter_module: Any, section_types: List[str], client_id: str = 'unknown', version: str = '1.0.0') -> Any:
    """
    Register adapters from a module containing adapter functions.

    Looks for functions named: {section_type}ToA2UI, {section_type}ToAGUI, etc.

    Usage:
    ```python
    from my_project.lib.agui_a2ui import adapters
    from Jotty.core.registry import register_agui_adapters_from_module

    register_agui_adapters_from_module(
        adapters,
        section_types=['data-table', 'chart', 'kanban-board'],
        client_id='justjot'
    )
    ```

    Args:
        adapter_module: Python module containing adapter functions
        section_types: List of section types to register
        client_id: Client identifier
        version: Adapter version
    """
    registry = get_agui_registry()
    count = 0

    for section_type in section_types:
        try:
            # Convert section-type to function name format: dataTableToA2UI
            func_name_base = ''.join(word.capitalize() for word in section_type.split('-'))
            func_name_base = func_name_base[0].lower() + func_name_base[1:]  # camelCase

            # Look for adapter functions
            to_a2ui_name = f'{func_name_base}ToA2UI'
            to_agui_name = f'{func_name_base}ToAGUI'

            to_a2ui_func = getattr(adapter_module, to_a2ui_name, None)
            to_agui_func = getattr(adapter_module, to_agui_name, None)

            if not to_a2ui_func and not to_agui_func:
                logger.debug(f"No adapter functions found for {section_type}")
                continue

            label = section_type.replace('-', ' ').replace('_', ' ').title()
            category = _guess_category_from_section_type(section_type)
            content_type = _guess_content_type(section_type)

            registry.register(
                section_type=section_type,
                label=label,
                category=category,
                to_a2ui_func=to_a2ui_func,
                to_agui_func=to_agui_func,
                description=f'{label} adapter for AGUI/A2UI protocols',
                content_type=content_type,
                client_id=client_id,
                version=version
            )
            count += 1
        except Exception as e:
            logger.error(f"Failed to register adapter for {section_type}: {e}")

    logger.info(f" Registered {count} AGUI adapters from {client_id} module")
    return count


def register_generic_agui_adapter(section_type: str, to_a2ui_func: Callable, to_agui_func: Optional[Callable] = None, client_id: str = 'unknown', **kwargs: Any) -> Any:
    """
    Register a single AGUI adapter.

    Usage:
    ```python
    from Jotty.core.registry import register_generic_agui_adapter

    register_generic_agui_adapter(
        section_type='custom-widget',
        to_a2ui_func=my_custom_to_a2ui,
        to_agui_func=my_custom_to_agui,
        client_id='justjot',
        category='Custom',
        description='My custom widget adapter'
    )
    ```

    Args:
        section_type: Section type identifier
        to_a2ui_func: Function to convert to A2UI
        to_agui_func: Optional function to convert to AGUI
        client_id: Client identifier
        **kwargs: Additional arguments passed to registry.register()
    """
    registry = get_agui_registry()

    # Set defaults
    if 'label' not in kwargs:
        kwargs['label'] = section_type.replace('-', ' ').replace('_', ' ').title()
    if 'category' not in kwargs:
        kwargs['category'] = _guess_category_from_section_type(section_type)
    if 'content_type' not in kwargs:
        kwargs['content_type'] = _guess_content_type(section_type)

    registry.register(
        section_type=section_type,
        to_a2ui_func=to_a2ui_func,
        to_agui_func=to_agui_func,
        client_id=client_id,
        **kwargs
    )

    logger.info(f" Registered AGUI adapter: {section_type} ({client_id})")


def get_registered_adapters_for_client(client_id: str) -> List[AGUIComponentAdapter]:
    """
    Get all adapters registered by a specific client.

    Usage:
    ```python
    from Jotty.core.registry import get_registered_adapters_for_client

    justjot_adapters = get_registered_adapters_for_client('justjot')
    print(f"JustJot.ai has {len(justjot_adapters)} registered adapters")
    ```

    Args:
        client_id: Client identifier

    Returns:
        List of AGUIComponentAdapter objects
    """
    registry = get_agui_registry()
    return registry.get_by_client(client_id)


def export_adapters_for_agent(client_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Export adapters in JSON format for agent consumption.

    This is useful when agents need to know what UI components are available
    from client projects.

    Usage:
    ```python
    from Jotty.core.registry import export_adapters_for_agent

    # Export all adapters
    all_adapters = export_adapters_for_agent()

    # Export only JustJot.ai adapters
    justjot_adapters = export_adapters_for_agent(client_id='justjot')
    ```

    Args:
        client_id: Optional client ID filter

    Returns:
        JSON-serializable dict with adapter metadata
    """
    registry = get_agui_registry()
    return registry.export_for_remote_agent(client_id=client_id)


# Helper functions

def _guess_category_from_section_type(section_type: str) -> str:
    """Guess category from section type name."""
    if any(kw in section_type.lower() for kw in ['chart', 'graph', 'plot']):
        return 'Visualization'
    if any(kw in section_type.lower() for kw in ['table', 'data', 'csv']):
        return 'Data'
    if any(kw in section_type.lower() for kw in ['kanban', 'board', 'todos']):
        return 'Project Management'
    if any(kw in section_type.lower() for kw in ['diagram', 'mermaid', 'excalidraw']):
        return 'Diagrams'
    if any(kw in section_type.lower() for kw in ['code', 'latex', 'markdown']):
        return 'Code & Text'
    if any(kw in section_type.lower() for kw in ['decision', 'log']):
        return 'Project Management'
    return 'Content'


def _guess_content_type(section_type: str) -> str:
    """Guess content type from section type name."""
    if section_type in ['text', 'markdown']:
        return 'markdown'
    if section_type in ['code', 'latex']:
        return 'text'
    if section_type == 'csv':
        return 'csv'
    # Most section types use JSON
    return 'json'

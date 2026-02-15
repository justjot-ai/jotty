from typing import Any

"""
UI Registry - Unified Frontend Components (Eyes)
=================================================

The "Eyes" of the Jotty system - handles all visual output and rendering.

Consolidates:
- WidgetSchema (section types with metadata)
- AGUIComponentAdapter (conversion functions for A2UI/AGUI protocols)

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                           UI REGISTRY (Eyes)                                 │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     UIComponent (Unified Schema)                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │   Widget    │  │    AGUI     │  │  Rendering  │  │   Client    │  │   │
│  │  │  Metadata   │  │  Adapters   │  │   Options   │  │   Binding   │  │   │
│  │  │ icon,label  │  │ to_a2ui()   │  │ contentType │  │ client_id   │  │   │
│  │  │ category    │  │ to_agui()   │  │ hasOwnUI    │  │ version     │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Features:                                                                   │
│  • Category indexing for discovery                                          │
│  • Client-specific component registration                                   │
│  • Bidirectional A2UI/AGUI conversion                                       │
│  • API-friendly serialization                                               │
│  • Backwards compatible with WidgetRegistry and AGUIComponentRegistry       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Integration with Swarm System:
- BaseSwarm uses UIRegistry to discover available output formats
- Agents select appropriate UI components based on task output
- SwarmIntelligence tracks UI component usage for optimization

Author: Jotty Team
Date: February 2026
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# UNIFIED UI COMPONENT SCHEMA
# =============================================================================


@dataclass
class UIComponent:
    """
    Unified UI component combining Widget metadata + AGUI adapters.

    This is the single schema for all frontend components in Jotty.
    Replaces both WidgetSchema and AGUIComponentAdapter with a unified structure.

    Attributes:
        # Identity
        component_type: Unique identifier (e.g., 'data-table', 'mermaid', 'kanban')
        label: Human-readable display name
        category: Grouping category (e.g., 'Data', 'Diagrams', 'Content')

        # Display
        icon: Emoji or icon identifier for UI
        description: What this component renders

        # Rendering
        content_type: Type of content ('json', 'text', 'markdown', 'code', 'csv')
        has_own_ui: Whether component has custom rendering logic
        content_schema: Example/default content structure

        # AGUI Adapters (for A2UI/AGUI protocol conversion)
        to_a2ui_func: Runtime callable to convert content → A2UI blocks
        to_agui_func: Runtime callable to convert content → AGUI format
        from_a2ui_func: Runtime callable to convert A2UI → content
        from_agui_func: Runtime callable to convert AGUI → content

        # Serialized adapters (for remote agents)
        to_a2ui: Serialized function code string
        to_agui: Serialized function code string
        from_a2ui: Serialized function code string
        from_agui: Serialized function code string

        # Client binding
        client_id: Which client registered this (e.g., 'justjot', 'jotty')
        version: Component version
        bidirectional: Supports round-trip conversion
    """

    # Identity
    component_type: str
    label: str
    category: str

    # Display
    icon: str = ""
    description: str = ""

    # Rendering
    content_type: str = "text"  # 'text', 'markdown', 'json', 'code', 'csv'
    has_own_ui: bool = False
    content_schema: str = ""

    # AGUI Adapters (runtime callables)
    to_a2ui_func: Optional[Callable] = field(default=None, repr=False)
    to_agui_func: Optional[Callable] = field(default=None, repr=False)
    from_a2ui_func: Optional[Callable] = field(default=None, repr=False)
    from_agui_func: Optional[Callable] = field(default=None, repr=False)

    # AGUI Adapters (serialized for remote)
    to_a2ui: Optional[str] = None
    to_agui: Optional[str] = None
    from_a2ui: Optional[str] = None
    from_agui: Optional[str] = None

    # Client binding
    client_id: str = "jotty"
    version: str = "1.0.0"
    bidirectional: bool = False

    # Example content
    example_input: Optional[str] = None
    example_output: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "component_type": self.component_type,
            "label": self.label,
            "category": self.category,
            "icon": self.icon,
            "description": self.description,
            "content_type": self.content_type,
            "has_own_ui": self.has_own_ui,
            "content_schema": self.content_schema,
            "client_id": self.client_id,
            "version": self.version,
            "bidirectional": self.bidirectional,
            "has_to_a2ui": self.to_a2ui is not None or self.to_a2ui_func is not None,
            "has_to_agui": self.to_agui is not None or self.to_agui_func is not None,
            "has_from_a2ui": self.from_a2ui is not None or self.from_a2ui_func is not None,
            "has_from_agui": self.from_agui is not None or self.from_agui_func is not None,
        }

    def to_widget_dict(self) -> Dict[str, Any]:
        """Convert to legacy WidgetSchema format for backwards compatibility."""
        return {
            "value": self.component_type,
            "label": self.label,
            "icon": self.icon,
            "description": self.description,
            "category": self.category,
            "hasOwnUI": self.has_own_ui,
            "contentType": self.content_type,
            "contentSchema": self.content_schema,
        }

    def to_agui_dict(self) -> Dict[str, Any]:
        """Convert to legacy AGUIComponentAdapter format for backwards compatibility."""
        return {
            "section_type": self.component_type,
            "label": self.label,
            "category": self.category,
            "description": self.description,
            "bidirectional": self.bidirectional,
            "content_type": self.content_type,
            "example_input": self.example_input,
            "example_output": self.example_output,
            "client_id": self.client_id,
            "version": self.version,
            "has_to_a2ui": self.to_a2ui is not None or self.to_a2ui_func is not None,
            "has_to_agui": self.to_agui is not None or self.to_agui_func is not None,
            "has_from_a2ui": self.from_a2ui is not None or self.from_a2ui_func is not None,
            "has_from_agui": self.from_agui is not None or self.from_agui_func is not None,
        }

    def to_json_serializable(self) -> Dict[str, Any]:
        """Convert to JSON-serializable format (includes function code strings)."""
        return {
            "component_type": self.component_type,
            "label": self.label,
            "category": self.category,
            "icon": self.icon,
            "description": self.description,
            "content_type": self.content_type,
            "has_own_ui": self.has_own_ui,
            "content_schema": self.content_schema,
            "to_a2ui": self.to_a2ui,
            "to_agui": self.to_agui,
            "from_a2ui": self.from_a2ui,
            "from_agui": self.from_agui,
            "client_id": self.client_id,
            "version": self.version,
            "bidirectional": self.bidirectional,
            "example_input": self.example_input,
            "example_output": self.example_output,
        }

    @property
    def has_adapters(self) -> bool:
        """Check if component has any AGUI adapters."""
        return any(
            [
                self.to_a2ui_func,
                self.to_agui_func,
                self.from_a2ui_func,
                self.from_agui_func,
                self.to_a2ui,
                self.to_agui,
                self.from_a2ui,
                self.from_agui,
            ]
        )


# =============================================================================
# UI REGISTRY (Eyes)
# =============================================================================


class UIRegistry:
    """
    Unified UI Registry - The "Eyes" of the Jotty system.

    Consolidates WidgetRegistry + AGUIComponentRegistry into a single,
    coherent registry for all frontend components.

    Features:
    - Unified UIComponent schema
    - Category and client indexing
    - A2UI/AGUI conversion support
    - API-friendly serialization
    - Backwards compatibility with legacy registries

    Usage:
        from Jotty.core.capabilities.registry import get_ui_registry

        ui = get_ui_registry()

        # Register a component
        ui.register(
            component_type='data-table',
            label='Data Table',
            category='Data',
            icon='📊',
            content_type='json',
            to_a2ui_func=my_converter,
            client_id='justjot'
        )

        # Get component
        component = ui.get('data-table')

        # Convert content to A2UI
        a2ui_blocks = ui.convert_to_a2ui('data-table', my_data)
    """

    def __init__(self) -> None:
        self._components: Dict[str, UIComponent] = {}
        self._by_category: Dict[str, List[str]] = {}
        self._by_client: Dict[str, List[str]] = {}
        self._by_content_type: Dict[str, List[str]] = {}
        logger.info("👁️ UIRegistry initialized (Eyes)")

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def register(
        self,
        component_type: str,
        label: str,
        category: str,
        icon: str = "",
        description: str = "",
        content_type: str = "text",
        has_own_ui: bool = False,
        content_schema: str = "",
        to_a2ui_func: Optional[Callable] = None,
        to_agui_func: Optional[Callable] = None,
        from_a2ui_func: Optional[Callable] = None,
        from_agui_func: Optional[Callable] = None,
        to_a2ui: Optional[str] = None,
        to_agui: Optional[str] = None,
        from_a2ui: Optional[str] = None,
        from_agui: Optional[str] = None,
        client_id: str = "jotty",
        version: str = "1.0.0",
        bidirectional: bool = False,
        example_input: Optional[str] = None,
        example_output: Optional[str] = None,
    ) -> UIComponent:
        """
        Register a UI component.

        Args:
            component_type: Unique identifier (e.g., 'data-table')
            label: Human-readable label
            category: Category grouping (e.g., 'Data', 'Diagrams')
            icon: Emoji or icon identifier
            description: What this component renders
            content_type: Type of content ('json', 'text', 'markdown', 'code')
            has_own_ui: Whether component has custom rendering
            content_schema: Example content structure
            to_a2ui_func: Runtime converter to A2UI
            to_agui_func: Runtime converter to AGUI
            from_a2ui_func: Runtime converter from A2UI
            from_agui_func: Runtime converter from AGUI
            to_a2ui: Serialized A2UI converter (for remote agents)
            to_agui: Serialized AGUI converter
            from_a2ui: Serialized from-A2UI converter
            from_agui: Serialized from-AGUI converter
            client_id: Client that registered this
            version: Component version
            bidirectional: Supports round-trip conversion
            example_input: Example input content
            example_output: Example output (A2UI/AGUI)

        Returns:
            The registered UIComponent
        """
        component = UIComponent(
            component_type=component_type,
            label=label,
            category=category,
            icon=icon,
            description=description,
            content_type=content_type,
            has_own_ui=has_own_ui,
            content_schema=content_schema,
            to_a2ui_func=to_a2ui_func,
            to_agui_func=to_agui_func,
            from_a2ui_func=from_a2ui_func,
            from_agui_func=from_agui_func,
            to_a2ui=to_a2ui,
            to_agui=to_agui,
            from_a2ui=from_a2ui,
            from_agui=from_agui,
            client_id=client_id,
            version=version,
            bidirectional=bidirectional,
            example_input=example_input,
            example_output=example_output,
        )

        self._components[component_type] = component

        # Update category index
        if category not in self._by_category:
            self._by_category[category] = []
        if component_type not in self._by_category[category]:
            self._by_category[category].append(component_type)

        # Update client index
        if client_id not in self._by_client:
            self._by_client[client_id] = []
        if component_type not in self._by_client[client_id]:
            self._by_client[client_id].append(component_type)

        # Update content type index
        if content_type not in self._by_content_type:
            self._by_content_type[content_type] = []
        if component_type not in self._by_content_type[content_type]:
            self._by_content_type[content_type].append(component_type)

        logger.debug(f"✅ Registered UI component: {component_type} ({category})")
        return component

    def register_batch(self, components: List[Dict[str, Any]]) -> List[UIComponent]:
        """Register multiple components at once."""
        registered = []
        for comp_data in components:
            registered.append(self.register(**comp_data))
        logger.info(f"📦 Registered {len(registered)} UI components in batch")
        return registered

    def register_from_widget(
        self,
        value: str,
        label: str,
        icon: str,
        description: str,
        category: str,
        hasOwnUI: bool = False,
        contentType: str = "text",
        contentSchema: str = "",
        **kwargs: Any,
    ) -> UIComponent:
        """
        Register from legacy WidgetSchema format.
        Provides backwards compatibility with WidgetRegistry.
        """
        return self.register(
            component_type=value,
            label=label,
            icon=icon,
            description=description,
            category=category,
            has_own_ui=hasOwnUI,
            content_type=contentType,
            content_schema=contentSchema,
            **kwargs,
        )

    def register_from_agui(
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
        description: str = "",
        bidirectional: bool = False,
        content_type: str = "json",
        client_id: str = "unknown",
        version: str = "1.0.0",
        **kwargs: Any,
    ) -> UIComponent:
        """
        Register from legacy AGUIComponentAdapter format.
        Provides backwards compatibility with AGUIComponentRegistry.
        """
        return self.register(
            component_type=section_type,
            label=label,
            category=category,
            description=description,
            content_type=content_type,
            to_a2ui=to_a2ui,
            to_agui=to_agui,
            from_a2ui=from_a2ui,
            from_agui=from_agui,
            to_a2ui_func=to_a2ui_func,
            to_agui_func=to_agui_func,
            from_a2ui_func=from_a2ui_func,
            from_agui_func=from_agui_func,
            bidirectional=bidirectional,
            client_id=client_id,
            version=version,
            **kwargs,
        )

    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    def get(self, component_type: str) -> Optional[UIComponent]:
        """Get component by type."""
        return self._components.get(component_type)

    def get_all(self) -> List[UIComponent]:
        """Get all components."""
        return list(self._components.values())

    def get_by_category(self, category: str) -> List[UIComponent]:
        """Get components in a category."""
        types = self._by_category.get(category, [])
        return [self._components[t] for t in types if t in self._components]

    def get_by_client(self, client_id: str) -> List[UIComponent]:
        """Get components registered by a specific client."""
        types = self._by_client.get(client_id, [])
        return [self._components[t] for t in types if t in self._components]

    def get_by_content_type(self, content_type: str) -> List[UIComponent]:
        """Get components that handle a specific content type."""
        types = self._by_content_type.get(content_type, [])
        return [self._components[t] for t in types if t in self._components]

    def get_with_adapters(self) -> List[UIComponent]:
        """Get components that have AGUI adapters."""
        return [c for c in self._components.values() if c.has_adapters]

    def get_categories(self) -> List[str]:
        """Get all categories."""
        return sorted(self._by_category.keys())

    def get_clients(self) -> List[str]:
        """Get all registered client IDs."""
        return sorted(self._by_client.keys())

    def list_types(self) -> List[str]:
        """List all component types."""
        return list(self._components.keys())

    # =========================================================================
    # CONVERSION (A2UI/AGUI)
    # =========================================================================

    def convert_to_a2ui(self, component_type: str, content: Any) -> Optional[List[Dict]]:
        """
        Convert content to A2UI blocks using component's adapter.

        Args:
            component_type: The component type
            content: The content to convert

        Returns:
            List of A2UI blocks, or None if no adapter
        """
        component = self._components.get(component_type)
        if not component:
            logger.warning(f"Component not found: {component_type}")
            return None

        if component.to_a2ui_func:
            try:
                return component.to_a2ui_func(content)
            except Exception as e:
                logger.error(f"A2UI conversion failed for {component_type}: {e}")
                return None

        logger.debug(f"No to_a2ui adapter for {component_type}")
        return None

    def convert_to_agui(self, component_type: str, content: Any) -> Optional[Dict]:
        """
        Convert content to AGUI format using component's adapter.

        Args:
            component_type: The component type
            content: The content to convert

        Returns:
            AGUI dict, or None if no adapter
        """
        component = self._components.get(component_type)
        if not component:
            logger.warning(f"Component not found: {component_type}")
            return None

        if component.to_agui_func:
            try:
                return component.to_agui_func(content)
            except Exception as e:
                logger.error(f"AGUI conversion failed for {component_type}: {e}")
                return None

        logger.debug(f"No to_agui adapter for {component_type}")
        return None

    def convert_from_a2ui(self, component_type: str, blocks: List[Dict]) -> Optional[Any]:
        """Convert A2UI blocks back to content."""
        component = self._components.get(component_type)
        if not component:
            return None

        if component.from_a2ui_func:
            try:
                return component.from_a2ui_func(blocks)
            except Exception as e:
                logger.error(f"from_a2ui conversion failed for {component_type}: {e}")
                return None
        return None

    def convert_from_agui(self, component_type: str, agui_data: Dict) -> Optional[Any]:
        """Convert AGUI format back to content."""
        component = self._components.get(component_type)
        if not component:
            return None

        if component.from_agui_func:
            try:
                return component.from_agui_func(agui_data)
            except Exception as e:
                logger.error(f"from_agui conversion failed for {component_type}: {e}")
                return None
        return None

    # =========================================================================
    # API RESPONSES
    # =========================================================================

    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "components": [c.to_dict() for c in self.get_all()],
            "categories": self.get_categories(),
            "clients": self.get_clients(),
            "count": len(self._components),
            "with_adapters": len(self.get_with_adapters()),
        }

    def export_for_remote_agent(self, client_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Export components for remote agents.
        Includes serialized adapter functions.
        """
        components = self.get_by_client(client_id) if client_id else self.get_all()
        return {
            "components": [c.to_json_serializable() for c in components],
            "client_id": client_id,
            "count": len(components),
        }

    # =========================================================================
    # UTILITY
    # =========================================================================

    def clear(self) -> None:
        """Clear all components (useful for testing)."""
        self._components.clear()
        self._by_category.clear()
        self._by_client.clear()
        self._by_content_type.clear()
        logger.info("🗑️ UIRegistry cleared")

    def merge_from_widget_registry(self, widget_registry: Any) -> int:
        """
        Import components from a legacy WidgetRegistry.
        Returns count of imported components.
        """
        count = 0
        for widget in widget_registry.get_all():
            self.register_from_widget(
                **(
                    widget.to_dict()
                    if hasattr(widget, "to_dict")
                    else {
                        "value": widget.value,
                        "label": widget.label,
                        "icon": widget.icon,
                        "description": widget.description,
                        "category": widget.category,
                        "hasOwnUI": widget.hasOwnUI,
                        "contentType": widget.contentType,
                        "contentSchema": widget.contentSchema,
                    }
                )
            )
            count += 1
        logger.info(f"📥 Imported {count} widgets from WidgetRegistry")
        return count

    def merge_from_agui_registry(self, agui_registry: Any) -> int:
        """
        Import components from a legacy AGUIComponentRegistry.
        Returns count of imported components.
        """
        count = 0
        for adapter in agui_registry.get_all():
            self.register_from_agui(
                section_type=adapter.section_type,
                label=adapter.label,
                category=adapter.category,
                description=adapter.description,
                content_type=adapter.content_type,
                to_a2ui=adapter.to_a2ui,
                to_agui=adapter.to_agui,
                from_a2ui=adapter.from_a2ui,
                from_agui=adapter.from_agui,
                to_a2ui_func=adapter.to_a2ui_func,
                to_agui_func=adapter.to_agui_func,
                from_a2ui_func=adapter.from_a2ui_func,
                from_agui_func=adapter.from_agui_func,
                bidirectional=adapter.bidirectional,
                client_id=adapter.client_id,
                version=adapter.version,
            )
            count += 1
        logger.info(f"📥 Imported {count} adapters from AGUIComponentRegistry")
        return count


# =============================================================================
# GLOBAL INSTANCE AND FACTORY
# =============================================================================

_global_ui_registry: Optional[UIRegistry] = None


def get_ui_registry() -> UIRegistry:
    """
    Get the global UI registry instance.

    The UIRegistry is the unified frontend component registry,
    combining widgets and AGUI adapters into a single coherent system.
    """
    global _global_ui_registry
    if _global_ui_registry is None:
        _global_ui_registry = UIRegistry()
        _load_builtin_components(_global_ui_registry)
    return _global_ui_registry


def _load_builtin_components(registry: UIRegistry) -> Any:
    """Load built-in UI components."""
    # Core UI components that are always available
    CORE_COMPONENTS = [
        {
            "component_type": "text",
            "label": "Text",
            "category": "Content",
            "icon": "📝",
            "description": "Plain text or markdown content",
            "content_type": "markdown",
        },
        {
            "component_type": "code",
            "label": "Code",
            "category": "Content",
            "icon": "💻",
            "description": "Syntax-highlighted code block",
            "content_type": "code",
        },
        {
            "component_type": "mermaid",
            "label": "Mermaid Diagram",
            "category": "Diagrams",
            "icon": "📊",
            "description": "Mermaid.js diagrams (flowcharts, sequence, etc.)",
            "content_type": "text",
        },
        {
            "component_type": "chart",
            "label": "Chart",
            "category": "Visualization",
            "icon": "📈",
            "description": "Data visualization charts",
            "content_type": "json",
        },
        {
            "component_type": "data-table",
            "label": "Data Table",
            "category": "Data",
            "icon": "📋",
            "description": "Tabular data display",
            "content_type": "json",
        },
        {
            "component_type": "kanban-board",
            "label": "Kanban Board",
            "category": "Project",
            "icon": "📌",
            "description": "Kanban-style task board",
            "content_type": "json",
        },
        {
            "component_type": "todos",
            "label": "Todo List",
            "category": "Project",
            "icon": "✅",
            "description": "Checkbox todo list",
            "content_type": "json",
        },
        {
            "component_type": "image",
            "label": "Image",
            "category": "Media",
            "icon": "🖼️",
            "description": "Image display",
            "content_type": "text",
        },
        {
            "component_type": "audio",
            "label": "Audio",
            "category": "Media",
            "icon": "🔊",
            "description": "Audio player",
            "content_type": "text",
        },
        {
            "component_type": "video",
            "label": "Video",
            "category": "Media",
            "icon": "🎬",
            "description": "Video player",
            "content_type": "text",
        },
        {
            "component_type": "timeline",
            "label": "Timeline",
            "category": "Visualization",
            "icon": "📅",
            "description": "Chronological timeline",
            "content_type": "json",
        },
        {
            "component_type": "card",
            "label": "Card",
            "category": "Layout",
            "icon": "🃏",
            "description": "Content card container",
            "content_type": "json",
        },
    ]

    # Register core components
    for comp in CORE_COMPONENTS:
        registry.register(**comp)
    logger.info(f"📦 Loaded {len(CORE_COMPONENTS)} core UI components")

    # Try to load supervisor widgets
    try:
        from .builtin_widgets import get_supervisor_widgets

        widgets = get_supervisor_widgets()
        for widget in widgets:
            # Convert supervisor widget format to UIComponent format
            registry.register(
                component_type=widget.get("id", widget.get("name", "").lower().replace(" ", "-")),
                label=widget.get("name", "Unknown"),
                category=widget.get("category", "Supervisor"),
                icon="🔧",
                description=widget.get("description", ""),
                content_type="json",
            )
        logger.info(f"📦 Loaded {len(widgets)} supervisor widgets")
    except ImportError:
        logger.debug("No supervisor widgets available")
    except Exception as e:
        logger.debug(f"Failed to load supervisor widgets: {e}")


def reset_ui_registry() -> None:
    """Reset the global UI registry (for testing)."""
    global _global_ui_registry
    _global_ui_registry = None


# =============================================================================
# BACKWARDS COMPATIBILITY ALIASES
# =============================================================================

# These allow old code using WidgetRegistry/AGUIComponentRegistry to still work


def get_widget_registry_compat() -> UIRegistry:
    """
    Backwards compatibility: Returns UIRegistry as widget registry.
    Old code can continue to work.
    """
    return get_ui_registry()


def get_agui_registry_compat() -> UIRegistry:
    """
    Backwards compatibility: Returns UIRegistry as AGUI registry.
    Old code can continue to work.
    """
    return get_ui_registry()

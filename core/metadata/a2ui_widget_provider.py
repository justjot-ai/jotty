from typing import Any
#!/usr/bin/env python3
"""
A2UI Widget Provider - Generic Widget Integration for Jotty SDK
================================================================

Enables AI agents to generate rich, interactive user interfaces using A2UI format.

**A2UI Overview:**
- Open-source framework from Google (v0.8 Public Preview)
- Declarative JSON format for agent-driven UIs
- Security-first: only pre-approved component catalog
- Framework-agnostic: works with React, Flutter, SwiftUI, etc.

**Architecture (DRY/SaaS Principles):**
1. Jotty SDK provides the framework (this class)
2. Clients provide implementation (widget catalog, data providers)
3. Dependency injection pattern (not hardcoded)
4. Extends BaseMetadataProvider (caching, budgeting, tool registry)

**Usage:**
```python
from Jotty.core.metadata import A2UIWidgetProvider, create_widget_provider

# Client provides widget catalog
catalog = {
    "weather": {
        "type": "Card",
        "title": "Weather Widget",
        "schema": {...}
    },
    "tasks": {
        "type": "List",
        "title": "Task List",
        "schema": {...}
    }
}

# Client provides data provider function
def get_widget_data(widget_type, params):
    if widget_type == "weather":
        return fetch_weather_data(params["location"])
    elif widget_type == "tasks":
        return fetch_tasks(params["user_id"])
    return {}

# Create provider (dependency injection)
provider = create_widget_provider(
    widget_catalog=catalog,
    data_provider_fn=get_widget_data
)

# DSPy agent can now generate A2UI widgets
agent = dspy.ReAct(ChatSignature, tools=provider.get_tools())
```

**A2UI Format Example:**
```json
{
  "components": [
    {
      "id": "card-1",
      "component": {
        "Card": {
          "title": "Weather in SF",
          "children": {"explicitList": ["text-1", "icon-1"]}
        }
      }
    },
    {
      "id": "text-1",
      "component": {
        "Text": {
          "value": "Sunny, 72°F",
          "style": "body"
        }
      }
    }
  ]
}
```

**References:**
- A2UI Spec: https://github.com/google/A2UI
- A2UI Composer: https://a2ui-composer.ag-ui.com/
- Developer Guide: https://developers.googleblog.com/introducing-a2ui-an-open-project-for-agent-driven-interfaces/
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
import json

from .base_metadata_provider import BaseMetadataProvider
from .widget_params_schema import (
    WidgetParamSchema,
    generate_param_docstring,
    generate_tool_examples
)

logger = logging.getLogger(__name__)


# =============================================================================
# A2UI Data Structures
# =============================================================================

@dataclass
class A2UIComponent:
    """
    Represents a single A2UI component.

    A2UI uses flat component lists with ID references (not nested trees).
    This is LLM-friendly and enables incremental updates.
    """
    id: str
    component_type: str  # "Card", "Text", "Row", etc.
    props: Dict[str, Any] = field(default_factory=dict)
    children: Optional[List[str]] = None  # List of component IDs
    data_binding: Optional[str] = None  # Data model reference

    def to_a2ui_json(self) -> Dict[str, Any]:
        """Convert to A2UI v0.8 JSON format."""
        result = {
            "id": self.id,
            "component": {
                self.component_type: self.props
            }
        }

        # Add children reference if present
        if self.children:
            result["component"][self.component_type]["children"] = {
                "explicitList": self.children
            }

        # Add data binding if present
        if self.data_binding:
            result["dataBinding"] = self.data_binding

        return result


@dataclass
class A2UIMessage:
    """
    Complete A2UI message with components and data model.

    This is what agents generate and clients render.
    """
    components: List[A2UIComponent]
    data_model: Dict[str, Any] = field(default_factory=dict)
    version: str = "0.8"  # A2UI spec version

    def to_json(self) -> str:
        """Serialize to A2UI JSON string."""
        return json.dumps({
            "version": self.version,
            "components": [c.to_a2ui_json() for c in self.components],
            "dataModel": self.data_model
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "A2UIMessage":
        """Parse A2UI JSON string."""
        data = json.loads(json_str)
        components = []

        for comp_data in data.get("components", []):
            comp_id = comp_data["id"]
            # Extract component type and props
            component_dict = comp_data["component"]
            comp_type = list(component_dict.keys())[0]
            props = component_dict[comp_type]

            # Extract children if present
            children = None
            if "children" in props:
                children_ref = props.pop("children")
                if isinstance(children_ref, dict) and "explicitList" in children_ref:
                    children = children_ref["explicitList"]

            # Extract data binding if present
            data_binding = comp_data.get("dataBinding")

            components.append(A2UIComponent(
                id=comp_id,
                component_type=comp_type,
                props=props,
                children=children,
                data_binding=data_binding
            ))

        return cls(
            components=components,
            data_model=data.get("dataModel", {}),
            version=data.get("version", "0.8")
        )


@dataclass
class WidgetDefinition:
    """
    Definition of a widget in the catalog.

    Clients register widgets with their component structure, data requirements,
    and parameter schemas for agent interaction.

    Attributes:
        id: Unique widget identifier
        name: Human-readable name
        description: Widget purpose and usage
        category: Widget category (layout, content, input, data_viz, etc.)
        component_tree: A2UI component structure template
        data_schema: JSON schema for widget data (what data widget displays)
        param_schema: JSON schema for widget parameters (what params agents pass)
        example_data: Example data for testing
        tags: Search/filter tags
    """
    id: str
    name: str
    description: str
    category: str  # "layout", "content", "input", "data_viz", etc.
    component_tree: List[A2UIComponent]  # Template structure
    data_schema: Dict[str, Any]  # JSON schema for required data
    param_schema: Optional['WidgetParamSchema'] = None  # JSON schema for parameters
    example_data: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)


# =============================================================================
# A2UI Widget Provider (Generic Framework)
# =============================================================================

class A2UIWidgetProvider(BaseMetadataProvider):
    """
    Generic A2UI widget integration for Jotty SDK.

    **Features:**
    - Widget catalog management (pre-approved components)
    - JSON schema generation for A2UI format
    - Security: only trusted widget types allowed
    - DSPy integration: agents generate A2UI JSON
    - Framework-agnostic backend (dependency injection)

    **Dependency Injection:**
    - widget_catalog: Dict of available widgets (client provides)
    - data_provider_fn: Function to fetch widget data (client provides)
    - renderer_config: Frontend renderer configuration (optional)

    **Client Responsibility:**
    1. Define widget catalog with A2UI component structures
    2. Implement data provider function for widget data
    3. Implement frontend renderer (React, Flutter, etc.)

    **Jotty Responsibility:**
    1. Manage widget registration and validation
    2. Generate JSON schemas for agents
    3. Provide tools for DSPy agents
    4. Cache and budget widget data
    """

    def __init__(self, widget_catalog: Optional[Dict[str, WidgetDefinition]] = None, data_provider_fn: Optional[Callable] = None, renderer_config: Optional[Dict[str, Any]] = None, token_budget: int = 100000, enable_caching: bool = True, **kwargs: Any) -> None:
        """
        Initialize A2UI widget provider.

        Args:
            widget_catalog: Dictionary of widget ID -> WidgetDefinition
            data_provider_fn: Function(widget_id, params) -> data dict
            renderer_config: Frontend renderer configuration
            token_budget: Maximum tokens for widget data
            enable_caching: Enable caching of widget data
            **kwargs: Additional BaseMetadataProvider options
        """
        super().__init__(
            name="A2UIWidgetProvider",
            token_budget=token_budget,
            enable_caching=enable_caching,
            **kwargs
        )

        self._widget_catalog = widget_catalog or {}
        self._data_provider_fn = data_provider_fn
        self._renderer_config = renderer_config or {}

        # A2UI standard component types (from spec v0.8)
        self._standard_components = {
            # Layout
            "Row", "Column", "List", "Card", "Tabs", "Modal",
            # Content
            "Text", "Image", "Icon", "Video", "AudioPlayer",
            # Input
            "TextField", "CheckBox", "Slider", "DateTime", "MultipleChoice",
            # Navigation
            "Button",
            # Decoration
            "Divider"
        }

        logger.info(f" A2UIWidgetProvider initialized")
        logger.info(f"   Widget catalog: {len(self._widget_catalog)} widgets")
        logger.info(f" Data provider: {'' if data_provider_fn else ''}")
        logger.info(f"   Caching: {'enabled' if enable_caching else 'disabled'}")

    # -------------------------------------------------------------------------
    # Widget Catalog Management
    # -------------------------------------------------------------------------

    def register_widget(self, widget: WidgetDefinition) -> None:
        """
        Register a widget in the catalog.

        Validates widget structure and adds to catalog.
        """
        # Validate component types are from standard A2UI spec
        for component in widget.component_tree:
            if component.component_type not in self._standard_components:
                logger.warning(
                    f" Widget '{widget.id}' uses non-standard component: "
                    f"'{component.component_type}'"
                )

        self._widget_catalog[widget.id] = widget
        logger.info(f" Registered widget: {widget.id} ({widget.name})")

    def get_widget_catalog(self) -> Dict[str, WidgetDefinition]:
        """Get all registered widgets."""
        return self._widget_catalog.copy()

    def get_widget(self, widget_id: str) -> Optional[WidgetDefinition]:
        """Get widget by ID."""
        return self._widget_catalog.get(widget_id)

    def list_widgets(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[WidgetDefinition]:
        """
        List widgets with optional filtering.

        Args:
            category: Filter by category ("layout", "data_viz", etc.)
            tags: Filter by tags

        Returns:
            List of matching widgets
        """
        widgets = list(self._widget_catalog.values())

        # Filter by category
        if category:
            widgets = [w for w in widgets if w.category == category]

        # Filter by tags
        if tags:
            widgets = [
                w for w in widgets
                if any(tag in w.tags for tag in tags)
            ]

        return widgets

    # -------------------------------------------------------------------------
    # Widget Data & Rendering
    # -------------------------------------------------------------------------

    def render_widget(
        self,
        widget_id: str,
        params: Optional[Dict[str, Any]] = None
    ) -> A2UIMessage:
        """
        Render widget with data.

        1. Get widget definition from catalog
        2. Fetch data using data_provider_fn
        3. Populate component tree with data
        4. Return A2UI message

        Args:
            widget_id: Widget ID from catalog
            params: Parameters for data fetching

        Returns:
            A2UIMessage ready for frontend rendering
        """
        widget = self.get_widget(widget_id)
        if not widget:
            raise ValueError(f"Widget '{widget_id}' not found in catalog")

        # Fetch widget data
        data = {}
        if self._data_provider_fn:
            try:
                data = self._data_provider_fn(widget_id, params or {})
                logger.info(f" Fetched data for widget: {widget_id}")
            except Exception as e:
                logger.error(f" Failed to fetch data for widget '{widget_id}': {e}")
                data = widget.example_data or {}
        else:
            logger.warning(f" No data provider, using example data for: {widget_id}")
            data = widget.example_data or {}

        # Create A2UI message with widget components and data
        return A2UIMessage(
            components=widget.component_tree.copy(),
            data_model=data
        )

    def render_widget_json(
        self,
        widget_id: str,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Render widget and return JSON string.

        Convenience method for agent responses.
        """
        message = self.render_widget(widget_id, params)
        return message.to_json()

    # -------------------------------------------------------------------------
    # DSPy Agent Integration
    # -------------------------------------------------------------------------

    def _generate_render_widget_docstring(self) -> str:
        """
        Generate dynamic docstring for render_widget_tool based on widget parameter schemas.

        This creates agent-friendly documentation with examples for each widget's parameters.
        """
        lines = [
            "DISPLAY rich visual widget to the user instead of plain text.",
            "",
            " IMPORTANT: Use this tool to show visual content to users!",
            "Instead of returning plain text responses, use widgets for better UX.",
            ""
        ]

        # Group widgets by whether they have parameter schemas
        widgets_with_params = []
        widgets_without_params = []

        for widget in self._widget_catalog.values():
            if widget.param_schema and widget.param_schema.properties:
                widgets_with_params.append(widget)
            else:
                widgets_without_params.append(widget)

        # Document widgets with parameter schemas
        if widgets_with_params:
            lines.append("Widgets with parameters:")
            lines.append("")
            for widget in widgets_with_params:
                lines.append(f"  {widget.id}:")
                # Generate examples from schema
                examples = generate_tool_examples(widget.param_schema, widget.id)
                for example in examples[:5]:  # Limit to 5 examples per widget
                    lines.append(f"    - {example}")
                lines.append("")

        # Document widgets without parameters
        if widgets_without_params:
            lines.append("Widgets without parameters (simple display):")
            for widget in widgets_without_params[:10]:  # Limit to avoid token bloat
                lines.append(f"  - {widget.id}: {widget.description}")
            lines.append("")

        lines.extend([
            "Args:",
            "    widget_id: Widget ID from catalog (use list_available_widgets to see all options)",
            "    params: Optional JSON string with widget parameters (e.g., '{\"status\": \"completed\"}')",
            "",
            "Returns:",
            "    A2UI JSON message with rich visual components (Card, List, Text, Image, Button, etc.)"
        ])

        return "\n".join(lines)

    def get_tools(self, actor_name: Optional[str] = None) -> List[Callable]:
        """
        Get tools for DSPy agents (BaseMetadataProvider interface).

        Agents can call these functions to generate A2UI widgets.
        """
        tools = []

        # Tool 1: List available widgets
        def list_available_widgets(category: Optional[str] = None) -> str:
            """
            List available A2UI widgets for displaying rich visual content.

            USE THIS when:
            - User asks "what widgets are available?"
            - You want to see which widgets you can use to display data

            Available widget types:
            - task_list: Display supervisor tasks (pending, completed, failed)
            - product_card, flight_status, email_compose, etc. (30+ standard widgets)

            Args:
                category: Optional category filter (communication, commerce, finance, travel)

            Returns:
                JSON string with widget catalog (id, name, description, category, tags)
            """
            widgets = self.list_widgets(category=category)
            return json.dumps([{
                "id": w.id,
                "name": w.name,
                "description": w.description,
                "category": w.category,
                "tags": w.tags
            } for w in widgets], indent=2)

        # Tool 2: Render widget (with schema-based validation)
        def render_widget_tool(widget_id: str, params: Optional[str] = None) -> str:
            # Parse params
            params_dict = json.loads(params) if params else {}

            # Validate parameters against widget schema
            widget = self.get_widget(widget_id)
            if widget and widget.param_schema:
                # Apply defaults
                params_dict = widget.param_schema.apply_defaults(params_dict)

                # Validate
                is_valid, error_msg = widget.param_schema.validate(params_dict)
                if not is_valid:
                    return json.dumps({
                        "error": f"Invalid parameters for widget '{widget_id}': {error_msg}",
                        "widget_id": widget_id,
                        "provided_params": params_dict
                    })

            return self.render_widget_json(widget_id, params_dict)

        # Set dynamic docstring based on widget schemas
        render_widget_tool.__doc__ = self._generate_render_widget_docstring()

        # Tool 3: Get widget schema
        def get_widget_schema(widget_id: str) -> str:
            """
            Get widget data requirements and example data.

            Use this to understand what data a widget needs before rendering it.
            Shows the expected data structure and an example.

            Args:
                widget_id: Widget ID from catalog

            Returns:
                JSON with data_schema (required fields) and example_data (sample)
            """
            widget = self.get_widget(widget_id)
            if not widget:
                return json.dumps({"error": f"Widget '{widget_id}' not found"})

            return json.dumps({
                "widget_id": widget.id,
                "name": widget.name,
                "data_schema": widget.data_schema,
                "example_data": widget.example_data
            }, indent=2)

        tools.extend([
            list_available_widgets,
            render_widget_tool,
            get_widget_schema
        ])

        logger.info(f" Generated {len(tools)} A2UI widget tools for agents")
        return tools

    def get_tool_names(self) -> List[str]:
        """Get list of tool names."""
        return [
            "list_available_widgets",
            "render_widget_tool",
            "get_widget_schema"
        ]


# =============================================================================
# Helper Functions (Simple Interface)
# =============================================================================

def create_widget_provider(widget_catalog: Dict[str, WidgetDefinition], data_provider_fn: Callable, **kwargs: Any) -> A2UIWidgetProvider:
    """
    Create A2UI widget provider from catalog and data provider.

    Simple interface for clients.

    Args:
        widget_catalog: Dict of widget ID -> WidgetDefinition
        data_provider_fn: Function(widget_id, params) -> data dict
        **kwargs: Additional A2UIWidgetProvider options

    Returns:
        Configured A2UIWidgetProvider instance

    Example:
        ```python
        provider = create_widget_provider(
            widget_catalog=my_widgets,
            data_provider_fn=fetch_widget_data
        )

        # Use in DSPy agent
        tools = provider.get_tools()
        agent = dspy.ReAct(ChatSignature, tools=tools)
        ```
    """
    return A2UIWidgetProvider(
        widget_catalog=widget_catalog,
        data_provider_fn=data_provider_fn,
        **kwargs
    )


def create_widget_from_dict(widget_dict: Dict[str, Any]) -> WidgetDefinition:
    """
    Create WidgetDefinition from dictionary.

    Convenience for loading widgets from config files.

    Args:
        widget_dict: Widget configuration dictionary

    Returns:
        WidgetDefinition instance
    """
    # Parse component tree
    components = []
    for comp_data in widget_dict.get("component_tree", []):
        components.append(A2UIComponent(
            id=comp_data["id"],
            component_type=comp_data["component_type"],
            props=comp_data.get("props", {}),
            children=comp_data.get("children"),
            data_binding=comp_data.get("data_binding")
        ))

    return WidgetDefinition(
        id=widget_dict["id"],
        name=widget_dict["name"],
        description=widget_dict["description"],
        category=widget_dict["category"],
        component_tree=components,
        data_schema=widget_dict.get("data_schema", {}),
        example_data=widget_dict.get("example_data"),
        tags=widget_dict.get("tags", [])
    )

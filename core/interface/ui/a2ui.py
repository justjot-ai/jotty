"""
A2UI Response Formatter

Simple A2UI v0.8 response formatter for Jotty agents.
Converts agent responses to A2UI widget format for rich UI rendering.

This is a lightweight formatter - it doesn't manage widget registry (that's handled by core/registry).
It just formats responses into A2UI structure.

Usage:
    # Format task list as A2UI
    from jotty.core.ui.a2ui import format_task_list

    tasks = [{"title": "Task 1", "status": "completed"}, ...]
    response = format_task_list(tasks)
    # Returns: {"role": "assistant", "content": [{"type": "list", "items": [...]}]}
"""

import json
from typing import Any, Dict, List, Optional, Union


class A2UIWidget:
    """Base class for A2UI widgets."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert widget to dictionary."""
        raise NotImplementedError

    def to_json(self) -> str:
        """Convert widget to JSON string."""
        return json.dumps(self.to_dict())


class A2UIText(A2UIWidget):
    """Text block widget."""

    def __init__(self, text: str, style: Optional[str] = None) -> None:
        """
        Create text block.

        Args:
            text: Text content
            style: Style ('bold', 'italic', or None)
        """
        self.text = text
        self.style = style

    def to_dict(self) -> Dict[str, Any]:
        widget = {"type": "text", "text": self.text}
        if self.style:
            widget["style"] = self.style
        return widget


class A2UICard(A2UIWidget):
    """Card widget with header, body, and footer."""

    def __init__(
        self,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        body: Optional[Union[A2UIWidget, List[A2UIWidget], str]] = None,
        footer: Optional[Union[A2UIWidget, List[A2UIWidget]]] = None,
    ) -> None:
        """
        Create card widget.

        Args:
            title: Card title
            subtitle: Card subtitle
            body: Card body (widget, list of widgets, or string)
            footer: Card footer (widget or list of widgets)
        """
        self.title = title
        self.subtitle = subtitle
        self.body = body
        self.footer = footer

    def to_dict(self) -> Dict[str, Any]:
        widget = {"type": "card"}

        if self.title:
            widget["title"] = self.title
        if self.subtitle:
            widget["subtitle"] = self.subtitle

        if self.body:
            if isinstance(self.body, str):
                widget["body"] = {"type": "text", "text": self.body}
            elif isinstance(self.body, list):
                widget["body"] = [
                    item.to_dict() if isinstance(item, A2UIWidget) else item for item in self.body
                ]
            elif isinstance(self.body, A2UIWidget):
                widget["body"] = self.body.to_dict()

        if self.footer:
            if isinstance(self.footer, list):
                widget["footer"] = [
                    item.to_dict() if isinstance(item, A2UIWidget) else item for item in self.footer
                ]
            elif isinstance(self.footer, A2UIWidget):
                widget["footer"] = self.footer.to_dict()

        return widget


class A2UIList(A2UIWidget):
    """List widget with items."""

    def __init__(self, items: List[Dict[str, Any]]) -> None:
        """
        Create list widget.

        Args:
            items: List of items, each with:
                - title: Item title
                - subtitle: Item subtitle (optional)
                - status: Status badge (optional)
                - icon: Icon name (optional)
                - metadata: List of metadata objects (optional)
        """
        self.items = items

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "list", "items": self.items}


class A2UIImage(A2UIWidget):
    """Image widget."""

    def __init__(self, url: str, alt: Optional[str] = None, caption: Optional[str] = None) -> None:
        """
        Create image widget.

        Args:
            url: Image URL
            alt: Alt text
            caption: Image caption
        """
        self.url = url
        self.alt = alt
        self.caption = caption

    def to_dict(self) -> Dict[str, Any]:
        widget = {"type": "image", "url": self.url}
        if self.alt:
            widget["alt"] = self.alt
        if self.caption:
            widget["caption"] = self.caption
        return widget


class A2UIButton(A2UIWidget):
    """Button widget."""

    def __init__(
        self, label: str, action: Optional[Dict[str, Any]] = None, variant: str = "secondary"
    ) -> None:
        """
        Create button widget.

        Args:
            label: Button label
            action: Action object (type, url/callback)
            variant: Button variant ('primary' or 'secondary')
        """
        self.label = label
        self.action = action
        self.variant = variant

    def to_dict(self) -> Dict[str, Any]:
        widget = {"type": "button", "label": self.label, "variant": self.variant}
        if self.action:
            widget["action"] = self.action
        return widget


class A2UISeparator(A2UIWidget):
    """Separator widget."""

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "separator"}


class A2UISection(A2UIWidget):
    """
    Section widget - renders JustJot.ai native section renderers.

    This is the DRY way to display content: instead of converting section content
    to generic A2UI blocks (list, card, etc.), we pass the native content directly
    to JustJot's section renderers which handle all 70+ types.

    Example:
        # Kanban board
        section = A2UISection(
            section_type="kanban-board",
            content={"columns": [...]},
            title="Project Tasks"
        )

        # Chart
        section = A2UISection(
            section_type="chart",
            content={"type": "bar", "data": {...}},
            title="Performance Metrics"
        )

        # Any of 70+ section types work!
    """

    def __init__(
        self,
        section_type: str,
        content: Union[Dict[str, Any], str],
        title: Optional[str] = None,
        props: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Create section widget.

        Args:
            section_type: JustJot section type (e.g., "kanban-board", "chart", "mermaid")
            content: Section content in native format (dict or JSON string)
            title: Optional title for the section
            props: Optional additional props for the section renderer
        """
        self.section_type = section_type
        self.content = content
        self.title = title
        self.props = props or {}

    def to_dict(self) -> Dict[str, Any]:
        widget = {"type": "section", "section_type": self.section_type, "content": self.content}

        if self.title:
            widget["title"] = self.title

        if self.props:
            widget["props"] = self.props

        return widget


class A2UIBuilder:
    """Builder for complex A2UI responses with multiple widgets."""

    def __init__(self) -> None:
        """Initialize builder."""
        self.widgets: List[A2UIWidget] = []

    def add_widget(self, widget: A2UIWidget) -> "A2UIBuilder":
        """Add a widget to the response."""
        self.widgets.append(widget)
        return self

    def add_text(self, text: str, style: Optional[str] = None) -> "A2UIBuilder":
        """Add a text block."""
        self.widgets.append(A2UIText(text, style))
        return self

    def add_card(
        self,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        body: Optional[Union[A2UIWidget, List[A2UIWidget], str]] = None,
        footer: Optional[Union[A2UIWidget, List[A2UIWidget]]] = None,
    ) -> "A2UIBuilder":
        """Add a card widget."""
        self.widgets.append(A2UICard(title, subtitle, body, footer))
        return self

    def add_list(self, items: List[Dict[str, Any]]) -> "A2UIBuilder":
        """Add a list widget."""
        self.widgets.append(A2UIList(items))
        return self

    def add_image(
        self, url: str, alt: Optional[str] = None, caption: Optional[str] = None
    ) -> "A2UIBuilder":
        """Add an image widget."""
        self.widgets.append(A2UIImage(url, alt, caption))
        return self

    def add_button(
        self, label: str, action: Optional[Dict[str, Any]] = None, variant: str = "secondary"
    ) -> "A2UIBuilder":
        """Add a button widget."""
        self.widgets.append(A2UIButton(label, action, variant))
        return self

    def add_separator(self) -> "A2UIBuilder":
        """Add a separator."""
        self.widgets.append(A2UISeparator())
        return self

    def add_section(
        self,
        section_type: str,
        content: Union[Dict[str, Any], str],
        title: Optional[str] = None,
        props: Optional[Dict[str, Any]] = None,
    ) -> "A2UIBuilder":
        """Add a section widget (JustJot.ai native renderer)."""
        self.widgets.append(A2UISection(section_type, content, title, props))
        return self

    def build(self) -> Dict[str, Any]:
        """
        Build the final A2UI response.

        Returns:
            A2UI response with role and content blocks
        """
        return {"role": "assistant", "content": [widget.to_dict() for widget in self.widgets]}

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.build())


def is_a2ui_response(response: Any) -> bool:
    """
    Check if response is an A2UI widget.

    Args:
        response: Response to check

    Returns:
        True if response is A2UI format
    """
    if isinstance(response, A2UIWidget):
        return True

    if isinstance(response, dict):
        # Check for A2UI structure
        if "role" in response and "content" in response:
            content = response.get("content", [])
            if isinstance(content, list) and len(content) > 0:
                # Check if first item has A2UI widget type
                first_item = content[0]
                if isinstance(first_item, dict) and "type" in first_item:
                    valid_types = [
                        "card",
                        "list",
                        "text",
                        "image",
                        "button",
                        "separator",
                        "section",
                    ]
                    return first_item.get("type") in valid_types

    return False


def convert_to_a2ui_response(response: Any) -> Dict[str, Any]:
    """
    Convert a response to A2UI format.

    Args:
        response: Response (widget, dict, or string)

    Returns:
        A2UI response dictionary
    """
    if isinstance(response, A2UIWidget):
        return {"role": "assistant", "content": [response.to_dict()]}

    if isinstance(response, dict) and is_a2ui_response(response):
        return response

    # Convert string or other types to text widget
    return {"role": "assistant", "content": [{"type": "text", "text": str(response)}]}


# ============================================================================
# Helper Functions - Simple API for Agents
# ============================================================================


def format_task_list(tasks: List[Dict[str, Any]], title: Optional[str] = None) -> Dict[str, Any]:
    """
    Format task list as A2UI response.

    Args:
        tasks: List of tasks with title, status, etc.
        title: Optional card title

    Returns:
        A2UI response
    """
    builder = A2UIBuilder()
    if title:
        builder.add_card(title=title, body=A2UIList(tasks))
    else:
        builder.add_list(tasks)
    return builder.build()


def format_card(title: str, body: str, subtitle: Optional[str] = None) -> Dict[str, Any]:
    """
    Format simple card as A2UI response.

    Args:
        title: Card title
        body: Card body text
        subtitle: Optional subtitle

    Returns:
        A2UI response
    """
    builder = A2UIBuilder()
    builder.add_card(title=title, subtitle=subtitle, body=body)
    return builder.build()


def format_text(text: str, style: Optional[str] = None) -> Dict[str, Any]:
    """
    Format text as A2UI response.

    Args:
        text: Text content
        style: Text style ('bold', 'italic', or None)

    Returns:
        A2UI response
    """
    builder = A2UIBuilder()
    builder.add_text(text, style)
    return builder.build()


def format_section(
    section_type: str,
    content: Union[Dict[str, Any], str],
    title: Optional[str] = None,
    props: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Format JustJot section as A2UI response (DRY way).

    This is the recommended way to display any of JustJot's 70+ section types
    in supervisor chat. Instead of using toA2UI() adapters that convert to
    generic list/card blocks, pass the native content directly.

    Args:
        section_type: JustJot section type:
            - Project: "kanban-board", "gantt", "sprint-planning", etc.
            - Diagrams: "mermaid", "excalidraw", "plantuml", etc.
            - Data: "chart", "data-table", "json", "csv", etc.
            - Business: "swot", "business-model-canvas", "okrs", etc.
            - And 60+ more!
        content: Section content in native format (dict or JSON string)
        title: Optional title displayed above the section
        props: Optional additional props for the renderer

    Returns:
        A2UI response with section block

    Example:
        # Kanban board
        response = format_section(
            section_type="kanban-board",
            content={
                "columns": [
                    {"id": "todo", "title": "To Do", "items": [...]},
                    {"id": "doing", "title": "Doing", "items": [...]},
                    {"id": "done", "title": "Done", "items": [...]}
                ]
            },
            title="Sprint Tasks"
        )

        # Mermaid diagram
        response = format_section(
            section_type="mermaid",
            content="graph TD; A-->B; B-->C;",
            title="System Architecture"
        )

        # Chart
        response = format_section(
            section_type="chart",
            content={
                "type": "bar",
                "data": {
                    "labels": ["Jan", "Feb", "Mar"],
                    "values": [10, 20, 30]
                }
            },
            title="Monthly Revenue"
        )
    """
    builder = A2UIBuilder()
    builder.add_section(section_type, content, title, props)
    return builder.build()

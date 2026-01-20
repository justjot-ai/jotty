"""
JustJot Section Helper - DRY Integration
=========================================

Helper functions for returning JustJot section data in supervisor chat.

IMPORTANT: This replaces the old toA2UI() adapters. Instead of converting
sections to generic A2UI blocks (list, card), we return section blocks that
let JustJot render the native components.

Usage:
    from jotty.core.ui.justjot_helper import return_section

    # Return any JustJot section type
    response = return_section(
        section_type="kanban-board",
        content={"columns": [...]},
        title="Sprint Tasks"
    )

This works for ALL 70+ JustJot section types automatically!
"""

from typing import Dict, Any, Union, Optional
from .a2ui import format_section


def return_section(
    section_type: str,
    content: Union[Dict[str, Any], str],
    title: Optional[str] = None,
    props: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Return a JustJot section for rendering in supervisor chat.

    This is the DRY way - returns section blocks that preserve full
    native functionality instead of converting to generic A2UI blocks.

    Args:
        section_type: JustJot section type (e.g., "kanban-board", "chart", "mermaid")
        content: Section content in native format (dict or JSON string)
        title: Optional title displayed above the section
        props: Optional additional props

    Returns:
        A2UI response with section block

    Example:
        # Kanban board
        response = return_section(
            section_type="kanban-board",
            content={"columns": [
                {"id": "todo", "title": "To Do", "items": [...]},
                {"id": "doing", "title": "Doing", "items": [...]},
                {"id": "done", "title": "Done", "items": [...]}
            ]},
            title="Sprint 23 Tasks"
        )

        # Mermaid diagram
        response = return_section(
            section_type="mermaid",
            content="graph TD; A-->B; B-->C;",
            title="Architecture"
        )

        # Chart
        response = return_section(
            section_type="chart",
            content={"type": "bar", "data": {...}},
            title="Performance Metrics"
        )
    """
    return format_section(section_type, content, title, props)


def return_kanban(
    columns: list,
    title: Optional[str] = "Tasks"
) -> Dict[str, Any]:
    """
    Return a kanban board (convenience wrapper).

    Args:
        columns: List of columns with items
        title: Board title

    Returns:
        A2UI response with kanban section

    Example:
        response = return_kanban(
            columns=[
                {
                    "id": "backlog",
                    "title": "Backlog",
                    "items": [
                        {
                            "id": "task-1",
                            "title": "Implement feature X",
                            "description": "Details...",
                            "priority": "high",
                            "assignee": "Alice"
                        }
                    ]
                },
                {"id": "doing", "title": "In Progress", "items": [...]},
                {"id": "done", "title": "Done", "items": [...]}
            ],
            title="Sprint 23"
        )
    """
    return return_section(
        section_type="kanban-board",
        content={"columns": columns},
        title=title
    )


def return_chart(
    chart_type: str,
    data: Dict[str, Any],
    title: Optional[str] = "Chart"
) -> Dict[str, Any]:
    """
    Return a chart (convenience wrapper).

    Args:
        chart_type: Chart type ("bar", "line", "pie", etc.)
        data: Chart data (labels, values, datasets)
        title: Chart title

    Returns:
        A2UI response with chart section

    Example:
        response = return_chart(
            chart_type="bar",
            data={
                "labels": ["Q1", "Q2", "Q3", "Q4"],
                "values": [100, 150, 200, 250]
            },
            title="Revenue Growth"
        )
    """
    return return_section(
        section_type="chart",
        content={"type": chart_type, "data": data},
        title=title
    )


def return_mermaid(
    diagram: str,
    title: Optional[str] = "Diagram"
) -> Dict[str, Any]:
    """
    Return a Mermaid diagram (convenience wrapper).

    Args:
        diagram: Mermaid diagram syntax
        title: Diagram title

    Returns:
        A2UI response with mermaid section

    Example:
        response = return_mermaid(
            diagram='''
            graph TD
                A[Start] --> B{Decision}
                B -->|Yes| C[Action 1]
                B -->|No| D[Action 2]
            ''',
            title="Process Flow"
        )
    """
    return return_section(
        section_type="mermaid",
        content=diagram,
        title=title
    )


def return_data_table(
    csv_data: str,
    title: Optional[str] = "Data"
) -> Dict[str, Any]:
    """
    Return a data table (convenience wrapper).

    Args:
        csv_data: CSV-formatted data
        title: Table title

    Returns:
        A2UI response with data-table section

    Example:
        response = return_data_table(
            csv_data='''Name,Role,Salary
Alice,Engineer,120000
Bob,Designer,95000
Charlie,PM,110000''',
            title="Team Directory"
        )
    """
    return return_section(
        section_type="data-table",
        content=csv_data,
        title=title
    )


# Map old adapter names to new section helper (for backwards compatibility during migration)
# TODO: Remove this once all code is migrated to use return_section() directly
SECTION_HELPERS = {
    "kanban-board": return_kanban,
    "chart": return_chart,
    "mermaid": return_mermaid,
    "data-table": return_data_table,
}

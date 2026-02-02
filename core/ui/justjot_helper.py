"""
JustJot Section Helper - DRY Integration with Auto-Validation
==============================================================

Helper functions for returning JustJot section data in supervisor chat.

IMPORTANT: This replaces the old toA2UI() adapters. Instead of converting
sections to generic A2UI blocks (list, card), we return section blocks that
let JustJot render the native components.

AUTO-VALIDATION: All helpers automatically validate and transform content
against section schemas fetched from JustJot.ai. This ensures data always
matches what the section renderers expect.

Usage:
    from jotty.core.ui.justjot_helper import return_section

    # Return any JustJot section type (auto-validated!)
    response = return_section(
        section_type="kanban-board",
        content={"columns": [...]},
        title="Sprint Tasks"
    )

This works for ALL 70+ JustJot section types automatically!
"""

from typing import Dict, Any, Union, Optional
from .a2ui import format_section
from .schema_validator import schema_registry
import logging

logger = logging.getLogger(__name__)


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

    AUTO-VALIDATES: Content is automatically validated and transformed
    to match the section renderer's schema (e.g., priority: 1 → 'low').

    Args:
        section_type: JustJot section type (e.g., "kanban-board", "chart", "mermaid")
        content: Section content in native format (dict or JSON string)
        title: Optional title displayed above the section
        props: Optional additional props

    Returns:
        A2UI response with section block

    Example:
        # Kanban board (auto-validates priority, assignee, etc.)
        response = return_section(
            section_type="kanban-board",
            content={"columns": [
                {"id": "todo", "title": "To Do", "items": [
                    {"id": "1", "title": "Task", "priority": 1, "assignee": "Alice"}
                    # ↑ Automatically transforms to: priority='low', assignee={'name': 'Alice'}
                ]},
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
    # Auto-validate and transform content against schema
    if isinstance(content, dict):
        try:
            content = schema_registry.validate_and_transform(section_type, content)
            logger.debug(f"✅ Validated {section_type} content")
        except Exception as e:
            logger.warning(f"⚠️  Schema validation failed for {section_type}: {e}")
            # Continue with original content

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


def _generate_preview_url(url: str, format: str) -> Optional[str]:
    """
    Generate preview URL for different file formats.

    - PDF: Direct URL (browsers render natively)
    - DOCX/PPTX/XLSX: Google Docs Viewer
    """
    from urllib.parse import quote

    format_lower = format.lower().replace('.', '')

    # PDF can be previewed directly by browsers
    if format_lower == 'pdf':
        return url

    # Office formats use Google Docs Viewer
    if format_lower in ['docx', 'pptx', 'xlsx', 'doc', 'ppt', 'xls']:
        # Google Docs Viewer requires a publicly accessible URL
        # If it's a relative path, we need to make it absolute
        if url.startswith('/'):
            # Relative URL - frontend will need to resolve this
            # Return a marker that frontend can process
            return f"gdocs:{url}"
        elif url.startswith('http'):
            # Absolute URL - use Google Docs Viewer directly
            encoded_url = quote(url, safe='')
            return f"https://docs.google.com/viewer?url={encoded_url}&embedded=true"
        else:
            # Local file path - can't preview with Google Docs
            return None

    # Other formats don't have preview support
    return None


def return_file_download(
    url: str,
    filename: str,
    format: str,
    size: Optional[str] = None,
    preview: bool = True,
    description: Optional[str] = None,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Return a downloadable file section for inline display in chat.

    This is the proper way to display generated documents (PDF, DOCX, PPTX)
    in the chat UI with download buttons and optional preview.

    Preview Support:
    - PDF: Native browser preview (iframe with direct URL)
    - DOCX/PPTX/XLSX: Google Docs Viewer (requires public URL)

    Args:
        url: Path or URL to the file
        filename: Display name for the file
        format: File format ('pdf', 'docx', 'pptx', 'xlsx', etc.)
        size: Optional human-readable file size (e.g., '2.5 MB')
        preview: Whether to show inline preview (default: True)
        description: Brief description of the file contents
        title: Section title

    Returns:
        A2UI response with file-download section

    Example:
        response = return_file_download(
            url='/outputs/report_20240115.pdf',
            filename='AI Trends Report.pdf',
            format='pdf',
            size='1.2 MB',
            description='Comprehensive analysis of AI trends for 2024',
            title='Your Report is Ready'
        )
    """
    format_lower = format.lower().replace('.', '')

    content = {
        "url": url,
        "filename": filename,
        "format": format_lower
    }

    if size:
        content["size"] = size
    if description:
        content["description"] = description

    # Always enable preview for supported formats
    content["preview"] = preview

    # Generate preview URL based on format
    if preview:
        preview_url = _generate_preview_url(url, format_lower)
        if preview_url:
            content["previewUrl"] = preview_url

    return return_section(
        section_type="file-download",
        content=content,
        title=title or f"📄 {filename}"
    )


def return_image(
    url: str,
    alt: Optional[str] = None,
    caption: Optional[str] = None,
    width: Optional[str] = None,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Return an image section for inline display in chat.

    Args:
        url: Path or URL to the image
        alt: Alt text for accessibility
        caption: Caption displayed below the image
        width: CSS width (e.g., '100%', '500px')
        title: Section title

    Returns:
        A2UI response with image section

    Example:
        response = return_image(
            url='/outputs/chart.png',
            caption='Sales performance Q4 2024',
            title='Generated Chart'
        )
    """
    content = {"url": url}

    if alt:
        content["alt"] = alt
    if caption:
        content["caption"] = caption
    if width:
        content["width"] = width

    return return_section(
        section_type="image",
        content=content,
        title=title
    )


# Map old adapter names to new section helper (for backwards compatibility during migration)
# TODO: Remove this once all code is migrated to use return_section() directly
SECTION_HELPERS = {
    "kanban-board": return_kanban,
    "chart": return_chart,
    "mermaid": return_mermaid,
    "data-table": return_data_table,
    "file-download": return_file_download,
    "image": return_image,
}

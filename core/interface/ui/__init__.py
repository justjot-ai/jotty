"""
UI Module

Provides response formatters for Jotty agents.
"""

from .a2ui import (
    A2UIBuilder,
    A2UISection,
    convert_to_a2ui_response,
    format_card,
    format_section,
    format_task_list,
    format_text,
    is_a2ui_response,
)
from .justjot_helper import (
    return_chart,
    return_data_table,
    return_file_download,
    return_image,
    return_kanban,
    return_mermaid,
    return_section,
)

__all__ = [
    "format_task_list",
    "format_card",
    "format_text",
    "format_section",
    "is_a2ui_response",
    "convert_to_a2ui_response",
    "A2UISection",
    "A2UIBuilder",
    # JustJot helpers
    "return_section",
    "return_kanban",
    "return_chart",
    "return_mermaid",
    "return_data_table",
    "return_file_download",
    "return_image",
]

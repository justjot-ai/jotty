"""
UI Module

Provides response formatters for Jotty agents.
"""

from .a2ui import (
    format_task_list,
    format_card,
    format_text,
    format_section,
    is_a2ui_response,
    convert_to_a2ui_response,
    A2UISection,
    A2UIBuilder,
)

from .justjot_helper import (
    return_section,
    return_kanban,
    return_chart,
    return_mermaid,
    return_data_table,
    return_file_download,
    return_image,
)

__all__ = [
    'format_task_list',
    'format_card',
    'format_text',
    'format_section',
    'is_a2ui_response',
    'convert_to_a2ui_response',
    'A2UISection',
    'A2UIBuilder',
    # JustJot helpers
    'return_section',
    'return_kanban',
    'return_chart',
    'return_mermaid',
    'return_data_table',
    'return_file_download',
    'return_image',
]

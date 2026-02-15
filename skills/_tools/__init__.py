"""
Skill Tools Infrastructure - Tool Generation, Selection & Management
====================================================================

Infrastructure for managing skills as tools:
- Tool selection: LLM-based agentic tool discovery (tool_selector.py)
- Tool generation: Convert skills to Claude API tool format (tool_generator.py)
- Tool management: Track performance, suggest replacements (tool_management.py)

Moved from core/intelligence/orchestration/ and core/capabilities/registry/
(Feb 2026) to follow skills/ infrastructure pattern (_infrastructure/, _providers/, _tools/).
"""

from .tool_generator import UnifiedToolGenerator, ToolDefinition
from .tool_management import ToolManager
from .tool_selector import (
    ToolShed,
    ToolShedSchema,
    AgenticToolSelector,
    CapabilityIndex,
)

__all__ = [
    # Tool selection (LLM-based)
    "ToolShed",
    "ToolShedSchema",
    "AgenticToolSelector",
    "CapabilityIndex",
    # Tool generation
    "UnifiedToolGenerator",
    "ToolDefinition",
    # Tool management
    "ToolManager",
]

"""
Skill Tools Infrastructure - Tool Generation & Management
=========================================================

Infrastructure for managing skills as tools:
- Tool generation: Convert skills to Claude API tool format
- Tool management: Track performance, suggest replacements

Moved from core/intelligence/orchestration/ (Feb 2026) to follow
skills/ infrastructure pattern (_infrastructure/, _providers/, _tools/).
"""

from .tool_generator import UnifiedToolGenerator
from .tool_management import ToolManager

__all__ = [
    # Tool generation
    "UnifiedToolGenerator",
    # Tool management
    "ToolManager",
]

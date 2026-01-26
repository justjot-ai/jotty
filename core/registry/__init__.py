"""
Jotty Tools and Widgets Registry
=================================

Unified registry for tools and widgets that can be used across projects.
Provides a generic, extensible system for managing AI tools and UI widgets.

Features:
- Tool registry with metadata (category, description, MCP support)
- Widget registry (section types) with metadata
- AGUI component registry for client adapters
- API endpoints for discovery
- Generic enough to work across different projects
"""

from .widget_registry import WidgetRegistry, WidgetSchema, get_widget_registry
from .tools_registry import ToolsRegistry, ToolSchema, get_tools_registry
from .unified_registry import UnifiedRegistry, get_unified_registry
from .agui_component_registry import AGUIComponentRegistry, AGUIComponentAdapter, get_agui_registry
from .client_registration_helpers import (
    register_agui_adapter_from_registry,
    register_agui_adapters_from_module,
    register_generic_agui_adapter,
    get_registered_adapters_for_client,
    export_adapters_for_agent
)
from .skills_registry import (
    SkillsRegistry,
    SkillDefinition,
    get_skills_registry,
)
from .skill_generator import (
    SkillGenerator,
    get_skill_generator,
)
from .skill_venv_manager import (
    SkillVenvManager,
    get_venv_manager,
)
from .skill_dependency_manager import (
    SkillDependencyManager,
    get_dependency_manager,
)

__all__ = [
    # Core registries
    'WidgetRegistry',
    'WidgetSchema',
    'get_widget_registry',
    'ToolsRegistry',
    'ToolSchema',
    'get_tools_registry',
    'UnifiedRegistry',
    'get_unified_registry',
    # Skills registry (framework-level)
    'SkillsRegistry',
    'SkillDefinition',
    'get_skills_registry',
    # Skill generator (AI-powered)
    'SkillGenerator',
    'get_skill_generator',
    # Skill venv manager
    'SkillVenvManager',
    'get_venv_manager',
    # Skill dependency manager
    'SkillDependencyManager',
    'get_dependency_manager',
    # AGUI component registry
    'AGUIComponentRegistry',
    'AGUIComponentAdapter',
    'get_agui_registry',
    # Client registration helpers
    'register_agui_adapter_from_registry',
    'register_agui_adapters_from_module',
    'register_generic_agui_adapter',
    'get_registered_adapters_for_client',
    'export_adapters_for_agent',
]

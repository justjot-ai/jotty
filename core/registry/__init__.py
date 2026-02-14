from typing import Any
"""
Jotty Registry System - Hands and Eyes
=======================================

The unified registry system for Jotty's swarm architecture.

Quick Start:
    from Jotty.core.registry import get_unified_registry

    registry = get_unified_registry()
    skill = registry.get_skill('web-search')
    tools = skill.tools
"""

import importlib as _importlib

# =============================================================================
# EAGER: Core registry classes used everywhere (lightweight, no DSPy)
# =============================================================================

from .unified_registry import (
    UnifiedRegistry,
    get_unified_registry,
    reset_unified_registry,
)

from .skills_registry import (
    SkillsRegistry,
    SkillDefinition,
    SkillType,
    ToolMetadata,
    get_skills_registry,
)

from .ui_registry import (
    UIRegistry,
    UIComponent,
    get_ui_registry,
    reset_ui_registry,
)

from .tool_collection import ToolCollection
from .tool_validation import ToolValidator, validate_tool_attributes, RegistryValidationResult

from .skill_dependency_manager import (
    SkillDependencyManager,
    get_dependency_manager,
)

# =============================================================================
# LAZY: Heavy modules (DSPy-dependent) loaded on first attribute access
# =============================================================================

_LAZY_IMPORTS: dict[str, str] = {
    # Skill generation (imports DSPy)
    "SkillGenerator": ".skill_generator",
    "get_skill_generator": ".skill_generator",
    # Venv management
    "SkillVenvManager": ".skill_venv_manager",
    "get_venv_manager": ".skill_venv_manager",
    # Legacy Widget Registry
    "WidgetRegistry": ".widget_registry",
    "WidgetSchema": ".widget_registry",
    "get_widget_registry": ".widget_registry",
    # Legacy Tools Registry
    "ToolsRegistry": ".tools_registry",
    "ToolSchema": ".tools_registry",
    "get_tools_registry": ".tools_registry",
    # Legacy AGUI Registry
    "AGUIComponentRegistry": ".agui_component_registry",
    "AGUIComponentAdapter": ".agui_component_registry",
    "get_agui_registry": ".agui_component_registry",
    # Legacy helpers
    "register_agui_adapter_from_registry": ".client_registration_helpers",
    "register_agui_adapters_from_module": ".client_registration_helpers",
    "register_generic_agui_adapter": ".client_registration_helpers",
    "get_registered_adapters_for_client": ".client_registration_helpers",
    "export_adapters_for_agent": ".client_registration_helpers",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Eager exports
    'UnifiedRegistry', 'get_unified_registry', 'reset_unified_registry',
    'SkillsRegistry', 'SkillDefinition', 'SkillType', 'ToolMetadata', 'get_skills_registry',
    'UIRegistry', 'UIComponent', 'get_ui_registry', 'reset_ui_registry',
    'ToolCollection', 'ToolValidator', 'validate_tool_attributes', 'ValidationResult',
    'SkillDependencyManager', 'get_dependency_manager',
    # Lazy exports
    *_LAZY_IMPORTS.keys(),
]

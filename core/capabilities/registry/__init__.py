from typing import Any

"""
Jotty Registry System - Hands and Eyes
=======================================

The unified registry system for Jotty's swarm architecture.

Quick Start:
    from Jotty.core.capabilities.registry import get_unified_registry

    registry = get_unified_registry()
    skill = registry.get_skill('web-search')
    tools = skill.tools
"""

import importlib as _importlib

from .skills_registry import (
    SkillDefinition,
    SkillsRegistry,
    SkillType,
    ToolMetadata,
    get_skills_registry,
)
from .tool_collection import ToolCollection
from .tool_validation import RegistryValidationResult, ToolValidator, validate_tool_attributes
from .ui_registry import UIComponent, UIRegistry, get_ui_registry, reset_ui_registry
from .unified_registry import UnifiedRegistry, get_unified_registry, reset_unified_registry

# =============================================================================
# EAGER: Core registry classes used everywhere (lightweight, no DSPy)
# =============================================================================


# =============================================================================
# LAZY: Heavy modules (DSPy-dependent) loaded on first attribute access
# =============================================================================

_LAZY_IMPORTS: dict[str, str] = {
    # Skill generation (imports DSPy)
    "SkillGenerator": ".skill_generator",
    "get_skill_generator": ".skill_generator",
    # Package/dependency management (now in skills/)
    "SkillDependencyManager": "Jotty.skills.skill-package-manager.skill_dependency_manager",
    "get_dependency_manager": "Jotty.skills.skill-package-manager.skill_dependency_manager",
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
        # Handle absolute imports (for skills with hyphens in names)
        if module_path.startswith("Jotty.skills."):
            # Use importlib.util for paths with hyphens
            import importlib.util
            import sys
            from pathlib import Path

            # Convert module path to file path
            # e.g., "Jotty.skills.skill-package-manager.skill_dependency_manager"
            #    -> "skills/skill-package-manager/skill_dependency_manager.py"
            parts = module_path.split(".")
            skill_name = parts[2]  # skill-package-manager
            file_name = parts[3]  # skill_dependency_manager

            base_path = Path(__file__).parent.parent.parent.parent
            file_path = base_path / "skills" / skill_name / f"{file_name}.py"

            # Load module from file path
            spec = importlib.util.spec_from_file_location(module_path, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_path] = module
                spec.loader.exec_module(module)
            else:
                raise ImportError(f"Could not load {module_path} from {file_path}")
        elif module_path.startswith("Jotty."):
            module = _importlib.import_module(module_path)
        else:
            # Relative imports
            module = _importlib.import_module(module_path, __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Eager exports
    "UnifiedRegistry",
    "get_unified_registry",
    "reset_unified_registry",
    "SkillsRegistry",
    "SkillDefinition",
    "SkillType",
    "ToolMetadata",
    "get_skills_registry",
    "UIRegistry",
    "UIComponent",
    "get_ui_registry",
    "reset_ui_registry",
    "ToolCollection",
    "ToolValidator",
    "validate_tool_attributes",
    "ValidationResult",
    "SkillDependencyManager",
    "get_dependency_manager",
    # Lazy exports
    *_LAZY_IMPORTS.keys(),
]

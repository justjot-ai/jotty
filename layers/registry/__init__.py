"""
REGISTRY Layer - Capabilities (Skills/Hands, UI/Eyes)
"""
from Jotty.core.registry import (
    UnifiedRegistry, get_unified_registry, reset_unified_registry,
    SkillsRegistry, SkillDefinition, ToolMetadata, get_skills_registry,
    UIRegistry, UIComponent, get_ui_registry, reset_ui_registry,
    ToolCollection, ToolValidator, validate_tool_attributes, ValidationResult,
    SkillGenerator, get_skill_generator,
    SkillVenvManager, get_venv_manager,
    SkillDependencyManager, get_dependency_manager,
)

# Semantic aliases
Hands = SkillsRegistry
Eyes = UIRegistry
get_hands = get_skills_registry
get_eyes = get_ui_registry

__all__ = [
    "UnifiedRegistry", "get_unified_registry", "reset_unified_registry",
    "SkillsRegistry", "SkillDefinition", "ToolMetadata", "get_skills_registry",
    "UIRegistry", "UIComponent", "get_ui_registry", "reset_ui_registry",
    "Hands", "Eyes", "get_hands", "get_eyes",
    "ToolCollection", "ToolValidator", "validate_tool_attributes", "ValidationResult",
    "SkillGenerator", "get_skill_generator", "SkillVenvManager", "get_venv_manager",
    "SkillDependencyManager", "get_dependency_manager",
]

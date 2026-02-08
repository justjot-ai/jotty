"""
Jotty Registry System - Hands and Eyes
=======================================

The unified registry system for Jotty's swarm architecture.

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                         JOTTY REGISTRY SYSTEM                                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     UNIFIED REGISTRY                                 │   │
│  │                 (Single Entry Point)                                 │   │
│  │                                                                      │   │
│  │  ┌─────────────────────────┐   ┌─────────────────────────┐          │   │
│  │  │  SKILLS (Backend/Hands) │   │   UI (Frontend/Eyes)    │          │   │
│  │  │                         │   │                         │          │   │
│  │  │  • SkillsRegistry       │   │  • UIRegistry           │          │   │
│  │  │  • SkillDefinition      │   │  • UIComponent          │          │   │
│  │  │  • ToolMetadata         │   │  • A2UI/AGUI adapters   │          │   │
│  │  │  • Lazy tool loading    │   │  • Content converters   │          │   │
│  │  │  • MCP support          │   │  • Category indexing    │          │   │
│  │  │                         │   │                         │          │   │
│  │  └─────────────────────────┘   └─────────────────────────┘          │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Integration with Swarm System:                                             │
│  • BaseSwarm queries UnifiedRegistry for available capabilities             │
│  • Agents use Skills (Hands) for task execution                            │
│  • Agents use UI (Eyes) for output rendering                               │
│  • SwarmIntelligence tracks usage for optimization                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Quick Start:
    from Jotty.core.registry import get_unified_registry

    registry = get_unified_registry()

    # Backend (Hands) - What agents can DO
    skill = registry.get_skill('web-search')
    tools = skill.tools

    # Frontend (Eyes) - How agents RENDER output
    component = registry.ui.get('data-table')
    a2ui_blocks = registry.ui.convert_to_a2ui('data-table', my_data)
"""

# =============================================================================
# PRIMARY EXPORTS - Use These
# =============================================================================

# Unified Registry (Main Entry Point)
from .unified_registry import (
    UnifiedRegistry,
    get_unified_registry,
    reset_unified_registry,
)

# Skills Registry (Backend/Hands)
from .skills_registry import (
    SkillsRegistry,
    SkillDefinition,
    SkillType,
    ToolMetadata,
    get_skills_registry,
)

# UI Registry (Frontend/Eyes)
from .ui_registry import (
    UIRegistry,
    UIComponent,
    get_ui_registry,
    reset_ui_registry,
)

# Tool Collections and Validation
from .tool_collection import ToolCollection
from .tool_validation import ToolValidator, validate_tool_attributes, ValidationResult

# Skill Generation and Management
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

# =============================================================================
# LEGACY EXPORTS - Deprecated but still work
# =============================================================================

# These are kept for backwards compatibility but log deprecation warnings

# Legacy Widget Registry (now part of UIRegistry)
from .widget_registry import WidgetRegistry, WidgetSchema, get_widget_registry

# Legacy Tools Registry (now part of SkillsRegistry)
from .tools_registry import ToolsRegistry, ToolSchema, get_tools_registry

# Legacy AGUI Component Registry (now part of UIRegistry)
from .agui_component_registry import AGUIComponentRegistry, AGUIComponentAdapter, get_agui_registry

# Legacy client registration helpers (still work with new system)
from .client_registration_helpers import (
    register_agui_adapter_from_registry,
    register_agui_adapters_from_module,
    register_generic_agui_adapter,
    get_registered_adapters_for_client,
    export_adapters_for_agent
)

# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # =========================================================================
    # PRIMARY (Use These)
    # =========================================================================

    # Unified Registry
    'UnifiedRegistry',
    'get_unified_registry',
    'reset_unified_registry',

    # Skills Registry (Backend/Hands)
    'SkillsRegistry',
    'SkillDefinition',
    'SkillType',
    'ToolMetadata',
    'get_skills_registry',

    # UI Registry (Frontend/Eyes)
    'UIRegistry',
    'UIComponent',
    'get_ui_registry',
    'reset_ui_registry',

    # Tool utilities
    'ToolCollection',
    'ToolValidator',
    'validate_tool_attributes',
    'ValidationResult',

    # Skill generation
    'SkillGenerator',
    'get_skill_generator',
    'SkillVenvManager',
    'get_venv_manager',
    'SkillDependencyManager',
    'get_dependency_manager',

    # =========================================================================
    # LEGACY (Deprecated but still work)
    # =========================================================================

    # Legacy Widget Registry
    'WidgetRegistry',
    'WidgetSchema',
    'get_widget_registry',

    # Legacy Tools Registry
    'ToolsRegistry',
    'ToolSchema',
    'get_tools_registry',

    # Legacy AGUI Registry
    'AGUIComponentRegistry',
    'AGUIComponentAdapter',
    'get_agui_registry',

    # Legacy helpers
    'register_agui_adapter_from_registry',
    'register_agui_adapters_from_module',
    'register_generic_agui_adapter',
    'get_registered_adapters_for_client',
    'export_adapters_for_agent',
]

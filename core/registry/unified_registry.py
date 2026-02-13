"""
Unified Registry - Combined Backend (Skills) + Frontend (UI)
=============================================================

The single entry point for all registry operations in Jotty.

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED REGISTRY                                     │
│                   (Single Entry Point for All Components)                   │
│                                                                              │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │      SKILLS REGISTRY (Hands)    │  │       UI REGISTRY (Eyes)        │  │
│  │                                 │  │                                 │  │
│  │  What the swarm can DO:         │  │  How the swarm can SEE/RENDER:  │  │
│  │  • Skills with tools            │  │  • UI Components                │  │
│  │  • Tool metadata (MCP)          │  │  • AGUI Adapters                │  │
│  │  • Tool implementations         │  │  • A2UI Converters              │  │
│  │  • Parameter schemas            │  │  • Widget metadata              │  │
│  │                                 │  │                                 │  │
│  └─────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                              │
│  Integration Points:                                                         │
│  • BaseSwarm queries available tools and UI components                      │
│  • Agents use skills for execution, UI for output rendering                 │
│  • SwarmIntelligence tracks tool/component usage for optimization           │
│  • AutoAgent discovers skills for autonomous execution                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

The Unified Registry combines:
- Backend: SkillsRegistry (what agents can DO)
- Frontend: UIRegistry (how agents RENDER output)

This replaces the old fragmented registry system:
- OLD: ToolsRegistry (orphaned), WidgetRegistry (orphaned), AGUIComponentRegistry
- NEW: SkillsRegistry + UIRegistry under UnifiedRegistry

Author: Jotty Team
Date: February 2026
"""

from typing import Dict, Any, Optional, List, Callable
import logging

logger = logging.getLogger(__name__)


class UnifiedRegistry:
    """
    Unified Registry - The single entry point for all Jotty components.

    Combines:
    - SkillsRegistry (Backend/Hands): What agents can DO
    - UIRegistry (Frontend/Eyes): How agents RENDER output

    This is the main interface that projects should use to access
    Jotty's full registry system.

    Usage:
        from Jotty.core.registry import get_unified_registry

        registry = get_unified_registry()

        # Access skills (backend)
        skill = registry.skills.get_skill('web-search')
        tools = skill.tools

        # Access UI components (frontend)
        component = registry.ui.get('data-table')
        a2ui_blocks = registry.ui.convert_to_a2ui('data-table', my_data)

        # Discover available capabilities
        all_skills = registry.list_skills()
        all_ui = registry.list_ui_components()

        # Get tools in Claude format
        claude_tools = registry.get_claude_tools()
    """

    def __init__(
        self,
        skills_registry=None,
        ui_registry=None,
        # Legacy support
        widget_registry=None,
        tools_registry=None,
    ):
        """
        Initialize unified registry.

        Args:
            skills_registry: SkillsRegistry instance (defaults to global)
            ui_registry: UIRegistry instance (defaults to global)
            widget_registry: Legacy WidgetRegistry (migrated to UIRegistry)
            tools_registry: Legacy ToolsRegistry (ignored, use SkillsRegistry)
        """
        # Import lazily to avoid circular imports
        from .skills_registry import get_skills_registry
        from .ui_registry import get_ui_registry

        self._skills = skills_registry or get_skills_registry()
        self._ui = ui_registry or get_ui_registry()

        # Migrate legacy widget registry if provided
        if widget_registry:
            self._ui.merge_from_widget_registry(widget_registry)

        logger.info(" UnifiedRegistry initialized (Skills + UI)")

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def skills(self):
        """Access SkillsRegistry (Backend/Hands)."""
        return self._skills

    @property
    def ui(self):
        """Access UIRegistry (Frontend/Eyes)."""
        return self._ui

    # Legacy aliases
    @property
    def tools(self):
        """Legacy alias for skills registry."""
        return self._skills

    @property
    def widgets(self):
        """Legacy alias for UI registry."""
        return self._ui

    # =========================================================================
    # SKILL OPERATIONS (Backend/Hands)
    # =========================================================================

    def list_skills(self) -> List[str]:
        """List all available skill names."""
        return list(self._skills.loaded_skills.keys())

    def get_skill(self, name: str):
        """Get a skill by name."""
        return self._skills.get_skill(name)

    def get_tool(self, skill_name: str, tool_name: str) -> Optional[Callable]:
        """Get a specific tool from a skill."""
        skill = self._skills.get_skill(skill_name)
        if skill:
            return skill.tools.get(tool_name)
        return None

    def get_claude_tools(self, skill_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get tools in Claude API format.

        Args:
            skill_names: Optional list of skill names to include (all if None)

        Returns:
            List of tool definitions for Claude API
        """
        tools = []
        skills_to_check = skill_names or self.list_skills()

        for skill_name in skills_to_check:
            skill = self._skills.get_skill(skill_name)
            if skill:
                tools.extend(skill.to_claude_tools())

        return tools

    def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Get all MCP-enabled tools."""
        mcp_tools = []
        for skill_name in self.list_skills():
            skill = self._skills.get_skill(skill_name)
            if skill and skill.mcp_enabled:
                mcp_tools.extend(skill.to_claude_tools())
        return mcp_tools

    # =========================================================================
    # UI OPERATIONS (Frontend/Eyes)
    # =========================================================================

    def list_ui_components(self) -> List[str]:
        """List all available UI component types."""
        return self._ui.list_types()

    def get_ui_component(self, component_type: str):
        """Get a UI component by type."""
        return self._ui.get(component_type)

    def get_ui_categories(self) -> List[str]:
        """Get all UI component categories."""
        return self._ui.get_categories()

    def convert_to_a2ui(self, component_type: str, content: Any) -> Optional[List[Dict]]:
        """Convert content to A2UI blocks using component's adapter."""
        return self._ui.convert_to_a2ui(component_type, content)

    def convert_to_agui(self, component_type: str, content: Any) -> Optional[Dict]:
        """Convert content to AGUI format using component's adapter."""
        return self._ui.convert_to_agui(component_type, content)

    # =========================================================================
    # UNIFIED OPERATIONS
    # =========================================================================

    def get_all(self) -> Dict[str, Any]:
        """
        Get all skills and UI components in API format.

        Returns:
            Dict with 'skills' and 'ui' keys
        """
        return {
            'skills': {
                'available': [self._skills.get_skill(n).to_dict() for n in self.list_skills()
                              if self._skills.get_skill(n)],
                'count': len(self.list_skills()),
            },
            'ui': self._ui.to_api_response(),
        }

    def get_tools(self) -> Dict[str, Any]:
        """Get skills registry data (legacy alias)."""
        return {
            'available': [self._skills.get_skill(n).to_dict() for n in self.list_skills()
                          if self._skills.get_skill(n)],
            'count': len(self.list_skills()),
        }

    def get_widgets(self) -> Dict[str, Any]:
        """Get UI registry data (legacy alias)."""
        return self._ui.to_api_response()

    def validate_tools(self, tool_names: List[str]) -> Dict[str, bool]:
        """
        Validate that tool names exist in any skill.

        Returns:
            Dict mapping tool_name -> exists
        """
        all_tools = set()
        for skill_name in self.list_skills():
            skill = self._skills.get_skill(skill_name)
            if skill:
                all_tools.update(skill.list_tools())
        return {name: name in all_tools for name in tool_names}

    def validate_widgets(self, widget_types: List[str]) -> Dict[str, bool]:
        """
        Validate that UI component types exist.

        Returns:
            Dict mapping component_type -> exists
        """
        available = set(self._ui.list_types())
        return {t: t in available for t in widget_types}

    def get_enabled_defaults(self) -> Dict[str, Any]:
        """
        Get default enabled tools and UI components.

        Projects can override this to provide their own defaults.
        """
        # Default: enable all skills, common UI components
        common_ui = ['text', 'mermaid', 'code', 'todos', 'chart', 'kanban-board', 'data-table']
        available_ui = self._ui.list_types()
        default_ui = [u for u in common_ui if u in available_ui]

        return {
            'skills': self.list_skills(),
            'ui_components': default_ui if default_ui else available_ui[:10],
        }

    def discover_for_task(self, task_description: str) -> Dict[str, Any]:
        """
        Discover relevant skills and UI components for a task.

        Delegates skill discovery to SkillsRegistry.discover() which uses
        keyword + capability scoring with type-aware boosting.

        Args:
            task_description: Description of the task

        Returns:
            Dict with 'skills' (sorted by relevance) and 'ui' suggestions
        """
        # Delegate skill discovery to the single source of truth
        relevant_skills = self._skills.discover(task_description)

        # Find relevant UI components
        task_lower = task_description.lower()
        relevant_ui = []
        ui_keywords = {
            'chart': ['chart', 'graph', 'visualiz', 'plot'],
            'data-table': ['table', 'data', 'list', 'records'],
            'mermaid': ['diagram', 'flowchart', 'architecture'],
            'code': ['code', 'script', 'program'],
            'kanban-board': ['kanban', 'board', 'tasks', 'project'],
            'todos': ['todo', 'task', 'checklist'],
        }

        for component_type, keywords in ui_keywords.items():
            if any(kw in task_lower for kw in keywords):
                component = self._ui.get(component_type)
                if component:
                    relevant_ui.append(component.to_dict())

        return {
            'skills': relevant_skills,
            'ui': relevant_ui,
            'task': task_description,
        }

    # =========================================================================
    # CONTEXT-SCOPED TOOLS (AgentScope tool-groups inspired)
    # =========================================================================

    def get_scoped_tools(
        self,
        task_description: str,
        max_tools: int = 10,
        format: str = 'claude',
    ) -> List[Any]:
        """
        Return a focused subset of tools relevant to the task.

        AgentScope insight: Agents perform better when they see fewer,
        more relevant tools instead of the full 126-tool catalog.
        This reduces context pollution and improves tool selection accuracy.

        DRY: Reuses existing discover_for_task() scoring + get_claude_tools().
        KISS: One method, no group activation state to manage.

        Args:
            task_description: What the agent is trying to do
            max_tools: Max tools to include in context (default 10)
            format: Tool format — 'claude', 'names', or 'full' (default 'claude')

        Returns:
            Focused list of tools in requested format

        Example:
            # Before (all 126 tools in context):
            tools = registry.get_claude_tools()

            # After (only relevant tools):
            tools = registry.get_scoped_tools("search the web for AI news", max_tools=8)
        """
        discovery = self.discover_for_task(task_description)
        relevant_names = [
            s['name'] if isinstance(s, dict) else s
            for s in discovery.get('skills', [])
        ][:max_tools]

        if not relevant_names:
            # Fallback: return first N skills
            relevant_names = self.list_skills()[:max_tools]

        if format == 'names':
            return relevant_names
        elif format == 'claude':
            return self.get_claude_tools(relevant_names)
        else:
            # 'full' — return skill objects
            return [
                self._skills.get_skill(name)
                for name in relevant_names
                if self._skills.get_skill(name) is not None
            ]

    # =========================================================================
    # INTROSPECTION
    # =========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the registry state."""
        return {
            'skills': {
                'count': len(self.list_skills()),
                'names': self.list_skills()[:10],  # First 10
                'has_more': len(self.list_skills()) > 10,
            },
            'ui': {
                'count': len(self._ui.list_types()),
                'categories': self._ui.get_categories(),
                'with_adapters': len(self._ui.get_with_adapters()),
            },
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_global_unified_registry: Optional[UnifiedRegistry] = None


def get_unified_registry() -> UnifiedRegistry:
    """
    Get the global unified registry instance.

    This is the main entry point for accessing all Jotty components.
    """
    global _global_unified_registry
    if _global_unified_registry is None:
        _global_unified_registry = UnifiedRegistry()
    return _global_unified_registry


def reset_unified_registry():
    """Reset the global unified registry (for testing)."""
    global _global_unified_registry
    _global_unified_registry = None


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

# These functions allow old code to continue working

def get_tools_registry():
    """Legacy: Returns skills registry as tools registry."""
    logger.warning("get_tools_registry() is deprecated. Use get_unified_registry().skills instead.")
    return get_unified_registry().skills


def get_widget_registry():
    """Legacy: Returns UI registry as widget registry."""
    logger.warning("get_widget_registry() is deprecated. Use get_unified_registry().ui instead.")
    return get_unified_registry().ui

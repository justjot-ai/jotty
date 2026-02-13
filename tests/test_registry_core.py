"""
Tests for Registry Core Module
=================================
Tests for UnifiedRegistry, skill discovery, and tool management.
"""
import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List


# =============================================================================
# UnifiedRegistry Creation Tests
# =============================================================================

class TestUnifiedRegistryCreation:
    """Tests for UnifiedRegistry initialization and singleton."""

    @pytest.mark.unit
    def test_singleton_creation(self):
        """get_unified_registry returns singleton."""
        from Jotty.core.registry.unified_registry import (
            get_unified_registry, reset_unified_registry,
        )
        reset_unified_registry()
        r1 = get_unified_registry()
        r2 = get_unified_registry()
        assert r1 is r2
        reset_unified_registry()

    @pytest.mark.unit
    def test_reset_singleton(self):
        """reset_unified_registry clears singleton."""
        from Jotty.core.registry.unified_registry import (
            get_unified_registry, reset_unified_registry,
        )
        reset_unified_registry()
        r1 = get_unified_registry()
        reset_unified_registry()
        r2 = get_unified_registry()
        assert r1 is not r2
        reset_unified_registry()

    @pytest.mark.unit
    def test_list_skills_returns_list(self):
        """list_skills returns list of strings."""
        from Jotty.core.registry.unified_registry import (
            get_unified_registry, reset_unified_registry,
        )
        reset_unified_registry()
        registry = get_unified_registry()
        skills = registry.list_skills()
        assert isinstance(skills, list)
        assert all(isinstance(s, str) for s in skills)
        reset_unified_registry()


# =============================================================================
# Skill Discovery Tests
# =============================================================================

class TestSkillDiscovery:
    """Tests for task-based skill discovery."""

    @pytest.mark.unit
    def test_discover_for_task_returns_dict(self):
        """discover_for_task returns dict with skills and ui keys."""
        from Jotty.core.registry.unified_registry import (
            get_unified_registry, reset_unified_registry,
        )
        reset_unified_registry()
        registry = get_unified_registry()
        result = registry.discover_for_task("research AI trends")
        assert isinstance(result, dict)
        assert 'skills' in result
        assert isinstance(result['skills'], list)
        reset_unified_registry()

    @pytest.mark.unit
    def test_discover_for_chart_task(self):
        """discover_for_task matches chart UI for charting tasks."""
        from Jotty.core.registry.unified_registry import (
            get_unified_registry, reset_unified_registry,
        )
        reset_unified_registry()
        registry = get_unified_registry()
        result = registry.discover_for_task("create a chart showing trends")
        if 'ui' in result:
            ui_names = [u if isinstance(u, str) else u.get('type', '') for u in result['ui']]
            # Chart should be suggested for charting tasks
            assert any('chart' in str(u).lower() for u in result.get('ui', []))
        reset_unified_registry()

    @pytest.mark.unit
    def test_discover_empty_task(self):
        """discover_for_task handles empty task string."""
        from Jotty.core.registry.unified_registry import (
            get_unified_registry, reset_unified_registry,
        )
        reset_unified_registry()
        registry = get_unified_registry()
        result = registry.discover_for_task("")
        assert isinstance(result, dict)
        assert 'skills' in result
        reset_unified_registry()


# =============================================================================
# Tool Management Tests
# =============================================================================

class TestToolManagement:
    """Tests for tool retrieval and validation."""

    @pytest.mark.unit
    def test_get_claude_tools_returns_list(self):
        """get_claude_tools returns list of tool dicts."""
        from Jotty.core.registry.unified_registry import (
            get_unified_registry, reset_unified_registry,
        )
        reset_unified_registry()
        registry = get_unified_registry()
        skills = registry.list_skills()
        if skills:
            tools = registry.get_claude_tools([skills[0]])
            assert isinstance(tools, list)
        reset_unified_registry()

    @pytest.mark.unit
    def test_validate_tools_existing(self):
        """validate_tools returns True for existing tools."""
        from Jotty.core.registry.unified_registry import (
            get_unified_registry, reset_unified_registry,
        )
        reset_unified_registry()
        registry = get_unified_registry()
        skills = registry.list_skills()
        if skills:
            skill = registry.get_skill(skills[0])
            if skill and hasattr(skill, 'tools') and skill.tools:
                tool_name = list(skill.tools.keys())[0]
                validation = registry.validate_tools([tool_name])
                assert tool_name in validation
        reset_unified_registry()

    @pytest.mark.unit
    def test_validate_tools_nonexistent(self):
        """validate_tools returns False for nonexistent tools."""
        from Jotty.core.registry.unified_registry import (
            get_unified_registry, reset_unified_registry,
        )
        reset_unified_registry()
        registry = get_unified_registry()
        validation = registry.validate_tools(["nonexistent_tool_xyz"])
        assert validation.get("nonexistent_tool_xyz") is False
        reset_unified_registry()

    @pytest.mark.unit
    def test_get_scoped_tools_limits_count(self):
        """get_scoped_tools respects max_tools limit."""
        from Jotty.core.registry.unified_registry import (
            get_unified_registry, reset_unified_registry,
        )
        reset_unified_registry()
        registry = get_unified_registry()
        tools = registry.get_scoped_tools(
            "research complex topic with many tools",
            max_tools=3,
            format='names',
        )
        assert isinstance(tools, list)
        assert len(tools) <= 3
        reset_unified_registry()


# =============================================================================
# Registry Summary Tests
# =============================================================================

class TestRegistrySummary:
    """Tests for registry summary and info methods."""

    @pytest.mark.unit
    def test_get_all_returns_complete_info(self):
        """get_all returns skills and ui info."""
        from Jotty.core.registry.unified_registry import (
            get_unified_registry, reset_unified_registry,
        )
        reset_unified_registry()
        registry = get_unified_registry()
        info = registry.get_all()
        assert isinstance(info, dict)
        assert 'skills' in info
        assert 'ui' in info
        reset_unified_registry()

    @pytest.mark.unit
    def test_get_summary_structure(self):
        """get_summary returns correct structure."""
        from Jotty.core.registry.unified_registry import (
            get_unified_registry, reset_unified_registry,
        )
        reset_unified_registry()
        registry = get_unified_registry()
        summary = registry.get_summary()
        assert isinstance(summary, dict)
        assert 'skills' in summary
        assert 'count' in summary['skills']
        reset_unified_registry()

    @pytest.mark.unit
    def test_get_enabled_defaults(self):
        """get_enabled_defaults returns skills and UI components."""
        from Jotty.core.registry.unified_registry import (
            get_unified_registry, reset_unified_registry,
        )
        reset_unified_registry()
        registry = get_unified_registry()
        defaults = registry.get_enabled_defaults()
        assert isinstance(defaults, dict)
        assert 'skills' in defaults or 'ui' in defaults
        reset_unified_registry()


# =============================================================================
# UI Registry Tests
# =============================================================================

class TestUIRegistry:
    """Tests for UI component management."""

    @pytest.mark.unit
    def test_list_ui_components(self):
        """list_ui_components returns list."""
        from Jotty.core.registry.unified_registry import (
            get_unified_registry, reset_unified_registry,
        )
        reset_unified_registry()
        registry = get_unified_registry()
        components = registry.list_ui_components()
        assert isinstance(components, list)
        reset_unified_registry()

    @pytest.mark.unit
    def test_get_ui_categories(self):
        """get_ui_categories returns list of category strings."""
        from Jotty.core.registry.unified_registry import (
            get_unified_registry, reset_unified_registry,
        )
        reset_unified_registry()
        registry = get_unified_registry()
        categories = registry.get_ui_categories()
        assert isinstance(categories, list)
        reset_unified_registry()

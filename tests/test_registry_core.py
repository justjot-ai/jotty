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


# =============================================================================
# CompositeSkill Tests
# =============================================================================

try:
    from Jotty.core.registry.composite_skill import (
        CompositeSkill, ExecutionMode, create_composite_skill,
    )
    COMPOSITE_AVAILABLE = True
except ImportError:
    COMPOSITE_AVAILABLE = False

try:
    from Jotty.core.registry.api import RegistryAPI
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

skipif_composite = pytest.mark.skipif(not COMPOSITE_AVAILABLE, reason="composite_skill not importable")
skipif_api = pytest.mark.skipif(not API_AVAILABLE, reason="registry api not importable")


@pytest.mark.unit
@skipif_composite
class TestExecutionModeEnum:
    """Tests for ExecutionMode enum."""

    def test_sequential_value(self):
        assert ExecutionMode.SEQUENTIAL.value == "sequential"

    def test_parallel_value(self):
        assert ExecutionMode.PARALLEL.value == "parallel"

    def test_mixed_value(self):
        assert ExecutionMode.MIXED.value == "mixed"


@pytest.mark.unit
@skipif_composite
class TestCompositeSkillInit:
    """Tests for CompositeSkill initialization."""

    def test_basic_creation(self):
        skill = CompositeSkill(
            name="test-skill",
            description="Test composite",
            steps=[{"skill_name": "a", "tool_name": "b"}],
        )
        assert skill.name == "test-skill"
        assert skill.description == "Test composite"
        assert len(skill.steps) == 1
        assert skill.execution_mode == ExecutionMode.SEQUENTIAL

    def test_with_parallel_mode(self):
        skill = CompositeSkill(
            name="parallel-skill",
            description="Runs in parallel",
            steps=[],
            execution_mode=ExecutionMode.PARALLEL,
        )
        assert skill.execution_mode == ExecutionMode.PARALLEL


@pytest.mark.unit
@skipif_composite
class TestCompositeSkillSequential:
    """Tests for sequential execution."""

    def _make_registry(self, tool_results):
        """Create mock registry with async tools returning given results."""
        registry = MagicMock()
        skill = MagicMock()
        tools = {}
        for name, result in tool_results.items():
            async def _tool(params, r=result):
                return r
            tools[name] = _tool
        skill.tools = tools
        registry.get_skill.return_value = skill
        return registry

    @pytest.mark.asyncio
    async def test_single_step_success(self):
        registry = self._make_registry({"do_thing": {"success": True, "data": "ok"}})
        skill = CompositeSkill(
            name="test", description="test",
            steps=[{"skill_name": "s1", "tool_name": "do_thing"}],
        )
        result = await skill.execute({"input": "val"}, registry)
        assert result["_success"] is True

    @pytest.mark.asyncio
    async def test_step_failure_stops(self):
        registry = self._make_registry({
            "bad": {"success": False, "error": "oops"},
            "good": {"success": True},
        })
        skill = CompositeSkill(
            name="fail", description="test",
            steps=[
                {"skill_name": "s1", "tool_name": "bad", "required": True},
                {"skill_name": "s1", "tool_name": "good"},
            ],
        )
        result = await skill.execute({}, registry)
        assert result["_success"] is False

    @pytest.mark.asyncio
    async def test_custom_output_key(self):
        registry = self._make_registry({"tool": {"success": True}})
        skill = CompositeSkill(
            name="keyed", description="test",
            steps=[{"skill_name": "s1", "tool_name": "tool", "output_key": "my_result"}],
        )
        result = await skill.execute({}, registry)
        assert "my_result" in result


@pytest.mark.unit
@skipif_composite
class TestCompositeSkillErrors:
    """Tests for error handling in CompositeSkill."""

    @pytest.mark.asyncio
    async def test_skill_not_found(self):
        registry = MagicMock()
        registry.get_skill.return_value = None
        skill = CompositeSkill(
            name="noskill", description="test",
            steps=[{"skill_name": "missing", "tool_name": "tool"}],
        )
        result = await skill.execute({}, registry)
        assert result["step_0"]["success"] is False

    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        mock_skill = MagicMock()
        mock_skill.tools = {}
        registry = MagicMock()
        registry.get_skill.return_value = mock_skill
        skill = CompositeSkill(
            name="notool", description="test",
            steps=[{"skill_name": "s1", "tool_name": "missing_tool"}],
        )
        result = await skill.execute({}, registry)
        assert result["step_0"]["success"] is False

    @pytest.mark.asyncio
    async def test_tool_exception_caught(self):
        mock_skill = MagicMock()
        async def _failing(params):
            raise RuntimeError("kaboom")
        mock_skill.tools = {"explode": _failing}
        registry = MagicMock()
        registry.get_skill.return_value = mock_skill
        skill = CompositeSkill(
            name="exploding", description="test",
            steps=[{"skill_name": "s1", "tool_name": "explode"}],
        )
        result = await skill.execute({}, registry)
        assert result["step_0"]["success"] is False
        assert "kaboom" in result["step_0"]["error"]


@pytest.mark.unit
@skipif_composite
class TestCreateCompositeSkillFactory:
    """Tests for create_composite_skill factory."""

    def test_returns_composite_skill(self):
        skill = create_composite_skill(
            name="factory-test",
            description="Made by factory",
            steps=[{"skill_name": "s", "tool_name": "t"}],
        )
        assert isinstance(skill, CompositeSkill)
        assert skill.name == "factory-test"


# =============================================================================
# RegistryAPI Tests
# =============================================================================

@pytest.mark.unit
@skipif_api
class TestRegistryAPIEndpoints:
    """Tests for RegistryAPI endpoints."""

    def _make_api(self):
        registry = MagicMock()
        registry.get_all.return_value = {"tools": [], "widgets": []}
        registry.get_tools.return_value = [{"name": "tool1"}]
        registry.get_widgets.return_value = [{"name": "widget1"}]
        registry.get_tool.return_value = {"name": "tool1", "desc": "A tool"}
        registry.get_widget.return_value = {"name": "w1"}
        registry.validate_tools.return_value = {"valid": ["t1"], "invalid": []}
        registry.validate_widgets.return_value = {"valid": ["w1"], "invalid": []}
        registry.get_enabled_defaults.return_value = {"tools": ["t1"]}
        return RegistryAPI(registry), registry

    def test_get_all(self):
        api, _ = self._make_api()
        result = api.get_all()
        assert result["success"] is True
        assert "data" in result

    def test_get_tools(self):
        api, _ = self._make_api()
        result = api.get_tools()
        assert result["success"] is True

    def test_get_widgets(self):
        api, _ = self._make_api()
        result = api.get_widgets()
        assert result["success"] is True

    def test_get_tool_found(self):
        api, _ = self._make_api()
        result = api.get_tool("tool1")
        assert result["success"] is True

    def test_get_tool_not_found(self):
        api, registry = self._make_api()
        registry.get_tool.return_value = None
        result = api.get_tool("missing")
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_get_widget_found(self):
        api, _ = self._make_api()
        result = api.get_widget("w1")
        assert result["success"] is True

    def test_get_widget_not_found(self):
        api, registry = self._make_api()
        registry.get_widget.return_value = None
        result = api.get_widget("missing")
        assert result["success"] is False

    def test_validate_tools(self):
        api, _ = self._make_api()
        result = api.validate_tools(["t1"])
        assert result["success"] is True

    def test_validate_widgets(self):
        api, _ = self._make_api()
        result = api.validate_widgets(["w1"])
        assert result["success"] is True

    def test_get_defaults(self):
        api, _ = self._make_api()
        result = api.get_defaults()
        assert result["success"] is True

    def test_get_skills(self):
        api, _ = self._make_api()
        with patch("Jotty.core.registry.api.get_skills_registry") as mock_sr:
            mock_reg = MagicMock()
            mock_reg.list_skills.return_value = [{"name": "s1"}]
            mock_reg.get_registered_tools.return_value = {"tool1": "func"}
            mock_sr.return_value = mock_reg
            result = api.get_skills()
            assert result["success"] is True

    def test_get_skill_found(self):
        api, _ = self._make_api()
        with patch("Jotty.core.registry.api.get_skills_registry") as mock_sr:
            mock_reg = MagicMock()
            mock_skill = MagicMock()
            mock_skill.name = "test-skill"
            mock_skill.description = "A test"
            mock_skill.tools = {"t1": "func"}
            mock_skill.metadata = {}
            mock_reg.get_skill.return_value = mock_skill
            mock_sr.return_value = mock_reg
            result = api.get_skill("test-skill")
            assert result["success"] is True

    def test_get_skill_not_found(self):
        api, _ = self._make_api()
        with patch("Jotty.core.registry.api.get_skills_registry") as mock_sr:
            mock_reg = MagicMock()
            mock_reg.get_skill.return_value = None
            mock_sr.return_value = mock_reg
            result = api.get_skill("missing")
            assert result["success"] is False


# =============================================================================
# UnifiedRegistry Isolated Tests (Mocked Sub-Registries)
# =============================================================================

@pytest.mark.unit
class TestUnifiedRegistryIsolated:
    """Tests for UnifiedRegistry with fully mocked sub-registries."""

    def _make_registry(self):
        """Create UnifiedRegistry with mock skills and UI registries."""
        from Jotty.core.registry.unified_registry import UnifiedRegistry

        mock_skills = MagicMock()
        ws_skill = MagicMock()
        ws_skill.name = "web-search"
        ws_skill.tools = {"search_tool": lambda p: p}
        ws_skill.mcp_enabled = True
        ws_skill.to_claude_tools = lambda: [{"name": "search_tool", "description": "Search"}]
        ws_skill.to_dict = lambda: {"name": "web-search"}
        ws_skill.list_tools = lambda: ["search_tool"]

        calc_skill = MagicMock()
        calc_skill.name = "calculator"
        calc_skill.tools = {"calc_tool": lambda p: p}
        calc_skill.mcp_enabled = False
        calc_skill.to_claude_tools = lambda: [{"name": "calc_tool", "description": "Calculate"}]
        calc_skill.to_dict = lambda: {"name": "calculator"}
        calc_skill.list_tools = lambda: ["calc_tool"]

        mock_skills.loaded_skills = {
            "web-search": ws_skill,
            "calculator": calc_skill,
        }
        mock_skills.get_skill = lambda name: mock_skills.loaded_skills.get(name)
        mock_skills.discover = MagicMock(return_value=[
            {"name": "web-search", "relevance_score": 5},
            {"name": "calculator", "relevance_score": 0},
        ])

        mock_ui = MagicMock()
        mock_ui.list_types.return_value = ["chart", "data-table", "text", "code"]
        mock_ui.get_categories.return_value = ["display", "input"]
        mock_ui.get.return_value = MagicMock(to_dict=lambda: {"type": "chart"})
        mock_ui.convert_to_a2ui.return_value = [{"type": "chart", "data": {}}]
        mock_ui.convert_to_agui.return_value = {"type": "chart"}
        mock_ui.to_api_response.return_value = {"components": 4}
        mock_ui.get_with_adapters.return_value = ["chart", "data-table"]

        return UnifiedRegistry(skills_registry=mock_skills, ui_registry=mock_ui)

    def test_skills_property(self):
        """skills property returns the skills registry."""
        reg = self._make_registry()
        assert reg.skills is not None
        assert "web-search" in reg.skills.loaded_skills

    def test_ui_property(self):
        """ui property returns the UI registry."""
        reg = self._make_registry()
        assert reg.ui is not None

    def test_tools_legacy_alias(self):
        """tools property is legacy alias for skills."""
        reg = self._make_registry()
        assert reg.tools is reg.skills

    def test_widgets_legacy_alias(self):
        """widgets property is legacy alias for ui."""
        reg = self._make_registry()
        assert reg.widgets is reg.ui

    def test_list_skills(self):
        """list_skills returns all skill names."""
        reg = self._make_registry()
        skills = reg.list_skills()
        assert set(skills) == {"web-search", "calculator"}

    def test_get_skill_found(self):
        """get_skill returns skill for existing name."""
        reg = self._make_registry()
        skill = reg.get_skill("web-search")
        assert skill is not None
        assert skill.name == "web-search"

    def test_get_skill_not_found(self):
        """get_skill returns None for nonexistent name."""
        reg = self._make_registry()
        assert reg.get_skill("nonexistent") is None

    def test_get_tool_found(self):
        """get_tool returns callable for existing skill+tool."""
        reg = self._make_registry()
        tool = reg.get_tool("web-search", "search_tool")
        assert tool is not None
        assert callable(tool)

    def test_get_tool_skill_not_found(self):
        """get_tool returns None when skill doesn't exist."""
        reg = self._make_registry()
        assert reg.get_tool("nonexistent", "tool") is None

    def test_get_tool_tool_not_found(self):
        """get_tool returns None when tool doesn't exist in skill."""
        reg = self._make_registry()
        assert reg.get_tool("web-search", "nonexistent_tool") is None

    def test_get_claude_tools_all(self):
        """get_claude_tools without filter returns tools from all skills."""
        reg = self._make_registry()
        tools = reg.get_claude_tools()
        assert len(tools) == 2
        names = {t["name"] for t in tools}
        assert "search_tool" in names
        assert "calc_tool" in names

    def test_get_claude_tools_filtered(self):
        """get_claude_tools with skill_names filters correctly."""
        reg = self._make_registry()
        tools = reg.get_claude_tools(["web-search"])
        assert len(tools) == 1
        assert tools[0]["name"] == "search_tool"

    def test_get_mcp_tools(self):
        """get_mcp_tools returns only MCP-enabled skill tools."""
        reg = self._make_registry()
        tools = reg.get_mcp_tools()
        names = {t["name"] for t in tools}
        assert "search_tool" in names
        assert "calc_tool" not in names

    def test_list_ui_components(self):
        """list_ui_components delegates to UI registry."""
        reg = self._make_registry()
        components = reg.list_ui_components()
        assert "chart" in components
        assert "data-table" in components

    def test_get_ui_component(self):
        """get_ui_component delegates to UI registry."""
        reg = self._make_registry()
        component = reg.get_ui_component("chart")
        assert component is not None

    def test_get_ui_categories(self):
        """get_ui_categories delegates to UI registry."""
        reg = self._make_registry()
        cats = reg.get_ui_categories()
        assert "display" in cats

    def test_convert_to_a2ui(self):
        """convert_to_a2ui delegates to UI registry."""
        reg = self._make_registry()
        blocks = reg.convert_to_a2ui("chart", {"x": [1], "y": [2]})
        assert isinstance(blocks, list)
        assert len(blocks) == 1

    def test_convert_to_agui(self):
        """convert_to_agui delegates to UI registry."""
        reg = self._make_registry()
        result = reg.convert_to_agui("chart", {"x": [1]})
        assert isinstance(result, dict)

    def test_get_all_structure(self):
        """get_all returns skills and ui sections."""
        reg = self._make_registry()
        data = reg.get_all()
        assert "skills" in data
        assert "ui" in data
        assert data["skills"]["count"] == 2
        assert len(data["skills"]["available"]) == 2

    def test_get_tools_legacy(self):
        """get_tools returns skills data (legacy)."""
        reg = self._make_registry()
        data = reg.get_tools()
        assert "available" in data
        assert data["count"] == 2

    def test_get_widgets_legacy(self):
        """get_widgets delegates to UI API response."""
        reg = self._make_registry()
        data = reg.get_widgets()
        assert data == {"components": 4}

    def test_validate_tools_mixed(self):
        """validate_tools checks existence of tool names."""
        reg = self._make_registry()
        result = reg.validate_tools(["search_tool", "nonexistent"])
        assert result["search_tool"] is True
        assert result["nonexistent"] is False

    def test_validate_widgets(self):
        """validate_widgets checks existence of UI component types."""
        reg = self._make_registry()
        result = reg.validate_widgets(["chart", "nonexistent"])
        assert result["chart"] is True
        assert result["nonexistent"] is False

    def test_get_enabled_defaults(self):
        """get_enabled_defaults returns skills and common UI."""
        reg = self._make_registry()
        defaults = reg.get_enabled_defaults()
        assert "skills" in defaults
        assert "ui_components" in defaults
        assert set(defaults["skills"]) == {"web-search", "calculator"}

    def test_discover_for_task(self):
        """discover_for_task returns skills and UI suggestions."""
        reg = self._make_registry()
        result = reg.discover_for_task("search the web for data")
        assert "skills" in result
        assert "ui" in result
        assert "task" in result
        assert result["task"] == "search the web for data"

    def test_discover_for_task_ui_matching(self):
        """discover_for_task matches UI keywords in task."""
        reg = self._make_registry()
        result = reg.discover_for_task("create a chart showing trends")
        assert len(result["ui"]) >= 1

    def test_get_scoped_tools_names(self):
        """get_scoped_tools with format='names' returns skill names."""
        reg = self._make_registry()
        names = reg.get_scoped_tools("search web", max_tools=5, format='names')
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_get_scoped_tools_claude(self):
        """get_scoped_tools with format='claude' returns tool dicts."""
        reg = self._make_registry()
        tools = reg.get_scoped_tools("search web", max_tools=5, format='claude')
        assert isinstance(tools, list)
        assert all(isinstance(t, dict) for t in tools)

    def test_get_scoped_tools_full(self):
        """get_scoped_tools with format='full' returns skill objects."""
        reg = self._make_registry()
        skills = reg.get_scoped_tools("search web", max_tools=5, format='full')
        assert isinstance(skills, list)

    def test_get_scoped_tools_max_limit(self):
        """get_scoped_tools respects max_tools."""
        reg = self._make_registry()
        tools = reg.get_scoped_tools("search", max_tools=1, format='names')
        assert len(tools) <= 1

    def test_get_scoped_tools_fallback_empty_discovery(self):
        """get_scoped_tools falls back when discovery returns empty."""
        reg = self._make_registry()
        reg._skills.discover.return_value = []
        names = reg.get_scoped_tools("anything", max_tools=5, format='names')
        assert isinstance(names, list)
        assert len(names) > 0  # Falls back to first N skills

    def test_get_summary_structure(self):
        """get_summary returns correct nested structure."""
        reg = self._make_registry()
        summary = reg.get_summary()
        assert summary["skills"]["count"] == 2
        assert "names" in summary["skills"]
        assert "has_more" in summary["skills"]
        assert summary["ui"]["count"] == 4
        assert "categories" in summary["ui"]
        assert "with_adapters" in summary["ui"]


# =============================================================================
# Legacy Functions Tests
# =============================================================================

@pytest.mark.unit
class TestLegacyFunctions:
    """Tests for deprecated get_tools_registry and get_widget_registry."""

    def test_get_tools_registry_returns_skills(self):
        """get_tools_registry returns skills registry with deprecation warning."""
        from Jotty.core.registry.unified_registry import (
            get_tools_registry, reset_unified_registry,
        )
        reset_unified_registry()
        import logging
        with patch.object(logging.getLogger("Jotty.core.registry.unified_registry"), "warning"):
            result = get_tools_registry()
        assert result is not None
        reset_unified_registry()

    def test_get_widget_registry_returns_ui(self):
        """get_widget_registry returns UI registry with deprecation warning."""
        from Jotty.core.registry.unified_registry import (
            get_widget_registry, reset_unified_registry,
        )
        reset_unified_registry()
        import logging
        with patch.object(logging.getLogger("Jotty.core.registry.unified_registry"), "warning"):
            result = get_widget_registry()
        assert result is not None
        reset_unified_registry()

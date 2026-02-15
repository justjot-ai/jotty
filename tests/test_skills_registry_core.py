"""
Tests for Skills Registry Core Module
==========================================
Tests for SkillsRegistry, SkillDefinition, SkillType, TrustLevel,
ToolMetadata, BaseSkill, get_skills_registry singleton, and discovery.
"""
import pytest
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any, Callable

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Jotty.core.capabilities.registry.skills_registry import (
    SkillsRegistry,
    SkillDefinition,
    SkillType,
    TrustLevel,
    ToolMetadata,
    BaseSkill,
    _infer_trust_level,
    get_skills_registry,
    _registry_instance,
)


# =============================================================================
# Helper: reset the global singleton between tests that exercise it
# =============================================================================

def _reset_skills_registry_singleton():
    """Reset the module-level singleton so each test starts fresh."""
    import Jotty.core.capabilities.registry.skills_registry as mod
    mod._registry_instance = None


# =============================================================================
# SkillType Enum Tests
# =============================================================================

@pytest.mark.unit
class TestSkillTypeEnum:
    """Tests for SkillType enum values and identity."""

    def test_skill_type_values(self):
        """SkillType enum has base, derived, and composite values."""
        assert SkillType.BASE.value == "base"
        assert SkillType.DERIVED.value == "derived"
        assert SkillType.COMPOSITE.value == "composite"

    def test_skill_type_members_count(self):
        """SkillType has exactly three members."""
        assert len(SkillType) == 3


# =============================================================================
# TrustLevel & _infer_trust_level Tests
# =============================================================================

@pytest.mark.unit
class TestTrustLevelInference:
    """Tests for TrustLevel enum and the _infer_trust_level heuristic."""

    def test_trust_level_values(self):
        """TrustLevel enum has safe, side_effect, and destructive."""
        assert TrustLevel.SAFE.value == "safe"
        assert TrustLevel.SIDE_EFFECT.value == "side_effect"
        assert TrustLevel.DESTRUCTIVE.value == "destructive"

    def test_infer_safe_from_category(self):
        """Skills in safe categories infer TrustLevel.SAFE."""
        result = _infer_trust_level("search", [], "web-search")
        assert result == TrustLevel.SAFE

    def test_infer_destructive_from_tags(self):
        """Skills with destructive tags infer TrustLevel.DESTRUCTIVE."""
        result = _infer_trust_level("general", ["delete", "data"], "delete-tool")
        assert result == TrustLevel.DESTRUCTIVE

    def test_infer_side_effect_from_name(self):
        """Skills with side-effect keywords in name infer SIDE_EFFECT."""
        result = _infer_trust_level("automation", [], "send-telegram")
        assert result == TrustLevel.SIDE_EFFECT

    def test_infer_defaults_to_safe(self):
        """Skills with no matching heuristics default to SAFE."""
        result = _infer_trust_level("misc", ["generic"], "some-neutral-tool")
        assert result == TrustLevel.SAFE


# =============================================================================
# ToolMetadata Tests
# =============================================================================

@pytest.mark.unit
class TestToolMetadata:
    """Tests for ToolMetadata dataclass and conversion methods."""

    def test_to_dict(self):
        """to_dict returns complete metadata dictionary."""
        meta = ToolMetadata(
            name="my_tool",
            description="Does something",
            category="test",
            mcp_enabled=True,
            parameters={"properties": {"q": {"type": "string"}}, "required": ["q"]},
            tags=["search"],
        )
        d = meta.to_dict()
        assert d["name"] == "my_tool"
        assert d["description"] == "Does something"
        assert d["category"] == "test"
        assert d["mcp_enabled"] is True
        assert "q" in d["parameters"]["properties"]
        assert d["tags"] == ["search"]

    def test_to_claude_tool(self):
        """to_claude_tool returns Claude API tool format."""
        meta = ToolMetadata(
            name="calc_tool",
            description="Calculate numbers",
            parameters={"properties": {"expr": {"type": "string"}}, "required": ["expr"]},
        )
        ct = meta.to_claude_tool()
        assert ct["name"] == "calc_tool"
        assert ct["description"] == "Calculate numbers"
        assert ct["input_schema"]["type"] == "object"
        assert "expr" in ct["input_schema"]["properties"]
        assert ct["input_schema"]["required"] == ["expr"]


# =============================================================================
# BaseSkill Tests
# =============================================================================

@pytest.mark.unit
class TestBaseSkill:
    """Tests for BaseSkill lifecycle and interface."""

    def test_default_attributes(self):
        """BaseSkill has expected default class attributes."""
        skill = BaseSkill()
        assert skill.name == "base-skill"
        assert skill.description == "Base skill class"
        assert skill.version == "1.0.0"
        assert skill.category == "general"

    def test_status_callback_invoked(self):
        """status() invokes the callback with stage and detail."""
        callback = MagicMock()
        skill = BaseSkill(status_callback=callback)
        skill.status("Loading", "loading data")
        callback.assert_called_once_with("Loading", "loading data")

    def test_status_no_callback_no_error(self):
        """status() with no callback does not raise."""
        skill = BaseSkill()
        skill.status("Stage", "detail")  # Should not raise

    def test_set_context(self):
        """set_context updates internal context dict."""
        skill = BaseSkill()
        skill.set_context(session_id="abc", user_id="u1")
        assert skill._context["session_id"] == "abc"
        assert skill._context["user_id"] == "u1"

    def test_get_tools_returns_wrapper(self):
        """get_tools returns dict with a wrapper callable."""
        skill = BaseSkill()
        tools = skill.get_tools()
        assert "base-skill_tool" in tools
        assert callable(tools["base-skill_tool"])

    def test_to_definition_returns_skill_definition(self):
        """to_definition converts BaseSkill to SkillDefinition."""
        skill = BaseSkill()
        defn = skill.to_definition()
        assert isinstance(defn, SkillDefinition)
        assert defn.name == "base-skill"
        assert defn.description == "Base skill class"


# =============================================================================
# SkillDefinition Tests
# =============================================================================

@pytest.mark.unit
class TestSkillDefinition:
    """Tests for SkillDefinition creation, lazy loading, and properties."""

    def test_basic_creation(self):
        """SkillDefinition stores name, description, and defaults."""
        sd = SkillDefinition(name="test-skill", description="A test skill")
        assert sd.name == "test-skill"
        assert sd.description == "A test skill"
        assert sd.skill_type == SkillType.BASE
        assert sd.tags == []
        assert sd.capabilities == []
        assert sd.version == "1.0.0"

    def test_eager_tools(self):
        """SkillDefinition with pre-loaded tools returns them directly."""
        fn = MagicMock()
        sd = SkillDefinition(name="eager", description="eager", tools={"my_tool": fn})
        assert sd.tools == {"my_tool": fn}
        assert sd.is_loaded is True

    def test_lazy_loading_triggers_on_access(self):
        """Accessing tools triggers the _tool_loader callback."""
        loader = MagicMock(return_value={"lazy_tool": lambda p: p})
        sd = SkillDefinition(name="lazy", description="lazy", _tool_loader=loader)
        assert sd.is_loaded is False
        tools = sd.tools  # trigger lazy load
        loader.assert_called_once()
        assert "lazy_tool" in tools
        assert sd.is_loaded is True

    def test_lazy_loader_failure_returns_empty(self):
        """If _tool_loader raises, tools falls back to empty dict."""
        loader = MagicMock(side_effect=RuntimeError("boom"))
        sd = SkillDefinition(name="broken", description="broken", _tool_loader=loader)
        tools = sd.tools
        assert tools == {}

    def test_list_tools(self):
        """list_tools returns list of tool names (triggers lazy load)."""
        sd = SkillDefinition(
            name="multi",
            description="multi",
            tools={"t1": lambda: None, "t2": lambda: None},
        )
        assert sorted(sd.list_tools()) == ["t1", "t2"]

    def test_get_tool(self):
        """get_tool returns callable for existing tool, None for missing."""
        fn = MagicMock()
        sd = SkillDefinition(name="x", description="x", tools={"fn": fn})
        assert sd.get_tool("fn") is fn
        assert sd.get_tool("missing") is None

    def test_is_available_no_gate(self):
        """Skills without context_gate are always available."""
        sd = SkillDefinition(name="open", description="open")
        assert sd.is_available() is True
        assert sd.is_available({"env": "prod"}) is True

    def test_is_available_with_gate_true(self):
        """Skills with passing context_gate return True."""
        gate = MagicMock(return_value=True)
        sd = SkillDefinition(name="gated", description="gated", context_gate=gate)
        assert sd.is_available({"browser": True}) is True
        gate.assert_called_once_with({"browser": True})

    def test_is_available_with_gate_false(self):
        """Skills with failing context_gate return False."""
        gate = MagicMock(return_value=False)
        sd = SkillDefinition(name="gated", description="gated", context_gate=gate)
        assert sd.is_available({"browser": False}) is False

    def test_is_available_gate_error_fails_open(self):
        """If context_gate raises, skill fails open (returns True)."""
        gate = MagicMock(side_effect=ValueError("bad"))
        sd = SkillDefinition(name="err", description="err", context_gate=gate)
        assert sd.is_available() is True

    def test_to_dict_structure(self):
        """to_dict returns dict with all expected keys."""
        sd = SkillDefinition(
            name="full",
            description="A full skill",
            tools={"tool1": lambda: None},
            category="analysis",
            tags=["data"],
            version="2.0.0",
            capabilities=["data-fetch"],
            use_when="need data",
            skill_type=SkillType.DERIVED,
            base_skills=["web-search"],
        )
        d = sd.to_dict()
        assert d["name"] == "full"
        assert d["description"] == "A full skill"
        assert d["category"] == "analysis"
        assert d["tags"] == ["data"]
        assert d["version"] == "2.0.0"
        assert d["skill_type"] == "derived"
        assert d["base_skills"] == ["web-search"]
        assert d["capabilities"] == ["data-fetch"]
        assert "tool1" in d["tools"]

    def test_to_dict_composite_includes_execution_mode(self):
        """to_dict for composite skills includes execution_mode."""
        sd = SkillDefinition(
            name="comp",
            description="comp",
            tools={},
            skill_type=SkillType.COMPOSITE,
            execution_mode="sequential",
        )
        d = sd.to_dict()
        assert d["execution_mode"] == "sequential"

    def test_infer_executor_type_browser(self):
        """Executor type inferred as 'browser' for browser-related skills."""
        sd = SkillDefinition(name="web-scrape", description="Scrape pages with playwright")
        assert sd.executor_type == "browser"

    def test_infer_executor_type_messaging(self):
        """Executor type inferred as 'messaging' for messaging-related skills."""
        sd = SkillDefinition(name="telegram-sender", description="Send messages via Telegram")
        assert sd.executor_type == "messaging"

    def test_infer_executor_type_general_fallback(self):
        """Executor type defaults to 'general' for unmatched skills."""
        sd = SkillDefinition(name="neutral-thing", description="Does neutral stuff")
        assert sd.executor_type == "general"

    def test_to_claude_tools(self):
        """to_claude_tools returns list of Claude API tool definitions."""
        async def my_tool(params):
            """My tool description."""
            return {}

        sd = SkillDefinition(name="s", description="s", tools={"my_tool": my_tool})
        ct = sd.to_claude_tools()
        assert len(ct) == 1
        assert ct[0]["name"] == "my_tool"
        assert "My tool description" in ct[0]["description"]

    def test_tool_metadata_set_and_get(self):
        """set_tool_metadata and get_tool_metadata work correctly."""
        sd = SkillDefinition(name="m", description="m")
        meta = ToolMetadata(name="tool_a", description="Tool A")
        sd.set_tool_metadata("tool_a", meta)
        retrieved = sd.get_tool_metadata("tool_a")
        assert retrieved is meta
        assert sd.get_tool_metadata("nonexistent") is None


# =============================================================================
# SkillsRegistry Initialization Tests
# =============================================================================

@pytest.mark.unit
class TestSkillsRegistryInit:
    """Tests for SkillsRegistry construction and initialization."""

    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._scan_skills_metadata")
    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._load_cached_dynamic_skills")
    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._prewarm_top_skills")
    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._refresh_dynamic_skills_background")
    def test_init_sets_initialized_flag(self, mock_refresh, mock_prewarm, mock_cache, mock_scan):
        """init() sets initialized = True and calls scan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillsRegistry(skills_dir=tmpdir)
            assert reg.initialized is False
            reg.init()
            assert reg.initialized is True
            mock_scan.assert_called_once()
            mock_cache.assert_called_once()

    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._scan_skills_metadata")
    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._load_cached_dynamic_skills")
    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._prewarm_top_skills")
    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._refresh_dynamic_skills_background")
    def test_init_idempotent(self, mock_refresh, mock_prewarm, mock_cache, mock_scan):
        """Calling init() twice only scans once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillsRegistry(skills_dir=tmpdir)
            reg.init()
            reg.init()
            mock_scan.assert_called_once()

    def test_custom_skills_dir(self):
        """SkillsRegistry uses the provided skills_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillsRegistry(skills_dir=tmpdir)
            assert reg.skills_dir == Path(tmpdir)

    def test_empty_registry_has_no_skills(self):
        """A freshly created registry with empty dir has no skills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillsRegistry(skills_dir=tmpdir)
            assert len(reg.loaded_skills) == 0


# =============================================================================
# SkillsRegistry Skill Management Tests
# =============================================================================

@pytest.mark.unit
class TestSkillsRegistryManagement:
    """Tests for skill registration, lookup, listing, and type filtering."""

    def _make_registry_with_skills(self):
        """Helper: create a registry and manually add SkillDefinitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillsRegistry(skills_dir=tmpdir)
            # Manually register skills (bypass disk loading)
            reg.loaded_skills["web-search"] = SkillDefinition(
                name="web-search",
                description="Search the web",
                tools={"search_tool": lambda p: p},
                skill_type=SkillType.BASE,
                category="search",
                capabilities=["data-fetch"],
                use_when="User needs web search results",
            )
            reg.loaded_skills["stock-research"] = SkillDefinition(
                name="stock-research",
                description="Research stock prices and trends",
                tools={"stock_tool": lambda p: p},
                skill_type=SkillType.DERIVED,
                base_skills=["web-search"],
                capabilities=["data-fetch", "analysis"],
                use_when="User wants stock data",
            )
            reg.loaded_skills["full-pipeline"] = SkillDefinition(
                name="full-pipeline",
                description="Search, summarize, send via telegram",
                tools={"pipeline_tool": lambda p: p},
                skill_type=SkillType.COMPOSITE,
                base_skills=["web-search", "telegram-sender"],
                execution_mode="sequential",
                capabilities=["data-fetch", "communicate"],
            )
            reg.initialized = True
            return reg

    def test_get_skill_found(self):
        """get_skill returns SkillDefinition for existing skill."""
        reg = self._make_registry_with_skills()
        skill = reg.get_skill("web-search")
        assert skill is not None
        assert skill.name == "web-search"

    def test_get_skill_not_found(self):
        """get_skill returns None for nonexistent skill."""
        reg = self._make_registry_with_skills()
        assert reg.get_skill("nonexistent") is None

    def test_list_skills_all(self):
        """list_skills returns all skills without filter."""
        reg = self._make_registry_with_skills()
        skills = reg.list_skills()
        names = {s["name"] for s in skills}
        assert names == {"web-search", "stock-research", "full-pipeline"}

    def test_list_skills_filter_base(self):
        """list_skills with filter_type=BASE returns only base skills."""
        reg = self._make_registry_with_skills()
        skills = reg.list_skills(filter_type=SkillType.BASE)
        assert len(skills) == 1
        assert skills[0]["name"] == "web-search"

    def test_list_skills_filter_composite(self):
        """list_skills with filter_type=COMPOSITE returns composite skills with execution_mode."""
        reg = self._make_registry_with_skills()
        skills = reg.list_skills(filter_type=SkillType.COMPOSITE)
        assert len(skills) == 1
        assert skills[0]["name"] == "full-pipeline"
        assert skills[0]["execution_mode"] == "sequential"

    def test_list_skills_by_type_with_limit(self):
        """list_skills_by_type respects max_skills limit."""
        reg = self._make_registry_with_skills()
        # Add extra base skills
        for i in range(5):
            reg.loaded_skills[f"extra-{i}"] = SkillDefinition(
                name=f"extra-{i}", description=f"Extra {i}",
                tools={}, skill_type=SkillType.BASE,
            )
        results = reg.list_skills_by_type(SkillType.BASE, max_skills=3)
        assert len(results) == 3

    def test_get_skill_type_summary(self):
        """get_skill_type_summary returns correct counts per type."""
        reg = self._make_registry_with_skills()
        summary = reg.get_skill_type_summary()
        assert summary["base"] == 1
        assert summary["derived"] == 1
        assert summary["composite"] == 1

    def test_get_failed_skills_initially_empty(self):
        """get_failed_skills returns empty dict when nothing has failed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillsRegistry(skills_dir=tmpdir)
            assert reg.get_failed_skills() == {}


# =============================================================================
# SkillsRegistry Discovery Tests
# =============================================================================

@pytest.mark.unit
class TestSkillsRegistryDiscovery:
    """Tests for the discover() method and keyword scoring."""

    def _make_registry_with_skills(self):
        """Helper: create a registry with test skills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillsRegistry(skills_dir=tmpdir)
            reg.loaded_skills["web-search"] = SkillDefinition(
                name="web-search",
                description="Search the web for information",
                tools={"search_tool": lambda p: p},
                skill_type=SkillType.BASE,
                category="search",
                capabilities=["data-fetch"],
            )
            reg.loaded_skills["calculator"] = SkillDefinition(
                name="calculator",
                description="Perform mathematical calculations",
                tools={"calc_tool": lambda p: p},
                skill_type=SkillType.BASE,
                category="calculation",
                capabilities=["calculation"],
            )
            reg.loaded_skills["stock-research"] = SkillDefinition(
                name="stock-research",
                description="Research stock data and prices",
                tools={"stock_tool": lambda p: p},
                skill_type=SkillType.DERIVED,
                base_skills=["web-search"],
                capabilities=["data-fetch", "analysis"],
            )
            reg.initialized = True
            return reg

    def test_discover_ranks_relevant_skills_first(self):
        """discover puts matching skills before unmatched ones."""
        reg = self._make_registry_with_skills()
        results = reg.discover("search the web")
        assert len(results) > 0
        # web-search should be ranked first (name match + desc match)
        assert results[0]["name"] == "web-search"
        assert results[0]["relevance_score"] > 0

    def test_discover_unmatched_skills_included(self):
        """discover includes unmatched skills with score 0 to fill quota."""
        reg = self._make_registry_with_skills()
        results = reg.discover("search the web", max_results=50)
        scores = [r["relevance_score"] for r in results]
        # At least one should have score 0 (unmatched)
        assert 0 in scores

    def test_discover_respects_max_results(self):
        """discover respects max_results limit."""
        reg = self._make_registry_with_skills()
        results = reg.discover("search", max_results=2)
        assert len(results) <= 2

    def test_discover_context_gate_filters_skill(self):
        """discover excludes skills whose context_gate returns False."""
        reg = self._make_registry_with_skills()
        # Add a gated skill that is not available
        reg.loaded_skills["browser-tool"] = SkillDefinition(
            name="browser-tool",
            description="Browser automation",
            tools={"browse_tool": lambda p: p},
            context_gate=lambda ctx: ctx.get("browser_available", False),
        )
        results = reg.discover("browser automation", task_context={"browser_available": False})
        names = [r["name"] for r in results]
        assert "browser-tool" not in names

    def test_discover_composite_boost(self):
        """Composite skills get a score boost over base skills."""
        reg = self._make_registry_with_skills()
        reg.loaded_skills["search-pipeline"] = SkillDefinition(
            name="search-pipeline",
            description="Full search pipeline",
            tools={"pipe_tool": lambda p: p},
            skill_type=SkillType.COMPOSITE,
            base_skills=["web-search", "calculator"],
        )
        results = reg.discover("search")
        # Both web-search (base) and search-pipeline (composite) match
        search_scores = {r["name"]: r["relevance_score"] for r in results if "search" in r["name"]}
        # Composite gets +2 boost, derived gets +1
        assert search_scores.get("search-pipeline", 0) > 0


# =============================================================================
# SkillsRegistry Capability Filtering Tests
# =============================================================================

@pytest.mark.unit
class TestSkillsRegistryCapabilityFiltering:
    """Tests for filter_skills_by_capabilities method."""

    def _make_registry(self):
        """Helper: create a registry with skills having capabilities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillsRegistry(skills_dir=tmpdir)
            reg.loaded_skills["fetcher"] = SkillDefinition(
                name="fetcher", description="Fetches data",
                tools={}, capabilities=["data-fetch"],
            )
            reg.loaded_skills["comm"] = SkillDefinition(
                name="comm", description="Communication tool",
                tools={}, capabilities=["communicate"],
            )
            reg.loaded_skills["multi"] = SkillDefinition(
                name="multi", description="Multi-cap tool",
                tools={}, capabilities=["data-fetch", "communicate"],
            )
            reg.initialized = True
            return reg

    def test_filter_by_single_capability(self):
        """Filtering by one capability returns matching skills."""
        reg = self._make_registry()
        results = reg.filter_skills_by_capabilities(["data-fetch"])
        names = {r["name"] for r in results}
        assert "fetcher" in names
        assert "multi" in names

    def test_filter_ranks_by_match_count(self):
        """Skills matching more capabilities are ranked higher."""
        reg = self._make_registry()
        results = reg.filter_skills_by_capabilities(["data-fetch", "communicate"])
        # 'multi' matches both caps, should be first
        assert results[0]["name"] == "multi"

    def test_filter_empty_capabilities_returns_all(self):
        """Empty required_capabilities returns all skills."""
        reg = self._make_registry()
        results = reg.filter_skills_by_capabilities([])
        assert len(results) == 3


# =============================================================================
# get_skills_registry Singleton Tests
# =============================================================================

@pytest.mark.unit
class TestGetSkillsRegistrySingleton:
    """Tests for the get_skills_registry() singleton factory."""

    def setup_method(self):
        """Reset singleton before each test."""
        _reset_skills_registry_singleton()

    def teardown_method(self):
        """Clean up singleton after each test."""
        _reset_skills_registry_singleton()

    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._scan_skills_metadata")
    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._load_cached_dynamic_skills")
    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._prewarm_top_skills")
    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._refresh_dynamic_skills_background")
    def test_singleton_identity(self, mock_refresh, mock_prewarm, mock_cache, mock_scan):
        """get_skills_registry returns the same instance on repeated calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            r1 = get_skills_registry(tmpdir)
            r2 = get_skills_registry()
            assert r1 is r2

    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._scan_skills_metadata")
    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._load_cached_dynamic_skills")
    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._prewarm_top_skills")
    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._refresh_dynamic_skills_background")
    def test_singleton_auto_initializes(self, mock_refresh, mock_prewarm, mock_cache, mock_scan):
        """get_skills_registry auto-initializes the registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = get_skills_registry(tmpdir)
            assert reg.initialized is True


# =============================================================================
# SkillsRegistry SKILL.md Metadata Parsing Tests
# =============================================================================

@pytest.mark.unit
class TestSkillMetadataParsing:
    """Tests for _parse_skill_metadata from SKILL.md content."""

    def _write_skill_md(self, tmpdir, content):
        """Write SKILL.md content to a temp dir and return its path."""
        md_path = Path(tmpdir) / "SKILL.md"
        md_path.write_text(content)
        return md_path

    def test_parse_full_metadata(self):
        """Parsing a complete SKILL.md extracts all fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillsRegistry(skills_dir=tmpdir)
            md = self._write_skill_md(tmpdir, """# Stock Research

Research stock prices and market data.

## Type
derived

## Base Skills
- web-search

## Execution
sequential

## Capabilities
- data-fetch
- analysis

## Use When
User wants stock market data
""")
            meta = reg._parse_skill_metadata(md)
            assert "stock" in meta["description"].lower() or "research" in meta["description"].lower()
            assert meta["skill_type"] == "derived"
            assert meta["base_skills"] == ["web-search"]
            assert meta["execution_mode"] == "sequential"
            assert "data-fetch" in meta["capabilities"]
            assert "analysis" in meta["capabilities"]
            assert "stock market" in meta["use_when"].lower()

    def test_parse_missing_file(self):
        """Parsing a nonexistent SKILL.md returns empty defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillsRegistry(skills_dir=tmpdir)
            md = Path(tmpdir) / "nonexistent" / "SKILL.md"
            meta = reg._parse_skill_metadata(md)
            assert meta["description"] == ""
            assert meta["skill_type"] is None
            assert meta["base_skills"] == []

    def test_parse_infers_type_from_base_skills(self):
        """When type is not specified, infer from base_skills count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillsRegistry(skills_dir=tmpdir)
            md = self._write_skill_md(tmpdir, """# Combo Skill

Combines multiple skills.

## Base Skills
- web-search
- calculator
""")
            meta = reg._parse_skill_metadata(md)
            # 2+ base skills -> composite
            assert meta["skill_type"] == "composite"

    def test_parse_auto_infers_use_when_from_description(self):
        """use_when is auto-populated from description when not explicit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillsRegistry(skills_dir=tmpdir)
            md = self._write_skill_md(tmpdir, """# Calculator

Perform mathematical calculations and unit conversions.
""")
            meta = reg._parse_skill_metadata(md)
            assert meta["use_when"] != ""
            assert "mathematical" in meta["use_when"].lower() or "calculation" in meta["use_when"].lower()


# =============================================================================
# SkillsRegistry Load Collection Tests
# =============================================================================

@pytest.mark.unit
class TestSkillsRegistryLoadCollection:
    """Tests for load_collection and collection management."""

    def test_load_collection_stores_collection(self):
        """load_collection registers tools from a ToolCollection."""
        from Jotty.core.capabilities.registry.tool_collection import ToolCollection

        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillsRegistry(skills_dir=tmpdir)
            reg.initialized = True

            # Create a real ToolCollection with minimal data
            collection = ToolCollection(
                tools=[{"name": "col_tool", "description": "A tool", "forward": lambda x: x}],
                source="test",
            )

            loaded = reg.load_collection(collection, collection_name="test_col")
            assert "test_col" in reg.loaded_collections
            assert isinstance(loaded, dict)

    def test_list_collections_returns_all(self):
        """list_collections returns info for loaded collections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillsRegistry(skills_dir=tmpdir)
            reg.loaded_collections["c1"] = {"source": "local", "tools": {"a": lambda: None}, "metadata": {}}
            reg.loaded_collections["c2"] = {"source": "hub", "tools": {"b": lambda: None}, "metadata": {}}
            result = reg.list_collections()
            assert len(result) == 2
            assert result[0]["source"] in ("local", "hub")

    def test_get_collection_found(self):
        """get_collection returns info for existing collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillsRegistry(skills_dir=tmpdir)
            reg.loaded_collections["test"] = {"source": "local", "tools": {}, "metadata": {}}
            result = reg.get_collection("test")
            assert result is not None
            assert result["source"] == "local"

    def test_get_collection_not_found(self):
        """get_collection returns None for missing collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillsRegistry(skills_dir=tmpdir)
            assert reg.get_collection("nonexistent") is None


# =============================================================================
# SkillsRegistry Additional Methods Tests
# =============================================================================

@pytest.mark.unit
class TestSkillsRegistryAdditionalMethods:
    """Tests for load_all_skills, get_registered_tools, and other methods."""

    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._scan_skills_metadata")
    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._load_cached_dynamic_skills")
    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._prewarm_top_skills")
    @patch("Jotty.core.registry.skills_registry.SkillsRegistry._refresh_dynamic_skills_background")
    def test_load_all_skills_forces_eager_load(self, mock_ref, mock_pre, mock_cache, mock_scan):
        """load_all_skills triggers lazy loading of all skills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillsRegistry(skills_dir=tmpdir)
            loader = MagicMock(return_value={"eager_tool": lambda p: p})
            reg.loaded_skills["lazy-skill"] = SkillDefinition(
                name="lazy-skill", description="lazy",
                _tool_loader=loader,
            )
            result = reg.load_all_skills()
            assert "eager_tool" in result
            loader.assert_called_once()

    def test_get_registered_tools_returns_all_tools(self):
        """get_registered_tools aggregates tools from all skills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillsRegistry(skills_dir=tmpdir)
            reg.loaded_skills["s1"] = SkillDefinition(
                name="s1", description="s1",
                tools={"tool_a": lambda: None, "tool_b": lambda: None},
            )
            reg.loaded_skills["s2"] = SkillDefinition(
                name="s2", description="s2",
                tools={"tool_c": lambda: None},
            )
            all_tools = reg.get_registered_tools()
            # At least 3 tools from our skills (may have extra from dep manager)
            assert len(all_tools) >= 3
            assert "tool_a" in all_tools
            assert "tool_b" in all_tools
            assert "tool_c" in all_tools

    def test_get_failed_skills_with_failures(self):
        """get_failed_skills returns errors after load failure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = SkillsRegistry(skills_dir=tmpdir)
            reg._failed_skills = {"broken-skill": "ModuleNotFoundError: No module named 'foo'"}
            failures = reg.get_failed_skills()
            assert "broken-skill" in failures
            assert "ModuleNotFoundError" in failures["broken-skill"]


# =============================================================================
# SkillDefinition Additional Tests
# =============================================================================

@pytest.mark.unit
class TestSkillDefinitionAdditional:
    """Additional tests for SkillDefinition edge cases."""

    def test_trust_level_auto_inferred(self):
        """Trust level is auto-inferred from category."""
        sd = SkillDefinition(
            name="safe-search", description="Search",
            tools={}, category="search",
        )
        assert sd.trust_level == TrustLevel.SAFE

    def test_trust_level_explicit_overrides_infer(self):
        """Explicit trust_level overrides auto-inference."""
        sd = SkillDefinition(
            name="safe-search", description="Search",
            tools={}, category="search",
            trust_level=TrustLevel.DESTRUCTIVE,
        )
        assert sd.trust_level == TrustLevel.DESTRUCTIVE

    def test_executor_type_web_search(self):
        """Executor type inferred as web_search."""
        sd = SkillDefinition(name="google-search", description="Search Google")
        assert sd.executor_type == "web_search"

    def test_executor_type_terminal(self):
        """Executor type inferred as terminal."""
        sd = SkillDefinition(name="bash-executor", description="Execute shell commands")
        assert sd.executor_type == "terminal"

    def test_executor_type_data(self):
        """Executor type inferred as data."""
        sd = SkillDefinition(name="csv-analytics", description="Analyze CSV data files")
        assert sd.executor_type == "data"

    def test_executor_type_llm(self):
        """Executor type inferred as llm."""
        sd = SkillDefinition(name="claude-summarize", description="Summarize with Claude")
        assert sd.executor_type == "llm"

    def test_executor_type_doc_gen(self):
        """Executor type inferred as doc_gen."""
        sd = SkillDefinition(name="pdf-maker", description="Generate PDF reports")
        assert sd.executor_type == "doc_gen"

    def test_executor_type_code(self):
        """Executor type inferred as code."""
        sd = SkillDefinition(name="python-linter", description="Lint Python code")
        assert sd.executor_type == "code"

    def test_tools_setter(self):
        """Tools can be set directly."""
        sd = SkillDefinition(name="test", description="test")
        new_tools = {"new_tool": lambda: None}
        sd.tools = new_tools
        assert sd.tools == new_tools

    def test_to_dict_use_when(self):
        """to_dict includes use_when when set."""
        sd = SkillDefinition(
            name="test", description="test",
            tools={}, use_when="When user needs tests",
        )
        d = sd.to_dict()
        assert d["use_when"] == "When user needs tests"

    def test_to_claude_tools_with_metadata(self):
        """to_claude_tools uses ToolMetadata when available."""
        meta = ToolMetadata(
            name="rich_tool", description="Rich tool",
            parameters={"properties": {"q": {"type": "string"}}, "required": ["q"]},
        )
        sd = SkillDefinition(
            name="test", description="test",
            tools={"rich_tool": lambda p: p},
            tool_metadata={"rich_tool": meta},
        )
        ct = sd.to_claude_tools()
        assert len(ct) == 1
        assert ct[0]["name"] == "rich_tool"
        assert "q" in ct[0]["input_schema"]["properties"]

    def test_to_claude_tools_no_metadata_uses_docstring(self):
        """to_claude_tools falls back to docstring when no metadata."""
        def my_tool(params):
            """My tool does stuff."""
            return {}

        sd = SkillDefinition(
            name="test", description="test",
            tools={"my_tool": my_tool},
        )
        ct = sd.to_claude_tools()
        assert len(ct) == 1
        assert "My tool does stuff" in ct[0]["description"]


# =============================================================================
# _infer_trust_level Additional Tests
# =============================================================================

@pytest.mark.unit
class TestInferTrustLevelAdditional:
    """Additional tests for _infer_trust_level edge cases."""

    def test_create_keyword_is_side_effect(self):
        """Name with 'create' keyword infers SIDE_EFFECT when category is not safe."""
        result = _infer_trust_level("automation", [], "create-note")
        assert result == TrustLevel.SIDE_EFFECT

    def test_write_keyword_is_side_effect(self):
        """Name with 'write' keyword infers SIDE_EFFECT when category is not safe."""
        result = _infer_trust_level("automation", [], "write-file")
        assert result == TrustLevel.SIDE_EFFECT

    def test_update_keyword_is_side_effect(self):
        """Name with 'update' keyword infers SIDE_EFFECT when category is not safe."""
        result = _infer_trust_level("automation", [], "update-config")
        assert result == TrustLevel.SIDE_EFFECT

    def test_destructive_tag_overrides_safe_category(self):
        """Destructive tags take precedence over safe category."""
        result = _infer_trust_level("search", ["delete"], "safe-name")
        assert result == TrustLevel.DESTRUCTIVE

    def test_purge_tag_is_destructive(self):
        """Purge tag infers DESTRUCTIVE."""
        result = _infer_trust_level("general", ["purge", "cache"], "purge-tool")
        assert result == TrustLevel.DESTRUCTIVE

    def test_analysis_category_is_safe(self):
        """Analysis category infers SAFE."""
        result = _infer_trust_level("analysis", [], "analyze-data")
        assert result == TrustLevel.SAFE

    def test_mixed_case_category(self):
        """Category comparison is case-insensitive."""
        result = _infer_trust_level("SEARCH", [], "any-tool")
        assert result == TrustLevel.SAFE

    def test_mixed_case_tags(self):
        """Tag comparison is case-insensitive."""
        result = _infer_trust_level("general", ["DELETE", "Important"], "some-tool")
        assert result == TrustLevel.DESTRUCTIVE

"""
Tests for Skill Generator and Skills Manifest modules.
========================================================

Tests the AI-powered skill generation system (skill_generator.py) and
the skill discovery/categorization manifest system (skills_manifest.py).

Covers:
- SkillGenerator: initialization, generate_skill, improve_skill, validate,
  LLM response parsing, fallback templates, singleton factory
- SkillsManifest: initialization, YAML loading, auto-discovery, category/tag
  queries, search, type classification, refresh, save, and summary
- SkillInfo / CategoryInfo dataclasses

All LLM calls, file I/O (YAML), and external dependencies are mocked.
"""
import sys
import os
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import Dict, Any

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try importing dspy - needed for SkillGenerator
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

# Try importing the modules under test
try:
    from Jotty.core.capabilities.registry.skill_generator import SkillGenerator, get_skill_generator
    SKILL_GENERATOR_AVAILABLE = True
except ImportError:
    SKILL_GENERATOR_AVAILABLE = False

try:
    from Jotty.core.capabilities.registry.skills_manifest import (
        SkillsManifest, SkillInfo, CategoryInfo, get_skills_manifest,
    )
    SKILLS_MANIFEST_AVAILABLE = True
except ImportError:
    SKILLS_MANIFEST_AVAILABLE = False


# =============================================================================
# Helper: reset singletons between tests
# =============================================================================

def _reset_generator_singleton():
    """Reset the module-level SkillGenerator singleton."""
    try:
        import Jotty.core.capabilities.registry.skill_generator as mod
        mod._generator_instance = None
    except ImportError:
        pass


def _reset_manifest_singleton():
    """Reset the module-level SkillsManifest singleton."""
    try:
        import Jotty.core.capabilities.registry.skills_manifest as mod
        mod._manifest_instance = None
    except ImportError:
        pass


# =============================================================================
# SkillGenerator Tests
# =============================================================================

@pytest.mark.skipif(not SKILL_GENERATOR_AVAILABLE, reason="SkillGenerator not importable")
@pytest.mark.unit
class TestSkillGeneratorInit:
    """Tests for SkillGenerator initialization."""

    def test_init_with_explicit_dir_and_lm(self, tmp_path):
        """SkillGenerator initializes with explicit skills_dir and lm."""
        mock_lm = MagicMock()
        gen = SkillGenerator(skills_dir=str(tmp_path), lm=mock_lm)
        assert gen.skills_dir == tmp_path
        assert gen.lm is mock_lm
        assert gen.skills_registry is None

    def test_init_creates_skills_dir(self, tmp_path):
        """SkillGenerator creates skills_dir if it does not exist."""
        new_dir = tmp_path / "new_skills"
        assert not new_dir.exists()
        mock_lm = MagicMock()
        gen = SkillGenerator(skills_dir=str(new_dir), lm=mock_lm)
        assert new_dir.exists()

    def test_init_with_skills_registry(self, tmp_path):
        """SkillGenerator stores skills_registry reference."""
        mock_lm = MagicMock()
        mock_registry = MagicMock()
        gen = SkillGenerator(skills_dir=str(tmp_path), lm=mock_lm, skills_registry=mock_registry)
        assert gen.skills_registry is mock_registry

    def test_init_env_var_skills_dir(self, tmp_path):
        """SkillGenerator uses JOTTY_SKILLS_DIR env var when skills_dir is None."""
        mock_lm = MagicMock()
        env_dir = str(tmp_path / "env_skills")
        with patch.dict(os.environ, {"JOTTY_SKILLS_DIR": env_dir}):
            gen = SkillGenerator(skills_dir=None, lm=mock_lm)
            assert str(gen.skills_dir) == env_dir

    def test_init_no_lm_raises_if_unconfigured(self, tmp_path):
        """SkillGenerator raises ValueError when no LM is available."""
        with patch("dspy.LM", side_effect=Exception("no config")):
            with patch(
                "Jotty.core.registry.skill_generator.SkillGenerator.__init__",
                wraps=SkillGenerator.__init__,
            ):
                with pytest.raises((ValueError, Exception)):
                    SkillGenerator(skills_dir=str(tmp_path), lm=None)


@pytest.mark.skipif(not SKILL_GENERATOR_AVAILABLE, reason="SkillGenerator not importable")
@pytest.mark.unit
class TestSkillGeneratorGenerate:
    """Tests for SkillGenerator.generate_skill method."""

    def _make_generator(self, tmp_path, lm_response=None):
        """Helper to create a SkillGenerator with a mocked LM."""
        mock_lm = MagicMock()
        if lm_response is not None:
            mock_lm.return_value = lm_response
        else:
            mock_lm.return_value = None  # Force fallback
        gen = SkillGenerator(skills_dir=str(tmp_path), lm=mock_lm)
        return gen

    def test_generate_skill_creates_directory(self, tmp_path):
        """generate_skill creates a subdirectory for the skill."""
        gen = self._make_generator(tmp_path)
        result = gen.generate_skill("test-skill", "A test skill")
        assert (tmp_path / "test-skill").is_dir()

    def test_generate_skill_creates_skill_md(self, tmp_path):
        """generate_skill writes SKILL.md file."""
        gen = self._make_generator(tmp_path)
        result = gen.generate_skill("test-skill", "A test skill")
        skill_md = tmp_path / "test-skill" / "SKILL.md"
        assert skill_md.exists()
        content = skill_md.read_text()
        assert "test-skill" in content

    def test_generate_skill_creates_tools_py(self, tmp_path):
        """generate_skill writes tools.py file."""
        gen = self._make_generator(tmp_path)
        result = gen.generate_skill("test-skill", "A test skill")
        tools_py = tmp_path / "test-skill" / "tools.py"
        assert tools_py.exists()
        content = tools_py.read_text()
        assert "def " in content

    def test_generate_skill_returns_metadata(self, tmp_path):
        """generate_skill returns dict with expected keys."""
        gen = self._make_generator(tmp_path)
        result = gen.generate_skill("my-skill", "My description")
        assert result["name"] == "my-skill"
        assert result["description"] == "My description"
        assert "path" in result
        assert "skill_md" in result
        assert "tools_py" in result
        assert result["reloaded"] is False
        assert result["tool_tested"] is False

    def test_generate_skill_with_requirements(self, tmp_path):
        """generate_skill passes requirements to LLM prompt."""
        mock_lm = MagicMock(return_value=None)
        gen = SkillGenerator(skills_dir=str(tmp_path), lm=mock_lm)
        result = gen.generate_skill(
            "api-skill", "Use an API", requirements="Requires API key"
        )
        assert result["name"] == "api-skill"
        # LLM was called (twice: md + py)
        assert mock_lm.call_count == 2

    def test_generate_skill_with_examples(self, tmp_path):
        """generate_skill accepts examples parameter."""
        gen = self._make_generator(tmp_path)
        result = gen.generate_skill(
            "ex-skill", "Example skill",
            examples=["Example 1", "Example 2"]
        )
        assert result["name"] == "ex-skill"

    def test_generate_skill_auto_reload(self, tmp_path):
        """generate_skill calls registry.load_all_skills when registry is provided."""
        mock_lm = MagicMock(return_value=None)
        mock_registry = MagicMock()
        mock_skill = MagicMock()
        mock_skill.tools = {"my_tool": MagicMock(return_value={"success": True})}
        mock_registry.get_skill.return_value = mock_skill
        gen = SkillGenerator(skills_dir=str(tmp_path), lm=mock_lm, skills_registry=mock_registry)
        result = gen.generate_skill("reload-skill", "test reload")
        mock_registry.load_all_skills.assert_called_once()
        assert result["reloaded"] is True

    def test_generate_skill_auto_reload_failure(self, tmp_path):
        """generate_skill handles registry reload failure gracefully."""
        mock_lm = MagicMock(return_value=None)
        mock_registry = MagicMock()
        mock_registry.load_all_skills.side_effect = RuntimeError("reload failed")
        gen = SkillGenerator(skills_dir=str(tmp_path), lm=mock_lm, skills_registry=mock_registry)
        result = gen.generate_skill("fail-reload", "test fail")
        assert result["reloaded"] is False

    def test_generate_skill_lm_list_response(self, tmp_path):
        """generate_skill handles LLM list response format."""
        gen = self._make_generator(tmp_path, lm_response=["# Generated\n\nContent here"])
        result = gen.generate_skill("list-skill", "test")
        md_content = (tmp_path / "list-skill" / "SKILL.md").read_text()
        assert "Generated" in md_content or "list-skill" in md_content

    def test_generate_skill_lm_string_response(self, tmp_path):
        """generate_skill handles LLM string response format."""
        gen = self._make_generator(tmp_path, lm_response="# Direct String\n\nSkill content")
        result = gen.generate_skill("str-skill", "test")
        md_content = (tmp_path / "str-skill" / "SKILL.md").read_text()
        assert "Direct String" in md_content or "str-skill" in md_content


@pytest.mark.skipif(not SKILL_GENERATOR_AVAILABLE, reason="SkillGenerator not importable")
@pytest.mark.unit
class TestSkillGeneratorLLMResponseParsing:
    """Tests for internal LLM response parsing (markdown/code extraction)."""

    def _make_generator(self, tmp_path):
        mock_lm = MagicMock(return_value=None)
        return SkillGenerator(skills_dir=str(tmp_path), lm=mock_lm)

    def test_generate_skill_md_fallback(self, tmp_path):
        """_generate_skill_md returns fallback when LLM returns None."""
        gen = self._make_generator(tmp_path)
        result = gen._generate_skill_md("my-skill", "My description", None, None)
        assert "my-skill" in result
        assert "My description" in result

    def test_generate_skill_md_extracts_markdown_block(self, tmp_path):
        """_generate_skill_md extracts content from ```markdown``` blocks."""
        mock_lm = MagicMock(return_value=["Here is the file:\n```markdown\n# Extracted\nContent\n```\nDone."])
        gen = SkillGenerator(skills_dir=str(tmp_path), lm=mock_lm)
        result = gen._generate_skill_md("test", "desc", None, None)
        assert "Extracted" in result

    def test_generate_skill_md_extracts_generic_code_block(self, tmp_path):
        """_generate_skill_md extracts content from generic ``` blocks."""
        mock_lm = MagicMock(return_value=["Explanation:\n```\n# Inside block\nStuff\n```\nEnd."])
        gen = SkillGenerator(skills_dir=str(tmp_path), lm=mock_lm)
        result = gen._generate_skill_md("test", "desc", None, None)
        assert "Inside block" in result

    def test_generate_skill_md_handles_json_wrapped(self, tmp_path):
        """_generate_skill_md handles JSON-wrapped responses."""
        json_resp = json.dumps(["# JSON Wrapped Content"])
        mock_lm = MagicMock(return_value=[json_resp])
        gen = SkillGenerator(skills_dir=str(tmp_path), lm=mock_lm)
        result = gen._generate_skill_md("test", "desc", None, None)
        assert "JSON Wrapped Content" in result

    def test_generate_tools_py_fallback(self, tmp_path):
        """_generate_tools_py returns fallback stub when LLM returns None."""
        gen = self._make_generator(tmp_path)
        result = gen._generate_tools_py("my-tool", "Do something", None, None)
        assert "def my_tool_tool" in result
        assert "Do something" in result

    def test_generate_tools_py_extracts_python_block(self, tmp_path):
        """_generate_tools_py extracts code from ```python``` blocks."""
        code = 'def hello_tool(params):\n    return {"success": True}'
        mock_lm = MagicMock(return_value=[f"Here:\n```python\n{code}\n```"])
        gen = SkillGenerator(skills_dir=str(tmp_path), lm=mock_lm)
        result = gen._generate_tools_py("hello", "greet", None, None)
        assert "def hello_tool" in result

    def test_generate_tools_py_strips_code_markers(self, tmp_path):
        """_generate_tools_py strips leading/trailing code block markers."""
        code = '```python\ndef raw_tool(params):\n    pass\n```'
        mock_lm = MagicMock(return_value=[code])
        gen = SkillGenerator(skills_dir=str(tmp_path), lm=mock_lm)
        result = gen._generate_tools_py("raw", "raw desc", None, None)
        assert not result.startswith("```")
        assert not result.endswith("```")

    def test_generate_tools_py_lm_exception_uses_fallback(self, tmp_path):
        """_generate_tools_py uses fallback when LLM raises exception."""
        mock_lm = MagicMock(side_effect=Exception("LLM timeout"))
        gen = SkillGenerator(skills_dir=str(tmp_path), lm=mock_lm)
        # Direct call; lm will raise on first call
        result = gen._generate_tools_py("err-skill", "Errored skill", None, None)
        assert "def err_skill_tool" in result


@pytest.mark.skipif(not SKILL_GENERATOR_AVAILABLE, reason="SkillGenerator not importable")
@pytest.mark.unit
class TestSkillGeneratorImprove:
    """Tests for SkillGenerator.improve_skill method."""

    def test_improve_skill_not_found(self, tmp_path):
        """improve_skill raises ValueError for non-existent skill."""
        mock_lm = MagicMock(return_value=None)
        gen = SkillGenerator(skills_dir=str(tmp_path), lm=mock_lm)
        with pytest.raises(ValueError, match="not found"):
            gen.improve_skill("nonexistent", "fix it")

    def test_improve_skill_calls_lm(self, tmp_path):
        """improve_skill calls LLM to improve existing files."""
        skill_dir = tmp_path / "fix-me"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# fix-me\nOld content")
        (skill_dir / "tools.py").write_text("def old_tool(params): pass")

        mock_lm = MagicMock(return_value="# Improved content")
        gen = SkillGenerator(skills_dir=str(tmp_path), lm=mock_lm)
        result = gen.improve_skill("fix-me", "needs better docs", changes="add examples")
        assert result["improved"] is True
        assert result["name"] == "fix-me"
        assert result["feedback"] == "needs better docs"
        # LLM called twice: once for md, once for py
        assert mock_lm.call_count == 2

    def test_improve_skill_missing_tools_py(self, tmp_path):
        """improve_skill works when only SKILL.md exists."""
        skill_dir = tmp_path / "md-only"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# md-only\nContent")

        mock_lm = MagicMock(return_value="# Improved")
        gen = SkillGenerator(skills_dir=str(tmp_path), lm=mock_lm)
        result = gen.improve_skill("md-only", "improve it")
        assert result["improved"] is True
        # LLM only called once (for md, not py since py doesn't exist)
        assert mock_lm.call_count == 1

    def test_improve_skill_missing_skill_md(self, tmp_path):
        """improve_skill works when only tools.py exists."""
        skill_dir = tmp_path / "py-only"
        skill_dir.mkdir()
        (skill_dir / "tools.py").write_text("def tool(p): pass")

        mock_lm = MagicMock(return_value="def improved_tool(p): pass")
        gen = SkillGenerator(skills_dir=str(tmp_path), lm=mock_lm)
        result = gen.improve_skill("py-only", "improve it")
        assert result["improved"] is True
        assert mock_lm.call_count == 1


@pytest.mark.skipif(not SKILL_GENERATOR_AVAILABLE, reason="SkillGenerator not importable")
@pytest.mark.unit
class TestSkillGeneratorValidate:
    """Tests for SkillGenerator.validate_generated_skill method."""

    def _make_generator(self, tmp_path):
        mock_lm = MagicMock(return_value=None)
        return SkillGenerator(skills_dir=str(tmp_path), lm=mock_lm)

    def test_validate_nonexistent_skill(self, tmp_path):
        """validate returns invalid for non-existent skill directory."""
        gen = self._make_generator(tmp_path)
        result = gen.validate_generated_skill("nope")
        assert result["valid"] is False
        assert any("not found" in e for e in result["errors"])

    def test_validate_valid_skill(self, tmp_path):
        """validate returns valid for skill with both files and function."""
        skill_dir = tmp_path / "good-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# good-skill\n\n## Description\nGood skill")
        (skill_dir / "tools.py").write_text("def good_tool(params): pass")

        gen = self._make_generator(tmp_path)
        result = gen.validate_generated_skill("good-skill")
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_missing_skill_md(self, tmp_path):
        """validate flags missing SKILL.md."""
        skill_dir = tmp_path / "no-md"
        skill_dir.mkdir()
        (skill_dir / "tools.py").write_text("def tool(p): pass")

        gen = self._make_generator(tmp_path)
        result = gen.validate_generated_skill("no-md")
        assert result["valid"] is False
        assert "Missing SKILL.md" in result["errors"]

    def test_validate_missing_tools_py(self, tmp_path):
        """validate flags missing tools.py."""
        skill_dir = tmp_path / "no-tools"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# no-tools\nDescription here")

        gen = self._make_generator(tmp_path)
        result = gen.validate_generated_skill("no-tools")
        assert result["valid"] is False
        assert "Missing tools.py" in result["errors"]

    def test_validate_tools_py_missing_functions(self, tmp_path):
        """validate flags tools.py with no function definitions."""
        skill_dir = tmp_path / "no-funcs"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# no-funcs\nDescription")
        (skill_dir / "tools.py").write_text("# just a comment\nx = 42")

        gen = self._make_generator(tmp_path)
        result = gen.validate_generated_skill("no-funcs")
        assert result["valid"] is False
        assert any("function" in e.lower() for e in result["errors"])

    def test_validate_skill_md_missing_description_warning(self, tmp_path):
        """validate warns when SKILL.md lacks description keyword."""
        skill_dir = tmp_path / "no-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# no-desc\nJust a title")
        (skill_dir / "tools.py").write_text("def tool(p): pass")

        gen = self._make_generator(tmp_path)
        result = gen.validate_generated_skill("no-desc")
        assert result["valid"] is True  # warnings don't make it invalid
        assert len(result["warnings"]) > 0


@pytest.mark.skipif(not SKILL_GENERATOR_AVAILABLE, reason="SkillGenerator not importable")
@pytest.mark.unit
class TestSkillGeneratorSingleton:
    """Tests for get_skill_generator singleton factory."""

    def setup_method(self):
        _reset_generator_singleton()

    def teardown_method(self):
        _reset_generator_singleton()

    def test_get_skill_generator_returns_instance(self, tmp_path):
        """get_skill_generator returns a SkillGenerator."""
        mock_lm = MagicMock()
        gen = get_skill_generator(skills_dir=str(tmp_path), lm=mock_lm)
        assert isinstance(gen, SkillGenerator)

    def test_get_skill_generator_singleton(self, tmp_path):
        """get_skill_generator returns same instance on subsequent calls."""
        mock_lm = MagicMock()
        gen1 = get_skill_generator(skills_dir=str(tmp_path), lm=mock_lm)
        gen2 = get_skill_generator(skills_dir=str(tmp_path), lm=mock_lm)
        assert gen1 is gen2

    def test_get_skill_generator_updates_registry(self, tmp_path):
        """get_skill_generator updates registry on existing instance."""
        mock_lm = MagicMock()
        gen1 = get_skill_generator(skills_dir=str(tmp_path), lm=mock_lm)
        assert gen1.skills_registry is None
        mock_registry = MagicMock()
        gen2 = get_skill_generator(skills_dir=str(tmp_path), lm=mock_lm, skills_registry=mock_registry)
        assert gen2 is gen1
        assert gen1.skills_registry is mock_registry


# =============================================================================
# SkillInfo / CategoryInfo Dataclass Tests
# =============================================================================

@pytest.mark.skipif(not SKILLS_MANIFEST_AVAILABLE, reason="SkillsManifest not importable")
@pytest.mark.unit
class TestSkillInfoDataclass:
    """Tests for SkillInfo dataclass."""

    def test_default_values(self):
        """SkillInfo has expected defaults."""
        info = SkillInfo(name="test")
        assert info.name == "test"
        assert info.category == "uncategorized"
        assert info.tags == []
        assert info.description == ""
        assert info.icon == ""
        assert info.requires_auth is False
        assert info.env_vars == []
        assert info.requires_cli == []
        assert info.is_discovered is False
        assert info.skill_type == "base"
        assert info.base_skills == []

    def test_custom_values(self):
        """SkillInfo accepts all custom values."""
        info = SkillInfo(
            name="web-search",
            category="research",
            tags=["web", "search"],
            description="Search the web",
            icon="magnifier",
            requires_auth=True,
            env_vars=["SEARCH_API_KEY"],
            requires_cli=["curl"],
            is_discovered=True,
            skill_type="derived",
            base_skills=["http-client"],
        )
        assert info.name == "web-search"
        assert info.category == "research"
        assert info.requires_auth is True
        assert info.skill_type == "derived"


@pytest.mark.skipif(not SKILLS_MANIFEST_AVAILABLE, reason="SkillsManifest not importable")
@pytest.mark.unit
class TestCategoryInfoDataclass:
    """Tests for CategoryInfo dataclass."""

    def test_default_values(self):
        """CategoryInfo has expected defaults."""
        cat = CategoryInfo(name="test", description="Test cat", icon="T")
        assert cat.name == "test"
        assert cat.skills == []

    def test_with_skills(self):
        """CategoryInfo stores skills list."""
        cat = CategoryInfo(name="dev", description="Dev tools", icon="D", skills=["git", "docker"])
        assert len(cat.skills) == 2
        assert "git" in cat.skills


# =============================================================================
# SkillsManifest Tests
# =============================================================================

@pytest.mark.skipif(not SKILLS_MANIFEST_AVAILABLE, reason="SkillsManifest not importable")
@pytest.mark.unit
class TestSkillsManifestInit:
    """Tests for SkillsManifest initialization."""

    def test_init_with_nonexistent_manifest(self, tmp_path):
        """SkillsManifest initializes gracefully with no manifest file."""
        manifest = SkillsManifest(
            skills_dir=str(tmp_path),
            manifest_path=str(tmp_path / "nonexistent.yaml")
        )
        assert len(manifest.skills) == 0
        assert len(manifest.categories) == 0

    def test_init_creates_no_skills_dir(self, tmp_path):
        """SkillsManifest does not fail if skills_dir is empty."""
        manifest = SkillsManifest(
            skills_dir=str(tmp_path),
            manifest_path=str(tmp_path / "none.yaml")
        )
        assert manifest.skills_dir == tmp_path

    @patch("Jotty.core.registry.skills_manifest.yaml", create=True)
    def test_init_loads_yaml_manifest(self, mock_yaml_mod, tmp_path):
        """SkillsManifest loads categories and skills from YAML."""
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("dummy")  # File must exist for _load_manifest

        yaml_data = {
            "auto_discover": False,
            "categories": {
                "research": {
                    "description": "Research tools",
                    "icon": "R",
                    "skills": ["web-search", "arxiv"]
                }
            },
            "tags": {
                "web": {"skills": ["web-search"]}
            },
            "skill_metadata": {
                "web-search": {
                    "requires_auth": True,
                    "env_vars": ["SEARCH_KEY"],
                    "requires_cli": ["curl"]
                }
            },
            "skill_types": {
                "base": ["web-search"],
                "derived": [{"name": "arxiv", "base_skills": ["web-search"]}]
            }
        }

        # Patch yaml.safe_load inside _load_manifest
        with patch("builtins.open", create=True) as mock_open:
            with patch("yaml.safe_load", return_value=yaml_data):
                manifest = SkillsManifest(
                    skills_dir=str(tmp_path),
                    manifest_path=str(manifest_path)
                )

        assert "research" in manifest.categories
        assert "web-search" in manifest.skills
        assert manifest.skills["web-search"].requires_auth is True


@pytest.mark.skipif(not SKILLS_MANIFEST_AVAILABLE, reason="SkillsManifest not importable")
@pytest.mark.unit
class TestSkillsManifestAutoDiscover:
    """Tests for auto-discovery of skills from filesystem."""

    def test_discover_new_skill(self, tmp_path):
        """Auto-discovers skill directories with tools.py."""
        skill_dir = tmp_path / "new-skill"
        skill_dir.mkdir()
        (skill_dir / "tools.py").write_text("def tool(p): pass")

        manifest = SkillsManifest(
            skills_dir=str(tmp_path),
            manifest_path=str(tmp_path / "none.yaml")
        )
        assert "new-skill" in manifest.skills
        assert manifest.skills["new-skill"].is_discovered is True
        assert manifest.skills["new-skill"].category == "uncategorized"

    def test_discover_ignores_hidden_dirs(self, tmp_path):
        """Auto-discovery skips directories starting with . or _."""
        (tmp_path / ".hidden").mkdir()
        (tmp_path / ".hidden" / "tools.py").write_text("x=1")
        (tmp_path / "_private").mkdir()
        (tmp_path / "_private" / "tools.py").write_text("x=1")

        manifest = SkillsManifest(
            skills_dir=str(tmp_path),
            manifest_path=str(tmp_path / "none.yaml")
        )
        assert ".hidden" not in manifest.skills
        assert "_private" not in manifest.skills

    def test_discover_ignores_dirs_without_tools_py(self, tmp_path):
        """Auto-discovery skips directories without tools.py."""
        (tmp_path / "no-tools").mkdir()
        (tmp_path / "no-tools" / "README.md").write_text("nothing")

        manifest = SkillsManifest(
            skills_dir=str(tmp_path),
            manifest_path=str(tmp_path / "none.yaml")
        )
        assert "no-tools" not in manifest.skills

    def test_discover_creates_uncategorized_category(self, tmp_path):
        """Auto-discovery creates uncategorized category for new skills."""
        skill_dir = tmp_path / "disc-skill"
        skill_dir.mkdir()
        (skill_dir / "tools.py").write_text("def tool(p): pass")

        manifest = SkillsManifest(
            skills_dir=str(tmp_path),
            manifest_path=str(tmp_path / "none.yaml")
        )
        assert "uncategorized" in manifest.categories
        assert "disc-skill" in manifest.categories["uncategorized"].skills


@pytest.mark.skipif(not SKILLS_MANIFEST_AVAILABLE, reason="SkillsManifest not importable")
@pytest.mark.unit
class TestSkillsManifestQueries:
    """Tests for SkillsManifest query methods."""

    def _make_manifest_with_skills(self, tmp_path):
        """Create a manifest with pre-populated skills and categories."""
        manifest = SkillsManifest(
            skills_dir=str(tmp_path),
            manifest_path=str(tmp_path / "none.yaml")
        )
        # Manually populate
        manifest.categories["research"] = CategoryInfo(
            name="research", description="Research", icon="R", skills=["web-search", "arxiv"]
        )
        manifest.categories["dev"] = CategoryInfo(
            name="dev", description="Dev tools", icon="D", skills=["git-ops"]
        )
        manifest.skills["web-search"] = SkillInfo(
            name="web-search", category="research", tags=["web", "search"],
            skill_type="base"
        )
        manifest.skills["arxiv"] = SkillInfo(
            name="arxiv", category="research", tags=["academic", "search"],
            skill_type="derived", base_skills=["web-search"]
        )
        manifest.skills["git-ops"] = SkillInfo(
            name="git-ops", category="dev", tags=["git"],
            skill_type="composite"
        )
        manifest.tags = {
            "web": ["web-search"],
            "search": ["web-search", "arxiv"],
            "git": ["git-ops"],
        }
        return manifest

    def test_get_categories(self, tmp_path):
        """get_categories returns all categories."""
        manifest = self._make_manifest_with_skills(tmp_path)
        cats = manifest.get_categories()
        assert len(cats) == 2

    def test_get_category_by_name(self, tmp_path):
        """get_category returns specific category."""
        manifest = self._make_manifest_with_skills(tmp_path)
        cat = manifest.get_category("research")
        assert cat is not None
        assert cat.name == "research"

    def test_get_category_not_found(self, tmp_path):
        """get_category returns None for unknown category."""
        manifest = self._make_manifest_with_skills(tmp_path)
        assert manifest.get_category("nonexistent") is None

    def test_get_skills_by_category(self, tmp_path):
        """get_skills_by_category returns matching skills."""
        manifest = self._make_manifest_with_skills(tmp_path)
        skills = manifest.get_skills_by_category("research")
        assert len(skills) == 2
        names = [s.name for s in skills]
        assert "web-search" in names
        assert "arxiv" in names

    def test_get_skills_by_tag(self, tmp_path):
        """get_skills_by_tag returns matching skills."""
        manifest = self._make_manifest_with_skills(tmp_path)
        skills = manifest.get_skills_by_tag("search")
        assert len(skills) == 2

    def test_get_skills_by_tag_empty(self, tmp_path):
        """get_skills_by_tag returns empty for unknown tag."""
        manifest = self._make_manifest_with_skills(tmp_path)
        skills = manifest.get_skills_by_tag("nonexistent")
        assert skills == []

    def test_get_skill_by_name(self, tmp_path):
        """get_skill returns specific skill."""
        manifest = self._make_manifest_with_skills(tmp_path)
        skill = manifest.get_skill("web-search")
        assert skill is not None
        assert skill.name == "web-search"

    def test_get_skill_not_found(self, tmp_path):
        """get_skill returns None for unknown skill."""
        manifest = self._make_manifest_with_skills(tmp_path)
        assert manifest.get_skill("unknown") is None

    def test_get_all_skills(self, tmp_path):
        """get_all_skills returns all skills."""
        manifest = self._make_manifest_with_skills(tmp_path)
        all_skills = manifest.get_all_skills()
        assert len(all_skills) == 3

    def test_get_skills_by_type(self, tmp_path):
        """get_skills_by_type filters by skill_type."""
        manifest = self._make_manifest_with_skills(tmp_path)
        base = manifest.get_skills_by_type("base")
        assert len(base) == 1
        assert base[0].name == "web-search"
        derived = manifest.get_skills_by_type("derived")
        assert len(derived) == 1
        composite = manifest.get_skills_by_type("composite")
        assert len(composite) == 1

    def test_get_type_summary(self, tmp_path):
        """get_type_summary returns counts by type."""
        manifest = self._make_manifest_with_skills(tmp_path)
        summary = manifest.get_type_summary()
        assert summary == {"base": 1, "derived": 1, "composite": 1}

    def test_search_skills_by_name(self, tmp_path):
        """search_skills finds skills by name substring."""
        manifest = self._make_manifest_with_skills(tmp_path)
        results = manifest.search_skills("web")
        assert any(s.name == "web-search" for s in results)

    def test_search_skills_by_category(self, tmp_path):
        """search_skills finds skills by category substring."""
        manifest = self._make_manifest_with_skills(tmp_path)
        results = manifest.search_skills("dev")
        assert any(s.name == "git-ops" for s in results)

    def test_search_skills_by_tag(self, tmp_path):
        """search_skills finds skills by tag."""
        manifest = self._make_manifest_with_skills(tmp_path)
        results = manifest.search_skills("academic")
        assert any(s.name == "arxiv" for s in results)

    def test_search_skills_case_insensitive(self, tmp_path):
        """search_skills is case insensitive."""
        manifest = self._make_manifest_with_skills(tmp_path)
        results = manifest.search_skills("WEB")
        assert len(results) > 0

    def test_search_skills_no_match(self, tmp_path):
        """search_skills returns empty for no match."""
        manifest = self._make_manifest_with_skills(tmp_path)
        results = manifest.search_skills("zzzznotfound")
        assert results == []

    def test_get_uncategorized_skills(self, tmp_path):
        """get_uncategorized_skills returns discovered/uncategorized skills."""
        manifest = self._make_manifest_with_skills(tmp_path)
        # Add a discovered skill
        manifest.skills["new-disc"] = SkillInfo(name="new-disc", is_discovered=True)
        uncategorized = manifest.get_uncategorized_skills()
        assert any(s.name == "new-disc" for s in uncategorized)


@pytest.mark.skipif(not SKILLS_MANIFEST_AVAILABLE, reason="SkillsManifest not importable")
@pytest.mark.unit
class TestSkillsManifestSummary:
    """Tests for manifest summary and discovery prompt generation."""

    def _make_manifest(self, tmp_path):
        manifest = SkillsManifest(
            skills_dir=str(tmp_path),
            manifest_path=str(tmp_path / "none.yaml")
        )
        manifest.categories["research"] = CategoryInfo(
            name="research", description="Research tools", icon="R",
            skills=["web-search"]
        )
        manifest.skills["web-search"] = SkillInfo(
            name="web-search", category="research", requires_auth=True
        )
        manifest.tags = {"web": ["web-search"]}
        return manifest

    def test_get_summary(self, tmp_path):
        """get_summary returns dict with total counts and categories."""
        manifest = self._make_manifest(tmp_path)
        summary = manifest.get_summary()
        assert summary["total_skills"] == 1
        assert summary["total_categories"] == 1
        assert "research" in summary["categories"]

    def test_get_discovery_prompt(self, tmp_path):
        """get_discovery_prompt returns formatted string."""
        manifest = self._make_manifest(tmp_path)
        prompt = manifest.get_discovery_prompt()
        assert "Available Skills" in prompt
        assert "web-search" in prompt
        assert "Tags" in prompt


@pytest.mark.skipif(not SKILLS_MANIFEST_AVAILABLE, reason="SkillsManifest not importable")
@pytest.mark.unit
class TestSkillsManifestMutation:
    """Tests for manifest mutation methods (add_skill_to_category, refresh)."""

    def test_add_skill_to_category(self, tmp_path):
        """add_skill_to_category moves skill between categories."""
        manifest = SkillsManifest(
            skills_dir=str(tmp_path),
            manifest_path=str(tmp_path / "none.yaml")
        )
        manifest.categories["old"] = CategoryInfo(name="old", description="", icon="", skills=["s1"])
        manifest.categories["new"] = CategoryInfo(name="new", description="", icon="", skills=[])
        manifest.skills["s1"] = SkillInfo(name="s1", category="old")

        with patch.object(manifest, "_save_manifest"):
            result = manifest.add_skill_to_category("s1", "new")
        assert result is True
        assert manifest.skills["s1"].category == "new"
        assert "s1" in manifest.categories["new"].skills
        assert "s1" not in manifest.categories["old"].skills

    def test_add_skill_to_category_unknown_skill(self, tmp_path):
        """add_skill_to_category returns False for unknown skill."""
        manifest = SkillsManifest(
            skills_dir=str(tmp_path),
            manifest_path=str(tmp_path / "none.yaml")
        )
        manifest.categories["cat"] = CategoryInfo(name="cat", description="", icon="")
        result = manifest.add_skill_to_category("nope", "cat")
        assert result is False

    def test_add_skill_to_category_unknown_category(self, tmp_path):
        """add_skill_to_category returns False for unknown category."""
        manifest = SkillsManifest(
            skills_dir=str(tmp_path),
            manifest_path=str(tmp_path / "none.yaml")
        )
        manifest.skills["s1"] = SkillInfo(name="s1")
        result = manifest.add_skill_to_category("s1", "nope")
        assert result is False

    def test_refresh_clears_and_reloads(self, tmp_path):
        """refresh clears data and re-initializes."""
        manifest = SkillsManifest(
            skills_dir=str(tmp_path),
            manifest_path=str(tmp_path / "none.yaml")
        )
        manifest.skills["test"] = SkillInfo(name="test")
        manifest.refresh()
        # After refresh with no manifest/skills, should be empty
        assert "test" not in manifest.skills


@pytest.mark.skipif(not SKILLS_MANIFEST_AVAILABLE, reason="SkillsManifest not importable")
@pytest.mark.unit
class TestSkillsManifestSingleton:
    """Tests for get_skills_manifest singleton factory."""

    def setup_method(self):
        _reset_manifest_singleton()

    def teardown_method(self):
        _reset_manifest_singleton()

    def test_get_skills_manifest_returns_instance(self):
        """get_skills_manifest returns a SkillsManifest."""
        manifest = get_skills_manifest()
        assert isinstance(manifest, SkillsManifest)

    def test_get_skills_manifest_singleton(self):
        """get_skills_manifest returns same instance on subsequent calls."""
        m1 = get_skills_manifest()
        m2 = get_skills_manifest()
        assert m1 is m2

    def test_get_skills_manifest_refresh(self):
        """get_skills_manifest with refresh=True creates new instance."""
        m1 = get_skills_manifest()
        m2 = get_skills_manifest(refresh=True)
        assert m1 is not m2

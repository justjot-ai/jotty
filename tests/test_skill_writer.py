"""Tests for the skill-writer skill."""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# LOAD MODULE BY FILE PATH (avoids name collision with other skills' tools.py)
# =============================================================================

_TOOLS_PATH = Path(__file__).parent.parent / "skills" / "skill-writer" / "tools.py"


def _load_tools():
    """Load the skill-writer tools.py by file path, bypassing module cache."""
    spec = importlib.util.spec_from_file_location("skill_writer_tools", str(_TOOLS_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tools():
    return _load_tools()


# =============================================================================
# TOOL IMPORT
# =============================================================================


class TestImports:
    """Verify all tools are importable."""

    def test_all_tools_importable(self, tools):
        assert hasattr(tools, "create_skill_tool")
        assert hasattr(tools, "improve_skill_tool")
        assert hasattr(tools, "list_skills_tool")

    def test_all_exports(self, tools):
        expected = {"create_skill_tool", "improve_skill_tool", "list_skills_tool"}
        assert set(tools.__all__) == expected


# =============================================================================
# NAME SANITIZATION
# =============================================================================


class TestNameSanitization:
    """Test skill name normalization."""

    def test_basic_kebab_case(self, tools):
        assert tools._sanitize_name("pdf-merger") == "pdf-merger"

    def test_spaces_to_hyphens(self, tools):
        assert tools._sanitize_name("My Cool Skill") == "my-cool-skill"

    def test_underscores_to_hyphens(self, tools):
        assert tools._sanitize_name("my_cool_skill") == "my-cool-skill"

    def test_special_chars_removed(self, tools):
        result = tools._sanitize_name("skill@v2.0!")
        assert "@" not in result
        assert "!" not in result

    def test_truncation(self, tools):
        long_name = "a" * 100
        assert len(tools._sanitize_name(long_name)) <= 60

    def test_empty_name(self, tools):
        assert tools._sanitize_name("") == "unnamed-skill"


# =============================================================================
# CREATE SKILL TOOL
# =============================================================================


class TestCreateSkillTool:
    """Test the create_skill_tool function."""

    def test_missing_name_returns_error(self, tools):
        result = tools.create_skill_tool({"description": "A test skill"})
        assert result.get("error") or not result.get("success", True)

    def test_missing_description_returns_error(self, tools):
        result = tools.create_skill_tool({"name": "test-skill"})
        assert result.get("error") or not result.get("success", True)

    def test_short_description_returns_error(self, tools):
        result = tools.create_skill_tool({"name": "test", "description": "hi"})
        assert result.get("error")

    def test_successful_creation(self, tools):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "test-skill"
            skill_dir.mkdir()
            tools_py = skill_dir / "tools.py"
            tools_py.write_text("def test_skill_tool(params):\n    pass\n")

            mock_gen = MagicMock()
            mock_gen.generate_skill.return_value = {
                "name": "test-skill",
                "description": "A test skill",
                "path": str(skill_dir),
                "tools_py": str(tools_py),
                "skill_md": str(skill_dir / "SKILL.md"),
                "reloaded": True,
                "tool_tested": True,
            }
            mock_gen.validate_generated_skill.return_value = {
                "valid": True,
                "errors": [],
                "warnings": [],
            }

            with patch.object(tools, "_get_generator", return_value=mock_gen):
                result = tools.create_skill_tool(
                    {
                        "name": "test-skill",
                        "description": "A test skill that does testing",
                    }
                )

            assert result.get("success") or result.get("skill_name")
            mock_gen.generate_skill.assert_called_once()

    def test_generator_failure(self, tools):
        with patch.object(tools, "_get_generator", side_effect=Exception("No LLM available")):
            result = tools.create_skill_tool(
                {
                    "name": "test-skill",
                    "description": "A test skill that does something useful",
                }
            )
        assert result.get("error")


# =============================================================================
# IMPROVE SKILL TOOL
# =============================================================================


class TestImproveSkillTool:
    """Test the improve_skill_tool function."""

    def test_missing_feedback_returns_error(self, tools):
        result = tools.improve_skill_tool({"name": "test-skill"})
        assert result.get("error") or not result.get("success", True)

    def test_short_feedback_returns_error(self, tools):
        result = tools.improve_skill_tool({"name": "test", "feedback": "hi"})
        assert result.get("error")

    def test_skill_not_found(self, tools):
        mock_gen = MagicMock()
        mock_gen.improve_skill.side_effect = ValueError("Skill not-real not found")

        with patch.object(tools, "_get_generator", return_value=mock_gen):
            result = tools.improve_skill_tool(
                {
                    "name": "not-real",
                    "feedback": "Make it better and faster",
                }
            )
        assert result.get("error")
        assert "not found" in result["error"].lower() or "not-real" in result["error"]

    def test_successful_improvement(self, tools):
        mock_gen = MagicMock()
        mock_gen.improve_skill.return_value = {
            "name": "test-skill",
            "improved": True,
            "feedback": "Add retry logic",
        }

        with patch.object(tools, "_get_generator", return_value=mock_gen):
            result = tools.improve_skill_tool(
                {
                    "name": "test-skill",
                    "feedback": "Add retry logic for network errors",
                }
            )
        assert not result.get("error")
        mock_gen.improve_skill.assert_called_once()


# =============================================================================
# LIST SKILLS TOOL
# =============================================================================


class TestListSkillsTool:
    """Test the list_skills_tool function."""

    def test_list_all_skills(self, tools):
        mock_reg = MagicMock()
        mock_reg.list_skills.return_value = [
            {
                "name": "calculator",
                "description": "Math calculations",
                "category": "workflow-automation",
            },
            {"name": "web-search", "description": "Search the web", "category": "research"},
        ]

        with patch("Jotty.core.skills.get_registry", return_value=mock_reg):
            result = tools.list_skills_tool({})
        assert result.get("total") == 2
        assert len(result.get("skills", [])) == 2

    def test_search_filter(self, tools):
        mock_reg = MagicMock()
        mock_reg.list_skills.return_value = [
            {
                "name": "calculator",
                "description": "Math calculations",
                "category": "workflow-automation",
            },
            {"name": "web-search", "description": "Search the web", "category": "research"},
        ]

        with patch("Jotty.core.skills.get_registry", return_value=mock_reg):
            result = tools.list_skills_tool({"search": "calc"})
        assert result.get("total") == 1
        assert result["skills"][0]["name"] == "calculator"

    def test_category_filter(self, tools):
        mock_reg = MagicMock()
        mock_reg.list_skills.return_value = [
            {
                "name": "calculator",
                "description": "Math calculations",
                "category": "workflow-automation",
            },
            {"name": "web-search", "description": "Search the web", "category": "research"},
        ]

        with patch("Jotty.core.skills.get_registry", return_value=mock_reg):
            result = tools.list_skills_tool({"category": "research"})
        assert result.get("total") == 1
        assert result["skills"][0]["name"] == "web-search"


# =============================================================================
# SKILL.md VALIDATION
# =============================================================================


class TestSkillMd:
    """Test the SKILL.md file is valid."""

    def test_skill_md_exists(self):
        skill_md = Path(__file__).parent.parent / "skills" / "skill-writer" / "SKILL.md"
        assert skill_md.exists()

    def test_skill_md_has_frontmatter(self):
        skill_md = Path(__file__).parent.parent / "skills" / "skill-writer" / "SKILL.md"
        content = skill_md.read_text()
        assert content.startswith("---")
        assert "name:" in content
        assert "description:" in content

    def test_skill_md_has_tools_section(self):
        skill_md = Path(__file__).parent.parent / "skills" / "skill-writer" / "SKILL.md"
        content = skill_md.read_text()
        assert "## Tools" in content
        assert "create_skill_tool" in content
        assert "improve_skill_tool" in content
        assert "list_skills_tool" in content

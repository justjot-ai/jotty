import pytest

from Jotty.core.capabilities.registry.skills_registry import SkillsRegistry


@pytest.mark.unit
def test_load_tools_from_file_fails_closed_on_import_error(tmp_path):
    """Broken tools.py must not produce fabricated success tools."""
    skill_dir = tmp_path / "broken-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    tools_file = skill_dir / "tools.py"
    tools_file.write_text(
        "import definitely_missing_package\n\n"
        "def sample_tool(params):\n"
        "    return {'success': True}\n"
    )

    registry = SkillsRegistry(skills_dir=str(tmp_path))
    tools = registry._load_tools_from_file(tools_file)

    assert tools == {}
    assert "sample_tool" not in tools
    assert "broken-skill" in registry.get_failed_skills()

"""
Tests for workspace identity files feature (SOUL.md, IDENTITY.md, USER.md).

Covers:
- load_identity_files() with single, multiple, zero, and oversized files
- load_identity_files() with nonexistent directory
- PromptComposer integration: identity injection and absence
"""

import pytest

from Jotty.core.capabilities.prompts.rules import (
    IDENTITY_FILES,
    _MAX_IDENTITY_SIZE,
    clear_rules_cache,
    load_identity_files,
)
from Jotty.core.capabilities.prompts.composer import PromptComposer


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the LRU cache before and after every test."""
    clear_rules_cache()
    yield
    clear_rules_cache()


@pytest.mark.unit
class TestLoadIdentityFilesSingle:
    """load_identity_files with a single SOUL.md file."""

    def test_returns_soul_content(self, tmp_path):
        (tmp_path / "SOUL.md").write_text("You are a helpful coding assistant.", encoding="utf-8")

        result = load_identity_files(str(tmp_path))

        assert result == "You are a helpful coding assistant."

    def test_strips_whitespace(self, tmp_path):
        (tmp_path / "SOUL.md").write_text("  \n  Be concise.  \n  ", encoding="utf-8")

        result = load_identity_files(str(tmp_path))

        assert result == "Be concise."


@pytest.mark.unit
class TestLoadIdentityFilesMultiple:
    """load_identity_files with multiple identity files merged."""

    def test_merges_soul_and_identity(self, tmp_path):
        (tmp_path / "SOUL.md").write_text("I am Jotty.", encoding="utf-8")
        (tmp_path / "IDENTITY.md").write_text("I specialize in code.", encoding="utf-8")

        result = load_identity_files(str(tmp_path))

        assert "I am Jotty." in result
        assert "I specialize in code." in result
        # Files are joined with double newline
        assert result == "I am Jotty.\n\nI specialize in code."

    def test_merges_all_three_files(self, tmp_path):
        (tmp_path / "SOUL.md").write_text("Soul content.", encoding="utf-8")
        (tmp_path / "IDENTITY.md").write_text("Identity content.", encoding="utf-8")
        (tmp_path / "USER.md").write_text("User preferences.", encoding="utf-8")

        result = load_identity_files(str(tmp_path))

        parts = result.split("\n\n")
        assert len(parts) == 3
        assert parts[0] == "Soul content."
        assert parts[1] == "Identity content."
        assert parts[2] == "User preferences."

    def test_skips_missing_files_in_sequence(self, tmp_path):
        """Only IDENTITY.md exists (SOUL.md and USER.md missing)."""
        (tmp_path / "IDENTITY.md").write_text("Just identity.", encoding="utf-8")

        result = load_identity_files(str(tmp_path))

        assert result == "Just identity."


@pytest.mark.unit
class TestLoadIdentityFilesEmpty:
    """load_identity_files with no identity files present."""

    def test_returns_empty_string(self, tmp_path):
        # Directory exists but has no identity files
        result = load_identity_files(str(tmp_path))

        assert result == ""

    def test_ignores_unrelated_files(self, tmp_path):
        (tmp_path / "README.md").write_text("Not an identity file.", encoding="utf-8")
        (tmp_path / ".clinerules").write_text("Some rules.", encoding="utf-8")

        result = load_identity_files(str(tmp_path))

        assert result == ""

    def test_ignores_empty_identity_file(self, tmp_path):
        """An identity file with only whitespace should be treated as absent."""
        (tmp_path / "SOUL.md").write_text("   \n\n   ", encoding="utf-8")

        result = load_identity_files(str(tmp_path))

        assert result == ""


@pytest.mark.unit
class TestLoadIdentityFilesNonexistentDir:
    """load_identity_files with a directory that does not exist."""

    def test_returns_empty_for_nonexistent_dir(self, tmp_path):
        fake_dir = str(tmp_path / "does_not_exist")

        result = load_identity_files(fake_dir)

        assert result == ""

    def test_returns_empty_for_file_path(self, tmp_path):
        """Passing a file path instead of a directory should return ''."""
        some_file = tmp_path / "somefile.txt"
        some_file.write_text("content", encoding="utf-8")

        result = load_identity_files(str(some_file))

        assert result == ""


@pytest.mark.unit
class TestLoadIdentityFilesSizeCap:
    """load_identity_files enforces _MAX_IDENTITY_SIZE total cap."""

    def test_constants_are_correct(self):
        assert IDENTITY_FILES == ["SOUL.md", "IDENTITY.md", "USER.md"]
        assert _MAX_IDENTITY_SIZE == 8192

    def test_single_file_truncated(self, tmp_path):
        oversized = "A" * (_MAX_IDENTITY_SIZE + 1000)
        (tmp_path / "SOUL.md").write_text(oversized, encoding="utf-8")

        result = load_identity_files(str(tmp_path))

        # Should be truncated to _MAX_IDENTITY_SIZE + truncation marker
        assert "[...truncated]" in result
        # The content portion before the marker should be exactly _MAX_IDENTITY_SIZE chars
        content_before_marker = result.split("\n[...truncated]")[0]
        assert len(content_before_marker) == _MAX_IDENTITY_SIZE

    def test_second_file_truncated_when_first_uses_most_budget(self, tmp_path):
        """First file uses most of the budget, second gets truncated."""
        first_content = "B" * (_MAX_IDENTITY_SIZE - 100)
        second_content = "C" * 500  # Would exceed remaining budget

        (tmp_path / "SOUL.md").write_text(first_content, encoding="utf-8")
        (tmp_path / "IDENTITY.md").write_text(second_content, encoding="utf-8")

        result = load_identity_files(str(tmp_path))

        # First file should be fully present
        assert first_content in result
        # Second file should be truncated since remaining budget < 500
        assert "[...truncated]" in result

    def test_third_file_skipped_when_budget_exhausted(self, tmp_path):
        """When first file exhausts the budget, remaining files are skipped."""
        full_budget = "D" * _MAX_IDENTITY_SIZE
        (tmp_path / "SOUL.md").write_text(full_budget, encoding="utf-8")
        (tmp_path / "IDENTITY.md").write_text("Should not appear.", encoding="utf-8")
        (tmp_path / "USER.md").write_text("Also should not appear.", encoding="utf-8")

        result = load_identity_files(str(tmp_path))

        assert "Should not appear" not in result
        assert "Also should not appear" not in result


@pytest.mark.unit
class TestPromptComposerIdentityInjection:
    """PromptComposer injects identity content when workspace_dir has identity files."""

    def test_injects_workspace_identity_section(self, tmp_path):
        (tmp_path / "SOUL.md").write_text("I am a research agent.", encoding="utf-8")

        composer = PromptComposer(model="generic")
        prompt = composer.compose(
            identity="Base identity",
            task="Do something",
            workspace_dir=str(tmp_path),
        )

        assert "Workspace Identity" in prompt
        assert "I am a research agent." in prompt

    def test_identity_appears_before_task(self, tmp_path):
        (tmp_path / "SOUL.md").write_text("Soul persona.", encoding="utf-8")

        composer = PromptComposer(model="generic")
        prompt = composer.compose(
            task="Perform the task",
            workspace_dir=str(tmp_path),
        )

        identity_pos = prompt.index("Soul persona.")
        task_pos = prompt.index("Perform the task")
        assert identity_pos < task_pos

    def test_identity_with_claude_model_uses_xml_tags(self, tmp_path):
        (tmp_path / "SOUL.md").write_text("Claude soul.", encoding="utf-8")

        composer = PromptComposer(model="claude-sonnet")
        prompt = composer.compose(
            workspace_dir=str(tmp_path),
        )

        assert "<workspace_identity>" in prompt
        assert "Claude soul." in prompt
        assert "</workspace_identity>" in prompt

    def test_identity_with_gpt_model_uses_markdown(self, tmp_path):
        (tmp_path / "SOUL.md").write_text("GPT soul.", encoding="utf-8")

        composer = PromptComposer(model="gpt-4o")
        prompt = composer.compose(
            workspace_dir=str(tmp_path),
        )

        assert "# Workspace Identity" in prompt
        assert "GPT soul." in prompt


@pytest.mark.unit
class TestPromptComposerNoIdentity:
    """PromptComposer with no identity files — no 'Workspace Identity' section."""

    def test_no_identity_section_when_no_files(self, tmp_path):
        composer = PromptComposer(model="generic")
        prompt = composer.compose(
            identity="Agent role",
            task="Do work",
            workspace_dir=str(tmp_path),
        )

        assert "Workspace Identity" not in prompt

    def test_no_identity_section_when_no_workspace_dir(self):
        composer = PromptComposer(model="generic")
        prompt = composer.compose(
            identity="Agent role",
            task="Do work",
        )

        assert "Workspace Identity" not in prompt
        assert "workspace_identity" not in prompt

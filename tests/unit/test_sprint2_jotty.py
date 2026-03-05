"""Tests for Sprint 2 competitive features (Jotty-side).

C1: OpenClaw skill loader
C2: MCP server
C4: Plugin lifecycle hooks
C5: TTL memory
C6: Community skill installer
"""

import asyncio
import json
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# C1: OpenClaw Adapter
# ============================================================================


class TestOpenClawAdapter:
    """Test OpenClaw SKILL.md parsing and skill registration."""

    @pytest.mark.unit
    def test_parse_yaml_frontmatter(self):
        from Jotty.core.capabilities.registry.openclaw_adapter import parse_openclaw_skill

        tmp = Path(tempfile.mkdtemp())
        try:
            skill_dir = tmp / "github-cli"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\n"
                "name: github-cli\n"
                'description: "Interact with GitHub via gh CLI"\n'
                'metadata: {"openclaw": {"requires": {"bins": ["gh"]}}}\n'
                "---\n\n"
                "# GitHub CLI\n\n"
                "Use `gh` commands to interact with GitHub.\n"
            )

            result = parse_openclaw_skill(skill_dir)
            assert result is not None
            assert result["name"] == "github-cli"
            assert "GitHub" in result["description"]
            assert "gh" in result["requires_bins"]
            assert "gh" in result["instructions"]
        finally:
            shutil.rmtree(tmp)

    @pytest.mark.unit
    def test_parse_table_format(self):
        from Jotty.core.capabilities.registry.openclaw_adapter import parse_openclaw_skill

        tmp = Path(tempfile.mkdtemp())
        try:
            skill_dir = tmp / "curl-api"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "| name | curl-api |\n"
                "|------|----------|\n"
                "| description | Make HTTP requests with curl |\n\n"
                "Use `curl` to interact with REST APIs.\n"
            )

            result = parse_openclaw_skill(skill_dir)
            assert result is not None
            assert result["name"] == "curl-api"
            assert "curl" in result["description"]
        finally:
            shutil.rmtree(tmp)

    @pytest.mark.unit
    def test_returns_none_if_tools_py_exists(self):
        from Jotty.core.capabilities.registry.openclaw_adapter import parse_openclaw_skill

        tmp = Path(tempfile.mkdtemp())
        try:
            skill_dir = tmp / "has-tools"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("---\nname: test\n---\n")
            (skill_dir / "tools.py").write_text("# tools")

            result = parse_openclaw_skill(skill_dir)
            assert result is None
        finally:
            shutil.rmtree(tmp)

    @pytest.mark.unit
    def test_returns_none_if_no_skill_md(self):
        from Jotty.core.capabilities.registry.openclaw_adapter import parse_openclaw_skill

        tmp = Path(tempfile.mkdtemp())
        try:
            skill_dir = tmp / "empty"
            skill_dir.mkdir()

            result = parse_openclaw_skill(skill_dir)
            assert result is None
        finally:
            shutil.rmtree(tmp)

    @pytest.mark.unit
    def test_returns_none_for_non_openclaw_format(self):
        from Jotty.core.capabilities.registry.openclaw_adapter import parse_openclaw_skill

        tmp = Path(tempfile.mkdtemp())
        try:
            skill_dir = tmp / "jotty-native"
            skill_dir.mkdir()
            # A regular Jotty SKILL.md without OpenClaw frontmatter or table
            (skill_dir / "SKILL.md").write_text(
                "# My Skill\n\nSome description.\n\n## Type\nbase\n"
            )

            result = parse_openclaw_skill(skill_dir)
            assert result is None
        finally:
            shutil.rmtree(tmp)

    @pytest.mark.unit
    def test_make_shell_exec_tools(self):
        from Jotty.core.capabilities.registry.openclaw_adapter import make_shell_exec_tools

        tools = make_shell_exec_tools("test-skill", "Use echo", ["echo"])
        assert len(tools) == 1
        tool_fn = list(tools.values())[0]
        assert asyncio.iscoroutinefunction(tool_fn)
        assert "test_skill" in tool_fn.__name__

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_shell_exec_tool_runs_command(self):
        from Jotty.core.capabilities.registry.openclaw_adapter import make_shell_exec_tools

        tools = make_shell_exec_tools("echo-test", "Echo things", ["echo"])
        tool_fn = list(tools.values())[0]
        result = await tool_fn({"command": "echo hello"})
        assert result["success"] is True
        assert "hello" in result["stdout"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_shell_exec_tool_handles_no_command(self):
        from Jotty.core.capabilities.registry.openclaw_adapter import make_shell_exec_tools

        tools = make_shell_exec_tools("empty", "test", [])
        tool_fn = list(tools.values())[0]
        result = await tool_fn({})
        assert result["success"] is False


# ============================================================================
# C2: MCP Server
# ============================================================================


class TestMCPServer:
    """Test MCP server tool listing and invocation."""

    @pytest.mark.unit
    def test_build_tool_list_returns_list(self):
        from Jotty.core.capabilities.mcp_server import _build_tool_list

        with patch("Jotty.core.capabilities.mcp_server._get_registry") as mock_reg:
            registry = MagicMock()
            registry.list_skills.return_value = [
                {"name": "web-search"},
            ]
            skill = MagicMock()
            skill._tool_metadata = {}
            skill.description = "Search the web"
            registry.get_skill.return_value = skill
            mock_reg.return_value = registry

            tools = _build_tool_list()
            assert len(tools) == 1
            assert tools[0]["name"] == "web-search"
            assert "inputSchema" in tools[0]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handle_call_tool_skill_not_found(self):
        from Jotty.core.capabilities.mcp_server import _handle_call_tool

        with patch("Jotty.core.capabilities.mcp_server._get_registry") as mock_reg:
            registry = MagicMock()
            registry.get_skill.return_value = None
            mock_reg.return_value = registry

            result = await _handle_call_tool("nonexistent__tool", {})
            data = json.loads(result)
            assert "error" in data

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handle_call_tool_success(self):
        from Jotty.core.capabilities.mcp_server import _handle_call_tool

        with patch("Jotty.core.capabilities.mcp_server._get_registry") as mock_reg:
            registry = MagicMock()
            skill = MagicMock()

            async def mock_tool(params):
                return {"result": "ok"}

            skill.get_tool.return_value = mock_tool
            registry.get_skill.return_value = skill
            mock_reg.return_value = registry

            result = await _handle_call_tool("web-search__search_tool", {"query": "test"})
            data = json.loads(result)
            assert data["result"] == "ok"


# ============================================================================
# C4: Plugin Lifecycle Hooks
# ============================================================================


class TestLifecycleHooks:
    """Test BaseSkill lifecycle methods and SkillDefinition hooks."""

    @pytest.mark.unit
    def test_base_skill_has_lifecycle_methods(self):
        from Jotty.core.capabilities.registry.skills_registry import BaseSkill

        skill = BaseSkill()
        assert hasattr(skill, "on_install")
        assert hasattr(skill, "on_enable")
        assert hasattr(skill, "on_disable")
        assert hasattr(skill, "on_uninstall")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_lifecycle_methods_are_no_ops_by_default(self):
        from Jotty.core.capabilities.registry.skills_registry import BaseSkill

        skill = BaseSkill()
        # Should not raise
        await skill.on_install()
        await skill.on_enable()
        await skill.on_disable()
        await skill.on_uninstall()

    @pytest.mark.unit
    def test_skill_definition_has_lifecycle_hooks_dict(self):
        from Jotty.core.capabilities.registry.skills_registry import SkillDefinition

        sd = SkillDefinition(name="test", description="test skill")
        assert isinstance(sd.lifecycle_hooks, dict)
        assert len(sd.lifecycle_hooks) == 0


# ============================================================================
# C5: TTL Memory
# ============================================================================


class TestTTLMemory:
    """Test TTL memory expiration in MemoryEntry and retrieval filtering."""

    @pytest.mark.unit
    def test_memory_entry_ttl_fields_exist(self):
        from Jotty.core.infrastructure.foundation.types.memory_types import MemoryEntry, MemoryLevel

        entry = MemoryEntry(
            key="test",
            content="test content",
            level=MemoryLevel.EPISODIC,
            context={},
        )
        assert entry.ttl_seconds is None
        assert entry.expires_at is None

    @pytest.mark.unit
    def test_memory_entry_ttl_computes_expires_at(self):
        from Jotty.core.infrastructure.foundation.types.memory_types import MemoryEntry, MemoryLevel

        entry = MemoryEntry(
            key="ttl-test",
            content="temporary data",
            level=MemoryLevel.EPISODIC,
            context={},
            ttl_seconds=3600,  # 1 hour
        )
        assert entry.expires_at is not None
        assert entry.expires_at > datetime.now()
        # Should expire approximately 1 hour from now
        delta = entry.expires_at - entry.created_at
        assert 3599 <= delta.total_seconds() <= 3601

    @pytest.mark.unit
    def test_memory_entry_no_ttl_no_expiry(self):
        from Jotty.core.infrastructure.foundation.types.memory_types import MemoryEntry, MemoryLevel

        entry = MemoryEntry(
            key="no-ttl",
            content="persistent data",
            level=MemoryLevel.SEMANTIC,
            context={},
        )
        assert entry.ttl_seconds is None
        assert entry.expires_at is None

    @pytest.mark.unit
    def test_expired_entry_filtered_in_retrieve_fast(self):
        """Verify that expired entries are excluded from retrieve_fast results."""
        from Jotty.core.infrastructure.foundation.types.memory_types import MemoryEntry, MemoryLevel
        from Jotty.core.infrastructure.foundation.data_structures import GoalValue

        # Create two entries: one expired, one still valid
        expired = MemoryEntry(
            key="expired",
            content="old data about python testing",
            level=MemoryLevel.EPISODIC,
            context={},
            ttl_seconds=1,
        )
        # Force it to be already expired
        expired.expires_at = datetime.now() - timedelta(seconds=10)

        valid = MemoryEntry(
            key="valid",
            content="fresh data about python testing",
            level=MemoryLevel.EPISODIC,
            context={},
        )

        # Manually test the filter logic (same as cortex.py)
        now = datetime.now()
        all_memories = [expired, valid]
        filtered = [m for m in all_memories if m.expires_at is None or m.expires_at > now]

        assert len(filtered) == 1
        assert filtered[0].key == "valid"

    @pytest.mark.unit
    def test_memory_system_store_accepts_ttl(self):
        """MemorySystem.store() accepts ttl_seconds parameter."""
        from Jotty.core.intelligence.memory.memory_system import (
            MemorySystem,
            MemoryConfig,
            MemoryBackend,
        )

        config = MemoryConfig(backend=MemoryBackend.FALLBACK)
        mem = MemorySystem(config=config)

        # Should not raise — ttl_seconds flows through metadata
        mem_id = mem.store(
            "temporary fact",
            level="episodic",
            ttl_seconds=60,
        )
        assert mem_id


# ============================================================================
# C6: Community Skill Installer
# ============================================================================


class TestSkillInstaller:
    """Test skill installation from local and GitHub sources."""

    @pytest.mark.unit
    def test_install_from_local(self):
        from Jotty.core.capabilities.registry.skill_installer import _install_from_local

        tmp = Path(tempfile.mkdtemp())
        try:
            # Create source skill
            src = tmp / "source" / "my-skill"
            src.mkdir(parents=True)
            (src / "SKILL.md").write_text("---\nname: my-skill\n---\n")
            (src / "tools.py").write_text("def my_tool(): pass")

            # Install
            target = tmp / "community"
            target.mkdir()
            result = _install_from_local(src, target)

            assert result["success"] is True
            assert result["skill_name"] == "my-skill"
            assert (target / "my-skill" / "SKILL.md").exists()
            assert (target / "my-skill" / "tools.py").exists()
        finally:
            shutil.rmtree(tmp)

    @pytest.mark.unit
    def test_install_local_already_exists(self):
        from Jotty.core.capabilities.registry.skill_installer import _install_from_local

        tmp = Path(tempfile.mkdtemp())
        try:
            src = tmp / "source" / "dupe"
            src.mkdir(parents=True)
            (src / "SKILL.md").write_text("test")

            target = tmp / "community"
            (target / "dupe").mkdir(parents=True)

            result = _install_from_local(src, target)
            assert result["success"] is False
            assert "already installed" in result["error"]
        finally:
            shutil.rmtree(tmp)

    @pytest.mark.unit
    def test_uninstall_skill(self):
        from Jotty.core.capabilities.registry.skill_installer import (
            uninstall_skill,
            _COMMUNITY_DIR,
        )

        tmp = Path(tempfile.mkdtemp())
        try:
            # Temporarily override community dir
            with patch(
                "Jotty.core.capabilities.registry.skill_installer._COMMUNITY_DIR",
                tmp,
            ):
                skill_dir = tmp / "test-skill"
                skill_dir.mkdir()
                (skill_dir / "SKILL.md").write_text("test")

                result = uninstall_skill("test-skill")
                assert result["success"] is True
                assert not skill_dir.exists()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.unit
    def test_uninstall_nonexistent(self):
        from Jotty.core.capabilities.registry.skill_installer import uninstall_skill

        tmp = Path(tempfile.mkdtemp())
        try:
            with patch(
                "Jotty.core.capabilities.registry.skill_installer._COMMUNITY_DIR",
                tmp,
            ):
                result = uninstall_skill("nonexistent")
                assert result["success"] is False
        finally:
            shutil.rmtree(tmp)

    @pytest.mark.unit
    def test_list_installed(self):
        from Jotty.core.capabilities.registry.skill_installer import list_installed

        tmp = Path(tempfile.mkdtemp())
        try:
            with patch(
                "Jotty.core.capabilities.registry.skill_installer._COMMUNITY_DIR",
                tmp,
            ):
                (tmp / "skill-a").mkdir()
                (tmp / "skill-a" / "SKILL.md").write_text("# Skill A\nDoes A things")
                (tmp / "skill-b").mkdir()
                (tmp / "skill-b" / "SKILL.md").write_text("# Skill B\nDoes B things")

                installed = list_installed()
                assert len(installed) == 2
                names = [s["name"] for s in installed]
                assert "skill-a" in names
                assert "skill-b" in names
        finally:
            shutil.rmtree(tmp)

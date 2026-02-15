"""
Comprehensive CLI Tests
=======================

Test all Jotty CLI features for automated testing and n8n workflows.
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCLICommands:
    """Test CLI command execution."""

    @pytest.fixture
    def cli(self):
        """Create CLI instance."""
        from Jotty.apps.cli.app import JottyCLI
        return JottyCLI()

    @pytest.mark.asyncio
    async def test_help_command(self, cli):
        """Test /help command."""
        result = await cli.run_once("/help")
        assert result is not None

    @pytest.mark.asyncio
    async def test_skills_list(self, cli):
        """Test /skills command."""
        result = await cli.run_once("/skills")
        assert result is not None

    @pytest.mark.asyncio
    async def test_agents_list(self, cli):
        """Test /agents command."""
        result = await cli.run_once("/agents")
        assert result is not None

    @pytest.mark.asyncio
    async def test_config_show(self, cli):
        """Test /config show."""
        result = await cli.run_once("/config show")
        assert result is not None

    @pytest.mark.asyncio
    async def test_stats_command(self, cli):
        """Test /stats command."""
        result = await cli.run_once("/stats")
        assert result is not None

    @pytest.mark.asyncio
    async def test_workflow_templates(self, cli):
        """Test /workflow templates."""
        result = await cli.run_once("/workflow templates")
        assert result is not None


class TestMLPipeline:
    """Test ML commands."""

    @pytest.fixture
    def cli(self):
        from Jotty.apps.cli.app import JottyCLI
        return JottyCLI()

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skipif(not os.getenv('ANTHROPIC_API_KEY'), reason="Requires ANTHROPIC_API_KEY for real LLM calls")
    async def test_ml_iris(self, cli):
        """Test /ml iris (fast dataset)."""
        result = await cli.run_once("/ml iris --iterations 1")
        assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skipif(not os.getenv('ANTHROPIC_API_KEY'), reason="Requires ANTHROPIC_API_KEY for real LLM calls")
    async def test_ml_wine(self, cli):
        """Test /ml wine."""
        result = await cli.run_once("/ml wine --iterations 1")
        assert result is not None


class TestFileOperations:
    """Test file-related commands."""

    @pytest.fixture
    def cli(self):
        from Jotty.apps.cli.app import JottyCLI
        return JottyCLI()

    @pytest.mark.asyncio
    async def test_preview_tools(self, cli):
        """Test /preview tools."""
        result = await cli.run_once("/preview tools")
        assert result is not None

    @pytest.mark.asyncio
    async def test_browse_current(self, cli):
        """Test /browse ."""
        result = await cli.run_once("/browse .")
        assert result is not None

    @pytest.mark.asyncio
    async def test_export_history(self, cli):
        """Test /export history."""
        result = await cli.run_once("/export history")
        assert result is not None


class TestResearch:
    """Test research command."""

    @pytest.fixture
    def cli(self):
        from Jotty.apps.cli.app import JottyCLI
        return JottyCLI()

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.skipif(not os.getenv('ANTHROPIC_API_KEY'), reason="Requires ANTHROPIC_API_KEY for real LLM calls")
    async def test_research_quick(self, cli):
        """Test /research with quick mode."""
        result = await cli.run_once("/research python --quick")
        assert result is not None


class TestSessionManagement:
    """Test session commands."""

    @pytest.fixture
    def cli(self):
        from Jotty.apps.cli.app import JottyCLI
        return JottyCLI()

    @pytest.mark.asyncio
    async def test_resume_list(self, cli):
        """Test /resume list."""
        result = await cli.run_once("/resume list")
        assert result is not None


class TestWorkflowIntegration:
    """Test workflow/n8n integration."""

    @pytest.fixture
    def cli(self):
        from Jotty.apps.cli.app import JottyCLI
        return JottyCLI()

    @pytest.mark.asyncio
    async def test_workflow_list(self, cli):
        """Test /workflow list."""
        result = await cli.run_once("/workflow list")
        assert result is not None

    @pytest.mark.asyncio
    async def test_workflow_export(self, cli):
        """Test /workflow export."""
        result = await cli.run_once("/workflow export")
        assert result is not None

    @pytest.mark.asyncio
    async def test_workflow_status(self, cli):
        """Test /workflow status."""
        result = await cli.run_once("/workflow status")
        assert result is not None


class TestCommandRegistry:
    """Test command registry."""

    def test_commands_registered(self):
        """Test all commands are registered."""
        from Jotty.apps.cli.commands import register_all_commands
        from Jotty.apps.cli.commands.base import CommandRegistry

        registry = CommandRegistry()
        register_all_commands(registry)

        # Should have 21+ commands
        assert len(registry._commands) >= 20

        # Check key commands exist
        expected = ['run', 'skills', 'agents', 'ml', 'research', 'workflow',
                   'preview', 'browse', 'export', 'resume', 'J']
        for cmd in expected:
            assert registry.get(cmd) is not None, f"Command {cmd} not found"

    def test_aliases_work(self):
        """Test command aliases resolve correctly."""
        from Jotty.apps.cli.commands import register_all_commands
        from Jotty.apps.cli.commands.base import CommandRegistry

        registry = CommandRegistry()
        register_all_commands(registry)

        # Test aliases
        assert registry.get('n8n').name == 'workflow'
        assert registry.get('schedule').name == 'workflow'
        assert registry.get('automl').name == 'ml'
        assert registry.get('search').name == 'research'


class TestCompleter:
    """Test autocomplete functionality."""

    def test_command_completions(self):
        """Test command completions."""
        from Jotty.apps.cli.commands import register_all_commands
        from Jotty.apps.cli.commands.base import CommandRegistry
        from Jotty.apps.cli.repl.completer import CommandCompleter
        from prompt_toolkit.document import Document

        registry = CommandRegistry()
        register_all_commands(registry)
        completer = CommandCompleter(registry)

        # Test / completions
        doc = Document('/')
        completions = list(completer.get_completions(doc, None))
        assert len(completions) > 10

    def test_ml_dataset_completions(self):
        """Test ML dataset completions."""
        from Jotty.apps.cli.commands import register_all_commands
        from Jotty.apps.cli.commands.base import CommandRegistry
        from Jotty.apps.cli.repl.completer import CommandCompleter
        from prompt_toolkit.document import Document

        registry = CommandRegistry()
        register_all_commands(registry)
        completer = CommandCompleter(registry)

        # Test /ml completions
        doc = Document('/ml ')
        completions = list(completer.get_completions(doc, None))
        completion_texts = [c.text for c in completions]

        assert 'titanic' in completion_texts
        assert 'iris' in completion_texts


class TestAPIServer:
    """Test API server components."""

    def test_api_server_creation(self):
        """Test API server can be created."""
        try:
            from Jotty.apps.cli.api import JottyAPIServer
            server = JottyAPIServer()
            assert server.port == 8765
        except ImportError:
            pytest.skip("FastAPI not installed")

    def test_api_app_creation(self):
        """Test FastAPI app can be created."""
        try:
            from Jotty.apps.cli.api import JottyAPIServer
            server = JottyAPIServer()
            app = server.create_app()
            assert app is not None
        except ImportError:
            pytest.skip("FastAPI not installed")


# Run specific test groups
def run_quick_tests():
    """Run quick tests only (no slow markers)."""
    pytest.main([__file__, "-v", "-m", "not slow", "--tb=short"])


def run_all_tests():
    """Run all tests."""
    pytest.main([__file__, "-v", "--tb=short"])


def run_ml_tests():
    """Run ML tests only."""
    pytest.main([__file__, "-v", "-k", "TestML", "--tb=short"])


def run_workflow_tests():
    """Run workflow tests only."""
    pytest.main([__file__, "-v", "-k", "TestWorkflow", "--tb=short"])


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "quick":
            run_quick_tests()
        elif test_type == "ml":
            run_ml_tests()
        elif test_type == "workflow":
            run_workflow_tests()
        else:
            run_all_tests()
    else:
        run_quick_tests()

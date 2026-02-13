"""
Unified Command Service
=======================

Single source of truth for all Jotty commands.
Used by CLI, Web API, Telegram bot - ensuring identical behavior everywhere.

Usage:
    from Jotty.core.services.command_service import CommandService

    service = CommandService()

    # List all commands
    commands = service.list_commands()

    # Execute a command
    result = await service.execute("/task list")
"""

import asyncio
import io
import re
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class CommandInfo:
    """Command metadata."""
    name: str
    description: str
    usage: str
    category: str
    aliases: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CommandExecutionResult:
    """Result from command execution."""
    success: bool
    output: str
    error: Optional[str] = None
    data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "data": self.data
        }


class CommandService:
    """
    Unified command service for all interfaces.

    Provides:
    - list_commands(): Get all available commands with metadata
    - execute(command_str): Execute a command string
    - get_command(name): Get info about a specific command

    All interfaces (CLI, Web, Telegram) should use this service
    to ensure consistent behavior.
    """

    def __init__(self):
        self._registry = None
        self._cli = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of command registry."""
        if self._initialized:
            return

        from Jotty.cli.commands import CommandRegistry, register_all_commands

        self._registry = CommandRegistry()
        register_all_commands(self._registry)
        self._initialized = True

    def _get_cli(self):
        """Get or create CLI instance for command execution."""
        if self._cli is None:
            from Jotty.cli.app import JottyCLI
            self._cli = JottyCLI(no_color=True)
        return self._cli

    def list_commands(self, category: Optional[str] = None) -> List[CommandInfo]:
        """
        List all available commands.

        Args:
            category: Optional filter by category

        Returns:
            List of CommandInfo objects
        """
        self._ensure_initialized()

        commands = []
        for name, cmd in self._registry._commands.items():
            # Skip aliases (only show primary names)
            if name in getattr(cmd, 'aliases', []):
                continue

            info = CommandInfo(
                name=name,
                description=getattr(cmd, 'description', ''),
                usage=getattr(cmd, 'usage', f'/{name}'),
                category=getattr(cmd, 'category', 'general'),
                aliases=getattr(cmd, 'aliases', [])
            )

            if category is None or info.category == category:
                commands.append(info)

        # Sort by category then name
        commands.sort(key=lambda c: (c.category, c.name))
        return commands

    def list_commands_dict(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List commands as dictionaries (for JSON serialization)."""
        return [cmd.to_dict() for cmd in self.list_commands(category)]

    def get_categories(self) -> List[str]:
        """Get all command categories."""
        commands = self.list_commands()
        categories = sorted(set(cmd.category for cmd in commands))
        return categories

    def get_command(self, name: str) -> Optional[CommandInfo]:
        """
        Get info about a specific command.

        Args:
            name: Command name (with or without /)

        Returns:
            CommandInfo or None if not found
        """
        self._ensure_initialized()

        # Strip leading slash if present
        name = name.lstrip('/')

        cmd = self._registry._commands.get(name)
        if cmd is None:
            # Check aliases
            for cmd_name, command in self._registry._commands.items():
                if name in getattr(command, 'aliases', []):
                    cmd = command
                    name = cmd_name
                    break

        if cmd is None:
            return None

        return CommandInfo(
            name=name,
            description=getattr(cmd, 'description', ''),
            usage=getattr(cmd, 'usage', f'/{name}'),
            category=getattr(cmd, 'category', 'general'),
            aliases=getattr(cmd, 'aliases', [])
        )

    def is_command(self, text: str) -> bool:
        """Check if text is a command (starts with /)."""
        return text.strip().startswith('/')

    def parse_command(self, text: str) -> tuple[str, str]:
        """
        Parse command text into command name and args.

        Args:
            text: Full command string like "/task list --status=pending"

        Returns:
            Tuple of (command_name, args_string)
        """
        text = text.strip()
        if not text.startswith('/'):
            return ('', text)

        # Remove leading slash
        text = text[1:]

        # Split into command and args
        parts = text.split(None, 1)
        command = parts[0] if parts else ''
        args = parts[1] if len(parts) > 1 else ''

        return (command, args)

    async def execute(self, command_str: str) -> CommandExecutionResult:
        """
        Execute a command string.

        Args:
            command_str: Full command like "/task list" or "task list"

        Returns:
            CommandExecutionResult with output
        """
        # Ensure it starts with /
        if not command_str.startswith('/'):
            command_str = '/' + command_str

        try:
            cli = self._get_cli()

            # Capture output
            captured_output = []

            def clean_text(text):
                """Remove ANSI and Rich markup."""
                clean = re.sub(r'\x1b\[[0-9;]*m', '', str(text))
                clean = re.sub(r'\[/?[^\]]*\]', '', clean)
                return clean.strip()

            # Save original renderer methods
            original_print = cli.renderer.print
            original_info = cli.renderer.info
            original_success = cli.renderer.success
            original_warning = cli.renderer.warning
            original_error = cli.renderer.error

            # Monkey-patch to capture output
            cli.renderer.print = lambda t, *a, **kw: captured_output.append(clean_text(t))
            cli.renderer.info = lambda t: captured_output.append(f"ℹ {clean_text(t)}")
            cli.renderer.success = lambda t: captured_output.append(f" {clean_text(t)}")
            cli.renderer.warning = lambda t: captured_output.append(f" {clean_text(t)}")
            cli.renderer.error = lambda t: captured_output.append(f" {clean_text(t)}")

            # Capture panel output
            original_panel = getattr(cli.renderer, 'panel', None)
            cli.renderer.panel = lambda content, **kwargs: captured_output.append(
                f" {kwargs.get('title', 'Panel')}:\n{clean_text(content)}"
            )

            # Capture tree output
            original_tree = getattr(cli.renderer, 'tree', None)
            def capture_tree(data, **kwargs):
                title = kwargs.get('title', 'Data')
                if isinstance(data, dict):
                    lines = [f" {title}:"]
                    for k, v in data.items():
                        lines.append(f"  • {k}: {v}")
                    captured_output.append("\n".join(lines))
                else:
                    captured_output.append(f" {title}: {data}")
            cli.renderer.tree = capture_tree

            # Capture table output
            original_print_table = cli.renderer.tables.print_table
            def capture_table(table):
                try:
                    from rich.console import Console
                    string_io = io.StringIO()
                    console = Console(file=string_io, force_terminal=False, no_color=True)
                    console.print(table)
                    captured_output.append(string_io.getvalue())
                except Exception:
                    captured_output.append(str(table))
            cli.renderer.tables.print_table = capture_table

            try:
                # Execute command
                await cli._handle_command(command_str)

                output = "\n".join(captured_output) if captured_output else "Command executed"
                return CommandExecutionResult(
                    success=True,
                    output=output
                )

            finally:
                # Restore original methods
                cli.renderer.print = original_print
                cli.renderer.info = original_info
                cli.renderer.success = original_success
                cli.renderer.warning = original_warning
                cli.renderer.error = original_error
                cli.renderer.tables.print_table = original_print_table
                if original_panel:
                    cli.renderer.panel = original_panel
                if original_tree:
                    cli.renderer.tree = original_tree

        except Exception as e:
            return CommandExecutionResult(
                success=False,
                output="",
                error=str(e)
            )

    def execute_sync(self, command_str: str) -> CommandExecutionResult:
        """Synchronous wrapper for execute()."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.execute(command_str))
        finally:
            loop.close()


# Singleton instance
_command_service: Optional[CommandService] = None


def get_command_service() -> CommandService:
    """Get the singleton CommandService instance."""
    global _command_service
    if _command_service is None:
        _command_service = CommandService()
    return _command_service

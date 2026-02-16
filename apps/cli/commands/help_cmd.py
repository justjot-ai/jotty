"""
Help Command
============

Help system for CLI.
"""

from typing import TYPE_CHECKING

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class HelpCommand(BaseCommand):
    """Help system."""

    name = "help"
    aliases = ["?", "h"]
    description = "Show help for commands"
    usage = "/help [command]"
    category = "system"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute help command."""
        if args.positional:
            # Help for specific command
            cmd_name = args.positional[0]
            return await self._command_help(cmd_name, cli)
        else:
            # General help
            return await self._general_help(cli)

    async def _general_help(self, cli: "JottyCLI") -> CommandResult:
        """Show general help."""
        commands = cli.command_registry.list_commands()

        # Group by category
        categories = {}  # type: ignore[var-annotated]
        for cmd in commands:
            cat = cmd.get("category", "general")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(cmd)

        # Display commands table
        table = cli.renderer.tables.commands_table(commands)  # type: ignore[attr-defined]
        cli.renderer.tables.print_table(table)  # type: ignore[attr-defined]

        # Tips
        cli.renderer.newline()  # type: ignore[attr-defined]
        cli.renderer.info("Tips:")  # type: ignore[attr-defined]
        cli.renderer.print("  • Type /run <task> to execute tasks with AI")  # type: ignore[attr-defined]
        cli.renderer.print("  • Type naturally without / for chat mode")  # type: ignore[attr-defined]
        cli.renderer.print("  • Use Tab for autocomplete")  # type: ignore[attr-defined]
        cli.renderer.print("  • Use /quit to exit")  # type: ignore[attr-defined]

        return CommandResult.ok(data=commands)

    async def _command_help(self, cmd_name: str, cli: "JottyCLI") -> CommandResult:
        """Show help for specific command."""
        # Remove leading / if present
        if cmd_name.startswith("/"):
            cmd_name = cmd_name[1:]

        cmd = cli.command_registry.get(cmd_name)

        if not cmd:
            cli.renderer.error(f"Unknown command: {cmd_name}")  # type: ignore[attr-defined]
            cli.renderer.info("Type /help for list of commands")  # type: ignore[attr-defined]
            return CommandResult.fail(f"Unknown command: {cmd_name}")

        # Show detailed help
        help_text = cmd.help_text()
        cli.renderer.panel(help_text, title=f"Help: /{cmd.name}", style="cyan")  # type: ignore[attr-defined]

        return CommandResult.ok(data={"command": cmd.name, "help": help_text})

    def get_completions(self, partial: str) -> list:
        """Get command completions for help."""
        # This will be populated from the command registry
        return []


class QuitCommand(BaseCommand):
    """Exit the CLI."""

    name = "quit"
    aliases = ["q", "exit"]
    description = "Exit the CLI"
    usage = "/quit"
    category = "system"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Exit CLI."""
        cli.renderer.goodbye()  # type: ignore[attr-defined]
        return CommandResult.exit()


class ClearCommand(BaseCommand):
    """Clear the screen."""

    name = "clear"
    aliases = ["cls"]
    description = "Clear the terminal screen"
    usage = "/clear"
    category = "system"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Clear screen."""
        cli.renderer.clear()  # type: ignore[attr-defined]
        return CommandResult.ok()


class HistoryCommand(BaseCommand):
    """Show command history."""

    name = "history"
    aliases = ["hist"]
    description = "Show command history"
    usage = "/history [limit]"
    category = "system"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Show history."""
        limit = int(args.positional[0]) if args.positional else 20

        history = cli.session.get_history(limit)

        if not history:
            cli.renderer.info("No history yet")  # type: ignore[attr-defined]
            return CommandResult.ok(data=[])

        # Format history
        lines = []
        for i, entry in enumerate(history, 1):
            role = entry.get("role", "user")
            content = entry.get("content", "")[:60]
            lines.append(f"{i}. [{role}] {content}")

        cli.renderer.panel("\n".join(lines), title=f"History (last {len(history)})", style="dim")  # type: ignore[attr-defined]

        return CommandResult.ok(data=history)

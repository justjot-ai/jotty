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
        categories = {}
        for cmd in commands:
            cat = cmd.get("category", "general")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(cmd)

        # Display commands table
        table = cli.renderer.tables.commands_table(commands)
        cli.renderer.tables.print_table(table)

        # Tips
        cli.renderer.newline()
        cli.renderer.info("Tips:")
        cli.renderer.print("  • Type /run <task> to execute tasks with AI")
        cli.renderer.print("  • Type naturally without / for chat mode")
        cli.renderer.print("  • Use Tab for autocomplete")
        cli.renderer.print("  • Use /quit to exit")

        return CommandResult.ok(data=commands)

    async def _command_help(self, cmd_name: str, cli: "JottyCLI") -> CommandResult:
        """Show help for specific command."""
        # Remove leading / if present
        if cmd_name.startswith("/"):
            cmd_name = cmd_name[1:]

        cmd = cli.command_registry.get(cmd_name)

        if not cmd:
            cli.renderer.error(f"Unknown command: {cmd_name}")
            cli.renderer.info("Type /help for list of commands")
            return CommandResult.fail(f"Unknown command: {cmd_name}")

        # Show detailed help
        help_text = cmd.help_text()
        cli.renderer.panel(help_text, title=f"Help: /{cmd.name}", style="cyan")

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
        cli.renderer.goodbye()
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
        cli.renderer.clear()
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
            cli.renderer.info("No history yet")
            return CommandResult.ok(data=[])

        # Format history
        lines = []
        for i, entry in enumerate(history, 1):
            role = entry.get("role", "user")
            content = entry.get("content", "")[:60]
            lines.append(f"{i}. [{role}] {content}")

        cli.renderer.panel("\n".join(lines), title=f"History (last {len(history)})", style="dim")

        return CommandResult.ok(data=history)

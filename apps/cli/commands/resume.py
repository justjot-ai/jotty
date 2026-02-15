"""
Resume Command
==============

/resume - Continue from last session
"""

import logging
from typing import TYPE_CHECKING

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI

logger = logging.getLogger(__name__)


class ResumeCommand(BaseCommand):
    """
    /resume - Continue from a previous session.

    Load conversation history and context from a saved session.
    """

    name = "resume"
    aliases = ["r", "continue", "load"]
    description = "Resume from a previous session"
    usage = "/resume [session_id] or /resume (loads most recent)"
    category = "session"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute resume command."""

        session_id = args.positional[0] if args.positional else None

        # List sessions if requested
        if session_id == "list" or args.flags.get("list"):
            return await self._list_sessions(cli)

        # Load specific or most recent session
        sessions = cli.session.list_sessions()

        if not sessions:
            cli.renderer.warning("No saved sessions found.")
            cli.renderer.info("Start a conversation and it will auto-save.")
            return CommandResult.ok()

        if session_id:
            # Load specific session
            matching = [s for s in sessions if s["session_id"].startswith(session_id)]
            if not matching:
                cli.renderer.error(f"Session not found: {session_id}")
                cli.renderer.info("Use /resume list to see available sessions")
                return CommandResult.fail("Session not found")
            target_session = matching[0]
        else:
            # Load most recent (skip current)
            other_sessions = [s for s in sessions if s["session_id"] != cli.session.session_id]
            if not other_sessions:
                cli.renderer.info("No previous sessions to resume.")
                return CommandResult.ok()
            target_session = other_sessions[0]

        # Load the session
        cli.session.load(target_session["session_id"])

        # Show summary
        msg_count = len(cli.session.conversation_history)
        cli.renderer.success(f"Resumed session: {cli.session.session_id}")
        cli.renderer.info(f"Loaded {msg_count} messages from conversation history")

        # Show last few messages as context
        if msg_count > 0:
            cli.renderer.newline()
            cli.renderer.print("[bold]Recent context:[/bold]")
            recent = cli.session.conversation_history[-3:]
            for msg in recent:
                role_color = "cyan" if msg.role == "user" else "green"
                preview = msg.content[:100].replace("\n", " ")
                if len(msg.content) > 100:
                    preview += "..."
                cli.renderer.print(f"  [{role_color}]{msg.role}:[/{role_color}] {preview}")

        # Restore output history if available
        if hasattr(cli, "_output_history"):
            cli._output_history = []
        for msg in cli.session.conversation_history:
            if msg.role == "assistant" and len(msg.content) > 100:
                if not hasattr(cli, "_output_history"):
                    cli._output_history = []
                cli._output_history.append(msg.content)

        cli.renderer.newline()
        cli.renderer.info("Continue the conversation or use /export to access previous outputs")

        return CommandResult.ok(output=f"Resumed session {cli.session.session_id}")

    async def _list_sessions(self, cli: "JottyCLI") -> CommandResult:
        """List available sessions."""
        sessions = cli.session.list_sessions()

        if not sessions:
            cli.renderer.warning("No saved sessions found.")
            return CommandResult.ok()

        cli.renderer.print("\n[bold]Available Sessions:[/bold]")
        cli.renderer.print("[dim]" + "─" * 60 + "[/dim]")

        for i, session in enumerate(sessions[:10], 1):
            session_id = session["session_id"]
            created = session.get("created_at", "unknown")[:16]
            msg_count = session.get("message_count", 0)

            # Mark current session
            current = " [yellow](current)[/yellow]" if session_id == cli.session.session_id else ""

            cli.renderer.print(
                f"  [cyan]{session_id}[/cyan]{current}"
                f"  [dim]{created}[/dim]  "
                f"[white]{msg_count} msgs[/white]"
            )

        cli.renderer.print("[dim]" + "─" * 60 + "[/dim]")
        cli.renderer.print("[dim]Use: /resume <session_id> or /resume (loads most recent)[/dim]")

        return CommandResult.ok()

    def get_completions(self, partial: str) -> list:
        """Get session ID completions."""
        # Would need access to session manager here
        return ["list"]

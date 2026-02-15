"""
Export Command
==============

/export - Export session outputs to various formats
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI

logger = logging.getLogger(__name__)


class ExportCommand(BaseCommand):
    """
    /export - Export session outputs to files.

    Export conversation history, outputs, or code to various formats.
    """

    name = "export"
    aliases = ["e", "save", "dump"]
    description = "Export session outputs"
    usage = "/export [format] [filename] or /export last"
    category = "session"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute export command."""

        export_type = args.positional[0] if args.positional else "last"
        filename = args.positional[1] if len(args.positional) > 1 else None

        if export_type == "last":
            return await self._export_last_output(cli, filename)
        elif export_type == "history":
            return await self._export_history(cli, filename)
        elif export_type == "code":
            return await self._export_code(cli, filename)
        else:
            cli.renderer.error(f"Unknown export type: {export_type}")
            cli.renderer.info("Available: last, history, code")
            return CommandResult.fail("Unknown export type")

    async def _export_last_output(self, cli: "JottyCLI", filename: str = None) -> CommandResult:
        """Export the last assistant output."""

        # Find last assistant message
        for msg in reversed(cli.session.conversation_history):
            if msg.role == "assistant" and msg.content:
                content = msg.content
                break
        else:
            cli.renderer.warning("No assistant output to export.")
            return CommandResult.ok()

        if filename:
            path = Path(filename).expanduser()
        else:
            path = Path.cwd() / "jotty_output.md"

        path.write_text(content)
        cli.renderer.success(f"Exported to: {path}")

        return CommandResult.ok(output=str(path))

    async def _export_history(self, cli: "JottyCLI", filename: str = None) -> CommandResult:
        """Export full conversation history."""

        if not cli.session.conversation_history:
            cli.renderer.warning("No conversation history to export.")
            return CommandResult.ok()

        if filename:
            path = Path(filename).expanduser()
        else:
            path = Path.cwd() / f"jotty_session_{cli.session.session_id[:8]}.md"

        lines = ["# Jotty Session Export\n"]
        lines.append(f"Session ID: {cli.session.session_id}\n\n")

        for msg in cli.session.conversation_history:
            role = "**User**" if msg.role == "user" else "**Assistant**"
            lines.append(f"## {role}\n\n{msg.content}\n\n---\n\n")

        path.write_text("".join(lines))
        cli.renderer.success(
            f"Exported {len(cli.session.conversation_history)} messages to: {path}"
        )

        return CommandResult.ok(output=str(path))

    async def _export_code(self, cli: "JottyCLI", filename: str = None) -> CommandResult:
        """Export code blocks from conversation."""
        import re

        code_blocks = []

        for msg in cli.session.conversation_history:
            if msg.role == "assistant":
                # Extract code blocks
                blocks = re.findall(r"```(\w*)\n(.*?)```", msg.content, re.DOTALL)
                code_blocks.extend(blocks)

        if not code_blocks:
            cli.renderer.warning("No code blocks found in conversation.")
            return CommandResult.ok()

        if filename:
            path = Path(filename).expanduser()
        else:
            path = Path.cwd() / "jotty_code.txt"

        lines = []
        for lang, code in code_blocks:
            lines.append(f"# === {lang or 'code'} ===\n{code.strip()}\n\n")

        path.write_text("".join(lines))
        cli.renderer.success(f"Exported {len(code_blocks)} code blocks to: {path}")

        return CommandResult.ok(output=str(path))

    def get_completions(self, partial: str) -> list:
        """Get export type completions."""
        options = ["last", "history", "code"]
        if not partial:
            return options
        return [o for o in options if o.startswith(partial)]

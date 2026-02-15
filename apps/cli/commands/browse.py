"""
Browse Command
==============

/browse - Interactive file explorer with preview
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI

logger = logging.getLogger(__name__)


class BrowseCommand(BaseCommand):
    """
    /browse - Interactive file explorer.

    Uses fzf for fuzzy file selection with preview.
    Falls back to simple listing if fzf not available.
    """

    name = "browse"
    aliases = ["files", "explore", "ls", "dir"]
    description = "Browse files interactively"
    usage = "/browse [path] [--preview] [--type TYPE]"
    category = "files"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute browse command."""

        # Get path
        start_path = Path(args.positional[0]).expanduser() if args.positional else Path.cwd()

        if not start_path.exists():
            cli.renderer.error(f"Path not found: {start_path}")
            return CommandResult.fail("Path not found")

        # Get options
        file_type = args.flags.get("type", None)
        show_preview = args.flags.get("preview", True)

        # Check for fzf
        fzf_path = shutil.which("fzf")

        if fzf_path:
            return await self._browse_with_fzf(cli, start_path, file_type, show_preview)
        else:
            return await self._browse_simple(cli, start_path, file_type)

    async def _browse_with_fzf(
        self, cli: "JottyCLI", path: Path, file_type: Optional[str], show_preview: bool
    ) -> CommandResult:
        """Interactive browse with fzf."""

        # Build find command
        if file_type:
            find_cmd = f"find {path} -type f -name '*.{file_type}' 2>/dev/null"
        else:
            find_cmd = f"find {path} -type f 2>/dev/null | head -1000"

        # Build preview command
        preview_cmd = self._build_preview_cmd()

        # Build fzf command
        fzf_args = [
            "fzf",
            "--height=80%",
            "--layout=reverse",
            "--border=rounded",
            "--info=inline",
            "--header=Select file (Enter to preview, Ctrl-C to cancel)",
            "--prompt=🔍 ",
            "--pointer=▶",
            "--marker=✓",
            "--color=dark,fg:white,bg:-1,hl:cyan,fg+:white,bg+:#3a3a3a,hl+:cyan",
            "--color=info:yellow,prompt:cyan,pointer:magenta,marker:green,spinner:yellow",
        ]

        if show_preview and preview_cmd:
            fzf_args.extend(
                [
                    "--preview",
                    preview_cmd,
                    "--preview-window=right:50%:wrap",
                ]
            )

        try:
            # Run fzf
            process = subprocess.Popen(
                fzf_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Get file list
            find_result = subprocess.run(
                find_cmd, shell=True, capture_output=True, text=True, timeout=10
            )

            stdout, stderr = process.communicate(input=find_result.stdout, timeout=60)

            if process.returncode == 0 and stdout.strip():
                selected_file = stdout.strip()
                cli.renderer.success(f"Selected: {selected_file}")

                # Preview the selected file
                from .preview import PreviewCommand

                preview_cmd = PreviewCommand()
                await preview_cmd._preview_file(cli, Path(selected_file), max_lines=100)

                # Store for later use
                cli._last_selected_file = selected_file

                return CommandResult.ok(output=selected_file)
            else:
                cli.renderer.info("No file selected.")
                return CommandResult.ok()

        except subprocess.TimeoutExpired:
            cli.renderer.warning("Browse timed out.")
            return CommandResult.ok()
        except Exception as e:
            cli.renderer.error(f"Browse failed: {e}")
            return await self._browse_simple(cli, path, file_type)

    async def _browse_simple(
        self, cli: "JottyCLI", path: Path, file_type: Optional[str]
    ) -> CommandResult:
        """Simple file listing without fzf."""

        cli.renderer.print(f"\n[bold]📁 {path}[/bold]")
        cli.renderer.print("[dim]" + "─" * 60 + "[/dim]")

        # Collect files
        files = []
        dirs = []

        try:
            for item in sorted(path.iterdir()):
                if item.name.startswith("."):
                    continue

                if item.is_dir():
                    dirs.append(item)
                elif item.is_file():
                    if file_type and not item.suffix.lower() == f".{file_type}":
                        continue
                    files.append(item)

            # Show directories
            for d in dirs[:20]:
                cli.renderer.print(f"  [blue]📁 {d.name}/[/blue]")

            # Show files
            for f in files[:50]:
                size = self._format_size(f.stat().st_size)
                icon = self._get_file_icon(f.suffix)
                cli.renderer.print(f"  {icon} {f.name} [dim]({size})[/dim]")

            total = len(dirs) + len(files)
            if total > 70:
                cli.renderer.print(f"[dim]... and {total - 70} more items[/dim]")

            cli.renderer.print("[dim]" + "─" * 60 + "[/dim]")
            cli.renderer.info("Install fzf for interactive browsing: apt install fzf")

            return CommandResult.ok()

        except PermissionError:
            cli.renderer.error(f"Permission denied: {path}")
            return CommandResult.fail("Permission denied")

    def _build_preview_cmd(self) -> str:
        """Build preview command for fzf."""
        # Check available tools
        bat = shutil.which("bat") or shutil.which("batcat")
        chafa = shutil.which("chafa")
        pdftotext = shutil.which("pdftotext")
        catdoc = shutil.which("catdoc")

        # Build conditional preview
        parts = []

        # Images
        if chafa:
            parts.append(f"[[ {{}} =~ \\.(png|jpg|jpeg|gif|webp)$ ]] && {chafa} --size=60x30 {{}}")

        # PDF
        if pdftotext:
            parts.append(f"[[ {{}} =~ \\.pdf$ ]] && {pdftotext} -layout -nopgbrk {{}} - | head -50")

        # DOCX
        if catdoc:
            parts.append(f"[[ {{}} =~ \\.docx?$ ]] && {catdoc} {{}} | head -50")

        # Code/text with bat
        if bat:
            parts.append(f"{bat} --color=always --style=numbers --line-range=:50 {{}} 2>/dev/null")
        else:
            parts.append("head -50 {} 2>/dev/null")

        return " || ".join(parts) if parts else "head -50 {}"

    def _get_file_icon(self, suffix: str) -> str:
        """Get icon for file type."""
        icons = {
            ".py": "🐍",
            ".js": "📜",
            ".ts": "📘",
            ".md": "📝",
            ".txt": "📄",
            ".pdf": "📕",
            ".docx": "📘",
            ".doc": "📘",
            ".xlsx": "📊",
            ".xls": "📊",
            ".csv": "📊",
            ".json": "📋",
            ".yaml": "⚙️",
            ".yml": "⚙️",
            ".html": "🌐",
            ".css": "🎨",
            ".png": "🖼️",
            ".jpg": "🖼️",
            ".jpeg": "🖼️",
            ".gif": "🖼️",
            ".svg": "🖼️",
            ".mp3": "🎵",
            ".mp4": "🎬",
            ".zip": "📦",
            ".tar": "📦",
            ".gz": "📦",
            ".sh": "⚡",
            ".bash": "⚡",
            ".zsh": "⚡",
        }
        return icons.get(suffix.lower(), "📄")

    def _format_size(self, size: int) -> str:
        """Format file size."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.0f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"

    def get_completions(self, partial: str) -> list:
        """Get path completions."""
        if not partial:
            return [".", "~", "/"]

        path = Path(partial).expanduser()

        if path.is_dir():
            return [str(p) for p in path.iterdir() if p.is_dir()][:20]
        elif path.parent.exists():
            return [str(p) for p in path.parent.glob(f"{path.name}*") if p.is_dir()][:20]

        return [".", "~", "/"]

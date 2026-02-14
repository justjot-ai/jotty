"""
Preview Command
===============

/preview - Rich file preview in terminal with inline rendering
"""

import logging
import subprocess
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Any

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI

logger = logging.getLogger(__name__)


class PreviewCommand(BaseCommand):
    """
    /preview - Preview files beautifully in terminal.

    Supports: PDF, DOCX, MD, images, code files, and more.
    Uses best available tools: glow, bat, catdoc, pdftotext, chafa.
    """

    name = "preview"
    aliases = ["view", "v", "cat", "show"]
    description = "Preview files in terminal"
    usage = "/preview <file> [--lines N] [--raw]"
    category = "files"

    # Tool detection cache
    _tools_cache = None

    @classmethod
    def detect_tools(cls) -> dict:
        """Detect available preview tools."""
        if cls._tools_cache is not None:
            return cls._tools_cache

        tools = {
            'glow': shutil.which('glow'),        # Markdown
            'bat': shutil.which('bat') or shutil.which('batcat'),  # Code/text
            'catdoc': shutil.which('catdoc'),    # DOCX
            'pdftotext': shutil.which('pdftotext'),  # PDF
            'chafa': shutil.which('chafa'),      # Images
            'timg': shutil.which('timg'),        # Images (alt)
            'viu': shutil.which('viu'),          # Images (alt)
            'pandoc': shutil.which('pandoc'),    # Universal converter
            'lynx': shutil.which('lynx'),        # HTML
            'w3m': shutil.which('w3m'),          # HTML (alt)
            'jq': shutil.which('jq'),            # JSON
            'hexyl': shutil.which('hexyl'),      # Binary/hex
            'exiftool': shutil.which('exiftool'), # Metadata
        }

        cls._tools_cache = tools
        return tools

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute preview command."""

        if not args.positional:
            # Show last generated file or usage
            if hasattr(cli, '_last_output_path') and cli._last_output_path:
                file_path = cli._last_output_path
            else:
                cli.renderer.info("Usage: /preview <file>")
                cli.renderer.info("       /preview last  - Preview last generated file")
                cli.renderer.info("       /preview tools - Show available preview tools")
                return CommandResult.ok()
        else:
            file_path = args.positional[0]

        # Special commands
        if file_path == "tools":
            return await self._show_tools(cli)
        elif file_path == "last":
            if hasattr(cli, '_last_output_path') and cli._last_output_path:
                file_path = cli._last_output_path
            else:
                cli.renderer.warning("No recent file to preview.")
                return CommandResult.ok()

        # Expand path
        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            cli.renderer.error(f"File not found: {path}")
            return CommandResult.fail("File not found")

        # Get options
        max_lines = args.flags.get('lines', 50)
        raw_mode = args.flags.get('raw', False)

        # Preview the file
        await self._preview_file(cli, path, max_lines, raw_mode)

        return CommandResult.ok()

    async def _show_tools(self, cli: "JottyCLI") -> CommandResult:
        """Show available preview tools."""
        tools = self.detect_tools()

        cli.renderer.print("\n[bold]Preview Tools Status:[/bold]")
        cli.renderer.print("[dim]" + "─" * 50 + "[/dim]")

        tool_info = [
            ('glow', 'Markdown', 'go install github.com/charmbracelet/glow@latest'),
            ('bat', 'Code/Text', 'apt install bat'),
            ('catdoc', 'DOCX', 'apt install catdoc'),
            ('pdftotext', 'PDF', 'apt install poppler-utils'),
            ('chafa', 'Images', 'apt install chafa'),
            ('pandoc', 'Universal', 'apt install pandoc'),
            ('jq', 'JSON', 'apt install jq'),
            ('hexyl', 'Binary', 'cargo install hexyl'),
        ]

        for tool, purpose, install in tool_info:
            status = "[green][/green]" if tools.get(tool) else "[red][/red]"
            cli.renderer.print(f"  {status} [cyan]{tool:12}[/cyan] {purpose:12} [dim]{install}[/dim]")

        cli.renderer.print("[dim]" + "─" * 50 + "[/dim]")
        return CommandResult.ok()

    async def _preview_file(self, cli: 'JottyCLI', path: Path, max_lines: int = 50, raw: bool = False) -> Any:
        """Preview a file with the best available tool."""
        tools = self.detect_tools()
        suffix = path.suffix.lower()

        # File info header
        size = path.stat().st_size
        size_str = self._format_size(size)
        cli.renderer.print(f"\n[bold cyan] {path.name}[/bold cyan] [dim]({size_str})[/dim]")
        cli.renderer.print("[dim]" + "─" * 60 + "[/dim]")

        try:
            # Route to appropriate previewer
            if suffix in ['.md', '.markdown']:
                await self._preview_markdown(cli, path, tools, raw)
            elif suffix in ['.pdf']:
                await self._preview_pdf(cli, path, tools, max_lines)
            elif suffix in ['.docx', '.doc']:
                await self._preview_docx(cli, path, tools, max_lines)
            elif suffix in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg']:
                await self._preview_image(cli, path, tools)
            elif suffix in ['.json']:
                await self._preview_json(cli, path, tools, max_lines)
            elif suffix in ['.html', '.htm']:
                await self._preview_html(cli, path, tools, max_lines)
            elif suffix in ['.csv', '.tsv']:
                await self._preview_csv(cli, path, max_lines)
            elif self._is_binary(path):
                await self._preview_binary(cli, path, tools)
            else:
                # Default: code/text preview
                await self._preview_code(cli, path, tools, max_lines, raw)

        except Exception as e:
            cli.renderer.error(f"Preview failed: {e}")
            # Fallback to raw
            await self._preview_raw(cli, path, max_lines)

        cli.renderer.print("[dim]" + "─" * 60 + "[/dim]")

    async def _preview_markdown(self, cli: 'JottyCLI', path: Path, tools: dict, raw: bool) -> Any:
        """Preview markdown with glow or rich."""
        if tools['glow'] and not raw:
            result = subprocess.run(
                [tools['glow'], '-s', 'dark', '-w', '100', str(path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print(result.stdout)
                return

        # Fallback to Rich markdown
        content = path.read_text()
        cli.renderer.markdown(content)

    async def _preview_pdf(self, cli: 'JottyCLI', path: Path, tools: dict, max_lines: int) -> Any:
        """Preview PDF with pdftotext."""
        if tools['pdftotext']:
            result = subprocess.run(
                [tools['pdftotext'], '-layout', '-nopgbrk', str(path), '-'],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')[:max_lines]
                for line in lines:
                    print(line)
                if len(result.stdout.split('\n')) > max_lines:
                    cli.renderer.print(f"[dim]... ({len(result.stdout.split(chr(10))) - max_lines} more lines)[/dim]")
                return

        # Try pandoc
        if tools['pandoc']:
            result = subprocess.run(
                [tools['pandoc'], '-f', 'pdf', '-t', 'plain', str(path)],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')[:max_lines]
                for line in lines:
                    print(line)
                return

        cli.renderer.warning("PDF preview requires pdftotext: apt install poppler-utils")
        # Show metadata at least
        await self._show_file_metadata(cli, path, tools)

    async def _preview_docx(self, cli: 'JottyCLI', path: Path, tools: dict, max_lines: int) -> Any:
        """Preview DOCX with catdoc or pandoc."""
        if tools['catdoc']:
            result = subprocess.run(
                [tools['catdoc'], '-w', str(path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')[:max_lines]
                for line in lines:
                    print(line)
                if len(result.stdout.split('\n')) > max_lines:
                    cli.renderer.print(f"[dim]... (truncated)[/dim]")
                return

        if tools['pandoc']:
            result = subprocess.run(
                [tools['pandoc'], '-f', 'docx', '-t', 'plain', str(path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')[:max_lines]
                for line in lines:
                    print(line)
                return

        cli.renderer.warning("DOCX preview requires catdoc or pandoc")
        cli.renderer.info("  apt install catdoc  OR  apt install pandoc")

    async def _preview_image(self, cli: 'JottyCLI', path: Path, tools: dict) -> Any:
        """Preview image in terminal."""
        # Try chafa (best quality)
        if tools['chafa']:
            result = subprocess.run(
                [tools['chafa'], '--size=80x40', '--colors=256', str(path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print(result.stdout)
                return

        # Try timg
        if tools['timg']:
            result = subprocess.run(
                [tools['timg'], '-g', '80x40', str(path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print(result.stdout)
                return

        # Try viu
        if tools['viu']:
            result = subprocess.run(
                [tools['viu'], '-w', '80', str(path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print(result.stdout)
                return

        cli.renderer.warning("Image preview requires chafa, timg, or viu")
        cli.renderer.info("  apt install chafa")
        # Show metadata
        await self._show_file_metadata(cli, path, tools)

    async def _preview_json(self, cli: 'JottyCLI', path: Path, tools: dict, max_lines: int) -> Any:
        """Preview JSON with syntax highlighting."""
        if tools['jq']:
            result = subprocess.run(
                [tools['jq'], '-C', '.', str(path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')[:max_lines]
                for line in lines:
                    print(line)
                return

        if tools['bat']:
            result = subprocess.run(
                [tools['bat'], '--color=always', '-l', 'json', str(path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')[:max_lines]
                for line in lines:
                    print(line)
                return

        # Fallback to python json
        import json
        content = json.loads(path.read_text())
        formatted = json.dumps(content, indent=2)
        lines = formatted.split('\n')[:max_lines]
        for line in lines:
            print(line)

    async def _preview_html(self, cli: 'JottyCLI', path: Path, tools: dict, max_lines: int) -> Any:
        """Preview HTML as text."""
        if tools['lynx']:
            result = subprocess.run(
                [tools['lynx'], '-dump', '-width=100', str(path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')[:max_lines]
                for line in lines:
                    print(line)
                return

        if tools['w3m']:
            result = subprocess.run(
                [tools['w3m'], '-dump', '-cols', '100', str(path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')[:max_lines]
                for line in lines:
                    print(line)
                return

        if tools['pandoc']:
            result = subprocess.run(
                [tools['pandoc'], '-f', 'html', '-t', 'plain', str(path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')[:max_lines]
                for line in lines:
                    print(line)
                return

        # Raw HTML with bat
        await self._preview_code(cli, path, tools, max_lines, False)

    async def _preview_csv(self, cli: 'JottyCLI', path: Path, max_lines: int) -> Any:
        """Preview CSV as table."""
        import csv

        with open(path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)[:max_lines + 1]

        if not rows:
            cli.renderer.print("[dim]Empty file[/dim]")
            return

        # Calculate column widths
        col_widths = []
        for col_idx in range(len(rows[0])):
            width = max(len(str(row[col_idx])) if col_idx < len(row) else 0 for row in rows)
            col_widths.append(min(width, 30))  # Cap at 30

        # Print header
        header = rows[0]
        header_str = " │ ".join(str(h)[:w].ljust(w) for h, w in zip(header, col_widths))
        cli.renderer.print(f"[bold]{header_str}[/bold]")
        cli.renderer.print("─" * len(header_str))

        # Print rows
        for row in rows[1:]:
            row_str = " │ ".join(str(c)[:w].ljust(w) if i < len(row) else " " * w
                                 for i, (c, w) in enumerate(zip(row + [''] * len(col_widths), col_widths)))
            print(row_str)

    async def _preview_code(self, cli: 'JottyCLI', path: Path, tools: dict, max_lines: int, raw: bool) -> Any:
        """Preview code/text with syntax highlighting."""
        if tools['bat'] and not raw:
            result = subprocess.run(
                [tools['bat'], '--color=always', '--style=numbers,changes',
                 '-r', f'1:{max_lines}', str(path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print(result.stdout)
                return

        # Fallback to raw
        await self._preview_raw(cli, path, max_lines)

    async def _preview_binary(self, cli: 'JottyCLI', path: Path, tools: dict) -> Any:
        """Preview binary file with hexyl."""
        if tools['hexyl']:
            result = subprocess.run(
                [tools['hexyl'], '-n', '256', str(path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print(result.stdout)
                return

        # xxd fallback
        result = subprocess.run(
            ['xxd', '-l', '256', str(path)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(result.stdout)
            return

        cli.renderer.print("[dim]Binary file - use hexyl for preview[/dim]")

    async def _preview_raw(self, cli: 'JottyCLI', path: Path, max_lines: int) -> Any:
        """Raw text preview."""
        try:
            with open(path, 'r', errors='replace') as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        cli.renderer.print(f"[dim]... (truncated at {max_lines} lines)[/dim]")
                        break
                    print(line.rstrip())
        except Exception as e:
            cli.renderer.error(f"Cannot read file: {e}")

    async def _show_file_metadata(self, cli: 'JottyCLI', path: Path, tools: dict) -> Any:
        """Show file metadata."""
        import os
        from datetime import datetime

        stat = path.stat()
        cli.renderer.print(f"[dim]Size: {self._format_size(stat.st_size)}[/dim]")
        cli.renderer.print(f"[dim]Modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}[/dim]")

        if tools['exiftool']:
            result = subprocess.run(
                [tools['exiftool'], '-s', '-s', '-s',
                 '-FileType', '-ImageSize', '-PageCount', '-Author', str(path)],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split('\n')[:5]:
                    cli.renderer.print(f"[dim]{line}[/dim]")

    def _format_size(self, size: int) -> str:
        """Format file size."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def _is_binary(self, path: Path) -> bool:
        """Check if file is binary."""
        try:
            with open(path, 'rb') as f:
                chunk = f.read(8192)
                return b'\x00' in chunk
        except:
            return False

    def get_completions(self, partial: str) -> list:
        """Get file path completions."""
        if not partial:
            return ['last', 'tools']

        path = Path(partial).expanduser()

        if path.is_dir():
            return [str(p) for p in path.iterdir()][:20]
        elif path.parent.exists():
            return [str(p) for p in path.parent.glob(f"{path.name}*")][:20]

        return ['last', 'tools']

"""
Rich Renderer for Jotty CLI
===========================

Main rendering class using Rich library.
"""

import math
import os
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from enum import Enum
from typing import Any, Optional, List, Dict, Iterator
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.tree import Tree
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .themes import Theme, get_theme
from .progress import ProgressManager
from .tables import TableRenderer
from ..config.schema import ColorDepth, TerminalDetector


class ShimmerEffect:
    """
    Cosine wave animation across characters for loading indicators.

    True-color ANSI rendering with bold/dim fallback for 16/256-color terminals.
    Runs as a background daemon thread at ~20 FPS with a 2s period and 5-char band.
    """

    FPS = 20
    PERIOD = 2.0  # seconds for full wave cycle
    BAND_WIDTH = 5  # characters wide for the shimmer band

    def __init__(self, color_depth: ColorDepth = None) -> None:
        self._color_depth = color_depth or TerminalDetector.detect_color_depth()
        self._message = ""
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self, message: str = 'Working...') -> Any:
        """Start the shimmer animation."""
        self._message = message
        self._running = True
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self) -> Any:
        """Stop the shimmer animation and clear the line."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        # Clear the animation line
        sys.stdout.write("\r" + " " * (len(self._message) + 20) + "\r")
        sys.stdout.flush()

    def update(self, message: str) -> Any:
        """Update the shimmer message."""
        with self._lock:
            self._message = message

    def _animate(self) -> Any:
        """Background animation loop."""
        frame = 0
        interval = 1.0 / self.FPS
        while self._running:
            with self._lock:
                msg = self._message
            rendered = self._render_frame(msg, frame)
            sys.stdout.write(f"\r{rendered}")
            sys.stdout.flush()
            frame += 1
            time.sleep(interval)

    def _render_frame(self, message: str, frame: int) -> str:
        """Render a single frame of the shimmer effect."""
        t = frame / self.FPS
        phase = (t % self.PERIOD) / self.PERIOD * 2 * math.pi

        if self._color_depth == ColorDepth.TRUE_COLOR:
            return self._render_truecolor(message, phase)
        return self._render_fallback(message, phase)

    def _render_truecolor(self, message: str, phase: float) -> str:
        """Render with true-color ANSI codes."""
        chars = []
        for i, ch in enumerate(message):
            # Cosine wave: brightness varies per character position
            wave = (math.cos(phase - i * 2 * math.pi / self.BAND_WIDTH) + 1) / 2
            # Interpolate brightness: dim (120) to bright (255)
            brightness = int(120 + wave * 135)
            chars.append(f"\033[38;2;{brightness};{brightness};{brightness}m{ch}")
        chars.append("\033[0m")
        return "".join(chars)

    def _render_fallback(self, message: str, phase: float) -> str:
        """Render with bold/dim fallback for 16/256-color terminals."""
        chars = []
        for i, ch in enumerate(message):
            wave = (math.cos(phase - i * 2 * math.pi / self.BAND_WIDTH) + 1) / 2
            if wave > 0.7:
                chars.append(f"\033[1m{ch}\033[0m")  # Bold
            elif wave < 0.3:
                chars.append(f"\033[2m{ch}\033[0m")  # Dim
            else:
                chars.append(ch)
        return "".join(chars)


class MarkdownStreamRenderer:
    """
    Incremental markdown renderer that buffers chunks and renders complete blocks.

    Detects fenced code blocks, paragraphs (double newline), and headers
    across chunk boundaries. Renders complete blocks via Rich Markdown.
    """

    def __init__(self, console: 'Console' = None) -> None:
        self._console = console
        self._buffer = ""
        self._in_code_block = False
        self._code_fence_pattern = re.compile(r'^```')
        self._rendered_up_to = 0

    def feed(self, chunk: str) -> Any:
        """
        Feed a new chunk of text into the stream renderer.

        Args:
            chunk: Partial markdown text to buffer
        """
        self._buffer += chunk
        self._try_render()

    def _try_render(self) -> Any:
        """Attempt to render complete blocks from the buffer."""
        remaining = self._buffer[self._rendered_up_to:]

        while remaining:
            # Track code fence toggles
            if self._in_code_block:
                # Look for closing fence
                close_idx = remaining.find("\n```")
                if close_idx == -1:
                    break  # Wait for more data
                # Include the closing fence line
                end_of_fence = remaining.find("\n", close_idx + 1)
                if end_of_fence == -1:
                    end_of_fence = close_idx + 4  # Just ```
                else:
                    end_of_fence += 1  # Include newline
                block = remaining[:end_of_fence]
                self._render_block(block)
                self._rendered_up_to += end_of_fence
                remaining = remaining[end_of_fence:]
                self._in_code_block = False
                continue

            # Check for code fence opening
            fence_match = self._code_fence_pattern.search(remaining)
            if fence_match and (fence_match.start() == 0 or remaining[fence_match.start() - 1] == "\n"):
                # Render any text before the fence
                if fence_match.start() > 0:
                    pre_text = remaining[:fence_match.start()]
                    if pre_text.strip():
                        self._render_block(pre_text)
                    self._rendered_up_to += fence_match.start()
                    remaining = remaining[fence_match.start():]
                self._in_code_block = True
                continue

            # Look for paragraph break (double newline)
            para_break = remaining.find("\n\n")
            if para_break != -1:
                block = remaining[:para_break + 2]
                if block.strip():
                    self._render_block(block)
                self._rendered_up_to += para_break + 2
                remaining = remaining[para_break + 2:]
                continue

            # No complete block found, wait for more data
            break

    def _render_block(self, block: str) -> Any:
        """Render a complete markdown block."""
        block = block.strip()
        if not block:
            return
        if self._console and RICH_AVAILABLE:
            try:
                md = Markdown(block)
                self._console.print(md)
            except Exception:
                self._console.print(block)
        else:
            print(block)

    def flush(self) -> Any:
        """Render any remaining buffer content."""
        remaining = self._buffer[self._rendered_up_to:]
        if remaining.strip():
            self._render_block(remaining)
        self._buffer = ""
        self._rendered_up_to = 0
        self._in_code_block = False


class REPLState(Enum):
    """REPL interaction states for footer hints."""
    INPUT = "input"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    EXPORTING = "exporting"


class FooterHints:
    """
    Context-aware footer toolbar hints for prompt_toolkit.

    Shows relevant keyboard shortcuts based on current REPL state.
    """

    HINTS = {
        REPLState.INPUT: [
            ("Tab", "complete"),
            ("Ctrl+R", "history"),
            ("Ctrl+C", "cancel"),
            ("/help", "commands"),
        ],
        REPLState.EXECUTING: [
            ("Ctrl+C", "abort"),
        ],
        REPLState.REVIEWING: [
            ("c", "copy"),
            ("d", "docx"),
            ("p", "pdf"),
            ("m", "markdown"),
            ("Enter", "done"),
        ],
        REPLState.EXPORTING: [
            ("1-3", "choose format"),
            ("s", "skip"),
        ],
    }

    def __init__(self) -> None:
        self._state = REPLState.INPUT

    @property
    def state(self) -> REPLState:
        return self._state

    @state.setter
    def state(self, value: REPLState) -> Any:
        self._state = value

    def get_toolbar_text(self) -> str:
        """
        Get formatted toolbar text for the current state.

        Returns:
            HTML-formatted string for prompt_toolkit bottom_toolbar
        """
        hints = self.HINTS.get(self._state, [])
        parts = []
        for key, desc in hints:
            parts.append(f"<b>{key}</b>:{desc}")
        text = "  ".join(parts)
        # Truncate if needed (prompt_toolkit handles width, but be safe)
        if len(text) > 200:
            text = text[:197] + "..."
        return text

    def get_rich_text(self, max_width: int = 80) -> str:
        """
        Get Rich-formatted toolbar text.

        Args:
            max_width: Terminal width for truncation

        Returns:
            Rich markup string
        """
        hints = self.HINTS.get(self._state, [])
        parts = []
        for key, desc in hints:
            parts.append(f"[bold]{key}[/bold]:{desc}")
        text = "  ".join(parts)
        if len(text) > max_width:
            text = text[:max_width - 3] + "..."
        return f"[dim]{text}[/dim]"


class DesktopNotifier:
    """
    Platform-aware desktop notification sender.

    Supports macOS (osascript), Linux (notify-send), Windows (plyer).
    Only notifies if task took longer than a configurable threshold.
    """

    def __init__(self, threshold_seconds: int = 10) -> None:
        self._threshold = threshold_seconds
        self._platform = sys.platform

    def notify(self, title: str, message: str, elapsed: float = 0) -> Any:
        """
        Send a desktop notification if elapsed time exceeds threshold.

        Args:
            title: Notification title
            message: Notification body
            elapsed: Task duration in seconds
        """
        if elapsed < self._threshold:
            return

        try:
            if self._platform == "darwin":
                self._notify_macos(title, message)
            elif self._platform == "linux":
                self._notify_linux(title, message)
            elif self._platform == "win32":
                self._notify_windows(title, message)
        except Exception:
            pass  # Silently fail - notifications are non-critical

    def _notify_macos(self, title: str, message: str) -> Any:
        """Send notification via osascript on macOS."""
        safe_title = title.replace('"', '\\"')
        safe_message = message.replace('"', '\\"')
        script = f'display notification "{safe_message}" with title "{safe_title}"'
        subprocess.Popen(
            ["osascript", "-e", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _notify_linux(self, title: str, message: str) -> Any:
        """Send notification via notify-send on Linux."""
        subprocess.Popen(
            ["notify-send", title, message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _notify_windows(self, title: str, message: str) -> Any:
        """Send notification via plyer on Windows."""
        try:
            from plyer import notification as plyer_notification
            plyer_notification.notify(
                title=title,
                message=message,
                timeout=5,
            )
        except ImportError:
            pass  # plyer not installed


class RichRenderer:
    """
    Main renderer for CLI output using Rich library.

    Provides:
    - Formatted output (panels, markdown, code)
    - Tables (skills, agents, stats)
    - Progress indicators (spinners, bars)
    - Color theming
    """

    def __init__(self, theme: Optional[str] = None, no_color: bool = False, max_width: int = 120) -> None:
        """
        Initialize renderer.

        Args:
            theme: Theme name (default, dark, light, minimal, matrix, ocean)
            no_color: Disable colored output
            max_width: Maximum output width
        """
        self.theme = get_theme(theme or "default")
        self.no_color = no_color
        self.max_width = max_width

        if RICH_AVAILABLE:
            self._console = Console(
                no_color=no_color,
                width=max_width,
                force_terminal=True,
            )
        else:
            self._console = None

        # Sub-renderers
        self.progress = ProgressManager(self._console, no_color)
        self.tables = TableRenderer(self.theme, self._console, no_color)

    @property
    def console(self) -> Any:
        """Get Rich console."""
        return self._console

    def print(self, *args: Any, **kwargs: Any) -> Any:
        """Print to console."""
        if self._console:
            self._console.print(*args, **kwargs)
        else:
            print(*args)

    def info(self, message: str) -> Any:
        """Print info message."""
        if self._console and not self.no_color:
            self._console.print(f"[{self.theme.info}]i[/{self.theme.info}] {message}")
        elif self._console:
            self._console.print(f"i {message}")
        else:
            print(f"[INFO] {message}")

    def success(self, message: str) -> Any:
        """Print success message."""
        if self._console and not self.no_color:
            self._console.print(f"[{self.theme.success}]✓[/{self.theme.success}] {message}")
        elif self._console:
            self._console.print(f"✓ {message}")
        else:
            print(f"[OK] {message}")

    def warning(self, message: str) -> Any:
        """Print warning message."""
        if self._console and not self.no_color:
            self._console.print(f"[{self.theme.warning}]![/{self.theme.warning}] {message}")
        elif self._console:
            self._console.print(f"! {message}")
        else:
            print(f"[WARN] {message}")

    def error(self, message: str) -> Any:
        """Print error message."""
        if self._console and not self.no_color:
            self._console.print(f"[{self.theme.error}]✗[/{self.theme.error}] {message}")
        elif self._console:
            self._console.print(f"✗ {message}")
        else:
            print(f"[ERROR] {message}")

    def header(self, text: str) -> Any:
        """Print section header."""
        if self._console and not self.no_color:
            self._console.print(f"\n[bold {self.theme.primary}]{'═' * 50}[/bold {self.theme.primary}]")
            self._console.print(f"[bold {self.theme.primary}]  {text}[/bold {self.theme.primary}]")
            self._console.print(f"[bold {self.theme.primary}]{'═' * 50}[/bold {self.theme.primary}]")
        else:
            print(f"\n{'=' * 50}")
            print(f"  {text}")
            print(f"{'=' * 50}")

    def subheader(self, text: str) -> Any:
        """Print subsection header."""
        if self._console and not self.no_color:
            self._console.print(f"\n[bold {self.theme.secondary}]── {text} ──[/bold {self.theme.secondary}]")
        else:
            print(f"\n── {text} ──")

    def status(self, text: str) -> Any:
        """Print progress/status message with spinner character."""
        if self._console and not self.no_color:
            self._console.print(f"[{self.theme.muted}]⏳[/{self.theme.muted}] {text}")
        else:
            print(f"⏳ {text}")

    def panel(self, content: str, title: Optional[str] = None, subtitle: Optional[str] = None, style: Optional[str] = None, expand: bool = False) -> Any:
        """
        Print content in a panel.

        Args:
            content: Panel content
            title: Panel title
            subtitle: Panel subtitle
            style: Border style
            expand: Expand to full width
        """
        if not RICH_AVAILABLE:
            if title:
                print(f"\n=== {title} ===")
            print(content)
            if subtitle:
                print(f"--- {subtitle} ---")
            return

        style = style or self.theme.primary
        panel = Panel(
            content,
            title=title,
            subtitle=subtitle,
            border_style=style,
            expand=expand,
            box=box.ROUNDED,
        )
        self._console.print(panel)

    def code(self, code: str, language: str = 'python', line_numbers: bool = False, title: Optional[str] = None) -> Any:
        """
        Print syntax-highlighted code.

        Args:
            code: Source code
            language: Language for highlighting
            line_numbers: Show line numbers
            title: Code block title
        """
        if not RICH_AVAILABLE:
            if title:
                print(f"\n--- {title} ({language}) ---")
            print(code)
            return

        syntax = Syntax(
            code,
            language,
            theme="monokai",
            line_numbers=line_numbers,
        )

        if title:
            self.panel(syntax, title=title, style=self.theme.code)
        else:
            self._console.print(syntax)

    def markdown(self, text: str) -> Any:
        """
        Print markdown-formatted text with LaTeX support.

        Args:
            text: Markdown content (may include LaTeX math)
        """
        # Convert LaTeX to Unicode for terminal display
        text = self._latex_to_unicode(text)

        if not RICH_AVAILABLE:
            print(text)
            return

        md = Markdown(text)
        self._console.print(md)

    def _latex_to_unicode(self, text: str) -> str:
        """
        Convert LaTeX math to Unicode for terminal display.

        Examples:
            $x^2$ → x²
            $\\alpha$ → α
            $\\sqrt{x}$ → √x
            $\\frac{a}{b}$ → a/b
        """
        import re

        # Greek letters
        greek = {
            r'\\alpha': 'α', r'\\beta': 'β', r'\\gamma': 'γ', r'\\delta': 'δ',
            r'\\epsilon': 'ε', r'\\zeta': 'ζ', r'\\eta': 'η', r'\\theta': 'θ',
            r'\\iota': 'ι', r'\\kappa': 'κ', r'\\lambda': 'λ', r'\\mu': 'μ',
            r'\\nu': 'ν', r'\\xi': 'ξ', r'\\pi': 'π', r'\\rho': 'ρ',
            r'\\sigma': 'σ', r'\\tau': 'τ', r'\\upsilon': 'υ', r'\\phi': 'φ',
            r'\\chi': 'χ', r'\\psi': 'ψ', r'\\omega': 'ω',
            r'\\Alpha': 'Α', r'\\Beta': 'Β', r'\\Gamma': 'Γ', r'\\Delta': 'Δ',
            r'\\Theta': 'Θ', r'\\Lambda': 'Λ', r'\\Xi': 'Ξ', r'\\Pi': 'Π',
            r'\\Sigma': 'Σ', r'\\Phi': 'Φ', r'\\Psi': 'Ψ', r'\\Omega': 'Ω',
        }

        # Math symbols
        symbols = {
            r'\\times': '×', r'\\div': '÷', r'\\pm': '±', r'\\mp': '∓',
            r'\\cdot': '·', r'\\ast': '∗', r'\\star': '⋆',
            r'\\leq': '≤', r'\\geq': '≥', r'\\neq': '≠', r'\\approx': '≈',
            r'\\equiv': '≡', r'\\sim': '∼', r'\\propto': '∝',
            r'\\infty': '∞', r'\\partial': '∂', r'\\nabla': '∇',
            r'\\sum': 'Σ', r'\\prod': 'Π', r'\\int': '∫',
            r'\\forall': '∀', r'\\exists': '∃', r'\\in': '∈', r'\\notin': '∉',
            r'\\subset': '⊂', r'\\supset': '⊃', r'\\cup': '∪', r'\\cap': '∩',
            r'\\emptyset': '∅', r'\\therefore': '∴', r'\\because': '∵',
            r'\\rightarrow': '→', r'\\leftarrow': '←', r'\\Rightarrow': '⇒',
            r'\\Leftarrow': '⇐', r'\\leftrightarrow': '↔', r'\\Leftrightarrow': '⇔',
            r'\\to': '→', r'\\gets': '←', r'\\implies': '⇒', r'\\iff': '⇔',
            r'\\land': '∧', r'\\lor': '∨', r'\\neg': '¬', r'\\not': '¬',
            r'\\ldots': '…', r'\\cdots': '⋯', r'\\vdots': '⋮', r'\\ddots': '⋱',
            r'\\prime': '′', r'\\degree': '°',
        }

        # Superscripts
        superscripts = {
            '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
            '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
            '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
            'n': 'ⁿ', 'i': 'ⁱ', 'x': 'ˣ', 'y': 'ʸ',
        }

        # Subscripts
        subscripts = {
            '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
            '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
            '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎',
            'a': 'ₐ', 'e': 'ₑ', 'i': 'ᵢ', 'o': 'ₒ', 'u': 'ᵤ',
            'x': 'ₓ', 'n': 'ₙ', 'm': 'ₘ',
        }

        def process_math(match: Any) -> Any:
            """Process a LaTeX math expression."""
            math = match.group(1)

            # Apply Greek letters
            for latex, unicode_char in greek.items():
                math = math.replace(latex, unicode_char)

            # Apply symbols
            for latex, unicode_char in symbols.items():
                math = math.replace(latex, unicode_char)

            # Handle sqrt
            math = re.sub(r'\\sqrt\{([^}]+)\}', r'√(\1)', math)
            math = re.sub(r'\\sqrt\s*(\w)', r'√\1', math)

            # Handle fractions: \frac{a}{b} → a/b
            math = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', math)

            # Handle superscripts: x^2 or x^{2n}
            def convert_super(m: Any) -> Any:
                base = m.group(1) if m.group(1) else ''
                exp = m.group(2)
                result = base
                for char in exp:
                    result += superscripts.get(char, f'^{char}')
                return result
            math = re.sub(r'(\w?)\^{([^}]+)}', convert_super, math)
            math = re.sub(r'(\w)\^(\w)', convert_super, math)

            # Handle subscripts: x_2 or x_{2n}
            def convert_sub(m: Any) -> Any:
                base = m.group(1) if m.group(1) else ''
                sub = m.group(2)
                result = base
                for char in sub:
                    result += subscripts.get(char, f'_{char}')
                return result
            math = re.sub(r'(\w?)_{([^}]+)}', convert_sub, math)
            math = re.sub(r'(\w)_(\w)', convert_sub, math)

            # Clean up remaining backslashes for common commands
            math = re.sub(r'\\text\{([^}]+)\}', r'\1', math)
            math = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', math)
            math = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', math)
            math = math.replace(r'\ ', ' ')
            math = math.replace(r'\,', ' ')
            math = math.replace(r'\;', ' ')
            math = math.replace(r'\!', '')

            return math

        # Process block math: $$...$$
        text = re.sub(r'\$\$(.+?)\$\$', lambda m: f'\n  {process_math(m)}\n', text, flags=re.DOTALL)

        # Process inline math: $...$
        text = re.sub(r'\$(.+?)\$', process_math, text)

        # Process \[...\] and \(...\)
        text = re.sub(r'\\\[(.+?)\\\]', lambda m: f'\n  {process_math(m)}\n', text, flags=re.DOTALL)
        text = re.sub(r'\\\((.+?)\\\)', process_math, text)

        return text

    def tree(self, data: Dict[str, Any], title: str = "Tree") -> Any:
        """
        Print tree structure.

        Args:
            data: Nested dictionary to display
            title: Root node title

        Returns:
            Tree object
        """
        if not RICH_AVAILABLE:
            self._print_dict(data, indent=0)
            return None

        tree = Tree(f"[bold]{title}[/bold]")
        self._build_tree(tree, data)
        self._console.print(tree)
        return tree

    def _build_tree(self, parent: Any, data: Any, key: str = None) -> Any:
        """Recursively build tree."""
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    branch = parent.add(f"[bold]{k}[/bold]")
                    self._build_tree(branch, v, k)
                else:
                    parent.add(f"[cyan]{k}[/cyan]: {v}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    branch = parent.add(f"[{i}]")
                    self._build_tree(branch, item)
                else:
                    parent.add(str(item))
        else:
            parent.add(str(data))

    def _print_dict(self, data: Any, indent: int = 0) -> Any:
        """Print dictionary (fallback)."""
        prefix = "  " * indent
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    print(f"{prefix}{k}:")
                    self._print_dict(v, indent + 1)
                else:
                    print(f"{prefix}{k}: {v}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self._print_dict(item, indent)
                else:
                    print(f"{prefix}- {item}")
        else:
            print(f"{prefix}{data}")

    def result(self, result: Any, title: str = 'Result') -> Any:
        """
        Print execution result.

        Args:
            result: Result object or dict (supports EpisodeResult)
            title: Result title
        """
        # Handle EpisodeResult and similar objects
        if hasattr(result, "to_dict"):
            data = result.to_dict()
        elif isinstance(result, dict):
            data = result
        elif hasattr(result, "success"):
            # EpisodeResult-like object
            data = {
                "success": result.success,
                "output": getattr(result, "output", None),
                "execution_time": getattr(result, "execution_time", None),
                "alerts": getattr(result, "alerts", []),
            }
        else:
            data = {"output": str(result)}

        # Format output
        output_lines = []

        if "success" in data:
            status = "Success" if data["success"] else "Failed"
            status_style = self.theme.success if data["success"] else self.theme.error
            output_lines.append(f"Status: [{status_style}]{status}[/{status_style}]" if RICH_AVAILABLE and not self.no_color else f"Status: {status}")

        if "output" in data and data["output"]:
            output = data["output"]
            if isinstance(output, dict):
                for k, v in output.items():
                    output_lines.append(f"  {k}: {v}")
            else:
                output_lines.append(str(output)[:500])

        # Handle both 'error' and 'alerts' (EpisodeResult uses alerts)
        if "error" in data and data["error"]:
            output_lines.append(f"Error: {data['error']}")
        elif "alerts" in data and data["alerts"]:
            output_lines.append(f"Alerts: {'; '.join(data['alerts'][:3])}")

        if "execution_time" in data and data["execution_time"]:
            output_lines.append(f"Time: {data['execution_time']:.2f}s")

        content = "\n".join(output_lines)
        self.panel(content, title=title, style=self.theme.primary if data.get("success", True) else self.theme.error)

    def welcome(self, version: str = '1.0.0') -> Any:
        """Print welcome banner."""
        banner = f"""
     ██╗ ██████╗ ████████╗████████╗██╗   ██╗
     ██║██╔═══██╗╚══██╔══╝╚══██╔══╝╚██╗ ██╔╝
     ██║██║   ██║   ██║      ██║    ╚████╔╝
██   ██║██║   ██║   ██║      ██║     ╚██╔╝
╚█████╔╝╚██████╔╝   ██║      ██║      ██║
 ╚════╝  ╚═════╝    ╚═╝      ╚═╝      ╚═╝

Multi-Agent AI Assistant v{version}
"""
        if RICH_AVAILABLE:
            self._console.print(f"[{self.theme.primary}]{banner}[/{self.theme.primary}]")
            self._console.print(f"Type [bold]/help[/bold] for commands, or just start typing!")
        else:
            print(banner)
            print("Type /help for commands, or just start typing!")

    def prompt(self) -> str:
        """Get prompt string (plain text for prompt_toolkit compatibility)."""
        # Return plain text - prompt_toolkit handles styling separately
        return "jotty> "

    def prompt_styled(self) -> str:
        """Get Rich-styled prompt for display."""
        if RICH_AVAILABLE:
            return f"[{self.theme.prompt}]jotty>[/{self.theme.prompt}] "
        return "jotty> "

    def goodbye(self) -> Any:
        """Print goodbye message."""
        if RICH_AVAILABLE:
            self._console.print(f"\n[{self.theme.muted}]Goodbye! Session saved.[/{self.theme.muted}]")
        else:
            print("\nGoodbye! Session saved.")

    def divider(self, char: str = '─', style: Optional[str] = None) -> Any:
        """Print horizontal divider."""
        style = style or self.theme.muted
        width = min(self.max_width, 80)
        if RICH_AVAILABLE:
            self._console.print(f"[{style}]{char * width}[/{style}]")
        else:
            print(char * width)

    def newline(self) -> Any:
        """Print blank line."""
        print()

    def clear(self) -> Any:
        """Clear screen."""
        if RICH_AVAILABLE:
            self._console.clear()
        else:
            print("\033[2J\033[H", end="")

    @contextmanager
    def shimmer(self, message: str = "Working...") -> Iterator[ShimmerEffect]:
        """
        Context manager for shimmer animation.

        Falls back to Rich spinner if not true-color or if Live display is active.

        Args:
            message: Status message to animate

        Yields:
            ShimmerEffect instance (or None when using fallback)
        """
        color_depth = TerminalDetector.detect_color_depth()
        use_shimmer = (
            color_depth == ColorDepth.TRUE_COLOR
            and TerminalDetector.supports_animations()
            and not (self.progress._active_spinner or self.progress._active_progress)
        )

        if use_shimmer:
            effect = ShimmerEffect(color_depth)
            effect.start(message)
            try:
                yield effect
            finally:
                effect.stop()
        else:
            # Fall back to Rich spinner
            with self.progress.spinner(message) as spinner:
                yield spinner

    # =========================================================================
    # Claude Code-style Output Methods
    # =========================================================================

    def task_start(self, task: str, explanation: str = None) -> Any:
        """
        Show task start message like Claude Code.

        Example:
            I'll search for information about BaFin KGAB framework...
        """
        if explanation:
            if RICH_AVAILABLE:
                self._console.print(f"\n[{self.theme.muted}]{explanation}[/{self.theme.muted}]")
            else:
                print(f"\n{explanation}")

    def steps_indicator(self, count: int) -> Any:
        """
        Show step count badge like Claude Code.

        Example:
            4 steps
        """
        if RICH_AVAILABLE:
            self._console.print(f"\n[bold cyan]{count} steps[/bold cyan]")
        else:
            print(f"\n{count} steps")

    def search_query(self, query: str, result_count: int = None) -> Any:
        """
        Show search query with result count like Claude Code.

        Example:
            BaFin KGAB framework requirements
            10 results
        """
        if RICH_AVAILABLE:
            self._console.print(f"\n[bold]{query}[/bold]")
            if result_count is not None:
                self._console.print(f"[{self.theme.muted}]{result_count} results[/{self.theme.muted}]")
        else:
            print(f"\n{query}")
            if result_count is not None:
                print(f"{result_count} results")

    def search_results(self, results: List[Dict[str, str]]) -> Any:
        """
        Show search results with favicons like Claude Code.

        Args:
            results: List of dicts with 'title', 'url', optional 'favicon'

        Example:
            🔗 BaFin Interpretation Guidance
               bafin.de
            🔗 SEC Outsourcing Rules
               sec.gov
        """
        if not results:
            return

        for result in results[:10]:  # Max 10 results
            title = result.get('title', 'Untitled')
            url = result.get('url', '')

            # Extract domain from URL
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc.replace('www.', '')
            except Exception:
                domain = url[:30] if url else ''

            if RICH_AVAILABLE:
                self._console.print(f"\n[bold]🔗[/bold] [link={url}]{title}[/link]")
                self._console.print(f"   [{self.theme.muted}]{domain}[/{self.theme.muted}]")
            else:
                print(f"\n🔗 {title}")
                print(f"   {domain}")

    def reading_file(self, filepath: str, description: str = None) -> Any:
        """
        Show file reading operation like Claude Code.

        Example:
            Reading the docx skill file for creating Word documents
        """
        desc = description or f"Reading {filepath}"
        if RICH_AVAILABLE:
            self._console.print(f"\n[{self.theme.muted}]📄 {desc}[/{self.theme.muted}]")
        else:
            print(f"\n📄 {desc}")

    def writing_file(self, filepath: str, description: str = None) -> Any:
        """
        Show file writing operation like Claude Code.

        Example:
            Creating comprehensive checklist document
            checklist.docx
        """
        desc = description or "Creating file"
        if RICH_AVAILABLE:
            self._console.print(f"\n[{self.theme.muted}]📝 {desc}[/{self.theme.muted}]")
            self._console.print(f"[bold cyan]{filepath}[/bold cyan]")
        else:
            print(f"\n📝 {desc}")
            print(filepath)

    def installing(self, package: str) -> Any:
        """
        Show package installation like Claude Code.

        Example:
            Installing docx library for document creation
        """
        if RICH_AVAILABLE:
            self._console.print(f"\n[{self.theme.muted}]📦 Installing {package}[/{self.theme.muted}]")
        else:
            print(f"\n📦 Installing {package}")

    def step_progress(self, step_num: int, total: int, description: str, status: str = 'running') -> Any:
        """
        Show step progress like Claude Code.

        Args:
            step_num: Current step number
            total: Total steps
            description: Step description
            status: 'running', 'done', 'failed'
        """
        icons = {
            'running': '⏳',
            'done': '✓',
            'failed': '✗',
        }
        icon = icons.get(status, '→')

        if RICH_AVAILABLE:
            if status == 'done':
                self._console.print(f"[{self.theme.success}]{icon}[/{self.theme.success}] Step {step_num}/{total}: {description}")
            elif status == 'failed':
                self._console.print(f"[{self.theme.error}]{icon}[/{self.theme.error}] Step {step_num}/{total}: {description}")
            else:
                self._console.print(f"[{self.theme.muted}]{icon}[/{self.theme.muted}] Step {step_num}/{total}: {description}")
        else:
            print(f"{icon} Step {step_num}/{total}: {description}")

    def tool_output(self, tool_name: str, output_path: str = None, summary: str = None) -> Any:
        """
        Show tool output like Claude Code.

        Example:
            ✓ Created checklist document
              /path/to/checklist.docx
        """
        if RICH_AVAILABLE:
            self._console.print(f"[{self.theme.success}]✓[/{self.theme.success}] {tool_name}")
            if output_path:
                self._console.print(f"  [bold cyan]{output_path}[/bold cyan]")
            if summary:
                self._console.print(f"  [{self.theme.muted}]{summary}[/{self.theme.muted}]")
        else:
            print(f"✓ {tool_name}")
            if output_path:
                print(f"  {output_path}")
            if summary:
                print(f"  {summary}")

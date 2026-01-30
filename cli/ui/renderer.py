"""
Rich Renderer for Jotty CLI
===========================

Main rendering class using Rich library.
"""

from typing import Any, Optional, List, Dict
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


class RichRenderer:
    """
    Main renderer for CLI output using Rich library.

    Provides:
    - Formatted output (panels, markdown, code)
    - Tables (skills, agents, stats)
    - Progress indicators (spinners, bars)
    - Color theming
    """

    def __init__(
        self,
        theme: Optional[str] = None,
        no_color: bool = False,
        max_width: int = 120
    ):
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

    def print(self, *args, **kwargs):
        """Print to console."""
        if self._console:
            self._console.print(*args, **kwargs)
        else:
            print(*args)

    def info(self, message: str):
        """Print info message."""
        if self._console and not self.no_color:
            self._console.print(f"[{self.theme.info}]i[/{self.theme.info}] {message}")
        elif self._console:
            self._console.print(f"i {message}")
        else:
            print(f"[INFO] {message}")

    def success(self, message: str):
        """Print success message."""
        if self._console and not self.no_color:
            self._console.print(f"[{self.theme.success}]✓[/{self.theme.success}] {message}")
        elif self._console:
            self._console.print(f"✓ {message}")
        else:
            print(f"[OK] {message}")

    def warning(self, message: str):
        """Print warning message."""
        if self._console and not self.no_color:
            self._console.print(f"[{self.theme.warning}]![/{self.theme.warning}] {message}")
        elif self._console:
            self._console.print(f"! {message}")
        else:
            print(f"[WARN] {message}")

    def error(self, message: str):
        """Print error message."""
        if self._console and not self.no_color:
            self._console.print(f"[{self.theme.error}]✗[/{self.theme.error}] {message}")
        elif self._console:
            self._console.print(f"✗ {message}")
        else:
            print(f"[ERROR] {message}")

    def header(self, text: str):
        """Print section header."""
        if self._console and not self.no_color:
            self._console.print(f"\n[bold {self.theme.primary}]{'═' * 50}[/bold {self.theme.primary}]")
            self._console.print(f"[bold {self.theme.primary}]  {text}[/bold {self.theme.primary}]")
            self._console.print(f"[bold {self.theme.primary}]{'═' * 50}[/bold {self.theme.primary}]")
        else:
            print(f"\n{'=' * 50}")
            print(f"  {text}")
            print(f"{'=' * 50}")

    def subheader(self, text: str):
        """Print subsection header."""
        if self._console and not self.no_color:
            self._console.print(f"\n[bold {self.theme.secondary}]── {text} ──[/bold {self.theme.secondary}]")
        else:
            print(f"\n── {text} ──")

    def status(self, text: str):
        """Print progress/status message with spinner character."""
        if self._console and not self.no_color:
            self._console.print(f"[{self.theme.muted}]⏳[/{self.theme.muted}] {text}")
        else:
            print(f"⏳ {text}")

    def panel(
        self,
        content: str,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        style: Optional[str] = None,
        expand: bool = False
    ):
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

    def code(
        self,
        code: str,
        language: str = "python",
        line_numbers: bool = False,
        title: Optional[str] = None
    ):
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

    def markdown(self, text: str):
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

        def process_math(match):
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
            def convert_super(m):
                base = m.group(1) if m.group(1) else ''
                exp = m.group(2)
                result = base
                for char in exp:
                    result += superscripts.get(char, f'^{char}')
                return result
            math = re.sub(r'(\w?)\^{([^}]+)}', convert_super, math)
            math = re.sub(r'(\w)\^(\w)', convert_super, math)

            # Handle subscripts: x_2 or x_{2n}
            def convert_sub(m):
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

    def _build_tree(self, parent: Any, data: Any, key: str = None):
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

    def _print_dict(self, data: Any, indent: int = 0):
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

    def result(self, result: Any, title: str = "Result"):
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

    def welcome(self, version: str = "1.0.0"):
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

    def goodbye(self):
        """Print goodbye message."""
        if RICH_AVAILABLE:
            self._console.print(f"\n[{self.theme.muted}]Goodbye! Session saved.[/{self.theme.muted}]")
        else:
            print("\nGoodbye! Session saved.")

    def divider(self, char: str = "─", style: Optional[str] = None):
        """Print horizontal divider."""
        style = style or self.theme.muted
        width = min(self.max_width, 80)
        if RICH_AVAILABLE:
            self._console.print(f"[{style}]{char * width}[/{style}]")
        else:
            print(char * width)

    def newline(self):
        """Print blank line."""
        print()

    def clear(self):
        """Clear screen."""
        if RICH_AVAILABLE:
            self._console.clear()
        else:
            print("\033[2J\033[H", end="")

    # =========================================================================
    # Claude Code-style Output Methods
    # =========================================================================

    def task_start(self, task: str, explanation: str = None):
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

    def steps_indicator(self, count: int):
        """
        Show step count badge like Claude Code.

        Example:
            4 steps
        """
        if RICH_AVAILABLE:
            self._console.print(f"\n[bold cyan]{count} steps[/bold cyan]")
        else:
            print(f"\n{count} steps")

    def search_query(self, query: str, result_count: int = None):
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

    def search_results(self, results: List[Dict[str, str]]):
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

    def reading_file(self, filepath: str, description: str = None):
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

    def writing_file(self, filepath: str, description: str = None):
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

    def installing(self, package: str):
        """
        Show package installation like Claude Code.

        Example:
            Installing docx library for document creation
        """
        if RICH_AVAILABLE:
            self._console.print(f"\n[{self.theme.muted}]📦 Installing {package}[/{self.theme.muted}]")
        else:
            print(f"\n📦 Installing {package}")

    def step_progress(self, step_num: int, total: int, description: str, status: str = "running"):
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

    def tool_output(self, tool_name: str, output_path: str = None, summary: str = None):
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

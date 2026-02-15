"""
Table Rendering for Jotty CLI
=============================

Rich-based table formatting.
"""

from typing import List, Dict, Any, Optional, Sequence

try:
    from rich.table import Table
    from rich.console import Console
    from rich.box import ROUNDED, SIMPLE, MINIMAL, DOUBLE, SQUARE
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .themes import Theme, get_theme


# Box style mapping
BOX_STYLES = {
    "rounded": ROUNDED if RICH_AVAILABLE else None,
    "simple": SIMPLE if RICH_AVAILABLE else None,
    "minimal": MINIMAL if RICH_AVAILABLE else None,
    "double": DOUBLE if RICH_AVAILABLE else None,
    "square": SQUARE if RICH_AVAILABLE else None,
}


class TableRenderer:
    """
    Table renderer for CLI output.

    Creates formatted tables using Rich.
    """

    def __init__(self, theme: Optional[Theme] = None, console: Optional[Any] = None, no_color: bool = False) -> None:
        """
        Initialize table renderer.

        Args:
            theme: Color theme
            console: Rich Console instance
            no_color: Disable colors
        """
        self.theme = theme or get_theme()
        self.no_color = no_color
        self._console = console

    @property
    def console(self) -> Any:
        """Get or create console."""
        if self._console is None and RICH_AVAILABLE:
            self._console = Console(no_color=self.no_color)
        return self._console

    def create_table(
        self,
        title: Optional[str] = None,
        columns: Optional[List[str]] = None,
        rows: Optional[List[Sequence[Any]]] = None,
        box_style: str = "rounded",
        show_header: bool = True,
        show_lines: bool = False,
        expand: bool = False,
    ) -> Any:
        """
        Create a Rich table.

        Args:
            title: Table title
            columns: Column headers
            rows: Row data
            box_style: Box style (rounded, simple, minimal, double, square)
            show_header: Show column headers
            show_lines: Show row separator lines
            expand: Expand to full width

        Returns:
            Rich Table object
        """
        if not RICH_AVAILABLE:
            return self._create_simple_table(title, columns, rows)

        box = BOX_STYLES.get(box_style, ROUNDED)

        table = Table(
            title=title,
            box=box,
            show_header=show_header,
            show_lines=show_lines,
            expand=expand,
            header_style=self.theme.table_header,
            border_style=self.theme.table_border,
        )

        # Add columns
        if columns:
            for i, col in enumerate(columns):
                style = self.theme.primary if i == 0 else None
                table.add_column(col, style=style)

        # Add rows
        if rows:
            for i, row in enumerate(rows):
                style = self.theme.table_row_odd if i % 2 == 0 else self.theme.table_row_even
                table.add_row(*[str(cell) for cell in row], style=style)

        return table

    def _create_simple_table(
        self,
        title: Optional[str],
        columns: Optional[List[str]],
        rows: Optional[List[Sequence[Any]]]
    ) -> str:
        """Create simple text table (fallback)."""
        lines = []

        if title:
            lines.append(f"\n{title}")
            lines.append("=" * len(title))

        if columns:
            # Calculate column widths
            widths = [len(str(col)) for col in columns]
            if rows:
                for row in rows:
                    for i, cell in enumerate(row):
                        if i < len(widths):
                            widths[i] = max(widths[i], len(str(cell)))

            # Header
            header = " | ".join(
                str(col).ljust(widths[i]) for i, col in enumerate(columns)
            )
            lines.append(header)
            lines.append("-" * len(header))

            # Rows
            if rows:
                for row in rows:
                    line = " | ".join(
                        str(cell).ljust(widths[i]) if i < len(widths) else str(cell)
                        for i, cell in enumerate(row)
                    )
                    lines.append(line)

        return "\n".join(lines)

    def print_table(self, table: Any) -> Any:
        """Print table to console."""
        if RICH_AVAILABLE and hasattr(table, "row_count"):
            self.console.print(table)
        else:
            print(table)

    def skills_table(self, skills: List[Dict[str, Any]]) -> Any:
        """
        Create skills listing table.

        Args:
            skills: List of skill dicts with name, description, tools

        Returns:
            Table
        """
        columns = ["Name", "Description", "Tools"]
        rows = []

        for skill in skills:
            name = skill.get("name", "unknown")
            description = skill.get("description", "")[:50]
            tools = ", ".join(skill.get("tools", [])[:3])
            if len(skill.get("tools", [])) > 3:
                tools += "..."
            rows.append((name, description, tools))

        return self.create_table(
            title="Available Skills",
            columns=columns,
            rows=rows,
        )

    def agents_table(self, agents: List[Dict[str, Any]]) -> Any:
        """
        Create agents listing table.

        Args:
            agents: List of agent dicts

        Returns:
            Table
        """
        columns = ["Name", "Type", "Specialization", "Success Rate"]
        rows = []

        for agent in agents:
            name = agent.get("name", "unknown")
            agent_type = agent.get("type", "auto")
            spec = agent.get("specialization", "general")
            rate = agent.get("success_rate", 0)
            rate_str = f"{rate:.1%}" if isinstance(rate, float) else str(rate)
            rows.append((name, agent_type, spec, rate_str))

        return self.create_table(
            title="Agents",
            columns=columns,
            rows=rows,
        )

    def stats_table(self, stats: Dict[str, Any]) -> Any:
        """
        Create statistics table.

        Args:
            stats: Statistics dictionary

        Returns:
            Table
        """
        columns = ["Metric", "Value"]
        rows = []

        for key, value in stats.items():
            if isinstance(value, float):
                value = f"{value:.4f}"
            elif isinstance(value, dict):
                value = f"{len(value)} items"
            elif isinstance(value, list):
                value = f"{len(value)} entries"
            rows.append((key, str(value)))

        return self.create_table(
            title="Learning Statistics",
            columns=columns,
            rows=rows,
        )

    def memory_table(self, memories: List[Dict[str, Any]]) -> Any:
        """
        Create memory listing table.

        Args:
            memories: List of memory entries

        Returns:
            Table
        """
        columns = ["Level", "Content", "Score"]
        rows = []

        for mem in memories[:20]:  # Limit display
            level = mem.get("level", "unknown")
            content = str(mem.get("content", ""))[:60]
            score = mem.get("relevance_score", 0)
            score_str = f"{score:.2f}" if isinstance(score, float) else str(score)
            rows.append((level, content, score_str))

        return self.create_table(
            title="Memory Contents",
            columns=columns,
            rows=rows,
        )

    def commands_table(self, commands: List[Dict[str, Any]]) -> Any:
        """
        Create commands help table.

        Args:
            commands: List of command dicts

        Returns:
            Table
        """
        columns = ["Command", "Aliases", "Description"]
        rows = []

        for cmd in commands:
            name = cmd.get("name", "")
            aliases = ", ".join(cmd.get("aliases", []))
            description = cmd.get("description", "")[:60]
            rows.append((name, aliases, description))

        return self.create_table(
            title="Available Commands",
            columns=columns,
            rows=rows,
        )

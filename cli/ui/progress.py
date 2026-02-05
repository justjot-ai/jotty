"""
Progress and Spinner Management
===============================

Rich-based progress bars and spinners for Jotty CLI.
"""

import asyncio
import os
import re
import select
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, Iterator, List

try:
    import termios
    import tty
    TERMIOS_AVAILABLE = True
except ImportError:
    TERMIOS_AVAILABLE = False

try:
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.spinner import Spinner
    from rich.live import Live
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.syntax import Syntax
    from rich.tree import Tree as RichTree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ProgressManager:
    """
    Progress and spinner manager for CLI.

    Provides:
    - Spinners for indeterminate tasks
    - Progress bars for determinate tasks
    - Status messages
    """

    def __init__(self, console: Optional[Any] = None, no_color: bool = False):
        """
        Initialize progress manager.

        Args:
            console: Rich Console instance
            no_color: Disable colors
        """
        self.no_color = no_color
        self._console = console
        self._active_spinner = None
        self._active_progress = None

    @property
    def console(self) -> Any:
        """Get or create console."""
        if self._console is None and RICH_AVAILABLE:
            self._console = Console(no_color=self.no_color)
        return self._console

    @contextmanager
    def spinner(self, message: str = "Working...", style: str = "cyan") -> Iterator[Any]:
        """
        Context manager for spinner.

        Args:
            message: Status message
            style: Spinner color style

        Yields:
            Spinner instance (or None if Rich not available)

        Example:
            with progress.spinner("Processing..."):
                do_work()
        """
        if not RICH_AVAILABLE:
            print(f"  {message}")
            yield None
            return

        with self.console.status(f"[{style}]{message}[/{style}]") as status:
            self._active_spinner = status
            try:
                yield status
            finally:
                self._active_spinner = None

    @contextmanager
    def progress_bar(
        self,
        description: str = "Progress",
        total: int = 100,
        show_time: bool = True
    ) -> Iterator[Any]:
        """
        Context manager for progress bar.

        Args:
            description: Task description
            total: Total steps
            show_time: Show elapsed/remaining time

        Yields:
            Progress task (use task.update(advance=1) to advance)

        Example:
            with progress.progress_bar("Processing", total=10) as task:
                for i in range(10):
                    task.update(advance=1)
        """
        if not RICH_AVAILABLE:
            print(f"  {description}...")
            yield None
            return

        columns = [
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ]
        if show_time:
            columns.extend([TimeElapsedColumn(), TimeRemainingColumn()])

        with Progress(*columns, console=self.console) as progress:
            self._active_progress = progress
            task_id = progress.add_task(description, total=total)
            try:
                yield ProgressTask(progress, task_id)
            finally:
                self._active_progress = None

    def update_status(self, message: str):
        """Update active spinner message."""
        if self._active_spinner is not None and RICH_AVAILABLE:
            self._active_spinner.update(message)

    async def spinner_async(self, message: str = "Working...", style: str = "cyan"):
        """
        Async context manager for spinner.

        Args:
            message: Status message
            style: Spinner color style

        Returns:
            AsyncSpinnerContext
        """
        return AsyncSpinnerContext(self, message, style)


class ProgressTask:
    """Wrapper for Rich progress task."""

    def __init__(self, progress: Any, task_id: int):
        self.progress = progress
        self.task_id = task_id

    def update(self, advance: int = 1, description: Optional[str] = None):
        """
        Update progress.

        Args:
            advance: Steps to advance
            description: New description (optional)
        """
        kwargs = {"advance": advance}
        if description is not None:
            kwargs["description"] = description
        self.progress.update(self.task_id, **kwargs)

    def complete(self):
        """Mark task as complete."""
        self.progress.update(self.task_id, completed=True)


class AsyncSpinnerContext:
    """Async context manager for spinner."""

    def __init__(self, manager: ProgressManager, message: str, style: str):
        self.manager = manager
        self.message = message
        self.style = style
        self._status = None

    async def __aenter__(self):
        if RICH_AVAILABLE:
            self._status = self.manager.console.status(
                f"[{self.style}]{self.message}[/{self.style}]"
            )
            self._status.__enter__()
            self.manager._active_spinner = self._status
        else:
            print(f"  {self.message}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._status is not None:
            self._status.__exit__(exc_type, exc_val, exc_tb)
            self.manager._active_spinner = None

    def update(self, message: str):
        """Update spinner message."""
        if self._status is not None:
            self._status.update(f"[{self.style}]{message}[/{self.style}]")


@dataclass
class SwarmState:
    """Real-time state of a CodingSwarm execution."""

    # Requirements
    requirements: str = ""
    scope: str = ""
    team: str = ""
    language: str = "python"

    # Pipeline tracking
    current_phase: str = ""
    current_agent: str = ""
    current_message: str = ""
    phases: List[Dict[str, str]] = field(default_factory=lambda: [
        {"id": "Phase 0", "name": "ScopeDetector", "status": "pending"},
        {"id": "Phase 1", "name": "Architect", "status": "pending"},
        {"id": "Phase 1.5", "name": "Researcher", "status": "pending"},
        {"id": "Phase 2", "name": "Developer", "status": "pending"},
        {"id": "Phase 3", "name": "Optimizer", "status": "pending"},
        {"id": "Phase 3.5", "name": "Validator", "status": "pending"},
        {"id": "Phase 4", "name": "TestWriter", "status": "pending"},
        {"id": "Phase 5", "name": "DocWriter", "status": "pending"},
        {"id": "Phase 5.5", "name": "Verifier", "status": "pending"},
        {"id": "Phase 6", "name": "TeamReview", "status": "pending"},
    ])

    # Files generated
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    main_file: str = ""
    total_loc: int = 0

    # Validation
    validation_status: str = ""
    fix_attempts: int = 0
    errors_fixed: List[str] = field(default_factory=list)

    # Review
    review_status: str = ""
    reviewers: List[Dict[str, str]] = field(default_factory=list)
    rework_attempts: int = 0
    arbitrator_decisions: List[str] = field(default_factory=list)

    # Log
    log_entries: List[str] = field(default_factory=list)
    max_log_entries: int = 50

    # Timing
    start_time: float = 0.0

    # Page navigation
    current_page: int = 1  # 1=Pipeline, 2=TaskBoard

    # Agent communication log (derived from progress events)
    agent_messages: List[Dict[str, str]] = field(default_factory=list)
    max_agent_messages: int = 30

    # Execution traces (from _trace_phase callback)
    traces: List[Dict[str, Any]] = field(default_factory=list)

    # Export
    output_path: str = ""

    # File content (available after completion, for Page 3 preview)
    file_contents: Dict[str, str] = field(default_factory=dict)

    # File explorer state
    selected_file_index: int = 0
    file_scroll_offset: int = 0

    # Component tracking
    components: List[str] = field(default_factory=list)
    component_count: int = 0

    # Review - detailed tracking
    reviewer_verdicts: List[Dict[str, str]] = field(default_factory=list)
    current_review_phase: str = ""

    def add_log(self, entry: str):
        """Add log entry, maintaining max size."""
        self.log_entries.append(entry)
        if len(self.log_entries) > self.max_log_entries:
            self.log_entries = self.log_entries[-self.max_log_entries:]

    def elapsed_str(self) -> str:
        """Format elapsed time as M:SS."""
        if not self.start_time:
            return "0:00"
        elapsed = time.time() - self.start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        return f"{mins}:{secs:02d}"

    def add_agent_message(self, from_agent: str, to_agent: str, event: str):
        """Track inter-agent communication."""
        self.agent_messages.append({
            "from": from_agent,
            "to": to_agent,
            "event": event,
            "time": self.elapsed_str(),
        })
        if len(self.agent_messages) > self.max_agent_messages:
            self.agent_messages = self.agent_messages[-self.max_agent_messages:]

    def add_trace(self, trace_dict: Dict[str, Any]):
        """Add execution trace entry."""
        trace_dict["elapsed"] = self.elapsed_str()
        self.traces.append(trace_dict)

    def expand_phases_fullstack(self):
        """Replace single-tier phases with full-stack sub-phases."""
        for p in self.phases:
            if p["id"] == "Phase 1":
                p["name"] = "SystemDesigner"
                break
        idx = next((i for i, p in enumerate(self.phases) if p["id"] == "Phase 2"), None)
        if idx is not None:
            self.phases.pop(idx)
            for i, sub in enumerate([
                {"id": "Phase 2a", "name": "DatabaseArchitect", "status": "pending"},
                {"id": "Phase 2b", "name": "APIDesigner", "status": "pending"},
                {"id": "Phase 2c", "name": "FrontendDeveloper", "status": "pending"},
                {"id": "Phase 2d", "name": "IntegrationEngineer", "status": "pending"},
            ]):
                self.phases.insert(idx + i, sub)

    def expand_phases_teamreview(self):
        """Expand Phase 6 to show team review sub-steps."""
        idx = next((i for i, p in enumerate(self.phases) if p["id"] == "Phase 6"), None)
        if idx is not None:
            self.phases.pop(idx)
            for i, sub in enumerate([
                {"id": "Phase 6", "name": "TeamReview", "status": "pending"},
                {"id": "Phase 6a", "name": "FunctionalReview", "status": "pending"},
                {"id": "Phase 6-arb", "name": "Arbitrator", "status": "pending"},
                {"id": "Phase 6b", "name": "QualityReview", "status": "pending"},
            ]):
                self.phases.insert(idx + i, sub)

    def update_from_progress(self, phase: str, agent: str, message: str):
        """Update state from a _progress() callback."""
        self.current_phase = phase
        self.current_agent = agent
        self.current_message = message
        self.add_log(f"[{phase}] {agent}: {message}")

        # Mark phase transitions
        msg_lower = message.lower()
        for p in self.phases:
            if p["id"] == phase:
                if any(kw in msg_lower for kw in ("done", "detected", "skipped", "all files validated", "final verdict", "max attempts")):
                    p["status"] = "completed"
                elif p["status"] == "pending":
                    p["status"] = "active"

        # Track files
        if "file(s)" in message and "done" in msg_lower:
            file_match = re.search(r':\s*(.+)$', message)
            if file_match:
                for fname in file_match.group(1).split(', '):
                    fname = fname.strip()
                    if fname and '.' in fname:
                        self.files[fname] = {"loc": 0, "validated": False}

        # Track validation
        if "syntax ok" in msg_lower or "run ok" in msg_lower:
            fname_match = message.split(': ')[-1].strip() if ': ' in message else ""
            if fname_match in self.files:
                self.files[fname_match]["validated"] = True

        if "all files validated" in msg_lower:
            self.validation_status = "passed"

        # Track scope
        if "detected scope" in msg_lower:
            self.scope = message.split(': ')[-1].strip() if ': ' in message else ""
            if self.scope == "full_stack":
                self.expand_phases_fullstack()

        # Team review expansion trigger
        if phase == "Phase 6" and "team review" in msg_lower:
            if not any(p["id"] == "Phase 6a" for p in self.phases):
                self.expand_phases_teamreview()

        # Component tracking
        if phase == "Phase 1" and agent in ("Architect", "SystemDesigner"):
            comp_count_match = re.search(r'(\d+)\s+component', message)
            if comp_count_match:
                self.component_count = int(comp_count_match.group(1))
            comp_name_match = re.match(r'\s*Component:\s*(.+)', message)
            if comp_name_match:
                comp_name = comp_name_match.group(1).strip()
                if comp_name and comp_name not in self.components:
                    self.components.append(comp_name)

        # Phase 6 sub-phase status tracking
        if phase == "Phase 6":
            if "functional review" in msg_lower:
                for p in self.phases:
                    if p["id"] == "Phase 6a" and p["status"] == "pending":
                        p["status"] = "active"
            elif "quality review" in msg_lower:
                for p in self.phases:
                    if p["id"] == "Phase 6b" and p["status"] == "pending":
                        p["status"] = "active"
            elif agent == "Arbitrator":
                for p in self.phases:
                    if p["id"] == "Phase 6-arb" and p["status"] == "pending":
                        p["status"] = "active"

            # Track reviewer verdicts
            verdict_match = re.match(r'(APPROVED|REJECTED)', message)
            if verdict_match and agent not in ("TeamReview", "Arbitrator", "Optimizer"):
                issue_match = re.search(r'\((\d+)\s+issue', message)
                self.reviewer_verdicts.append({
                    "persona": agent,
                    "verdict": verdict_match.group(1),
                    "issues": issue_match.group(1) if issue_match else "0",
                    "phase": self.current_review_phase,
                })

            # Track review phase type
            if re.match(r'(Functional|Quality)\s+review:', message, re.IGNORECASE):
                self.current_review_phase = message.split()[0].lower()
                self.review_status = "reviewing"

            # Mark sub-phases completed
            if "final verdict" in msg_lower:
                for p in self.phases:
                    if p["id"] in ("Phase 6a", "Phase 6b", "Phase 6-arb"):
                        if p["status"] == "active":
                            p["status"] = "completed"

        # Track review
        if "approved" in msg_lower and phase == "Phase 6":
            self.review_status = "approved"
        if "overruled" in msg_lower:
            self.arbitrator_decisions.append(message)
        if "confirmed" in msg_lower and "arbitrator" in agent.lower():
            self.arbitrator_decisions.append(message)

        # Track agent-to-agent communication (phase transitions imply handoffs)
        if "done" in msg_lower:
            self.add_agent_message(agent, "Pipeline", f"{phase} completed")
        if "architecture_designed" in msg_lower or "designing" in msg_lower:
            self.add_agent_message(agent, "Developer", "architecture handoff")
        if "optimiz" in msg_lower and phase == "Phase 3":
            self.add_agent_message("Developer", agent, "code handoff")
        if "validat" in msg_lower and phase == "Phase 3.5":
            self.add_agent_message("Optimizer", agent, "optimized code handoff")
        if "review" in msg_lower and phase == "Phase 6":
            self.add_agent_message("Verifier", agent, "verified code for review")
        if "overruled" in msg_lower or "confirmed" in msg_lower:
            self.add_agent_message(
                "Arbitrator",
                agent.split()[0] if agent else "Reviewer",
                message[:50],
            )
        if "rework" in msg_lower:
            self.add_agent_message("TeamReview", "Optimizer", "rework request")


class SwarmDashboard:
    """Real-time TUI dashboard for CodingSwarm execution."""

    def __init__(self, console: "Console", requirements: str = ""):
        self.console = console
        self.state = SwarmState(requirements=requirements)
        self._live = None
        self._key_thread = None
        self._key_running = False
        self._old_term_settings = None

    def on_progress(self, phase: str, agent: str, message: str):
        """Callback for _progress() -- updates state and refreshes dashboard."""
        self.state.update_from_progress(phase, agent, message)
        if self._live:
            self._live.update(self._build_layout())

    def on_trace(self, trace_dict: Dict[str, Any]):
        """Callback for _record_trace() -- adds execution trace to state."""
        self.state.add_trace(trace_dict)
        if self._live:
            self._live.update(self._build_layout())

    def _start_key_listener(self):
        """Start background thread that listens for 1/2 key presses."""
        if not TERMIOS_AVAILABLE:
            return
        try:
            self._old_term_settings = termios.tcgetattr(sys.stdin)
        except (termios.error, ValueError):
            return  # Not a terminal, skip keyboard input

        self._key_running = True

        def _listener():
            try:
                tty.setcbreak(sys.stdin.fileno())
                while self._key_running:
                    if sys.stdin in select.select([sys.stdin], [], [], 0.2)[0]:
                        ch = sys.stdin.read(1)
                        if ch == '1':
                            self.state.current_page = 1
                            if self._live:
                                self._live.update(self._build_layout())
                        elif ch == '2':
                            self.state.current_page = 2
                            if self._live:
                                self._live.update(self._build_layout())
                        elif ch == '3':
                            self.state.current_page = 3
                            if self._live:
                                self._live.update(self._build_layout())
                        elif ch == 'j' and self.state.current_page == 3:
                            file_count = len(self.state.files)
                            if file_count > 0:
                                self.state.selected_file_index = min(
                                    self.state.selected_file_index + 1, file_count - 1
                                )
                                self.state.file_scroll_offset = 0
                                if self._live:
                                    self._live.update(self._build_layout())
                        elif ch == 'k' and self.state.current_page == 3:
                            if self.state.selected_file_index > 0:
                                self.state.selected_file_index -= 1
                                self.state.file_scroll_offset = 0
                                if self._live:
                                    self._live.update(self._build_layout())
                        elif ch == 'd' and self.state.current_page == 3:
                            self.state.file_scroll_offset += 10
                            if self._live:
                                self._live.update(self._build_layout())
                        elif ch == 'u' and self.state.current_page == 3:
                            self.state.file_scroll_offset = max(
                                0, self.state.file_scroll_offset - 10
                            )
                            if self._live:
                                self._live.update(self._build_layout())
            except Exception:
                pass
            finally:
                if self._old_term_settings:
                    try:
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term_settings)
                    except Exception:
                        pass

        self._key_thread = threading.Thread(target=_listener, daemon=True)
        self._key_thread.start()

    def _stop_key_listener(self):
        """Stop keyboard listener and restore terminal."""
        self._key_running = False
        if self._old_term_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term_settings)
            except Exception:
                pass
        self._old_term_settings = None

    def _build_layout(self) -> "Layout":
        """Build the full dashboard layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=1),
        )

        layout["header"].update(self._render_header())
        layout["footer"].update(self._render_footer())

        if self.state.current_page == 1:
            self._build_page1(layout["body"])
        elif self.state.current_page == 2:
            self._build_page2(layout["body"])
        elif self.state.current_page == 3:
            self._build_page3(layout["body"])

        return layout

    def _build_page1(self, body: "Layout"):
        """Page 1: Pipeline + Details + Log (existing layout)."""
        body.split_column(
            Layout(name="main", ratio=1),
            Layout(name="log", size=12),
        )
        body["main"].split_row(
            Layout(name="pipeline", ratio=1),
            Layout(name="details", ratio=1),
        )
        body["main"]["pipeline"].update(self._render_pipeline())
        body["main"]["details"].update(self._render_details())
        body["log"].update(self._render_log())

    def _build_page2(self, body: "Layout"):
        """Page 2: Task Board + Agent Comms + Traces."""
        body.split_column(
            Layout(name="top", ratio=1),
            Layout(name="bottom", ratio=1),
        )
        body["top"].split_row(
            Layout(name="taskboard", ratio=1),
            Layout(name="comms", ratio=1),
        )
        body["bottom"].update(self._render_traces())
        body["top"]["taskboard"].update(self._render_taskboard())
        body["top"]["comms"].update(self._render_comms())

    def _build_page3(self, body: "Layout"):
        """Page 3: File Explorer + Content Preview."""
        body.split_row(
            Layout(name="file_tree", ratio=1, minimum_size=25),
            Layout(name="file_preview", ratio=3),
        )
        body["file_tree"].update(self._render_file_tree())
        body["file_preview"].update(self._render_file_preview())

    def _render_file_tree(self) -> "Panel":
        """Render file tree with selection marker and validation icons."""
        file_list = sorted(self.state.files.keys())
        if not file_list:
            return Panel("[dim]No files generated yet[/dim]", title="[bold]Files[/bold]", border_style="cyan")

        # Group files by directory
        dirs: Dict[str, List[str]] = {}
        for fname in file_list:
            parts = fname.rsplit("/", 1)
            if len(parts) == 2:
                dir_name, base_name = parts
            else:
                dir_name, base_name = ".", parts[0]
            dirs.setdefault(dir_name, []).append(fname)

        tree = RichTree("[bold]Project[/bold]")
        idx = 0
        for dir_name in sorted(dirs.keys()):
            if dir_name == ".":
                branch = tree
            else:
                branch = tree.add(f"[bold blue]{dir_name}/[/bold blue]")
            for fname in sorted(dirs[dir_name]):
                info = self.state.files.get(fname, {})
                validated = info.get("validated", False)
                v_icon = "[green]✓[/green]" if validated else "[dim]○[/dim]"
                display_name = fname.rsplit("/", 1)[-1]
                if idx == self.state.selected_file_index:
                    branch.add(f"{v_icon} [bold reverse] {display_name} [/bold reverse]")
                else:
                    branch.add(f"{v_icon} {display_name}")
                idx += 1

        return Panel(tree, title="[bold]Files[/bold]", border_style="cyan")

    def _render_file_preview(self) -> "Panel":
        """Render syntax-highlighted file content preview."""
        file_list = sorted(self.state.files.keys())
        if not file_list:
            return Panel("[dim]Select a file to preview[/dim]", title="[bold]Preview[/bold]", border_style="green")

        sel_idx = max(0, min(self.state.selected_file_index, len(file_list) - 1))
        selected_file = file_list[sel_idx]
        content = self.state.file_contents.get(selected_file, "")

        if not content:
            return Panel(
                f"[dim]Content for [bold]{selected_file}[/bold] will be available after completion[/dim]",
                title=f"[bold]{selected_file}[/bold]",
                border_style="green",
            )

        # Detect language from extension
        ext_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".html": "html", ".css": "css", ".json": "json",
            ".yaml": "yaml", ".yml": "yaml", ".md": "markdown",
            ".sql": "sql", ".sh": "bash", ".go": "go",
            ".rs": "rust", ".java": "java", ".tsx": "tsx",
            ".jsx": "jsx", ".rb": "ruby", ".php": "php",
        }
        ext = "." + selected_file.rsplit(".", 1)[-1] if "." in selected_file else ""
        lang = ext_map.get(ext, "text")

        # Apply scroll offset
        lines = content.split("\n")
        offset = self.state.file_scroll_offset
        visible_lines = lines[offset:]
        visible_content = "\n".join(visible_lines)

        scroll_info = ""
        if offset > 0:
            scroll_info = f" (line {offset + 1}+)"

        try:
            syntax = Syntax(
                visible_content, lang,
                line_numbers=True,
                start_line=offset + 1,
                theme="monokai",
            )
            return Panel(syntax, title=f"[bold]{selected_file}{scroll_info}[/bold]", border_style="green")
        except Exception:
            return Panel(visible_content, title=f"[bold]{selected_file}{scroll_info}[/bold]", border_style="green")

    def _render_header(self) -> "Panel":
        """Render header with title, config, elapsed time."""
        req_display = (self.state.requirements[:70] + "...") if len(self.state.requirements) > 70 else self.state.requirements
        elapsed = self.state.elapsed_str()

        parts = ["[bold cyan]JOTTY CODING SWARM[/bold cyan]"]
        if self.state.team:
            parts.append(f"team=[yellow]{self.state.team}[/yellow]")
        if self.state.scope:
            parts.append(f"scope=[yellow]{self.state.scope}[/yellow]")
        if self.state.language:
            parts.append(f"lang=[yellow]{self.state.language}[/yellow]")
        parts.append(f"[bold green]{elapsed}[/bold green]")

        pg = self.state.current_page
        nav = (
            f"[{'bold' if pg == 1 else 'dim'}]1:Pipeline[/{'bold' if pg == 1 else 'dim'}]"
            f" [{'bold' if pg == 2 else 'dim'}]2:Tasks[/{'bold' if pg == 2 else 'dim'}]"
            f" [{'bold' if pg == 3 else 'dim'}]3:Files[/{'bold' if pg == 3 else 'dim'}]"
        )
        parts.append(nav)

        header_text = "  ".join(parts) + f"\n[dim]{req_display}[/dim]"
        return Panel(header_text, style="cyan", height=3)

    def _render_pipeline(self) -> "Panel":
        """Render pipeline phases with status indicators."""
        lines = []
        icons = {
            "pending": "[dim]○[/dim]",
            "active": "[yellow]⏳[/yellow]",
            "completed": "[green]✓[/green]",
        }

        for phase in self.state.phases:
            icon = icons.get(phase["status"], "○")
            name = phase["name"]
            phase_id = phase["id"]

            if phase["status"] == "active":
                lines.append(f" {icon} [bold]{phase_id}[/bold] [yellow]{name}[/yellow]")
            elif phase["status"] == "completed":
                lines.append(f" {icon} {phase_id} [green]{name}[/green]")
            else:
                lines.append(f" {icon} [dim]{phase_id} {name}[/dim]")

        return Panel("\n".join(lines), title="[bold]Pipeline[/bold]", border_style="cyan")

    def _render_details(self) -> "Panel":
        """Render right panel: active agent + files + stats."""
        sections = []

        # Active agent
        if self.state.current_agent:
            sections.append(f"[bold yellow]Active:[/bold yellow] {self.state.current_agent}")
            if self.state.current_message:
                msg = (self.state.current_message[:60] + "...") if len(self.state.current_message) > 60 else self.state.current_message
                sections.append(f"  [dim]{msg}[/dim]")
            sections.append("")

        # Components
        if self.state.components or self.state.component_count:
            sections.append(f"[bold]Components:[/bold] {self.state.component_count} designed")
            for comp in self.state.components:
                sections.append(f"  [cyan]>[/cyan] {comp}")
            sections.append("")

        # Files
        if self.state.files:
            sections.append("[bold]Files Generated:[/bold]")
            for fname, info in self.state.files.items():
                loc = info.get("loc", 0)
                validated = info.get("validated", False)
                status_icon = "[green]✓[/green]" if validated else "[dim]○[/dim]"
                loc_str = f"{loc} lines" if loc > 0 else ""
                main = " [cyan]*[/cyan]" if fname == self.state.main_file else ""
                sections.append(f"  {status_icon} {fname}{main} {loc_str}")
            sections.append(f"  [dim]Total LOC: {self.state.total_loc}[/dim]")
            sections.append("")

        # Validation
        if self.state.validation_status:
            v_icon = {
                "passed": "[green]✓[/green]",
                "failed": "[red]✗[/red]",
                "validating": "[yellow]⏳[/yellow]",
            }.get(self.state.validation_status, "")
            sections.append(f"[bold]Validation:[/bold] {v_icon} {self.state.validation_status}")
            if self.state.fix_attempts:
                sections.append(f"  Fix attempts: {self.state.fix_attempts}")
            if self.state.errors_fixed:
                for err in self.state.errors_fixed[-3:]:
                    sections.append(f"  [dim]Fixed: {err}[/dim]")
            sections.append("")

        # Review
        if self.state.review_status:
            r_icon = {
                "approved": "[green]✓[/green]",
                "rejected": "[red]✗[/red]",
                "reviewing": "[yellow]⏳[/yellow]",
            }.get(self.state.review_status, "")
            sections.append(f"[bold]Review:[/bold] {r_icon} {self.state.review_status}")
            if self.state.rework_attempts:
                sections.append(f"  Rework cycles: {self.state.rework_attempts}")
            # Per-reviewer verdicts
            if self.state.reviewer_verdicts:
                for rv in self.state.reviewer_verdicts:
                    v_icon = "[green]✓[/green]" if rv["verdict"] == "APPROVED" else "[red]✗[/red]"
                    issues = f" ({rv['issues']} issues)" if rv["issues"] != "0" else ""
                    phase_tag = f" [{rv['phase']}]" if rv.get("phase") else ""
                    sections.append(f"  {v_icon} {rv['persona']}: {rv['verdict']}{issues}{phase_tag}")
            # Arbitrator decisions
            for dec in self.state.arbitrator_decisions[-3:]:
                sections.append(f"  [dim]{dec}[/dim]")

        return Panel(
            "\n".join(sections) if sections else "[dim]Waiting...[/dim]",
            title="[bold]Details[/bold]",
            border_style="green",
        )

    def _render_log(self) -> "Panel":
        """Render scrolling log panel."""
        visible = self.state.log_entries[-10:]
        log_text = "\n".join(f"[dim]{entry}[/dim]" for entry in visible)
        return Panel(
            log_text or "[dim]No log entries yet[/dim]",
            title="[bold]Log[/bold]",
            border_style="dim",
        )

    def _render_footer(self) -> "Panel":
        """Render footer with navigation hint."""
        pg = self.state.current_page
        return Panel(
            f"[dim]Press [bold]1[/bold]=Pipeline  [bold]2[/bold]=Tasks/Comms  [bold]3[/bold]=Files  |  Page {pg}/3[/dim]",
            style="dim",
            height=1,
        )

    def _render_taskboard(self) -> "Panel":
        """Render task board showing agent execution status and timing."""
        table = Table(show_header=True, header_style="bold", box=None, expand=True)
        table.add_column("Agent", style="cyan", width=16)
        table.add_column("Status", width=10)
        table.add_column("Time", width=8, justify="right")
        table.add_column("Output", ratio=1)

        status_icons = {
            "pending": "[dim]○[/dim]",
            "active": "[yellow]⏳[/yellow]",
            "completed": "[green]✓[/green]",
        }

        for phase in self.state.phases:
            status = phase["status"]
            icon = status_icons.get(status, "○")
            agent_name = phase["name"]

            # Find trace timing for this agent
            trace_time = ""
            trace_output = ""
            for t in self.state.traces:
                if t.get("agent") == agent_name:
                    trace_time = f"{t.get('time', 0):.1f}s"
                    if t.get("success"):
                        trace_output = str(t.get("output_summary", ""))[:40]
                    else:
                        trace_output = f"[red]{t.get('error', 'failed')}[/red]"

            if status == "active":
                table.add_row(
                    f"[bold yellow]{agent_name}[/bold yellow]",
                    f"{icon} running",
                    trace_time,
                    trace_output,
                )
            elif status == "completed":
                table.add_row(
                    f"[green]{agent_name}[/green]",
                    f"{icon} done",
                    trace_time,
                    trace_output,
                )
            else:
                table.add_row(
                    f"[dim]{agent_name}[/dim]",
                    f"{icon} pending",
                    "",
                    "",
                )

        return Panel(table, title="[bold]Task Board[/bold]", border_style="yellow")

    def _render_comms(self) -> "Panel":
        """Render agent communication panel."""
        lines = []
        for msg in self.state.agent_messages[-12:]:
            from_a = msg["from"]
            to_a = msg["to"]
            event = msg["event"]
            t = msg["time"]
            lines.append(
                f"[dim]{t}[/dim] [cyan]{from_a}[/cyan] -> [green]{to_a}[/green]: {event}"
            )

        return Panel(
            "\n".join(lines) if lines else "[dim]No agent communication yet[/dim]",
            title="[bold]Agent Communication[/bold]",
            border_style="magenta",
        )

    def _render_traces(self) -> "Panel":
        """Render execution trace timeline."""
        table = Table(show_header=True, header_style="bold", box=None, expand=True)
        table.add_column("Time", width=6, justify="right")
        table.add_column("Agent", style="cyan", width=16)
        table.add_column("Role", width=10)
        table.add_column("Duration", width=8, justify="right")
        table.add_column("Status", width=8)
        table.add_column("Details", ratio=1)

        for t in self.state.traces[-8:]:
            elapsed = t.get("elapsed", "")
            agent = t.get("agent", "")
            role = t.get("role", "")
            duration = f"{t.get('time', 0):.1f}s"
            success = "[green]✓[/green]" if t.get("success") else "[red]✗[/red]"
            details = str(t.get("output_summary", ""))[:50]

            table.add_row(elapsed, agent, role, duration, success, details)

        return Panel(
            table if self.state.traces else "[dim]No execution traces yet[/dim]",
            title="[bold]Execution Traces[/bold]",
            border_style="blue",
        )

    def start(self):
        """Start the Live display."""
        self.state.start_time = time.time()
        self._live = Live(
            self._build_layout(),
            console=self.console,
            refresh_per_second=4,
            screen=True,
        )
        self._live.start()
        self._start_key_listener()

    def stop(self):
        """Stop the Live display."""
        self._stop_key_listener()
        if self._live:
            self._live.stop()
            self._live = None

    def show_final_summary(self):
        """Print final summary after dashboard closes."""
        s = self.state
        completed = sum(1 for p in s.phases if p["status"] == "completed")
        total = len(s.phases)

        self.console.print(f"\n[bold cyan]Swarm Complete[/bold cyan] ({s.elapsed_str()})")
        self.console.print(f"  Phases: {completed}/{total}")
        self.console.print(f"  Files: {len(s.files)}")
        self.console.print(f"  LOC: {s.total_loc}")
        if s.validation_status:
            self.console.print(f"  Validation: {s.validation_status}")
        if s.review_status:
            self.console.print(f"  Review: {s.review_status}")

    def show_export_menu(self, result, output_path: str = ""):
        """Interactive post-run export menu."""
        self.console.print(f"\n[bold cyan]Export Options[/bold cyan]")

        if output_path:
            self.console.print(f"  [green]Auto-saved to:[/green] {output_path}")

        if not result or not hasattr(result, 'code') or not result.code:
            self.console.print("  [dim]No code to export[/dim]")
            return

        self.console.print(f"  [bold]1[/bold]) Save code files to current directory")
        self.console.print(f"  [bold]2[/bold]) Copy main file to clipboard")
        self.console.print(f"  [bold]3[/bold]) Export as markdown report")
        self.console.print(f"  [bold]s[/bold]) Skip")

        try:
            choice = input("\n  Choice [1/2/3/s]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return

        if choice == "1":
            self._export_code_files(result)
        elif choice == "2":
            self._copy_to_clipboard(result)
        elif choice == "3":
            self._export_markdown_report(result, self.state)

    def _export_code_files(self, result):
        """Write code files to current directory."""
        from pathlib import Path
        for fname, content in result.code.files.items():
            path = Path.cwd() / fname
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            self.console.print(f"  [green]Wrote:[/green] {path}")
        if result.code.tests:
            tests_dir = Path.cwd() / "tests"
            tests_dir.mkdir(exist_ok=True)
            for fname, content in result.code.tests.items():
                path = tests_dir / fname
                path.write_text(content, encoding="utf-8")
                self.console.print(f"  [green]Wrote:[/green] {path}")

    def _copy_to_clipboard(self, result):
        """Copy main file content to clipboard."""
        import subprocess
        main = result.code.files.get(result.code.main_file, "")
        if not main:
            main = next(iter(result.code.files.values()), "")
        for cmd in [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard"], ["pbcopy"]]:
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                proc.communicate(main.encode())
                self.console.print(f"  [green]Copied to clipboard[/green] ({len(main)} chars)")
                return
            except FileNotFoundError:
                continue
        self.console.print("  [yellow]No clipboard tool found (xclip/xsel/pbcopy)[/yellow]")

    def _export_markdown_report(self, result, state):
        """Export full markdown report."""
        from pathlib import Path
        lines = [f"# Swarm Generation Report\n"]
        lines.append(f"**Requirements:** {state.requirements}\n")
        lines.append(
            f"**Team:** {state.team} | **Language:** {state.language}"
            f" | **Time:** {state.elapsed_str()}\n"
        )
        lines.append(f"**Files:** {len(state.files)} | **LOC:** {state.total_loc}\n")
        lines.append(
            f"**Validation:** {state.validation_status}"
            f" | **Review:** {state.review_status}\n"
        )
        lines.append("\n## Generated Files\n")
        for fname, content in result.code.files.items():
            loc = content.count('\n') + 1
            lines.append(f"\n### {fname} ({loc} lines)\n\n```python\n{content}\n```\n")
        if result.code.tests:
            lines.append("\n## Tests\n")
            for fname, content in result.code.tests.items():
                lines.append(f"\n### {fname}\n\n```python\n{content}\n```\n")
        if result.code.docs:
            lines.append(f"\n## Documentation\n\n{result.code.docs}\n")

        path = Path.cwd() / "swarm_report.md"
        path.write_text("".join(lines), encoding="utf-8")
        self.console.print(f"  [green]Report saved:[/green] {path}")

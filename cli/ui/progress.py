"""
Progress and Spinner Management
===============================

Rich-based progress bars and spinners for Jotty CLI.
"""

import asyncio
from contextlib import contextmanager
from typing import Optional, Any, Iterator

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

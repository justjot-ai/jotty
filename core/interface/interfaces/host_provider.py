"""
HostProvider — abstract interface between Jotty core and host environment.

Cline pattern: same core runs in VSCode extension or standalone process
by abstracting all host interactions behind HostProvider.

In Jotty: CLI, Web, Telegram, SDK all implement the same interface.
Core never imports host-specific code — it calls HostProvider methods.

KISS: 5 methods, no framework. Singleton with swap-in implementations.

Usage:
    # In CLI startup:
    HostProvider.initialize(CLIHost())

    # In core code (agent_runner, swarm_manager, etc.):
    host = HostProvider.get()
    host.notify("Task complete", level="info")
    answer = await host.prompt_user("Proceed with deletion?")
    host.display_progress(TaskProgress(...))
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Host(ABC):
    """Abstract host interface. Implementations: CLIHost, WebHost, TelegramHost, etc."""

    @abstractmethod
    def notify(self, message: str, level: str = "info") -> None:
        """Send a notification to the user. Levels: info, warning, error."""
        ...

    @abstractmethod
    async def prompt_user(self, question: str, default: str = "") -> str:
        """Ask the user a question and return their response."""
        ...

    @abstractmethod
    def display_progress(self, progress: Any) -> None:
        """Display task progress (TaskProgress, dict, or string)."""
        ...

    @abstractmethod
    def display_diff(self, diff_text: str, title: str = "") -> None:
        """Display a diff (workspace checkpoint diff, code changes, etc.)."""
        ...

    @abstractmethod
    def log_tool_use(
        self, tool_name: str, trust_level: str, allowed: bool, reason: str = ""
    ) -> None:
        """Log a tool use decision (for ToolGuard visibility)."""
        ...


class NullHost(Host):
    """Silent host — does nothing. Default before initialization."""

    # DRY: Level mapping constant
    _LOG_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}

    def notify(self, message: str, level: str = "info") -> None:
        log_level = self._LOG_LEVELS.get(level, logging.INFO)
        logger.log(log_level, f"[NullHost] {message}")

    async def prompt_user(self, question: str, default: str = "") -> str:
        logger.warning(f"[NullHost] prompt_user called but no host: {question}")
        return default

    def display_progress(self, progress: Any) -> None:
        if hasattr(progress, "render"):
            logger.info(f"[NullHost] {progress.render()}")

    def display_diff(self, diff_text: str, title: str = "") -> None:
        preview = diff_text[:500]
        logger.info(f"[NullHost] diff: {title}\n{preview}")

    def log_tool_use(
        self, tool_name: str, trust_level: str, allowed: bool, reason: str = ""
    ) -> None:
        status = "ALLOWED" if allowed else f"BLOCKED: {reason}"
        logger.debug(f"[ToolGuard] {tool_name} ({trust_level}) → {status}")


class CLIHost(Host):
    """CLI terminal host implementation."""

    # DRY: Level icons constant
    _ICONS = {"info": "ℹ", "warning": "", "error": ""}

    # DRY: ANSI color codes
    _COLOR_GREEN = "\033[32m"
    _COLOR_RED = "\033[31m"
    _COLOR_RESET = "\033[0m"
    _MAX_DIFF_LINES = 50

    def notify(self, message: str, level: str = "info") -> None:
        icon = self._ICONS.get(level, "")
        logger.info(f" {icon} {message}")

    async def prompt_user(self, question: str, default: str = "") -> str:
        import asyncio

        prompt = f" {question}"
        if default:
            prompt += f" [{default}]"
        prompt += ": "
        # Run input() in thread to not block event loop
        response = await asyncio.to_thread(input, prompt)
        return response.strip() or default

    def display_progress(self, progress: Any) -> None:
        if hasattr(progress, "render"):
            logger.info(progress.render())
        elif isinstance(progress, str):
            logger.info(progress)

    def display_diff(self, diff_text: str, title: str = "") -> None:
        if title:
            logger.info(f"\n{'='*40} {title} {'='*40}")

        # DRY: Colorize diff lines
        lines = diff_text.split("\n")
        for line in lines[: self._MAX_DIFF_LINES]:
            colored_line = self._colorize_diff_line(line)
            logger.info(colored_line)

        # Show truncation message if needed
        remaining = len(lines) - self._MAX_DIFF_LINES
        if remaining > 0:
            logger.info(f"  ... ({remaining} more lines)")

    def _colorize_diff_line(self, line: str) -> str:
        """DRY helper: colorize a diff line based on prefix."""
        if line.startswith("+"):
            return f"{self._COLOR_GREEN}{line}{self._COLOR_RESET}"
        elif line.startswith("-"):
            return f"{self._COLOR_RED}{line}{self._COLOR_RESET}"
        else:
            return line

    def log_tool_use(
        self, tool_name: str, trust_level: str, allowed: bool, reason: str = ""
    ) -> None:
        if not allowed:
            logger.warning(f"Tool blocked: {tool_name} ({trust_level}) — {reason}")


class HostProvider:
    """
    Singleton access point for the current host.

    Core code calls HostProvider.get() — never imports CLI/Web directly.
    Host environment calls HostProvider.initialize() at startup.
    """

    _instance: Optional[Host] = None

    @classmethod
    def initialize(cls, host: Host) -> None:
        """Set the active host implementation."""
        cls._instance = host
        logger.info(f"HostProvider initialized: {type(host).__name__}")

    @classmethod
    def get(cls) -> Host:
        """Get the current host. Returns NullHost if not initialized."""
        if cls._instance is None:
            cls._instance = NullHost()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset to NullHost (for testing)."""
        cls._instance = None

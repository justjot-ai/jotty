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
from typing import Optional, Any, Dict

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
    def log_tool_use(self, tool_name: str, trust_level: str, allowed: bool, reason: str = "") -> None:
        """Log a tool use decision (for ToolGuard visibility)."""
        ...


class NullHost(Host):
    """Silent host — does nothing. Default before initialization."""

    def notify(self, message: str, level: str = "info") -> None:
        logger.log(
            {'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR}.get(level, logging.INFO),
            f"[NullHost] {message}"
        )

    async def prompt_user(self, question: str, default: str = "") -> str:
        logger.warning(f"[NullHost] prompt_user called but no host: {question}")
        return default

    def display_progress(self, progress: Any) -> None:
        if hasattr(progress, 'render'):
            logger.info(f"[NullHost] {progress.render()}")

    def display_diff(self, diff_text: str, title: str = "") -> None:
        logger.info(f"[NullHost] diff: {title}\n{diff_text[:500]}")

    def log_tool_use(self, tool_name: str, trust_level: str, allowed: bool, reason: str = "") -> None:
        status = "ALLOWED" if allowed else f"BLOCKED: {reason}"
        logger.debug(f"[ToolGuard] {tool_name} ({trust_level}) → {status}")


class CLIHost(Host):
    """CLI terminal host implementation."""

    def notify(self, message: str, level: str = "info") -> None:
        icons = {'info': 'ℹ', 'warning': '', 'error': ''}
        icon = icons.get(level, '')
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
        if hasattr(progress, 'render'):
            logger.info(progress.render())
        elif isinstance(progress, str):
            logger.info(progress)

    def display_diff(self, diff_text: str, title: str = "") -> None:
        if title:
            logger.info(f"\n{'='*40} {title} {'='*40}")
        # Basic colorization for +/- lines
        for line in diff_text.split('\n')[:50]:
            if line.startswith('+'):
                logger.info(f"\033[32m{line}\033[0m")
            elif line.startswith('-'):
                logger.info(f"\033[31m{line}\033[0m")
            else:
                logger.info(line)
        if diff_text.count('\n') > 50:
            logger.info(f"  ... ({diff_text.count(chr(10)) - 50} more lines)")

    def log_tool_use(self, tool_name: str, trust_level: str, allowed: bool, reason: str = "") -> None:
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

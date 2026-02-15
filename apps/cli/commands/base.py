"""
Base Command Class
==================

Abstract base for CLI commands.
"""

import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..app import JottyCLI


@dataclass
class CommandResult:
    """Result from command execution."""

    success: bool = True
    output: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[str] = None
    should_exit: bool = False

    @classmethod
    def ok(cls, output: str = None, data: Any = None) -> "CommandResult":
        """Create success result."""
        return cls(success=True, output=output, data=data)

    @classmethod
    def fail(cls, error: str, data: Any = None) -> "CommandResult":
        """Create failure result."""
        return cls(success=False, error=error, data=data)

    @classmethod
    def exit(cls) -> "CommandResult":
        """Create exit result."""
        return cls(success=True, should_exit=True)


@dataclass
class ParsedArgs:
    """Parsed command arguments."""

    positional: List[str] = field(default_factory=list)
    flags: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""


class BaseCommand(ABC):
    """
    Base class for CLI commands.

    Subclasses implement execute() method.
    """

    name: str = "command"
    aliases: List[str] = []
    description: str = "A command"
    usage: str = "/command [args]"
    category: str = "general"

    @abstractmethod
    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """
        Execute the command.

        Args:
            args: Parsed command arguments
            cli: JottyCLI instance

        Returns:
            CommandResult
        """
        pass

    def parse_args(self, raw_args: str) -> ParsedArgs:
        """
        Parse raw argument string.

        Args:
            raw_args: Raw argument string

        Returns:
            ParsedArgs with positional args and flags
        """
        args = ParsedArgs(raw=raw_args)

        if not raw_args.strip():
            return args

        try:
            parts = shlex.split(raw_args)
        except ValueError:
            parts = raw_args.split()

        i = 0
        while i < len(parts):
            part = parts[i]

            # Long flag: --flag or --flag=value
            if part.startswith("--"):
                if "=" in part:
                    key, value = part[2:].split("=", 1)
                    args.flags[key] = value
                else:
                    key = part[2:]
                    # Check if next is a value
                    if i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                        args.flags[key] = parts[i + 1]
                        i += 1
                    else:
                        args.flags[key] = True

            # Short flag: -f or -f value
            elif part.startswith("-") and len(part) > 1:
                key = part[1:]
                # Check if next is a value
                if i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                    args.flags[key] = parts[i + 1]
                    i += 1
                else:
                    args.flags[key] = True

            else:
                args.positional.append(part)

            i += 1

        return args

    def get_completions(self, partial: str) -> List[str]:
        """
        Get autocomplete suggestions.

        Args:
            partial: Partial input

        Returns:
            List of completion suggestions
        """
        return []

    def help_text(self) -> str:
        """Get detailed help text."""
        text = f"/{self.name}"
        if self.aliases:
            text += f" (aliases: {', '.join('/' + a for a in self.aliases)})"
        text += f"\n\n{self.description}\n\nUsage: {self.usage}"
        return text


class CommandRegistry:
    """
    Registry for CLI commands.

    Maps command names and aliases to command instances.
    """

    def __init__(self) -> None:
        self._commands: Dict[str, BaseCommand] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, command: BaseCommand) -> Any:
        """
        Register a command.

        Args:
            command: Command instance
        """
        self._commands[command.name] = command

        for alias in command.aliases:
            self._aliases[alias] = command.name

    def get(self, name: str) -> Optional[BaseCommand]:
        """
        Get command by name or alias.

        Args:
            name: Command name or alias

        Returns:
            Command instance or None
        """
        # Check direct name
        if name in self._commands:
            return self._commands[name]

        # Check alias
        if name in self._aliases:
            return self._commands[self._aliases[name]]

        return None

    def list_commands(self) -> List[Dict[str, Any]]:
        """
        List all registered commands.

        Returns:
            List of command info dicts
        """
        commands = []
        for cmd in self._commands.values():
            commands.append(
                {
                    "name": cmd.name,
                    "aliases": cmd.aliases,
                    "description": cmd.description,
                    "usage": cmd.usage,
                    "category": cmd.category,
                }
            )
        return sorted(commands, key=lambda x: x["name"])

    def get_completions(self, partial: str) -> List[str]:
        """
        Get command name completions.

        Args:
            partial: Partial command name (without /)

        Returns:
            List of matching command names
        """
        matches = []

        for name in self._commands.keys():
            if name.startswith(partial):
                matches.append(name)

        for alias in self._aliases.keys():
            if alias.startswith(partial):
                matches.append(alias)

        return sorted(set(matches))

    @property
    def commands(self) -> Dict[str, BaseCommand]:
        """Get all commands."""
        return self._commands

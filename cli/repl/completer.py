"""
Command Completer
=================

Autocomplete for CLI commands.
"""

from typing import TYPE_CHECKING, List, Optional, Iterable

try:
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.document import Document
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False
    Completer = object
    Completion = None
    Document = None

if TYPE_CHECKING:
    from ..commands.base import CommandRegistry


class CommandCompleter(Completer if PROMPT_TOOLKIT_AVAILABLE else object):
    """
    Autocomplete for CLI commands.

    Provides completions for:
    - Slash commands (/run, /skills, etc.)
    - Command arguments
    - File paths
    """

    def __init__(self, command_registry: "CommandRegistry"):
        """
        Initialize completer.

        Args:
            command_registry: Command registry for command completions
        """
        self.command_registry = command_registry
        self._skill_names: List[str] = []

    def set_skill_names(self, names: List[str]):
        """Set available skill names for completion."""
        self._skill_names = names

    def get_completions(
        self,
        document: "Document",
        complete_event
    ) -> Iterable["Completion"]:
        """
        Get completions for current input.

        Args:
            document: Current document
            complete_event: Completion event

        Yields:
            Completion objects
        """
        if not PROMPT_TOOLKIT_AVAILABLE:
            return

        text = document.text_before_cursor
        word = document.get_word_before_cursor()

        # Slash command completion
        if text.startswith("/"):
            yield from self._complete_command(text, word)
        elif " " in text and text.split()[0].startswith("/"):
            # Command argument completion
            yield from self._complete_args(text, word)
        else:
            # General text completion (skills, etc.)
            yield from self._complete_general(text, word)

    def _complete_command(
        self,
        text: str,
        word: str
    ) -> Iterable["Completion"]:
        """Complete slash commands."""
        # Get the part after /
        cmd_text = text[1:].split()[0] if text[1:] else ""

        # Get matching commands
        matches = self.command_registry.get_completions(cmd_text)

        for match in matches:
            # Calculate replacement start position
            start_position = -len(cmd_text) if cmd_text else 0
            yield Completion(
                match,
                start_position=start_position,
                display_meta=self._get_command_meta(match)
            )

    def _complete_args(
        self,
        text: str,
        word: str
    ) -> Iterable["Completion"]:
        """Complete command arguments."""
        parts = text.split()
        if not parts:
            return

        cmd_name = parts[0][1:]  # Remove /
        cmd = self.command_registry.get(cmd_name)

        if not cmd:
            return

        # Get command-specific completions
        partial = parts[-1] if len(parts) > 1 else ""
        completions = cmd.get_completions(partial)

        for comp in completions:
            yield Completion(
                comp,
                start_position=-len(partial) if partial else 0,
            )

        # Common flags
        if word.startswith("-"):
            common_flags = ["--help", "--verbose", "-v", "--debug"]
            for flag in common_flags:
                if flag.startswith(word):
                    yield Completion(flag, start_position=-len(word))

    def _complete_general(
        self,
        text: str,
        word: str
    ) -> Iterable["Completion"]:
        """General completions."""
        # Skill names
        for skill in self._skill_names:
            if skill.startswith(word.lower()):
                yield Completion(skill, start_position=-len(word))

    def _get_command_meta(self, cmd_name: str) -> str:
        """Get command description for display."""
        cmd = self.command_registry.get(cmd_name)
        if cmd:
            return cmd.description[:40]
        return ""


class SimpleCompleter:
    """
    Simple completer for non-prompt_toolkit environments.

    Provides basic completion functionality.
    """

    def __init__(self, command_registry: "CommandRegistry"):
        self.command_registry = command_registry
        self._skill_names: List[str] = []

    def set_skill_names(self, names: List[str]):
        """Set available skill names."""
        self._skill_names = names

    def complete(self, text: str) -> List[str]:
        """
        Get completions for text.

        Args:
            text: Input text

        Returns:
            List of completion strings
        """
        if text.startswith("/"):
            cmd_text = text[1:].split()[0] if text[1:] else ""
            return ["/" + c for c in self.command_registry.get_completions(cmd_text)]

        # Skill completions
        return [s for s in self._skill_names if s.startswith(text.lower())]

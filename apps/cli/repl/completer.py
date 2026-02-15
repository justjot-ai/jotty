"""
Command Completer
=================

Autocomplete for CLI commands.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Iterable, Any

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
    - Command arguments and flags
    - Tool/skill names
    - File paths
    """

    # Command-specific flags and parameters
    COMMAND_FLAGS = {
        'run': ['--verbose', '--debug', '--timeout'],
        'ml': ['--target', '--context', '--iterations',
               # Seaborn datasets
               'titanic', 'iris', 'tips', 'penguins', 'diamonds', 'mpg',
               # Sklearn datasets
               'breast_cancer', 'wine', 'digits', 'california', 'diabetes'],
        'research': ['--quick', '--deep', '--sources', '--emit'],
        'skills': ['--category', '--search', 'list', 'info', 'run'],
        'tools': ['web-search', 'file-read', 'file-write', 'telegram-sender', 'docx-tools', 'document-converter'],
        'preview': ['--lines', '--raw', 'last', 'tools'],
        'browse': ['--type', '--preview', '.', '~', '/'],
        'export': ['last', 'history', 'code', 'c', 'd', 'p', 'm', 'cdpm'],
        'resume': ['list'],
        'config': ['show', 'set', 'reset', 'edit'],
        'learn': ['warmup', 'train', 'status'],
        'stats': ['--detailed', '--export'],
        'agents': ['list', 'status', 'add', 'remove'],
        'swarm': ['status', 'reset', 'config'],
        'memory': ['show', 'clear', 'export'],
        'plan': ['--steps', '--output'],
        'git': ['status', 'commit', 'push', 'pull', 'diff', 'undo', 'auto-commit'],
        'J': ['--tags', '--status'],
    }

    MAX_FILE_COMPLETIONS = 30

    def __init__(self, command_registry: 'CommandRegistry') -> None:
        """
        Initialize completer.

        Args:
            command_registry: Command registry for command completions
        """
        self.command_registry = command_registry
        self._skill_names: List[str] = []
        self._tool_names: List[str] = []
        self._code_tokens: List[str] = []

    def set_skill_names(self, names: List[str]) -> Any:
        """Set available skill names for completion."""
        self._skill_names = names
        # Also set as tool names
        self._tool_names = names

    def set_code_tokens(self, tokens: List[str]) -> Any:
        """
        Set recent function/class names from LLM output for code completion.

        Args:
            tokens: List of code identifiers (function/class names)
        """
        # Deduplicate, keep last 100
        seen = set()
        unique = []
        for t in reversed(tokens):
            if t not in seen and len(t) > 2:
                seen.add(t)
                unique.append(t)
        self._code_tokens = list(reversed(unique[:100]))

    def get_completions(self, document: 'Document', complete_event: Any) -> Iterable['Completion']:
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

        # Determine completion type
        if text.startswith("/"):
            # Check if we're still typing the command or already at arguments
            if " " in text:
                # Has space - we're in arguments territory
                yield from self._complete_args(text, word)
            else:
                # No space yet - completing the command name
                yield from self._complete_command(text, word)
        elif text.split() and text.split()[0].startswith("/"):
            # Command argument completion (alternate check)
            yield from self._complete_args(text, word)
        else:
            # General text completion (skills, etc.)
            yield from self._complete_general(text, word)

    def _complete_command(
        self,
        text: str,
        word: str
    ) -> Iterable["Completion"]:
        """Complete slash commands with descriptions and usage hints."""
        # Get the part after /
        cmd_text = text[1:].split()[0] if text[1:] else ""

        # Get matching commands
        matches = self.command_registry.get_completions(cmd_text)

        for match in matches:
            # Calculate replacement start position
            start_position = -len(cmd_text) if cmd_text else 0

            # Get command info
            cmd = self.command_registry.get(match)
            if cmd:
                # Build display with usage hint
                display = f"/{match}"
                meta = f"{cmd.description}"

                # Add usage hint
                if cmd.usage and cmd.usage != f"/{match}":
                    usage_hint = cmd.usage.replace(f"/{match}", "").strip()
                    if usage_hint:
                        display = f"{match} {usage_hint[:30]}"

                yield Completion(
                    match,
                    start_position=start_position,
                    display=display,
                    display_meta=meta[:50]
                )
            else:
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
        """Complete command arguments with flags and parameters."""
        parts = text.split()
        if not parts:
            return

        cmd_name = parts[0][1:]  # Remove /
        cmd = self.command_registry.get(cmd_name)

        if not cmd:
            return

        # Get the partial word being typed
        partial = word if word else ""

        # Get command-specific completions from the command itself
        cmd_completions = cmd.get_completions(partial)
        for comp in cmd_completions:
            if comp.startswith(partial) or not partial:
                yield Completion(
                    comp,
                    start_position=-len(partial) if partial else 0,
                    display_meta="argument"
                )

        # Get predefined flags and params for this command
        cmd_key = cmd.name
        if cmd_key in self.COMMAND_FLAGS:
            for flag in self.COMMAND_FLAGS[cmd_key]:
                if flag.startswith(partial) or not partial:
                    # Don't suggest already-used flags
                    if flag not in parts:
                        meta = "flag" if flag.startswith("-") else "option"
                        yield Completion(
                            flag,
                            start_position=-len(partial) if partial else 0,
                            display_meta=meta
                        )

        # Special case: /tools - show skill names
        if cmd_name == "tools" and len(parts) == 1:
            for skill in self._tool_names[:20]:
                if skill.startswith(partial) or not partial:
                    yield Completion(
                        skill,
                        start_position=-len(partial) if partial else 0,
                        display_meta="skill/tool"
                    )

        # Common flags for all commands
        common_flags = ["--help", "-h"]
        for flag in common_flags:
            if flag.startswith(partial) and flag not in parts:
                yield Completion(
                    flag,
                    start_position=-len(partial) if partial else 0,
                    display_meta="help"
                )

    def _complete_file_paths(
        self,
        partial: str,
    ) -> Iterable["Completion"]:
        """
        Complete relative file paths.

        Limits to MAX_FILE_COMPLETIONS per directory, skips hidden files.

        Args:
            partial: Partial file path typed so far

        Yields:
            Completion objects for matching paths
        """
        if not PROMPT_TOOLKIT_AVAILABLE:
            return

        try:
            # Determine base directory and prefix
            path_obj = Path(partial)
            if partial.endswith(os.sep) or partial.endswith("/"):
                base_dir = path_obj
                prefix = ""
            else:
                base_dir = path_obj.parent
                prefix = path_obj.name

            # Resolve relative to cwd
            if not base_dir.is_absolute():
                base_dir = Path.cwd() / base_dir

            if not base_dir.is_dir():
                return

            count = 0
            for entry in sorted(base_dir.iterdir()):
                # Skip hidden files
                if entry.name.startswith("."):
                    continue
                if count >= self.MAX_FILE_COMPLETIONS:
                    break
                if entry.name.startswith(prefix) or not prefix:
                    display_name = entry.name
                    if entry.is_dir():
                        display_name += "/"
                    # Calculate replacement
                    completion_text = display_name
                    yield Completion(
                        completion_text,
                        start_position=-len(prefix) if prefix else 0,
                        display_meta="dir" if entry.is_dir() else "file",
                    )
                    count += 1
        except (PermissionError, OSError):
            pass

    def _complete_general(
        self,
        text: str,
        word: str
    ) -> Iterable["Completion"]:
        """General completions: skills, file paths, and code tokens."""
        # Skill names
        for skill in self._skill_names:
            if skill.startswith(word.lower()):
                yield Completion(skill, start_position=-len(word))

        # File path completions (if word contains path separator or starts with .)
        if word and (os.sep in word or word.startswith(".") or "/" in word):
            yield from self._complete_file_paths(word)

        # Code token completions
        for token in self._code_tokens:
            if token.startswith(word) and word:
                yield Completion(
                    token,
                    start_position=-len(word),
                    display_meta="code",
                )

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

    def __init__(self, command_registry: 'CommandRegistry') -> None:
        self.command_registry = command_registry
        self._skill_names: List[str] = []

    def set_skill_names(self, names: List[str]) -> Any:
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
